"""One command: tunnel, webhook registration, receiver, conversation, cleanup.

The manual flow needs three terminals and a dashboard visit. This does the whole
thing in one process:

1. opens an ngrok tunnel to a local port
2. registers a webhook with ElevenLabs pointing at it — the create call hands
   back the signing secret, so no copy-paste from the dashboard
3. points *the one agent you are calling* at that webhook, with
   ``transcript_format=opentelemetry``
4. starts the receiver
5. holds the conversation, then waits for the post-call webhooks to land
6. puts the agent's webhook config back and deletes the webhook it created

Step 3 and step 6 are the reason this is scoped to a per-agent override rather
than the workspace-level post-call webhook: the workspace setting is shared with
everyone else in your ElevenLabs workspace, and clobbering it for the length of
a demo call would break their traces too.

Needs an ngrok authtoken on top of the usual ``ELEVENLABS_API_KEY`` /
``ELEVENLABS_AGENT_ID``. If you have ever run ``ngrok config add-authtoken`` or
signed in to the ngrok CLI, that is already on disk and this finds it — the
embedded agent used here reads only ``NGROK_AUTHTOKEN``, unlike the CLI, so we
fall back to the CLI's own config file rather than making you copy it out.
"""

from __future__ import annotations

import logging
import os
import re
import sys
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import uvicorn
from elevenlabs import ElevenLabs
from elevenlabs.types.webhook_hmac_settings import WebhookHmacSettings

from ..webhook import build_app

logger = logging.getLogger(__name__)

WEBHOOK_NAME = "voice-demo (temporary)"
# Post-call analysis has to finish before ElevenLabs sends anything, and there
# is no documented SLA for it, so this is deliberately patient.
WEBHOOK_WAIT_SECONDS = 240


def _say(step: str, detail: str = "") -> None:
    print(f"[voice-demo] {step}{' ' + detail if detail else ''}", file=sys.stderr)


# Where the ngrok CLI keeps its config, newest location first. The embedded
# agent does not read these itself.
_NGROK_CONFIGS = (
    Path.home() / "Library/Application Support/ngrok/ngrok.yml",  # macOS
    Path(os.environ.get("LOCALAPPDATA", "")) / "ngrok/ngrok.yml",  # Windows
    Path.home() / ".config/ngrok/ngrok.yml",  # Linux
    Path.home() / ".ngrok2/ngrok.yml",  # ngrok v2
)
_AUTHTOKEN_RE = re.compile(r"^\s*authtoken:\s*(\S+)\s*$", re.MULTILINE)


def _ngrok_authtoken() -> str | None:
    """The env var, else the token the ngrok CLI already has on disk."""
    if token := os.environ.get("NGROK_AUTHTOKEN"):
        return token
    for config in _NGROK_CONFIGS:
        try:
            match = _AUTHTOKEN_RE.search(config.read_text())
        except OSError:
            continue
        if match:
            logger.debug("using the ngrok authtoken from %s", config)
            return match.group(1).strip("\"'")
    return None


@contextmanager
def _tunnel(port: int) -> Iterator[str]:
    """An ngrok tunnel to ``port``, closed on the way out."""
    try:
        import ngrok
    except ImportError:
        sys.exit(
            "[voice-demo] The one-command flow needs ngrok: uv sync --extra elevenlabs"
        )
    authtoken = _ngrok_authtoken()
    if not authtoken:
        sys.exit(
            "[voice-demo] No ngrok authtoken found. Either run "
            "`ngrok config add-authtoken <token>` or set NGROK_AUTHTOKEN. "
            "Get one free at https://dashboard.ngrok.com/get-started/your-authtoken"
        )
    listener = ngrok.forward(port, authtoken=authtoken)
    try:
        yield listener.url()
    finally:
        try:
            ngrok.disconnect(listener.url())
        except Exception:
            logger.debug("ngrok tunnel did not close cleanly", exc_info=True)


def _needs_permission(error: Exception, permission: str, action: str) -> None:
    """Turn ElevenLabs' permission errors into something you can act on."""
    if getattr(error, "status_code", None) not in (401, 403):
        return
    sys.exit(
        f"[voice-demo] This ElevenLabs API key cannot {action}: it is missing the "
        f"`{permission}` permission.\n"
        "[voice-demo] Either create a key with that permission at "
        "https://elevenlabs.io/app/settings/api-keys, or skip --tunnel and wire "
        "the webhook up by hand (see the README)."
    )


@contextmanager
def _webhook(client: ElevenLabs, url: str) -> Iterator[tuple[str, str]]:
    """A temporary ElevenLabs webhook pointing at ``url``, deleted on the way out."""
    try:
        created = client.webhooks.create(
            settings=WebhookHmacSettings(name=WEBHOOK_NAME, webhook_url=url)
        )
    except Exception as error:
        _needs_permission(error, "webhooks_write", "register a webhook")
        raise
    if not created.webhook_secret:
        # Only returned at creation; without it nothing can be verified.
        client.webhooks.delete(created.webhook_id)
        sys.exit("[voice-demo] ElevenLabs did not return a webhook signing secret.")
    try:
        yield created.webhook_id, created.webhook_secret
    finally:
        try:
            client.webhooks.delete(created.webhook_id)
            _say("cleaned up: deleted webhook", created.webhook_id)
        except Exception:  # noqa: BLE001 - teardown is best-effort
            _say(
                "could not delete webhook", f"{created.webhook_id} — remove it by hand"
            )


@contextmanager
def _agent_override(
    client: ElevenLabs, agent_id: str, webhook_id: str
) -> Iterator[None]:
    """Point one agent's post-call webhook at ours, restoring it afterwards."""
    try:
        agent = client.conversational_ai.agents.get(agent_id)
    except Exception as error:
        _needs_permission(error, "convai_read", "read the agent")
        raise
    settings = getattr(agent, "platform_settings", None)
    previous = getattr(settings, "workspace_overrides", None)

    client.conversational_ai.agents.update(
        agent_id,
        platform_settings={
            "workspace_overrides": {
                "webhooks": {
                    "post_call_webhook_id": webhook_id,
                    # Only the trace. The recording is pulled from the
                    # conversations API instead — see webhook.py for why.
                    "events": ["transcript"],
                    # Without this ElevenLabs sends a JSON transcript, not a trace.
                    "transcript_format": "opentelemetry",
                }
            }
        },
    )
    try:
        yield
    finally:
        try:
            restored: Any = (
                previous.model_dump(exclude_none=True) if previous is not None else None
            )
            client.conversational_ai.agents.update(
                agent_id, platform_settings={"workspace_overrides": restored}
            )
            _say("cleaned up: restored the agent's webhook settings")
        except Exception:  # noqa: BLE001 - teardown is best-effort
            _say(
                "could not restore the agent's webhook settings — "
                f"check agent {agent_id} in the dashboard"
            )


@contextmanager
def _receiver(project_name: str, secret: str, port: int) -> Iterator[Any]:
    """The webhook receiver, served on a background thread."""
    app = build_app(project_name, secret=secret)
    server = uvicorn.Server(
        uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning")
    )
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    while not server.started and thread.is_alive():
        time.sleep(0.05)
    if not server.started:
        sys.exit(f"[voice-demo] The webhook receiver failed to bind port {port}.")
    try:
        yield app.state.receiver
    finally:
        server.should_exit = True
        thread.join(timeout=10)


def _wait_for_export(receiver: Any, conversation_id: str | None) -> bool:
    """Block until the conversation's webhooks land and export, or we give up."""
    _say(
        f"waiting up to {WEBHOOK_WAIT_SECONDS}s for ElevenLabs to finish "
        "post-call analysis and send the webhooks..."
    )
    deadline = time.monotonic() + WEBHOOK_WAIT_SECONDS
    while time.monotonic() < deadline:
        # No id (the session ended abnormally) — settle for any export.
        if conversation_id in receiver.exported or (
            conversation_id is None and receiver.exported
        ):
            return True
        time.sleep(1.0)
    return False


def run(project_name: str, *, port: int) -> None:
    """Run the whole ElevenLabs demo loop in one process."""
    from ..agent import converse

    api_key = os.environ.get("ELEVENLABS_API_KEY") or os.environ.get("ELEVEN_API_KEY")
    agent_id = os.environ.get("ELEVENLABS_AGENT_ID")
    if not api_key:
        sys.exit("[voice-demo] ELEVENLABS_API_KEY is required to register a webhook.")
    if not agent_id:
        sys.exit("[voice-demo] ELEVENLABS_AGENT_ID is not set.")

    client = ElevenLabs(api_key=api_key)

    with _tunnel(port) as public_url:
        _say("tunnel:", public_url)
        with _webhook(client, f"{public_url}/webhook") as (webhook_id, secret):
            _say("registered webhook:", webhook_id)
            with _agent_override(client, agent_id, webhook_id):
                _say("agent now reports to it:", agent_id)
                with _receiver(project_name, secret, port) as receiver:
                    _say(f"receiver listening on 127.0.0.1:{port}")
                    conversation_id = converse(client, agent_id, requires_auth=True)
                    if _wait_for_export(receiver, conversation_id):
                        _say(f"traced to LangSmith project '{project_name}'.")
                    else:
                        _say(
                            "gave up waiting. The webhooks may still arrive — "
                            "run the receiver on its own to catch them."
                        )

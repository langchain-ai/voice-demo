"""Receive ElevenLabs post-call webhooks and forward them to LangSmith.

ElevenLabs POSTs `post_call_transcription_otel` after each call — the finished
OTLP trace, at `data.otlp_traces` (only when the webhook's `transcript_format`
is `opentelemetry`). This server verifies it, attaches the conversation audio,
and hands both to `langsmith.integrations.elevenlabs.aexport_elevenlabs_trace`.

The audio does NOT come from the `post_call_audio` webhook, even though one
exists. That webhook arrives independently of the trace, in either order, and is
never retried — so pairing them means holding the trace back on a timer and
still losing the recording whenever the audio delivery fails. Instead the audio
is pulled from `GET /v1/convai/conversations/{id}/audio` the moment the trace
lands: it takes well under a second, it cannot arrive out of order, and it can
be retried. Nothing waits.

That the export must happen exactly once is not a preference. LangSmith's OTLP
ingest answers a repeat of a span it already has with:

    409  Run create payload already received.
         Duplicate run create requests for the same run are not supported.

So there is no "export now, attach the audio later" — whatever the trace carries
on its first and only export is what LangSmith keeps.

ElevenLabs delivers to a public URL, so put a tunnel in front of this::

    uv run voice-demo --backend elevenlabs-webhook --port 8080
    ngrok http 8080

or let `--backend elevenlabs --tunnel` do the whole thing (see `tunnel.py`).
"""

from __future__ import annotations

import asyncio
import logging
import os
import sys
from collections.abc import Mapping
from typing import Any

from elevenlabs import ElevenLabs
from fastapi import FastAPI, Request, Response
from langsmith.integrations.elevenlabs import aexport_elevenlabs_trace

logger = logging.getLogger(__name__)

OTEL_EVENT = "post_call_transcription_otel"

# The trace webhook fires once post-call analysis is done, so the recording is
# normally ready too. These bound the rare case where it is not quite.
AUDIO_FETCH_ATTEMPTS = 3
AUDIO_RETRY_SECONDS = 1.0
# The trace is JSON text; without the base64 audio webhook it stays small.
MAX_BODY_BYTES = 32 * 1024 * 1024


class WebhookReceiver:
    """Verifies post-call webhooks and exports each conversation exactly once."""

    def __init__(
        self, *, client: ElevenLabs, secret: str, project_name: str, with_audio: bool
    ) -> None:
        self._client = client
        self._secret = secret
        self._project_name = project_name
        self._with_audio = with_audio
        # Conversation ids already sent. Guards against ElevenLabs' retries (it
        # retries the transcript webhook on 5xx/429/408) hitting the 409 above,
        # and lets the one-command flow know when it can shut down.
        self.exported: set[str] = set()

    def verify(self, body: bytes, signature: str | None) -> dict[str, Any]:
        """Authenticate one delivery and return its parsed event.

        Delegates to the ElevenLabs SDK, which checks the ``t=,v0=`` HMAC over
        ``{timestamp}.{body}`` and rejects stale timestamps. Raises on failure.
        """
        return self._client.webhooks.construct_event(
            rawBody=body.decode("utf-8"),
            sig_header=signature or "",
            secret=self._secret,
        )

    async def receive(self, event: dict[str, Any]) -> None:
        """Export one verified trace webhook, audio and all."""
        if event.get("type") != OTEL_EVENT:
            # Includes `post_call_audio`: harmless, just not what we trace from.
            logger.info("ignoring webhook %r", event.get("type"))
            return

        data = event.get("data") or {}
        conversation_id = data.get("conversation_id")
        if not conversation_id:
            logger.warning("trace webhook has no conversation_id; dropping")
            return
        if conversation_id in self.exported:
            logger.info("already exported %s; ignoring retry", conversation_id)
            return

        audio = await self._audio(conversation_id) if self._with_audio else None
        await self._export(conversation_id, event, audio)

    async def _audio(self, conversation_id: str) -> bytes | None:
        """Fetch the conversation recording as raw MP3 bytes."""
        for attempt in range(1, AUDIO_FETCH_ATTEMPTS + 1):
            try:
                chunks = await asyncio.to_thread(
                    lambda: list(
                        self._client.conversational_ai.conversations.audio.get(
                            conversation_id=conversation_id
                        )
                    )
                )
            except Exception as error:  # noqa: BLE001 - audio is best-effort
                status = getattr(error, "status_code", None)
                logger.info(
                    "audio for %s not ready (attempt %d/%d): %s",
                    conversation_id,
                    attempt,
                    AUDIO_FETCH_ATTEMPTS,
                    f"HTTP {status}" if status else type(error).__name__,
                )
                if status == 404:
                    # No such conversation — retrying will not change that.
                    break
                if attempt < AUDIO_FETCH_ATTEMPTS:
                    await asyncio.sleep(AUDIO_RETRY_SECONDS)
                continue
            raw = b"".join(chunks)
            if raw:
                logger.info(
                    "fetched %d bytes of audio for %s", len(raw), conversation_id
                )
                return raw
        logger.warning(
            "no audio for %s; exporting the trace without it", conversation_id
        )
        return None

    async def _export(
        self, conversation_id: str, event: dict[str, Any], audio: bytes | None
    ) -> None:
        # The SDK takes the OTLP envelope itself, and tolerates a re-delivery.
        otlp_traces = (event.get("data") or {}).get("otlp_traces")
        if not isinstance(otlp_traces, Mapping):
            logger.warning("%s has no data.otlp_traces; dropping", conversation_id)
            return
        try:
            await aexport_elevenlabs_trace(
                otlp_traces, audio=audio, project_name=self._project_name
            )
        except Exception:
            logger.exception("failed exporting %s to LangSmith", conversation_id)
            return
        self.exported.add(conversation_id)
        logger.info(
            "exported %s to LangSmith project %r (audio=%s)",
            conversation_id,
            self._project_name,
            audio is not None,
        )


def build_app(project_name: str, secret: str | None = None) -> FastAPI:
    """Build the receiver's ASGI app, failing fast on missing configuration.

    ``secret`` overrides ``ELEVENLABS_WEBHOOK_SECRET`` — the one-command flow
    registers the webhook itself and passes back the secret ElevenLabs returns.
    The receiver is published on ``app.state.receiver`` so that caller can watch
    what has been exported.
    """
    secret = secret or os.environ.get("ELEVENLABS_WEBHOOK_SECRET")
    if not secret:
        sys.exit(
            "[voice-demo] ELEVENLABS_WEBHOOK_SECRET is not set. Copy it from the "
            "post-call webhook you created in the ElevenLabs dashboard."
        )
    api_key = os.environ.get("ELEVENLABS_API_KEY") or os.environ.get("ELEVEN_API_KEY")
    if not api_key:
        logger.warning(
            "no ELEVENLABS_API_KEY: traces will be exported without their audio, "
            "since the recording is fetched from the conversations API."
        )

    receiver = WebhookReceiver(
        client=ElevenLabs(api_key=api_key),
        secret=secret,
        project_name=project_name,
        with_audio=bool(api_key),
    )
    app = FastAPI()
    app.state.receiver = receiver

    @app.post("/webhook")
    async def webhook(request: Request) -> Response:
        # Streamed so an oversized delivery is refused rather than buffered, and
        # kept as raw bytes because the HMAC covers the exact body ElevenLabs
        # sent — re-serializing parsed JSON would invalidate it.
        body = bytearray()
        async for chunk in request.stream():
            body.extend(chunk)
            if len(body) > MAX_BODY_BYTES:
                return Response(status_code=413)

        try:
            event = receiver.verify(
                bytes(body), request.headers.get("elevenlabs-signature")
            )
        except Exception:  # noqa: BLE001 - any failure to verify is a rejection
            # No detail to the caller, and never the body or the secret.
            logger.warning("rejected a webhook with an invalid signature")
            return Response(status_code=401)

        await receiver.receive(event)
        # ElevenLabs disables a webhook after repeated non-2xx replies, so this
        # acknowledges delivery; export failures are logged, not signalled back.
        return Response(status_code=204)

    return app


def run(project_name: str, *, port: int) -> None:
    """Serve the webhook receiver until interrupted."""
    import uvicorn

    print(
        f"[voice-demo] Webhook receiver on http://localhost:{port}/webhook\n"
        f"[voice-demo] Expose it with: ngrok http {port}\n"
        f"[voice-demo] Traces land in LangSmith project '{project_name}'.",
        file=sys.stderr,
    )
    uvicorn.run(build_app(project_name), host="127.0.0.1", port=port, log_level="info")

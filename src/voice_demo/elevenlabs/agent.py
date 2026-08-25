"""ElevenLabs Agents conversation, run from the terminal.

This backend is the odd one out. Every other agent here runs in-process, so the
LangSmith integration can watch it live. ElevenLabs runs the whole agent
server-side and emits the trace *after* the call, as a post-call webhook — so
this module only holds the conversation, and the trace arrives separately at the
``elevenlabs-webhook`` backend, which verifies it and forwards it to LangSmith.

That means nothing is traced unless the webhook receiver is already running and
reachable from the public internet before you hang up. See
``voice_demo.elevenlabs.webhook`` for the ngrok setup.

Audio is the local mic and speaker via ``SoundDeviceAudioInterface``; the agent's
prompt, voice, and tools all live in its ElevenLabs dashboard configuration
rather than in this repo.
"""

from __future__ import annotations

import os
import signal
import sys
from types import FrameType

from elevenlabs import ElevenLabs
from elevenlabs.conversational_ai.conversation import Conversation

from .utils.audio import SoundDeviceAudioInterface


def converse(client: ElevenLabs, agent_id: str, *, requires_auth: bool) -> str | None:
    """Hold one conversation, returning its id once the caller hangs up."""
    conversation = Conversation(
        client,
        agent_id,
        # A private agent needs the API key; a public one can connect without.
        requires_auth=requires_auth,
        audio_interface=SoundDeviceAudioInterface(),
        callback_agent_response=lambda text: print(f"agent> {text}"),
        callback_user_transcript=lambda text: print(f"you>   {text}"),
    )

    def _hang_up(_signum: int, _frame: FrameType | None) -> None:
        conversation.end_session()

    signal.signal(signal.SIGINT, _hang_up)

    print("[voice-demo] Talk to the agent. Ctrl-C to hang up.", file=sys.stderr)
    conversation.start_session()
    conversation_id = conversation.wait_for_session_end()
    print(f"\n[voice-demo] Conversation ended: {conversation_id}", file=sys.stderr)
    return conversation_id


def run(project_name: str) -> None:
    """Hold one conversation, leaving its trace to a separately-run receiver."""
    api_key = os.environ.get("ELEVENLABS_API_KEY") or os.environ.get("ELEVEN_API_KEY")
    agent_id = os.environ.get("ELEVENLABS_AGENT_ID")
    if not agent_id:
        sys.exit(
            "[voice-demo] ELEVENLABS_AGENT_ID is not set. Create an agent at "
            "https://elevenlabs.io/app/agents and copy its id into .env."
        )

    converse(ElevenLabs(api_key=api_key), agent_id, requires_auth=bool(api_key))
    print(
        "[voice-demo] ElevenLabs will POST the trace to your webhook URL shortly; "
        f"it lands in LangSmith project '{project_name}'.",
        file=sys.stderr,
    )

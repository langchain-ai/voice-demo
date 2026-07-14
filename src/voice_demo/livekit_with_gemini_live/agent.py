"""LiveKit voice agent using Gemini Live (S2S), traced via OpenTelemetry.

This is the `livekit` cascade backend with its LLM slot swapped for a
speech-to-speech model: Google's Gemini Live model ingests audio and emits audio
directly, so there is no separate STT / LLM / TTS stage, no VAD, and no turn
detector — the model does it all. Everything else (the tracing wiring, the
native `@function_tool`, the console entrypoint) is identical to the cascade;
see `livekit/agent.py` for the fully-commented version. The two live in separate
folders, boilerplate and all, so each is a self-contained example of one setup.

Why OTEL and not SDK: LiveKit's Agents SDK emits OTel spans natively; we call
`configure_livekit` (the LangSmith LiveKit integration), which translates
LiveKit's `lk.*` attributes into the `gen_ai.*` / `langsmith.*` namespaces the
LangSmith UI keys off, and attaches audio for free.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# The conversation's thread id is set once per session via the SDK's
# ``set_thread_id`` (a ContextVar) in the entrypoint; the processor reads it per
# span. The audio path is handed to the processor as an app-owned provider.
#
# A module global, not a ContextVar: deferred root spans can be exported at
# flush/shutdown, outside the job's task tree. Console mode runs one job per
# process, so a single slot is enough.
_audio_file_path: Path | None = None
# This demo runs in console mode, so it uses the local-file recording path
# (audio_path_provider) below. In a deployed worker ctx.session_directory is
# ephemeral; production captures audio with LiveKit Egress and attaches it via
# the processor's expect_recording / complete_recording methods. See the
# "Record in production with Egress" section of the LangSmith LiveKit docs.


def run(project_name: str) -> None:
    """Launch a LiveKit console agent (Gemini Live). Blocks until Ctrl-C.

    Args:
        project_name: LangSmith project (informational — the OTLP headers wired
            by `tracing.configure` decide where spans land).
    """
    if not os.environ.get("GOOGLE_API_KEY"):
        print(
            "GOOGLE_API_KEY is not set (Gemini Live needs it).",
            file=sys.stderr,
        )
        sys.exit(1)

    # Import after the env vars are set in tracing.configure() so the OTLP
    # exporter picks them up at module import time.
    from livekit import agents
    from livekit.agents import Agent, AgentSession, function_tool, room_io

    from ..prompts import GREETING, SYSTEM_PROMPT
    from langsmith.integrations.livekit import configure_livekit, set_thread_id
    from ..weather import fetch_weather

    # --- Tracing wiring --- (see livekit/agent.py for the full rationale)
    processor = None
    if os.environ.get("LANGSMITH_API_KEY"):
        processor = configure_livekit(
            audio_path_provider=lambda: _audio_file_path,
            project=project_name,
        )
        print("[livekit-gemini-live] LangSmith OTel tracing enabled.", file=sys.stderr)
    else:
        print(
            "[livekit-gemini-live] LANGSMITH_API_KEY not set — running without tracing.",
            file=sys.stderr,
        )

    # Plugins must be imported on the main thread (LiveKit registers them at
    # import time), so this can't be deferred into the job entrypoint.
    from livekit.plugins import google as lk_google

    def _build_session() -> AgentSession:
        """A single speech-to-speech model fills the whole session."""
        return AgentSession(llm=lk_google.beta.realtime.RealtimeModel(voice="Puck"))

    class _Assistant(Agent):
        """Instructions + one native function tool."""

        def __init__(self) -> None:
            super().__init__(instructions=SYSTEM_PROMPT)

        @function_tool
        async def lookup_weather(self, city: str) -> dict:
            """Get the current weather for a single city. Call once per city."""
            return await fetch_weather(city)

    server = agents.AgentServer()

    @server.rtc_session()
    async def _entrypoint(ctx: agents.JobContext) -> None:
        global _audio_file_path
        # ctx.job.id is unique per dispatch; ctx.room.name is "console" in
        # console mode and would collide every session into one giant thread.
        set_thread_id(ctx.job.id)
        _audio_file_path = ctx.session_directory / "audio.ogg"

        session = _build_session()
        if processor is not None:
            processor.instrument_session(session, ctx.job.id)
        await session.start(
            room=ctx.room,
            agent=_Assistant(),
            room_options=room_io.RoomOptions(),
            record={"audio": True},  # see livekit/agent.py for the console/prod note
        )

        # Speak first. Realtime models report supports_say == False — there is no
        # TTS to synthesize a fixed string — so the model speaks the opener
        # itself from an instruction.
        await session.generate_reply(instructions=f'Say: "{GREETING}"')

    # Force `console` + `--record` so the user gets a local mic+speaker session
    # with recording without remembering the subcommand. LiveKit's CLI owns argv
    # from here. (See livekit/agent.py for the full explanation.)
    sys.argv = [sys.argv[0], "console", "--record"]
    agents.cli.run_app(server)

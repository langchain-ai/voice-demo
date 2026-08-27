"""LiveKit voice agent using OpenAI Realtime (S2S), traced via OpenTelemetry.

This is the `livekit` cascade backend with its LLM slot swapped for a
speech-to-speech model: OpenAI's Realtime model ingests audio and emits audio
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


def run(project_name: str) -> None:
    """Launch a LiveKit console agent (OpenAI Realtime). Blocks until Ctrl-C.

    Args:
        project_name: LangSmith project (informational — the OTLP headers wired
            by `tracing.configure` decide where spans land).
    """
    if not os.environ.get("OPENAI_API_KEY"):
        print(
            "OPENAI_API_KEY is not set (OpenAI Realtime needs it).",
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
            project=project_name,
        )
        print(
            "[livekit-openai-realtime] LangSmith OTel tracing enabled.", file=sys.stderr
        )
    else:
        print(
            "[livekit-openai-realtime] LANGSMITH_API_KEY not set — running without tracing.",
            file=sys.stderr,
        )

    # Plugins must be imported on the main thread (LiveKit registers them at
    # import time), so this can't be deferred into the job entrypoint.
    from livekit.plugins import openai as lk_openai

    def _build_session() -> AgentSession:
        """A single speech-to-speech model fills the whole session."""
        return AgentSession(llm=lk_openai.realtime.RealtimeModel(voice="marin"))

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
        # ctx.job.id is unique per dispatch; ctx.room.name is "console" in
        # console mode and would collide every session into one giant thread.
        set_thread_id(ctx.job.id)

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

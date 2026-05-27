"""LiveKit voice agent, traced via OpenTelemetry into LangSmith.

Why OTEL and not SDK: LiveKit's Agents SDK emits OTel spans natively across
its STT, LLM, TTS, turn-detection, and EOU pipeline. The framework does the
hard work — we just wire a TracerProvider with the vendored
`LangSmithSpanProcessor` (in `processor.py`) that translates LiveKit's `lk.*`
vendor-prefixed attributes into the `gen_ai.*` / `langsmith.*` namespaces
that the LangSmith UI keys off. Audio attachments and per-turn latency
aggregation come for free because the processor already does them.

The tracing setup is the only "tracing code" in this file:

    provider = TracerProvider()
    provider.add_span_processor(LangSmithSpanProcessor())   # wraps OTLPSpanExporter
    set_tracer_provider(provider)                            # LiveKit's hook

Everything else is the standard LiveKit AgentSession boilerplate — STT, LLM,
TTS, VAD, turn detector — driven through `agents.cli.run_app()` in console
mode so the local mic + speaker are wired in by the SDK itself.

(Unlike the OpenAI and ADK backends, we do not use `voice_demo.audio` here.
LiveKit owns its own audio I/O when running in console mode, and going around
it would lose the framework-emitted per-turn telemetry that is the whole
point of using LiveKit.)
"""

from __future__ import annotations

import os
import sys


def run(project_name: str) -> None:
    """Launch LiveKit's console agent. Blocks until Ctrl-C."""
    if not os.environ.get("OPENAI_API_KEY"):
        print("OPENAI_API_KEY is not set (LiveKit's STT/LLM/TTS need it).", file=sys.stderr)
        sys.exit(1)

    # Import after the env vars are set in tracing.configure() so the OTLP
    # exporter picks them up at module import time.
    from livekit import agents
    from livekit.agents import Agent, AgentSession, room_io
    from livekit.agents.telemetry import set_tracer_provider
    from livekit.plugins import openai as lk_openai
    from livekit.plugins import silero
    from livekit.plugins.turn_detector.multilingual import MultilingualModel
    from opentelemetry import trace as otel_trace
    from opentelemetry.sdk.trace import TracerProvider

    from ._thread_id import set_active_thread_id, set_audio_file_path
    from .processor import LangSmithSpanProcessor

    # --- Tracing wiring ---
    if os.environ.get("LANGSMITH_API_KEY"):
        provider = TracerProvider()
        # The processor wraps a BatchSpanProcessor(OTLPSpanExporter()) internally
        # — it reads OTEL_EXPORTER_OTLP_ENDPOINT / _HEADERS from env.
        provider.add_span_processor(LangSmithSpanProcessor())
        set_tracer_provider(provider)           # LiveKit's hook
        otel_trace.set_tracer_provider(provider) # OTel global (for LangChain etc.)
        print("[livekit] LangSmith OTel tracing enabled.", file=sys.stderr)
    else:
        print(
            "[livekit] LANGSMITH_API_KEY not set — running without tracing.",
            file=sys.stderr,
        )

    # --- Agent + dispatch ---
    _INSTRUCTIONS = (
        "You are a friendly voice assistant who can chat about anything. "
        "Keep replies short, conversational, and free of formatting "
        "(no asterisks, no bullet points, no emoji)."
    )

    class _Assistant(Agent):
        def __init__(self) -> None:
            super().__init__(instructions=_INSTRUCTIONS)

    server = agents.AgentServer()

    @server.rtc_session()
    async def _entrypoint(ctx: agents.JobContext) -> None:
        # ctx.job.id is unique per dispatch; ctx.room.name is "console" in
        # console mode and would collide every session into one giant thread.
        set_active_thread_id(ctx.job.id)
        set_audio_file_path(ctx.session_directory / "audio.ogg")

        session = AgentSession(
            stt=lk_openai.STT(model="gpt-4o-mini-transcribe"),
            llm=lk_openai.LLM(model="gpt-4o-mini"),
            tts=lk_openai.TTS(model="tts-1", voice="alloy"),
            vad=silero.VAD.load(),
            turn_detection=MultilingualModel(),
        )

        await session.start(
            room=ctx.room,
            agent=_Assistant(),
            room_options=room_io.RoomOptions(),
            # `record={"audio": True}` enables the writer in `dev`/`start` modes;
            # in `console` mode you additionally need the CLI flag `--record`.
            record={"audio": True},
        )

    # LiveKit's CLI owns argv parsing from here. We force `console` so the user
    # gets a local mic+speaker session without having to remember the subcommand.
    sys.argv = [sys.argv[0], "console"]
    agents.cli.run_app(server)

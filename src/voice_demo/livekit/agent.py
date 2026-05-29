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

The "LLM" here is a LangGraph ReAct agent (with a weather tool) plugged in via
the langchain plugin's `LLMAdapter`, rather than a bare chat model. The graph
is kept stateless (no checkpointer / thread_id) so LiveKit's own ChatContext
remains the single source of truth — which is what keeps interruption
truncation correct: the adapter rebuilds the graph input from that context
each turn, so an interrupted, never-heard response never lingers in memory.

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
    from langchain_openai import ChatOpenAI
    from langgraph.prebuilt import create_react_agent
    from livekit import agents
    from livekit.agents import Agent, AgentSession, room_io
    from livekit.agents.telemetry import set_tracer_provider
    from livekit.plugins import openai as lk_openai
    from livekit.plugins import silero
    from livekit.plugins.turn_detector.multilingual import MultilingualModel
    from opentelemetry import trace as otel_trace
    from opentelemetry.sdk.trace import TracerProvider

    from ..weather import fetch_weather
    from ._adapter import AITextOnlyLLMAdapter
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
        "You are a friendly voice assistant who can chat about anything and can "
        "look up the current weather for any city. When the user asks about "
        "weather in one or more places, call the lookup_weather tool — once per "
        "city — and summarize the results naturally in one or two short spoken "
        "sentences. Keep replies short, conversational, and free of formatting "
        "(no asterisks, no bullet points, no emoji)."
    )

    async def lookup_weather(city: str) -> dict:
        """Get the current weather for a single city.

        Call once per city for multi-city questions.

        Args:
            city: City name, e.g. 'San Francisco' or 'Tokyo'.
        """
        return await fetch_weather(city)

    class _Assistant(Agent):
        def __init__(self) -> None:
            # Instructions become the system message in the chat context, which
            # the LLMAdapter forwards into the graph each turn — so the system
            # prompt lives here, not in create_react_agent's `prompt=`.
            super().__init__(instructions=_INSTRUCTIONS)

    server = agents.AgentServer()

    @server.rtc_session()
    async def _entrypoint(ctx: agents.JobContext) -> None:
        # ctx.job.id is unique per dispatch; ctx.room.name is "console" in
        # console mode and would collide every session into one giant thread.
        set_active_thread_id(ctx.job.id)
        set_audio_file_path(ctx.session_directory / "audio.ogg")

        # A LangGraph ReAct agent is the "brain" plugged into LiveKit via the
        # langchain plugin's LLMAdapter. The graph is STATELESS — no
        # checkpointer, no thread_id — so LiveKit's ChatContext stays the
        # single source of truth. That matters for barge-in: when the user
        # interrupts, LiveKit truncates the assistant turn to what was actually
        # spoken, and the adapter rebuilds the graph input from that truncated
        # context on the next turn (no stale, never-heard output lingers).
        graph = create_react_agent(
            ChatOpenAI(model="gpt-4o-mini"),
            tools=[lookup_weather],
        )

        session = AgentSession(
            stt=lk_openai.STT(model="gpt-4o-mini-transcribe"),
            # Custom adapter: speaks only the model's own text, not the raw
            # JSON a tool returns (the stock LLMAdapter would speak both).
            llm=AITextOnlyLLMAdapter(graph=graph),
            tts=lk_openai.TTS(model="tts-1", voice="alloy"),
            vad=silero.VAD.load(),
            turn_detection=MultilingualModel(),
        )

        await session.start(
            room=ctx.room,
            agent=_Assistant(),
            room_options=room_io.RoomOptions(),
            # Enables the audio writer; in console mode it also needs the
            # `--record` CLI flag, which we force into argv below. The processor
            # reads the resulting audio.ogg and attaches it to the root span.
            record={"audio": True},
        )

    # LiveKit's CLI owns argv parsing from here. We force `console` so the user
    # gets a local mic+speaker session without having to remember the subcommand.
    # `--record` is required in console mode for LiveKit to actually write the
    # audio file (the `record={"audio": True}` dict alone only takes effect in
    # dev/start modes); without it the processor has no audio.ogg to attach to
    # the root span.
    sys.argv = [sys.argv[0], "console", "--record"]
    agents.cli.run_app(server)

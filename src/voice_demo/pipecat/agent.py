"""Pipecat voice agent, traced via OpenTelemetry into LangSmith.

Why OTEL and not SDK: Pipecat runs the whole STT → LLM → TTS pipeline
in-process and emits OTel spans for it natively (a `conversation` root, a
`turn` span per exchange, and `stt` / `llm` / `tts` child spans), gated behind
`PipelineTask(enable_tracing=True, enable_turn_tracking=True)`. The framework
does the hard work; we just register the `LangSmithSpanProcessor` (in
`processor.py`) that rewrites those spans into the `gen_ai.*` / `langsmith.*`
namespaces LangSmith keys off, and attach the recorded audio.

This mirrors the official LangChain × Pipecat tracing demo
(github.com/langchain-ai/voice-agents-tracing), adapted to this repo: OpenAI STT
instead of local Whisper, and the same weather-assistant + `lookup_weather` tool
as the other backends (so tool calls show up in the trace).

## Interruption handling

Barge-in is a first-class part of the pipeline, not an afterthought:

  * `SileroVADAnalyzer` on the transport input detects when the user starts
    speaking; Pipecat's turn strategies then immediately cut off the bot's
    in-flight TTS — it truncates the spoken output and discards the rest. (This
    is on by default once a VAD analyzer is present, which is why we don't set
    the deprecated `allow_interruptions` flag.)
  * `register_function(..., cancel_on_interruption=True)` cancels an in-flight
    tool call if the user interrupts before it returns.
  * The turn tracker (`enable_turn_tracking=True`) records `was_interrupted` on
    each turn; the span processor surfaces that onto the LangSmith `turn` span,
    and we also log it here. The per-turn audio recorder captures exactly what
    was spoken before the cut, so an interrupted turn's audio reflects reality.

Audio I/O is owned by Pipecat's `LocalAudioTransport` (the framework's console
transport), which is why this backend doesn't use `voice_demo.audio` — the same
reason the LiveKit backend doesn't.
"""

from __future__ import annotations

import os
import sys
import uuid
from datetime import datetime
from pathlib import Path


def _build_weather_tools():
    """Define the `lookup_weather` tool schema + async handler.

    Returns `(tools_schema, handler)`. The handler calls the shared Open-Meteo
    fetch and feeds the result back into the conversation via the result
    callback so the model can summarize it in its next spoken turn.
    """
    from pipecat.adapters.schemas.function_schema import FunctionSchema
    from pipecat.adapters.schemas.tools_schema import ToolsSchema
    from pipecat.services.llm_service import FunctionCallParams

    from ..weather import fetch_weather

    weather_fn = FunctionSchema(
        name="lookup_weather",
        description=(
            "Get the current weather for a single city. "
            "Call once per city for multi-city questions."
        ),
        properties={
            "city": {
                "type": "string",
                "description": "City name, e.g. 'San Francisco' or 'Tokyo'.",
            },
        },
        required=["city"],
    )

    async def handle_lookup_weather(params: FunctionCallParams) -> None:
        city = (params.arguments.get("city") or "").strip()
        result = await fetch_weather(city) if city else {"error": "missing city"}
        await params.result_callback(result)

    return ToolsSchema(standard_tools=[weather_fn]), handle_lookup_weather


SYSTEM_PROMPT = (
    "You are a friendly voice assistant who can look up the weather for any "
    "city. Keep replies short, conversational, and free of formatting (no "
    "asterisks, no bullet points, no emoji). When the user asks about weather "
    "in one or more places, call the lookup_weather tool — once per city — and "
    "then summarize the results naturally in one or two short spoken sentences."
)

LLM_MODEL = os.getenv("PIPECAT_LLM_MODEL", "gpt-4o-mini")
STT_MODEL = os.getenv("PIPECAT_STT_MODEL", "gpt-4o-mini-transcribe")
TTS_VOICE = os.getenv("PIPECAT_TTS_VOICE", "alloy")


async def run(project_name: str) -> None:
    """Build and run the Pipecat pipeline with LangSmith tracing.

    Blocks until Ctrl-C. `project_name` is informational here — the trace's
    LangSmith project is set by the OTLP headers wired in `tracing.configure`.
    """
    if not os.environ.get("OPENAI_API_KEY"):
        print(
            "OPENAI_API_KEY is not set (Pipecat's STT/LLM/TTS need it).",
            file=sys.stderr,
        )
        sys.exit(1)

    from loguru import logger
    from pipecat.audio.vad.silero import SileroVADAnalyzer
    from pipecat.frames.frames import LLMRunFrame
    from pipecat.pipeline.pipeline import Pipeline
    from pipecat.pipeline.runner import PipelineRunner
    from pipecat.pipeline.task import PipelineParams, PipelineTask
    from pipecat.processors.aggregators.llm_context import LLMContext
    from pipecat.processors.aggregators.llm_response_universal import (
        LLMContextAggregatorPair,
        LLMUserAggregatorParams,
    )
    from pipecat.services.openai.llm import OpenAILLMService
    from pipecat.services.openai.stt import OpenAISTTService
    from pipecat.services.openai.tts import OpenAITTSService
    from pipecat.transports.local.audio import (
        LocalAudioTransport,
        LocalAudioTransportParams,
    )

    from .audio_recorder import AudioRecorder
    from .processor import setup_langsmith_tracing
    from .turn_audio_recorder import TurnAudioRecorder

    conversation_id = str(uuid.uuid4())

    # Wire the OTel → LangSmith span processor. tracing.configure() has already
    # set OTEL_EXPORTER_OTLP_ENDPOINT / _HEADERS (only when a LangSmith key is
    # present), so this no-ops gracefully into an un-exported provider otherwise.
    span_processor = setup_langsmith_tracing() if os.environ.get("LANGSMITH_API_KEY") else None
    if span_processor is not None:
        logger.info(f"[pipecat] LangSmith tracing enabled · project={project_name}")
    else:
        logger.info("[pipecat] LANGSMITH_API_KEY not set — running without tracing.")
    logger.info(f"[pipecat] conversation_id={conversation_id}")

    # Per-conversation recording directory.
    recordings_dir = Path.cwd() / "pipecat-recordings"
    recordings_dir.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    recording_path = recordings_dir / f"conversation_{timestamp}.wav"

    transport = LocalAudioTransport(
        LocalAudioTransportParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
        )
    )

    stt = OpenAISTTService(settings=OpenAISTTService.Settings(model=STT_MODEL))
    llm = OpenAILLMService(
        settings=OpenAILLMService.Settings(
            model=LLM_MODEL, system_instruction=SYSTEM_PROMPT
        )
    )
    tts = OpenAITTSService(settings=OpenAITTSService.Settings(voice=TTS_VOICE))

    tools, handle_lookup_weather = _build_weather_tools()
    # cancel_on_interruption: if the user barges in before the lookup returns,
    # drop the in-flight call rather than speaking a now-stale result.
    llm.register_function(
        "lookup_weather", handle_lookup_weather, cancel_on_interruption=True
    )

    # Universal LLM context (pipecat ≥1.x). Tools live on the context; the
    # aggregator pair feeds user/assistant turns in and out of it. The VAD
    # analyzer goes on the USER aggregator (not the transport) — that's what
    # delimits the user's turn so STT commits a segment and the LLM runs.
    context = LLMContext(tools=tools)
    context_aggregator = LLMContextAggregatorPair(
        context,
        user_params=LLMUserAggregatorParams(vad_analyzer=SileroVADAnalyzer()),
    )

    audio_recorder = AudioRecorder(str(recording_path))
    turn_audio_recorder = TurnAudioRecorder(
        span_processor=span_processor,
        conversation_id=conversation_id,
        recordings_dir=recordings_dir,
    )

    if span_processor is not None:
        span_processor.register_recording(
            conversation_id, str(recording_path), audio_recorder=audio_recorder
        )
        span_processor.register_turn_audio_recorder(
            conversation_id, turn_audio_recorder
        )

    pipeline = Pipeline(
        [
            transport.input(),  # mic in
            stt,  # speech → text
            context_aggregator.user(),  # add user msg to context
            llm,  # text → response (+ tool calls)
            tts,  # response → speech
            audio_recorder,  # whole-conversation WAV
            turn_audio_recorder,  # per-turn WAV snippets
            transport.output(),  # speaker out
            context_aggregator.assistant(),  # add assistant msg to context
        ]
    )

    task = PipelineTask(
        pipeline,
        params=PipelineParams(enable_metrics=True),
        enable_tracing=True,
        enable_turn_tracking=True,  # required for tracing; records was_interrupted
        conversation_id=conversation_id,
    )

    # Connect the turn tracker so per-turn audio is captured, and log barge-ins.
    turn_tracker = task.turn_tracking_observer
    if turn_tracker is not None:
        turn_audio_recorder.connect_to_turn_tracker(turn_tracker)

        async def _on_turn_ended(_observer, turn_number, duration, was_interrupted):
            if was_interrupted:
                logger.info(
                    f"[pipecat] turn {turn_number} interrupted "
                    f"(barge-in) after {duration:.1f}s"
                )

        turn_tracker.add_event_handler("on_turn_ended", _on_turn_ended)
    else:
        logger.warning("[pipecat] no turn tracker — per-turn audio disabled")

    # Kick off with a short spoken greeting so the bot speaks first — this also
    # makes the opening turn a real (completing) turn rather than an idle one.
    context.add_message(
        {"role": "developer", "content": "Briefly greet the user and offer to check the weather."}
    )
    await task.queue_frames([LLMRunFrame()])

    logger.info("[pipecat] connected — say something (Ctrl-C to quit).")
    runner = PipelineRunner(
        handle_sigint=False if sys.platform == "win32" else True
    )
    try:
        await runner.run(task)
    finally:
        # Save before the conversation span finalizes (the processor attaches
        # this file when the `conversation` span ends).
        audio_recorder.save_recording()
        logger.info(f"[pipecat] recording saved to {recording_path}")

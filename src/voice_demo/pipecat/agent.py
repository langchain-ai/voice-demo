"""Pipecat voice agent, traced via OpenTelemetry into LangSmith.

Why OTEL and not SDK: Pipecat runs the whole STT → LLM → TTS pipeline
in-process and emits OTel spans for it natively (a `conversation` root, a
`turn` span per exchange, and `stt` / `llm` / `tts` child spans), gated behind
`PipelineTask(enable_tracing=True, enable_turn_tracking=True)`. The framework
does the hard work; we just register the `LangSmithSpanProcessor` (in
`processor.py`) that rewrites those spans into the `gen_ai.*` / `langsmith.*`
namespaces LangSmith keys off, and attach the recorded audio.

The "brain" is an in-process **LangGraph** graph (see `graph.py`) plugged in via
`LangGraphLLMService` (see `langgraph_llm_service.py`) — not Pipecat's stock
`OpenAILLMService`. Because that service subclasses `OpenAILLMService` and runs
the graph inside Pipecat's `@traced_llm` `llm` span, every graph node (a content
guardrail, the model, tool calls) nests as a **subspan of the `llm` span** when
`LANGSMITH_TRACING_MODE=otel` is set — so you can watch steps that are traced but
never spoken (the guardrail) and fully customize the graph. This mirrors the
LiveKit backend's "LangGraph brain + OTel" design (and the official LangChain ×
Pipecat tracing demo's span-translation approach in `processor.py`).

## Interruption handling

Barge-in is a first-class part of the pipeline, not an afterthought:

  * `SileroVADAnalyzer` on the user aggregator detects when the user starts
    speaking; Pipecat's turn strategies then immediately cut off the bot's
    in-flight TTS — it truncates the spoken output and discards the rest. (This
    is on by default once a VAD analyzer is present, which is why we don't set
    the deprecated `allow_interruptions` flag.)
  * The graph is stateless and Pipecat's `LLMContext` is the single source of
    truth, so an interrupted, never-heard turn never lingers in graph memory.
  * The turn tracker (`enable_turn_tracking=True`) records `was_interrupted` on
    each turn; the span processor surfaces that onto the LangSmith `turn` span,
    and we also log it here.

Recording captures *what was heard*, not what was generated: a
`RecordingLocalAudioTransport` (see `recording_transport.py`) taps the played
agent audio at the device-write boundary — reached only after barge-in
truncation — plus the user's mic, and writes one stereo WAV (L=user, R=agent)
via the shared `sdk_tracing.build_stereo_session_wav`, exactly as the OpenAI/ADK
backends do. (Per-turn audio snippets were dropped — full conversation only.)

Audio I/O is otherwise owned by Pipecat's `LocalAudioTransport` (the framework's
console transport), which is why this backend doesn't use `voice_demo.audio` —
the same reason the LiveKit backend doesn't.
"""

from __future__ import annotations

import os
import sys
import uuid
from datetime import datetime
from pathlib import Path


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
    from pipecat.services.openai.stt import OpenAISTTService
    from pipecat.services.openai.tts import OpenAITTSService
    from pipecat.transports.local.audio import LocalAudioTransportParams

    from .graph import build_graph
    from .langgraph_llm_service import LangGraphLLMService
    from .processor import setup_langsmith_tracing
    from .recording_transport import ConversationRecorder, RecordingLocalAudioTransport

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

    # Record at the device-write boundary so the WAV captures what was *heard*,
    # not what was generated: the recording transport taps played agent audio in
    # write_audio_frame (reached only after barge-in truncation) and user audio
    # off the input callback, then writes one stereo WAV via the shared
    # build_stereo_session_wav — the same artifact the OpenAI/ADK backends emit.
    recorder = ConversationRecorder(recording_path)
    transport = RecordingLocalAudioTransport(
        LocalAudioTransportParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
        ),
        recorder,
    )

    stt = OpenAISTTService(settings=OpenAISTTService.Settings(model=STT_MODEL))
    # The LLM stage is a LangGraph graph (guardrail → agent ⇄ tools). It runs
    # in-process inside Pipecat's `llm` span, so its nodes nest as subspans.
    # Tools live in the graph, not on the Pipecat context.
    llm = LangGraphLLMService(
        graph=build_graph(SYSTEM_PROMPT),
        settings=LangGraphLLMService.Settings(
            model=LLM_MODEL, system_instruction=SYSTEM_PROMPT
        ),
    )
    tts = OpenAITTSService(settings=OpenAITTSService.Settings(voice=TTS_VOICE))

    # Universal LLM context (pipecat ≥1.x). The aggregator pair feeds
    # user/assistant turns in and out of it. The VAD analyzer goes on the USER
    # aggregator (not the transport) — that's what delimits the user's turn so
    # STT commits a segment and the LLM runs.
    context = LLMContext()
    context_aggregator = LLMContextAggregatorPair(
        context,
        user_params=LLMUserAggregatorParams(vad_analyzer=SileroVADAnalyzer()),
    )

    # The span processor attaches this file to the conversation root span when
    # that span ends — it calls recorder.save_recording() first (duck-typed).
    if span_processor is not None:
        span_processor.register_recording(
            conversation_id, str(recording_path), audio_recorder=recorder
        )

    pipeline = Pipeline(
        [
            transport.input(),  # mic in (taps user audio for the recording)
            stt,  # speech → text
            context_aggregator.user(),  # add user msg to context
            llm,  # text → response (+ tool calls)
            tts,  # response → speech
            transport.output(),  # speaker out (taps played agent audio)
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

    # Log barge-ins (the per-turn audio recorder is gone; full-conversation only).
    turn_tracker = task.turn_tracking_observer
    if turn_tracker is not None:

        async def _on_turn_ended(_observer, turn_number, duration, was_interrupted):
            if was_interrupted:
                logger.info(
                    f"[pipecat] turn {turn_number} interrupted "
                    f"(barge-in) after {duration:.1f}s"
                )

        turn_tracker.add_event_handler("on_turn_ended", _on_turn_ended)

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
        # Idempotent — also called by the span processor when the conversation
        # span ends (so the file exists in time to be attached). This covers the
        # no-tracing path and any shutdown ordering.
        recorder.save_recording()
        logger.info(f"[pipecat] recording saved to {recording_path}")

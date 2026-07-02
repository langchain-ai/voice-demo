"""Pipecat voice agent, traced via OpenTelemetry into LangSmith.

Why OTEL and not SDK: Pipecat runs the whole STT → LLM → TTS pipeline
in-process and emits OTel spans for it natively (a `conversation` root, a
`turn` span per exchange, and `stt` / `llm` / `tts` child spans), gated behind
`PipelineTask(enable_tracing=True, enable_turn_tracking=True)`. The framework
does the hard work; we just call `configure_pipecat` (the LangSmith voice
integration, `langsmith.integrations.pipecat`), which installs a span processor
that rewrites those spans into the `gen_ai.*` / `langsmith.*` namespaces
LangSmith keys off, and attaches the recorded audio.

The "brain" is an in-process **LangGraph** graph (see `graph.py`) plugged in via
`LangGraphLLMService` (see `langgraph_llm_service.py`) — not Pipecat's stock
`OpenAILLMService`. Because that service subclasses `OpenAILLMService` and runs
the graph inside Pipecat's `@traced_llm` `llm` span, every graph node (the model,
tool calls) nests as a **subspan of the `llm` span** when
`LANGSMITH_TRACING_MODE=otel` is set — so you can watch the tool-deciding turn
and tool execution (traced but never spoken) and fully customize the graph.

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
    each turn; the span processor surfaces that onto the LangSmith `turn` span.

## Recording

Audio is recorded with Pipecat's own `AudioBufferProcessor`, placed **after**
`transport.output()` so it captures the bot audio as actually played (reached
only after barge-in truncation) plus the user's mic — i.e. *what was heard*. We
wire it to the span processor with `attach_audio_buffer`, which accumulates the
merged stereo PCM (user left, bot right) and attaches one WAV to the
conversation root span when it ends. A non-zero `buffer_size` makes the buffer
stream periodically, so the audio is accumulated before the root span exports.

Audio I/O is otherwise owned by Pipecat's `LocalAudioTransport` (the framework's
console transport), which is why this backend doesn't use `voice_demo.audio` —
the same reason the LiveKit backend doesn't.
"""

from __future__ import annotations

import os
import sys
import uuid

LLM_MODEL = os.getenv("PIPECAT_LLM_MODEL", "gpt-4o-mini")
STT_MODEL = os.getenv("PIPECAT_STT_MODEL", "gpt-4o-mini-transcribe")
TTS_VOICE = os.getenv("PIPECAT_TTS_VOICE", "alloy")

# Per-track bytes buffered before the AudioBufferProcessor emits an `on_audio_data`
# event. Non-zero so audio streams in periodically and is accumulated before the
# conversation span exports (the default 0 emits only once, at stop).
AUDIO_BUFFER_SIZE = 32_000


async def run(project_name: str) -> None:
    """Build and run the Pipecat pipeline with LangSmith tracing.

    Blocks until Ctrl-C. `project_name` is the LangSmith project the trace lands
    in (passed to `configure_pipecat`).
    """
    if not os.environ.get("OPENAI_API_KEY"):
        print(
            "OPENAI_API_KEY is not set (Pipecat's STT/LLM/TTS need it).",
            file=sys.stderr,
        )
        sys.exit(1)

    from loguru import logger
    from pipecat.audio.vad.silero import SileroVADAnalyzer
    from pipecat.frames.frames import TTSSpeakFrame
    from pipecat.pipeline.pipeline import Pipeline
    from pipecat.pipeline.runner import PipelineRunner
    from pipecat.pipeline.task import PipelineParams, PipelineTask
    from pipecat.processors.aggregators.llm_context import LLMContext
    from pipecat.processors.aggregators.llm_response_universal import (
        LLMContextAggregatorPair,
        LLMUserAggregatorParams,
    )
    from pipecat.processors.audio.audio_buffer_processor import AudioBufferProcessor
    from pipecat.services.openai.stt import OpenAISTTService
    from pipecat.services.openai.tts import OpenAITTSService
    from pipecat.transports.local.audio import (
        LocalAudioTransport,
        LocalAudioTransportParams,
    )

    from .graph import GREETING, SYSTEM_PROMPT, build_graph
    from langsmith.integrations.pipecat import configure_pipecat, set_thread_id
    from .langgraph_llm_service import LangGraphLLMService

    conversation_id = str(uuid.uuid4())
    # Bind the thread id in this conversation's context. The span processor reads
    # it per span via its default provider (a ContextVar), so this scales to
    # concurrent conversations: each task running run() sees its own id, where a
    # shared closure would not.
    set_thread_id(conversation_id)

    # Wire the OTel → LangSmith span processor. configure_pipecat resolves the
    # LangSmith exporter config from the standard LANGSMITH_* env, so it no-ops
    # gracefully (returns None) when no API key is present.
    # llm_span_kind="chain" is paired with LangGraphLLMService below: Pipecat's
    # `llm` span only orchestrates the graph; the nested model runs are the real
    # inference. With a stock LLM service, keep the default "llm".
    #
    # Two ways to install the provider: configure_pipecat() installs an OTel
    # TracerProvider + the LangSmith processor for you (used here). To use your
    # own provider, skip it and add PipecatLangSmithSpanProcessor(...) to that
    # provider directly.
    span_processor = (
        configure_pipecat(
            llm_span_kind="chain",
            # thread_id comes from set_thread_id(conversation_id) above, which
            # groups all spans into a LangSmith thread.
            project=project_name,
            service_name="voice-demo-pipecat",
        )
        if os.environ.get("LANGSMITH_API_KEY")
        else None
    )
    if span_processor is not None:
        logger.info(f"[pipecat] LangSmith tracing enabled · project={project_name}")
    else:
        logger.info("[pipecat] LANGSMITH_API_KEY not set — running without tracing.")
    logger.info(f"[pipecat] conversation_id={conversation_id}")

    transport = LocalAudioTransport(
        LocalAudioTransportParams(audio_in_enabled=True, audio_out_enabled=True)
    )

    stt = OpenAISTTService(settings=OpenAISTTService.Settings(model=STT_MODEL))
    # The LLM stage is a LangGraph graph (weather agent ⇄ tools). It runs
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

    # Pipecat's official recorder: stereo (user left, bot right). Placed after
    # transport.output() so it taps the bot audio as actually played
    # (post-barge-in-truncation) plus the user's mic. The span processor reads
    # its `on_audio_data` events and attaches the WAV to the root span.
    audiobuffer = AudioBufferProcessor(num_channels=2, buffer_size=AUDIO_BUFFER_SIZE)
    if span_processor is not None:
        span_processor.attach_audio_buffer(audiobuffer, conversation_id)

    pipeline = Pipeline(
        [
            transport.input(),  # mic in
            stt,  # speech → text
            context_aggregator.user(),  # add user msg to context
            llm,  # text → response
            tts,  # response → speech
            transport.output(),  # speaker out
            audiobuffer,  # record what was heard (after output)
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

    # Speak first with an explicit, fixed greeting (no LLM call): a TTSSpeakFrame
    # streams the text straight to TTS, and append_to_context=True records it as
    # the assistant's opening message so the model has context on the next turn.
    # Mirrors the LiveKit backend's session.say(GREETING). It adds no `llm` span,
    # so the conversation root aggregates from the first real user turn onward.
    await task.queue_frames([TTSSpeakFrame(text=GREETING, append_to_context=True)])

    # Begin recording. start_recording() only flips a flag and resets buffers (no
    # sample-rate dependency); the StartFrame sets the rate before any audio flows.
    await audiobuffer.start_recording()

    logger.info("[pipecat] connected — say something (Ctrl-C to quit).")
    runner = PipelineRunner(handle_sigint=False if sys.platform == "win32" else True)
    await runner.run(task)

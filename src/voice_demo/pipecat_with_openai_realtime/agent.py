"""Pipecat voice agent with an OpenAI Realtime (S2S) LLM stage, traced into LangSmith.

This is the `pipecat` cascade backend with its STT → LLM → TTS trio replaced by
a single speech-to-speech model. `OpenAIRealtimeLLMService` speaks WebSocket to
OpenAI's Realtime API: it takes the user's mic audio in and streams the bot's
spoken audio out, so there is no separate `OpenAISTTService` / `OpenAITTSService`
stage and no `stt` / `tts` span — just the one `llm` span for the realtime model.
For the fully-commented tracing/recording rationale shared with this file, read
`pipecat/agent.py`; only the realtime-specific differences are called out here.

Why OTEL and not SDK: unchanged from the cascade. Pipecat emits OTel spans for
the pipeline natively (a `conversation` root, a `turn` per exchange, and an `llm`
child), gated behind `PipelineWorker(enable_tracing=True,
enable_turn_tracking=True)`. `configure_pipecat` (the LangSmith Pipecat
integration) installs the span processor that rewrites those into the
`gen_ai.*` / `langsmith.*` namespaces and attaches the recorded audio. The `llm`
span here is a real OpenAI inference span (the realtime model *is* the
inference), so we keep `configure_pipecat`'s default `llm_span_kind="llm"` —
unlike the LangGraph variant, which overrides it to `"chain"`.

## Turn detection (no local VAD)

The realtime service runs on OpenAI's *server-side* VAD: it emits
`UserStartedSpeakingFrame` / `UserStoppedSpeakingFrame` from the model's own
turn-detection events, which also drive barge-in. So, unlike the cascade, we
attach no `SileroVADAnalyzer` — a second local VAD would double-broadcast
user-turn frames. We pair the aggregators with `realtime_service_mode=True` so
context writes are driven by the content stream (transcripts,
`LLMFullResponseStartFrame`) instead of those turn frames, which is how the
universal aggregator supports a continuously-running S2S model.

## Tools

Identical to the cascade: `lookup_weather` is a `FunctionSchema` that carries its
own handler, so advertising it on `LLMContext(tools=[...])` auto-registers it.
Pipecat owns the tool loop — the realtime model requests the call, Pipecat runs
the handler and feeds the result back over the WebSocket, and the call/result
land in the context so the model sees its own prior tool usage on later turns.

## Greeting

Realtime models have no TTS stage to synthesize a fixed string, so the cascade's
`TTSSpeakFrame(GREETING)` trick doesn't apply (mirroring the LiveKit realtime
backend, where `supports_say` is False). Instead the seeded system message asks
the model to open with the exact greeting, and we kick the first turn with an
`LLMRunFrame`: the aggregator pushes the initial context, and the service
auto-creates the opening response — the model speaks the greeting itself.
"""

from __future__ import annotations

import os
import sys
import uuid

# Connection-level realtime model (set on the WebSocket URL; can't change mid-session)
# and the voice the model speaks in. "marin" matches the LiveKit realtime backend.
REALTIME_MODEL = os.getenv("PIPECAT_REALTIME_MODEL", "gpt-realtime")
REALTIME_VOICE = os.getenv("PIPECAT_REALTIME_VOICE", "marin")

# Per-track bytes buffered before the AudioBufferProcessor emits `on_audio_data`.
# Non-zero so audio streams in and is accumulated before the conversation span
# exports (the default 0 emits only once, at stop). See pipecat/agent.py.
AUDIO_BUFFER_SIZE = 32_000


async def run(project_name: str) -> None:
    """Build and run the Pipecat + OpenAI Realtime pipeline with LangSmith tracing.

    Blocks until Ctrl-C. `project_name` is the LangSmith project the trace lands
    in (passed to `configure_pipecat`).
    """
    if not os.environ.get("OPENAI_API_KEY"):
        print(
            "OPENAI_API_KEY is not set (the OpenAI Realtime model needs it).",
            file=sys.stderr,
        )
        sys.exit(1)

    from loguru import logger
    from pipecat.adapters.schemas.function_schema import FunctionSchema
    from pipecat.frames.frames import LLMRunFrame
    from pipecat.pipeline.pipeline import Pipeline
    from pipecat.pipeline.worker import PipelineParams, PipelineWorker
    from pipecat.processors.aggregators.llm_context import LLMContext
    from pipecat.processors.aggregators.llm_response_universal import (
        LLMContextAggregatorPair,
    )
    from pipecat.processors.audio.audio_buffer_processor import AudioBufferProcessor
    from pipecat.services.llm_service import FunctionCallParams
    from pipecat.services.openai.realtime.events import (
        AudioConfiguration,
        AudioOutput,
        SessionProperties,
    )
    from pipecat.services.openai.realtime.llm import OpenAIRealtimeLLMService
    from pipecat.transports.local.audio import (
        LocalAudioTransport,
        LocalAudioTransportParams,
    )
    from pipecat.workers.runner import WorkerRunner

    from ..prompts import GREETING, SYSTEM_PROMPT
    from ..weather import fetch_weather
    from langsmith.integrations.pipecat import configure_pipecat, set_thread_id

    conversation_id = str(uuid.uuid4())
    # Bind the thread id in this conversation's context (a ContextVar the span
    # processor reads per span). See pipecat/agent.py for why this scales to
    # concurrent conversations where a shared closure would not.
    set_thread_id(conversation_id)

    # Wire the OTel → LangSmith span processor. Same as the cascade: keep the
    # default llm_span_kind="llm" — here the `llm` span is a real OpenAI Realtime
    # inference call. configure_pipecat resolves the LangSmith exporter config
    # from the standard LANGSMITH_* env, so we only call it when a key is present.
    span_processor = (
        configure_pipecat(
            project=project_name,
            service_name="voice-demo-pipecat-openai-realtime",
        )
        if os.environ.get("LANGSMITH_API_KEY")
        else None
    )
    if span_processor is not None:
        logger.info(
            f"[pipecat-openai-realtime] LangSmith tracing enabled · project={project_name}"
        )
    else:
        logger.info(
            "[pipecat-openai-realtime] LANGSMITH_API_KEY not set — running without tracing."
        )
    logger.info(f"[pipecat-openai-realtime] conversation_id={conversation_id}")

    transport = LocalAudioTransport(
        LocalAudioTransportParams(audio_in_enabled=True, audio_out_enabled=True)
    )

    # The single speech-to-speech service that fills the STT+LLM+TTS slot. The
    # voice is set via the session's audio-output config; the system instructions
    # and tools come from the LLMContext below (the service reads the context's
    # system message as its session instructions). Server-side turn detection is
    # on by default, which is what emits the user-speaking frames and handles
    # barge-in — so we add no local VAD.
    llm = OpenAIRealtimeLLMService(
        api_key=os.environ["OPENAI_API_KEY"],
        settings=OpenAIRealtimeLLMService.Settings(
            model=REALTIME_MODEL,
            session_properties=SessionProperties(
                audio=AudioConfiguration(output=AudioOutput(voice=REALTIME_VOICE)),
            ),
        ),
    )

    # A native Pipecat function tool — identical to the cascade. The handler
    # receives FunctionCallParams and returns via result_callback; because the
    # schema carries the handler, advertising it on the LLMContext auto-registers
    # it and Pipecat runs the tool loop over the realtime WebSocket.
    async def lookup_weather(params: FunctionCallParams) -> None:
        weather = await fetch_weather(params.arguments["city"])
        await params.result_callback(weather)

    weather_tool = FunctionSchema(
        name="lookup_weather",
        description="Get the current weather for a single city. Call once per city.",
        properties={
            "city": {"type": "string", "description": "City name, e.g. 'Paris'."}
        },
        required=["city"],
        handler=lookup_weather,
    )

    # Seed the context with the shared system prompt plus a one-time directive to
    # open with the exact greeting (see the "Greeting" note in the module
    # docstring — a realtime model has no TTS stage to speak a fixed string, so
    # it says the opener itself). The service reads this system message as the
    # session instructions.
    context = LLMContext(
        messages=[
            {
                "role": "system",
                "content": (
                    f"{SYSTEM_PROMPT}\n\n"
                    f'To begin, greet the user by saying exactly: "{GREETING}"'
                ),
            }
        ],
        tools=[weather_tool],
    )
    # realtime_service_mode=True: context writes follow the content stream, not
    # turn frames, and no VAD analyzer on the user aggregator (the realtime
    # service supplies the user-turn frames from server-side VAD).
    context_aggregator = LLMContextAggregatorPair(
        context, realtime_service_mode=True
    )

    # Pipecat's official recorder: stereo (user left, bot right), placed after
    # transport.output() so it taps the bot audio as actually played plus the
    # user's mic. The span processor reads its `on_audio_data` events and attaches
    # the WAV to the root span. See pipecat/agent.py.
    audiobuffer = AudioBufferProcessor(num_channels=2, buffer_size=AUDIO_BUFFER_SIZE)
    if span_processor is not None:
        span_processor.attach_audio_buffer(audiobuffer, conversation_id)

    # No stt/tts stages: the realtime `llm` sits directly between the transport's
    # mic in and speaker out.
    pipeline = Pipeline(
        [
            transport.input(),  # mic in
            context_aggregator.user(),  # add user msg to context
            llm,  # speech in → speech out (+ tool calls)
            transport.output(),  # speaker out
            audiobuffer,  # record what was heard (after output)
            context_aggregator.assistant(),  # add assistant msg to context
        ]
    )

    worker = PipelineWorker(
        pipeline,
        params=PipelineParams(enable_metrics=True),
        enable_tracing=True,
        enable_turn_tracking=True,  # required for tracing; records was_interrupted
        conversation_id=conversation_id,
    )

    # Begin recording before any audio flows. start_recording() only flips a flag
    # and resets buffers; the StartFrame sets the sample rate.
    await audiobuffer.start_recording()

    # Kick the opening turn: the aggregator pushes the initial context, and the
    # realtime service auto-creates a response — the model speaks the greeting
    # seeded above. (There's no TTSSpeakFrame here; a realtime model has no TTS
    # stage to synthesize a fixed string.)
    await worker.queue_frames([LLMRunFrame()])

    logger.info("[pipecat-openai-realtime] connected — say something (Ctrl-C to quit).")
    runner = WorkerRunner(handle_sigint=False if sys.platform == "win32" else True)
    await runner.add_workers(worker)
    await runner.run()

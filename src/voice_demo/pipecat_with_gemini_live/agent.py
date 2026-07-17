"""Pipecat voice agent with a Gemini Live (S2S) LLM stage, traced into LangSmith.

This is the `pipecat` cascade backend with its STT → LLM → TTS trio replaced by
a single speech-to-speech model. `GeminiLiveLLMService` speaks WebSocket to
Google's Gemini Live API: it takes the user's mic audio in and streams the bot's
spoken audio out, so there is no separate STT / TTS stage — just the one `llm`
span for the Gemini Live model. It is the Gemini twin of
`pipecat_with_openai_realtime/agent.py`; read that file and `pipecat/agent.py`
first — only the Gemini-specific differences are called out here.

Why OTEL and not SDK: unchanged from the cascade — Pipecat emits the OTel spans
(`conversation` root, a `turn` per exchange, an `llm` child) and
`configure_pipecat` (the LangSmith Pipecat integration) rewrites them into the
`gen_ai.*` / `langsmith.*` shape. The `llm` span here is a real Gemini inference
span, so we keep the default `llm_span_kind="llm"`.

## Turn detection: LOCAL VAD, unlike the OpenAI Realtime twin

This is *the* difference from `pipecat_with_openai_realtime/`. OpenAI Realtime
emits `UserStartedSpeakingFrame` / `UserStoppedSpeakingFrame` from its
server-side VAD, so that backend needs no local VAD. **Gemini Live does not** —
its API exposes an `interrupted` event but no turn-start/-end signals. Pipecat's
`TurnTrackingObserver` opens a turn on `UserStartedSpeakingFrame`, so with
Gemini's server-VAD-only setup there would be no user-turn frames, hence no
`turn` spans and no turn-based barge-in — gutting the very thing this repo is
here to show.

So we run Gemini Live in *locally-driven-turns* mode:

  * Disable Gemini's server VAD (`GeminiVADParams(disabled=True)`).
  * Attach a local `SileroVADAnalyzer` to the user aggregator. It emits the
    user-turn frames, which the `TurnTrackingObserver` turns into `turn` spans
    and which the service converts into Gemini `activity_start` / `activity_end`
    signals so the model knows when the user's turn begins and ends.

The trade-off (a fair teaching point): local turn boundaries are a VAD heuristic
and may not match what Gemini's own server-side VAD would have decided. We still
pair the aggregators with `realtime_service_mode=True` — Gemini Live is a
continuously-running S2S service, so context writes must follow the content
stream rather than the turn frames.

## Tools

Identical to the other Pipecat backends: `lookup_weather` is a `FunctionSchema`
carrying its own handler, so advertising it on `LLMContext(tools=[...])`
auto-registers it and Pipecat runs the tool loop over the Gemini WebSocket.

## Greeting

Like the OpenAI Realtime twin, there is no TTS stage to speak a fixed string, so
the seeded system message asks the model to open with the exact greeting.
`GeminiLiveLLMService` generates a response when its context is first set
(`inference_on_context_initialization=True`, the default), so kicking the first
turn with an `LLMRunFrame` makes the model speak the greeting itself.
"""

from __future__ import annotations

import os
import sys
import uuid

# The Gemini Live model (a native-audio S2S model) and the voice it speaks in.
# "Puck" matches the LiveKit Gemini backend. The default model tracks Pipecat's
# own default so it's known-valid for the installed pipecat version.
GEMINI_MODEL = os.getenv(
    "PIPECAT_GEMINI_MODEL", "models/gemini-2.5-flash-native-audio-preview-12-2025"
)
GEMINI_VOICE = os.getenv("PIPECAT_GEMINI_VOICE", "Puck")

# Per-track bytes buffered before the AudioBufferProcessor emits `on_audio_data`.
# Non-zero so audio streams in and is accumulated before the conversation span
# exports (the default 0 emits only once, at stop). See pipecat/agent.py.
AUDIO_BUFFER_SIZE = 32_000


async def run(project_name: str) -> None:
    """Build and run the Pipecat + Gemini Live pipeline with LangSmith tracing.

    Blocks until Ctrl-C. `project_name` is the LangSmith project the trace lands
    in (passed to `configure_pipecat`).
    """
    if not os.environ.get("GOOGLE_API_KEY"):
        print(
            "GOOGLE_API_KEY is not set (the Gemini Live model needs it).",
            file=sys.stderr,
        )
        sys.exit(1)

    from loguru import logger
    from pipecat.adapters.schemas.function_schema import FunctionSchema
    from pipecat.audio.vad.silero import SileroVADAnalyzer
    from pipecat.frames.frames import LLMRunFrame
    from pipecat.pipeline.pipeline import Pipeline
    from pipecat.pipeline.worker import PipelineParams, PipelineWorker
    from pipecat.processors.aggregators.llm_context import LLMContext
    from pipecat.processors.aggregators.llm_response_universal import (
        LLMContextAggregatorPair,
        LLMUserAggregatorParams,
    )
    from pipecat.processors.audio.audio_buffer_processor import AudioBufferProcessor
    from pipecat.services.google.gemini_live.llm import (
        GeminiLiveLLMService,
        GeminiModalities,
        GeminiVADParams,
    )
    from pipecat.services.llm_service import FunctionCallParams
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
    # default llm_span_kind="llm" — here the `llm` span is a real Gemini Live
    # inference call. configure_pipecat resolves the LangSmith exporter config
    # from the standard LANGSMITH_* env, so we only call it when a key is present.
    span_processor = (
        configure_pipecat(
            project=project_name,
            service_name="voice-demo-pipecat-gemini-live",
        )
        if os.environ.get("LANGSMITH_API_KEY")
        else None
    )
    if span_processor is not None:
        logger.info(
            f"[pipecat-gemini-live] LangSmith tracing enabled · project={project_name}"
        )
    else:
        logger.info(
            "[pipecat-gemini-live] LANGSMITH_API_KEY not set — running without tracing."
        )
    logger.info(f"[pipecat-gemini-live] conversation_id={conversation_id}")

    transport = LocalAudioTransport(
        LocalAudioTransportParams(audio_in_enabled=True, audio_out_enabled=True)
    )

    # The single speech-to-speech service that fills the STT+LLM+TTS slot. We
    # disable Gemini's server-side VAD (vad=disabled) and drive turns with a
    # local Silero VAD instead — see the "Turn detection" note in the module
    # docstring. The system instructions and tools come from the LLMContext below
    # (the service reads the context's system message as its instructions).
    llm = GeminiLiveLLMService(
        api_key=os.environ["GOOGLE_API_KEY"],
        settings=GeminiLiveLLMService.Settings(
            model=GEMINI_MODEL,
            voice=GEMINI_VOICE,
            modalities=GeminiModalities.AUDIO,
            vad=GeminiVADParams(disabled=True),
        ),
    )

    # A native Pipecat function tool — identical to the other Pipecat backends.
    # Because the schema carries the handler, advertising it on the LLMContext
    # auto-registers it and Pipecat runs the tool loop over the Gemini WebSocket.
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
    # open with the exact greeting (a realtime model has no TTS stage to speak a
    # fixed string, so it says the opener itself). The service reads this system
    # message as the session instructions.
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
    # Local VAD on the user aggregator supplies the user-turn frames (Gemini Live
    # doesn't emit them); realtime_service_mode=True keeps context writes driven
    # by the content stream, as required for a continuously-running S2S service.
    context_aggregator = LLMContextAggregatorPair(
        context,
        user_params=LLMUserAggregatorParams(vad_analyzer=SileroVADAnalyzer()),
        realtime_service_mode=True,
    )

    # Pipecat's official recorder: stereo (user left, bot right), placed after
    # transport.output() so it taps the bot audio as actually played plus the
    # user's mic. The span processor reads its `on_audio_data` events and attaches
    # the WAV to the root span. See pipecat/agent.py.
    audiobuffer = AudioBufferProcessor(num_channels=2, buffer_size=AUDIO_BUFFER_SIZE)
    if span_processor is not None:
        span_processor.attach_audio_buffer(audiobuffer, conversation_id)
        # Gemini Live delivers the user's finalized text through the user context
        # aggregator (on_user_turn_message_added), not an OTel span, so register
        # it here — otherwise the user turns never reach the trace. This folds
        # them onto the root and into each llm response's input history.
        span_processor.instrument_user_aggregator(context_aggregator, conversation_id)

    # No stt/tts stages: the Gemini Live `llm` sits directly between the
    # transport's mic in and speaker out.
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
    # Gemini Live service generates a response on first context
    # (inference_on_context_initialization=True) — the model speaks the greeting
    # seeded above.
    await worker.queue_frames([LLMRunFrame()])

    logger.info("[pipecat-gemini-live] connected — say something (Ctrl-C to quit).")
    runner = WorkerRunner(handle_sigint=False if sys.platform == "win32" else True)
    await runner.add_workers(worker)
    await runner.run()

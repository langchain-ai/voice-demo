# voice-demo

Four voice-agent backends — **OpenAI Realtime v2v**, **Google ADK Live multi-modal**, **LiveKit STT/LLM/TTS**, and **Pipecat STT/LLM/TTS** — each instrumented for LangSmith *the way its framework intends*.

The point isn't the agents (they're toy weather assistants). The point is **how each one is traced**, side-by-side — and how to structure a voice agent so the **frontend is swappable** from the agent itself.

## The tracing principle

> Use the **SDK** when you're consuming an event stream from a remote service. Use **OTEL** when the framework runs the pipeline in-process and emits its own spans.

| Backend | Tracing path | Why |
|---|---|---|
| OpenAI Realtime | SDK `RunTree`, one span per event | Realtime is a remote WebSocket: we observe an event stream (`input_audio_buffer.speech_started`, `response.created`, …) and build the trace ourselves. Every event becomes its own child span under the session root, carrying that event's full payload — so the trace mirrors *exactly what the server sent*. |
| Google ADK Live | SDK `RunTree`, one span per event | Same situation: `Runner.run_live` is a remote stream. ADK's OTel instrumentation covers its non-live paths but **doesn't emit spans for `run_live`** — going OTel-only produces an empty root span. We span each event we observe, in the same RunTree shape as OpenAI. |
| LiveKit | OTEL + processor that translates `lk.*` → `gen_ai.*` / `langsmith.*` | LiveKit runs the full STT/LLM/TTS/VAD pipeline in-process and emits OTel spans for every stage. Translating attribute names lets us inherit all of it — transcripts, per-stage latencies, model names, EOU probabilities, audio attachments — for free. |
| Pipecat | OTEL + processor that maps Pipecat spans → `gen_ai.*` / `langsmith.*` | Pipecat also runs in-process and emits OTel spans (`conversation` / `turn` / `stt` / `llm` / `tts`) behind `enable_tracing=True`. A span processor rewrites them into LangSmith's namespaces and attaches the whole-conversation audio. |

The folder layout reflects this: the two **OTEL** backends (LiveKit, Pipecat) each pair an `agent.py` with a `processor.py` that enriches OTel spans before export. The two **SDK** backends (OpenAI, ADK) have no processor — they build the trace via `voice_demo.sdk_tracing` directly.

## The frontend is swappable from the agent

Every backend is the same three things: an **agent brain** + **tracing** + a **transport** that moves audio in and out.

- **LiveKit** and **Pipecat** get their transport (and console UX) from their *own frameworks* — LiveKit's console mode, Pipecat's `LocalAudioTransport`. The swappable part is the brain + the OTel processor.
- **OpenAI** and **ADK** ship no console, so this repo provides one: `MicStream` / `SpeakerStream` (the "console transport") and a `ConsoleStatus` meter. But the agents don't depend on those classes — they depend on small protocols:

  | Protocol | Where | Console implementation |
  |---|---|---|
  | `AudioInput` / `AudioOutput` | `voice_demo/audio.py` | `MicStream` / `SpeakerStream` (sounddevice, PCM16) |
  | `StatusUI` | `voice_demo/console.py` | `ConsoleStatus` (or the no-op `NullUI`) |

  The agent's `run()` takes those by injection; `cli.py` (the frontend) constructs the console versions and passes them in. **To drive the same OpenAI Realtime or ADK Live agent from a web app, a phone call, or a websocket bridge, implement those three interfaces and inject your own — without touching the agent's event loop, tracing, or tool logic.**

## Layout

```
src/voice_demo/
├── cli.py                 # the frontend: arg parsing + builds & injects the console transport/UI
├── tracing.py             # shared LangSmith env wiring (SDK key / OTLP exporter vars)
├── sdk_tracing.py         # shared RunTree-per-event machinery (OpenAI + ADK)
├── audio.py               # AudioInput/AudioOutput protocols + MicStream/SpeakerStream (console transport)
├── console.py             # StatusUI protocol + ConsoleStatus / NullUI
├── weather.py             # shared Open-Meteo lookup
├── openai/
│   ├── agent.py           # Realtime event loop → RunTree span per event
│   ├── guardrail.py       # @traceable LangChain structured-output guardrail
│   └── tools.py           # @traceable Open-Meteo weather lookup
├── adk/
│   └── agent.py           # ADK Runner.run_live() driver → RunTree span per event
├── livekit/
│   ├── agent.py           # AgentSession in console mode + tracer setup (LangGraph brain)
│   ├── processor.py        # OTEL processor (lk.* → gen_ai.*, audio attachment, per-turn latency)
│   └── _thread_id.py      # ContextVar for stable per-session thread_id
└── pipecat/
    ├── agent.py           # Pipeline (OpenAI STT/TTS, Silero VAD, LangGraph LLM) + interruption
    ├── graph.py           # in-process LangGraph brain: guardrail → agent ⇄ tools
    ├── langgraph_llm_service.py  # runs the graph inside Pipecat's `llm` span (nests its subspans)
    ├── processor.py        # OTEL processor (Pipecat spans → gen_ai.*, audio attachment)
    └── recording_transport.py  # LocalAudioTransport that records what was *heard* (stereo WAV)
```

## Setup

Python 3.11+, [uv](https://docs.astral.sh/uv/).

```bash
cd voice-demo
uv sync --all-extras            # or just one: --extra openai / --extra adk / --extra livekit / --extra pipecat
cp .env.example .env            # then fill in keys
```

### Required env

- `LANGSMITH_API_KEY` — for tracing (optional, but the whole point of the demo)
- `OPENAI_API_KEY` — used by the OpenAI Realtime backend, the LangChain guardrail, LiveKit's STT/LLM/TTS plugins, *and* the Pipecat backend's STT/LLM/TTS
- `GOOGLE_API_KEY` — used by the ADK Live backend (Gemini)
- `LIVEKIT_API_KEY` / `LIVEKIT_API_SECRET` / `LIVEKIT_URL` — dummy values fine; LiveKit's SDK only validates them at startup

## Run

```bash
uv run voice-demo --backend openai
uv run voice-demo --backend adk
uv run voice-demo --backend livekit
uv run voice-demo --backend pipecat
```

All four open the local mic + speaker. For LiveKit, press the spacebar to talk (LiveKit's console UX). For OpenAI, ADK, and Pipecat, just start speaking — server/local VAD handles turn boundaries, and you can **barge in** to interrupt the agent mid-sentence.

Try:
- "What's the weather in San Francisco?"
- "How's the weather in Tokyo and Rome right now?"
- Interrupt the agent while it's talking — watch it stop and listen.
- (OpenAI & Pipecat) "What do you think about pineapple on pizza?" — guardrail trips, theatrical refusal. On Pipecat the guardrail is a non-spoken node you can watch as a subspan under the `llm` span.

## What you see in LangSmith

Each backend lands in its own project: `voice-demo-openai`, `voice-demo-adk`, `voice-demo-livekit`, `voice-demo-pipecat`.

**OpenAI** — one conversation is one trace. Every WebSocket event becomes its own span under the session root, in arrival order, carrying that event's full (scrubbed) payload — in `inputs` if the user sent it toward the model, in `outputs` if the model/server sent it back:
```
realtime_session                                 (root; one stereo conversation.wav: L=user, R=agent)
│   metadata: event_count, duration_s
├── input_audio_buffer.speech_started            (event)
├── input_audio_buffer.speech_stopped            (event)
├── conversation.item.input_audio_transcription.completed   (event)
│   └── guardrail                                (ran while handling this event)
├── response.created                             (event)
├── response.function_call_arguments.done        (event)
├── response.done                                (event)
│   └── lookup_weather × N                       (ran while handling this event)
├── response.done                                (event — post-tool follow-up)
└── error                                        (event, if the server sent one)
```
Raw audio (`response.output_audio.delta`) is played but not spanned, and every other streaming partial (`*.delta`) is dropped — the terminal `*.done` events carry the complete payload. The session root carries a single **stereo WAV** reconstructed from timestamped chunks (user on left, agent on right — interruption shows as overlap).

**ADK** — same shape as OpenAI (one conversation = one trace), spanning each `run_live` event:
```
realtime_session                    (root; one stereo conversation.wav: L=user, R=agent)
│   metadata: thread_id, model, event_count, duration_s
├── input_transcription            (event — user speech chunk)
├── output_transcription           (event — agent speech chunk)
├── function_call: get_weather     (event)
│   └── execute_tool: get_weather  (real tool child; finalized when the matching function_response arrives)
├── function_response: get_weather (event)
├── turn_complete                  (event)
└── interrupted                    (event)
```
Built explicitly with `RunTree` (the shared `voice_demo.sdk_tracing.EventSession`) because ADK's OTel instrumentation doesn't cover `Runner.run_live`. Events whose *only* payload is an audio chunk are played but not spanned; every recorded event has its audio bytes scrubbed to a `<N bytes>` placeholder.

**LiveKit**:
```
job_entrypoint                      (root, attaches the full conversation WAV)
└── agent_session
    └── agent_turn × N              (per-turn metadata: turn_e2e_latency, turn_llm_ttft, turn_tts_ttfb, turn_eou_*)
        ├── user_turn               (STT — chain kind, holds the user transcript)
        ├── llm_node                (LLM — the model response)
        └── tts_node                (TTS — voice + audio_duration)
```

**Pipecat** — the LLM stage is an in-process **LangGraph** graph, so its nodes nest as subspans *under* Pipecat's `llm` span (one trace). Steps that are traced but never spoken — like the content guardrail — show up too:
```
conversation                        (root; attaches the stereo what-was-heard WAV: L=user, R=agent)
└── turn × N                        (one exchange; carries turn.was_interrupted)
    ├── stt                         (audio → transcript)
    ├── llm                         (Pipecat's LLM span — LangGraphLLMService)
    │   └── LangGraph
    │       ├── guardrail           (structured-output check — traced, NOT spoken)
    │       ├── call_model          (ChatOpenAI; may emit tool calls)
    │       ├── tools: lookup_weather
    │       └── call_model          (final answer — spoken)
    └── tts                         (response text → audio)
```
This works because `LangGraphLLMService` subclasses Pipecat's `OpenAILLMService` and runs the graph inside its `@traced_llm` `llm` span (opened with `start_as_current_span`); with `LANGSMITH_TRACING_MODE=otel`, LangChain/LangGraph emit OTel spans through the shared provider and inherit that span as parent. It's the same "LangGraph brain + OTel" design as the LiveKit backend — only the final assistant text is pushed to TTS, everything else is traced but not voiced.

## Interruption handling

Barge-in (the user interrupting the agent mid-utterance) is handled by every realtime backend, and it's a first-class part of the demo:

- **OpenAI** — server VAD with `interrupt_response: true`; on `input_audio_buffer.speech_started` the agent flushes the speaker buffer (`AudioOutput.clear()`).
- **ADK** — Gemini emits an `interrupted` event; the agent flushes the speaker and (carefully) distinguishes a real interrupt from speaker bleed.
- **LiveKit** — the framework truncates the assistant turn to what was actually spoken; the LangGraph brain is kept stateless so an interrupted, never-heard response never lingers in the context.
- **Pipecat** — the Silero VAD drives barge-in (default turn-strategy behavior); tool calls use `cancel_on_interruption=True`; the turn tracker records `was_interrupted`, which the processor surfaces onto the LangSmith `turn` span.

### Recording what was *heard*, not what was generated

A realtime agent generates more audio than it plays — barge-in truncates the bot mid-sentence. All three local backends record at the **playout boundary**, so the recording reflects what each party actually heard:

- **OpenAI / ADK** — the agent side is recorded in `SpeakerStream`'s device callback (the bytes actually pulled to the speaker); audio that `clear()` drops on barge-in is never recorded.
- **Pipecat** — a `RecordingLocalAudioTransport` taps played agent audio in `write_audio_frame` (reached only *after* the output clock queue truncates on barge-in) plus the user's mic, and writes one stereo WAV via the same `sdk_tracing.build_stereo_session_wav` the SDK backends use.
- **LiveKit** — the framework's own media recorder already records the transmitted track post-truncation.

In production (telephony, web push-to-talk) the same principle applies: record at the transport/media layer (e.g. dual-channel call recording, or your SFU's recording egress), never from the model/TTS output.

## Notes

- Each backend lazily imports its framework, so a missing optional dep for one backend doesn't break the others.
- Model overrides via env: OpenAI `REALTIME_MODEL` (default `gpt-realtime-2`); ADK `ADK_LIVE_MODEL` (default `gemini-2.5-flash-native-audio-latest`); Pipecat `PIPECAT_LLM_MODEL` / `PIPECAT_STT_MODEL` / `PIPECAT_TTS_VOICE`.
- Event payloads are scrubbed before they reach LangSmith — raw audio `bytes` become a `<N bytes>` placeholder and long strings are truncated, so no audio blobs or oversized payloads are ever shipped.

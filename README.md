# voice-demo

Seven voice-agent backends — **OpenAI Realtime** (raw WebSocket), **OpenAI
Realtime via the Agents SDK**, **Google ADK Live (Gemini)**, **LiveKit
STT/LLM/TTS**, **Pipecat STT/LLM/TTS**, and **LiveKit speech-to-speech**
(OpenAI Realtime / Gemini Live) — each instrumented for LangSmith *the way its
framework intends*.

The point isn't the agents (they're toy weather assistants). The point is
**how each one is traced**, side-by-side — and how to structure a voice agent
so the **frontend is swappable** from the agent itself.

Each backend's `agent.py` and `processor.py` carry module docstrings that
document the framework's tracing model in depth — event/span taxonomies,
mapping decisions, and the reasoning behind each choice.

## The tracing principle

> Use the **SDK** when you're consuming an event stream from a remote service.
> Use **OTel** when the framework runs the pipeline in-process and emits its
> own spans.

| Backend | Tracing path | Why |
|---|---|---|
| OpenAI Realtime | SDK `RunTree`, one span per event | Realtime is a remote WebSocket with no telemetry of its own: we observe the event stream (`input_audio_buffer.speech_started`, `response.done`, …) and build the trace ourselves. Every non-noise event becomes its own child span under the session root, carrying that event's full payload — the trace mirrors *exactly what the server sent*. |
| OpenAI Realtime (Agents SDK) | SDK `RunTree`, one span per event | The same Realtime model, but driven through the OpenAI Agents SDK (`RealtimeAgent` / `RealtimeRunner`) — so the runner owns the turn/tool loop and we observe its **semantic** event stream (`agent_start`, `tool_start` / `tool_end`, `audio_interrupted`, `history_updated`) instead of the raw wire protocol. Same per-event span machinery; the low-level `raw_model_event` passthrough is dropped as noise. |
| Google ADK Live | SDK `RunTree`, one span per event | Same situation: `Runner.run_live` is a remote stream, and ADK's OTel instrumentation **doesn't cover the live path** — OTel-only setups produce one empty root span. We span each event we observe, in the same shape as the OpenAI backend. |
| LiveKit | OTel + a span processor (`lk.*` → `gen_ai.*` / `langsmith.*`) | LiveKit Agents runs the full STT/LLM/TTS/VAD pipeline in-process and emits OTel spans for every stage — but under a vendor prefix LangSmith doesn't read. The processor translates them, so transcripts, messages (with tool calls), token usage, per-stage latencies, and the call recording all land in LangSmith. |
| Pipecat | OTel + a span processor (Pipecat spans → `gen_ai.*` / `langsmith.*`) | Pipecat also runs in-process and emits OTel spans (`conversation` / `turn` / `stt` / `llm` / `tts`) behind `enable_tracing=True`. Its processor does the same translation job and attaches the whole-conversation audio to the root. |
| LiveKit speech-to-speech | OTel — reuses LiveKit's processor **unchanged** | Two variants (`livekit-openai-realtime`, `livekit-google-realtime`) swap the STT/LLM/TTS pipeline for one speech-to-speech model. LiveKit still emits the same span vocabulary (turns, tools, realtime metrics), so the same processor handles it with no changes. |

The folder layout reflects this: the **OTel** backends (LiveKit and its two
speech-to-speech variants, Pipecat) each pair an `agent.py` with a
`processor.py` that rewrites OTel spans before export. The **SDK** backends
(OpenAI, ADK) have no processor — they build the trace via
`voice_demo.sdk_tracing` directly.

## The frontend is swappable from the agent

Every backend is the same three things: an **agent brain** + **tracing** + a
**transport** that moves audio in and out.

- **LiveKit** and **Pipecat** get their transport (and console UX) from their
  *own frameworks* — LiveKit's console mode, Pipecat's `LocalAudioTransport`.
- **OpenAI** and **ADK** ship no console, so this repo provides one — but the
  agents don't depend on it. They depend on small protocols, injected by the
  CLI:

  | Protocol | Where | Console implementation |
  |---|---|---|
  | `AudioInput` / `AudioOutput` | `voice_demo/audio.py` | `MicStream` / `SpeakerStream` (sounddevice, PCM16) |
  | `StatusUI` | `voice_demo/console.py` | `ConsoleStatus` (or the no-op `NullUI`) |

  To drive the same OpenAI Realtime or ADK Live agent from a web app, a phone
  call, or a websocket bridge, implement those interfaces and inject your own —
  without touching the agent's event loop, tracing, or tool logic.

## Layout

```
src/voice_demo/
├── cli.py                 # the frontend: arg parsing + builds & injects the console transport/UI
├── tracing.py             # shared LangSmith env wiring (SDK vars / OTLP exporter vars)
├── sdk_tracing.py         # shared RunTree-per-event machinery (OpenAI, OpenAI Agents, ADK)
├── audio.py               # AudioInput/AudioOutput protocols + MicStream/SpeakerStream
├── console.py             # StatusUI protocol + ConsoleStatus / NullUI
├── prompts.py             # shared system prompt + greeting
├── weather.py             # shared Open-Meteo lookup (no API key)
├── graph.py               # LangGraph brain (weather agent ⇄ tools) — used by Pipecat
├── openai/
│   ├── agent.py           # raw Realtime WebSocket event loop → RunTree span per event
│   ├── utils.py           # event direction + tool dispatch
│   └── tools.py           # traceable weather lookup
├── openai_agents/
│   ├── agent.py           # Agents SDK (RealtimeRunner) → RunTree span per semantic event
│   ├── utils.py           # describe_event(): event → clean span payload + direction
│   └── tools.py           # weather lookup as an Agents @function_tool
├── adk/
│   ├── agent.py           # Runner.run_live() driver → RunTree span per event
│   ├── events.py          # LiveEvent: readable view over ADK's all-optional-fields events
│   └── utils.py           # event_context(): span-or-skip per event
├── livekit/
│   ├── agent.py           # LiveKit-native agent (cascade or realtime) + tracer wiring
│   └── processor.py       # OTel processor: lk.* + span events → gen_ai.*/langsmith.*; generic to any LiveKit agent
└── pipecat/
    ├── agent.py           # Pipeline (OpenAI STT/TTS, Silero VAD, LangGraph LLM) + interruption
    ├── langgraph_llm_service.py  # runs the graph inside Pipecat's `llm` span (nests its runs)
    ├── processor.py       # OTel processor: Pipecat spans → gen_ai.*/langsmith.*; audio attachment
    └── recording_transport.py    # LocalAudioTransport that records what was *heard* (stereo WAV)
```

Each backend's module docstrings are the deep-dive companion to its code.

## Setup

Python 3.11+, [uv](https://docs.astral.sh/uv/).

```bash
cd voice-demo
uv sync --all-extras            # or just one: --extra openai / --extra openai-agents / --extra adk / --extra livekit / --extra pipecat
cp .env.example .env            # then fill in keys
```

### Required env

- `LANGSMITH_API_KEY` — for tracing (optional, but the whole point of the demo)
- `OPENAI_API_KEY` — the two OpenAI Realtime backends (raw WebSocket + Agents
  SDK), LiveKit's STT/LLM/TTS plugins and its Realtime variant, and Pipecat's
  STT/TTS + LangGraph brain
- `GOOGLE_API_KEY` — the ADK Live backend and LiveKit's Gemini Live variant
- `LIVEKIT_API_KEY` / `LIVEKIT_API_SECRET` / `LIVEKIT_URL` — dummy values are
  fine; LiveKit's SDK only validates them at startup in console mode

## Run

```bash
uv run voice-demo --backend openai           # OpenAI Realtime, raw WebSocket
uv run voice-demo --backend openai-agents    # OpenAI Realtime, via the Agents SDK
uv run voice-demo --backend adk
uv run voice-demo --backend livekit
uv run voice-demo --backend pipecat
uv run voice-demo --backend livekit --llm openai-realtime   # LiveKit + OpenAI Realtime (S2S)
uv run voice-demo --backend livekit --llm google-realtime   # LiveKit + Gemini Live (S2S)
```

The speech-to-speech variants are the `livekit` backend with its LLM slot
swapped via `--llm` (the only backends are `openai`, `openai-agents`, `adk`,
`livekit`, `pipecat`); `--llm` applies to `livekit` only.

All of them open the local mic + speaker. For LiveKit (all three variants),
press the spacebar to talk (LiveKit's console UX). For OpenAI, ADK, and
Pipecat, just start speaking — server/local VAD handles turn boundaries, and
you can **barge in** to interrupt the agent mid-sentence.

Try:
- "What's the weather in Tokyo?" / "How's the weather in Rome and Berlin right now?"
- Interrupt the agent while it's talking — watch it stop and listen.
- Ask about several cities at once — watch the tool get called once per city
  in the trace.

## What you see in LangSmith

Each backend lands in its own project: `voice-demo-openai`,
`voice-demo-openai-agents`, `voice-demo-adk`, `voice-demo-livekit`,
`voice-demo-pipecat`, `voice-demo-livekit-openai-realtime`,
`voice-demo-livekit-google-realtime`.

**OpenAI** — one conversation = one trace; every non-noise WebSocket event is
its own span, in arrival order (payload in `inputs` for user→model events,
`outputs` for model→user). Streaming `*.delta` events are skipped (the `*.done`
event repeats the full payload), and agent audio is played but never spanned:

```
realtime_session                                 (root; stereo conversation.wav: L=user, R=agent)
├── input_audio_buffer.speech_started
├── input_audio_buffer.speech_stopped
├── conversation.item.input_audio_transcription.completed
├── response.created
├── response.done
│   └── lookup_weather × N                       (real tool runs — the client executes tools)
└── response.done                                (the spoken follow-up)
```

**OpenAI (Agents SDK)** — the same Realtime model and the same per-event span
machinery, but the runner owns the turn/tool loop, so the spans are the SDK's
**semantic** events rather than the wire protocol. The model's `function_call`
items never surface — the SDK runs the tool and reports `tool_start` /
`tool_end`. The low-level `raw_model_event` passthrough is dropped as noise:

```
realtime_session                                 (root; stereo conversation.wav: L=user, R=agent)
├── agent_start
├── history_added            (user transcript → span inputs)
├── tool_start               (lookup_weather — the SDK is about to run it)
├── tool_end                 (weather result → span outputs)
├── history_updated          (agent transcript)
└── audio_interrupted        (barge-in)
```

**ADK** — same shape (one span per `run_live` event), with span names derived
from which fields are populated (ADK events have no type tag). ADK executes
tools itself, so tool usage appears as the two observed events:

```
realtime_session                    (root; stereo conversation.wav)
├── input_transcription            (user speech → span inputs)
├── output_transcription           (agent speech → span outputs)
├── function_call: get_weather
├── function_response: get_weather
├── turn_complete
└── interrupted                    (barge-in)
```

**LiveKit** — the framework's own OTel spans, translated. The genuine
inference calls (`user_turn` STT, `llm_request`, `tts_request`) are `llm`-kind
runs with real I/O — `llm_request` carries the full message history including
structured tool calls, plus native token usage for cost tracking; the
pipeline wrappers are chains. The root holds the whole-conversation
transcript (released when the session ends — the entrypoint span itself ends
right after the greeting) and the call recording. A turn that uses a tool has
two model calls — that's the tool loop, not a bug:

```
<job entrypoint>                    (root; full transcript + audio.ogg)
└── agent_session
    └── agent_turn × N              (lk.user_input / lk.response.text + latency metadata)
        ├── user_turn               (STT — llm kind)
        ├── eou_detection           (turn detection — chain)
        ├── llm_node                (chain wrapper)
        │   └── llm_request         (the model call — llm kind; messages + tool calls + tokens)
        │       └── llm_request_run (retry attempt — chain)
        ├── function_tool           (tool kind; args + output)
        └── tts_node                (chain wrapper)
            └── tts_request         (synthesis — llm kind; text in)
                └── tts_request_run (retry attempt — chain)
```

Every `lk.*` attribute LiveKit emits also lands as `langsmith.metadata.lk_*`
(JSON blobs flattened), so nothing the framework measures is dropped —
per-stage latencies live on the span that measured them. The two
**speech-to-speech variants** produce the same root/session/turn shape with no
pipeline spans; per-turn token counts come from LiveKit's realtime metrics,
folded onto each `agent_turn`.

**Pipecat** — the LLM stage is an in-process **LangGraph** agent, so its runs
(natively LangSmith-shaped, tool calls included) nest under Pipecat's `llm`
span in one trace — which is why that span is classified `chain`, not `llm`
(the real inference is the nested model nodes; llm-inside-llm would
double-count). Only the final answer is spoken; the tool-deciding turn and
tool execution are traced but never voiced:

```
conversation                        (root; full transcript + stereo what-was-heard WAV)
└── turn × N                        (chain; turn_number / was_interrupted metadata)
    ├── stt                         (audio → transcript)
    ├── llm                         (chain — orchestrates the graph)
    │   ├── model                   (ChatOpenAI — llm kind; may emit tool calls)
    │   ├── tools: lookup_weather   (tool run)
    │   └── model                   (final answer — spoken)
    └── tts                         (text → audio)
```

This nesting works because `LangGraphLLMService` runs the graph inside
Pipecat's `@traced_llm` span and `LANGSMITH_TRACING_MODE=otel` routes
LangChain/LangGraph runs through the same OTel provider. With a stock Pipecat
LLM service instead, the `llm` span *is* the inference — construct the
processor with the default `llm_span_kind="llm"`.

In every backend, the root span carries `ls_modality: "audio"` and the
conversation recording as an attachment, and raw audio bytes never reach
LangSmith (scrubbed to `<N bytes>` placeholders on the SDK path; never copied
into attributes on the OTel path).

## Interruption handling

Barge-in (the user interrupting the agent mid-utterance) is handled by every
backend, and it's a first-class part of the demo:

- **OpenAI** — server VAD with `interrupt_response: true`; on
  `input_audio_buffer.speech_started` the agent flushes the speaker buffer.
- **ADK** — Gemini emits an `interrupted` event; the agent flushes the speaker.
- **LiveKit** — the framework truncates the assistant turn to what was
  actually spoken before updating its chat context.
- **Pipecat** — Silero VAD drives barge-in; the turn tracker records
  `was_interrupted`, surfaced as metadata on the LangSmith `turn` span.

### Recording what was *heard*, not what was generated

A realtime agent generates more audio than it plays — barge-in truncates the
bot mid-sentence. Every backend records at the **playout boundary**, so the
recording reflects what each party actually heard:

- **OpenAI / ADK** — the agent side is recorded in `SpeakerStream`'s device
  callback (the bytes actually pulled to the speaker); audio flushed on
  barge-in is never recorded.
- **Pipecat** — `RecordingLocalAudioTransport` taps played agent audio in
  `write_audio_frame` (reached only *after* barge-in truncation) plus the
  user's mic, and writes one stereo WAV.
- **LiveKit** — the framework's own recorder records the transmitted track
  post-truncation (console mode needs the `--record` flag).

In production (telephony, web push-to-talk) the same principle applies:
record at the transport/media layer, never from the model/TTS output.

## Notes

- Each backend lazily imports its framework, so a missing optional dep for
  one backend doesn't break the others.
- Model overrides via env: OpenAI `REALTIME_MODEL` (default `gpt-realtime-2`);
  ADK `ADK_LIVE_MODEL` (default `gemini-3.1-flash-live-preview`); LiveKit
  `LIVEKIT_LLM_MODEL` / `LIVEKIT_STT_MODEL` / `LIVEKIT_TTS_VOICE`; Pipecat
  `PIPECAT_LLM_MODEL` / `PIPECAT_STT_MODEL` / `PIPECAT_TTS_VOICE`.
- The two OTel processors are deliberately parallel in design: they wrap
  their downstream exporter (rewrites always land before export), give the
  inference spans real message I/O (singular LangChain-format JSON — the
  indexed `gen_ai.prompt.{n}.*` form can't carry tool calls), keep wrappers
  as bare chains, and take per-conversation identity (`thread_id_provider`)
  and audio location by injection rather than owning hidden state. The
  processor module docstrings document every decision.

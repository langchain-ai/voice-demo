# voice-demo

Three voice-agent backends — **LiveKit STT/LLM/TTS**, **OpenAI Realtime v2v**, **Google ADK Live multi-modal** — sharing one console interface. Each backend is instrumented for LangSmith *the way its framework intends*.

The point isn't the agents (they're toy weather assistants). The point is **how each one is traced**, side-by-side.

## The tracing principle

> Use **OTEL** when the framework runs in-process and emits its own spans. Use the **SDK** when you're consuming an event stream from a remote service.

| Backend | Tracing path | Why |
|---|---|---|
| LiveKit | OTEL + processor that translates `lk.*` → `gen_ai.*` / `langsmith.*` | LiveKit runs the full STT/LLM/TTS/VAD pipeline in-process and emits OTel spans for every stage. Translating attribute names lets us inherit all of it — transcripts, per-stage latencies, model names, EOU probabilities, audio attachments — for free. |
| OpenAI Realtime | SDK `RunTree`, one span per event | Realtime is a remote WebSocket: we observe an event stream (`input_audio_buffer.speech_started`, `response.created`, etc.) and build the trace ourselves. Every event becomes its own child span under the session root, carrying that event's full payload — so the trace mirrors *exactly what the server sent*. The SDK gives us multipart audio attachments and tool calls as proper child runs that nest under the event being handled. |
| Google ADK Live | SDK `RunTree`, one span per event | Same situation: `Runner.run_live` is a remote stream over the wire. ADK's OTel instrumentation covers its non-live paths but **doesn't emit spans for `run_live`** — going OTel-only here produces an empty root span with no children. We span each event we observe (`input_transcription`, `function_call`, `turn_complete`, …) in the same RunTree shape as OpenAI. |

The folder layout reflects this: LiveKit pairs its `agent.py` with a `processor.py` that enriches OTel spans before export. OpenAI and ADK each have just an `agent.py` (no processor) — the visible signal that they build the trace via the SDK directly.

## Layout

```
src/voice_demo/
├── cli.py                 # `voice-demo --backend openai|adk|livekit`
├── audio.py               # shared MicStream + SpeakerStream (sounddevice, PCM16)
├── tracing.py             # shared LangSmith env wiring
├── openai/
│   ├── agent.py           # Realtime event loop + RunTree span per event
│   ├── guardrail.py       # @traceable LangChain structured-output guardrail
│   └── tools.py           # @traceable Open-Meteo weather lookup
├── adk/
│   └── agent.py           # ADK Runner.run_live() driver + RunTree span per event
└── livekit/
    ├── agent.py           # AgentSession in console mode + tracer setup
    ├── processor.py       # OTEL processor (lk.* translation, audio attachment)
    └── _thread_id.py      # ContextVar for stable per-session thread_id
```

## Setup

Python 3.11+, [uv](https://docs.astral.sh/uv/).

```bash
cd voice-demo
uv sync --all-extras            # or just one: --extra openai / --extra adk / --extra livekit
cp .env.example .env            # then fill in keys
```

### Required env

- `LANGSMITH_API_KEY` — for tracing (optional but the whole point of the demo)
- `OPENAI_API_KEY` — used by the OpenAI Realtime backend, the LangChain guardrail, *and* LiveKit's STT/LLM/TTS plugins
- `GOOGLE_API_KEY` — used by the ADK Live backend (Gemini)
- `LIVEKIT_API_KEY` / `LIVEKIT_API_SECRET` / `LIVEKIT_URL` — dummy values fine, LiveKit's SDK only validates them at startup

## Run

```bash
uv run voice-demo --backend openai
uv run voice-demo --backend adk
uv run voice-demo --backend livekit
```

All three open the local mic + speaker. For LiveKit, press the spacebar to talk (LiveKit's console UX). For OpenAI and ADK, just start speaking — server VAD handles turn boundaries.

Try:
- "What's the weather in San Francisco?"
- "How's the weather in Tokyo and Rome right now?"
- (OpenAI only) "What do you think about pineapple on pizza?" — guardrail trips, theatrical refusal.

## What you see in LangSmith

Each backend lands in its own project: `voice-demo-openai`, `voice-demo-adk`, `voice-demo-livekit`.

**OpenAI** — one conversation is one trace. Every WebSocket event becomes its own span under the session root, in arrival order, carrying that event's full (scrubbed) payload — in `inputs` if the user sent it toward the model (speech buffer, transcription), in `outputs` if the model/server sent it back (`response.*`, `error`):
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
Raw audio (`response.output_audio.delta`) is played but not spanned, and every other streaming partial (`*.delta`) is dropped — the terminal `*.done` events carry the complete payload. Work the agent does *while handling* an event (the guardrail check, tool execution) nests inside that event's span, exactly as a tool call would in any other traced app. The session root carries a single **stereo WAV** reconstructed from timestamped chunks (user on left, agent on right — interruption shows as overlap).

**ADK** — same shape as OpenAI (one conversation = one trace), spanning each `run_live` event:
```
realtime_session                    (root; one stereo conversation.wav: L=user, R=agent)
│   metadata: thread_id, model, event_count, duration_s
├── input_transcription            (event — user speech chunk)
├── output_transcription           (event — agent speech chunk)
├── function_call: get_weather     (event)
│   └── execute_tool: get_weather  (real tool child; finalized when the matching
│                                   function_response arrives)
├── function_response: get_weather (event)
├── turn_complete                  (event)
└── interrupted                    (event)
```
Built explicitly with `RunTree` because ADK's OTel instrumentation doesn't cover `Runner.run_live`. Events whose *only* payload is an audio chunk are played but not spanned (there are too many), and every recorded event has its audio bytes scrubbed to a `<N bytes>` placeholder. Tool calls trace as proper child runs — the parents are simply the literal events instead of synthesized turns.

**LiveKit**:
```
job_entrypoint                      (root, attaches the full conversation WAV)
└── agent_session
    └── agent_turn × N              (per-turn metadata: turn_e2e_latency, turn_llm_ttft, turn_tts_ttfb, turn_eou_*)
        ├── user_turn               (STT — chain kind, holds the user transcript)
        ├── llm_node                (LLM — the model response)
        └── tts_node                (TTS — voice + audio_duration)
```

## Notes

- Each backend lazily imports its framework, so a missing optional dep for one backend doesn't break the others.
- The OpenAI backend uses `gpt-realtime-2` by default; override with `REALTIME_MODEL`. The ADK backend uses `gemini-2.5-flash-native-audio-latest`; override with `ADK_LIVE_MODEL`.

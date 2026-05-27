# voice-demo

Three voice-agent backends — **LiveKit STT/LLM/TTS**, **OpenAI Realtime v2v**, **Google ADK Live multi-modal** — sharing one console interface. Each backend is instrumented for LangSmith *the way its framework intends*.

The point isn't the agents (they're toy weather assistants). The point is **how each one is traced**, side-by-side.

## The tracing principle

> Use **OTEL** when the framework runs in-process and emits its own spans. Use the **SDK** when you're consuming an event stream from a remote service.

| Backend | Tracing path | Why |
|---|---|---|
| LiveKit | OTEL + processor that translates `lk.*` → `gen_ai.*` / `langsmith.*` | LiveKit runs the full STT/LLM/TTS/VAD pipeline in-process and emits OTel spans for every stage. Translating attribute names lets us inherit all of it — transcripts, per-stage latencies, model names, EOU probabilities, audio attachments — for free. |
| OpenAI Realtime | SDK `RunTree`, built explicitly per turn | Realtime is a remote WebSocket: we observe an event stream (`input_audio_buffer.speech_started`, `response.created`, etc.) and build the trace ourselves. The SDK gives us multipart audio attachments, mid-run patching, first-class interruption status, and tool calls as proper child runs. |
| Google ADK Live | SDK `RunTree`, built explicitly per turn | Same situation: `Runner.run_live` is a remote stream over the wire. ADK's OTel instrumentation covers its non-live paths but **doesn't emit spans for `run_live`** — going OTel-only here produces an empty root span with no children. We translate the events we observe (`input_transcription`, `inline_data`, `function_call`, `turn_complete`) into the same RunTree shape as OpenAI. |

The folder layout reflects this: LiveKit pairs its `agent.py` with a `processor.py` that enriches OTel spans before export. OpenAI and ADK each have just an `agent.py` (no processor) — the visible signal that they build the trace via the SDK directly.

## Layout

```
src/voice_demo/
├── cli.py                 # `voice-demo --backend openai|adk|livekit`
├── audio.py               # shared MicStream + SpeakerStream (sounddevice, PCM16)
├── tracing.py             # shared LangSmith env wiring
├── openai/
│   ├── agent.py           # Realtime event loop + RunTree per turn
│   ├── guardrail.py       # @traceable LangChain structured-output guardrail
│   └── tools.py           # @traceable Open-Meteo weather lookup
├── adk/
│   └── agent.py           # ADK Runner.run_live() driver + RunTree per turn
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

**OpenAI** — one conversation is one trace:
```
realtime_session                    (root; one stereo conversation.wav: L=user, R=agent)
├── user_turn 1                     (opened on speech_started, audio: user_utterance.wav)
│   │   metadata: turn_user_speech_duration_ms, turn_transcription_latency_ms,
│   │             turn_think_latency_ms, turn_e2e_latency_ms
│   ├── guardrail
│   └── agent_response (1.1)        (per response.create→done cycle; response_audio.wav)
│       │   metadata: response_ttfb_ms, response_duration_ms
│       │   usage_metadata: { input_tokens, output_tokens, total_tokens }
│       └── lookup_weather × N      (nested under the response that called them)
├── user_turn 2 [tag: interrupted]
│   └── agent_response (2.1) [tag: cancelled]   (partial transcript + cut-off audio)
├── user_turn 3 [tag: no_transcript]   (noise / cough; outputs: { noop: true })
└── user_turn 4
    ├── agent_response (4.1)        (emits tool calls)
    │   └── lookup_weather × 2
    └── agent_response (4.2)        (post-tool follow-up speech)
```
Turn boundaries are VAD events (`speech_started` → terminal `response.done`), not transcription — so per-stage latency is observable end-to-end and noise turns stay in the trace tagged `no_transcript` for debugging "why didn't the agent hear me?". Each response cycle is its own `agent_response` child, so cancelled responses don't pollute the turn. The session root carries a single **stereo WAV** reconstructed from timestamped chunks (user on left, agent on right — interruption shows as overlap).

**ADK** — same shape as OpenAI (one conversation = one trace):
```
realtime_session                    (root; one stereo conversation.wav: L=user, R=agent)
├── user_turn 1                     (opens on first event after last turn_complete;
│   │                                attachments: user_utterance.wav)
│   │   metadata: turn_duration_ms, agent_audio_duration_ms, e2e_latency_ms
│   │   inputs:  user_transcript
│   │   outputs: agent_transcript
│   └── execute_tool: get_weather   (per function_call seen)
├── user_turn 2 [tag: interrupted]
└── user_turn 3 [tag: no_transcript]   (noise / cough; outputs: { noop: true })
```
Built explicitly with `RunTree` because ADK's OTel instrumentation doesn't cover `Runner.run_live`. Per-stage latency is coarser than OpenAI's (no explicit `speech_started/stopped` events from ADK — we get an `e2e_latency_ms` from first `input_transcription` to first agent audio instead).

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

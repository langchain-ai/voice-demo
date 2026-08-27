# Tracing Voice Agents to LangSmith - Demo Implementations

This repo implements a variety of voice agent backends across different frameworks, each instrumented for
[LangSmith](https://smith.langchain.com/) tracing using best practices.

## The demo implementations


| `--backend` | Stack |
|---|---|
| `openai` | OpenAI Realtime, raw WebSocket |
| `openai-agents` | OpenAI Realtime, via the Agents SDK |
| `gemini` | Gemini Live, raw WebSocket via the official `google-genai` SDK |
| `adk` | Google ADK Live (Gemini) |
| `livekit` | LiveKit STT → LLM → TTS cascade (AssemblyAI STT · OpenAI LLM · Cartesia TTS) |
| `livekit-with-langgraph` | LiveKit, but the Agent's `llm_node` runs an in-process LangGraph agent — LiveKit owns the ChatContext (barge-in-truncated transcript), LangGraph owns the control flow |
| `livekit-with-recording-egress` | LiveKit cascade that delivers the report and recording separately, reading audio from a distinct simulated Egress path |
| `livekit-with-openai-realtime` | LiveKit, LLM slot swapped for OpenAI Realtime (speech-to-speech) |
| `livekit-with-gemini-live` | LiveKit, LLM slot swapped for Gemini Live (speech-to-speech) |
| `pipecat` | Pipecat STT / LLM / TTS (Deepgram STT + Aura TTS · OpenAI LLM) |
| `pipecat-with-langgraph` | Pipecat, LLM stage is an in-process LangGraph agent |
| `pipecat-with-openai-realtime` | Pipecat, STT/LLM/TTS cascade swapped for OpenAI Realtime (speech-to-speech) |
| `pipecat-with-gemini-live` | Pipecat, STT/LLM/TTS cascade swapped for Gemini Live (speech-to-speech) |

Each backend lives in its own self-contained folder under `src/voice_demo/`
(the `livekit*` and `pipecat*` folders repeat some framework boilerplate on
purpose, so each one reads as one complete example).

## Workshops

Notebook walkthroughs live in `src/workshop/`:

- `openai_realtime_langsmith.ipynb`
- `google_adk_live_langsmith.ipynb`
- `livekit_agent_langsmith.ipynb`
- `pipecat_cascade_langsmith.ipynb`

Each workshop follows the same flow: build the agent, configure LangSmith
tracing, put the framework pieces together, then run the maintained backend code
live so the audio plumbing stays out of the lesson.

## Setup

Requires Python 3.11+ and [uv](https://docs.astral.sh/uv/).

```bash
uv sync --all-extras     # or one backend's deps: --extra openai / openai-agents / gemini / adk / livekit / pipecat
cp .env.example .env      # then fill in the keys below
```

Fill in `.env` with your own API keys.

> The LiveKit API keys are not required in console mode — the dummy values in `.env.example` will work fine.


## Run

```bash
uv run voice-demo --backend openai
uv run voice-demo --backend openai-agents
uv run voice-demo --backend gemini
uv run voice-demo --backend adk
uv run voice-demo --backend livekit
uv run voice-demo --backend livekit-with-langgraph
uv run voice-demo --backend livekit-with-recording-egress
uv run voice-demo --backend livekit-with-openai-realtime
uv run voice-demo --backend livekit-with-gemini-live
uv run voice-demo --backend pipecat
uv run voice-demo --backend pipecat-with-langgraph
uv run voice-demo --backend pipecat-with-openai-realtime
uv run voice-demo --backend pipecat-with-gemini-live
```

Each backend opens your local mic and speaker.

The `livekit-with-recording-egress` backend copies LiveKit's completed local
recording to a separate temporary path in the shutdown callback and calls
`complete_recording`; the LangSmith processor captures the session report
automatically. Its five-second recording delay produces an offset near 5000
milliseconds for testing player alignment and span-click seeking.

Things to try:

- "What's the weather in Tokyo?"
- "How's the weather in Rome and Berlin right now?" — watch the weather tool get called once per city in the trace.
- Interrupt the agent while it's talking — watch it stop and listen.

Traces land in a LangSmith project per backend: `voice-demo-openai`,
`voice-demo-gemini`, `voice-demo-adk`, `voice-demo-livekit`, and so on. Override the name with
`--project`, or pass `--debug` for verbose tracing logs.

## How the tracing works

All tracing comes from the published LangSmith SDK's voice integrations, under
`langsmith.integrations`. Each backend wires one up in a single line that leaves
the app's own event loop untouched.

## Layout

```
src/voice_demo/
├── cli.py         # entry point: arg parsing + builds/injects the console mic, speaker, and status UI
├── tracing.py     # shared LANGSMITH_* env wiring
├── audio.py       # AudioInput/AudioOutput protocols + MicStream/SpeakerStream
├── console.py     # StatusUI protocol + ConsoleStatus / NullUI
├── prompts.py     # shared system prompt + greeting
├── weather.py     # shared Open-Meteo lookup (no API key)
│
├── openai/, openai_agents/, gemini/, adk/  # event-stream backends (each has an agent.py)
└── livekit*/, pipecat*/                     # in-process backends (each has an agent.py)
```

The OpenAI, Gemini, and ADK backends ship no console of their own, so `cli.py` builds one
and injects it through small protocols (`AudioInput` / `AudioOutput` /
`StatusUI`). To drive the same agent from a web app or a phone call, implement
those interfaces and inject your own — without touching the agent's event loop,
tracing, or tool logic. LiveKit and Pipecat bring their own transport, so for
those the CLI just wires the tracer and hands over control.

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
| `livekit-with-openai-realtime` | LiveKit, LLM slot swapped for OpenAI Realtime (speech-to-speech) |
| `livekit-with-gemini-live` | LiveKit, LLM slot swapped for Gemini Live (speech-to-speech) |
| `pipecat` | Pipecat STT / LLM / TTS (Deepgram STT + Aura TTS · OpenAI LLM) |
| `pipecat-with-langgraph` | Pipecat, LLM stage is an in-process LangGraph agent |
| `pipecat-with-openai-realtime` | Pipecat, STT/LLM/TTS cascade swapped for OpenAI Realtime (speech-to-speech) |
| `pipecat-with-gemini-live` | Pipecat, STT/LLM/TTS cascade swapped for Gemini Live (speech-to-speech) |
| `elevenlabs` | ElevenLabs Agents — the agent runs on ElevenLabs' servers; traced post-call from its webhooks |
| `elevenlabs-webhook` | Not an agent: the receiver that verifies ElevenLabs' post-call webhooks and forwards them to LangSmith |

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
uv run voice-demo --backend livekit-with-openai-realtime
uv run voice-demo --backend livekit-with-gemini-live
uv run voice-demo --backend pipecat
uv run voice-demo --backend pipecat-with-langgraph
uv run voice-demo --backend pipecat-with-openai-realtime
uv run voice-demo --backend pipecat-with-gemini-live
```

`elevenlabs` needs a second process and some setup — see below.

Each backend opens your local mic and speaker.

Things to try:

- "What's the weather in Tokyo?"
- "How's the weather in Rome and Berlin right now?" — watch the weather tool get called once per city in the trace.
- Interrupt the agent while it's talking — watch it stop and listen.

Traces land in a LangSmith project per backend: `voice-demo-openai`,
`voice-demo-gemini`, `voice-demo-adk`, `voice-demo-livekit`, and so on. Override the name with
`--project`, or pass `--debug` for verbose tracing logs.

## ElevenLabs

Every other backend runs its agent in-process, so the LangSmith integration
watches it live. ElevenLabs runs the agent on its own servers and emits a
finished OTLP trace **after** the call, as a webhook. So this backend is a pair:
one process holds the conversation, another receives the trace.

Nothing is traced unless the receiver is running and reachable from the public
internet *before* you hang up. ElevenLabs never retries the audio webhook.

### One command

```bash
uv run voice-demo --backend elevenlabs --tunnel
```

Opens an ngrok tunnel, registers a webhook with ElevenLabs (the create call
returns the signing secret, so there is nothing to copy out of the dashboard),
points **the one agent you are calling** at it, starts the receiver, holds the
conversation, then puts the agent's webhook settings back and deletes the
webhook it made.

Needs `ELEVENLABS_AGENT_ID`, an ngrok authtoken, and an `ELEVENLABS_API_KEY`
**with the `webhooks_write` permission** — a default key does not have it, and
without it this exits with a message telling you so. Create one at
[API keys](https://elevenlabs.io/app/settings/api-keys), or use the by-hand flow
below, which needs no special permission.

For ngrok: if you have ever signed in with the CLI (`ngrok config add-authtoken`),
that token is found automatically — `NGROK_AUTHTOKEN` is only needed if you have
not. The embedded agent this uses does not read the CLI's config on its own, so
the demo reads it for you.

It scopes the change to a single agent on purpose: the workspace-level post-call
webhook is shared with everyone else in your ElevenLabs workspace.

After you hang up it waits for ElevenLabs to finish post-call analysis, which is
the one delay nobody can remove — no SLA is documented for it.

### Or, by hand

```bash
uv run voice-demo --backend elevenlabs-webhook --port 8080   # terminal 1
ngrok http 8080                                              # terminal 2
```

### 2. Configure the webhook in ElevenLabs

In [Agents settings](https://elevenlabs.io/app/agents/settings), under
**Post-Call Webhook**, create a webhook:

- **URL** — the `https://` address ngrok printed, plus `/webhook`
- **Transcript** event — on
- **OpenTelemetry transcript payloads** — on. This is what makes ElevenLabs send
  `post_call_transcription_otel` instead of the plain JSON transcript; without
  it there is no trace to forward.
- **Send audio data** — on, so the conversation MP3 is attached to the trace
- Copy the signing secret it shows on creation into `ELEVENLABS_WEBHOOK_SECRET`
  (it is shown once)

Both toggles can also be set per agent, under
`platform_settings.workspace_overrides.webhooks`.

Then set `ELEVENLABS_AGENT_ID` (from [Agents](https://elevenlabs.io/app/agents))
and, if that agent is private, `ELEVENLABS_API_KEY`.

### 3. Talk to it

```bash
uv run voice-demo --backend elevenlabs      # terminal 3
```

Hang up with Ctrl-C. When the trace webhook lands, the receiver fetches the
recording from the conversations API and exports one trace to
`voice-demo-elevenlabs`.

### Where the audio comes from

Not from the `post_call_audio` webhook, even though one exists. That webhook
arrives independently of the trace, in either order, and is **never retried** —
so pairing them means holding the trace on a timer and still losing the audio
whenever delivery fails. The receiver pulls the recording from
`GET /v1/convai/conversations/{id}/audio` instead, the moment the trace lands
(0.2–0.6 s for a one-minute call, scaling with duration). Nothing waits, and nothing is lost to a dropped
delivery.

That the export happens exactly once is not a preference. LangSmith's OTLP
ingest answers a repeat of a span it already has with a `409`, so there is no
attaching the audio in a second pass.

## How the tracing works

All tracing comes from the LangSmith SDK's voice integrations, under
`langsmith.integrations`. Each backend wires one up in a single line that leaves
the app's own event loop untouched — except `elevenlabs`, which has no live
agent to instrument and instead forwards the vendor's own post-call OTLP trace
through `langsmith.integrations.elevenlabs`.

> `langsmith.integrations.elevenlabs` has not shipped in a published `langsmith`
> release yet, so the `elevenlabs` extra alone is not enough. Until it lands,
> install the SDK from a local checkout alongside this repo:
> `uv pip install -e ../langsmith-sdk/python`.

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
├── livekit*/, pipecat*/                     # in-process backends (each has an agent.py)
└── elevenlabs/    # agent.py (the conversation) + webhook.py (the post-call
                   # receiver); utils/ holds the mic and tunnel plumbing
```

The OpenAI, Gemini, and ADK backends ship no console of their own, so `cli.py` builds one
and injects it through small protocols (`AudioInput` / `AudioOutput` /
`StatusUI`). To drive the same agent from a web app or a phone call, implement
those interfaces and inject your own — without touching the agent's event loop,
tracing, or tool logic. LiveKit and Pipecat bring their own transport, so for
those the CLI just wires the tracer and hands over control.

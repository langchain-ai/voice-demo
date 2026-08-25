"""Entry point: `voice-demo --backend <name>`.

Backends: openai · openai-agents · gemini · adk · livekit · livekit-with-openai-realtime ·
livekit-with-gemini-live · pipecat · pipecat-with-langgraph ·
pipecat-with-openai-realtime · pipecat-with-gemini-live · elevenlabs ·
elevenlabs-webhook. The
`*-with-openai-realtime` / `*-with-gemini-live` / `livekit-with-*` backends swap
their framework's STT→LLM→TTS cascade for a speech-to-speech realtime model
(OpenAI Realtime / Gemini Live); `pipecat` uses Pipecat's stock OpenAI LLM
service, while `pipecat-with-langgraph` runs an in-process LangGraph graph as the
LLM stage.

This module is the *frontend*. It owns everything backend-agnostic — argument
parsing, LangSmith env wiring, and (for the SDK backends) constructing the local
mic + speaker + status meter and injecting them into the agent.

The agents themselves know nothing about the console: the OpenAI and ADK
backends receive their audio I/O and status UI through small protocols
(`AudioInput` / `AudioOutput` / `StatusUI`), so swapping this terminal frontend
for a web or telephony one means reimplementing those interfaces here, not
touching the agents. LiveKit and Pipecat own their own audio path through their
frameworks, so for those we just wire the tracer and hand control over.

ElevenLabs is the exception to all of it: the agent runs on ElevenLabs' servers
and the trace arrives afterwards by webhook, so it comes as a pair — the
`elevenlabs` backend holds the conversation, and `elevenlabs-webhook` runs the
receiver that verifies those webhooks and forwards them to LangSmith.

Each backend lazily imports its framework, so a missing optional dependency for
one backend doesn't break the others. `uv sync --extra openai` is enough to run
the OpenAI backend.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from pathlib import Path

from dotenv import load_dotenv

from . import tracing

_HERE = Path(__file__).resolve().parent
for _candidate in (_HERE / ".env", _HERE.parent.parent / ".env"):
    if _candidate.exists():
        load_dotenv(_candidate, override=False)
        break

# All direct SDK backends run the local mic + speaker at 24 kHz (Gemini and ADK
# resample to 16 kHz internally before sending audio to Gemini Live).
_CONSOLE_SAMPLE_RATE = 24_000


def _run_console_backend(run, project: str) -> None:
    """Build the local console frontend and inject it into an SDK agent.

    `run` is the agent's async `run(project_name, *, audio_in, audio_out, ui)`.
    """
    from .audio import MicStream, SpeakerStream
    from .console import ConsoleStatus

    mic = MicStream(sample_rate=_CONSOLE_SAMPLE_RATE)
    speaker = SpeakerStream(sample_rate=_CONSOLE_SAMPLE_RATE)
    status = ConsoleStatus()
    asyncio.run(run(project, audio_in=mic, audio_out=speaker, ui=status))


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="voice-demo",
        description="Run one of the voice-agent backends with LangSmith tracing.",
    )
    parser.add_argument(
        "--backend",
        required=True,
        choices=(
            "openai",
            "openai-agents",
            "gemini",
            "adk",
            "livekit",
            "livekit-with-langgraph",
            "livekit-with-openai-realtime",
            "livekit-with-gemini-live",
            "pipecat",
            "pipecat-with-langgraph",
            "pipecat-with-openai-realtime",
            "pipecat-with-gemini-live",
            "elevenlabs",
            "elevenlabs-webhook",
        ),
        help="Which voice-agent stack to launch.",
    )
    parser.add_argument(
        "--project",
        default=None,
        help="LangSmith project name. Defaults to '<prefix>-<backend>'.",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8080,
        help="Port for the elevenlabs-webhook receiver. Ignored by other backends.",
    )
    parser.add_argument(
        "--tunnel",
        action="store_true",
        help=(
            "elevenlabs only: do the whole loop in one process — open an ngrok "
            "tunnel, register a webhook, point the agent at it, run the "
            "receiver, converse, then undo the ElevenLabs-side changes."
        ),
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Verbose tracing-processor logs to stderr.",
    )
    args = parser.parse_args()

    if args.debug:
        # The LangSmith voice integrations log under this package.
        logging.basicConfig(level=logging.INFO)
        logging.getLogger("langsmith.integrations").setLevel(logging.DEBUG)

    project = tracing.configure(args.backend, project=args.project)

    if args.backend == "openai":
        from .openai.agent import run as run_openai

        _run_console_backend(run_openai, project)

    elif args.backend == "openai-agents":
        # Same OpenAI Realtime model as `openai`, but driven through the OpenAI
        # Agents SDK (RealtimeAgent/RealtimeRunner) instead of the raw WebSocket
        # event loop. Same console transport via the shared protocols.
        from .openai_agents.agent import run as run_openai_agents

        _run_console_backend(run_openai_agents, project)

    elif args.backend == "gemini":
        # Gemini Live over google-genai's raw WebSocket session, without ADK.
        from .gemini.agent import run as run_gemini

        _run_console_backend(run_gemini, project)

    elif args.backend == "adk":
        from .adk.agent import run as run_adk

        _run_console_backend(run_adk, project)

    elif args.backend == "livekit":
        # LiveKit's console mode is its own CLI under the hood; we just hand
        # control over to it after wiring the tracer. STT→LLM→TTS cascade.
        from .livekit.agent import run as run_livekit

        run_livekit(project_name=project)

    elif args.backend == "livekit-with-langgraph":
        # Same LiveKit console flow as `livekit`, but the Agent's `llm_node` is
        # overridden to run an in-process LangGraph graph. LiveKit owns the
        # ChatContext (truthful, barge-in-truncated transcript); the graph owns
        # the control flow (the ReAct tool loop) and keeps no transcript.
        from .livekit_with_langgraph.agent import run as run_livekit_langgraph

        run_livekit_langgraph(project_name=project)

    elif args.backend == "livekit-with-openai-realtime":
        # Same LiveKit console flow, but the session's LLM slot is a
        # speech-to-speech OpenAI Realtime model instead of the cascade.
        from .livekit_with_openai_realtime.agent import run as run_livekit_openai

        run_livekit_openai(project_name=project)

    elif args.backend == "livekit-with-gemini-live":
        # Same LiveKit console flow, but the session's LLM slot is a
        # speech-to-speech Gemini Live model instead of the cascade.
        from .livekit_with_gemini_live.agent import run as run_livekit_gemini

        run_livekit_gemini(project_name=project)

    elif args.backend == "pipecat":
        # Pipecat owns its own LocalAudioTransport; we wire the OTel tracer and
        # run its pipeline. Stock OpenAI LLM service.
        from .pipecat.agent import run as run_pipecat

        asyncio.run(run_pipecat(project_name=project))

    elif args.backend == "pipecat-with-langgraph":
        # Same Pipecat pipeline, but the LLM stage is an in-process LangGraph
        # graph whose nodes nest under the `llm` span.
        from .pipecat_with_langgraph.agent import run as run_pipecat_langgraph

        asyncio.run(run_pipecat_langgraph(project_name=project))

    elif args.backend == "pipecat-with-openai-realtime":
        # Same Pipecat pipeline, but the STT/LLM/TTS cascade is collapsed into a
        # single speech-to-speech OpenAI Realtime model in the LLM slot.
        from .pipecat_with_openai_realtime.agent import run as run_pipecat_realtime

        asyncio.run(run_pipecat_realtime(project_name=project))

    elif args.backend == "elevenlabs":
        # The agent runs on ElevenLabs' servers, so there is nothing to trace in
        # this process — the post-call trace arrives by webhook. `--tunnel` puts
        # the receiver, the tunnel, and the ElevenLabs-side wiring in this same
        # process; without it, start `elevenlabs-webhook` yourself first.
        if args.tunnel:
            from .elevenlabs.utils.tunnel import run as run_elevenlabs_tunnel

            run_elevenlabs_tunnel(project_name=project, port=args.port)
        else:
            from .elevenlabs.agent import run as run_elevenlabs

            run_elevenlabs(project_name=project)

    elif args.backend == "elevenlabs-webhook":
        # Receives ElevenLabs' post-call webhooks, verifies their HMAC, pairs the
        # OTLP trace with the audio, and forwards both to LangSmith.
        from .elevenlabs.webhook import run as run_elevenlabs_webhook

        run_elevenlabs_webhook(project_name=project, port=args.port)

    elif args.backend == "pipecat-with-gemini-live":
        # Same Pipecat pipeline, but the STT/LLM/TTS cascade is collapsed into a
        # single speech-to-speech Gemini Live model (turns driven by a local VAD
        # since Gemini Live emits no turn frames of its own).
        from .pipecat_with_gemini_live.agent import run as run_pipecat_gemini

        asyncio.run(run_pipecat_gemini(project_name=project))


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(0)

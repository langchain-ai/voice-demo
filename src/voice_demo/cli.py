"""Entry point: `voice-demo --backend <name>`.

Backends: openai · openai-agents · adk · livekit · pipecat. The LiveKit backend additionally
takes `--llm {openai-realtime,google-realtime}` to swap its STT→LLM→TTS
cascade for a speech-to-speech realtime model (OpenAI Realtime / Gemini Live).

This module is the *frontend*. It owns everything backend-agnostic — argument
parsing, LangSmith env wiring, and (for the SDK backends) constructing the local
mic + speaker + status meter and injecting them into the agent.

The agents themselves know nothing about the console: the OpenAI and ADK
backends receive their audio I/O and status UI through small protocols
(`AudioInput` / `AudioOutput` / `StatusUI`), so swapping this terminal frontend
for a web or telephony one means reimplementing those interfaces here, not
touching the agents. LiveKit and Pipecat own their own audio path through their
frameworks, so for those we just wire the tracer and hand control over.

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

# Both SDK backends run the local mic + speaker at 24 kHz (ADK resamples to
# 16 kHz internally before sending to Gemini).
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
        choices=("openai", "openai-agents", "adk", "livekit", "pipecat"),
        help="Which voice-agent stack to launch.",
    )
    parser.add_argument(
        "--llm",
        default=None,
        choices=("openai-realtime", "google-realtime"),
        help=(
            "LiveKit only: fill the session's LLM slot with a speech-to-speech "
            "realtime model (OpenAI Realtime / Gemini Live) instead of the "
            "default STT→LLM→TTS cascade."
        ),
    )
    parser.add_argument(
        "--project",
        default=None,
        help="LangSmith project name. Defaults to '<prefix>-<backend>'.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Verbose tracing-processor logs to stderr.",
    )
    args = parser.parse_args()

    if args.llm is not None and args.backend != "livekit":
        parser.error("--llm only applies to --backend livekit")

    if args.debug:
        # The LangSmith voice integrations log under this package.
        logging.basicConfig(level=logging.INFO)
        logging.getLogger("langsmith.integrations").setLevel(logging.DEBUG)

    # Realtime LiveKit modes keep their own default LangSmith project, as when
    # they were separate backends (e.g. "<prefix>-livekit-openai-realtime").
    backend_label = f"{args.backend}-{args.llm}" if args.llm else args.backend
    project = tracing.configure(backend_label, project=args.project)

    if args.backend == "openai":
        from .openai.agent import run as run_openai

        _run_console_backend(run_openai, project)

    elif args.backend == "openai-agents":
        # Same OpenAI Realtime model as `openai`, but driven through the OpenAI
        # Agents SDK (RealtimeAgent/RealtimeRunner) instead of the raw WebSocket
        # event loop. Same console transport via the shared protocols.
        from .openai_agents.agent import run as run_openai_agents

        _run_console_backend(run_openai_agents, project)

    elif args.backend == "adk":
        from .adk.agent import run as run_adk

        _run_console_backend(run_adk, project)

    elif args.backend == "livekit":
        # LiveKit's console mode is its own CLI under the hood; we just hand
        # control over to it after wiring the tracer. --llm swaps the session's
        # LLM slot from the STT→LLM→TTS cascade to a realtime (S2S) model.
        from .livekit.agent import run as run_livekit

        realtime_provider = args.llm.removesuffix("-realtime") if args.llm else None
        run_livekit(project_name=project, realtime_provider=realtime_provider)

    elif args.backend == "pipecat":
        # Pipecat owns its own LocalAudioTransport; we wire the OTel tracer and
        # run its pipeline.
        from .pipecat.agent import run as run_pipecat

        asyncio.run(run_pipecat(project_name=project))


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(0)

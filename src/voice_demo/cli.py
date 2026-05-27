"""Entry point: `voice-demo --backend openai|adk|livekit`.

Lazy-imports each backend so a missing optional dependency for one backend
doesn't break the others. `uv sync --extra openai` is enough to run the OpenAI
backend; you do not need the LiveKit or ADK extras installed.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

from . import tracing

_HERE = Path(__file__).resolve().parent
for _candidate in (_HERE / ".env", _HERE.parent.parent / ".env"):
    if _candidate.exists():
        load_dotenv(_candidate, override=False)
        break


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="voice-demo",
        description="Run one of three voice-agent backends with LangSmith tracing.",
    )
    parser.add_argument(
        "--backend",
        required=True,
        choices=("openai", "adk", "livekit"),
        help="Which voice-agent stack to launch.",
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

    if args.debug:
        os.environ["LANGSMITH_PROCESSOR_DEBUG"] = "true"

    project = tracing.configure(args.backend, project=args.project)

    if args.backend == "openai":
        from .openai.agent import run as run_openai

        asyncio.run(run_openai(project_name=project))

    elif args.backend == "adk":
        from .adk.agent import run as run_adk

        asyncio.run(run_adk(project_name=project))

    elif args.backend == "livekit":
        # LiveKit's console mode is its own CLI under the hood; we just hand
        # control over to it after wiring the tracer.
        from .livekit.agent import run as run_livekit

        run_livekit(project_name=project)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(0)

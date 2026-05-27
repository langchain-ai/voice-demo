"""Shared LangSmith env wiring.

Each backend traces under a project name derived from a single prefix, so the
three sit next to each other in the LangSmith UI:

    voice-demo-openai
    voice-demo-adk
    voice-demo-livekit

For the OTEL-based backends (LiveKit, ADK), this also wires the OTLP exporter
env vars that LangSmith's `/otel/v1/traces` endpoint expects. For the SDK-based
backend (OpenAI), only the LangSmith API key and project name matter — the SDK
reads them directly.
"""

from __future__ import annotations

import os
import sys
from typing import Literal

Backend = Literal["openai", "adk", "livekit"]

LANGSMITH_OTLP_ENDPOINT = "https://api.smith.langchain.com/otel"


def project_name_for(backend: Backend, override: str | None = None) -> str:
    if override:
        return override
    prefix = os.environ.get("LANGSMITH_PROJECT_PREFIX", "voice-demo")
    return f"{prefix}-{backend}"


def configure(backend: Backend, project: str | None = None) -> str:
    """Set up env vars before any backend-specific imports.

    Returns the resolved project name so callers can pass it to
    `configure_google_adk(project_name=...)` etc.
    """
    api_key = os.environ.get("LANGSMITH_API_KEY")
    project_name = project_name_for(backend, project)

    if not api_key:
        print(
            f"[voice-demo] LANGSMITH_API_KEY not set — running '{backend}' without tracing.",
            file=sys.stderr,
        )
        return project_name

    # Used by both the SDK (LANGSMITH_PROJECT) and explicit RunTree(project_name=...) callers.
    os.environ.setdefault("LANGSMITH_TRACING", "true")
    os.environ["LANGSMITH_PROJECT"] = project_name

    if backend in ("livekit", "adk"):
        # OTLP exporter env vars — read by OTLPSpanExporter(). The
        # `Langsmith-Project` header lets the receiver assign spans to the
        # right project without us having to set it on every span.
        os.environ.setdefault("OTEL_EXPORTER_OTLP_ENDPOINT", LANGSMITH_OTLP_ENDPOINT)
        existing_headers = os.environ.get("OTEL_EXPORTER_OTLP_HEADERS", "")
        if "x-api-key" not in existing_headers:
            os.environ["OTEL_EXPORTER_OTLP_HEADERS"] = (
                f"x-api-key={api_key},Langsmith-Project={project_name}"
            )

    if backend == "adk":
        # ADK only writes message content onto span attributes when the
        # experimental gen_ai semconv opt-in is set. Without these two, the
        # spans land in LangSmith with no input/output text.
        os.environ.setdefault(
            "OTEL_SEMCONV_STABILITY_OPT_IN", "gen_ai_latest_experimental"
        )
        os.environ.setdefault(
            "OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT", "SPAN_AND_EVENT"
        )

    print(f"[voice-demo] LangSmith project: {project_name}", file=sys.stderr)
    return project_name

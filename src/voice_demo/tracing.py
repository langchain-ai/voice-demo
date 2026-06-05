"""Shared LangSmith env wiring.

Each backend traces under a project name derived from a single prefix, so they
sit next to each other in the LangSmith UI:

    voice-demo-openai
    voice-demo-adk
    voice-demo-livekit
    voice-demo-pipecat
    voice-demo-livekit-openai-realtime
    voice-demo-livekit-google-realtime

There are two tracing paths (see each backend's own module docstring for why):

  * OTEL — LiveKit (including the two realtime backends) and Pipecat run a
    framework in-process that emits its own OTel spans, so this wires the OTLP
    exporter env vars that LangSmith's `/otel/v1/traces` endpoint expects.
  * SDK  — OpenAI Realtime and ADK Live consume a remote event stream and build
    the trace themselves with the LangSmith SDK (`RunTree`). Only the API key
    and project name matter — the SDK reads them directly; no OTLP wiring.
"""

from __future__ import annotations

import os
import sys
from typing import Literal

Backend = Literal[
    "openai",
    "adk",
    "livekit",
    "pipecat",
    "livekit-openai-realtime",
    "livekit-google-realtime",
]

DEFAULT_LANGSMITH_ENDPOINT = "https://api.smith.langchain.com"


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

    if backend.startswith("livekit") or backend == "pipecat":
        # OTLP exporter env vars — read by OTLPSpanExporter(). The
        # `startswith("livekit")` also covers the livekit-*-realtime backends.
        # (ADK and OpenAI are SDK-traced via RunTree and need no OTLP wiring.)
        # `Langsmith-Project` header lets the receiver assign spans to the
        # right project without us having to set it on every span.
        #
        # Derive the OTLP endpoint from LANGSMITH_ENDPOINT (the same var the
        # LangSmith SDK honors) so the OTEL backends point at the same base URL
        # as the SDK-based OpenAI backend — including a self-hosted/localhost
        # instance. The OTel ingestion path is `<base>/otel`; OTLPSpanExporter
        # then appends `/v1/traces`.
        base = os.environ.get(
            "LANGSMITH_ENDPOINT", DEFAULT_LANGSMITH_ENDPOINT
        ).rstrip("/")
        os.environ.setdefault("OTEL_EXPORTER_OTLP_ENDPOINT", f"{base}/otel")
        existing_headers = os.environ.get("OTEL_EXPORTER_OTLP_HEADERS", "")
        if "x-api-key" not in existing_headers:
            os.environ["OTEL_EXPORTER_OTLP_HEADERS"] = (
                f"x-api-key={api_key},Langsmith-Project={project_name}"
            )

    if backend in ("livekit", "pipecat"):
        # Both these backends run a LangGraph agent as their LLM "brain". Put
        # the LangSmith SDK in OTel mode so the langchain/langgraph runs emit
        # through the shared global TracerProvider instead of posting straight
        # to the LangSmith API. That makes them inherit the active OTel context
        # and nest UNDER the framework's LLM span — LiveKit's agent_turn/llm_node,
        # or Pipecat's `llm` span — rather than forming a separate top-level
        # trace. (openai/adk stay in default langsmith mode — they trace via
        # RunTree directly.)
        os.environ.setdefault("LANGSMITH_TRACING_MODE", "otel")

    print(f"[voice-demo] LangSmith project: {project_name}", file=sys.stderr)
    return project_name

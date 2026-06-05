"""Helpers for the ADK Live agent: tracing glue."""

from __future__ import annotations

from contextlib import AbstractContextManager, nullcontext

from ..sdk_tracing import EventSession
from .events import LiveEvent


def event_context(session: EventSession, event: LiveEvent) -> AbstractContextManager:
    """The tracing span for one event — or a no-op for the agent-audio flood."""
    if event.is_audio_only:
        return nullcontext()
    return session.event_span(
        event.raw, session.now(), name=event.label, inbound=event.is_inbound
    )

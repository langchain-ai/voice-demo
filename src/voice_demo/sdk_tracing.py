"""Shared LangSmith *SDK* tracing for the event-stream backends.

The OpenAI Realtime and Google ADK Live backends are in the same situation:
each consumes an **event stream from a remote service** (a WebSocket / a
`run_live` generator) rather than running a framework in-process. Neither
service emits its own usable telemetry for the live loop, so we build the trace
ourselves with the LangSmith SDK — one root span per conversation, one child
span per received event.

This module factors out everything that pattern needs and that the two backends
would otherwise duplicate:

  * `scrub()` / `dump_event()` — turn an arbitrary event object into a compact,
    safe payload (raw audio `bytes` → `<N bytes>`, long strings truncated) so we
    never ship megabytes of audio to LangSmith.
  * `build_stereo_session_wav()` — reconstruct one stereo conversation WAV from
    the timestamped PCM16 chunks each side recorded (L=user, R=agent), laid out
    at natural play time so bursts don't overlap.
  * `EventSession` — the RunTree lifecycle: a root `realtime_session` span, a
    child span per event (`event_span`), audio-chunk recording, and finalization
    that rolls up `event_count` / `duration_s` and attaches the WAV.

Each backend keeps only what is genuinely backend-specific: which events count
as `inbound` (user→model, so the payload lands in span `inputs`) and how to
label each event span. Those are passed into `event_span()` per event.
"""

from __future__ import annotations

import io
import math
import time
import wave
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from langsmith import RunTree

# Longest string we keep on a span before truncating. Transcripts are short;
# this only ever trims an unexpectedly large blob.
MAX_STR = 2000


# ---------------------------------------------------------------------------
# Event payload scrubbing
# ---------------------------------------------------------------------------

def scrub(obj: Any) -> Any:
    """Make an event payload safe + compact for a span.

    Replaces raw `bytes` (audio / base64 blobs) with a `<N bytes>` placeholder
    and truncates very long strings, so we never ship megabytes of payload to
    LangSmith or blow up JSON serialization.
    """
    if isinstance(obj, bytes):
        return f"<{len(obj)} bytes>"
    if isinstance(obj, str):
        if len(obj) > MAX_STR:
            return obj[:MAX_STR] + f"... <+{len(obj) - MAX_STR} chars>"
        return obj
    if isinstance(obj, dict):
        return {k: scrub(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [scrub(v) for v in obj]
    return obj


def dump_event(event: Any) -> dict[str, Any]:
    """Best-effort conversion of an event object to a plain dict."""
    if hasattr(event, "model_dump"):
        try:
            return event.model_dump()
        except Exception:
            pass
    if isinstance(event, dict):
        return event
    return {"repr": repr(event)}


# ---------------------------------------------------------------------------
# Stereo conversation WAV
# ---------------------------------------------------------------------------

def _layout_chunks_to_play_time(
    chunks: list[tuple[float, bytes]], sample_rate: int
) -> list[tuple[float, bytes]]:
    """Rewrite receipt timestamps into natural-play timestamps.

    Receipt times reflect when bytes arrived from the source, not when they
    play. The agent channel especially arrives in bursts faster than realtime —
    multiple chunks can land within a few ms of each other, and placing them at
    receipt time makes them overlap and overwrite each other (you hear scrambled
    tail-ends). The correct natural play time for a chunk is the LATER of where
    the previous chunk ended and when this chunk arrived, which preserves real
    gaps between bursts and keeps consecutive bursts contiguous.
    """
    out: list[tuple[float, bytes]] = []
    cur_time = 0.0
    for i, (t_recv, data) in enumerate(chunks):
        cur_time = t_recv if i == 0 else max(cur_time, t_recv)
        out.append((cur_time, data))
        cur_time += (len(data) // 2) / sample_rate
    return out


def build_stereo_session_wav(
    user_chunks: list[tuple[float, bytes]],
    agent_chunks: list[tuple[float, bytes]],
    sample_rate: int,
) -> bytes:
    """Reconstruct a stereo WAV from timestamped PCM16 chunks.

    Left channel = user, right channel = agent. Both channels are laid out at
    natural play time (see `_layout_chunks_to_play_time`). Gaps between bursts
    are silence; overlap between user and agent (during a barge-in) is preserved
    because they live on different channels.
    """
    if not user_chunks and not agent_chunks:
        return b""

    user = _layout_chunks_to_play_time(user_chunks, sample_rate)
    agent = _layout_chunks_to_play_time(agent_chunks, sample_rate)

    def chunk_end(t: float, data: bytes) -> float:
        return t + (len(data) // 2) / sample_rate

    user_end = max((chunk_end(t, d) for t, d in user), default=0.0)
    agent_end = max((chunk_end(t, d) for t, d in agent), default=0.0)
    total_samples = int(math.ceil(max(user_end, agent_end) * sample_rate))

    stereo = np.zeros((total_samples, 2), dtype=np.int16)

    def write_channel(chunks: list[tuple[float, bytes]], channel: int) -> None:
        for t, data in chunks:
            offset = int(t * sample_rate)
            samples = np.frombuffer(data, dtype=np.int16)
            end = min(offset + len(samples), total_samples)
            n = end - offset
            if n > 0:
                stereo[offset:end, channel] = samples[:n]

    write_channel(user, 0)
    write_channel(agent, 1)

    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(2)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(stereo.tobytes())
    return buf.getvalue()


# ---------------------------------------------------------------------------
# Session: one root span per conversation, one child span per event
# ---------------------------------------------------------------------------

@dataclass
class EventSession:
    """Conversation-level tracing state. One per process lifetime.

    A conversation doesn't really have inputs/outputs — it *is* the unit of
    work, so the root span carries roll-up stats in metadata and the stereo
    conversation WAV as an attachment. Each received event becomes a child span
    via `event_span()`.
    """

    run: RunTree
    thread_id: str
    project_name: str
    sample_rate: int
    # Monotonic clock origin. Everything else stores `now() - t0`.
    t0: float = field(default_factory=time.monotonic)
    # Time-stamped audio chunks for the stereo session WAV. Each entry is
    # (offset_seconds_from_t0, pcm16_bytes). Reconstructed at finalize.
    user_chunks: list[tuple[float, bytes]] = field(default_factory=list)
    agent_chunks: list[tuple[float, bytes]] = field(default_factory=list)
    event_count: int = 0

    def now(self) -> float:
        """Seconds since the session started — the timeline used for the WAV."""
        return time.monotonic() - self.t0

    def record_user(self, t: float, data: bytes) -> None:
        self.user_chunks.append((t, data))

    def record_agent(self, t: float, data: bytes) -> None:
        self.agent_chunks.append((t, data))

    @contextmanager
    def event_span(
        self, event: Any, t_now: float, *, name: str, inbound: bool
    ) -> Iterator[RunTree]:
        """Open a child span for one received event; close it on body exit.

        Wrapping the handler body means any real work done while handling the
        event (e.g. tool execution) nests inside this span — the same way a tool
        call nests under the LLM step that triggered it in any traced app.

        The event payload lands in `inputs` for user→model (`inbound`) events and
        in `outputs` for model→user events, so the trace reads in the natural
        direction of flow.
        """
        self.event_count += 1
        payload = scrub(dump_event(event))
        run = self.run.create_child(
            name=name,
            run_type="chain",
            inputs=payload if inbound else {},
            tags=["event"],
            extra={"metadata": {"received_at_s": round(t_now, 3)}},
        )
        run.post()
        try:
            yield run
        finally:
            run.end(outputs={} if inbound else payload)
            run.patch()

    def finalize(self) -> None:
        """Roll up stats, attach the stereo WAV, and close the root span."""
        extra: dict[str, Any] = self.run.extra or {}
        metadata: dict[str, Any] = dict(extra.get("metadata") or {})
        metadata["event_count"] = self.event_count
        metadata["duration_s"] = round(time.monotonic() - self.t0, 2)
        extra["metadata"] = metadata
        self.run.extra = extra

        wav = build_stereo_session_wav(
            self.user_chunks, self.agent_chunks, self.sample_rate
        )
        if wav:
            # One audio asset for the whole conversation — stereo so you can
            # hear both sides AND see interruption overlap.
            self.run.attachments = {"conversation": ("audio/wav", wav)}
        self.run.end(outputs={})
        self.run.patch()


def start_session(
    *,
    thread_id: str,
    project_name: str,
    sample_rate: int,
    tags: list[str],
    metadata: dict[str, Any],
    name: str = "realtime_session",
) -> EventSession:
    """Create and post the conversation root span, returning an EventSession."""
    # Mark the root as an audio-modality run (these are voice conversations).
    # Shared by the OpenAI and ADK backends, which both call start_session.
    metadata = {"ls_modality": "audio", **metadata}
    run = RunTree(
        name=name,
        run_type="chain",
        inputs={},
        project_name=project_name,
        tags=tags,
        extra={"metadata": metadata},
    )
    run.post()
    return EventSession(
        run=run,
        thread_id=thread_id,
        project_name=project_name,
        sample_rate=sample_rate,
    )

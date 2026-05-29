"""OpenAI Realtime API voice agent, traced with the LangSmith SDK.

Why SDK and not OTEL: OpenAI Realtime has no native telemetry — it's a raw
WebSocket event stream of `input_audio_buffer.*`, `response.*`, etc. We build
the trace directly from that stream so the trace mirrors *exactly what the
server sent us*, rather than a curated abstraction layered on top.

## Trace shape — one conversation = one trace

Every received WebSocket event becomes its own span under the session root, in
arrival order, carrying that event's full (scrubbed) payload — in `inputs` if
the user sent it toward the model (speech buffer, transcription), in `outputs`
if the model/server sent it back (`response.*`, `error`). Raw
audio (`response.output_audio.delta`) is played but not spanned, and every other
streaming partial (`*.delta`) is dropped too — the terminal `*.done` events
carry the complete payload. Work the agent does *while handling* an event (the
guardrail check, tool execution) nests inside that event's span, exactly as a
tool call would in any other traced app.

    realtime_session                                   (root)
    │  attachments: conversation.wav                  (stereo: L=user, R=agent)
    │  metadata: event_count, duration_s
    │
    ├── input_audio_buffer.speech_started             (event)
    ├── input_audio_buffer.speech_stopped             (event)
    ├── conversation.item.input_audio_transcription.completed   (event)
    │   └── guardrail                                 (ran while handling this event)
    ├── response.created                              (event)
    ├── response.function_call_arguments.done         (event)
    ├── response.done                                 (event)
    │   └── lookup_weather × N                        (ran while handling this event)
    ├── response.done                                 (event — post-tool follow-up)
    └── error                                         (event, if the server sent one)

Audio playback, barge-in/interruption, the guardrail, and the tool follow-up
loop all behave exactly as before — only the trace representation changed: it's
now the literal event stream instead of synthesized turn/response spans.
"""

from __future__ import annotations

import asyncio
import base64
import io
import json
import math
import os
import sys
import time
import uuid
import wave
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from langsmith import RunTree
from langsmith.run_helpers import tracing_context
from openai import AsyncOpenAI

from ..audio import MicStream, SpeakerStream
from ..console import ConsoleStatus, frame_level
from .guardrail import REFUSAL_INSTRUCTIONS, check_guardrail
from .tools import lookup_weather


SYSTEM_PROMPT = """You are a friendly voice assistant who can look up the
weather for any city. Keep replies short, conversational, and free of
formatting (no asterisks, no bullet points, no emoji). When the user asks
about weather in one or more places, call the lookup_weather tool — once per
city — and then summarize the results naturally in one or two short spoken
sentences."""


WEATHER_TOOL = {
    "type": "function",
    "name": "lookup_weather",
    "description": "Get the current weather for a single city. Call once per city for multi-city questions.",
    "parameters": {
        "type": "object",
        "properties": {
            "city": {
                "type": "string",
                "description": "City name, e.g. 'San Francisco' or 'Tokyo'.",
            },
        },
        "required": ["city"],
    },
}


DEFAULT_MODEL = os.getenv("REALTIME_MODEL", "gpt-realtime-2")
SAMPLE_RATE = 24_000


# ---------------------------------------------------------------------------
# Run-tree state
# ---------------------------------------------------------------------------

@dataclass
class _Session:
    """Conversation-level state. One per process lifetime."""

    run: RunTree
    thread_id: str
    project_name: str
    # Monotonic clock origin. Everything else stores `now() - t0`.
    t0: float
    # Time-stamped audio chunks for the stereo session WAV. Each entry is
    # (offset_seconds_from_t0, pcm16_bytes). Reconstructed at finalize.
    user_chunks: list[tuple[float, bytes]] = field(default_factory=list)
    agent_chunks: list[tuple[float, bytes]] = field(default_factory=list)
    event_count: int = 0


# ---------------------------------------------------------------------------
# WAV helpers
# ---------------------------------------------------------------------------

def _layout_chunks_to_play_time(
    chunks: list[tuple[float, bytes]],
) -> list[tuple[float, bytes]]:
    """Rewrite receipt timestamps into natural-play timestamps.

    Receipt times reflect when bytes arrived from the source, not when they
    play. For the agent channel especially, the OpenAI server sends bursts
    faster than realtime — multiple 100 ms chunks can arrive within 20 ms of
    each other, and if we naively place them at receipt time they overlap and
    overwrite each other (you hear scrambled tail-ends of each chunk).

    The correct natural play time for a chunk is the LATER of:
      (a) where the previous chunk ended, and
      (b) when this chunk arrived (can't play before received)

    That preserves real gaps between bursts (e.g., between two responses) and
    keeps consecutive bursts contiguous.
    """
    out: list[tuple[float, bytes]] = []
    cur_time = 0.0
    for i, (t_recv, data) in enumerate(chunks):
        if i == 0:
            cur_time = t_recv
        else:
            cur_time = max(cur_time, t_recv)
        out.append((cur_time, data))
        # Advance by this chunk's natural duration.
        cur_time += (len(data) // 2) / SAMPLE_RATE
    return out


def _build_stereo_session_wav(session: _Session) -> bytes:
    """Reconstruct a stereo WAV from the timestamped chunks.

    Left channel = user, right channel = agent. Both channels are laid out at
    natural play time (see _layout_chunks_to_play_time). Gaps between bursts
    are silence (zeros). Overlap between user and agent (during a barge-in)
    is preserved because they live on different channels.
    """
    if not session.user_chunks and not session.agent_chunks:
        return b""

    user = _layout_chunks_to_play_time(session.user_chunks)
    agent = _layout_chunks_to_play_time(session.agent_chunks)

    def chunk_end(t: float, data: bytes) -> float:
        return t + (len(data) // 2) / SAMPLE_RATE

    user_end = max((chunk_end(t, d) for t, d in user), default=0.0)
    agent_end = max((chunk_end(t, d) for t, d in agent), default=0.0)
    total_samples = int(math.ceil(max(user_end, agent_end) * SAMPLE_RATE))

    stereo = np.zeros((total_samples, 2), dtype=np.int16)

    def write_channel(chunks: list[tuple[float, bytes]], channel: int) -> None:
        for t, data in chunks:
            offset = int(t * SAMPLE_RATE)
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
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(stereo.tobytes())
    return buf.getvalue()


# ---------------------------------------------------------------------------
# Session lifecycle
# ---------------------------------------------------------------------------

def _start_session(thread_id: str, project_name: str) -> _Session:
    # A conversation doesn't really have inputs/outputs — it IS the unit of
    # work. Model, thread_id, and roll-up stats belong in metadata.
    run = RunTree(
        name="realtime_session",
        run_type="chain",
        inputs={},
        project_name=project_name,
        tags=["voice-demo", "openai", "session"],
        extra={"metadata": {"thread_id": thread_id, "model": DEFAULT_MODEL}},
    )
    run.post()
    return _Session(run=run, thread_id=thread_id, project_name=project_name, t0=time.monotonic())


def _finalize_session(session: _Session) -> None:
    extra: dict[str, Any] = session.run.extra or {}
    metadata: dict[str, Any] = dict(extra.get("metadata") or {})
    metadata["event_count"] = session.event_count
    metadata["duration_s"] = round(time.monotonic() - session.t0, 2)
    extra["metadata"] = metadata
    session.run.extra = extra

    stereo_wav = _build_stereo_session_wav(session)
    if stereo_wav:
        # Single audio asset for the whole conversation — stereo so you can
        # hear both sides AND see interruption overlap.
        session.run.attachments = {
            "conversation": ("audio/wav", stereo_wav),
        }
    session.run.end(outputs={})
    session.run.patch()


# ---------------------------------------------------------------------------
# Event spans — one span per received WebSocket event
# ---------------------------------------------------------------------------

_MAX_STR = 2000


def _scrub(obj: Any) -> Any:
    """Make an event payload safe + compact for a span.

    Replaces raw `bytes` with a `<N bytes>` placeholder (audio/base64 blobs)
    and truncates very long strings, so we never ship megabytes of payload to
    LangSmith or blow up JSON serialization.
    """
    if isinstance(obj, bytes):
        return f"<{len(obj)} bytes>"
    if isinstance(obj, str):
        if len(obj) > _MAX_STR:
            return obj[:_MAX_STR] + f"... <+{len(obj) - _MAX_STR} chars>"
        return obj
    if isinstance(obj, dict):
        return {k: _scrub(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_scrub(v) for v in obj]
    return obj


def _dump_event(event: Any) -> dict[str, Any]:
    """Best-effort conversion of a Realtime event to a plain dict."""
    if hasattr(event, "model_dump"):
        try:
            return event.model_dump()
        except Exception:
            pass
    if isinstance(event, dict):
        return event
    return {"repr": repr(event)}


def _is_inbound(et: str) -> bool:
    """Direction of an event relative to the model.

    Inbound = something the user sent toward the model (their speech buffer,
    their transcription) → goes in span `inputs`. Everything else (`response.*`,
    `error`, `session.*`) is the model/server talking back → span `outputs`.
    """
    return et.startswith("input_audio_buffer") or "input_audio_transcription" in et


@contextmanager
def _start_event(session: _Session, event: Any, t_now: float) -> Iterator[RunTree]:
    """Open a child span for one received event; close it when the body exits.

    Wrapping the handler body means any real work done while handling the event
    (the guardrail check, tool execution) nests inside this span — the same way
    a tool call nests under the LLM step that triggered it in any traced app.

    The event payload lands in `inputs` for user→model events and `outputs` for
    model→user events, so the trace reads in the natural direction of flow.
    """
    session.event_count += 1
    et = getattr(event, "type", "event")
    payload = _scrub(_dump_event(event))
    inbound = _is_inbound(et)
    run = session.run.create_child(
        name=et,
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


# ---------------------------------------------------------------------------
# Misc helpers
# ---------------------------------------------------------------------------

async def _execute_tool(name: str, raw_args: str) -> dict:
    if name != "lookup_weather":
        return {"error": f"unknown tool: {name}"}
    try:
        args = json.loads(raw_args or "{}")
    except json.JSONDecodeError:
        args = {}
    city = (args.get("city") or "").strip()
    if not city:
        return {"error": "missing city"}
    return await lookup_weather(city)


# ---------------------------------------------------------------------------
# Main session loop
# ---------------------------------------------------------------------------

async def run(project_name: str) -> None:
    if not os.getenv("OPENAI_API_KEY"):
        print("OPENAI_API_KEY is not set.", file=sys.stderr)
        sys.exit(1)

    client = AsyncOpenAI()
    thread_id = str(uuid.uuid4())

    status = ConsoleStatus()
    status.log(f"[openai] thread_id={thread_id}")
    status.log("[openai] connecting to OpenAI Realtime API...")

    mic = MicStream(sample_rate=SAMPLE_RATE)
    speaker = SpeakerStream(sample_rate=SAMPLE_RATE)

    session = _start_session(thread_id, project_name)
    pending_tool_calls: list[tuple[str, str, str]] = []

    def _now() -> float:
        return time.monotonic() - session.t0

    with tracing_context(
        metadata={"thread_id": thread_id},
        tags=["voice-demo", "openai"],
        project_name=project_name,
    ):
        try:
            async with client.realtime.connect(model=DEFAULT_MODEL) as connection:
                await connection.session.update(
                    session={
                        "type": "realtime",
                        "instructions": SYSTEM_PROMPT,
                        "output_modalities": ["audio"],
                        "audio": {
                            "input": {
                                "format": {"type": "audio/pcm", "rate": SAMPLE_RATE},
                                "transcription": {"model": "gpt-4o-mini-transcribe"},
                                "noise_reduction": {"type": "near_field"},
                                "turn_detection": {
                                    "type": "server_vad",
                                    "create_response": False,
                                    "interrupt_response": True,
                                },
                            },
                            "output": {
                                "format": {"type": "audio/pcm", "rate": SAMPLE_RATE},
                                "voice": "alloy",
                            },
                        },
                        "tools": [WEATHER_TOOL],
                        "tool_choice": "auto",
                    }
                )

                mic.start()
                speaker.start()
                status.log("[openai] connected. Talk into your mic — Ctrl-C to quit.")
                status.set_state("listening")

                async def pump_mic() -> None:
                    async for frame in mic.frames():
                        await connection.input_audio_buffer.append(
                            audio=base64.b64encode(frame).decode("ascii")
                        )
                        # Record into the session's timestamped timeline for
                        # the stereo conversation WAV.
                        session.user_chunks.append((_now(), frame))
                        status.update_level(frame_level(frame))

                mic_task = asyncio.create_task(pump_mic())

                async for event in connection:
                    et = event.type
                    t_now = _now()

                    # Raw audio: hundreds of chunks per response. Play it, but
                    # don't span it.
                    if et == "response.output_audio.delta":
                        chunk = base64.b64decode(event.delta)
                        speaker.write(chunk)
                        status.set_state("speaking")
                        session.agent_chunks.append((t_now, chunk))
                        continue

                    # Skip every other streaming partial (`*.delta`) — they're
                    # too noisy and the terminal `*.done` event carries the
                    # complete payload.
                    if et.endswith(".delta"):
                        continue

                    # Every other event becomes its own span, carrying the
                    # event's payload. Work done while handling it (guardrail,
                    # tools) nests inside via tracing_context(parent=ev).
                    with _start_event(session, event, t_now) as ev:
                        if et == "input_audio_buffer.speech_started":
                            # Barge-in: flush whatever the agent was still saying.
                            speaker.clear()
                            status.set_state("hearing you")

                        elif et == "input_audio_buffer.speech_stopped":
                            status.set_state("transcribing")

                        elif et == "conversation.item.input_audio_transcription.completed":
                            transcript = (event.transcript or "").strip()
                            if not transcript:
                                # Noise — nothing usable was transcribed.
                                status.log("[openai] (no transcript — noise turn)")
                                status.set_state("listening")
                            else:
                                status.log(f"user:  {transcript}")
                                status.set_state("guardrail")
                                with tracing_context(parent=ev):
                                    verdict = await check_guardrail(transcript)

                                if verdict.blocked:
                                    status.log(
                                        f"[openai] guardrail tripped: {verdict.reason}"
                                    )
                                    status.set_state("refusing")
                                    await connection.response.create(
                                        response={"instructions": REFUSAL_INSTRUCTIONS}
                                    )
                                else:
                                    status.set_state("thinking")
                                    await connection.response.create()

                        elif et == "response.output_audio_transcript.done":
                            status.log(f"agent: {event.transcript or ''}")

                        elif et == "response.function_call_arguments.done":
                            pending_tool_calls.append(
                                (event.name, event.call_id, event.arguments)
                            )

                        elif et == "response.done":
                            if pending_tool_calls:
                                # Model emitted tool calls — run them nested
                                # under THIS event, then ask for the follow-up.
                                status.set_state("running tools")
                                calls, pending_tool_calls = pending_tool_calls, []
                                for name, call_id, raw_args in calls:
                                    with tracing_context(parent=ev):
                                        result = await _execute_tool(name, raw_args)
                                    await connection.conversation.item.create(
                                        item={
                                            "type": "function_call_output",
                                            "call_id": call_id,
                                            "output": json.dumps(result),
                                        }
                                    )
                                status.set_state("thinking")
                                await connection.response.create()
                            else:
                                status.set_state("listening")

                        elif et == "error":
                            status.log(f"[openai] server error: {event.error}")

        finally:
            try:
                mic_task.cancel()  # type: ignore[name-defined]
            except NameError:
                pass
            mic.stop()
            speaker.stop()
            _finalize_session(session)
            status.finish()

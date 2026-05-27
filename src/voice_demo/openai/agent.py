"""OpenAI Realtime API voice agent, traced with the LangSmith SDK.

Why SDK and not OTEL: OpenAI Realtime has no native telemetry — it's a raw
WebSocket event stream of `input_audio_buffer.*`, `response.*`, etc. We have
to build the trace from scratch either way, and the SDK gives us four things
OTEL can't: (1) multipart audio attachments (no base64 overhead), (2) mid-run
patching so a long response shows up in real time, (3) first-class interruption
status via tags + outputs on each run, (4) tool calls as proper child runs
instead of synthetic span attributes.

## Trace shape — one conversation = one trace

    realtime_session                          (root, opened at startup)
    │  attachments: conversation.wav         (stereo: L=user, R=agent)
    │
    ├── user_turn 1                          (opened on speech_started)
    │   │  attachments: user_utterance.wav   (just this user's utterance)
    │   │  metadata: turn_user_speech_duration_ms, turn_transcription_latency_ms,
    │   │            turn_think_latency_ms, turn_e2e_latency_ms
    │   ├── guardrail
    │   └── agent_response (1.1)             (one per response.create→done cycle)
    │       │  attachments: response_audio.wav
    │       │  usage_metadata: { input_tokens, output_tokens, total_tokens }
    │       │  metadata: response_ttfb_ms, response_duration_ms
    │       └── lookup_weather × N           (nested under the calling response)
    │
    ├── user_turn 2 [tag: interrupted]       (user spoke before turn 1 wrapped up)
    │   └── agent_response (2.1) [tag: cancelled]   (partial transcript + cut-off audio)
    │
    ├── user_turn 3 [tag: no_transcript]     (noise / cough — no usable speech)
    │       outputs: { noop: true }
    │
    └── user_turn 4
        ├── agent_response (4.1)             (emits tool calls)
        │   └── lookup_weather × 2
        └── agent_response (4.2)             (post-tool follow-up speech)

## Boundary choice

Turn = `speech_started → terminal response.done`. We use VAD events, not
transcription, because:
  - VAD is what the agent actually reacts to (it can cancel responses on
    `speech_started`, before any transcription exists).
  - Per-stage latency requires speech timestamps. If you key off
    `transcription.completed`, you've lost the "how long did the user talk",
    "how long did transcription take", and "what was the end-to-end perceived
    latency" signals — which are the most useful numbers for a voice agent.
  - Noise turns (no transcript) are real events — debugging "why didn't the
    agent hear me?" needs them in the trace, just tagged so they're filterable.
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
    turn_count: int = 0
    response_count: int = 0


@dataclass
class _Turn:
    """One user utterance and everything the agent did in response.

    Opened on `input_audio_buffer.speech_started`, finalized on the terminal
    `response.done` (or on the next `speech_started` if the user interrupted).
    """

    run: RunTree
    # Timestamps relative to session.t0 (seconds).
    speech_started_at: float
    speech_stopped_at: float | None = None
    transcription_at: float | None = None
    response_created_at: float | None = None
    first_response_audio_at: float | None = None

    user_transcript: str = ""
    # Per-turn user-utterance audio (speech_started → speech_stopped window).
    user_audio: bytearray = field(default_factory=bytearray)
    collecting_user_audio: bool = True

    blocked: bool = False
    block_reason: str = ""
    interrupted: bool = False
    agent_transcript: str = ""
    response_count: int = 0


@dataclass
class _Response:
    """One response.create→done cycle (a turn may have several)."""

    run: RunTree
    created_at: float
    first_audio_at: float | None = None
    done_at: float | None = None
    audio: bytearray = field(default_factory=bytearray)
    transcript: str = ""
    tool_calls: list[dict] = field(default_factory=list)
    cancelled: bool = False
    usage: dict | None = None


# ---------------------------------------------------------------------------
# WAV helpers
# ---------------------------------------------------------------------------

def _mono_pcm16_to_wav(data: bytes, sample_rate: int = SAMPLE_RATE) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(bytes(data))
    return buf.getvalue()


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
# Session / turn / response lifecycle
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
    metadata["turn_count"] = session.turn_count
    metadata["response_count"] = session.response_count
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


def _start_turn(session: _Session, t_started: float) -> _Turn:
    session.turn_count += 1
    # `create_child()` correctly sets trace_id, parent_run_id, and dotted_order
    # so this nests under the session root. `RunTree(parent=...)` would NOT —
    # `parent` isn't a recognized kwarg on the constructor and gets silently
    # ignored, leaving the run as a fresh root with no parent (the bug that
    # made every turn show up as a separate trace).
    run = session.run.create_child(
        name="user_turn",
        run_type="chain",
        inputs={
            "turn_index": session.turn_count,
            "speech_started_at_s": round(t_started, 3),
        },
        tags=["turn"],
        extra={"metadata": {
            "thread_id": session.thread_id,
            "turn_index": session.turn_count,
        }},
    )
    run.post()
    return _Turn(run=run, speech_started_at=t_started)


def _finalize_turn(turn: _Turn) -> None:
    outputs: dict[str, Any] = {
        "user_transcript": turn.user_transcript,
        "agent_transcript": turn.agent_transcript,
        "response_count": turn.response_count,
    }
    if not turn.user_transcript:
        outputs["noop"] = True
    if turn.blocked:
        outputs["blocked"] = True
        outputs["block_reason"] = turn.block_reason
    if turn.interrupted:
        outputs["interrupted"] = True

    # Per-stage latencies → metadata.turn_*  (mirrors livekit-demo's namespace).
    extra: dict[str, Any] = turn.run.extra or {}
    metadata: dict[str, Any] = dict(extra.get("metadata") or {})
    if turn.speech_stopped_at is not None:
        metadata["turn_user_speech_duration_ms"] = int(
            (turn.speech_stopped_at - turn.speech_started_at) * 1000
        )
    if turn.transcription_at is not None and turn.speech_stopped_at is not None:
        metadata["turn_transcription_latency_ms"] = int(
            (turn.transcription_at - turn.speech_stopped_at) * 1000
        )
    if turn.response_created_at is not None and turn.transcription_at is not None:
        metadata["turn_think_latency_ms"] = int(
            (turn.response_created_at - turn.transcription_at) * 1000
        )
    if turn.first_response_audio_at is not None and turn.speech_stopped_at is not None:
        # The number users actually perceive: how long after they stopped
        # talking before the agent started talking.
        metadata["turn_e2e_latency_ms"] = int(
            (turn.first_response_audio_at - turn.speech_stopped_at) * 1000
        )
    extra["metadata"] = metadata
    turn.run.extra = extra

    if turn.user_audio:
        turn.run.attachments = {
            "user_utterance": ("audio/wav", _mono_pcm16_to_wav(bytes(turn.user_audio))),
        }

    tags = list(turn.run.tags or [])
    if turn.interrupted and "interrupted" not in tags:
        tags.append("interrupted")
    if not turn.user_transcript and "no_transcript" not in tags:
        tags.append("no_transcript")
    turn.run.tags = tags

    # Patch the latest inputs in case transcription updated them.
    turn.run.inputs = {
        "turn_index": metadata.get("turn_index"),
        "user_transcript": turn.user_transcript,
        "speech_started_at_s": round(turn.speech_started_at, 3),
    }
    turn.run.end(outputs=outputs)
    turn.run.patch()


def _start_response(session: _Session, turn: _Turn, t_created: float) -> _Response:
    session.response_count += 1
    turn.response_count += 1
    # Same as in _start_turn — must use create_child(), not RunTree(parent=...).
    run = turn.run.create_child(
        name="agent_response",
        # llm-kind so usage_metadata renders as token counts in the UI.
        run_type="llm",
        inputs={
            "model": DEFAULT_MODEL,
            "user_transcript": turn.user_transcript,
            "turn_response_index": turn.response_count,
        },
        tags=["response"],
    )
    run.post()
    return _Response(run=run, created_at=t_created)


def _finalize_response(resp: _Response) -> None:
    outputs: dict[str, Any] = {"transcript": resp.transcript}
    if resp.tool_calls:
        outputs["tool_calls"] = resp.tool_calls
    if resp.cancelled:
        outputs["cancelled"] = True

    extra: dict[str, Any] = resp.run.extra or {}
    metadata: dict[str, Any] = dict(extra.get("metadata") or {})
    if resp.first_audio_at is not None:
        metadata["response_ttfb_ms"] = int(
            (resp.first_audio_at - resp.created_at) * 1000
        )
    if resp.done_at is not None:
        metadata["response_duration_ms"] = int(
            (resp.done_at - resp.created_at) * 1000
        )
    if resp.usage:
        metadata["usage"] = resp.usage
        # LangSmith reads `usage_metadata` to render tokens + cost.
        extra["usage_metadata"] = {
            "input_tokens": int(resp.usage.get("input_tokens") or 0),
            "output_tokens": int(resp.usage.get("output_tokens") or 0),
            "total_tokens": int(resp.usage.get("total_tokens") or 0),
        }
    extra["metadata"] = metadata
    resp.run.extra = extra

    if resp.audio:
        resp.run.attachments = {
            "response_audio": ("audio/wav", _mono_pcm16_to_wav(bytes(resp.audio))),
        }

    if resp.cancelled:
        tags = list(resp.run.tags or [])
        if "cancelled" not in tags:
            tags.append("cancelled")
        resp.run.tags = tags

    resp.run.end(outputs=outputs)
    resp.run.patch()


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


def _coerce_usage(raw: Any) -> dict | None:
    if raw is None:
        return None
    if hasattr(raw, "model_dump"):
        return raw.model_dump()
    if isinstance(raw, dict):
        return raw
    try:
        return dict(raw)
    except (TypeError, ValueError):
        return None


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
    turn: _Turn | None = None
    resp: _Response | None = None
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
                        t = _now()
                        # Always record into the session's timestamped timeline.
                        session.user_chunks.append((t, frame))
                        # While the user is actively speaking (between
                        # speech_started and speech_stopped), also record
                        # into the per-turn utterance buffer.
                        if turn is not None and turn.collecting_user_audio:
                            turn.user_audio.extend(frame)
                        status.update_level(frame_level(frame))

                mic_task = asyncio.create_task(pump_mic())

                async for event in connection:
                    et = event.type
                    t_now = _now()

                    if et == "input_audio_buffer.speech_started":
                        # New utterance. Always opens a new turn — if there
                        # was a turn still in flight, the user is interrupting.
                        speaker.clear()
                        status.set_state("hearing you")

                        if resp is not None:
                            resp.cancelled = True
                            resp.done_at = t_now
                            _finalize_response(resp)
                            resp = None
                        if turn is not None:
                            turn.interrupted = True
                            turn.collecting_user_audio = False
                            _finalize_turn(turn)
                            turn = None

                        turn = _start_turn(session, t_now)

                    elif et == "input_audio_buffer.speech_stopped":
                        if turn is not None:
                            turn.speech_stopped_at = t_now
                            turn.collecting_user_audio = False
                        status.set_state("transcribing")

                    elif et == "conversation.item.input_audio_transcription.completed":
                        transcript = (event.transcript or "").strip()
                        if turn is None:
                            # Shouldn't happen — transcription without a
                            # turn means we missed speech_started. Skip.
                            continue
                        turn.transcription_at = t_now

                        if not transcript:
                            # Noise turn — finalize as noop. Stays in the
                            # trace tagged `no_transcript` for debugging.
                            status.log("[openai] (no transcript — noise turn)")
                            _finalize_turn(turn)
                            turn = None
                            status.set_state("listening")
                            continue

                        status.log(f"user:  {transcript}")
                        turn.user_transcript = transcript

                        status.set_state("guardrail")
                        with tracing_context(parent=turn.run):
                            verdict = await check_guardrail(transcript)

                        if verdict.blocked:
                            status.log(f"[openai] guardrail tripped: {verdict.reason}")
                            turn.blocked = True
                            turn.block_reason = verdict.reason
                            status.set_state("refusing")
                            await connection.response.create(
                                response={"instructions": REFUSAL_INSTRUCTIONS}
                            )
                        else:
                            status.set_state("thinking")
                            await connection.response.create()

                    elif et == "response.created":
                        if turn is not None:
                            if turn.response_created_at is None:
                                turn.response_created_at = t_now
                            resp = _start_response(session, turn, t_now)

                    elif et == "response.output_audio.delta":
                        chunk = base64.b64decode(event.delta)
                        speaker.write(chunk)
                        status.set_state("speaking")
                        session.agent_chunks.append((t_now, chunk))
                        if resp is not None:
                            if resp.first_audio_at is None:
                                resp.first_audio_at = t_now
                            resp.audio.extend(chunk)
                        if turn is not None and turn.first_response_audio_at is None:
                            turn.first_response_audio_at = t_now

                    elif et == "response.output_audio_transcript.done":
                        text = event.transcript or ""
                        status.log(f"agent: {text}")
                        if resp is not None:
                            resp.transcript = text
                        if turn is not None:
                            if turn.agent_transcript:
                                turn.agent_transcript += " "
                            turn.agent_transcript += text

                    elif et == "response.function_call_arguments.done":
                        pending_tool_calls.append(
                            (event.name, event.call_id, event.arguments)
                        )

                    elif et == "response.done":
                        if resp is not None:
                            resp.done_at = t_now
                            resp.usage = _coerce_usage(
                                getattr(getattr(event, "response", None), "usage", None)
                            )

                        if pending_tool_calls:
                            # Intermediate response.done — model emitted tool
                            # calls. Run them nested under THIS response, then
                            # close it so the follow-up becomes its own child.
                            status.set_state("running tools")
                            calls, pending_tool_calls = pending_tool_calls, []
                            for name, call_id, raw_args in calls:
                                parent = resp.run if resp is not None else (
                                    turn.run if turn is not None else None
                                )
                                with tracing_context(parent=parent):
                                    result = await _execute_tool(name, raw_args)
                                if resp is not None:
                                    resp.tool_calls.append(
                                        {"name": name, "arguments": raw_args, "result": result}
                                    )
                                await connection.conversation.item.create(
                                    item={
                                        "type": "function_call_output",
                                        "call_id": call_id,
                                        "output": json.dumps(result),
                                    }
                                )
                            if resp is not None:
                                _finalize_response(resp)
                                resp = None
                            status.set_state("thinking")
                            await connection.response.create()
                        else:
                            # Terminal response.done — close response + turn.
                            if resp is not None:
                                _finalize_response(resp)
                                resp = None
                            if turn is not None:
                                _finalize_turn(turn)
                                turn = None
                            status.set_state("listening")

                    elif et == "error":
                        status.log(f"[openai] server error: {event.error}")

        finally:
            # Close children before the session root.
            if resp is not None:
                resp.cancelled = True
                resp.done_at = _now()
                _finalize_response(resp)
            if turn is not None:
                turn.interrupted = True
                turn.collecting_user_audio = False
                _finalize_turn(turn)
            try:
                mic_task.cancel()  # type: ignore[name-defined]
            except NameError:
                pass
            mic.stop()
            speaker.stop()
            _finalize_session(session)
            status.finish()

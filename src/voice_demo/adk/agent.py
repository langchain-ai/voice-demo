"""Google ADK Live voice agent, traced with the LangSmith SDK.

Why SDK not OTEL
----------------
ADK emits standard `gen_ai.*` OTel spans for its non-live execution paths
(Runner.run, BaseAgent.run_async, FunctionTool.run_async, etc.) — but
`Runner.run_live`, which is the entire voice loop, doesn't appear to be
instrumented. An OTel-only setup produces a single empty root span and
nothing inside it. So we treat ADK's `run_live` event stream the same way
the OpenAI backend treats OpenAI Realtime's WebSocket events: observe each
event, build the trace explicitly with RunTree.

That refines the demo's "OTel vs SDK" rule:

    OTel  → framework runs in-process and emits its own spans (LiveKit)
    SDK   → framework hands us an event stream from a remote service
            (OpenAI Realtime, ADK Live)

Trace shape
-----------
One conversation = one trace:

    realtime_session                       (root; conversation.wav stereo)
    │   metadata: thread_id, model, turn_count, duration_s
    │
    ├── user_turn 1                        (opens on first event after last
    │   │                                   turn_complete; closes on
    │   │                                   turn_complete)
    │   │   attachments: user_utterance.wav
    │   │   metadata: turn_duration_ms, agent_audio_duration_ms,
    │   │             e2e_latency_ms (rough — first input_transcription to
    │   │             first agent audio)
    │   │   inputs:  user_transcript
    │   │   outputs: agent_transcript
    │   └── execute_tool: get_weather      (per function_call event)
    │
    ├── user_turn 2 [tag: interrupted]
    │   └── ...
    └── user_turn 3 [tag: no_transcript]   (noise / cough; outputs.noop = true)

What ADK doesn't give us
------------------------
No explicit `speech_started` / `speech_stopped` events, so fine-grained
per-stage latencies (transcription latency, think latency) aren't observable
the way they are for OpenAI Realtime. We get coarser turn-level timing.
"""

from __future__ import annotations

import asyncio
import io
import logging
import math
import os
import sys
import time
import uuid
import warnings
import wave
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any
from zoneinfo import ZoneInfo

# ADK prints noisy startup messages (an experimental-feature warning, plus a
# log line about MCP not being installed) that smash into our status line.
# Mute them before importing anything from google.adk.
warnings.filterwarnings("ignore", category=UserWarning, module="google.adk")
logging.getLogger("google_adk").setLevel(logging.ERROR)
logging.getLogger("google.adk").setLevel(logging.ERROR)

import numpy as np
from langsmith import RunTree
from langsmith.run_helpers import tracing_context
from scipy import signal as scipy_signal

from ..audio import MicStream, SpeakerStream
from ..console import ConsoleStatus, frame_level


APP_NAME = "voice-demo-adk"
USER_ID = "console-user"

SEND_SAMPLE_RATE = 16_000  # ADK Live wants 16 kHz in
RECV_SAMPLE_RATE = 24_000  # …and 24 kHz out

MODEL = os.getenv("ADK_LIVE_MODEL", "gemini-2.5-flash-native-audio-latest")


# ---------------------------------------------------------------------------
# Tools — ADK introspects type hints + docstrings to build the tool schema.
# ---------------------------------------------------------------------------

def get_time(timezone: str = "America/Los_Angeles") -> dict:
    """Return the current time in the requested IANA timezone."""
    try:
        tz = ZoneInfo(timezone)
    except Exception:
        return {"error": f"Unknown timezone: {timezone}"}
    now = datetime.now(tz)
    return {
        "timezone": timezone,
        "iso": now.isoformat(),
        "human": now.strftime("%A, %B %d %Y at %I:%M %p %Z"),
    }


def get_weather(city: str) -> dict:
    """Return a stubbed weather report for the given city.

    (Demo stub — ADK's tool dispatch runs the function synchronously, so we
    don't call out to Open-Meteo here. Use the OpenAI backend for the real
    weather lookup.)
    """
    return {
        "city": city,
        "temperature_f": 72,
        "condition": "sunny with light clouds",
        "note": "Stubbed demo data, not a real forecast.",
    }


_INSTRUCTIONS = (
    "You are a friendly voice assistant speaking with the user in real time. "
    "Keep responses short and natural — one or two sentences. "
    "Use the get_time tool when asked about the current time, and the "
    "get_weather tool when asked about weather."
)


# ---------------------------------------------------------------------------
# Run-tree state (same shape as the OpenAI backend, adapted to ADK events)
# ---------------------------------------------------------------------------

@dataclass
class _Session:
    run: RunTree
    thread_id: str
    project_name: str
    t0: float
    # Time-stamped audio chunks for the stereo conversation WAV.
    user_chunks: list[tuple[float, bytes]] = field(default_factory=list)
    agent_chunks: list[tuple[float, bytes]] = field(default_factory=list)
    turn_count: int = 0


@dataclass
class _Turn:
    run: RunTree
    started_at: float
    user_transcript: str = ""
    agent_transcript: str = ""
    user_audio: bytearray = field(default_factory=bytearray)
    collecting_user_audio: bool = True
    first_input_transcription_at: float | None = None
    first_agent_audio_at: float | None = None
    last_agent_audio_at: float | None = None
    interrupted: bool = False
    tool_runs: dict[str, RunTree] = field(default_factory=dict)
    tool_calls: list[dict] = field(default_factory=list)


# ---------------------------------------------------------------------------
# WAV helpers (lifted from openai/agent.py — same audio model)
# ---------------------------------------------------------------------------

def _mono_pcm16_to_wav(data: bytes, sample_rate: int) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(bytes(data))
    return buf.getvalue()


def _layout_chunks_to_play_time(
    chunks: list[tuple[float, bytes]], sample_rate: int
) -> list[tuple[float, bytes]]:
    """Rewrite receipt timestamps into natural-play timestamps.

    Agent audio chunks arrive over WebSocket in bursts (server sends faster
    than realtime). Naively placing them at receipt time makes them overlap
    and overwrite each other. Place each chunk at max(prev_end, receipt_time)
    so bursts play consecutively and real gaps between bursts are preserved.
    """
    out: list[tuple[float, bytes]] = []
    cur_time = 0.0
    for i, (t_recv, data) in enumerate(chunks):
        if i == 0:
            cur_time = t_recv
        else:
            cur_time = max(cur_time, t_recv)
        out.append((cur_time, data))
        cur_time += (len(data) // 2) / sample_rate
    return out


def _build_stereo_session_wav(session: _Session) -> bytes:
    """Stereo WAV: channel 0 = user, channel 1 = agent.

    User and agent are captured at different sample rates (24k mic, 24k agent
    output) — both happen to be 24k so we use that as the file rate.
    """
    if not session.user_chunks and not session.agent_chunks:
        return b""

    sr = RECV_SAMPLE_RATE
    user = _layout_chunks_to_play_time(session.user_chunks, sr)
    agent = _layout_chunks_to_play_time(session.agent_chunks, sr)

    def chunk_end(t: float, data: bytes) -> float:
        return t + (len(data) // 2) / sr

    user_end = max((chunk_end(t, d) for t, d in user), default=0.0)
    agent_end = max((chunk_end(t, d) for t, d in agent), default=0.0)
    total_samples = int(math.ceil(max(user_end, agent_end) * sr))

    stereo = np.zeros((total_samples, 2), dtype=np.int16)

    def write_channel(chunks: list[tuple[float, bytes]], channel: int) -> None:
        for t, data in chunks:
            offset = int(t * sr)
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
        wf.setframerate(sr)
        wf.writeframes(stereo.tobytes())
    return buf.getvalue()


# ---------------------------------------------------------------------------
# Session / turn / tool lifecycle
# ---------------------------------------------------------------------------

def _start_session(thread_id: str, project_name: str) -> _Session:
    run = RunTree(
        name="realtime_session",
        run_type="chain",
        inputs={},
        project_name=project_name,
        tags=["voice-demo", "adk", "session"],
        extra={"metadata": {"thread_id": thread_id, "model": MODEL}},
    )
    run.post()
    return _Session(run=run, thread_id=thread_id, project_name=project_name, t0=time.monotonic())


def _finalize_session(session: _Session) -> None:
    extra: dict[str, Any] = session.run.extra or {}
    metadata: dict[str, Any] = dict(extra.get("metadata") or {})
    metadata["turn_count"] = session.turn_count
    metadata["duration_s"] = round(time.monotonic() - session.t0, 2)
    extra["metadata"] = metadata
    session.run.extra = extra

    stereo_wav = _build_stereo_session_wav(session)
    if stereo_wav:
        session.run.attachments = {
            "conversation": ("audio/wav", stereo_wav),
        }
    session.run.end(outputs={})
    session.run.patch()


def _start_turn(session: _Session, started_at: float) -> _Turn:
    session.turn_count += 1
    run = session.run.create_child(
        name="user_turn",
        run_type="chain",
        inputs={
            "turn_index": session.turn_count,
            "started_at_s": round(started_at, 3),
        },
        tags=["turn"],
        extra={"metadata": {
            "thread_id": session.thread_id,
            "turn_index": session.turn_count,
        }},
    )
    run.post()
    return _Turn(run=run, started_at=started_at)


def _finalize_turn(turn: _Turn, completed_at: float) -> None:
    outputs: dict[str, Any] = {
        "user_transcript": turn.user_transcript,
        "agent_transcript": turn.agent_transcript,
    }
    if not turn.user_transcript:
        outputs["noop"] = True
    if turn.interrupted:
        outputs["interrupted"] = True
    if turn.tool_calls:
        outputs["tool_calls"] = turn.tool_calls

    extra: dict[str, Any] = turn.run.extra or {}
    metadata: dict[str, Any] = dict(extra.get("metadata") or {})
    metadata["turn_duration_ms"] = int((completed_at - turn.started_at) * 1000)
    if turn.first_agent_audio_at is not None and turn.last_agent_audio_at is not None:
        metadata["agent_audio_duration_ms"] = int(
            (turn.last_agent_audio_at - turn.first_agent_audio_at) * 1000
        )
    if (
        turn.first_agent_audio_at is not None
        and turn.first_input_transcription_at is not None
    ):
        # Rough perceived latency — time from first user transcription chunk
        # to first agent audio. Coarser than OpenAI's `speech_stopped → first
        # audio` because we lack explicit speech_stopped here.
        metadata["e2e_latency_ms"] = int(
            (turn.first_agent_audio_at - turn.first_input_transcription_at) * 1000
        )
    extra["metadata"] = metadata
    turn.run.extra = extra

    if turn.user_audio:
        turn.run.attachments = {
            "user_utterance": (
                "audio/wav",
                _mono_pcm16_to_wav(bytes(turn.user_audio), RECV_SAMPLE_RATE),
            ),
        }

    tags = list(turn.run.tags or [])
    if turn.interrupted and "interrupted" not in tags:
        tags.append("interrupted")
    if not turn.user_transcript and "no_transcript" not in tags:
        tags.append("no_transcript")
    turn.run.tags = tags

    turn.run.inputs = {
        "turn_index": metadata.get("turn_index"),
        "user_transcript": turn.user_transcript,
        "started_at_s": round(turn.started_at, 3),
    }
    turn.run.end(outputs=outputs)
    turn.run.patch()


def _start_tool(turn: _Turn, name: str, args: dict, t: float) -> RunTree:
    run = turn.run.create_child(
        name=f"execute_tool: {name}",
        run_type="tool",
        inputs={"args": args},
        tags=["tool"],
    )
    run.post()
    return run


def _finalize_tool(tool_run: RunTree, response: Any) -> None:
    tool_run.end(outputs={"response": response})
    tool_run.patch()


# ---------------------------------------------------------------------------
# Audio resampling — our streams are 24k; ADK wants 16k on input.
# ---------------------------------------------------------------------------

def _resample_pcm16(data: bytes, src_rate: int, dst_rate: int) -> bytes:
    if src_rate == dst_rate:
        return data
    samples = np.frombuffer(data, dtype=np.int16)
    if samples.size == 0:
        return b""
    n_out = int(round(samples.size * dst_rate / src_rate))
    out = scipy_signal.resample(samples, n_out).astype(np.int16)
    return out.tobytes()


# ---------------------------------------------------------------------------
# Main run loop
# ---------------------------------------------------------------------------

async def run(project_name: str) -> None:
    if not os.environ.get("GOOGLE_API_KEY"):
        print("GOOGLE_API_KEY is not set.", file=sys.stderr)
        sys.exit(1)

    # Lazy ADK imports — quiets startup until we know we'll use them.
    from google.adk.agents import LlmAgent
    from google.adk.agents.live_request_queue import LiveRequestQueue
    from google.adk.agents.run_config import RunConfig, StreamingMode
    from google.adk.runners import Runner
    from google.adk.sessions import InMemorySessionService
    from google.genai import types as genai_types

    root_agent = LlmAgent(
        name="voice_assistant",
        model=MODEL,
        instruction=_INSTRUCTIONS,
        tools=[get_time, get_weather],
    )

    session_service = InMemorySessionService()
    adk_session = await session_service.create_session(app_name=APP_NAME, user_id=USER_ID)
    runner = Runner(
        app_name=APP_NAME, agent=root_agent, session_service=session_service
    )
    queue = LiveRequestQueue()

    mic = MicStream(sample_rate=RECV_SAMPLE_RATE)
    speaker = SpeakerStream(sample_rate=RECV_SAMPLE_RATE)

    thread_id = str(uuid.uuid4())
    status = ConsoleStatus()
    status.log(f"[adk] thread_id={thread_id}")
    status.log(f"[adk] connected with model={MODEL}.")
    status.log("[adk] talk into your mic — Ctrl-C to quit.")

    session = _start_session(thread_id, project_name)
    turn: _Turn | None = None
    # Most recent agent audio chunk receipt — used to distinguish real user
    # interrupts from speaker bleed transcriptions (see input_transcription
    # handler below).
    last_agent_chunk_at = 0.0
    AGENT_QUIET_THRESHOLD_S = 0.5

    def _now() -> float:
        return time.monotonic() - session.t0

    mic.start()
    speaker.start()
    status.set_state("listening")

    stop = asyncio.Event()

    async def pump_mic() -> None:
        async for frame in mic.frames():
            # Resample 24k → 16k for ADK Live.
            resampled = _resample_pcm16(frame, RECV_SAMPLE_RATE, SEND_SAMPLE_RATE)
            queue.send_realtime(
                genai_types.Blob(
                    data=resampled, mime_type=f"audio/pcm;rate={SEND_SAMPLE_RATE}"
                )
            )
            status.update_level(frame_level(frame))
            # Always record into the session-level timeline (for the
            # conversation stereo WAV). Per-turn user audio when active.
            t = _now()
            session.user_chunks.append((t, frame))
            if turn is not None and turn.collecting_user_audio:
                turn.user_audio.extend(frame)

    async def pump_responses() -> None:
        nonlocal turn, last_agent_chunk_at

        run_config = RunConfig(
            response_modalities=["AUDIO"],
            streaming_mode=StreamingMode.BIDI,
            input_audio_transcription=genai_types.AudioTranscriptionConfig(),
            output_audio_transcription=genai_types.AudioTranscriptionConfig(),
            realtime_input_config=genai_types.RealtimeInputConfig(
                automatic_activity_detection=genai_types.AutomaticActivityDetection(
                    start_of_speech_sensitivity=genai_types.StartSensitivity.START_SENSITIVITY_HIGH,
                    end_of_speech_sensitivity=genai_types.EndSensitivity.END_SENSITIVITY_LOW,
                    prefix_padding_ms=200,
                    silence_duration_ms=800,
                ),
            ),
        )

        user_buf: list[str] = []
        agent_buf: list[str] = []

        def ensure_turn() -> _Turn:
            nonlocal turn
            if turn is None:
                turn = _start_turn(session, _now())
            return turn

        try:
            async for event in runner.run_live(
                user_id=USER_ID,
                session_id=adk_session.id,
                live_request_queue=queue,
                run_config=run_config,
            ):
                t_now = _now()

                if getattr(event, "interrupted", False):
                    speaker.clear()
                    status.set_state("hearing you")
                    if turn is not None:
                        turn.interrupted = True

                content = getattr(event, "content", None)
                if content and getattr(content, "parts", None):
                    for part in content.parts:
                        # Agent audio chunk
                        inline = getattr(part, "inline_data", None)
                        if inline and inline.data:
                            chunk = inline.data
                            speaker.write(chunk)
                            status.set_state("speaking")
                            session.agent_chunks.append((t_now, chunk))
                            last_agent_chunk_at = time.monotonic()
                            t = ensure_turn()
                            if t.first_agent_audio_at is None:
                                t.first_agent_audio_at = t_now
                            t.last_agent_audio_at = t_now

                        # Tool call (server invoked a function)
                        fc = getattr(part, "function_call", None)
                        if fc is not None and getattr(fc, "name", None):
                            t = ensure_turn()
                            args = dict(fc.args) if getattr(fc, "args", None) else {}
                            call_id = getattr(fc, "id", None) or fc.name
                            tool_run = _start_tool(t, fc.name, args, t_now)
                            t.tool_runs[call_id] = tool_run
                            t.tool_calls.append({"name": fc.name, "args": args})

                        # Tool response (server returning the function's output)
                        fr = getattr(part, "function_response", None)
                        if fr is not None and getattr(fr, "name", None):
                            response = getattr(fr, "response", None)
                            t = ensure_turn()
                            call_id = getattr(fr, "id", None) or fr.name
                            tool_run = t.tool_runs.pop(call_id, None)
                            if tool_run is not None:
                                _finalize_tool(tool_run, response)
                            # Stamp the result onto the last matching call entry.
                            for tc in reversed(t.tool_calls):
                                if tc.get("name") == fr.name and "result" not in tc:
                                    tc["result"] = response
                                    break

                in_tx = getattr(event, "input_transcription", None)
                if in_tx and in_tx.text:
                    user_buf.append(in_tx.text)
                    t = ensure_turn()
                    if t.first_input_transcription_at is None:
                        t.first_input_transcription_at = t_now
                    # Disambiguate real interrupt vs mic bleed (Gemini
                    # transcribes whatever the mic picks up — including
                    # speaker bleed). If agent audio is still arriving
                    # within QUIET_THRESHOLD, this is almost certainly bleed.
                    quiet_for = time.monotonic() - last_agent_chunk_at
                    speaker_has_audio = speaker.buffered_bytes() > 0
                    if not speaker_has_audio:
                        status.set_state("hearing you")
                    elif quiet_for > AGENT_QUIET_THRESHOLD_S:
                        # Drain phase — flush stale audio so the interrupt feels real.
                        speaker.clear()
                        status.set_state("hearing you")
                    # else: active generation, likely bleed — ignore.

                out_tx = getattr(event, "output_transcription", None)
                if out_tx and out_tx.text:
                    agent_buf.append(out_tx.text)
                    ensure_turn()

                if getattr(event, "turn_complete", False):
                    user_text = "".join(user_buf).strip()
                    agent_text = "".join(agent_buf).strip()
                    if user_text:
                        status.log(f"user:  {user_text}")
                    if agent_text:
                        status.log(f"agent: {agent_text}")
                    user_buf.clear()
                    agent_buf.clear()

                    if turn is not None:
                        turn.user_transcript = user_text
                        turn.agent_transcript = agent_text
                        # Best-effort close-out for any tools that didn't
                        # see a matching function_response (shouldn't happen
                        # normally — defensive).
                        for tool_run in turn.tool_runs.values():
                            _finalize_tool(tool_run, {"error": "no response captured"})
                        turn.tool_runs.clear()
                        _finalize_turn(turn, t_now)
                        turn = None

                    status.set_state("listening")
        except asyncio.CancelledError:
            pass

    with tracing_context(
        metadata={"thread_id": thread_id},
        tags=["voice-demo", "adk"],
        project_name=project_name,
    ):
        mic_task = asyncio.create_task(pump_mic())
        play_task = asyncio.create_task(pump_responses())

        try:
            await stop.wait()
        except asyncio.CancelledError:
            pass
        finally:
            queue.close()
            for t in (mic_task, play_task):
                t.cancel()
            await asyncio.gather(mic_task, play_task, return_exceptions=True)
            mic.stop()
            speaker.stop()
            status.finish()

            if turn is not None:
                turn.interrupted = True
                _finalize_turn(turn, _now())
            _finalize_session(session)

"""OpenAI Realtime API voice agent, traced with the LangSmith SDK.

Why SDK and not OTEL: OpenAI Realtime has no native telemetry — it's a raw
WebSocket event stream of `input_audio_buffer.*`, `response.*`, etc. We trace it
with the LangSmith voice integration `langsmith.integrations.openai_realtime.wrap_realtime`,
which wraps the connection and builds the trace directly from that stream — so the
trace mirrors *exactly what the server sent us*.

## Frontend-agnostic

`run()` takes its audio frontend by dependency injection: an `AudioInput`
(mic), an `AudioOutput` (speaker), and an optional `StatusUI`. The console CLI
supplies the local-machine implementations, but the agent itself imports no
console code.

## Tracing is one wrapper

`wrap_realtime(connection)` returns a transparent proxy: the event loop below is
left untouched (`async for event in connection`), and the wrapper owns every
trace decision. The app only (optionally) feeds audio for the stereo WAV via
`connection.record_user_audio` / `record_agent_audio` and supplies
`is_agent_speaking` so barge-ins are flagged. See the SDK module for the trace
shape.
"""

from __future__ import annotations

import asyncio
import base64
import json
import os
import sys
import uuid

from openai import AsyncOpenAI

from ..audio import AudioInput, AudioOutput
from ..console import NullUI, StatusUI, frame_level
from langsmith.integrations.openai_realtime import wrap_realtime
from .utils import execute_tool

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


async def run(
    project_name: str,
    *,
    audio_in: AudioInput,
    audio_out: AudioOutput,
    ui: StatusUI | None = None,
) -> None:
    """Drive an OpenAI Realtime conversation over the given audio frontend.

    Args:
        project_name: LangSmith project the trace lands in.
        audio_in: PCM16 mic source (24 kHz).
        audio_out: PCM16 speaker sink (24 kHz); `clear()` is called on barge-in.
        ui: Optional status observer; defaults to a no-op for headless use.
    """
    if not os.getenv("OPENAI_API_KEY"):
        print("OPENAI_API_KEY is not set.", file=sys.stderr)
        sys.exit(1)

    ui = ui or NullUI()
    client = AsyncOpenAI()
    thread_id = str(uuid.uuid4())

    ui.log(f"[openai] thread_id={thread_id}")
    ui.log("[openai] connecting to OpenAI Realtime API...")

    mic_task: asyncio.Task | None = None
    try:
        # `wrap_realtime` builds the LangSmith trace from the event stream; the
        # loop below is pure application behavior.
        async with client.realtime.connect(model=DEFAULT_MODEL) as raw, wrap_realtime(
            raw,
            thread_id=thread_id,
            sample_rate=SAMPLE_RATE,
            project_name=project_name,
            tags=["voice-demo", "openai"],
            metadata={"model": DEFAULT_MODEL},
            is_agent_speaking=lambda: audio_out.buffered_bytes() > 0,
        ) as connection:
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

            # Record the AGENT side at the speaker, not at receipt: this callback
            # fires with the bytes actually played, so audio that barge-in
            # flushes (clear()) is never recorded — the WAV reflects what was
            # *heard*, not what was generated.
            audio_out.set_played_callback(connection.record_agent_audio)

            audio_in.start()
            audio_out.start()
            ui.log("[openai] connected. Talk into your mic — Ctrl-C to quit.")
            ui.set_state("listening")

            async def pump_mic() -> None:
                async for frame in audio_in.frames():
                    await connection.input_audio_buffer.append(
                        audio=base64.b64encode(frame).decode("ascii")
                    )
                    # Record into the session's timestamped timeline for the
                    # stereo conversation WAV.
                    connection.record_user_audio(frame)
                    ui.update_level(frame_level(frame))

            mic_task = asyncio.create_task(pump_mic())

            # Application event loop — pure application behavior (play audio, set
            # UI state, run tools, ask the model to respond). Tool calls
            # auto-nest under the event's span because `wrap_realtime` keeps it
            # the active LangSmith context while the body runs.
            async for event in connection:
                event_type = event.type

                if event_type == "response.output_audio.delta":
                    # Hundreds of chunks per response — play each as it arrives.
                    audio_out.write(base64.b64decode(event.delta))
                    ui.set_state("speaking")

                elif event_type == "input_audio_buffer.speech_started":
                    # Barge-in: flush whatever the agent was still saying.
                    audio_out.clear()
                    ui.set_state("hearing you")

                elif event_type == "input_audio_buffer.speech_stopped":
                    ui.set_state("transcribing")

                elif (
                    event_type
                    == "conversation.item.input_audio_transcription.completed"
                ):
                    transcript = (event.transcript or "").strip()
                    if not transcript:
                        ui.log("[openai] (no speech detected)")
                        ui.set_state("listening")
                    else:
                        ui.log(f"user:  {transcript}")
                        # Turn detection uses create_response=False, so we
                        # explicitly ask the model to respond once transcribed.
                        ui.set_state("thinking")
                        await connection.response.create()

                elif (
                    event_type
                    == "conversation.item.input_audio_transcription.failed"
                ):
                    ui.log(
                        f"[openai] transcription failed: {getattr(event, 'error', None)}"
                    )
                    ui.set_state("listening")

                elif event_type == "response.output_audio_transcript.done":
                    transcript = (event.transcript or "").strip()
                    if transcript:
                        ui.log(f"agent: {transcript}")

                elif event_type == "response.done":
                    # The done event carries the complete response, including any
                    # function calls — no need to collect them from streaming.
                    tool_calls = [
                        item
                        for item in (event.response.output or [])
                        if item.type == "function_call"
                    ]
                    if tool_calls:
                        # Run them, then ask for the spoken follow-up.
                        ui.set_state("running tools")
                        for call in tool_calls:
                            result = await execute_tool(call.name, call.arguments)
                            await connection.conversation.item.create(
                                item={
                                    "type": "function_call_output",
                                    "call_id": call.call_id,
                                    "output": json.dumps(result),
                                }
                            )
                        ui.set_state("thinking")
                        await connection.response.create()
                    else:
                        ui.set_state("listening")

                elif event_type == "error":
                    ui.log(f"[openai] server error: {event.error}")

    except Exception as exc:
        # The error is recorded on the root span by `wrap_realtime`'s teardown;
        # here we just surface it to the console.
        ui.log(f"[openai] error: {exc}")
    finally:
        if mic_task is not None:
            mic_task.cancel()
        audio_in.stop()
        audio_out.stop()
        ui.finish()

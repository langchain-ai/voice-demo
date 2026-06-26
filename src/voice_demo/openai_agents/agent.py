"""OpenAI Realtime via the **Agents SDK**, traced with the LangSmith SDK.

Same conversation as `voice_demo.openai` — same model, tool, console transport —
but driven through the OpenAI Agents SDK (`agents.realtime`) instead of a
hand-written `client.realtime.connect()` loop. The `RealtimeRunner` owns the
tool-call loop; we feed mic PCM with `session.send_audio()` and consume a stream
of **semantic** events.

## Tracing is one wrapper

The Agents SDK's own realtime tracing is server-side (it uploads to OpenAI's
dashboard), and the batch `OpenAIAgentsTracingProcessor` sees nothing for a
realtime session. We trace via the LangSmith voice integration
`langsmith.integrations.openai_realtime.wrap_realtime_session`, which wraps the
session and builds the trace from its semantic event stream. The loop below is pure
application behavior; the wrapper owns every trace decision. We pass `on_message`
so finalized transcript lines still print to the console, and feed audio for the
stereo WAV via `conn.record_user_audio` / `record_agent_audio`.
"""

from __future__ import annotations

import asyncio
import os
import sys
import uuid

from agents import set_tracing_disabled
from agents.realtime import RealtimeAgent, RealtimeRunner

from ..audio import AudioInput, AudioOutput
from ..console import NullUI, StatusUI, frame_level
from langsmith.integrations.openai_realtime import wrap_realtime_session
from .tools import lookup_weather

SYSTEM_PROMPT = """You are a friendly voice assistant who can look up the
weather for any city. Keep replies short, conversational, and free of
formatting (no asterisks, no bullet points, no emoji). When the user asks
about weather in one or more places, call the lookup_weather tool — once per
city — and then summarize the results naturally in one or two short spoken
sentences."""


DEFAULT_MODEL = os.getenv("REALTIME_MODEL", "gpt-realtime-2")
SAMPLE_RATE = 24_000


def _model_settings() -> dict:
    """RealtimeSessionModelSettings: the model/audio/turn config for the session."""
    return {
        "model_name": DEFAULT_MODEL,
        "output_modalities": ["audio"],
        "voice": "alloy",
        "input_audio_format": "pcm16",
        "output_audio_format": "pcm16",
        "input_audio_transcription": {"model": "gpt-4o-mini-transcribe"},
        "input_audio_noise_reduction": {"type": "near_field"},
        "turn_detection": {"type": "server_vad", "interrupt_response": True},
        "tool_choice": "auto",
    }


async def run(
    project_name: str,
    *,
    audio_in: AudioInput,
    audio_out: AudioOutput,
    ui: StatusUI | None = None,
) -> None:
    """Drive an OpenAI Realtime conversation through the Agents SDK.

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
    thread_id = str(uuid.uuid4())

    # The Agents SDK ships its own tracer (uploads to OpenAI's dashboard). We
    # trace via LangSmith from the session's event stream instead, so turn the
    # SDK's exporter off to avoid a second, surprise upload path.
    set_tracing_disabled(True)

    ui.log(f"[openai-agents] thread_id={thread_id}")
    ui.log("[openai-agents] connecting to OpenAI Realtime API...")

    agent = RealtimeAgent(
        name="weather-assistant",
        instructions=SYSTEM_PROMPT,
        tools=[lookup_weather],
    )
    runner = RealtimeRunner(
        starting_agent=agent,
        config={"model_settings": _model_settings()},
    )

    def _on_message(role: str, text: str) -> None:
        ui.log(f"{'user: ' if role == 'user' else 'agent:'} {text}")

    mic_task: asyncio.Task | None = None
    try:
        session = await runner.run()
        # `wrap_realtime_session` enters the underlying session and builds the
        # LangSmith trace from its semantic event stream.
        async with wrap_realtime_session(
            session,
            thread_id=thread_id,
            sample_rate=SAMPLE_RATE,
            project_name=project_name,
            tags=["voice-demo", "openai-agents"],
            metadata={"model": DEFAULT_MODEL},
            on_message=_on_message,
        ) as conn:
            # Record the AGENT side at the speaker (bytes actually played), so
            # barge-in-flushed audio is excluded — the WAV reflects what was heard.
            audio_out.set_played_callback(conn.record_agent_audio)

            audio_in.start()
            audio_out.start()
            ui.log("[openai-agents] connected. Talk into your mic — Ctrl-C to quit.")
            ui.set_state("listening")

            async def pump_mic() -> None:
                async for frame in audio_in.frames():
                    await conn.send_audio(frame)
                    conn.record_user_audio(frame)
                    ui.update_level(frame_level(frame))

            mic_task = asyncio.create_task(pump_mic())

            # Application event loop — pure application behavior (play audio, set
            # UI state). All tracing lives in the wrapper.
            async for event in conn:
                event_type = event.type

                if event_type == "audio":
                    audio_out.write(event.audio.data)
                    ui.set_state("speaking")
                elif event_type == "agent_start":
                    ui.set_state("listening")
                elif event_type == "audio_interrupted":
                    audio_out.clear()  # barge-in
                    ui.set_state("hearing you")
                elif event_type == "audio_end":
                    ui.set_state("listening")
                elif event_type == "tool_start":
                    ui.set_state("running tools")
                elif event_type == "tool_end":
                    ui.set_state("thinking")
                elif event_type == "error":
                    ui.log(
                        f"[openai-agents] server error: {getattr(event, 'error', None)}"
                    )

    except Exception as exc:
        # The error is recorded on the root span by the wrapper's teardown.
        ui.log(f"[openai-agents] error: {exc}")
    finally:
        if mic_task is not None:
            mic_task.cancel()
        audio_in.stop()
        audio_out.stop()
        ui.finish()

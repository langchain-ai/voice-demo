"""Gemini Live voice agent over the official SDK's raw WebSocket session.

This backend is parallel to ``voice_demo.openai``: it uses no agent framework.
The application directly drives ``client.aio.live.connect(...)``, streams PCM
microphone frames, consumes provider messages, plays audio, handles barge-in,
and dispatches tools. ``langsmith.integrations.gemini_live.wrap_gemini_live``
wraps the session at the event-stream boundary and owns all trace decisions.

The agent is frontend-agnostic. Its local console, mic, and speaker are injected
through ``AudioInput`` / ``AudioOutput`` / ``StatusUI`` protocols.
"""

from __future__ import annotations

import asyncio
import os
import sys
import uuid

from google import genai
from google.genai import types
from langsmith.integrations.gemini_live import wrap_gemini_live

from ..audio import AudioInput, AudioOutput, resample_pcm16
from ..console import NullUI, StatusUI, frame_level
from .events import LiveMessage
from .tools import execute_tool

SYSTEM_PROMPT = """You are a friendly voice assistant who can look up the
weather for any city. Keep replies short, conversational, and free of
formatting (no asterisks, no bullet points, no emoji). When the user asks
about weather in one or more places, call the lookup_weather tool once per
city, then summarize the results naturally in one or two short sentences."""

DEFAULT_MODEL = os.getenv("GEMINI_LIVE_MODEL", "gemini-3.1-flash-live-preview")
SEND_SAMPLE_RATE = 16_000
RECV_SAMPLE_RATE = 24_000

WEATHER_TOOL = types.Tool(
    function_declarations=[
        types.FunctionDeclaration(
            name="lookup_weather",
            description=(
                "Get the current weather for a single city. Call once per city "
                "for multi-city questions."
            ),
            parameters_json_schema={
                "type": "object",
                "properties": {
                    "city": {
                        "type": "string",
                        "description": "City name, such as San Francisco or Tokyo.",
                    }
                },
                "required": ["city"],
                "additionalProperties": False,
            },
        )
    ]
)


def _live_config() -> types.LiveConnectConfig:
    return types.LiveConnectConfig(
        response_modalities=[types.Modality.AUDIO],
        system_instruction=SYSTEM_PROMPT,
        tools=[WEATHER_TOOL],
        input_audio_transcription=types.AudioTranscriptionConfig(),
        output_audio_transcription=types.AudioTranscriptionConfig(),
        speech_config=types.SpeechConfig(
            voice_config=types.VoiceConfig(
                prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name="Aoede")
            )
        ),
        realtime_input_config=types.RealtimeInputConfig(
            automatic_activity_detection=types.AutomaticActivityDetection(
                start_of_speech_sensitivity=types.StartSensitivity.START_SENSITIVITY_HIGH,
                end_of_speech_sensitivity=types.EndSensitivity.END_SENSITIVITY_LOW,
                prefix_padding_ms=200,
                silence_duration_ms=800,
            )
        ),
    )


async def run(
    project_name: str,
    *,
    audio_in: AudioInput,
    audio_out: AudioOutput,
    ui: StatusUI | None = None,
) -> None:
    """Drive a raw Gemini Live conversation over the supplied audio frontend."""
    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        print("GOOGLE_API_KEY is not set.", file=sys.stderr)
        sys.exit(1)

    ui = ui or NullUI()
    thread_id = str(uuid.uuid4())
    client = genai.Client(api_key=api_key)

    ui.log(f"[gemini] thread_id={thread_id}")
    ui.log(f"[gemini] connecting to Gemini Live with model={DEFAULT_MODEL}...")

    mic_task: asyncio.Task | None = None
    receive_task: asyncio.Task | None = None
    try:
        async with (
            client.aio.live.connect(model=DEFAULT_MODEL, config=_live_config()) as raw,
            wrap_gemini_live(
                raw,
                model=DEFAULT_MODEL,
                thread_id=thread_id,
                sample_rate=RECV_SAMPLE_RATE,
                project_name=project_name,
                tags=["voice-demo", "gemini"],
                metadata={"model": DEFAULT_MODEL},
                is_agent_speaking=lambda: audio_out.buffered_bytes() > 0,
            ) as session,
        ):
            # Capture what the listener actually heard. Audio removed by
            # ``clear()`` during barge-in never reaches this callback.
            audio_out.set_played_callback(session.record_agent_audio)
            audio_in.start()
            audio_out.start()
            ui.log("[gemini] connected. Talk into your mic — Ctrl-C to quit.")
            ui.set_state("listening")

            async def pump_mic() -> None:
                async for frame in audio_in.frames():
                    await session.send_realtime_input(
                        audio=types.Blob(
                            data=resample_pcm16(
                                frame, audio_in.sample_rate, SEND_SAMPLE_RATE
                            ),
                            mime_type=f"audio/pcm;rate={SEND_SAMPLE_RATE}",
                        )
                    )
                    session.record_user_audio(
                        resample_pcm16(frame, audio_in.sample_rate, RECV_SAMPLE_RATE)
                    )
                    ui.update_level(frame_level(frame))

            async def pump_responses() -> None:
                # Gemini's receive() generator ends at each turn_complete, so
                # open a new generator for every subsequent user turn.
                while True:
                    async for raw_message in session.receive():
                        message = LiveMessage(raw_message)

                        if message.interrupted:
                            audio_out.clear()
                            ui.set_state("hearing you")

                        for chunk in message.audio_chunks:
                            audio_out.write(chunk)
                            ui.set_state("speaking")

                        if message.user_transcript:
                            ui.set_state("hearing you")
                        if message.final_user_transcript:
                            ui.log(f"user:  {message.final_user_transcript}")
                        if message.final_agent_transcript:
                            ui.log(f"agent: {message.final_agent_transcript}")

                        if message.function_calls:
                            ui.set_state("running tools")
                            responses: list[types.FunctionResponse] = []
                            for call in message.function_calls:
                                if not getattr(call, "id", None):
                                    ui.log(
                                        "[gemini] ignored malformed tool call without an id"
                                    )
                                    continue
                                result = await execute_tool(
                                    getattr(call, "name", None),
                                    getattr(call, "args", None),
                                )
                                responses.append(
                                    types.FunctionResponse(
                                        id=call.id,
                                        name=call.name,
                                        response=result,
                                    )
                                )
                            if responses:
                                ui.set_state("thinking")
                                await session.send_tool_response(
                                    function_responses=responses
                                )

                        if message.turn_complete:
                            ui.set_state("listening")

            mic_task = asyncio.create_task(pump_mic())
            receive_task = asyncio.create_task(pump_responses())
            await asyncio.gather(mic_task, receive_task)

    except asyncio.CancelledError:
        pass
    except Exception as exc:
        ui.log(f"[gemini] error: {exc}")
    finally:
        for task in (mic_task, receive_task):
            if task is not None:
                task.cancel()
        await asyncio.gather(
            *(task for task in (mic_task, receive_task) if task is not None),
            return_exceptions=True,
        )
        audio_out.set_played_callback(None)
        audio_in.stop()
        audio_out.stop()
        ui.finish()

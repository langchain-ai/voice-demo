"""Deepgram Voice Agent API through the official async Python SDK.

Deepgram owns the full STT → LLM → TTS loop on one connection. The application
only configures the agent, streams PCM from the injected audio frontend, plays
the returned PCM, and executes client-side tools. ``wrap_deepgram_voice``
observes the SDK's unchanged event stream and turns it into one LangSmith trace.

The SDK yields typed pydantic events (``frame.type`` is a ``Literal``), so the
loop below branches on that field directly — the same shape as the OpenAI
Realtime backend's ``event.type`` dispatch.
"""

from __future__ import annotations

import asyncio
import functools
import json
import os
import sys
import uuid

from deepgram import AsyncDeepgramClient
from deepgram.agent.v1.types import (
    AgentV1FunctionCallRequestFunctionsItem,
    AgentV1SendFunctionCallResponse,
    AgentV1Settings,
)
from langsmith.integrations.deepgram_voice import wrap_deepgram_voice

from ..audio import AudioInput, AudioOutput
from ..console import NullUI, StatusUI, frame_level
from ..prompts import GREETING, SYSTEM_PROMPT
from .tools import WEATHER_FUNCTION, execute_tool

SAMPLE_RATE = 24_000
LISTEN_MODEL = os.getenv("DEEPGRAM_LISTEN_MODEL", "nova-3")
THINK_MODEL = os.getenv("DEEPGRAM_THINK_MODEL", "gpt-4o-mini")
SPEAK_MODEL = os.getenv("DEEPGRAM_SPEAK_MODEL", "aura-2-thalia-en")


def _settings() -> AgentV1Settings:
    return AgentV1Settings(
        audio={
            "input": {"encoding": "linear16", "sample_rate": SAMPLE_RATE},
            "output": {
                "encoding": "linear16",
                "sample_rate": SAMPLE_RATE,
                "container": "none",
            },
        },
        agent={
            "language": "en",
            "listen": {"provider": {"type": "deepgram", "model": LISTEN_MODEL}},
            "think": {
                "provider": {"type": "open_ai", "model": THINK_MODEL},
                "prompt": SYSTEM_PROMPT,
                "functions": [WEATHER_FUNCTION],
            },
            "speak": {"provider": {"type": "deepgram", "model": SPEAK_MODEL}},
            "greeting": GREETING,
        },
    )


async def _expect(connection, expected_type: str) -> None:
    """Read control frames until the expected Deepgram handshake event.

    ``recv()`` returns bytes for audio, a typed event for known messages, or a
    raw dict for a message type this SDK build doesn't know — skip the latter two
    shapes rather than probing them.
    """
    while True:
        frame = await connection.recv()
        if isinstance(frame, (bytes, dict)):
            continue
        if frame.type == expected_type:
            return
        if frame.type == "Error":
            raise RuntimeError(frame.description)


async def _run_tool(
    connection, function: AgentV1FunctionCallRequestFunctionsItem
) -> None:
    result = await execute_tool(function.name, function.arguments)
    await connection.send_function_call_response(
        AgentV1SendFunctionCallResponse(
            id=function.id, name=function.name, content=json.dumps(result)
        )
    )


async def run(
    project_name: str,
    *,
    audio_in: AudioInput,
    audio_out: AudioOutput,
    ui: StatusUI | None = None,
) -> None:
    """Run a Deepgram Voice Agent conversation over the supplied audio frontend."""
    api_key = os.getenv("DEEPGRAM_API_KEY")
    if not api_key:
        print("DEEPGRAM_API_KEY is not set.", file=sys.stderr)
        sys.exit(1)

    ui = ui or NullUI()
    thread_id = str(uuid.uuid4())
    mic_task: asyncio.Task[None] | None = None
    # Tools run as background tasks so the loop keeps draining audio while a
    # lookup is in flight. Deepgram may issue several client-side calls in one
    # FunctionCallRequest, so each gets its own task keyed by call id.
    tool_tasks: dict[str, asyncio.Task[None]] = {}

    ui.log(f"[deepgram] thread_id={thread_id}")
    ui.log("[deepgram] connecting to Voice Agent API...")

    def tool_done(call_id: str, task: asyncio.Task[None]) -> None:
        if tool_tasks.get(call_id) is task:
            del tool_tasks[call_id]
        if not task.cancelled() and (error := task.exception()) is not None:
            ui.log(f"[deepgram] tool failed: {error}")

    try:
        client = AsyncDeepgramClient(api_key=api_key, session_id=thread_id)
        async with (
            client.agent.v1.connect() as raw,
            wrap_deepgram_voice(
                raw,
                thread_id=thread_id,
                sample_rate=SAMPLE_RATE,
                project_name=project_name,
                tags=["voice-demo", "deepgram"],
                metadata={
                    "listen_model": LISTEN_MODEL,
                    "think_model": THINK_MODEL,
                    "speak_model": SPEAK_MODEL,
                },
                is_agent_speaking=lambda: audio_out.buffered_bytes() > 0,
            ) as connection,
        ):
            await _expect(connection, "Welcome")
            await connection.send_settings(_settings())
            await _expect(connection, "SettingsApplied")

            # Record the agent side at the speaker so audio flushed by barge-in
            # never reaches the conversation WAV — it was never heard.
            audio_out.set_played_callback(connection.record_agent_audio)
            audio_in.start()
            audio_out.start()
            ui.log("[deepgram] connected. Talk into your mic — Ctrl-C to quit.")
            ui.set_state("listening")

            async def pump_mic() -> None:
                async for frame in audio_in.frames():
                    await connection.send_media(frame)
                    connection.record_user_audio(frame)
                    ui.update_level(frame_level(frame))

            mic_task = asyncio.create_task(pump_mic())

            # The SDK's iterator yields bytes for audio and a typed event for
            # every known control message; unknown types are dropped upstream.
            async for frame in connection:
                if isinstance(frame, bytes):
                    audio_out.write(frame)
                    ui.set_state("speaking")
                    continue

                if frame.type == "UserStartedSpeaking":
                    # Barge-in: flush whatever the agent was still saying.
                    audio_out.clear()
                    ui.set_state("hearing you")

                elif frame.type == "ConversationText":
                    if content := frame.content.strip():
                        label = "user" if frame.role == "user" else "agent"
                        ui.log(f"{label}: {content}")
                    if frame.role == "user":
                        ui.set_state("thinking")

                elif frame.type == "AgentThinking":
                    ui.set_state("thinking")

                elif frame.type == "AgentAudioDone":
                    ui.set_state("listening")

                elif frame.type == "FunctionCallRequest":
                    ui.set_state("running tools")
                    for function in frame.functions:
                        if not function.client_side:
                            continue
                        task = asyncio.create_task(_run_tool(connection, function))
                        tool_tasks[function.id] = task
                        task.add_done_callback(
                            functools.partial(tool_done, function.id)
                        )

                elif frame.type == "Warning":
                    ui.log(f"[deepgram] warning: {frame.description}")

                elif frame.type == "Error":
                    raise RuntimeError(frame.description)

    except Exception as exc:
        # The error is recorded on the root span by the wrapper's teardown;
        # here we just surface it to the console.
        ui.log(f"[deepgram] error: {exc}")
    finally:
        tasks = [task for task in (mic_task, *tool_tasks.values()) if task is not None]
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        audio_out.set_played_callback(None)
        audio_in.stop()
        audio_out.stop()
        ui.finish()

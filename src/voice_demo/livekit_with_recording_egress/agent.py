"""LiveKit recording-mode test with a distinct simulated Egress file."""

from __future__ import annotations

import asyncio
import os
import shutil
import sys
import tempfile
from pathlib import Path

LLM_MODEL = os.getenv("LIVEKIT_LLM_MODEL", "gpt-4o-mini")
STT_MODEL = os.getenv("LIVEKIT_STT_MODEL") or None
TTS_VOICE = os.getenv("LIVEKIT_TTS_VOICE") or None
TTS_MODEL = os.getenv("LIVEKIT_TTS_MODEL") or None
RECORDING_START_DELAY_SECONDS = 5


def run(project_name: str) -> None:
    """Launch a LiveKit cascade that delivers report and Egress audio separately."""
    for variable, purpose in (
        ("OPENAI_API_KEY", "the LLM stage"),
        ("ASSEMBLYAI_API_KEY", "STT"),
        ("CARTESIA_API_KEY", "Cartesia TTS"),
    ):
        if not os.environ.get(variable):
            print(
                f"{variable} is not set (this mode needs it for {purpose}).",
                file=sys.stderr,
            )
            sys.exit(1)

    from livekit import agents
    from livekit.agents import (
        Agent,
        AgentSession,
        TurnHandlingOptions,
        function_tool,
        room_io,
    )

    from langsmith.integrations.livekit import configure_livekit, set_thread_id

    from ..prompts import GREETING, SYSTEM_PROMPT
    from ..weather import fetch_weather

    if os.environ.get("LANGSMITH_API_KEY"):
        processor = configure_livekit(
            project=project_name,
            recording_mode="egress",
        )
        print(
            "[livekit-with-recording-egress] LangSmith OTel tracing enabled.",
            file=sys.stderr,
        )
    else:
        processor = None
        print(
            "[livekit-with-recording-egress] LANGSMITH_API_KEY not set, running without tracing.",
            file=sys.stderr,
        )

    from livekit.plugins import assemblyai
    from livekit.plugins import cartesia
    from livekit.plugins import openai as lk_openai
    from livekit.plugins import silero
    from livekit.plugins.turn_detector.multilingual import MultilingualModel

    def _build_session() -> AgentSession:
        tts_kwargs = {}
        if TTS_MODEL:
            tts_kwargs["model"] = TTS_MODEL
        if TTS_VOICE:
            tts_kwargs["voice"] = TTS_VOICE
        return AgentSession(
            stt=assemblyai.STT(model=STT_MODEL) if STT_MODEL else assemblyai.STT(),
            llm=lk_openai.LLM(model=LLM_MODEL, temperature=0.3),
            tts=cartesia.TTS(**tts_kwargs),
            vad=silero.VAD.load(),
            turn_handling=TurnHandlingOptions(turn_detection=MultilingualModel()),
        )

    class _Assistant(Agent):
        def __init__(self) -> None:
            super().__init__(instructions=SYSTEM_PROMPT)

        @function_tool
        async def lookup_weather(self, city: str) -> dict:
            """Get the current weather for a single city. Call once per city."""
            return await fetch_weather(city)

    server = agents.AgentServer()

    @server.rtc_session()
    async def _entrypoint(ctx: agents.JobContext) -> None:
        # External recording delivery is routed to the trace by thread id.
        thread_id = ctx.job.id
        set_thread_id(thread_id)

        async def _attach_egress_recording() -> None:
            if processor is None:
                return

            report = ctx.make_session_report()
            report_path = report.audio_recording_path
            if report_path is None:
                processor.complete_recording(thread_id, None)
                return

            with tempfile.TemporaryDirectory(prefix="voice-demo-egress-") as directory:
                egress_path = Path(directory, "egress-recording.ogg").resolve()
                shutil.copyfile(Path(report_path).resolve(), egress_path)
                processor.complete_recording(
                    thread_id,
                    data=egress_path.read_bytes(),
                    started_at=report.audio_recording_started_at,
                )

        ctx.add_shutdown_callback(_attach_egress_recording)

        await asyncio.sleep(RECORDING_START_DELAY_SECONDS)
        session = _build_session()
        await session.start(
            room=ctx.room,
            agent=_Assistant(),
            room_options=room_io.RoomOptions(),
            record={"audio": True},
        )
        await session.say(GREETING)

    sys.argv = [sys.argv[0], "console", "--record"]
    agents.cli.run_app(server)

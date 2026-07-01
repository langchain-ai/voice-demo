"""LiveKit voice agent (STT → LLM → TTS cascade), traced via OpenTelemetry.

This is the classic cascade pipeline: OpenAI STT → OpenAI LLM → OpenAI TTS, with
Silero VAD and multilingual turn detection. For the speech-to-speech variants
that collapse the whole pipeline into one realtime model, see
`livekit_with_openai_realtime/` and `livekit_with_gemini_live/` — they share this
file's tracing wiring, agent, and console entrypoint, differing only in how the
`AgentSession` is built.

Why OTEL and not SDK: LiveKit's Agents SDK emits OTel spans natively across its
STT, LLM, TTS, turn-detection, and EOU pipeline. The framework does the hard
work — we just call `configure_livekit` (the LangSmith voice integration,
`langsmith.integrations.livekit`), which wires a TracerProvider with a span
processor that translates LiveKit's `lk.*` vendor-prefixed attributes into the
`gen_ai.*` / `langsmith.*` namespaces that the LangSmith UI keys off. Audio
attachments come for free because the processor already does them.

Tools are LiveKit-native: `lookup_weather` is a `@function_tool` on the Agent,
and the framework owns the tool loop — it executes the function and appends the
call/result pair to its ChatContext itself, so the model always sees its prior
tool usage acknowledged on later turns.

(Unlike the OpenAI and ADK backends, we do not use `voice_demo.audio` here.
LiveKit owns its own audio I/O when running in console mode, and going around it
would lose the framework-emitted per-turn telemetry that is the whole point of
using LiveKit.)
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

LLM_MODEL = os.getenv("LIVEKIT_LLM_MODEL", "gpt-4o-mini")
STT_MODEL = os.getenv("LIVEKIT_STT_MODEL", "gpt-4o-mini-transcribe")
TTS_VOICE = os.getenv("LIVEKIT_TTS_VOICE", "alloy")

# The conversation's thread id is set once per session via the SDK's
# ``set_thread_id`` (a ContextVar) in the entrypoint; the processor reads it per
# span. The audio path is handed to the processor as an app-owned provider.
#
# A module global, not a ContextVar: deferred root spans can be exported at
# flush/shutdown, outside the job's task tree. Console mode runs one job per
# process, so a single slot is enough.
_audio_file_path: Path | None = None
# This demo runs in console mode, so it uses the local-file recording path
# (audio_path_provider) below. In a deployed worker ctx.session_directory is
# ephemeral; production captures audio with LiveKit Egress and attaches it via
# the processor's expect_recording / complete_recording methods. See the
# "Record in production with Egress" section of the LangSmith LiveKit docs.


def run(project_name: str) -> None:
    """Launch a LiveKit console agent (cascade). Blocks until Ctrl-C.

    Args:
        project_name: LangSmith project (informational — the OTLP headers wired
            by `tracing.configure` decide where spans land).
    """
    if not os.environ.get("OPENAI_API_KEY"):
        print(
            "OPENAI_API_KEY is not set (this LiveKit mode needs it for STT/LLM/TTS).",
            file=sys.stderr,
        )
        sys.exit(1)

    # Import after the env vars are set in tracing.configure() so the OTLP
    # exporter picks them up at module import time.
    from livekit import agents
    from livekit.agents import Agent, AgentSession, function_tool, room_io

    from ..prompts import GREETING, SYSTEM_PROMPT
    from langsmith.integrations.livekit import configure_livekit, set_thread_id
    from ..weather import fetch_weather

    # --- Tracing wiring ---
    # `configure_livekit` builds the TracerProvider, registers the
    # LangSmith span processor (which targets LangSmith's OTLP endpoint via the
    # SDK's own exporter), and wires it as both LiveKit's and the OTel global
    # provider. The audio-path provider stays app-owned.
    #
    # Two ways to install the provider: configure_livekit() builds + registers a
    # global provider for you (used here). To use your own provider, construct
    # LiveKitLangSmithSpanProcessor(...), add it to that provider, and register
    # it with LiveKit via livekit.agents.telemetry.set_tracer_provider(...).
    if os.environ.get("LANGSMITH_API_KEY"):
        configure_livekit(
            audio_path_provider=lambda: _audio_file_path,
            project=project_name,
        )
        print("[livekit] LangSmith OTel tracing enabled.", file=sys.stderr)
    else:
        print(
            "[livekit] LANGSMITH_API_KEY not set — running without tracing.",
            file=sys.stderr,
        )

    # Plugins must be imported on the main thread (LiveKit registers them at
    # import time), so none of these can be deferred into the job entrypoint.
    from livekit.plugins import openai as lk_openai
    from livekit.plugins import silero
    from livekit.plugins.turn_detector.multilingual import MultilingualModel

    def _build_session() -> AgentSession:
        """The STT → LLM → TTS cascade with VAD and turn detection."""
        return AgentSession(
            stt=lk_openai.STT(model=STT_MODEL),
            llm=lk_openai.LLM(model=LLM_MODEL, temperature=0.3),
            tts=lk_openai.TTS(model="tts-1", voice=TTS_VOICE),
            vad=silero.VAD.load(),
            turn_detection=MultilingualModel(),
        )

    class _Assistant(Agent):
        """Instructions + one native function tool."""

        def __init__(self) -> None:
            super().__init__(instructions=SYSTEM_PROMPT)

        @function_tool
        async def lookup_weather(self, city: str) -> dict:
            """Get the current weather for a single city. Call once per city."""
            return await fetch_weather(city)

    server = agents.AgentServer()

    @server.rtc_session()
    async def _entrypoint(ctx: agents.JobContext) -> None:
        global _audio_file_path
        # ctx.job.id is unique per dispatch; ctx.room.name is "console" in
        # console mode and would collide every session into one giant thread.
        set_thread_id(ctx.job.id)
        _audio_file_path = ctx.session_directory / "audio.ogg"

        session = _build_session()
        await session.start(
            room=ctx.room,
            agent=_Assistant(),
            room_options=room_io.RoomOptions(),
            # Turns on LiveKit's session audio recording. In console mode the
            # recording is written under ctx.session_directory only when the
            # `--record` CLI flag is set (forced into argv below); the processor
            # reads that audio.ogg and attaches it to the root span. This
            # local-file path is a console/dev pattern: in a deployed worker
            # ctx.session_directory is an ephemeral temp dir deleted at session
            # end (audio is uploaded to LiveKit's backend), so production should
            # use egress + attach the recording URL instead.
            record={"audio": True},
        )

        # Speak first. The cascade has a TTS stage, so session.say streams the
        # fixed greeting verbatim (and records it as the assistant's opening
        # message).
        await session.say(GREETING)

    # LiveKit's CLI owns argv parsing from here. We force `console` so the user
    # gets a local mic+speaker session without having to remember the subcommand.
    # `--record` is the console flag that makes LiveKit write the recording to a
    # local file under ctx.session_directory; without it the processor has no
    # audio.ogg to attach. (In dev/start modes, record={"audio": True} alone
    # enables recording, but the file is ephemeral — see the note above.)
    sys.argv = [sys.argv[0], "console", "--record"]
    agents.cli.run_app(server)

"""Google ADK Live voice agent, traced with the LangSmith SDK.

Why SDK not OTEL
----------------
ADK emits standard `gen_ai.*` OTel spans for its non-live execution paths
(Runner.run, BaseAgent.run_async, FunctionTool.run_async, etc.) — but
`Runner.run_live`, which is the entire voice loop, doesn't appear to be
instrumented. An OTel-only setup produces a single empty root span and
nothing inside it. So we treat ADK's `run_live` event stream the same way
the OpenAI backend treats OpenAI Realtime's WebSocket events: observe each
event, build the trace explicitly via the shared `voice_demo.sdk_tracing`
`EventSession`.

That refines the demo's "OTel vs SDK" rule:

    OTel  → framework runs in-process and emits its own spans (LiveKit, Pipecat)
    SDK   → framework hands us an event stream from a remote service
            (OpenAI Realtime, ADK Live)

Frontend-agnostic
-----------------
Like the OpenAI backend, `run()` takes its audio frontend by injection
(`AudioInput` / `AudioOutput` / optional `StatusUI`); the agent imports no
console code. The CLI supplies the local-machine implementations.

Trace shape
-----------
One conversation = one trace. Every `run_live` event becomes its own span under
the session root, in arrival order, carrying that event's full (scrubbed)
payload — in `inputs` if it's the user's transcribed speech heading to the
model, in `outputs` if it's the model/server replying. Audio is embedded inside
events as `inline_data` bytes; events whose *only* payload is an audio chunk are
played but not spanned (there are too many), and every recorded event has its
audio bytes scrubbed to a `<N bytes>` placeholder.

    realtime_session                       (root; conversation.wav stereo)
    │   metadata: thread_id, model, event_count, duration_s
    │
    ├── input_transcription               (event — user speech chunk)
    ├── output_transcription              (event — agent speech chunk)
    ├── function_call: get_weather        (event)
    ├── function_response: get_weather    (event)
    ├── turn_complete                     (event)
    └── interrupted                       (event)

Tool execution belongs to ADK — it runs the registered function itself between
the `function_call` and `function_response` events — so the trace records
exactly those two events rather than synthesizing a separate tool run. The
event stream is the source of truth.
"""

from __future__ import annotations

import asyncio
import logging
import os
import sys
import uuid
import warnings
from datetime import datetime
from zoneinfo import ZoneInfo

from langsmith.run_helpers import tracing_context

from ..audio import AudioInput, AudioOutput, resample_pcm16
from ..console import NullUI, StatusUI, frame_level
from ..sdk_tracing import start_session
from ..weather import fetch_weather
from .events import LiveEvent
from .utils import event_context

# ADK prints noisy startup messages (an experimental-feature warning, plus a
# log line about MCP not being installed) that smash into our status line. Mute
# them here, at import time — well before `run()` lazily imports google.adk.
warnings.filterwarnings("ignore", category=UserWarning, module="google.adk")
logging.getLogger("google_adk").setLevel(logging.ERROR)
logging.getLogger("google.adk").setLevel(logging.ERROR)


APP_NAME = "voice-demo-adk"
USER_ID = "console-user"

SEND_SAMPLE_RATE = 16_000  # ADK Live wants 16 kHz in
RECV_SAMPLE_RATE = 24_000  # …and 24 kHz out (and what our mic/speaker run at)

MODEL = os.getenv("ADK_LIVE_MODEL", "gemini-3.1-flash-live-preview")


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


async def get_weather(city: str) -> dict:
    """Geocode a city name and return current weather.

    Delegates to the shared `voice_demo.weather.fetch_weather` (Open-Meteo, no
    API key) — the same real lookup the OpenAI and LiveKit backends use. ADK
    awaits coroutine tools, so this runs inside the live event loop; its call
    and result appear in the trace as function_call/function_response events.

    Returns one of:
      {"city": str, "country": str, "weather": {...}}    on success
      {"city": str, "error": "not_found" | "http_error"} on failure
    """
    return await fetch_weather(city)


_INSTRUCTIONS = (
    "You are a friendly voice assistant speaking with the user in real time. "
    "Keep responses short and natural — one or two sentences. "
    "Use the get_time tool when asked about the current time, and the "
    "get_weather tool when asked about weather."
)


# ---------------------------------------------------------------------------
# Main run loop
# ---------------------------------------------------------------------------

async def run(
    project_name: str,
    *,
    audio_in: AudioInput,
    audio_out: AudioOutput,
    ui: StatusUI | None = None,
) -> None:
    """Drive an ADK Live conversation over the given audio frontend.

    Args:
        project_name: LangSmith project the trace lands in.
        audio_in: PCM16 mic source (24 kHz; resampled to 16 kHz for ADK).
        audio_out: PCM16 speaker sink (24 kHz); `clear()` is called on barge-in.
        ui: Optional status observer; defaults to a no-op for headless use.
    """
    if not os.environ.get("GOOGLE_API_KEY"):
        print("GOOGLE_API_KEY is not set.", file=sys.stderr)
        sys.exit(1)

    ui = ui or NullUI()

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

    thread_id = str(uuid.uuid4())
    ui.log(f"[adk] thread_id={thread_id}")
    ui.log(f"[adk] connected with model={MODEL}.")
    ui.log("[adk] talk into your mic — Ctrl-C to quit.")

    session = start_session(
        thread_id=thread_id,
        project_name=project_name,
        sample_rate=RECV_SAMPLE_RATE,
        tags=["voice-demo", "adk", "session"],
        metadata={"thread_id": thread_id, "model": MODEL},
    )
    # Record the AGENT side at the speaker (what was played), not at receipt —
    # so audio that barge-in flushes via clear() is excluded and the WAV
    # reflects what was heard.
    audio_out.set_played_callback(
        lambda data: session.record_agent(session.now(), data)
    )

    audio_in.start()
    audio_out.start()
    ui.set_state("listening")

    stop = asyncio.Event()

    async def pump_mic() -> None:
        async for frame in audio_in.frames():
            queue.send_realtime(
                genai_types.Blob(
                    data=resample_pcm16(frame, RECV_SAMPLE_RATE, SEND_SAMPLE_RATE),
                    mime_type=f"audio/pcm;rate={SEND_SAMPLE_RATE}"
                )
            )
            ui.update_level(frame_level(frame))

            session.record_user(session.now(), frame)

    async def pump_responses() -> None:
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

        try:
            async for raw_event in runner.run_live(
                user_id=USER_ID,
                session_id=adk_session.id,
                live_request_queue=queue,
                run_config=run_config,
            ):
                event = LiveEvent(raw_event)
                with event_context(session, event):
                    if event.interrupted:
                        audio_out.clear()
                        ui.set_state("hearing you")

                    for chunk in event.audio_chunks:
                        audio_out.write(chunk)
                        ui.set_state("speaking")

                    if event.user_transcript:
                        ui.set_state("hearing you")
                    if event.final_user_transcript:
                        ui.log(f"user : {event.final_user_transcript}")

                    if event.final_agent_transcript:
                        ui.log(f"agent: {event.final_agent_transcript}")

                    if event.turn_complete:
                        ui.set_state("listening")
        except asyncio.CancelledError:
            pass
        except Exception as exc:
            # Surface unexpected failures (network drop, auth, …) on the root
            # span — otherwise the session finalizes as if it ended cleanly —
            # and trigger teardown instead of hanging on a dead response pump.
            session.run.error = f"{type(exc).__name__}: {exc}"
            ui.log(f"[adk] error: {exc}")
            stop.set()

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
            audio_in.stop()
            audio_out.stop()
            ui.finish()

            session.finalize()

"""OpenAI Realtime API voice agent, traced with the LangSmith SDK.

Why SDK and not OTEL: OpenAI Realtime has no native telemetry — it's a raw
WebSocket event stream of `input_audio_buffer.*`, `response.*`, etc. We build
the trace directly from that stream (via the shared `voice_demo.sdk_tracing`
`EventSession`) so the trace mirrors *exactly what the server sent us*, rather
than a curated abstraction layered on top.

## Frontend-agnostic

`run()` takes its audio frontend by dependency injection: an `AudioInput`
(mic), an `AudioOutput` (speaker), and an optional `StatusUI`. The console CLI
supplies the local-machine implementations, but the agent itself imports no
console code — point it at a web/telephony transport and the same Realtime loop,
tracing, and tool handling run unchanged.

## Trace shape — one conversation = one trace

Every received WebSocket event becomes its own span under the session root, in
arrival order, carrying that event's full (scrubbed) payload — in `inputs` if
the user sent it toward the model (speech buffer, transcription), in `outputs`
if the model/server sent it back (`response.*`, `error`). Raw audio
(`response.output_audio.delta`) is played but not spanned, and every other
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
"""

from __future__ import annotations

import asyncio
import base64
import json
import os
import sys
import uuid

from langsmith.run_helpers import tracing_context
from openai import AsyncOpenAI

from ..audio import AudioInput, AudioOutput
from ..console import NullUI, StatusUI, frame_level
from ..sdk_tracing import start_session
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


def _is_inbound(et: str) -> bool:
    """Direction of an event relative to the model.

    Inbound = something the user sent toward the model (their speech buffer,
    their transcription) → goes in span `inputs`. Everything else (`response.*`,
    `error`, `session.*`) is the model/server talking back → span `outputs`.
    """
    return et.startswith("input_audio_buffer") or "input_audio_transcription" in et


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

    session = start_session(
        thread_id=thread_id,
        project_name=project_name,
        sample_rate=SAMPLE_RATE,
        tags=["voice-demo", "openai", "session"],
        metadata={"thread_id": thread_id, "model": DEFAULT_MODEL},
    )
    pending_tool_calls: list[tuple[str, str, str]] = []

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

                # Record the AGENT side at the speaker, not at receipt: this
                # callback fires with the bytes actually played, so audio that
                # barge-in flushes (clear()) is never recorded — the WAV
                # reflects what was *heard*, not what was generated.
                audio_out.set_played_callback(
                    lambda data: session.record_agent(session.now(), data)
                )

                audio_in.start()
                audio_out.start()
                ui.log("[openai] connected. Talk into your mic — Ctrl-C to quit.")
                ui.set_state("listening")

                async def pump_mic() -> None:
                    async for frame in audio_in.frames():
                        await connection.input_audio_buffer.append(
                            audio=base64.b64encode(frame).decode("ascii")
                        )
                        # Record into the session's timestamped timeline for
                        # the stereo conversation WAV.
                        session.record_user(session.now(), frame)
                        ui.update_level(frame_level(frame))

                mic_task = asyncio.create_task(pump_mic())

                async for event in connection:
                    et = event.type
                    t_now = session.now()

                    # Raw audio: hundreds of chunks per response. Play it, but
                    # don't span it.
                    if et == "response.output_audio.delta":
                        chunk = base64.b64decode(event.delta)
                        audio_out.write(chunk)
                        ui.set_state("speaking")
                        # NB: not recorded here — the speaker's played-callback
                        # records what's actually played (see set_played_callback
                        # above), so barge-in truncation is captured correctly.
                        continue

                    # Skip every other streaming partial (`*.delta`) — they're
                    # too noisy and the terminal `*.done` event carries the
                    # complete payload.
                    if et.endswith(".delta"):
                        continue

                    # Every other event becomes its own span, carrying the
                    # event's payload. Work done while handling it (guardrail,
                    # tools) nests inside via tracing_context(parent=ev).
                    with session.event_span(
                        event, t_now, name=et, inbound=_is_inbound(et)
                    ) as ev:
                        if et == "input_audio_buffer.speech_started":
                            # Barge-in: flush whatever the agent was still saying.
                            audio_out.clear()
                            ui.set_state("hearing you")

                        elif et == "input_audio_buffer.speech_stopped":
                            ui.set_state("transcribing")

                        elif et == "conversation.item.input_audio_transcription.completed":
                            transcript = (event.transcript or "").strip()
                            if not transcript:
                                # Noise — nothing usable was transcribed.
                                ui.log("[openai] (no transcript — noise turn)")
                                ui.set_state("listening")
                            else:
                                ui.log(f"user:  {transcript}")
                                ui.set_state("guardrail")
                                with tracing_context(parent=ev):
                                    verdict = await check_guardrail(transcript)

                                if verdict.blocked:
                                    ui.log(
                                        f"[openai] guardrail tripped: {verdict.reason}"
                                    )
                                    ui.set_state("refusing")
                                    await connection.response.create(
                                        response={"instructions": REFUSAL_INSTRUCTIONS}
                                    )
                                else:
                                    ui.set_state("thinking")
                                    await connection.response.create()

                        elif et == "response.output_audio_transcript.done":
                            ui.log(f"agent: {event.transcript or ''}")

                        elif et == "response.function_call_arguments.done":
                            pending_tool_calls.append(
                                (event.name, event.call_id, event.arguments)
                            )

                        elif et == "response.done":
                            if pending_tool_calls:
                                # Model emitted tool calls — run them nested
                                # under THIS event, then ask for the follow-up.
                                ui.set_state("running tools")
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
                                ui.set_state("thinking")
                                await connection.response.create()
                            else:
                                ui.set_state("listening")

                        elif et == "error":
                            ui.log(f"[openai] server error: {event.error}")

        finally:
            try:
                mic_task.cancel()  # type: ignore[name-defined]
            except NameError:
                pass
            audio_in.stop()
            audio_out.stop()
            session.finalize()
            ui.finish()

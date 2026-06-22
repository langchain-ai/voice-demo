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

Every *meaningful* received WebSocket event becomes its own span under the
session root, in arrival order, carrying that event's full (scrubbed) payload —
in `inputs` if the user sent it toward the model (speech buffer, transcription),
in `outputs` if the model/server sent it back (`response.*`, `error`). The root
span's `outputs` carries the running conversation transcript, so the LangSmith
preview pane shows the whole exchange at a glance.

Three classes of event are dropped to keep the trace readable: raw audio
(`response.output_audio.delta`) is played but not spanned; every other streaming
partial (`*.delta`) is dropped since the terminal `*.done` repeats the full
payload; and purely structural bookkeeping events (`NOISE_EVENTS` — item/part
add/done lifecycle markers that duplicate what `response.done` and the
transcript events already carry) are dropped too. Work the agent does *while
handling* an event (e.g. tool execution) nests inside that event's span, exactly
as a tool call would in any other traced app.

Spans are grouped into conversational `turn`s: a new turn opens each time the
user starts speaking (`input_audio_buffer.speech_started`, which is also the
barge-in signal), so one user utterance plus the agent's full response — tool
calls and the spoken follow-up — sit under a single turn span. Pre-conversation
`session.*` events arrive before any turn and stay under the root. This mirrors
the `turn` grouping of the LiveKit and Pipecat backends.

Readable events carry curated, conversation-shaped I/O (a user/assistant
message) with the raw wire payload preserved under `metadata.raw_event`. Each
`response.done` is a `chain` wrapper holding a point-in-time `model` (`llm`)
span — assistant message (with token usage for cost rollup and finish status),
`inputs` = the conversation context — alongside any tool runs it triggered, so
local tool latency is never attributed to the model call. The agent transcript
(`response.output_audio_transcript.done`) and the tool args
(`response.function_call_arguments.done`) are *not* spanned separately — the
`model` span already carries both — but the transcript still builds the root
conversation + prints to the console. The root is titled with the first user
utterance; each turn records `latency_to_first_audio_ms` and, on a barge-in,
`was_interrupted`.

    realtime_session                                   (root; metadata.title = first utterance)
    │  outputs: { messages: [...] }                   (full conversation transcript)
    │  attachments: conversation.wav                  (stereo: L=user, R=agent)
    │
    ├── session.created / session.updated             (pre-conversation setup)
    ├── turn                                           (metadata: latency_ms, was_interrupted)
    │   ├── input_audio_buffer.speech_started         (event)
    │   ├── input_audio_buffer.speech_stopped         (event)
    │   ├── conversation.item.input_audio_transcription.completed   (user message)
    │   ├── response.done                             (chain wrapper)
    │   │   ├── model                                 (llm — inputs=context, msg + tokens + status)
    │   │   └── lookup_weather × N                    (tool — sibling of the model span)
    │   └── response.done                             (chain → model llm, spoken follow-up)
    └── turn                                           (next user utterance …)
"""

from __future__ import annotations

import asyncio
import base64
import json
import os
import sys
import uuid
from typing import Any

from langsmith.run_helpers import tracing_context
from openai import AsyncOpenAI

from ..audio import AudioInput, AudioOutput
from ..console import NullUI, StatusUI, frame_level
from ..sdk_tracing import start_session
from .utils import (
    execute_tool,
    is_inbound,
    response_assistant_output,
    response_usage_metadata,
)


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

# Structural bookkeeping events the server sends to track item/content-part
# lifecycle. They carry no behavioral handler and only duplicate state already
# captured by `response.done` and the transcript events, so spanning them just
# floods the trace. Dropped before a span is opened (cf. the `*.delta` skip).
NOISE_EVENTS = frozenset(
    {
        "input_audio_buffer.committed",
        "conversation.item.added",
        "conversation.item.done",
        "response.created",
        "response.output_item.added",
        "response.output_item.done",
        "response.content_part.added",
        "response.content_part.done",
        "response.output_audio.done",
        # Redundant with the `model` llm span's tool_calls + the tool span's args.
        "response.function_call_arguments.done",
    }
)


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
    mic_task: asyncio.Task | None = None

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

                # Per-turn latency: end-of-user-speech → first agent audio
                # (perceived response lag). Set at speech_stopped, recorded on
                # the first audio chunk (then cleared), reset per user turn.
                # `None` = not armed.
                await_audio_since: float | None = None

                async for event in connection:
                    event_type = event.type
                    received_at = session.now()

                    # Raw audio: hundreds of chunks per response. Play it, but
                    # don't span it.
                    if event_type == "response.output_audio.delta":
                        if await_audio_since is not None:
                            session.add_turn_metadata(
                                latency_to_first_audio_ms=round(
                                    (received_at - await_audio_since) * 1000
                                )
                            )
                            await_audio_since = None
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
                    if event_type.endswith(".delta"):
                        continue

                    # Skip purely structural lifecycle events (see NOISE_EVENTS):
                    # they have no behavioral handler and only duplicate state
                    # `response.done` / the transcript events already carry.
                    if event_type in NOISE_EVENTS:
                        continue

                    # A user starting to speak begins a new conversational turn
                    # (this is also the barge-in signal). Open the turn span
                    # before this event's span so speech_started and everything
                    # up to the next speech_started nest under it. Pre-conversation
                    # session.* events precede any turn and stay under the root.
                    if event_type == "input_audio_buffer.speech_started":
                        # Barge-in: the user spoke while the agent still had
                        # audio queued. Flag the turn being interrupted before
                        # start_turn() closes it.
                        if audio_out.buffered_bytes() > 0:
                            session.add_turn_metadata(was_interrupted=True)
                        session.start_turn()

                    # The agent transcript is already the `model` llm span's
                    # content (recorded at response.done); don't span it twice.
                    # Use it to build the root transcript + print to the console,
                    # then drop the redundant event span.
                    if event_type == "response.output_audio_transcript.done":
                        transcript = (event.transcript or "").strip()
                        if transcript:
                            ui.log(f"agent: {transcript}")
                            session.add_message("assistant", transcript)
                        continue

                    # Curated, conversation-shaped I/O for the readable events;
                    # everything else keeps its raw wire payload (see event_span).
                    span_kwargs: dict[str, Any] = {"inbound": is_inbound(event_type)}
                    if event_type == "conversation.item.input_audio_transcription.completed":
                        user_text = (event.transcript or "").strip()
                        if user_text:
                            span_kwargs["inputs"] = {"role": "user", "content": user_text}
                    elif event_type == "response.done":
                        # Keep this event span a `chain` wrapper; the model payload
                        # (assistant message, tokens, status) goes on a child `llm`
                        # span recorded in the handler below, so local tool latency
                        # in this handler is never attributed to the model call.
                        # Empty outputs → the raw response is preserved under
                        # metadata.raw_event; the child `llm` span is the readable one.
                        span_kwargs["outputs"] = {}

                    # Every other event becomes its own span, carrying the
                    # event's payload. Work done while handling it (e.g. tool
                    # execution) nests inside via tracing_context(parent=event_run).
                    with session.event_span(
                        event, received_at, name=event_type, **span_kwargs
                    ) as event_run:
                        if event_type == "input_audio_buffer.speech_started":
                            # A new turn supersedes any latency measurement still
                            # pending from the previous turn (e.g. the user spoke
                            # again before the last response produced audio) —
                            # discard it so it can't be attributed to this turn.
                            await_audio_since = None
                            # Barge-in: flush whatever the agent was still saying.
                            audio_out.clear()
                            ui.set_state("hearing you")

                        elif event_type == "input_audio_buffer.speech_stopped":
                            # Arm the latency timer: from here to the first agent
                            # audio chunk is the perceived response lag.
                            await_audio_since = received_at
                            ui.set_state("transcribing")

                        elif event_type == "conversation.item.input_audio_transcription.completed":
                            transcript = (event.transcript or "").strip()
                            if not transcript:
                                # Noise — nothing usable was transcribed.
                                ui.log("[openai] (no speech detected)")
                                await_audio_since = None
                                ui.set_state("listening")
                            else:
                                ui.log(f"user:  {transcript}")
                                session.add_message("user", transcript)
                                session.set_title(transcript)  # first utterance → trace title
                                # Turn detection is configured with
                                # create_response=False, so we explicitly ask the
                                # model to respond once the turn is transcribed.
                                ui.set_state("thinking")
                                await connection.response.create()

                        elif event_type == "conversation.item.input_audio_transcription.failed":
                            # The model couldn't transcribe the turn — surface it
                            # (errored span + log) so "it didn't understand me" is
                            # debuggable, and don't leave the latency timer armed.
                            reason = getattr(event, "error", None)
                            ui.log(f"[openai] transcription failed: {reason}")
                            event_run.error = f"transcription_failed: {reason}"
                            await_audio_since = None
                            ui.set_state("listening")

                        elif event_type == "response.done":
                            # Point-in-time `llm` span for the model payload:
                            # assistant message, token usage (cost), finish status.
                            session.record_llm(
                                event_run,
                                outputs=response_assistant_output(event.response),
                                usage_metadata=response_usage_metadata(event.response),
                                metadata={"status": getattr(event.response, "status", None)},
                            )
                            # The done event carries the complete response,
                            # including any function calls — no need to collect
                            # them from earlier streaming events.
                            tool_calls = [
                                item
                                for item in (event.response.output or [])
                                if item.type == "function_call"
                            ]
                            if tool_calls:
                                # Run them nested under THIS event, then ask for
                                # the spoken follow-up. (response.create() must
                                # wait for response.done — the API allows one
                                # active response at a time.)
                                ui.set_state("running tools")
                                for call in tool_calls:
                                    with tracing_context(parent=event_run):
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
            # Surface unexpected failures (network drop, auth, …) on the root
            # span — otherwise the session finalizes as if it ended cleanly.
            session.run.error = f"{type(exc).__name__}: {exc}"
            ui.log(f"[openai] error: {exc}")
        finally:
            if mic_task is not None:
                mic_task.cancel()
            audio_in.stop()
            audio_out.stop()
            session.finalize()
            ui.finish()

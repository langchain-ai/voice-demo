"""OpenAI Realtime via the **Agents SDK**, traced with the LangSmith SDK.

This is the same conversation as `voice_demo.openai` — same model, same weather
tool, same console transport — but driven through the OpenAI Agents SDK
(`agents.realtime`) instead of a hand-written `client.realtime.connect()` loop.

## What the SDK changes

The raw backend talks the wire protocol directly: it appends to the input audio
buffer, watches `input_audio_buffer.*` / `response.*` events stream by, pulls
function calls out of `response.done`, runs them, posts `function_call_output`,
and asks for the follow-up — all by hand. Here the `RealtimeRunner` owns that
loop. We hand it a `RealtimeAgent` (instructions + `@function_tool`s), feed mic
PCM in with `session.send_audio()`, and consume a stream of **semantic** events:
`agent_start`, `tool_start` / `tool_end`, `audio`, `audio_interrupted`,
`history_added` / `history_updated`, `error`. The tool call loop happens *inside*
the SDK — we never see `function_call` items or post their outputs ourselves.

## Frontend-agnostic

Identical to the raw backend: `run()` takes its audio frontend by injection (an
`AudioInput`, an `AudioOutput`, an optional `StatusUI`). The console CLI supplies
the local-machine versions; the agent imports no console code.

## Trace shape — one conversation = one trace

Same machinery as the raw backend (`voice_demo.sdk_tracing.EventSession`): one
root `realtime_session` span (full transcript in `outputs`, stereo WAV attached)
grouped into `turn`s.

The conversation itself is reconstructed from the **full `history` snapshot** the
session delivers on every `history_updated` — not from the streamed events, which
mis-deliver user messages and emit the assistant transcript as growing partials.
We fold the snapshot into an item map keyed by stable `item_id` (latest text per
id wins → partials collapse, every user/assistant message is captured), and emit
one span per message once it's finalized: a curated `user_message` span or an
assistant `model` (`llm`) span. A new user item opens a `turn`. (Token usage
isn't available — the SDK has no per-response usage events yet.)

Everything else stays on the live event stream: `tool_start`/`tool_end` (the
SDK-run tool, a `tool` span), `audio_end`/`agent_*` (the timeline that drives the
audio-player sync), `latency_to_first_audio_ms`, and `was_interrupted` on
barge-in. Raw audio (`audio`) isn't spanned; the SDK's `raw_model_event`
passthrough isn't spanned either, but we mine it for the wire-level input
transcript that `history` can omit on a barge-in, merging it into the item map.

    realtime_session                                   (root; transcript in outputs)
    │  attachments: conversation.wav                  (stereo: L=user, R=agent)
    │
    ├── turn                                           (metadata: latency_ms, was_interrupted)
    │   ├── user_message         (user message)       (curated user msg, from history)
    │   ├── tool_start           (lookup_weather)     (event)
    │   ├── tool_end             (weather result)     (tool — args in / output out)
    │   ├── model                (assistant message)  (llm — history; inputs=context, partials collapsed)
    │   └── audio_interrupted    (barge-in)           (event)
    └── turn                                           (next user utterance …)

`agent_start`/`agent_end` are dropped (bookkeeping markers with no I/O).
"""

from __future__ import annotations

import asyncio
import os
import sys
import uuid
from typing import Any

from agents import set_tracing_disabled
from agents.realtime import RealtimeAgent, RealtimeRunner
from langsmith.run_helpers import tracing_context

from ..audio import AudioInput, AudioOutput
from ..console import NullUI, StatusUI, frame_level
from ..sdk_tracing import start_session
from .tools import lookup_weather
from .utils import (
    HistoryTracker,
    describe_event,
    history_item,
    history_messages,
    raw_input_transcript,
)

SYSTEM_PROMPT = """You are a friendly voice assistant who can look up the
weather for any city. Keep replies short, conversational, and free of
formatting (no asterisks, no bullet points, no emoji). When the user asks
about weather in one or more places, call the lookup_weather tool — once per
city — and then summarize the results naturally in one or two short spoken
sentences."""


DEFAULT_MODEL = os.getenv("REALTIME_MODEL", "gpt-realtime-2")
SAMPLE_RATE = 24_000


def _model_settings() -> dict:
    """RealtimeSessionModelSettings: the model/audio/turn config for the session.

    The same knobs the raw backend passes to `session.update`, in the Agents
    SDK's flat shape. We let the SDK's turn loop auto-create responses (no
    `create_response=False` here), since driving the response/tool cycle is
    exactly what the runner is for.
    """
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

    # The Agents SDK ships its own tracer (uploads to OpenAI's traces dashboard).
    # We trace via LangSmith from the session's event stream instead, so turn the
    # SDK's exporter off to avoid a second, surprise upload path.
    set_tracing_disabled(True)

    ui.log(f"[openai-agents] thread_id={thread_id}")
    ui.log("[openai-agents] connecting to OpenAI Realtime API...")

    session_trace = start_session(
        thread_id=thread_id,
        project_name=project_name,
        sample_rate=SAMPLE_RATE,
        tags=["voice-demo", "openai-agents", "session"],
        metadata={"thread_id": thread_id, "model": DEFAULT_MODEL},
    )
    mic_task: asyncio.Task | None = None

    # Authoritative conversation state, reconstructed from the full `history`
    # snapshot the session delivers on every history event (the streamed events
    # alone mis-deliver user messages and partials). The tracker folds snapshots
    # into a per-item map, emits one span per finalized message, groups them into
    # turns, and owns the per-turn first-audio latency timer.
    tracker = HistoryTracker(session_trace, ui)

    agent = RealtimeAgent(
        name="weather-assistant",
        instructions=SYSTEM_PROMPT,
        tools=[lookup_weather],
    )
    runner = RealtimeRunner(
        starting_agent=agent,
        config={"model_settings": _model_settings()},
    )

    with tracing_context(
        metadata={"thread_id": thread_id},
        tags=["voice-demo", "openai-agents"],
        project_name=project_name,
    ):
        try:
            session = await runner.run()
            async with session:
                # Record the AGENT side at the speaker, not at receipt: this
                # callback fires with the bytes actually played, so audio that
                # barge-in flushes (clear()) is never recorded — the WAV
                # reflects what was *heard*.
                audio_out.set_played_callback(
                    lambda data: session_trace.record_agent(session_trace.now(), data)
                )

                audio_in.start()
                audio_out.start()
                ui.log("[openai-agents] connected. Talk into your mic — Ctrl-C to quit.")
                ui.set_state("listening")

                async def pump_mic() -> None:
                    async for frame in audio_in.frames():
                        await session.send_audio(frame)
                        # Record into the session's timestamped timeline for the
                        # stereo conversation WAV.
                        session_trace.record_user(session_trace.now(), frame)
                        ui.update_level(frame_level(frame))

                mic_task = asyncio.create_task(pump_mic())

                async for event in session:
                    event_type = event.type
                    received_at = session_trace.now()

                    # Raw agent audio: many chunks per turn. Play it, don't span
                    # it. The speaker's played-callback records what's heard.
                    if event_type == "audio":
                        tracker.record_first_audio(received_at)
                        audio_out.write(event.audio.data)
                        ui.set_state("speaking")
                        continue

                    # The SDK's passthrough of the underlying wire protocol — noisy,
                    # not spanned. But it carries the wire-level user transcript that
                    # the semantic `history` sometimes omits (notably a barge-in), so
                    # mine that one event and merge it into the item map.
                    if event_type == "raw_model_event":
                        rec = raw_input_transcript(event)
                        if rec:
                            item_id, transcript = rec
                            tracker.observe([(item_id, "user", transcript)], received_at)
                            tracker.flush(hold_last=True)
                        continue

                    # History events feed the authoritative item map; they're not
                    # spanned directly (they stream partials). Messages are emitted
                    # as user/`model` spans from the map once an item is finalized.
                    if event_type == "history_added":
                        tracker.observe([history_item(event)], received_at)
                        tracker.flush(hold_last=True)
                        continue
                    if event_type == "history_updated":
                        tracker.observe(history_messages(event), received_at)
                        tracker.flush(hold_last=True)
                        continue

                    # agent_start/agent_end are bookkeeping markers (the agent
                    # re-activates on each model pass in the tool loop); they carry
                    # no I/O and just clutter the trace. Keep agent_start's UI cue
                    # but don't span them.
                    if event_type == "agent_start":
                        ui.set_state("listening")
                        continue
                    if event_type == "agent_end":
                        continue

                    # Remaining real-time events: the live timeline (audio-player
                    # sync), tool runs, latency/barge-in. The full describe_event
                    # payload is preserved under metadata.raw_event by event_span.
                    name, payload, inbound = describe_event(event)
                    span_kwargs: dict[str, Any] = {"inbound": inbound}
                    if event_type == "tool_end":
                        # The SDK ran the tool; record it as a `tool` run (args in,
                        # result out) — a sibling of the assistant `model` span.
                        span_kwargs["run_type"] = "tool"
                        span_kwargs["inputs"] = {"arguments": payload.get("arguments")}
                        span_kwargs["outputs"] = {"output": payload.get("output")}
                    elif event_type == "audio_interrupted":
                        session_trace.add_turn_metadata(was_interrupted=True)

                    with session_trace.event_span(
                        payload, received_at, name=name, **span_kwargs
                    ):
                        if event_type == "audio_interrupted":
                            # Barge-in: flush whatever the agent was still saying.
                            audio_out.clear()
                            ui.set_state("hearing you")

                        elif event_type == "audio_end":
                            ui.set_state("listening")

                        elif event_type == "tool_start":
                            ui.set_state("running tools")

                        elif event_type == "tool_end":
                            ui.set_state("thinking")

                        elif event_type == "error":
                            ui.log(f"[openai-agents] server error: {payload.get('error')}")

        except Exception as exc:
            # Surface unexpected failures (network drop, auth, …) on the root
            # span — otherwise the session finalizes as if it ended cleanly.
            session_trace.run.error = f"{type(exc).__name__}: {exc}"
            ui.log(f"[openai-agents] error: {exc}")
        finally:
            if mic_task is not None:
                mic_task.cancel()
            audio_in.stop()
            audio_out.stop()
            # Emit any messages still pending when the session ended (e.g. Ctrl-C
            # mid-utterance) before the turn/root spans close.
            tracker.flush(hold_last=False)
            session_trace.finalize()
            ui.finish()

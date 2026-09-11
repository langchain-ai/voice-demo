"""GPT-Live 1 weather agent over the official OpenAI Live SDK connection.

GPT-Live owns the full-duplex spoken conversation. A delegated Responses model
does the weather reasoning and selects the local ``lookup_weather`` function.
The application validates and executes every function call, returns all results,
and then explicitly continues the delegated response.
"""

from __future__ import annotations

import asyncio
import base64
import binascii
import json
import os
import signal
import sys
import uuid
from collections import deque
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any

from openai import AsyncOpenAI
from openai.resources.live.live import AsyncLiveConnection
from openai.types.live.function_tool_param import FunctionToolParam
from openai.types.live.session_config_param import SessionConfigParam
from openai.types.responses.response_input_item_param import ResponseInputItemParam

from ..audio import AudioInput, AudioOutput
from ..console import NullUI, StatusUI, frame_level
from .tools import execute_tool
from .tracing import LiveTracer

LIVE_MODEL = "gpt-live-1"
BACKEND_MODEL = os.getenv("OPENAI_LIVE_BACKEND_MODEL", "gpt-5.6-luna")
SAMPLE_RATE = 24_000
MAX_EVENT_BYTES = 4 * 1024 * 1024
MAX_PENDING_RESPONSES = 64
MAX_TOOL_CALLS_PER_RESPONSE = 16
MAX_TRANSCRIPT_CHARS = 4_000

CONVERSATION_PROMPT = """You are a friendly voice assistant. Keep spoken replies
short, conversational, and free of formatting. Delegate every weather request
to the backend. You may acknowledge that you are checking while it works. Only
state weather facts returned by the backend."""

BACKEND_PROMPT = """Handle weather questions for a live spoken conversation.
Call lookup_weather exactly once for each requested city, including every city
in a comparison. Never invent current conditions. After the tool results arrive,
return a concise, natural summary that is easy to say aloud."""

WEATHER_TOOL: FunctionToolParam = {
    "type": "function",
    "name": "lookup_weather",
    "description": (
        "Get current weather for one city. Call once per city for multi-city questions."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "city": {
                "type": "string",
                "maxLength": 120,
                "description": "A city name such as San Francisco or Tokyo.",
            }
        },
        "required": ["city"],
        "additionalProperties": False,
    },
    "strict": True,
}


def session_config() -> SessionConfigParam:
    """Return the immutable GPT-Live startup configuration."""
    return {
        "model": LIVE_MODEL,
        "instructions": CONVERSATION_PROMPT,
        "store": False,
        "audio": {
            "format": {"type": "audio/pcm", "rate": SAMPLE_RATE},
            "output": {"voice": "marin"},
        },
        "delegation": {
            "type": "responses",
            "responses": {
                "model": BACKEND_MODEL,
                "instructions": BACKEND_PROMPT,
                "tools": [WEATHER_TOOL],
                "tool_choice": "auto",
                "parallel_tool_calls": True,
            },
        },
    }


class LiveAPIError(RuntimeError):
    """A terminal error reported by the GPT-Live service."""


@dataclass(frozen=True)
class PendingToolCall:
    call_id: str
    name: str
    arguments: str


@dataclass
class _PendingResponse:
    tool_calls: list[PendingToolCall] = field(default_factory=list)


class TranscriptRows:
    """Small, revisable caption groups for console logging."""

    def __init__(self, gap_ms: int = 1_000) -> None:
        self.gap_ms = gap_ms
        self._text = {"user": "", "assistant": ""}
        self._end_ms: dict[str, int | None] = {"user": None, "assistant": None}

    def add(
        self, role: str, delta: Any, start_ms: Any, end_ms: Any
    ) -> str | None:
        if role not in self._text or not isinstance(delta, str) or not delta:
            return None
        previous = self._text[role]
        previous_end = self._end_ms[role]
        complete: str | None = None
        if (
            previous
            and isinstance(start_ms, int)
            and isinstance(previous_end, int)
            and start_ms - previous_end > self.gap_ms
        ):
            complete = previous
            previous = ""
        remaining = MAX_TRANSCRIPT_CHARS - len(previous)
        self._text[role] = (
            previous if remaining <= 0 else previous + delta[:remaining]
        )
        if isinstance(end_ms, int):
            self._end_ms[role] = end_ms
        return complete

    def flush(self) -> list[tuple[str, str]]:
        rows = [(role, text) for role, text in self._text.items() if text]
        self._text = {"user": "", "assistant": ""}
        self._end_ms = {"user": None, "assistant": None}
        return rows


class LiveEventProcessor:
    """Apply GPT-Live server events to audio, tools, UI, and tracing."""

    def __init__(
        self,
        *,
        connection: AsyncLiveConnection,
        audio_out: AudioOutput,
        ui: StatusUI,
        tracer: LiveTracer,
        tool_runner: Callable[[str, str], Awaitable[dict[str, Any]]] = execute_tool,
    ) -> None:
        self.connection = connection
        self.audio_out = audio_out
        self.ui = ui
        self.tracer = tracer
        self.tool_runner = tool_runner
        self.started = asyncio.Event()
        self.finalized = asyncio.Event()
        self.rows = TranscriptRows()
        self._pending: dict[tuple[str, str], _PendingResponse] = {}
        self._current_response: dict[str, str] = {}
        self._completed: set[tuple[str, str]] = set()
        self._completed_order: deque[tuple[str, str]] = deque()

    async def handle(self, event: dict[str, Any]) -> bool:
        """Handle one event; return false after a graceful terminal event."""
        event_type = event.get("type")
        if event_type == "session.started":
            self.started.set()
            session_id = (event.get("session") or {}).get("id", "unknown")
            self.ui.log(f"[openai-live] session_id={session_id}")
            self.ui.log(
                "[openai-live] connected. Talk into your mic — Ctrl-C to quit."
            )
            self.ui.set_state("listening")
        elif event_type == "session.output_audio.delta":
            delta = event.get("delta")
            if isinstance(delta, str):
                try:
                    audio = base64.b64decode(delta, validate=True)
                except (binascii.Error, ValueError) as exc:
                    raise LiveAPIError("invalid output audio from GPT-Live") from exc
                self.audio_out.write(audio)
                self.ui.set_state("speaking")
        elif event_type == "session.input_transcript.delta":
            self._transcript("user", event)
            self.ui.set_state("hearing you")
        elif event_type == "session.output_transcript.delta":
            self._transcript("assistant", event)
        elif event_type == "session.delegation.created":
            delegation_id = event.get("delegation_id") or (
                event.get("delegation") or {}
            ).get("id")
            self.tracer.start_delegation(delegation_id, event)
            self.ui.set_state("thinking")
        elif event_type == "response.event":
            await self._handle_response_event(event)
        elif event_type == "session.closed":
            for role, text in self.rows.flush():
                self.ui.log(f"{role}:  {text}")
            self.tracer.set_final_usage(event.get("usage"))
            self.finalized.set()
            return False
        elif event_type == "error":
            error = event.get("error")
            if isinstance(error, dict):
                message = str(
                    error.get("message") or error.get("type") or "unknown error"
                )
            else:
                message = str(error or "unknown error")
            raise LiveAPIError(message[:500])
        return True

    def _transcript(self, role: str, event: dict[str, Any]) -> None:
        delta = event.get("delta")
        self.tracer.add_transcript(role, delta)
        complete = self.rows.add(
            role, delta, event.get("start_ms"), event.get("end_ms")
        )
        if complete:
            self.ui.log(f"{role}:  {complete}")

    async def _handle_response_event(self, envelope: dict[str, Any]) -> None:
        delegation_id = envelope.get("delegation_id")
        inner = envelope.get("event")
        if not isinstance(delegation_id, str) or not isinstance(inner, dict):
            return
        event_type = inner.get("type")

        if event_type == "response.created":
            response = inner.get("response") or {}
            response_id = response.get("id") or inner.get("response_id")
            if isinstance(response_id, str):
                self._current_response[delegation_id] = response_id
                self._response_state(delegation_id, response_id)
                self.tracer.start_model(delegation_id, response_id)
        elif event_type == "response.output_text.delta":
            response_id = self._response_id(delegation_id, inner)
            if response_id:
                self.tracer.add_model_text(
                    delegation_id, response_id, inner.get("delta")
                )
        elif event_type == "response.output_item.done":
            response_id = self._response_id(delegation_id, inner)
            item = inner.get("item") or {}
            if (
                response_id
                and isinstance(item, dict)
                and item.get("type") == "function_call"
            ):
                self._collect_tool_call(delegation_id, response_id, item)
        elif event_type in {
            "response.completed",
            "response.failed",
            "response.incomplete",
        }:
            response = inner.get("response") or {}
            response_id = response.get("id") or self._response_id(delegation_id, inner)
            if not isinstance(response_id, str):
                return
            self.tracer.finish_model(delegation_id, response_id, response)
            if event_type != "response.completed":
                raise LiveAPIError(f"delegated backend ended with {event_type}")
            await self._complete_response(delegation_id, response_id)

    def _response_id(self, delegation_id: str, event: dict[str, Any]) -> str | None:
        value = event.get("response_id")
        return (
            value
            if isinstance(value, str)
            else self._current_response.get(delegation_id)
        )

    def _response_state(
        self, delegation_id: str, response_id: str
    ) -> _PendingResponse:
        key = (delegation_id, response_id)
        state = self._pending.get(key)
        if state is not None:
            return state
        if len(self._pending) >= MAX_PENDING_RESPONSES:
            raise LiveAPIError("too many pending delegated responses")
        state = _PendingResponse()
        self._pending[key] = state
        return state

    def _collect_tool_call(
        self, delegation_id: str, response_id: str, item: dict[str, Any]
    ) -> None:
        state = self._response_state(delegation_id, response_id)
        if len(state.tool_calls) >= MAX_TOOL_CALLS_PER_RESPONSE:
            raise LiveAPIError("too many tool calls in one delegated response")
        call_id, name, arguments = (
            item.get("call_id"),
            item.get("name"),
            item.get("arguments"),
        )
        if not all(
            isinstance(value, str) and value
            for value in (call_id, name, arguments)
        ):
            self.ui.log("[openai-live] ignored malformed delegated tool call")
            return
        call = PendingToolCall(call_id=call_id, name=name, arguments=arguments[:8_192])
        state.tool_calls.append(call)
        self.tracer.add_model_tool_call(
            delegation_id,
            response_id,
            call_id=call.call_id,
            name=call.name,
            arguments=call.arguments,
        )

    async def _complete_response(self, delegation_id: str, response_id: str) -> None:
        key = (delegation_id, response_id)
        if key in self._completed:
            return
        self._remember_completed(key)
        state = self._pending.pop(key, _PendingResponse())
        if not state.tool_calls:
            self._current_response.pop(delegation_id, None)
            self.tracer.finish_delegation(delegation_id)
            self.ui.set_state("listening")
            return

        self.ui.set_state("running tools")

        async def run_one(call: PendingToolCall) -> dict[str, Any]:
            return await self.tracer.run_tool(
                delegation_id=delegation_id,
                name=call.name,
                arguments=call.arguments,
                operation=lambda call=call: self.tool_runner(call.name, call.arguments),
            )

        # Independent city lookups can run together, but every result must be
        # appended before the single response.create that continues the backend.
        results = await asyncio.gather(*(run_one(call) for call in state.tool_calls))
        for call, result in zip(state.tool_calls, results, strict=True):
            item: ResponseInputItemParam = {
                "type": "function_call_output",
                "call_id": call.call_id,
                "output": json.dumps(result, separators=(",", ":")),
            }
            await self.connection.response.item.create(
                event_id=f"tool_result_{uuid.uuid4().hex}",
                item=item,
            )
        self.ui.set_state("thinking")
        await self.connection.response.create(
            event_id=f"continue_{uuid.uuid4().hex}"
        )

    def _remember_completed(self, key: tuple[str, str]) -> None:
        self._completed.add(key)
        self._completed_order.append(key)
        while len(self._completed_order) > MAX_PENDING_RESPONSES:
            self._completed.discard(self._completed_order.popleft())


async def run(
    project_name: str,
    *,
    audio_in: AudioInput,
    audio_out: AudioOutput,
    ui: StatusUI | None = None,
) -> None:
    """Run a local-mic GPT-Live conversation until Ctrl-C."""
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("OPENAI_API_KEY is not set.", file=sys.stderr)
        sys.exit(1)
    if audio_in.sample_rate != SAMPLE_RATE or audio_out.sample_rate != SAMPLE_RATE:
        raise ValueError("GPT-Live audio input and output must both be PCM16 at 24 kHz")

    ui = ui or NullUI()
    thread_id = str(uuid.uuid4())
    tracer = LiveTracer(
        project_name=project_name,
        thread_id=thread_id,
        live_model=LIVE_MODEL,
        backend_model=BACKEND_MODEL,
        sample_rate=SAMPLE_RATE,
    )
    tracer.start()
    ui.log(f"[openai-live] thread_id={thread_id}")
    ui.log(f"[openai-live] connecting with backend={BACKEND_MODEL}...")

    mic_task: asyncio.Task[None] | None = None
    receiver_task: asyncio.Task[None] | None = None
    closer_task: asyncio.Task[None] | None = None
    close_requested = asyncio.Event()
    closing = asyncio.Event()
    send_lock = asyncio.Lock()
    trace_error: str | None = None
    loop = asyncio.get_running_loop()
    signal_installed = False

    try:
        loop.add_signal_handler(signal.SIGINT, close_requested.set)
        signal_installed = True
    except (NotImplementedError, RuntimeError):
        pass

    try:
        async with AsyncOpenAI(api_key=api_key) as client:
            async with client.live.connect(
                websocket_connection_options={
                    "compression": None,
                    "max_size": MAX_EVENT_BYTES,
                    "max_queue": 32,
                }
            ) as connection:
                processor = LiveEventProcessor(
                    connection=connection,
                    audio_out=audio_out,
                    ui=ui,
                    tracer=tracer,
                )
                await connection.session.start(
                    event_id=f"start_{uuid.uuid4().hex}",
                    session=session_config(),
                )

                # Capture the assistant at the speaker callback, not when bytes
                # arrive from the network. Buffered audio dropped on interruption
                # therefore never enters the trace because it was never heard.
                audio_out.set_played_callback(tracer.record_agent_audio)
                audio_in.start()
                audio_out.start()

                async def pump_mic() -> None:
                    await processor.started.wait()
                    pending = b""
                    async for frame in audio_in.frames():
                        if closing.is_set():
                            return
                        chunk = pending + frame
                        complete = len(chunk) - len(chunk) % 2
                        pending = chunk[complete:]
                        if not complete:
                            continue
                        sent_audio = chunk[:complete]
                        async with send_lock:
                            await connection.session.input_audio.append(
                                audio=base64.b64encode(sent_audio).decode("ascii")
                            )
                        tracer.record_user_audio(sent_audio)
                        ui.update_level(frame_level(frame))

                async def receive_events() -> None:
                    async for event in connection:
                        event_data = event.model_dump(mode="python")
                        if not await processor.handle(event_data):
                            return

                async def close_on_request() -> None:
                    await close_requested.wait()
                    closing.set()
                    await asyncio.wait_for(processor.started.wait(), timeout=15)
                    async with send_lock:
                        await connection.session.close(
                            event_id=f"close_{uuid.uuid4().hex}"
                        )
                    await asyncio.wait_for(processor.finalized.wait(), timeout=15)

                mic_task = asyncio.create_task(pump_mic())
                receiver_task = asyncio.create_task(receive_events())
                closer_task = asyncio.create_task(close_on_request())

                done, _ = await asyncio.wait(
                    {receiver_task, closer_task},
                    return_when=asyncio.FIRST_COMPLETED,
                )
                for task in done:
                    error = task.exception()
                    if error is not None:
                        raise error
                if receiver_task in done and not processor.finalized.is_set():
                    raise LiveAPIError("connection closed before session.closed")

    except asyncio.CancelledError:
        trace_error = "session cancelled"
        raise
    except Exception as exc:
        trace_error = f"{type(exc).__name__}: {exc}"
        ui.log(f"[openai-live] error: {exc}")
    finally:
        closing.set()
        for task in (mic_task, receiver_task, closer_task):
            if task is not None:
                task.cancel()
        await asyncio.gather(
            *(
                task
                for task in (mic_task, receiver_task, closer_task)
                if task is not None
            ),
            return_exceptions=True,
        )
        if signal_installed:
            loop.remove_signal_handler(signal.SIGINT)
        audio_in.stop()
        audio_out.stop()
        audio_out.set_played_callback(None)
        tracer.finish(error=trace_error)
        ui.finish()

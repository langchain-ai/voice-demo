"""LangSmith tracing built from GPT-Live's event stream.

``langsmith.integrations.openai_realtime`` cannot be reused here: it expects
Realtime's flat ``input_audio_buffer.*`` and ``response.*`` events. GPT-Live
uses ``session.*`` events and wraps delegated Responses events inside
``response.event``. The delegated Responses request also runs on OpenAI's
server, so wrapping a local OpenAI client cannot observe it.

This small adapter records the boundaries the application can actually see:
one Live session, each delegation, every delegated model response, and local
tool execution. A bounded stereo WAV is attached to the session root (user on
the left, assistant audio actually played on the right); audio and credentials
never enter JSON trace payloads.
"""

from __future__ import annotations

import array
import io
import os
import threading
import time
import wave
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, TypeVar

from langsmith.run_trees import RunTree

MAX_TRANSCRIPT_CHARS = 20_000
MAX_MODEL_TEXT_CHARS = 12_000
MAX_TOOL_CALLS_PER_RESPONSE = 16
MAX_OPEN_RUNS = 64
MAX_AUDIO_SECONDS = 10 * 60

T = TypeVar("T")


def _layout_audio(
    chunks: list[tuple[float, bytes]], sample_rate: int
) -> list[tuple[float, bytes]]:
    """Place bursty PCM chunks at natural playback time without overlap."""
    laid_out: list[tuple[float, bytes]] = []
    cursor = 0.0
    for index, (received_at, data) in enumerate(chunks):
        cursor = received_at if index == 0 else max(cursor, received_at)
        laid_out.append((cursor, data))
        cursor += (len(data) // 2) / sample_rate
    return laid_out


def _build_stereo_wav(
    user_chunks: list[tuple[float, bytes]],
    agent_chunks: list[tuple[float, bytes]],
    sample_rate: int,
) -> bytes:
    """Build a user-left/assistant-right PCM16 WAV from timestamped chunks."""
    if not user_chunks and not agent_chunks:
        return b""
    user = _layout_audio(user_chunks, sample_rate)
    agent = _layout_audio(agent_chunks, sample_rate)

    def end_time(chunks: list[tuple[float, bytes]]) -> float:
        return max(
            (offset + (len(data) // 2) / sample_rate for offset, data in chunks),
            default=0.0,
        )

    total_samples = min(
        int(max(end_time(user), end_time(agent)) * sample_rate + 0.999),
        MAX_AUDIO_SECONDS * sample_rate,
    )
    if total_samples <= 0:
        return b""

    def channel(chunks: list[tuple[float, bytes]]) -> array.array[int]:
        samples = array.array("h", bytes(total_samples * 2))
        for offset, data in chunks:
            start = int(offset * sample_rate)
            chunk = array.array("h")
            chunk.frombytes(data[: len(data) - len(data) % 2])
            count = min(len(chunk), total_samples - start)
            if count > 0:
                samples[start : start + count] = chunk[:count]
        return samples

    left = channel(user)
    right = channel(agent)
    stereo = array.array("h", bytes(total_samples * 4))
    stereo[0::2] = left
    stereo[1::2] = right
    output = io.BytesIO()
    with wave.open(output, "wb") as wav:
        wav.setnchannels(2)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(stereo.tobytes())
    return output.getvalue()


class _ConversationAudio:
    """Thread-safe, duration-bounded PCM capture for one trace attachment."""

    def __init__(self, sample_rate: int) -> None:
        self.sample_rate = sample_rate
        self.started_at = time.monotonic()
        self.max_channel_bytes = MAX_AUDIO_SECONDS * sample_rate * 2
        self.user_chunks: list[tuple[float, bytes]] = []
        self.agent_chunks: list[tuple[float, bytes]] = []
        self.user_bytes = 0
        self.agent_bytes = 0
        self.truncated = False
        self._lock = threading.Lock()

    def record_user(self, data: bytes) -> None:
        self._record("user", data)

    def record_agent(self, data: bytes) -> None:
        self._record("agent", data)

    def _record(self, role: str, data: bytes) -> None:
        data = data[: len(data) - len(data) % 2]
        if not data:
            return
        with self._lock:
            current = self.user_bytes if role == "user" else self.agent_bytes
            remaining = self.max_channel_bytes - current
            if remaining <= 0:
                self.truncated = True
                return
            if len(data) > remaining:
                data = data[: remaining - remaining % 2]
                self.truncated = True
            target = self.user_chunks if role == "user" else self.agent_chunks
            target.append((time.monotonic() - self.started_at, bytes(data)))
            if role == "user":
                self.user_bytes += len(data)
            else:
                self.agent_bytes += len(data)

    def build(self) -> tuple[bytes, bool]:
        with self._lock:
            user = list(self.user_chunks)
            agent = list(self.agent_chunks)
            truncated = self.truncated
        return _build_stereo_wav(user, agent, self.sample_rate), truncated


def _append_bounded(current: str, fragment: Any, limit: int) -> str:
    if not isinstance(fragment, str) or not fragment:
        return current
    remaining = limit - len(current)
    return current if remaining <= 0 else current + fragment[:remaining]


def _usage_metadata(value: Any) -> dict[str, Any] | None:
    """Keep only LangSmith's canonical numeric token fields."""
    if not isinstance(value, dict):
        return None
    allowed = (
        "input_tokens",
        "output_tokens",
        "total_tokens",
        "input_token_details",
        "output_token_details",
    )
    result = {key: value[key] for key in allowed if key in value}
    return result or None


def _compact_usage(value: Any) -> dict[str, Any] | None:
    """Bound the final duration-usage payload retained on the root trace."""
    if not isinstance(value, dict):
        return None
    compact: dict[str, Any] = {}
    for key, item in list(value.items())[:32]:
        name = str(key)[:80]
        if item is None or isinstance(item, (str, int, float, bool)):
            compact[name] = item
        elif isinstance(item, dict):
            compact[name] = {
                str(nested_key)[:80]: nested_value
                for nested_key, nested_value in list(item.items())[:32]
                if nested_value is None
                or isinstance(nested_value, (str, int, float, bool))
            }
    return compact or None


@dataclass
class _ModelRun:
    run: RunTree
    text: str = ""
    tool_calls: list[dict[str, Any]] = field(default_factory=list)


class LiveTracer:
    """Create a compact LangSmith trace from meaningful GPT-Live events."""

    def __init__(
        self,
        *,
        project_name: str,
        thread_id: str,
        live_model: str,
        backend_model: str,
        sample_rate: int,
        enabled: bool | None = None,
    ) -> None:
        self.enabled = (
            bool(os.environ.get("LANGSMITH_API_KEY"))
            if enabled is None
            else enabled
        )
        self.project_name = project_name
        self.thread_id = thread_id
        self.live_model = live_model
        self.backend_model = backend_model
        self.sample_rate = sample_rate
        self.root: RunTree | None = None
        self._audio: _ConversationAudio | None = None
        self._delegations: dict[str, RunTree] = {}
        self._models: dict[tuple[str, str], _ModelRun] = {}
        self._user_transcript = ""
        self._assistant_transcript = ""
        self._final_usage: dict[str, Any] | None = None

    def start(self) -> None:
        if not self.enabled:
            return
        self._audio = _ConversationAudio(self.sample_rate)
        self.root = RunTree(
            name="gpt_live_session",
            run_type="chain",
            inputs={
                "live_model": self.live_model,
                "backend_model": self.backend_model,
                "transport": "websocket",
            },
            project_name=self.project_name,
            tags=["voice-demo", "openai-live"],
            extra={
                "metadata": {
                    "thread_id": self.thread_id,
                    "ls_provider": "openai",
                    "ls_model_name": self.live_model,
                }
            },
        )
        self._safe(self.root.post)

    def record_user_audio(self, data: bytes) -> None:
        """Record PCM16 microphone bytes that were successfully sent."""
        if self._audio is not None:
            self._audio.record_user(data)

    def record_agent_audio(self, data: bytes) -> None:
        """Record PCM16 assistant bytes reported as played by the speaker."""
        if self._audio is not None:
            self._audio.record_agent(data)

    def add_transcript(self, role: str, delta: Any) -> None:
        if role == "user":
            self._user_transcript = _append_bounded(
                self._user_transcript, delta, MAX_TRANSCRIPT_CHARS
            )
        elif role == "assistant":
            self._assistant_transcript = _append_bounded(
                self._assistant_transcript, delta, MAX_TRANSCRIPT_CHARS
            )

    def start_delegation(self, delegation_id: Any, event: dict[str, Any]) -> None:
        if not self.root or not isinstance(delegation_id, str) or not delegation_id:
            return
        if delegation_id in self._delegations:
            return
        if len(self._delegations) >= MAX_OPEN_RUNS:
            return
        run = self.root.create_child(
            name="responses_delegation",
            run_type="chain",
            inputs={
                "target": event.get("target"),
                "response_id": event.get("response_id"),
            },
            extra={"metadata": {"delegation_id": delegation_id}},
        )
        self._delegations[delegation_id] = run
        self._safe(run.post)

    def start_model(self, delegation_id: str, response_id: str) -> None:
        if not self.root:
            return
        key = (delegation_id, response_id)
        if key in self._models:
            return
        if len(self._models) >= MAX_OPEN_RUNS:
            return
        parent = self._delegations.get(delegation_id, self.root)
        run = parent.create_child(
            name=self.backend_model,
            run_type="llm",
            inputs={"context": "managed by GPT-Live Responses delegation"},
            extra={
                "metadata": {
                    "delegation_id": delegation_id,
                    "response_id": response_id,
                    "ls_provider": "openai",
                    "ls_model_name": self.backend_model,
                }
            },
        )
        self._models[key] = _ModelRun(run=run)
        self._safe(run.post)

    def add_model_text(self, delegation_id: str, response_id: str, delta: Any) -> None:
        state = self._models.get((delegation_id, response_id))
        if state:
            state.text = _append_bounded(
                state.text, delta, MAX_MODEL_TEXT_CHARS
            )

    def add_model_tool_call(
        self,
        delegation_id: str,
        response_id: str,
        *,
        call_id: str,
        name: str,
        arguments: str,
    ) -> None:
        state = self._models.get((delegation_id, response_id))
        if not state or len(state.tool_calls) >= MAX_TOOL_CALLS_PER_RESPONSE:
            return
        state.tool_calls.append(
            {
                "id": call_id,
                "type": "function",
                "function": {"name": name, "arguments": arguments},
            }
        )

    def finish_model(
        self,
        delegation_id: str,
        response_id: str,
        response: Any,
    ) -> None:
        state = self._models.pop((delegation_id, response_id), None)
        if not state:
            return
        message: dict[str, Any] = {
            "role": "assistant",
            "content": state.text,
        }
        if state.tool_calls:
            message["tool_calls"] = state.tool_calls
        if isinstance(response, dict):
            usage = _usage_metadata(response.get("usage"))
            if usage:
                self._safe(lambda: state.run.set(usage_metadata=usage))
        self._finish_run(state.run, outputs=message)

    async def run_tool(
        self,
        *,
        delegation_id: str,
        name: str,
        arguments: str,
        operation: Callable[[], Awaitable[T]],
    ) -> T:
        run: RunTree | None = None
        if self.root:
            parent = self._delegations.get(delegation_id, self.root)
            run = parent.create_child(
                name=name[:80] or "tool",
                run_type="tool",
                inputs={"arguments": arguments[:8_192]},
            )
            self._safe(run.post)
        try:
            result = await operation()
        except Exception as exc:
            if run:
                self._finish_run(run, error=f"{type(exc).__name__}: {exc}")
            raise
        if run:
            self._finish_run(run, outputs={"result": result})
        return result

    def finish_delegation(self, delegation_id: str) -> None:
        run = self._delegations.pop(delegation_id, None)
        if run:
            self._finish_run(run, outputs={"status": "completed"})

    def set_final_usage(self, usage: Any) -> None:
        self._final_usage = _compact_usage(usage)

    def finish(self, error: str | None = None) -> None:
        for state in list(self._models.values()):
            self._finish_run(state.run, error="session ended before response completed")
        self._models.clear()
        for run in list(self._delegations.values()):
            self._finish_run(run, error="session ended before delegation completed")
        self._delegations.clear()
        if not self.root:
            return
        outputs: dict[str, Any] = {
            "transcript": [
                {"role": "user", "content": self._user_transcript},
                {"role": "assistant", "content": self._assistant_transcript},
            ]
        }
        if self._final_usage is not None:
            outputs["usage"] = self._final_usage
        audio = self._audio
        self._audio = None
        if audio is not None:
            try:
                wav, truncated = audio.build()
            except Exception:
                wav, truncated = b"", False
            if wav:
                self.root.attachments = {  # type: ignore[assignment]
                    "conversation": ("audio/wav", wav)
                }
            if truncated:
                self._safe(lambda: self.root.add_metadata({"audio_truncated": True}))
        self._finish_run(self.root, outputs=outputs, error=error)
        self.root = None

    @staticmethod
    def _safe(operation: Callable[[], Any]) -> None:
        try:
            operation()
        except Exception:
            # Observability must never take down the voice session.
            pass

    def _finish_run(
        self,
        run: RunTree,
        *,
        outputs: dict[str, Any] | None = None,
        error: str | None = None,
    ) -> None:
        self._safe(lambda: run.end(outputs=outputs, error=error))
        self._safe(run.patch)

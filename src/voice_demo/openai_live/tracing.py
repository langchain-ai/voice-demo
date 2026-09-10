"""LangSmith tracing built from GPT-Live's event stream.

``langsmith.integrations.openai_realtime`` cannot be reused here: it expects
Realtime's flat ``input_audio_buffer.*`` and ``response.*`` events. GPT-Live
uses ``session.*`` events and wraps delegated Responses events inside
``response.event``. The delegated Responses request also runs on OpenAI's
server, so wrapping a local OpenAI client cannot observe it.

This small adapter records the boundaries the application can actually see:
one Live session, each delegation, every delegated model response, and local
tool execution. Raw audio and credentials never enter trace payloads.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, TypeVar

from langsmith.run_trees import RunTree

MAX_TRANSCRIPT_CHARS = 20_000
MAX_MODEL_TEXT_CHARS = 12_000
MAX_TOOL_CALLS_PER_RESPONSE = 16
MAX_OPEN_RUNS = 64

T = TypeVar("T")


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
        self.root: RunTree | None = None
        self._delegations: dict[str, RunTree] = {}
        self._models: dict[tuple[str, str], _ModelRun] = {}
        self._user_transcript = ""
        self._assistant_transcript = ""
        self._final_usage: dict[str, Any] | None = None

    def start(self) -> None:
        if not self.enabled:
            return
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

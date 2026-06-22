"""Helpers for the OpenAI Realtime agent: tracing glue and tool dispatch."""

from __future__ import annotations

import json
from typing import Any

from .tools import lookup_weather


def is_inbound(event_type: str) -> bool:
    """Direction of an event relative to the model.

    Inbound = something the user sent toward the model (their speech buffer,
    their transcription) → goes in span `inputs`. Everything else (`response.*`,
    `error`, `session.*`) is the model/server talking back → span `outputs`.
    """
    return event_type.startswith("input_audio_buffer") or "input_audio_transcription" in event_type


def _safe_json(raw: str | None) -> Any:
    """Parse a JSON arguments string, falling back to {} on bad/missing input."""
    try:
        return json.loads(raw or "{}")
    except (json.JSONDecodeError, TypeError):
        return {}


def response_assistant_output(response: Any) -> dict[str, Any]:
    """Curated assistant message from a `response.done` payload.

    Used as the `llm` span's `outputs`: the spoken text the model produced this
    response, plus any tool calls it requested — the readable, AIMessage-shaped
    view of what the model returned, rather than the raw wire object.
    """
    texts: list[str] = []
    tool_calls: list[dict[str, Any]] = []
    for item in getattr(response, "output", None) or []:
        if getattr(item, "type", None) == "function_call":
            tool_calls.append(
                {
                    "name": getattr(item, "name", None),
                    "args": _safe_json(getattr(item, "arguments", None)),
                    "id": getattr(item, "call_id", None),
                }
            )
            continue
        for part in getattr(item, "content", None) or []:
            text = getattr(part, "transcript", None) or getattr(part, "text", None)
            if text:
                texts.append(text)
    out: dict[str, Any] = {"role": "assistant", "content": " ".join(texts).strip()}
    if tool_calls:
        out["tool_calls"] = tool_calls
    return out


def response_usage_metadata(response: Any) -> dict[str, int] | None:
    """Map Realtime token `usage` onto LangSmith `usage_metadata` (for cost).

    Returns None when the response carries no usage (e.g. a cancelled turn), so
    the caller can skip attaching it.
    """
    usage = getattr(response, "usage", None)
    if usage is None:
        return None

    def field(key: str) -> int | None:
        val = getattr(usage, key, None)
        if val is None and hasattr(usage, "get"):
            val = usage.get(key)
        return val

    inp, out, total = field("input_tokens"), field("output_tokens"), field("total_tokens")
    if inp is None and out is None and total is None:
        return None
    return {
        "input_tokens": inp or 0,
        "output_tokens": out or 0,
        "total_tokens": total if total is not None else (inp or 0) + (out or 0),
    }


async def execute_tool(name: str, arguments: str) -> dict:
    """Run one model-requested tool call and return its JSON-able result."""
    if name != "lookup_weather":
        return {"error": f"unknown tool: {name}"}
    try:
        args = json.loads(arguments or "{}")
    except json.JSONDecodeError:
        args = {}
    city = (args.get("city") or "").strip()
    if not city:
        return {"error": "missing city"}
    return await lookup_weather(city)

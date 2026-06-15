"""Turn an Agents-SDK realtime session event into a clean span payload.

The session emits *semantic* events (`agents.realtime.events`) — dataclasses
holding live SDK objects (the `RealtimeAgent`, tool objects, history items).
Dumping those wholesale would put unserializable junk on a span, so we walk the
event recursively and keep everything that reads cleanly while stripping what
would break or bloat the span: raw audio bytes, callables, private attributes,
and anything past the depth/width caps (see `_clean`). On top of that generic
view we overlay a few *promoted* fields per event type (role/text, tool args,
…) so the most useful data sits at the top level, plus the direction (`inbound`
= the user talking toward the model → span `inputs`).
"""

from __future__ import annotations

import dataclasses
from typing import Any

_MAX_DEPTH = 4  # how deep we recurse into nested SDK objects before bailing
_MAX_ITEMS = 50  # cap on collection width so one event never balloons a span
_MAX_REPR = 200  # truncate the repr() fallback for exotic objects


def _stringify(value: Any) -> Any:
    """Keep JSON-able values as-is; repr anything exotic so a span never breaks."""
    if value is None or isinstance(value, (str, int, float, bool, dict, list)):
        return value
    return repr(value)


def _shallow(value: Any) -> Any:
    """Last-resort view of a value we won't (or can't) recurse into."""
    name = getattr(value, "name", None)
    if isinstance(name, str):
        return name
    text = repr(value)
    return text if len(text) <= _MAX_REPR else text[:_MAX_REPR] + "…"


def _public_fields(value: Any) -> dict[str, Any] | None:
    """Public, non-callable attributes of a dataclass/object, or None."""
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        fields = {f.name: getattr(value, f.name, None) for f in dataclasses.fields(value)}
    else:
        fields = getattr(value, "__dict__", None)
    if not fields:
        return None
    return {k: v for k, v in fields.items() if not k.startswith("_") and not callable(v)}


def _clean(value: Any, depth: int = 0, seen: frozenset[int] = frozenset()) -> Any:
    """Recursively coerce any value into a compact, JSON-able span payload.

    Keeps readable data (scalars, dicts, lists, nested objects' public fields)
    and strips what would break or bloat a span: raw bytes, callables, private
    attributes, and anything past the depth/width caps. Cycles are broken via
    `seen`, the set of object ids already on the current path.
    """
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, (bytes, bytearray, memoryview)):
        return f"<{len(bytes(value))} bytes>"

    if depth >= _MAX_DEPTH:
        return _shallow(value)
    if id(value) in seen:
        return "<circular>"
    seen = seen | {id(value)}

    if isinstance(value, dict):
        out: dict[str, Any] = {}
        for i, (key, val) in enumerate(value.items()):
            if i >= _MAX_ITEMS:
                out["…"] = f"+{len(value) - _MAX_ITEMS} more"
                break
            if isinstance(key, str) and key.startswith("_"):
                continue
            out[str(key)] = _clean(val, depth + 1, seen)
        return out

    if isinstance(value, (list, tuple, set)):
        items = list(value)
        cleaned = [_clean(v, depth + 1, seen) for v in items[:_MAX_ITEMS]]
        if len(items) > _MAX_ITEMS:
            cleaned.append(f"… +{len(items) - _MAX_ITEMS} more")
        return cleaned

    fields = _public_fields(value)
    if fields:
        return {k: _clean(v, depth + 1, seen) for k, v in fields.items()}
    return _shallow(value)


def _tool_name(tool: Any) -> str | None:
    return getattr(tool, "name", None) or (repr(tool) if tool is not None else None)


def _item_text(item: Any) -> tuple[str | None, str | None]:
    """Best-effort (role, text) for a realtime history item.

    History items carry a `role` and a `content` list whose parts expose the
    spoken text as `transcript` (audio) or `text`. Field shapes vary across
    item kinds, so every access is defensive — missing pieces just yield None.
    """
    if item is None:
        return None, None
    role = getattr(item, "role", None)
    content = getattr(item, "content", None) or []
    parts = []
    for part in content:
        text = getattr(part, "transcript", None) or getattr(part, "text", None)
        if text:
            parts.append(text)
    return role, (" ".join(parts) or None)


def describe_event(event: Any) -> tuple[str, dict[str, Any], bool]:
    """Map a session event to `(name, span_payload, inbound)`.

    `name` labels the span, `span_payload` is a serializable view of the event —
    a full recursive dump (via `_clean`) with the most useful fields *promoted*
    to the top level — and `inbound` routes the payload to `inputs` (user→model)
    vs `outputs`.
    """
    etype = getattr(event, "type", None) or repr(event)
    dumped = _clean(event)
    payload: dict[str, Any] = dumped if isinstance(dumped, dict) else {"value": dumped}
    payload["type"] = etype
    inbound = False

    if etype in ("agent_start", "agent_end"):
        payload["agent"] = getattr(getattr(event, "agent", None), "name", None)

    elif etype == "tool_start":
        payload["tool"] = _tool_name(getattr(event, "tool", None))
        payload["arguments"] = _stringify(getattr(event, "arguments", None))
        inbound = True  # the model is asking us to run something → its input

    elif etype == "tool_end":
        payload["tool"] = _tool_name(getattr(event, "tool", None))
        payload["arguments"] = _stringify(getattr(event, "arguments", None))
        payload["output"] = _stringify(getattr(event, "output", None))

    elif etype == "audio_interrupted":
        payload["item_id"] = getattr(event, "item_id", None)

    elif etype == "history_added":
        role, text = _item_text(getattr(event, "item", None))
        payload["role"], payload["text"] = role, text
        inbound = role == "user"

    elif etype == "history_updated":
        history = getattr(event, "history", None) or []
        role, text = _item_text(history[-1]) if history else (None, None)
        payload["last_role"], payload["last_text"] = role, text
        payload["length"] = len(history)
        inbound = role == "user"

    elif etype == "handoff":
        payload["from_agent"] = getattr(getattr(event, "from_agent", None), "name", None)
        payload["to_agent"] = getattr(getattr(event, "to_agent", None), "name", None)

    elif etype == "guardrail_tripped":
        payload["message"] = _stringify(getattr(event, "message", None))

    elif etype == "error":
        payload["error"] = _stringify(getattr(event, "error", None))

    return etype, payload, inbound

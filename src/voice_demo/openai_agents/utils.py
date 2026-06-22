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
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ..console import StatusUI
    from ..sdk_tracing import EventSession

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


def raw_input_transcript(event: Any) -> tuple[str | None, str] | None:
    """User transcript from a `raw_model_event` wrapping an input-audio
    transcription completion.

    This is the wire-level source the raw OpenAI backend reads directly. The
    Agents SDK passes it through as a `raw_model_event`, and it sometimes carries
    a user utterance the semantic `history` never surfaces (notably a barge-in).
    Returns `(item_id, transcript)` for a non-empty transcript, else None.
    """
    data = getattr(event, "data", None)
    if getattr(data, "type", None) != "input_audio_transcription_completed":
        return None
    transcript = (getattr(data, "transcript", None) or "").strip()
    if not transcript:
        return None
    return getattr(data, "item_id", None), transcript


def history_item(event: Any) -> tuple[str | None, str | None, str | None]:
    """`(item_id, role, text)` for a `history_added` event's single item."""
    item = getattr(event, "item", None)
    role, text = _item_text(item)
    return getattr(item, "item_id", None), role, text


def history_messages(event: Any) -> list[tuple[str | None, str | None, str | None]]:
    """`(item_id, role, text)` for every item in a `history_updated` snapshot.

    The realtime session delivers the full conversation `history` on every
    update, so this is the authoritative source for the transcript: walking it
    (rather than the streamed partials) reliably yields *all* user and assistant
    messages, each with its stable `item_id` so callers can collapse partials by
    keeping the latest text per id.
    """
    out: list[tuple[str | None, str | None, str | None]] = []
    for item in getattr(event, "history", None) or []:
        role, text = _item_text(item)
        out.append((getattr(item, "item_id", None), role, text))
    return out


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


class HistoryTracker:
    """Reconstructs the conversation from history snapshots and emits its spans.

    The Agents SDK delivers the full conversation `history` on every history
    event — the streamed events alone mis-deliver user messages and emit the
    assistant transcript as growing partials. This tracker folds those snapshots
    into a map keyed by stable `item_id` (latest non-empty text per id wins, so
    partials collapse and every user/assistant message is captured) and emits one
    span per message once it finalizes: a curated `user_message` span or an
    assistant `model` (`llm`) span, grouped into per-user-turn `turn` spans. It
    also owns the per-turn first-audio latency timer (armed when a new user turn
    opens, recorded on the first agent audio chunk).

    All trace writes go through the injected `EventSession`; console lines go
    through the injected `StatusUI`. The tracker does no I/O of its own.
    """

    def __init__(self, trace: EventSession, ui: StatusUI) -> None:
        self._trace = trace
        self._ui = ui
        # item_id → {"role", "text"}, in arrival order (dict preserves insertion).
        self._items: dict[str, dict[str, Any]] = {}
        self._emitted: set[str] = set()  # item_ids already turned into spans
        self._logged: set[tuple[str, str]] = set()  # (role, text) already printed
        # Per-turn latency: when the current user turn opened; None = not armed.
        self._latency_since: float | None = None

    def observe(self, seq: list[tuple], received_at: float) -> None:
        """Fold `(item_id, role, text)` tuples into the map.

        A brand-new user item begins a turn: the previous turn's messages are
        flushed first (so they stay grouped under it), then a new `turn` span
        opens and the latency timer is armed.
        """
        for iid, role, text in seq:
            if not iid:
                continue
            if iid not in self._items and role == "user":
                self.flush(hold_last=False)
                self._trace.start_turn()
                self._latency_since = received_at
            cur = self._items.get(iid)
            if cur is None:
                self._items[iid] = {"role": role, "text": text}
            else:
                if role:
                    cur["role"] = role
                if text:  # latest non-empty text wins → collapses partials
                    cur["text"] = text
            # Print to the console promptly as text arrives (deduped), so the
            # agent's reply shows immediately — independent of when its span is
            # emitted (which waits for the item to be superseded).
            if role in ("user", "assistant") and text:
                self._log_line(role, text)

    def flush(self, hold_last: bool) -> None:
        """Emit not-yet-emitted items in order, optionally holding back the last
        (still-streaming) item until it is superseded or the session ends."""
        ids = list(self._items)
        cutoff = len(ids) - 1 if hold_last else len(ids)
        for iid in ids[: max(0, cutoff)]:
            if iid not in self._emitted:
                self._record_item(iid)

    def record_first_audio(self, received_at: float) -> None:
        """Record `latency_to_first_audio_ms` on the open turn, once per turn."""
        if self._latency_since is not None:
            self._trace.add_turn_metadata(
                latency_to_first_audio_ms=round((received_at - self._latency_since) * 1000)
            )
            self._latency_since = None

    def _log_line(self, role: str | None, text: str | None) -> None:
        """Print one transcript line to the console once.

        Console-only and deduped (history repeats/streams text). Kept separate
        from the trace transcript (`add_message`) so the console can print
        promptly as text arrives, while span emission still waits for an item to
        finalize — otherwise the agent's reply wouldn't print until the next user
        turn supersedes it.
        """
        if not role or not text or (role, text) in self._logged:
            return
        self._logged.add((role, text))
        self._ui.log(f"{'user: ' if role == 'user' else 'agent:'} {text}")

    def _record_item(self, iid: str) -> None:
        """Emit one message span for a finalized history item (once)."""
        role = self._items[iid]["role"]
        text = (self._items[iid]["text"] or "").strip()
        if role not in ("user", "assistant"):
            self._emitted.add(iid)  # tool/system items are never messages
            return
        if not text:
            return  # message text not in yet (may arrive late, e.g. a barge-in
            # transcript recovered from raw_model_event) — leave pending
        self._emitted.add(iid)
        self._log_line(role, text)  # console (no-op if already printed promptly)
        self._trace.add_message(role, text)  # the trace transcript
        if role == "user":
            with self._trace.event_span(
                {"item_id": iid, "role": "user", "content": text},
                self._trace.now(),
                name="user_message",
                inbound=True,
                inputs={"role": "user", "content": text},
            ):
                pass
        else:
            # Assistant message → a point-in-time `model` `llm` span under the turn.
            self._trace.record_llm(outputs={"role": "assistant", "content": text})

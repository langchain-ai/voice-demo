"""LangSmith tracing for the OpenAI Agents SDK realtime backend.

This module owns *everything* that turns the Agents SDK's semantic event stream
into a LangSmith trace, kept apart from the application logic in `agent.py`: the
agent's loop wraps each event in `HistoryTracker.track_event(...)` and its body
holds only application side-effects (play audio, set UI state). All span, turn,
transcript, and metric decisions live here.

Two layers:

- **Payload shaping.** The session emits *semantic* events
  (`agents.realtime.events`) — dataclasses holding live SDK objects (the
  `RealtimeAgent`, tool objects, history items). Dumping those wholesale would
  put unserializable junk on a span, so `_clean` walks the event recursively,
  keeping everything that reads cleanly while stripping what would break or
  bloat the span (raw audio bytes, callables, private attributes, anything past
  the depth/width caps). `describe_event` overlays a few *promoted* fields per
  event type (role/text, tool args, …) plus the direction (`inbound`).
- **`HistoryTracker`.** Reconstructs the conversation from `history` snapshots,
  emits spans grouped into turns, and routes every event via `track_event`.

## Trace shape — one conversation = one trace

Same machinery as the raw backend (`voice_demo.sdk_tracing.EventSession`): one
root `realtime_session` span (full transcript in `outputs`, stereo WAV attached)
grouped into `turn`s.

The conversation is reconstructed from the **full `history` snapshot** the
session delivers on every `history_updated` — not from the streamed events,
which mis-deliver user messages and emit the assistant transcript as growing
partials. We fold the snapshot into an item map keyed by stable `item_id`
(latest text per id wins → partials collapse, every user/assistant message is
captured), and emit one span per message once it's finalized: a curated
`user_message` span or an assistant `model` (`llm`) span. A new user item opens
a `turn`. (Token usage isn't available — the SDK has no per-response usage
events yet.)

Everything else stays on the live event stream: `tool_start`/`tool_end` (the
SDK-run tool, a `tool` span), `audio_end`/`agent_*` (the timeline that drives
the audio-player sync), `latency_to_first_audio_ms`, and `was_interrupted` on
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

import dataclasses
from collections.abc import Iterator
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any

from langsmith.run_helpers import tracing_context

if TYPE_CHECKING:
    from langsmith import RunTree

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
        fields = {
            f.name: getattr(value, f.name, None) for f in dataclasses.fields(value)
        }
    else:
        fields = getattr(value, "__dict__", None)
    if not fields:
        return None
    return {
        k: v for k, v in fields.items() if not k.startswith("_") and not callable(v)
    }


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
        payload["from_agent"] = getattr(
            getattr(event, "from_agent", None), "name", None
        )
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

    All trace writes go through the injected `EventSession`; console lines
    derived from the reconstructed transcript go through the injected `StatusUI`
    (the tracker is the only place that knows when a transcript line is new, so
    that one bit of console output lives here rather than in the app). The
    tracker does no I/O of its own.

    `track_event` is the single entry point the agent loop calls per event; it
    routes audio/history/raw/tool/barge-in events to the logic below.
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

    @contextmanager
    def trace_context(self, *, tags: list[str]) -> Iterator[None]:
        """Session-level LangSmith context (project + tags) for the whole run."""
        with tracing_context(
            metadata={"thread_id": self._trace.thread_id},
            tags=tags,
            project_name=self._trace.project_name,
        ):
            yield

    @contextmanager
    def track_event(self, event: Any) -> Iterator[RunTree | None]:
        """Observe one session event and span it where warranted.

        The body of the `with` is the application's handler (audio, UI state).
        Most events are folded into the conversation map or dropped and yield
        `None`; the remaining live events get a span.
        """
        received_at = self._trace.now()
        etype = event.type

        # Raw agent audio: played by the app, not spanned. Record first-audio
        # latency on the way through.
        if etype == "audio":
            self.record_first_audio(received_at)
            yield None
            return

        # The SDK's wire-protocol passthrough — not spanned, but it carries the
        # wire-level user transcript the semantic `history` sometimes omits
        # (notably a barge-in). Mine that one event into the item map.
        if etype == "raw_model_event":
            rec = raw_input_transcript(event)
            if rec:
                item_id, transcript = rec
                self.observe([(item_id, "user", transcript)], received_at)
                self.flush(hold_last=True)
            yield None
            return

        # History events feed the authoritative item map; not spanned directly
        # (they stream partials). Messages emit as user/`model` spans from the
        # map once an item is finalized.
        if etype == "history_added":
            self.observe([history_item(event)], received_at)
            self.flush(hold_last=True)
            yield None
            return
        if etype == "history_updated":
            self.observe(history_messages(event), received_at)
            self.flush(hold_last=True)
            yield None
            return

        # Bookkeeping markers (the agent re-activates on each model pass in the
        # tool loop): no I/O, not spanned. The app still reacts to agent_start.
        if etype in ("agent_start", "agent_end"):
            yield None
            return

        # Remaining live events get a span. tool_end becomes a `tool` run; a
        # barge-in flags the open turn. The full describe_event payload is
        # preserved under metadata.raw_event by event_span.
        name, payload, inbound = describe_event(event)
        span_kwargs: dict[str, Any] = {"inbound": inbound}
        if etype == "tool_end":
            span_kwargs["run_type"] = "tool"
            span_kwargs["inputs"] = {"arguments": payload.get("arguments")}
            span_kwargs["outputs"] = {"output": payload.get("output")}
        elif etype == "audio_interrupted":
            self._trace.add_turn_metadata(was_interrupted=True)

        with self._trace.event_span(
            payload, received_at, name=name, **span_kwargs
        ) as run:
            yield run

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
                latency_to_first_audio_ms=round(
                    (received_at - self._latency_since) * 1000
                )
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

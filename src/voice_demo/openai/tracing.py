"""LangSmith tracing for the raw OpenAI Realtime backend.

This module owns *everything* that turns the Realtime WebSocket event stream
into a LangSmith trace. It is deliberately kept apart from the application logic
in `agent.py`: the agent's event loop wraps each event in
`RealtimeTracer.track_event(...)` and its body contains only application
side-effects (play audio, set UI state, run tools, ask the model to respond).
Every trace decision lives here — which events become spans, how they group into
turns, the conversation transcript rollup, the per-turn latency/interruption
metrics, and the point-in-time `llm` spans.

The tracer wraps the shared `voice_demo.sdk_tracing.EventSession` (root span,
turns, WAV attachment, payload scrubbing); this module is the OpenAI-Realtime
*policy* layered on that mechanism.

## Trace shape — one conversation = one trace

Every *meaningful* received WebSocket event becomes its own span, in arrival
order, carrying that event's full (scrubbed) payload — in `inputs` if the user
sent it toward the model (speech buffer, transcription), in `outputs` if the
model/server sent it back (`response.*`, `error`). The root span's `outputs`
carries the running conversation transcript, so the LangSmith preview pane shows
the whole exchange at a glance.

Three classes of event are dropped to keep the trace readable: raw audio
(`response.output_audio.delta`) is played but not spanned; every other streaming
partial (`*.delta`) is dropped since the terminal `*.done` repeats the full
payload; and purely structural bookkeeping events (`NOISE_EVENTS` — item/part
add/done lifecycle markers that duplicate what `response.done` and the
transcript events already carry) are dropped too. Work the agent does *while
handling* an event (e.g. tool execution) nests inside that event's span: the
tracer sets `tracing_context(parent=...)` for the duration of the `with` body,
so the app's tool calls auto-nest without ever touching the run object.

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
conversation. The root is titled with the first user utterance; each turn
records `latency_to_first_audio_ms` and, on a barge-in, `was_interrupted`.

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

import json
from collections.abc import Iterator
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any

from langsmith.run_helpers import tracing_context

if TYPE_CHECKING:
    from langsmith import RunTree

    from ..audio import AudioOutput
    from ..sdk_tracing import EventSession


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


class RealtimeTracer:
    """Turns the Realtime event stream into a LangSmith trace, one event at a time.

    The agent's loop wraps every event in `track_event(event)`; the tracer
    decides — privately — whether to open a span, open/close a turn, append to
    the transcript rollup, set the trace title, time first-audio latency, flag a
    barge-in, and record the `llm` span. All trace writes go through the injected
    `EventSession`.

    The tracer also holds a *read-only* reference to the speaker (`audio_out`).
    The wire stream has no "playout finished" event, so the only way to know the
    agent was still audible when the user spoke (a barge-in) is the speaker's
    queued-byte count. The tracer only ever reads it; the app owns clearing it.
    """

    def __init__(self, session: EventSession, audio_out: AudioOutput) -> None:
        self._session = session
        self._audio_out = audio_out
        # Per-turn latency: end-of-user-speech → first agent audio. Armed at
        # speech_stopped, recorded on the first audio chunk, then cleared.
        # None = not armed.
        self._await_audio_since: float | None = None

    @contextmanager
    def trace_context(self, *, tags: list[str]) -> Iterator[None]:
        """Session-level LangSmith context (project + tags) for the whole run.

        Wraps the conversation so any traceable calls (e.g. tool execution) land
        in the right project/thread even outside an open span.
        """
        with tracing_context(
            metadata={"thread_id": self._session.thread_id},
            tags=tags,
            project_name=self._session.project_name,
        ):
            yield

    @contextmanager
    def track_event(self, event: Any) -> Iterator[RunTree | None]:
        """Observe one event and span it where warranted; yields for its handler.

        The body of the `with` is the application's handler for this event. For
        events that get a span, anything the handler traces (tool execution)
        nests under it. For events that get no span, the tracer still observes
        them (latency, transcript, turn boundaries) and yields `None`.
        """
        received_at = self._session.now()
        etype = event.type

        # No-span events: observed for side-state, but never get their own span.
        if etype == "response.output_audio.delta":
            self._record_first_audio(received_at)
            yield None
            return
        if etype.endswith(".delta") or etype in NOISE_EVENTS:
            yield None
            return
        if etype == "response.output_audio_transcript.done":
            # Already the `model` span's content (recorded at response.done);
            # fold it into the rollup but don't span it twice. The app logs it.
            self._session.add_message("assistant", event.transcript or "")
            yield None
            return

        # Turn boundaries + transcript rollup happen before the span opens.
        self._before_span(event, received_at)

        with self._session.event_span(
            event, received_at, name=etype, **self._span_kwargs(event)
        ) as run:
            if etype == "conversation.item.input_audio_transcription.failed":
                # Surface a failed transcription as an errored span so "it didn't
                # understand me" is debuggable.
                run.error = f"transcription_failed: {getattr(event, 'error', None)}"
            elif etype == "response.done":
                self._record_response_llm(run, event.response)
            # The app's handler runs here; whatever it traces nests under `run`.
            with tracing_context(parent=run):
                yield run

    def _before_span(self, event: Any, received_at: float) -> None:
        """Turn/transcript/title side-state applied as a span is about to open."""
        etype = event.type
        if etype == "input_audio_buffer.speech_started":
            # A user starting to speak opens a new turn (also the barge-in
            # signal). If the speaker still has audio queued, the turn being
            # closed was talked over — flag it before start_turn() closes it.
            if self._audio_out.buffered_bytes() > 0:
                self._session.add_turn_metadata(was_interrupted=True)
            self._session.start_turn()
            # A new turn supersedes any latency measurement still pending from
            # the previous turn — discard it so it can't be misattributed.
            self._await_audio_since = None
        elif etype == "input_audio_buffer.speech_stopped":
            # Arm the latency timer: from here to the first agent audio chunk is
            # the perceived response lag.
            self._await_audio_since = received_at
        elif etype == "conversation.item.input_audio_transcription.completed":
            text = (event.transcript or "").strip()
            if text:
                self._session.add_message("user", text)
                self._session.set_title(text)  # first utterance → trace title
            else:
                self._await_audio_since = None  # nothing transcribed
        elif etype == "conversation.item.input_audio_transcription.failed":
            self._await_audio_since = None

    def _span_kwargs(self, event: Any) -> dict[str, Any]:
        """Curated, conversation-shaped I/O for the readable events.

        Everything else keeps its raw wire payload (`event_span`'s default).
        """
        etype = event.type
        kwargs: dict[str, Any] = {"inbound": is_inbound(etype)}
        if etype == "conversation.item.input_audio_transcription.completed":
            text = (event.transcript or "").strip()
            if text:
                kwargs["inputs"] = {"role": "user", "content": text}
        elif etype == "response.done":
            # Keep this span a `chain` wrapper; the model payload (assistant
            # message, tokens, status) goes on the child `llm` span so local tool
            # latency in the app's handler is never attributed to the model. The
            # raw response is still preserved under metadata.raw_event.
            kwargs["outputs"] = {}
        return kwargs

    def _record_first_audio(self, received_at: float) -> None:
        """Record `latency_to_first_audio_ms` on the open turn, once per turn."""
        if self._await_audio_since is not None:
            self._session.add_turn_metadata(
                latency_to_first_audio_ms=round(
                    (received_at - self._await_audio_since) * 1000
                )
            )
            self._await_audio_since = None

    def _record_response_llm(self, run: RunTree, response: Any) -> None:
        """Point-in-time `llm` child span for a `response.done` payload."""
        self._session.record_llm(
            run,
            outputs=response_assistant_output(response),
            usage_metadata=response_usage_metadata(response),
            metadata={"status": getattr(response, "status", None)},
        )


def is_inbound(event_type: str) -> bool:
    """Direction of an event relative to the model.

    Inbound = something the user sent toward the model (their speech buffer,
    their transcription) → goes in span `inputs`. Everything else (`response.*`,
    `error`, `session.*`) is the model/server talking back → span `outputs`.
    """
    return (
        event_type.startswith("input_audio_buffer")
        or "input_audio_transcription" in event_type
    )


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

    inp, out, total = (
        field("input_tokens"),
        field("output_tokens"),
        field("total_tokens"),
    )
    if inp is None and out is None and total is None:
        return None
    return {
        "input_tokens": inp or 0,
        "output_tokens": out or 0,
        "total_tokens": total if total is not None else (inp or 0) + (out or 0),
    }

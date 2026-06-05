"""OTel → LangSmith bridge for LiveKit Agents.

LangSmith's OTel ingester reads the standard `gen_ai.*` and `langsmith.*`
attribute namespaces. LiveKit emits its data under a `lk.*` vendor prefix the
ingester doesn't recognize — so without translation, transcripts, confidence
scores, TTS metrics, EOU probabilities, latency numbers, etc. all evaporate at
ingest. This module is an in-process OTel `SpanProcessor` that rewrites `lk.*`
into the keys LangSmith understands, before each span is exported downstream.

Design — generic to any LiveKit agent, not this demo
----------------------------------------------------
The processor only ever reads LiveKit's own `lk.*` attributes and the span's
name/position in the trace. It carries **no knowledge of what the LLM stage is**
— whether that's a bare chat model, this demo's LangGraph brain, or anything
else. Two rules make that work:

  1. **Translate LiveKit spans from their own data.** Each known LiveKit span
     (`user_turn`, `llm_request`, `tts_*`, `agent_turn`, `function_tool`, …)
     is classified and rendered from the `lk.*` attributes and `gen_ai.*`
     span events LiveKit always sets on it (`lk.user_transcript`,
     `gen_ai.user.message` events, `lk.tts_metrics`, `lk.function_tool.*`, …).
  2. **Leave everything else untouched.** Any span that isn't a LiveKit span —
     e.g. the LangChain/LangGraph runs that ride through the same OTel provider
     when `LANGSMITH_TRACING_MODE=otel` — already arrives in LangSmith's native
     shape (run type, `gen_ai.prompt`/`gen_ai.completion`), so the processor
     exports it verbatim. It never reshapes another framework's runs.

That's the whole point: drop a different brain into the LiveKit `llm` slot and
this processor needs no changes.

Two translation layers for LiveKit spans
----------------------------------------
  1. **Per-span classification** — set `langsmith.span.kind` and pull the
     conversation into `gen_ai.prompt` / `gen_ai.completion` so it renders as
     messages. The genuine inference nodes (STT `user_turn`, the chat
     completion `llm_request`, the TTS synthesis `tts_request`) are
     `llm`-kind; the framework wrappers (`llm_node`, `tts_node`,
     retry-attempt spans) are `chain` with no fabricated I/O.
  2. **Blanket `lk.*` pass-through** — every `lk.*` scalar lands as
     `langsmith.metadata.lk_<name>`, and JSON-object blobs (`lk.tts_metrics`,
     `lk.llm_metrics`, …) are flattened so their scalar leaves become discrete
     `lk_<name>_<subkey>` fields. No data is silently dropped, and new LiveKit
     attributes surface automatically. Per-stage latencies (transcription
     delay, eou endpointing delay, llm ttft, tts ttfb, e2e latency) surface
     this way too, each on its own stage's span.

The whole-conversation transcript on the root span is accumulated from the
per-turn `agent_turn` spans (LiveKit's own turn boundary), and the call
recording is attached to the root via `langsmith.attachments`.

Thread grouping is opt-in, matching the rest of the LangSmith ecosystem (the
user supplies the id; the integration only propagates it). Pass
`thread_id_provider` — a zero-arg callable returning the current conversation's
id, e.g. a ContextVar's `.get` — and the processor stamps
`langsmith.metadata.thread_id` on every span, children included, which
LangSmith needs for thread-level filtering and token/cost aggregation.
Likewise `audio_path_provider` tells the processor where the call recording
lives so it can attach it to the root span. With neither set, the processor
needs nothing from the host application and sets no thread metadata.
"""

from __future__ import annotations

import base64
import json
import logging
from pathlib import Path
from typing import Any, Callable, Optional

from opentelemetry.sdk.trace import ReadableSpan, Span, SpanProcessor
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter

logger = logging.getLogger("langsmith_processor")

# LiveKit's span names, grouped by how we treat them. The genuine inference
# calls are `llm`-kind; the framework wrappers around them are `chain`.
_STT_SPAN = "user_turn"                       # audio → transcript (inference)
_LLM_INFERENCE_SPAN = "llm_request"           # chat completion (inference)
_LLM_WRAPPER_SPANS = {"llm_node", "llm_request_run"}
_TTS_INFERENCE_SPAN = "tts_request"           # text → audio (inference)
_TTS_WRAPPER_SPANS = {"tts_node", "tts_request_run"}
_TURN_SPAN = "agent_turn"
_SESSION_SPAN = "agent_session"
_TOOL_SPAN = "function_tool"
_REALTIME_METRICS_SPAN = "realtime_metrics"

# LiveKit records each chat-context item sent to the model as a span event on
# `llm_request` — the event name implies the message role — followed by a
# `gen_ai.choice` event carrying the generated message.
_LLM_EVENT_ROLES = {
    "gen_ai.system.message": "system",
    "gen_ai.user.message": "user",
    "gen_ai.assistant.message": "assistant",
    "gen_ai.tool.message": "tool",
}
_LLM_CHOICE_EVENT = "gen_ai.choice"


class LangSmithSpanProcessor(SpanProcessor):
    """Enriches LiveKit Agents' OTel spans with LangSmith-compatible attributes.

    Enables conversation tracking and message visualization in LangSmith's UI by
    translating LiveKit's `lk.*` namespace into `gen_ai.*` / `langsmith.*`.
    """

    def __init__(
        self,
        downstream_processor: Optional[SpanProcessor] = None,
        *,
        thread_id_provider: Optional[Callable[[], Optional[str]]] = None,
        audio_path_provider: Optional[Callable[[], Optional[Path]]] = None,
    ):
        super().__init__()
        if downstream_processor is None:
            downstream_processor = BatchSpanProcessor(OTLPSpanExporter())
        self.downstream = downstream_processor
        # App-owned context providers (see module docstring); None disables.
        self.thread_id_provider = thread_id_provider
        self.audio_path_provider = audio_path_provider
        # Whole-conversation transcript, accumulated turn-by-turn from the
        # `agent_turn` spans. trace_id -> flat [{role, content}, ...]. Rendered
        # onto the root span and cleared when the trace ends.
        self._conversation_by_trace: dict[int, list[dict]] = {}
        # The trace root span (job entrypoint) ends before the conversation is
        # complete when the agent greets first, so we hold it until `agent_session`
        # ends with the full conversation. trace_id -> ReadableSpan.
        self.deferred_root_spans: dict[str, ReadableSpan] = {}
        # trace_ids whose `agent_session` span has ended — a root ending after
        # that exports immediately instead of deferring.
        self._ended_sessions: set[str] = set()
        # Realtime-model metrics cached from a `realtime_metrics` child span and
        # drained onto its parent `agent_turn`. Keyed by the parent span_id.
        self._realtime_metrics_by_parent: dict[int, dict] = {}

    # -- span lifecycle -------------------------------------------------------

    def on_start(self, span: Span, parent_context=None) -> None:
        self.downstream.on_start(span, parent_context)

    def on_end(self, span: ReadableSpan) -> None:
        """Translate one ended span into LangSmith's attribute shape, then export."""
        trace_id = format(span.context.trace_id, "032x")

        # Thread grouping (opt-in): LangSmith needs the thread_id on every run
        # for thread-level filtering and token/cost aggregation. Never clobber
        # an id set upstream.
        if (
            self.thread_id_provider is not None
            and "langsmith.metadata.thread_id" not in span.attributes
        ):
            thread_id = self.thread_id_provider()
            if thread_id:
                span._attributes["langsmith.metadata.thread_id"] = str(thread_id)

        name = span.name

        if name == _STT_SPAN:
            self._handle_stt(span)
        elif name == _LLM_INFERENCE_SPAN:
            self._handle_llm_request(span)
        elif name in _LLM_WRAPPER_SPANS:
            # llm_node (pipeline stage) / llm_request_run (retry attempt):
            # wrappers, no fabricated I/O.
            span._attributes["langsmith.span.kind"] = "chain"
        elif name == _TTS_INFERENCE_SPAN or name in _TTS_WRAPPER_SPANS:
            self._handle_tts(span)
        elif name == _TURN_SPAN:
            self._handle_turn(span)
        elif name == _SESSION_SPAN:
            # Session end: a framework wrapper (chain). The conversation is now
            # complete — release the deferred root span (if any).
            span._attributes["langsmith.span.kind"] = "chain"
            self._ended_sessions.add(trace_id)
            self._release_root_span(trace_id)
        elif name == "eou_detection":
            # End-of-utterance detection: a framework step.
            span._attributes["langsmith.span.kind"] = "chain"
        elif name == _TOOL_SPAN:
            self._handle_tool(span)
        elif name == _REALTIME_METRICS_SPAN:
            # These metrics belong ON the turn, not as their own span: cache them
            # for the parent agent_turn and suppress this span entirely.
            self._cache_realtime_metrics(span)
            return
        elif span.parent is None:
            # The trace root (LiveKit's job entrypoint). Render the rolled-up
            # conversation, attach the recording, and treat it as the LangSmith
            # root — or defer if the conversation hasn't started yet.
            self._handle_root(span, trace_id)
            return
        # Any other span is not a LiveKit span — e.g. a LangChain/LangGraph run
        # riding through the same OTel provider. It already arrives in LangSmith's
        # native shape, so we leave it untouched and just export it.

        self._export_span(span)

    # -- per-span-type handlers ----------------------------------------------

    def _handle_stt(self, span: ReadableSpan) -> None:
        """STT (`user_turn`): audio input → transcribed text. An inference node."""
        span._attributes["langsmith.span.kind"] = "llm"
        model = span.attributes.get("gen_ai.request.model")
        if model:
            span._attributes["langsmith.metadata.model_name"] = str(model)

        transcript = span.attributes.get("lk.user_transcript")
        self._set_messages(span, prompt=[{"role": "user", "content": f"Audio for: \"{transcript}\""}])
        if transcript:
            self._set_messages(
                span, completion=[{"role": "assistant", "content": str(transcript)}]
            )
        # The audio → transcript pair isn't a conversation turn; keep it out of
        # the Messages view (it stays visible in the trace tree).
        span._attributes["langsmith.metadata.ls_message_view_exclude"] = True

    def _handle_llm_request(self, span: ReadableSpan) -> None:
        """`llm_request`: the actual chat-completion call. An inference node.

        LiveKit records the request's I/O as span events: one
        `gen_ai.{system,user,assistant,tool}.message` event per chat-context
        item (the full conversation sent to the model, in order), then a
        `gen_ai.choice` event with the generated message. LangSmith's own event
        mapping routes those by role — the conversation's assistant history
        would land in *outputs* — and the plain message events carry no `role`
        attribute (it's implied by the event name), so they render as type
        "unknown". We rebuild instead: every message event becomes part of
        `gen_ai.prompt`, the choice becomes `gen_ai.completion`, and the
        translated events are stripped so the ingester doesn't render them a
        second time. Model and token usage (`gen_ai.request.model`,
        `gen_ai.usage.*`) are already native attributes on this span.

        Unlike the other handlers, this writes ONLY the singular JSON
        attributes, as LangChain-format messages (`{"messages": [...]}` on
        both sides — see the "Log LLM calls" LangSmith docs). The indexed
        `gen_ai.prompt.{n}.*` form would take precedence at ingest and can
        only carry role/content, which loses the structured tool calls.
        """
        span._attributes["langsmith.span.kind"] = "llm"

        prompt: list[dict] = []
        completion: list[dict] = []
        for event in span.events:
            if event.name == _LLM_CHOICE_EVENT:
                completion.append(self._message_from_event("assistant", event))
            elif (role := _LLM_EVENT_ROLES.get(event.name)) is not None:
                prompt.append(self._message_from_event(role, event))
        if prompt:
            span._attributes["gen_ai.prompt"] = json.dumps({"messages": prompt})
        if completion:
            span._attributes["gen_ai.completion"] = json.dumps({"messages": completion})

        # Drop the translated events (keep any others, e.g. exceptions) so the
        # ingester's own event mapping doesn't render the same messages again.
        span._events = [
            e
            for e in span.events
            if e.name != _LLM_CHOICE_EVENT and e.name not in _LLM_EVENT_ROLES
        ]

    @staticmethod
    def _message_from_event(role: str, event: Any) -> dict:
        """Build a LangChain-format message dict from a LiveKit gen_ai span event.

        Plain messages carry `content`. Tool-calling assistant messages (and
        choices) carry `tool_calls` as JSON strings in the OpenAI shape
        `{"function": {"name", "arguments"}, "id", "type"}`; LangChain format
        wants them flat — `{"name", "args", "id", "type": "tool_call"}` with
        `args` as an object — so LangSmith renders proper tool-call blocks on
        the assistant message. Tool-result events carry the call id, linked
        back via top-level `tool_call_id`.
        """
        attrs = event.attributes or {}
        msg: dict = {
            "role": str(attrs.get("role") or role),
            "content": str(attrs.get("content") or ""),
        }
        tool_calls = []
        for raw_call in attrs.get("tool_calls") or ():
            if isinstance(raw_call, str):
                try:
                    raw_call = json.loads(raw_call)
                except json.JSONDecodeError:
                    continue
            if not isinstance(raw_call, dict):
                continue
            fn = raw_call.get("function") or {}
            args = fn.get("arguments")
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except json.JSONDecodeError:
                    pass
            tool_calls.append(
                {
                    "type": "tool_call",
                    "id": raw_call.get("id"),
                    "name": fn.get("name"),
                    "args": args if isinstance(args, dict) else {},
                }
            )
        if tool_calls:
            msg["tool_calls"] = tool_calls
        if role == "tool":
            if attrs.get("id"):
                msg["tool_call_id"] = str(attrs["id"])
            if attrs.get("name"):
                msg["name"] = str(attrs["name"])
        return msg

    def _handle_tts(self, span: ReadableSpan) -> None:
        """TTS spans: text → audio.

        `tts_request` is the actual synthesis call, so it's the `llm` node:
        input text in, generated-audio marker out, with the model/voice as
        metadata (from the `lk.tts_metrics` blob). `tts_node` (pipeline-stage
        wrapper) and `tts_request_run` (one retry attempt) are `chain`s with no
        fabricated I/O — they carry just their duration and `lk.*` metadata
        (e.g. retry count) via the blanket pass-through.
        """
        if span.name != _TTS_INFERENCE_SPAN:
            span._attributes["langsmith.span.kind"] = "chain"
            return

        span._attributes["langsmith.span.kind"] = "llm"
        # The text → audio pair isn't a conversation turn; keep it out of the
        # Messages view (still visible in the trace tree).
        span._attributes["langsmith.metadata.ls_message_view_exclude"] = True

        text = (
            span.attributes.get("lk.input_text")
            or span.attributes.get("lk.request.text")
            or span.attributes.get("lk.text")
            or ""
        )
        self._set_messages(
            span,
            prompt=[{"role": "user", "content": str(text)}],
            completion=[{"role": "assistant", "content": f"Generated audio for: \"{text}\""}],
        )

        # Model/voice are metadata, not conversation content. LiveKit puts the
        # model name inside tts_request's lk.tts_metrics blob.
        model = span.attributes.get("gen_ai.request.model")
        if not model:
            metrics = self._try_parse_json_object(span.attributes.get("lk.tts_metrics"))
            if isinstance(metrics, dict):
                model = (metrics.get("metadata") or {}).get("model_name") or metrics.get(
                    "model_name"
                )
        if model:
            span._attributes["gen_ai.request.model"] = str(model)
            span._attributes["langsmith.metadata.model_name"] = str(model)

    def _handle_turn(self, span: ReadableSpan) -> None:
        """`agent_turn`: one user↔assistant exchange.

        Renders directly from LiveKit's per-turn attributes — `lk.user_input`
        (what the user said) and `lk.response.text` (what the agent said) — and
        appends them to the trace's running conversation, which the root span
        rolls up. This is the LiveKit-native turn boundary, so it works for both
        the STT/LLM/TTS pipeline and the speech-to-speech realtime backends.
        """
        span._attributes["langsmith.span.kind"] = "chain"
        # Fold in any realtime-model metrics cached from a child span.
        self._drain_realtime_metrics(span)

        user_input = span.attributes.get("lk.user_input")
        response = span.attributes.get("lk.response.text")
        conversation = self._conversation_by_trace.setdefault(span.context.trace_id, [])
        if user_input:
            msg = {"role": "user", "content": str(user_input)}
            self._set_messages(span, prompt=[msg])
            conversation.append(msg)
        if response:
            msg = {"role": "assistant", "content": str(response)}
            self._set_messages(span, completion=[msg])
            conversation.append(msg)

    def _handle_tool(self, span: ReadableSpan) -> None:
        """`function_tool`: render as a proper `tool` run with its I/O."""
        span._attributes["langsmith.span.kind"] = "tool"
        tool_name = span.attributes.get("lk.function_tool.name")
        if tool_name:
            span._attributes["langsmith.metadata.tool_name"] = str(tool_name)
        args = span.attributes.get("lk.function_tool.arguments")
        if args is not None:
            span._attributes["gen_ai.prompt"] = (
                args if isinstance(args, str) else json.dumps(args)
            )
        output = span.attributes.get("lk.function_tool.output")
        if output is not None:
            span._attributes["gen_ai.completion"] = (
                output if isinstance(output, str) else json.dumps(output)
            )

    def _handle_root(self, span: ReadableSpan, trace_id: str) -> None:
        """The trace root (job entrypoint): the LangSmith conversation root.

        Marks it the root, sets audio modality, and renders the rolled-up
        conversation. In console mode the root ends right after the greeting
        (the entrypoint returns while the conversation continues), so unless
        `agent_session` has already ended we defer the root and release it,
        complete, when the session ends.
        """
        span._attributes["langsmith.span.kind"] = "chain"
        span._attributes["langsmith.root_span"] = True
        span._attributes["langsmith.metadata.ls_modality"] = "audio"

        if trace_id in self._ended_sessions:
            self._render_conversation(span)
            self._attach_audio_recording(span)
            self._export_span(span)
            self._cleanup_trace(trace_id)
        else:
            self.deferred_root_spans[trace_id] = span

    # -- deferred root release ------------------------------------------------

    def _release_root_span(self, trace_id: str) -> None:
        span = self.deferred_root_spans.pop(trace_id, None)
        if span is None:
            return
        self._render_conversation(span)
        self._attach_audio_recording(span)
        self._export_span(span)
        self._cleanup_trace(trace_id)

    def _render_conversation(self, span: ReadableSpan) -> bool:
        """Render the accumulated conversation onto a root span.

        Input = the opening message (the greeting, for an agent that speaks
        first); output = everything after it. Returns True if anything was
        rendered.
        """
        messages = self._conversation_by_trace.get(span.context.trace_id, [])
        if not messages:
            return False
        self._set_messages(span, prompt=messages[:1])
        if len(messages) > 1:
            self._set_messages(span, completion=messages[1:])
        return True

    def _cleanup_trace(self, trace_id: str) -> None:
        self._conversation_by_trace.pop(int(trace_id, 16), None)
        self._ended_sessions.discard(trace_id)

    def shutdown(self) -> None:
        self._flush_deferred_root_spans()
        self.downstream.shutdown()

    def force_flush(self, timeout_millis: int = 30000) -> bool:
        self._flush_deferred_root_spans()
        return self.downstream.force_flush(timeout_millis)

    def _flush_deferred_root_spans(self) -> None:
        """Export any root spans still held (a session that ended abnormally)."""
        for trace_id, span in list(self.deferred_root_spans.items()):
            if not self._render_conversation(span):
                self._set_messages(
                    span,
                    prompt=[{"role": "system", "content": "Conversation not captured"}],
                    completion=[
                        {"role": "assistant", "content": "No conversation turns recorded."}
                    ],
                )
            self._attach_audio_recording(span)
            self._export_span(span)
            del self.deferred_root_spans[trace_id]

    # -- audio attachment -----------------------------------------------------

    def _attach_audio_recording(self, span: ReadableSpan) -> None:
        """Attach the call recording (if any) via `langsmith.attachments`.

        Uses the OTel attachment path documented at
        https://docs.langchain.com/langsmith/trace-with-opentelemetry — the value
        is a JSON list of {name, content (base64), mime_type} dicts.
        """
        if self.audio_path_provider is None:
            return
        audio_path = self.audio_path_provider()
        if audio_path is None:
            return
        try:
            if not audio_path.exists():
                logger.debug(
                    "no audio recording at %s (in console mode, pass `--record`)",
                    audio_path,
                )
                return
            audio_bytes = audio_path.read_bytes()
        except Exception as e:  # pragma: no cover
            logger.debug("failed to read audio recording %s: %s", audio_path, e)
            return

        attachment = {
            "name": audio_path.name,
            "content": base64.b64encode(audio_bytes).decode("ascii"),
            "mime_type": "audio/ogg",
        }
        span._attributes["langsmith.attachments"] = json.dumps([attachment])
        logger.debug("attached %d bytes of audio to root span", len(audio_bytes))

    # -- realtime-model metrics ----------------------------------------------

    def _cache_realtime_metrics(self, span: ReadableSpan) -> None:
        """Stash a `realtime_metrics` span's data for its parent `agent_turn`.

        Caches the `lk.*` attributes and any `gen_ai.usage.*` token counts, keyed
        by the parent span_id; the parent agent_turn drains them. Children end
        before parents, so the cache is populated before the turn drains it.
        """
        parent_id = span.parent.span_id if span.parent else None
        if parent_id is None:
            return
        cached = {
            k: v
            for k, v in span.attributes.items()
            if k.startswith("lk.") or k.startswith("gen_ai.usage")
        }
        # Lift token counts out of the metrics blob into gen_ai.usage.* for cost
        # tracking, not just opaque metadata.
        metrics = self._try_parse_json_object(
            span.attributes.get("lk.realtime_model_metrics")
        )
        if isinstance(metrics, dict):
            for src, dst in (
                ("input_tokens", "gen_ai.usage.input_tokens"),
                ("output_tokens", "gen_ai.usage.output_tokens"),
                ("total_tokens", "gen_ai.usage.total_tokens"),
            ):
                v = metrics.get(src)
                if isinstance(v, (int, float)):
                    cached[dst] = v
        if cached:
            self._realtime_metrics_by_parent[parent_id] = cached

    def _drain_realtime_metrics(self, span: ReadableSpan) -> None:
        """Copy cached realtime metrics onto this (agent_turn) span.

        Sets each attribute only if absent, so the blanket `lk.*` pass-through
        then surfaces them as `langsmith.metadata.lk_*`, and the `gen_ai.usage.*`
        counts drive token/cost tracking.
        """
        cached = self._realtime_metrics_by_parent.pop(span.context.span_id, None)
        if not cached:
            return
        for k, v in cached.items():
            if k not in span._attributes:
                span._attributes[k] = v

    # -- attribute helpers ----------------------------------------------------

    def _set_messages(
        self,
        span: ReadableSpan,
        *,
        prompt: Optional[list[dict]] = None,
        completion: Optional[list[dict]] = None,
    ) -> None:
        """Write messages in both the indexed and singular `gen_ai.*` forms.

        LangSmith maps both the indexed `gen_ai.prompt.{n}.role/content`
        attributes and the singular `gen_ai.prompt` JSON; only the singular
        form can carry the structured fields (`tool_calls`, `tool_call_id`).
        """
        if prompt is not None:
            for i, msg in enumerate(prompt):
                span._attributes[f"gen_ai.prompt.{i}.role"] = msg.get("role", "user")
                span._attributes[f"gen_ai.prompt.{i}.content"] = str(msg.get("content", ""))
            span._attributes["gen_ai.prompt"] = json.dumps(prompt)
        if completion is not None:
            for i, msg in enumerate(completion):
                span._attributes[f"gen_ai.completion.{i}.role"] = msg.get(
                    "role", "assistant"
                )
                span._attributes[f"gen_ai.completion.{i}.content"] = str(
                    msg.get("content", "")
                )
            span._attributes["gen_ai.completion"] = json.dumps(completion)

    # -- blanket lk.* pass-through (the only place lk.* metadata is surfaced) --

    def _passthrough_lk_attrs(self, span: ReadableSpan) -> None:
        """Forward every `lk.*` attribute to `langsmith.metadata.lk_<name>`.

        Runs right before export, after the per-span branches. Scalars and
        sequences of scalars are forwarded directly; JSON-object blobs (e.g.
        `lk.tts_metrics`) are flattened so each metric is its own sidebar field.
        """
        for key in list(span.attributes.keys()):
            if not key.startswith("lk."):
                continue
            v = span.attributes[key]
            if v is None or isinstance(v, dict):
                continue
            ms_key = f"langsmith.metadata.{key.replace('.', '_')}"
            parsed = self._try_parse_json_object(v)
            if parsed is not None:
                self._flatten_into_metadata(span, ms_key, parsed)
                continue
            if ms_key in span._attributes:  # don't clobber what a branch set
                continue
            span._attributes[ms_key] = v

    @staticmethod
    def _try_parse_json_object(value: Any) -> Optional[dict]:
        """Return `value` parsed as a dict if it's a JSON-object string, else None."""
        if not isinstance(value, str):
            return None
        s = value.strip()
        if not (s.startswith("{") and s.endswith("}")):
            return None
        try:
            obj = json.loads(s)
        except (json.JSONDecodeError, ValueError):
            return None
        return obj if isinstance(obj, dict) else None

    def _flatten_into_metadata(
        self, span: ReadableSpan, prefix: str, obj: dict, _depth: int = 0
    ) -> None:
        """Flatten a dict's scalar leaves to `langsmith.metadata.<prefix>_<key>`.

        Recurses into nested dicts (joining keys with `_`); forwards scalar leaves
        and sequences of scalars; skips lists of objects (OTel can't store them).
        Depth-bounded as a guard against pathologically nested blobs.
        """
        if _depth > 4:
            return
        for k, v in obj.items():
            name = f"{prefix}_{k}"
            if isinstance(v, dict):
                self._flatten_into_metadata(span, name, v, _depth + 1)
            elif isinstance(v, (str, int, float, bool)):
                if name not in span._attributes:
                    span._attributes[name] = v
            elif (
                isinstance(v, (list, tuple))
                and v
                and all(isinstance(item, (str, int, float, bool)) for item in v)
            ):
                if name not in span._attributes:
                    span._attributes[name] = list(v)

    def _export_span(self, span: ReadableSpan) -> None:
        self._passthrough_lk_attrs(span)
        self.downstream.on_end(span)

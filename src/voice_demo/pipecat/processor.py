"""OTel → LangSmith bridge for Pipecat.

Pipecat emits OTel spans named `conversation`, `turn`, `stt`, `llm`, and `tts`,
but with attributes (`transcript`, `input`/`output`, `text`, `turn.number`, …)
that LangSmith's OTLP ingester doesn't recognize. This `SpanProcessor` rewrites
each span type into the `gen_ai.*` / `langsmith.*` namespaces LangSmith keys
off, renders the whole conversation onto the root span, and attaches the
recorded audio there. It wraps its downstream exporter (mirroring the LiveKit
processor in `..livekit.processor`) so attributes are always rewritten before
the span is queued for export.

Adapted from the official LangChain × Pipecat tracing demo
(github.com/langchain-ai/voice-agents-tracing/blob/main/pipecat/langsmith_processor.py).

The trace shape in LangSmith:

    conversation                      (root; whole transcript + conversation WAV)
    └── turn × N                      (per exchange; carries was_interrupted)
        ├── stt                       (audio → transcript)
        ├── llm                       (chain — the LangGraph brain; inference is
        │   │                          in the nested model nodes below)
        │   ├── model                 (ChatOpenAI inference; may emit tool calls)
        │   ├── tools: lookup_weather (tool execution)
        │   └── model                 (final answer — spoken)
        └── tts                       (response text → audio)

Note: the kind of the span Pipecat names `llm` is set by `llm_span_kind` at
construction. This demo passes "chain" because `LangGraphLLMService` only
orchestrates the graph — the real `llm`-kind runs are the nested model nodes.
With a stock service doing its own inference, keep the default "llm". See the
comment in `_handle_llm`.

Thread grouping is opt-in, as in the LiveKit processor: pass
`thread_id_provider` (a zero-arg callable returning the conversation's id) and
every span gets `langsmith.metadata.thread_id`; without it, none is set.
"""

from __future__ import annotations

import base64
import json
from pathlib import Path
from typing import Callable, Optional

from loguru import logger
from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.trace import ReadableSpan, SpanProcessor, TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor


class LangSmithSpanProcessor(SpanProcessor):
    """Enriches Pipecat's OTel spans with LangSmith-compatible attributes."""

    def __init__(
        self,
        downstream_processor: Optional[SpanProcessor] = None,
        *,
        llm_span_kind: str = "llm",
        thread_id_provider: Optional[Callable[[], Optional[str]]] = None,
    ) -> None:
        """Create the processor.

        Args:
            downstream_processor: where rewritten spans are forwarded; defaults
                to `BatchSpanProcessor(OTLPSpanExporter())` (reads the OTLP
                endpoint/headers from env).
            llm_span_kind: LangSmith run kind for Pipecat's `llm` span. Keep
                the default `"llm"` when the LLM stage does its own inference
                (stock services such as `OpenAILLMService`). Pass `"chain"`
                when it only orchestrates nested runs that are exported to
                LangSmith themselves (our `LangGraphLLMService`) — see the
                comment in `_handle_llm`.
            thread_id_provider: opt-in conversation id for LangSmith thread
                grouping; called per span, None disables.
        """
        super().__init__()
        if downstream_processor is None:
            downstream_processor = BatchSpanProcessor(OTLPSpanExporter())
        self.downstream = downstream_processor
        self._llm_span_kind = llm_span_kind
        self.thread_id_provider = thread_id_provider
        # The latest llm request's full context per trace — each request
        # carries the whole history, so the last snapshot IS the conversation.
        # Rendered onto the root `conversation` span, which ends last.
        self._conversation_by_trace: dict[str, list] = {}
        self.conversation_recordings: dict[str, str] = {}  # conversation_id -> path
        self.conversation_recorders: dict[str, object] = {}  # conversation_id -> recorder

    # -- recorder registration ------------------------------------------------

    def register_recording(self, conversation_id, recording_path, audio_recorder=None):
        """Register the whole-conversation recording for the root span.

        `audio_recorder`, if given, is any object with a `save_recording()`
        method — it's invoked when the conversation span ends so the file is on
        disk before we read and attach it.
        """
        self.conversation_recordings[conversation_id] = recording_path
        if audio_recorder:
            self.conversation_recorders[conversation_id] = audio_recorder

    # -- span lifecycle -------------------------------------------------------

    def on_start(self, span: ReadableSpan, parent_context=None) -> None:
        self.downstream.on_start(span, parent_context)

    def on_end(self, span: ReadableSpan) -> None:
        """Rewrite a Pipecat span into LangSmith's expected attribute shape."""
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

        if span.name == "stt":
            self._handle_stt(span)
            self._exclude_from_message_view(span)
        elif span.name == "llm":
            self._handle_llm(span, trace_id)
        elif span.name == "tts":
            self._handle_tts(span)
            self._exclude_from_message_view(span)
        elif span.name == "turn":
            self._handle_turn(span)
        elif span.name == "conversation":
            self._handle_conversation(span, trace_id)
        # Nested LangGraph/LangChain runs (the model node, its ChatOpenAI calls,
        # tools/lookup_weather) need no rewriting — they arrive in LangSmith's
        # native shape with their own run type. With the framework span now a
        # `chain`, they form the clean llm-within-chain multi-call shape the
        # Messages view reconstructs from, so the processor needs no knowledge of
        # the graph's nodes. (A graph node that is traced but never spoken can opt
        # out of the Messages view by setting `ls_message_view_exclude` metadata
        # on itself in graph.py — keeping that concern out of this processor.)

        self.downstream.on_end(span)

    def _exclude_from_message_view(self, span: ReadableSpan) -> None:
        """Drop a framework `stt`/`tts` span from the conversation Messages view.

        That view reconstructs the chat from `llm`/`tool` runs (see
        smith-go/runs/v2/messages). The framework span (named `llm`) is now a
        `chain`, so the real inference runs nested under it — the model node and
        its ChatOpenAI calls, plus the lookup_weather tool — form the clean
        llm-within-chain multi-call shape LangSmith reconstructs reliably, and
        they're left visible. `stt`/`tts` are tagged `llm`-kind for the trace
        tree, but as conversation turns they'd inject fake "assistant" messages
        (the raw transcript; "Generated audio for: …"), so we drop them here.

        A nested graph node that is traced but never spoken can drop *itself*
        from the Messages view by setting `ls_message_view_exclude` metadata in
        graph.py, keeping graph knowledge out of this processor.

        LangSmith honors the `ls_message_view_exclude` metadata key to drop a run
        from the Messages view *only* — these runs stay fully visible in the
        trace tree.
        """
        span._attributes["langsmith.metadata.ls_message_view_exclude"] = True

    # -- per-span-type handlers ----------------------------------------------

    def _handle_stt(self, span: ReadableSpan) -> None:
        """STT span: audio input → transcribed text. Rendered like LiveKit's."""
        transcript = span.attributes.get("transcript", "")
        span._attributes["langsmith.span.kind"] = "llm"
        self._set_prompt_attributes(
            span, [{"role": "user", "content": f'Audio for: "{transcript}"'}]
        )
        if transcript:
            self._set_completion_attributes(
                span, [{"role": "assistant", "content": str(transcript)}]
            )

    def _handle_llm(self, span: ReadableSpan, trace_id: str) -> None:
        """Framework `llm` span: the LLM stage of the pipeline."""
        input_data = span.attributes.get("input", "")
        output_data = span.attributes.get("output", "")

        # =====================================================================
        # WHY WOULD A SPAN NAMED `llm` BE CLASSIFIED AS A `chain`?
        #
        # With our LangGraphLLMService, this span does NO inference itself —
        # it only orchestrates the graph run. The real inference is the nested
        # `model` runs (ChatOpenAI), which arrive in LangSmith's native shape
        # with their own `llm` kind. Tagging the wrapper `llm` too would give
        # LangSmith llm-inside-llm and double-count the conversation in the
        # Messages view (which reconstructs the chat from llm/tool runs) — so
        # the wrapper must be a `chain`. That's what this demo configures.
        #
        # With Pipecat's stock LLM services (e.g. plain OpenAILLMService), the
        # opposite is true: this span IS the inference, nothing nests under
        # it, and the default `llm` is the correct kind. IF YOU COPY THIS
        # PROCESSOR INTO A STOCK-SERVICE PIPELINE, KEEP THE DEFAULT.
        #
        # This can't be auto-detected at span-end time: LangChain's runs are
        # exported to OTel from LangSmith's background queue — with backdated
        # timestamps, *after* this span has already ended — so the nested
        # runs aren't visible yet when this handler fires. The code that
        # builds the pipeline knows which LLM service it wired, so it states
        # the choice explicitly via `llm_span_kind` at construction.
        # =====================================================================
        span._attributes["langsmith.span.kind"] = self._llm_span_kind

        try:
            raw_messages = json.loads(input_data)
        except (json.JSONDecodeError, TypeError):
            raw_messages = []
        messages = [
            self._to_langchain_message(m)
            for m in (raw_messages if isinstance(raw_messages, list) else [])
            if isinstance(m, dict)
        ]

        # Singular-only JSON in LangChain message format ({"messages": [...]}).
        # The indexed gen_ai.prompt.{n}.role/content form takes precedence at
        # ingest and can only carry role/content — it would drop tool_calls.
        if messages:
            span._attributes["gen_ai.prompt"] = json.dumps({"messages": messages})
        if output_data:
            span._attributes["gen_ai.completion"] = json.dumps(
                {"messages": [{"role": "assistant", "content": output_data}]}
            )

        # Each request's input carries the full history, so the latest snapshot
        # IS the conversation — kept per trace for the root span to render.
        transcript = [m for m in messages if m.get("role") != "system"]
        if output_data:
            transcript.append({"role": "assistant", "content": output_data})
        if transcript:
            self._conversation_by_trace[trace_id] = transcript

    def _handle_tts(self, span: ReadableSpan) -> None:
        """TTS span: text → audio. Rendered like LiveKit's: the voice is
        metadata, not conversation content."""
        text = span.attributes.get("text", "")
        span._attributes["langsmith.span.kind"] = "llm"
        voice_id = span.attributes.get("voice_id")
        if voice_id:
            span._attributes["langsmith.metadata.voice_id"] = str(voice_id)
        self._set_prompt_attributes(span, [{"role": "user", "content": str(text)}])
        self._set_completion_attributes(
            span, [{"role": "assistant", "content": f'Generated audio for: "{text}"'}]
        )

    def _handle_turn(self, span: ReadableSpan) -> None:
        """Turn span: a framework wrapper around one exchange — a chain with no
        fabricated I/O (the llm span beneath it carries the content)."""
        span._attributes["langsmith.span.kind"] = "chain"
        turn_number = span.attributes.get("turn.number")
        if turn_number is not None:
            span._attributes["langsmith.metadata.turn_number"] = turn_number
        was_interrupted = span.attributes.get("turn.was_interrupted")
        if was_interrupted is not None:
            span._attributes["langsmith.metadata.turn_was_interrupted"] = was_interrupted

    def _handle_conversation(self, span: ReadableSpan, trace_id: str) -> None:
        """Conversation span: the whole session; the LangSmith root.

        Input = the opening message; output = everything after it (same split
        as the LiveKit root). Pipecat's conversation span genuinely ends last,
        so no deferral is needed.
        """
        conversation_id = span.attributes.get("conversation.id", "") or span.attributes.get(
            "conversation_id", ""
        )
        span._attributes["langsmith.span.kind"] = "chain"
        span._attributes["langsmith.root_span"] = True
        span._attributes["langsmith.metadata.ls_modality"] = "audio"

        # Singular-only, like _handle_llm — the transcript can contain
        # tool-call messages the indexed form can't represent.
        messages = self._conversation_by_trace.get(trace_id, [])
        if messages:
            span._attributes["gen_ai.prompt"] = json.dumps({"messages": messages[:1]})
            if len(messages) > 1:
                span._attributes["gen_ai.completion"] = json.dumps(
                    {"messages": messages[1:]}
                )

        self._attach_conversation_audio(span, conversation_id)
        self._cleanup_conversation(trace_id, conversation_id)

    # -- audio attachment -----------------------------------------------------

    def _attach_conversation_audio(self, span: ReadableSpan, conversation_id) -> None:
        recorder = self.conversation_recorders.get(conversation_id)
        if recorder is not None:
            try:
                recorder.save_recording()
            except Exception as e:  # pragma: no cover
                logger.warning(f"Failed to save recording for {conversation_id}: {e}")

        path_str = self.conversation_recordings.get(conversation_id)
        if path_str is None and len(self.conversation_recordings) == 1:
            path_str = next(iter(self.conversation_recordings.values()))
        if not path_str:
            return

        encoded = self._load_audio_file(path_str)
        if encoded:
            span._attributes["langsmith.attachments"] = json.dumps(
                [
                    {
                        "name": Path(path_str).name,
                        "content": encoded,
                        "mime_type": "audio/wav",
                    }
                ]
            )
            logger.debug(f"Attached recording {Path(path_str).name} to conversation span")

    def _cleanup_conversation(self, trace_id: str, conversation_id) -> None:
        self._conversation_by_trace.pop(trace_id, None)
        self.conversation_recordings.pop(conversation_id, None)
        self.conversation_recorders.pop(conversation_id, None)

    # -- attribute helpers ----------------------------------------------------

    @staticmethod
    def _to_langchain_message(msg: dict) -> dict:
        """Convert one OpenAI-format context message to LangChain flat format.

        Assistant tool calls arrive as `{"id", "type": "function", "function":
        {"name", "arguments": "<json string>"}}`; LangSmith renders tool-call
        blocks from the flat `{"type": "tool_call", "id", "name", "args"
        (object)}` shape, with tool-result messages linking back via top-level
        `tool_call_id`.
        """
        content = msg.get("content")
        if not isinstance(content, str):
            content = "" if content is None else json.dumps(content)
        out: dict = {"role": str(msg.get("role", "")), "content": content}
        tool_calls = []
        for tc in msg.get("tool_calls") or []:
            if not isinstance(tc, dict):
                continue
            fn = tc.get("function") or {}
            args = fn.get("arguments")
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except json.JSONDecodeError:
                    pass
            tool_calls.append(
                {
                    "type": "tool_call",
                    "id": tc.get("id"),
                    "name": fn.get("name"),
                    "args": args if isinstance(args, dict) else {},
                }
            )
        if tool_calls:
            out["tool_calls"] = tool_calls
        if msg.get("tool_call_id"):
            out["tool_call_id"] = str(msg["tool_call_id"])
        if msg.get("role") == "tool" and msg.get("name"):
            out["name"] = str(msg["name"])
        return out

    def _set_prompt_attributes(self, span: ReadableSpan, messages: list) -> None:
        for i, msg in enumerate(messages):
            span._attributes[f"gen_ai.prompt.{i}.role"] = msg.get("role", "")
            span._attributes[f"gen_ai.prompt.{i}.content"] = msg.get("content", "")

    def _set_completion_attributes(self, span: ReadableSpan, messages: list) -> None:
        for i, msg in enumerate(messages):
            span._attributes[f"gen_ai.completion.{i}.role"] = msg.get("role", "")
            span._attributes[f"gen_ai.completion.{i}.content"] = msg.get("content", "")

    def _load_audio_file(self, file_path) -> str | None:
        try:
            path = Path(file_path)
            if path.exists():
                data = path.read_bytes()
                if data:
                    return base64.b64encode(data).decode("utf-8")
        except Exception as e:  # pragma: no cover
            logger.warning(f"Failed to load audio file {file_path}: {e}")
        return None

    def shutdown(self) -> None:
        self.downstream.shutdown()

    def force_flush(self, timeout_millis: int = 30000) -> bool:
        return self.downstream.force_flush(timeout_millis)


def setup_langsmith_tracing(
    *,
    llm_span_kind: str = "llm",
    thread_id_provider: Optional[Callable[[], Optional[str]]] = None,
) -> LangSmithSpanProcessor:
    """Configure OTel export to LangSmith and register the span processor.

    Pipecat's `setup_tracing` only creates and installs the TracerProvider
    (`exporter=None` adds no export pipeline); the processor registered here
    wraps the OTLP exporter itself, so spans are always rewritten before they
    are queued for export. The exporter reads `OTEL_EXPORTER_OTLP_ENDPOINT` /
    `OTEL_EXPORTER_OTLP_HEADERS` from the environment (wired by
    `voice_demo.tracing.configure`). Returns the processor instance so the
    agent can register its audio recorders on it.

    `llm_span_kind` and `thread_id_provider` are forwarded to
    `LangSmithSpanProcessor` — see its docstring.
    """
    from pipecat.utils.tracing.setup import setup_tracing

    setup_tracing(service_name="voice-demo-pipecat", exporter=None, console_export=False)

    span_processor = LangSmithSpanProcessor(
        llm_span_kind=llm_span_kind, thread_id_provider=thread_id_provider
    )
    provider = trace.get_tracer_provider()
    if isinstance(provider, TracerProvider):
        provider.add_span_processor(span_processor)
    return span_processor

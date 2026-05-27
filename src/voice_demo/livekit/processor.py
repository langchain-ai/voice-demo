"""
OTel -> LangSmith bridge for LiveKit Agents.

LangSmith's OTel ingester reads the standard `gen_ai.*` and `langsmith.*`
attribute namespaces. LiveKit emits its data under a `lk.*` vendor prefix the
ingester doesn't recognize -- so without translation, transcripts, confidence
scores, TTS metrics, EOU probabilities, latency numbers, etc. all evaporate at
ingest. This module is an in-process OTel `SpanProcessor` that rewrites `lk.*`
into the keys LangSmith understands, before each span is exported downstream.

Translates in three layers:

  1. Per-span-type unpacking -- rewrite `lk.*` keys into `gen_ai.*` /
     `langsmith.metadata.*` for each LiveKit span (user_turn, tts_node,
     llm_node, agent_turn, eou_detection, etc.). Also overrides
     `langsmith.span.kind` where the default `gen_ai.request.model` autoclassify
     would mislabel (e.g. STT tagged `chain`, not `llm`).
  2. Blanket `lk.*` pass-through -- catch-all so any remaining vendor scalar
     attribute lands as `langsmith.metadata.lk_<flattened_name>`. Ensures no
     data is silently dropped and future-proofs against new LiveKit attributes.
  3. Per-turn aggregation onto `agent_turn` -- pipeline-stage latencies
     (transcription_delay, eou.endpointing_delay, llm ttft, tts ttfb,
     e2e_latency) roll up as `langsmith.metadata.turn_*` so each turn has a
     one-look breakdown mirroring LiveKit's own `ChatMessage.metrics` shape.

Also handles LangChain runs that ride through the same OTel provider
(`LANGSMITH_TRACING_MODE=otel`): reads the singular `gen_ai.prompt` /
`gen_ai.completion` JSON strings, unwraps the serialized LLMResult to the
inner text, and surfaces `generation_info` fields (finish_reason, model_name,
system_fingerprint) as metadata.

Uses `ctx.job.id` as the LangSmith `thread_id` (via the ContextVar in
`livekit_demo._thread_id`) so all turns of one session share a thread.
`ctx.room.name` is unreliable in console mode (always `"console"`), so the
job id is the more stable per-conversation key.
"""
import base64
import json
import os
import logging
from copy import deepcopy
from typing import Optional
from opentelemetry.sdk.trace import SpanProcessor, ReadableSpan
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter

# Pull the active job/room id out of a ContextVar so it can be used as a
# stable, per-conversation thread_id. Falls back gracefully if the import
# fails (e.g. running this processor outside this project).
try:
    from ._thread_id import (
        active_thread_id as _active_thread_id,
        last_set_thread_id as _last_set_thread_id,
        get_audio_file_path as _get_audio_file_path,
    )
except Exception:  # pragma: no cover
    _active_thread_id = None
    _last_set_thread_id = None
    _get_audio_file_path = None

# Optional verbose logging for local debugging
DEBUG = os.getenv("LANGSMITH_PROCESSOR_DEBUG", "false").lower() in ("true", "1", "yes")
logger = logging.getLogger("langsmith_processor")
if DEBUG and not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter('%(asctime)s [LANGSMITH] %(levelname)s %(message)s'))
    logger.addHandler(handler)
    logger.setLevel(logging.DEBUG)


def _dbg(*args, **kwargs):
    """Debug print gated on the LANGSMITH_PROCESSOR_DEBUG env var.

    Existing call sites pass `file=sys.stderr, flush=True`; setdefault keeps
    those kwargs honored, but they're not required.
    """
    if not DEBUG:
        return
    import sys
    kwargs.setdefault("file", sys.stderr)
    kwargs.setdefault("flush", True)
    print(*args, **kwargs)


class LangSmithSpanProcessor(SpanProcessor):
    """
    Custom OpenTelemetry span processor that enriches LiveKit Agents spans with LangSmith-compatible attributes.
    This enables proper conversation tracking and message visualization in LangSmith's UI.
    """

    def __init__(self, downstream_processor: Optional[SpanProcessor] = None):
        super().__init__()
        if downstream_processor is None:
            downstream_processor = BatchSpanProcessor(OTLPSpanExporter())
        self.downstream = downstream_processor
        # Track conversation messages across spans for proper LangSmith grouping
        self.conversation_messages = {}  # trace_id -> list of messages
        self.trace_to_conversation_id = {}  # trace_id -> conversation_id
        # Hold root/job spans until conversation data is ready
        self.deferred_job_spans = {}  # trace_id -> ReadableSpan
        # LiveKit writes lk.tts_metrics + lk.input_text onto the inner tts_request
        # span but not the outer tts_node wrapper. Cache those values when
        # tts_request ends so we can mirror them onto tts_node when it ends next.
        # Keyed by OTel trace_id (int). Cleared on tts_node end.
        self._tts_cache_by_trace_id = {}  # int -> {"metrics": dict, "input_text": str, "model_name": str|None}
        # Per-turn latency aggregation cache. Each pipeline-stage span pushes
        # its latency in as it ends; we drain onto the agent_turn span so the
        # whole turn breakdown is visible in one place (mirrors LiveKit's own
        # ChatMessage.metrics shape). Keyed by trace_id, popped on agent_turn end.
        self._turn_latencies_by_trace_id: dict[int, dict] = {}
        # Cache the per-turn LLM response text from `llm_node` so agent_turn
        # can render exactly what LiveKit considers the turn's output —
        # bypassing the trace-wide accumulator that mixes outputs from every
        # inner LangChain/ChatOpenAI/Runnable child. Popped on agent_turn end.
        self._llm_response_by_trace_id: dict[int, str] = {}

    def on_start(self, span: ReadableSpan, parent_context=None) -> None:
        if self.downstream:
            self.downstream.on_start(span, parent_context)

    def on_end(self, span: ReadableSpan) -> None:
        """
        Enriches spans with LangSmith-compatible attributes before they're exported.
        Maps LiveKit Agents span types to LangSmith's expected format.
        """
        # Always log that we're processing a span (even without DEBUG mode)
        # Use print to stderr to ensure it's visible
        import sys
        _dbg(f"[LANGSMITH-PROCESSOR] Processing span: {span.name}", file=sys.stderr, flush=True)
        
        # Track each conversation as a thread in LangSmith. Prefer a
        # per-conversation thread_id (the LiveKit job id) over the per-trace
        # OTel trace_id, so multi-turn sessions group under one LangSmith
        # thread. Two-step fallback: ContextVar (cleanest, per-task) ->
        # module global (catches callbacks that escape the task tree) ->
        # trace_id (last resort).
        trace_id = format(span.context.trace_id, '032x')
        thread_id_override = _active_thread_id.get() if _active_thread_id is not None else None
        if thread_id_override is None and _last_set_thread_id is not None:
            thread_id_override = _last_set_thread_id()
        span._attributes["langsmith.metadata.thread_id"] = thread_id_override or trace_id

        # Link all spans to their conversation for proper grouping in LangSmith
        if trace_id in self.trace_to_conversation_id:
            conversation_id = self.trace_to_conversation_id[trace_id]
            span._attributes["conversation.id"] = conversation_id
            span._attributes["langsmith.parent_span_id"] = "conversation"

        span_name = span.name.lower()

        # Collect per-stage latencies into a per-turn cache. Drained when
        # agent_turn ends — see _collect_turn_latencies for details.
        self._collect_turn_latencies(span, span_name)

        # STT span: audio input -> transcribed text.
        # LiveKit wraps STT in a span literally named `user_turn` and writes the
        # transcript/confidence/timing onto it via `lk.*` keys; the LangSmith
        # ingester doesn't recognize that namespace, so we rename them here.
        if (
            "stt" in span_name
            or "speech_to_text" in span_name
            or "transcription" in span_name
            or "user_turn" in span_name
        ):
            # STT is a pipeline step, not a model invocation — tag as chain so
            # it doesn't pollute "LLM" filters/dashboards. LangSmith renders
            # chain-kind Input/Output panels from the *singular* gen_ai.prompt
            # / gen_ai.completion JSON attrs (not the per-N indexed form, which
            # is read only for llm-kind runs), so we set both.
            span._attributes["langsmith.span.kind"] = "chain"
            transcript = (
                span.attributes.get("lk.user_transcript")
                or span.attributes.get("transcript")
                or span.attributes.get("text")
                or span.attributes.get("output", "")
            )
            prompt_msgs = [{"role": "user", "content": "audio_segment"}]
            self._set_prompt_attributes(span, prompt_msgs)
            span._attributes["gen_ai.prompt"] = json.dumps(prompt_msgs)
            if transcript:
                completion_msgs = [{"role": "assistant", "content": str(transcript)}]
                self._set_completion_attributes(span, completion_msgs)
                span._attributes["gen_ai.completion"] = json.dumps(completion_msgs)

            # Surface LiveKit STT metadata so it shows up in the run sidebar.
            # OTel attribute values must be scalars/sequences of scalars — skip dicts/lists.
            for src, dst in (
                ("lk.transcript_confidence", "transcript_confidence"),
                ("lk.transcription_delay", "transcription_delay"),
                ("lk.end_of_turn_delay", "end_of_turn_delay"),
                ("lk.provider_request_ids", "provider_request_ids"),
            ):
                v = span.attributes.get(src)
                if v is None or isinstance(v, dict):
                    continue
                # Some providers (e.g. OpenAI gpt-4o-mini-transcribe) don't
                # report a real confidence — LiveKit defaults to 0.0. Skip the
                # "fake zero" so the UI doesn't render misleading data.
                if src == "lk.transcript_confidence" and not (isinstance(v, (int, float)) and v > 0):
                    continue
                span._attributes[f"langsmith.metadata.{dst}"] = v

        # LLM span: conversation messages -> AI response
        elif "llm" in span_name or "chat" in span_name or "completion" in span_name or "openai" in span_name:
            span._attributes["langsmith.span.kind"] = "llm"
            messages = self._extract_llm_messages(span)
            if not messages:
                messages = self._fallback_messages(span, span_name)
            self._set_prompt_attributes(span, messages)

            output_data = self._extract_llm_output(span)
            if output_data:
                # LangChain's OTel emission packs the whole LLMResult into one
                # JSON string. Unwrap to the inner text so the run output shows
                # the assistant message, not a `{"generations": [...]}` blob.
                output_data = self._unwrap_langchain_llmresult(output_data, span)
                completion = [{"role": "assistant", "content": str(output_data)}]
                self._set_completion_attributes(span, completion)
                self._track_messages(self.conversation_messages, trace_id, messages, str(output_data))

            # Cache LiveKit's canonical per-turn LLM response so agent_turn
            # can render it directly instead of pulling from the accumulator
            # (which also collects from inner LangGraph / ChatOpenAI spans).
            if span_name == "llm_node":
                lk_text = span.attributes.get("lk.response.text")
                if lk_text:
                    self._llm_response_by_trace_id[span.context.trace_id] = str(lk_text)

        # TTS span: text -> audio
        elif "tts" in span_name or "text_to_speech" in span_name or "synthesis" in span_name:
            span._attributes["langsmith.span.kind"] = "llm"

            # Debug TTS spans - always print attributes to see what LiveKit uses
            import sys
            _dbg(f"\n[LANGSMITH-PROCESSOR] 🔊 TTS SPAN: {span.name}", file=sys.stderr, flush=True)
            _dbg(f"  📋 All attributes for {span.name} ({len(span.attributes)} total):", file=sys.stderr, flush=True)
            for key, value in sorted(span.attributes.items()):
                value_str = str(value)
                if len(value_str) > 500:
                    value_str = value_str[:500] + "... (truncated)"
                _dbg(f"    • {key} = {value_str}", file=sys.stderr, flush=True)

            # Cache bridges data from the inner `tts_request` span (where
            # LiveKit actually writes lk.input_text + lk.tts_metrics) up to the
            # outer `tts_node` wrapper that ends after it. Keyed by trace_id.
            trace_int = span.context.trace_id
            cached = self._tts_cache_by_trace_id.get(trace_int, {})
            cached_metrics = cached.get("metrics") or {}
            cached_text = cached.get("input_text") or ""
            cached_model = cached.get("model_name") or None

            # Try LiveKit-specific attributes first; fall back to whatever the
            # child span cached.
            text = (
                span.attributes.get("lk.input_text") or
                span.attributes.get("lk.request.text") or
                span.attributes.get("lk.text") or
                span.attributes.get("text") or
                span.attributes.get("input") or
                span.attributes.get("prompt") or
                cached_text or
                ""
            )

            # Model: LiveKit sets `gen_ai.request.model` directly on tts_node.
            # For tts_request, the model lives inside lk.tts_metrics.metadata.
            voice_id = "unknown"
            gen_ai_model = span.attributes.get("gen_ai.request.model")
            if gen_ai_model:
                voice_id = str(gen_ai_model)

            # Track which scalar metric keys we wrote from THIS span's own
            # lk.tts_metrics so we know which gaps to fill from the cache.
            local_metric_keys: set[str] = set()
            scalar_keys = (
                "audio_duration",
                "ttfb",
                "characters_count",
                "duration",
                "input_tokens",
                "output_tokens",
                "streamed",
                "cancelled",
                "request_id",
                "segment_id",
                "speech_id",
            )
            metrics_data: dict = {}
            tts_metrics = span.attributes.get("lk.tts_metrics")
            if tts_metrics:
                try:
                    if isinstance(tts_metrics, str):
                        metrics_data = json.loads(tts_metrics)
                    else:
                        metrics_data = tts_metrics
                    if isinstance(metrics_data, dict):
                        metadata = metrics_data.get("metadata", {})
                        model_name = metadata.get("model_name") or metrics_data.get("model_name")
                        if model_name and voice_id == "unknown":
                            voice_id = str(model_name)
                        # Unpack scalar metric fields. Explicit allowlist — we
                        # only forward known-safe scalar keys, never nested
                        # dicts/lists (OTel rejects them as attribute values).
                        for k in scalar_keys:
                            v = metrics_data.get(k)
                            if v is not None and not isinstance(v, (dict, list)):
                                span._attributes[f"langsmith.metadata.tts_{k}"] = v
                                local_metric_keys.add(k)
                except (json.JSONDecodeError, TypeError, KeyError):
                    metrics_data = {}

            # Fill any metric keys this span didn't carry locally from the
            # cached values pushed by the inner `tts_request`.
            for k, v in cached_metrics.items():
                if k in local_metric_keys:
                    continue
                if v is None or isinstance(v, (dict, list)):
                    continue
                span._attributes[f"langsmith.metadata.tts_{k}"] = v

            # `lk.response.ttfb` is also set directly on the TTS span (not just
            # inside the metrics blob) — forward it for parity.
            ttfb_attr = span.attributes.get("lk.response.ttfb")
            if ttfb_attr is not None and not isinstance(ttfb_attr, (dict, list)):
                span._attributes["langsmith.metadata.tts_response_ttfb"] = ttfb_attr

            # Final voice fallbacks: any explicit lk.voice, then the cached
            # model from the child span, then "unknown".
            if voice_id == "unknown":
                voice_id = (
                    span.attributes.get("lk.voice") or
                    span.attributes.get("voice") or
                    span.attributes.get("voice_id") or
                    cached_model or
                    "unknown"
                )

            _dbg(f"  ✅ Extracted text: length={len(str(text))}, voice={voice_id}", file=sys.stderr, flush=True)

            self._set_prompt_attributes(span, [
                {"role": "system", "content": f"Convert to speech with voice: {voice_id}"},
                {"role": "user", "content": str(text) if text else "text_to_speech"}
            ])
            self._set_completion_attributes(span, [{"role": "assistant", "content": f"Generated audio for: {text}"}])

            # tts_request is the inner span that ends before its parent
            # tts_node. Stash its data so tts_node can render the same content.
            # tts_node is the consumer — clear the cache once it ends.
            if span_name == "tts_request":
                self._tts_cache_by_trace_id[trace_int] = {
                    "metrics": {
                        k: metrics_data.get(k)
                        for k in scalar_keys
                        if metrics_data.get(k) is not None
                        and not isinstance(metrics_data.get(k), (dict, list))
                    },
                    "input_text": str(text) if text else "",
                    "model_name": voice_id if voice_id != "unknown" else None,
                }
            elif span_name == "tts_node":
                self._tts_cache_by_trace_id.pop(trace_int, None)

        # Agent/Chain/Job spans: aggregate conversation
        elif (
            "agent" in span_name
            or "session" in span_name
            or "conversation" in span_name
            or "job" in span_name
        ):
            span._attributes["langsmith.span.kind"] = "chain"
            is_job_span = "job" in span_name
            
            # Try to extract conversation ID
            conversation_id = (
                span.attributes.get("conversation.id") or
                span.attributes.get("conversation_id") or
                span.attributes.get("session_id") or
                (span.attributes.get("lk.job_id") if is_job_span else "") or
                ""
            )
            if conversation_id:
                self.trace_to_conversation_id[trace_id] = str(conversation_id)
                span._attributes["conversation.id"] = str(conversation_id)
                span._attributes["langsmith.root_span"] = True
            elif is_job_span:
                # Ensure the root job span is treated as the LangSmith conversation root
                span._attributes["conversation.id"] = trace_id
                span._attributes["langsmith.root_span"] = True

            # Special case: agent_turn renders directly from LiveKit's per-turn
            # attrs. The default chain rollup uses self.conversation_messages,
            # which accumulates outputs from every llm-flavored child span —
            # including the LangGraph's internal ChatOpenAI calls that serialize
            # responses as `{"generations":[...]}` JSON, causing duplicate
            # outputs (clean text + JSON blob). Use what LiveKit itself
            # considers the per-turn input (`lk.user_input`) and output
            # (cached `lk.response.text` from the per-turn `llm_node`).
            if span_name == "agent_turn":
                user_input = span.attributes.get("lk.user_input")
                cached_response = self._llm_response_by_trace_id.pop(
                    span.context.trace_id, None
                )
                if user_input:
                    prompt_msgs = [{"role": "user", "content": str(user_input)}]
                    self._set_prompt_attributes(span, prompt_msgs)
                    span._attributes["gen_ai.prompt"] = json.dumps(prompt_msgs)
                if cached_response:
                    completion_msgs = [{"role": "assistant", "content": cached_response}]
                    self._set_completion_attributes(span, completion_msgs)
                    span._attributes["gen_ai.completion"] = json.dumps(completion_msgs)
                # Fall through to deferred-release / cleanup logic below, but
                # skip the conversation_messages rollup that would re-pollute.
                self._export_span(span)
                return

            # Aggregate messages from conversation
            conv_msgs = self.conversation_messages.get(trace_id, [])
            if conv_msgs:
                system_msg, first_user_msg, remaining_msgs = self._split_conversation_messages(conv_msgs)

                # Add input (first user message only, exclude system message)
                # System message is only shown in LLM call spans, not in job entrypoint
                prompt_msgs = []
                if first_user_msg:
                    prompt_msgs.append(first_user_msg)
                if prompt_msgs:
                    self._set_prompt_attributes(span, prompt_msgs)

                # Add output (remaining conversation)
                if remaining_msgs:
                    self._set_completion_attributes(span, remaining_msgs)

                # LangSmith renders the chain-run "Input"/"Output" panels from
                # the singular `gen_ai.prompt` / `gen_ai.completion` JSON attrs,
                # not the enumerated gen_ai.prompt.N.* form (that path is read
                # only for LLM-kind runs). Set both so the trace root shows
                # the conversation instead of "no messages to display".
                if prompt_msgs:
                    span._attributes["gen_ai.prompt"] = json.dumps(prompt_msgs)
                if remaining_msgs:
                    span._attributes["gen_ai.completion"] = json.dumps(remaining_msgs)

                # Only release the deferred job_entrypoint when the actual
                # session ends — not on each per-turn `agent_turn` end (which
                # also matches "agent" in span_name). Agent_turn fires mid-
                # conversation, before the recorder has been closed, so
                # releasing then would miss audio.
                should_release_deferred = is_job_span or span_name == "agent_session"
                if should_release_deferred:
                    self._release_job_span_if_waiting(trace_id, prompt_msgs, remaining_msgs)
            elif is_job_span:
                # Defer export until conversation data becomes available
                self._defer_job_span(trace_id, span)
                return

            # Cleanup
            should_cleanup_trace = is_job_span or span.parent is None
            if should_cleanup_trace:
                if trace_id in self.conversation_messages:
                    del self.conversation_messages[trace_id]
                if trace_id in self.trace_to_conversation_id:
                    del self.trace_to_conversation_id[trace_id]

        # Default: mark as chain if no specific type detected
        else:
            # Check if it has LLM-like attributes
            if span.attributes.get("input") or span.attributes.get("output"):
                span._attributes["langsmith.span.kind"] = "llm"
                input_val = span.attributes.get("input", "")
                output_val = span.attributes.get("output", "")
                if input_val:
                    self._set_prompt_attributes(span, [{"role": "user", "content": str(input_val)}])
                if output_val:
                    self._set_completion_attributes(span, [{"role": "assistant", "content": str(output_val)}])
            else:
                span._attributes["langsmith.span.kind"] = "chain"

        # Export span downstream (unless it was deferred earlier)
        self._export_span(span)

    def _set_prompt_attributes(self, span: ReadableSpan, messages: list, start_idx: int = 0, log: bool = False):
        """Set gen_ai.prompt.* attributes from a list of messages."""
        import sys
        for i, msg in enumerate(messages):
            idx = start_idx + i
            if isinstance(msg, dict):
                role = msg.get("role", "user")
                content = str(msg.get("content", ""))
                span._attributes[f"gen_ai.prompt.{idx}.role"] = role
                span._attributes[f"gen_ai.prompt.{idx}.content"] = content
                if log:
                    content_preview = content[:100] + "..." if len(content) > 100 else content
                    _dbg(f"    Set gen_ai.prompt.{idx}.role = '{role}', gen_ai.prompt.{idx}.content = '{content_preview}' (length: {len(content)})", file=sys.stderr, flush=True)
            else:
                # Handle string messages
                content = str(msg)
                span._attributes[f"gen_ai.prompt.{idx}.role"] = "user"
                span._attributes[f"gen_ai.prompt.{idx}.content"] = content
                if log:
                    content_preview = content[:100] + "..." if len(content) > 100 else content
                    _dbg(f"    Set gen_ai.prompt.{idx}.role = 'user', gen_ai.prompt.{idx}.content = '{content_preview}' (length: {len(content)})", file=sys.stderr, flush=True)

    def _set_completion_attributes(self, span: ReadableSpan, messages: list, start_idx: int = 0, log: bool = False):
        """Set gen_ai.completion.* attributes from a list of messages."""
        import sys
        for i, msg in enumerate(messages):
            idx = start_idx + i
            if isinstance(msg, dict):
                role = msg.get("role", "assistant")
                content = str(msg.get("content", ""))
                span._attributes[f"gen_ai.completion.{idx}.role"] = role
                span._attributes[f"gen_ai.completion.{idx}.content"] = content
                if log:
                    content_preview = content[:200] + "..." if len(content) > 200 else content
                    _dbg(f"    Set gen_ai.completion.{idx}.role = '{role}', gen_ai.completion.{idx}.content = '{content_preview}' (length: {len(content)})", file=sys.stderr, flush=True)
            else:
                # Handle string messages
                content = str(msg)
                span._attributes[f"gen_ai.completion.{idx}.role"] = "assistant"
                span._attributes[f"gen_ai.completion.{idx}.content"] = content
                if log:
                    content_preview = content[:200] + "..." if len(content) > 200 else content
                    _dbg(f"    Set gen_ai.completion.{idx}.role = 'assistant', gen_ai.completion.{idx}.content = '{content_preview}' (length: {len(content)})", file=sys.stderr, flush=True)

    def _fallback_messages(self, span: ReadableSpan, span_name: str) -> list:
        """Use system/user attributes or span name when no chat context is available."""
        # NOTE: `gen_ai.system` is the OTel-semconv AI provider NAME
        # (e.g. "openai", "anthropic") — NOT a system prompt. We deliberately
        # don't read it here; otherwise the run renders a fake system message
        # with body "openai".
        system_prompt = span.attributes.get("system") or span.attributes.get("system_prompt") or ""
        user_prompt = (
            span.attributes.get("gen_ai.user")
            or span.attributes.get("user")
            or span.attributes.get("input")
            or ""
        )
        fallback = []
        if system_prompt:
            fallback.append({"role": "system", "content": str(system_prompt)})
        if user_prompt:
            fallback.append({"role": "user", "content": str(user_prompt)})
        if not fallback:
            fallback.append({"role": "user", "content": f"LLM request: {span_name}"})
        return fallback

    def _split_conversation_messages(self, messages: list) -> tuple:
        """
        Split conversation messages into system, first user, and remaining messages.
        Returns: (system_msg, first_user_msg, remaining_msgs)
        """
        system_msg = None
        first_user_msg = None
        remaining_msgs = []
        first_user_found = False

        for msg in messages:
            role = msg.get("role", "") if isinstance(msg, dict) else "user"
            if role == "system" and system_msg is None:
                system_msg = msg
            elif role == "user" and not first_user_found:
                first_user_msg = msg
                first_user_found = True
            elif first_user_found:
                remaining_msgs.append(msg)

        return (system_msg, first_user_msg, remaining_msgs)

    def _extract_llm_messages(self, span: ReadableSpan) -> list:
        """
        Extract LLM input messages from span attributes using multiple strategies.
        Returns a list of message dicts with 'role' and 'content' keys.
        """
        import sys
        _dbg(f"  🔍 Strategy 1: Checking lk.chat_ctx...", file=sys.stderr, flush=True)
        
        # Strategy 1: LiveKit-specific attribute: lk.chat_ctx
        chat_ctx = span.attributes.get("lk.chat_ctx")
        if chat_ctx:
            _dbg(f"    ✓ Found lk.chat_ctx, type={type(chat_ctx)}, length={len(str(chat_ctx)) if isinstance(chat_ctx, str) else 'N/A'}", file=sys.stderr, flush=True)
            try:
                if isinstance(chat_ctx, str):
                    ctx_data = json.loads(chat_ctx)
                else:
                    ctx_data = chat_ctx
                
                # Extract messages from items array
                if isinstance(ctx_data, dict) and "items" in ctx_data:
                    messages = []
                    for item in ctx_data["items"]:
                        if isinstance(item, dict) and item.get("type") == "message":
                            role = item.get("role", "user")
                            content = item.get("content", "")
                            # Content might be a list of strings or a single string
                            if isinstance(content, list):
                                content = " ".join(str(c) for c in content)
                            if content:
                                messages.append({"role": str(role), "content": str(content)})
                    
                    if messages:
                        _dbg(f"    ✅ Strategy 1 SUCCESS: Found {len(messages)} messages from lk.chat_ctx", file=sys.stderr, flush=True)
                        return messages
            except (json.JSONDecodeError, TypeError, KeyError, AttributeError) as e:
                _dbg(f"    ✗ Strategy 1 FAILED: {type(e).__name__}: {e}", file=sys.stderr, flush=True)
        else:
            _dbg(f"    ✗ lk.chat_ctx not found", file=sys.stderr, flush=True)
        
        # Strategy 2: Check for OpenTelemetry semantic convention attributes
        # gen_ai.request.prompt.* or gen_ai.prompt.*
        _dbg(f"  🔍 Strategy 2: Checking gen_ai.request.prompt.*...", file=sys.stderr, flush=True)
        messages = []
        idx = 0
        while True:
            role_key = f"gen_ai.request.prompt.{idx}.role"
            content_key = f"gen_ai.request.prompt.{idx}.content"
            if role_key in span.attributes or content_key in span.attributes:
                role = span.attributes.get(role_key, "user")
                content = span.attributes.get(content_key, "")
                if content:
                    messages.append({"role": str(role), "content": str(content)})
                idx += 1
            else:
                break

        if messages:
            _dbg(f"    ✅ Strategy 2 SUCCESS: Found {len(messages)} messages from gen_ai.request.prompt.*", file=sys.stderr, flush=True)
            return messages
        else:
            _dbg(f"    ✗ No gen_ai.request.prompt.* attributes found", file=sys.stderr, flush=True)

        # Strategy 2b: Check for gen_ai.prompt.* (alternative format)
        _dbg(f"  🔍 Strategy 2b: Checking gen_ai.prompt.*...", file=sys.stderr, flush=True)
        idx = 0
        while True:
            role_key = f"gen_ai.prompt.{idx}.role"
            content_key = f"gen_ai.prompt.{idx}.content"
            if role_key in span.attributes or content_key in span.attributes:
                role = span.attributes.get(role_key, "user")
                content = span.attributes.get(content_key, "")
                if content:
                    messages.append({"role": str(role), "content": str(content)})
                idx += 1
            else:
                break

        if messages:
            _dbg(f"    ✅ Strategy 2b SUCCESS: Found {len(messages)} messages from gen_ai.prompt.*", file=sys.stderr, flush=True)
            return messages
        else:
            _dbg(f"    ✗ No gen_ai.prompt.* attributes found", file=sys.stderr, flush=True)

        # Strategy 2c: LangChain's OTel exporter packs the entire serialized
        # inputs as a JSON string in a single `gen_ai.prompt` attribute (see
        # langsmith/_internal/otel/_otel_exporter.py:699). Symmetric to the
        # output side which lands in `gen_ai.completion`. Without this strategy,
        # ChatOpenAI runs fall through to the "LLM request: <span_name>" fallback.
        _dbg(f"  🔍 Strategy 2c: Checking gen_ai.prompt (singular JSON string)...", file=sys.stderr, flush=True)
        prompt_singular = span.attributes.get("gen_ai.prompt")
        if isinstance(prompt_singular, str):
            try:
                parsed = json.loads(prompt_singular)
            except (json.JSONDecodeError, TypeError) as e:
                _dbg(f"    ✗ Strategy 2c parse failed: {type(e).__name__}: {e}", file=sys.stderr, flush=True)
                parsed = None
            if isinstance(parsed, dict):
                raw = parsed.get("messages") or parsed.get("prompts") or []
                # LangChain chat models pass `messages` as List[List[BaseMessage]]
                # for batched calls; unwrap one level if present.
                if isinstance(raw, list) and raw and isinstance(raw[0], list):
                    raw = raw[0]
                normalized = []
                if isinstance(raw, list):
                    for m in raw:
                        if not isinstance(m, dict):
                            continue
                        # Role: OpenAI-API style (`role`) or LangChain serialized (`type`).
                        role = m.get("role") or m.get("type") or "user"
                        # Map LangChain BaseMessage types to chat roles.
                        if role == "human":
                            role = "user"
                        elif role == "ai":
                            role = "assistant"
                        # Content: direct, or nested in `kwargs.content` for LangChain BaseMessage.
                        content = m.get("content")
                        if not content and isinstance(m.get("kwargs"), dict):
                            content = m["kwargs"].get("content")
                        if content:
                            normalized.append({"role": str(role), "content": str(content)})
                if normalized:
                    _dbg(f"    ✅ Strategy 2c SUCCESS: Found {len(normalized)} messages from gen_ai.prompt singular", file=sys.stderr, flush=True)
                    return normalized
                _dbg(f"    ✗ Strategy 2c: parsed JSON had no usable messages", file=sys.stderr, flush=True)
        else:
            _dbg(f"    ✗ No gen_ai.prompt singular string attribute", file=sys.stderr, flush=True)

        # Strategy 3: Check for messages attribute (JSON string or list)
        _dbg(f"  🔍 Strategy 3: Checking messages/llm.messages/input attributes...", file=sys.stderr, flush=True)
        messages_attr = span.attributes.get("messages") or span.attributes.get("llm.messages") or span.attributes.get("input")
        _dbg(f"    Checking: messages={bool(span.attributes.get('messages'))}, llm.messages={bool(span.attributes.get('llm.messages'))}, input={bool(span.attributes.get('input'))}", file=sys.stderr, flush=True)
        if messages_attr:
            try:
                if isinstance(messages_attr, str):
                    if DEBUG:
                        logger.debug(f"  Parsing JSON string, length={len(messages_attr)}")
                    parsed = json.loads(messages_attr)
                    if isinstance(parsed, list):
                        # Validate and normalize message format
                        normalized = []
                        for msg in parsed:
                            if isinstance(msg, dict) and "content" in msg:
                                normalized.append({
                                    "role": msg.get("role", "user"),
                                    "content": str(msg.get("content", ""))
                                })
                        if normalized:
                            _dbg(f"    ✅ Strategy 3 SUCCESS: Found {len(normalized)} messages from JSON string", file=sys.stderr, flush=True)
                            return normalized
                elif isinstance(messages_attr, list):
                    _dbg(f"    Found list type, length={len(messages_attr)}", file=sys.stderr, flush=True)
                    # Validate and normalize message format
                    normalized = []
                    for msg in messages_attr:
                        if isinstance(msg, dict) and "content" in msg:
                            normalized.append({
                                "role": msg.get("role", "user"),
                                "content": str(msg.get("content", ""))
                            })
                    if normalized:
                        _dbg(f"    ✅ Strategy 3 SUCCESS: Found {len(normalized)} messages from list", file=sys.stderr, flush=True)
                        return normalized
            except (json.JSONDecodeError, TypeError, AttributeError) as e:
                _dbg(f"    ✗ Strategy 3 FAILED: {type(e).__name__}: {e}", file=sys.stderr, flush=True)
        else:
            _dbg(f"    ✗ No messages attribute found", file=sys.stderr, flush=True)

        # Strategy 4: Check for individual system/user/assistant attributes
        _dbg(f"  🔍 Strategy 4: Checking individual system/user/assistant attributes...", file=sys.stderr, flush=True)
        # NOTE: `gen_ai.system` is the AI provider NAME per OTel semconv
        # ("openai", "anthropic"), not a system prompt. Don't read it here.
        system = span.attributes.get("system") or span.attributes.get("system_prompt")
        user = span.attributes.get("gen_ai.user") or span.attributes.get("user") or span.attributes.get("user_input")
        assistant = span.attributes.get("gen_ai.assistant") or span.attributes.get("assistant")
        
        _dbg(f"    system={bool(system)}, user={bool(user)}, assistant={bool(assistant)}", file=sys.stderr, flush=True)
        
        if system or user or assistant:
            result = []
            if system:
                result.append({"role": "system", "content": str(system)})
            if user:
                result.append({"role": "user", "content": str(user)})
            if assistant:
                result.append({"role": "assistant", "content": str(assistant)})
            if result:
                _dbg(f"    ✅ Strategy 4 SUCCESS: Found {len(result)} messages from individual attributes", file=sys.stderr, flush=True)
                return result
        else:
            _dbg(f"    ✗ No individual attributes found", file=sys.stderr, flush=True)

        _dbg(f"  ⚠️  All strategies failed - no messages extracted", file=sys.stderr, flush=True)
        return []

    def _extract_llm_output(self, span: ReadableSpan) -> str:
        """
        Extract LLM output/completion from span attributes using multiple strategies.
        Returns the output as a string.
        """
        import sys
        _dbg(f"  🔍 EXTRACTING LLM OUTPUT:", file=sys.stderr, flush=True)
        
        # Strategy 1: LiveKit-specific attribute: lk.response.text
        _dbg(f"    Strategy 1: Checking lk.response.text...", file=sys.stderr, flush=True)
        output = span.attributes.get("lk.response.text")
        if output:
            _dbg(f"      ✅ Strategy 1 SUCCESS: Found output, length={len(str(output))}", file=sys.stderr, flush=True)
            return str(output)
        else:
            _dbg(f"      ✗ lk.response.text not found", file=sys.stderr, flush=True)
        
        # Strategy 2: OpenTelemetry semantic convention
        _dbg(f"    Strategy 2: Checking gen_ai.response.text / gen_ai.completion.text...", file=sys.stderr, flush=True)
        output = span.attributes.get("gen_ai.response.text") or span.attributes.get("gen_ai.completion.text")
        if output:
            _dbg(f"      ✅ Strategy 2 SUCCESS: Found output, length={len(str(output))}", file=sys.stderr, flush=True)
            return str(output)
        else:
            _dbg(f"      ✗ gen_ai.response.text and gen_ai.completion.text not found", file=sys.stderr, flush=True)

        # Strategy 3: Common attribute names
        _dbg(f"    Strategy 3: Checking common attribute names...", file=sys.stderr, flush=True)
        output = (
            span.attributes.get("gen_ai.response") or
            span.attributes.get("gen_ai.completion") or
            span.attributes.get("output") or
            span.attributes.get("response") or
            span.attributes.get("completion") or
            span.attributes.get("llm.output") or
            span.attributes.get("llm.response") or
            span.attributes.get("text") or
            ""
        )

        if output:
            _dbg(f"      ✅ Strategy 3 SUCCESS: Found output, length={len(str(output))}", file=sys.stderr, flush=True)
            return str(output)
        else:
            _dbg(f"      ✗ No common output attributes found", file=sys.stderr, flush=True)

        # Strategy 4: Check for completion.* attributes
        _dbg(f"    Strategy 4: Checking gen_ai.completion.* attributes...", file=sys.stderr, flush=True)
        idx = 0
        completion_parts = []
        while True:
            content_key = f"gen_ai.completion.{idx}.content"
            if content_key in span.attributes:
                completion_parts.append(str(span.attributes[content_key]))
                idx += 1
            else:
                break

        if completion_parts:
            _dbg(f"      ✅ Strategy 4 SUCCESS: Found {len(completion_parts)} completion parts", file=sys.stderr, flush=True)
            return "\n".join(completion_parts)
        else:
            _dbg(f"      ✗ No gen_ai.completion.* attributes found", file=sys.stderr, flush=True)

        _dbg(f"    ⚠️  All strategies failed - no output extracted", file=sys.stderr, flush=True)
        return ""

    def _get_messages_from_attributes(self, span: ReadableSpan) -> list:
        """Extract messages from span attributes as fallback."""
        messages = []
        # NOTE: `gen_ai.system` is the AI provider NAME per OTel semconv,
        # not a system prompt — don't read it here.
        system = span.attributes.get("system")
        user = span.attributes.get("gen_ai.user") or span.attributes.get("user") or span.attributes.get("input")
        
        if system:
            messages.append({"role": "system", "content": str(system)})
        if user:
            messages.append({"role": "user", "content": str(user)})
        
        return messages

    def _track_messages(self, target_dict: dict, key: str, messages: list, output_data: str):
        """
        Track messages in target_dict, avoiding duplicates.
        Preserves deduplication logic: case-insensitive content comparison.
        """
        if key not in target_dict:
            target_dict[key] = []
            # Add system prompt once at the start
            for msg in messages:
                if isinstance(msg, dict) and msg.get("role") == "system":
                    target_dict[key].append(msg)
                    break

        # Add the latest user message if it's new
        last_user_msg = next(
            (msg for msg in reversed(messages) if isinstance(msg, dict) and msg.get("role") == "user"),
            None
        )
        if last_user_msg:
            new_content = str(last_user_msg.get("content", "")).strip().lower()
            existing_contents = [
                str(m.get("content", "")).strip().lower()
                for m in target_dict[key]
                if isinstance(m, dict) and m.get("role") == "user"
            ]
            if new_content and new_content not in existing_contents:
                target_dict[key].append(last_user_msg)

        # Add the assistant response if it's new
        if output_data:
            new_assistant_content = str(output_data).strip().lower()
            existing_assistant_contents = [
                str(m.get("content", "")).strip().lower()
                for m in target_dict[key]
                if isinstance(m, dict) and m.get("role") == "assistant"
            ]
            if new_assistant_content not in existing_assistant_contents:
                target_dict[key].append({"role": "assistant", "content": output_data})

    def shutdown(self) -> None:
        self._flush_deferred_job_spans()
        if self.downstream:
            self.downstream.shutdown()

    def force_flush(self, timeout_millis: int = 30000) -> bool:
        self._flush_deferred_job_spans()
        if self.downstream:
            return self.downstream.force_flush(timeout_millis)
        return True

    def _defer_job_span(self, trace_id: str, span: ReadableSpan):
        import sys
        self.deferred_job_spans[trace_id] = span
        _dbg(f"⏸️  Deferring export of job span for trace {trace_id}", file=sys.stderr, flush=True)

    def _release_job_span_if_waiting(self, trace_id: str, prompt_msgs: list, completion_msgs: list):
        job_span = self.deferred_job_spans.pop(trace_id, None)
        if not job_span:
            return
        import sys
        _dbg(f"🧩 Releasing deferred job span for trace {trace_id}", file=sys.stderr, flush=True)
        if prompt_msgs:
            self._set_prompt_attributes(job_span, deepcopy(prompt_msgs))
            job_span._attributes["gen_ai.prompt"] = json.dumps(prompt_msgs)
        if completion_msgs:
            self._set_completion_attributes(job_span, deepcopy(completion_msgs))
            job_span._attributes["gen_ai.completion"] = json.dumps(completion_msgs)
        self._attach_audio_recording(job_span)
        self._export_span(job_span)

    def _attach_audio_recording(self, span: ReadableSpan) -> None:
        """If a recording exists, attach it via `langsmith.attachments`.

        Uses the OTel attribute path documented at
        https://docs.langchain.com/langsmith/trace-with-opentelemetry — value is
        a JSON-serialized list of {name, content (base64), mime_type} dicts.
        Avoids needing `projects:read` on the API key (which the LangSmith
        Client `list_runs` path required).
        """
        if _get_audio_file_path is None:
            return
        audio_path = _get_audio_file_path()
        if audio_path is None:
            return
        import sys
        try:
            if not audio_path.exists():
                _dbg(
                    f"[livekit-demo] no audio recording at {audio_path} "
                    "(in console mode, pass `--record` to enable the writer)",
                    file=sys.stderr,
                    flush=True,
                )
                return
            audio_bytes = audio_path.read_bytes()
        except Exception as e:  # pragma: no cover
            _dbg(
                f"[livekit-demo] failed to read audio recording {audio_path}: {e}",
                file=sys.stderr,
                flush=True,
            )
            return

        attachment = {
            "name": "call.ogg",
            "content": base64.b64encode(audio_bytes).decode("ascii"),
            "mime_type": "audio/ogg",
        }
        span._attributes["langsmith.attachments"] = json.dumps([attachment])
        _dbg(
            f"📎 Attached {len(audio_bytes):,} bytes of audio to root span",
            file=sys.stderr,
            flush=True,
        )

    def _flush_deferred_job_spans(self):
        if not self.deferred_job_spans:
            return
        import sys
        _dbg(f"⚠️  Flushing {len(self.deferred_job_spans)} deferred job span(s) without conversation data", file=sys.stderr, flush=True)
        for trace_id, span in list(self.deferred_job_spans.items()):
            self._set_prompt_attributes(span, [{"role": "system", "content": "Conversation not captured"}])
            self._set_completion_attributes(span, [{"role": "assistant", "content": "No conversation turns recorded."}])
            self._export_span(span)
            del self.deferred_job_spans[trace_id]

    def _unwrap_langchain_llmresult(self, output_data: str, span: ReadableSpan) -> str:
        """If output_data is a serialized LangChain LLMResult/ChatResult,
        extract the inner text and surface `generation_info` fields as
        `langsmith.metadata.langchain_*`. Otherwise return the input unchanged.

        Shape we're looking for (LangChain's `LLMResult.dict()` output):
            {
              "generations": [[{
                "text": "...",
                "generation_info": {"finish_reason": "stop", "model_name": ..., ...},
                "type": "ChatGeneration",
                "message": {"kwargs": {"content": "..."}, ...}
              }]],
              ...
            }

        Uses `json.loads` (primitive types only — no class reconstruction) and
        an explicit shape allowlist (must have a `generations` array of arrays
        of dicts). If any check fails, returns the original string untouched.
        """
        if not isinstance(output_data, str) or not output_data.lstrip().startswith("{"):
            return output_data
        try:
            parsed = json.loads(output_data)
        except (json.JSONDecodeError, TypeError):
            return output_data
        if not isinstance(parsed, dict) or "generations" not in parsed:
            return output_data
        gens = parsed.get("generations")
        if not (isinstance(gens, list) and gens and isinstance(gens[0], list) and gens[0]):
            return output_data
        first = gens[0][0]
        if not isinstance(first, dict):
            return output_data

        text = first.get("text")
        if not text:
            msg = first.get("message") or {}
            kwargs = msg.get("kwargs") if isinstance(msg, dict) else None
            if isinstance(kwargs, dict):
                text = kwargs.get("content")
        if not text:
            return output_data

        gen_info = first.get("generation_info")
        if isinstance(gen_info, dict):
            for k, v in gen_info.items():
                if v is not None and not isinstance(v, (dict, list)):
                    span._attributes[f"langsmith.metadata.langchain_{k}"] = v

        return str(text)

    def _collect_turn_latencies(self, span: ReadableSpan, span_name: str) -> None:
        """Aggregate per-pipeline-stage latencies onto the agent_turn span.

        LiveKit emits each stage's latency on its own span (`lk.transcription_delay`
        on user_turn, `lk.response.ttft` on llm_node, etc.). To get a per-turn
        view in one place (mirroring LiveKit's own `ChatMessage.metrics`), we
        cache each value as the child span ends, then drain onto `agent_turn`
        which ends last in the turn. Result: agent_turn carries
        `langsmith.metadata.turn_stt_transcription_delay`, `turn_llm_ttft`,
        `turn_tts_ttfb`, `turn_e2e_latency`, etc.

        Caveat: under interruption, the next user_turn could in rare cases end
        before agent_turn_n drains. The pipeline is sequential enough in the
        common case that we accept the small mix-up risk for now.
        """
        trace_int = span.context.trace_id

        def cache(name: str, src_attr: str) -> None:
            v = span.attributes.get(src_attr)
            if isinstance(v, (int, float)):
                self._turn_latencies_by_trace_id.setdefault(trace_int, {})[name] = v

        if "user_turn" in span_name:
            cache("stt_transcription_delay", "lk.transcription_delay")
            cache("stt_end_of_turn_delay", "lk.end_of_turn_delay")
        elif span_name == "eou_detection":
            cache("eou_endpointing_delay", "lk.eou.endpointing_delay")
            cache("eou_probability", "lk.eou.probability")
            lang = span.attributes.get("lk.eou.language")
            if lang:
                self._turn_latencies_by_trace_id.setdefault(trace_int, {})["eou_language"] = str(lang)
        elif span_name == "llm_node":
            cache("llm_ttft", "lk.response.ttft")
        elif span_name == "tts_node":
            cache("tts_ttfb", "lk.response.ttfb")
        elif span_name == "agent_turn":
            latencies = self._turn_latencies_by_trace_id.pop(trace_int, {})
            e2e = span.attributes.get("lk.e2e_latency")
            if isinstance(e2e, (int, float)):
                latencies["e2e_latency"] = e2e
            sid = span.attributes.get("lk.speech_id")
            if sid:
                latencies["speech_id"] = str(sid)
            for k, v in latencies.items():
                span._attributes[f"langsmith.metadata.turn_{k}"] = v
            # Synthetic check: do the stage parts roughly sum to e2e? Useful
            # signal that nothing's missing from the breakdown.
            components = (
                "stt_transcription_delay",
                "eou_endpointing_delay",
                "llm_ttft",
                "tts_ttfb",
            )
            parts = [
                latencies[k]
                for k in components
                if isinstance(latencies.get(k), (int, float))
            ]
            if len(parts) >= 2:
                span._attributes["langsmith.metadata.turn_total_breakdown"] = sum(parts)

    def _passthrough_lk_attrs(self, span: ReadableSpan) -> None:
        """Blanket forward every scalar `lk.*` attribute to
        `langsmith.metadata.lk_<flattened_name>`.

        LangSmith's OTel ingester doesn't recognize the `lk.*` vendor
        namespace, so anything we don't translate explicitly evaporates.
        This catch-all runs right before export — after the per-span-type
        branches in `on_end` — so explicit unpacking still produces nicer
        keys (e.g. `tts_audio_duration`), and this just fills the gaps and
        future-proofs against new LiveKit attributes (e2e_latency,
        eou.*, interruption.*, response.ttft, the raw lk.llm_metrics
        JSON blob, etc.).

        Denylist-style: forward all `lk.*` scalars and sequences thereof,
        skipping dicts (OTel forbids dict attribute values). Safe because
        the source is the in-process LiveKit SDK emitting its own OTel
        attributes, not external/user-controlled input.
        """
        for key in list(span.attributes.keys()):
            if not key.startswith("lk."):
                continue
            v = span.attributes[key]
            if v is None or isinstance(v, dict):
                continue
            ms_key = f"langsmith.metadata.{key.replace('.', '_')}"
            # Don't overwrite anything an explicit branch already set with
            # a nicer name (e.g. tts_audio_duration vs lk_tts_metrics).
            if ms_key in span._attributes:
                continue
            span._attributes[ms_key] = v

    def _export_span(self, span: ReadableSpan):
        self._passthrough_lk_attrs(span)
        if self.downstream:
            self.downstream.on_end(span)


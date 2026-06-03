"""OTel → LangSmith bridge for Pipecat.

Pipecat emits OTel spans named `conversation`, `turn`, `stt`, `llm`, and `tts`,
but with attributes (`transcript`, `input`/`output`, `text`, `turn.number`, …)
that LangSmith's OTLP ingester doesn't recognize. This `SpanProcessor` rewrites
each span type into the `gen_ai.*` / `langsmith.*` namespaces LangSmith keys
off, aggregates per-turn and per-conversation messages, and attaches the
recorded audio (whole conversation on the root span; per-turn snippets on each
`turn` span).

Adapted from the official LangChain × Pipecat tracing demo
(github.com/langchain-ai/voice-agents-tracing/blob/main/pipecat/langsmith_processor.py).
The notable change is `setup_langsmith_tracing()`: rather than configuring the
tracer and registering the processor as an import side effect, the agent calls
it explicitly once the LangSmith env is wired and confirmed present.

The trace shape in LangSmith:

    conversation                      (root; attaches the whole-conversation WAV)
    └── turn × N                      (per exchange; turn.was_interrupted, turn audio)
        ├── stt                       (audio → transcript)
        ├── llm                       (messages → response, incl. tool calls)
        └── tts                       (response text → audio)
"""

from __future__ import annotations

import base64
import json
from pathlib import Path

from loguru import logger
from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.trace import ReadableSpan, SpanProcessor, TracerProvider


class LangSmithSpanProcessor(SpanProcessor):
    """Enriches Pipecat's OTel spans with LangSmith-compatible attributes."""

    def __init__(self) -> None:
        super().__init__()
        # Cross-span state, keyed for proper LangSmith grouping.
        self.conversation_messages: dict = {}  # trace_id -> list of messages
        self.turn_messages: dict = {}  # parent_span_id -> list of messages
        self.trace_to_conversation_id: dict = {}  # trace_id -> conversation_id
        self.conversation_recordings: dict = {}  # conversation_id -> recording_path
        self.conversation_recorders: dict = {}  # conversation_id -> AudioRecorder
        self.turn_audio_recorders: dict = {}  # conversation_id -> TurnAudioRecorder

    # -- recorder registration ------------------------------------------------

    def register_recording(self, conversation_id, recording_path, audio_recorder=None):
        """Register the whole-conversation recording for the root span."""
        self.conversation_recordings[conversation_id] = recording_path
        if audio_recorder:
            self.conversation_recorders[conversation_id] = audio_recorder

    def register_turn_audio_recorder(self, conversation_id, turn_audio_recorder):
        """Register the per-turn recorder so turn spans can attach their audio."""
        self.turn_audio_recorders[conversation_id] = turn_audio_recorder

    # -- span lifecycle -------------------------------------------------------

    def on_start(self, span: ReadableSpan, parent_context=None) -> None:
        pass

    def on_end(self, span: ReadableSpan) -> None:
        """Rewrite a Pipecat span into LangSmith's expected attribute shape."""
        # Track each conversation as a thread in LangSmith.
        trace_id = format(span.context.trace_id, "032x")
        span._attributes["langsmith.metadata.thread_id"] = trace_id

        # Link every span to its conversation for grouping.
        if trace_id in self.trace_to_conversation_id:
            conversation_id = self.trace_to_conversation_id[trace_id]
            span._attributes["conversation.id"] = conversation_id
            span._attributes["langsmith.parent_span_id"] = "conversation"

        if span.name == "stt":
            self._handle_stt(span)
        elif span.name == "llm":
            self._handle_llm(span, trace_id)
        elif span.name == "tts":
            self._handle_tts(span)
        elif span.name == "turn":
            self._handle_turn(span, trace_id)
        elif span.name == "conversation":
            self._handle_conversation(span, trace_id)

    # -- per-span-type handlers ----------------------------------------------

    def _handle_stt(self, span: ReadableSpan) -> None:
        """STT span: audio input → transcribed text."""
        transcript = span.attributes.get("transcript", "")
        span._attributes["langsmith.span.kind"] = "llm"
        self._set_prompt_attributes(span, [{"role": "user", "content": "audio_segment"}])
        self._set_completion_attributes(
            span, [{"role": "assistant", "content": transcript}]
        )

    def _handle_llm(self, span: ReadableSpan, trace_id: str) -> None:
        """LLM span: conversation messages → AI response."""
        input_data = span.attributes.get("input", "")
        output_data = span.attributes.get("output", "")
        span._attributes["langsmith.span.kind"] = "llm"

        messages = []
        try:
            messages = json.loads(input_data)
            self._set_prompt_attributes(span, messages)
        except (json.JSONDecodeError, TypeError):
            pass

        if output_data:
            self._set_completion_attributes(
                span, [{"role": "assistant", "content": output_data}]
            )

        # Aggregate for the enclosing turn and conversation spans.
        parent_span_id = format(span.parent.span_id, "016x") if span.parent else None
        self._track_messages(self.conversation_messages, trace_id, messages, output_data)
        if parent_span_id:
            self._track_messages(self.turn_messages, parent_span_id, messages, output_data)

    def _handle_tts(self, span: ReadableSpan) -> None:
        """TTS span: text → audio."""
        text = span.attributes.get("text", "")
        voice_id = span.attributes.get("voice_id", "")
        span._attributes["langsmith.span.kind"] = "llm"
        self._set_prompt_attributes(
            span,
            [
                {"role": "system", "content": f"Convert to speech with voice: {voice_id}"},
                {"role": "user", "content": text},
            ],
        )
        self._set_completion_attributes(
            span, [{"role": "assistant", "content": f"Generated audio for: {text}"}]
        )

    def _handle_turn(self, span: ReadableSpan, trace_id: str) -> None:
        """Turn span: one user↔assistant exchange; carries was_interrupted."""
        turn_number = span.attributes.get("turn.number", 0)
        was_interrupted = span.attributes.get("turn.was_interrupted", False)
        span._attributes["langsmith.span.kind"] = "chain"

        span_id = format(span.context.span_id, "016x")
        turn_msgs = self.turn_messages.get(span_id, [])
        user_msgs = [m for m in turn_msgs if m.get("role") == "user"]
        assistant_msgs = [m for m in turn_msgs if m.get("role") == "assistant"]

        if user_msgs:
            self._set_prompt_attributes(span, user_msgs)
        else:
            self._set_prompt_attributes(
                span, [{"role": "user", "content": f"Turn {turn_number}"}]
            )

        if assistant_msgs:
            self._set_completion_attributes(span, assistant_msgs)
        else:
            status = "interrupted" if was_interrupted else "no response"
            self._set_completion_attributes(
                span, [{"role": "assistant", "content": status}]
            )

        self._attach_turn_audio(span, trace_id, turn_number)

        if span_id in self.turn_messages:
            del self.turn_messages[span_id]

    def _handle_conversation(self, span: ReadableSpan, trace_id: str) -> None:
        """Conversation span: the whole session; the LangSmith root."""
        conversation_id = span.attributes.get("conversation.id", "") or span.attributes.get(
            "conversation_id", ""
        )
        conversation_type = span.attributes.get("conversation.type", "voice")
        self.trace_to_conversation_id[trace_id] = conversation_id

        span._attributes["langsmith.span.kind"] = "chain"
        span._attributes["langsmith.root_span"] = True

        conv_msgs = self.conversation_messages.get(trace_id, [])
        if conv_msgs:
            _system, first_user, remaining = self._split_conversation_messages(conv_msgs)
            prompt_msgs = [first_user] if first_user else []
            self._set_prompt_attributes(span, prompt_msgs)
            if remaining:
                self._set_completion_attributes(span, remaining)
            else:
                self._set_completion_attributes(
                    span, [{"role": "assistant", "content": "No responses yet"}]
                )
        else:
            self._set_prompt_attributes(
                span,
                [{"role": "system", "content": f"Starting {conversation_type} conversation"}],
            )
            self._set_completion_attributes(
                span, [{"role": "assistant", "content": "No messages"}]
            )

        self._attach_conversation_audio(span, conversation_id)
        self._cleanup_conversation(trace_id, conversation_id)

    # -- audio attachment -----------------------------------------------------

    def _attach_turn_audio(self, span: ReadableSpan, trace_id: str, turn_number) -> None:
        conversation_id = span.attributes.get(
            "conversation.id", ""
        ) or self.trace_to_conversation_id.get(trace_id, "")
        recorder = self.turn_audio_recorders.get(conversation_id)
        if recorder is None:
            return

        turn_files = recorder.save_turn_audio_sync(turn_number)
        attachments = []
        for role, key in (("user", "user"), ("ai", "ai")):
            path = turn_files.get(key)
            if not path:
                continue
            encoded = self._load_audio_file(path)
            if encoded:
                attachments.append(
                    {
                        "name": f"turn_{turn_number}_{role}.wav",
                        "content": encoded,
                        "mime_type": "audio/wav",
                    }
                )
        if attachments:
            span._attributes["langsmith.attachments"] = json.dumps(attachments)

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
        self.conversation_messages.pop(trace_id, None)
        self.trace_to_conversation_id.pop(trace_id, None)
        self.conversation_recordings.pop(conversation_id, None)
        self.conversation_recorders.pop(conversation_id, None)
        self.turn_audio_recorders.pop(conversation_id, None)

    # -- attribute helpers ----------------------------------------------------

    def _set_prompt_attributes(self, span: ReadableSpan, messages: list) -> None:
        for i, msg in enumerate(messages):
            span._attributes[f"gen_ai.prompt.{i}.role"] = msg.get("role", "")
            span._attributes[f"gen_ai.prompt.{i}.content"] = msg.get("content", "")

    def _set_completion_attributes(self, span: ReadableSpan, messages: list) -> None:
        for i, msg in enumerate(messages):
            span._attributes[f"gen_ai.completion.{i}.role"] = msg.get("role", "")
            span._attributes[f"gen_ai.completion.{i}.content"] = msg.get("content", "")

    def _split_conversation_messages(self, messages: list) -> tuple:
        """Split into (system_msg, first_user_msg, remaining_msgs)."""
        system_msg = None
        first_user_msg = None
        remaining_msgs: list = []
        first_user_found = False
        for msg in messages:
            role = msg.get("role", "")
            if role == "system" and system_msg is None:
                system_msg = msg
            elif role == "user" and not first_user_found:
                first_user_msg = msg
                first_user_found = True
            elif first_user_found:
                remaining_msgs.append(msg)
        return system_msg, first_user_msg, remaining_msgs

    def _track_messages(self, target: dict, key, messages: list, output_data: str) -> None:
        """Append new user/assistant messages, deduping on lowercased content."""
        if key not in target:
            target[key] = []
            for msg in messages:  # keep the system prompt once, at the start
                if msg.get("role") == "system":
                    target[key].append(msg)
                    break

        last_user = next(
            (m for m in reversed(messages) if m.get("role") == "user"), None
        )
        if last_user:
            content = last_user.get("content", "").strip().lower()
            existing = [
                m.get("content", "").strip().lower()
                for m in target[key]
                if m.get("role") == "user"
            ]
            if content and content not in existing:
                target[key].append(last_user)

        if output_data:
            content = output_data.strip().lower()
            existing = [
                m.get("content", "").strip().lower()
                for m in target[key]
                if m.get("role") == "assistant"
            ]
            if content not in existing:
                target[key].append({"role": "assistant", "content": output_data})

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
        pass

    def force_flush(self, timeout_millis: int = 30000) -> bool:
        return True


def setup_langsmith_tracing() -> LangSmithSpanProcessor:
    """Configure OTel export to LangSmith and register the span processor.

    Reads `OTEL_EXPORTER_OTLP_ENDPOINT` / `OTEL_EXPORTER_OTLP_HEADERS` from the
    environment (wired by `voice_demo.tracing.configure`). Returns the processor
    instance so the agent can register its audio recorders on it.
    """
    from pipecat.utils.tracing.setup import setup_tracing

    setup_tracing(
        service_name="voice-demo-pipecat",
        exporter=OTLPSpanExporter(),
        console_export=False,  # avoid base64-audio spam on the console
    )

    span_processor = LangSmithSpanProcessor()
    provider = trace.get_tracer_provider()
    if isinstance(provider, TracerProvider):
        provider.add_span_processor(span_processor)
    return span_processor

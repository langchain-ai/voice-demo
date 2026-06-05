"""A readable view over raw ADK `run_live` events.

Every item the runner yields is the same ADK Event class with all-optional
fields; "what kind of event is this" is implicit in which fields are populated
(unlike OpenAI Realtime's flat `.type` tag). LiveEvent centralizes that field
inspection so the run loop reads as intent.
"""

from __future__ import annotations

from typing import Any


class LiveEvent:
    """Wraps one raw ADK `run_live` event."""

    def __init__(self, raw: Any):
        self.raw = raw

    @property
    def interrupted(self) -> bool:
        """The server detected the user barging in over the agent."""
        return bool(self.raw.interrupted)

    @property
    def turn_complete(self) -> bool:
        """The server finished its half of the exchange."""
        return bool(self.raw.turn_complete)

    @property
    def user_transcript(self) -> str | None:
        """Transcribed fragment of the user's speech, if any."""
        tx = self.raw.input_transcription
        return tx.text if tx and tx.text else None

    @property
    def agent_transcript(self) -> str | None:
        """Transcribed fragment of the agent's speech, if any."""
        tx = self.raw.output_transcription
        return tx.text if tx and tx.text else None

    @property
    def final_user_transcript(self) -> str | None:
        """The user's whole utterance — only on the `finished=True` event.

        Transcription streams in fragments, then a final event repeats the
        full text with `finished` set. Print this one and skip the fragments.
        """
        tx = self.raw.input_transcription
        return tx.text if tx and tx.finished and tx.text else None

    @property
    def final_agent_transcript(self) -> str | None:
        """The agent's whole utterance — only on the `finished=True` event."""
        tx = self.raw.output_transcription
        return tx.text if tx and tx.finished and tx.text else None

    @property
    def _parts(self) -> list[Any]:
        content = self.raw.content
        return list(content.parts) if content and content.parts else []

    @property
    def audio_chunks(self) -> list[bytes]:
        """PCM16 agent audio carried in this event."""
        return [
            part.inline_data.data
            for part in self._parts
            if part.inline_data and part.inline_data.data
        ]

    @property
    def function_calls(self) -> list[Any]:
        """Tool invocations the model requested in this event."""
        return [
            part.function_call
            for part in self._parts
            if part.function_call and part.function_call.name
        ]

    @property
    def function_responses(self) -> list[Any]:
        """Results of tool calls ADK executed, on their way back to the model."""
        return [
            part.function_response
            for part in self._parts
            if part.function_response and part.function_response.name
        ]

    @property
    def is_audio_only(self) -> bool:
        """True when the event's only payload is agent audio.

        ADK streams agent speech as a flood of audio-chunk events; spanning
        each one would bury the trace, so these are played but not traced.
        """
        return bool(self.audio_chunks) and not (
            self.interrupted
            or self.turn_complete
            or self.user_transcript
            or self.agent_transcript
            or self.function_calls
            or self.function_responses
        )

    @property
    def is_inbound(self) -> bool:
        """True for user speech heading toward the model (traced as span
        `inputs`); everything else is the model/server replying (`outputs`)."""
        return self.user_transcript is not None

    @property
    def label(self) -> str:
        """A readable span name derived from the populated fields."""
        if self.interrupted:
            return "interrupted"
        if self.turn_complete:
            return "turn_complete"
        if self.user_transcript:
            return "input_transcription"
        if self.agent_transcript:
            return "output_transcription"
        if self.function_calls:
            return f"function_call: {self.function_calls[0].name}"
        if self.function_responses:
            return f"function_response: {self.function_responses[0].name}"
        return "event"

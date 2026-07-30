"""Readable view over raw ``google.genai`` Gemini Live server messages."""

from __future__ import annotations

from typing import Any

MAX_TRANSCRIPT_CHARS = 2_000


def append_transcript(current: str, fragment: str) -> str:
    """Append one Gemini transcription delta, retaining the per-turn cap."""
    remaining = MAX_TRANSCRIPT_CHARS - len(current)
    return current if remaining <= 0 else current + fragment[:remaining]


class LiveMessage:
    """Expose the application-relevant fields of a ``LiveServerMessage``."""

    def __init__(self, raw: Any) -> None:
        self.raw = raw

    @property
    def server_content(self) -> Any:
        return getattr(self.raw, "server_content", None)

    @property
    def interrupted(self) -> bool:
        content = self.server_content
        return bool(content is not None and getattr(content, "interrupted", False))

    @property
    def turn_complete(self) -> bool:
        content = self.server_content
        return bool(content is not None and getattr(content, "turn_complete", False))

    def _transcription(self, attr: str) -> Any:
        content = self.server_content
        return getattr(content, attr, None) if content is not None else None

    def _transcript(self, attr: str) -> str | None:
        tx = self._transcription(attr)
        text = getattr(tx, "text", None) if tx is not None else None
        return str(text) if text else None

    def _transcript_finished(self, attr: str) -> bool:
        tx = self._transcription(attr)
        return bool(tx is not None and getattr(tx, "finished", False))

    @property
    def user_transcript(self) -> str | None:
        return self._transcript("input_transcription")

    @property
    def user_transcript_finished(self) -> bool:
        return self._transcript_finished("input_transcription")

    @property
    def agent_transcript(self) -> str | None:
        return self._transcript("output_transcription")

    @property
    def agent_transcript_finished(self) -> bool:
        return self._transcript_finished("output_transcription")

    @property
    def audio_chunks(self) -> list[bytes]:
        content = self.server_content
        turn = getattr(content, "model_turn", None) if content is not None else None
        parts = getattr(turn, "parts", None) if turn is not None else None
        chunks: list[bytes] = []
        for part in parts or []:
            blob = getattr(part, "inline_data", None)
            data = getattr(blob, "data", None) if blob is not None else None
            if isinstance(data, bytes) and data:
                chunks.append(data)
        return chunks

    @property
    def function_calls(self) -> list[Any]:
        tool_call = getattr(self.raw, "tool_call", None)
        return list(getattr(tool_call, "function_calls", None) or [])

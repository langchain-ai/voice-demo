"""Readable view over raw ``google.genai`` Gemini Live server messages."""

from __future__ import annotations

from typing import Any


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

    def _transcript(self, attr: str, *, final_only: bool) -> str | None:
        content = self.server_content
        tx = getattr(content, attr, None) if content is not None else None
        text = getattr(tx, "text", None) if tx is not None else None
        if not text or (final_only and not getattr(tx, "finished", False)):
            return None
        return str(text)

    @property
    def user_transcript(self) -> str | None:
        return self._transcript("input_transcription", final_only=False)

    @property
    def final_user_transcript(self) -> str | None:
        return self._transcript("input_transcription", final_only=True)

    @property
    def final_agent_transcript(self) -> str | None:
        return self._transcript("output_transcription", final_only=True)

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

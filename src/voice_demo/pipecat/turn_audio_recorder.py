"""Per-turn audio recorder (a Pipecat `FrameProcessor`).

Subscribes to the `TurnTrackingObserver` to find turn boundaries and saves
separate WAVs for the user's speech and the AI's response within each turn —
so an interrupted turn's audio reflects exactly what was spoken before the
barge-in. Bounded per-turn buffers prevent runaway memory on long turns.

Ported from the official LangChain × Pipecat tracing demo.
"""

from __future__ import annotations

import wave
from pathlib import Path

import numpy as np
from loguru import logger
from pipecat.frames.frames import (
    AudioRawFrame,
    Frame,
    InputAudioRawFrame,
    OutputAudioRawFrame,
    TTSAudioRawFrame,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from scipy import signal


class TurnAudioRecorder(FrameProcessor):
    """Captures user and AI audio separately, one WAV pair per turn."""

    MAX_BUFFER_FRAMES = 1000  # ~20s per source per turn

    def __init__(
        self,
        span_processor,
        conversation_id: str,
        recordings_dir: Path,
        turn_tracker=None,
        user_sample_rate: int = 16000,
        ai_sample_rate: int = 24000,
        channels: int = 1,
    ):
        super().__init__()
        self._span_processor = span_processor
        self._conversation_id = conversation_id
        self._recordings_dir = Path(recordings_dir)
        self._turn_tracker = turn_tracker
        self._channels = channels

        self._current_turn_number = 0
        self._is_turn_active = False

        self._current_user_frames: list[tuple[bytes, int | None]] = []
        self._current_ai_frames: list[tuple[bytes, int | None]] = []
        self._user_detected_rates: set[int] = set()
        self._ai_detected_rates: set[int] = set()
        self._default_user_rate = user_sample_rate
        self._default_ai_rate = ai_sample_rate
        # Sides ("user"/"AI") already warned about hitting the cap this turn.
        self._warned: set[str] = set()

        self._recordings_dir.mkdir(parents=True, exist_ok=True)

    def connect_to_turn_tracker(self, turn_tracker) -> None:
        """Register turn start/end handlers (call after the task is created)."""
        self._turn_tracker = turn_tracker
        turn_tracker.add_event_handler("on_turn_started", self._on_turn_started)
        turn_tracker.add_event_handler("on_turn_ended", self._on_turn_ended)

    async def _on_turn_started(self, _observer, turn_number: int) -> None:
        self._current_turn_number = turn_number
        self._is_turn_active = True
        # Clear previous turn's buffers (prevents memory accumulation).
        self._current_user_frames = []
        self._current_ai_frames = []
        self._user_detected_rates.clear()
        self._ai_detected_rates.clear()
        self._warned.clear()

    async def _on_turn_ended(
        self, _observer, turn_number: int, duration: float, was_interrupted: bool
    ) -> None:
        self._is_turn_active = False

    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        await super().process_frame(frame, direction)

        if isinstance(frame, AudioRawFrame) and self._is_turn_active:
            frame_sample_rate = getattr(frame, "sample_rate", None)
            if isinstance(frame, InputAudioRawFrame):
                self._buffer(
                    self._current_user_frames,
                    self._user_detected_rates,
                    frame,
                    frame_sample_rate,
                    "user",
                )
            elif isinstance(frame, (OutputAudioRawFrame, TTSAudioRawFrame)):
                self._buffer(
                    self._current_ai_frames,
                    self._ai_detected_rates,
                    frame,
                    frame_sample_rate,
                    "AI",
                )

        await self.push_frame(frame, direction)

    def _buffer(self, frames, rates, frame, sample_rate, label) -> None:
        if sample_rate:
            rates.add(sample_rate)
        if len(frames) < self.MAX_BUFFER_FRAMES:
            frames.append((frame.audio, sample_rate))
        elif label not in self._warned:
            # Cap reached — keep the first MAX_BUFFER_FRAMES and warn exactly
            # once per side per turn (not on every dropped frame).
            self._warned.add(label)
            logger.warning(
                f"{label} audio buffer limit reached for turn "
                f"{self._current_turn_number}; keeping the first "
                f"{self.MAX_BUFFER_FRAMES} frames, dropping the rest."
            )

    def save_turn_audio_sync(self, turn_number: int) -> dict:
        """Save WAVs for `turn_number`; return {'user': path, 'ai': path}."""
        saved: dict[str, str] = {}
        if turn_number != self._current_turn_number:
            return saved
        try:
            if self._current_user_frames:
                path = self._recordings_dir / f"turn_{turn_number}_user.wav"
                rate = (
                    max(self._user_detected_rates)
                    if self._user_detected_rates
                    else self._default_user_rate
                )
                self._save_wav_file(path, self._current_user_frames, rate)
                saved["user"] = str(path)
            if self._current_ai_frames:
                path = self._recordings_dir / f"turn_{turn_number}_ai.wav"
                rate = (
                    max(self._ai_detected_rates)
                    if self._ai_detected_rates
                    else self._default_ai_rate
                )
                self._save_wav_file(path, self._current_ai_frames, rate)
                saved["ai"] = str(path)
        except Exception as e:
            logger.error(f"Failed to save turn {turn_number} audio: {e}")
        return saved

    def _save_wav_file(self, output_path: Path, frames: list, target_sample_rate: int) -> None:
        if not frames:
            return
        with wave.open(str(output_path), "wb") as wav_file:
            wav_file.setnchannels(self._channels)
            wav_file.setsampwidth(2)  # 16-bit PCM
            wav_file.setframerate(target_sample_rate)
            for audio_data, frame_sample_rate in frames:
                if frame_sample_rate and frame_sample_rate != target_sample_rate:
                    arr = np.frombuffer(audio_data, dtype=np.int16)
                    n = int(len(arr) * target_sample_rate / frame_sample_rate)
                    audio_data = signal.resample(arr, n).astype(np.int16).tobytes()
                wav_file.writeframes(audio_data)

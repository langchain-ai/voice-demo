"""Whole-conversation audio recorder (a Pipecat `FrameProcessor`).

Captures every `AudioRawFrame` flowing through the pipeline (mic input + TTS
output) and writes one WAV. Input and output can arrive at different sample
rates, so frames are resampled to the highest detected rate. A bounded buffer
is flushed to disk periodically to avoid unbounded memory on long calls.

Ported from the official LangChain × Pipecat tracing demo.
"""

from __future__ import annotations

import wave

import numpy as np
from loguru import logger
from pipecat.frames.frames import AudioRawFrame, Frame
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from scipy import signal


class AudioRecorder(FrameProcessor):
    """Records all pipeline audio to a single WAV, resampling as needed."""

    # Flush to disk after this many buffered frames (~10s at typical rates).
    MAX_BUFFER_FRAMES = 500

    def __init__(self, output_path: str, sample_rate: int = 24000, channels: int = 1):
        super().__init__()
        self.output_path = output_path
        self._default_sample_rate = sample_rate
        self._target_sample_rate: int | None = None
        self.channels = channels
        self.audio_frames: list[tuple[bytes, int | None]] = []
        self.detected_sample_rates: set[int] = set()
        self._wav_file: wave.Wave_write | None = None
        self._is_initialized = False

    def _initialize_wav_file(self) -> None:
        if self._is_initialized:
            return
        try:
            self._target_sample_rate = (
                max(self.detected_sample_rates)
                if self.detected_sample_rates
                else self._default_sample_rate
            )
            self._wav_file = wave.open(self.output_path, "wb")
            self._wav_file.setnchannels(self.channels)
            self._wav_file.setsampwidth(2)  # 16-bit PCM
            self._wav_file.setframerate(self._target_sample_rate)
            self._is_initialized = True
        except Exception as e:
            logger.error(f"Failed to initialize WAV file {self.output_path}: {e}")

    def _flush_buffer(self) -> None:
        if not self.audio_frames or not self._wav_file:
            return
        try:
            for audio_data, frame_sample_rate in self.audio_frames:
                if frame_sample_rate and frame_sample_rate != self._target_sample_rate:
                    arr = np.frombuffer(audio_data, dtype=np.int16)
                    n = int(len(arr) * self._target_sample_rate / frame_sample_rate)
                    audio_data = signal.resample(arr, n).astype(np.int16).tobytes()
                self._wav_file.writeframes(audio_data)
            self.audio_frames.clear()
        except Exception as e:
            logger.error(f"Failed to flush audio buffer to {self.output_path}: {e}")

    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        await super().process_frame(frame, direction)

        if isinstance(frame, AudioRawFrame):
            frame_sample_rate = getattr(frame, "sample_rate", None)
            if frame_sample_rate:
                self.detected_sample_rates.add(frame_sample_rate)
            if not self._is_initialized:
                self._initialize_wav_file()
            self.audio_frames.append((frame.audio, frame_sample_rate))
            if len(self.audio_frames) >= self.MAX_BUFFER_FRAMES:
                self._flush_buffer()

        await self.push_frame(frame, direction)

    def save_recording(self) -> None:
        """Flush remaining audio and close the WAV file."""
        if not self._is_initialized:
            logger.warning("No audio frames to save — WAV file was never initialized")
            return
        try:
            self._flush_buffer()
            if self._wav_file:
                self._wav_file.close()
                self._wav_file = None
            logger.info(f"Recording saved to: {self.output_path}")
        except Exception as e:
            logger.error(f"Failed to save recording to {self.output_path}: {e}")
        finally:
            self._is_initialized = False

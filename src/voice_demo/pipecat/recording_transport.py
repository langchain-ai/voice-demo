"""A `LocalAudioTransport` that records *what was heard*, not what was generated.

The default approach of recording with a `FrameProcessor` placed in the pipeline
taps audio *upstream* of `transport.output()` — i.e. the TTS frames as generated,
before the output transport's clock queue discards the unplayed tail on barge-in.
So that recording over-captures: it includes audio the user never heard.

This subclass instead taps at the **device-write boundary**:

  * agent audio is recorded in `write_audio_frame`, which the output transport's
    clock queue only reaches *after* interruption truncation — so flushed audio
    is never recorded (it was never played);
  * user audio is recorded straight off the input callback.

That's the exact same principle the OpenAI/ADK backends use with
`SpeakerStream.set_played_callback`, and we deliberately reuse the *same* stereo
WAV builder (`sdk_tracing.build_stereo_session_wav`) so all three backends emit
the identical artifact: one stereo file, left = user, right = agent, laid out at
natural play time, of what was actually heard.
"""

from __future__ import annotations

import time
from pathlib import Path

from pipecat.transports.local.audio import (
    LocalAudioInputTransport,
    LocalAudioOutputTransport,
    LocalAudioTransport,
    LocalAudioTransportParams,
)

from ..audio import resample_pcm16
from ..sdk_tracing import build_stereo_session_wav


class ConversationRecorder:
    """Collects timestamped played/captured PCM16 and writes one stereo WAV.

    Duck-typed to the span processor's recorder hook: it calls
    `save_recording()` when the conversation span ends, then attaches the file.
    """

    def __init__(self, output_path: str | Path) -> None:
        self._path = str(output_path)
        self._t0 = time.monotonic()
        # (offset_seconds, pcm16_bytes, sample_rate)
        self._user: list[tuple[float, bytes, int]] = []
        self._agent: list[tuple[float, bytes, int]] = []
        self._saved = False

    def _now(self) -> float:
        return time.monotonic() - self._t0

    def record_user(self, data: bytes, sample_rate: int) -> None:
        self._user.append((self._now(), data, sample_rate))

    def record_agent(self, data: bytes, sample_rate: int) -> None:
        self._agent.append((self._now(), data, sample_rate))

    def save_recording(self) -> None:
        """Build the stereo WAV (user L / agent R) and write it to disk."""
        if self._saved:
            return
        self._saved = True
        # Snapshot to avoid racing the audio threads still appending.
        user = list(self._user)
        agent = list(self._agent)
        rates = [r for *_, r in (user + agent) if r]
        target = max(rates) if rates else 24_000
        user_chunks = [(t, resample_pcm16(d, r or target, target)) for t, d, r in user]
        agent_chunks = [(t, resample_pcm16(d, r or target, target)) for t, d, r in agent]
        wav = build_stereo_session_wav(user_chunks, agent_chunks, target)
        if wav:
            Path(self._path).write_bytes(wav)


class _RecordingInputTransport(LocalAudioInputTransport):
    def __init__(self, py_audio, params, recorder: ConversationRecorder) -> None:
        super().__init__(py_audio, params)
        self._recorder = recorder

    def _audio_in_callback(self, in_data, frame_count, time_info, status):  # noqa: ANN001
        # User mic isn't truncated; record straight off the capture callback.
        self._recorder.record_user(in_data, self._sample_rate)
        return super()._audio_in_callback(in_data, frame_count, time_info, status)


class _RecordingOutputTransport(LocalAudioOutputTransport):
    def __init__(self, py_audio, params, recorder: ConversationRecorder) -> None:
        super().__init__(py_audio, params)
        self._recorder = recorder

    async def write_audio_frame(self, frame) -> bool:  # noqa: ANN001
        # Reached only for audio the clock queue actually plays — post barge-in
        # truncation — so this is exactly what was heard.
        ok = await super().write_audio_frame(frame)
        if ok:
            rate = getattr(frame, "sample_rate", 0) or self._sample_rate
            self._recorder.record_agent(frame.audio, rate)
        return ok


class RecordingLocalAudioTransport(LocalAudioTransport):
    """`LocalAudioTransport` whose input/output tap a `ConversationRecorder`."""

    def __init__(
        self, params: LocalAudioTransportParams, recorder: ConversationRecorder
    ) -> None:
        super().__init__(params)
        self._recorder = recorder

    def input(self):
        if not self._input:
            self._input = _RecordingInputTransport(
                self._pyaudio, self._params, self._recorder
            )
        return self._input

    def output(self):
        if not self._output:
            self._output = _RecordingOutputTransport(
                self._pyaudio, self._params, self._recorder
            )
        return self._output

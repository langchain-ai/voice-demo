"""Local mic + speaker I/O — the voice-demo "console transport".

`MicStream` and `SpeakerStream` are the local-machine implementation of the
`AudioInput` / `AudioOutput` transport protocols defined here. They are the
direct analog of the framework-owned transports the other backends use
(LiveKit's console mode, Pipecat's `LocalAudioTransport`): they move PCM16 audio
between the default mic/speaker and the agent, and nothing more.

Because the agents depend only on the *protocols*, a different frontend (a web
app, a phone call, a websocket bridge) can drive the exact same OpenAI Realtime
or ADK Live agent by supplying its own `AudioInput`/`AudioOutput` — without
touching the agent's event loop, tracing, or tool logic.

Both streams are PCM16 mono, backed by `sounddevice` RawStreams. PortAudio runs
the callbacks on its own thread; we shuttle bytes to/from the asyncio loop
through queues so nothing blocks the event loop. Realtime is symmetric at
24 kHz; ADK Live wants 16 kHz in, 24 kHz out — so the streams take an explicit
sample rate at construction.
"""

from __future__ import annotations

import asyncio
import queue
import threading
from collections.abc import AsyncIterator, Callable
from typing import Protocol, runtime_checkable

import numpy as np
import sounddevice as sd
from scipy import signal as scipy_signal

CHANNELS = 1
DTYPE = "int16"
CHUNK_MS = 20


def resample_pcm16(data: bytes, src_rate: int, dst_rate: int) -> bytes:
    """Resample raw PCM16 mono audio between sample rates.

    Used wherever a backend's device rate differs from what its service wants
    (e.g. 24 kHz mic → 16 kHz for ADK Live) or when mixing chunks of different
    rates into one recording (the Pipecat conversation WAV).
    """
    if not data or not src_rate or src_rate == dst_rate:
        return data
    samples = np.frombuffer(data, dtype=np.int16)
    if samples.size == 0:
        return b""
    n_out = int(round(samples.size * dst_rate / src_rate))
    return scipy_signal.resample(samples, n_out).astype(np.int16).tobytes()


# ---------------------------------------------------------------------------
# Transport protocols — the contract an agent needs from its audio frontend
# ---------------------------------------------------------------------------

@runtime_checkable
class AudioInput(Protocol):
    """A source of PCM16 mono audio frames (the agent's "ears")."""

    sample_rate: int

    def start(self) -> None: ...

    def frames(self) -> AsyncIterator[bytes]:
        """Async-iterate PCM16 mono frames until the input is stopped."""
        ...

    def stop(self) -> None: ...


@runtime_checkable
class AudioOutput(Protocol):
    """A sink for PCM16 mono audio frames (the agent's "voice")."""

    sample_rate: int

    def start(self) -> None: ...

    def write(self, data: bytes) -> None: ...

    def buffered_bytes(self) -> int:
        """Bytes still queued/unplayed — used to reason about barge-in timing."""
        ...

    def clear(self) -> None:
        """Drop any buffered audio. Used on barge-in / interruption."""
        ...

    def set_played_callback(self, callback: Callable[[bytes], None] | None) -> None:
        """Register a sink called with the PCM16 bytes *actually played*.

        Invoked from the audio device thread as audio is consumed by the
        speaker, so it reflects what the listener heard — audio dropped by
        `clear()` on barge-in is never played and so never reaches this
        callback. This is the correct tap point for "what was heard" recording,
        as opposed to recording at `write()` (which is what was *generated*).
        """
        ...

    def stop(self) -> None: ...


def _bytes_per_chunk(sample_rate: int) -> int:
    # 20 ms frame, int16 mono = sample_rate * 0.02 * 2 bytes
    return int(sample_rate * CHUNK_MS / 1000) * 2


class MicStream:
    """Async-iterable source of PCM16 mono frames from the default input device."""

    def __init__(self, sample_rate: int = 24_000) -> None:
        self.sample_rate = sample_rate
        self._frame_bytes = _bytes_per_chunk(sample_rate)
        self._loop: asyncio.AbstractEventLoop | None = None
        self._queue: asyncio.Queue[bytes] = asyncio.Queue(maxsize=100)
        self._stream: sd.RawInputStream | None = None

    def start(self) -> None:
        self._loop = asyncio.get_running_loop()

        def callback(indata, frames, time_info, status):  # noqa: ANN001
            if self._loop is None:
                return
            data = bytes(indata)
            self._loop.call_soon_threadsafe(self._safe_put, data)

        self._stream = sd.RawInputStream(
            samplerate=self.sample_rate,
            channels=CHANNELS,
            dtype=DTYPE,
            blocksize=self._frame_bytes // 2,  # frames, not bytes
            callback=callback,
        )
        self._stream.start()

    def _safe_put(self, data: bytes) -> None:
        try:
            self._queue.put_nowait(data)
        except asyncio.QueueFull:
            # Realtime audio prefers freshness over completeness.
            pass

    async def frames(self):
        while True:
            yield await self._queue.get()

    def stop(self) -> None:
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None


class SpeakerStream:
    """PCM16 mono sink that plays through the default output device.

    Backed by a `queue.Queue` because PortAudio's callback runs on its own
    thread. `.clear()` drops any buffered audio — used on barge-in.

    For accurate "what was heard" recording, register a played-callback via
    `set_played_callback`: it fires from the device thread with exactly the
    PCM16 bytes pulled into the speaker, so audio that `clear()` drops on
    barge-in is never reported (it was never heard).
    """

    def __init__(self, sample_rate: int = 24_000) -> None:
        self.sample_rate = sample_rate
        self._frame_bytes = _bytes_per_chunk(sample_rate)
        self._queue: queue.Queue[bytes] = queue.Queue()
        self._buffer = bytearray()
        self._stream: sd.RawOutputStream | None = None
        self._lock = threading.Lock()
        self._on_played: Callable[[bytes], None] | None = None

    def set_played_callback(self, callback: Callable[[bytes], None] | None) -> None:
        self._on_played = callback

    def start(self) -> None:
        def callback(outdata, frames, time_info, status):  # noqa: ANN001
            needed = len(outdata)
            with self._lock:
                while len(self._buffer) < needed:
                    try:
                        chunk = self._queue.get_nowait()
                    except queue.Empty:
                        break
                    self._buffer.extend(chunk)
                if len(self._buffer) >= needed:
                    played = bytes(self._buffer[:needed])
                    outdata[:] = played
                    del self._buffer[:needed]
                else:
                    # Underrun: only the real (non-silence) prefix was heard.
                    played = bytes(self._buffer)
                    outdata[:] = played + b"\x00" * (needed - len(played))
                    self._buffer.clear()
            # Report what was actually played, outside the lock. Silence padding
            # on underrun is intentionally excluded — it's a gap, not content.
            if self._on_played is not None and played:
                self._on_played(played)

        self._stream = sd.RawOutputStream(
            samplerate=self.sample_rate,
            channels=CHANNELS,
            dtype=DTYPE,
            blocksize=self._frame_bytes // 2,
            callback=callback,
        )
        self._stream.start()

    def write(self, data: bytes) -> None:
        self._queue.put(data)

    def buffered_bytes(self) -> int:
        with self._lock:
            return len(self._buffer) + sum(
                len(self._queue.queue[i]) for i in range(self._queue.qsize())
            )

    def clear(self) -> None:
        """Drop any buffered audio. Used on barge-in / interruption."""
        with self._lock:
            self._buffer.clear()
        while True:
            try:
                self._queue.get_nowait()
            except queue.Empty:
                return

    def stop(self) -> None:
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None

"""Local mic + speaker I/O for the voice-demo backends.

Two streams, both PCM16 mono, backed by `sounddevice` RawStreams. PortAudio
runs the callbacks on its own thread; we shuttle bytes to/from the asyncio
loop through queues so nothing blocks the event loop.

Both OpenAI Realtime and ADK Live want PCM16. Realtime is symmetric at 24 kHz;
ADK Live wants 16 kHz in, 24 kHz out — so the streams take an explicit sample
rate at construction. LiveKit owns its own audio path and does not use this
module (its console mode wires the mic+speaker through the Agents SDK directly).
"""

from __future__ import annotations

import asyncio
import queue
import threading

import sounddevice as sd

CHANNELS = 1
DTYPE = "int16"
CHUNK_MS = 20


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
    """

    def __init__(self, sample_rate: int = 24_000) -> None:
        self.sample_rate = sample_rate
        self._frame_bytes = _bytes_per_chunk(sample_rate)
        self._queue: queue.Queue[bytes] = queue.Queue()
        self._buffer = bytearray()
        self._stream: sd.RawOutputStream | None = None
        self._lock = threading.Lock()

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
                    outdata[:] = bytes(self._buffer[:needed])
                    del self._buffer[:needed]
                else:
                    out = bytes(self._buffer) + b"\x00" * (needed - len(self._buffer))
                    outdata[:] = out
                    self._buffer.clear()

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

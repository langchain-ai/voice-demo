"""ElevenLabs ``AudioInterface`` backed by sounddevice.

The SDK ships a ``DefaultAudioInterface`` built on pyaudio, which needs a
compiled portaudio binding. Every other backend in this repo already runs on
sounddevice (see ``voice_demo.audio``), so this keeps the demo to one audio
stack and one system dependency.

The contract is fixed by the SDK: PCM16 mono at 16 kHz in both directions,
``output`` must return immediately, and ``interrupt`` drops whatever the agent
has queued so barge-in stops playback at once.
"""

from __future__ import annotations

import threading
from collections.abc import Callable

import sounddevice as sd
from elevenlabs.conversational_ai.conversation import AudioInterface

SAMPLE_RATE = 16_000
# 250 ms — the chunk size the SDK recommends for the input callback.
CHUNK_SAMPLES = 4_000
BYTES_PER_SAMPLE = 2


class SoundDeviceAudioInterface(AudioInterface):
    """Local mic and speaker for an ElevenLabs conversation."""

    def __init__(self) -> None:
        self._input_callback: Callable[[bytes], None] | None = None
        self._in_stream: sd.RawInputStream | None = None
        self._out_stream: sd.RawOutputStream | None = None
        # Playback is buffered here and drained by PortAudio's own thread, so
        # `output` never blocks the SDK's websocket loop. Guarded because both
        # threads touch it.
        self._playback = bytearray()
        self._lock = threading.Lock()

    def start(self, input_callback: Callable[[bytes], None]) -> None:
        self._input_callback = input_callback
        self._in_stream = sd.RawInputStream(
            samplerate=SAMPLE_RATE,
            channels=1,
            dtype="int16",
            blocksize=CHUNK_SAMPLES,
            callback=self._on_mic,
        )
        self._out_stream = sd.RawOutputStream(
            samplerate=SAMPLE_RATE,
            channels=1,
            dtype="int16",
            blocksize=CHUNK_SAMPLES,
            callback=self._on_speaker,
        )
        self._in_stream.start()
        self._out_stream.start()

    def stop(self) -> None:
        # Cleared first: the SDK forbids calling the input callback after stop.
        self._input_callback = None
        for stream in (self._in_stream, self._out_stream):
            if stream is not None:
                stream.stop()
                stream.close()
        self._in_stream = self._out_stream = None
        self.interrupt()

    def output(self, audio: bytes) -> None:
        with self._lock:
            self._playback.extend(audio)

    def interrupt(self) -> None:
        with self._lock:
            self._playback.clear()

    # -- PortAudio's thread ---------------------------------------------------

    def _on_mic(self, indata, _frames, _time, _status) -> None:
        callback = self._input_callback
        if callback is not None:
            callback(bytes(indata))

    def _on_speaker(self, outdata, frames, _time, _status) -> None:
        wanted = frames * BYTES_PER_SAMPLE
        with self._lock:
            chunk = bytes(self._playback[:wanted])
            del self._playback[:wanted]
        outdata[: len(chunk)] = chunk
        if len(chunk) < wanted:  # underrun — pad with silence rather than glitch
            outdata[len(chunk) :] = b"\x00" * (wanted - len(chunk))

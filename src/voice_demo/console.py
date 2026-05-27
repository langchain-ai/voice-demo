"""Shared terminal UI: single-line mic level meter + agent state.

Used by the OpenAI Realtime and Google ADK backends so their consoles look
and behave identically. The LiveKit backend has its own console UI from the
`livekit-agents` CLI and doesn't use this module.
"""

from __future__ import annotations

import array
import math
import sys
import time


def frame_level(frame: bytes) -> float:
    """0..1 level for a PCM16 mono frame, mapped from roughly -60..0 dBFS."""
    samples = array.array("h")
    samples.frombytes(frame)
    n = len(samples)
    if not n:
        return 0.0
    mean_sq = sum(s * s for s in samples) / n
    if mean_sq <= 0:
        return 0.0
    rms = math.sqrt(mean_sq)
    db = 20.0 * math.log10(rms / 32767.0)
    return max(0.0, min(1.0, (db + 60.0) / 60.0))


class ConsoleStatus:
    """Single-line stderr status: mic meter + agent state.

    `log()` clears the status line, prints a message, then redraws — so
    transcript lines and the meter coexist instead of overwriting each other.
    Falls back to plain line-based output when stderr isn't a TTY.
    """

    _BARS = 20

    def __init__(self) -> None:
        self._enabled = sys.stderr.isatty()
        self._state = "starting"
        self._level = 0.0
        self._last_render = 0.0
        self._line_len = 0

    def set_state(self, state: str) -> None:
        if state == self._state:
            return
        self._state = state
        if not self._enabled:
            sys.stderr.write(f"[{state}]\n")
            sys.stderr.flush()
            return
        self._render()

    def update_level(self, level: float) -> None:
        self._level = max(0.0, min(1.0, level))
        if not self._enabled:
            return
        now = time.monotonic()
        if now - self._last_render < 0.1:
            return
        self._render()

    def log(self, msg: str) -> None:
        if not self._enabled:
            sys.stderr.write(msg + "\n")
            sys.stderr.flush()
            return
        sys.stderr.write("\r" + " " * self._line_len + "\r")
        sys.stderr.write(msg + "\n")
        sys.stderr.flush()
        self._render()

    def finish(self) -> None:
        if self._enabled:
            sys.stderr.write("\n")
            sys.stderr.flush()

    def _render(self) -> None:
        self._last_render = time.monotonic()
        filled = int(self._level * self._BARS)
        meter = "█" * filled + "░" * (self._BARS - filled)
        line = f"mic {meter} [{self._state}]"
        sys.stderr.write("\r" + line.ljust(self._line_len))
        sys.stderr.flush()
        self._line_len = max(self._line_len, len(line))

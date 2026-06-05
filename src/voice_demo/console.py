"""Terminal frontend UI: single-line mic level meter + agent state.

`ConsoleStatus` is the local-terminal implementation of the `StatusUI` protocol
— the small observer interface the OpenAI Realtime and ADK Live agents use to
surface what they're doing (state changes, transcript lines, mic level). It's
purely a *frontend* concern: the agents depend on the protocol, not this class,
so a non-terminal frontend can pass its own `StatusUI` (or the no-op `NullUI`)
and render those events however it likes — or ignore them entirely.

This is the piece the raw OpenAI/ADK SDKs don't ship and we built ourselves; the
LiveKit and Pipecat backends get their equivalent console chatter from their own
frameworks, which is why they don't use this module.
"""

from __future__ import annotations

import array
import math
import sys
import time
from typing import Protocol, runtime_checkable


@runtime_checkable
class StatusUI(Protocol):
    """What an agent reports to its frontend as a conversation progresses.

    All methods are best-effort and side-effect-only — an agent never reads
    anything back, so a frontend is free to implement these as no-ops, log
    lines, structured events, or rich UI updates.
    """

    def set_state(self, state: str) -> None:
        """The agent changed phase (e.g. listening, thinking, speaking)."""
        ...

    def update_level(self, level: float) -> None:
        """Latest input level, 0..1 — for a mic meter."""
        ...

    def log(self, msg: str) -> None:
        """A discrete line worth surfacing (a transcript, a notice)."""
        ...

    def finish(self) -> None:
        """The session ended; flush/close any UI state."""
        ...


class NullUI:
    """A `StatusUI` that does nothing — the default for headless frontends."""

    def set_state(self, state: str) -> None: ...

    def update_level(self, level: float) -> None: ...

    def log(self, msg: str) -> None: ...

    def finish(self) -> None: ...


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

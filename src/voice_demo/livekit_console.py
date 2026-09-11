"""Shared local-console launcher for the LiveKit demo backends."""

from __future__ import annotations

import os
import sys
from typing import Any


def run_livekit_console(server: Any) -> None:
    """Run LiveKit's audio console with optional explicit device selection.

    ``LIVEKIT_INPUT_DEVICE`` and ``LIVEKIT_OUTPUT_DEVICE`` accept the same
    numeric IDs or name substrings as LiveKit's console flags.
    """
    from livekit import agents

    argv = [sys.argv[0], "console", "--record"]
    for env_name, flag in (
        ("LIVEKIT_INPUT_DEVICE", "--input-device"),
        ("LIVEKIT_OUTPUT_DEVICE", "--output-device"),
    ):
        if device := os.environ.get(env_name, "").strip():
            argv.extend((flag, device))

    sys.argv = argv
    agents.cli.run_app(server)

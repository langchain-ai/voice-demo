"""Active thread_id + recording path for the current LiveKit conversation.

`processor.py` reads these when building span attributes:
  - `active_thread_id` (ContextVar) propagates through asyncio tasks
  - `last_set_thread_id()` is a fallback for callbacks that escape the task tree
  - `get_audio_file_path()` lets the processor attach the call recording to the
    root span via `langsmith.attachments`

The agent calls `set_active_thread_id(ctx.job.id)` and
`set_audio_file_path(ctx.session_directory / "audio.ogg")` once per session.
"""

from __future__ import annotations

import contextvars
from pathlib import Path

active_thread_id: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "voice_demo_livekit_active_thread_id", default=None
)

# Single-worker assumption: one job per process at a time. If we ever run
# multiple concurrent jobs per worker, this needs a {pid: job_id} map.
_last_job_id: str | None = None
_audio_file_path: Path | None = None


def set_active_thread_id(job_id: str) -> None:
    global _last_job_id
    active_thread_id.set(job_id)
    _last_job_id = job_id


def last_set_thread_id() -> str | None:
    return _last_job_id


def set_audio_file_path(path: Path) -> None:
    global _audio_file_path
    _audio_file_path = path


def get_audio_file_path() -> Path | None:
    return _audio_file_path

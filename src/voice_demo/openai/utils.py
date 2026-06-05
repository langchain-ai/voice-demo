"""Helpers for the OpenAI Realtime agent: tracing glue and tool dispatch."""

from __future__ import annotations

import json

from .tools import lookup_weather


def is_inbound(event_type: str) -> bool:
    """Direction of an event relative to the model.

    Inbound = something the user sent toward the model (their speech buffer,
    their transcription) → goes in span `inputs`. Everything else (`response.*`,
    `error`, `session.*`) is the model/server talking back → span `outputs`.
    """
    return event_type.startswith("input_audio_buffer") or "input_audio_transcription" in event_type


async def execute_tool(name: str, arguments: str) -> dict:
    """Run one model-requested tool call and return its JSON-able result."""
    if name != "lookup_weather":
        return {"error": f"unknown tool: {name}"}
    try:
        args = json.loads(arguments or "{}")
    except json.JSONDecodeError:
        args = {}
    city = (args.get("city") or "").strip()
    if not city:
        return {"error": "missing city"}
    return await lookup_weather(city)

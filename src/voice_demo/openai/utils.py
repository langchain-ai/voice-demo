"""Application helpers for the OpenAI Realtime agent: tool dispatch.

Tracing glue lives in `tracing.py`; this module is the application side.
"""

from __future__ import annotations

import json

from .tools import lookup_weather


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

"""Allowlisted application tool dispatch for the Deepgram Voice Agent backend.

Tracing lives in the LangSmith ``deepgram_voice`` integration that wraps the
connection (see ``agent.py``); this module is the application side.
"""

from __future__ import annotations

import json

from ..weather import fetch_weather

WEATHER_FUNCTION = {
    "name": "lookup_weather",
    "description": "Get the current weather for one city. Call once per city.",
    "parameters": {
        "type": "object",
        "properties": {
            "city": {
                "type": "string",
                "description": "City name, such as Paris or Tokyo.",
            }
        },
        "required": ["city"],
    },
}


async def execute_tool(name: str, arguments: str) -> dict:
    """Run one agent-requested tool call and return its JSON-able result.

    Deepgram delivers ``arguments`` as a JSON string; malformed or unknown
    calls return safe error payloads instead of raising.
    """
    if name != "lookup_weather":
        return {"error": f"unknown tool: {name}"}
    try:
        args = json.loads(arguments or "{}")
    except json.JSONDecodeError:
        args = {}
    if not isinstance(args, dict):
        return {"error": "tool arguments must be an object"}
    city = args.get("city")
    if not isinstance(city, str) or not city.strip():
        return {"error": "missing city"}
    return await fetch_weather(city.strip())

"""Validated application tool dispatch for the GPT-Live weather backend."""

from __future__ import annotations

import json
from typing import Any

from ..weather import fetch_weather

MAX_ARGUMENT_CHARS = 8_192
MAX_CITY_CHARS = 120


async def execute_tool(name: str, arguments: str) -> dict[str, Any]:
    """Validate and run one delegated function call.

    Model output is untrusted input.  Only the allowlisted weather function is
    executable, its arguments must be a small JSON object, and the city value is
    bounded before it reaches the fixed-host Open-Meteo client.
    """
    if name != "lookup_weather":
        return {"error": "unknown_tool", "tool": name[:80]}
    if not isinstance(arguments, str) or len(arguments) > MAX_ARGUMENT_CHARS:
        return {"error": "invalid_arguments"}

    try:
        parsed = json.loads(arguments or "{}")
    except json.JSONDecodeError:
        return {"error": "invalid_arguments"}
    if not isinstance(parsed, dict) or set(parsed) - {"city"}:
        return {"error": "invalid_arguments"}

    city = parsed.get("city")
    if not isinstance(city, str):
        return {"error": "missing_city"}
    city = city.strip()
    if not city:
        return {"error": "missing_city"}
    if len(city) > MAX_CITY_CHARS:
        return {"error": "city_too_long"}

    return await fetch_weather(city)

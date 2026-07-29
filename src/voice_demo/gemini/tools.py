"""Allowlisted application tool dispatch for the raw Gemini Live backend."""

from __future__ import annotations

from typing import Any

from langsmith import traceable

from ..weather import fetch_weather


@traceable(run_type="tool", name="lookup_weather")
async def lookup_weather(city: str) -> dict:
    """Get the current weather for one city from the shared Open-Meteo client."""
    return await fetch_weather(city)


async def execute_tool(name: str | None, arguments: Any) -> dict:
    """Execute one known tool; malformed or unknown calls return safe errors."""
    if name != "lookup_weather":
        return {"error": f"unknown tool: {name or '<missing>'}"}
    if not isinstance(arguments, dict):
        return {"error": "tool arguments must be an object"}
    city = arguments.get("city")
    if not isinstance(city, str) or not city.strip():
        return {"error": "missing city"}
    return await lookup_weather(city.strip())

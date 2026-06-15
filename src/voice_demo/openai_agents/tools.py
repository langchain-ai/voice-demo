"""The weather tool, as an Agents-SDK `@function_tool`.

Unlike the raw-WebSocket backend — where we hand the model a JSON schema and
dispatch the call ourselves in `execute_tool()` — the Agents SDK derives the
schema from this function's signature/docstring and runs the call for us inside
its own turn loop. We don't trace the tool with `@traceable` here: the SDK
surfaces every call as `tool_start` / `tool_end` session events, and the agent
loop spans those directly (see `agent.py`).

The actual HTTP work lives in the shared `voice_demo.weather` module, so all
backends share one Open-Meteo lookup.
"""

from __future__ import annotations

from agents import function_tool

from ..weather import fetch_weather


@function_tool
async def lookup_weather(city: str) -> dict:
    """Get the current weather for a single city. Call once per city for multi-city questions.

    Args:
        city: City name, e.g. 'San Francisco' or 'Tokyo'.

    Returns one of:
      {"city": str, "country": str, "weather": {...}}    on success
      {"city": str, "error": "not_found" | "http_error"} on failure
    """
    return await fetch_weather(city)

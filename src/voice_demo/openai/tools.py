"""Open-Meteo weather lookup, traced for the OpenAI backend.

`@traceable` makes each lookup a child run under the active event span in
LangSmith. The actual HTTP work lives in the shared `voice_demo.weather`
module so the pipecat-with-langgraph backend can reuse it as a LangGraph tool.
"""

from __future__ import annotations

from langsmith import traceable

from ..weather import fetch_weather


@traceable(run_type="tool", name="lookup_weather")
async def lookup_weather(city: str) -> dict:
    """Geocode a city name and return current weather.

    Returns one of:
      {"city": str, "country": str, "weather": {...}}    on success
      {"city": str, "error": "not_found" | "http_error"} on failure
    """
    return await fetch_weather(city)

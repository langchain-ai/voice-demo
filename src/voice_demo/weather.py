"""Open-Meteo weather lookup. Free, public, no API key.

A pure async fetch shared across backends. Each backend wraps it in whatever
tracing its stack uses: the OpenAI backend wraps it with LangSmith's
`@traceable` (see `openai/tools.py`); the pipecat-with-langgraph backend
exposes it as a LangGraph tool whose execution is traced through OTel.

The endpoints are fixed constants — the only caller-supplied value is the city
name, passed as a query parameter — so there's no user-controlled base URL.
"""

from __future__ import annotations

import httpx

GEOCODE_URL = "https://geocoding-api.open-meteo.com/v1/search"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"

_TIMEOUT = httpx.Timeout(10.0, connect=5.0)


async def fetch_weather(city: str) -> dict:
    """Geocode a city name and return current weather.

    Returns one of:
      {"city": str, "country": str, "weather": {...}}    on success
      {"city": str, "error": "not_found" | "http_error"} on failure
    """
    async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
        try:
            geo = await client.get(GEOCODE_URL, params={"name": city, "count": 1})
            geo.raise_for_status()
            results = geo.json().get("results") or []
            if not results:
                return {"city": city, "error": "not_found"}
            loc = results[0]

            wx = await client.get(
                FORECAST_URL,
                params={
                    "latitude": loc["latitude"],
                    "longitude": loc["longitude"],
                    "current_weather": True,
                    "temperature_unit": "fahrenheit",
                    "wind_speed_unit": "mph",
                },
            )
            wx.raise_for_status()
            current = wx.json().get("current_weather") or {}
            return {
                "city": loc.get("name") or city,
                "country": loc.get("country") or "",
                "weather": current,
            }
        except httpx.HTTPError:
            return {"city": city, "error": "http_error"}

"""OpenAI Realtime via the OpenAI **Agents SDK** (`agents.realtime`).

A second take on the OpenAI Realtime backend: same model, same weather tool,
same console frontend — but driven through the Agents SDK's `RealtimeAgent` /
`RealtimeRunner` / `RealtimeSession` instead of the raw `client.realtime.connect()`
WebSocket loop in `voice_demo.openai`. See `agent.py` for how the trace shape
differs.
"""

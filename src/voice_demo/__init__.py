"""voice-demo: three voice-agent backends sharing one console interface.

Run `voice-demo --backend openai|adk|livekit` to launch a console session
against the chosen backend. Each backend is traced to LangSmith using the
optimal path for its framework (OTEL for LiveKit + ADK, SDK RunTree for OpenAI
Realtime). See README.md for the rationale.
"""

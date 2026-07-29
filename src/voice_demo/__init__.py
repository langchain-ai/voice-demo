"""voice-demo: several voice-agent backends sharing one console interface.

Run `voice-demo --backend <name>` to launch a console session against the chosen
backend: openai, openai-agents, gemini, adk, livekit, livekit-with-openai-realtime,
livekit-with-gemini-live, pipecat, or pipecat-with-langgraph. Each backend is
traced to LangSmith using the optimal path for its framework (OTEL for LiveKit,
Pipecat, and ADK; SDK RunTree for OpenAI Realtime and raw Gemini Live). See README.md for the
rationale.
"""

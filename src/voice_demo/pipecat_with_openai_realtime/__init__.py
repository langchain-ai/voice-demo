"""Pipecat voice-agent backend whose LLM stage is OpenAI Realtime (S2S).

This is the `pipecat` cascade backend with its STT → LLM → TTS trio collapsed
into a single speech-to-speech model: Pipecat's `OpenAIRealtimeLLMService`
ingests the user's audio and emits the bot's audio directly, so there is no
separate STT or TTS stage and no local VAD — OpenAI's server-side turn
detection drives the turns. Everything else (the OTel → LangSmith tracing
wiring, the native Pipecat function tool, the recorder) is the same as the
cascade; see `pipecat/agent.py` for the fully-commented version.

See `pipecat_with_langgraph/` for the variant that keeps the STT/TTS cascade but
swaps the *text* LLM stage for an in-process LangGraph graph.
"""

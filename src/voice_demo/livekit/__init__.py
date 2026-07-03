"""LiveKit voice-agent backend (STT → LLM → TTS cascade), traced into LangSmith.

LiveKit's Agents SDK emits OTel spans natively across its STT, LLM, TTS,
turn-detection, and EOU pipeline; `configure_livekit` (the LangSmith LiveKit
integration) translates them into the shape LangSmith ingests. See
`livekit_with_openai_realtime/` and `livekit_with_gemini_live/` for the
speech-to-speech variants.
"""

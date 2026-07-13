"""Pipecat voice-agent backend whose LLM stage is Gemini Live (S2S).

This is the `pipecat` cascade backend with its STT → LLM → TTS trio collapsed
into a single speech-to-speech model: Pipecat's `GeminiLiveLLMService` ingests
the user's audio and emits the bot's audio directly. It is the Gemini twin of
`pipecat_with_openai_realtime/`, with one structural difference: Gemini Live
doesn't emit user turn-start/-end frames from its server-side VAD, so this
backend drives turns with a *local* Silero VAD (and disables Gemini's server
VAD) to keep the LangSmith `turn` spans and turn-based barge-in that the tracing
demo depends on. See `pipecat_with_gemini_live/agent.py` for the full rationale
and `pipecat/agent.py` for the shared tracing/recording commentary.
"""

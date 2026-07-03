"""LiveKit voice agent using Gemini Live (speech-to-speech), traced.

A variant of the `livekit` backend: instead of the STT → LLM → TTS cascade, the
session's LLM slot is filled with Google's Gemini Live model, which ingests audio
and emits audio directly — collapsing the whole pipeline into one model. LiveKit
still emits the same OTel span vocabulary (turns, tools, realtime metrics), so
`configure_livekit` handles it with no changes.
"""

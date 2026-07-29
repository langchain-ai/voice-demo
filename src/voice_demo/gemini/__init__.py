"""Gemini Live voice agent over the raw ``google-genai`` WebSocket session.

This is the framework-free Gemini twin of :mod:`voice_demo.openai`: the app
owns the provider event loop, audio playback, barge-in handling, and tool
dispatch, while LangSmith's ``wrap_gemini_live`` integration owns tracing.
"""

"""ElevenLabs Agents, traced to LangSmith from its post-call webhooks.

The only backend here whose agent runs entirely on the vendor's servers, so its
tracing is post-call rather than live: :mod:`voice_demo.elevenlabs.agent` holds
the conversation, and :mod:`voice_demo.elevenlabs.webhook` receives the OTLP
trace and audio ElevenLabs sends afterwards, verifies them, and forwards them
through ``langsmith.integrations.elevenlabs``.
"""

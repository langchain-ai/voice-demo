"""Pipecat voice-agent backend, traced into LangSmith via OpenTelemetry.

Pipecat runs the full STT → LLM → TTS pipeline in-process and emits its own
OTel spans (`conversation`, `turn`, `stt`, `llm`, `tts`), so — like the LiveKit
backend — this pairs an `agent.py` (the pipeline) with a `processor.py` (an OTel
`SpanProcessor` that translates Pipecat's spans into the `gen_ai.*` /
`langsmith.*` shape LangSmith ingests).
"""

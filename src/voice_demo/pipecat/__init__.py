"""Minimal Pipecat voice-agent backend, traced into LangSmith via OpenTelemetry.

Pipecat runs the full STT → LLM → TTS pipeline in-process and emits its own OTel
spans (`conversation`, `turn`, `stt`, `llm`, `tts`), which `configure_pipecat`
(the LangSmith Pipecat integration) rewrites into the `gen_ai.*` / `langsmith.*`
shape LangSmith ingests. The LLM stage is Pipecat's stock `OpenAILLMService`
with a native function tool.

See `pipecat_with_langgraph/` for the variant that runs an in-process LangGraph
graph as the LLM stage instead.
"""

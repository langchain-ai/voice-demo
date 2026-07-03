"""Pipecat voice agent whose LLM stage is an in-process LangGraph "brain".

Like the plain `pipecat` backend, Pipecat runs the full STT → LLM → TTS pipeline
in-process and emits its own OTel spans (`conversation`, `turn`, `stt`, `llm`,
`tts`), rewritten into LangSmith's shape by `configure_pipecat` (the LangSmith
Pipecat integration). What differs here is the `llm` stage: instead of Pipecat's
stock `OpenAILLMService`, it runs a LangChain `create_agent` graph (`graph.py`)
via `LangGraphLLMService` (`langgraph_llm_service.py`), so every graph node —
model and tools — nests as a subspan of Pipecat's `llm` span in one trace.

See `pipecat/` (no LangGraph) for the minimal stock-service version.
"""

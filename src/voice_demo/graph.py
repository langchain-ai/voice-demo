"""In-process LangChain agent "brain" — used by the Pipecat backend.

A prebuilt `create_agent` (LangChain v1) ReAct agent over a single `lookup_weather`
tool. It compiles to a standard LangGraph (Pregel) graph:

    START → model  ⇄  tools        (ReAct loop; lookup_weather)
              │
              └────────────────→ END   (no more tool calls)

Only the final assistant message is spoken; the tool-deciding turn and tool
execution are traced but never voiced. `LangGraphLLMService` runs the graph
inside Pipecat's `llm` span and streams only the model node's text to TTS,
relying on `LANGSMITH_TRACING_MODE=otel` so the agent's spans nest under the
framework's LLM span in one trace.

The agent is **stateless** (no checkpointer): the framework's chat context is the
single source of truth and is passed in fresh each turn, which keeps barge-in
truncation correct (an interrupted, never-heard turn never lingers in agent
memory).

(The LiveKit backend previously drove this same graph through an LLMAdapter;
it now uses LiveKit's native agent framework with a `@function_tool` instead.)
"""

from __future__ import annotations

from langchain.agents import create_agent
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI

from .prompts import GREETING, SYSTEM_PROMPT
from .weather import fetch_weather

__all__ = ["AGENT_MODEL", "GREETING", "SYSTEM_PROMPT", "build_graph", "lookup_weather"]

AGENT_MODEL = "gpt-4o-mini"


@tool
async def lookup_weather(city: str) -> dict:
    """Get the current weather for a single city. Call once per city."""
    return await fetch_weather(city)


def build_graph(system_prompt: str = SYSTEM_PROMPT):
    """Compile the weather-agent graph.

    `system_prompt` is the spoken assistant's instructions, passed to the
    prebuilt agent as its `prompt`. The returned object is a compiled LangGraph
    supporting `.astream` / `.astream_events` with `{"messages": [...]}` input —
    a drop-in for both backends.
    """
    return create_agent(
        ChatOpenAI(model=AGENT_MODEL, temperature=0.3),
        tools=[lookup_weather],
        system_prompt=system_prompt,
    )

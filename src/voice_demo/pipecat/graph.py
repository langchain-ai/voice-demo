"""In-process LangGraph "brain" for the Pipecat backend.

A small, custom graph so you can watch a *non-spoken* step (a content guardrail)
trace as a subspan under Pipecat's `llm` span:

    START → guardrail ──blocked?──yes──→ refuse ──────────────→ END
               │
               no
               ▼
            call_model  ⇄  tools         (ReAct loop; lookup_weather)
               │
               └──────────────────────→ END   (no more tool calls)

Only the final assistant message is spoken (see `LangGraphLLMService`); the
guardrail verdict, the tool-deciding turn, tool execution, and routing are all
traced but never voiced.

The graph is **stateless** (no checkpointer): Pipecat's `LLMContext` is the
single source of truth and is passed in fresh each turn, which keeps barge-in
truncation correct (an interrupted, never-heard turn never lingers in graph
memory) — the same design the LiveKit backend uses.
"""

from __future__ import annotations

from typing import Annotated, TypedDict

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition

from ..openai.guardrail import REFUSAL_INSTRUCTIONS, check_guardrail
from ..weather import fetch_weather

AGENT_MODEL = "gpt-4o-mini"


@tool
async def lookup_weather(city: str) -> dict:
    """Get the current weather for a single city. Call once per city."""
    return await fetch_weather(city)


class GraphState(TypedDict):
    """Conversation messages plus the guardrail verdict for this turn."""

    messages: Annotated[list, add_messages]
    blocked: bool
    reason: str


def _latest_user_text(messages: list) -> str:
    for m in reversed(messages):
        if isinstance(m, HumanMessage):
            return m.content if isinstance(m.content, str) else str(m.content)
    return ""


def build_graph(system_prompt: str):
    """Compile the guardrail + weather-agent graph. `system_prompt` is the
    spoken assistant's instructions (the guardrail/refusal have their own)."""
    model_with_tools = ChatOpenAI(model=AGENT_MODEL, temperature=0.3).bind_tools(
        [lookup_weather]
    )
    refuse_model = ChatOpenAI(model=AGENT_MODEL, temperature=0.7)

    async def guardrail(state: GraphState) -> dict:
        # A structured-output LLM call — traced as a subspan, never spoken.
        verdict = await check_guardrail(_latest_user_text(state["messages"]))
        return {"blocked": verdict.blocked, "reason": verdict.reason}

    def route(state: GraphState) -> str:
        return "refuse" if state.get("blocked") else "call_model"

    async def call_model(state: GraphState) -> dict:
        msgs = [SystemMessage(content=system_prompt), *state["messages"]]
        resp: AIMessage = await model_with_tools.ainvoke(msgs)
        return {"messages": [resp]}

    async def refuse(state: GraphState) -> dict:
        resp = await refuse_model.ainvoke(
            [
                SystemMessage(content=REFUSAL_INSTRUCTIONS),
                HumanMessage(content=_latest_user_text(state["messages"])),
            ]
        )
        return {"messages": [resp]}

    builder = StateGraph(GraphState)
    builder.add_node("guardrail", guardrail)
    builder.add_node("call_model", call_model)
    builder.add_node("tools", ToolNode([lookup_weather]))
    builder.add_node("refuse", refuse)

    builder.add_edge(START, "guardrail")
    builder.add_conditional_edges(
        "guardrail", route, {"refuse": "refuse", "call_model": "call_model"}
    )
    # tools_condition routes to "tools" when the model emitted tool calls, else END.
    builder.add_conditional_edges("call_model", tools_condition)
    builder.add_edge("tools", "call_model")
    builder.add_edge("refuse", END)
    return builder.compile()

"""The LangGraph "brain" — a ReAct agent with a checkpointer and persistent
workflow state, where the transcript stays transient.

This is the architectural counterpart to `pipecat_with_langgraph/graph.py`,
extended to show the *other* half of the state story: **LangGraph owns
call/workflow state, LiveKit owns the transcript.** Three things to notice:

1. **The transcript is NOT checkpointed state.** Each turn, `agent.py` passes
   LiveKit's `ChatContext` (the authoritative, barge-in-truncated transcript)
   into the graph via `config["configurable"]["transcript"]` — a transient
   per-invocation input, never written to a LangGraph state channel. LiveKit is
   the single source of truth for what was said and heard; the graph keeps no
   competing copy. (This is the rule from the architecture writeup: an
   interrupted, never-heard turn must never linger in agent memory.) Within a
   turn, the ReAct loop accumulates the tool exchange in a *local* variable, not
   in state — so nothing transcript-shaped is ever checkpointed.

2. **Workflow state IS checkpointed.** `collected_cities` — the cities the
   caller has asked about, the receptionist "slots" analog — lives in the
   `WorkflowState` schema and is restored via `thread_id` (the LiveKit job id)
   on every turn. It survives crashes and, in production (AsyncRedisSaver), is
   what you'd fork on barge-in — *workflow* state forks; *transcript* state is
   just re-read from LiveKit. The demo uses `InMemorySaver`; swap in
   `AsyncRedisSaver` for persistence across process restarts.

3. **The control flow (ReAct tool loop) is LangGraph's, not LiveKit's.** The
   `model ⇄ tools` loop runs inside the graph node, so the tool-deciding turn is
   a traced LangSmith subspan (`on_chat_model_stream` from the bound
   `ChatOpenAI`) and is never spoken — only the final answer's content deltas
   are voiced (see `_SPOKEN_NODES` in `agent.py`). There are no LiveKit
   `@function_tool`s. LiveKit only sees the final spoken text; it speaks it,
   truncates it on barge-in, and writes the authoritative assistant message
   back into `ChatContext`.

Why a single node with an internal ReAct loop (instead of `model ⇄ tools` graph
edges): the transcript is transient (config, not state), so the within-turn tool
exchange can't be handed between nodes via a checkpointed `messages` channel
without re-introducing a parallel transcript. Keeping the loop local to the node
holds the tool exchange in a local variable — nothing transcript-shaped is ever
checkpointed, which is the whole point. The control-flow ownership question
(graph vs. LiveKit) is unaffected: the tool loop is still in LangGraph, traced
and unvoiced, not in LiveKit. For a real receptionist with *phase* routing
(Option C), the macro graph edges are between phase nodes; the same
transcript-transient / workflow-state-persistent split applies.
"""

from __future__ import annotations

from typing import Annotated, Any

from typing_extensions import TypedDict

from langchain_core.messages import AIMessage, AnyMessage, SystemMessage, ToolMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph

from ..prompts import GREETING, SYSTEM_PROMPT
from ..weather import fetch_weather

__all__ = [
    "AGENT_MODEL",
    "GREETING",
    "MODEL_NODE",
    "SYSTEM_PROMPT",
    "build_graph",
    "lookup_weather",
]

AGENT_MODEL = "gpt-4o-mini"

# The graph node name. agent.py keys _SPOKEN_NODES off this: only this node's
# streamed text deltas are the user-facing answer; the tool-deciding turn
# streams tool-call deltas with empty content (traced, never spoken).
MODEL_NODE = "model"

# Safety cap on the within-turn ReAct loop (model -> tools -> model -> ...).
_MAX_TOOL_ITERS = 5


@tool
async def lookup_weather(city: str) -> dict:
    """Get the current weather for a single city. Call once per city."""
    return await fetch_weather(city)


def _add_list(left: list[str] | None, right: list[str] | None) -> list[str]:
    """Reducer for the persistent workflow-state channel (accumulates across turns)."""
    return (left or []) + (right or [])


class WorkflowState(TypedDict):
    """Checkpointed state — workflow fields only, NO transcript.

    `collected_cities` is the "slots" analog: cities the caller has asked about,
    accumulated across turns and restored from the checkpoint each turn via
    `thread_id`. The transcript never lives here — it arrives transiently via
    `config.configurable.transcript` (see `agent.py`), so the graph keeps no
    competing copy of what was said.
    """

    collected_cities: Annotated[list[str], _add_list]


async def _react(
    transcript: list[AnyMessage],
) -> tuple[AIMessage, list[str]]:
    """Run the model ⇄ tools ReAct loop over the transcript.

    Returns the final AIMessage (the spoken answer) and the cities looked up
    this turn (to fold into `collected_cities`). The accumulating message list
    is a LOCAL variable — the transcript and the within-turn tool exchange are
    never written to graph state, so nothing transcript-shaped is ever
    checkpointed. Streaming happens inside `model.astream`, so `astream_events`
    on the compiled graph emits `on_chat_model_stream` deltas for each model
    call (tool-deciding turns have empty content and are filtered out by
    `agent.py`; the final turn's content is voiced).
    """
    model = ChatOpenAI(model=AGENT_MODEL, temperature=0.3).bind_tools([lookup_weather])
    messages: list[AnyMessage] = [SystemMessage(content=SYSTEM_PROMPT), *transcript]
    cities: list[str] = []

    for _ in range(_MAX_TOOL_ITERS):
        ai: AIMessage | None = None
        async for chunk in model.astream(messages):
            ai = chunk if ai is None else ai + chunk  # merges content + tool_calls
        assert ai is not None

        if not ai.tool_calls:
            return ai, cities  # final spoken answer

        # Tool-deciding turn: execute the calls, append the exchange to the
        # LOCAL message list (not to graph state), and loop for the final answer.
        messages.append(ai)
        for tc in ai.tool_calls:
            out = await lookup_weather.ainvoke(tc["args"])
            messages.append(
                ToolMessage(content=str(out), tool_call_id=tc["id"], name=tc["name"])
            )
            if isinstance(tc.get("args"), dict) and isinstance(
                tc["args"].get("city"), str
            ):
                cities.append(tc["args"]["city"])

    # Loop cap hit — return whatever the model last produced.
    return ai, cities  # type: ignore[possibly-undefined]


async def agent_node(state: WorkflowState, config: RunnableConfig) -> dict[str, Any]:
    """One LiveKit turn → one graph invocation.

    The transcript comes from `config.configurable.transcript` (transient). The
    ReAct loop runs inside `_react` with local variables, so the only thing
    written to state is `collected_cities` — the persistent workflow state. The
    final AIMessage is yielded for streaming by `_react`'s internal
    `model.astream`; it is NOT stored in state, because LiveKit records the
    spoken answer in its `ChatContext` after playout and re-reads it from there
    next turn. That is what keeps the graph from owning a parallel transcript.
    """
    transcript: list[AnyMessage] = config.get("configurable", {}).get("transcript", [])
    _ai, cities = await _react(transcript)
    return {"collected_cities": cities}


def build_graph():
    """Compile the weather-agent graph with an in-memory checkpointer.

    The checkpointer keys on `configurable.thread_id` (the LiveKit job id — see
    `agent.py`). It persists `collected_cities` across turns; the transcript is
    never persisted because it is not a state channel. Swap `InMemorySaver` for
    `AsyncRedisSaver` in production for crash recovery and barge-in state
    forking (of workflow state only — the transcript is re-read from LiveKit
    regardless).
    """
    g = StateGraph(WorkflowState)
    g.add_node(MODEL_NODE, agent_node)
    g.add_edge(START, MODEL_NODE)
    g.add_edge(MODEL_NODE, END)
    return g.compile(checkpointer=InMemorySaver())

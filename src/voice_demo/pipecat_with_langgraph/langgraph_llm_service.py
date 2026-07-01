"""A Pipecat LLM service whose "brain" is an in-process LangGraph graph.

Why this and not Pipecat's `LangchainProcessor`: `LangchainProcessor` is a plain
`FrameProcessor`, so it produces no `llm` span — there'd be nothing for the
graph's spans to nest under. By instead **subclassing `OpenAILLMService` and
overriding its `@traced_llm`-decorated `_process_context`**, the graph runs
*inside* Pipecat's `llm` span (which `traced_llm` opens with
`start_as_current_span`). With `LANGSMITH_TRACING_MODE=otel`, LangChain/LangGraph
emit their runs as OTel spans through the shared provider, so every node —
model, tools — becomes a **subspan of that `llm` span**, in one trace:

    turn
    └── llm                         (this service — only orchestrates the graph)
        ├── model                   (ChatOpenAI, may emit tool calls)
        ├── tools: lookup_weather   (tool execution)
        └── model                   (final answer — spoken)

Although Pipecat names this span `llm`, it does no inference itself (see the NB
below) — it just runs the graph. The LangSmith processor (`processor.py`)
therefore classifies it as a `chain`; the only `llm`-kind runs are the nested
model nodes.

Only the final assistant text is pushed as `LLMTextFrame` (and thus spoken); the
tool-deciding turn and tool output are traced but never voiced. They *are*,
however, written back into Pipecat's `LLMContext` after the run (see
`_process_context`), so the model can see its own prior tool calls and results on
later turns — without ever speaking them. This mirrors the LiveKit backend's
"LangGraph brain + OTel" design.

NB: this makes no OpenAI API call of its own — the parent's client stays unused;
we only inherit its `llm` span, metrics, and frame plumbing.
"""

from __future__ import annotations

from typing import Any

from langchain_core.messages import (
    AIMessage,
    ToolMessage,
    convert_to_messages,
    convert_to_openai_messages,
)
from pipecat.frames.frames import LLMTextFrame
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.services.openai.llm import OpenAILLMService
from pipecat.utils.tracing.service_decorators import traced_llm

# Graph nodes whose streamed tokens are the user-facing answer (so they're
# spoken). LangChain v1's `create_agent` names its model node `model`.
_SPOKEN_NODES = {"model"}


def _spoken_text(event: dict) -> str | None:
    """Text to voice from a stream event, or None.

    Only non-empty text deltas from the spoken nodes are voiced. The
    tool-deciding turn streams tool-call deltas with *empty* content, so it
    (and tool output) is traced but never spoken.
    """
    if event.get("event") != "on_chat_model_stream":
        return None
    node = (event.get("metadata") or {}).get("langgraph_node")
    if node not in _SPOKEN_NODES:
        return None
    chunk = event.get("data", {}).get("chunk")
    content = getattr(chunk, "content", None)
    return content if isinstance(content, str) and content else None


def _final_state(event: dict, best: list | None) -> list | None:
    """Track the graph's final message list across `on_chain_end` events.

    The root run's end event carries the full accumulated message list;
    node-level end events carry only their own delta — so the longest list
    wins (robust without relying on event parent_ids).
    """
    if event.get("event") != "on_chain_end":
        return best
    output = event.get("data", {}).get("output")
    if isinstance(output, dict) and isinstance(output.get("messages"), list):
        messages = output["messages"]
        if best is None or len(messages) > len(best):
            return messages
    return best


def _tool_exchange(input_messages: list, final_messages: list) -> list:
    """This turn's tool-call exchange, to persist back into Pipecat's context.

    Only the tool-deciding AIMessages and the ToolMessages they produced —
    NOT the final spoken answer: the assistant aggregator records that from
    the pushed LLMTextFrames. Persisting these first keeps the context order
    correct: [..., user, AI(tool_calls), ToolMessage, AI(final answer)].
    For tool-free turns this is empty and behaviour is unchanged.
    """
    new_messages = final_messages[len(input_messages) :]
    return [
        m
        for m in new_messages
        if isinstance(m, ToolMessage) or (isinstance(m, AIMessage) and m.tool_calls)
    ]


class LangGraphLLMService(OpenAILLMService):
    """Runs a compiled LangGraph graph as the Pipecat LLM stage."""

    def __init__(self, *, graph: Any, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._graph = graph

    @traced_llm  # re-applied so the `llm` span wraps the graph run (and nests it)
    async def _process_context(self, context: LLMContext) -> None:
        # Pipecat's context is OpenAI-format dicts; convert to LangChain
        # messages, dropping system messages — the graph owns its own prompt.
        messages = convert_to_messages(
            [m for m in context.get_messages() if m.get("role") != "system"]
        )

        await self.start_ttfb_metrics()
        first_token = True
        final_messages: list | None = None
        async for event in self._graph.astream_events(
            {"messages": messages}, version="v2"
        ):
            if event.get("event") == "on_chain_end":
                final_messages = _final_state(event, final_messages)
            elif text := _spoken_text(event):
                if first_token:
                    await self.stop_ttfb_metrics()
                    first_token = False
                await self.push_frame(LLMTextFrame(text))

        # Persist the tool-call exchange so the model sees it next turn.
        if final_messages is not None:
            to_persist = _tool_exchange(messages, final_messages)
            if to_persist:
                context.add_messages(convert_to_openai_messages(to_persist))

"""A Pipecat LLM service whose "brain" is an in-process LangGraph graph.

Why this and not Pipecat's `LangchainProcessor`: `LangchainProcessor` is a plain
`FrameProcessor`, so it produces no `llm` span — there'd be nothing for the
graph's spans to nest under. By instead **subclassing `OpenAILLMService` and
overriding its `@traced_llm`-decorated `_process_context`**, the graph runs
*inside* Pipecat's `llm` span (which `traced_llm` opens with
`start_as_current_span`). With `LANGSMITH_TRACING_MODE=otel`, LangChain/LangGraph
emit their runs as OTel spans through the shared provider, so every node —
guardrail, model, tools — becomes a **subspan of that `llm` span**, in one trace:

    turn
    └── llm                         (this service)
        ├── guardrail               (structured-output check — traced, not spoken)
        ├── call_model              (ChatOpenAI, may emit tool calls)
        ├── tools: lookup_weather   (tool execution)
        └── call_model              (final answer — spoken)

Only the final assistant text is pushed as `LLMTextFrame` (and thus spoken); the
guardrail verdict, tool-deciding turn, and tool output are traced but never
voiced. This mirrors the LiveKit backend's "LangGraph brain + OTel" design.

NB: this makes no OpenAI API call of its own — the parent's client stays unused;
we only inherit its `llm` span, metrics, and frame plumbing.
"""

from __future__ import annotations

from typing import Any

from langchain_core.messages import convert_to_messages
from pipecat.frames.frames import LLMTextFrame
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.services.openai.llm import OpenAILLMService
from pipecat.utils.tracing.service_decorators import traced_llm

# Graph nodes whose streamed tokens are the user-facing answer (so they're
# spoken). Everything else (the `guardrail` node, the tool-deciding turn's
# empty-content deltas) is traced but filtered out here.
_SPOKEN_NODES = {"call_model", "refuse"}


class LangGraphLLMService(OpenAILLMService):
    """Runs a compiled LangGraph graph as the Pipecat LLM stage."""

    def __init__(self, *, graph: Any, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._graph = graph

    @traced_llm  # re-applied so the `llm` span wraps the graph run (and nests it)
    async def _process_context(self, context: LLMContext) -> None:
        # Pipecat's context is OpenAI-format dicts; convert to LangChain
        # messages. Drop system messages — the graph supplies its own prompts.
        raw = [m for m in context.get_messages() if m.get("role") != "system"]
        messages = convert_to_messages(raw)

        await self.start_ttfb_metrics()
        first_token = True
        async for event in self._graph.astream_events(
            {"messages": messages}, version="v2"
        ):
            if event.get("event") != "on_chat_model_stream":
                continue
            node = (event.get("metadata") or {}).get("langgraph_node")
            if node not in _SPOKEN_NODES:
                continue
            chunk = event.get("data", {}).get("chunk")
            content = getattr(chunk, "content", None)
            # The tool-deciding turn streams tool-call deltas with empty content;
            # the guardrail's structured output likewise carries no spoken text.
            # Only non-empty plain-text content is voiced.
            if isinstance(content, str) and content:
                if first_token:
                    await self.stop_ttfb_metrics()
                    first_token = False
                await self.push_frame(LLMTextFrame(content))

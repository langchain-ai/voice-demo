"""An LLMAdapter that only speaks the model's own text.

The stock livekit `langchain.LLMAdapter` streams *every* message a LangGraph
graph emits under `stream_mode="messages"` — including the `ToolMessage` that
carries a tool's raw return value. For a ReAct agent that means the JSON weather
payload gets read aloud before (or instead of) the model's natural-language
summary, because the adapter's `_to_chat_chunk` happily turns any object with a
`.content` string into spoken text.

We subclass the adapter's stream and forward only `AIMessageChunk` text — the
model's own generated tokens. That drops:
  - ToolMessages (raw tool output), and
  - the tool-calling AIMessageChunk (which has tool_calls but no text).

`_chat_ctx_to_state` is inherited unchanged, so LiveKit's ChatContext remains
the single source of truth and interruption truncation keeps working.

This intentionally reaches into a couple of the plugin's internal helpers
(`langgraph._extract_message_chunk` / `_to_chat_chunk`) to keep the stream-shape
handling identical to upstream; if a future plugin version moves them, this is
the one place to update.
"""

from __future__ import annotations

from typing import Any

from langchain_core.messages import AIMessageChunk
from livekit.agents import llm
from livekit.agents.types import DEFAULT_API_CONNECT_OPTIONS, APIConnectOptions
from livekit.plugins.langchain import LLMAdapter
from livekit.plugins.langchain import langgraph as _lk_langgraph


class _AITextOnlyStream(_lk_langgraph.LangGraphStream):
    """Like LangGraphStream, but only emits the model's own text tokens."""

    async def _run(self) -> None:
        state = self._chat_ctx_to_state()
        # Mirror upstream's astream call (with graceful fallback for older
        # LangGraph versions that don't accept context/subgraphs kwargs).
        try:
            aiter = self._graph.astream(
                state,
                self._config,
                context=self._context,
                stream_mode="messages",
                subgraphs=self._subgraphs,
            )
        except TypeError:
            aiter = self._graph.astream(
                state, self._config, stream_mode="messages"
            )

        async for item in aiter:
            token = _lk_langgraph._extract_message_chunk(item)
            # Only the model's own streamed text. ToolMessages (raw tool JSON)
            # and tool-call-only chunks (no text) are skipped.
            if not isinstance(token, AIMessageChunk):
                continue
            chunk = _lk_langgraph._to_chat_chunk(token)
            if chunk is not None:
                self._event_ch.send_nowait(chunk)


class AITextOnlyLLMAdapter(LLMAdapter):
    """LLMAdapter whose stream speaks only the model's text, not tool output."""

    def chat(
        self,
        *,
        chat_ctx: Any,
        tools: list[llm.Tool] | None = None,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
        **_kwargs: Any,
    ) -> _AITextOnlyStream:
        return _AITextOnlyStream(
            self,
            chat_ctx=chat_ctx,
            tools=tools or [],
            graph=self._graph,
            conn_options=conn_options,
            config=self._config,
            context=self._context,
            subgraphs=self._subgraphs,
            stream_mode=self._stream_mode,
        )

"""LiveKit voice agent whose LLM stage is an in-process LangGraph graph.

Mirrors `livekit/agent.py` (the plain cascade) in every respect — tracing
wiring, console entrypoint, AgentSession build — except for one thing: the
`llm_node` of the `Agent` is overridden to run a LangGraph graph instead of
LiveKit's stock OpenAI LLM. See the package docstring for the architectural
thesis (LiveKit owns the transcript; LangGraph owns the control flow and the
workflow state) and `graph.py` for the brain.

Two things to notice in the wiring below that make that thesis real:

  * **The transcript is passed as transient config, not as graph input.** Each
    turn, `llm_node` converts LiveKit's `ChatContext` to LangChain messages
    (using the LiveKit item ids as stable message ids) and hands them to the
    graph under `config["configurable"]["transcript"]`. The graph's
    `WorkflowState` has no `messages` channel, so the transcript is never
    checkpointed — LiveKit stays the single source of truth, and an
    interrupted, never-heard turn can never linger in agent memory.

  * **The checkpointer keys on the LiveKit job id.** `thread_id` is the LiveKit
    `ctx.job.id`, so the graph's persistent workflow state (`collected_cities`)
    is restored per call across all turns of that call. In production, swap the
    graph's `InMemorySaver` for `AsyncRedisSaver` and the same `thread_id` gives
    you crash recovery and a barge-in fork point — for *workflow* state only;
    the transcript is re-read from LiveKit regardless.

A note on the unused `llm=` passed to `AgentSession`: LiveKit requires an
`llm.LLM` for capabilities/metadata and to construct the default `llm_node`, but
because we override `llm_node`, that plugin is never called for inference — the
graph holds its own `ChatOpenAI`. This mirrors the Pipecat sibling, where the
parent `OpenAILLMService` client stays unused while the graph does the work.
"""

from __future__ import annotations

import os
import sys
from typing import Any

LLM_MODEL = os.getenv("LIVEKIT_LLM_MODEL", "gpt-4o-mini")
STT_MODEL = os.getenv("LIVEKIT_STT_MODEL") or None
TTS_VOICE = os.getenv("LIVEKIT_TTS_VOICE") or None
TTS_MODEL = os.getenv("LIVEKIT_TTS_MODEL") or None


# The graph node whose streamed text deltas are the user-facing answer. The
# tool-deciding turn streams tool-call deltas with empty content (traced, never
# spoken). This must match MODEL_NODE in graph.py.
_SPOKEN_NODES = frozenset({"model"})


def _spoken_text(event: dict) -> str | None:
    """Text to voice from a graph stream event, or None.

    Only non-empty string content from the spoken (model) node is voiced — the
    tool-deciding turn and tool output are traced but never spoken.
    """
    if event.get("event") != "on_chat_model_stream":
        return None
    node = (event.get("metadata") or {}).get("langgraph_node")
    if node not in _SPOKEN_NODES:
        return None
    chunk = event.get("data", {}).get("chunk")
    content = getattr(chunk, "content", None)
    return content if isinstance(content, str) and content else None


def _livekit_chat_ctx_to_langchain(chat_ctx: Any) -> list:
    """Convert LiveKit's authoritative ChatContext to LangChain messages.

    Each LiveKit ChatItem id becomes the LangChain message id — a stable
    identity across turns. On barge-in, LiveKit rewrites the interrupted
    assistant message in place (same id, truncated `synchronized_transcript`
    content, `interrupted=True`); we just re-read it here next turn. The graph
    keeps no copy of its own.

    System/developer messages are dropped: the graph owns its own prompt (see
    `graph.py`). Function calls / outputs are handled too, for robustness, but
    in this architecture they are not persisted into LiveKit's ChatContext (the
    tool exchange is within-turn control flow owned by the graph, not
    transcript owned by LiveKit), so they normally do not appear here.
    """
    from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

    from livekit.agents import llm

    out: list = []
    for item in chat_ctx.items:
        if isinstance(item, llm.ChatMessage):
            if item.role in ("system", "developer"):
                continue  # the graph owns its own prompt
            text = item.text_content or ""
            if item.role == "user":
                out.append(HumanMessage(content=text, id=item.id))
            else:  # assistant (includes barge-in-truncated messages)
                out.append(AIMessage(content=text, id=item.id))
        elif isinstance(item, llm.FunctionCall):
            # Not normally present (we don't persist tool calls to LiveKit), but
            # handled so the transcript stays a valid ReAct history if it ever is.
            out.append(
                AIMessage(
                    content="",
                    id=item.id,
                    tool_calls=[
                        {"id": item.call_id, "name": item.name, "args": item.arguments}
                    ],
                )
            )
        elif isinstance(item, llm.FunctionCallOutput):
            out.append(
                ToolMessage(
                    content=item.output,
                    tool_call_id=item.call_id,
                    name=item.name or "",
                    id=item.id,
                )
            )
        # AgentHandoff / AgentConfigUpdate: ignored — not transcript.
    return out


def run(project_name: str) -> None:
    """Launch a LiveKit console agent whose brain is a LangGraph graph.

    Blocks until Ctrl-C. `project_name` is informational — the OTLP headers wired
    by `tracing.configure` decide where spans land.
    """
    if not os.environ.get("OPENAI_API_KEY"):
        print(
            "OPENAI_API_KEY is not set (this LiveKit mode needs it for the graph's LLM).",
            file=sys.stderr,
        )
        sys.exit(1)
    if not os.environ.get("ASSEMBLYAI_API_KEY"):
        print(
            "ASSEMBLYAI_API_KEY is not set (this LiveKit mode needs it for STT).",
            file=sys.stderr,
        )
        sys.exit(1)
    if not os.environ.get("CARTESIA_API_KEY"):
        print(
            "CARTESIA_API_KEY is not set (this LiveKit mode needs it for Cartesia TTS).",
            file=sys.stderr,
        )
        sys.exit(1)

    from livekit import agents
    from livekit.agents import (
        Agent,
        AgentSession,
        TurnHandlingOptions,
        llm,
        room_io,
    )

    from ..prompts import GREETING, SYSTEM_PROMPT
    from langsmith.integrations.livekit import configure_livekit

    from .graph import build_graph

    if os.environ.get("LANGSMITH_API_KEY"):
        configure_livekit(
            project=project_name,
        )
        print(
            "[livekit-with-langgraph] LangSmith OTel tracing enabled.", file=sys.stderr
        )
    else:
        print(
            "[livekit-with-langgraph] LANGSMITH_API_KEY not set — running without tracing.",
            file=sys.stderr,
        )

    from livekit.plugins import assemblyai
    from livekit.plugins import cartesia
    from livekit.plugins import openai as lk_openai
    from livekit.plugins import silero
    from livekit.plugins.turn_detector.multilingual import MultilingualModel

    def _build_session() -> AgentSession:
        """The STT → LLM → TTS cascade. The LLM slot is unused for inference —
        `llm_node` is overridden to run the LangGraph graph instead — but
        LiveKit still requires an `llm.LLM` for capabilities/metadata."""
        tts_kwargs = {}
        if TTS_MODEL:
            tts_kwargs["model"] = TTS_MODEL
        if TTS_VOICE:
            tts_kwargs["voice"] = TTS_VOICE
        return AgentSession(
            stt=assemblyai.STT(model=STT_MODEL) if STT_MODEL else assemblyai.STT(),
            llm=lk_openai.LLM(model=LLM_MODEL, temperature=0.3),
            tts=cartesia.TTS(**tts_kwargs),
            vad=silero.VAD.load(),
            turn_handling=TurnHandlingOptions(turn_detection=MultilingualModel()),
        )

    class _Assistant(Agent):
        """Instructions + an `llm_node` that runs a LangGraph brain.

        No `@function_tool`s live here: the tool loop is owned by the graph, so
        that control flow (including the traced-but-unspoken tool-deciding turn)
        stays in LangGraph. LiveKit only sees the final spoken text. The graph's
        checkpointer (keyed on the LiveKit job id) persists workflow state
        (`collected_cities`) across turns; the transcript is passed in fresh
        each turn via config and is never checkpointed.
        """

        def __init__(self) -> None:
            super().__init__(instructions=SYSTEM_PROMPT)
            self._graph = build_graph()
            self._thread_id: str | None = None

        def _config(self, transcript: list) -> dict:
            """Per-turn config: the transient transcript + the checkpointer key."""
            return {
                "configurable": {"thread_id": self._thread_id, "transcript": transcript}
            }

        async def llm_node(
            self,
            chat_ctx: llm.ChatContext,
            tools: list[llm.Tool],
            model_settings,
        ):
            # `chat_ctx` is LiveKit's truthful transcript: on barge-in the
            # interrupted assistant message has already been rewritten in place
            # to the `synchronized_transcript` (only what played). Convert it to
            # LangChain messages and pass it as TRANSIENT config — the graph has
            # no `messages` channel, so it is never checkpointed.
            transcript = _livekit_chat_ctx_to_langchain(chat_ctx)
            config = self._config(transcript)

            # Drive the graph. Input is empty: `collected_cities` (the only
            # state channel) is restored from the checkpoint for this thread_id.
            # The transcript rides in config; the graph's final answer streams
            # out via on_chat_model_stream events from the bound ChatOpenAI.
            async for event in self._graph.astream_events({}, config, version="v2"):
                if text := _spoken_text(event):
                    # Yielding str is supported by llm_node; LiveKit voices it
                    # and writes the (barge-in-truncated) assistant message back
                    # into ChatContext after playout — closing the loop.
                    yield text

    server = agents.AgentServer()

    @server.rtc_session()
    async def _entrypoint(ctx: agents.JobContext) -> None:
        assistant = _Assistant()
        # The checkpointer keys on this id; set it once the Agent exists.
        assistant._thread_id = ctx.job.id

        session = _build_session()
        await session.start(
            room=ctx.room,
            agent=assistant,
            room_options=room_io.RoomOptions(),
            record={"audio": True},
        )

        # Speak first. The cascade has a TTS stage, so session.say streams the
        # fixed greeting verbatim and records it as the assistant's opening
        # message in ChatContext.
        await session.say(GREETING)

    sys.argv = [sys.argv[0], "console", "--record"]
    agents.cli.run_app(server)

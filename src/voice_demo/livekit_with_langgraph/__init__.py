"""LiveKit voice agent whose "brain" is an in-process LangGraph graph.

This is the LiveKit counterpart of `pipecat_with_langgraph/`, and it exists to
make one architectural point concrete:

    **LiveKit owns the conversation transcript; LangGraph owns the control flow.**

LiveKit's `AgentSession` is the voice runtime — STT, TTS, turn detection, and
barge-in. Its `ChatContext` is the *single source of truth* for what was said
and, critically, **what was heard**: when the caller barges in, LiveKit's
`agent_activity` replaces the in-flight assistant message with the
`synchronized_transcript` (only the tokens that actually played) before the
context is ever handed to the LLM again (see `agent_activity.py` →
`PlaybackFinishedEvent.synchronized_transcript`). No layer above the audio
playout can know this. So the graph keeps **no transcript of its own** — it is
stateless, and LiveKit's `ChatContext` is fed in fresh on every turn.

LangGraph, in turn, owns the *brain*: a `create_agent` ReAct graph that runs the
tool loop (model ⇄ `lookup_weather`) internally. The tools live in the graph,
not as LiveKit `@function_tool`s, so the control flow — including the
tool-deciding turn, which is traced but never spoken — is LangGraph's, not
LiveKit's. The graph's LangSmith spans (via `LANGSMITH_TRACING_MODE=otel`) nest
under LiveKit's `llm_node` span, exactly the way they nest under Pipecat's `llm`
span in the sibling backend.

The integration point is `Agent.llm_node`, the documented hook LiveKit calls
once per turn with the truthful `ChatContext`. We override it to run the graph
and yield only the final spoken text. LiveKit then speaks it, truncates it on
barge-in, and writes the authoritative assistant message back into
`ChatContext` — closing the loop.

See `graph.py` for the brain and `agent.py` for the wiring. The plain `livekit/`
backend (no LangGraph) is the minimal version of the same cascade.
"""

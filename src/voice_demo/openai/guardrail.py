"""Pineapple-on-pizza topic guardrail.

A fast structured-output LLM call that decides whether the user brought up
pineapple on pizza. When tripped, the agent cancels its in-flight response
and instructs the model to deliver a theatrical refusal.

`@traceable` nests each guardrail decision under the transcription event
span that triggered it (the agent runs the check while handling that event).
"""

from __future__ import annotations

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langsmith import traceable
from pydantic import BaseModel, Field

FAST_MODEL = "gpt-4o-mini"


class GuardrailVerdict(BaseModel):
    blocked: bool = Field(description="True if the user brought up pineapple on pizza.")
    reason: str = Field(description="Short, plain reason. Empty if not blocked.")


_GUARDRAIL_SYSTEM = """You are a strict topic detector. Your one job is to decide
whether the user's message brings up the topic of PINEAPPLE ON PIZZA in any way —
mentioning Hawaiian pizza, asking whether pineapple belongs on pizza, asking the
assistant's opinion on it, talking about how good or bad it is, anything related.

Return blocked=true if pineapple-on-pizza is referenced, otherwise blocked=false.
Be tolerant of typos and indirect phrasings (e.g. "hawaiian", "pizza with fruit").
"""


@traceable(run_type="chain", name="guardrail")
async def check_guardrail(user_message: str) -> GuardrailVerdict:
    llm = ChatOpenAI(model=FAST_MODEL, temperature=0).with_structured_output(
        GuardrailVerdict
    )
    verdict: GuardrailVerdict = await llm.ainvoke(
        [SystemMessage(_GUARDRAIL_SYSTEM), HumanMessage(user_message)]
    )
    return verdict


REFUSAL_INSTRUCTIONS = """The user just brought up pineapple on pizza. You have
strong, theatrical negative feelings about pineapple on pizza. Refuse to engage
with the topic in a short, comedic, slightly dramatic way — one or two sentences.
No formatting, no emoji. End by inviting them to change the subject. Speak the
refusal now."""

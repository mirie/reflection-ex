"""ADK example of an iterative reflection workflow.

Source: Agent Design Patterns by Antonio Gulli.
"""

from typing import Literal

from google.adk import Event
from google.adk.agents import LlmAgent
from google.adk.agents.context import Context
from google.adk.events import EventActions
from google.adk.workflow import START, Workflow
from pydantic import BaseModel


MODEL = "gemini-3.5-flash-lite"
MAX_REVIEW_ROUNDS = 3


class ReviewResult(BaseModel):
    """The reviewer's decision and its explanation."""

    status: Literal["APPROVED", "REVISE"]
    feedback: str


def initialize_workflow_state(node_input: str) -> Event:
    """Save the user's request and initialize the review counter."""

    # ADK passes START's value to the special `node_input` parameter.
    user_request = node_input
    return Event(
        actions=EventActions(
            state_delta={"topic": user_request, "review_round": 0}
        )
    )


# The producer creates the first draft from the user's request.
producer = LlmAgent(
    name="DraftWriter",
    model=MODEL,
    description="Writes an initial draft about the user's request.",
    instruction=(
        "Write a short, informative paragraph about this request: {topic}"
    ),
    output_key="draft_text",
)

# The reviewer evaluates the current draft and makes a structured decision.
reviewer = LlmAgent(
    name="FactChecker",
    model=MODEL,
    description="Reviews the current draft and decides whether it needs revision.",
    instruction=(
        "Review the draft below against the user's request. Check relevance, clarity, "
        "and whether claims are supported by the request. Do not claim to verify "
        "outside facts without sources.\n\n"
        "Request:\n{topic}\n\nDraft:\n{draft_text}\n\n"
        f"This is review round {{review_round}} of at most {MAX_REVIEW_ROUNDS}. "
        "Return APPROVED only if the draft is clear, relevant, and supported. "
        "Otherwise return REVISE and explain the concrete changes needed."
    ),
    output_schema=ReviewResult,
    output_key="review_result",
)

# The reviser applies the review to the existing draft.
reviser = LlmAgent(
    name="Reviser",
    model=MODEL,
    description="Revises the current draft using the reviewer's feedback.",
    instruction=(
        "Revise the draft using the reviewer's feedback. Preserve the user's "
        "intended meaning and do not add unsupported claims.\n\n"
        "Request:\n{topic}\n\nCurrent draft:\n{draft_text}\n\n"
        "Review result and feedback:\n{review_result}"
    ),
    output_key="draft_text",
)


def route_review(review_result: ReviewResult, context: Context) -> Event:
    """Approve the draft or send it around for another review and revision."""

    review_round = context.state.get("review_round", 0) + 1
    context.state["review_round"] = review_round

    if review_result.status == "APPROVED" or review_round >= MAX_REVIEW_ROUNDS:
        return Event(actions=EventActions(route="finish"))

    return Event(actions=EventActions(route="revise"))


def return_final_draft(draft_text: str) -> str:
    """Return the latest draft as the workflow's final output."""

    return draft_text


root_agent = Workflow(
    name="reflection_workflow",
    edges=[
        (START, initialize_workflow_state, producer, reviewer, route_review),
        (route_review, {"revise": reviser, "finish": return_final_draft}),
        (reviser, reviewer),
    ],
)

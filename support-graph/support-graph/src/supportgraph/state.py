"""Graph state and the structured models the LLM is required to produce.

The state is a TypedDict rather than a free-form dict so every node has a
contract, and the Pydantic models below are what turn "the model said something"
into "the model returned a decision the graph can route on". An agent that
routes on parsed prose is an agent that fails in production the first time the
model phrases something differently.
"""

from __future__ import annotations

import operator
from enum import Enum
from typing import Annotated, Any, Literal, TypedDict

from pydantic import BaseModel, Field, field_validator


class Priority(str, Enum):
    P1 = "P1"
    P2 = "P2"
    P3 = "P3"
    P4 = "P4"


class Intent(str, Enum):
    BILLING = "billing"
    REFUND = "refund"
    TECHNICAL = "technical"
    ACCOUNT_ACCESS = "account_access"
    SECURITY = "security"
    FEATURE_REQUEST = "feature_request"
    OTHER = "other"


class Sentiment(str, Enum):
    CALM = "calm"
    FRUSTRATED = "frustrated"
    ANGRY = "angry"
    AT_RISK = "at_risk"


class Classification(BaseModel):
    """First structured decision. Routing depends on it, so it is validated."""

    intent: Intent = Intent.OTHER
    priority: Priority = Priority.P3
    sentiment: Sentiment = Sentiment.CALM
    product_area: str = "unknown"
    # Free-text summary the drafting node uses instead of re-reading the ticket.
    summary: str = ""
    # Model's own confidence. Treated as a weak signal: it is combined with
    # verification results, never trusted on its own.
    confidence: float = 0.5
    requested_amount_usd: float = 0.0
    reasoning: str = ""

    @field_validator("confidence")
    @classmethod
    def _clamp(cls, v: float) -> float:
        return max(0.0, min(1.0, v))

    @field_validator("requested_amount_usd")
    @classmethod
    def _non_negative(cls, v: float) -> float:
        return max(0.0, v)


class ToolCall(BaseModel):
    name: str
    arguments: dict[str, Any] = Field(default_factory=dict)
    reason: str = ""


class ToolPlan(BaseModel):
    calls: list[ToolCall] = Field(default_factory=list)
    done: bool = False
    reasoning: str = ""


class ToolResult(BaseModel):
    name: str
    arguments: dict[str, Any] = Field(default_factory=dict)
    ok: bool = True
    output: Any = None
    error: str = ""
    latency_ms: float = 0.0

    def as_context(self) -> str:
        if not self.ok:
            return f"{self.name}({self.arguments}) -> ERROR: {self.error}"
        return f"{self.name}({self.arguments}) -> {self.output}"


class Verification(BaseModel):
    """Output-side check. The draft is not sent until this passes."""

    grounded: bool = False
    policy_compliant: bool = True
    within_authority: bool = True
    makes_commitment: bool = False
    issues: list[str] = Field(default_factory=list)
    confidence: float = 0.0

    @property
    def passed(self) -> bool:
        return (
            self.grounded
            and self.policy_compliant
            and self.within_authority
            and not self.makes_commitment
        )


class EscalationReason(str, Enum):
    SECURITY = "security_concern"
    AUTHORITY = "exceeds_agent_authority"
    PRIORITY = "p1_requires_human"
    LOW_CONFIDENCE = "low_confidence"
    VERIFICATION_FAILED = "verification_failed"
    UNGROUNDED = "no_supporting_policy"
    INJECTION = "prompt_injection_detected"
    LOOP = "tool_loop_limit_reached"


class AgentState(TypedDict, total=False):
    """Shared state. `operator.add` reducers let nodes append without having to
    read-modify-write, which is what keeps the tool loop from clobbering
    earlier results."""

    # input
    ticket_id: str
    customer_message: str
    customer_id: str
    workspace_id: str
    channel: str

    # guardrails
    sanitised_message: str
    injection_detected: bool
    injection_kinds: list[str]
    redactions: dict[str, int]

    # reasoning
    classification: Classification | None
    plan: ToolPlan | None
    tool_results: Annotated[list[ToolResult], operator.add]
    kb_passages: list[dict[str, Any]]
    iterations: int

    # output
    draft: str
    verification: Verification | None
    draft_attempts: int
    final_reply: str
    resolution: Literal["auto_resolved", "escalated", "needs_input"]

    # escalation
    escalated: bool
    escalation_reasons: list[str]
    escalation_note: str
    human_decision: str | None

    # observability
    trace: Annotated[list[dict[str, Any]], operator.add]

    # Set from settings at invoke time. Exists so the eval harness can measure
    # how much worse the agent is with the escalation guardrails removed --
    # a number worth having rather than asserting. Never off in production.
    _escalation_enabled: bool

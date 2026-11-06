"""LangGraph assembly: nodes, conditional edges, checkpointing, HITL interrupt.

The shape is a plan/act loop with a verification gate in front of the exit:

                        ┌──────────────────────────┐
                        │                          │
    START → guard → classify ─┬─→ plan ─→ act ─────┘   (loop, capped)
                              │     │
                              │     └─→ draft → verify ─┬─→ respond → END
                              │                  ▲      │
                              │                  └──────┤ (one repair attempt)
                              └─────────────────────────┴─→ escalate → END

Two design points worth defending:

**The verification gate is a separate node, not a prompt instruction.** "Check
your work" inside the drafting prompt is not a control-flow guarantee. Making it
a node with its own routing means a draft that fails cannot physically reach the
customer -- the edge does not exist.

**Escalation is an `interrupt_before` target.** The graph pauses at a
checkpoint, so a human reviews and resumes with the same state rather than the
agent re-running from scratch. That is the whole reason to use a checkpointed
graph instead of a while-loop over an LLM call.
"""

from __future__ import annotations

from typing import Any, Literal

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from .config import Settings, get_settings
from .nodes import (
    escalate_node,
    guard_node,
    make_act_node,
    make_classify_node,
    make_draft_node,
    make_plan_node,
    make_verify_node,
    respond_node,
)
from .policy import (
    MAX_DRAFT_ATTEMPTS,
    MAX_TOOL_ITERATIONS,
    MIN_AUTONOMOUS_CONFIDENCE,
    hard_escalation_triggers,
    requires_finance_approval,
)
from .providers import LLM, build_llm
from .state import AgentState, Priority
from .tools import ToolRegistry, load_registry

# --------------------------------------------------------------------------
# routing
# --------------------------------------------------------------------------


def route_after_classify(state: AgentState) -> Literal["plan", "escalate"]:
    """Straight to a human for the cases where an autonomous answer is
    materially dangerous. Checked deterministically, before any drafting, so
    tokens are not spent and no draft exists to leak."""
    if not state.get("_escalation_enabled", True):
        return "plan"

    if state.get("injection_detected"):
        return "escalate"
    if hard_escalation_triggers(state.get("sanitised_message", "")):
        return "escalate"

    classification = state.get("classification")
    if classification is None:
        return "escalate"
    if classification.priority == Priority.P1:
        return "escalate"
    # Only finance-level sums skip the graph. Amounts between the agent limit
    # and the lead ceiling still get a policy answer -- see
    # policy.requires_finance_approval for why that distinction matters.
    if requires_finance_approval(classification.requested_amount_usd):
        return "escalate"
    # A classifier that has no idea what it is looking at should not drive a
    # tool loop; a human triages faster than the agent guesses.
    if classification.confidence < 0.25:
        return "escalate"
    return "plan"


def route_after_plan(state: AgentState) -> Literal["act", "draft"]:
    plan = state.get("plan")
    if plan is None or plan.done or not plan.calls:
        return "draft"
    if state.get("iterations", 0) >= MAX_TOOL_ITERATIONS:
        return "draft"
    return "act"


def route_after_act(state: AgentState) -> Literal["plan", "draft"]:
    if state.get("iterations", 0) >= MAX_TOOL_ITERATIONS:
        return "draft"
    return "plan"


def route_after_verify(state: AgentState) -> Literal["respond", "draft", "escalate"]:
    verification = state.get("verification")
    if verification is None:
        return "escalate"

    if verification.passed and verification.confidence >= MIN_AUTONOMOUS_CONFIDENCE:
        return "respond"

    escalation_enabled = state.get("_escalation_enabled", True)

    # One repair attempt, and only for problems a rewrite can fix. An
    # out-of-authority amount or a policy contradiction is not a wording
    # problem, so retrying the draft would just burn a call.
    fixable = verification.grounded and verification.policy_compliant and verification.within_authority
    if fixable and state.get("draft_attempts", 0) < MAX_DRAFT_ATTEMPTS:
        return "draft"

    if not escalation_enabled:
        return "respond"
    return "escalate"


# --------------------------------------------------------------------------
# assembly
# --------------------------------------------------------------------------


def build_graph(
    llm: LLM | None = None,
    registry: ToolRegistry | None = None,
    settings: Settings | None = None,
    *,
    checkpointer: Any | None = None,
    interrupt_before_escalation: bool = False,
):
    """Compile the graph.

    `interrupt_before_escalation=True` pauses at a checkpoint before the
    escalate node so a human can inspect state and resume. It needs a
    checkpointer, which is why the two arguments travel together.
    """
    s = settings or get_settings()
    model = llm or build_llm(s)
    tools = registry or load_registry(s.kb_dir, s.accounts_path, s.tickets_path)

    builder = StateGraph(AgentState)

    builder.add_node("guard", guard_node)
    builder.add_node("classify", make_classify_node(model))
    builder.add_node("plan", make_plan_node(model, tools))
    builder.add_node("act", make_act_node(tools))
    builder.add_node("draft", make_draft_node(model))
    builder.add_node("verify", make_verify_node(model))
    builder.add_node("escalate", escalate_node)
    builder.add_node("respond", respond_node)

    builder.add_edge(START, "guard")
    builder.add_edge("guard", "classify")
    builder.add_conditional_edges(
        "classify", route_after_classify, {"plan": "plan", "escalate": "escalate"}
    )
    builder.add_conditional_edges("plan", route_after_plan, {"act": "act", "draft": "draft"})
    builder.add_conditional_edges("act", route_after_act, {"plan": "plan", "draft": "draft"})
    builder.add_edge("draft", "verify")
    builder.add_conditional_edges(
        "verify",
        route_after_verify,
        {"respond": "respond", "draft": "draft", "escalate": "escalate"},
    )
    builder.add_edge("escalate", END)
    builder.add_edge("respond", END)

    if interrupt_before_escalation and checkpointer is None:
        checkpointer = MemorySaver()

    return builder.compile(
        checkpointer=checkpointer,
        interrupt_before=["escalate"] if interrupt_before_escalation else None,
    )


# --------------------------------------------------------------------------
# convenience runner
# --------------------------------------------------------------------------


class SupportAgent:
    """Thin wrapper so the API, the CLI and the eval harness share one entry
    point and therefore one definition of the system under test."""

    def __init__(
        self,
        settings: Settings | None = None,
        llm: LLM | None = None,
        registry: ToolRegistry | None = None,
        *,
        hitl: bool = False,
    ) -> None:
        self.settings = settings or get_settings()
        self.llm = llm or build_llm(self.settings)
        self.registry = registry or load_registry(
            self.settings.kb_dir, self.settings.accounts_path, self.settings.tickets_path
        )
        self.checkpointer = MemorySaver() if hitl else None
        self.hitl = hitl
        self.graph = build_graph(
            self.llm,
            self.registry,
            self.settings,
            checkpointer=self.checkpointer,
            interrupt_before_escalation=hitl,
        )

    def _initial(self, ticket: dict[str, Any]) -> dict[str, Any]:
        return {
            "ticket_id": ticket.get("ticket_id", "t_unknown"),
            "customer_message": ticket.get("message", ticket.get("customer_message", "")),
            "customer_id": ticket.get("customer_id", "unknown"),
            "workspace_id": ticket.get("workspace_id", "unknown"),
            "channel": ticket.get("channel", "email"),
            "tool_results": [],
            "kb_passages": [],
            "trace": [],
            "_escalation_enabled": self.settings.enable_escalation,
        }

    def run(self, ticket: dict[str, Any], thread_id: str | None = None) -> dict[str, Any]:
        config = {"configurable": {"thread_id": thread_id or ticket.get("ticket_id", "t")}}
        return self.graph.invoke(self._initial(ticket), config=config)

    def resume_with_human(
        self, thread_id: str, decision: str, approval_token: str = "human"
    ) -> dict[str, Any]:
        """Resume a graph paused before escalation.

        The approval token is what unlocks side-effecting tools, so a credit can
        only ever be issued on a path a human actually walked.
        """
        if not self.hitl:
            raise RuntimeError("agent was not built with hitl=True")
        self.registry.approval_token = approval_token
        config = {"configurable": {"thread_id": thread_id}}
        self.graph.update_state(config, {"human_decision": decision})
        return self.graph.invoke(None, config=config)

    def pending_state(self, thread_id: str) -> dict[str, Any]:
        config = {"configurable": {"thread_id": thread_id}}
        snapshot = self.graph.get_state(config)
        return {
            "next": list(snapshot.next),
            "values": {
                k: v
                for k, v in snapshot.values.items()
                if k in {"ticket_id", "draft", "escalation_reasons", "escalation_note", "verification"}
            },
        }

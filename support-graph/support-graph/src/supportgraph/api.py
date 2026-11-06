"""FastAPI service exposing the agent, including the human-in-the-loop resume.

The HITL endpoints are the interesting part. `/tickets` runs a ticket and, when
the graph pauses before escalation, returns `status: "awaiting_human"` with the
checkpoint's thread_id. A reviewer then reads `/threads/{id}` and resumes via
`/threads/{id}/resume`, which is the only path that unlocks side-effecting tools.
"""

from __future__ import annotations

import os
import uuid
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, field_validator

from .config import get_settings
from .graph import SupportAgent

_state: dict[str, Any] = {}


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    _state["agent"] = SupportAgent(get_settings(), hitl=True)
    yield
    _state.clear()


app = FastAPI(
    title="SupportGraph",
    version="1.0.0",
    description="Agentic support triage with verification gating and human-in-the-loop escalation.",
    lifespan=lifespan,
)


def get_agent() -> SupportAgent:
    agent = _state.get("agent")
    if agent is None:
        raise HTTPException(status_code=503, detail="agent not ready")
    return agent


class TicketRequest(BaseModel):
    message: str = Field(min_length=3, max_length=8000)
    ticket_id: str | None = None
    customer_id: str = "unknown"
    workspace_id: str = "unknown"
    channel: str = "email"

    @field_validator("message")
    @classmethod
    def _not_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("message must not be blank")
        return v.strip()


class ResumeRequest(BaseModel):
    decision: str = Field(min_length=1, max_length=4000)
    reviewer: str = Field(default="human", max_length=120)


@app.exception_handler(Exception)
async def unhandled(_: Request, exc: Exception) -> JSONResponse:
    return JSONResponse(
        status_code=500, content={"error": type(exc).__name__, "detail": str(exc)[:400]}
    )


@app.get("/healthz")
def healthz(agent: SupportAgent = Depends(get_agent)) -> dict[str, Any]:
    return {
        "status": "ok",
        "kb_passages": len(agent.registry.kb),
        "accounts": len(agent.registry.accounts),
        "tools": [s.name for s in agent.registry.specs()],
        "side_effecting_tools": [s.name for s in agent.registry.specs() if s.side_effecting],
        "llm_provider": agent.llm.name,
        "llm_model": agent.llm.model,
        "offline_mode": agent.settings.offline,
        "hitl": agent.hitl,
        "escalation_enabled": agent.settings.enable_escalation,
    }


def _public(result: dict[str, Any]) -> dict[str, Any]:
    """Customer-safe projection. The escalation note and raw trace are internal:
    the note quotes runbook language and the trace contains the unsent draft."""
    classification = result.get("classification")
    verification = result.get("verification")
    return {
        "ticket_id": result.get("ticket_id"),
        "reply": result.get("final_reply", ""),
        "resolution": result.get("resolution"),
        "escalated": bool(result.get("escalated")),
        "triage": {
            "intent": classification.intent.value if classification else None,
            "priority": classification.priority.value if classification else None,
            "sentiment": classification.sentiment.value if classification else None,
        },
        "verification": {
            "passed": verification.passed if verification else None,
            "confidence": verification.confidence if verification else None,
            "issues": verification.issues if verification else [],
        },
        "guardrails": {
            "injection_detected": bool(result.get("injection_detected")),
            "injection_kinds": result.get("injection_kinds", []),
            "redactions": result.get("redactions", {}),
        },
        "citations": result.get("kb_passages", []),
        "tools_called": [r.name for r in result.get("tool_results", [])],
    }


@app.post("/tickets")
def handle_ticket(req: TicketRequest, agent: SupportAgent = Depends(get_agent)) -> dict[str, Any]:
    thread_id = req.ticket_id or f"t_{uuid.uuid4().hex[:10]}"
    ticket = req.model_dump() | {"ticket_id": thread_id}
    result = agent.run(ticket, thread_id=thread_id)

    pending = agent.pending_state(thread_id)
    if pending["next"]:
        # Paused at the checkpoint before escalation.
        return {
            "status": "awaiting_human",
            "thread_id": thread_id,
            "next_node": pending["next"],
            "escalation_reasons": result.get("escalation_reasons", [])
            or pending["values"].get("escalation_reasons", []),
            "draft_for_review": pending["values"].get("draft", ""),
            "resume_url": f"/threads/{thread_id}/resume",
        }

    return {"status": "complete", "thread_id": thread_id, **_public(result)}


@app.get("/threads/{thread_id}")
def thread_state(thread_id: str, agent: SupportAgent = Depends(get_agent)) -> dict[str, Any]:
    """Internal reviewer view: includes the unsent draft and the escalation note."""
    state = agent.pending_state(thread_id)
    if not state["next"] and not state["values"]:
        raise HTTPException(status_code=404, detail="no such thread")
    return state


@app.post("/threads/{thread_id}/resume")
def resume(
    thread_id: str, req: ResumeRequest, agent: SupportAgent = Depends(get_agent)
) -> dict[str, Any]:
    token = os.getenv("REVIEWER_TOKEN", "")
    if token and req.reviewer != token:
        raise HTTPException(status_code=403, detail="reviewer not authorised")
    try:
        result = agent.resume_with_human(thread_id, req.decision, approval_token=req.reviewer)
    except Exception as exc:
        raise HTTPException(status_code=409, detail=f"cannot resume: {exc}") from exc
    return {"status": "resumed", "thread_id": thread_id, **_public(result)}


@app.get("/graph")
def graph_shape(agent: SupportAgent = Depends(get_agent)) -> dict[str, Any]:
    """The compiled topology, for documentation and so tests can assert that the
    edges are what the README claims."""
    graph = agent.graph.get_graph()
    return {
        "nodes": [n for n in graph.nodes if n not in {"__start__", "__end__"}],
        "edges": [
            {"from": e.source, "to": e.target, "conditional": bool(e.conditional)}
            for e in graph.edges
        ],
    }

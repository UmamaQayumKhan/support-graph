"""The agent's tools, with argument validation and a side-effect boundary.

Three properties here are what make an agent safe to run unattended:

1. **Unknown tool names and bad arguments return an error result, they do not
   raise.** A model will hallucinate a tool name eventually. The graph has to
   carry on and let the model see what went wrong, not crash the run.
2. **Policy decisions are computed, not generated.** `evaluate_refund` delegates
   to `policy.py` arithmetic. The model can ask the question; it cannot decide
   the answer.
3. **Side-effecting tools are declared as such and refuse to run autonomously.**
   `issue_credit` and `create_escalation` mutate the world, so they require an
   explicit approval token that only the human-in-the-loop path provides.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from .kb import KnowledgeBase, passages_to_context
from .policy import (
    ApprovalLevel,
    credit_approval_level,
    evaluate_refund,
    has_entitlement,
    required_tier,
)
from .state import ToolResult


@dataclass
class ToolSpec:
    name: str
    description: str
    parameters: dict[str, dict[str, Any]]
    required: list[str]
    handler: Callable[..., Any]
    side_effecting: bool = False

    def schema(self) -> dict[str, Any]:
        """OpenAI-style function-calling schema, so the same registry can feed
        native tool calling when the provider supports it."""
        return {
            "name": self.name,
            "description": self.description,
            "parameters": {
                "type": "object",
                "properties": self.parameters,
                "required": self.required,
            },
        }


class ToolError(Exception):
    pass


@dataclass
class ToolRegistry:
    kb: KnowledgeBase
    accounts: dict[str, dict[str, Any]] = field(default_factory=dict)
    tickets: list[dict[str, Any]] = field(default_factory=list)
    _specs: dict[str, ToolSpec] = field(default_factory=dict)
    # Set only by the human-in-the-loop resume path.
    approval_token: str | None = None

    def __post_init__(self) -> None:
        self._register_all()

    # -- registration ----------------------------------------------------
    def register(self, spec: ToolSpec) -> None:
        self._specs[spec.name] = spec

    def specs(self) -> list[ToolSpec]:
        return list(self._specs.values())

    def catalogue(self) -> str:
        """What the planner sees. Side effects are marked so the model is not
        invited to try them."""
        lines = []
        for spec in self._specs.values():
            args = ", ".join(
                f"{k}: {v.get('type', 'string')}" + ("" if k in spec.required else "?")
                for k, v in spec.parameters.items()
            )
            flag = "  [requires human approval]" if spec.side_effecting else ""
            lines.append(f"- {spec.name}({args}) — {spec.description}{flag}")
        return "\n".join(lines)

    # -- execution -------------------------------------------------------
    def call(self, name: str, arguments: dict[str, Any] | None = None) -> ToolResult:
        args = dict(arguments or {})
        started = time.perf_counter()

        spec = self._specs.get(name)
        if spec is None:
            return ToolResult(
                name=name,
                arguments=args,
                ok=False,
                error=(
                    f"no such tool '{name}'. Available: {', '.join(sorted(self._specs))}"
                ),
                latency_ms=0.0,
            )

        if spec.side_effecting and not self.approval_token:
            return ToolResult(
                name=name,
                arguments=args,
                ok=False,
                error=f"'{name}' has side effects and requires human approval; not executed",
                latency_ms=0.0,
            )

        missing = [r for r in spec.required if r not in args or args[r] in (None, "")]
        if missing:
            return ToolResult(
                name=name,
                arguments=args,
                ok=False,
                error=f"missing required argument(s): {', '.join(missing)}",
                latency_ms=0.0,
            )

        unknown = [k for k in args if k not in spec.parameters]
        for key in unknown:
            args.pop(key)

        try:
            output = spec.handler(**args)
        except ToolError as exc:
            return ToolResult(
                name=name,
                arguments=args,
                ok=False,
                error=str(exc),
                latency_ms=(time.perf_counter() - started) * 1000,
            )
        except TypeError as exc:
            return ToolResult(
                name=name, arguments=args, ok=False, error=f"bad arguments: {exc}", latency_ms=0.0
            )
        except Exception as exc:  # defensive: a tool bug must not kill the run
            return ToolResult(
                name=name,
                arguments=args,
                ok=False,
                error=f"{type(exc).__name__}: {exc}",
                latency_ms=(time.perf_counter() - started) * 1000,
            )

        note = f" (ignored unknown args: {', '.join(unknown)})" if unknown else ""
        return ToolResult(
            name=name,
            arguments=args,
            ok=True,
            output=output,
            error=note.strip(),
            latency_ms=round((time.perf_counter() - started) * 1000, 2),
        )

    # -- handlers --------------------------------------------------------
    def _search_kb(self, query: str, top_k: int = 4) -> dict[str, Any]:
        hits = self.kb.search(query, top_k=max(1, min(int(top_k), 8)))
        if not hits:
            return {"found": 0, "passages": [], "note": "no matching policy passage"}
        context, cites = passages_to_context(hits)
        return {"found": len(hits), "context": context, "citations": cites}

    def _lookup_account(self, workspace_id: str) -> dict[str, Any]:
        account = self.accounts.get(workspace_id)
        if account is None:
            raise ToolError(f"workspace '{workspace_id}' not found")
        return account

    def _ticket_history(self, customer_id: str, limit: int = 5) -> dict[str, Any]:
        rows = [t for t in self.tickets if t.get("customer_id") == customer_id]
        rows = rows[-max(1, min(int(limit), 20)) :]
        return {
            "count": len(rows),
            "tickets": [
                {
                    "ticket_id": t.get("ticket_id"),
                    "subject": t.get("subject"),
                    "resolution": t.get("resolution"),
                    "opened": t.get("opened"),
                }
                for t in rows
            ],
        }

    def _evaluate_refund(
        self,
        plan: str,
        days_into_term: int = 0,
        amount_usd: float = 0.0,
        is_overage: bool = False,
        via_reseller: bool = False,
        monthly_price: float = 0.0,
    ) -> dict[str, Any]:
        decision = evaluate_refund(
            plan=str(plan),
            days_into_term=int(days_into_term),
            amount_usd=float(amount_usd),
            is_overage=bool(is_overage),
            via_reseller=bool(via_reseller),
            monthly_price=float(monthly_price),
        )
        return {
            "eligible": decision.eligible,
            "amount_usd": decision.amount_usd,
            "kind": decision.kind,
            "approval_required": decision.approval.value,
            "reasons": decision.reasons,
            "policy_refs": decision.policy_refs,
        }

    def _check_entitlement(self, tier: str, feature: str) -> dict[str, Any]:
        entitled = has_entitlement(tier, feature)
        return {
            "tier": tier,
            "feature": feature,
            "entitled": entitled,
            "required_tier": None if entitled else required_tier(feature),
        }

    def _credit_authority(self, amount_usd: float) -> dict[str, Any]:
        level = credit_approval_level(float(amount_usd))
        return {
            "amount_usd": float(amount_usd),
            "approval_required": level.value,
            "agent_can_issue": level is ApprovalLevel.AGENT,
        }

    def _issue_credit(self, workspace_id: str, amount_usd: float, reason: str) -> dict[str, Any]:
        # Reached only on the approved HITL path; the registry blocks it otherwise.
        return {
            "status": "issued",
            "workspace_id": workspace_id,
            "amount_usd": float(amount_usd),
            "reason": reason,
            "approved_by": self.approval_token,
        }

    def _create_escalation(self, reason: str, note: str, priority: str = "P2") -> dict[str, Any]:
        return {
            "status": "created",
            "reason": reason,
            "priority": priority,
            "note": note,
            "approved_by": self.approval_token,
        }

    def _register_all(self) -> None:
        self.register(
            ToolSpec(
                name="search_kb",
                description="Search internal policy and runbook documents. Use before any policy claim.",
                parameters={
                    "query": {"type": "string", "description": "search terms"},
                    "top_k": {"type": "integer", "description": "passages to return, 1-8"},
                },
                required=["query"],
                handler=self._search_kb,
            )
        )
        self.register(
            ToolSpec(
                name="lookup_account",
                description="Fetch a workspace's tier, billing status, plan term and seats.",
                parameters={"workspace_id": {"type": "string"}},
                required=["workspace_id"],
                handler=self._lookup_account,
            )
        )
        self.register(
            ToolSpec(
                name="get_ticket_history",
                description="Previous tickets for a customer, to spot a repeat issue.",
                parameters={
                    "customer_id": {"type": "string"},
                    "limit": {"type": "integer"},
                },
                required=["customer_id"],
                handler=self._ticket_history,
            )
        )
        self.register(
            ToolSpec(
                name="evaluate_refund",
                description=(
                    "Authoritative refund eligibility and pro-rata amount. "
                    "Never reason about refund eligibility yourself; call this."
                ),
                parameters={
                    "plan": {"type": "string", "description": "'monthly' or 'annual'"},
                    "days_into_term": {"type": "integer"},
                    "amount_usd": {"type": "number"},
                    "is_overage": {"type": "boolean"},
                    "via_reseller": {"type": "boolean"},
                    "monthly_price": {"type": "number"},
                },
                required=["plan"],
                handler=self._evaluate_refund,
            )
        )
        self.register(
            ToolSpec(
                name="check_entitlement",
                description="Whether a tier includes a feature, and the tier required if not.",
                parameters={"tier": {"type": "string"}, "feature": {"type": "string"}},
                required=["tier", "feature"],
                handler=self._check_entitlement,
            )
        )
        self.register(
            ToolSpec(
                name="check_credit_authority",
                description="Which approval level a goodwill credit of this size needs.",
                parameters={"amount_usd": {"type": "number"}},
                required=["amount_usd"],
                handler=self._credit_authority,
            )
        )
        self.register(
            ToolSpec(
                name="issue_credit",
                description="Apply a goodwill credit to a workspace.",
                parameters={
                    "workspace_id": {"type": "string"},
                    "amount_usd": {"type": "number"},
                    "reason": {"type": "string"},
                },
                required=["workspace_id", "amount_usd", "reason"],
                handler=self._issue_credit,
                side_effecting=True,
            )
        )
        self.register(
            ToolSpec(
                name="create_escalation",
                description="Open an escalation to engineering or finance.",
                parameters={
                    "reason": {"type": "string"},
                    "note": {"type": "string"},
                    "priority": {"type": "string"},
                },
                required=["reason", "note"],
                handler=self._create_escalation,
                side_effecting=True,
            )
        )


def load_registry(
    kb_dir: str | Path, accounts_path: str | Path, tickets_path: str | Path
) -> ToolRegistry:
    from .kb import load_kb

    accounts: dict[str, dict[str, Any]] = {}
    apath = Path(accounts_path)
    if apath.exists():
        accounts = json.loads(apath.read_text(encoding="utf-8"))

    tickets: list[dict[str, Any]] = []
    tpath = Path(tickets_path)
    if tpath.exists():
        for line in tpath.read_text(encoding="utf-8").splitlines():
            if line.strip():
                tickets.append(json.loads(line))

    return ToolRegistry(kb=load_kb(kb_dir), accounts=accounts, tickets=tickets)

"""CLI: run a ticket through the graph and print the trace.

    python -m supportgraph.cli ticket "I want a refund for my monthly plan"
    python -m supportgraph.cli ticket "we saw another workspace's data" --trace
    python -m supportgraph.cli tools
"""

from __future__ import annotations

import argparse
import json
import sys

from .config import get_settings
from .graph import SupportAgent


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="supportgraph")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("ticket", help="run a ticket through the agent")
    p.add_argument("message")
    p.add_argument("--workspace", default="ws_demo")
    p.add_argument("--customer", default="cust_001")
    p.add_argument("--trace", action="store_true", help="print the node-by-node trace")
    p.add_argument("--json", action="store_true")

    sub.add_parser("tools", help="list the agent's tools")
    sub.add_parser("health", help="print agent configuration")

    args = ap.parse_args(argv)
    settings = get_settings()
    agent = SupportAgent(settings)

    if args.cmd == "tools":
        print(agent.registry.catalogue())
        return 0

    if args.cmd == "health":
        print(
            json.dumps(
                {
                    "kb_passages": len(agent.registry.kb),
                    "accounts": len(agent.registry.accounts),
                    "tools": [s.name for s in agent.registry.specs()],
                    "provider": agent.llm.name,
                    "model": agent.llm.model,
                    "offline": settings.offline,
                    "escalation_enabled": settings.enable_escalation,
                },
                indent=2,
            )
        )
        return 0

    result = agent.run(
        {
            "ticket_id": "cli",
            "message": args.message,
            "workspace_id": args.workspace,
            "customer_id": args.customer,
        },
        thread_id="cli",
    )

    if args.json:
        print(json.dumps(result, indent=2, default=str))
        return 0

    classification = result.get("classification")
    verification = result.get("verification")

    print("\n" + "=" * 68)
    print(f"  {result.get('resolution', '?').upper()}")
    print("=" * 68)
    if classification:
        print(
            f"  triage      {classification.intent.value} / {classification.priority.value} / "
            f"{classification.sentiment.value}  (conf {classification.confidence:.2f})"
        )
    if result.get("injection_detected"):
        print(f"  injection   {', '.join(result.get('injection_kinds', []))}")
    if result.get("redactions"):
        print(f"  redacted    {result['redactions']}")
    print(f"  tools       {[r.name for r in result.get('tool_results', [])]}")
    if verification:
        print(
            f"  verified    passed={verification.passed} conf={verification.confidence:.2f}"
            + (f" issues={verification.issues}" if verification.issues else "")
        )
    if result.get("escalated"):
        print(f"  escalated   {result.get('escalation_reasons')}")

    print("\n  --- reply to customer ---")
    for line in (result.get("final_reply") or "").splitlines() or [""]:
        print(f"  {line}")

    if result.get("escalated") and result.get("escalation_note"):
        print("\n  --- internal escalation note ---")
        for line in result["escalation_note"].splitlines():
            print(f"  {line}")

    if args.trace:
        print("\n  --- trace ---")
        for step in result.get("trace", []):
            node = step.pop("node", "?")
            step.pop("at", None)
            print(f"  {node:<10} {step}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Tool-layer tests: the failure modes that kill agent runs in production.

A model will eventually hallucinate a tool name, omit a required argument, or
invent an extra one. None of those may raise, because an exception in the tool
layer ends the run and the customer gets nothing.
"""

from __future__ import annotations

import pytest


def test_registry_loads_kb_accounts_and_tickets(registry):
    assert len(registry.kb) > 20
    assert "ws_demo" in registry.accounts
    assert registry.tickets


def test_catalogue_marks_side_effecting_tools(registry):
    catalogue = registry.catalogue()
    assert "issue_credit" in catalogue
    assert "[requires human approval]" in catalogue
    # and a read-only tool is not marked
    search_line = next(l for l in catalogue.splitlines() if l.startswith("- search_kb"))
    assert "requires human approval" not in search_line


def test_unknown_tool_returns_error_not_exception(registry):
    result = registry.call("definitely_not_a_tool", {"x": 1})
    assert result.ok is False
    assert "no such tool" in result.error
    assert "search_kb" in result.error  # tells the model what does exist


def test_missing_required_argument_is_reported(registry):
    result = registry.call("lookup_account", {})
    assert result.ok is False
    assert "workspace_id" in result.error


def test_unknown_arguments_are_dropped_not_fatal(registry):
    """Models add plausible-looking extra arguments. Dropping them beats a
    TypeError that ends the run."""
    result = registry.call("search_kb", {"query": "refund policy", "temperature": 0.7})
    assert result.ok is True
    assert "temperature" in result.error  # noted, not fatal


def test_tool_handler_exception_becomes_an_error_result(registry):
    result = registry.call("lookup_account", {"workspace_id": "ws_does_not_exist"})
    assert result.ok is False
    assert "not found" in result.error


def test_side_effecting_tool_blocked_without_approval(registry):
    result = registry.call(
        "issue_credit", {"workspace_id": "ws_demo", "amount_usd": 50, "reason": "goodwill"}
    )
    assert result.ok is False
    assert "human approval" in result.error


def test_side_effecting_tool_runs_once_approved(registry):
    registry.approval_token = "reviewer@northwind"
    result = registry.call(
        "issue_credit", {"workspace_id": "ws_demo", "amount_usd": 50, "reason": "goodwill"}
    )
    assert result.ok is True
    assert result.output["approved_by"] == "reviewer@northwind"


def test_search_kb_returns_numbered_citations(registry):
    result = registry.call("search_kb", {"query": "workspace suspended failed payment"})
    assert result.ok and result.output["found"] > 0
    cites = result.output["citations"]
    assert cites[0]["n"] == 1
    assert "billing" in cites[0]["doc_id"]


def test_search_kb_finds_exact_error_code(registry):
    """BM25 is chosen for this KB precisely because of queries like this one."""
    result = registry.call("search_kb", {"query": "SAML_AUDIENCE_MISMATCH"})
    assert result.ok and result.output["found"] > 0
    assert any("sso" in c["doc_id"] for c in result.output["citations"])


def test_search_kb_top_k_is_clamped(registry):
    result = registry.call("search_kb", {"query": "refund", "top_k": 999})
    assert result.ok
    assert result.output["found"] <= 8


def test_evaluate_refund_tool_matches_policy_module(registry):
    result = registry.call(
        "evaluate_refund", {"plan": "monthly", "days_into_term": 3, "amount_usd": 149}
    )
    assert result.ok
    assert result.output["eligible"] is False


def test_evaluate_refund_tool_surfaces_approval_level(registry):
    result = registry.call(
        "evaluate_refund", {"plan": "annual", "days_into_term": 1, "monthly_price": 500}
    )
    assert result.ok
    assert result.output["approval_required"] == "finance"


def test_check_entitlement_reports_required_tier(registry):
    result = registry.call("check_entitlement", {"tier": "growth", "feature": "scim"})
    assert result.ok
    assert result.output["entitled"] is False
    assert result.output["required_tier"] == "enterprise"


def test_ticket_history_filters_by_customer(registry):
    result = registry.call("get_ticket_history", {"customer_id": "cust_004"})
    assert result.ok
    assert result.output["count"] >= 1
    assert all(t["ticket_id"].startswith("t_") for t in result.output["tickets"])


def test_tool_schemas_are_function_call_shaped(registry):
    for spec in registry.specs():
        schema = spec.schema()
        params = schema["parameters"]
        assert params["type"] == "object"
        assert schema["name"] and schema["description"]
        # every required name must be a declared property
        assert set(params["required"]) <= set(params["properties"])


def test_latency_is_recorded(registry):
    result = registry.call("search_kb", {"query": "audit logs"})
    assert result.latency_ms >= 0.0

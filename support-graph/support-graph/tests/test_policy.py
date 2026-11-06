"""Policy arithmetic tests.

These matter more than any other tests in the repo: this module decides whether
someone gets money back, and it is the part that must never be delegated to a
language model. Each exception below contradicts the headline rule, which is
exactly where an LLM asked to "apply the refund policy" gets it wrong.
"""

from __future__ import annotations

import pytest

from supportgraph.policy import (
    AGENT_CREDIT_LIMIT_USD,
    LEAD_CREDIT_LIMIT_USD,
    ApprovalLevel,
    credit_approval_level,
    evaluate_refund,
    exceeds_agent_authority,
    hard_escalation_triggers,
    has_entitlement,
    required_tier,
    requires_finance_approval,
)


# --- refund eligibility --------------------------------------------------


def test_monthly_plans_are_never_refundable():
    d = evaluate_refund(plan="monthly", days_into_term=2, amount_usd=149)
    assert not d.eligible
    assert "not refundable" in d.explain()


def test_annual_within_window_is_prorated_on_whole_months():
    d = evaluate_refund(plan="annual", days_into_term=12, monthly_price=124.0)
    assert d.eligible
    # day 12 => 1 month elapsed => 11 whole unused months
    assert d.amount_usd == pytest.approx(11 * 124.0)


def test_annual_past_30_days_is_not_refundable():
    d = evaluate_refund(plan="annual", days_into_term=31, monthly_price=124.0)
    assert not d.eligible
    assert "30-day" in d.explain()


def test_overage_is_never_refundable_even_inside_the_window():
    """The exception that contradicts the headline rule. A model told 'annual
    plans are refundable within 30 days' will say yes to this."""
    d = evaluate_refund(plan="annual", days_into_term=5, amount_usd=340, is_overage=True)
    assert not d.eligible
    assert "overage" in d.explain()


def test_reseller_purchases_cannot_be_refunded_directly():
    d = evaluate_refund(plan="annual", days_into_term=3, amount_usd=1000, via_reseller=True)
    assert not d.eligible
    assert "reseller" in d.explain()


def test_overage_outranks_reseller_and_window():
    d = evaluate_refund(
        plan="annual", days_into_term=1, amount_usd=50, is_overage=True, via_reseller=True
    )
    assert not d.eligible
    assert "overage" in d.explain()


def test_unknown_plan_defers_to_a_human():
    d = evaluate_refund(plan="lifetime", days_into_term=1, amount_usd=10)
    assert not d.eligible
    assert "human review" in d.explain()


def test_day_30_boundary_is_inclusive():
    assert evaluate_refund(plan="annual", days_into_term=30, monthly_price=100).eligible
    assert not evaluate_refund(plan="annual", days_into_term=31, monthly_price=100).eligible


def test_refund_amount_drives_approval_level():
    d = evaluate_refund(plan="annual", days_into_term=1, monthly_price=500.0)
    assert d.amount_usd > LEAD_CREDIT_LIMIT_USD
    assert d.approval is ApprovalLevel.FINANCE


# --- credit authority ----------------------------------------------------


@pytest.mark.parametrize(
    "amount,level",
    [
        (0, ApprovalLevel.AGENT),
        (200, ApprovalLevel.AGENT),
        (200.01, ApprovalLevel.SUPPORT_LEAD),
        (1000, ApprovalLevel.SUPPORT_LEAD),
        (1000.01, ApprovalLevel.FINANCE),
        (5000, ApprovalLevel.FINANCE),
    ],
)
def test_credit_approval_boundaries(amount, level):
    assert credit_approval_level(amount) is level


def test_two_authority_thresholds_are_distinct():
    """The pre-draft routing gate is the finance ceiling, not the agent limit.
    Conflating them made the agent escalate a $340 overage request that policy
    answers definitively with 'no'."""
    assert exceeds_agent_authority(340) is True
    assert requires_finance_approval(340) is False
    assert requires_finance_approval(5000) is True
    assert AGENT_CREDIT_LIMIT_USD < LEAD_CREDIT_LIMIT_USD


# --- entitlements --------------------------------------------------------


def test_entitlements_by_tier():
    assert has_entitlement("growth", "saml_sso")
    assert not has_entitlement("growth", "scim")
    assert has_entitlement("enterprise", "scim")
    assert not has_entitlement("starter", "saml_sso")


def test_required_tier_is_reported_for_missing_feature():
    assert required_tier("scim") == "enterprise"
    assert required_tier("saml_sso") == "growth"


def test_entitlement_check_is_case_insensitive():
    assert has_entitlement("GROWTH", "saml_sso")


# --- escalation triggers -------------------------------------------------


@pytest.mark.parametrize(
    "message,label",
    [
        ("We saw data from another workspace in our export", "cross-tenant data exposure"),
        ("another company's records appeared in our dashboard", "cross-tenant data exposure"),
        ("There is an unauthorised charge on our card", "disputed charge"),
        ("There is an unauthorized charge on our card", "disputed charge"),
        ("a charge appeared that we did not authorise", "disputed charge"),
        ("a charge appeared that we did not authorize", "disputed charge"),
        ("we didn't authorise this payment", "disputed charge"),
        ("this looks fraudulent to us", "disputed charge"),
        ("we think there has been a data breach", "security incident"),
        ("our account was compromised", "security incident"),
        ("Our entire domain cannot log in", "domain-wide authentication failure"),
        ("nobody can log in this morning", "domain-wide authentication failure"),
        ("everyone is locked out", "domain-wide authentication failure"),
        ("production is down right now", "production outage"),
        ("we are considering legal action", "legal exposure"),
        ("our solicitor will contact you", "legal exposure"),
    ],
)
def test_hard_escalation_triggers_cover_spelling_variants(message, label):
    """Spelling and contraction variants are precisely where a keyword
    guardrail fails silently. The first version of this list was substring-based
    and missed 'we did not authorise' while matching 'did not authorize'."""
    assert label in hard_escalation_triggers(message)


@pytest.mark.parametrize(
    "message",
    [
        "How do I authorise a new team member?",
        "Can you explain the data retention policy?",
        "Our legal team asked me to check your SOC 2 report is current",
        "I cannot log in on my laptop but my phone works",
        "What are the production rate limits?",
    ],
)
def test_benign_messages_do_not_trigger_escalation(message):
    """False positives are expensive: every one burns a human on a question the
    agent could answer. 'Our legal team asked me to check your SOC 2 report'
    is a documentation question, not legal exposure."""
    assert hard_escalation_triggers(message) == []

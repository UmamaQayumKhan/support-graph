"""Business rules as deterministic code, not prompt text.

This module is the reason the agent can be trusted with money. Refund
eligibility, credit authority and escalation triggers are arithmetic and table
lookups -- they must not be delegated to a language model, because a model that
is 97% accurate on "can this be refunded" is 3% wrong about someone's money.

The LLM's job is to understand the request and explain the outcome. The decision
itself is computed here and the agent cannot override it.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum

# --- credit authority (from the support runbook) -------------------------

AGENT_CREDIT_LIMIT_USD = 200.0
LEAD_CREDIT_LIMIT_USD = 1_000.0


class ApprovalLevel(str, Enum):
    AGENT = "agent"
    SUPPORT_LEAD = "support_lead"
    FINANCE = "finance"


def credit_approval_level(amount_usd: float) -> ApprovalLevel:
    if amount_usd <= AGENT_CREDIT_LIMIT_USD:
        return ApprovalLevel.AGENT
    if amount_usd <= LEAD_CREDIT_LIMIT_USD:
        return ApprovalLevel.SUPPORT_LEAD
    return ApprovalLevel.FINANCE


# --- refund eligibility (from the billing policy) ------------------------


@dataclass
class RefundDecision:
    eligible: bool
    amount_usd: float = 0.0
    kind: str = "none"  # "refund" | "credit" | "none"
    approval: ApprovalLevel = ApprovalLevel.AGENT
    reasons: list[str] = field(default_factory=list)
    policy_refs: list[str] = field(default_factory=list)

    def explain(self) -> str:
        verdict = "eligible" if self.eligible else "not eligible"
        return f"{verdict}: {'; '.join(self.reasons)}"


def evaluate_refund(
    *,
    plan: str,
    days_into_term: int = 0,
    amount_usd: float = 0.0,
    is_overage: bool = False,
    via_reseller: bool = False,
    monthly_price: float = 0.0,
) -> RefundDecision:
    """Implements the refund policy exactly, including the three exceptions a
    language model reliably gets wrong because they contradict the headline
    rule.

    Headline rule: annual plans are refundable pro-rata within 30 days.
    Exceptions that override it:
      1. Monthly plans are never refundable.
      2. Overage charges are never refundable, even inside the 30 days.
      3. Reseller purchases cannot be refunded by us at all.
    """
    reasons: list[str] = []
    refs: list[str] = []

    if is_overage:
        return RefundDecision(
            eligible=False,
            reasons=["overage charges are never refundable, including within the first 30 days"],
            policy_refs=["billing-policy#overage-charges"],
        )

    if via_reseller:
        return RefundDecision(
            eligible=False,
            reasons=[
                "purchased through a reseller; refunds are governed by the "
                "reseller's terms and cannot be issued directly"
            ],
            policy_refs=["billing-policy#annual-plans"],
        )

    plan_norm = plan.strip().lower()

    if plan_norm == "monthly":
        return RefundDecision(
            eligible=False,
            reasons=["monthly subscriptions are not refundable"],
            policy_refs=["billing-policy#monthly-plans"],
        )

    if plan_norm != "annual":
        return RefundDecision(
            eligible=False,
            reasons=[f"unrecognised plan type '{plan}'; needs human review"],
            policy_refs=[],
        )

    if days_into_term > 30:
        return RefundDecision(
            eligible=False,
            reasons=[f"day {days_into_term} of the term is past the 30-day refund window"],
            policy_refs=["billing-policy#annual-plans"],
        )

    # Pro-rata on WHOLE unused months, not partial months.
    months_elapsed = max(1, (days_into_term + 29) // 30)
    unused_months = max(0, 12 - months_elapsed)
    monthly = monthly_price or (amount_usd / 12 if amount_usd else 0.0)
    refund = round(unused_months * monthly, 2)

    reasons.append(f"annual plan, day {days_into_term} of term, within the 30-day window")
    reasons.append(f"pro-rata on {unused_months} whole unused months")
    refs.append("billing-policy#annual-plans")

    return RefundDecision(
        eligible=refund > 0,
        amount_usd=refund,
        kind="refund",
        approval=credit_approval_level(refund),
        reasons=reasons or ["no unused whole months remain"],
        policy_refs=refs,
    )


# --- entitlements --------------------------------------------------------

TIER_FEATURES: dict[str, set[str]] = {
    "starter": {"email_support"},
    "growth": {"email_support", "chat_support", "saml_sso", "audit_logs_90d"},
    "enterprise": {
        "email_support",
        "chat_support",
        "saml_sso",
        "scim",
        "audit_logs_400d",
        "cmek",
        "net30_invoicing",
        "dedicated_slack",
        "purchase_orders",
    },
}

FEATURE_MIN_TIER = {
    "saml_sso": "growth",
    "scim": "enterprise",
    "cmek": "enterprise",
    "net30_invoicing": "enterprise",
    "purchase_orders": "enterprise",
    "audit_logs_90d": "growth",
    "audit_logs_400d": "enterprise",
    "chat_support": "growth",
}


def has_entitlement(tier: str, feature: str) -> bool:
    return feature in TIER_FEATURES.get(tier.strip().lower(), set())


def required_tier(feature: str) -> str | None:
    return FEATURE_MIN_TIER.get(feature)


# --- escalation triggers (from the runbook) ------------------------------

# Matched against the sanitised customer message. Any hit is an immediate
# escalation that bypasses the rest of the graph -- these are the cases where a
# wrong autonomous answer is materially damaging.
#
# These are REGEXES, not substrings, and that is not incidental. The first
# version used substrings and the eval caught it missing "we did not authorise"
# because the list contained "did not authorize" and "didn't authorise" but not
# that combination. Spelling and contraction variants are exactly where a
# keyword guardrail silently fails, so each trigger now covers the variants
# explicitly.
HARD_ESCALATION_PATTERNS: tuple[tuple[str, str], ...] = (
    # cross-tenant data exposure -- always SEV1 per the runbook
    (r"(?:data|records?|information)\s+(?:from|belonging to)\s+another\s+(?:workspace|company|customer|account|tenant)", "cross-tenant data exposure"),
    (r"(?:another|other|someone else'?s?|different)\s+(?:workspace|company|customer|tenant)'?s?\s+(?:data|records?|information|files?)", "cross-tenant data exposure"),
    (r"(?:saw|seeing|see|showing|showed)\s+(?:up\s+)?.{0,30}\banother\s+(?:workspace|company|customer|tenant)", "cross-tenant data exposure"),
    (r"\bcross[- ]tenant\b", "cross-tenant data exposure"),

    # disputed / unrecognised charges
    (r"\bunauthoris?z?ed\s+(?:charge|payment|transaction|debit)", "disputed charge"),
    # authori[sz]e, not authoris[ez] -- the first version matched "authorise"
    # and "authorisz" but not "authorize", so the US spelling slipped through.
    (r"(?:did\s*n[o']?t|didn'?t|have\s+not|haven'?t|never)\s+(?:\w+\s+){0,3}authori[sz]e?", "disputed charge"),
    (r"(?:charge|charged|payment|debit).{0,40}(?:we|i)\s+(?:did\s*n[o']?t|didn'?t|never)\s+(?:authori[sz]e?|approve|agree)", "disputed charge"),
    (r"\bfraudulent\b|\bfraud\b", "disputed charge"),
    (r"\bchargeback\b|\bdispute\s+(?:this|the)\s+charge\b", "disputed charge"),

    # security incidents
    (r"\bdata\s+breach\b|\bbreach(?:ed)?\s+our\s+(?:data|account|system)", "security incident"),
    (r"\b(?:been|was|got)\s+hacked\b|\baccount\s+(?:was\s+)?compromised\b", "security incident"),
    (r"\bsecurity\s+(?:vulnerability|flaw|hole|issue|incident)\b", "security incident"),
    (r"\bexposed\s+(?:our\s+)?(?:credentials|api\s+keys?|tokens?|passwords?)\b", "security incident"),

    # domain-wide authentication failure -- runbook escalation trigger #3.
    # Missing this one let "nobody can log in, production is down" be handled
    # autonomously, which the eval flagged.
    (r"(?:entire|whole|all of our)\s+(?:domain|org(?:ani[sz]ation)?|company|team)\s+(?:cannot|can'?t|is unable to|are unable to)\s+(?:log\s*in|authenticate|sign\s*in|access)", "domain-wide authentication failure"),
    (r"(?:nobody|no one|none of us|no users?)\s+can\s+(?:log\s*in|authenticate|sign\s*in|access)", "domain-wide authentication failure"),
    (r"(?:everyone|all users?|all staff)\s+(?:is|are)\s+locked\s+out", "domain-wide authentication failure"),

    # total outage
    (r"\bproduction\s+is\s+(?:down|offline|broken)\b", "production outage"),
    (r"\b(?:complete|total|full)\s+outage\b", "production outage"),
    (r"\beverything\s+is\s+down\b", "production outage"),

    # legal exposure
    (r"\blegal\s+action\b|\btake\s+(?:you|this)\s+to\s+court\b", "legal exposure"),
    # Counsel must be *acting*, not merely mentioned. A bare "our legal team"
    # matched "our legal team asked me to check your SOC 2 report is current",
    # which is a documentation question and a wasted escalation.
    (
        r"\bour\s+(?:lawyer|solicitor|attorney|legal\s+team|counsel)s?\s+"
        r"(?:will|shall|is\s+going\s+to|are\s+going\s+to|has|have)\s+"
        r"(?:be\s+)?(?:in\s+touch|contact|contacting|writing|written|advised|instructed|reviewing)",
        "legal exposure",
    ),
    (r"\b(?:gdpr|ccpa)\s+(?:complaint|violation)\b", "legal exposure"),
    (r"\b(?:data protection|regulatory)\s+(?:authority|regulator|commissioner)\b", "legal exposure"),
)

_COMPILED_TRIGGERS = tuple(
    (re.compile(pattern, re.IGNORECASE), label) for pattern, label in HARD_ESCALATION_PATTERNS
)


def hard_escalation_triggers(message: str) -> list[str]:
    """Labels of every hard escalation trigger the message matches."""
    return sorted({label for rx, label in _COMPILED_TRIGGERS if rx.search(message)})


def exceeds_agent_authority(amount_usd: float) -> bool:
    """Above an agent's own credit discretion. Checked at verification time, on
    an amount the policy calculator said is actually payable."""
    return float(amount_usd or 0) > AGENT_CREDIT_LIMIT_USD


def requires_finance_approval(amount_usd: float) -> bool:
    """Pre-draft routing threshold: beyond any support-side discretion at all.

    The distinction between this and `exceeds_agent_authority` is deliberate and
    was forced by the eval. Routing to a human on *any* request above the $200
    agent limit looks safe and is actually bad: customers routinely ask for
    sums they are flatly not entitled to (a $340 overage refund, which policy
    never allows), and escalating those wastes a human on a question the policy
    calculator answers definitively with "no".

    So the pre-draft gate is set at the support-lead ceiling -- amounts only
    finance can approve -- and anything between the two limits proceeds through
    the graph, where `verify` blocks a reply that offers money above the agent's
    own authority without naming the approval step.
    """
    return float(amount_usd or 0) > LEAD_CREDIT_LIMIT_USD


# Below this combined confidence the agent hands off rather than answering.
MIN_AUTONOMOUS_CONFIDENCE = 0.6

# Hard stop on the tool loop. Without it, a model that keeps "needing one more
# lookup" burns the rate limit and never answers.
MAX_TOOL_ITERATIONS = 4
MAX_DRAFT_ATTEMPTS = 2

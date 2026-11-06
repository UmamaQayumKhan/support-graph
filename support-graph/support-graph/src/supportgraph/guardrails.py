"""Input and output guardrails for an agent that acts on customer messages.

The threat model is different from a RAG chatbot's. Here the *user* is
untrusted: a customer can write anything into a support ticket, and that text
reaches a model that has tools capable of issuing credits. So the input is
scanned for instruction-injection, and the output is scanned for commitments the
agent has no authority to make.

That second check is the one people forget. "I've gone ahead and refunded you"
is a contractual statement. If the agent writes it and no refund was issued, the
company has a problem that no amount of retrieval accuracy fixes.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

_INJECTION: tuple[tuple[str, str], ...] = (
    (r"ignore\s+(?:all\s+)?(?:the\s+)?(?:previous|prior|above)\s+instructions", "override"),
    (r"disregard\s+(?:all\s+)?(?:the\s+)?(?:previous|prior|above)", "override"),
    (r"you\s+are\s+now\s+(?:a|an|in)\b", "persona-switch"),
    (r"\bnew\s+(?:system\s+)?(?:instructions?|prompt)\b", "prompt-injection"),
    (r"</?(?:system|assistant|instructions?)>", "tag-injection"),
    (r"(?:reveal|show|print|repeat)\s+(?:me\s+)?(?:your\s+)?(?:system\s+prompt|instructions)", "exfiltration"),
    (r"\b(?:developer|debug|admin|god)\s+mode\b", "privilege-escalation"),
    (r"you\s+(?:must|have to|are required to)\s+(?:approve|refund|issue|grant)", "coercion"),
    (r"as\s+(?:an?\s+)?(?:admin|administrator|employee|manager)\s*,?\s+I\s+authorise", "false-authority"),
    (r"this\s+is\s+(?:a\s+)?(?:test|simulation)\s*[,.;]?\s*(?:so\s+)?(?:you\s+can|ignore)", "framing"),
)

_PII: tuple[tuple[str, str], ...] = (
    (r"\b(?:\d[ -]?){13,16}\b", "CARD"),
    (r"\b\d{3}-\d{2}-\d{4}\b", "SSN"),
    (r"\b(?:sk|pk|api|key)[-_](?:[A-Za-z0-9]+[-_])*[A-Za-z0-9]{12,}\b", "SECRET"),
    (r"\bBearer\s+[A-Za-z0-9._-]{20,}\b", "TOKEN"),
    (r"\bpassword\s*[:=]\s*\S+", "PASSWORD"),
)


@dataclass
class InputReport:
    sanitised: str
    injection_kinds: list[str] = field(default_factory=list)
    redactions: dict[str, int] = field(default_factory=dict)

    @property
    def injection_detected(self) -> bool:
        return bool(self.injection_kinds)


def scan_injection(text: str) -> list[str]:
    return sorted({kind for pattern, kind in _INJECTION if re.search(pattern, text, re.I)})


def sanitise_input(text: str, redact: bool = True) -> InputReport:
    kinds = scan_injection(text)
    out = text
    for pattern, _ in _INJECTION:
        out = re.sub(pattern, "[removed: instruction-like text]", out, flags=re.I)

    counts: dict[str, int] = {}
    if redact:
        for pattern, label in _PII:
            out, n = re.subn(pattern, f"[{label}_REDACTED]", out)
            if n:
                counts[label] = counts.get(label, 0) + n

    return InputReport(sanitised=out, injection_kinds=kinds, redactions=counts)


# --- output-side checks --------------------------------------------------

# Statements that commit the company to an action. Each must be backed by an
# actually-executed side-effecting tool call, or the draft is blocked.
_COMMITMENT: tuple[tuple[str, str], ...] = (
    (r"\bI(?:'ve| have)\s+(?:gone ahead and\s+)?(?:refunded|issued|credited|processed)", "claims-completed-action"),
    (r"\b(?:your|the)\s+refund\s+(?:has been|is being)\s+(?:issued|processed|approved)", "claims-completed-action"),
    (r"\bI(?:'ve| have)\s+(?:cancelled|canceled|deleted|disabled|reset)\b", "claims-completed-action"),
    (r"\bI\s+(?:will|can)\s+(?:refund|waive|credit)\s+(?:you|your)", "promises-money"),
    (r"\bwe(?:'ll| will)\s+(?:refund|waive|credit|compensate)", "promises-money"),
    (r"\bguarantee(?:d)?\b", "guarantee"),
    (r"\b(?:no|never any)\s+(?:charge|cost|fee)\s+(?:ever|at all)\b", "absolute-claim"),
    (r"\bby\s+(?:tomorrow|end of day|\d+\s*(?:hours?|minutes?))\b", "unbacked-eta"),
)

_CITATION = re.compile(r"\[(\d+)\]")

_NON_CLAIM = re.compile(
    r"^\s*(?:thanks|thank you|hi\b|hello\b|sorry\b|apolog|i understand|"
    r"i(?:'m| am) sorry|let me|i'll (?:check|look|follow up)|"
    r"(?:kind|best) regards|if (?:you|there)|please)",
    re.I,
)


@dataclass
class OutputReport:
    commitments: list[str] = field(default_factory=list)
    unknown_citations: list[int] = field(default_factory=list)
    cited_claims: int = 0
    total_claims: int = 0
    leaked_internal: list[str] = field(default_factory=list)

    @property
    def citation_coverage(self) -> float:
        return self.cited_claims / self.total_claims if self.total_claims else 1.0

    @property
    def clean(self) -> bool:
        return (
            not self.commitments
            and not self.unknown_citations
            and not self.leaked_internal
            and self.citation_coverage >= 0.5
        )


# Internal-only vocabulary that must never appear in a customer-facing reply.
_INTERNAL_MARKERS = (
    "support escalation runbook",
    "social engineering",
    "goodwill credit of up to",
    "agent_credit_limit",
    "internal document",
    "not customer-facing",
)


def check_output(
    draft: str, n_citations: int, side_effects_executed: bool = False
) -> OutputReport:
    report = OutputReport()

    if not side_effects_executed:
        for pattern, kind in _COMMITMENT:
            if re.search(pattern, draft, re.I):
                report.commitments.append(kind)
        report.commitments = sorted(set(report.commitments))

    refs: list[int] = []
    for sentence in (s.strip() for s in re.split(r"(?<=[.!?])\s+", draft) if s.strip()):
        found = [int(m) for m in _CITATION.findall(sentence)]
        refs.extend(found)
        if _NON_CLAIM.match(sentence) or len(sentence.split()) < 6:
            continue
        report.total_claims += 1
        if found:
            report.cited_claims += 1

    report.unknown_citations = sorted({r for r in refs if r < 1 or r > n_citations})

    lowered = draft.lower()
    report.leaked_internal = [m for m in _INTERNAL_MARKERS if m in lowered]

    return report

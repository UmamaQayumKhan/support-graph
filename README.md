# SupportGraph — a LangGraph support agent that knows when to stop

An agent that triages inbound support tickets, looks up policy and account
state with tools, drafts a grounded reply, and then **refuses to send it** if
the reply fails verification. Security reports, disputed charges, jailbreak
attempts and anything above its spending authority go to a human via a
checkpointed human-in-the-loop interrupt.

Runs with **no API keys** — a deterministic stub model drives every node and
routing branch, which is how the 107 tests and the CI safety gate execute.

```bash
pip install -r requirements-dev.txt
make test        # 107 tests, offline, <1s
make gate        # the safety gate CI runs
make ticket M="I want a refund for my monthly plan"
```

---

## The problem with most agent demos

They measure whether the answer was good. The failure modes that matter for an
agent holding tools are different:

- It confidently handles a ticket reporting **another customer's data**.
- It writes **"I've gone ahead and refunded you"** when it cannot issue refunds.
- A customer writes **"ignore all previous instructions"** and it complies.
- It reasons its way to the wrong refund answer because the policy has three
  exceptions that contradict the headline rule.
- Its tool loop never terminates and burns the rate limit.

Each of those is a control-flow or authority problem, not a prompt-quality
problem. So this agent is built around them, and the eval measures them.

---

## Graph

```
                         ┌──────────────────────────────┐
                         │                              │
 START → guard → classify ─┬─→ plan ──→ act ────────────┘  (loop, max 4)
                           │     │
                           │     └─→ draft → verify ─┬─→ respond → END
                           │                   ▲     │
                           │                   └─────┤ one repair attempt
                           └─────────────────────────┴─→ escalate → END
                                                              ▲
                                            interrupt_before ─┘ (HITL pause)
```

| node | job |
| --- | --- |
| `guard` | sanitise the untrusted customer message; strip injection, redact secrets |
| `classify` | structured triage (intent, priority, sentiment, requested amount) |
| `plan` | choose tools; drop side-effecting and duplicate calls |
| `act` | execute tools, accumulate results and citations |
| `draft` | write the reply, grounded in tool evidence, with `[n]` citations |
| `verify` | deterministic + model checks; the gate before any send |
| `escalate` | build an internal handoff note; pause for a human |
| `respond` | send the verified reply |

### Two decisions worth defending

**The verification gate is a node, not a prompt instruction.** "Check your work
before replying" is not a control-flow guarantee. Making verification a node
with its own routing means a failed draft *cannot* reach the customer — the edge
does not exist. `test_there_is_no_edge_from_draft_straight_to_respond` asserts
that structurally rather than trusting the prose.

**Policy decisions are arithmetic, not generation.** `policy.py` computes refund
eligibility and approval levels in plain Python. The model can ask the question;
it cannot decide the answer. A model that is 97% accurate on "can this be
refunded" is 3% wrong about someone's money.

---

## Measured results

From `make eval` — 31 scenarios across must-escalate, injection, refund-policy,
entitlement, routine and ambiguous categories. Offline provider, reproducible
on clone.

| metric | value | |
| --- | --- | --- |
| **escalation recall** | **1.000** | of tickets that must reach a human, all did |
| **injection catch rate** | **1.000** | all 6 jailbreak attempts routed to a human |
| **commitment leak rate** | **0.000** | no reply claimed an action the agent cannot perform |
| escalation precision | 1.000 | nothing escalated that didn't need it |
| over-escalation rate | 0.000 | no routine ticket sent to a human |
| tool correctness | 1.000 | every refund question reached the policy calculator |
| priority accuracy | 1.000 | |
| intent accuracy | 0.833 | limited by the offline stub classifier |
| auto-resolution rate | 0.613 | |

### What the guardrails are actually worth

`make ablation` re-runs with escalation routing disabled:

| metric | escalation on | escalation off |
| --- | --- | --- |
| escalation recall | **1.000** | **0.000** |
| commitment leak rate | 0.000 | 0.000 |

With the routing layer off, **all 12 tickets that must reach a human get an
autonomous reply instead** — including cross-tenant data exposure and disputed
charges. The leak rate staying at zero is the useful second finding: the
output-side guardrails in `verify` are independent of the routing layer, so the
two failures do not share a cause. That is defence in depth, demonstrated rather
than claimed.

### Three safety bugs the eval found

Every one of these passed a manual read-through of the code and was caught only
by running the scenarios:

| bug | symptom | fix |
| --- | --- | --- |
| Substring triggers missed a spelling variant | *"a charge we did not authorise"* was handled autonomously; the list had `did not authorize` and `didn't authorise` but not that combination | triggers are regexes covering spelling and contraction variants; `authori[sz]e`, not `authoris[ez]` |
| No domain-outage trigger | *"Our entire domain cannot log in. Production is down."* got an autonomous reply, despite being runbook escalation trigger #3 | added domain-wide auth failure and production outage patterns |
| No amount-based authority routing | a **$5,000** goodwill-credit request was handled autonomously because no phrase matched | routing on the requested amount, at the finance ceiling |

The third fix needed a second iteration, and the reasoning is the interesting
part. Routing *any* request above the \$200 agent limit to a human looked safer
and was worse: it escalated a \$340 overage refund, which policy refuses
outright, wasting a human on a question the calculator answers definitively. So
there are deliberately **two thresholds** — the pre-draft gate sits at the
\$1,000 support-lead ceiling, and amounts between the two proceed through the
graph where `verify` blocks any reply offering money above an agent's own
authority without naming the approval step. `test_two_authority_thresholds_are_distinct`
pins the distinction.

---

## Safety controls

| control | where | tested by |
| --- | --- | --- |
| Injection detection (10 pattern classes) | `guard` | `test_guardrails.py` |
| Secret / card / password redaction | `guard` | `test_secrets_and_cards_are_redacted` |
| Hard escalation triggers (6 categories) | `classify` routing | `test_policy.py` |
| Commitment detection ("I've refunded you") | `verify` | `test_commitments_are_detected` |
| Internal-language leak detection | `verify` | `test_internal_language_leak_is_caught` |
| Citation validity | `verify` | `test_citation_to_missing_evidence_is_caught` |
| Side-effect boundary (approval token) | `tools` | `test_side_effecting_tool_blocked_without_approval` |
| Tool loop bound (max 4 iterations) | `plan` | `test_tool_loop_is_bounded` |
| Duplicate call suppression | `plan` | `test_duplicate_tool_calls_are_suppressed` |

False positives are tested as carefully as true positives. A guardrail that
flags *"Our legal team asked me to check your SOC 2 report is current"* as legal
exposure burns a human on a documentation question — that exact string is a test
case, and fixing it is why the legal trigger now requires counsel to be *acting*
rather than merely mentioned.

---

## Tools

Eight tools; the two that mutate anything are marked and refuse to run without
an approval token that only the HITL resume path supplies.

| tool | side effects |
| --- | --- |
| `search_kb` | — |
| `lookup_account` | — |
| `get_ticket_history` | — |
| `evaluate_refund` | — (delegates to `policy.py` arithmetic) |
| `check_entitlement` | — |
| `check_credit_authority` | — |
| `issue_credit` | **yes** — requires human approval |
| `create_escalation` | **yes** — requires human approval |

The tool layer never raises. A hallucinated tool name returns an error result
listing the real tools; a missing argument is reported; an invented extra
argument is dropped with a note. All three are tested, because an exception in
the tool layer ends the run and the customer gets nothing.

---

## Human in the loop

```bash
# 1. ticket arrives, graph pauses at the escalate checkpoint
curl -s localhost:8000/tickets -H 'content-type: application/json' \
  -d '{"message":"We can see another workspace'\''s data","workspace_id":"ws_demo"}'
# → {"status":"awaiting_human","thread_id":"t_...","escalation_reasons":[...]}

# 2. reviewer inspects the paused state (internal view: unsent draft + note)
curl -s localhost:8000/threads/t_.../

# 3. reviewer resumes; this is the only path that unlocks side-effecting tools
curl -s localhost:8000/threads/t_.../resume -H 'content-type: application/json' \
  -d '{"decision":"confirmed, routing to finance","reviewer":"lead@northwind"}'
```

The graph resumes from the checkpoint with the same state rather than re-running
from scratch. That is the whole reason to use a checkpointed graph instead of a
`while` loop around an LLM call.

---

## API

| method | path | purpose |
| --- | --- | --- |
| `POST` | `/tickets` | run a ticket; returns a reply or `awaiting_human` |
| `GET` | `/threads/{id}` | internal reviewer view of a paused thread |
| `POST` | `/threads/{id}/resume` | resume with a human decision |
| `GET` | `/graph` | compiled topology (nodes and edges) |
| `GET` | `/healthz` | KB size, tool list, provider, escalation state |

Customer-facing responses go through a projection that strips the escalation
note, the unsent draft and the trace — the note quotes runbook language and
approval limits, so shipping the raw state would itself be a leak.

---

## Layout

```
src/supportgraph/
  state.py       TypedDict state + validated Pydantic decision models
  policy.py      refund arithmetic, authority thresholds, escalation triggers
  tools.py       8 tools, argument validation, side-effect boundary
  kb.py          BM25 policy retrieval, one passage per section
  guardrails.py  injection, PII, commitment and leak detection
  nodes.py       guard · classify · plan · act · draft · verify · escalate · respond
  graph.py       StateGraph assembly, conditional edges, checkpointing
  providers.py   Groq / Gemini / OpenAI / offline + validated structured output
  api.py         FastAPI including HITL resume
  cli.py         run a ticket and print the trace
eval/
  scenarios.jsonl  31 scenarios incl. 6 jailbreaks and 5 must-escalate cases
  run_eval.py      failure-mode metrics, guardrail ablation, CI gate
tests/             107 tests, fully offline
```

## Design notes

**Structured output is validated, with a repair retry.** `complete_structured`
parses into a Pydantic model; on a validation error it retries once with the
error and the schema as a repair instruction, then falls back to a safe default
so the graph degrades instead of crashing. Every parse problem lands in the
trace. A graph that routes on unvalidated model output fails the first time the
model phrases something differently.

**Deterministic checks run before the model-based ones in `verify`.** Commitment
detection, citation validity and the refund cross-check are regex and arithmetic
— they are not opinions and must not be delegated. The LLM judgement only
tightens the result; it can never loosen it.

**The offline stub is scenario-aware on purpose.** It inspects the prompt and
returns plausible structured JSON for each node, so the tests exercise the real
graph — all eight nodes, every conditional edge, the loop bound, the HITL pause
— rather than mocking the thing under test.

## Known limits

- `intent_accuracy` of 0.833 reflects the offline stub's keyword classifier, not
  a real model. Set `GROQ_API_KEY` and re-run `make eval` for a meaningful
  number.
- 31 scenarios is enough to catch the failure classes above, not enough to
  estimate production deflection rate.
- `MemorySaver` checkpointing is in-process; a real deployment needs the Postgres
  or Redis checkpointer so a pause survives a restart.
- The KB is BM25-only. Past a few hundred documents it needs a dense arm —
  `kb.py` keeps the same interface as the companion AtlasRAG service so it can
  be swapped for a call to it.
- Policy is hardcoded to one fictional company's rules. Real deployment needs
  these loaded from a versioned source with an audit trail.

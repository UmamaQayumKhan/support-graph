"""Provider-agnostic LLM access with validated structured output.

The important function here is `complete_structured`. Agent graphs route on
model output, so "the model returned something that does not parse" has to be a
handled case, not an exception that kills the run. It retries with a repair
prompt carrying the validation error, then falls back to the caller's default.

The offline provider returns deterministic, scenario-aware structured responses,
which is what lets the full graph -- every node, every routing branch -- be
tested with no API key.
"""

from __future__ import annotations

import json
import random
import re
import time
from dataclasses import dataclass, field
from typing import Any, Protocol, Sequence, TypeVar

import httpx
from pydantic import BaseModel, ValidationError

from .config import Settings, get_settings

T = TypeVar("T", bound=BaseModel)


class ProviderError(RuntimeError):
    pass


class LLM(Protocol):
    name: str
    model: str

    def complete(self, messages: Sequence[dict[str, str]], **kw: Any) -> str: ...


def _retry(client: httpx.Client, method: str, url: str, *, tries: int, **kw: Any) -> httpx.Response:
    last: Exception | None = None
    for attempt in range(tries):
        try:
            resp = client.request(method, url, **kw)
        except (httpx.TimeoutException, httpx.TransportError) as exc:
            last = exc
        else:
            if resp.status_code < 400:
                return resp
            if resp.status_code != 429 and resp.status_code < 500:
                raise ProviderError(f"{resp.status_code}: {resp.text[:300]}")
            wait = resp.headers.get("retry-after")
            if wait:
                try:
                    time.sleep(min(float(wait), 30.0))
                    continue
                except ValueError:
                    pass
            last = ProviderError(f"{resp.status_code}: {resp.text[:200]}")
        if attempt < tries - 1:
            time.sleep(min(2**attempt + random.random(), 20.0))
    raise ProviderError(f"{url} failed after {tries} attempts: {last}")


@dataclass
class OpenAICompatLLM:
    api_key: str
    model: str
    base_url: str
    timeout: float = 60.0
    tries: int = 3
    name: str = "openai-compatible"

    def __post_init__(self) -> None:
        self._client = httpx.Client(
            timeout=self.timeout,
            headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
        )

    def complete(self, messages: Sequence[dict[str, str]], **kw: Any) -> str:
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": list(messages),
            "temperature": kw.get("temperature", 0.1),
            "max_tokens": kw.get("max_tokens", 900),
        }
        if kw.get("json_mode"):
            payload["response_format"] = {"type": "json_object"}
        resp = _retry(
            self._client, "POST", f"{self.base_url}/chat/completions", tries=self.tries, json=payload
        )
        return resp.json()["choices"][0]["message"]["content"] or ""


@dataclass
class GeminiLLM:
    api_key: str
    model: str
    base_url: str
    timeout: float = 60.0
    tries: int = 3
    name: str = "gemini"

    def __post_init__(self) -> None:
        self._client = httpx.Client(timeout=self.timeout)

    def complete(self, messages: Sequence[dict[str, str]], **kw: Any) -> str:
        system = [m["content"] for m in messages if m["role"] == "system"]
        contents = [
            {"role": "model" if m["role"] == "assistant" else "user", "parts": [{"text": m["content"]}]}
            for m in messages
            if m["role"] != "system"
        ]
        cfg: dict[str, Any] = {
            "temperature": kw.get("temperature", 0.1),
            "maxOutputTokens": kw.get("max_tokens", 900),
        }
        if kw.get("json_mode"):
            cfg["responseMimeType"] = "application/json"
        body: dict[str, Any] = {"contents": contents, "generationConfig": cfg}
        if system:
            body["systemInstruction"] = {"parts": [{"text": "\n\n".join(system)}]}
        resp = _retry(
            self._client,
            "POST",
            f"{self.base_url}/models/{self.model}:generateContent",
            tries=self.tries,
            params={"key": self.api_key},
            json=body,
        )
        parts = []
        for cand in resp.json().get("candidates", []):
            for part in cand.get("content", {}).get("parts", []):
                if "text" in part:
                    parts.append(part["text"])
        return "".join(parts)


@dataclass
class OfflineLLM:
    """Deterministic scenario-aware stub.

    It inspects the prompt for the shape of answer the node needs and returns
    plausible structured JSON. That makes the whole graph -- classification,
    planning, drafting, verification, every routing branch -- testable offline.
    Without it, the tests would mock the graph instead of exercising it.
    """

    name: str = "offline"
    model: str = "offline-stub"
    calls: list[dict[str, Any]] = field(default_factory=list)

    def complete(self, messages: Sequence[dict[str, str]], **kw: Any) -> str:
        system = " ".join(m["content"] for m in messages if m["role"] == "system").lower()
        user = " ".join(m["content"] for m in messages if m["role"] == "user")
        self.calls.append({"system": system[:80], "user": user[:200]})
        low = user.lower()

        if "classify" in system:
            intent, priority = "other", "P3"
            amount = 0.0
            if any(w in low for w in ("refund", "money back", "charged")):
                intent, priority = "refund", "P2"
            elif any(w in low for w in ("invoice", "billing", "payment", "card")):
                intent, priority = "billing", "P3"
            elif any(w in low for w in ("sso", "saml", "login", "log in", "locked out")):
                intent, priority = "account_access", "P2"
            elif any(w in low for w in ("breach", "hacked", "vulnerability", "another workspace")):
                intent, priority = "security", "P1"
            elif any(w in low for w in ("api", "429", "rate limit", "webhook", "error")):
                intent, priority = "technical", "P3"
            elif "feature" in low or "could you add" in low:
                intent, priority = "feature_request", "P4"
            match = re.search(r"\$\s?([\d,]+(?:\.\d{2})?)", user)
            if match:
                amount = float(match.group(1).replace(",", ""))
            sentiment = "angry" if any(w in low for w in ("unacceptable", "furious", "ridiculous")) else (
                "frustrated" if any(w in low for w in ("still", "again", "third time")) else "calm"
            )
            return json.dumps(
                {
                    "intent": intent,
                    "priority": priority,
                    "sentiment": sentiment,
                    "product_area": intent,
                    "summary": user[:160],
                    "confidence": 0.82,
                    "requested_amount_usd": amount,
                    "reasoning": "offline deterministic classification",
                }
            )

        if "plan" in system and "tool" in system:
            # One round of lookups, then stop. Mirrors what a competent model
            # does and keeps the loop-limit branch reachable in tests.
            if "tool results so far" in low and "none" not in low[-400:]:
                return json.dumps({"calls": [], "done": True, "reasoning": "have enough"})
            calls = [{"name": "search_kb", "arguments": {"query": user[:120]}, "reason": "policy"}]

            ws = re.search(r"\b(ws_[a-z0-9_]+)", user)
            if ws or any(w in low for w in ("workspace", "account", "refund", "charge", "tier")):
                calls.append(
                    {
                        "name": "lookup_account",
                        "arguments": {"workspace_id": ws.group(1) if ws else "ws_demo"},
                        "reason": "tier and billing state",
                    }
                )

            # A refund question must reach the policy calculator, never the
            # model's own judgement. Mirroring that here keeps the offline path
            # exercising the same control flow the keyed path takes.
            if any(w in low for w in ("refund", "money back", "charged", "overage", "credit")):
                plan_term = "monthly" if "monthly" in low else ("annual" if "annual" in low else "monthly")
                calls.append(
                    {
                        "name": "evaluate_refund",
                        "arguments": {
                            "plan": plan_term,
                            "days_into_term": 8,
                            "amount_usd": 0.0,
                            "is_overage": "overage" in low,
                            "via_reseller": "reseller" in low,
                        },
                        "reason": "authoritative refund eligibility",
                    }
                )
            return json.dumps({"calls": calls, "done": False, "reasoning": "gather context"})

        if "verify" in system:
            return json.dumps(
                {
                    "grounded": True,
                    "policy_compliant": True,
                    "within_authority": True,
                    "makes_commitment": False,
                    "issues": [],
                    "confidence": 0.8,
                }
            )

        if kw.get("json_mode"):
            return "{}"

        return (
            "Thanks for getting in touch, and sorry for the trouble. "
            "Based on our policy [1], here is what applies to your account. "
            "I'll follow up if anything changes."
        )


def build_llm(settings: Settings | None = None) -> LLM:
    s = settings or get_settings()
    p = s.llm_provider.lower()
    if p == "offline" or s.offline:
        return OfflineLLM()
    if p == "groq":
        return OpenAICompatLLM(
            s.groq_api_key, s.groq_model, s.groq_base_url, s.request_timeout, s.max_retries, "groq"
        )
    if p == "gemini":
        return GeminiLLM(
            s.gemini_api_key, s.gemini_model, s.gemini_base_url, s.request_timeout, s.max_retries
        )
    if p in {"openai", "openai-compatible", "vllm", "ollama"}:
        return OpenAICompatLLM(
            s.openai_api_key, s.openai_model, s.openai_base_url, s.request_timeout, s.max_retries, p
        )
    raise ProviderError(f"unknown LLM_PROVIDER: {s.llm_provider}")


def _extract_json(text: str) -> str | None:
    text = text.strip()
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if fenced:
        return fenced.group(1)
    brace = re.search(r"\{.*\}", text, re.DOTALL)
    return brace.group(0) if brace else None


def complete_structured(
    llm: LLM,
    model_cls: type[T],
    messages: Sequence[dict[str, str]],
    *,
    default: T | None = None,
    retries: int = 2,
    **kw: Any,
) -> tuple[T, list[str]]:
    """Call the model and validate into `model_cls`.

    Returns (instance, problems). On a parse or validation failure it retries
    with the error appended as a repair instruction -- which recovers the common
    case of a model emitting a near-correct object -- then falls back to
    `default` so the graph degrades instead of crashing. Every problem is
    returned so it lands in the trace rather than disappearing.
    """
    problems: list[str] = []
    convo = list(messages)

    for attempt in range(max(1, retries + 1)):
        raw = llm.complete(convo, json_mode=True, **kw)
        blob = _extract_json(raw)
        if blob is None:
            problems.append(f"attempt {attempt + 1}: no JSON object in response")
        else:
            try:
                return model_cls.model_validate_json(blob), problems
            except ValidationError as exc:
                problems.append(f"attempt {attempt + 1}: {exc.error_count()} validation error(s)")
                detail = "; ".join(
                    f"{'.'.join(str(p) for p in e['loc'])}: {e['msg']}" for e in exc.errors()[:4]
                )
            except json.JSONDecodeError as exc:
                problems.append(f"attempt {attempt + 1}: malformed JSON ({exc.msg})")
                detail = exc.msg

            if attempt < retries:
                convo = list(messages) + [
                    {"role": "assistant", "content": raw[:1500]},
                    {
                        "role": "user",
                        "content": (
                            f"That response was not valid. Problem: {detail}. "
                            f"Reply with ONLY a JSON object matching this schema:\n"
                            f"{json.dumps(model_cls.model_json_schema())[:1200]}"
                        ),
                    },
                ]

    if default is not None:
        problems.append("fell back to default after exhausting retries")
        return default, problems
    return model_cls(), problems

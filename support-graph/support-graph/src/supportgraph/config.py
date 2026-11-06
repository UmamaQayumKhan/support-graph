from __future__ import annotations

import os
from dataclasses import dataclass, field
from functools import lru_cache


def _env(k: str, d: str) -> str:
    return os.getenv(k, d)


def _env_f(k: str, d: float) -> float:
    try:
        return float(os.getenv(k, str(d)))
    except ValueError:
        return d


def _env_i(k: str, d: int) -> int:
    try:
        return int(os.getenv(k, str(d)))
    except ValueError:
        return d


def _env_b(k: str, d: bool) -> bool:
    raw = os.getenv(k)
    return d if raw is None else raw.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class Settings:
    llm_provider: str = field(default_factory=lambda: _env("LLM_PROVIDER", "groq"))

    groq_api_key: str = field(default_factory=lambda: _env("GROQ_API_KEY", ""))
    groq_model: str = field(default_factory=lambda: _env("GROQ_MODEL", "llama-3.3-70b-versatile"))
    groq_base_url: str = field(
        default_factory=lambda: _env("GROQ_BASE_URL", "https://api.groq.com/openai/v1")
    )

    gemini_api_key: str = field(default_factory=lambda: _env("GEMINI_API_KEY", ""))
    gemini_model: str = field(default_factory=lambda: _env("GEMINI_MODEL", "gemini-2.0-flash"))
    gemini_base_url: str = field(
        default_factory=lambda: _env(
            "GEMINI_BASE_URL", "https://generativelanguage.googleapis.com/v1beta"
        )
    )

    openai_api_key: str = field(default_factory=lambda: _env("OPENAI_API_KEY", ""))
    openai_model: str = field(default_factory=lambda: _env("OPENAI_MODEL", "gpt-4o-mini"))
    openai_base_url: str = field(
        default_factory=lambda: _env("OPENAI_BASE_URL", "https://api.openai.com/v1")
    )

    temperature: float = field(default_factory=lambda: _env_f("TEMPERATURE", 0.1))
    max_tokens: int = field(default_factory=lambda: _env_i("MAX_TOKENS", 900))
    request_timeout: float = field(default_factory=lambda: _env_f("REQUEST_TIMEOUT", 60.0))
    max_retries: int = field(default_factory=lambda: _env_i("MAX_RETRIES", 3))
    # Structured-output parse failures are retried with a repair prompt before
    # the node gives up and falls back to a safe default.
    structured_retries: int = field(default_factory=lambda: _env_i("STRUCTURED_RETRIES", 2))

    kb_dir: str = field(default_factory=lambda: _env("KB_DIR", "data/kb"))
    accounts_path: str = field(default_factory=lambda: _env("ACCOUNTS_PATH", "data/accounts.json"))
    tickets_path: str = field(default_factory=lambda: _env("TICKETS_PATH", "data/tickets.jsonl"))

    # When false the graph answers autonomously even on escalation triggers.
    # Only used to measure how much worse that is; never a production setting.
    enable_escalation: bool = field(default_factory=lambda: _env_b("ENABLE_ESCALATION", True))

    @property
    def offline(self) -> bool:
        return self.llm_provider == "offline" or not (
            self.groq_api_key or self.gemini_api_key or self.openai_api_key
        )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()


def reset_settings_cache() -> None:
    get_settings.cache_clear()

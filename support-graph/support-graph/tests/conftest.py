from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))


@pytest.fixture(autouse=True)
def offline_env(monkeypatch):
    """Whole graph runs offline: deterministic stub model, no network, no keys.
    Every routing branch is reachable, so these are real integration tests
    rather than mocks of the thing under test."""
    from supportgraph.config import reset_settings_cache

    monkeypatch.setenv("LLM_PROVIDER", "offline")
    monkeypatch.setenv("KB_DIR", str(ROOT / "data" / "kb"))
    monkeypatch.setenv("ACCOUNTS_PATH", str(ROOT / "data" / "accounts.json"))
    monkeypatch.setenv("TICKETS_PATH", str(ROOT / "data" / "tickets.jsonl"))
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("ENABLE_ESCALATION", raising=False)
    reset_settings_cache()
    yield
    reset_settings_cache()


@pytest.fixture
def settings():
    from supportgraph.config import Settings

    return Settings()


@pytest.fixture
def registry(settings):
    from supportgraph.tools import load_registry

    return load_registry(settings.kb_dir, settings.accounts_path, settings.tickets_path)


@pytest.fixture
def agent(settings):
    from supportgraph.graph import SupportAgent

    return SupportAgent(settings)


@pytest.fixture
def hitl_agent(settings):
    from supportgraph.graph import SupportAgent

    return SupportAgent(settings, hitl=True)


def run(agent, message: str, **kw):
    ticket = {
        "ticket_id": kw.pop("ticket_id", "t_test"),
        "message": message,
        "workspace_id": kw.pop("workspace_id", "ws_demo"),
        "customer_id": kw.pop("customer_id", "cust_001"),
    }
    return agent.run(ticket, thread_id=ticket["ticket_id"])

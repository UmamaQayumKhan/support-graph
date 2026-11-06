"""SupportGraph — a LangGraph agent that triages support tickets and knows when to stop."""

from .config import Settings, get_settings
from .graph import SupportAgent, build_graph
from .kb import KnowledgeBase, load_kb
from .policy import evaluate_refund, has_entitlement, hard_escalation_triggers
from .state import AgentState, Classification, Priority, Verification
from .tools import ToolRegistry, load_registry

__version__ = "1.0.0"

__all__ = [
    "AgentState",
    "Classification",
    "KnowledgeBase",
    "Priority",
    "Settings",
    "SupportAgent",
    "ToolRegistry",
    "Verification",
    "build_graph",
    "evaluate_refund",
    "get_settings",
    "hard_escalation_triggers",
    "has_entitlement",
    "load_kb",
    "load_registry",
    "__version__",
]

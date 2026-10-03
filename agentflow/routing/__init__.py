"""Routing layer: automatic intent detection and automatic model selection."""

from __future__ import annotations

from agentflow.routing.intent import (
    PROTOTYPES,
    RULE_PRIORITY,
    RULES,
    Intent,
    IntentDetector,
    IntentResult,
    tokenize,
)
from agentflow.routing.router import (
    INTENT_TASK,
    TASK_CAPABILITIES,
    AutoRouter,
    RoutingDecision,
)

__all__ = [
    "INTENT_TASK",
    "PROTOTYPES",
    "RULE_PRIORITY",
    "RULES",
    "TASK_CAPABILITIES",
    "AutoRouter",
    "Intent",
    "IntentDetector",
    "IntentResult",
    "RoutingDecision",
    "tokenize",
]
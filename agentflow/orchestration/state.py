"""LangGraph state.

The state is a reducer-annotated TypedDict so parallel branches can write to the
same field without clobbering each other. That annotation is the whole reason
LangGraph is used here rather than a hand-rolled dict: fan-out writes are merged
by ``add``/``operator.add`` instead of last-writer-wins.
"""

from __future__ import annotations

import operator
from dataclasses import dataclass, field
from typing import Annotated, Any, TypedDict

from agentflow.agents.base import AgentResult
from agentflow.llm.types import Message
from agentflow.routing.intent import IntentResult
from agentflow.routing.router import RoutingDecision


def merge_results(
    left: dict[str, AgentResult], right: dict[str, AgentResult]
) -> dict[str, AgentResult]:
    return {**left, **right}


def merge_messages(left: list[Message], right: list[Message]) -> list[Message]:
    return [*left, *right]


class WorkflowState(TypedDict, total=False):
    """Shared state threaded through the graph.

    ``agent_results`` and ``trace`` use additive reducers because several nodes
    may write them concurrently; everything else is single-writer.
    """

    # input
    task: str
    session_id: str
    history: Annotated[list[Message], merge_messages]

    # routing
    intent: IntentResult
    routing: RoutingDecision
    selected_agents: list[str]

    # fan-out / fan-in
    agent_results: Annotated[dict[str, AgentResult], merge_results]
    agent_failures: dict[str, str]

    # output
    answer: str
    ground_truth: list[str]
    escalated: bool
    iteration: int

    # observability
    trace: Annotated[list[dict[str, Any]], operator.add]
    errors: Annotated[list[str], operator.add]


@dataclass(slots=True)
class WorkflowOutcome:
    """Result of one orchestrator run, including the full execution trace."""

    answer: str
    intent: IntentResult | None
    routing: RoutingDecision | None
    agents_run: list[str] = field(default_factory=list)
    agent_results: dict[str, AgentResult] = field(default_factory=dict)
    agent_failures: dict[str, str] = field(default_factory=dict)
    escalated: bool = False
    iterations: int = 0
    errors: list[str] = field(default_factory=list)
    tokens_used: int = 0
    tokens_saved: int = 0
    cost_usd: float = 0.0
    guardrail_findings: list[dict[str, Any]] = field(default_factory=list)
    trace: list[dict[str, Any]] = field(default_factory=list)
    wall_ms: float = 0.0
    #: serial_ms / wall_ms for the last fan-out; 1.0 when nothing ran in parallel.
    parallel_speedup: float = 1.0

    @property
    def ok(self) -> bool:
        return not self.errors and not self.agent_failures and bool(self.answer.strip())

    def to_dict(self) -> dict[str, Any]:
        return {
            "answer": self.answer,
            "ok": self.ok,
            "intent": self.intent.to_dict() if self.intent else None,
            "routing": self.routing.to_dict() if self.routing else None,
            "agents_run": list(self.agents_run),
            "agent_failures": dict(self.agent_failures),
            "agent_results": {k: v.to_dict() for k, v in self.agent_results.items()},
            "escalated": self.escalated,
            "iterations": self.iterations,
            "errors": list(self.errors),
            "tokens_used": self.tokens_used,
            "tokens_saved": self.tokens_saved,
            "cost_usd": round(self.cost_usd, 6),
            "guardrail_findings": list(self.guardrail_findings),
            "trace": list(self.trace),
            "wall_ms": round(self.wall_ms, 2),
            "parallel_speedup": self.parallel_speedup,
        }


__all__ = ["WorkflowOutcome", "WorkflowState", "merge_messages", "merge_results"]
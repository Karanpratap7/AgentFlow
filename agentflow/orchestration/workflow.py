"""The compiled agentic workflow.

Builds the LangGraph ``StateGraph`` from the supervisor's nodes and compiles it
once. Two details matter:

* **Conditional fan-in.** ``fan_out`` is followed by an always-taken ``fan_in``
  edge. That explicit join node is what guarantees LangGraph waits for every
  parallel branch before ``synthesize`` reads ``agent_results``, instead of
  racing the aggregator.
* **Bounded escalation.** ``verify`` routes back to ``fan_out`` at most once,
  driven by ``escalated`` and the iteration counter, so a request cannot loop.
"""

from __future__ import annotations

import logging
from typing import Any

from agentflow.agents.pool import AgentRegistry
from agentflow.config import Settings
from agentflow.config import settings as default_settings
from agentflow.guardrails.pipeline import GuardrailPipeline
from agentflow.llm.types import LLMClient
from agentflow.orchestration.state import WorkflowOutcome, WorkflowState
from agentflow.orchestration.supervisor import Supervisor
from agentflow.routing.intent import IntentDetector
from agentflow.routing.router import AutoRouter
from agentflow.tokens.budget import TokenBudget
from agentflow.tools import ToolRegistry

log = logging.getLogger("agentflow.workflow")


def build_graph(supervisor: Supervisor) -> Any:
    """Compile the supervisor into a LangGraph state machine."""
    from langgraph.graph import END, START, StateGraph

    graph = StateGraph(WorkflowState)

    graph.add_node("guardrail", supervisor.guardrail)
    graph.add_node("detect_intent", supervisor.detect_intent)
    graph.add_node("route", supervisor.route)
    graph.add_node("plan", supervisor.plan)
    graph.add_node("fan_out", supervisor.fan_out)
    graph.add_node("fan_in", _fan_in)
    graph.add_node("synthesize", supervisor.synthesize)
    graph.add_node("verify", supervisor.verify)

    graph.add_edge(START, "guardrail")
    graph.add_conditional_edges(
        "guardrail",
        _after_guardrail,
        {"blocked": END, "ok": "detect_intent"},
    )
    graph.add_edge("detect_intent", "route")
    graph.add_edge("route", "plan")
    graph.add_edge("plan", "fan_out")
    graph.add_edge("fan_out", "fan_in")
    graph.add_edge("fan_in", "synthesize")
    graph.add_edge("synthesize", "verify")
    graph.add_conditional_edges(
        "verify",
        _after_verify,
        {"escalate": "fan_out", "done": END},
    )

    return graph.compile()


def _fan_in(state: WorkflowState) -> dict[str, Any]:
    """Explicit parallel join. Aggregates what the branches produced."""
    results = dict(state.get("agent_results") or {})
    failures = dict(state.get("agent_failures") or {})
    return {
        "trace": [
            {
                "node": "fan_in",
                "collected": sorted(results),
                "failed": sorted(failures),
            }
        ]
    }


def _after_guardrail(state: WorkflowState) -> str:
    return "blocked" if state.get("errors") else "ok"


def _after_verify(state: WorkflowState) -> str:
    iteration = int(state.get("iteration", 0))
    if state.get("escalated") and iteration < 2:
        return "escalate"
    return "done"


class AgentFlowOrchestrator:
    """Public entry point: ``run(task) -> WorkflowOutcome``."""

    def __init__(
        self,
        client: LLMClient,
        *,
        registry: AgentRegistry,
        tools: ToolRegistry,
        settings: Settings | None = None,
        guardrails: GuardrailPipeline | None = None,
        router: AutoRouter | None = None,
        intent_detector: IntentDetector | None = None,
        budget: TokenBudget | None = None,
        forced_agents: list[str] | None = None,
        rag: Any = None,
        thread_name: str = "agentflow",
    ) -> None:
        self.settings = settings or default_settings
        self.supervisor = Supervisor(
            client,
            registry=registry,
            tools=tools,
            settings=self.settings,
            guardrails=guardrails,
            router=router,
            intent_detector=intent_detector,
            rag=rag,
            budget=budget,
            forced_agents=forced_agents,
        )
        self.graph = build_graph(self.supervisor)
        self.thread_name = thread_name

    def run(
        self,
        task: str,
        *,
        session_id: str = "default",
        history: list[Any] | None = None,
    ) -> WorkflowOutcome:
        from agentflow.llm.types import now_ms

        started = now_ms()
        initial: WorkflowState = {
            "task": task,
            "session_id": session_id,
            "history": list(history or []),
            "agent_results": {},
            "agent_failures": {},
            "trace": [],
            "errors": [],
            "iteration": 0,
            "escalated": False,
        }
        config = {
            "configurable": {"thread_id": f"{self.thread_name}:{session_id}"},
            "recursion_limit": 25,
        }
        final = self.graph.invoke(initial, config)

        results = dict(final.get("agent_results") or {})
        parallel = self.supervisor.last_parallel
        speedup = parallel.speedup if parallel is not None else 1.0
        return WorkflowOutcome(
            answer=final.get("answer", ""),
            intent=final.get("intent"),
            routing=final.get("routing"),
            agents_run=sorted(results) or list(final.get("selected_agents") or []),
            agent_results=results,
            agent_failures=dict(final.get("agent_failures") or {}),
            escalated=bool(final.get("escalated")),
            iterations=int(final.get("iteration", 0)),
            errors=list(final.get("errors") or []),
            tokens_used=sum(r.usage_total for r in results.values())
            + int(final.get("tokens_used", 0)),
            tokens_saved=sum(r.tokens_saved for r in results.values())
            + int(final.get("tokens_saved", 0)),
            cost_usd=self.supervisor.budget.snapshot().spent_usd,
            guardrail_findings=[
                f
                for r in results.values()
                for f in r.guardrail_findings
            ],
            trace=list(final.get("trace") or []),
            wall_ms=now_ms() - started,
            parallel_speedup=speedup,
        )

    def stream(self, task: str, **kwargs: Any) -> Any:
        """Stream node-by-node updates. Useful for a SSE endpoint."""
        return self.graph.stream(
            {
                "task": task,
                "session_id": kwargs.get("session_id", "default"),
                "history": list(kwargs.get("history") or []),
                "agent_results": {},
                "agent_failures": {},
                "trace": [],
                "errors": [],
                "iteration": 0,
                "escalated": False,
            },
            config={
                "configurable": {"thread_id": f"{self.thread_name}:{kwargs.get('session_id', 'default')}"},
                "recursion_limit": 25,
            },
        )

    @property
    def budget(self) -> TokenBudget:
        return self.supervisor.budget

    def graph_repr(self) -> str:
        """ASCII rendering of the compiled graph, for the CLI."""
        try:
            return str(self.graph.get_graph().draw_ascii())
        except Exception:  # noqa: BLE001 - graphviz may not be installed
            return (
                "guardrail -> detect_intent -> route -> plan -> fan_out "
                "-> fan_in -> synthesize -> verify -> END\n"
                "verify --escalate--> fan_out (bounded to 1 retry)"
            )


__all__ = ["AgentFlowOrchestrator", "build_graph"]
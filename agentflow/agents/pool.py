"""Agent registry and the parallel execution engine.

``AgentPool.run`` is the fan-out half of the orchestration: it decides which
specialists a task needs, runs them concurrently, and isolates failures. Agents
are independent by construction (no shared mutable conversation state), so a
thread pool is the correct primitive here — each thread gets its own message
list and its own tool budget.
"""

from __future__ import annotations

import logging
import threading
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from concurrent.futures import TimeoutError as FuturesTimeout
from dataclasses import dataclass, field
from typing import Any

from agentflow.agents.base import Agent, AgentResult
from agentflow.agents.specialists import (
    CodeAgent,
    ResearchAgent,
    SQLAgent,
    SynthesizerAgent,
    TriageAgent,
)
from agentflow.routing.intent import Intent, IntentResult
from agentflow.telemetry import metrics
from agentflow.tokens.budget import TokenBudget

log = logging.getLogger("agentflow.agents.pool")


class AgentRegistry:
    """Name -> agent factory, so the orchestrator can build agents on demand."""

    def __init__(self) -> None:
        self._factories: dict[str, type[Agent]] = {}
        self._instances: dict[str, Agent] = {}
        self._lock = threading.Lock()

    def register(self, name: str, factory: type[Agent]) -> None:
        self._factories[name] = factory
        self._instances.pop(name, None)

    def names(self) -> list[str]:
        return sorted(self._factories)

    def build(self, name: str, **kwargs: Any) -> Agent:
        with self._lock:
            if name in self._instances:
                return self._instances[name]
            try:
                factory = self._factories[name]
            except KeyError as exc:
                raise KeyError(f"unknown agent {name!r}; have {self.names()}") from exc
            agent = factory(**self._applicable(factory, kwargs))
            self._instances[name] = agent
            return agent

    @staticmethod
    def _applicable(factory: Any, kwargs: dict[str, Any]) -> dict[str, Any]:
        """Drop keyword arguments a factory does not declare.

        Callers pass one shared dependency bag (client, tools, router, and
        optionally ``rag``) to every specialist. Only ResearchAgent takes ``rag``;
        forwarding it blindly would raise ``TypeError`` on the other four.
        """
        import inspect

        parameters = inspect.signature(factory).parameters
        if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in parameters.values()):
            return kwargs
        return {key: value for key, value in kwargs.items() if key in parameters}

    def __len__(self) -> int:
        return len(self._factories)


def default_registry(**kwargs: Any) -> AgentRegistry:
    """The five production agents, wired with the shared client and toolset."""
    registry = AgentRegistry()
    for factory in (TriageAgent, ResearchAgent, CodeAgent, SQLAgent, SynthesizerAgent):
        registry.register(factory.name, factory)
    return registry


@dataclass(slots=True)
class ParallelOutcome:
    """Fan-in payload for the orchestrator."""

    results: dict[str, AgentResult] = field(default_factory=dict)
    failures: dict[str, str] = field(default_factory=dict)
    wall_ms: float = 0.0
    serial_ms: float = 0.0

    @property
    def speedup(self) -> float:
        return round(self.serial_ms / self.wall_ms, 2) if self.wall_ms > 0 else 1.0

    @property
    def ok(self) -> bool:
        return not self.failures and all(r.ok for r in self.results.values())

    def total_tokens(self) -> int:
        return sum(r.usage_total for r in self.results.values())

    def total_saved(self) -> int:
        return sum(r.tokens_saved for r in self.results.values())

    def total_cost(self) -> float:
        return sum(r.cost_usd for r in self.results.values())

    def to_dict(self) -> dict[str, Any]:
        return {
            "agents": list(self.results),
            "failures": dict(self.failures),
            "wall_ms": round(self.wall_ms, 2),
            "serial_ms": round(self.serial_ms, 2),
            "speedup": self.speedup,
            "total_tokens": self.total_tokens(),
            "tokens_saved": self.total_saved(),
            "total_cost_usd": round(self.total_cost(), 6),
        }


class AgentPool:
    """Runs a set of agents concurrently and collects their results."""

    def __init__(
        self,
        registry: AgentRegistry,
        *,
        max_workers: int = 4,
        timeout_s: float = 45.0,
        budget: TokenBudget | None = None,
    ) -> None:
        self.registry = registry
        self.max_workers = max_workers
        self.timeout_s = timeout_s
        self.budget = budget

    # -- planning ---------------------------------------------------------
    @staticmethod
    def plan(intent: IntentResult | None, *, force: list[str] | None = None) -> list[str]:
        """Which specialists does this request need?

        Deliberately fan-out heavy for compound work: a request that mentions
        both a symptom and a documentation question gets both the code agent and
        the researcher, because they are independent and the wall-clock win is
        real.
        """
        if force:
            return list(dict.fromkeys(force))
        if intent is None:
            return ["triage", "researcher"]

        chosen: list[str] = []
        label = intent.intent
        if label in {Intent.GREETING, Intent.CHITCHAT}:
            # Trivial: a specialist fan-out would cost more than it returns.
            return []
        if label in {Intent.KNOWLEDGE, Intent.SUMMARIZE, Intent.UNKNOWN}:
            chosen.append("researcher")
        if label is Intent.SQL:
            chosen.extend(["sql", "researcher"])
        elif label is Intent.CODE:
            chosen.extend(["coder", "researcher"])
        elif label is Intent.ANALYSIS:
            chosen.extend(["researcher", "coder"])
        elif label in {Intent.SAFETY_RISK}:
            return ["triage"]
        else:
            chosen.extend(["triage", "researcher"])
        return list(dict.fromkeys(chosen))

    # -- execution --------------------------------------------------------
    def run(
        self,
        agents: list[str],
        task: str,
        *,
        context: str = "",
        per_agent_context: dict[str, str] | None = None,
        **agent_kwargs: Any,
    ) -> ParallelOutcome:
        """Fan out to ``agents``, fan in whatever comes back."""
        from agentflow.llm.types import now_ms

        started = now_ms()
        per_agent_context = per_agent_context or {}
        results: dict[str, AgentResult] = {}
        failures: dict[str, str] = {}
        futures: dict[Future[AgentResult], str] = {}

        with ThreadPoolExecutor(max_workers=min(self.max_workers, len(agents) or 1)) as pool:
            for name in agents:
                try:
                    agent = self.registry.build(name, budget=self.budget, **agent_kwargs)
                except (KeyError, TypeError) as exc:
                    failures[name] = f"build failed: {exc}"
                    continue
                task_context = f"{context}\n\n{per_agent_context.get(name, '')}".strip()
                futures[pool.submit(agent.run, task, context=task_context)] = name

            try:
                for future in as_completed(futures, timeout=self.timeout_s):
                    name = futures[future]
                    try:
                        results[name] = future.result()
                    except Exception as exc:  # noqa: BLE001 - isolate agent failures
                        log.warning("agent %s failed: %s", name, exc, exc_info=exc)
                        failures[name] = f"{type(exc).__name__}: {exc}"
            except FuturesTimeout:
                # One slow agent must not discard the work the others already did.
                for future, name in futures.items():
                    if future.done():
                        continue
                    future.cancel()
                    failures[name] = f"timed out after {self.timeout_s}s"
                    log.warning("agent %s exceeded %.1fs", name, self.timeout_s)

        wall = now_ms() - started
        serial = sum(r.latency_ms for r in results.values())
        outcome = ParallelOutcome(
            results=results, failures=failures, wall_ms=wall, serial_ms=serial or wall
        )

        metrics.incr("agent_fanout_total")
        metrics.incr("agent_invocations_total", len(results))
        metrics.incr("agent_failures_total", len(failures))
        metrics.observe("agent_wall_ms", wall)
        metrics.observe("agent_serial_ms", outcome.serial_ms)
        return outcome


__all__ = ["AgentPool", "AgentRegistry", "ParallelOutcome", "default_registry"]
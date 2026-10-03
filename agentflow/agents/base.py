"""The agent contract.

One base class implements the whole loop — plan, act, observe, re-plan — so a
concrete agent only declares its system prompt, its tool allow-list and its
capabilities. Guardrails, token optimisation, budget checks and routing all wrap
the loop from the base class, so no agent can accidentally opt out of them.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from agentflow.config import Settings
from agentflow.config import settings as default_settings
from agentflow.errors import GuardrailViolation
from agentflow.guardrails.pipeline import GuardrailPipeline
from agentflow.llm.catalog import ModelSpec, catalog
from agentflow.llm.types import ContentBlock, LLMClient, LLMResponse, Message, ToolSchema
from agentflow.routing.router import AutoRouter, RoutingDecision
from agentflow.tokens.budget import TokenBudget, cost_usd
from agentflow.tokens.optimizer import TokenOptimizer
from agentflow.tools import ToolRegistry

log = logging.getLogger("agentflow.agents")


@dataclass(slots=True)
class StepTrace:
    """One model call plus whatever tools it triggered."""

    iteration: int
    model: str
    prompt_tokens: int
    saved_tokens: int
    tool_calls: list[str] = field(default_factory=list)
    latency_ms: float = 0.0
    text: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "iteration": self.iteration,
            "model": self.model,
            "prompt_tokens": self.prompt_tokens,
            "saved_tokens": self.saved_tokens,
            "tool_calls": list(self.tool_calls),
            "latency_ms": round(self.latency_ms, 2),
            "text": self.text[:300],
        }


@dataclass(slots=True)
class AgentResult:
    """Everything one agent produced, including how it got there."""

    agent: str
    output: str
    model: str
    routing: RoutingDecision | None = None
    steps: list[StepTrace] = field(default_factory=list)
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    usage_total: int = 0
    tokens_saved: int = 0
    cost_usd: float = 0.0
    guardrail_findings: list[dict[str, Any]] = field(default_factory=list)
    blocked: bool = False
    error: str | None = None
    latency_ms: float = 0.0

    @property
    def ok(self) -> bool:
        return self.error is None and not self.blocked

    @property
    def tools_used(self) -> list[str]:
        return [c["tool"] for c in self.tool_calls]

    def to_dict(self) -> dict[str, Any]:
        return {
            "agent": self.agent,
            "output": self.output,
            "model": self.model,
            "routing": self.routing.to_dict() if self.routing else None,
            "steps": [s.to_dict() for s in self.steps],
            "tool_calls": list(self.tool_calls),
            "tools_used": self.tools_used,
            "usage_total": self.usage_total,
            "tokens_saved": self.tokens_saved,
            "cost_usd": round(self.cost_usd, 6),
            "guardrail_findings": list(self.guardrail_findings),
            "blocked": self.blocked,
            "error": self.error,
            "latency_ms": round(self.latency_ms, 2),
            "ok": self.ok,
        }


class Agent(ABC):
    """Base class: routing, guardrails, optimisation, budget, tool loop."""

    name: str = "agent"
    description: str = ""
    capabilities: set[str] = field(default_factory=set)
    allowed_tools: set[str] | None = None  # None => all registered tools
    max_iterations: int = 6

    def __init__(
        self,
        client: LLMClient,
        *,
        registry: ToolRegistry,
        settings: Settings | None = None,
        guardrails: GuardrailPipeline | None = None,
        router: AutoRouter | None = None,
        optimizer: TokenOptimizer | None = None,
        budget: TokenBudget | None = None,
        model: str | None = None,
    ) -> None:
        self.settings = settings or default_settings
        self.client = client
        self.registry = registry
        self.guardrails = guardrails or GuardrailPipeline(self.settings)
        self.router = router or AutoRouter(self.settings)
        self.optimizer = optimizer or TokenOptimizer(
            cache_min_tokens=self.settings.cache_min_tokens
        )
        self.budget = budget
        self.forced_model = model
        self.max_iterations = min(self.max_iterations, self.settings.max_tool_iterations)

    # -- subclass contract ------------------------------------------------
    @abstractmethod
    def system_prompt(self) -> str:
        """Stable prefix. Kept byte-identical across calls so it caches."""

    def on_result(self, result: AgentResult) -> AgentResult:
        """Hook for post-processing. Default is a no-op."""
        return result

    def extra_tools(self) -> list[ToolSchema]:
        return []

    # -- tools ------------------------------------------------------------
    def tools(self) -> list[ToolSchema]:
        schemas = list(self.extra_tools())
        if self.allowed_tools is None:
            schemas.extend(self.registry.schemas())
        else:
            schemas.extend(self.registry.schemas(include=self.allowed_tools))
        return schemas

    def _tool_names(self) -> set[str]:
        return {s.name for s in self.tools()}

    # -- the loop ---------------------------------------------------------
    def run(self, task: str, *, context: str = "", history: list[Message] | None = None) -> AgentResult:
        """Execute the agentic loop for one task."""
        from agentflow.llm.types import now_ms

        started = now_ms()
        result = AgentResult(agent=self.name, output="", model=self.forced_model or "n/a")

        try:
            self.guardrails.check_input(task)
        except GuardrailViolation as exc:
            result.blocked = True
            result.error = f"{exc.rule}: {exc}"
            result.guardrail_findings.append(
                {"rule": exc.rule, "stage": exc.stage, "message": str(exc)}
            )
            result.latency_ms = now_ms() - started
            return result

        messages: list[Message] = [
            *(history or []),
            Message("user", f"{task}\n\n{context}".strip() if context else task),
        ]
        system = self.system_prompt()

        for iteration in range(1, self.max_iterations + 1):
            routing = self.router.route(
                task,
                expected_input_tokens=self.optimizer.measure(
                    messages, system, self.tools() or None
                ),
                requires_tools=bool(self.tools()),
                requires_retrieval="prompt-caching" in self.capabilities,
                force_model=self.forced_model,
                escalate=result.steps and self._last_call_failed(messages),
            )
            result.routing = result.routing or routing

            payload = self.optimizer.optimize(messages, system=system, tools=self.tools() or None)
            prompt_tokens = payload.report.tokens_after
            if self.budget:
                self.budget.check(prompt_tokens)

            response = self.client.complete(
                payload.messages,
                system=payload.system,
                model=routing.model,
                tools=self.tools() or None,
                max_output_tokens=routing.max_output_tokens,
                temperature=0.2,
            )
            result.model = response.model
            result.tokens_saved += payload.report.saved_tokens

            if self.budget:
                result.cost_usd += self.budget.charge(response.model, response.usage, cached=response.cached)
            else:
                # Without a budget there is nothing to charge, but the cost is
                # still real: price the usage off the catalog so reporting never
                # silently understates what the run spent.
                spec = catalog.get(response.model) if response.model in catalog else None
                result.cost_usd += cost_usd(spec, response.usage) if spec else 0.0
            result.usage_total += response.usage.total

            step = StepTrace(
                iteration=iteration,
                model=response.model,
                prompt_tokens=prompt_tokens,
                saved_tokens=payload.report.saved_tokens,
                tool_calls=[b.get("name", "") for b in response.tool_uses],
                latency_ms=response.latency_ms,
                text=response.text,
            )
            result.steps.append(step)

            if not response.wants_tools:
                guard = self.guardrails.check_output(response.text, context=task)
                result.output = guard.text
                result.guardrail_findings.extend(guard.to_dict()["findings"])  # type: ignore[arg-type]
                result.latency_ms = now_ms() - started
                return self.on_result(result)

            messages.append(Message("assistant", response.blocks))
            messages.append(Message("tool", self._observe(response, result)))

        result.output = result.steps[-1].text if result.steps else ""
        result.error = f"stopped after {self.max_iterations} iterations without a final answer"
        result.latency_ms = now_ms() - started
        return self.on_result(result)

    # -- internals --------------------------------------------------------
    def _observe(self, response: LLMResponse, result: AgentResult) -> list[ContentBlock]:
        """Execute every tool_use block and return the tool_result turn."""
        blocks: list[ContentBlock] = []
        for use in response.tool_uses:
            name = str(use.get("name", ""))
            arguments = dict(use.get("input") or {})
            if self.allowed_tools is not None and name not in self._tool_names():
                invocation_text = f"tool {name!r} is not available to the {self.name} agent"
                is_error = True
            else:
                invocation = self.registry.invoke(name, **arguments)
                invocation_text = invocation.as_text()
                is_error = invocation.is_error
                result.tool_calls.append(invocation.to_dict())
            blocks.append(
                ContentBlock.tool_result(str(use.get("id", "")), invocation_text, is_error=is_error)
            )
        return blocks

    @staticmethod
    def _last_call_failed(messages: Sequence[Message]) -> bool:
        return any(
            block.get("type") == "tool_result" and block.get("is_error")
            for message in messages
            for block in message.as_blocks()
        )

    def spec(self) -> ModelSpec | None:
        return catalog.get(self.forced_model) if self.forced_model else None


__all__ = ["Agent", "AgentResult", "StepTrace"]
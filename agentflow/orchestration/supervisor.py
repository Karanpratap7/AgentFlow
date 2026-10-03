"""The supervisor: node implementations for the agentic workflow.

Graph shape::

    guardrail -> intent -> route -> plan -> fanout -> fanin -> synthesize -> verify
                                                      ^                        |
                                                      +---- escalate ----------+

* ``guardrail`` blocks unsafe input before a single token is spent.
* ``route``    picks the model for the synthesis step.
* ``plan``     asks the pool which specialists the request needs.
* ``fanout``   runs them in parallel; the fan-in step collects whatever landed.
* ``synthesize`` merges findings into one answer.
* ``verify``   checks the answer for grounding and tool failures, and either
  escalates (retry once on a stronger model) or terminates.
"""

from __future__ import annotations

import logging
from typing import Any

from agentflow.agents.base import AgentResult
from agentflow.agents.pool import AgentPool, AgentRegistry, ParallelOutcome
from agentflow.agents.specialists import SynthesizerAgent
from agentflow.config import Settings
from agentflow.config import settings as default_settings
from agentflow.errors import GuardrailViolation
from agentflow.guardrails.pipeline import GuardrailPipeline
from agentflow.llm.types import LLMClient
from agentflow.orchestration.state import WorkflowState
from agentflow.routing.intent import IntentDetector
from agentflow.routing.router import AutoRouter
from agentflow.tokens.budget import TokenBudget
from agentflow.tools import ToolRegistry

log = logging.getLogger("agentflow.orchestrator")

MAX_ITERATIONS = 2
_MIN_USABLE_AGENTS = 1


class Supervisor:
    """Holds every collaborator and exposes one method per graph node."""

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
        max_parallel_agents: int | None = None,
        forced_agents: list[str] | None = None,
        rag: Any = None,
    ) -> None:
        self.client = client
        self.settings = settings or default_settings
        self.registry = registry
        self.tools = tools
        self.guardrails = guardrails or GuardrailPipeline(self.settings)
        self.router = router or AutoRouter(self.settings)
        self.intent_detector = intent_detector or IntentDetector(client)
        self.budget = budget or TokenBudget()
        self.pool = AgentPool(
            registry,
            max_workers=max_parallel_agents or self.settings.max_parallel_agents,
            timeout_s=self.settings.agent_timeout_s,
            budget=self.budget,
        )
        self.forced_agents = forced_agents
        #: Shared dependency bag handed to every specialist. ``rag`` is only
        #: consumed by ResearchAgent, which then skips the generic tool loop and
        #: answers through the full RAG pipeline instead.
        self._agent_kwargs: dict[str, Any] = {
            "registry": tools,
            "client": client,
            "settings": self.settings,
            "guardrails": self.guardrails,
            "router": self.router,
            "optimizer": None,
            "rag": rag,
        }
        self.last_parallel: ParallelOutcome | None = None

    # -- nodes ------------------------------------------------------------
    def guardrail(self, state: WorkflowState) -> dict[str, Any]:
        """Input gate. A violation terminates the run via a raised exception."""
        try:
            result = self.guardrails.check_input(state["task"])
        except GuardrailViolation as exc:
            return {
                "answer": f"Request blocked by guardrail `{exc.rule}`: {exc}",
                "errors": [f"guardrail:{exc.rule}"],
                "iterations": 0,
            }
        return {
            "trace": [
                {
                    "node": "guardrail",
                    "redacted": result.redaction is not None and result.redaction.redacted,
                    "kinds": result.redaction.kinds() if result.redaction else [],
                    "flagged": result.flagged,
                }
            ]
        }

    def detect_intent(self, state: WorkflowState) -> dict[str, Any]:
        intent = self.intent_detector.detect(state["task"])
        return {
            "intent": intent,
            "trace": [
                {
                    "node": "intent",
                    "intent": str(intent.intent),
                    "confidence": intent.confidence,
                    "stage": intent.stage,
                }
            ],
        }

    def route(self, state: WorkflowState) -> dict[str, Any]:
        intent = state.get("intent")
        decision = self.router.route(
            state["task"],
            intent=intent,
            requires_tools=False,
            escalate=state.get("escalated", False),
        )
        return {"routing": decision, "trace": [{"node": "route", **decision.to_dict()}]}

    def plan(self, state: WorkflowState) -> dict[str, Any]:
        selected = self.pool.plan(
            state.get("intent"), force=self.forced_agents or None
        )
        return {
            "selected_agents": selected,
            "trace": [{"node": "plan", "selected": selected}],
        }

    def fan_out(self, state: WorkflowState) -> dict[str, Any]:
        """Parallel fan-out. Returns a single merged update; the pool does the
        concurrency so one worker thread per agent is used."""
        selected = state.get("selected_agents") or []
        if not selected:
            return {"trace": [{"node": "fanout", "skipped": "no specialists required"}]}

        outcome = self.pool.run(selected, state["task"], **self._agent_kwargs)
        self.last_parallel = outcome

        results = dict(outcome.results)
        state_dict: dict[str, Any] = {
            "agent_results": results,
            "agent_failures": dict(outcome.failures),
            "trace": [
                {
                    "node": "fanout",
                    **outcome.to_dict(),
                    "per_agent": {k: v.to_dict() for k, v in results.items()},
                }
            ],
        }
        unusable = [n for n, r in results.items() if not r.ok]
        if unusable:
            state_dict["errors"] = [f"agent:{n}" for n in unusable]
        return state_dict

    def synthesize(self, state: WorkflowState) -> dict[str, Any]:
        """Fan-in. Merge the parallel findings, or answer directly when there
        are none (trivial requests short-circuit the specialist layer)."""
        results: dict[str, AgentResult] = dict(state.get("agent_results") or {})
        usable = {n: r for n, r in results.items() if r.ok and r.output.strip()}
        routing = state.get("routing")

        if not usable:
            decision = self.router.route(state["task"], intent=state.get("intent"))
            response = self.client.complete(
                [self._user_turn(state["task"])],
                system=self._direct_system_prompt(),
                model=decision.model,
                max_output_tokens=decision.max_output_tokens,
            )
            self.budget.charge(response.model, response.usage, cached=response.cached)
            guard = self.guardrails.check_output(response.text, context=state["task"])
            return {
                "answer": guard.text,
                "routing": decision,
                "trace": [
                    {
                        "node": "synthesize",
                        "mode": "direct",
                        "model": response.model,
                        "reason": "no specialist produced a usable result",
                    }
                ],
            }

        if len(usable) == 1:
            only = next(iter(usable.values()))
            return {
                "answer": only.output,
                "trace": [
                    {"node": "synthesize", "mode": "pass_through", "agent": only.agent}
                ],
            }

        synthesizer = self.registry.build("synthesizer", **self._agent_kwargs)
        assert isinstance(synthesizer, SynthesizerAgent)
        context = SynthesizerAgent.build_context(usable)
        routing = routing or self.router.route(
            state["task"],
            intent=state.get("intent"),
            requires_tools=False,
            escalate=state.get("escalated", False),
        )

        guard = self.guardrails.check_input(context)
        payload = synthesizer.optimizer.optimize(
            [self._user_turn(state["task"])],
            system=synthesizer.system_prompt(),
            summary=None,
        )
        prompt = (
            f"{state['task']}\n\n"
            "## Parallel agent findings\n"
            f"{guard.text}\n\n"
            "Synthesise one answer. Attribute each point to the agent it came from."
        )
        response = self.client.complete(
            [self._user_turn(prompt)],
            system=payload.system,
            model=routing.model,
            max_output_tokens=routing.max_output_tokens,
        )
        self.budget.charge(response.model, response.usage, cached=response.cached)
        output = self.guardrails.check_output(response.text, context=state["task"])
        return {
            "answer": output.text,
            "tokens_saved": payload.report.saved_tokens,
            "trace": [
                {
                    "node": "synthesize",
                    "mode": "merged",
                    "agents": list(usable),
                    "model": response.model,
                    "tokens_saved": payload.report.saved_tokens,
                }
            ],
        }

    def verify(self, state: WorkflowState) -> dict[str, Any]:
        """Quality gate. Escalates once, then accepts."""
        results = dict(state.get("agent_results") or {})
        selected = list(state.get("selected_agents") or [])
        usable = [r for r in results.values() if r.ok and r.output.strip()]
        failures = list(state.get("agent_failures") or {})
        iteration = int(state.get("iteration", 0)) + 1

        problems: list[str] = []
        # Only a *planned* agent run can fail. A request that short-circuited to
        # the direct path legitimately has no agent results.
        if selected and len(usable) < _MIN_USABLE_AGENTS:
            problems.append(
                f"no planned agent produced a usable result (planned={selected}, failed={failures})"
            )
        if not (state.get("answer") or "").strip():
            problems.append("synthesiser produced an empty answer")

        needs_escalation = bool(problems) and not state.get("escalated", False)
        return {
            "iteration": iteration,
            "escalated": bool(state.get("escalated", False)) or needs_escalation,
            "trace": [
                {
                    "node": "verify",
                    "usable_agents": len(usable),
                    "failures": failures,
                    "problems": problems,
                    "escalate": needs_escalation,
                }
            ],
        }

    # -- helpers ----------------------------------------------------------
    @staticmethod
    def _user_turn(text: str) -> Any:
        from agentflow.llm.types import Message

        return Message("user", text)

    @staticmethod
    def _direct_system_prompt() -> str:
        return (
            "You are AgentFlow, a direct assistant. Answer in at most 120 words. "
            "If the request is ambiguous, state the assumption you made."
        )


__all__ = ["MAX_ITERATIONS", "Supervisor"]
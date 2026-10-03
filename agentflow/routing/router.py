"""The auto-router: intent + difficulty -> model + task shape.

Scoring is explicit and inspectable. Every routing decision returns a
:class:`RoutingDecision` carrying the factor breakdown, so a wrong route can be
debugged instead of guessed at.

Cost model
----------
For an ``n``-token prompt and ``m``-token answer the naive route costs
``3*P_in + P_out`` per 1M tokens; a cached route costs
``3*(w*P_write + (1-w)*P_read) + P_out`` where ``w`` is the expected write
fraction. ``expected_saving_usd`` uses that formula against the frontier model,
which is what the ``/route/compare`` endpoint reports.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from agentflow.config import Settings
from agentflow.config import settings as default_settings
from agentflow.errors import RoutingError
from agentflow.llm.catalog import ModelCatalog, ModelSpec, Quality
from agentflow.llm.catalog import catalog as default_catalog
from agentflow.llm.types import Message, Usage
from agentflow.routing.intent import Intent, IntentResult
from agentflow.tokens.budget import cost_usd, uncached_cost_usd
from agentflow.tokens.counter import estimate_tokens
from agentflow.tokens.optimizer import cap_output_tokens

#: Capability tags implied by each task shape.
TASK_CAPABILITIES: dict[str, set[str]] = {
    "triage": {"streaming"},
    "extractive": {"streaming"},
    "chat": {"streaming"},
    "rag": {"tool-use", "prompt-caching", "streaming"},
    "sql": {"tool-use", "extended-thinking"},
    "code": {"tool-use", "extended-thinking"},
    "agentic": {"tool-use", "prompt-caching", "extended-thinking", "long-horizon-agentic"},
}

#: Highest task shape that may be run on a downgraded model. Degradation only
#: ever happens for shapes whose capability tags are still satisfied.
DEGRADABLE_TASKS = frozenset({"triage", "extractive", "chat", "rag"})

#: Human-readable task label per intent, used for output capping.
INTENT_TASK: dict[Intent, str] = {
    Intent.GREETING: "chat",
    Intent.CHITCHAT: "chat",
    Intent.KNOWLEDGE: "rag",
    Intent.CODE: "code",
    Intent.SQL: "sql",
    Intent.SUMMARIZE: "extractive",
    Intent.ANALYSIS: "analysis",
    Intent.TRANSLATION: "chat",
    Intent.MATH: "code",
    Intent.SAFETY_RISK: "triage",
    Intent.UNKNOWN: "chat",
}


@dataclass(slots=True)
class RoutingDecision:
    """One routing outcome plus the evidence behind it."""

    model: str
    tier: str
    task: str
    quality: str
    reason: str
    scores: dict[str, float] = field(default_factory=dict)
    capabilities: list[str] = field(default_factory=list)
    max_output_tokens: int = 1_200
    needs_retrieval: bool = False
    needs_tools: bool = False
    expected_input_tokens: int = 0
    cache_prefix_tokens: int = 0
    fallback_model: str = ""
    alternatives: list[dict[str, Any]] = field(default_factory=list)

    @property
    def spec(self) -> ModelSpec:
        return default_catalog.get(self.model)

    def to_dict(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "tier": self.tier,
            "task": self.task,
            "quality": self.quality,
            "reason": self.reason,
            "scores": {k: round(v, 4) for k, v in self.scores.items()},
            "capabilities": list(self.capabilities),
            "max_output_tokens": self.max_output_tokens,
            "needs_retrieval": self.needs_retrieval,
            "needs_tools": self.needs_tools,
            "expected_input_tokens": self.expected_input_tokens,
            "cache_prefix_tokens": self.cache_prefix_tokens,
            "fallback_model": self.fallback_model,
            "alternatives": list(self.alternatives),
        }


_QUALITY_RANK: dict[Quality, int] = {"economy": 0, "balanced": 1, "frontier": 2}
_QUALITY_DOWN: dict[Quality, Quality] = {"frontier": "balanced", "balanced": "economy", "economy": "economy"}


def _relax_quality(floor: Quality) -> Quality:
    return _QUALITY_DOWN[floor]


class AutoRouter:
    """Selects a model and a task shape for every request."""

    def __init__(
        self,
        settings: Settings | None = None,
        *,
        catalog: ModelCatalog | None = None,
        cost_sensitivity: float = 1.0,
    ) -> None:
        self.settings = settings or default_settings
        self.catalog = catalog or default_catalog
        #: >1 pushes towards cheaper models, <1 towards stronger ones.
        self.cost_sensitivity = cost_sensitivity

    # -- main entry point -------------------------------------------------
    def route(
        self,
        text: str,
        *,
        intent: IntentResult | None = None,
        expected_input_tokens: int | None = None,
        requires_tools: bool = False,
        requires_retrieval: bool = False,
        cache_prefix_tokens: int = 0,
        force_model: str | None = None,
        escalate: bool = False,
    ) -> RoutingDecision:
        from agentflow.routing.intent import IntentDetector

        result = intent or IntentDetector().detect(text)
        tokens = (
            expected_input_tokens
            if expected_input_tokens is not None
            else estimate_tokens([Message("user", text)])
        )
        task = INTENT_TASK.get(result.intent, "chat")
        capabilities = set(TASK_CAPABILITIES.get(task, {"streaming"}))
        if requires_tools:
            capabilities.add("tool-use")
        if requires_retrieval:
            capabilities.add("prompt-caching")
        if cache_prefix_tokens >= self.settings.cache_min_tokens:
            capabilities.add("prompt-caching")

        scores = self._scores(result, capabilities, tokens, cache_prefix_tokens)
        if escalate:
            # Orchestrator-driven escalation: a previous attempt on this request
            # failed, so retry on a stronger model rather than looping.
            scores["complexity"] = max(scores["complexity"], 0.6)
            scores["forced_escalation"] = 1.0

        if force_model:
            if force_model not in self.catalog:
                raise RoutingError(f"cannot force unknown model {force_model!r}")
            spec = self.catalog.get(force_model)
            reason = "model pinned by caller"
        else:
            spec, reason = self._select(result, capabilities, tokens, scores, task)

        alternatives = self._alternatives(spec, capabilities, tokens)

        return RoutingDecision(
            model=spec.id,
            tier=spec.tier,
            task=task,
            quality=spec.quality,
            reason=reason,
            scores=scores,
            capabilities=sorted(capabilities),
            max_output_tokens=cap_output_tokens(task, None, spec=spec),
            needs_retrieval=requires_retrieval or result.needs_retrieval,
            needs_tools=requires_tools or result.needs_tools,
            expected_input_tokens=tokens,
            cache_prefix_tokens=cache_prefix_tokens,
            fallback_model=self.settings.fallback_model,
            alternatives=alternatives,
        )

    # -- scoring ----------------------------------------------------------
    def _scores(
        self,
        result: IntentResult,
        capabilities: set[str],
        tokens: int,
        cache_prefix_tokens: int,
    ) -> dict[str, float]:
        floor = self.catalog.requirements_for(capabilities)
        scores: dict[str, float] = {
            "intent_confidence": result.confidence,
            "quality_floor": {"economy": 0.0, "balanced": 0.35, "frontier": 0.7}[floor],
            "token_pressure": min(1.0, tokens / max(1, self.settings.request_token_budget)),
            "cacheable_prefix": min(1.0, cache_prefix_tokens / max(1, self.settings.cache_min_tokens)),
            "cost_sensitivity": self.cost_sensitivity,
        }
        if result.intent in {Intent.CODE, Intent.SQL, Intent.ANALYSIS}:
            scores["reasoning_required"] = 0.4 + 0.4 * result.confidence
        else:
            scores["reasoning_required"] = 0.0
        scores["complexity"] = round(
            0.45 * scores["token_pressure"]
            + 0.3 * scores["reasoning_required"]
            + 0.25 * scores["quality_floor"],
            4,
        )
        return scores

    def _select(
        self,
        result: IntentResult,
        capabilities: set[str],
        tokens: int,
        scores: dict[str, float],
        task: str,
    ) -> tuple[ModelSpec, str]:
        floor = self.catalog.requirements_for(capabilities)
        complexity = scores["complexity"]

        # Three-step ladder. Low classification confidence must not by itself
        # buy the most expensive model — it buys a *mid* tier. Only genuine
        # complexity escalates to frontier.
        complexity_high = complexity >= 0.6
        low_confidence = result.confidence < 0.45

        if complexity_high:
            min_quality: Quality = "frontier"
            objective = "quality"
        elif low_confidence:
            # Raise the floor one step, but still buy the *cheapest* model that
            # meets it. Escalating straight to the objective would pick the most
            # expensive model in the tier for no extra benefit.
            min_quality = (
                "balanced" if _QUALITY_RANK[floor] < _QUALITY_RANK["balanced"] else floor
            )
            objective = "cost"
        else:
            min_quality = floor
            objective = "cost"

        try:
            spec = self.catalog.select(
                required_capabilities=capabilities,
                min_quality=min_quality,
                required_tokens=tokens,
                must_support_cache="prompt-caching" in capabilities,
                objective=objective,  # type: ignore[arg-type]
            )
        except RoutingError:
            spec = self.catalog.select(
                required_capabilities={"streaming"},
                min_quality="economy",
                objective="cost",
            )
            return spec, f"no model met {sorted(capabilities)}; degraded to {spec.id}"

        # Cost downgrade: a short, confident, low-complexity request does not
        # need the capability tier its tags imply. Relax the floor one level so
        # the router can pick a cheaper model with identical capabilities.
        if self._may_degrade(task, min_quality != floor, complexity, result):
            relaxed = _relax_quality(floor)
            cheaper = self.catalog.filter(
                required_capabilities=capabilities,
                min_quality=relaxed,
                required_tokens=tokens,
                must_support_cache="prompt-caching" in capabilities,
            )
            if cheaper:
                pick = self.catalog.select(
                    required_capabilities=capabilities,
                    min_quality=relaxed,
                    required_tokens=tokens,
                    must_support_cache="prompt-caching" in capabilities,
                    objective="cost",
                )
                if pick.id != spec.id:
                    return pick, (
                        f"complexity {complexity:.2f} <= downgrade threshold and confidence "
                        f"{result.confidence:.2f} high: relaxed quality {floor} -> {relaxed} "
                        f"for a cheaper model"
                    )

        if complexity_high:
            reason = (
                f"complexity {complexity:.2f} >= 0.60: escalated to quality={min_quality}"
            )
        elif low_confidence:
            reason = (
                f"intent confidence {result.confidence:.2f} < 0.45: "
                f"raised quality floor {floor} -> {min_quality} rather than going to frontier"
            )
        elif "prompt-caching" in capabilities:
            reason = f"capabilities {sorted(capabilities)} met by cheapest qualifying model"
        else:
            reason = f"intent {result.intent} -> cheapest model meeting floor {floor}"
        return spec, reason

    def _may_degrade(
        self, task: str, already_raised: bool, complexity: float, result: IntentResult
    ) -> bool:
        if already_raised or task not in DEGRADABLE_TASKS:
            return False
        threshold = 0.45 / max(0.1, self.cost_sensitivity)
        return complexity <= threshold and result.confidence >= 0.8

    def _alternatives(
        self, chosen: ModelSpec, capabilities: set[str], tokens: int
    ) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for spec in self.catalog.filter(required_capabilities=capabilities, required_tokens=tokens):
            if spec.id == chosen.id:
                continue
            out.append(
                {
                    "model": spec.id,
                    "tier": spec.tier,
                    "quality": spec.quality,
                    "blended_cost_per_mtok": round(self.catalog.cost_key(spec)[0], 4),
                }
            )
        return out[:3]

    # -- economics --------------------------------------------------------
    def compare_models(
        self,
        *,
        input_tokens: int,
        output_tokens: int,
        cache_prefix_tokens: int = 0,
        cache_hit_ratio: float = 0.9,
    ) -> dict[str, Any]:
        """Per-model cost for one request shape, plus the counterfactual."""
        rows: list[dict[str, Any]] = []
        for spec in self.catalog.specs():
            plain = uncached_cost_usd(spec, input_tokens, output_tokens)
            cached_usage = Usage(
                input_tokens=max(0, input_tokens - cache_prefix_tokens),
                output_tokens=output_tokens,
                cache_read_input_tokens=cache_prefix_tokens * cache_hit_ratio,
                cache_creation_input_tokens=cache_prefix_tokens * (1 - cache_hit_ratio),
            )
            cached = cost_usd(spec, cached_usage)
            rows.append(
                {
                    "model": spec.id,
                    "tier": spec.tier,
                    "uncached_usd": round(plain, 6),
                    "cached_usd": round(cached, 6),
                    "cache_savings_usd": round(plain - cached, 6),
                    "cache_savings_pct": round((plain - cached) / plain, 4) if plain else 0.0,
                }
            )
        rows.sort(key=lambda r: r["cached_usd"])
        cheapest = rows[0]
        frontier = next(
            (r for r in rows if self.catalog.get(r["model"]).quality == "frontier"), rows[-1]
        )
        return {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "cache_prefix_tokens": cache_prefix_tokens,
            "models": rows,
            "recommended": cheapest["model"],
            "vs_frontier_savings_usd": round(frontier["cached_usd"] - cheapest["cached_usd"], 6),
        }


__all__ = [
    "DEGRADABLE_TASKS",
    "INTENT_TASK",
    "TASK_CAPABILITIES",
    "AutoRouter",
    "RoutingDecision",
]
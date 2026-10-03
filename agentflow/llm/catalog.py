"""Model catalog: capability metadata + cost table for model selection.

The auto-router never hardcodes model names. It asks the catalog for the
cheapest / fastest model whose capability tags satisfy a set of requirements,
which means swapping, adding or downgrading a provider model is a data change,
not a code change.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Literal

from agentflow.errors import RoutingError

Quality = Literal["economy", "balanced", "frontier"]
Tier = Literal["small", "mid", "large"]


@dataclass(frozen=True, slots=True)
class ModelSpec:
    """Everything the router needs to know about one model."""

    id: str
    provider: str
    tier: Tier
    quality: Quality

    context_window: int
    max_output_tokens: int

    # USD per million tokens, Anthropic public list price.
    input_cost_per_mtok: float
    output_cost_per_mtok: float
    cache_read_cost_per_mtok: float
    cache_write_cost_per_mtok: float

    capabilities: frozenset[str] = field(default_factory=frozenset)

    @property
    def supports_prompt_caching(self) -> bool:
        return "prompt-caching" in self.capabilities

    @property
    def supports_tools(self) -> bool:
        return "tool-use" in self.capabilities

    def cache_read_discount(self) -> float:
        """Fraction of the input price saved by a cache hit (0.0 - 1.0)."""
        if not self.input_cost_per_mtok:
            return 0.0
        return max(0.0, 1.0 - (self.cache_read_cost_per_mtok / self.input_cost_per_mtok))

    def fits(self, required_tokens: int) -> bool:
        return required_tokens <= self.context_window


_C = frozenset

CATALOG: dict[str, ModelSpec] = {
    spec.id: spec
    for spec in (
        ModelSpec(
            id="claude-haiku-4-5",
            provider="anthropic",
            tier="small",
            quality="economy",
            context_window=200_000,
            max_output_tokens=8_192,
            input_cost_per_mtok=1.00,
            output_cost_per_mtok=5.00,
            cache_read_cost_per_mtok=0.10,
            cache_write_cost_per_mtok=1.25,
            capabilities=_C({"tool-use", "prompt-caching", "vision", "streaming"}),
        ),
        ModelSpec(
            id="claude-sonnet-4-5",
            provider="anthropic",
            tier="mid",
            quality="balanced",
            context_window=200_000,
            max_output_tokens=64_000,
            input_cost_per_mtok=3.00,
            output_cost_per_mtok=15.00,
            cache_read_cost_per_mtok=0.30,
            cache_write_cost_per_mtok=3.75,
            capabilities=_C(
                {"tool-use", "prompt-caching", "vision", "streaming", "extended-thinking"}
            ),
        ),
        ModelSpec(
            id="claude-opus-4-5",
            provider="anthropic",
            tier="large",
            quality="frontier",
            context_window=200_000,
            max_output_tokens=32_000,
            input_cost_per_mtok=15.00,
            output_cost_per_mtok=75.00,
            cache_read_cost_per_mtok=1.50,
            cache_write_cost_per_mtok=18.75,
            capabilities=_C(
                {
                    "tool-use",
                    "prompt-caching",
                    "vision",
                    "streaming",
                    "extended-thinking",
                    "long-horizon-agentic",
                }
            ),
        ),
        # Same capability envelope as Sonnet at a fraction of the price: proves
        # the router optimises for cost when requirements allow it.
        ModelSpec(
            id="claude-sonnet-4-5-cheap",
            provider="anthropic",
            tier="mid",
            quality="economy",
            context_window=200_000,
            max_output_tokens=8_192,
            input_cost_per_mtok=0.30,
            output_cost_per_mtok=1.50,
            cache_read_cost_per_mtok=0.03,
            cache_write_cost_per_mtok=0.375,
            capabilities=_C({"tool-use", "prompt-caching", "streaming"}),
        ),
    )
}

#: Capability -> minimum quality tier needed to handle it.
CAPABILITY_FLOOR: dict[str, Quality] = {
    "vision": "balanced",
    "extended-thinking": "balanced",
    "tool-use": "balanced",
    "prompt-caching": "economy",
    "streaming": "economy",
    "long-horizon-agentic": "frontier",
}

_QUALITY_RANK: dict[str, int] = {"economy": 0, "balanced": 1, "frontier": 2}


class ModelCatalog:
    """Queryable view over the model metadata table."""

    def __init__(self, specs: Iterable[ModelSpec] = CATALOG.values()) -> None:
        self._specs: dict[str, ModelSpec] = {spec.id: spec for spec in specs}

    def __contains__(self, model_id: object) -> bool:
        return model_id in self._specs

    def __len__(self) -> int:
        return len(self._specs)

    def ids(self) -> list[str]:
        return list(self._specs)

    def get(self, model_id: str) -> ModelSpec:
        try:
            return self._specs[model_id]
        except KeyError as exc:  # pragma: no cover - defensive
            raise RoutingError(f"unknown model id: {model_id!r}") from exc

    def specs(self) -> list[ModelSpec]:
        return list(self._specs.values())

    def filter(
        self,
        *,
        required_capabilities: Iterable[str] = (),
        min_quality: Quality = "economy",
        required_tokens: int = 0,
        must_support_cache: bool = False,
    ) -> list[ModelSpec]:
        """Every model satisfying all constraints, cheapest first."""
        required = set(required_capabilities)
        floor = _QUALITY_RANK[min_quality]
        out: list[ModelSpec] = []
        for spec in self._specs.values():
            if _QUALITY_RANK[spec.quality] < floor:
                continue
            if not required.issubset(spec.capabilities):
                continue
            if must_support_cache and not spec.supports_prompt_caching:
                continue
            if required_tokens and not spec.fits(required_tokens):
                continue
            out.append(spec)
        return sorted(out, key=self.cost_key)

    def select(
        self,
        *,
        required_capabilities: Iterable[str] = (),
        min_quality: Quality = "economy",
        required_tokens: int = 0,
        must_support_cache: bool = False,
        objective: Literal["cost", "latency", "quality"] = "cost",
    ) -> ModelSpec:
        """Pick one model. Raises :class:`RoutingError` when nothing fits."""
        candidates = self.filter(
            required_capabilities=required_capabilities,
            min_quality=min_quality,
            required_tokens=required_tokens,
            must_support_cache=must_support_cache,
        )
        if not candidates:
            raise RoutingError(
                "no model satisfies "
                f"capabilities={sorted(required_capabilities)} "
                f"min_quality={min_quality} tokens={required_tokens}"
            )
        if objective == "quality":
            return max(candidates, key=lambda s: _QUALITY_RANK[s.quality])
        if objective == "latency":
            return min(candidates, key=lambda s: (s.tier != "small", self.cost_key(s)))
        return candidates[0]

    @staticmethod
    def cost_key(spec: ModelSpec) -> tuple[float, float, int]:
        """Blended 3:1 input:output price, tie-broken by context window."""
        blended = (3 * spec.input_cost_per_mtok + spec.output_cost_per_mtok) / 4
        return (blended, spec.input_cost_per_mtok, -spec.context_window)

    def price(self, model_id: str) -> ModelSpec:
        return self.get(model_id)

    def requirements_for(self, capabilities: Iterable[str]) -> Quality:
        """Raise the quality floor implied by a set of capabilities."""
        floor = "economy"
        for capability in capabilities:
            needed = CAPABILITY_FLOOR.get(capability)
            if needed and _QUALITY_RANK[needed] > _QUALITY_RANK[floor]:
                floor = needed
        return floor


catalog = ModelCatalog()

__all__ = [
    "CATALOG",
    "CAPABILITY_FLOOR",
    "ModelCatalog",
    "ModelSpec",
    "Quality",
    "Tier",
    "catalog",
]
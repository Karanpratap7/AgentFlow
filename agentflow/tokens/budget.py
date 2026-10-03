"""Token budgets and per-session accounting.

Every request is bounded twice: a per-request ceiling (protects the context
window) and a per-session ceiling (protects the wallet over a conversation).
``TokenBudget`` records spend so :mod:`agentflow.telemetry` can attribute cost to
a model, an agent or a session.
"""

from __future__ import annotations

import threading
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

from agentflow.errors import TokenBudgetExceeded
from agentflow.llm.catalog import ModelSpec, catalog
from agentflow.llm.types import Usage


def cost_usd(spec: ModelSpec, usage: Usage) -> float:
    """Cost of one call. Cache reads are billed at their own (much lower) rate."""
    return (
        usage.input_tokens / 1e6 * spec.input_cost_per_mtok
        + usage.output_tokens / 1e6 * spec.output_cost_per_mtok
        + usage.cache_read_input_tokens / 1e6 * spec.cache_read_cost_per_mtok
        + usage.cache_creation_input_tokens / 1e6 * spec.cache_write_cost_per_mtok
    )


def uncached_cost_usd(spec: ModelSpec, prompt_tokens: int, output_tokens: int) -> float:
    """What the same call would have cost with caching disabled."""
    usage = Usage(input_tokens=prompt_tokens, output_tokens=output_tokens)
    return cost_usd(spec, usage)


@dataclass(slots=True)
class BudgetSnapshot:
    session_id: str
    spent_tokens: int
    token_limit: int
    spent_usd: float
    calls: int
    cache_hits: int

    @property
    def remaining_tokens(self) -> int:
        return max(0, self.token_limit - self.spent_tokens)

    @property
    def utilisation(self) -> float:
        return self.spent_tokens / self.token_limit if self.token_limit else 0.0

    @property
    def cache_hit_rate(self) -> float:
        return self.cache_hits / self.calls if self.calls else 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "session_id": self.session_id,
            "spent_tokens": self.spent_tokens,
            "token_limit": self.token_limit,
            "remaining_tokens": self.remaining_tokens,
            "utilisation": round(self.utilisation, 4),
            "spent_usd": round(self.spent_usd, 6),
            "calls": self.calls,
            "cache_hits": self.cache_hits,
            "cache_hit_rate": round(self.cache_hit_rate, 4),
        }


@dataclass
class TokenBudget:
    """Thread-safe spend tracker for one conversation."""

    session_id: str = "default"
    token_limit: int = 180_000
    _spent: int = field(default=0, init=False)
    _usd: float = field(default=0.0, init=False)
    _calls: int = field(default=0, init=False)
    _cache_hits: int = field(default=0, init=False)
    _by_model: dict[str, int] = field(default_factory=lambda: defaultdict(int), init=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, init=False)

    # -- limits -----------------------------------------------------------
    def check(self, incoming_tokens: int) -> None:
        """Pre-flight a call. Raises if it would exceed what is already spent.

        This is advisory: several agents can pass this check concurrently before
        any of them is charged, so :meth:`charge` is what actually enforces the
        ceiling.
        """
        with self._lock:
            if self._spent + incoming_tokens > self.token_limit:
                raise TokenBudgetExceeded(
                    f"session {self.session_id!r} budget exhausted: "
                    f"{self._spent}+{incoming_tokens} > {self.token_limit} tokens"
                )

    def remaining(self) -> int:
        with self._lock:
            return max(0, self.token_limit - self._spent)

    # -- spend ------------------------------------------------------------
    def charge(self, model_id: str, usage: Usage, *, cached: bool = False) -> float:
        """Record a completed call and return its USD cost.

        Enforces the limit rather than merely reporting it: a call that pushes the
        session past its ceiling raises, so the recorded spend can never exceed
        ``token_limit``.
        """
        spec = catalog.get(model_id) if model_id in catalog else None
        charged = cost_usd(spec, usage) if spec else 0.0
        with self._lock:
            if self._spent + usage.total > self.token_limit:
                raise TokenBudgetExceeded(
                    f"session {self.session_id!r} budget exhausted on {model_id!r}: "
                    f"{self._spent}+{usage.total} > {self.token_limit} tokens"
                )
            self._spent += usage.total
            self._usd += charged
            self._calls += 1
            self._by_model[model_id] += usage.total
            if cached:
                self._cache_hits += 1
        return charged

    def snapshot(self) -> BudgetSnapshot:
        with self._lock:
            return BudgetSnapshot(
                session_id=self.session_id,
                spent_tokens=self._spent,
                token_limit=self.token_limit,
                spent_usd=self._usd,
                calls=self._calls,
                cache_hits=self._cache_hits,
            )

    def per_model_tokens(self) -> dict[str, int]:
        with self._lock:
            return dict(self._by_model)

    def reset(self) -> None:
        with self._lock:
            self._spent = 0
            self._usd = 0.0
            self._calls = 0
            self._cache_hits = 0
            self._by_model.clear()


class BudgetRegistry:
    """Named budgets, so the API can look one up per conversation id."""

    def __init__(self, default_limit: int = 180_000) -> None:
        self._default_limit = default_limit
        self._budgets: dict[str, TokenBudget] = {}
        self._lock = threading.Lock()

    def get(self, session_id: str) -> TokenBudget:
        with self._lock:
            budget = self._budgets.get(session_id)
            if budget is None:
                budget = TokenBudget(session_id=session_id, token_limit=self._default_limit)
                self._budgets[session_id] = budget
            return budget

    def drop(self, session_id: str) -> None:
        with self._lock:
            self._budgets.pop(session_id, None)

    def __len__(self) -> int:
        return len(self._budgets)


__all__ = [
    "BudgetRegistry",
    "BudgetSnapshot",
    "TokenBudget",
    "cost_usd",
    "uncached_cost_usd",
]
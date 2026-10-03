"""Metrics.

A small in-process metrics sink. No Prometheus dependency, but the shape
(counters, labelled observations, histogram summary) maps onto one directly, and
``to_prometheus()`` renders it so a real scrape endpoint is ~10 lines.

Everything worth measuring in this runtime:
* ``intent_stage``      how often intent detection stops at the free stages
* ``route_model``       which model each intent actually selected
* ``tokens_*``          prompt tokens before and after optimisation
* ``cost_usd``          real spend, split by model
* ``guardrail_*``       which rules fire, and how often they block
* ``agent_*``           fan-out width, failures, wall-clock vs serial
"""

from __future__ import annotations

import math
import threading
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class Histogram:
    """Count/sum/min/max plus a p50/p95 approximation from bounded samples."""

    count: int = 0
    total: float = 0.0
    minimum: float = math.inf
    maximum: float = 0.0
    samples: list[float] = field(default_factory=list)

    def observe(self, value: float, *, keep: int = 2_000) -> None:
        self.count += 1
        self.total += value
        self.minimum = min(self.minimum, value)
        self.maximum = max(self.maximum, value)
        self.samples.append(value)
        if len(self.samples) > keep:
            del self.samples[: len(self.samples) - keep]

    @property
    def mean(self) -> float:
        return self.total / self.count if self.count else 0.0

    def percentile(self, pct: float) -> float:
        if not self.samples:
            return 0.0
        ordered = sorted(self.samples)
        index = min(len(ordered) - 1, int(round(pct / 100 * (len(ordered) - 1))))
        return ordered[index]

    def to_dict(self) -> dict[str, float | int]:
        return {
            "count": self.count,
            "total": round(self.total, 4),
            "mean": round(self.mean, 4),
            "min": 0.0 if self.minimum is math.inf else round(self.minimum, 4),
            "max": round(self.maximum, 4),
            "p50": round(self.percentile(50), 4),
            "p95": round(self.percentile(95), 4),
        }


class Metrics:
    """Thread-safe counters and histograms with optional label sets."""

    def __init__(self, *, enabled: bool = True, namespace: str = "agentflow") -> None:
        self.enabled = enabled
        self.namespace = namespace
        self._counters: dict[str, int] = defaultdict(int)
        self._labels: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
        self._histograms: dict[str, Histogram] = defaultdict(Histogram)
        self._lock = threading.Lock()

    # -- recording --------------------------------------------------------
    def incr(self, name: str, value: int = 1, **labels: Any) -> None:
        if not self.enabled:
            return
        with self._lock:
            self._counters[name] += value
            if labels:
                key = self._label_key(labels)
                self._labels[name][key] += value

    def observe(self, name: str, value: float, **labels: Any) -> None:
        if not self.enabled:
            return
        with self._lock:
            self._histograms[name].observe(value)

    def labelled(self, name: str, **labels: Any) -> int:
        with self._lock:
            return self._labels[name][self._label_key(labels)]

    @staticmethod
    def _label_key(labels: dict[str, Any]) -> str:
        return ",".join(f"{k}={labels[k]}" for k in sorted(labels))

    # -- reporting --------------------------------------------------------
    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return {
                "counters": dict(sorted(self._counters.items())),
                "labelled": {
                    name: dict(sorted(values.items()))
                    for name, values in self._labels.items()
                    if values
                },
                "histograms": {
                    name: hist.to_dict() for name, hist in sorted(self._histograms.items())
                },
            }

    def summary(self) -> dict[str, Any]:
        """The few numbers worth putting on a dashboard."""
        snap = self.snapshot()
        tokens_in = snap["counters"].get("tokens_input_total", 0)
        tokens_saved = snap["counters"].get("tokens_saved_total", 0)
        fanout = snap["counters"].get("agent_fanout_total", 0)
        wall = float(snap["histograms"].get("agent_wall_ms", {}).get("total", 0.0))
        serial = float(snap["histograms"].get("agent_serial_ms", {}).get("total", 0.0))
        cache_reads = snap["counters"].get("prompt_cache_reads_total", 0)
        cache_writes = snap["counters"].get("prompt_cache_writes_total", 0)
        return {
            "requests": snap["counters"].get("requests_total", 0),
            "tokens_input": tokens_in,
            "tokens_output": snap["counters"].get("tokens_output_total", 0),
            "tokens_saved": tokens_saved,
            "token_savings_ratio": round(tokens_saved / tokens_in, 4) if tokens_in else 0.0,
            "cost_usd": round(snap["counters"].get("cost_usd_micros", 0) / 1e6, 6),
            "guardrail_blocks": snap["counters"].get("guardrail_blocks_total", 0),
            "blocked_fraction": round(
                snap["counters"].get("guardrail_blocks_total", 0)
                / max(1, snap["counters"].get("requests_total", 0)),
                4,
            ),
            "agent_fanout_runs": fanout,
            "agents_per_run": round(fanout and (snap["counters"].get("agent_invocations_total", 0) / fanout) or 0, 2),
            "parallel_speedup": round(serial / wall, 2) if wall > 0 else 1.0,
            # A cache read only pays off against a write of the same prefix.
            "cache_hit_rate": round(
                cache_reads / (cache_reads + cache_writes) if cache_reads + cache_writes else 0.0,
                4,
            ),
            "intents": snap["labelled"].get("intent_total", {}),
            "models": snap["labelled"].get("route_model_total", {}),
        }

    def to_prometheus(self) -> str:
        """Render the current state in Prometheus text exposition format."""
        lines: list[str] = []

        def emit(name: str, value: float, kind: str, help_text: str) -> None:
            metric = f"{self.namespace}_{name}"
            lines.append(f"# HELP {metric} {help_text}")
            lines.append(f"# TYPE {metric} {kind}")
            lines.append(f"{metric} {value}")

        for name, value in sorted(self._counters.items()):
            emit(name, value, "counter", f"{name} counter")
        for name, values in sorted(self._labels.items()):
            if not values:
                continue
            metric = f"{self.namespace}_{name}"
            lines.append(f"# HELP {metric} {name}")
            lines.append(f"# TYPE {metric} counter")
            for key, value in sorted(values.items()):
                labels = ",".join(f"{k}=\"{v}\"" for k, _, v in (p.partition("=") for p in key.split(",")))
                lines.append(f"{metric}{{{labels}}} {value}")
        for name, hist in sorted(self._histograms.items()):
            emit(f"{name}_sum", round(hist.total, 4), "counter", f"{name} sum")
            emit(f"{name}_count", hist.count, "counter", f"{name} count")
        return "\n".join(lines) + "\n"

    def reset(self) -> None:
        with self._lock:
            self._counters.clear()
            self._labels.clear()
            self._histograms.clear()

metrics = Metrics()


def record_usage(
    usage: Any,
    *,
    model: str,
    latency_ms: float = 0.0,
    sink: Metrics | None = None,
) -> None:
    """Record one model call from any client's ``Usage``.

    Every provider implementation funnels through here, so token, cache and cost
    totals stay consistent whether the run was live or on the offline scripted
    client. Cost reuses ``tokens.budget.cost_usd`` so a telemetry total can never
    drift from what the router priced. Costs accumulate in micro-dollars because
    ``incr`` is integer-only and one frontier call is often well under a cent.
    """
    from agentflow.llm.catalog import catalog
    from agentflow.llm.types import Usage
    from agentflow.tokens.budget import cost_usd

    m = sink or metrics
    if not m.enabled:
        return

    usage = Usage(
        input_tokens=int(getattr(usage, "input_tokens", 0) or 0),
        output_tokens=int(getattr(usage, "output_tokens", 0) or 0),
        cache_read_input_tokens=int(getattr(usage, "cache_read_input_tokens", 0) or 0),
        cache_creation_input_tokens=int(getattr(usage, "cache_creation_input_tokens", 0) or 0),
    )

    m.incr("llm_calls_total", model=model)
    m.incr("tokens_input_total", usage.input_tokens)
    m.incr("tokens_output_total", usage.output_tokens)
    if latency_ms:
        m.observe("llm_latency_ms", latency_ms, model=model)

    if usage.cache_read_input_tokens:
        m.incr("prompt_cache_reads_total", usage.cache_read_input_tokens)
    if usage.cache_creation_input_tokens:
        m.incr("prompt_cache_writes_total", usage.cache_creation_input_tokens)

    try:
        spec = catalog.get(model)
    except Exception:  # noqa: BLE001 - an unknown model must not break accounting
        return
    m.incr("cost_usd_micros", int(round(cost_usd(spec, usage) * 1e6)))


__all__ = ["Histogram", "Metrics", "metrics", "record_usage"]
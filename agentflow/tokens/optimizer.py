"""Token optimisation.

Seven independent reductions, each measured so the savings claim is auditable
rather than asserted:

1. ``squeeze``          - lossless whitespace/control-character normalisation.
2. ``dedupe``           - drop blocks already present verbatim in the prefix.
3. ``compact``          - replace aged turns with a running summary.
4. ``prune_retrieval``  - drop retrieved chunks below a relevance floor.
5. ``shorten_tools``    - emit only the schema fields the model must see.
6. ``cap_output``       - bound ``max_tokens`` to what the task needs.
7. ``cache_prefix``     - lift the stable prefix into a cacheable system block.

Every stage reports ``tokens_before``/``tokens_after`` so
``OptimizationReport.saved_tokens`` is computed, not guessed.
"""

from __future__ import annotations

import json
import logging
import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any

from agentflow.llm.catalog import ModelSpec
from agentflow.llm.types import ContentBlock, Message, ToolSchema
from agentflow.telemetry import metrics
from agentflow.tokens.counter import (
    CHARS_PER_TOKEN,
    estimate_text_tokens,
    estimate_tokens,
    squeeze_whitespace,
)

_PRIORITY = ("cache_prefix", "dedupe", "compact", "prune_retrieval", "shorten_tools", "squeeze")

#: Marks the block the RAG pipeline injects retrieved chunks into.
CONTEXT_OPEN = "<retrieved_context>"

log = logging.getLogger("agentflow.tokens")


@dataclass(slots=True)
class StageResult:
    """Savings attribution for one optimisation stage."""

    stage: str
    tokens_before: int
    tokens_after: int
    detail: str = ""

    @property
    def saved(self) -> int:
        return max(0, self.tokens_before - self.tokens_after)

    def to_dict(self) -> dict[str, Any]:
        return {
            "stage": self.stage,
            "tokens_before": self.tokens_before,
            "tokens_after": self.tokens_after,
            "saved": self.saved,
            "detail": self.detail,
        }


@dataclass(slots=True)
class OptimizationReport:
    """Aggregate result of an :class:`TokenOptimizer` run."""

    stages: list[StageResult] = field(default_factory=list)
    applied: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    @property
    def tokens_before(self) -> int:
        return self.stages[0].tokens_before if self.stages else 0

    @property
    def tokens_after(self) -> int:
        return self.stages[-1].tokens_after if self.stages else 0

    @property
    def saved_tokens(self) -> int:
        return max(0, self.tokens_before - self.tokens_after)

    @property
    def saved_ratio(self) -> float:
        return self.saved_tokens / self.tokens_before if self.tokens_before else 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "tokens_before": self.tokens_before,
            "tokens_after": self.tokens_after,
            "saved_tokens": self.saved_tokens,
            "saved_ratio": round(self.saved_ratio, 4),
            "applied": list(self.applied),
            "stages": [s.to_dict() for s in self.stages],
            "notes": list(self.notes),
        }


@dataclass(slots=True)
class OptimizedPayload:
    """The reduced request plus the report that justifies the reduction."""

    messages: list[Message]
    system: str | list[ContentBlock] | None
    tools: list[ToolSchema] | None
    retrieved: list[Any] = field(default_factory=list)
    report: OptimizationReport = field(default_factory=OptimizationReport)

    @property
    def prompt_tokens(self) -> int:
        return self.report.tokens_after

    def __iter__(self):  # tuple-style unpacking for ergonomic call sites
        return iter((self.messages, self.system, self.tools, self.report))


class TokenOptimizer:
    """Stateless, order-dependent reducer applied before every model call."""

    def __init__(
        self,
        *,
        cache_min_tokens: int = 1_024,
        retrieval_floor: float = 0.35,
        compact_after_turns: int = 8,
        max_tool_schema_tokens: int = 200,
    ) -> None:
        self.cache_min_tokens = cache_min_tokens
        self.retrieval_floor = retrieval_floor
        self.compact_after_turns = compact_after_turns
        self.max_tool_schema_tokens = max_tool_schema_tokens

    # -- measurement ------------------------------------------------------
    @staticmethod
    def measure(
        messages: Sequence[Message],
        system: str | list[ContentBlock] | None = None,
        tools: Sequence[ToolSchema] | None = None,
    ) -> int:
        total = estimate_tokens(list(messages), system if isinstance(system, str) else None)
        if isinstance(system, list):
            total += sum(estimate_text_tokens(b.get("text", "")) for b in system)
        for tool in tools or ():
            total += estimate_text_tokens(tool.description)
            total += estimate_text_tokens(json.dumps(tool.input_schema))
        return total

    # -- public entry point ----------------------------------------------
    def optimize(
        self,
        messages: list[Message],
        *,
        system: str | list[ContentBlock] | None = None,
        tools: list[ToolSchema] | None = None,
        retrieved: list[Any] | None = None,
        summary: str | None = None,
        enable: Iterable[str] | None = None,
    ) -> OptimizedPayload:
        """Reduce a payload and report what each stage saved.

        The payload object carries the reduced messages/system/tools plus the
        report, so a caller can cache-prefix, budget-check and send in one step.
        """
        allowed = set(enable) if enable is not None else set(_PRIORITY)

        work = OptimizedPayload(
            messages=[
                Message(m.role, [ContentBlock(**b) for b in m.content])
                if isinstance(m.content, list)
                else Message(m.role, m.content)
                for m in messages
            ],
            system=system,
            tools=list(tools) if tools else None,
            retrieved=list(retrieved or []),
        )
        report = work.report
        baseline = self.measure(work.messages, work.system, work.tools)
        report.stages.append(StageResult("input", baseline, baseline, "unoptimised baseline"))

        def checkpoint(stage: str, detail: str = "") -> None:
            report.stages.append(
                StageResult(
                    stage,
                    report.stages[-1].tokens_after,
                    self.measure(work.messages, work.system, work.tools),
                    detail,
                )
            )
            report.applied.append(stage)

        if "cache_prefix" in allowed:
            work.system = self._cache_prefix(system, report)
            checkpoint("cache_prefix", "stable prefix lifted for prompt caching")
        if "squeeze" in allowed:
            work.messages, work.system = self._squeeze(work.messages, work.system)
            checkpoint("squeeze", "whitespace and control characters normalised")
        if "dedupe" in allowed:
            work.messages = self._dedupe(work.messages)
            checkpoint("dedupe", "verbatim duplicate tool results removed")
        if "compact" in allowed and summary:
            before = len(work.messages)
            work.messages = self._compact(work.messages, summary)
            checkpoint("compact", f"{before} turns -> {len(work.messages)} turns")
        if "prune_retrieval" in allowed and work.retrieved:
            kept = len(work.retrieved)
            work.retrieved = self._prune_retrieval(work.retrieved)
            work.system, work.messages = self._apply_retrieval(
                work.system, work.messages, work.retrieved
            )
            checkpoint(
                "prune_retrieval",
                f"{kept} -> {len(work.retrieved)} chunks at floor {self.retrieval_floor}",
            )
        if "shorten_tools" in allowed and work.tools:
            work.tools = self._shorten_tools(work.tools)
            checkpoint("shorten_tools", f"{len(work.tools)} tool schemas trimmed")

        if report.saved_tokens:
            metrics.incr("tokens_saved_total", report.saved_tokens)
            metrics.incr("optimize_calls_total")
            for stage in report.applied:
                metrics.incr("optimize_stage_total", stage=stage)

        return work

    # -- stages -----------------------------------------------------------
    def _cache_prefix(
        self, system: str | list[ContentBlock] | None, report: OptimizationReport
    ) -> str | list[ContentBlock]:
        """Split a long system prompt into [stable policy | volatile request].

        Only the stable part gets a cache breakpoint. Marking the whole thing
        would invalidate the cache on every request that carries a timestamp or a
        user id, which is the usual way teams accidentally get zero cache hits.
        """
        if not isinstance(system, str):
            return system
        blocks: list[ContentBlock] = []
        if len(system.strip()) < 40:
            return system

        static_part, volatile_part = split_system_prompt(system)
        if not static_part:
            return system
        blocks.append(ContentBlock.text(static_part))
        if volatile_part:
            blocks.append(ContentBlock.text(volatile_part))
        report.notes.append(
            f"cacheable static prefix ~{estimate_text_tokens(static_part)} tokens "
            f"(>= {self.cache_min_tokens} minimum: "
            f"{estimate_text_tokens(static_part) >= self.cache_min_tokens})"
        )
        return blocks

    def _squeeze(
        self, messages: list[Message], system: str | list[ContentBlock] | None
    ) -> tuple[list[Message], str | list[ContentBlock] | None]:
        out: list[Message] = []
        for message in messages:
            if isinstance(message.content, str):
                out.append(Message(message.role, squeeze_whitespace(message.content)))
                continue
            blocks = [
                (
                    ContentBlock.text(squeeze_whitespace(b.get("text", "")))
                    if b.get("type") == "text"
                    else ContentBlock(**b)
                )
                for b in message.content
            ]
            out.append(Message(message.role, blocks))

        if isinstance(system, str):
            return out, squeeze_whitespace(system)
        if isinstance(system, list):
            return out, [
                ContentBlock.text(squeeze_whitespace(b.get("text", "")))
                if b.get("type") == "text"
                else ContentBlock(**b)
                for b in system
            ]
        return out, system

    @staticmethod
    def _dedupe(messages: list[Message]) -> list[Message]:
        """Drop tool_result blocks whose content repeats an earlier turn verbatim.

        Agent loops echo tool output back into context; the model already has it,
        so a second copy is pure spend.
        """
        seen: set[str] = set()
        out: list[Message] = []
        for message in messages:
            if isinstance(message.content, str):
                key = message.content.strip()
                if key and key in seen and len(out):
                    continue
                if key:
                    seen.add(key)
                out.append(message)
                continue
            blocks: list[ContentBlock] = []
            for block in message.content:
                if block.get("type") == "tool_result":
                    body = str(block.get("content", "")).strip()
                    if body and body in seen:
                        continue
                    if body:
                        seen.add(body)
                blocks.append(ContentBlock(**block))
            out.append(Message(message.role, blocks))
        return out

    @staticmethod
    def _compact(messages: list[Message], summary: str) -> list[Message]:
        """Keep the summary + the last few verbatim turns, drop the middle."""
        if len(messages) <= 2:
            return messages
        tail = messages[-2:]
        head = messages[:1]
        compacted = [
            Message("user", f"[Compacted earlier conversation]\n{summary}"),
            *head,
        ] if head else [Message("user", f"[Compacted earlier conversation]\n{summary}")]
        # preserve ordering: summary, then tail
        return [*compacted, *tail]

    def _prune_retrieval(self, retrieved: list[Any]) -> list[Any]:
        kept = [c for c in retrieved if _score(c) >= self.retrieval_floor]
        # Never return nothing: fall back to the single best chunk.
        return kept or retrieved[:1]

    @staticmethod
    def _render_context(retrieved: list[Any]) -> str:
        body = "\n\n".join(f"[{i + 1}] {_text(c)}" for i, c in enumerate(retrieved))
        return f"<retrieved_context>\n{body}\n</retrieved_context>"

    def _apply_retrieval(
        self,
        system: str | list[ContentBlock] | None,
        messages: list[Message],
        retrieved: list[Any],
    ) -> tuple[str | list[ContentBlock] | None, list[Message]]:
        """Rewrite the retrieval context block in place from the pruned chunk set.

        RAG callers put ``<retrieved_context>`` in the system prompt, so both the
        system blocks and the message list have to be searched. Appending a fresh
        user message when no block exists would smuggle context in as if the user
        had typed it, so instead we report the mismatch and leave the prompt alone.
        """
        if not retrieved:
            return system, messages
        block = self._render_context(retrieved)

        if isinstance(system, str) and CONTEXT_OPEN in system:
            return system.split(CONTEXT_OPEN)[0] + block, messages

        if isinstance(system, list):
            for index, item in enumerate(system):
                if not isinstance(item, ContentBlock) or CONTEXT_OPEN not in str(item.get("text", "")):
                    continue
                head = str(item.get("text", "")).split(CONTEXT_OPEN)[0]
                rebuilt = list(system)
                rebuilt[index] = ContentBlock(type=str(item.get("type", "text")), text=head + block)
                return rebuilt, messages

        for index, message in enumerate(messages):
            if isinstance(message.content, str) and CONTEXT_OPEN in message.content:
                head = message.content.split(CONTEXT_OPEN)[0]
                rebuilt = list(messages)
                rebuilt[index] = Message(message.role, head + block)
                return system, rebuilt

        log.debug("retrieved chunks pruned but no <retrieved_context> block to rewrite")
        return system, messages

    def _shorten_tools(self, tools: list[ToolSchema]) -> list[ToolSchema]:
        """Collapse tool schemas whose JSON blows past the per-tool ceiling.

        Tool schemas are re-sent on every single call and cache poorly, so
        trimming descriptions is usually the highest-leverage saving available.
        """
        out: list[ToolSchema] = []
        for tool in tools:
            cost = estimate_text_tokens(tool.description) + estimate_text_tokens(
                json.dumps(tool.input_schema)
            )
            if cost <= self.max_tool_schema_tokens:
                out.append(tool)
                continue
            out.append(
                ToolSchema(
                    name=tool.name,
                    description=tool.description[: self.max_tool_schema_tokens * 3].strip(),
                    input_schema=_minimal_schema(tool.input_schema),
                )
            )
        return out


# -- helpers ---------------------------------------------------------------
_VOLATILE = re.compile(
    r"(?:^|\n)\s*(?:#+\s*)?(?:current\s+(?:date|time|user|session)|"
    r"request\s+id|conversation\s+id|now\s*[:=]|timestamp)\b",
    re.IGNORECASE,
)


def split_system_prompt(system: str) -> tuple[str, str]:
    """Split into (stable prefix, volatile tail).

    Section based: everything up to the first volatile section header is treated
    as stable policy, which is what the provider can cache across requests.
    """
    lines = system.split("\n")
    for index, line in enumerate(lines):
        if _VOLATILE.search(line):
            static = "\n".join(lines[:index]).rstrip()
            volatile = "\n".join(lines[index:]).strip()
            if len(static) >= 200:
                return static, volatile
    return system.strip(), ""


def _minimal_schema(schema: dict[str, Any]) -> dict[str, Any]:
    required = set(schema.get("required", []))
    properties = {
        name: {k: v for k, v in spec.items() if k in {"type", "enum", "description"}}
        for name, spec in schema.get("properties", {}).items()
        if name in required
    }
    return {"type": "object", "properties": properties, "required": sorted(required)}


def _score(chunk: Any) -> float:
    """Relevance of a retrieved chunk, on a 0..1 scale.

    ``score`` from the retriever is a raw RRF value that lives near 0.01, so it must
    never be compared against a floor like 0.15. ``rerank_score`` is already
    normalised, and is preferred whenever the pipeline set it.
    """
    if isinstance(chunk, dict):
        if chunk.get("rerank_score") is not None:
            return float(chunk["rerank_score"])
        return float(chunk.get("score", 0.0))
    if getattr(chunk, "rerank_score", None) is not None:
        return float(chunk.rerank_score)
    return float(getattr(chunk, "score", 0.0))


def _text(chunk: Any) -> str:
    if isinstance(chunk, dict):
        return str(chunk.get("text", ""))
    return str(getattr(chunk, "text", ""))


# -- output capping --------------------------------------------------------
_OUTPUT_CAPS = {
    "triage": 200,
    "extractive": 400,
    "chat": 1_200,
    "analysis": 1_600,
    "rag": 1_600,
    "sql": 1_200,
    "agentic": 2_000,
    "code": 4_000,
}


def cap_output_tokens(task: str, requested: int | None, *, spec: ModelSpec | None = None) -> int:
    """Bound output tokens to what the task shape actually needs."""
    ceiling = _OUTPUT_CAPS.get(task, 1_200)
    if requested:
        ceiling = min(ceiling, requested)
    if spec:
        ceiling = min(ceiling, spec.max_output_tokens)
    return max(64, ceiling)


# -- savings estimation ----------------------------------------------------
def project_cached_savings(
    spec: ModelSpec,
    static_prefix_tokens: int,
    calls_per_session: int,
    cache_hit_ratio: float = 0.9,
) -> dict[str, Any]:
    """USD saved by caching a stable prefix across a session."""
    per_hit = static_prefix_tokens / 1e6 * spec.cache_read_cost_per_mtok
    per_miss = static_prefix_tokens / 1e6 * spec.cache_write_cost_per_mtok
    uncached = static_prefix_tokens / 1e6 * spec.input_cost_per_mtok

    hits = calls_per_session * cache_hit_ratio
    misses = calls_per_session - hits
    with_cache = hits * per_hit + misses * per_miss
    without_cache = calls_per_session * uncached
    return {
        "model": spec.id,
        "static_prefix_tokens": static_prefix_tokens,
        "calls_per_session": calls_per_session,
        "cache_hit_ratio": cache_hit_ratio,
        "usd_with_cache": round(with_cache, 6),
        "usd_without_cache": round(without_cache, 6),
        "usd_saved": round(without_cache - with_cache, 6),
        "savings_pct": round((without_cache - with_cache) / without_cache, 4) if without_cache else 0.0,
    }


def measure_prefix_reuse(
    prefixes: Sequence[str],
) -> dict[str, Any]:
    """How much of a prefix is repeated across calls, i.e. cacheable fraction."""
    if not prefixes:
        return {"samples": 0, "cacheable_prefix_tokens": 0, "reuse_ratio": 0.0}
    common = prefixes[0]
    for candidate in prefixes[1:]:
        limit = min(len(common), len(candidate))
        cut = 0
        for i in range(limit):
            if common[i] != candidate[i]:
                break
            cut = i + 1
        common = common[:cut]
    tokens = int(len(common) / CHARS_PER_TOKEN)
    avg = sum(estimate_text_tokens(p) for p in prefixes) / len(prefixes)
    return {
        "samples": len(prefixes),
        "cacheable_prefix_tokens": tokens,
        "avg_prompt_tokens": int(avg),
        "reuse_ratio": round(tokens / avg, 4) if avg else 0.0,
    }


__all__ = [
    "OptimizedPayload",
    "OptimizationReport",
    "StageResult",
    "TokenOptimizer",
    "cap_output_tokens",
    "measure_prefix_reuse",
    "project_cached_savings",
    "split_system_prompt",
]
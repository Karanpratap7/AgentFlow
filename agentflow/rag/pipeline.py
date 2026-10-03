"""The RAG pipeline.

    ingest:  documents -> chunk -> embed -> hybrid index
    query:   question
              -> guardrail check
              -> auto intent detection
              -> query rewrite (decompose / normalise)
              -> hybrid retrieve (k1)
              -> rerank
              -> context compression (drop to k2 by rerank score)
              -> auto-router model selection
              -> grounded generation with mandatory citations
              -> hallucination check (are the citations real?)
              -> output guardrails
              -> answer + trace

The trace returned alongside every answer is what makes the pipeline auditable:
it records the intent, the model the router chose, the chunks that survived
compression and how many tokens each stage cost.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from agentflow.config import Settings
from agentflow.config import settings as default_settings
from agentflow.errors import NotIndexedError
from agentflow.guardrails.pipeline import GuardrailPipeline
from agentflow.llm.types import LLMClient, Message
from agentflow.rag.chunking import Chunker, count_tokens
from agentflow.rag.embeddings import Embedder, default_embedder
from agentflow.rag.store import HybridVectorStore, RetrievedChunk
from agentflow.routing.intent import IntentDetector, IntentResult
from agentflow.routing.router import AutoRouter, RoutingDecision
from agentflow.telemetry import metrics
from agentflow.tokens.counter import estimate_text_tokens
from agentflow.tokens.optimizer import TokenOptimizer

SYSTEM_PROMPT = """You are AgentFlow's retrieval-augmented assistant.

Answer only from the retrieved context below. If the context does not contain the
answer, say so explicitly instead of guessing.

Cite every claim with the bracketed chunk marker it came from, for example [1].
Keep the answer under {max_words} words and lead with the direct answer.

## Retrieved context
{context}
"""

_CITATION = re.compile(r"\[(\d+)\]")


@dataclass(slots=True)
class RAGTrace:
    """Everything that happened, for debugging and for the API response."""

    question: str
    intent: IntentResult | None = None
    routing: RoutingDecision | None = None
    retrieved: list[RetrievedChunk] = field(default_factory=list)
    kept: list[RetrievedChunk] = field(default_factory=list)
    context_tokens: int = 0
    prompt_tokens: int = 0
    optimization: dict[str, Any] = field(default_factory=dict)
    cited_markers: list[int] = field(default_factory=list)
    unsupported_citations: list[int] = field(default_factory=list)
    grounded: bool = True
    latency_ms: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "question": self.question,
            "intent": self.intent.to_dict() if self.intent else None,
            "routing": self.routing.to_dict() if self.routing else None,
            "retrieved_chunk_ids": [c.chunk.chunk_id for c in self.retrieved],
            "kept_chunk_ids": [c.chunk.chunk_id for c in self.kept],
            "retrieved": len(self.retrieved),
            "kept": len(self.kept),
            "context_tokens": self.context_tokens,
            "prompt_tokens": self.prompt_tokens,
            "optimization": self.optimization,
            "cited_markers": list(self.cited_markers),
            "unsupported_citations": list(self.unsupported_citations),
            "grounded": self.grounded,
            "latency_ms": round(self.latency_ms, 2),
        }


@dataclass(slots=True)
class RAGAnswer:
    text: str
    trace: RAGTrace
    guardrails: dict[str, Any] = field(default_factory=dict)


class RAGPipeline:
    """Ingest + retrieval-augmented generation, with a full audit trail."""

    def __init__(
        self,
        client: LLMClient,
        *,
        settings: Settings | None = None,
        embedder: Embedder | None = None,
        store: HybridVectorStore | None = None,
        chunker: Chunker | None = None,
        guardrails: GuardrailPipeline | None = None,
        router: AutoRouter | None = None,
        intent_detector: IntentDetector | None = None,
        optimizer: TokenOptimizer | None = None,
        top_k: int = 8,
        keep_k: int = 4,
        compress_below: float = 0.15,
        relative_floor: float = 0.6,
        max_words: int = 180,
    ) -> None:
        self.settings = settings or default_settings
        self.client = client
        self.embedder = embedder or default_embedder()
        self.chunker = chunker or Chunker()
        # ``is None``, not ``or``: HybridVectorStore defines __len__, so an empty
        # store passed in by a caller is falsy and would be swapped for a new one,
        # throwing away an index the caller already built.
        self.store = (
            store
            if store is not None
            else HybridVectorStore(self.embedder, persist_dir=self.settings.index_dir)
        )
        self.guardrails = guardrails or GuardrailPipeline(self.settings)
        self.router = router or AutoRouter(self.settings)
        self.intent_detector = intent_detector or IntentDetector(client)
        self.optimizer = optimizer or TokenOptimizer(cache_min_tokens=self.settings.cache_min_tokens)
        self.top_k = top_k
        self.keep_k = keep_k
        self.compress_below = compress_below
        self.relative_floor = relative_floor
        self.max_words = max_words

    # -- ingest -----------------------------------------------------------
    def ingest(
        self,
        documents: dict[str, str],
        *,
        chunker: Chunker | None = None,
        persist: bool = False,
    ) -> list[Any]:
        chunks = self.store.index_documents(documents, chunker=chunker or self.chunker)
        if persist:
            self.store.save()
        metrics.incr("rag_ingests_total")
        metrics.incr("rag_chunks_indexed_total", len(chunks))
        return chunks

    def ingest_files(self, paths: list[Path], *, persist: bool = True) -> list[Any]:
        documents = {p.stem: p.read_text(encoding="utf-8") for p in paths}
        return self.ingest(documents, persist=persist)

    def load(self) -> int:
        return self.store.load()

    # -- query ------------------------------------------------------------
    def ask(self, question: str, *, history: list[Message] | None = None) -> RAGAnswer:
        if self.store.is_empty:
            self.load()
        if self.store.is_empty:
            raise NotIndexedError("RAG index is empty; ingest documents first")

        from agentflow.llm.types import now_ms

        started = now_ms()
        trace = RAGTrace(question=question)

        # 1. safety before any spend
        self.guardrails.check_input(question)

        # 2. intent
        trace.intent = self.intent_detector.detect(question)

        # 3. rewrite
        rewritten = self.rewrite(question, trace)

        # 4. hybrid recall
        trace.retrieved = self.store.search(rewritten, k=self.top_k)

        # 5. rerank
        ranked = self.store.rerank(list(trace.retrieved), rewritten)

        # 6. compress
        trace.kept = self._compress(ranked)
        context = self._format_context(trace.kept)
        trace.context_tokens = count_tokens(context)

        # 7. route
        trace.routing = self.router.route(
            question,
            intent=trace.intent,
            expected_input_tokens=trace.context_tokens + estimate_text_tokens(question),
            requires_retrieval=True,
            cache_prefix_tokens=trace.context_tokens,
        )

        # 8. token optimisation, then generate.
        # Chunks are pruned before the context is rendered (step 6), so the prompt
        # and ``trace.kept`` cannot disagree about how much context was sent.
        system = SYSTEM_PROMPT.format(
            context=context,
            max_words=self.max_words,
        )
        payload = self.optimizer.optimize(
            [*(history or []), Message("user", rewritten)],
            system=system,
            summary=history[0].plain_text() if history else None,
        )
        trace.prompt_tokens = payload.report.tokens_after
        trace.optimization = payload.report.to_dict()
        trace.optimization["chunks_pruned"] = {
            "retrieved": len(ranked),
            "kept": len(trace.kept),
            "floor": self.compress_below,
            "relative_floor": self.relative_floor,
            "tokens_saved": max(0, trace.prompt_tokens - trace.context_tokens),
        }

        response = self.client.complete(
            payload.messages,
            system=payload.system,
            model=trace.routing.model,
            max_output_tokens=trace.routing.max_output_tokens,
            temperature=0.0,
        )

        # 9. grounded-ness + output safety
        trace.cited_markers = sorted({int(m) for m in _CITATION.findall(response.text)})
        trace.unsupported_citations = [
            m for m in trace.cited_markers if not 1 <= m <= len(trace.kept)
        ]
        trace.grounded = not trace.unsupported_citations
        guard = self.guardrails.check_output(response.text, context=question)
        trace.latency_ms = now_ms() - started

        metrics.incr("requests_total")
        metrics.incr("rag_queries_total")
        metrics.incr("route_model_total", model=trace.routing.model)
        metrics.observe("rag_prompt_tokens", trace.prompt_tokens)
        metrics.observe("rag_chunks_kept", len(trace.kept))
        metrics.observe("rag_latency_ms", trace.latency_ms)
        if not trace.grounded:
            metrics.incr("rag_ungrounded_total")

        return RAGAnswer(text=guard.text, trace=trace, guardrails=guard.to_dict())

    # -- stages -----------------------------------------------------------
    def rewrite(self, question: str, trace: RAGTrace) -> str:
        """Turn a conversational question into a retrieval-shaped one.

        Strips filler that dilutes BM25, then appends sub-questions for compound
        queries so recall covers every conjunct instead of just the first.
        """
        cleaned = re.sub(
            r"^\s*(?:hey|hi|hello|ok|okay|so|well|now|um)\b[,!\s]*",
            "",
            question.strip(),
            flags=re.IGNORECASE,
        ).strip()

        sub = _split_compound(cleaned)
        if sub:
            trace.optimization.setdefault("query", {})
            trace.optimization["query"] = {"rewritten": cleaned, "sub_questions": sub}
            return " ".join([cleaned, *sub])
        return cleaned

    def _compress(self, ranked: list[RetrievedChunk]) -> list[RetrievedChunk]:
        """Keep the chunks that clear the relevance floor, capped at ``keep_k``.

        The floor is relative as well as absolute: a chunk only earns context space
        if it scores at least ``relative_floor`` of the best hit. Without that, a
        single confident match drags in whatever merely scrapes past the absolute
        threshold, and the model pays tokens for a passage about an unrelated
        document. The absolute floor still applies, and the top hit is always
        kept so the context can never come back empty.
        """
        floor = self.compress_below
        scores = [(c.rerank_score or 0.0) for c in ranked]
        relative = max(scores) * self.relative_floor if scores else 0.0
        threshold = max(floor, relative)
        kept = [c for c, score in zip(ranked, scores, strict=True) if score >= threshold][: self.keep_k]
        return kept or ranked[:1]

    @staticmethod
    def _format_context(chunks: list[RetrievedChunk]) -> str:
        blocks = []
        for i, chunk in enumerate(chunks, start=1):
            where = f" — {chunk.chunk.breadcrumb}" if chunk.chunk.breadcrumb else ""
            blocks.append(f"[{i}] {chunk.chunk.text}{where}")
        return "\n\n".join(blocks)

    # -- reporting --------------------------------------------------------
    def stats(self) -> dict[str, Any]:
        return {
            **self.store.stats(),
            "top_k": self.top_k,
            "keep_k": self.keep_k,
            "compression_ratio": round(self.keep_k / self.top_k, 3) if self.top_k else 1.0,
            "rerank_floor": self.compress_below,
            "relative_floor": self.relative_floor,
        }


_COMPOUND = re.compile(
    r"\band\b(?:\s+(?:also|then|how|what|why|does|do|is|are)\b|\s+to\b)", re.IGNORECASE
)


def _split_compound(question: str) -> list[str]:
    """Extract the trailing conjunct of an ``and ...`` question."""
    match = _COMPOUND.search(question)
    if not match:
        return []
    tail = question[match.end() :].strip(" ?.")
    if len(tail.split()) < 3:
        return []
    return [tail]


def cite_sources(chunks: list[RetrievedChunk]) -> str:
    return "\n".join(f"[{i}] {c.citation}" for i, c in enumerate(chunks, start=1))


__all__ = ["SYSTEM_PROMPT", "RAGAnswer", "RAGPipeline", "RAGTrace", "cite_sources"]
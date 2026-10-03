"""Hybrid vector store: dense recall + lexical recall + RRF fusion.

Pure vector search misses exact identifiers (an error code, a table name, a
ticket ID); pure keyword search misses paraphrase. Running both and fusing with
Reciprocal Rank Fusion gets the best of each without needing a cross-encoder.

The index persists to a single ``.npz`` plus a JSON sidecar, so an index built
once survives restarts and never needs a vector database to run.
"""

from __future__ import annotations

import json
import math
import re
from collections import Counter, defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from agentflow.errors import NotIndexedError
from agentflow.rag.chunking import Chunk, count_tokens
from agentflow.rag.embeddings import Embedder, l2_normalize

_WORD = re.compile(r"[a-z0-9]+")
_TOKEN_IDF_SMOOTH = 1.0


@dataclass(slots=True)
class RetrievedChunk:
    """One search hit, with the full score breakdown kept for introspection."""

    chunk: Chunk
    score: float
    dense_rank: int | None = None
    lexical_rank: int | None = None
    rrf_score: float = 0.0
    rerank_score: float | None = None
    token_count: int = 0

    @property
    def text(self) -> str:
        return self.chunk.text

    @property
    def citation(self) -> str:
        where = self.chunk.breadcrumb or self.chunk.doc_id
        return f"{self.chunk.chunk_id} ({where})"

    def to_dict(self) -> dict[str, Any]:
        return {
            "chunk_id": self.chunk.chunk_id,
            "doc_id": self.chunk.doc_id,
            "breadcrumb": self.chunk.breadcrumb,
            "score": round(self.score, 6),
            "rrf_score": round(self.rrf_score, 6),
            "rerank_score": round(self.rerank_score, 6) if self.rerank_score is not None else None,
            "dense_rank": self.dense_rank,
            "lexical_rank": self.lexical_rank,
            "token_count": self.token_count or self.chunk.token_count,
            "text": self.chunk.text,
            "citation": self.citation,
        }


def _analyse(text: str) -> list[str]:
    return _WORD.findall(text.lower())


def _bm25_scores(
    query_terms: Sequence[str],
    doc_term_freqs: Sequence[Counter[str]],
    doc_lengths: Sequence[int],
    avg_length: float,
    k1: float = 1.5,
    b: float = 0.75,
) -> list[float]:
    n = len(doc_term_freqs)
    df: Counter[str] = Counter()
    for tf in doc_term_freqs:
        df.update(tf.keys())

    scores = [0.0] * n
    for term in query_terms:
        n_q = df.get(term, 0)
        if n_q == 0:
            continue
        idf = math.log(1 + (n - n_q + _TOKEN_IDF_SMOOTH) / (n_q + _TOKEN_IDF_SMOOTH))
        for i, tf in enumerate(doc_term_freqs):
            freq = tf.get(term, 0)
            if not freq:
                continue
            denom = freq + k1 * (1 - b + b * doc_lengths[i] / (avg_length or 1.0))
            scores[i] += idf * (freq * (k1 + 1)) / denom
    return scores


class HybridVectorStore:
    """In-memory dense + BM25 index with Reciprocal Rank Fusion."""

    def __init__(
        self,
        embedder: Embedder,
        *,
        rrf_k: int = 60,
        persist_dir: Path | None = None,
    ) -> None:
        self.embedder = embedder
        self.rrf_k = rrf_k
        self.persist_dir = persist_dir

        self.chunks: list[Chunk] = []
        self._matrix: np.ndarray = np.zeros((0, embedder.dimensions), dtype=np.float32)
        self._term_freqs: list[Counter[str]] = []
        self._doc_lengths: list[int] = []
        self._avg_length: float = 0.0
        self._by_id: dict[str, int] = {}

    # -- indexing ---------------------------------------------------------
    def __len__(self) -> int:
        return len(self.chunks)

    @property
    def is_empty(self) -> bool:
        return not self.chunks

    def add(self, chunks: Iterable[Chunk], *, embed: bool = True) -> int:
        incoming = list(chunks)
        if not incoming:
            return 0
        vectors = (
            l2_normalize(np.vstack([self.embedder.embed_one(_indexable(c)) for c in incoming]))
            if embed
            else np.zeros((len(incoming), self.embedder.dimensions), dtype=np.float32)
        )
        if self._matrix.shape[0] == 0:
            self._matrix = vectors
        else:
            self._matrix = np.vstack([self._matrix, vectors])
        for chunk in incoming:
            terms = _analyse(chunk.text)
            self._term_freqs.append(Counter(terms))
            self._doc_lengths.append(len(terms))
            self._by_id[chunk.chunk_id] = len(self.chunks)
            self.chunks.append(chunk)
        self._avg_length = sum(self._doc_lengths) / len(self._doc_lengths)
        return len(incoming)

    def index_documents(
        self, documents: dict[str, str], *, chunker: Any | None = None, metadata: dict | None = None
    ) -> list[Chunk]:
        from agentflow.rag.chunking import Chunker

        chunker = chunker or Chunker()
        made: list[Chunk] = []
        for doc_id, text in documents.items():
            made.extend(chunker.split(text, doc_id=doc_id, metadata=metadata))
        self.add(made)
        return made

    # -- retrieval --------------------------------------------------------
    def search(
        self,
        query: str,
        *,
        k: int = 8,
        dense_weight: float = 0.6,
        candidate_multiplier: int = 4,
    ) -> list[RetrievedChunk]:
        """Hybrid search: dense and BM25, fused with RRF."""
        if self.is_empty:
            raise NotIndexedError("index is empty; call index_documents() first")

        pool = min(len(self.chunks), max(k * candidate_multiplier, k))
        dense_order = self._dense_rank(query, pool)
        lexical_order = self._lexical_rank(query, pool)

        fused: dict[str, float] = defaultdict(float)
        for rank, idx in enumerate(dense_order, start=1):
            fused[self.chunks[idx].chunk_id] += dense_weight / (self.rrf_k + rank)
        for rank, idx in enumerate(lexical_order, start=1):
            fused[self.chunks[idx].chunk_id] += (1 - dense_weight) / (self.rrf_k + rank)

        dense_rank = {idx: r for r, idx in enumerate(dense_order, start=1)}
        lexical_rank = {idx: r for r, idx in enumerate(lexical_order, start=1)}

        scored = [
            RetrievedChunk(
                chunk=self.chunks[idx],
                score=fused[self.chunks[idx].chunk_id],
                rrf_score=fused[self.chunks[idx].chunk_id],
                dense_rank=dense_rank.get(idx),
                lexical_rank=lexical_rank.get(idx),
                token_count=self.chunks[idx].token_count,
            )
            for idx in sorted(self._by_id.values())
            if self.chunks[idx].chunk_id in fused
        ]
        scored.sort(key=lambda c: -c.rrf_score)
        return scored[:k]

    def _dense_rank(self, query: str, pool: int) -> list[int]:
        if self._matrix.shape[0] == 0:
            return []
        vector = self.embedder.embed_one(_indexable_query(query)).reshape(1, -1)
        sims = (self._matrix @ vector.T).ravel()
        order = np.argsort(-sims)[:pool]
        return [int(i) for i in order]

    def _lexical_rank(self, query: str, pool: int) -> list[int]:
        terms = _analyse(query)
        if not terms or not self._term_freqs:
            return []
        scores = _bm25_scores(terms, self._term_freqs, self._doc_lengths, self._avg_length)
        order = sorted(range(len(scores)), key=lambda i: -scores[i])[:pool]
        return [i for i in order if scores[i] > 0]

    # -- reranking --------------------------------------------------------
    @staticmethod
    def rerank(
        hits: list[RetrievedChunk],
        query: str,
        *,
        boost_exact: float = 0.25,
        length_penalty: float = 0.15,
        proximity_bonus: float = 0.1,
    ) -> list[RetrievedChunk]:
        """Cheap feature-based rerank: exact-term coverage + proximity + length fit.

        Cross-encoder rerankers are the usual next step; this keeps the pipeline
        dependency-free while still fixing the ordering problems pure RRF leaves.
        """
        q_terms = set(_analyse(query))
        for hit in hits:
            d_terms = _analyse(hit.chunk.text)
            d_set = set(d_terms)
            coverage = len(q_terms & d_set) / len(q_terms) if q_terms else 0.0
            proximity = _proximity(d_terms, q_terms)
            target, spread = 220, 120
            length_fit = math.exp(-abs(hit.chunk.token_count - target) / spread)
            score = (
                0.55 * coverage
                + boost_exact * proximity
                + 0.2 * length_fit
                - length_penalty * min(1.0, hit.chunk.token_count / 600)
            )
            hit.rerank_score = round(score, 6)
        hits.sort(key=lambda h: -float(h.rerank_score or 0.0))
        return hits

    # -- persistence ------------------------------------------------------
    def save(self, path: Path | None = None) -> Path:
        target = path or (self.persist_dir / "index.npz" if self.persist_dir else None)
        if target is None:
            raise ValueError("no persist_dir configured")
        target.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(target, matrix=self._matrix)
        target.with_suffix(".chunks.json").write_text(
            json.dumps([c.to_dict() for c in self.chunks], indent=1), encoding="utf-8"
        )
        return target

    def load(self, path: Path | None = None) -> int:
        source = path or (self.persist_dir / "index.npz" if self.persist_dir else None)
        if source is None or not source.exists():
            return 0
        with np.load(source) as data:
            self._matrix = data["matrix"]
        sidecar = source.with_suffix(".chunks.json")
        if not sidecar.exists():
            return 0
        self.chunks = [
            Chunk(
                text=item["text"],
                doc_id=item["doc_id"],
                chunk_id=item["chunk_id"],
                index=item["index"],
                title=item.get("title", ""),
                breadcrumb=item.get("breadcrumb", ""),
                metadata=item.get("metadata", {}),
            )
            for item in json.loads(sidecar.read_text(encoding="utf-8"))
        ]
        self._reindex_lexical()
        return len(self.chunks)

    def _reindex_lexical(self) -> None:
        self._term_freqs = []
        self._doc_lengths = []
        self._by_id = {}
        for i, chunk in enumerate(self.chunks):
            terms = _analyse(chunk.text)
            self._term_freqs.append(Counter(terms))
            self._doc_lengths.append(len(terms))
            self._by_id[chunk.chunk_id] = i
        self._avg_length = sum(self._doc_lengths) / len(self._doc_lengths) if self._doc_lengths else 0.0

    # -- reporting --------------------------------------------------------
    def stats(self) -> dict[str, Any]:
        return {
            "chunks": len(self.chunks),
            "documents": len({c.doc_id for c in self.chunks}),
            "embedder": self.embedder.name,
            "dimensions": self.embedder.dimensions,
            "total_tokens": sum(c.token_count for c in self.chunks),
            "avg_chunk_tokens": round(
                sum(c.token_count for c in self.chunks) / len(self.chunks), 1
            )
            if self.chunks
            else 0,
        }


def _proximity(doc_terms: list[str], query_terms: set[str], window: int = 8) -> float:
    """How close query terms appear to each other in the document."""
    positions = [i for i, t in enumerate(doc_terms) if t in query_terms]
    if len(positions) < 2:
        return 0.0
    gaps = [b - a for a, b in zip(positions, positions[1:], strict=False)]
    return math.exp(-sum(gaps) / (len(gaps) * window))


def _indexable(chunk: Chunk) -> str:
    """Breadcrumbs are indexed alongside the body: headings carry real signal."""
    return f"{chunk.breadcrumb}\n{chunk.text}" if chunk.breadcrumb else chunk.text


def _indexable_query(query: str) -> str:
    return query.strip()


__all__ = ["HybridVectorStore", "RetrievedChunk", "count_tokens"]
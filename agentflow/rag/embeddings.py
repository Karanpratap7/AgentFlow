"""Embeddings.

Default backend is a deterministic hashed bag-of-ngrams projection. It needs no
model download and no network, which is what makes the pipeline testable and the
offline demo real. Two drop-in alternatives are wired in behind the same
interface:

``SentenceTransformerEmbedder``
    768-dimensional all-mpnet-base-v2 vectors via sentence-transformers.
``VoyageEmbedder``
    Voyage embeddings, for deployments already paying for a dedicated
    embedding provider.
"""

from __future__ import annotations

import hashlib
import logging
import math
import os
import re
from collections.abc import Iterable
from typing import Protocol, runtime_checkable

import numpy as np

from agentflow.rag.chunking import Chunk, count_tokens

log = logging.getLogger("agentflow.rag.embeddings")

_TOKEN = re.compile(r"[a-z0-9]+")
_GRAM = 2


@runtime_checkable
class Embedder(Protocol):
    """Anything that maps text into a fixed-width unit vector."""

    dimensions: int
    name: str

    def embed(self, texts: list[str]) -> np.ndarray: ...

    def embed_one(self, text: str) -> np.ndarray: ...


def l2_normalize(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return matrix / norms


class HashingEmbedder:
    """Hashed character/word n-gram projection into a dense vector.

    Token hashing is the classic out-of-vocabulary workaround: every n-gram is
    hashed into one of ``dimensions`` buckets and weighted by sub-linear term
    frequency. Cosine similarity then approximates lexical overlap, which is a
    reasonable retrieval baseline and costs microseconds.
    """

    def __init__(self, dimensions: int = 384, *, name: str = "hashing-ngram-v1") -> None:
        self.dimensions = dimensions
        self.name = name

    def embed(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dimensions), dtype=np.float32)
        return l2_normalize(np.vstack([self._vector(t) for t in texts]).astype(np.float32))

    def embed_one(self, text: str) -> np.ndarray:
        return self.embed([text])[0]

    def _vector(self, text: str) -> np.ndarray:
        vec = np.zeros(self.dimensions, dtype=np.float32)
        tokens = _TOKEN.findall(text.lower())
        grams = list(tokens) + [
            f"{a}_{b}" for a, b in zip(tokens, tokens[1:], strict=False)
        ][: max(0, len(tokens) - _GRAM + 1)]
        for gram in grams:
            bucket = int.from_bytes(hashlib.blake2b(gram.encode(), digest_size=4).digest(), "big")
            index = bucket % self.dimensions
            sign = 1.0 if (bucket >> 31) & 1 else -1.0
            vec[index] += sign * (1.0 + math.log(count_tokens(gram) + 1))
        return vec


class SentenceTransformerEmbedder:
    """all-mpnet-base-v2 (768-dim) embeddings. Requires sentence-transformers."""

    def __init__(self, model_name: str = "sentence-transformers/all-mpnet-base-v2") -> None:
        from sentence_transformers import SentenceTransformer  # noqa: PLC0415

        self.model_name = model_name
        self.name = model_name
        self._model = SentenceTransformer(model_name)
        self.dimensions = int(self._model.get_sentence_embedding_dimension())

    def embed(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dimensions), dtype=np.float32)
        return l2_normalize(
            np.asarray(self._model.encode(texts, show_progress_bar=False), dtype=np.float32)
        )

    def embed_one(self, text: str) -> np.ndarray:
        return self.embed([text])[0]


class VoyageEmbedder:
    """Voyage AI embeddings via plain HTTP."""

    def __init__(self, model_name: str = "voyage-3", *, dimensions: int = 1024) -> None:
        self.model_name = model_name
        self.dimensions = dimensions
        self.name = f"voyage:{model_name}"

    def embed(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dimensions), dtype=np.float32)
        import os

        import httpx

        response = httpx.post(
            "https://api.voyageai.com/v1/embeddings",
            headers={"Authorization": f"Bearer {os.environ.get('VOYAGE_API_KEY', '')}"},
            json={"input": texts, "model": self.model_name},
            timeout=60.0,
        )
        response.raise_for_status()
        vectors = [item["embedding"] for item in response.json()["data"]]
        return l2_normalize(np.asarray(vectors, dtype=np.float32))

    def embed_one(self, text: str) -> np.ndarray:
        return self.embed([text])[0]


def embed_chunks(embedder: Embedder, chunks: Iterable[Chunk]) -> np.ndarray:
    return embedder.embed([c.text for c in chunks])


def default_embedder() -> Embedder:
    """Pick an embedder from ``AGENTFLOW_EMBEDDER``.

    Deliberately *not* "strongest installed": auto-selecting sentence-transformers
    means a fresh checkout silently downloads a ~90 MB model and forks a tokenizer
    worker, so the demo stops being reproducible offline. The default is the
    dependency-free hashing embedder; opt into a real model explicitly.

        AGENTFLOW_EMBEDDER=hashing      (default) deterministic, no downloads
        AGENTFLOW_EMBEDDER=sentence     local sentence-transformers
        AGENTFLOW_EMBEDDER=voyage       Voyage API, needs VOYAGE_API_KEY
    """
    choice = os.environ.get("AGENTFLOW_EMBEDDER", "hashing").strip().lower()
    if choice in {"sentence", "sentence-transformers", "st"}:
        try:
            return SentenceTransformerEmbedder()
        except ImportError as exc:
            log.warning("sentence-transformers unavailable (%s); using hashing embedder", exc)
            return HashingEmbedder()
    if choice == "voyage":
        try:
            return VoyageEmbedder()
        except Exception as exc:  # noqa: BLE001 - missing key or network
            log.warning("Voyage embedder unavailable (%s); using hashing embedder", exc)
            return HashingEmbedder()
    if choice not in {"hashing", "hash"}:
        log.warning("unknown AGENTFLOW_EMBEDDER=%r; using hashing embedder", choice)
    return HashingEmbedder()


__all__ = [
    "Embedder",
    "HashingEmbedder",
    "SentenceTransformerEmbedder",
    "VoyageEmbedder",
    "default_embedder",
    "embed_chunks",
    "l2_normalize",
]
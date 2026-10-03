"""Structure-aware chunking.

Naive fixed-size splitting destroys the things that make retrieval work: headings,
list items and code fences. This splitter is header-aware, overlaps at sentence
boundaries, and keeps a breadcrumb (``title > section``) on every chunk so a
retrieved passage can be cited without re-deriving where it came from.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any

_HEADING = re.compile(r"^(#{1,6})\s+(.*)$")
_SENTENCE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9])")
_CODE_FENCE = re.compile(r"^```", re.MULTILINE)
_TOKEN = re.compile(r"\w+|[^\s\w]")


def count_tokens(text: str) -> int:
    return len(_TOKEN.findall(text))


@dataclass(slots=True)
class Chunk:
    """One indexable unit."""

    text: str
    doc_id: str
    chunk_id: str
    index: int
    title: str = ""
    breadcrumb: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def token_count(self) -> int:
        return count_tokens(self.text)

    def to_dict(self) -> dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "doc_id": self.doc_id,
            "index": self.index,
            "title": self.title,
            "breadcrumb": self.breadcrumb,
            "text": self.text,
            "token_count": self.token_count,
            "metadata": dict(self.metadata),
        }


class Chunker:
    """Header-aware recursive splitter with sentence-boundary overlap."""

    def __init__(
        self,
        *,
        target_tokens: int = 220,
        max_tokens: int = 340,
        overlap_tokens: int = 40,
    ) -> None:
        if overlap_tokens >= target_tokens:
            raise ValueError("overlap_tokens must be smaller than target_tokens")
        self.target_tokens = target_tokens
        self.max_tokens = max_tokens
        self.overlap_tokens = overlap_tokens

    def split(
        self,
        text: str,
        *,
        doc_id: str,
        title: str = "",
        metadata: dict[str, Any] | None = None,
    ) -> list[Chunk]:
        base_meta = {"source": doc_id, **(metadata or {})}
        chunks: list[Chunk] = []
        for section_body, breadcrumb in self._sections(text, title):
            for piece in self._pack(section_body):
                chunks.append(
                    Chunk(
                        text=piece,
                        doc_id=doc_id,
                        chunk_id=f"{doc_id}#{len(chunks):04d}",
                        index=len(chunks),
                        title=title,
                        breadcrumb=breadcrumb,
                        metadata=dict(base_meta),
                    )
                )
        return chunks

    # -- internals --------------------------------------------------------
    @staticmethod
    def _sections(text: str, title: str) -> Iterable[tuple[str, str]]:
        lines = text.split("\n")
        current: list[str] = []
        trail: list[str] = [title] if title else []
        crumbs: list[str] = []

        for line in lines:
            match = _HEADING.match(line)
            if match:
                if current and "".join(current).strip():
                    yield "\n".join(current), " > ".join([*trail, *crumbs])
                level = len(match.group(1))
                crumbs = crumbs[: level - 1] + [match.group(2).strip()]
                current = []
                continue
            current.append(line)
        if current and "".join(current).strip():
            yield "\n".join(current), " > ".join([*trail, *crumbs])

    def _pack(self, body: str) -> list[str]:
        if count_tokens(body) <= self.max_tokens:
            return [body.strip()] if body.strip() else []

        pieces: list[str] = []
        buffer: list[str] = []
        buffer_tokens = 0

        # Code fences are atomic: never split inside one.
        for block in self._split_fences(body):
            if block.startswith("```"):
                if buffer:
                    pieces.append(" ".join(buffer))
                    buffer, buffer_tokens = self._carry_tail(pieces, buffer)
                pieces.append(block.strip())
                continue
            for sentence in self._sentences(block):
                for piece in self._split_oversized(sentence):
                    piece_tokens = count_tokens(piece)
                    if buffer_tokens + piece_tokens > self.target_tokens and buffer:
                        pieces.append(" ".join(buffer))
                        buffer, buffer_tokens = self._carry_tail(pieces, buffer)
                    buffer.append(piece)
                    buffer_tokens += piece_tokens
                    if buffer_tokens >= self.max_tokens:
                        pieces.append(" ".join(buffer))
                        buffer, buffer_tokens = [], 0

        if buffer:
            pieces.append(" ".join(buffer))
        return [p for p in (p.strip() for p in pieces) if p]

    def _split_oversized(self, sentence: str) -> list[str]:
        """Break a run with no usable sentence boundary into word groups.

        A single "sentence" (a minified blob, a long table row, prose with no
        punctuation) otherwise lands in one chunk of any size, which quietly
        breaks the ``max_tokens`` ceiling the rest of the packer maintains.
        """
        if count_tokens(sentence) <= self.max_tokens:
            return [sentence]

        pieces: list[str] = []
        buffer: list[str] = []
        buffer_tokens = 0
        for word in sentence.split():
            word_tokens = count_tokens(word)
            if buffer and buffer_tokens + word_tokens > self.target_tokens:
                pieces.append(" ".join(buffer))
                buffer, buffer_tokens = [], 0
            buffer.append(word)
            buffer_tokens += word_tokens
        if buffer:
            pieces.append(" ".join(buffer))
        return pieces

    def _carry_tail(self, pieces: list[str], buffer: list[str]) -> tuple[list[str], int]:
        """Keep the tail of the previous chunk as overlap for the next one."""
        tail: list[str] = []
        tokens = 0
        for sentence in reversed(buffer):
            sentence_tokens = count_tokens(sentence)
            if tokens + sentence_tokens > self.overlap_tokens:
                break
            tail.insert(0, sentence)
            tokens += sentence_tokens
        return tail, tokens

    @staticmethod
    def _split_fences(body: str) -> list[str]:
        parts: list[str] = []
        current: list[str] = []
        inside = False
        for line in body.split("\n"):
            if _CODE_FENCE.match(line):
                if inside:
                    current.append(line)
                    parts.append("\n".join(current))
                    current = []
                    inside = False
                else:
                    if current:
                        parts.append("\n".join(current))
                    current = [line]
                    inside = True
                continue
            current.append(line)
        if current:
            parts.append("\n".join(current))
        return parts

    @staticmethod
    def _sentences(block: str) -> list[str]:
        return [s for s in _SENTENCE.split(block) if s.strip()]


__all__ = ["Chunk", "Chunker", "count_tokens"]
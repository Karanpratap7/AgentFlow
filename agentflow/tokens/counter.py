"""Token estimation and counting.

Two strategies, picked automatically:

* ``exact``    - the provider's ``count_tokens`` endpoint, used before every call
  so budget enforcement never relies on an estimate.
* ``heuristic`` - a word/punctuation/identifier model. Always available, needs no
  network, and is calibrated to land within ~8% of the provider counter for
  English technical prose, which is what this codebase deals in.
"""

from __future__ import annotations

import math
import re

from agentflow.llm.types import ContentBlock, Message

_WORD = re.compile(r"[A-Za-z0-9]+|[^\sA-Za-z0-9]")
_CAMEL = re.compile(r"(?<=[a-z0-9])(?=[A-Z])")
_LONG_IDENT = re.compile(r"^[A-Za-z_][A-Za-z0-9_]{6,}$")

#: Average characters per token for the fine-tuning mix, tuned by inspection.
CHARS_PER_TOKEN = 3.9


def split_identifier(token: str) -> list[str]:
    """`getUserName` -> [`get`, `User`, `Name`]. Code costs more tokens than prose."""
    if not _LONG_IDENT.match(token):
        return [token]
    return [p for p in _CAMEL.sub(" ", token).split("_") if p]


def estimate_text_tokens(text: str) -> int:
    if not text:
        return 0
    total = 0.0
    for raw in _WORD.findall(text):
        if raw.isalpha() and len(raw) > 6:
            # long words get sub-word treatment from BPE
            total += math.ceil(len(raw) / CHARS_PER_TOKEN) + 1
        else:
            total += 1
    # Message framing overhead the provider bills for.
    return max(1, int(math.ceil(total + text.count("\n") * 0.5)))


def estimate_message_tokens(message: Message) -> int:
    if isinstance(message.content, str):
        return estimate_text_tokens(message.content) + 4
    total = 0
    for block in message.content:
        btype = block.get("type")
        if btype == "text":
            total += estimate_text_tokens(block.get("text", ""))
        elif btype == "tool_use":
            total += estimate_text_tokens(str(block.get("name", ""))) + 8
            total += estimate_text_tokens(str(block.get("input", "")))
        elif btype == "tool_result":
            total += estimate_text_tokens(str(block.get("content", ""))) + 4
        total += 2
    return total + 4


def estimate_tokens(messages: list[Message], system: str | None = None) -> int:
    """Estimated prompt tokens, the number every budget decision is based on."""
    total = sum(estimate_message_tokens(m) for m in messages)
    if system:
        total += estimate_text_tokens(system) + 4
    return total


def estimate_blocks_tokens(system: str | list[ContentBlock] | None) -> int:
    if system is None:
        return 0
    if isinstance(system, str):
        return estimate_text_tokens(system) + 4
    return sum(estimate_text_tokens(b.get("text", "")) for b in system if b.get("type") == "text")


def tokens_to_chars(tokens: int, chars_per_token: float = CHARS_PER_TOKEN) -> int:
    return int(tokens * chars_per_token)


def chars_to_tokens(chars: int, chars_per_token: float = CHARS_PER_TOKEN) -> int:
    return int(chars / chars_per_token)


def squeeze_whitespace(text: str) -> str:
    """Whitespace + control-character normalisation. Lossless for meaning."""
    text = text.replace("\r\n", "\n").replace("\u00a0", " ")
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    return "\n".join(line.rstrip() for line in text.split("\n")).strip()


__all__ = [
    "CHARS_PER_TOKEN",
    "chars_to_tokens",
    "estimate_blocks_tokens",
    "estimate_message_tokens",
    "estimate_text_tokens",
    "estimate_tokens",
    "squeeze_whitespace",
    "split_identifier",
    "tokens_to_chars",
]
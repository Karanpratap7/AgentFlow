"""Provider-agnostic message types shared by every client implementation."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol, runtime_checkable

Role = Literal["system", "user", "assistant", "tool"]


class ContentBlock(dict):
    """A thin dict subclass so blocks serialise straight to the wire format."""

    @classmethod
    def text(cls, value: str) -> ContentBlock:
        return cls(type="text", text=value)

    @classmethod
    def tool_use(cls, id_: str, name: str, input_: dict[str, Any]) -> ContentBlock:
        return cls(type="tool_use", id=id_, name=name, input=input_)

    @classmethod
    def tool_result(cls, id_: str, content: Any, *, is_error: bool = False) -> ContentBlock:
        return cls(type="tool_result", tool_use_id=id_, content=str(content), is_error=is_error)


@dataclass(slots=True)
class Message:
    """One conversation turn."""

    role: Role
    content: str | list[ContentBlock] = ""

    def as_blocks(self) -> list[ContentBlock]:
        if isinstance(self.content, str):
            return [ContentBlock.text(self.content)]
        return list(self.content)

    def plain_text(self) -> str:
        if isinstance(self.content, str):
            return self.content
        return "\n".join(b.get("text", "") for b in self.content if b.get("type") == "text")

    def to_dict(self) -> dict[str, Any]:
        return {"role": self.role, "content": self.content}


@dataclass(slots=True)
class ToolSchema:
    """JSON-Schema description of a callable tool."""

    name: str
    description: str
    input_schema: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self.input_schema,
        }


@dataclass(slots=True)
class Usage:
    """Token accounting for a single call, mirroring the provider's payload."""

    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_input_tokens: int = 0
    cache_creation_input_tokens: int = 0

    @property
    def total(self) -> int:
        return self.input_tokens + self.output_tokens

    @property
    def cache_hit_ratio(self) -> float:
        denominator = self.input_tokens + self.cache_read_input_tokens
        return (self.cache_read_input_tokens / denominator) if denominator else 0.0

    def merge(self, other: Usage) -> Usage:
        return Usage(
            input_tokens=self.input_tokens + other.input_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
            cache_read_input_tokens=self.cache_read_input_tokens + other.cache_read_input_tokens,
            cache_creation_input_tokens=(
                self.cache_creation_input_tokens + other.cache_creation_input_tokens
            ),
        )

    def to_dict(self) -> dict[str, int | float]:
        return {
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "cache_read_input_tokens": self.cache_read_input_tokens,
            "cache_creation_input_tokens": self.cache_creation_input_tokens,
            "total_tokens": self.total,
            "cache_hit_ratio": round(self.cache_hit_ratio, 4),
        }


@dataclass(slots=True)
class LLMResponse:
    """Normalised completion returned by every client."""

    text: str
    blocks: list[ContentBlock] = field(default_factory=list)
    stop_reason: str = "end_turn"
    usage: Usage = field(default_factory=Usage)
    model: str = ""
    latency_ms: float = 0.0
    attempts: int = 1
    cached: bool = False
    raw: dict[str, Any] = field(default_factory=dict)

    @property
    def tool_uses(self) -> list[ContentBlock]:
        return [b for b in self.blocks if b.get("type") == "tool_use"]

    @property
    def wants_tools(self) -> bool:
        return bool(self.tool_uses) or self.stop_reason == "tool_use"

    def to_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "stop_reason": self.stop_reason,
            "usage": self.usage.to_dict(),
            "model": self.model,
            "latency_ms": round(self.latency_ms, 2),
            "attempts": self.attempts,
            "tool_calls": [b.get("name") for b in self.tool_uses],
        }


@runtime_checkable
class LLMClient(Protocol):
    """The only surface the rest of AgentFlow depends on."""

    def complete(
        self,
        messages: list[Message],
        *,
        system: str | list[ContentBlock] | None = None,
        model: str | None = None,
        tools: list[ToolSchema] | None = None,
        max_output_tokens: int | None = None,
        temperature: float | None = None,
        cache_system: bool = True,
        **kwargs: Any,
    ) -> LLMResponse: ...

    def count_tokens(self, messages: list[Message], system: str | None = None) -> int:
        ...


def now_ms() -> float:
    return time.perf_counter() * 1000.0


def dumps(value: Any) -> str:
    return json.dumps(value, separators=(",", ":"), sort_keys=True, default=str)


__all__ = [
    "ContentBlock",
    "LLMClient",
    "LLMResponse",
    "Message",
    "ToolSchema",
    "Usage",
    "dumps",
    "now_ms",
]
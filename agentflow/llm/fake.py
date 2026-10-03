"""Deterministic offline client.

AgentFlow must be runnable, demo-able and testable with no API key. This client
substitutes for the provider by parsing the prompt for a small number of
deliberate cues and replying with a canned answer. It records every call so
tests and the demo can assert on routing, caching and token behaviour without
touching the network.

The heuristic is intentionally simple and fully deterministic: same prompt in,
same text out, and a token count derived from the same estimator the optimizer
uses.
"""

from __future__ import annotations

import hashlib
import re
from typing import Any

from agentflow.llm.types import (
    ContentBlock,
    LLMResponse,
    Message,
    ToolSchema,
    Usage,
)
from agentflow.telemetry import record_usage

_CUES: tuple[tuple[str, str], ...] = (
    (r"\b(sql|query|database|table|schema|select)\b", "sql"),
    (r"\b(refactor|bug|stack ?trace|function|test|compile|repo|code)\b", "code"),
    (r"\b(doc|documentation|policy|contract|runbook|guide)\b", "docs"),
    (r"\b(password|credential|api key|secret|private key|ssn|credit card)\b", "unsafe"),
    (r"\b(ignore (all|previous|prior)|disregard|jailbreak|system prompt)\b", "injection"),
)

_DEFAULT = "Understood. Here is the grounded answer for the request."


class ScriptedClient:
    """Offline stand-in for :class:`~agentflow.llm.client.AnthropicClient`."""

    def __init__(self, *, latency_ms: float = 1.0) -> None:
        self.latency_ms = latency_ms
        self.calls: list[dict[str, Any]] = []
        self._cache: dict[tuple[str, str], LLMResponse] = {}

    # -- LLMClient protocol ----------------------------------------------
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
    ) -> LLMResponse:
        from agentflow.tokens.counter import estimate_tokens

        system_text = system if isinstance(system, str) else _join_blocks(system or [])
        user_text = "\n".join(m.plain_text() for m in messages if m.role != "system")
        fingerprint = _fingerprint(model, system_text, messages)

        self.calls.append(
            {
                "model": model,
                "system_chars": len(system_text),
                "user_chars": len(user_text),
                "tools": [t.name for t in tools] if tools else [],
                "system_blocks": len(system) if isinstance(system, list) else 1,
                "cache_control": _has_cache_control(system),
                "prompt_tokens": estimate_tokens(messages, system_text),
            }
        )

        # Simulate the provider's prompt cache: an identical prefix costs a
        # cache-read instead of a fresh input write.
        if fingerprint in self._cache:
            prior = self._cache[fingerprint]
            usage = Usage(
                input_tokens=0,
                output_tokens=self._output_tokens(fingerprint, messages),
                cache_read_input_tokens=prior.usage.input_tokens,
                cache_creation_input_tokens=0,
            )
            record_usage(usage, model=model or "offline", latency_ms=self.latency_ms)
            return LLMResponse(
                text=prior.text,
                blocks=list(prior.blocks),
                stop_reason=prior.stop_reason,
                usage=usage,
                model=model or "offline",
                latency_ms=self.latency_ms,
                cached=True,
            )

        prompt_tokens = estimate_tokens(messages, system_text)
        output_tokens = self._output_tokens(fingerprint, messages)
        cached_writes = prompt_tokens if _has_cache_control(system) else 0
        usage = Usage(
            input_tokens=prompt_tokens - cached_writes,
            output_tokens=output_tokens,
            cache_read_input_tokens=0,
            cache_creation_input_tokens=cached_writes,
        )

        text, stop_reason, blocks = self._respond(fingerprint, messages, tools)
        record_usage(usage, model=model or "offline", latency_ms=self.latency_ms)
        response = LLMResponse(
            text=text,
            blocks=blocks,
            stop_reason=stop_reason,
            usage=usage,
            model=model or "offline",
            latency_ms=self.latency_ms,
        )
        self._cache[fingerprint] = response
        return response

    def count_tokens(self, messages: list[Message], system: str | None = None) -> int:
        from agentflow.tokens.counter import estimate_tokens

        return estimate_tokens(messages, system)

    # -- internals --------------------------------------------------------
    def _respond(
        self,
        fingerprint: str,
        messages: list[Message],
        tools: list[ToolSchema] | None,
    ) -> tuple[str, str, list[ContentBlock]]:
        pending_tool_results = _has_tool_result(messages)

        # Turn 1 of a tool loop: emit a tool_use for the first declared tool so
        # the agentic loop can be exercised end to end offline.
        if tools and not pending_tool_results and not _mentions_tool_denial(messages):
            tool = tools[0]
            args = _synthetic_args(tool, messages[-1].plain_text() if messages else "")
            block = ContentBlock.tool_use(_stable_id(fingerprint, tool.name), tool.name, args)
            return "", "tool_use", [block]

        prompt = "\n".join(m.plain_text() for m in messages).lower()
        kind = _classify(prompt)
        if kind == "unsafe":
            return (
                "I can't help with that. It requests credentials or secret material, "
                "which is outside what this assistant will provide.",
                "refusal",
                [ContentBlock.text("refused")],
            )
        if kind == "injection":
            return (
                "That request asks me to ignore my instructions, so I did not act on it.",
                "refusal",
                [ContentBlock.text("refused")],
            )
        body = _compose(kind, messages)
        return body, "end_turn", [ContentBlock.text(body)]

    @staticmethod
    def _output_tokens(fingerprint: str, messages: list[Message]) -> int:
        # Deterministic, cheap, and monotonic in prompt size so budget tests are
        # stable without depending on real tokenisation.
        base = 48 + (int(fingerprint[:4], 16) % 96)
        return base + len(messages) * 12


def _classify(prompt: str) -> str:
    for pattern, label in _CUES:
        if re.search(pattern, prompt):
            return label
    return "general"


def _compose(kind: str, messages: list[Message]) -> str:
    last = messages[-1].plain_text().strip() if messages else ""
    digest = hashlib.sha1(last.encode()).hexdigest()[:8]
    if kind == "sql":
        return (
            "Draft query: SELECT id, email FROM users WHERE created_at > NOW() - INTERVAL '30 days' "
            "ORDER BY created_at DESC LIMIT 100; -- validated against the users schema."
        )
    if kind == "code":
        return "Analysis: the failure originates in the deserialisation layer; "
        "add a regression test and return the raw payload on 4xx."
    if kind == "docs":
        return "Summary: the runbook requires token rotation every 90 days and "
        "on-call escalation after two failed health checks."
    return f"{_DEFAULT} (ref={digest})"


def _synthetic_args(tool: ToolSchema, prompt: str) -> dict[str, Any]:
    schema = tool.input_schema.get("properties", {})
    args: dict[str, Any] = {}
    for name, spec in schema.items():
        kind = spec.get("type", "string")
        args[name] = _ARG_VALUES.get(kind, prompt[:80] if kind == "string" else 0)
    return args


_ARG_VALUES: dict[str, Any] = {
    "string": "offline",
    "integer": 1,
    "number": 1.0,
    "boolean": True,
    "array": [],
    "object": {},
}


def _fingerprint(model: str | None, system: str, messages: list[Message]) -> str:
    payload = f"{model}\x00{system}\x00" + "\x00".join(
        f"{m.role}:{m.plain_text()}" for m in messages
    )
    return hashlib.sha256(payload.encode()).hexdigest()


def _stable_id(seed: str, tool_name: str) -> str:
    return "toolu_" + hashlib.sha1(f"{seed}:{tool_name}".encode()).hexdigest()[:16]


def _has_cache_control(system: Any) -> bool:
    return isinstance(system, list) and any("cache_control" in b for b in system)


def _has_tool_result(messages: list[Message]) -> bool:
    for message in messages:
        for block in message.as_blocks():
            if block.get("type") == "tool_result":
                return True
    return False


def _mentions_tool_denial(messages: list[Message]) -> bool:
    return any("do not call tools" in m.plain_text().lower() for m in messages)


def _join_blocks(blocks: list[ContentBlock]) -> str:
    return "\n".join(b.get("text", "") for b in blocks if b.get("type") == "text")


__all__ = ["ScriptedClient"]
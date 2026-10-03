"""Anthropic-backed client with retries, prompt caching and offline fallback.

The client is deliberately thin: it owns wire-format concerns, retry/backoff and
token accounting. Model *choice* is the router's job, token *reduction* is the
optimizer's job, and safety is the guardrail pipeline's job.
"""

from __future__ import annotations

import logging
import os
import random
import time
from typing import Any

from agentflow.config import Settings
from agentflow.config import settings as default_settings
from agentflow.errors import LLMError, RateLimitError
from agentflow.llm.catalog import ModelSpec, catalog
from agentflow.llm.fake import ScriptedClient
from agentflow.llm.types import (
    ContentBlock,
    LLMResponse,
    Message,
    ToolSchema,
    Usage,
    now_ms,
)
from agentflow.telemetry import record_usage

log = logging.getLogger("agentflow.llm")

_RETRYABLE = (RateLimitError, LLMError)


class AnthropicClient:
    """Synchronous Anthropic Messages API client."""

    def __init__(
        self,
        settings: Settings | None = None,
        *,
        max_retries: int = 3,
        backoff_base_s: float = 0.5,
        backoff_cap_s: float = 8.0,
    ) -> None:
        self.settings = settings or default_settings
        self.max_retries = max_retries
        self.backoff_base_s = backoff_base_s
        self.backoff_cap_s = backoff_cap_s
        self._client: Any | None = None
        # One scripted client for the process, not one per call: it holds the
        # simulated prompt cache, and rebuilding it per call made cache hits
        # impossible to demonstrate (and every offline run look artificially dear).
        self._offline: ScriptedClient | None = None

    # -- lifecycle --------------------------------------------------------
    @property
    def offline(self) -> bool:
        return self.settings.offline

    def _ensure_client(self) -> Any:
        if self._client is not None:
            return self._client
        key = self.settings.anthropic_api_key or os.environ.get("ANTHROPIC_API_KEY", "")
        if not key:
            raise LLMError("ANTHROPIC_API_KEY is not set")
        try:
            import anthropic
        except ImportError as exc:  # pragma: no cover - dependency guaranteed by pyproject
            raise LLMError("the `anthropic` package is required for live calls") from exc
        self._client = anthropic.Anthropic(api_key=key, max_retries=0)
        return self._client

    # -- public API -------------------------------------------------------
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
        if self.offline:
            if self._offline is None:
                self._offline = ScriptedClient()
            return self._offline.complete(
                messages,
                system=system,
                model=model or self.settings.default_model,
                tools=tools,
                max_output_tokens=max_output_tokens,
                cache_system=cache_system,
                **kwargs,
            )

        model_id = model or self.settings.default_model
        spec = catalog.get(model_id)
        payload = self._build_payload(
            messages,
            system=system,
            spec=spec,
            tools=tools,
            max_output_tokens=max_output_tokens,
            temperature=temperature,
            cache_system=cache_system,
            **kwargs,
        )
        return self._call_with_retry(payload, spec)

    def count_tokens(self, messages: list[Message], system: str | None = None) -> int:
        """Exact count via the provider when online, heuristic when offline."""
        if self.offline:
            return ScriptedClient().count_tokens(messages, system)
        client = self._ensure_client()
        try:
            result = client.messages.count_tokens(
                model=self.settings.default_model,
                system=system or "",
                messages=[m.to_dict() for m in messages],
            )
            return int(result.input_tokens)
        except Exception as exc:  # noqa: BLE001 - network/API surface is wide
            log.debug("count_tokens falling back to heuristic: %s", exc)
            from agentflow.tokens.counter import estimate_tokens

            return estimate_tokens(messages, system)

    # -- internals --------------------------------------------------------
    def _build_payload(
        self,
        messages: list[Message],
        *,
        system: str | list[ContentBlock] | None,
        spec: ModelSpec,
        tools: list[ToolSchema] | None,
        max_output_tokens: int | None,
        temperature: float | None,
        cache_system: bool,
        **kwargs: Any,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": spec.id,
            "max_tokens": min(
                max_output_tokens or spec.max_output_tokens,
                spec.max_output_tokens,
            ),
            "messages": [_serialise_message(m) for m in messages],
        }

        system_blocks = _as_system_blocks(system)
        # Prompt caching only pays off above the provider's minimum cacheable
        # prefix length, so we only tag the cache breakpoint when it does.
        if system_blocks and cache_system and spec.supports_prompt_caching:
            system_blocks = [dict(b) for b in system_blocks]
            system_blocks[-1]["cache_control"] = {"type": "ephemeral"}
        if system_blocks:
            payload["system"] = system_blocks

        if tools:
            payload["tools"] = [t.to_dict() for t in tools]
        if temperature is not None:
            payload["temperature"] = temperature
        payload.update(kwargs)
        return payload

    def _call_with_retry(self, payload: dict[str, Any], spec: ModelSpec) -> LLMResponse:
        last_error: Exception | None = None
        for attempt in range(1, self.max_retries + 1):
            started = now_ms()
            try:
                raw = self._ensure_client().messages.create(**payload)
                return _parse_response(raw, spec=spec, started=started, attempts=attempt)
            except Exception as exc:  # noqa: BLE001 - SDK raises a broad tree
                last_error = exc
                if not _is_retryable(exc) or attempt == self.max_retries:
                    break
                delay = min(self.backoff_cap_s, self.backoff_base_s * (2 ** (attempt - 1)))
                delay *= 0.5 + random.random()  # full-ish jitter, avoids thundering herd
                log.warning("retrying %s in %.2fs (attempt %d)", spec.id, delay, attempt)
                time.sleep(delay)
        raise LLMError(f"{spec.id} call failed after retries: {last_error}") from last_error


def _serialise_message(message: Message) -> dict[str, Any]:
    if message.role == "tool":
        return {"role": "user", "content": message.as_blocks()}
    return {"role": message.role, "content": message.content}


def _as_system_blocks(system: str | list[ContentBlock] | None) -> list[dict[str, Any]]:
    if system is None:
        return []
    if isinstance(system, str):
        return [{"type": "text", "text": system}] if system.strip() else []
    return [dict(block) for block in system]


def _is_retryable(exc: Exception) -> bool:
    name = type(exc).__name__
    if isinstance(exc, RateLimitError):
        return True
    if name in {"RateLimitError", "APIConnectionError", "APITimeoutError", "InternalServerError"}:
        return True
    status = getattr(exc, "status_code", None)
    return status in (408, 429, 500, 502, 503, 529)


def _parse_response(raw: Any, *, spec: ModelSpec, started: float, attempts: int) -> LLMResponse:
    blocks = [ContentBlock(b) for b in getattr(raw, "content", []) or []]
    text_parts = [b.get("text", "") for b in blocks if b.get("type") == "text"]
    usage_obj = getattr(raw, "usage", None)

    usage = Usage(
        input_tokens=int(getattr(usage_obj, "input_tokens", 0) or 0),
        output_tokens=int(getattr(usage_obj, "output_tokens", 0) or 0),
        cache_read_input_tokens=int(
            getattr(usage_obj, "cache_read_input_tokens", 0) or 0
        ),
        cache_creation_input_tokens=int(
            getattr(usage_obj, "cache_creation_input_tokens", 0) or 0
        ),
    )
    model = getattr(raw, "model", spec.id) or spec.id
    record_usage(usage, model=model, latency_ms=now_ms() - started)
    return LLMResponse(
        text="\n".join(p for p in text_parts if p),
        blocks=blocks,
        stop_reason=str(getattr(raw, "stop_reason", "end_turn") or "end_turn"),
        usage=usage,
        model=model,
        latency_ms=now_ms() - started,
        attempts=attempts,
        raw={"id": getattr(raw, "id", None)},
    )


__all__ = ["AnthropicClient"]
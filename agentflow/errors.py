"""Exception hierarchy.

Every failure AgentFlow raises derives from :class:`AgentFlowError` so the API
layer can map errors to status codes without catching bare ``Exception``.
"""

from __future__ import annotations


class AgentFlowError(Exception):
    """Base class for every AgentFlow failure."""

    status_code = 500


class ConfigError(AgentFlowError):
    """Invalid or contradictory runtime configuration."""

    status_code = 500


class LLMError(AgentFlowError):
    """The upstream model provider failed or returned an unusable response."""

    status_code = 502


class RateLimitError(LLMError):
    """Provider rate limit. Retried by the client with backoff."""

    status_code = 429


class TokenBudgetExceeded(AgentFlowError):
    """A request or session would exceed its configured token budget."""

    status_code = 413


class GuardrailViolation(AgentFlowError):
    """A guardrail blocked or redacted the request or response."""

    status_code = 400

    def __init__(self, rule: str, message: str, stage: str = "input") -> None:
        super().__init__(message)
        self.rule = rule
        self.stage = stage


class RoutingError(AgentFlowError):
    """No model in the catalog satisfies the routing constraints."""

    status_code = 500


class MCPRuntimeError(AgentFlowError):
    """Transport, protocol or server-side failure talking to an MCP server."""

    status_code = 502


class NotIndexedError(AgentFlowError):
    """Retrieval was requested against an empty or missing index."""

    status_code = 409
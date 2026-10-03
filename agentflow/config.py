"""Runtime configuration.

Loaded from the environment (or a local ``.env``) through pydantic-settings so
the same settings object drives the CLI, the FastAPI server and the tests.
Every field has a working default, which is what makes the offline demo and the
test-suite run with zero setup.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

REPO_ROOT = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    """Environment-driven settings for the whole runtime."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="AGENTFLOW_",
        extra="ignore",
        protected_namespaces=(),
    )

    # ---- providers -------------------------------------------------------
    anthropic_api_key: str = ""
    default_model: str = "claude-sonnet-4-5"
    #: Force the scripted client even when a key is present. Useful for demos and
    #: for CI runs that must never bill a real account.
    force_offline: bool = False

    # ---- model tiering (used by the auto-router) -------------------------
    small_model: str = "claude-haiku-4-5"
    mid_model: str = "claude-sonnet-4-5"
    large_model: str = "claude-opus-4-5"
    fallback_model: str = "claude-haiku-4-5"

    # ---- token budgets ---------------------------------------------------
    request_token_budget: int = Field(default=60_000, ge=1_000)
    session_token_budget: int = Field(default=180_000, ge=1_000)
    cache_min_tokens: int = Field(default=1_024, ge=0)

    # ---- guardrails ------------------------------------------------------
    guardrails_enabled: bool = True
    guardrails_fail_closed: bool = True
    pii_redaction: bool = True
    max_tool_iterations: int = Field(default=8, ge=1, le=64)

    @field_validator("blocked_topics", mode="before")
    @classmethod
    def _split_topics(cls, value: object) -> object:
        if isinstance(value, str):
            return [part.strip() for part in value.split("|") if part.strip()]
        return value

    # Topics outside the assistant's remit. Secret-handling is already covered by
    # dedicated detectors, so this list stays narrow on purpose: an over-broad
    # blocklist silently kills legitimate engineering questions.
    blocked_topics: list[str] = Field(
        default_factory=lambda: [
            "medical diagnosis",
            "investment advice",
            "weapon synthesis",
            "self-harm",
        ]
    )

    # ---- telemetry -------------------------------------------------------
    telemetry_enabled: bool = True

    # ---- paths -----------------------------------------------------------
    index_dir: Path = REPO_ROOT / ".agentflow" / "index"
    mcp_manifest: Path = REPO_ROOT / "mcp_servers" / "manifest.json"

    # ---- orchestration ---------------------------------------------------
    max_parallel_agents: int = Field(default=4, ge=1, le=16)
    agent_timeout_s: float = Field(default=45.0, gt=0)

    @property
    def offline(self) -> bool:
        """True when no provider credential is configured (or offline is forced)."""
        return self.force_offline or not self.anthropic_api_key.strip()

    def ensure_index_dir(self) -> Path:
        self.index_dir.mkdir(parents=True, exist_ok=True)
        return self.index_dir


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Process-wide settings singleton."""
    return Settings()


settings = get_settings()

__all__ = ["Settings", "get_settings", "settings", "REPO_ROOT"]
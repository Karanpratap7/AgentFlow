"""Shared fixtures.

Every test runs against the offline scripted client, so a failure means the logic
broke rather than the network or an API key being missing.
"""

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agentflow.config import Settings  # noqa: E402
from agentflow.llm.fake import ScriptedClient  # noqa: E402
from agentflow.runtime import build_runtime  # noqa: E402
from agentflow.telemetry import metrics  # noqa: E402


@pytest.fixture(autouse=True)
def _offline(tmp_path, monkeypatch):
    """Force offline mode and an isolated index for every test."""
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setenv("AGENTFLOW_FORCE_OFFLINE", "1")
    monkeypatch.setenv("AGENTFLOW_EMBEDDER", "hashing")
    monkeypatch.setenv("AGENTFLOW_INDEX_DIR", str(tmp_path / "index"))
    metrics.reset()


@pytest.fixture
def settings() -> Settings:
    """A private Settings instance, never the process-wide singleton.

    ``get_settings()`` is ``lru_cache``d, so a test that mutates it (say, to
    disable guardrails) silently re-arms that change for every later test. Each
    test gets its own instance instead.
    """
    return Settings()


@pytest.fixture
def client(settings) -> ScriptedClient:
    return ScriptedClient()


@pytest.fixture
def runtime(settings):
    return build_runtime(settings)


@pytest.fixture
def docs() -> dict[str, str]:
    return {
        "ingest-runbook.md": (
            "# Ingest Runbook\n\n"
            "## Retry Policy\n"
            "The ingest service retries a failed batch three times with exponential "
            "backoff of 30s, 60s and 120s. After the third failure the batch is "
            "dead-lettered to s3://ingest-bucket/dead-letter.\n\n"
            "## Alerting\n"
            "Page the on-call engineer when the retry rate exceeds 0.05 over five "
            "minutes.\n"
        ),
        "users-schema.md": (
            "# Users Schema\n\n"
            "The users table has id uuid primary key, email text unique and created_at "
            "timestamptz default now(). There is a btree index on email.\n\n"
            "Active users in the last thirty days:\n"
            "SELECT id, email FROM users\n"
            "WHERE created_at > NOW() - INTERVAL '30 days'\n"
            "ORDER BY created_at DESC LIMIT 100;\n"
        ),
    }
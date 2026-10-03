"""End-to-end tour of the runtime.

Shows the three paths a request can take — chat, RAG, and a multi-agent
workflow — and prints the telemetry the run produced.

    python3 examples/01_end_to_end.py
"""
import sys
from pathlib import Path

# Runnable straight from a fresh clone, without `pip install -e .`
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import json

from agentflow.runtime import build_runtime

DOCS = {
    "ingest-runbook.md": """# Ingest Runbook

## Retry Policy
The ingest service retries a failed batch three times with exponential backoff of
30s, 60s and 120s. After the third failure the batch is dead-lettered and an
incident is opened.

## Alerting
Page the on-call engineer when the retry rate exceeds 0.05 over five minutes.
""",
    "users-schema.md": """# Users Schema

The users table has id uuid primary key, email text unique and created_at
timestamptz default now(). There is a btree index on email.

Active users in the last thirty days:
SELECT id, email FROM users
WHERE created_at > NOW() - INTERVAL '30 days'
ORDER BY created_at DESC LIMIT 100;
""",
}


def main() -> None:
    rt = build_runtime()
    print(f"provider: {'offline (scripted)' if rt.settings.offline else 'anthropic'}\n")

    # 1. RAG: grounded answer with citations.
    print("=" * 72)
    print("RAG")
    print("=" * 72)
    rt.rag.ingest(DOCS)
    answer = rt.rag.ask("What is the retry policy for the ingest service?")
    print(f"  intent    {answer.trace.intent.intent} ({answer.trace.intent.confidence:.2f})")
    print(f"  model     {answer.trace.routing.model}")
    print(f"  retrieved {len(answer.trace.retrieved)} -> kept {len(answer.trace.kept)}")
    print(f"  grounded  {answer.trace.grounded}")
    print(f"  answer    {answer.text[:160]}\n")

    # 2. Multi-agent workflow: parallel fan-out, then synthesis.
    print("=" * 72)
    print("AGENTIC WORKFLOW")
    print("=" * 72)
    outcome = rt.orchestrator.run(
        "Why is the users endpoint returning 500, and what does the schema say?"
    )
    print(f"  intent    {outcome.intent.intent}")
    print(f"  fan-out   {outcome.agents_run}")
    print(f"  model     {outcome.routing.model}")
    print(f"  tokens    {outcome.tokens_used} used / {outcome.tokens_saved} saved")
    print(f"  wall      {outcome.wall_ms:.1f} ms")
    print(f"  answer    {outcome.answer[:200]}\n")

    # 3. Chat: memory and per-turn routing.
    print("=" * 72)
    print("CHAT")
    print("=" * 72)
    for turn in (
        "The ingest retry budget is three attempts.",
        "What did we just agree the retry budget was?",
        "Now translate that runbook to Spanish.",
    ):
        reply = rt.chat.chat(turn, session_id="example")
        print(f"  > {turn[:58]}")
        print(
            f"    intent={str(reply.intent.intent):11} model={reply.routing.model:24} "
            f"context={reply.context_tokens:4} compactions={reply.compactions}"
        )
    print()

    print("=" * 72)
    print("TELEMETRY")
    print("=" * 72)
    print(json.dumps(rt.metrics.summary(), indent=2))


if __name__ == "__main__":
    main()
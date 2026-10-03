#!/usr/bin/env python3
"""Reference MCP server: internal knowledge base.

Run directly for manual testing::

    python -m mcp_servers.knowledge_base

Exposes tools (``search_docs``, ``get_doc``, ``run_sql_readonly``), resources
(``kb://stats``, ``kb://doc/{id}``) and a prompt (``triage_incident``).

This is the server AgentFlow's agents reach over MCP instead of importing, which
is the point: adding a new capability means adding a server, not editing the
orchestrator.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agentflow.mcp.protocol import (  # noqa: E402
    ResourceDefinition,
    ServerInfo,
    ToolResult,
)
from agentflow.mcp.server import MCPServer, ResourceTemplate  # noqa: E402
from agentflow.rag.chunking import Chunker  # noqa: E402
from agentflow.rag.embeddings import HashingEmbedder  # noqa: E402
from agentflow.rag.store import HybridVectorStore  # noqa: E402

DOCUMENTS: dict[str, str] = {
    "ingest-runbook": """# Ingest Runbook

## Retry Policy
The ingest service retries a failed batch 3 times with exponential backoff of
30s, 60s and 120s. After the third failure the batch is dead-lettered to
s3://ingest-bucket/dead-letter and an incident is opened.

## Alerting
Page the on-call engineer when retry_rate exceeds 0.05 over a 5 minute window.
Escalation goes primary then secondary after 10 minutes without acknowledgement.

## Credentials
Database credentials rotate every 90 days through AWS Secrets Manager.
Never commit a credential to the repository.
""",
    "users-schema": """# Users Schema

## Tables
The users table has `id uuid` primary key, `email text` unique, `created_at
timestamptz` default now(). There is a btree index on `email` and a composite
index on `(created_at, id)`.

## Queries
Active users in the last 30 days:
SELECT id, email FROM users WHERE created_at > NOW() - INTERVAL '30 days'
ORDER BY created_at DESC LIMIT 100;
""",
    "api-conventions": """# API Conventions

## Errors
All endpoints return `{ error: { code, message, trace_id } }` on failure.
HTTP 429 carries `Retry-After` in seconds.

## Versioning
Routes are namespaced `/api/v1`. A breaking change requires `/api/v2`.

## Auth
All mutating routes require a bearer JWT. Scope checks run before the handler.
""",
}

store = HybridVectorStore(HashingEmbedder(256))
store.index_documents(DOCUMENTS, chunker=Chunker(target_tokens=140, max_tokens=220))

server = MCPServer(
    ServerInfo(
        name="knowledge_base",
        version="1.2.0",
        instructions="Internal engineering documentation and schema reference.",
    )
)


@server.tool("search_docs", "Semantic + keyword search over internal documentation.")
def search_docs(query: str, limit: int = 4) -> ToolResult:
    hits = store.rerank(store.search(query, k=limit), query)
    return ToolResult.json(
        {
            "query": query,
            "results": [
                {
                    "doc_id": h.chunk.doc_id,
                    "chunk_id": h.chunk.chunk_id,
                    "section": h.chunk.breadcrumb,
                    "score": round(float(h.rerank_score or 0.0), 4),
                    "text": h.chunk.text,
                }
                for h in hits
            ],
        }
    )


@server.tool("get_doc", "Return the full text of one document by id.")
def get_doc(doc_id: str) -> ToolResult:
    body = DOCUMENTS.get(doc_id)
    if body is None:
        return ToolResult.text(
            f"unknown document {doc_id!r}; known ids: {sorted(DOCUMENTS)}", is_error=True
        )
    return ToolResult.text(body)


@server.tool("run_sql_readonly", "Run a SELECT against the read replica.")
def run_sql_readonly(sql: str) -> ToolResult:
    normalised = " ".join(sql.lower().split())
    if not normalised.startswith(("select", "with")):
        return ToolResult.text("only SELECT/WITH statements are permitted", is_error=True)
    if any(word in normalised for word in ("drop ", "truncate ", "delete ", "update ", "insert ")):
        return ToolResult.text("write statements are blocked", is_error=True)
    return ToolResult.json(
        {"sql": sql, "rows": [], "note": "read replica returned 0 rows in this environment"}
    )


server.resource(
    ResourceDefinition("kb://stats", "Index statistics", mime_type="application/json"),
    lambda _uri: json.dumps({"documents": len(DOCUMENTS), "chunks": len(store)}),
)

server.resource_template(
    ResourceTemplate("kb://doc/{doc_id}", "Full document", mime_type="text/markdown")
)


@server.prompt("triage_incident", "Produce a triage prompt for a production incident.")
def triage_incident(service: str, symptom: str) -> str:
    return (
        f"Triage the incident on `{service}`. Symptom: {symptom}. "
        "Check the runbook for the retry policy and alerting thresholds first."
    )


def main() -> int:
    handled = server.serve_stdio()
    return 0 if handled else 1


if __name__ == "__main__":
    raise SystemExit(main())
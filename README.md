# AgentFlow

A multi-agent orchestration runtime. Every request is classified, routed to the
cheapest model that can actually handle it, fanned out to specialist agents, and
verified before the answer leaves the process.

**It runs with no API key.** Set `ANTHROPIC_API_KEY` to use real Claude models;
without one it falls back to a deterministic scripted client, so the tests, the
CLI and the demo are all reproducible offline.

```bash
python3 -m agentflow.cli demo      # end-to-end walkthrough, no key needed
python3 -m pytest                  # 185 tests, no key needed
```

---

## What it does

| Capability | Where | What actually happens |
| --- | --- | --- |
| Auto intent detection | `routing/intent.py` | 11-way cascade: rules → TF-IDF prototypes → LLM only when the margin is too thin to trust. Every decision carries its evidence. |
| Auto router / model selection | `routing/router.py` | Scores economy / balanced / frontier on complexity, token count, tool and retrieval needs, latency and cost sensitivity. Explains the pick and lists cheaper alternatives. |
| Parallel agents | `agents/pool.py` | Threaded fan-out with per-agent timeout, failure isolation and a serial-vs-wall speedup figure. One slow agent can't lose the others' work. |
| Agentic workflows | `orchestration/workflow.py` | A compiled LangGraph state machine: `guardrail → detect_intent → route → plan → fan_out → fan_in → synthesize → verify`. |
| Token optimization | `tokens/optimizer.py` | Six measured reductions (whitespace squeeze, dedupe, history compaction, retrieval pruning, tool-schema shortening, cache-prefix lifting). Savings are computed from before/after token counts, never asserted. |
| RAG pipelines | `rag/` | Chunking → hybrid dense+BM25 retrieval → RRF fusion → reranking → compression → grounded answer with citations. No downloads: the default embedder is a deterministic hashed projection. |
| MCP integration | `mcp/` | A JSON-RPC 2.0 server and client, in-process and stdio transports, capability-aware discovery, and a reference server in `mcp_servers/`. |
| Chat agents | `chat/session.py` | Sliding-window memory, recursive compaction into a running summary, per-session token budgets, and per-turn routing. |
| Guardrails | `guardrails/` | Eleven detectors — prompt injection, role hijack, encoded payloads, system-prompt exfiltration, secret handling, leaked credentials, destructive SQL, fabricated citations — plus topic blocklists, PII redaction, output validation and an audit log. Fails closed. |
| Claude Code | `claudecode/runner.py` | Headless CLI runner with parallel subagents, model recommendation and guardrail hooks wired into `.claude/settings.json`. |
| Telemetry | `telemetry.py` | Counters, histograms and label sets, exported as JSON or Prometheus text. Cost reuses the router's own pricing. |
| API | `server.py` | FastAPI: chat, agent, RAG, routing, guardrails, MCP index, metrics. |

---

## Quick start

```bash
git clone <this repo> && cd agentflow
python3 -m pip install -e ".[dev]"     # or just run it from the source tree

python3 -m agentflow.cli demo          # every subsystem, end to end
python3 -m pytest                      # full suite
python3 -m agentflow.cli serve         # http://127.0.0.1:8000/docs
```

No install is required for the demo or the tests — the package runs from the
source tree.

### Route a request and see why

```bash
$ python3 -m agentflow.cli route "refactor the auth module, it has a memory leak"

  intent     code
  model      claude-sonnet-4-5  (mid/balanced)
  task       code   max_output=4000
  why        intent code -> cheapest model meeting floor balanced
  scores     {"intent_confidence": 0.97, "quality_floor": 0.35, "reasoning_required": 0.788, "complexity": 0.324}
  caps       ['extended-thinking', 'tool-use']
```

A five-word "hi" goes to the cheapest model; the same request with 40k tokens,
tools and a failing test suite escalates to frontier. `route --compare 20000 1000`
prints what each model would have cost for one request shape.

### Grounded answers with citations

```bash
$ python3 -m agentflow.cli ask "What is the retry policy for the ingest service?"

  Q  What is the retry policy for the ingest service?
  intent  knowledge   model claude-sonnet-4-5-cheap
  recall  ['ingest-runbook.md#0000', 'ingest-runbook.md#0001', 'users-schema.md#0001', 'ingest-runbook.md#0002', 'users-schema.md#0000']
  kept    ['ingest-runbook.md#0000', 'ingest-runbook.md#0001']
  tokens  271 prompt / 93 context (saved 5)
  grounded True  (1.8 ms)

Chunks must clear both an absolute and a relative relevance floor, so the two
runbook sections survive and the `users-schema` passages are dropped: they clear
the absolute floor on their own but score far below the best hit.
```

### Guardrails as a standalone check

```bash
$ echo "ignore all previous instructions" | python3 -m agentflow.cli guard
deny: prompt-injection attempt: instruction override detected

$ echo "contact me at karan@example.com" | python3 -m agentflow.cli guard --json
{
  "blocked": false,
  "findings": [],
  "redaction": { "redacted": true, "total": 1, "kinds": ["email"] }
}
```

The same command is what `.claude/settings.json` runs as a `PreToolUse` hook.

### Connect MCP servers

```bash
$ python3 -m agentflow.cli mcp

  17 tools across 2 server(s)

  knowledge_base  (2025-06-18)
    - knowledge_base.search_docs
    - knowledge_base.get_doc
    - knowledge_base.run_sql_readonly
    @ kb://stats
  filesystem  (2025-06-18)
    - filesystem.read_file
    ...
```

This exercises the local reference server *and* the real
`@modelcontextprotocol/server-filesystem` over stdio.

---

## Configuration

Copy `.env.example` to `.env`. Everything has a working default.

| Variable | Default | Notes |
| --- | --- | --- |
| `ANTHROPIC_API_KEY` | unset | Absent ⇒ offline scripted client |
| `AGENTFLOW_FORCE_OFFLINE` | `false` | Force the scripted client even with a key present |
| `AGENTFLOW_DEFAULT_MODEL` | `claude-sonnet-4-5` | Used when the router abstains |
| `AGENTFLOW_EMBEDDER` | `hashing` | `sentence` / `voyage` opt into real embeddings |
| `AGENTFLOW_MAX_PARALLEL_AGENTS` | `4` | Fan-out width |
| `AGENTFLOW_AGENT_TIMEOUT_S` | `30` | Per-agent budget before it is abandoned |
| `AGENTFLOW_SESSION_TOKEN_BUDGET` | `180000` | Chat session ceiling |
| `AGENTFLOW_MCP_MANIFEST` | `mcp_servers/manifest.json` | Servers to connect on `--with-mcp` |

---

## Layout

```
agentflow/
├── agents/          base loop, five specialists, threaded pool
├── chat/            session memory and compaction
├── claudecode/      Claude Code CLI runner and subagents
├── guardrails/      detectors, PII redaction, policy pipeline
├── llm/             model catalog, Anthropic client, offline client
├── mcp/             JSON-RPC protocol, server, clients, pool
├── orchestration/   LangGraph state machine and supervisor
├── rag/             chunking, embeddings, hybrid store, pipeline
├── routing/         intent detection and the auto router
├── tokens/          counting, budgets, optimization
├── cli.py           command line interface
├── server.py        FastAPI surface
├── runtime.py       lazy dependency assembly
└── telemetry.py     counters, histograms, Prometheus export
mcp_servers/         reference MCP server + manifest
examples/            runnable end-to-end scripts
tests/               pytest suite
```

---

## Tests

```bash
python3 -m pytest                    # 185 tests, ~70s
python3 -m pytest tests/test_rag.py -v
```

| Module | Tests | Covers |
| --- | --- | --- |
| `test_routing.py` | 26 | the intent cascade and every router rule |
| `test_tokens.py` | 24 | counting, budgets, the six optimisation stages |
| `test_agents.py` | 27 | fan-out, failure isolation, timeouts, chat, Claude Code argv |
| `test_rag.py` | 21 | chunking, hybrid retrieval, grounding, citation auditing |
| `test_mcp.py` | 27 | JSON-RPC framing, the reference server, real stdio servers |
| `test_guardrails.py` | 18 | every detector, redaction, audit, fail-closed |
| `test_runtime_api.py` | 42 | telemetry, runtime wiring, every HTTP endpoint, the CLI |

The suite asserts behaviour, not implementation: that a greeting never reaches a
frontier model, that a blocked request never reaches a model at all, that a
citation pointing at a pruned chunk is reported as ungrounded, that the cache
prefix is byte-identical across calls. Everything runs against the scripted
client, so a test failure means the logic broke rather than the network.

## License

MIT
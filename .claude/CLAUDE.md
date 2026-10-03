# AgentFlow — Claude Code project context

Multi-agent orchestration runtime. Python 3.10+, LangGraph, Anthropic SDK.

## Commands

```bash
pip install -e ".[dev]"
pytest                                    # test suite (runs fully offline)
python -m agentflow.cli demo              # end-to-end offline walkthrough
python -m agentflow.cli route "..."       # show the routing decision for a request
python -m agentflow.cli mcp               # connect to every server in the manifest
python -m agentflow.cli serve             # FastAPI on :8000
ruff check . && ruff format --check .
```

## Architecture

```
agentflow/
  llm/           provider client, model catalog (capabilities + pricing), offline stub
  routing/       intent.py  -> three-signal cascade classifier
                 router.py  -> intent + complexity -> model + task shape
  tokens/        counter.py, budget.py, optimizer.py (6 reduction stages)
  guardrails/    detectors.py, redaction.py, pipeline.py
  rag/           chunking, embeddings, hybrid store, pipeline
  mcp/           protocol.py, server.py, client.py (JSON-RPC 2.0 over stdio)
  agents/        base.py (the loop), specialists.py, pool.py (fan-out)
  orchestration/ state.py, supervisor.py, workflow.py (LangGraph)
  chat/          session.py (sliding window + compaction)
  claudecode/    runner.py (headless CLI, subagents, hooks)
```

## Conventions that matter here

- **Never break the offline path.** With no `ANTHROPIC_API_KEY`, `AnthropicClient`
  delegates to `ScriptedClient`. Every test and the demo must pass without network.
- **Routing decisions must be explainable.** `RoutingDecision.reason` and `.scores`
  are part of the contract; do not return a bare model id.
- **Guardrails wrap, they do not decorate.** Subclasses must not bypass the base
  `Agent.run` safety/budget/optimisation layers. If an agent needs a different
  generation path, override `run` and re-implement the guards, as `ResearchAgent`
  does for RAG.
- **No new state without a reducer.** `WorkflowState` fields that parallel nodes
  write must use `Annotated[..., merge_fn]` or branches will clobber each other.
- **Cost claims need measurement.** Use `OptimizationReport.saved_tokens` and
  `BudgetSnapshot`, never hand-computed numbers.

## Working agreements

- One concern per module. Do not put agent logic in `routing/` or MCP code in `llm/`.
- Tool errors are data, not exceptions, on the agent path (`ToolInvocation.is_error`).
  Exceptions are reserved for transport failures.
- Add a test with each new guardrail rule and each new intent.
- When changing the model catalog, re-run `agentflow route --compare` and update the
  expected routing table in `tests/test_router.py`.
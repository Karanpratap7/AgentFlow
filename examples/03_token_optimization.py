"""Measure token optimisation and routing decisions.

Every number here is computed from before/after token counts, so the savings
figure is reproducible rather than a claim.

    python3 examples/03_token_optimization.py
"""
import sys
from pathlib import Path

# Runnable straight from a fresh clone, without `pip install -e .`
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


from agentflow.llm.catalog import catalog
from agentflow.llm.types import Message, ToolSchema
from agentflow.routing.router import AutoRouter
from agentflow.tokens.budget import uncached_cost_usd
from agentflow.tokens.optimizer import TokenOptimizer


def optimization_report() -> None:
    optimizer = TokenOptimizer()

    system = "You are AgentFlow.\n" + "Policy text that stays constant. " * 120
    messages = [
        Message("user", "Summarise the retry policy.\n\n\n"),
        # A duplicate tool result: the second copy is pure waste.
        Message("tool", '{"rows": 128, "status": "ok"}'),
        Message("tool", '{"rows": 128, "status": "ok"}'),
        Message("assistant", "The service retries three times."),
        Message("user", "And the alert threshold?"),
    ]
    tools = [
        ToolSchema(
            name="search_docs",
            description="Semantic and keyword search over internal documentation. " * 40,
            input_schema={
                "type": "object",
                "properties": {"query": {"type": "string"}, "top_k": {"type": "integer"}},
                "required": ["query"],
            },
        )
    ]
    retrieved = [
        {"text": f"chunk {i}", "rerank_score": score}
        for i, score in enumerate([0.94, 0.61, 0.22, 0.08])
    ]

    payload = optimizer.optimize(messages, system=system, tools=tools, retrieved=retrieved)
    report = payload.report

    print("TOKEN OPTIMIZATION")
    print(f"  {'stage':<18}{'before':>8}{'after':>8}{'saved':>8}  detail")
    for stage in report.stages:
        print(
            f"  {stage.stage:<18}{stage.tokens_before:>8}{stage.tokens_after:>8}"
            f"{stage.tokens_before - stage.tokens_after:>8}  {stage.detail}"
        )
    print(
        f"\n  total   {report.tokens_before} -> {report.tokens_after} tokens "
        f"({report.saved_tokens} saved, {report.saved_ratio * 100:.1f}%)"
    )
    print(f"  stages  {', '.join(report.applied)}\n")


def routing_report() -> None:
    router = AutoRouter()
    print("MODEL ROUTING")
    for text in (
        "hi",
        "what does the retry policy say?",
        "write a SQL query joining orders and users",
        "refactor and debug every module: memory leak, deadlock, race condition "
        + "across the whole service " * 300,
    ):
        decision = router.route(text)
        shown = text if len(text) < 46 else text[:43] + "..."
        print(
            f"  {shown:46} -> {decision.model:26} "
            f"{decision.task:8} complexity={decision.scores['complexity']:.2f}"
        )
        print(f"  {'':46}    {decision.reason}")

    print("\nCOST FOR ONE 20k-in / 1k-out REQUEST")
    print("(18k of the prompt is a stable prefix, so it is charged at the cache-read rate)")
    comparison = router.compare_models(
        input_tokens=20_000, output_tokens=1_000, cache_prefix_tokens=18_000
    )
    print(f"  {'model':26}{'tier':10}{'uncached $':>12}{'cached $':>11}{'saved %':>9}")
    for row in comparison["models"]:
        print(
            f"  {row['model']:26}{row['tier']:10}"
            f"{row['uncached_usd']:>12.6f}{row['cached_usd']:>11.6f}"
            f"{row['cache_savings_pct'] * 100:>8.1f}%"
        )
    print(f"\n  recommended {comparison['recommended']}, "
          f"saves ${comparison['vs_frontier_savings_usd']:.6f} vs frontier")

    frontier = catalog.get("claude-opus-4-5")
    print(f"\n  frontier uncached cost for the same shape: "
          f"${uncached_cost_usd(frontier, 20_000, 1_000):.6f}")
    print()


if __name__ == "__main__":
    optimization_report()
    routing_report()
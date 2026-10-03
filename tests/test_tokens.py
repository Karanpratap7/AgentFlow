"""Token counting, budgets and the seven optimisation stages."""

import pytest

from agentflow.llm.catalog import catalog
from agentflow.llm.types import ContentBlock, Message, ToolSchema
from agentflow.tokens.budget import TokenBudget, TokenBudgetExceeded, cost_usd, uncached_cost_usd
from agentflow.tokens.counter import (
    estimate_text_tokens,
    estimate_tokens,
    squeeze_whitespace,
)
from agentflow.tokens.optimizer import CONTEXT_OPEN, TokenOptimizer


# -- counting ---------------------------------------------------------------
def test_token_estimate_scales_with_length():
    short = estimate_text_tokens("hello world")
    long = estimate_text_tokens("hello world " * 100)
    assert 0 < short < long


def test_token_estimate_is_stable():
    text = "The ingest service retries a failed batch three times."
    assert estimate_text_tokens(text) == estimate_text_tokens(text)


def test_squeeze_is_lossless_for_content():
    messy = "line one\n\n\n\n   line two   \t\n"
    squeezed = squeeze_whitespace(messy)
    assert "line one" in squeezed and "line two" in squeezed
    assert estimate_text_tokens(squeezed) < estimate_text_tokens(messy)


def test_estimate_counts_the_system_prompt():
    messages = [Message("user", "hello")]
    system = "You are a helpful assistant. " * 20
    assert estimate_tokens(messages, system) > estimate_tokens(messages)


def test_tool_schemas_are_measured_separately():
    """Tool JSON is re-sent on every call, so it has to be counted."""
    messages = [Message("user", "hello")]
    tool = ToolSchema(
        name="t",
        description="A tool. " * 50,
        input_schema={"type": "object", "properties": {}},
    )
    optimizer = TokenOptimizer()
    assert optimizer.measure(messages, None, [tool]) > optimizer.measure(messages)


# -- budget -----------------------------------------------------------------
def test_budget_raises_before_an_oversized_call():
    from agentflow.llm.types import Usage

    budget = TokenBudget(token_limit=1_000)
    budget.check(600)
    budget.charge("claude-haiku-4-5", Usage(input_tokens=400, output_tokens=0))
    with pytest.raises(TokenBudgetExceeded):
        budget.check(700)


def test_charge_enforces_the_ceiling_even_when_checks_all_passed():
    """Pre-flight checks are advisory; ``charge`` is what actually caps spend."""
    from agentflow.llm.types import Usage

    budget = TokenBudget(token_limit=1_000)
    budget.check(400)
    budget.check(400)  # both pre-flights pass: nothing is spent yet
    budget.charge("claude-haiku-4-5", Usage(input_tokens=400, output_tokens=0))
    with pytest.raises(TokenBudgetExceeded):
        budget.charge("claude-haiku-4-5", Usage(input_tokens=400, output_tokens=400))
    assert budget.snapshot().spent_tokens == 400


def test_charge_accumulates_cost_and_counts_cache_hits():
    from agentflow.llm.types import Usage

    budget = TokenBudget(token_limit=10_000)
    usage = Usage(input_tokens=1_000, output_tokens=200)
    first = budget.charge("claude-sonnet-4-5", usage)
    second = budget.charge("claude-sonnet-4-5", usage, cached=True)
    snapshot = budget.snapshot()
    assert snapshot.spent_tokens == 2_400
    assert snapshot.calls == 2
    assert snapshot.cache_hits == 1
    assert snapshot.cache_hit_rate == pytest.approx(0.5)
    assert snapshot.spent_usd == pytest.approx(first + second)
    assert first > 0


def test_charge_of_an_unknown_model_costs_nothing():
    from agentflow.llm.types import Usage

    assert TokenBudget().charge("no-such-model", Usage(input_tokens=10, output_tokens=1)) == 0.0


def test_budget_snapshot_reports_utilisation():
    from agentflow.llm.types import Usage

    budget = TokenBudget(token_limit=1_000)
    budget.charge("claude-haiku-4-5", Usage(input_tokens=250, output_tokens=250))
    snapshot = budget.snapshot()
    assert snapshot.spent_tokens == 500
    assert snapshot.remaining_tokens == 500
    assert snapshot.utilisation == pytest.approx(0.5)
    assert snapshot.calls == 1


def test_cost_matches_the_catalog():
    spec = catalog.get("claude-sonnet-4-5")
    usage = uncached_cost_usd(spec, 10_000, 1_000)
    assert usage == pytest.approx(
        10_000 / 1e6 * spec.input_cost_per_mtok + 1_000 / 1e6 * spec.output_cost_per_mtok
    )


def test_cache_reads_are_cheaper_than_fresh_input():
    from agentflow.llm.types import Usage

    spec = catalog.get("claude-sonnet-4-5")
    fresh = cost_usd(spec, Usage(input_tokens=20_000, output_tokens=0))
    cached = cost_usd(
        spec,
        Usage(input_tokens=2_000, output_tokens=0, cache_read_input_tokens=18_000),
    )
    assert cached < fresh


# -- optimisation -----------------------------------------------------------
def test_every_stage_is_measured():
    optimizer = TokenOptimizer()
    payload = optimizer.optimize(
        [
            Message("user", "hi\n\n\n\n"),
            Message("tool", '{"a": 1}'),
            Message("tool", '{"a": 1}'),
        ],
        system="System. " * 50,
    )
    report = payload.report
    assert report.stages[0].stage == "input"
    assert all(s.tokens_after <= s.tokens_before for s in report.stages)
    assert report.tokens_after <= report.tokens_before
    assert report.saved_tokens == report.tokens_before - report.tokens_after


def test_dedupe_removes_a_verbatim_repeat():
    payload = TokenOptimizer().optimize(
        [Message("tool", '{"rows": 128}'), Message("tool", '{"rows": 128}')]
    )
    assert "dedupe" in payload.report.applied
    assert len(payload.messages) == 1


def test_dedupe_keeps_distinct_tool_results():
    payload = TokenOptimizer().optimize(
        [Message("tool", '{"a": 1}'), Message("tool", '{"a": 2}')]
    )
    assert len(payload.messages) == 2


def test_prune_retrieval_drops_low_relevance_chunks():
    payload = TokenOptimizer(retrieval_floor=0.5).optimize(
        [Message("user", "q")],
        retrieved=[
            {"text": "keep me", "rerank_score": 0.9},
            {"text": "maybe", "rerank_score": 0.6},
            {"text": "drop me", "rerank_score": 0.1},
        ],
    )
    assert len(payload.retrieved) == 2


def test_prune_retrieval_never_empties_the_context():
    payload = TokenOptimizer(retrieval_floor=0.99).optimize(
        [Message("user", "q")],
        retrieved=[{"text": "a", "rerank_score": 0.1}, {"text": "b", "rerank_score": 0.2}],
    )
    assert len(payload.retrieved) == 1


def test_pruning_rewrites_the_context_block_in_the_system_prompt():
    """The context must not be pruned from the chunk list and left in the prompt."""
    optimizer = TokenOptimizer(retrieval_floor=0.5)
    system = f"Rules.\n{CONTEXT_OPEN}\n[1] keep me\n\n[2] drop me\n</retrieved_context>"
    payload = optimizer.optimize(
        [Message("user", "q")],
        system=system,
        retrieved=[
            {"text": "keep me", "rerank_score": 0.9},
            {"text": "drop me", "rerank_score": 0.1},
        ],
    )
    rendered = payload.system if isinstance(payload.system, str) else str(payload.system)
    assert "drop me" not in rendered
    assert "keep me" in rendered


def test_pruning_rewrites_a_context_block_inside_system_blocks():
    optimizer = TokenOptimizer(retrieval_floor=0.5)
    system = [ContentBlock.text(f"Rules.\n{CONTEXT_OPEN}\n[1] keep\n[2] drop\n</retrieved_context>")]
    payload = optimizer.optimize(
        [Message("user", "q")],
        system=system,
        retrieved=[
            {"text": "keep", "rerank_score": 0.9},
            {"text": "drop", "rerank_score": 0.1},
        ],
    )
    rendered = " ".join(str(b.get("text", "")) for b in payload.system)
    assert "drop" not in rendered
    assert "keep" in rendered


def test_shorten_tools_trims_a_bloated_schema():
    tool = ToolSchema(
        name="search",
        description="Search. " * 300,
        input_schema={"type": "object", "properties": {"q": {"type": "string"}}},
    )
    payload = TokenOptimizer().optimize([Message("user", "q")], tools=[tool])
    assert payload.tools[0].description != tool.description
    assert len(payload.tools[0].description) < len(tool.description)
    assert payload.tools[0].name == "search"


def test_cache_prefix_is_byte_identical_across_calls():
    optimizer = TokenOptimizer()
    system = "Stable policy text. " * 60
    first = optimizer.optimize([Message("user", "a")], system=system)
    second = optimizer.optimize([Message("user", "b")], system=system)
    assert first.system == second.system
    assert "cache_prefix" in first.report.applied


def test_disable_a_stage_and_it_is_skipped():
    payload = TokenOptimizer().optimize(
        [Message("tool", '{"a": 1}'), Message("tool", '{"a": 1}')],
        enable=["squeeze"],
    )
    assert "dedupe" not in payload.report.applied
    assert len(payload.messages) == 2


def test_optimisation_never_grows_the_payload():
    optimizer = TokenOptimizer()
    messages = [Message("user", "x " * 200), Message("assistant", "y " * 200)]
    payload = optimizer.optimize(messages, system="z " * 100)
    assert payload.report.tokens_after <= payload.report.tokens_before


def test_savings_are_recorded_in_telemetry():
    from agentflow.telemetry import metrics

    TokenOptimizer().optimize(
        [Message("tool", '{"a": 1}'), Message("tool", '{"a": 1}')], system="s " * 50
    )
    assert metrics.snapshot()["counters"].get("tokens_saved_total", 0) > 0

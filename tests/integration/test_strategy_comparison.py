"""Multi-strategy comparison integration tests.

Tests the MultiStrategyEvaluator, individual strategy execution, and the
auto-mode selection logic end-to-end with mock LLM clients.
"""

import pytest

from rlmkit import MockLLMClient
from rlmkit.api import _determine_auto_mode
from rlmkit.core.budget import estimate_tokens
from rlmkit.strategies.direct import DirectStrategy
from rlmkit.strategies.rlm_strategy import RLMStrategy
from rlmkit.strategies.evaluator import MultiStrategyEvaluator


pytestmark = pytest.mark.integration


class TestStrategyComparison:
    """Run multiple strategies on the same content and compare results."""

    def test_comparison_runs_all_strategies(self, sample_document):
        """Evaluator executes every registered strategy and collects results."""
        # Direct strategy needs a simple answer; RLM needs FINAL-prefixed answer.
        # MockLLMClient repeats the last response once exhausted, so ordering matters.
        # We create separate clients so each strategy gets its own response stream.
        direct_client = MockLLMClient(["Direct answer about the document"])
        rlm_client = MockLLMClient(["FINAL: RLM answer about the document"])

        evaluator = MultiStrategyEvaluator([
            DirectStrategy(client=direct_client),
            RLMStrategy(client=rlm_client),
        ])

        result = evaluator.evaluate(sample_document, "Summarize this")

        assert "direct" in result.results
        assert "rlm" in result.results
        assert result.results["direct"].success
        assert result.results["rlm"].success

    def test_comparison_metrics_populated(self, sample_document):
        """Each strategy result contains populated token, cost, and time metrics."""
        direct_client = MockLLMClient(["The summary is here."])
        rlm_client = MockLLMClient([
            "```python\nprint(peek(0, 100))\n```",
            "FINAL: Explored and found the answer",
        ])

        evaluator = MultiStrategyEvaluator([
            DirectStrategy(client=direct_client),
            RLMStrategy(client=rlm_client),
        ])

        result = evaluator.evaluate(sample_document, "What is this about?")

        for name, sr in result.results.items():
            assert sr.tokens.total_tokens > 0, f"{name} should have token usage"
            assert sr.elapsed_time >= 0, f"{name} should have non-negative elapsed time"
            assert sr.steps >= 1, f"{name} should have at least one step"

    def test_comparison_summary_identifies_winners(self, sample_document):
        """get_summary() identifies fastest and cheapest strategies."""
        direct_client = MockLLMClient(["Quick answer"])
        rlm_client = MockLLMClient(["FINAL: Thorough answer"])

        evaluator = MultiStrategyEvaluator([
            DirectStrategy(client=direct_client),
            RLMStrategy(client=rlm_client),
        ])

        result = evaluator.evaluate(sample_document, "Summarize")
        summary = result.get_summary()

        assert "fastest" in summary
        assert "cheapest" in summary
        assert "fewest_tokens" in summary
        assert summary["fastest"] in ("direct", "rlm")

    def test_comparison_pairwise(self, sample_document):
        """get_comparison() returns delta metrics between two strategies."""
        direct_client = MockLLMClient(["Answer A"])
        rlm_client = MockLLMClient(["FINAL: Answer B"])

        evaluator = MultiStrategyEvaluator([
            DirectStrategy(client=direct_client),
            RLMStrategy(client=rlm_client),
        ])

        result = evaluator.evaluate(sample_document, "Compare")
        cmp = result.get_comparison("direct", "rlm")

        assert cmp is not None
        assert "tokens" in cmp
        assert "cost" in cmp
        assert "time" in cmp
        assert "delta" in cmp["tokens"]

    def test_comparison_handles_strategy_failure(self, sample_document):
        """If one strategy fails, the other still succeeds in the evaluator."""
        good_client = MockLLMClient(["Works fine"])

        class _FailingClient:
            def complete(self, messages):
                raise RuntimeError("provider down")

        evaluator = MultiStrategyEvaluator([
            DirectStrategy(client=good_client),
            DirectStrategy(client=_FailingClient()),
        ])

        # Both strategies have name="direct", so the second overwrites the first
        # in the result dict. Use a custom wrapper instead.
        class _NamedDirect:
            def __init__(self, client, name_val):
                self._inner = DirectStrategy(client=client)
                self._name = name_val

            @property
            def name(self):
                return self._name

            def run(self, content, query):
                return self._inner.run(content, query)

        evaluator = MultiStrategyEvaluator([
            _NamedDirect(good_client, "good"),
            _NamedDirect(_FailingClient(), "bad"),
        ])

        result = evaluator.evaluate(sample_document, "Test failure handling")

        assert result.results["good"].success
        assert not result.results["bad"].success
        assert "provider down" in result.results["bad"].error


class TestAutoModeSelection:
    """Verify auto-mode picks the right strategy based on content size."""

    def test_short_content_selects_direct(self):
        """Content under 8K tokens resolves to direct mode."""
        # 8000 tokens * 4 chars/token = 32000 chars threshold
        short = "a" * 1000  # ~250 tokens, well under 8K
        assert _determine_auto_mode(short) == "direct"

    def test_medium_content_selects_rag(self):
        """Content between 8K and 100K tokens resolves to rag mode."""
        # Need >= 8000 tokens -> >= 32000 chars
        medium = "a" * 40000  # ~10000 tokens
        assert _determine_auto_mode(medium) == "rag"

    def test_large_content_selects_rlm(self):
        """Content over 100K tokens resolves to rlm mode."""
        # Need >= 100000 tokens -> >= 400000 chars
        large = "a" * 500000  # ~125000 tokens
        assert _determine_auto_mode(large) == "rlm"

    def test_boundary_at_8k_tokens(self):
        """Content exactly at the 8K-token boundary selects rag."""
        # estimate_tokens uses len(text) // 4, so 32000 chars = 8000 tokens
        boundary = "a" * 32000
        assert estimate_tokens(boundary) == 8000
        assert _determine_auto_mode(boundary) == "rag"

    def test_boundary_at_100k_tokens(self):
        """Content exactly at the 100K-token boundary selects rlm."""
        boundary = "a" * 400000
        assert estimate_tokens(boundary) == 100000
        assert _determine_auto_mode(boundary) == "rlm"


class TestEvaluationBatch:
    """Batch evaluation across multiple queries."""

    def test_evaluate_batch_returns_per_query_results(self, sample_document):
        """evaluate_batch processes each query independently."""
        client = MockLLMClient(["FINAL: batch answer"])

        evaluator = MultiStrategyEvaluator([RLMStrategy(client=client)])
        queries = ["Question 1", "Question 2", "Question 3"]
        results = evaluator.evaluate_batch(sample_document, queries)

        assert len(results) == 3
        for i, er in enumerate(results):
            assert er.query == queries[i]
            assert "rlm" in er.results

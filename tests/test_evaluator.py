"""Tests for MultiStrategyEvaluator."""

import pytest
from rlmkit import MockLLMClient
from rlmkit.strategies.base import StrategyResult
from rlmkit.strategies.direct import DirectStrategy
from rlmkit.strategies.rlm_strategy import RLMStrategy
from rlmkit.strategies.evaluator import MultiStrategyEvaluator, EvaluationResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class StubStrategy:
    """Minimal strategy for testing the evaluator."""

    def __init__(self, name_val: str, answer: str = "ok"):
        self._name = name_val
        self._answer = answer

    @property
    def name(self) -> str:
        return self._name

    def run(self, content: str, query: str) -> StrategyResult:
        return StrategyResult(
            strategy=self._name,
            answer=self._answer,
            steps=1,
            elapsed_time=0.1,
            cost=0.001,
        )


class FailingStrategy:
    """Strategy that always raises."""

    @property
    def name(self) -> str:
        return "failing"

    def run(self, content: str, query: str) -> StrategyResult:
        raise RuntimeError("strategy exploded")


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestMultiStrategyEvaluator:
    def test_evaluate_all_succeed(self):
        ev = MultiStrategyEvaluator([StubStrategy("a"), StubStrategy("b")])
        result = ev.evaluate("content", "query")

        assert "a" in result.results
        assert "b" in result.results
        assert result.results["a"].success
        assert result.results["b"].success

    def test_evaluate_with_failure(self):
        ev = MultiStrategyEvaluator([StubStrategy("good"), FailingStrategy()])
        result = ev.evaluate("content", "query")

        assert result.results["good"].success
        assert not result.results["failing"].success
        assert "strategy exploded" in result.results["failing"].error

    def test_evaluate_batch(self):
        ev = MultiStrategyEvaluator([StubStrategy("s1")])
        results = ev.evaluate_batch("content", ["q1", "q2", "q3"])

        assert len(results) == 3
        assert results[0].query == "q1"
        assert results[2].query == "q3"

    def test_with_real_strategies(self):
        client = MockLLMClient(["FINAL: rlm answer", "direct answer"])
        ev = MultiStrategyEvaluator([
            RLMStrategy(client=client),
            DirectStrategy(client=client),
        ])
        result = ev.evaluate("some text", "what?")

        assert "rlm" in result.results
        assert "direct" in result.results
        assert result.results["rlm"].success
        assert result.results["direct"].success


class TestEvaluationResult:
    def test_get_comparison(self):
        er = EvaluationResult(query="q", content_length=100)
        er.results["a"] = StrategyResult(
            strategy="a", answer="x", elapsed_time=1.0, cost=0.01
        )
        er.results["b"] = StrategyResult(
            strategy="b", answer="y", elapsed_time=2.0, cost=0.02
        )

        cmp = er.get_comparison("a", "b")
        assert cmp is not None
        assert cmp["time"]["delta"] == pytest.approx(-1.0)
        assert cmp["cost"]["delta"] == pytest.approx(-0.01)

    def test_get_comparison_missing(self):
        er = EvaluationResult(query="q", content_length=100)
        er.results["a"] = StrategyResult(strategy="a", answer="x")
        assert er.get_comparison("a", "z") is None

    def test_get_summary(self):
        er = EvaluationResult(query="q", content_length=100)
        er.results["fast"] = StrategyResult(
            strategy="fast", answer="x", elapsed_time=0.5, cost=0.1
        )
        er.results["cheap"] = StrategyResult(
            strategy="cheap", answer="y", elapsed_time=1.0, cost=0.01
        )

        s = er.get_summary()
        assert s["fastest"] == "fast"
        assert s["cheapest"] == "cheap"
        assert "fast" in s["per_strategy"]

    def test_get_summary_empty(self):
        er = EvaluationResult(query="q", content_length=0)
        s = er.get_summary()
        assert s["strategies"] == []

    def test_to_comparison_result(self):
        er = EvaluationResult(query="q", content_length=100)
        er.results["rlm"] = StrategyResult(strategy="rlm", answer="a", steps=3)
        er.results["direct"] = StrategyResult(strategy="direct", answer="b", steps=0)

        cr = er.to_comparison_result()
        assert cr.rlm_metrics is not None
        assert cr.direct_metrics is not None
        assert cr.rlm_metrics.mode == "rlm"
        assert cr.direct_metrics.mode == "direct"

    def test_to_comparison_result_partial(self):
        er = EvaluationResult(query="q", content_length=100)
        er.results["rlm"] = StrategyResult(strategy="rlm", answer="a")

        cr = er.to_comparison_result()
        assert cr.rlm_metrics is not None
        assert cr.direct_metrics is None

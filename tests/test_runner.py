"""Tests for BenchmarkRunner."""

import pytest
from rlmkit.strategies.base import StrategyResult
from rlmkit.benchmark.dataset import BenchmarkCase, BenchmarkDataset
from rlmkit.benchmark.runner import BenchmarkRunner, BenchmarkRun, CaseResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class StubStrategy:
    """Minimal strategy that returns a fixed answer."""

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
            elapsed_time=0.01,
            cost=0.001,
        )


class FailingStrategy:
    @property
    def name(self) -> str:
        return "failing"

    def run(self, content: str, query: str) -> StrategyResult:
        raise RuntimeError("boom")


def _make_dataset(n: int = 3) -> BenchmarkDataset:
    cases = [
        BenchmarkCase(
            id=f"case_{i}",
            content=f"content {i}",
            query=f"query {i}",
            category="test",
        )
        for i in range(n)
    ]
    return BenchmarkDataset(name="test_ds", cases=cases)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestBenchmarkRunner:
    def test_run_all_cases(self):
        runner = BenchmarkRunner(strategies=[StubStrategy("a"), StubStrategy("b")])
        result = runner.run(_make_dataset(3))

        assert isinstance(result, BenchmarkRun)
        assert result.case_count == 3
        assert result.dataset_name == "test_ds"
        assert result.strategy_names == ["a", "b"]

    def test_run_collects_results(self):
        runner = BenchmarkRunner(strategies=[StubStrategy("s1")])
        result = runner.run(_make_dataset(2))

        for cr in result.case_results:
            assert "s1" in cr.evaluation.results
            assert cr.evaluation.results["s1"].success
            assert cr.evaluation.results["s1"].answer == "ok"

    def test_run_with_case_ids_filter(self):
        runner = BenchmarkRunner(strategies=[StubStrategy("s1")])
        result = runner.run(_make_dataset(5), case_ids=["case_1", "case_3"])

        assert result.case_count == 2
        ids = [cr.case.id for cr in result.case_results]
        assert ids == ["case_1", "case_3"]

    def test_run_with_callback(self):
        completed = []
        runner = BenchmarkRunner(
            strategies=[StubStrategy("s1")],
            on_case_complete=lambda cr: completed.append(cr.case.id),
        )
        runner.run(_make_dataset(3))
        assert completed == ["case_0", "case_1", "case_2"]

    def test_run_with_failing_strategy(self):
        runner = BenchmarkRunner(strategies=[StubStrategy("good"), FailingStrategy()])
        result = runner.run(_make_dataset(2))

        assert result.case_count == 2
        for cr in result.case_results:
            assert cr.evaluation.results["good"].success
            assert not cr.evaluation.results["failing"].success
            assert "boom" in cr.evaluation.results["failing"].error

    def test_run_case_single(self):
        runner = BenchmarkRunner(strategies=[StubStrategy("s1")])
        case = BenchmarkCase(id="single", content="hi", query="q")
        cr = runner.run_case(case)

        assert isinstance(cr, CaseResult)
        assert cr.case.id == "single"
        assert "s1" in cr.evaluation.results

    def test_elapsed_time_tracked(self):
        runner = BenchmarkRunner(strategies=[StubStrategy("s1")])
        result = runner.run(_make_dataset(2))

        assert result.total_elapsed_time > 0
        for cr in result.case_results:
            assert cr.elapsed_time > 0


class TestBenchmarkRun:
    def _make_run(self):
        runner = BenchmarkRunner(
            strategies=[StubStrategy("fast", "a"), StubStrategy("slow", "b")]
        )
        return runner.run(_make_dataset(3))

    def test_success_rate(self):
        run = self._make_run()
        rates = run.success_rate
        assert rates["fast"] == 1.0
        assert rates["slow"] == 1.0

    def test_success_rate_with_failures(self):
        runner = BenchmarkRunner(strategies=[StubStrategy("good"), FailingStrategy()])
        run = runner.run(_make_dataset(4))
        rates = run.success_rate
        assert rates["good"] == 1.0
        assert rates["failing"] == 0.0

    def test_get_strategy_metrics(self):
        run = self._make_run()
        m = run.get_strategy_metrics("fast")
        assert m["strategy"] == "fast"
        assert m["cases"] == 3
        assert m["successes"] == 3
        assert m["success_rate"] == 1.0
        assert m["total_tokens"] >= 0
        assert m["avg_cost"] == pytest.approx(0.001)

    def test_get_strategy_metrics_missing(self):
        run = self._make_run()
        m = run.get_strategy_metrics("nonexistent")
        assert m["cases"] == 0

    def test_to_dict(self):
        run = self._make_run()
        d = run.to_dict()
        assert d["dataset_name"] == "test_ds"
        assert d["case_count"] == 3
        assert "fast" in d["per_strategy"]
        assert "slow" in d["per_strategy"]
        assert len(d["cases"]) == 3


class TestCaseResult:
    def test_to_dict(self):
        runner = BenchmarkRunner(strategies=[StubStrategy("s1")])
        case = BenchmarkCase(id="t", content="x", query="q")
        cr = runner.run_case(case)
        d = cr.to_dict()
        assert d["case"]["id"] == "t"
        assert "evaluation" in d
        assert "elapsed_time" in d

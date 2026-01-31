"""Tests for BenchmarkReport."""

import json
import csv
import pytest
from io import StringIO

from rlmkit.strategies.base import StrategyResult
from rlmkit.benchmark.dataset import BenchmarkCase, BenchmarkDataset
from rlmkit.benchmark.runner import BenchmarkRunner, BenchmarkRun
from rlmkit.benchmark.report import BenchmarkReport, _pct_delta


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class StubStrategy:
    def __init__(self, name_val: str, answer: str = "ok", cost: float = 0.01, time: float = 0.1):
        self._name = name_val
        self._answer = answer
        self._cost = cost
        self._time = time

    @property
    def name(self) -> str:
        return self._name

    def run(self, content: str, query: str) -> StrategyResult:
        return StrategyResult(
            strategy=self._name,
            answer=self._answer,
            steps=1,
            elapsed_time=self._time,
            cost=self._cost,
        )


def _make_run(n_cases: int = 3) -> BenchmarkRun:
    cases = [
        BenchmarkCase(id=f"c{i}", content=f"text {i}", query=f"q{i}", category="test")
        for i in range(n_cases)
    ]
    ds = BenchmarkDataset(name="test_ds", cases=cases)
    runner = BenchmarkRunner(
        strategies=[
            StubStrategy("fast", cost=0.001, time=0.05),
            StubStrategy("expensive", cost=0.1, time=0.5),
        ]
    )
    return runner.run(ds)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestBenchmarkReport:
    def test_summary(self):
        run = _make_run()
        report = BenchmarkReport(run)
        s = report.summary()

        assert s["dataset"] == "test_ds"
        assert s["cases"] == 3
        assert "fast" in s["strategies"]
        assert "expensive" in s["strategies"]
        assert s["winners"]["cheapest"] == "fast"
        assert s["winners"]["fastest"] == "fast"

    def test_summary_with_no_cases(self):
        run = _make_run(0)
        report = BenchmarkReport(run)
        s = report.summary()
        assert s["cases"] == 0
        assert s["winners"] == {}

    def test_pairwise_comparison(self):
        run = _make_run()
        report = BenchmarkReport(run)
        cmp = report.pairwise_comparison("fast", "expensive")

        assert cmp["strategies"] == ("fast", "expensive")
        assert cmp["cost"]["delta"] < 0  # fast is cheaper
        assert cmp["time"]["delta"] < 0  # fast is faster
        assert cmp["cost"]["delta_pct"] < 0

    def test_pairwise_comparison_missing_strategy(self):
        run = _make_run()
        report = BenchmarkReport(run)
        cmp = report.pairwise_comparison("fast", "nonexistent")
        assert "error" in cmp

    def test_per_case_table(self):
        run = _make_run(2)
        report = BenchmarkReport(run)
        rows = report.per_case_table()

        # 2 cases x 2 strategies = 4 rows
        assert len(rows) == 4
        assert all("case_id" in r for r in rows)
        assert all("strategy" in r for r in rows)
        assert all("tokens_total" in r for r in rows)
        assert all("cost" in r for r in rows)

    def test_save_json(self, tmp_path):
        run = _make_run()
        report = BenchmarkReport(run)
        path = str(tmp_path / "results.json")
        report.save_json(path)

        with open(path) as f:
            data = json.load(f)
        assert "summary" in data
        assert "cases" in data
        assert data["summary"]["dataset"] == "test_ds"

    def test_save_csv(self, tmp_path):
        run = _make_run(2)
        report = BenchmarkReport(run)
        path = str(tmp_path / "results.csv")
        report.save_csv(path)

        with open(path) as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        assert len(rows) == 4  # 2 cases x 2 strategies
        assert "case_id" in rows[0]
        assert "strategy" in rows[0]

    def test_save_csv_empty(self, tmp_path):
        run = _make_run(0)
        report = BenchmarkReport(run)
        path = str(tmp_path / "empty.csv")
        report.save_csv(path)

        with open(path) as f:
            content = f.read()
        assert content == ""

    def test_to_csv_string(self):
        run = _make_run(2)
        report = BenchmarkReport(run)
        csv_str = report.to_csv_string()

        reader = csv.DictReader(StringIO(csv_str))
        rows = list(reader)
        assert len(rows) == 4

    def test_save_json_creates_directories(self, tmp_path):
        run = _make_run(1)
        report = BenchmarkReport(run)
        path = str(tmp_path / "sub" / "dir" / "results.json")
        report.save_json(path)

        with open(path) as f:
            data = json.load(f)
        assert data["summary"]["cases"] == 1


class TestPctDelta:
    def test_positive(self):
        assert _pct_delta(120, 100) == pytest.approx(20.0)

    def test_negative(self):
        assert _pct_delta(80, 100) == pytest.approx(-20.0)

    def test_zero_base(self):
        assert _pct_delta(100, 0) == 0.0

    def test_equal(self):
        assert _pct_delta(50, 50) == pytest.approx(0.0)

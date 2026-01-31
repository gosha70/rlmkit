# Copyright (c) EGOGE - All Rights Reserved.
# This software may be used and distributed according to the terms of the MIT license.

"""Benchmark report generation with aggregation and export."""

import csv
import io
import json
from pathlib import Path
from typing import Dict, Any, List, Optional

from .runner import BenchmarkRun


class BenchmarkReport:
    """Generate reports from a benchmark run.

    Aggregates metrics across strategies and cases, and exports
    results to JSON or CSV.

    Usage::

        report = BenchmarkReport(run)
        report.save_json("results.json")
        report.save_csv("results.csv")
        summary = report.summary()
    """

    def __init__(self, run: BenchmarkRun):
        self.run = run

    def summary(self) -> Dict[str, Any]:
        """High-level summary of the benchmark run."""
        per_strategy = {
            name: self.run.get_strategy_metrics(name)
            for name in self.run.strategy_names
        }

        # Find winners
        winners: Dict[str, Any] = {}
        successful_strategies = {
            name: m for name, m in per_strategy.items() if m.get("cases", 0) > 0
        }

        if successful_strategies:
            winners["fastest"] = min(
                successful_strategies,
                key=lambda n: successful_strategies[n].get("avg_time", float("inf")),
            )
            winners["cheapest"] = min(
                successful_strategies,
                key=lambda n: successful_strategies[n].get("avg_cost", float("inf")),
            )
            winners["fewest_tokens"] = min(
                successful_strategies,
                key=lambda n: successful_strategies[n].get("avg_tokens", float("inf")),
            )
            winners["most_reliable"] = max(
                successful_strategies,
                key=lambda n: successful_strategies[n].get("success_rate", 0),
            )

        return {
            "dataset": self.run.dataset_name,
            "cases": self.run.case_count,
            "strategies": self.run.strategy_names,
            "total_time": self.run.total_elapsed_time,
            "success_rates": self.run.success_rate,
            "per_strategy": per_strategy,
            "winners": winners,
        }

    def pairwise_comparison(self, strategy_a: str, strategy_b: str) -> Dict[str, Any]:
        """Compare two strategies across all cases."""
        a_metrics = self.run.get_strategy_metrics(strategy_a)
        b_metrics = self.run.get_strategy_metrics(strategy_b)

        if a_metrics.get("cases", 0) == 0 or b_metrics.get("cases", 0) == 0:
            return {"error": "One or both strategies have no results"}

        return {
            "strategies": (strategy_a, strategy_b),
            "tokens": {
                strategy_a: a_metrics["avg_tokens"],
                strategy_b: b_metrics["avg_tokens"],
                "delta": a_metrics["avg_tokens"] - b_metrics["avg_tokens"],
                "delta_pct": _pct_delta(a_metrics["avg_tokens"], b_metrics["avg_tokens"]),
            },
            "cost": {
                strategy_a: a_metrics["avg_cost"],
                strategy_b: b_metrics["avg_cost"],
                "delta": a_metrics["avg_cost"] - b_metrics["avg_cost"],
                "delta_pct": _pct_delta(a_metrics["avg_cost"], b_metrics["avg_cost"]),
            },
            "time": {
                strategy_a: a_metrics["avg_time"],
                strategy_b: b_metrics["avg_time"],
                "delta": a_metrics["avg_time"] - b_metrics["avg_time"],
                "delta_pct": _pct_delta(a_metrics["avg_time"], b_metrics["avg_time"]),
            },
            "success_rate": {
                strategy_a: a_metrics["success_rate"],
                strategy_b: b_metrics["success_rate"],
            },
        }

    def per_case_table(self) -> List[Dict[str, Any]]:
        """Flat table of per-case, per-strategy metrics for CSV export."""
        rows: List[Dict[str, Any]] = []
        for cr in self.run.case_results:
            for name in self.run.strategy_names:
                sr = cr.evaluation.results.get(name)
                if sr is None:
                    continue
                rows.append({
                    "case_id": cr.case.id,
                    "category": cr.case.category,
                    "difficulty": cr.case.difficulty,
                    "content_length": cr.case.content_length,
                    "strategy": name,
                    "success": sr.success,
                    "answer_length": len(sr.answer),
                    "steps": sr.steps,
                    "tokens_input": sr.tokens.input_tokens,
                    "tokens_output": sr.tokens.output_tokens,
                    "tokens_total": sr.tokens.total_tokens,
                    "cost": sr.cost,
                    "elapsed_time": sr.elapsed_time,
                    "error": sr.error or "",
                })
        return rows

    def save_json(self, path: str) -> None:
        """Export full benchmark results to JSON."""
        filepath = Path(path)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "summary": self.summary(),
            "cases": self.per_case_table(),
        }
        with open(filepath, "w") as f:
            json.dump(data, f, indent=2)

    def save_csv(self, path: str) -> None:
        """Export per-case metrics to CSV."""
        filepath = Path(path)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        rows = self.per_case_table()
        if not rows:
            filepath.write_text("")
            return

        fieldnames = list(rows[0].keys())
        with open(filepath, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    def to_csv_string(self) -> str:
        """Return CSV content as a string."""
        rows = self.per_case_table()
        if not rows:
            return ""
        buf = io.StringIO()
        writer = csv.DictWriter(buf, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
        return buf.getvalue()


def _pct_delta(a: float, b: float) -> float:
    """Percentage delta of a relative to b: (a - b) / b * 100."""
    if b == 0:
        return 0.0
    return (a - b) / b * 100.0

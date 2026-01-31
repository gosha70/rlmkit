# Copyright (c) EGOGE - All Rights Reserved.
# This software may be used and distributed according to the terms of the MIT license.

"""Multi-strategy evaluator for comparing N strategies on the same query."""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional

from .base import LLMStrategy, StrategyResult


@dataclass
class EvaluationResult:
    """Results from running multiple strategies on the same query."""

    query: str
    content_length: int
    results: Dict[str, StrategyResult] = field(default_factory=dict)

    def get_comparison(self, a: str, b: str) -> Optional[Dict[str, Any]]:
        """Pairwise comparison between two strategies."""
        ra, rb = self.results.get(a), self.results.get(b)
        if ra is None or rb is None:
            return None
        return {
            "strategies": (a, b),
            "tokens": {
                a: ra.tokens.total_tokens,
                b: rb.tokens.total_tokens,
                "delta": ra.tokens.total_tokens - rb.tokens.total_tokens,
            },
            "cost": {a: ra.cost, b: rb.cost, "delta": ra.cost - rb.cost},
            "time": {
                a: ra.elapsed_time,
                b: rb.elapsed_time,
                "delta": ra.elapsed_time - rb.elapsed_time,
            },
            "steps": {a: ra.steps, b: rb.steps},
            "success": {a: ra.success, b: rb.success},
        }

    def get_summary(self) -> Dict[str, Any]:
        """Summary across all strategies."""
        summary: Dict[str, Any] = {
            "query": self.query,
            "content_length": self.content_length,
            "strategies": list(self.results.keys()),
        }
        if not self.results:
            return summary

        successful = {k: v for k, v in self.results.items() if v.success}
        if successful:
            summary["fastest"] = min(successful, key=lambda k: successful[k].elapsed_time)
            summary["cheapest"] = min(successful, key=lambda k: successful[k].cost)
            summary["fewest_tokens"] = min(
                successful, key=lambda k: successful[k].tokens.total_tokens
            )

        summary["per_strategy"] = {k: v.to_dict() for k, v in self.results.items()}
        return summary

    def to_comparison_result(self) -> "ComparisonResult":
        """Convert to existing ComparisonResult for backward-compat UI."""
        from rlmkit.core.comparison import ComparisonResult

        cr = ComparisonResult()
        if "rlm" in self.results:
            cr.rlm_metrics = self.results["rlm"].to_execution_metrics()
        if "direct" in self.results:
            cr.direct_metrics = self.results["direct"].to_execution_metrics()
        return cr


class MultiStrategyEvaluator:
    """Run N strategies on the same content + query and collect results."""

    def __init__(self, strategies: List[LLMStrategy]):
        self.strategies = strategies

    def evaluate(self, content: str, query: str) -> EvaluationResult:
        result = EvaluationResult(query=query, content_length=len(content))
        for strategy in self.strategies:
            try:
                sr = strategy.run(content, query)
                result.results[strategy.name] = sr
            except Exception as e:
                result.results[strategy.name] = StrategyResult(
                    strategy=strategy.name,
                    answer="",
                    success=False,
                    error=str(e),
                )
        return result

    def evaluate_batch(
        self, content: str, queries: List[str]
    ) -> List[EvaluationResult]:
        return [self.evaluate(content, q) for q in queries]

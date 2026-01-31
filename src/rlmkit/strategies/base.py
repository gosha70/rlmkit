# Copyright (c) EGOGE - All Rights Reserved.
# This software may be used and distributed according to the terms of the MIT license.

"""Strategy protocol and unified result type for multi-strategy evaluation."""

from typing import Protocol, List, Dict, Any, Optional, runtime_checkable
from dataclasses import dataclass, field

from rlmkit.core.budget import TokenUsage


@runtime_checkable
class LLMStrategy(Protocol):
    """Common interface for all evaluation strategies."""

    @property
    def name(self) -> str: ...

    def run(self, content: str, query: str) -> "StrategyResult": ...


@dataclass
class StrategyResult:
    """Unified result from any strategy execution."""

    strategy: str
    answer: str
    success: bool = True
    error: Optional[str] = None
    steps: int = 0
    tokens: TokenUsage = field(default_factory=TokenUsage)
    cost: float = 0.0
    elapsed_time: float = 0.0
    trace: List[Dict[str, Any]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "strategy": self.strategy,
            "answer": self.answer,
            "success": self.success,
            "error": self.error,
            "steps": self.steps,
            "tokens": self.tokens.to_dict(),
            "cost": self.cost,
            "elapsed_time": self.elapsed_time,
            "trace_length": len(self.trace),
            "metadata": self.metadata,
        }

    def to_execution_metrics(self) -> "ExecutionMetrics":
        """Convert to existing comparison.ExecutionMetrics for UI compat."""
        from rlmkit.core.comparison import ExecutionMetrics

        return ExecutionMetrics(
            mode=self.strategy,
            answer=self.answer,
            steps=self.steps,
            tokens=self.tokens,
            elapsed_time=self.elapsed_time,
            cost=self.cost,
            success=self.success,
            error=self.error,
            trace=self.trace,
        )

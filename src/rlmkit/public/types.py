"""Public type definitions for the RLMKit API.

These provide stable, user-facing types that shield callers from
internal implementation changes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class PublicRunResult:
    """Result from a single-mode run through the public API.

    Attributes:
        answer: Final answer text.
        mode_used: Execution mode that was actually used.
        success: Whether execution completed without error.
        error: Error description, if any.
        total_tokens: Total tokens consumed (input + output).
        input_tokens: Input tokens consumed.
        output_tokens: Output tokens consumed.
        total_cost: Total cost in USD.
        elapsed_time: Wall-clock seconds.
        steps: Number of execution steps.
        trace: Serialized execution trace for inspection.
        metadata: Additional result metadata.
    """

    answer: str = ""
    mode_used: str = ""
    success: bool = True
    error: Optional[str] = None
    total_tokens: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    total_cost: float = 0.0
    elapsed_time: float = 0.0
    steps: int = 0
    trace: List[Dict[str, Any]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __str__(self) -> str:
        return self.answer

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to a plain dictionary."""
        return {
            "answer": self.answer,
            "mode_used": self.mode_used,
            "success": self.success,
            "error": self.error,
            "total_tokens": self.total_tokens,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_cost": self.total_cost,
            "elapsed_time": self.elapsed_time,
            "steps": self.steps,
            "has_trace": len(self.trace) > 0,
            "metadata": self.metadata,
        }


@dataclass
class PublicInteractResult:
    """Result from an interact() call, backward-compatible with the
    existing ``InteractResult`` shape.

    Attributes:
        answer: The generated response text.
        mode_used: Which strategy was actually used.
        metrics: Token usage, cost, and timing information.
        trace: Optional execution trace.
    """

    answer: str
    mode_used: str
    metrics: Dict[str, Any] = field(default_factory=dict)
    trace: Optional[Any] = None

    def __str__(self) -> str:
        return self.answer

    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary format."""
        return {
            "answer": self.answer,
            "mode_used": self.mode_used,
            "metrics": self.metrics,
            "has_trace": self.trace is not None,
        }

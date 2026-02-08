"""Domain entities: core business objects for RLMKit.

All entities are plain Python dataclasses with NO external dependencies.
They represent the fundamental concepts of the RLM paradigm.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional


@dataclass
class Query:
    """A user query to be processed by the RLM system.

    Attributes:
        content: The document or text content to analyze.
        question: The user's question about the content.
        mode: Requested execution mode.
        metadata: Arbitrary key-value metadata attached to the query.
    """

    content: str
    question: str
    mode: Literal["direct", "rag", "rlm", "auto", "compare"] = "auto"
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Response:
    """The result produced by an RLM execution.

    Attributes:
        answer: Final answer text.
        mode_used: The mode that actually produced this answer.
        success: Whether execution completed without error.
        error: Error description if execution failed.
        steps: Number of execution steps taken.
        metadata: Additional response metadata.
    """

    answer: str
    mode_used: str
    success: bool = True
    error: Optional[str] = None
    steps: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TraceStep:
    """A single step in the RLM execution trace.

    Attributes:
        index: Step number (0-indexed).
        action_type: Type of action taken (inspect, subcall, final, error).
        code: Python code executed, if any.
        output: Result of code execution, if any.
        tokens_used: Number of tokens consumed in this step.
        timestamp: Unix timestamp when this step occurred.
        recursion_depth: Depth in the recursion tree (0 = root).
        cost: Estimated monetary cost for this step.
        duration: Wall-clock time for this step in seconds.
        model: Model identifier used for this step.
        raw_response: Full LLM response text.
        error: Error message if this step failed.
    """

    index: int
    action_type: Literal["inspect", "subcall", "final", "error"]
    code: Optional[str] = None
    output: Optional[str] = None
    tokens_used: int = 0
    timestamp: float = 0.0
    recursion_depth: int = 0
    cost: float = 0.0
    duration: float = 0.0
    model: Optional[str] = None
    raw_response: Optional[str] = None
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to a plain dictionary."""
        return {
            "index": self.index,
            "action_type": self.action_type,
            "code": self.code,
            "output": self.output,
            "tokens_used": self.tokens_used,
            "timestamp": self.timestamp,
            "recursion_depth": self.recursion_depth,
            "cost": self.cost,
            "duration": self.duration,
            "model": self.model,
            "raw_response": self.raw_response,
            "error": self.error,
        }


@dataclass
class ExecutionTrace:
    """Complete execution trace for an RLM run.

    Attributes:
        steps: Ordered list of trace steps.
        start_time: Unix timestamp when execution started.
        end_time: Unix timestamp when execution finished (None if still running).
        metadata: Additional trace-level metadata.
    """

    steps: List[TraceStep] = field(default_factory=list)
    start_time: float = 0.0
    end_time: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def add_step(self, step: TraceStep) -> None:
        """Append a step to the trace."""
        self.steps.append(step)

    @property
    def total_tokens(self) -> int:
        """Sum of tokens across all steps."""
        return sum(s.tokens_used for s in self.steps)

    @property
    def total_cost(self) -> float:
        """Sum of costs across all steps."""
        return sum(s.cost for s in self.steps)

    @property
    def max_depth(self) -> int:
        """Maximum recursion depth reached."""
        if not self.steps:
            return 0
        return max(s.recursion_depth for s in self.steps)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize the entire trace to a plain dictionary."""
        return {
            "steps": [s.to_dict() for s in self.steps],
            "start_time": self.start_time,
            "end_time": self.end_time,
            "total_tokens": self.total_tokens,
            "total_cost": self.total_cost,
            "max_depth": self.max_depth,
            "metadata": self.metadata,
        }


@dataclass
class BudgetConfig:
    """Immutable budget configuration for an execution.

    Attributes:
        max_steps: Maximum number of execution steps (None = unlimited).
        max_tokens: Maximum total tokens (None = unlimited).
        max_cost: Maximum cost in USD (None = unlimited).
        max_time_seconds: Maximum wall-clock time in seconds (None = unlimited).
        max_recursion_depth: Maximum recursion depth (None = unlimited).
    """

    max_steps: Optional[int] = None
    max_tokens: Optional[int] = None
    max_cost: Optional[float] = None
    max_time_seconds: Optional[float] = None
    max_recursion_depth: Optional[int] = None


@dataclass
class BudgetState:
    """Mutable budget consumption state.

    Attributes:
        steps: Number of steps consumed.
        input_tokens: Number of input tokens consumed.
        output_tokens: Number of output tokens consumed.
        cost: Accumulated cost in USD.
        elapsed_seconds: Accumulated wall-clock time in seconds.
        recursion_depth: Current recursion depth.
    """

    steps: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cost: float = 0.0
    elapsed_seconds: float = 0.0
    recursion_depth: int = 0

    @property
    def total_tokens(self) -> int:
        """Total tokens consumed (input + output)."""
        return self.input_tokens + self.output_tokens

    def is_within(self, config: BudgetConfig) -> bool:
        """Check whether the current state is within the given budget limits."""
        if config.max_steps is not None and self.steps >= config.max_steps:
            return False
        if config.max_tokens is not None and self.total_tokens >= config.max_tokens:
            return False
        if config.max_cost is not None and self.cost >= config.max_cost:
            return False
        if config.max_time_seconds is not None and self.elapsed_seconds >= config.max_time_seconds:
            return False
        if config.max_recursion_depth is not None and self.recursion_depth >= config.max_recursion_depth:
            return False
        return True

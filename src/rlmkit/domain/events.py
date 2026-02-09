"""Domain events: signals emitted during RLM execution.

Events are immutable records of something that happened. They carry
no external dependencies and can be consumed by any layer.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional


@dataclass(frozen=True)
class StepCompleted:
    """Emitted when a single execution step finishes.

    Attributes:
        step_index: Zero-based step number.
        action_type: The kind of action taken (inspect, subcall, final, error).
        duration: Wall-clock seconds for this step.
        tokens_used: Tokens consumed in this step.
        recursion_depth: Depth at which this step executed.
        model: Model identifier used, if any.
    """

    step_index: int
    action_type: str
    duration: float = 0.0
    tokens_used: int = 0
    recursion_depth: int = 0
    model: Optional[str] = None


@dataclass(frozen=True)
class BudgetExceeded:
    """Emitted when a budget limit is hit.

    Attributes:
        budget_type: Which budget was exceeded (steps, tokens, cost, time, depth).
        limit: The configured limit value.
        current: The current consumption value that triggered the event.
    """

    budget_type: str
    limit: float
    current: float


@dataclass(frozen=True)
class ExecutionStarted:
    """Emitted when an RLM execution begins.

    Attributes:
        query: The user question.
        mode: Execution mode selected.
        model: Primary model identifier.
        content_length: Length of the content in characters.
    """

    query: str
    mode: str
    model: Optional[str] = None
    content_length: int = 0


@dataclass(frozen=True)
class ExecutionCompleted:
    """Emitted when an RLM execution finishes.

    Attributes:
        success: Whether execution completed without error.
        total_steps: Number of steps taken.
        total_tokens: Total tokens consumed.
        total_cost: Total cost in USD.
        duration: Total wall-clock seconds.
        error: Error message if execution failed.
    """

    success: bool
    total_steps: int = 0
    total_tokens: int = 0
    total_cost: float = 0.0
    duration: float = 0.0
    error: Optional[str] = None

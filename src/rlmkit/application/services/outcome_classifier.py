"""Outcome classifier — single source of truth for execution failure categorization.

Classifies an execution result into one of five outcome categories
based on the full run result (success flag, error string, and answer content).
No other module should re-parse error strings for failure categorization.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class OutcomeCategory(Enum):
    """Categorization of an execution outcome."""

    SUCCESS = "success"
    TIMEOUT = "timeout"
    BUDGET_EXHAUSTED = "budget_exhausted"
    CONTEXT_OVERFLOW = "context_overflow"
    GENERAL_ERROR = "general_error"


# Warning prefix used by RLM/Direct/RAG use cases for degraded outcomes
_DEGRADED_WARNING_PREFIX = "\u26a0\ufe0f"


@dataclass(frozen=True)
class ExecutionOutcome:
    """Result of classifying an execution."""

    category: OutcomeCategory
    is_usable: bool


def classify_execution_outcome(
    success: bool,
    error: str | None,
    answer: str,
) -> ExecutionOutcome:
    """Classify from the full result, not just the error string.

    Handles three states:
    1. ``success=False`` — hard failure, classify from error string keywords.
    2. ``success=True`` + answer starts with warning prefix — degraded,
       classify from answer content.
    3. ``success=True`` + normal answer — real success.
    """
    if not success:
        return _classify_hard_failure(error)

    # success=True but answer may be a degraded warning
    if answer.startswith(_DEGRADED_WARNING_PREFIX):
        return _classify_degraded(answer)

    return ExecutionOutcome(OutcomeCategory.SUCCESS, is_usable=True)


def _classify_hard_failure(error: str | None) -> ExecutionOutcome:
    """Classify a hard failure (success=False) from the error string."""
    error_lower = (error or "").lower()
    if "timeout" in error_lower or "timed out" in error_lower:
        return ExecutionOutcome(OutcomeCategory.TIMEOUT, is_usable=False)
    if "context window" in error_lower or "context_length" in error_lower:
        return ExecutionOutcome(OutcomeCategory.CONTEXT_OVERFLOW, is_usable=False)
    if "budget" in error_lower:
        return ExecutionOutcome(OutcomeCategory.BUDGET_EXHAUSTED, is_usable=False)
    return ExecutionOutcome(OutcomeCategory.GENERAL_ERROR, is_usable=False)


def _classify_degraded(answer: str) -> ExecutionOutcome:
    """Classify a degraded outcome (success=True, answer starts with warning prefix)."""
    answer_lower = answer.lower()
    if "timed out" in answer_lower:
        return ExecutionOutcome(OutcomeCategory.TIMEOUT, is_usable=False)
    if "context window" in answer_lower:
        return ExecutionOutcome(OutcomeCategory.CONTEXT_OVERFLOW, is_usable=False)
    if "budget" in answer_lower or "step" in answer_lower:
        return ExecutionOutcome(OutcomeCategory.BUDGET_EXHAUSTED, is_usable=False)
    return ExecutionOutcome(OutcomeCategory.SUCCESS, is_usable=True)

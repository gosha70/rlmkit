"""AC-7 — PREFILL_TIMEOUT classifier refinement.

A timeout failure where >=2 steps are prefill-dominated promotes to
PREFILL_TIMEOUT; otherwise it stays TIMEOUT. Classifier keyword
accepts the optional `trace` arg; when None, back-compat behavior.
"""

from __future__ import annotations

from rlmkit.application.services.outcome_classifier import (
    OutcomeCategory,
    classify_execution_outcome,
)
from rlmkit.domain.entities import ExecutionTrace, TraceStep


def _prefill_dominated_step(*, ttft_ms: int, duration_s: float) -> TraceStep:
    return TraceStep(
        index=0,
        action_type="inspect",
        ttft_ms=ttft_ms,
        duration=duration_s,
    )


class TestPrefillTimeoutPromotion:
    def test_two_prefill_dominated_steps_promotes_to_prefill_timeout(self):
        # Each step: ttft=800 ms of 1000 ms total → 0.8 ratio (> 0.7).
        trace = ExecutionTrace(
            steps=[
                _prefill_dominated_step(ttft_ms=800, duration_s=1.0),
                _prefill_dominated_step(ttft_ms=800, duration_s=1.0),
            ]
        )
        outcome = classify_execution_outcome(
            success=False,
            error="LLM_TIMEOUT: request timed out after 30s",
            answer="",
            trace=trace,
        )
        assert outcome.category == OutcomeCategory.PREFILL_TIMEOUT
        assert outcome.is_usable is False

    def test_only_one_prefill_step_stays_timeout(self):
        trace = ExecutionTrace(
            steps=[
                _prefill_dominated_step(ttft_ms=800, duration_s=1.0),  # prefill
                _prefill_dominated_step(ttft_ms=200, duration_s=2.0),  # decode-heavy
            ]
        )
        outcome = classify_execution_outcome(
            success=False,
            error="timed out",
            answer="",
            trace=trace,
        )
        assert outcome.category == OutcomeCategory.TIMEOUT

    def test_trace_with_no_ttft_stays_timeout(self):
        trace = ExecutionTrace(steps=[TraceStep(index=0, action_type="inspect")])
        outcome = classify_execution_outcome(success=False, error="timeout", answer="", trace=trace)
        assert outcome.category == OutcomeCategory.TIMEOUT

    def test_none_trace_back_compat(self):
        """Back-compat: callers that can't materialize a trace pass None."""
        outcome = classify_execution_outcome(success=False, error="timeout", answer="", trace=None)
        assert outcome.category == OutcomeCategory.TIMEOUT

    def test_prefill_threshold_boundary(self):
        # Exactly 0.7 does NOT trigger (spec says "> 0.7", not ">=").
        trace = ExecutionTrace(
            steps=[
                _prefill_dominated_step(ttft_ms=700, duration_s=1.0),
                _prefill_dominated_step(ttft_ms=700, duration_s=1.0),
            ]
        )
        outcome = classify_execution_outcome(success=False, error="timeout", answer="", trace=trace)
        assert outcome.category == OutcomeCategory.TIMEOUT

    def test_non_timeout_failure_ignores_trace(self):
        """Prefill refinement only applies to timeout failures."""
        trace = ExecutionTrace(
            steps=[
                _prefill_dominated_step(ttft_ms=800, duration_s=1.0),
                _prefill_dominated_step(ttft_ms=800, duration_s=1.0),
            ]
        )
        outcome = classify_execution_outcome(
            success=False,
            error="budget exceeded",
            answer="",
            trace=trace,
        )
        assert outcome.category == OutcomeCategory.BUDGET_EXHAUSTED

    def test_success_with_trace_returns_success(self):
        """A successful run is still SUCCESS regardless of trace shape."""
        trace = ExecutionTrace(steps=[_prefill_dominated_step(ttft_ms=800, duration_s=1.0)])
        outcome = classify_execution_outcome(success=True, error=None, answer="ok", trace=trace)
        assert outcome.category == OutcomeCategory.SUCCESS

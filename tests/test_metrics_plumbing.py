"""AC-25 — `_RunPoint.outcome_category` plumbing through metrics route.

- `_point_from_telemetry` forwards `RunSummary.outcome_category` verbatim.
- `_point_from_message` always sets `None` (legacy path).
- `_resolve_outcome` prefers persisted; falls back to re-classification
  when `None` or when the persisted value isn't a known category.
"""

from __future__ import annotations

from datetime import datetime, timezone

from rlmstudio.application.services.outcome_classifier import OutcomeCategory
from rlmstudio.server.routes.metrics import (
    _point_from_message,
    _point_from_telemetry,
    _resolve_outcome,
    _RunPoint,
)
from rlmstudio.telemetry.store import RunSummary


def _make_run_summary(outcome_category: str | None) -> RunSummary:
    return RunSummary(
        id="r1",
        created_at=1000.0,
        mode="rlm",
        provider="openai",
        model="gpt-4o",
        query="q",
        total_tokens=100,
        total_cost=0.1,
        elapsed_seconds=1.5,
        success=True,
        error=None,
        session_id="s1",
        chat_provider_id=None,
        chat_provider_name=None,
        steps_count=3,
        answer_length=10,
        answer="answer",
        outcome_category=outcome_category,
    )


class TestPointFromTelemetry:
    def test_forwards_outcome_category_verbatim(self):
        run = _make_run_summary(outcome_category="prefill_timeout")
        point = _point_from_telemetry(run)
        assert point.outcome_category == "prefill_timeout"

    def test_forwards_none(self):
        run = _make_run_summary(outcome_category=None)
        point = _point_from_telemetry(run)
        assert point.outcome_category is None


class TestPointFromMessage:
    def test_always_none(self):
        msg = {
            "role": "assistant",
            "content": "ok",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "mode_used": "rlm",
            "provider": "openai",
            "execution_id": "e1",
        }
        metrics = {"total_tokens": 100, "cost_usd": 0.1, "elapsed_seconds": 1.0}
        point = _point_from_message(msg, metrics)
        assert point is not None
        assert point.outcome_category is None


class TestResolveOutcome:
    def _point(self, **overrides) -> _RunPoint:
        base = {
            "execution_id": "e1",
            "timestamp": datetime.now(timezone.utc),
            "tokens": 0,
            "cost": 0.0,
            "latency": 0.0,
            "mode": "rlm",
            "provider": "openai",
            "chat_provider_name": None,
            "success": False,
            "error": "timed out",
            "answer": "",
            "outcome_category": None,
        }
        base.update(overrides)
        return _RunPoint(**base)

    def test_prefers_persisted_category(self):
        """A persisted 'prefill_timeout' is returned even when the
        error string alone would resolve to 'timeout' (no trace path)."""
        point = self._point(outcome_category="prefill_timeout")
        outcome = _resolve_outcome(point)
        assert outcome.category == OutcomeCategory.PREFILL_TIMEOUT

    def test_falls_back_to_reclassify_when_none(self):
        """Legacy rows have no persisted category — re-derive from scalars."""
        point = self._point(outcome_category=None, error="timed out")
        outcome = _resolve_outcome(point)
        assert outcome.category == OutcomeCategory.TIMEOUT

    def test_legacy_timeout_never_promoted_to_prefill_timeout(self):
        """AC-15: no retroactive promotion — legacy rows stayed TIMEOUT."""
        point = self._point(outcome_category=None, error="timeout")
        outcome = _resolve_outcome(point)
        assert outcome.category == OutcomeCategory.TIMEOUT

    def test_unknown_persisted_category_falls_back(self):
        """Defensive: a garbage string in the column falls back rather
        than raising."""
        point = self._point(outcome_category="bogus_value", error="timeout")
        outcome = _resolve_outcome(point)
        assert outcome.category == OutcomeCategory.TIMEOUT

    def test_success_point_returns_success(self):
        point = self._point(success=True, error=None, answer="ok")
        outcome = _resolve_outcome(point)
        assert outcome.category == OutcomeCategory.SUCCESS
        assert outcome.is_usable is True

    def test_persisted_success_is_usable(self):
        point = self._point(success=True, error=None, answer="ok", outcome_category="success")
        outcome = _resolve_outcome(point)
        assert outcome.is_usable is True

    def test_persisted_failure_is_not_usable(self):
        point = self._point(outcome_category="prefill_timeout")
        outcome = _resolve_outcome(point)
        assert outcome.is_usable is False

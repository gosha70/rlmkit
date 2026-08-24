"""AC-21 — `TelemetryStore.record_run` accepts `outcome_category`.

The store is a dumb sink: it persists the string verbatim and does
not validate. The caller-level contract (that only valid
OutcomeCategory.value strings are passed) is checked separately by
the chat route's integration tests.
"""

from __future__ import annotations

from pathlib import Path

from rlmstudio.application.services.outcome_classifier import OutcomeCategory
from rlmstudio.telemetry.store import TelemetryStore


def _make_store(tmp_path: Path) -> TelemetryStore:
    return TelemetryStore(db_path=tmp_path / "outcome.db")


class TestRecordRunOutcomeCategory:
    def test_accepts_optional_outcome_category(self, tmp_path: Path) -> None:
        store = _make_store(tmp_path)
        try:
            store.record_run(
                run_id="r-valid",
                created_at=1000.0,
                mode="rlm",
                outcome_category=OutcomeCategory.PREFILL_TIMEOUT.value,
            )
            runs = store.list_runs(limit=10)
            row = next(r for r in runs if r.id == "r-valid")
            assert row.outcome_category == "prefill_timeout"
        finally:
            store.close()

    def test_none_is_permitted(self, tmp_path: Path) -> None:
        store = _make_store(tmp_path)
        try:
            store.record_run(
                run_id="r-none",
                created_at=1000.0,
                mode="rlm",
                outcome_category=None,
            )
            runs = store.list_runs(limit=10)
            row = next(r for r in runs if r.id == "r-none")
            assert row.outcome_category is None
        finally:
            store.close()

    def test_store_does_not_validate_category(self, tmp_path: Path) -> None:
        """Dumb sink — unknown values persist verbatim; validation is
        the caller's responsibility. The dashboard fallback path
        re-derives when it can't parse the persisted string."""
        store = _make_store(tmp_path)
        try:
            store.record_run(
                run_id="r-weird",
                created_at=1000.0,
                mode="rlm",
                outcome_category="something_unknown",
            )
            runs = store.list_runs(limit=10)
            row = next(r for r in runs if r.id == "r-weird")
            assert row.outcome_category == "something_unknown"
        finally:
            store.close()

    def test_every_outcome_category_value_roundtrips(self, tmp_path: Path) -> None:
        store = _make_store(tmp_path)
        try:
            for i, cat in enumerate(OutcomeCategory):
                store.record_run(
                    run_id=f"r-cat-{i}",
                    created_at=1000.0 + i,
                    mode="rlm",
                    outcome_category=cat.value,
                )
            runs = {r.id: r for r in store.list_runs(limit=100)}
            for i, cat in enumerate(OutcomeCategory):
                assert runs[f"r-cat-{i}"].outcome_category == cat.value
        finally:
            store.close()

"""AC-27 — end-to-end PREFILL_TIMEOUT guard.

Simulates a prefill-dominated RLM run whose raw trace writer emits the
four new telemetry keys and whose terminal error is a timeout. Verifies
that the chat-route record_run call persists
``runs.outcome_category = "prefill_timeout"`` and that the failures
endpoint returns a ``prefill_timeout`` bucket.

This is the guard against the v1.4 regression class where the raw-DTO
translator and the classifier's `from_dict` would disagree on keys.
"""

from __future__ import annotations

from collections.abc import Generator

import pytest
from fastapi.testclient import TestClient

from rlmstudio.application.dto import RunResultDTO
from rlmstudio.application.sandbox_vars import (
    TRACE_KEY_ELAPSED_SECONDS,
    TRACE_KEY_INPUT_TOKENS,
    TRACE_KEY_MODEL,
    TRACE_KEY_OUTPUT_TOKENS,
    TRACE_KEY_ROLE,
    TRACE_KEY_STEP,
)
from rlmstudio.server.app import create_app
from rlmstudio.server.dependencies import (
    ExecutionRecord,
    SessionRecord,
    get_state,
    reset_state,
)
from rlmstudio.server.routes.chat import _record_telemetry

pytestmark = pytest.mark.e2e


@pytest.fixture(autouse=True)
def _clean_state() -> Generator[None, None, None]:
    reset_state()
    yield
    reset_state()


@pytest.fixture
def client() -> TestClient:
    return TestClient(create_app())


def _prefill_step(step: int, ttft_ms: int, duration_s: float) -> dict:
    return {
        TRACE_KEY_STEP: step,
        TRACE_KEY_ROLE: "assistant",
        TRACE_KEY_INPUT_TOKENS: 500,
        TRACE_KEY_OUTPUT_TOKENS: 20,
        TRACE_KEY_ELAPSED_SECONDS: duration_s,
        TRACE_KEY_MODEL: "gpt-4o",
        "ttft_ms": ttft_ms,
        "decode_ms": max(0, int(duration_s * 1000) - ttft_ms),
        "cached_tokens": 0,
        "cache_write_tokens": 0,
    }


class TestPrefillTimeoutEndToEnd:
    def test_prefill_dominated_rlm_run_classifies_as_prefill_timeout(
        self, client: TestClient
    ) -> None:
        from datetime import datetime, timezone

        state = get_state()
        # Seed a session so the failures endpoint can find the run.
        session_id = "sess-pft"
        state.sessions[session_id] = SessionRecord(
            id=session_id,
            name="Test",
            created_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc),
        )

        execution = ExecutionRecord(
            execution_id="exec-pft",
            session_id=session_id,
            query="q",
            mode="rlm",
            status="complete",
            started_at=datetime.now(timezone.utc),
        )
        state.executions[execution.execution_id] = execution

        # Build a prefill-dominated trace: 3 steps, each with 80% of
        # duration spent on TTFT. Terminal error is a timeout.
        result = RunResultDTO(
            answer="",
            mode_used="rlm",
            success=False,
            error="LLM_TIMEOUT: request timed out after 30s",
            steps=3,
            input_tokens=1500,
            output_tokens=60,
            total_cost=0.0,
            elapsed_time=3.0,
            trace=[
                _prefill_step(1, ttft_ms=800, duration_s=1.0),
                _prefill_step(2, ttft_ms=800, duration_s=1.0),
                _prefill_step(3, ttft_ms=800, duration_s=1.0),
            ],
        )

        _record_telemetry(
            state=state,
            execution=execution,
            result=result,
            provider="openai",
            model="gpt-4o",
        )

        # Read back directly from the store to confirm the column.
        runs = state.telemetry.list_runs(session_id=session_id, limit=10)
        persisted = next(r for r in runs if r.id == "exec-pft")
        assert persisted.outcome_category == "prefill_timeout"

        # Failures endpoint surfaces the prefill_timeout bucket.
        resp = client.get(f"/api/metrics/failures/{session_id}")
        assert resp.status_code == 200
        categories = {bucket["category"]: bucket for bucket in resp.json()["by_category"]}
        assert "prefill_timeout" in categories
        assert categories["prefill_timeout"]["count"] == 1


class TestJudgeUnchanged:
    """AC-28 — spec explicitly scopes judge out. No structural changes."""

    def test_judge_not_touched_by_phase_4(self) -> None:
        """server/judge.py must not grow new record_run / outcome
        persistence surface. This guard is coarse — it checks that
        `record_run` is not called inside judge.py at all."""
        import pathlib

        judge_src = (
            pathlib.Path(__file__).resolve().parent.parent.parent
            / "src"
            / "rlmstudio"
            / "server"
            / "judge.py"
        )
        text = judge_src.read_text()
        assert "record_run" not in text, (
            "server/judge.py must not call record_run — spec v1.7 AC-28 "
            "explicitly scopes judge-side persistence out of this spec. "
            "If a new judge-telemetry row is needed, it ships under a "
            "follow-up spec."
        )

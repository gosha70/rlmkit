"""AC-9 — `GET /api/traces/{id}` exposes the six new step fields.

For traces recorded after Phase 3, the fields carry actual values.
For legacy traces without the fields, they appear with defaults
(0 / None) — additive-only wire shape.
"""

from __future__ import annotations

from collections.abc import Generator

import pytest
from fastapi.testclient import TestClient

from rlmkit.server.app import create_app
from rlmkit.server.dependencies import get_state, reset_state

pytestmark = pytest.mark.e2e


@pytest.fixture(autouse=True)
def _clean_state() -> Generator[None, None, None]:
    reset_state()
    yield
    reset_state()


@pytest.fixture
def client() -> TestClient:
    return TestClient(create_app())


_EXPECTED_FIELDS = {
    "prompt_tokens",
    "completion_tokens",
    "ttft_ms",
    "decode_ms",
    "cached_tokens",
    "cache_write_tokens",
}


def _seed_execution(
    *,
    execution_id: str,
    steps: list[dict],
) -> None:
    """Inject an execution directly into AppState so the e2e trace
    endpoint has real data to render, without needing a live LLM."""
    from rlmkit.server.dependencies import ExecutionRecord

    state = get_state()
    state.executions[execution_id] = ExecutionRecord(
        execution_id=execution_id,
        session_id="sess-1",
        query="q",
        mode="direct",
        status="complete",
        result={
            "answer": "ok",
            "success": True,
            "input_tokens": 5,
            "output_tokens": 2,
            "total_cost": 0.0,
            "total_tokens": 7,
        },
        steps=steps,
    )


class TestTraceResponseShape:
    def test_new_trace_includes_six_new_step_fields(self, client: TestClient) -> None:
        """A trace recorded after Phase 3 carries the six new fields
        with populated values."""
        _seed_execution(
            execution_id="new-exec-1",
            steps=[
                {
                    "role": "assistant",
                    "content": "ok",
                    "input_tokens": 10,
                    "output_tokens": 4,
                    "elapsed_seconds": 0.25,
                    "model": "gpt-4o",
                    "ttft_ms": 120,
                    "decode_ms": 45,
                    "cached_tokens": 6,
                    "cache_write_tokens": 1,
                }
            ],
        )

        resp = client.get("/api/traces/new-exec-1")
        assert resp.status_code == 200
        step = resp.json()["steps"][0]
        missing = _EXPECTED_FIELDS - set(step.keys())
        assert not missing, f"missing fields: {missing}"
        assert step["ttft_ms"] == 120
        assert step["decode_ms"] == 45
        assert step["cached_tokens"] == 6
        assert step["cache_write_tokens"] == 1

    def test_telemetry_persisted_trace_surfaces_six_new_fields(self, client: TestClient) -> None:
        """Regression guard for P1 review finding on Phase 3.

        Write through ``record_run`` + ``record_step`` and read back via
        ``GET /api/traces/{id}``. Exercises the telemetry-store read
        path (``_trace_from_telemetry``), not the in-memory
        ``ExecutionRecord.steps`` path that the other tests use. Before
        the fix this returned zeros for every new field because
        ``record_step`` dropped them.
        """
        from rlmkit.server.dependencies import get_state

        state = get_state()
        # Remove any in-memory execution for this id so the traces route
        # falls through to the telemetry-store path.
        exec_id = "persisted-exec-1"
        state.executions.pop(exec_id, None)

        run_id = state.telemetry.record_run(
            run_id=exec_id,
            created_at=1_000.0,
            mode="direct",
            provider="openai",
            model="gpt-4o",
            query="q",
            answer="ok",
            input_tokens=200,
            output_tokens=40,
            total_tokens=240,
            success=True,
            steps_count=1,
        )
        state.telemetry.record_step(
            run_id=run_id,
            step_index=0,
            action_type="final",
            code=None,
            output="ok",
            input_tokens=200,
            output_tokens=40,
            duration=0.3,
            model="gpt-4o",
            prompt_tokens=200,
            completion_tokens=40,
            ttft_ms=180,
            decode_ms=55,
            cached_tokens=120,
            cache_write_tokens=3,
        )

        resp = client.get(f"/api/traces/{exec_id}")
        assert resp.status_code == 200
        step = resp.json()["steps"][0]
        assert step["ttft_ms"] == 180
        assert step["decode_ms"] == 55
        assert step["cached_tokens"] == 120
        assert step["cache_write_tokens"] == 3
        assert step["prompt_tokens"] == 200
        assert step["completion_tokens"] == 40

    def test_legacy_step_dict_surfaces_defaults(self, client: TestClient) -> None:
        """A legacy trace (no new keys) materializes the six new fields
        with default values — additive wire shape, no breakage."""
        _seed_execution(
            execution_id="legacy-exec-1",
            steps=[
                {
                    "role": "assistant",
                    "content": "ok",
                    "input_tokens": 5,
                    "output_tokens": 2,
                    "elapsed_seconds": 0.1,
                    "model": "legacy-model",
                }
            ],
        )

        resp = client.get("/api/traces/legacy-exec-1")
        assert resp.status_code == 200
        step = resp.json()["steps"][0]
        assert step["ttft_ms"] is None
        assert step["decode_ms"] == 0
        assert step["cached_tokens"] == 0
        assert step["cache_write_tokens"] == 0
        # Legacy tokens feed prompt/completion for back-compat.
        assert step["prompt_tokens"] == 5
        assert step["completion_tokens"] == 2

"""E2E tests for GET /api/replays/{execution_id}.

Covers the route's three pinned cases (NEXT.md §3f test matrix):

- telemetry-backed execution id round-trips
- in-memory execution id round-trips *after canonicalization*
  (the in-memory branch stores raw "assistant"/"execution"
  roles; the shared load_canonical_trace helper normalizes via
  _canonical_action_type before the converter runs)
- unknown execution id → 404 with the standard error envelope

The converter's internal behavior is covered in depth by
``tests/test_trace_to_replay.py``; this file exercises the
integration from HTTP to route to loader to service.
"""

from __future__ import annotations

import time
import uuid

import pytest
from fastapi.testclient import TestClient

from rlmkit.server.dependencies import ExecutionRecord, get_state

pytestmark = [pytest.mark.e2e]


class TestReplayEndpoint:
    """GET /api/replays/{execution_id}."""

    def test_unknown_execution_id_returns_404(self, client: TestClient) -> None:
        resp = client.get(f"/api/replays/{uuid.uuid4().hex}")
        assert resp.status_code == 404
        # Standard error envelope from server/app.py exception handler.
        body = resp.json()
        assert "error" in body
        assert body["error"]["code"] == "NOT_FOUND"

    def test_telemetry_backed_execution_id_converts(self, client: TestClient) -> None:
        """Historical execution persisted in telemetry — not in memory."""
        state = get_state()
        run_id = uuid.uuid4().hex
        state.telemetry.record_run(
            run_id=run_id,
            created_at=time.time(),
            mode="rlm",
            query="What is the answer?",
            answer="42",
            success=True,
            elapsed_seconds=2.5,
            steps_count=3,
            total_cost=0.0042,
            input_tokens=500,
            output_tokens=120,
        )
        # Canonical action_types stored directly on the steps (the save
        # path already normalizes; see chat.py _canonical_action_type).
        state.telemetry.record_step(
            run_id=run_id,
            step_index=0,
            action_type="inspect",
            code='hits = grep(prompt, pattern="answer")',
            output=None,
            input_tokens=200,
            output_tokens=40,
            duration=0.8,
        )
        state.telemetry.record_step(
            run_id=run_id,
            step_index=1,
            action_type="subcall",
            code=None,
            output="3 hits",
            duration=0.05,
        )
        state.telemetry.record_step(
            run_id=run_id,
            step_index=2,
            action_type="final",
            code=None,
            output="42",
        )

        resp = client.get(f"/api/replays/{run_id}")
        assert resp.status_code == 200
        data = resp.json()
        assert data["metadata"]["source"] == "trace"
        assert data["metadata"]["executionId"] == run_id
        assert data["metadata"]["convertorVersion"] == 1
        assert data["metadata"].get("failed") in (None, False)
        kinds = [s["kind"] for s in data["steps"]]
        # question + code + result + answer (final folds into answer)
        assert kinds == ["question", "code", "result", "answer"]
        # Steps must carry no plan/decision kinds for trace-sourced.
        assert "plan" not in kinds
        assert "decision" not in kinds

    def test_in_memory_execution_id_converts_after_canonicalization(
        self, client: TestClient
    ) -> None:
        """In-memory steps store raw ``assistant``/``execution`` roles.

        The route must canonicalize via ``load_canonical_trace`` before
        calling the converter; otherwise the converter sees raw roles
        and emits no trace-derived steps. This test fails if the
        canonicalization pre-step from v2b step 2 ever regresses.
        """
        state = get_state()
        eid = uuid.uuid4().hex
        state.executions[eid] = ExecutionRecord(
            execution_id=eid,
            session_id="sess-1",
            query="Summarize the quarterly report",
            mode="rlm",
            status="complete",
            result={
                "answer": "Q3 revenue was $4.2B.",
                "success": True,
                "total_cost": 0.01,
                "input_tokens": 600,
                "output_tokens": 150,
            },
            steps=[
                # Raw roles — this is what the in-memory path stores.
                {
                    "role": "assistant",
                    "code": "section = peek(prompt, start=5000, length=500)",
                    "input_tokens": 300,
                    "output_tokens": 60,
                    "elapsed_seconds": 0.9,
                },
                {
                    "role": "execution",
                    "content": "Q3 revenue of $4.2B...",
                    "elapsed_seconds": 0.05,
                },
                # Last step with success=True → canonicalizes to "final".
                {
                    "role": "assistant",
                    "code": "FINAL_VAR = 'Q3 revenue was $4.2B.'",
                    "input_tokens": 250,
                    "output_tokens": 40,
                    "elapsed_seconds": 0.7,
                },
            ],
        )

        resp = client.get(f"/api/replays/{eid}")
        assert resp.status_code == 200
        data = resp.json()
        kinds = [s["kind"] for s in data["steps"]]
        # Without canonicalization, the converter would see unknown
        # action_types and emit zero trace-derived steps. With it,
        # the first two raw steps pair into code+result and the third
        # (last+success) becomes `final` (folded into answer).
        assert "code" in kinds
        assert "result" in kinds
        assert kinds[0] == "question"
        assert kinds[-1] == "answer"
        assert sum(1 for k in kinds if k == "answer") == 1

    def test_failed_run_folds_error_into_answer_with_metadata_failed(
        self, client: TestClient
    ) -> None:
        state = get_state()
        run_id = uuid.uuid4().hex
        state.telemetry.record_run(
            run_id=run_id,
            created_at=time.time(),
            mode="rlm",
            query="Break the sandbox",
            answer="",
            success=False,
            error="Provider timeout after 30s",
            elapsed_seconds=30.0,
            steps_count=1,
        )
        state.telemetry.record_step(
            run_id=run_id,
            step_index=0,
            action_type="inspect",
            code="x = 1",
            output=None,
        )

        resp = client.get(f"/api/replays/{run_id}")
        assert resp.status_code == 200
        data = resp.json()
        assert data["metadata"]["failed"] is True
        # Exactly one answer step, summary surfaces the error.
        answer_steps = [s for s in data["steps"] if s["kind"] == "answer"]
        assert len(answer_steps) == 1
        assert "timeout" in answer_steps[0]["summary"].lower()

    def test_response_is_bare_learn_replay_not_wrapped(self, client: TestClient) -> None:
        """Single-resource endpoint returns the resource directly.

        Mirrors GET /api/traces/{id} and GET /api/docs/{slug}; wrappers
        like {replay: ...} are reserved for collection endpoints.
        """
        state = get_state()
        run_id = uuid.uuid4().hex
        state.telemetry.record_run(
            run_id=run_id,
            created_at=time.time(),
            mode="direct",
            query="Hi",
            answer="Hello",
            success=True,
        )
        data = client.get(f"/api/replays/{run_id}").json()
        # Expected top-level keys are LearnReplay fields, not a wrapper.
        assert set(data.keys()) >= {"id", "title", "description", "steps", "metadata"}
        assert "replay" not in data

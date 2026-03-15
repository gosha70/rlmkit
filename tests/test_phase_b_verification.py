"""Phase B verification tests: polling metrics, session persistence, dashboard data.

These tests verify the fixes for Bugs #2, #4, and #5 from the outstanding bugs handoff.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from unittest.mock import patch

import pytest
from starlette.testclient import TestClient

from rlmkit.server.app import app
from rlmkit.server.dependencies import AppState, ExecutionRecord, reset_state


@pytest.fixture(autouse=True)
def _clean_state() -> None:
    reset_state()


@pytest.fixture()
def client() -> TestClient:
    return TestClient(app)


# ---------------------------------------------------------------------------
# Bug #2: Trace result exposes input/output tokens for polling path
# ---------------------------------------------------------------------------


class TestTraceTokenMetrics:
    def _create_execution_with_result(self, state: AppState) -> str:
        """Create a completed execution with token metrics."""
        session = state.get_or_create_session()
        exec_id = "test-exec-001"
        now = datetime.now(timezone.utc)
        execution = ExecutionRecord(
            execution_id=exec_id,
            session_id=session.id,
            query="test query",
            mode="direct",
            status="complete",
            started_at=now,
            completed_at=now,
            result={
                "answer": "test answer",
                "success": True,
                "input_tokens": 150,
                "output_tokens": 75,
                "total_tokens": 225,
                "total_cost": 0.0042,
                "elapsed_time": 1.5,
                "steps_count": 0,
            },
            steps=[
                {
                    "step": 0,
                    "role": "assistant",
                    "content": "test answer",
                    "input_tokens": 150,
                    "output_tokens": 75,
                    "elapsed_seconds": 1.5,
                }
            ],
        )
        state.executions[exec_id] = execution
        return exec_id

    def test_trace_result_includes_token_breakdown(self, client: TestClient) -> None:
        """GET /api/traces/{id} includes input_tokens, output_tokens, total_cost in result."""
        from rlmkit.server.dependencies import get_state

        state = get_state()
        exec_id = self._create_execution_with_result(state)

        resp = client.get(f"/api/traces/{exec_id}")
        assert resp.status_code == 200
        data = resp.json()

        result = data["result"]
        assert result["input_tokens"] == 150
        assert result["output_tokens"] == 75
        assert result["total_cost"] == 0.0042

    def test_trace_budget_still_has_totals(self, client: TestClient) -> None:
        """Budget fields are still populated for backward compatibility."""
        from rlmkit.server.dependencies import get_state

        state = get_state()
        exec_id = self._create_execution_with_result(state)

        resp = client.get(f"/api/traces/{exec_id}")
        data = resp.json()

        budget = data["budget"]
        assert budget["tokens_used"] == 225
        assert budget["cost_used"] == 0.0042

    def test_trace_result_defaults_to_zero(self, client: TestClient) -> None:
        """Missing token data defaults to 0, not None."""
        from rlmkit.server.dependencies import get_state

        state = get_state()
        session = state.get_or_create_session()
        execution = ExecutionRecord(
            execution_id="test-exec-empty",
            session_id=session.id,
            query="test",
            mode="direct",
            status="complete",
            result={"answer": "ok", "success": True},
            steps=[],
        )
        state.executions["test-exec-empty"] = execution

        resp = client.get("/api/traces/test-exec-empty")
        result = resp.json()["result"]
        assert result["input_tokens"] == 0
        assert result["output_tokens"] == 0
        assert result["total_cost"] == 0.0


# ---------------------------------------------------------------------------
# Bug #4: Dashboard empty state shown incorrectly
# (Backend side: metrics endpoint works with populated sessions)
# ---------------------------------------------------------------------------


class TestDashboardMetrics:
    def test_metrics_aggregates_by_provider(self, client: TestClient) -> None:
        """Metrics endpoint populates by_provider from assistant messages."""
        from rlmkit.server.dependencies import get_state

        state = get_state()
        session = state.get_or_create_session()
        now = datetime.now(timezone.utc)

        # Add two assistant messages from different providers
        state.add_message(
            session.id,
            {
                "role": "assistant",
                "content": "answer 1",
                "mode_used": "direct",
                "provider": "openai",
                "chat_provider_name": "GPT-4o Config",
                "metrics": {
                    "total_tokens": 200,
                    "cost_usd": 0.003,
                    "elapsed_seconds": 1.0,
                },
                "timestamp": now.isoformat(),
            },
        )
        state.add_message(
            session.id,
            {
                "role": "assistant",
                "content": "answer 2",
                "mode_used": "direct",
                "provider": "anthropic",
                "chat_provider_name": "Claude Config",
                "metrics": {
                    "total_tokens": 300,
                    "cost_usd": 0.005,
                    "elapsed_seconds": 2.0,
                },
                "timestamp": now.isoformat(),
            },
        )

        resp = client.get(f"/api/metrics/{session.id}")
        assert resp.status_code == 200
        data = resp.json()

        assert data["summary"]["total_queries"] == 2
        assert data["summary"]["total_tokens"] == 500
        assert "openai" in data["by_provider"]
        assert "anthropic" in data["by_provider"]
        assert data["by_provider"]["openai"]["queries"] == 1
        assert data["by_provider"]["anthropic"]["queries"] == 1

    def test_metrics_aggregates_by_chat_provider(self, client: TestClient) -> None:
        """Metrics endpoint populates by_chat_provider from assistant messages."""
        from rlmkit.server.dependencies import get_state

        state = get_state()
        session = state.get_or_create_session()
        now = datetime.now(timezone.utc)

        state.add_message(
            session.id,
            {
                "role": "assistant",
                "content": "answer",
                "mode_used": "direct",
                "provider": "openai",
                "chat_provider_name": "My GPT Config",
                "metrics": {
                    "total_tokens": 100,
                    "cost_usd": 0.001,
                    "elapsed_seconds": 0.5,
                },
                "timestamp": now.isoformat(),
            },
        )

        resp = client.get(f"/api/metrics/{session.id}")
        data = resp.json()

        assert "My GPT Config" in data["by_chat_provider"]
        assert data["by_chat_provider"]["My GPT Config"]["queries"] == 1


# ---------------------------------------------------------------------------
# Bug #5: Session persistence survives restart
# ---------------------------------------------------------------------------


class TestSessionPersistence:
    def test_sessions_loaded_from_disk(self) -> None:
        """AppState._load_sessions() correctly restores sessions."""
        state = AppState(load_from_disk=False)
        session = state.get_or_create_session()
        now = datetime.now(timezone.utc)
        state.add_message(
            session.id,
            {
                "role": "user",
                "content": "Hello",
                "timestamp": now.isoformat(),
            },
        )
        state.add_message(
            session.id,
            {
                "role": "assistant",
                "content": "Hi there",
                "mode_used": "direct",
                "metrics": {"total_tokens": 50, "cost_usd": 0.001, "elapsed_seconds": 0.3},
                "timestamp": now.isoformat(),
            },
        )

        # Manually serialize and deserialize (simulates restart)
        serialized = [
            {
                "id": session.id,
                "name": session.name,
                "created_at": session.created_at.isoformat(),
                "updated_at": session.updated_at.isoformat(),
                "messages": session.messages,
                "conversations": session.conversations,
            }
        ]

        state2 = AppState(load_from_disk=False)
        assert len(state2.sessions) == 0

        # Simulate loading from disk
        with patch("rlmkit.server.dependencies._SESSIONS_FILE") as mock_file:
            mock_file.exists.return_value = True
            mock_file.read_text.return_value = json.dumps(serialized, default=str)
            state2._load_sessions()

        assert len(state2.sessions) == 1
        loaded = state2.sessions[session.id]
        assert len(loaded.messages) == 2
        assert loaded.messages[0]["content"] == "Hello"
        assert loaded.messages[1]["content"] == "Hi there"

    def test_rest_error_path_saves_sessions(self) -> None:
        """_run_execution error path calls save_sessions() so errors persist."""
        import inspect

        from rlmkit.server.routes import chat

        source = inspect.getsource(chat._run_execution)
        except_idx = source.rfind("except Exception")
        assert except_idx != -1
        after_except = source[except_idx:]
        assert "save_sessions()" in after_except, (
            "_run_execution error path must call save_sessions()"
        )

    def test_ws_error_path_saves_sessions(self) -> None:
        """_ws_execute error path calls save_sessions() so WS errors persist."""
        import inspect

        from rlmkit.server.routes import chat

        source = inspect.getsource(chat.websocket_chat)
        # Find the _ws_execute inner function's except block
        ws_exec_idx = source.find("async def _ws_execute")
        assert ws_exec_idx != -1, "websocket_chat should contain _ws_execute"
        ws_source = source[ws_exec_idx:]
        except_idx = ws_source.rfind("except Exception")
        assert except_idx != -1
        after_except = ws_source[except_idx:]
        assert "save_sessions()" in after_except, "_ws_execute error path must call save_sessions()"
        assert "add_message" in after_except, (
            "_ws_execute error path must add error message to session"
        )

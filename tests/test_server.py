"""Tests for the FastAPI server using TestClient with mocked use cases."""

from __future__ import annotations

import io
from datetime import datetime, timezone

import pytest
from fastapi.testclient import TestClient

from rlmkit.server.app import create_app
from rlmkit.server.dependencies import (
    ExecutionRecord,
    FileRecord,
    SessionRecord,
    get_state,
    reset_state,
)


@pytest.fixture(autouse=True)
def _clean_state():
    """Reset shared state before each test."""
    reset_state()
    yield
    reset_state()


@pytest.fixture
def client():
    app = create_app()
    return TestClient(app)


# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------


class TestHealthCheck:
    def test_health(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert data["version"] == "0.1.0"
        assert "uptime_seconds" in data


# ---------------------------------------------------------------------------
# Sessions
# ---------------------------------------------------------------------------


class TestSessions:
    def test_list_empty(self, client):
        resp = client.get("/api/sessions")
        assert resp.status_code == 200
        assert resp.json() == []

    def test_list_sessions(self, client):
        state = get_state()
        now = datetime.now(timezone.utc)
        state.sessions["s1"] = SessionRecord(
            id="s1",
            name="Session 1",
            created_at=now,
            updated_at=now,
            messages=[
                {"id": "m1", "role": "user", "content": "hello", "timestamp": now.isoformat()}
            ],
        )
        resp = client.get("/api/sessions")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data) == 1
        assert data[0]["id"] == "s1"
        assert data[0]["message_count"] == 1

    def test_get_session(self, client):
        state = get_state()
        now = datetime.now(timezone.utc)
        state.sessions["s1"] = SessionRecord(
            id="s1",
            name="Session 1",
            created_at=now,
            updated_at=now,
            messages=[
                {"id": "m1", "role": "user", "content": "hello", "timestamp": now.isoformat()}
            ],
        )
        resp = client.get("/api/sessions/s1")
        assert resp.status_code == 200
        data = resp.json()
        assert data["id"] == "s1"
        assert len(data["messages"]) == 1

    def test_get_session_not_found(self, client):
        resp = client.get("/api/sessions/nonexistent")
        assert resp.status_code == 404

    def test_delete_session(self, client):
        state = get_state()
        now = datetime.now(timezone.utc)
        state.sessions["s1"] = SessionRecord(
            id="s1",
            name="Session 1",
            created_at=now,
            updated_at=now,
        )
        resp = client.delete("/api/sessions/s1")
        assert resp.status_code == 204
        assert "s1" not in state.sessions

    def test_delete_session_not_found(self, client):
        resp = client.delete("/api/sessions/nonexistent")
        assert resp.status_code == 404

    def test_pagination(self, client):
        state = get_state()
        now = datetime.now(timezone.utc)
        for i in range(5):
            state.sessions[f"s{i}"] = SessionRecord(
                id=f"s{i}",
                name=f"Session {i}",
                created_at=now,
                updated_at=now,
            )
        resp = client.get("/api/sessions?limit=2&offset=0")
        assert resp.status_code == 200
        assert len(resp.json()) == 2

        resp = client.get("/api/sessions?limit=2&offset=4")
        assert resp.status_code == 200
        assert len(resp.json()) == 1


# ---------------------------------------------------------------------------
# File upload
# ---------------------------------------------------------------------------


class TestFileUpload:
    def test_upload_text_file(self, client):
        content = b"Hello, this is a test document with some text content."
        resp = client.post(
            "/api/files/upload",
            files={"file": ("test.txt", io.BytesIO(content), "text/plain")},
        )
        assert resp.status_code == 201
        data = resp.json()
        assert data["name"] == "test.txt"
        assert data["size_bytes"] == len(content)
        assert data["token_count"] > 0

    def test_upload_md_file(self, client):
        content = b"# Hello\n\nThis is markdown."
        resp = client.post(
            "/api/files/upload",
            files={"file": ("readme.md", io.BytesIO(content), "text/markdown")},
        )
        assert resp.status_code == 201
        assert resp.json()["name"] == "readme.md"

    def test_upload_unsupported_type(self, client):
        resp = client.post(
            "/api/files/upload",
            files={"file": ("image.png", io.BytesIO(b"fake"), "image/png")},
        )
        assert resp.status_code == 400

    def test_get_file(self, client):
        content = b"some text"
        resp = client.post(
            "/api/files/upload",
            files={"file": ("doc.txt", io.BytesIO(content), "text/plain")},
        )
        file_id = resp.json()["id"]

        resp = client.get(f"/api/files/{file_id}")
        assert resp.status_code == 200
        assert resp.json()["id"] == file_id

    def test_get_file_not_found(self, client):
        resp = client.get("/api/files/nonexistent")
        assert resp.status_code == 404


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


class TestMetrics:
    def test_metrics_not_found(self, client):
        resp = client.get("/api/metrics/nonexistent")
        assert resp.status_code == 404

    def test_metrics_empty_session(self, client):
        state = get_state()
        now = datetime.now(timezone.utc)
        state.sessions["s1"] = SessionRecord(
            id="s1",
            name="S1",
            created_at=now,
            updated_at=now,
        )
        resp = client.get("/api/metrics/s1")
        assert resp.status_code == 200
        data = resp.json()
        assert data["summary"]["total_queries"] == 0

    def test_metrics_with_messages(self, client):
        state = get_state()
        now = datetime.now(timezone.utc)
        state.sessions["s1"] = SessionRecord(
            id="s1",
            name="S1",
            created_at=now,
            updated_at=now,
            messages=[
                {"id": "m1", "role": "user", "content": "q", "timestamp": now.isoformat()},
                {
                    "id": "m2",
                    "role": "assistant",
                    "content": "a",
                    "mode_used": "rlm",
                    "timestamp": now.isoformat(),
                    "metrics": {
                        "input_tokens": 100,
                        "output_tokens": 50,
                        "total_tokens": 150,
                        "cost_usd": 0.015,
                        "elapsed_seconds": 2.5,
                        "steps": 3,
                    },
                },
            ],
        )
        resp = client.get("/api/metrics/s1")
        assert resp.status_code == 200
        data = resp.json()
        assert data["summary"]["total_queries"] == 1
        assert data["summary"]["total_tokens"] == 150
        assert "rlm" in data["by_mode"]


# ---------------------------------------------------------------------------
# Traces
# ---------------------------------------------------------------------------


class TestTraces:
    def test_trace_not_found(self, client):
        resp = client.get("/api/traces/nonexistent")
        assert resp.status_code == 404

    def test_trace_found(self, client):
        state = get_state()
        now = datetime.now(timezone.utc)
        state.executions["ex1"] = ExecutionRecord(
            execution_id="ex1",
            session_id="s1",
            query="test query",
            mode="rlm",
            status="complete",
            started_at=now,
            completed_at=now,
            result={"answer": "42", "success": True},
            steps=[
                {
                    "role": "assistant",
                    "content": "exploring",
                    "input_tokens": 10,
                    "output_tokens": 5,
                },
            ],
        )
        resp = client.get("/api/traces/ex1")
        assert resp.status_code == 200
        data = resp.json()
        assert data["execution_id"] == "ex1"
        assert data["result"]["answer"] == "42"
        assert len(data["steps"]) == 1


# ---------------------------------------------------------------------------
# Providers
# ---------------------------------------------------------------------------


class TestProviders:
    def test_list_providers(self, client):
        resp = client.get("/api/providers")
        assert resp.status_code == 200
        data = resp.json()
        assert isinstance(data, list)
        assert len(data) >= 3
        names = [p["name"] for p in data]
        assert "openai" in names
        assert "anthropic" in names
        assert "ollama" in names


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


class TestConfig:
    def test_get_config(self, client):
        resp = client.get("/api/config")
        assert resp.status_code == 200
        data = resp.json()
        assert "active_provider" in data
        assert "budget" in data
        assert "sandbox" in data
        assert "appearance" in data

    def test_update_config(self, client):
        """PUT /api/config updates budget/sandbox/appearance but not active_provider/model."""
        resp = client.put(
            "/api/config",
            json={"active_provider": "anthropic", "active_model": "claude-sonnet-4-5-20250929"},
        )
        assert resp.status_code == 200
        data = resp.json()
        # active_provider/model are only set via PUT /api/providers/{name}
        assert data["active_provider"] == "openai"
        assert data["active_model"] == "gpt-4o"

    def test_update_budget(self, client):
        resp = client.put(
            "/api/config",
            json={
                "budget": {
                    "max_steps": 32,
                    "max_tokens": 100000,
                    "max_cost_usd": 5.0,
                    "max_time_seconds": 60,
                    "max_recursion_depth": 10,
                }
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["budget"]["max_steps"] == 32

    def test_update_preserves_unset_fields(self, client):
        # Get initial config
        resp = client.get("/api/config")
        initial = resp.json()

        # Update only budget — active_provider should remain unchanged
        resp = client.put("/api/config", json={"budget": {"max_steps": 64}})
        data = resp.json()
        assert data["active_provider"] == initial["active_provider"]
        assert data["budget"]["max_steps"] == 64


# ---------------------------------------------------------------------------
# Chat (basic validation)
# ---------------------------------------------------------------------------


class TestChat:
    def test_chat_creates_session(self, client):
        resp = client.post(
            "/api/chat",
            json={"query": "What is this?", "content": "Hello world", "mode": "direct"},
        )
        # Will return 202 even though background task will fail (no real LLM)
        assert resp.status_code == 202
        data = resp.json()
        assert "execution_id" in data
        assert "session_id" in data
        assert data["status"] == "running"

    def test_chat_with_existing_session(self, client):
        state = get_state()
        now = datetime.now(timezone.utc)
        state.sessions["s1"] = SessionRecord(
            id="s1",
            name="S1",
            created_at=now,
            updated_at=now,
        )
        resp = client.post(
            "/api/chat",
            json={"query": "Follow up", "content": "text", "session_id": "s1"},
        )
        assert resp.status_code == 202
        assert resp.json()["session_id"] == "s1"

    def test_chat_with_file_id(self, client):
        state = get_state()
        now = datetime.now(timezone.utc)
        state.files["f1"] = FileRecord(
            id="f1",
            name="doc.txt",
            size_bytes=100,
            content_type="text/plain",
            text_content="Some document text",
            token_count=5,
            created_at=now,
        )
        resp = client.post(
            "/api/chat",
            json={"query": "Summarize", "file_id": "f1"},
        )
        assert resp.status_code == 202

    def test_chat_missing_file_id_returns_404(self, client):
        resp = client.post(
            "/api/chat",
            json={"query": "Summarize", "file_id": "nonexistent"},
        )
        assert resp.status_code == 404
        data = resp.json()
        assert data["error"]["code"] == "NOT_FOUND"
        assert "File not found" in data["error"]["message"]

    def test_chat_missing_content_and_file_id_returns_400(self, client):
        resp = client.post(
            "/api/chat",
            json={"query": "What?"},
        )
        assert resp.status_code == 400
        data = resp.json()
        assert data["error"]["code"] == "VALIDATION_ERROR"
        assert "content or file_id" in data["error"]["message"]

    def test_chat_rejects_invalid_mode(self, client):
        resp = client.post(
            "/api/chat",
            json={"query": "What?", "content": "text", "mode": "invalid_mode"},
        )
        assert resp.status_code == 422


# ---------------------------------------------------------------------------
# Error Response Format (CRITICAL-1)
# ---------------------------------------------------------------------------


class TestErrorResponseFormat:
    def test_404_error_format(self, client):
        resp = client.get("/api/sessions/nonexistent")
        assert resp.status_code == 404
        data = resp.json()
        assert "error" in data
        assert data["error"]["code"] == "NOT_FOUND"
        assert isinstance(data["error"]["message"], str)
        assert "details" in data["error"]

    def test_400_error_format(self, client):
        resp = client.post(
            "/api/files/upload",
            files={"file": ("image.png", io.BytesIO(b"fake"), "image/png")},
        )
        assert resp.status_code == 400
        data = resp.json()
        assert "error" in data
        assert data["error"]["code"] == "VALIDATION_ERROR"

    def test_404_file_error_format(self, client):
        resp = client.get("/api/files/nonexistent")
        assert resp.status_code == 404
        data = resp.json()
        assert "error" in data
        assert data["error"]["code"] == "NOT_FOUND"
        assert "File not found" in data["error"]["message"]

    def test_404_trace_error_format(self, client):
        resp = client.get("/api/traces/nonexistent")
        assert resp.status_code == 404
        data = resp.json()
        assert "error" in data
        assert data["error"]["code"] == "NOT_FOUND"

    def test_404_metrics_error_format(self, client):
        resp = client.get("/api/metrics/nonexistent")
        assert resp.status_code == 404
        data = resp.json()
        assert "error" in data
        assert data["error"]["code"] == "NOT_FOUND"


# ---------------------------------------------------------------------------
# Config Merge (MAJOR-5)
# ---------------------------------------------------------------------------


class TestConfigMerge:
    def test_budget_merge_preserves_unset_fields(self, client):
        # Set initial budget to known values
        client.put(
            "/api/config",
            json={
                "budget": {
                    "max_steps": 32,
                    "max_tokens": 100000,
                    "max_cost_usd": 5.0,
                    "max_time_seconds": 60,
                    "max_recursion_depth": 10,
                }
            },
        )

        # Update only max_steps
        resp = client.put(
            "/api/config",
            json={"budget": {"max_steps": 64}},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["budget"]["max_steps"] == 64
        # Other budget fields should remain from previous update
        assert data["budget"]["max_tokens"] == 100000
        assert data["budget"]["max_cost_usd"] == 5.0
        assert data["budget"]["max_time_seconds"] == 60
        assert data["budget"]["max_recursion_depth"] == 10

    def test_appearance_merge_preserves_unset_fields(self, client):
        # Set initial appearance
        client.put(
            "/api/config",
            json={"appearance": {"theme": "dark", "sidebar_collapsed": True}},
        )

        # Update only theme
        resp = client.put(
            "/api/config",
            json={"appearance": {"theme": "light"}},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["appearance"]["theme"] == "light"
        assert data["appearance"]["sidebar_collapsed"] is True


# ---------------------------------------------------------------------------
# WebSocket (JSON validation - MINOR-3)
# ---------------------------------------------------------------------------


class TestWebSocket:
    def test_websocket_malformed_json(self, client):
        with client.websocket_connect("/ws/chat/test-session") as ws:
            # Read the connected message
            connected = ws.receive_json()
            assert connected["type"] == "connected"

            # Send malformed JSON
            ws.send_text("not valid json{{{")
            error = ws.receive_json()
            assert error["type"] == "error"
            assert error["data"]["code"] == "INVALID_JSON"
            assert error["data"]["recoverable"] is True

    def test_websocket_connected_message(self, client):
        with client.websocket_connect("/ws/chat/my-session") as ws:
            msg = ws.receive_json()
            assert msg["type"] == "connected"
            assert msg["session_id"] == "my-session"

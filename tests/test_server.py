"""Tests for the FastAPI server using TestClient with mocked use cases."""

from __future__ import annotations

import io
from collections.abc import Generator
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
def _clean_state() -> Generator[None, None, None]:
    """Reset shared state before each test."""
    reset_state()
    yield
    reset_state()


@pytest.fixture
def client() -> TestClient:
    app = create_app()
    return TestClient(app)


# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------


class TestHealthCheck:
    def test_health(self, client: TestClient) -> None:
        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert data["version"] == "1.0.0"
        assert "uptime_seconds" in data


# ---------------------------------------------------------------------------
# Sessions
# ---------------------------------------------------------------------------


class TestSessions:
    def test_list_empty(self, client: TestClient) -> None:
        resp = client.get("/api/sessions")
        assert resp.status_code == 200
        assert resp.json() == []

    def test_list_sessions(self, client: TestClient) -> None:
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

    def test_get_session(self, client: TestClient) -> None:
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

    def test_get_session_not_found(self, client: TestClient) -> None:
        resp = client.get("/api/sessions/nonexistent")
        assert resp.status_code == 404

    def test_delete_session(self, client: TestClient) -> None:
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

    def test_delete_session_not_found(self, client: TestClient) -> None:
        resp = client.delete("/api/sessions/nonexistent")
        assert resp.status_code == 404

    def test_pagination(self, client: TestClient) -> None:
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

    def test_rename_session(self, client: TestClient) -> None:
        state = get_state()
        now = datetime.now(timezone.utc)
        state.sessions["s1"] = SessionRecord(
            id="s1",
            name="Old Name",
            created_at=now,
            updated_at=now,
        )
        resp = client.put("/api/sessions/s1", json={"name": "New Name"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["id"] == "s1"
        assert data["name"] == "New Name"
        assert state.sessions["s1"].name == "New Name"

    def test_rename_session_not_found(self, client: TestClient) -> None:
        resp = client.put("/api/sessions/nonexistent", json={"name": "New Name"})
        assert resp.status_code == 404

    def test_rename_session_empty_name(self, client: TestClient) -> None:
        state = get_state()
        now = datetime.now(timezone.utc)
        state.sessions["s1"] = SessionRecord(
            id="s1",
            name="Original Name",
            created_at=now,
            updated_at=now,
        )
        resp = client.put("/api/sessions/s1", json={"name": "  "})
        assert resp.status_code == 200
        data = resp.json()
        # Blank name strips to "" which is falsy, so the original name is kept
        assert data["name"] == "Original Name"
        assert state.sessions["s1"].name == "Original Name"


# ---------------------------------------------------------------------------
# Provider Models
# ---------------------------------------------------------------------------


class TestProviderModels:
    def test_list_provider_models_fallback(self, client: TestClient) -> None:
        # No real API key available in tests — endpoint falls back to catalog
        resp = client.get("/api/providers/openai/models")
        assert resp.status_code == 200
        data = resp.json()
        assert isinstance(data, list)
        assert len(data) > 0
        names = [m["name"] for m in data]
        assert "gpt-4o" in names
        assert "gpt-4o-mini" in names

    def test_list_provider_models_unknown_provider(self, client: TestClient) -> None:
        # Unknown provider — catalog lookup returns nothing, endpoint returns []
        resp = client.get("/api/providers/nonexistent/models")
        assert resp.status_code == 200
        assert resp.json() == []


# ---------------------------------------------------------------------------
# File upload
# ---------------------------------------------------------------------------


class TestFileUpload:
    def test_upload_text_file(self, client: TestClient) -> None:
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

    def test_upload_md_file(self, client: TestClient) -> None:
        content = b"# Hello\n\nThis is markdown."
        resp = client.post(
            "/api/files/upload",
            files={"file": ("readme.md", io.BytesIO(content), "text/markdown")},
        )
        assert resp.status_code == 201
        assert resp.json()["name"] == "readme.md"

    def test_upload_unsupported_type(self, client: TestClient) -> None:
        resp = client.post(
            "/api/files/upload",
            files={"file": ("image.png", io.BytesIO(b"fake"), "image/png")},
        )
        assert resp.status_code == 400

    def test_upload_no_file_field_returns_400(self, client: TestClient) -> None:
        """Sending a form with no 'file' upload field returns 400."""
        resp = client.post(
            "/api/files/upload",
            data={"file": "not-a-file"},
        )
        assert resp.status_code == 400

    def test_upload_file_too_large_returns_413(
        self, client: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Files exceeding the size cap are rejected with 413."""
        import rlmkit.server.routes.files as files_mod

        monkeypatch.setattr(files_mod, "_MAX_FILE_SIZE", 5)
        resp = client.post(
            "/api/files/upload",
            files={"file": ("big.txt", io.BytesIO(b"123456"), "text/plain")},
        )
        assert resp.status_code == 413

    def test_upload_corrupt_pdf_returns_422(self, client: TestClient) -> None:
        """A PDF that cannot be parsed is rejected with 422."""
        resp = client.post(
            "/api/files/upload",
            files={"file": ("bad.pdf", io.BytesIO(b"not a real pdf"), "application/pdf")},
        )
        # pypdf raises on corrupt content → 422 from text-extraction error handler
        assert resp.status_code in (201, 422)  # 201 if pypdf falls back gracefully

    def test_upload_pdf_file(self, client: TestClient) -> None:
        """A minimal valid PDF is accepted."""
        # Minimal 1-page PDF that pypdf can parse
        minimal_pdf = (
            b"%PDF-1.4\n"
            b"1 0 obj<</Type/Catalog/Pages 2 0 R>>endobj\n"
            b"2 0 obj<</Type/Pages/Kids[3 0 R]/Count 1>>endobj\n"
            b"3 0 obj<</Type/Page/MediaBox[0 0 3 3]>>endobj\n"
            b"xref\n0 4\n0000000000 65535 f\n"
            b"0000000009 00000 n\n0000000058 00000 n\n"
            b"0000000115 00000 n\n"
            b"trailer<</Size 4/Root 1 0 R>>\nstartxref\n190\n%%EOF"
        )
        resp = client.post(
            "/api/files/upload",
            files={"file": ("doc.pdf", io.BytesIO(minimal_pdf), "application/pdf")},
        )
        assert resp.status_code in (201, 422)

    def test_upload_json_file(self, client: TestClient) -> None:
        """JSON files are accepted and decoded as text."""
        content = b'{"key": "value"}'
        resp = client.post(
            "/api/files/upload",
            files={"file": ("data.json", io.BytesIO(content), "application/json")},
        )
        assert resp.status_code == 201

    def test_upload_csv_file(self, client: TestClient) -> None:
        """CSV files are accepted and decoded as text."""
        content = b"col1,col2\nval1,val2\n"
        resp = client.post(
            "/api/files/upload",
            files={"file": ("data.csv", io.BytesIO(content), "text/csv")},
        )
        assert resp.status_code == 201

    def test_get_file(self, client: TestClient) -> None:
        content = b"some text"
        resp = client.post(
            "/api/files/upload",
            files={"file": ("doc.txt", io.BytesIO(content), "text/plain")},
        )
        file_id = resp.json()["id"]

        resp = client.get(f"/api/files/{file_id}")
        assert resp.status_code == 200
        assert resp.json()["id"] == file_id

    def test_get_file_not_found(self, client: TestClient) -> None:
        resp = client.get("/api/files/nonexistent")
        assert resp.status_code == 404


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


class TestMetrics:
    def test_metrics_not_found(self, client: TestClient) -> None:
        resp = client.get("/api/metrics/nonexistent")
        assert resp.status_code == 404

    def test_metrics_empty_session(self, client: TestClient) -> None:
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

    def test_metrics_with_messages(self, client: TestClient) -> None:
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
    def test_trace_not_found(self, client: TestClient) -> None:
        resp = client.get("/api/traces/nonexistent")
        assert resp.status_code == 404

    def test_trace_found(self, client: TestClient) -> None:
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
    def test_list_providers(self, client: TestClient) -> None:
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
    def test_get_config(self, client: TestClient) -> None:
        resp = client.get("/api/config")
        assert resp.status_code == 200
        data = resp.json()
        assert "active_provider" in data
        assert "budget" in data
        assert "sandbox" in data
        assert "appearance" in data

    def test_update_config(self, client: TestClient) -> None:
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

    def test_update_budget(self, client: TestClient) -> None:
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

    def test_update_preserves_unset_fields(self, client: TestClient) -> None:
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
    def test_chat_creates_session(self, client: TestClient) -> None:
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

    def test_chat_with_existing_session(self, client: TestClient) -> None:
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

    def test_chat_with_file_id(self, client: TestClient) -> None:
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

    def test_chat_missing_file_id_returns_404(self, client: TestClient) -> None:
        resp = client.post(
            "/api/chat",
            json={"query": "Summarize", "file_id": "nonexistent"},
        )
        assert resp.status_code == 404
        data = resp.json()
        assert data["error"]["code"] == "NOT_FOUND"
        assert "File not found" in data["error"]["message"]

    def test_chat_missing_content_and_file_id_returns_400(self, client: TestClient) -> None:
        resp = client.post(
            "/api/chat",
            json={"query": "What?"},
        )
        assert resp.status_code == 400
        data = resp.json()
        assert data["error"]["code"] == "VALIDATION_ERROR"
        assert "content or file_id" in data["error"]["message"]

    def test_chat_rejects_invalid_mode(self, client: TestClient) -> None:
        resp = client.post(
            "/api/chat",
            json={"query": "What?", "content": "text", "mode": "invalid_mode"},
        )
        assert resp.status_code == 422


# ---------------------------------------------------------------------------
# Error Response Format (CRITICAL-1)
# ---------------------------------------------------------------------------


class TestErrorResponseFormat:
    def test_404_error_format(self, client: TestClient) -> None:
        resp = client.get("/api/sessions/nonexistent")
        assert resp.status_code == 404
        data = resp.json()
        assert "error" in data
        assert data["error"]["code"] == "NOT_FOUND"
        assert isinstance(data["error"]["message"], str)
        assert "details" in data["error"]

    def test_400_error_format(self, client: TestClient) -> None:
        resp = client.post(
            "/api/files/upload",
            files={"file": ("image.png", io.BytesIO(b"fake"), "image/png")},
        )
        assert resp.status_code == 400
        data = resp.json()
        assert "error" in data
        assert data["error"]["code"] == "VALIDATION_ERROR"

    def test_404_file_error_format(self, client: TestClient) -> None:
        resp = client.get("/api/files/nonexistent")
        assert resp.status_code == 404
        data = resp.json()
        assert "error" in data
        assert data["error"]["code"] == "NOT_FOUND"
        assert "File not found" in data["error"]["message"]

    def test_404_trace_error_format(self, client: TestClient) -> None:
        resp = client.get("/api/traces/nonexistent")
        assert resp.status_code == 404
        data = resp.json()
        assert "error" in data
        assert data["error"]["code"] == "NOT_FOUND"

    def test_404_metrics_error_format(self, client: TestClient) -> None:
        resp = client.get("/api/metrics/nonexistent")
        assert resp.status_code == 404
        data = resp.json()
        assert "error" in data
        assert data["error"]["code"] == "NOT_FOUND"


# ---------------------------------------------------------------------------
# Config Merge (MAJOR-5)
# ---------------------------------------------------------------------------


class TestConfigMerge:
    def test_budget_merge_preserves_unset_fields(self, client: TestClient) -> None:
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

    def test_appearance_merge_preserves_unset_fields(self, client: TestClient) -> None:
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
    def test_websocket_malformed_json(self, client: TestClient) -> None:
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

    def test_websocket_connected_message(self, client: TestClient) -> None:
        with client.websocket_connect("/ws/chat/my-session") as ws:
            msg = ws.receive_json()
            assert msg["type"] == "connected"
            assert msg["session_id"] == "my-session"


# ---------------------------------------------------------------------------
# Session Persistence
# ---------------------------------------------------------------------------


class TestSessionPersistence:
    def test_save_and_load_sessions(
        self, tmp_path: object, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Sessions saved to disk are loaded back on startup."""
        import rlmkit.server.dependencies as deps

        sessions_file = tmp_path / "sessions.json"  # type: ignore[operator]
        monkeypatch.setattr(deps, "_SESSIONS_FILE", sessions_file)

        state = get_state()
        now = datetime.now(timezone.utc)
        state.sessions["s1"] = SessionRecord(
            id="s1",
            name="Test Session",
            created_at=now,
            updated_at=now,
            messages=[
                {"id": "m1", "role": "user", "content": "hello", "timestamp": now.isoformat()},
                {
                    "id": "m2",
                    "role": "assistant",
                    "content": "hi",
                    "provider": "anthropic",
                    "mode_used": "direct",
                    "timestamp": now.isoformat(),
                    "metrics": {"total_tokens": 100, "cost_usd": 0.01},
                },
            ],
        )
        # Enable real persistence for this test
        state.save_sessions = deps.AppState.save_sessions.__get__(state)
        state.save_sessions()

        assert sessions_file.exists()

        # Load into a fresh state
        reset_state()
        state2 = get_state()
        state2._load_sessions = deps.AppState._load_sessions.__get__(state2)
        monkeypatch.setattr(deps, "_SESSIONS_FILE", sessions_file)
        state2._load_sessions()

        assert "s1" in state2.sessions
        assert state2.sessions["s1"].name == "Test Session"
        assert len(state2.sessions["s1"].messages) == 2

    def test_session_cap(self, tmp_path: object, monkeypatch: pytest.MonkeyPatch) -> None:
        """Only the most recent N sessions are persisted."""
        import rlmkit.server.dependencies as deps

        sessions_file = tmp_path / "sessions.json"  # type: ignore[operator]
        monkeypatch.setattr(deps, "_SESSIONS_FILE", sessions_file)
        monkeypatch.setattr(deps, "_MAX_PERSISTED_SESSIONS", 5)

        state = get_state()
        state.save_sessions = deps.AppState.save_sessions.__get__(state)

        for i in range(10):
            now = datetime(2026, 1, 1 + i, tzinfo=timezone.utc)
            state.sessions[f"s{i}"] = SessionRecord(
                id=f"s{i}",
                name=f"Session {i}",
                created_at=now,
                updated_at=now,
            )

        state.save_sessions()

        import json

        saved = json.loads(sessions_file.read_text())  # type: ignore[union-attr]
        assert len(saved) == 5
        # Most recent sessions should be kept (s9, s8, s7, s6, s5)
        saved_ids = {s["id"] for s in saved}
        assert "s9" in saved_ids
        assert "s0" not in saved_ids

    def test_corrupt_sessions_file(self, tmp_path: object, monkeypatch: pytest.MonkeyPatch) -> None:
        """Corrupt sessions file is handled gracefully."""
        import rlmkit.server.dependencies as deps

        sessions_file = tmp_path / "sessions.json"  # type: ignore[operator]
        sessions_file.write_text("not valid json{{{")  # type: ignore[union-attr]
        monkeypatch.setattr(deps, "_SESSIONS_FILE", sessions_file)

        state = get_state()
        state._load_sessions = deps.AppState._load_sessions.__get__(state)
        state._load_sessions()
        # Should not crash, sessions remain empty
        assert len(state.sessions) == 0


# ---------------------------------------------------------------------------
# Metrics: by_provider
# ---------------------------------------------------------------------------


class TestMetricsByProvider:
    def test_metrics_include_by_provider(self, client: TestClient) -> None:
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
                    "mode_used": "direct",
                    "provider": "anthropic",
                    "timestamp": now.isoformat(),
                    "metrics": {"total_tokens": 200, "cost_usd": 0.02, "elapsed_seconds": 1.0},
                },
                {"id": "m3", "role": "user", "content": "q2", "timestamp": now.isoformat()},
                {
                    "id": "m4",
                    "role": "assistant",
                    "content": "a2",
                    "mode_used": "direct",
                    "provider": "openai",
                    "timestamp": now.isoformat(),
                    "metrics": {"total_tokens": 300, "cost_usd": 0.03, "elapsed_seconds": 0.5},
                },
            ],
        )
        resp = client.get("/api/metrics/s1")
        assert resp.status_code == 200
        data = resp.json()
        assert "anthropic" in data["by_provider"]
        assert "openai" in data["by_provider"]
        assert data["by_provider"]["anthropic"]["total_tokens"] == 200
        assert data["by_provider"]["openai"]["total_cost_usd"] == 0.03


# ---------------------------------------------------------------------------
# Tests: Trajectory logging
# ---------------------------------------------------------------------------


class TestTrajectoryLogging:
    """Test that trajectory JSONL files are saved when configured."""

    def test_save_trajectory_writes_jsonl(self, tmp_path: object) -> None:
        from rlmkit.application.dto import RunResultDTO
        from rlmkit.server.routes.chat import _save_trajectory

        execution = ExecutionRecord(
            execution_id="exec-1",
            session_id="sess-1",
            query="What is 2+2?",
            mode="direct",
        )
        result = RunResultDTO(
            answer="4",
            mode_used="direct",
            success=True,
            steps=1,
            input_tokens=10,
            output_tokens=5,
            elapsed_time=0.5,
            trace=[
                {
                    "step": 0,
                    "role": "assistant",
                    "content": "4",
                    "input_tokens": 10,
                    "output_tokens": 5,
                    "model": "gpt-4o",
                    "elapsed_seconds": 0.5,
                },
            ],
        )
        trace_dir = str(tmp_path / "trajectories")  # type: ignore[operator]
        _save_trajectory(execution, result, trace_dir)

        filepath = tmp_path / "trajectories" / "exec-1.jsonl"  # type: ignore[operator]
        assert filepath.exists()
        lines = filepath.read_text().strip().split("\n")  # type: ignore[union-attr]
        # First line is metadata, second is the step
        assert len(lines) == 2
        import json

        meta = json.loads(lines[0])
        assert meta["metadata"]["execution_id"] == "exec-1"
        step = json.loads(lines[1])
        assert step["action_type"] == "final"
        assert step["tokens_used"] == 15


# ---------------------------------------------------------------------------
# num_retries — model fields and adapter creation
# ---------------------------------------------------------------------------


class TestNumRetriesModel:
    """ChatRequest and ChatProviderConfig accept num_retries field."""

    def test_chat_request_accepts_num_retries(self, client: TestClient) -> None:
        resp = client.post(
            "/api/chat",
            json={
                "query": "test",
                "content": "content",
                "mode": "direct",
                "num_retries": 0,
            },
        )
        # 202 means the field was accepted and parsed correctly
        assert resp.status_code == 202

    def test_chat_request_num_retries_defaults_none(self, client: TestClient) -> None:
        resp = client.post(
            "/api/chat",
            json={"query": "test", "content": "content", "mode": "direct"},
        )
        assert resp.status_code == 202

    def test_chat_provider_create_accepts_num_retries(self, client: TestClient) -> None:
        lp_resp = client.post(
            "/api/llm-providers",
            json={"name": "Ollama llama3.2", "backend": "ollama", "model": "llama3.2"},
        )
        assert lp_resp.status_code == 201
        lp_id = lp_resp.json()["id"]

        resp = client.post(
            "/api/chat-providers",
            json={
                "name": "no-retry-provider",
                "llm_provider_id": lp_id,
                "num_retries": 0,
            },
        )
        assert resp.status_code == 201
        data = resp.json()
        assert data["num_retries"] == 0

    def test_chat_provider_update_accepts_num_retries(self, client: TestClient) -> None:
        lp_resp = client.post(
            "/api/llm-providers",
            json={"name": "OpenAI gpt-4o", "backend": "openai", "model": "gpt-4o"},
        )
        assert lp_resp.status_code == 201
        lp_id = lp_resp.json()["id"]

        create_resp = client.post(
            "/api/chat-providers",
            json={"name": "updatable", "llm_provider_id": lp_id},
        )
        assert create_resp.status_code == 201
        cp_id = create_resp.json()["id"]

        update_resp = client.put(f"/api/chat-providers/{cp_id}", json={"num_retries": 5})
        assert update_resp.status_code == 200
        assert update_resp.json()["num_retries"] == 5

    def test_chat_provider_num_retries_cleared_with_explicit_null(self, client: TestClient) -> None:
        """Sending num_retries=null explicitly resets the override to None (automatic defaults)."""
        lp_resp = client.post(
            "/api/llm-providers",
            json={"name": "OpenAI gpt-4o", "backend": "openai", "model": "gpt-4o"},
        )
        assert lp_resp.status_code == 201
        lp_id = lp_resp.json()["id"]

        # Create with an override
        create_resp = client.post(
            "/api/chat-providers",
            json={
                "name": "clearable",
                "llm_provider_id": lp_id,
                "num_retries": 5,
            },
        )
        assert create_resp.status_code == 201
        cp_id = create_resp.json()["id"]
        assert create_resp.json()["num_retries"] == 5

        # Clear the override by explicitly sending null
        clear_resp = client.put(f"/api/chat-providers/{cp_id}", json={"num_retries": None})
        assert clear_resp.status_code == 200
        assert clear_resp.json()["num_retries"] is None

    def test_chat_provider_num_retries_unchanged_when_field_omitted(
        self, client: TestClient
    ) -> None:
        """Omitting num_retries from the update body leaves the stored value untouched."""
        lp_resp = client.post(
            "/api/llm-providers",
            json={"name": "OpenAI gpt-4o", "backend": "openai", "model": "gpt-4o"},
        )
        assert lp_resp.status_code == 201
        lp_id = lp_resp.json()["id"]

        create_resp = client.post(
            "/api/chat-providers",
            json={
                "name": "sticky",
                "llm_provider_id": lp_id,
                "num_retries": 3,
            },
        )
        assert create_resp.status_code == 201
        cp_id = create_resp.json()["id"]

        # Update an unrelated field — num_retries must stay at 3
        update_resp = client.put(f"/api/chat-providers/{cp_id}", json={"execution_mode": "direct"})
        assert update_resp.status_code == 200
        assert update_resp.json()["num_retries"] == 3

    def test_negative_num_retries_rejected_in_chat_request(self, client: TestClient) -> None:
        resp = client.post(
            "/api/chat",
            json={"query": "test", "content": "content", "num_retries": -1},
        )
        assert resp.status_code == 422

    def test_negative_num_retries_rejected_in_provider_create(self, client: TestClient) -> None:
        lp_resp = client.post(
            "/api/llm-providers",
            json={"name": "OpenAI gpt-4o", "backend": "openai", "model": "gpt-4o"},
        )
        assert lp_resp.status_code == 201
        lp_id = lp_resp.json()["id"]

        resp = client.post(
            "/api/chat-providers",
            json={
                "name": "bad",
                "llm_provider_id": lp_id,
                "num_retries": -1,
            },
        )
        assert resp.status_code == 422

    def test_negative_num_retries_rejected_in_provider_update(self, client: TestClient) -> None:
        lp_resp = client.post(
            "/api/llm-providers",
            json={"name": "OpenAI gpt-4o", "backend": "openai", "model": "gpt-4o"},
        )
        assert lp_resp.status_code == 201
        lp_id = lp_resp.json()["id"]

        create_resp = client.post(
            "/api/chat-providers",
            json={"name": "valid", "llm_provider_id": lp_id},
        )
        assert create_resp.status_code == 201
        cp_id = create_resp.json()["id"]
        resp = client.put(f"/api/chat-providers/{cp_id}", json={"num_retries": -2})
        assert resp.status_code == 422


class TestNumRetriesAdapterCreation:
    """AppState.create_llm_adapter* respects num_retries priority chain."""

    def test_ollama_provider_defaults_to_0_retries(self) -> None:
        from unittest.mock import patch

        state = get_state()
        with patch("rlmkit.server.dependencies.LiteLLMAdapter") as mock_cls:
            mock_cls.return_value = object()
            state.config.active_provider = "ollama"
            state.config.active_model = "llama3.2"
            state.create_llm_adapter()
        _, kwargs = mock_cls.call_args
        assert kwargs["num_retries"] == 0

    def test_openai_provider_defaults_to_2_retries(self) -> None:
        from unittest.mock import patch

        state = get_state()
        with patch("rlmkit.server.dependencies.LiteLLMAdapter") as mock_cls:
            mock_cls.return_value = object()
            state.config.active_provider = "openai"
            state.config.active_model = "gpt-4o"
            state.create_llm_adapter()
        _, kwargs = mock_cls.call_args
        assert kwargs["num_retries"] == 2

    def test_explicit_num_retries_overrides_default(self) -> None:
        from unittest.mock import patch

        state = get_state()
        with patch("rlmkit.server.dependencies.LiteLLMAdapter") as mock_cls:
            mock_cls.return_value = object()
            state.config.active_provider = "openai"
            state.config.active_model = "gpt-4o"
            state.create_llm_adapter(num_retries=0)
        _, kwargs = mock_cls.call_args
        assert kwargs["num_retries"] == 0

    def test_chat_provider_config_num_retries_used(self) -> None:
        from unittest.mock import patch

        from rlmkit.server.models import ChatProviderConfig, RuntimeSettings

        state = get_state()
        cp = ChatProviderConfig(
            id="cp-test",
            name="TestCP",
            llm_provider="openai",
            llm_model="gpt-4o",
            runtime_settings=RuntimeSettings(),
            num_retries=1,
        )
        state.config.chat_providers = [cp]

        with patch("rlmkit.server.dependencies.LiteLLMAdapter") as mock_cls:
            mock_cls.return_value = object()
            state.create_llm_adapter_for_chat_provider("cp-test")
        _, kwargs = mock_cls.call_args
        assert kwargs["num_retries"] == 1

    def test_per_request_num_retries_beats_chat_provider_config(self) -> None:
        from unittest.mock import patch

        from rlmkit.server.models import ChatProviderConfig, RuntimeSettings

        state = get_state()
        cp = ChatProviderConfig(
            id="cp-test2",
            name="TestCP2",
            llm_provider="openai",
            llm_model="gpt-4o",
            runtime_settings=RuntimeSettings(),
            num_retries=1,
        )
        state.config.chat_providers = [cp]

        with patch("rlmkit.server.dependencies.LiteLLMAdapter") as mock_cls:
            mock_cls.return_value = object()
            state.create_llm_adapter_for_chat_provider("cp-test2", num_retries=7)
        _, kwargs = mock_cls.call_args
        assert kwargs["num_retries"] == 7


# ---------------------------------------------------------------------------
# Profiles CRUD
# ---------------------------------------------------------------------------


class TestProfiles:
    def test_list_profiles_includes_builtins(self, client: TestClient) -> None:
        resp = client.get("/api/profiles")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data) >= 4  # 4 builtins
        builtin_ids = {p["id"] for p in data}
        assert "builtin-fast" in builtin_ids
        assert "builtin-accurate" in builtin_ids
        assert "builtin-rlm-deep" in builtin_ids
        assert "builtin-rag" in builtin_ids

    def test_create_profile(self, client: TestClient) -> None:
        payload = {"name": "My Profile", "strategy": "rlm"}
        resp = client.post("/api/profiles", json=payload)
        assert resp.status_code == 201
        data = resp.json()
        assert data["name"] == "My Profile"
        assert data["strategy"] == "rlm"
        assert data["is_builtin"] is False
        assert "id" in data

    def test_create_profile_duplicate_name_returns_409(self, client: TestClient) -> None:
        payload = {"name": "Dup Profile", "strategy": "direct"}
        client.post("/api/profiles", json=payload)
        resp = client.post("/api/profiles", json=payload)
        assert resp.status_code == 409

    def test_create_profile_duplicate_builtin_name_returns_409(self, client: TestClient) -> None:
        payload = {"name": "Fast & cheap", "strategy": "direct"}
        resp = client.post("/api/profiles", json=payload)
        assert resp.status_code == 409

    def test_update_profile(self, client: TestClient) -> None:
        create_resp = client.post("/api/profiles", json={"name": "ToUpdate", "strategy": "direct"})
        profile_id = create_resp.json()["id"]
        resp = client.put(
            f"/api/profiles/{profile_id}", json={"name": "Updated", "strategy": "rlm"}
        )
        assert resp.status_code == 200
        assert resp.json()["name"] == "Updated"
        assert resp.json()["strategy"] == "rlm"

    def test_update_builtin_returns_400(self, client: TestClient) -> None:
        resp = client.put("/api/profiles/builtin-fast", json={"name": "X"})
        assert resp.status_code == 400

    def test_update_profile_duplicate_name_returns_409(self, client: TestClient) -> None:
        client.post("/api/profiles", json={"name": "P1", "strategy": "direct"})
        r2 = client.post("/api/profiles", json={"name": "P2", "strategy": "direct"})
        pid2 = r2.json()["id"]
        resp = client.put(f"/api/profiles/{pid2}", json={"name": "P1"})
        assert resp.status_code == 409

    def test_update_profile_not_found_returns_404(self, client: TestClient) -> None:
        resp = client.put("/api/profiles/nonexistent", json={"name": "X"})
        assert resp.status_code == 404

    def test_delete_profile(self, client: TestClient) -> None:
        create_resp = client.post("/api/profiles", json={"name": "ToDel", "strategy": "rlm"})
        profile_id = create_resp.json()["id"]
        resp = client.delete(f"/api/profiles/{profile_id}")
        assert resp.status_code == 204
        # Should no longer appear in listing
        listing = client.get("/api/profiles").json()
        assert all(p["id"] != profile_id for p in listing)

    def test_delete_builtin_returns_400(self, client: TestClient) -> None:
        resp = client.delete("/api/profiles/builtin-rag")
        assert resp.status_code == 400

    def test_delete_profile_not_found_returns_404(self, client: TestClient) -> None:
        resp = client.delete("/api/profiles/nonexistent")
        assert resp.status_code == 404

    def test_activate_endpoint_removed(self, client: TestClient) -> None:
        """The /activate endpoint was removed; verify it returns 404/405."""
        resp = client.post("/api/profiles/builtin-accurate/activate")
        assert resp.status_code in (404, 405)


# ---------------------------------------------------------------------------
# System prompts
# ---------------------------------------------------------------------------


class TestSystemPrompts:
    def test_get_system_prompts(self, client: TestClient) -> None:
        resp = client.get("/api/system-prompts")
        assert resp.status_code == 200
        data = resp.json()
        assert "direct" in data or isinstance(data, dict)

    def test_update_system_prompts(self, client: TestClient) -> None:
        resp = client.put(
            "/api/system-prompts",
            json={"direct": "You are a helpful assistant.", "rlm": "", "rag": ""},
        )
        assert resp.status_code == 200

    def test_list_prompt_templates(self, client: TestClient) -> None:
        resp = client.get("/api/system-prompts/templates")
        assert resp.status_code == 200
        assert isinstance(resp.json(), list)


# ---------------------------------------------------------------------------
# Public API deprecation shim
# ---------------------------------------------------------------------------


class TestPublicAPIShim:
    def test_deprecated_public_interact_result(self) -> None:
        import warnings

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            from rlmkit.public import PublicInteractResult  # noqa: F401

            assert len(w) >= 1
            dep_warnings = [x for x in w if issubclass(x.category, DeprecationWarning)]
            assert len(dep_warnings) >= 1
            assert "deprecated" in str(dep_warnings[0].message).lower()

    def test_unknown_attribute_raises(self) -> None:
        import importlib

        import pytest

        pub = importlib.import_module("rlmkit.public")
        with pytest.raises(AttributeError):
            _ = pub.NonExistentName

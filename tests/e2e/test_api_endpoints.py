"""E2E tests for REST API endpoints.

Tests exercise real FastAPI routes via TestClient against the in-memory
backend. Each test resets state via the autouse _clean_state fixture.
"""

from __future__ import annotations

import uuid

import pytest

from rlmkit.server.dependencies import get_state

pytestmark = [pytest.mark.e2e]


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------


class TestHealthEndpoint:
    """GET /health -- server liveness check."""

    def test_health_returns_200_with_status_ok(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert data["version"] == "0.1.0"
        assert "uptime_seconds" in data
        assert isinstance(data["uptime_seconds"], (int, float))


# ---------------------------------------------------------------------------
# Chat
# ---------------------------------------------------------------------------


class TestChatEndpoints:
    """POST /api/chat -- submit a query (returns 202, async execution)."""

    def test_chat_submit_returns_202(self, client):
        resp = client.post(
            "/api/chat",
            json={
                "query": "What is RLM?",
                "content": "RLM is a recursive language model approach.",
                "mode": "direct",
                "provider": "openai",
            },
        )
        assert resp.status_code == 202
        data = resp.json()
        assert "execution_id" in data
        assert "session_id" in data
        assert data["status"] == "running"

    def test_chat_creates_session(self, client):
        resp = client.post(
            "/api/chat",
            json={"query": "Hello", "content": "Some content.", "mode": "direct"},
        )
        session_id = resp.json()["session_id"]
        state = get_state()
        assert session_id in state.sessions

    def test_chat_uses_provided_session_id(self, client):
        sid = str(uuid.uuid4())
        resp = client.post(
            "/api/chat",
            json={"query": "Hi", "content": "Content.", "mode": "direct", "session_id": sid},
        )
        assert resp.json()["session_id"] == sid

    def test_chat_with_file_id_not_found(self, client):
        resp = client.post(
            "/api/chat",
            json={"query": "Summarize", "file_id": "nonexistent", "mode": "direct"},
        )
        assert resp.status_code == 404

    def test_chat_requires_content_or_file_id(self, client):
        resp = client.post(
            "/api/chat",
            json={"query": "Hello", "mode": "direct"},
        )
        assert resp.status_code == 400

    def test_chat_with_uploaded_file(self, client, uploaded_file_id):
        resp = client.post(
            "/api/chat",
            json={"query": "Summarize", "file_id": uploaded_file_id, "mode": "direct"},
        )
        assert resp.status_code == 202
        assert "execution_id" in resp.json()


# ---------------------------------------------------------------------------
# File upload
# ---------------------------------------------------------------------------


class TestFileUploadEndpoints:
    """POST /api/files/upload -- upload a document."""

    def test_upload_text_file(self, client):
        resp = client.post(
            "/api/files/upload",
            files={"file": ("test.txt", b"Hello world content", "text/plain")},
        )
        assert resp.status_code == 201
        data = resp.json()
        assert "id" in data
        assert data["name"] == "test.txt"
        assert data["size_bytes"] == len(b"Hello world content")
        assert data["token_count"] > 0

    def test_upload_markdown_file(self, client):
        content = b"# Title\n\nSome markdown content.\n"
        resp = client.post(
            "/api/files/upload",
            files={"file": ("readme.md", content, "text/markdown")},
        )
        assert resp.status_code == 201
        assert resp.json()["name"] == "readme.md"

    def test_upload_unsupported_type_returns_400(self, client):
        resp = client.post(
            "/api/files/upload",
            files={"file": ("image.png", b"\x89PNG", "image/png")},
        )
        assert resp.status_code == 400

    def test_get_uploaded_file(self, client, uploaded_file_id):
        resp = client.get(f"/api/files/{uploaded_file_id}")
        assert resp.status_code == 200
        data = resp.json()
        assert data["id"] == uploaded_file_id
        assert data["name"] == "test.txt"

    def test_get_nonexistent_file_returns_404(self, client):
        resp = client.get("/api/files/no-such-id")
        assert resp.status_code == 404


# ---------------------------------------------------------------------------
# Sessions
# ---------------------------------------------------------------------------


class TestSessionEndpoints:
    """Session management endpoints."""

    def test_list_sessions_initially_empty(self, client):
        resp = client.get("/api/sessions")
        assert resp.status_code == 200
        assert resp.json() == []

    def test_list_sessions_after_chat(self, client):
        client.post(
            "/api/chat",
            json={"query": "Hello", "content": "Some content.", "mode": "direct"},
        )
        resp = client.get("/api/sessions")
        assert resp.status_code == 200
        sessions = resp.json()
        assert len(sessions) == 1
        assert "id" in sessions[0]
        assert "name" in sessions[0]
        assert "message_count" in sessions[0]

    def test_get_session_detail(self, client):
        chat_resp = client.post(
            "/api/chat",
            json={"query": "Hi", "content": "Content.", "mode": "direct"},
        )
        sid = chat_resp.json()["session_id"]
        resp = client.get(f"/api/sessions/{sid}")
        assert resp.status_code == 200
        data = resp.json()
        assert data["id"] == sid
        assert "messages" in data
        assert len(data["messages"]) >= 1

    def test_get_nonexistent_session_returns_404(self, client):
        resp = client.get("/api/sessions/nonexistent")
        assert resp.status_code == 404

    def test_delete_session(self, client):
        chat_resp = client.post(
            "/api/chat",
            json={"query": "Hi", "content": "Content.", "mode": "direct"},
        )
        sid = chat_resp.json()["session_id"]
        resp = client.delete(f"/api/sessions/{sid}")
        assert resp.status_code == 204
        resp = client.get(f"/api/sessions/{sid}")
        assert resp.status_code == 404

    def test_delete_nonexistent_session_returns_404(self, client):
        resp = client.delete("/api/sessions/nonexistent")
        assert resp.status_code == 404


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


class TestMetricsEndpoints:
    """Metrics aggregation endpoints."""

    def test_get_metrics_for_session(self, client):
        chat_resp = client.post(
            "/api/chat",
            json={"query": "Hi", "content": "Content.", "mode": "direct"},
        )
        sid = chat_resp.json()["session_id"]
        resp = client.get(f"/api/metrics/{sid}")
        assert resp.status_code == 200
        data = resp.json()
        assert data["session_id"] == sid
        assert "summary" in data
        assert "by_mode" in data
        assert "timeline" in data

    def test_get_metrics_nonexistent_session_returns_404(self, client):
        resp = client.get("/api/metrics/no-such-session")
        assert resp.status_code == 404


# ---------------------------------------------------------------------------
# Traces
# ---------------------------------------------------------------------------


class TestTraceEndpoints:
    """Execution trace endpoints."""

    def test_get_trace_for_execution(self, client):
        chat_resp = client.post(
            "/api/chat",
            json={"query": "Hello", "content": "Content.", "mode": "direct"},
        )
        exec_id = chat_resp.json()["execution_id"]
        resp = client.get(f"/api/traces/{exec_id}")
        assert resp.status_code == 200
        data = resp.json()
        assert data["execution_id"] == exec_id
        assert data["query"] == "Hello"
        assert data["mode"] == "direct"
        assert data["status"] in ("running", "complete", "error")
        assert "steps" in data
        assert "budget" in data
        assert "result" in data

    def test_get_trace_nonexistent_returns_404(self, client):
        resp = client.get("/api/traces/no-such-execution")
        assert resp.status_code == 404


# ---------------------------------------------------------------------------
# Providers
# ---------------------------------------------------------------------------


class TestProviderEndpoints:
    """Provider listing and test endpoints."""

    def test_list_providers(self, client):
        resp = client.get("/api/providers")
        assert resp.status_code == 200
        providers = resp.json()
        assert isinstance(providers, list)
        assert len(providers) >= 3  # openai, anthropic, ollama
        names = [p["name"] for p in providers]
        assert "openai" in names
        assert "anthropic" in names
        assert "ollama" in names
        for p in providers:
            assert "display_name" in p
            assert "status" in p
            assert "configured" in p
            assert isinstance(p["models"], list)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


class TestConfigEndpoints:
    """Configuration get/update endpoints."""

    def test_get_config_defaults(self, client):
        resp = client.get("/api/config")
        assert resp.status_code == 200
        data = resp.json()
        assert data["active_provider"] == "openai"
        assert data["active_model"] == "gpt-4o"
        assert "budget" in data
        assert data["budget"]["max_steps"] == 16
        assert "sandbox" in data
        assert "appearance" in data

    def test_update_config_partial(self, client):
        """PUT /api/config no longer sets active_provider/model (use providers endpoint)."""
        resp = client.put(
            "/api/config",
            json={"active_provider": "anthropic"},
        )
        assert resp.status_code == 200
        data = resp.json()
        # active_provider is only set via PUT /api/providers/{name}
        assert data["active_provider"] == "openai"
        assert data["active_model"] == "gpt-4o"

    def test_update_config_budget(self, client):
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
        assert resp.json()["budget"]["max_steps"] == 32

    def test_update_config_appearance(self, client):
        resp = client.put(
            "/api/config",
            json={"appearance": {"theme": "dark", "sidebar_collapsed": True}},
        )
        assert resp.status_code == 200
        assert resp.json()["appearance"]["theme"] == "dark"
        assert resp.json()["appearance"]["sidebar_collapsed"] is True

    def test_config_persists_across_requests(self, client):
        """Budget changes via PUT /api/config persist across requests."""
        client.put("/api/config", json={"budget": {"max_steps": 99}})
        resp = client.get("/api/config")
        assert resp.json()["budget"]["max_steps"] == 99

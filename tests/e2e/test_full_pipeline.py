"""E2E tests for full application pipelines.

Tests exercise multi-step workflows that cross API boundaries:
upload -> chat -> session -> metrics -> trace.
"""

from __future__ import annotations

import pytest

from rlmkit.server.dependencies import get_state

pytestmark = [pytest.mark.e2e]


# ---------------------------------------------------------------------------
# Upload -> Query pipeline
# ---------------------------------------------------------------------------


class TestUploadThenQuery:
    """Upload a document, then reference it in a chat query."""

    def test_upload_then_chat_with_file_id(self, client):
        # 1. Upload a text file
        upload_resp = client.post(
            "/api/files/upload",
            files={"file": ("doc.txt", b"RLM uses recursive code execution to explore documents.", "text/plain")},
        )
        assert upload_resp.status_code == 201
        file_id = upload_resp.json()["id"]

        # 2. Submit a chat referencing the file
        chat_resp = client.post(
            "/api/chat",
            json={"query": "What does this document describe?", "file_id": file_id, "mode": "direct"},
        )
        assert chat_resp.status_code == 202
        data = chat_resp.json()
        assert "execution_id" in data
        assert "session_id" in data

        # 3. Verify the file is retrievable
        file_resp = client.get(f"/api/files/{file_id}")
        assert file_resp.status_code == 200
        assert file_resp.json()["name"] == "doc.txt"

    def test_upload_then_trace_created(self, client):
        # Upload
        upload_resp = client.post(
            "/api/files/upload",
            files={"file": ("notes.txt", b"Testing trace creation.", "text/plain")},
        )
        file_id = upload_resp.json()["id"]

        # Chat
        chat_resp = client.post(
            "/api/chat",
            json={"query": "Summarize", "file_id": file_id, "mode": "direct"},
        )
        exec_id = chat_resp.json()["execution_id"]

        # Trace should exist
        trace_resp = client.get(f"/api/traces/{exec_id}")
        assert trace_resp.status_code == 200
        assert trace_resp.json()["query"] == "Summarize"


# ---------------------------------------------------------------------------
# Session lifecycle
# ---------------------------------------------------------------------------


class TestSessionLifecycle:
    """Create sessions via chat, list them, inspect, then delete."""

    def test_full_session_lifecycle(self, client):
        # 1. Submit two queries (creates a session)
        r1 = client.post(
            "/api/chat",
            json={"query": "First question", "content": "Content for analysis.", "mode": "direct"},
        )
        sid = r1.json()["session_id"]

        r2 = client.post(
            "/api/chat",
            json={"query": "Second question", "content": "More content.", "mode": "direct", "session_id": sid},
        )
        assert r2.json()["session_id"] == sid

        # 2. List sessions -- should have exactly one
        list_resp = client.get("/api/sessions")
        assert list_resp.status_code == 200
        sessions = list_resp.json()
        assert len(sessions) == 1
        assert sessions[0]["id"] == sid
        assert sessions[0]["message_count"] >= 2  # at least 2 user messages

        # 3. Get session detail -- should have messages
        detail_resp = client.get(f"/api/sessions/{sid}")
        assert detail_resp.status_code == 200
        messages = detail_resp.json()["messages"]
        assert len(messages) >= 2
        assert messages[0]["role"] == "user"
        assert messages[0]["content"] == "First question"

        # 4. Get metrics for the session
        metrics_resp = client.get(f"/api/metrics/{sid}")
        assert metrics_resp.status_code == 200
        assert metrics_resp.json()["session_id"] == sid

        # 5. Delete the session
        del_resp = client.delete(f"/api/sessions/{sid}")
        assert del_resp.status_code == 204

        # 6. Verify session is gone
        assert client.get(f"/api/sessions/{sid}").status_code == 404
        assert client.get("/api/sessions").json() == []


# ---------------------------------------------------------------------------
# Config -> Chat integration
# ---------------------------------------------------------------------------


class TestConfigIntegration:
    """Verify that config changes are reflected in subsequent operations."""

    def test_change_config_then_chat(self, client):
        # Change budget config
        client.put(
            "/api/config",
            json={"budget": {"max_steps": 8, "max_tokens": 25000, "max_cost_usd": 1.0, "max_time_seconds": 15, "max_recursion_depth": 3}},
        )

        # Verify config took effect
        config = client.get("/api/config").json()
        assert config["budget"]["max_steps"] == 8

        # Submit a chat (should use updated config internally)
        chat_resp = client.post(
            "/api/chat",
            json={"query": "Test with new budget", "content": "Some content.", "mode": "direct"},
        )
        assert chat_resp.status_code == 202

        # Trace should reflect the budget limit from config
        exec_id = chat_resp.json()["execution_id"]
        trace_resp = client.get(f"/api/traces/{exec_id}")
        assert trace_resp.status_code == 200
        assert trace_resp.json()["budget"]["steps_limit"] == 8

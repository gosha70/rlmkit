"""E2E tests for REST API endpoints.

All tests are skipped until the FastAPI backend (Bet 2.1a) is implemented.
Each test documents the expected endpoint contract so the backend team
can enable tests as features land.
"""

import pytest


pytestmark = [
    pytest.mark.e2e,
    pytest.mark.skip(reason="Waiting for FastAPI server (Bet 2.1a)"),
]


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------


class TestHealthEndpoint:
    """GET /health — server liveness check."""

    def test_health_endpoint(self, api_base_url):
        """Health endpoint returns 200 with status ok."""
        # response = client.get(f"{api_base_url}/health")
        # assert response.status_code == 200
        # assert response.json()["status"] == "ok"


# ---------------------------------------------------------------------------
# Chat
# ---------------------------------------------------------------------------


class TestChatEndpoints:
    """POST /api/chat — submit a query against uploaded content."""

    def test_chat_submit_query(self, api_base_url, sample_document):
        """Submitting a query returns a structured response with answer and metrics."""
        # payload = {
        #     "content": sample_document,
        #     "query": "What is RLM?",
        #     "mode": "direct",
        # }
        # response = client.post(f"{api_base_url}/api/chat", json=payload)
        # assert response.status_code == 200
        # data = response.json()
        # assert "answer" in data
        # assert "mode_used" in data
        # assert data["success"] is True


# ---------------------------------------------------------------------------
# File upload
# ---------------------------------------------------------------------------


class TestFileUploadEndpoints:
    """POST /api/upload — upload a document for analysis."""

    def test_file_upload(self, api_base_url):
        """Uploading a text file returns a content hash and metadata."""
        # files = {"file": ("test.txt", b"Hello world content", "text/plain")}
        # response = client.post(f"{api_base_url}/api/upload", files=files)
        # assert response.status_code == 200
        # data = response.json()
        # assert "content_hash" in data
        # assert "char_count" in data


# ---------------------------------------------------------------------------
# Sessions
# ---------------------------------------------------------------------------


class TestSessionEndpoints:
    """Session management endpoints."""

    def test_list_sessions(self, api_base_url):
        """GET /api/sessions returns a list of conversation sessions."""
        # response = client.get(f"{api_base_url}/api/sessions")
        # assert response.status_code == 200
        # assert isinstance(response.json(), list)

    def test_get_session_messages(self, api_base_url, session_id):
        """GET /api/sessions/{id}/messages returns message history."""
        # response = client.get(f"{api_base_url}/api/sessions/{session_id}/messages")
        # assert response.status_code in (200, 404)


# ---------------------------------------------------------------------------
# Metrics & Trace
# ---------------------------------------------------------------------------


class TestMetricsEndpoints:
    """Metrics and trace inspection endpoints."""

    def test_get_metrics(self, api_base_url, session_id):
        """GET /api/sessions/{id}/metrics returns token/cost/time metrics."""
        # response = client.get(f"{api_base_url}/api/sessions/{session_id}/metrics")
        # assert response.status_code in (200, 404)
        # if response.status_code == 200:
        #     data = response.json()
        #     assert "total_tokens" in data
        #     assert "total_cost" in data

    def test_get_trace(self, api_base_url, session_id):
        """GET /api/sessions/{id}/trace returns execution trace steps."""
        # response = client.get(f"{api_base_url}/api/sessions/{session_id}/trace")
        # assert response.status_code in (200, 404)
        # if response.status_code == 200:
        #     data = response.json()
        #     assert isinstance(data.get("steps"), list)


# ---------------------------------------------------------------------------
# Provider configuration
# ---------------------------------------------------------------------------


class TestProviderEndpoints:
    """Provider listing and health-check endpoints."""

    def test_provider_list(self, api_base_url):
        """GET /api/providers returns available LLM providers."""
        # response = client.get(f"{api_base_url}/api/providers")
        # assert response.status_code == 200
        # providers = response.json()
        # assert isinstance(providers, list)
        # assert any(p["name"] == "litellm" for p in providers)

    def test_provider_health_check(self, api_base_url):
        """POST /api/providers/health tests connectivity to a provider."""
        # payload = {"provider": "mock", "model": "mock-model"}
        # response = client.post(f"{api_base_url}/api/providers/health", json=payload)
        # assert response.status_code == 200
        # assert response.json()["healthy"] is True


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


class TestConfigEndpoints:
    """Configuration get/update endpoints."""

    def test_config_get_and_update(self, api_base_url):
        """GET/PUT /api/config reads and writes runtime configuration."""
        # # Read current config
        # response = client.get(f"{api_base_url}/api/config")
        # assert response.status_code == 200
        # config = response.json()
        # assert "max_steps" in config
        #
        # # Update a setting
        # response = client.put(
        #     f"{api_base_url}/api/config",
        #     json={"max_steps": 10},
        # )
        # assert response.status_code == 200
        #
        # # Verify the update
        # response = client.get(f"{api_base_url}/api/config")
        # assert response.json()["max_steps"] == 10

"""E2E tests for full application pipelines.

All tests are skipped until the FastAPI backend and frontend are functional.
These tests exercise multi-step workflows that cross API boundaries.
"""

import pytest


pytestmark = [
    pytest.mark.e2e,
    pytest.mark.skip(reason="Waiting for full stack (Bets 2.1a + 2.1b)"),
]


# ---------------------------------------------------------------------------
# Upload -> Query pipeline
# ---------------------------------------------------------------------------


class TestUploadThenQuery:
    """Upload a document, then run queries against it."""

    def test_upload_then_query_rlm(self, api_base_url, large_document):
        """Upload content, then query it in RLM mode and get a traced answer."""
        # # 1. Upload document
        # upload_resp = client.post(
        #     f"{api_base_url}/api/upload",
        #     files={"file": ("doc.txt", large_document.encode(), "text/plain")},
        # )
        # assert upload_resp.status_code == 200
        # content_hash = upload_resp.json()["content_hash"]
        #
        # # 2. Submit RLM query referencing the uploaded content
        # query_resp = client.post(
        #     f"{api_base_url}/api/chat",
        #     json={
        #         "content_hash": content_hash,
        #         "query": "What are the main topics covered?",
        #         "mode": "rlm",
        #     },
        # )
        # assert query_resp.status_code == 200
        # data = query_resp.json()
        # assert data["success"] is True
        # assert data["mode_used"] == "rlm"
        # assert len(data.get("trace", [])) >= 1


# ---------------------------------------------------------------------------
# Comparison mode
# ---------------------------------------------------------------------------


class TestComparisonMode:
    """Run comparison mode and verify both strategies produce results."""

    def test_comparison_mode(self, api_base_url, sample_document):
        """Compare mode runs both direct and RLM, returning results for each."""
        # response = client.post(
        #     f"{api_base_url}/api/chat",
        #     json={
        #         "content": sample_document,
        #         "query": "Summarize the key points.",
        #         "mode": "compare",
        #     },
        # )
        # assert response.status_code == 200
        # data = response.json()
        # assert "direct" in data["results"]
        # assert "rlm" in data["results"]
        # assert data["results"]["direct"]["success"] is True
        # assert data["results"]["rlm"]["success"] is True


# ---------------------------------------------------------------------------
# Session persistence
# ---------------------------------------------------------------------------


class TestSessionPersistence:
    """Verify that conversation history persists across requests."""

    def test_session_persistence(self, api_base_url, sample_document):
        """Messages sent in one request appear in session history."""
        # # 1. Create a session with a first query
        # r1 = client.post(
        #     f"{api_base_url}/api/chat",
        #     json={
        #         "content": sample_document,
        #         "query": "What is RLM?",
        #         "mode": "direct",
        #     },
        # )
        # assert r1.status_code == 200
        # session_id = r1.json()["session_id"]
        #
        # # 2. Send a follow-up query in the same session
        # r2 = client.post(
        #     f"{api_base_url}/api/chat",
        #     json={
        #         "session_id": session_id,
        #         "query": "Tell me more about budget management.",
        #         "mode": "direct",
        #     },
        # )
        # assert r2.status_code == 200
        #
        # # 3. Fetch session messages
        # r3 = client.get(f"{api_base_url}/api/sessions/{session_id}/messages")
        # assert r3.status_code == 200
        # messages = r3.json()
        # assert len(messages) >= 2  # At least 2 Q&A pairs

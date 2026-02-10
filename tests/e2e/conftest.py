"""Shared fixtures for E2E tests.

Provides TestClient, state reset, and test data helpers for end-to-end
testing of the FastAPI backend.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from rlmkit.server.app import app
from rlmkit.server.dependencies import get_state, reset_state


# ---------------------------------------------------------------------------
# Client and state management
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _clean_state():
    """Reset in-memory state before and after every test."""
    reset_state()
    yield
    reset_state()


@pytest.fixture()
def client():
    """FastAPI TestClient backed by the real application."""
    return TestClient(app)


# ---------------------------------------------------------------------------
# Document fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def sample_document():
    """Small document (~3000 chars) suitable for direct-mode queries."""
    return (
        "The Recursive Language Model (RLM) paradigm represents a significant advancement "
        "in how large language models interact with extensive textual content. Unlike "
        "traditional approaches that attempt to fit entire documents into a context window, "
        "RLM treats the content as a variable in a programming environment and allows the "
        "LLM to write code that explores the content programmatically.\n\n"
    ) * 10


@pytest.fixture()
def large_document():
    """Large document (~50000 chars) suitable for RLM-mode queries."""
    sections = []
    topics = [
        ("Introduction", "This chapter covers the foundational concepts."),
        ("Architecture", "The system follows a hexagonal architecture pattern."),
        ("Recursion", "Sub-agents can spawn their own child agents for focused tasks."),
        ("Budget Management", "Token, cost, and step budgets prevent runaway execution."),
        ("Evaluation", "Automated evaluation compares RLM against direct and RAG baselines."),
    ]
    for i in range(100):
        title, body = topics[i % len(topics)]
        sections.append(
            f"Section {i + 1}: {title}\n"
            f"{body} This section provides detailed analysis of {title.lower()} "
            f"covering multiple aspects and considerations for production deployment. "
            f"Key findings include improved performance metrics and cost optimization.\n"
        )
    return "\n".join(sections)


# ---------------------------------------------------------------------------
# Helper to upload a text file via the API
# ---------------------------------------------------------------------------


@pytest.fixture()
def uploaded_file_id(client):
    """Upload a small text file and return its file ID."""
    resp = client.post(
        "/api/files/upload",
        files={"file": ("test.txt", b"Hello world content for testing.", "text/plain")},
    )
    assert resp.status_code == 201
    return resp.json()["id"]

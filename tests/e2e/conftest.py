"""Shared fixtures for E2E tests.

These fixtures provide test data and client helpers for end-to-end testing
of the FastAPI backend and WebSocket streaming once they are built.
"""

import pytest


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
# API configuration fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def api_base_url():
    """Base URL for the FastAPI server under test."""
    return "http://localhost:8000"


@pytest.fixture()
def ws_base_url():
    """WebSocket URL for the streaming endpoint under test."""
    return "ws://localhost:8000/ws"


# ---------------------------------------------------------------------------
# Session management helpers
# ---------------------------------------------------------------------------


@pytest.fixture()
def session_id():
    """A deterministic session ID for test isolation."""
    return "test-session-e2e-001"


@pytest.fixture()
def auth_headers():
    """Default authentication headers (placeholder for future auth)."""
    return {"Authorization": "Bearer test-token", "Content-Type": "application/json"}

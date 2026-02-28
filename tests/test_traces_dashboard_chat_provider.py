"""Acceptance tests for Chat Provider tracking on Traces and Dashboard."""

from __future__ import annotations

from collections.abc import Generator
from datetime import datetime, timezone
from typing import Any

import pytest
from fastapi.testclient import TestClient

from rlmkit.server.app import create_app
from rlmkit.server.dependencies import ExecutionRecord, SessionRecord, get_state, reset_state
from rlmkit.server.models import MetricsResponse, ProviderSummary, TraceResponse

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _clean_state() -> Generator[None, None, None]:
    reset_state()
    yield
    reset_state()


@pytest.fixture
def client() -> TestClient:
    return TestClient(create_app())


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _make_execution(
    execution_id: str = "exec-1",
    session_id: str = "sess-1",
    query: str = "What is 2+2?",
    mode: str = "direct",
    chat_provider_id: str | None = None,
    chat_provider_name: str | None = None,
    status: str = "complete",
    total_tokens: int = 100,
    total_cost: float = 0.01,
) -> ExecutionRecord:
    now = _now()
    return ExecutionRecord(
        execution_id=execution_id,
        session_id=session_id,
        query=query,
        mode=mode,
        status=status,
        started_at=now,
        completed_at=now,
        result={
            "answer": "4",
            "success": True,
            "total_tokens": total_tokens,
            "total_cost": total_cost,
            "error": None,
        },
        steps=[],
        chat_provider_id=chat_provider_id,
        chat_provider_name=chat_provider_name,
    )


def _make_assistant_message(
    provider: str = "openai",
    chat_provider_name: str | None = None,
    mode_used: str = "direct",
    total_tokens: int = 100,
    cost_usd: float = 0.01,
    elapsed_seconds: float = 1.0,
) -> dict[str, Any]:
    return {
        "id": "msg-1",
        "role": "assistant",
        "content": "Answer",
        "mode_used": mode_used,
        "provider": provider,
        "chat_provider_name": chat_provider_name,
        "metrics": {
            "input_tokens": 80,
            "output_tokens": 20,
            "total_tokens": total_tokens,
            "cost_usd": cost_usd,
            "elapsed_seconds": elapsed_seconds,
            "steps": 1,
        },
        "timestamp": _now().isoformat(),
    }


def _make_session(
    session_id: str = "sess-1", messages: list[dict[str, Any]] | None = None
) -> SessionRecord:
    now = _now()
    return SessionRecord(
        id=session_id,
        name="Test Session",
        created_at=now,
        updated_at=now,
        messages=messages or [],
    )


# ---------------------------------------------------------------------------
# Test 1: ExecutionRecord stores chat_provider_id/name
# ---------------------------------------------------------------------------


class TestExecutionRecordFields:
    def test_stores_chat_provider_id_and_name(self) -> None:
        rec = _make_execution(
            chat_provider_id="cp-1",
            chat_provider_name="DIRECT-CLAUDE",
        )
        assert rec.chat_provider_id == "cp-1"
        assert rec.chat_provider_name == "DIRECT-CLAUDE"

    def test_defaults_to_none_when_not_set(self) -> None:
        rec = _make_execution()
        assert rec.chat_provider_id is None
        assert rec.chat_provider_name is None


# ---------------------------------------------------------------------------
# Test 2: GET /api/executions returns chat_provider_id/name
# ---------------------------------------------------------------------------


class TestListExecutions:
    def test_returns_chat_provider_fields(self, client: TestClient) -> None:
        state = get_state()
        state.executions["exec-1"] = _make_execution(
            execution_id="exec-1",
            session_id="sess-1",
            chat_provider_id="cp-1",
            chat_provider_name="DIRECT-CLAUDE",
        )

        resp = client.get("/api/executions")

        assert resp.status_code == 200
        data = resp.json()
        assert len(data) == 1
        assert data[0]["execution_id"] == "exec-1"
        assert data[0]["chat_provider_id"] == "cp-1"
        assert data[0]["chat_provider_name"] == "DIRECT-CLAUDE"

    def test_null_when_no_chat_provider(self, client: TestClient) -> None:
        state = get_state()
        state.executions["exec-2"] = _make_execution(execution_id="exec-2")

        resp = client.get("/api/executions")

        assert resp.status_code == 200
        data = resp.json()
        assert data[0]["chat_provider_id"] is None
        assert data[0]["chat_provider_name"] is None


# ---------------------------------------------------------------------------
# Test 3: GET /api/traces/{id} returns chat_provider_id/name
# ---------------------------------------------------------------------------


class TestGetTrace:
    def test_returns_chat_provider_fields(self, client: TestClient) -> None:
        state = get_state()
        state.executions["exec-1"] = _make_execution(
            execution_id="exec-1",
            session_id="sess-1",
            chat_provider_id="cp-1",
            chat_provider_name="DIRECT-CLAUDE",
        )

        resp = client.get("/api/traces/exec-1")

        assert resp.status_code == 200
        data = resp.json()
        assert data["execution_id"] == "exec-1"
        assert data["chat_provider_id"] == "cp-1"
        assert data["chat_provider_name"] == "DIRECT-CLAUDE"

    def test_404_for_unknown_execution(self, client: TestClient) -> None:
        resp = client.get("/api/traces/nonexistent")
        assert resp.status_code == 404

    def test_null_when_no_chat_provider(self, client: TestClient) -> None:
        state = get_state()
        state.executions["exec-3"] = _make_execution(execution_id="exec-3")

        resp = client.get("/api/traces/exec-3")

        assert resp.status_code == 200
        data = resp.json()
        assert data["chat_provider_id"] is None
        assert data["chat_provider_name"] is None


# ---------------------------------------------------------------------------
# Test 4: GET /api/metrics/{session_id} returns by_chat_provider
# ---------------------------------------------------------------------------


class TestMetricsByChatProvider:
    def test_by_chat_provider_populated(self, client: TestClient) -> None:
        state = get_state()
        state.sessions["sess-1"] = _make_session(
            session_id="sess-1",
            messages=[
                _make_assistant_message(
                    provider="anthropic",
                    chat_provider_name="DIRECT-CLAUDE",
                    total_tokens=200,
                    cost_usd=0.02,
                    elapsed_seconds=1.5,
                ),
            ],
        )

        resp = client.get("/api/metrics/sess-1")

        assert resp.status_code == 200
        data = resp.json()
        assert "by_chat_provider" in data
        by_cp = data["by_chat_provider"]
        assert "DIRECT-CLAUDE" in by_cp
        cp_data = by_cp["DIRECT-CLAUDE"]
        assert cp_data["queries"] == 1
        assert cp_data["total_tokens"] == 200
        assert abs(cp_data["total_cost_usd"] - 0.02) < 1e-6
        assert cp_data["avg_latency_seconds"] == 1.5


# ---------------------------------------------------------------------------
# Test 5: Two Chat Providers on same LLM provider → separate by_chat_provider,
#          merged in by_provider
# ---------------------------------------------------------------------------


class TestSameLLMProviderTwoChatProviders:
    def test_separate_by_chat_provider_merged_by_provider(self, client: TestClient) -> None:
        state = get_state()
        state.sessions["sess-2"] = _make_session(
            session_id="sess-2",
            messages=[
                {
                    **_make_assistant_message(
                        provider="anthropic",
                        chat_provider_name="DIRECT-CLAUDE",
                        total_tokens=100,
                        cost_usd=0.01,
                        elapsed_seconds=1.0,
                    ),
                    "id": "msg-a",
                },
                {
                    **_make_assistant_message(
                        provider="anthropic",
                        chat_provider_name="RLM-CLAUDE",
                        total_tokens=300,
                        cost_usd=0.03,
                        elapsed_seconds=3.0,
                    ),
                    "id": "msg-b",
                },
            ],
        )

        resp = client.get("/api/metrics/sess-2")

        assert resp.status_code == 200
        data = resp.json()

        # by_provider merges both under "anthropic"
        by_provider = data["by_provider"]
        assert len(by_provider) == 1
        assert "anthropic" in by_provider
        assert by_provider["anthropic"]["queries"] == 2
        assert by_provider["anthropic"]["total_tokens"] == 400

        # by_chat_provider separates them
        by_cp = data["by_chat_provider"]
        assert len(by_cp) == 2
        assert "DIRECT-CLAUDE" in by_cp
        assert "RLM-CLAUDE" in by_cp
        assert by_cp["DIRECT-CLAUDE"]["queries"] == 1
        assert by_cp["DIRECT-CLAUDE"]["total_tokens"] == 100
        assert by_cp["RLM-CLAUDE"]["queries"] == 1
        assert by_cp["RLM-CLAUDE"]["total_tokens"] == 300


# ---------------------------------------------------------------------------
# Test 6: Legacy messages without chat_provider_name don't appear in
#          by_chat_provider but do appear in by_provider
# ---------------------------------------------------------------------------


class TestLegacyMessagesWithoutChatProvider:
    def test_appear_in_by_provider_not_by_chat_provider(self, client: TestClient) -> None:
        state = get_state()
        state.sessions["sess-3"] = _make_session(
            session_id="sess-3",
            messages=[
                _make_assistant_message(
                    provider="openai",
                    chat_provider_name=None,
                    total_tokens=150,
                ),
            ],
        )

        resp = client.get("/api/metrics/sess-3")

        assert resp.status_code == 200
        data = resp.json()

        # Appears in by_provider
        assert "openai" in data["by_provider"]
        assert data["by_provider"]["openai"]["queries"] == 1

        # Does NOT appear in by_chat_provider
        assert data["by_chat_provider"] == {}

    def test_mixed_legacy_and_chat_provider_messages(self, client: TestClient) -> None:
        state = get_state()
        state.sessions["sess-4"] = _make_session(
            session_id="sess-4",
            messages=[
                {
                    **_make_assistant_message(
                        provider="openai",
                        chat_provider_name=None,
                        total_tokens=100,
                    ),
                    "id": "msg-legacy",
                },
                {
                    **_make_assistant_message(
                        provider="anthropic",
                        chat_provider_name="DIRECT-CLAUDE",
                        total_tokens=200,
                    ),
                    "id": "msg-new",
                },
            ],
        )

        resp = client.get("/api/metrics/sess-4")

        assert resp.status_code == 200
        data = resp.json()

        # by_provider has both providers
        assert len(data["by_provider"]) == 2
        assert data["by_provider"]["openai"]["queries"] == 1
        assert data["by_provider"]["anthropic"]["queries"] == 1

        # by_chat_provider has only the one with chat_provider_name
        assert len(data["by_chat_provider"]) == 1
        assert "DIRECT-CLAUDE" in data["by_chat_provider"]


# ---------------------------------------------------------------------------
# Test 7: TraceResponse model accepts and serializes chat_provider_id/name
# ---------------------------------------------------------------------------


class TestTraceResponseModel:
    def test_accepts_chat_provider_fields(self) -> None:
        tr = TraceResponse(
            execution_id="exec-1",
            session_id="sess-1",
            query="test",
            mode="direct",
            chat_provider_id="cp-1",
            chat_provider_name="DIRECT-CLAUDE",
        )
        assert tr.chat_provider_id == "cp-1"
        assert tr.chat_provider_name == "DIRECT-CLAUDE"

    def test_serializes_chat_provider_fields(self) -> None:
        tr = TraceResponse(
            execution_id="exec-1",
            session_id="sess-1",
            query="test",
            mode="direct",
            chat_provider_id="cp-99",
            chat_provider_name="RLM-GPT4",
        )
        d = tr.model_dump()
        assert d["chat_provider_id"] == "cp-99"
        assert d["chat_provider_name"] == "RLM-GPT4"

    def test_chat_provider_fields_default_to_none(self) -> None:
        tr = TraceResponse(
            execution_id="exec-1",
            session_id="sess-1",
            query="test",
            mode="direct",
        )
        assert tr.chat_provider_id is None
        assert tr.chat_provider_name is None


# ---------------------------------------------------------------------------
# Test 8: MetricsResponse model has by_chat_provider field
# ---------------------------------------------------------------------------


class TestMetricsResponseModel:
    def test_has_by_chat_provider_field(self) -> None:
        mr = MetricsResponse(session_id="sess-1")
        assert hasattr(mr, "by_chat_provider")
        assert isinstance(mr.by_chat_provider, dict)
        assert mr.by_chat_provider == {}

    def test_accepts_provider_summary_values(self) -> None:
        mr = MetricsResponse(
            session_id="sess-1",
            by_chat_provider={
                "DIRECT-CLAUDE": ProviderSummary(queries=3, total_tokens=500),
            },
        )
        assert "DIRECT-CLAUDE" in mr.by_chat_provider
        assert mr.by_chat_provider["DIRECT-CLAUDE"].queries == 3
        assert mr.by_chat_provider["DIRECT-CLAUDE"].total_tokens == 500

    def test_serializes_by_chat_provider(self) -> None:
        mr = MetricsResponse(
            session_id="sess-1",
            by_chat_provider={
                "RLM-GPT4": ProviderSummary(queries=1, total_tokens=100, total_cost_usd=0.05),
            },
        )
        d = mr.model_dump()
        assert "by_chat_provider" in d
        assert "RLM-GPT4" in d["by_chat_provider"]
        assert d["by_chat_provider"]["RLM-GPT4"]["queries"] == 1

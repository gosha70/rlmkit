"""Targeted tests to bring coverage above the 80% threshold.

Covers the specific gaps introduced in the latest session:
- server/dependencies.py: _get_instance_api_key (keyring branch + env-var fallback)
- server/routes/evaluations.py: DELETE /evaluations/{session_id} and GET summary
- server/quality.py: get_recommendation with judge scores (sort key logic)
- infrastructure/sandbox/restricted_sandbox.py: set_variable("P", ...) tool rebinding
"""

from __future__ import annotations

import os
from collections.abc import Generator
from datetime import datetime, timezone
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from rlmkit.server.app import create_app
from rlmkit.server.dependencies import (
    AppState,
    SessionRecord,
    get_state,
    reset_state,
    _get_instance_api_key,
)


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


def _make_session(state: AppState, session_id: str = "sess-1") -> SessionRecord:
    now = datetime.now(timezone.utc)
    rec = SessionRecord(
        id=session_id,
        name="Test Session",
        created_at=now,
        updated_at=now,
    )
    state.sessions[session_id] = rec
    return rec


# ---------------------------------------------------------------------------
# _get_instance_api_key — dependencies.py lines 41, 45, 46
# ---------------------------------------------------------------------------


class TestGetInstanceApiKey:
    def test_returns_key_from_store(self) -> None:
        """When a key is stored in the secret store, it is returned immediately."""
        mock_store = MagicMock()
        mock_store.get.return_value = "stored-api-key-123"

        with (
            patch("rlmkit.server.dependencies.KeyringSecretStore.is_available", return_value=True),
            patch("rlmkit.server.dependencies.KeyringSecretStore", return_value=mock_store),
        ):
            result = _get_instance_api_key("lp-uuid-1", "openai")

        assert result == "stored-api-key-123"
        mock_store.get.assert_called_once_with("llm_provider:lp-uuid-1")

    def test_falls_back_to_env_var(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """When no stored key, falls back to catalog env-var."""
        monkeypatch.setenv("OPENAI_API_KEY", "env-key-xyz")
        mock_store = MagicMock()
        mock_store.get.return_value = None  # Nothing in store

        with (
            patch("rlmkit.server.dependencies.KeyringSecretStore.is_available", return_value=True),
            patch("rlmkit.server.dependencies.KeyringSecretStore", return_value=mock_store),
        ):
            result = _get_instance_api_key("lp-uuid-2", "openai")

        assert result == "env-key-xyz"

    def test_returns_none_for_unknown_backend(self) -> None:
        """When backend has no env-var in catalog, returns None."""
        mock_store = MagicMock()
        mock_store.get.return_value = None

        with (
            patch("rlmkit.server.dependencies.KeyringSecretStore.is_available", return_value=False),
            patch("rlmkit.server.dependencies.FileSecretStore", return_value=mock_store),
        ):
            result = _get_instance_api_key("lp-uuid-3", "unknown-backend-xyz")

        assert result is None

    def test_uses_file_store_when_keyring_unavailable(self) -> None:
        """Falls back to FileSecretStore when KeyringSecretStore is not available."""
        mock_store = MagicMock()
        mock_store.get.return_value = "file-stored-key"

        with (
            patch("rlmkit.server.dependencies.KeyringSecretStore.is_available", return_value=False),
            patch("rlmkit.server.dependencies.FileSecretStore", return_value=mock_store),
        ):
            result = _get_instance_api_key("lp-uuid-4", "anthropic")

        assert result == "file-stored-key"


# ---------------------------------------------------------------------------
# DELETE /api/evaluations/{session_id} — evaluations.py lines 114-124
# ---------------------------------------------------------------------------


class TestResetSessionEvaluations:
    def _populate_evaluations(self, state: AppState, session_id: str) -> None:
        state.evaluations["thumb_ratings"].append(
            {
                "id": "r1",
                "execution_id": "e1",
                "session_id": session_id,
                "rating": "up",
                "chat_provider_id": "cp1",
            }
        )
        state.evaluations["best_picks"].append(
            {"id": "p1", "session_id": session_id, "winner_execution_id": "e1"}
        )
        state.evaluations["judge_scores"].append(
            {"id": "j1", "session_id": session_id, "chat_provider_id": "cp1", "overall_score": 4.0}
        )
        state.evaluations["judge_pairwise"].append(
            {
                "id": "jj1",
                "session_id": session_id,
                "execution_id_a": "e1",
                "execution_id_b": "e2",
                "winner": "a",
            }
        )

    def test_delete_clears_all_evaluation_types(self, client: TestClient) -> None:
        state = get_state()
        _make_session(state, "sess-del")
        self._populate_evaluations(state, "sess-del")

        resp = client.delete("/api/evaluations/sess-del")
        assert resp.status_code == 204

        for key in ("thumb_ratings", "best_picks", "judge_scores", "judge_pairwise"):
            assert all(e["session_id"] != "sess-del" for e in state.evaluations[key])

    def test_delete_leaves_other_sessions_intact(self, client: TestClient) -> None:
        state = get_state()
        _make_session(state, "sess-a")
        _make_session(state, "sess-b")
        self._populate_evaluations(state, "sess-a")
        self._populate_evaluations(state, "sess-b")

        resp = client.delete("/api/evaluations/sess-a")
        assert resp.status_code == 204

        # sess-b data should remain
        assert any(e["session_id"] == "sess-b" for e in state.evaluations["thumb_ratings"])

    def test_delete_on_empty_session_returns_204(self, client: TestClient) -> None:
        """Deleting when there's nothing to delete is idempotent."""
        resp = client.delete("/api/evaluations/no-such-session")
        assert resp.status_code == 204


# ---------------------------------------------------------------------------
# GET /api/evaluations/{session_id}/summary — evaluations.py lines 127-143
# ---------------------------------------------------------------------------


class TestGetEvaluationSummary:
    def test_summary_empty_session(self, client: TestClient) -> None:
        state = get_state()
        _make_session(state, "sess-empty")

        resp = client.get("/api/evaluations/sess-empty/summary")
        assert resp.status_code == 200
        data = resp.json()
        assert data["session_id"] == "sess-empty"
        assert isinstance(data["by_chat_provider"], dict)

    def test_summary_with_thumb_ratings(self, client: TestClient) -> None:
        state = get_state()
        _make_session(state, "sess-thumbs")

        # Add thumb ratings for two providers
        for i in range(3):
            state.evaluations["thumb_ratings"].append(
                {
                    "id": f"r{i}",
                    "execution_id": f"e{i}",
                    "session_id": "sess-thumbs",
                    "chat_provider_id": "cp-a",
                    "rating": "up",
                }
            )
        state.evaluations["thumb_ratings"].append(
            {
                "id": "r3",
                "execution_id": "e3",
                "session_id": "sess-thumbs",
                "chat_provider_id": "cp-b",
                "rating": "down",
            }
        )

        resp = client.get("/api/evaluations/sess-thumbs/summary")
        assert resp.status_code == 200
        data = resp.json()
        assert "cp-a" in data["by_chat_provider"]
        assert data["by_chat_provider"]["cp-a"]["thumb_up"] == 3

    def test_summary_includes_recommendation_field(self, client: TestClient) -> None:
        state = get_state()
        _make_session(state, "sess-rec")

        resp = client.get("/api/evaluations/sess-rec/summary")
        assert resp.status_code == 200
        data = resp.json()
        assert "recommendation" in data
        assert "recommendation_reason" in data


# ---------------------------------------------------------------------------
# QualityEngine.get_recommendation — quality.py _sort_key logic
# ---------------------------------------------------------------------------


class TestQualityEngineRecommendation:
    def _make_state_with_scores(
        self,
        providers: dict[str, dict[str, Any]],
    ) -> AppState:
        """Build a minimal AppState with evaluation data for the given providers."""
        state = AppState(load_from_disk=False)
        state.save_config = lambda: None  # type: ignore[assignment]
        state.save_sessions = lambda: None  # type: ignore[assignment]
        state.save_evaluations = lambda: None  # type: ignore[assignment]
        return state

    def test_judge_score_beats_combined_score(self) -> None:
        """Provider with higher judge_avg_score wins over higher combined_score."""
        from rlmkit.server.models import ProviderQualityScore
        from rlmkit.server.quality import QualityEngine

        engine = QualityEngine()
        scores = {
            "cp-rlm": ProviderQualityScore(
                chat_provider_id="cp-rlm",
                chat_provider_name="RLM",
                judge_avg_score=3.0,
                combined_score=0.20,
            ),
            "cp-direct": ProviderQualityScore(
                chat_provider_id="cp-direct",
                chat_provider_name="Direct",
                judge_avg_score=2.0,
                combined_score=0.35,  # higher combined but lower judge
            ),
        }
        provider_id, reason = engine.get_recommendation(scores)
        assert provider_id == "cp-rlm"
        assert "judge score" in reason

    def test_returns_none_when_no_data(self) -> None:
        from rlmkit.server.quality import QualityEngine
        from rlmkit.server.models import ProviderQualityScore

        engine = QualityEngine()
        scores = {
            "cp-1": ProviderQualityScore(
                chat_provider_id="cp-1",
                chat_provider_name="P1",
                judge_avg_score=None,
                combined_score=0.0,
            ),
        }
        provider_id, reason = engine.get_recommendation(scores)
        assert provider_id is None

    def test_returns_none_when_empty_scores(self) -> None:
        from rlmkit.server.quality import QualityEngine

        engine = QualityEngine()
        provider_id, reason = engine.get_recommendation({})
        assert provider_id is None
        assert "No evaluation data" in reason

    def test_winner_by_judge_score_only(self) -> None:
        """When only judge scores exist (no combined score data), winner is highest judge."""
        from rlmkit.server.models import ProviderQualityScore
        from rlmkit.server.quality import QualityEngine

        engine = QualityEngine()
        scores = {
            "cp-a": ProviderQualityScore(
                chat_provider_id="cp-a",
                chat_provider_name="A",
                judge_avg_score=4.5,
                combined_score=0.10,
            ),
            "cp-b": ProviderQualityScore(
                chat_provider_id="cp-b",
                chat_provider_name="B",
                judge_avg_score=3.0,
                combined_score=0.10,
            ),
        }
        provider_id, _ = engine.get_recommendation(scores)
        assert provider_id == "cp-a"

    def test_combined_score_as_tiebreaker(self) -> None:
        """When judge scores are equal, combined_score breaks the tie."""
        from rlmkit.server.models import ProviderQualityScore
        from rlmkit.server.quality import QualityEngine

        engine = QualityEngine()
        scores = {
            "cp-a": ProviderQualityScore(
                chat_provider_id="cp-a",
                chat_provider_name="A",
                judge_avg_score=3.0,
                combined_score=0.30,
            ),
            "cp-b": ProviderQualityScore(
                chat_provider_id="cp-b",
                chat_provider_name="B",
                judge_avg_score=3.0,
                combined_score=0.50,  # tiebreak winner
            ),
        }
        provider_id, _ = engine.get_recommendation(scores)
        assert provider_id == "cp-b"


# ---------------------------------------------------------------------------
# RestrictedSandboxAdapter.set_variable("P", ...) — restricted_sandbox.py 199-204
# ---------------------------------------------------------------------------


class TestRestrictedSandboxPToolRebinding:
    def test_set_P_rebinds_peek(self) -> None:
        """After set_variable('P', content), peek() is rebound to that content."""
        from rlmkit.infrastructure.sandbox.restricted_sandbox import RestrictedSandboxAdapter

        sb = RestrictedSandboxAdapter()
        content = "hello world"
        sb.set_variable("P", content)

        # peek(start, end) — partial(peek, content) so peek(0, 5) = peek(content, 0, 5)
        result = sb.execute("x = peek(0, 5)\nprint(x)")
        assert result.success
        assert "hello" in result.stdout

    def test_set_P_rebinds_grep(self) -> None:
        """After set_variable('P', content), grep() searches within that content."""
        from rlmkit.infrastructure.sandbox.restricted_sandbox import RestrictedSandboxAdapter

        sb = RestrictedSandboxAdapter()
        content = "line one\nline two\nline three"
        sb.set_variable("P", content)

        result = sb.execute("hits = grep('two')\nprint(hits)")
        assert result.success
        assert "two" in result.stdout

    def test_set_P_rebinds_chunk(self) -> None:
        """After set_variable('P', content), chunk() splits that content."""
        from rlmkit.infrastructure.sandbox.restricted_sandbox import RestrictedSandboxAdapter

        sb = RestrictedSandboxAdapter()
        content = "word " * 100
        sb.set_variable("P", content)

        result = sb.execute("parts = chunk()\nprint(len(parts))")
        assert result.success
        assert result.exception is None

    def test_set_P_rebinds_select(self) -> None:
        """After set_variable('P', content), select() slices that content."""
        from rlmkit.infrastructure.sandbox.restricted_sandbox import RestrictedSandboxAdapter

        sb = RestrictedSandboxAdapter()
        content = "abcdefghij"
        sb.set_variable("P", content)

        # select(ranges) — partial(select, content) so select([(0, 3)]) = select(content, [(0, 3)])
        result = sb.execute("s = select([(0, 3)])\nprint(s)")
        assert result.success
        assert "abc" in result.stdout

    def test_set_non_P_does_not_rebind_tools(self) -> None:
        """set_variable with a name other than 'P' does not touch globals."""
        from rlmkit.infrastructure.sandbox.restricted_sandbox import RestrictedSandboxAdapter

        sb = RestrictedSandboxAdapter()
        original_peek = sb._globals["peek"]
        sb.set_variable("Q", "some content")
        assert sb._globals["peek"] is original_peek

    def test_P_variable_accessible_in_code(self) -> None:
        """After set_variable('P', ...), the variable P is usable in executed code."""
        from rlmkit.infrastructure.sandbox.restricted_sandbox import RestrictedSandboxAdapter

        sb = RestrictedSandboxAdapter()
        sb.set_variable("P", "test content here")

        result = sb.execute("print(P[:4])")
        assert result.success
        assert "test" in result.stdout


# ---------------------------------------------------------------------------
# JudgeService.score_pointwise and compare_pairwise — judge.py lines 64-166
# ---------------------------------------------------------------------------


class _FakeLLMResult:
    content: str

    def __init__(self, content: str) -> None:
        self.content = content


class TestJudgeServiceScoring:
    """Test JudgeService async scoring without hitting a real LLM."""

    def _make_judge_state(self) -> AppState:
        """Return an AppState with a fake judge Chat Provider."""
        from datetime import datetime, timezone
        from rlmkit.server.models import ChatProviderConfig, LLMProviderConfig

        state = AppState(load_from_disk=False)
        state.save_config = lambda: None  # type: ignore[assignment]
        state.save_sessions = lambda: None  # type: ignore[assignment]
        state.save_evaluations = lambda: None  # type: ignore[assignment]

        now = datetime.now(timezone.utc)
        lp = LLMProviderConfig(
            id="lp-judge",
            name="Judge LLM",
            backend="openai",
            model="gpt-4",
            created_at=now,
            updated_at=now,
        )
        cp = ChatProviderConfig(
            id="cp-judge",
            name="Judge CP",
            llm_provider_id="lp-judge",
            execution_mode="direct",
            created_at=now,
            updated_at=now,
        )
        state.config.llm_providers.append(lp)
        state.config.chat_providers.append(cp)
        state.config.judge_chat_provider_id = "cp-judge"
        return state

    def _add_execution(
        self, state: AppState, session_id: str, exec_id: str, cp_id: str = "cp-a"
    ) -> None:
        """Inject a session + execution context for testing."""
        now = datetime.now(timezone.utc)
        if session_id not in state.sessions:
            from rlmkit.server.dependencies import SessionRecord

            state.sessions[session_id] = SessionRecord(
                id=session_id, name="Test", created_at=now, updated_at=now
            )
        msg_user = {"id": "u1", "role": "user", "content": "What is X?"}
        msg_asst = {
            "id": "a1",
            "role": "assistant",
            "content": "X is Y.",
            "execution_id": exec_id,
            "chat_provider_id": cp_id,
        }
        state.sessions[session_id].messages.extend([msg_user, msg_asst])

    @pytest.mark.asyncio
    async def test_score_pointwise_returns_judge_score(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """score_pointwise calls the judge adapter and returns a JudgeScore."""
        from unittest.mock import AsyncMock
        from rlmkit.server.judge import JudgeService
        from rlmkit.infrastructure.llm.litellm_adapter import LiteLLMAdapter

        state = self._make_judge_state()
        self._add_execution(state, "sess-j", "exec-j1")

        judge_response = (
            '{"dimensions": {"relevance": 4.0, "correctness": 4.0, '
            '"completeness": 3.0, "coherence": 4.0, "conciseness": 3.0}, '
            '"overall_score": 3.6, "reasoning": "Good answer"}'
        )
        fake_adapter = MagicMock(spec=LiteLLMAdapter)
        fake_adapter.complete_async = AsyncMock(return_value=_FakeLLMResult(judge_response))

        monkeypatch.setattr(
            state, "create_llm_adapter_for_chat_provider", lambda _cp_id, **kw: fake_adapter
        )

        svc = JudgeService(state)
        score = await svc.score_pointwise("exec-j1")

        assert score.overall_score == pytest.approx(3.6)
        assert score.session_id == "sess-j"
        assert score.chat_provider_id == "cp-a"
        # Verify it was persisted
        assert len(state.evaluations["judge_scores"]) == 1

    @pytest.mark.asyncio
    async def test_score_pointwise_handles_parse_failure(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """score_pointwise falls back to default scores on parse failure."""
        from unittest.mock import AsyncMock
        from rlmkit.server.judge import JudgeService

        state = self._make_judge_state()
        self._add_execution(state, "sess-j2", "exec-j2")

        fake_adapter = MagicMock()
        fake_adapter.complete_async = AsyncMock(return_value=_FakeLLMResult("not valid json"))
        monkeypatch.setattr(
            state, "create_llm_adapter_for_chat_provider", lambda _cp_id, **kw: fake_adapter
        )

        svc = JudgeService(state)
        score = await svc.score_pointwise("exec-j2")

        # Fallback score is 3.0
        assert score.overall_score == pytest.approx(3.0)
        assert "Failed to parse" in score.reasoning

    @pytest.mark.asyncio
    async def test_compare_pairwise_returns_winner(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """compare_pairwise debiases and returns final winner."""
        from unittest.mock import AsyncMock
        from rlmkit.server.judge import JudgeService

        state = self._make_judge_state()
        self._add_execution(state, "sess-j3", "exec-pa", cp_id="cp-pa")
        self._add_execution(state, "sess-j3", "exec-pb", cp_id="cp-pb")

        # Both runs agree 'a' wins (position-swapped: run2 says 'b' which maps back to 'a')
        resp_run1 = '{"winner": "a", "reasoning": "A is better"}'
        resp_run2 = (
            '{"winner": "b", "reasoning": "B was worse"}'  # position-swapped → means 'a' wins
        )

        call_count = 0

        async def _fake_complete(messages: list) -> _FakeLLMResult:
            nonlocal call_count
            call_count += 1
            return _FakeLLMResult(resp_run1 if call_count == 1 else resp_run2)

        fake_adapter = MagicMock()
        fake_adapter.complete_async = _fake_complete
        monkeypatch.setattr(
            state, "create_llm_adapter_for_chat_provider", lambda _cp_id, **kw: fake_adapter
        )

        svc = JudgeService(state)
        result = await svc.compare_pairwise("exec-pa", "exec-pb")

        # run1 says 'a', run2 (position-swapped) says 'b' → re-mapped to 'a' → agree on 'a'
        assert result.winner == "a"
        assert len(state.evaluations["judge_pairwise"]) == 1

    @pytest.mark.asyncio
    async def test_compare_pairwise_disagreement_is_tie(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """When runs disagree, final_winner is 'tie'."""
        from unittest.mock import AsyncMock
        from rlmkit.server.judge import JudgeService

        state = self._make_judge_state()
        self._add_execution(state, "sess-j4", "exec-pc", cp_id="cp-pc")
        self._add_execution(state, "sess-j4", "exec-pd", cp_id="cp-pd")

        # run1 says 'a', run2 says 'a' → position-swapped means original 'b' → disagreement
        call_count = 0

        async def _fake_complete(messages: list) -> _FakeLLMResult:
            nonlocal call_count
            call_count += 1
            return _FakeLLMResult('{"winner": "a", "reasoning": "test"}')

        fake_adapter = MagicMock()
        fake_adapter.complete_async = _fake_complete
        monkeypatch.setattr(
            state, "create_llm_adapter_for_chat_provider", lambda _cp_id, **kw: fake_adapter
        )

        svc = JudgeService(state)
        result = await svc.compare_pairwise("exec-pc", "exec-pd")

        # run1='a', run2_raw='a' → position-swapped → 'b' → 'a' vs 'b' → tie
        assert result.winner == "tie"


# ---------------------------------------------------------------------------
# POST /api/evaluations/judge endpoint — evaluations.py lines 171-195
# ---------------------------------------------------------------------------


class TestTriggerJudgeEndpoint:
    def _setup_judge_provider(self, state: AppState) -> None:
        """Configure state with a judge chat provider."""
        from datetime import datetime, timezone
        from rlmkit.server.models import ChatProviderConfig, LLMProviderConfig

        now = datetime.now(timezone.utc)
        lp = LLMProviderConfig(
            id="lp-j",
            name="JudgeLLM",
            backend="openai",
            model="gpt-4",
            created_at=now,
            updated_at=now,
        )
        cp = ChatProviderConfig(
            id="cp-j",
            name="Judge",
            llm_provider_id="lp-j",
            execution_mode="direct",
            created_at=now,
            updated_at=now,
        )
        state.config.llm_providers.append(lp)
        state.config.chat_providers.append(cp)
        state.config.judge_chat_provider_id = "cp-j"

    def test_trigger_judge_no_provider_returns_400(self, client: TestClient) -> None:
        """Without judge_chat_provider_id, endpoint returns 400."""
        resp = client.post(
            "/api/evaluations/judge",
            json={
                "session_id": "sess-1",
                "execution_ids": ["exec-1", "exec-2"],
                "mode": "pointwise",
            },
        )
        assert resp.status_code == 400
        body = resp.json()
        # Server wraps HTTPException in {"error": {"message": ...}}
        error_msg = body.get("detail") or body.get("error", {}).get("message", "")
        assert "judge Chat Provider" in error_msg

    def test_trigger_judge_pointwise_with_mock(
        self, client: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """trigger_judge runs pointwise scoring and returns results."""
        from unittest.mock import AsyncMock, MagicMock
        from rlmkit.server.models import JudgeScore

        state = get_state()
        self._setup_judge_provider(state)

        # Add session with messages so resolve_execution_context works
        now = datetime.now(timezone.utc)
        from rlmkit.server.dependencies import SessionRecord

        state.sessions["sess-tj"] = SessionRecord(
            id="sess-tj", name="TJ", created_at=now, updated_at=now
        )
        state.sessions["sess-tj"].messages = [
            {"id": "u1", "role": "user", "content": "What?"},
            {
                "id": "a1",
                "role": "assistant",
                "content": "Answer.",
                "execution_id": "exec-tj1",
                "chat_provider_id": "cp-test",
            },
        ]

        fake_score = JudgeScore(
            id="score-1",
            execution_id="exec-tj1",
            session_id="sess-tj",
            chat_provider_id="cp-test",
            judge_provider_id="cp-j",
            dimensions={
                "relevance": 4.0,
                "correctness": 4.0,
                "completeness": 4.0,
                "coherence": 4.0,
                "conciseness": 4.0,
            },
            overall_score=4.0,
            reasoning="Good",
            created_at=now,
        )

        _captured_score = fake_score

        import rlmkit.server.judge as judge_module

        class _FakeJudgeService:
            def __init__(self, _state: Any) -> None:
                pass

            async def score_pointwise(self, exec_id: str) -> JudgeScore:
                return _captured_score

        monkeypatch.setattr(judge_module, "JudgeService", _FakeJudgeService)

        resp = client.post(
            "/api/evaluations/judge",
            json={
                "session_id": "sess-tj",
                "execution_ids": ["exec-tj1"],
                "mode": "pointwise",
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "pointwise" in data
        assert len(data["pointwise"]) == 1
        assert data["pointwise"][0]["overall_score"] == pytest.approx(4.0)

    def test_trigger_judge_pairwise_exception_is_caught(
        self, client: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Exceptions in pairwise scoring are caught and logged (not 500)."""
        state = get_state()
        self._setup_judge_provider(state)

        import rlmkit.server.judge as judge_module

        class _FailingJudgeService:
            def __init__(self, _state: Any) -> None:
                pass

            async def score_pointwise(self, exec_id: str) -> None:
                raise RuntimeError("LLM timeout")

            async def compare_pairwise(self, id_a: str, id_b: str) -> None:
                raise RuntimeError("LLM timeout")

        monkeypatch.setattr(judge_module, "JudgeService", _FailingJudgeService)

        resp = client.post(
            "/api/evaluations/judge",
            json={
                "session_id": "sess-fail",
                "execution_ids": ["exec-1", "exec-2"],
                "mode": "both",
            },
        )
        # Exceptions are caught per-item — endpoint still returns 200
        assert resp.status_code == 200
        data = resp.json()
        assert data["pointwise"] == []
        assert data["pairwise"] == []


# ---------------------------------------------------------------------------
# QualityEngine.compute_scores — extra branches (quality.py lines 52-53, 191-196)
# ---------------------------------------------------------------------------


class TestQualityEngineComputeScores:
    """Exercise specific uncovered branches in compute_scores."""

    def test_cost_and_speed_branches_covered(self) -> None:
        """Verify cost/speed score branches run (all_costs/all_speeds > 0)."""
        from rlmkit.server.quality import QualityEngine
        from rlmkit.server.dependencies import AppState, SessionRecord, ExecutionRecord

        state = AppState(load_from_disk=False)
        state.save_config = lambda: None  # type: ignore[assignment]
        state.save_sessions = lambda: None  # type: ignore[assignment]
        state.save_evaluations = lambda: None  # type: ignore[assignment]

        now = datetime.now(timezone.utc)
        session = SessionRecord(id="s-q", name="Q", created_at=now, updated_at=now)
        session.messages = [
            {"id": "u1", "role": "user", "content": "query"},
            {
                "id": "a1",
                "role": "assistant",
                "content": "resp",
                "execution_id": "e-q1",
                "chat_provider_id": "cp-q1",
                "metrics": {"cost_usd": 0.01, "elapsed_seconds": 2.5},
            },
        ]
        state.sessions["s-q"] = session

        # Also add in-memory execution that's NOT in messages (fallback path, lines 52-53)
        state.executions["e-q2"] = ExecutionRecord(
            execution_id="e-q2",
            session_id="s-q",
            query="q2",
            mode="direct",
            chat_provider_id="cp-q1",
            result={"total_cost": 0.02, "elapsed_time": 3.0},
        )

        state.evaluations["thumb_ratings"] = [
            {
                "id": "r1",
                "execution_id": "e-q1",
                "session_id": "s-q",
                "chat_provider_id": "cp-q1",
                "rating": "up",
            },
            {
                "id": "r2",
                "execution_id": "e-q1",
                "session_id": "s-q",
                "chat_provider_id": "cp-q1",
                "rating": "up",
            },
            {
                "id": "r3",
                "execution_id": "e-q1",
                "session_id": "s-q",
                "chat_provider_id": "cp-q1",
                "rating": "up",
            },
        ]

        engine = QualityEngine()
        scores = engine.compute_scores("s-q", state)

        assert "cp-q1" in scores
        # combined_score should be > 0 since there are thumbs + cost/speed data
        assert scores["cp-q1"].combined_score > 0

    def test_no_data_combined_score_is_zero(self) -> None:
        """When a provider has no data at all, combined_score stays 0."""
        from rlmkit.server.quality import QualityEngine
        from rlmkit.server.dependencies import AppState, ExecutionRecord

        state = AppState(load_from_disk=False)
        state.save_config = lambda: None  # type: ignore[assignment]
        state.save_sessions = lambda: None  # type: ignore[assignment]
        state.save_evaluations = lambda: None  # type: ignore[assignment]

        # Only an in-memory execution, no ratings, no cost data
        state.executions["e-bare"] = ExecutionRecord(
            execution_id="e-bare",
            session_id="s-bare",
            query="q",
            mode="direct",
            chat_provider_id="cp-bare",
        )

        engine = QualityEngine()
        scores = engine.compute_scores("s-bare", state)

        assert "cp-bare" in scores
        assert scores["cp-bare"].combined_score == 0.0

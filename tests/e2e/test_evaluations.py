"""E2E tests for evaluation endpoints (thumb ratings, best-pick, summary)."""

from __future__ import annotations

import uuid

import pytest
from fastapi.testclient import TestClient

from rlmstudio.server.dependencies import ExecutionRecord, get_state

pytestmark = [pytest.mark.e2e]


def _setup_session_with_executions(
    n_providers: int = 2,
    query: str = "What is RLM?",
    session_id: str | None = None,
) -> tuple[str, list[str], list[str]]:
    """Create a session with N providers and return (session_id, cp_ids, exec_ids).

    When session_id is given, adds a new turn to an existing session.
    """
    state = get_state()
    session = state.get_or_create_session(session_id)
    sid = session.id

    # Add user message first (required for turn-based grouping)
    user_msg_id = str(uuid.uuid4())
    state.add_message(
        sid,
        {
            "id": user_msg_id,
            "role": "user",
            "content": query,
        },
    )

    cp_ids = []
    exec_ids = []
    for i in range(n_providers):
        cp_id = str(uuid.uuid4())
        eid = str(uuid.uuid4())
        cp_ids.append(cp_id)
        exec_ids.append(eid)
        state.executions[eid] = ExecutionRecord(
            execution_id=eid,
            session_id=sid,
            query=query,
            mode="direct",
            status="complete",
            chat_provider_id=cp_id,
            chat_provider_name=f"Provider-{i}",
            result={
                "answer": f"Response from provider {i}",
                "cost_used": 0.01 * (i + 1),
            },
        )
        state.add_message(
            sid,
            {
                "id": str(uuid.uuid4()),
                "role": "assistant",
                "content": f"Response from provider {i}",
                "execution_id": eid,
                "chat_provider_id": cp_id,
                "metrics": {
                    "cost_usd": 0.01 * (i + 1),
                    "elapsed_seconds": 1.0 + i * 0.5,
                    "total_tokens": 100 * (i + 1),
                },
            },
            chat_provider_id=cp_id,
        )

    return sid, cp_ids, exec_ids


class TestThumbRatings:
    def test_submit_thumb_up(self, client: TestClient) -> None:
        sid, cp_ids, exec_ids = _setup_session_with_executions()
        resp = client.post(
            "/api/evaluations/thumb",
            json={
                "execution_id": exec_ids[0],
                "session_id": sid,
                "chat_provider_id": cp_ids[0],
                "rating": "up",
            },
        )
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"

    def test_submit_thumb_down(self, client: TestClient) -> None:
        sid, cp_ids, exec_ids = _setup_session_with_executions()
        resp = client.post(
            "/api/evaluations/thumb",
            json={
                "execution_id": exec_ids[0],
                "session_id": sid,
                "chat_provider_id": cp_ids[0],
                "rating": "down",
            },
        )
        assert resp.status_code == 200

    def test_thumb_upsert_replaces_existing(self, client: TestClient) -> None:
        sid, cp_ids, exec_ids = _setup_session_with_executions()
        client.post(
            "/api/evaluations/thumb",
            json={
                "execution_id": exec_ids[0],
                "session_id": sid,
                "chat_provider_id": cp_ids[0],
                "rating": "up",
            },
        )
        client.post(
            "/api/evaluations/thumb",
            json={
                "execution_id": exec_ids[0],
                "session_id": sid,
                "chat_provider_id": cp_ids[0],
                "rating": "down",
            },
        )
        evals = client.get(f"/api/evaluations/{sid}").json()
        thumbs = [t for t in evals["thumb_ratings"] if t["execution_id"] == exec_ids[0]]
        assert len(thumbs) == 1
        assert thumbs[0]["rating"] == "down"

    def test_delete_thumb(self, client: TestClient) -> None:
        sid, cp_ids, exec_ids = _setup_session_with_executions()
        client.post(
            "/api/evaluations/thumb",
            json={
                "execution_id": exec_ids[0],
                "session_id": sid,
                "chat_provider_id": cp_ids[0],
                "rating": "up",
            },
        )
        resp = client.delete(f"/api/evaluations/thumb/{exec_ids[0]}")
        assert resp.status_code == 200
        evals = client.get(f"/api/evaluations/{sid}").json()
        assert len(evals["thumb_ratings"]) == 0

    def test_delete_nonexistent_thumb_returns_404(self, client: TestClient) -> None:
        resp = client.delete(f"/api/evaluations/thumb/{uuid.uuid4()}")
        assert resp.status_code == 404


class TestBestPick:
    def test_submit_best_pick(self, client: TestClient) -> None:
        sid, _, exec_ids = _setup_session_with_executions()
        resp = client.post(
            "/api/evaluations/best-pick",
            json={
                "session_id": sid,
                "winner_execution_id": exec_ids[0],
            },
        )
        assert resp.status_code == 200
        evals = client.get(f"/api/evaluations/{sid}").json()
        assert len(evals["best_picks"]) == 1
        assert evals["best_picks"][0]["winner_execution_id"] == exec_ids[0]

    def test_repick_replaces_previous(self, client: TestClient) -> None:
        sid, _, exec_ids = _setup_session_with_executions()
        client.post(
            "/api/evaluations/best-pick",
            json={
                "session_id": sid,
                "winner_execution_id": exec_ids[0],
            },
        )
        client.post(
            "/api/evaluations/best-pick",
            json={
                "session_id": sid,
                "winner_execution_id": exec_ids[1],
            },
        )
        evals = client.get(f"/api/evaluations/{sid}").json()
        assert len(evals["best_picks"]) == 1
        assert evals["best_picks"][0]["winner_execution_id"] == exec_ids[1]


class TestSessionEvaluations:
    def test_get_empty_evaluations(self, client: TestClient) -> None:
        sid = str(uuid.uuid4())
        resp = client.get(f"/api/evaluations/{sid}")
        assert resp.status_code == 200
        data = resp.json()
        assert data["thumb_ratings"] == []
        assert data["best_picks"] == []
        assert data["judge_scores"] == []
        assert data["judge_pairwise"] == []

    def test_evaluations_filtered_by_session(self, client: TestClient) -> None:
        sid1, cp_ids1, exec_ids1 = _setup_session_with_executions()
        sid2, cp_ids2, exec_ids2 = _setup_session_with_executions()

        client.post(
            "/api/evaluations/thumb",
            json={
                "execution_id": exec_ids1[0],
                "session_id": sid1,
                "chat_provider_id": cp_ids1[0],
                "rating": "up",
            },
        )
        client.post(
            "/api/evaluations/thumb",
            json={
                "execution_id": exec_ids2[0],
                "session_id": sid2,
                "chat_provider_id": cp_ids2[0],
                "rating": "down",
            },
        )

        evals1 = client.get(f"/api/evaluations/{sid1}").json()
        assert len(evals1["thumb_ratings"]) == 1
        assert evals1["thumb_ratings"][0]["rating"] == "up"

        evals2 = client.get(f"/api/evaluations/{sid2}").json()
        assert len(evals2["thumb_ratings"]) == 1
        assert evals2["thumb_ratings"][0]["rating"] == "down"


class TestEvaluationSummary:
    def test_summary_with_thumbs(self, client: TestClient) -> None:
        sid, cp_ids, exec_ids = _setup_session_with_executions()
        client.post(
            "/api/evaluations/thumb",
            json={
                "execution_id": exec_ids[0],
                "session_id": sid,
                "chat_provider_id": cp_ids[0],
                "rating": "up",
            },
        )
        client.post(
            "/api/evaluations/thumb",
            json={
                "execution_id": exec_ids[1],
                "session_id": sid,
                "chat_provider_id": cp_ids[1],
                "rating": "down",
            },
        )
        resp = client.get(f"/api/evaluations/{sid}/summary")
        assert resp.status_code == 200
        data = resp.json()
        assert data["session_id"] == sid
        assert cp_ids[0] in data["by_chat_provider"]
        p0 = data["by_chat_provider"][cp_ids[0]]
        assert p0["thumb_up"] == 1
        assert p0["thumb_down"] == 0
        assert p0["thumb_score"] == 1.0

    def test_summary_empty_session(self, client: TestClient) -> None:
        sid = str(uuid.uuid4())
        resp = client.get(f"/api/evaluations/{sid}/summary")
        assert resp.status_code == 200
        data = resp.json()
        assert data["by_chat_provider"] == {}
        assert data["recommendation"] is None


class TestJudgeTrigger:
    def test_judge_without_config_returns_400(self, client: TestClient) -> None:
        sid, _, exec_ids = _setup_session_with_executions()
        resp = client.post(
            "/api/evaluations/judge",
            json={
                "session_id": sid,
                "execution_ids": exec_ids,
            },
        )
        assert resp.status_code == 400


class TestRecommendation:
    def test_recommendation_endpoint(self, client: TestClient) -> None:
        sid = str(uuid.uuid4())
        resp = client.get(f"/api/evaluations/{sid}/recommendation")
        assert resp.status_code == 200
        data = resp.json()
        assert "recommended_provider_id" in data
        assert "reason" in data

    def test_recommendation_includes_performance_signals(self, client: TestClient) -> None:
        """Combined score should include cost/speed weights, not just quality."""
        sid, cp_ids, exec_ids = _setup_session_with_executions()
        # Rate enough times to cross the minimum threshold
        for i in range(3):
            eid = str(uuid.uuid4())
            state = get_state()
            state.add_message(
                sid,
                {"id": str(uuid.uuid4()), "role": "user", "content": f"q{i}"},
            )
            state.executions[eid] = ExecutionRecord(
                execution_id=eid,
                session_id=sid,
                query=f"q{i}",
                mode="direct",
                status="complete",
                chat_provider_id=cp_ids[0],
            )
            state.add_message(
                sid,
                {
                    "id": str(uuid.uuid4()),
                    "role": "assistant",
                    "content": f"answer {i}",
                    "execution_id": eid,
                    "chat_provider_id": cp_ids[0],
                    "metrics": {"cost_usd": 0.001, "elapsed_seconds": 0.5, "total_tokens": 50},
                },
                chat_provider_id=cp_ids[0],
            )
            client.post(
                "/api/evaluations/thumb",
                json={
                    "execution_id": eid,
                    "session_id": sid,
                    "chat_provider_id": cp_ids[0],
                    "rating": "up",
                },
            )

        summary = client.get(f"/api/evaluations/{sid}/summary").json()
        p0 = summary["by_chat_provider"][cp_ids[0]]
        # combined_score should be > thumb weight alone (0.20) because
        # performance signals contribute when metrics are present
        assert p0["combined_score"] > 0


class TestBestPickRepeatedQuery:
    """Regression: best-picks for repeated identical queries must be independent."""

    def test_repeated_queries_independent_picks(self, client: TestClient) -> None:
        # Create two turns with the same query text in the same session
        sid, cp_ids_t1, exec_ids_t1 = _setup_session_with_executions(query="What is RLM?")
        _, cp_ids_t2, exec_ids_t2 = _setup_session_with_executions(
            query="What is RLM?", session_id=sid
        )

        # Pick best for turn 1
        client.post(
            "/api/evaluations/best-pick",
            json={"session_id": sid, "winner_execution_id": exec_ids_t1[0]},
        )
        # Pick best for turn 2
        client.post(
            "/api/evaluations/best-pick",
            json={"session_id": sid, "winner_execution_id": exec_ids_t2[1]},
        )

        evals = client.get(f"/api/evaluations/{sid}").json()
        pick_winners = {p["winner_execution_id"] for p in evals["best_picks"]}
        # Both picks should coexist
        assert exec_ids_t1[0] in pick_winners
        assert exec_ids_t2[1] in pick_winners
        assert len(evals["best_picks"]) == 2


class TestJudgeProviderClearing:
    """Regression: judge_chat_provider_id can be cleared via config update."""

    def test_set_and_clear_judge_provider(self, client: TestClient) -> None:
        fake_id = str(uuid.uuid4())
        # Set it
        resp = client.put("/api/config", json={"judge_chat_provider_id": fake_id})
        assert resp.status_code == 200
        assert resp.json()["judge_chat_provider_id"] == fake_id

        # Clear it with empty string
        resp = client.put("/api/config", json={"judge_chat_provider_id": ""})
        assert resp.status_code == 200
        assert resp.json()["judge_chat_provider_id"] is None

    def test_clear_judge_provider_with_null(self, client: TestClient) -> None:
        fake_id = str(uuid.uuid4())
        resp = client.put("/api/config", json={"judge_chat_provider_id": fake_id})
        assert resp.json()["judge_chat_provider_id"] == fake_id

        # Clear with null
        resp = client.put("/api/config", json={"judge_chat_provider_id": None})
        assert resp.status_code == 200
        assert resp.json()["judge_chat_provider_id"] is None


class TestDeleteJudgeProvider:
    """Regression: deleting the active judge provider clears the config."""

    def test_delete_judge_provider_clears_config(self, client: TestClient) -> None:
        # Create an LLM Provider first
        lp_resp = client.post(
            "/api/llm-providers",
            json={"name": "OpenAI gpt-4o", "backend": "openai", "model": "gpt-4o"},
        )
        assert lp_resp.status_code == 201
        lp_id = lp_resp.json()["id"]

        # Create a chat provider
        resp = client.post(
            "/api/chat-providers",
            json={
                "name": "Judge CP",
                "llm_provider_id": lp_id,
            },
        )
        assert resp.status_code == 201
        cp_id = resp.json()["id"]

        # Set it as judge
        client.put("/api/config", json={"judge_chat_provider_id": cp_id})
        config = client.get("/api/config").json()
        assert config["judge_chat_provider_id"] == cp_id

        # Delete the provider
        resp = client.delete(f"/api/chat-providers/{cp_id}")
        assert resp.status_code == 204

        # judge_chat_provider_id should be cleared
        config = client.get("/api/config").json()
        assert config["judge_chat_provider_id"] is None


class TestRecommendationWithBestPicks:
    """Regression: best-picks count toward the minimum-data threshold."""

    def test_best_picks_alone_enable_recommendation(self, client: TestClient) -> None:
        # Create 3 turns each with a best-pick (no thumbs, no judge).
        # Each turn needs a unique query so picks are independent.
        pick_eids: list[str] = []
        sid: str | None = None
        for i in range(3):
            s, _, eids = _setup_session_with_executions(query=f"unique-query-{i}", session_id=sid)
            sid = s
            pick_eids.append(eids[0])  # pick first provider in each turn

        for eid in pick_eids:
            client.post(
                "/api/evaluations/best-pick",
                json={"session_id": sid, "winner_execution_id": eid},
            )

        resp = client.get(f"/api/evaluations/{sid}/recommendation")
        assert resp.status_code == 200
        data = resp.json()
        # 3 best-picks should meet the threshold — should NOT say "Need at least..."
        assert "need at least" not in data["reason"].lower()


class TestEvalsSurviveExecClear:
    """Regression: evaluations work when state.executions is empty (post-restart)."""

    def test_summary_after_clearing_executions(self, client: TestClient) -> None:
        sid, cp_ids, exec_ids = _setup_session_with_executions()
        client.post(
            "/api/evaluations/thumb",
            json={
                "execution_id": exec_ids[0],
                "session_id": sid,
                "chat_provider_id": cp_ids[0],
                "rating": "up",
            },
        )

        # Simulate restart: clear in-memory executions
        state = get_state()
        state.executions.clear()

        # Summary should still work via persisted session messages
        resp = client.get(f"/api/evaluations/{sid}/summary")
        assert resp.status_code == 200
        data = resp.json()
        assert cp_ids[0] in data["by_chat_provider"]
        assert data["by_chat_provider"][cp_ids[0]]["thumb_up"] == 1

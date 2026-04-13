"""Tests for LLM judge auto-scoring of non-usable execution outcomes."""

from __future__ import annotations

from typing import Any

import pytest

from rlmkit.server.judge import _AUTO_SCORE_BUDGET_PARTIAL, _AUTO_SCORE_FAILURE, JudgeService


class _FakeTelemetry:
    def get_run(self, run_id: str) -> None:
        return None


class _FakeConfig:
    judge_chat_provider_id: str | None = "judge-cp-1"


class _FakeState:
    """Minimal stub satisfying JudgeService requirements."""

    def __init__(
        self,
        *,
        execution_ctx: tuple[str, str, str, str] | None = None,
        exec_result: dict[str, Any] | None = None,
    ) -> None:
        self.config = _FakeConfig()
        self.telemetry = _FakeTelemetry()
        self.evaluations: dict[str, list[dict[str, Any]]] = {
            "thumb_ratings": [],
            "best_picks": [],
            "judge_scores": [],
            "judge_pairwise": [],
        }
        self._execution_ctx = execution_ctx
        self._exec_result = exec_result
        self.executions: dict[str, Any] = {}

        # Pre-populate in-memory execution record if result provided
        if exec_result is not None:

            class _Rec:
                result = exec_result

            self.executions["exec-1"] = _Rec()

    def resolve_execution_context(self, execution_id: str) -> tuple[str, str, str, str] | None:
        return self._execution_ctx

    def save_evaluations(self) -> None:
        pass


class TestJudgeAutoScore:
    """Verify that non-usable outcomes get auto-scored without calling the LLM judge."""

    def _make_service(
        self,
        response: str,
        *,
        success: bool = True,
        error: str | None = None,
    ) -> JudgeService:
        ctx = ("sess-1", "What is X?", response, "cp-abc")
        result = {"success": success, "error": error, "answer": response}
        state = _FakeState(execution_ctx=ctx, exec_result=result)
        return JudgeService(state)  # type: ignore[arg-type]

    @pytest.mark.asyncio
    async def test_auto_score_timeout_failure(self) -> None:
        svc = self._make_service("", success=False, error="Request timeout")
        score = await svc.score_pointwise("exec-1")
        assert score.overall_score == _AUTO_SCORE_FAILURE
        assert score.judge_provider_id == "auto"
        assert "timeout" in score.reasoning.lower()

    @pytest.mark.asyncio
    async def test_auto_score_context_overflow_failure(self) -> None:
        svc = self._make_service("", success=False, error="context window exceeded")
        score = await svc.score_pointwise("exec-1")
        assert score.overall_score == _AUTO_SCORE_FAILURE
        assert score.judge_provider_id == "auto"

    @pytest.mark.asyncio
    async def test_auto_score_general_error(self) -> None:
        svc = self._make_service("", success=False, error="Unknown crash")
        score = await svc.score_pointwise("exec-1")
        assert score.overall_score == _AUTO_SCORE_FAILURE
        assert "general_error" in score.reasoning.lower()

    @pytest.mark.asyncio
    async def test_auto_score_degraded_timeout(self) -> None:
        answer = "⚠️ **Execution timed out** after 5 steps.\n\nPartial results..."
        svc = self._make_service(answer, success=True)
        score = await svc.score_pointwise("exec-1")
        assert score.overall_score == _AUTO_SCORE_FAILURE
        assert score.judge_provider_id == "auto"

    @pytest.mark.asyncio
    async def test_auto_score_degraded_budget_short_answer(self) -> None:
        """Budget exhausted with a very short answer gets score 1."""
        answer = "⚠️ **Step budget exhausted** (16/16)."
        svc = self._make_service(answer, success=True)
        score = await svc.score_pointwise("exec-1")
        assert score.overall_score == _AUTO_SCORE_FAILURE

    @pytest.mark.asyncio
    async def test_auto_score_degraded_budget_with_partial(self) -> None:
        """Budget exhausted with a substantial partial answer gets score 2."""
        partial = "A" * 100  # >50 chars of content
        answer = f"⚠️ **Step budget exhausted** (16/16 steps used).\n\n{partial}"
        svc = self._make_service(answer, success=True)
        score = await svc.score_pointwise("exec-1")
        assert score.overall_score == _AUTO_SCORE_BUDGET_PARTIAL

    @pytest.mark.asyncio
    async def test_auto_score_stored_in_evaluations(self) -> None:
        svc = self._make_service("", success=False, error="timeout")
        await svc.score_pointwise("exec-1")
        scores = svc.state.evaluations["judge_scores"]
        assert len(scores) == 1
        assert scores[0]["judge_provider_id"] == "auto"

    @pytest.mark.asyncio
    async def test_auto_score_dimensions_match_overall(self) -> None:
        svc = self._make_service("", success=False, error="timeout")
        score = await svc.score_pointwise("exec-1")
        for dim_score in score.dimensions.values():
            assert dim_score == score.overall_score

    @pytest.mark.asyncio
    async def test_fallback_to_telemetry_for_result(self) -> None:
        """When no in-memory execution record exists, fall back to telemetry."""
        ctx = ("sess-1", "What is X?", "", "cp-abc")
        state = _FakeState(execution_ctx=ctx, exec_result=None)

        # Override telemetry to return a run with success=False
        class _FakeTelemetryWithRun:
            def get_run(self, run_id: str) -> object:
                class _Run:
                    success = False
                    error = "context window exceeded"

                return _Run()

        state.telemetry = _FakeTelemetryWithRun()  # type: ignore[assignment]
        svc = JudgeService(state)  # type: ignore[arg-type]
        score = await svc.score_pointwise("exec-1")
        assert score.overall_score == _AUTO_SCORE_FAILURE
        assert score.judge_provider_id == "auto"

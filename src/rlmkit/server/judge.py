"""LLM-as-Judge service for evaluating response quality."""

from __future__ import annotations

import json
import logging
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

from rlmkit.application.services.outcome_classifier import (
    OutcomeCategory,
    classify_execution_outcome,
)
from rlmkit.prompts.templates import load_prompt_from_file
from rlmkit.server.models import JudgePairwise, JudgeScore

if TYPE_CHECKING:
    from rlmkit.server.dependencies import AppState

logger = logging.getLogger(__name__)

_PROMPTS_DIR = Path(__file__).resolve().parent.parent / "prompts"

# Auto-scores for non-usable outcomes.
# 0 for hard failures (not scored at all — excluded from quality math).
# 1 for budget exhaustion with partial content (minimal credit).
AUTO_SCORE_FAILURE = 0.0
AUTO_SCORE_BUDGET_PARTIAL = 1.0
# Minimum answer length to qualify as a "partial" budget-exhausted answer
# rather than an empty failure. 50 chars covers a short sentence but
# excludes bare error prefixes like "Error: ...".
AUTO_SCORE_PARTIAL_THRESHOLD_CHARS = 50


def _parse_json_from_response(text: str) -> dict[str, Any]:
    """Extract JSON from LLM response, handling markdown fences."""
    text = text.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        # Drop first and last lines (fences)
        lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines)
    result: dict[str, Any] = json.loads(text)
    return result


class JudgeService:
    def __init__(self, state: AppState) -> None:
        self.state = state

    def _get_judge_provider_id(self) -> str:
        judge_id = self.state.config.judge_chat_provider_id
        if not judge_id:
            raise ValueError("No judge Chat Provider configured")
        return str(judge_id)

    def _get_execution_context(self, execution_id: str) -> tuple[str, str, str, str]:
        """Return (session_id, query, response_content, chat_provider_id).

        Uses AppState.resolve_execution_context which derives data from
        persisted session messages, surviving server restarts.
        """
        ctx = self.state.resolve_execution_context(execution_id)
        if ctx is None:
            raise ValueError(f"Execution {execution_id} not found")
        session_id: str = ctx[0]
        query: str = ctx[1]
        response_content: str = ctx[2]
        chat_provider_id: str = ctx[3]
        return session_id, query, response_content, chat_provider_id

    def _get_execution_result(self, execution_id: str) -> tuple[bool, str | None]:
        """Return (success, error) for an execution.

        Checks in-memory ExecutionRecord first, then falls back to the
        telemetry store.
        """
        rec = self.state.executions.get(execution_id)
        if rec and rec.result:
            return bool(rec.result.get("success", True)), rec.result.get("error")

        run = self.state.telemetry.get_run(execution_id)
        if run:
            return run.success, run.error

        # Assume success if we can't find status — the judge will evaluate normally
        return True, None

    def _auto_score_for_outcome(
        self,
        execution_id: str,
        session_id: str,
        cp_id: str,
        category: OutcomeCategory,
        response: str,
    ) -> JudgeScore:
        """Build a JudgeScore with auto-assigned score for a non-usable outcome."""
        # Budget-exhausted with a partial answer gets a slightly higher score
        has_partial = (
            bool(response.strip()) and len(response.strip()) > AUTO_SCORE_PARTIAL_THRESHOLD_CHARS
        )
        if category == OutcomeCategory.BUDGET_EXHAUSTED and has_partial:
            auto_score = AUTO_SCORE_BUDGET_PARTIAL
        else:
            auto_score = AUTO_SCORE_FAILURE

        score = JudgeScore(
            id=str(uuid.uuid4()),
            execution_id=execution_id,
            session_id=session_id,
            chat_provider_id=cp_id,
            judge_provider_id="auto",
            dimensions={
                "relevance": auto_score,
                "correctness": auto_score,
                "completeness": auto_score,
                "coherence": auto_score,
                "conciseness": auto_score,
            },
            overall_score=auto_score,
            reasoning=f"Auto-scored: {category.value} (LLM judge skipped)",
            created_at=datetime.now(timezone.utc),
        )
        self.state.evaluations["judge_scores"].append(score.model_dump(mode="json"))
        self.state.save_evaluations()
        return score

    def _resolve_source_content(self, execution_id: str) -> str | None:
        """Resolve the source document content for an execution.

        Walks the session messages to find the user message (with file_ids)
        that preceded this execution's assistant response, then reads the
        file content from state.files.
        """
        rec = self.state.executions.get(execution_id)
        if not rec:
            return None
        session = self.state.sessions.get(rec.session_id)
        if not session:
            return None

        # Find the user message that preceded this execution's response.
        # Walk messages in order; the last user message before our execution's
        # assistant message is the one that triggered it.
        last_user_msg: dict[str, Any] | None = None
        for msg in session.messages:
            if msg.get("role") == "user":
                last_user_msg = msg
            elif msg.get("execution_id") == execution_id and msg.get("role") == "assistant":
                break

        if not last_user_msg:
            return None

        # Try file_ids first (multi-file), then file_id (legacy single-file)
        file_ids: list[str] = last_user_msg.get("file_ids") or []
        if not file_ids:
            fid = last_user_msg.get("file_id")
            if fid:
                file_ids = [fid]

        if not file_ids:
            return None

        # Concatenate text content from all referenced files
        parts: list[str] = []
        for fid in file_ids:
            frec = self.state.files.get(fid)
            if frec and frec.text_content:
                parts.append(frec.text_content)
        return "\n\n".join(parts) if parts else None

    async def score_pointwise(
        self,
        execution_id: str,
        source_content: str | None = None,
    ) -> JudgeScore:
        """Score a single response on 5 dimensions using the judge LLM.

        Non-usable outcomes (failures, degraded warnings) are auto-scored
        without calling the judge LLM.

        Args:
            execution_id: The execution to score.
            source_content: Optional source document text. If not provided,
                the method attempts to resolve it from the session's file
                uploads. The judge prompt evaluates Correctness and
                Completeness against this when available.
        """
        judge_cp_id = self._get_judge_provider_id()
        session_id, query, response, cp_id = self._get_execution_context(execution_id)

        # Check outcome before sending to the LLM judge
        success, error = self._get_execution_result(execution_id)
        outcome = classify_execution_outcome(success, error, response)
        if not outcome.is_usable:
            logger.info(
                "Skipping LLM judge for non-usable execution %s (%s)",
                execution_id[:8],
                outcome.category.value,
            )
            return self._auto_score_for_outcome(
                execution_id, session_id, cp_id, outcome.category, response
            )

        # Resolve source document for source-aware evaluation
        source_cap = 8000
        if source_content is None:
            source_content = self._resolve_source_content(execution_id)
        if source_content:
            if len(source_content) > source_cap:
                source_block = (
                    source_content[:source_cap]
                    + f"\n\n[Source truncated at {source_cap:,} characters"
                    f" — full document is {len(source_content):,} characters]"
                )
            else:
                source_block = source_content
        else:
            source_block = "Not provided — evaluate based on the response alone."

        template = load_prompt_from_file(_PROMPTS_DIR / "judge_pointwise.yaml")
        prompt = template.format(
            query=query,
            response=response,
            source_document=source_block,
        )

        adapter = self.state.create_llm_adapter_for_chat_provider(judge_cp_id)
        llm_result = await adapter.complete_async([{"role": "user", "content": prompt}])
        response_text = llm_result.content

        try:
            parsed = _parse_json_from_response(response_text)
        except (json.JSONDecodeError, ValueError):
            logger.warning("Failed to parse judge pointwise response: %s", response_text[:200])
            parsed = {
                "dimensions": {
                    "relevance": 3.0,
                    "correctness": 3.0,
                    "completeness": 3.0,
                    "coherence": 3.0,
                    "conciseness": 3.0,
                },
                "reasoning": f"Failed to parse judge response: {response_text[:200]}",
            }

        # Clamp dimensions to [1.0, 5.0] and compute overall as mean
        raw_dims = parsed.get("dimensions", {})
        dimensions = {
            k: max(1.0, min(5.0, float(v)))
            for k, v in raw_dims.items()
            if isinstance(v, (int, float))
        }
        dim_values = list(dimensions.values())
        overall = round(sum(dim_values) / len(dim_values), 2) if dim_values else 0.0
        reasoning = parsed.get("reasoning", "")

        score = JudgeScore(
            id=str(uuid.uuid4()),
            execution_id=execution_id,
            session_id=session_id,
            chat_provider_id=cp_id,
            judge_provider_id=judge_cp_id,
            dimensions=dimensions,
            overall_score=overall,
            reasoning=reasoning,
            created_at=datetime.now(timezone.utc),
        )

        self.state.evaluations["judge_scores"].append(score.model_dump(mode="json"))
        self.state.save_evaluations()
        return score

    async def compare_pairwise(self, execution_id_a: str, execution_id_b: str) -> JudgePairwise:
        """Compare two responses, with position-swap debiasing."""
        judge_cp_id = self._get_judge_provider_id()
        session_id_a, query_a, response_a, _ = self._get_execution_context(execution_id_a)
        _, _, response_b, _ = self._get_execution_context(execution_id_b)

        template = load_prompt_from_file(_PROMPTS_DIR / "judge_pairwise.yaml")
        adapter = self.state.create_llm_adapter_for_chat_provider(judge_cp_id)

        # Run 1: A first, B second
        prompt_1 = template.format(query=query_a, response_a=response_a, response_b=response_b)
        result_1 = await adapter.complete_async([{"role": "user", "content": prompt_1}])
        result_1_str = result_1.content

        # Run 2: B first, A second (position swap)
        prompt_2 = template.format(query=query_a, response_a=response_b, response_b=response_a)
        result_2 = await adapter.complete_async([{"role": "user", "content": prompt_2}])
        result_2_str = result_2.content

        try:
            parsed_1 = _parse_json_from_response(result_1_str)
            parsed_2 = _parse_json_from_response(result_2_str)
        except (json.JSONDecodeError, ValueError):
            logger.warning("Failed to parse pairwise judge responses")
            parsed_1 = {"winner": "tie", "reasoning": "Parse failure"}
            parsed_2 = {"winner": "tie", "reasoning": "Parse failure"}

        winner_1 = parsed_1.get("winner", "tie").lower()
        # Swap interpretation for run 2
        winner_2_raw = parsed_2.get("winner", "tie").lower()
        winner_2 = {"a": "b", "b": "a", "tie": "tie"}.get(winner_2_raw, "tie")

        # Disagreement → tie
        if winner_1 != winner_2:
            final_winner = "tie"
        else:
            final_winner = winner_1

        if final_winner not in ("a", "b", "tie"):
            final_winner = "tie"

        reasoning = parsed_1.get("reasoning", "")

        result = JudgePairwise(
            id=str(uuid.uuid4()),
            session_id=session_id_a,
            execution_id_a=execution_id_a,
            execution_id_b=execution_id_b,
            winner=final_winner,  # type: ignore[arg-type]
            judge_provider_id=judge_cp_id,
            reasoning=reasoning,
            created_at=datetime.now(timezone.utc),
        )

        self.state.evaluations["judge_pairwise"].append(result.model_dump(mode="json"))
        self.state.save_evaluations()
        return result

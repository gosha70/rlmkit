"""LLM-as-Judge service for evaluating response quality."""

from __future__ import annotations

import json
import logging
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

from rlmkit.prompts.templates import load_prompt_from_file
from rlmkit.server.models import JudgePairwise, JudgeScore

if TYPE_CHECKING:
    from rlmkit.server.dependencies import AppState

logger = logging.getLogger(__name__)

_PROMPTS_DIR = Path(__file__).resolve().parent.parent / "prompts"


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

    async def score_pointwise(self, execution_id: str) -> JudgeScore:
        """Score a single response on 5 dimensions using the judge LLM."""
        judge_cp_id = self._get_judge_provider_id()
        session_id, query, response, cp_id = self._get_execution_context(execution_id)

        template = load_prompt_from_file(_PROMPTS_DIR / "judge_pointwise.yaml")
        prompt = template.format(query=query, response=response)

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
                "overall_score": 3.0,
                "reasoning": f"Failed to parse judge response: {response_text[:200]}",
            }

        dimensions = parsed.get("dimensions", {})
        overall = parsed.get("overall_score", 0.0)
        reasoning = parsed.get("reasoning", "")

        score = JudgeScore(
            id=str(uuid.uuid4()),
            execution_id=execution_id,
            session_id=session_id,
            chat_provider_id=cp_id,
            judge_provider_id=judge_cp_id,
            dimensions=dimensions,
            overall_score=float(overall),
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

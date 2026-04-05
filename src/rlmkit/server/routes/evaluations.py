"""Evaluation endpoints: thumb ratings, best-pick, judge trigger, summaries."""

from __future__ import annotations

import logging
import uuid
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException

from rlmkit.server.dependencies import AppState, get_state
from rlmkit.server.models import (
    BestPickRequest,
    EvaluationSummaryResponse,
    JudgeRequest,
    ThumbRatingRequest,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/evaluations", tags=["evaluations"])


@router.post("/thumb")
async def submit_thumb_rating(
    req: ThumbRatingRequest,
    state: AppState = Depends(get_state),
) -> dict[str, str]:
    """Submit or update a thumbs up/down rating for a response."""
    ratings = state.evaluations["thumb_ratings"]
    # Upsert: replace existing rating for this execution_id
    ratings[:] = [r for r in ratings if r["execution_id"] != req.execution_id]
    ratings.append(
        {
            "id": str(uuid.uuid4()),
            "execution_id": req.execution_id,
            "session_id": req.session_id,
            "chat_provider_id": req.chat_provider_id,
            "rating": req.rating,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
    )
    state.save_evaluations()
    return {"status": "ok"}


@router.delete("/thumb/{execution_id}")
async def remove_thumb_rating(
    execution_id: str,
    state: AppState = Depends(get_state),
) -> dict[str, str]:
    """Remove a thumb rating."""
    ratings = state.evaluations["thumb_ratings"]
    before = len(ratings)
    ratings[:] = [r for r in ratings if r["execution_id"] != execution_id]
    if len(ratings) == before:
        raise HTTPException(status_code=404, detail="Rating not found")
    state.save_evaluations()
    return {"status": "ok"}


@router.post("/best-pick")
async def submit_best_pick(
    req: BestPickRequest,
    state: AppState = Depends(get_state),
) -> dict[str, str]:
    """Submit a best-response selection for a turn."""
    picks = state.evaluations["best_picks"]
    # Find all execution_ids in the same turn using message-based grouping.
    # This correctly handles repeated identical queries as separate turns.
    same_turn_exec_ids = state.get_execution_ids_for_turn(req.session_id, req.winner_execution_id)

    if same_turn_exec_ids:
        picks[:] = [
            p
            for p in picks
            if not (
                p["session_id"] == req.session_id
                and p.get("winner_execution_id") in same_turn_exec_ids
            )
        ]
    picks.append(
        {
            "id": str(uuid.uuid4()),
            "session_id": req.session_id,
            "winner_execution_id": req.winner_execution_id,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
    )
    state.save_evaluations()
    return {"status": "ok"}


@router.get("/{session_id}")
async def get_session_evaluations(
    session_id: str,
    state: AppState = Depends(get_state),
) -> dict[str, list[dict]]:
    """Get all evaluations for a session."""
    return {
        "thumb_ratings": [
            r for r in state.evaluations["thumb_ratings"] if r["session_id"] == session_id
        ],
        "best_picks": [p for p in state.evaluations["best_picks"] if p["session_id"] == session_id],
        "judge_scores": [
            s for s in state.evaluations["judge_scores"] if s["session_id"] == session_id
        ],
        "judge_pairwise": [
            p for p in state.evaluations["judge_pairwise"] if p["session_id"] == session_id
        ],
    }


@router.delete("/{session_id}", status_code=204)
async def reset_session_evaluations(
    session_id: str,
    state: AppState = Depends(get_state),
) -> None:
    """Delete all evaluations (ratings, picks, judge scores) for a session."""
    for key in ("thumb_ratings", "best_picks", "judge_scores", "judge_pairwise"):
        state.evaluations[key] = [
            e for e in state.evaluations[key] if e["session_id"] != session_id
        ]
    state.save_evaluations()


@router.get("/{session_id}/summary")
async def get_evaluation_summary(
    session_id: str,
    state: AppState = Depends(get_state),
) -> EvaluationSummaryResponse:
    """Get aggregated quality scores per provider for a session."""
    from rlmkit.server.quality import QualityEngine

    engine = QualityEngine()
    scores = engine.compute_scores(session_id, state)
    rec = engine.get_recommendation(scores)
    return EvaluationSummaryResponse(
        session_id=session_id,
        by_chat_provider=scores,
        recommendation=rec[0],
        recommendation_reason=rec[1],
    )


@router.get("/{session_id}/recommendation")
async def get_recommendation(
    session_id: str,
    state: AppState = Depends(get_state),
) -> dict[str, str | None]:
    """Get combined quality recommendation for a session."""
    from rlmkit.server.quality import QualityEngine

    engine = QualityEngine()
    scores = engine.compute_scores(session_id, state)
    provider_id, reason = engine.get_recommendation(scores)
    return {"recommended_provider_id": provider_id, "reason": reason}


@router.post("/judge")
async def trigger_judge(
    req: JudgeRequest,
    state: AppState = Depends(get_state),
) -> dict[str, list[dict]]:
    """Trigger LLM-as-Judge scoring for the given execution_ids."""
    from rlmkit.server.judge import JudgeService

    if not state.config.judge_chat_provider_id:
        raise HTTPException(status_code=400, detail="No judge Chat Provider configured")

    service = JudgeService(state)

    pointwise_results: list[dict] = []
    pairwise_results: list[dict] = []

    if req.mode in ("pointwise", "both"):
        for eid in req.execution_ids:
            try:
                score = await service.score_pointwise(eid)
                pointwise_results.append(score.model_dump(mode="json"))
            except Exception as exc:
                logger.warning("Pointwise judge failed for %s: %s", eid, exc)

    if req.mode in ("pairwise", "both") and len(req.execution_ids) >= 2:
        # Compare all pairs
        ids = req.execution_ids
        for i in range(len(ids)):
            for j in range(i + 1, len(ids)):
                try:
                    result = await service.compare_pairwise(ids[i], ids[j])
                    pairwise_results.append(result.model_dump(mode="json"))
                except Exception as exc:
                    logger.warning("Pairwise judge failed for %s vs %s: %s", ids[i], ids[j], exc)

    return {"pointwise": pointwise_results, "pairwise": pairwise_results}

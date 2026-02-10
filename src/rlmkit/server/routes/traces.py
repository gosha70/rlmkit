"""Trace endpoints."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException

from rlmkit.server.dependencies import AppState, get_state
from rlmkit.server.models import (
    TraceBudget,
    TraceResponse,
    TraceResult,
    TraceStep,
)

router = APIRouter()


@router.get("/api/traces/{execution_id}")
async def get_trace(
    execution_id: str,
    state: AppState = Depends(get_state),
) -> TraceResponse:
    """Get execution trace with steps."""
    execution = state.executions.get(execution_id)
    if execution is None:
        raise HTTPException(status_code=404, detail="Execution not found")

    steps = []
    for i, step_data in enumerate(execution.steps):
        steps.append(
            TraceStep(
                index=i,
                action_type=step_data.get("role", "inspect"),
                code=step_data.get("code"),
                output=step_data.get("content", ""),
                input_tokens=step_data.get("input_tokens", 0),
                output_tokens=step_data.get("output_tokens", 0),
            )
        )

    result_data = execution.result or {}
    return TraceResponse(
        execution_id=execution.execution_id,
        session_id=execution.session_id,
        query=execution.query,
        mode=execution.mode,
        status=execution.status,
        started_at=execution.started_at,
        completed_at=execution.completed_at,
        result=TraceResult(
            answer=result_data.get("answer", ""),
            success=result_data.get("success", False),
        ),
        budget=TraceBudget(
            steps_used=len(execution.steps),
            steps_limit=state.config.budget.max_steps,
        ),
        steps=steps,
    )

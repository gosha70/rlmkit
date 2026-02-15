"""Trace endpoints."""

from __future__ import annotations

from typing import List

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel

from rlmkit.server.dependencies import AppState, get_state
from rlmkit.server.models import (
    TraceBudget,
    TraceResponse,
    TraceResult,
    TraceStep,
)

router = APIRouter()


class ExecutionSummary(BaseModel):
    execution_id: str
    session_id: str
    query: str
    mode: str
    status: str
    started_at: str | None = None
    completed_at: str | None = None
    total_tokens: int = 0
    total_cost: float = 0.0


@router.get("/api/executions")
async def list_executions(
    limit: int = Query(default=20, ge=1, le=100),
    state: AppState = Depends(get_state),
) -> List[ExecutionSummary]:
    """List recent executions (newest first)."""
    execs = sorted(
        state.executions.values(),
        key=lambda e: e.started_at or "",
        reverse=True,
    )[:limit]
    result = []
    for e in execs:
        rd = e.result or {}
        result.append(ExecutionSummary(
            execution_id=e.execution_id,
            session_id=e.session_id,
            query=e.query,
            mode=e.mode,
            status=e.status,
            started_at=e.started_at.isoformat() if e.started_at else None,
            completed_at=e.completed_at.isoformat() if e.completed_at else None,
            total_tokens=rd.get("total_tokens", 0),
            total_cost=rd.get("total_cost", 0.0),
        ))
    return result


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
            tokens_used=result_data.get("total_tokens", 0),
            tokens_limit=state.config.budget.max_tokens,
            cost_used=result_data.get("total_cost", 0.0),
            cost_limit=state.config.budget.max_cost_usd,
        ),
        steps=steps,
    )

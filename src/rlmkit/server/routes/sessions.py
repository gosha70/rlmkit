"""Session management endpoints."""

from __future__ import annotations

from datetime import datetime
from typing import List

from fastapi import APIRouter, Depends, HTTPException, Query

from rlmkit.server.dependencies import AppState, get_state
from rlmkit.server.models import (
    SessionDetail,
    SessionMessage,
    SessionSummary,
    MessageMetrics,
)

router = APIRouter()


@router.get("/api/sessions")
async def list_sessions(
    limit: int = Query(default=20, ge=1, le=100),
    offset: int = Query(default=0, ge=0),
    state: AppState = Depends(get_state),
) -> List[SessionSummary]:
    """List conversation sessions."""
    sessions = sorted(
        state.sessions.values(),
        key=lambda s: s.updated_at,
        reverse=True,
    )
    page = sessions[offset : offset + limit]
    return [
        SessionSummary(
            id=s.id,
            name=s.name,
            created_at=s.created_at,
            updated_at=s.updated_at,
            message_count=len(s.messages),
        )
        for s in page
    ]


@router.get("/api/sessions/{session_id}")
async def get_session(
    session_id: str,
    state: AppState = Depends(get_state),
) -> SessionDetail:
    """Get a session with all its messages."""
    session = state.sessions.get(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")

    messages = []
    for m in session.messages:
        metrics = None
        if m.get("metrics"):
            metrics = MessageMetrics(**m["metrics"])
        messages.append(
            SessionMessage(
                id=m["id"],
                role=m["role"],
                content=m["content"],
                file_id=m.get("file_id"),
                mode=m.get("mode"),
                mode_used=m.get("mode_used"),
                execution_id=m.get("execution_id"),
                metrics=metrics,
                timestamp=datetime.fromisoformat(m["timestamp"]),
            )
        )

    return SessionDetail(
        id=session.id,
        name=session.name,
        created_at=session.created_at,
        updated_at=session.updated_at,
        messages=messages,
    )


@router.delete("/api/sessions/{session_id}", status_code=204)
async def delete_session(
    session_id: str,
    state: AppState = Depends(get_state),
) -> None:
    """Delete a session."""
    if session_id not in state.sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    del state.sessions[session_id]

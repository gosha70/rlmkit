"""Session management endpoints."""

from __future__ import annotations

from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException, Query

from rlmstudio.server.dependencies import AppState, get_state
from rlmstudio.server.models import (
    MessageMetrics,
    SessionDetail,
    SessionMessage,
    SessionRenameRequest,
    SessionSummary,
)

router = APIRouter()


@router.get("/api/sessions")
async def list_sessions(
    limit: int = Query(default=20, ge=1, le=100),
    offset: int = Query(default=0, ge=0),
    state: AppState = Depends(get_state),
) -> list[SessionSummary]:
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


def _deserialize_message(m: dict) -> SessionMessage:
    """Convert a raw message dict to a SessionMessage model."""
    metrics = None
    if m.get("metrics"):
        metrics = MessageMetrics(**m["metrics"])
    return SessionMessage(
        id=m["id"],
        role=m["role"],
        content=m["content"],
        file_id=m.get("file_id"),
        file_ids=m.get("file_ids"),
        mode=m.get("mode"),
        mode_used=m.get("mode_used"),
        execution_id=m.get("execution_id"),
        metrics=metrics,
        chat_provider_id=m.get("chat_provider_id"),
        chat_provider_name=m.get("chat_provider_name"),
        timestamp=datetime.fromisoformat(m["timestamp"]),
    )


@router.get("/api/sessions/{session_id}")
async def get_session(
    session_id: str,
    state: AppState = Depends(get_state),
) -> SessionDetail:
    """Get a session with all its messages."""
    session = state.sessions.get(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")

    # Legacy flat messages
    messages = [_deserialize_message(m) for m in session.messages]

    # Per-Chat-Provider conversations
    conversations: dict[str, list[SessionMessage]] = {}
    for cp_id, conv_msgs in session.conversations.items():
        conversations[cp_id] = [_deserialize_message(m) for m in conv_msgs]

    return SessionDetail(
        id=session.id,
        name=session.name,
        created_at=session.created_at,
        updated_at=session.updated_at,
        messages=messages,
        conversations=conversations,
    )


@router.put("/api/sessions/{session_id}")
async def rename_session(
    session_id: str,
    req: SessionRenameRequest,
    state: AppState = Depends(get_state),
) -> SessionSummary:
    """Rename a session."""
    session = state.sessions.get(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")
    session.name = req.name.strip() or session.name
    session.updated_at = datetime.now(timezone.utc)
    state.save_sessions()
    return SessionSummary(
        id=session.id,
        name=session.name,
        created_at=session.created_at,
        updated_at=session.updated_at,
        message_count=len(session.messages),
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

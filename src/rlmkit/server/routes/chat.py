"""Chat endpoints: POST /api/chat and WS /ws/chat/{session_id}."""

from __future__ import annotations

import asyncio
import json
import time
import uuid
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, WebSocket, WebSocketDisconnect

from rlmkit.application.use_cases.run_direct import RunDirectUseCase
from rlmkit.application.use_cases.run_rlm import RunRLMUseCase
from rlmkit.server.dependencies import AppState, ExecutionRecord, get_state
from rlmkit.server.models import (
    ChatRequest,
    ChatResponse,
    ErrorDetail,
    ErrorResponse,
)

router = APIRouter()


@router.post("/api/chat", status_code=202)
async def submit_chat(
    req: ChatRequest,
    state: AppState = Depends(get_state),
) -> ChatResponse | ErrorResponse:
    """Submit a chat query for execution."""
    # Resolve content
    content = req.content
    if content is None and req.file_id:
        file_rec = state.files.get(req.file_id)
        if file_rec is None:
            return ErrorResponse(
                error=ErrorDetail(code="NOT_FOUND", message="File not found")
            )
        content = file_rec.text_content
    if content is None:
        content = ""

    # Get or create session
    session = state.get_or_create_session(req.session_id)

    # Create execution record
    exec_id = str(uuid.uuid4())
    now = datetime.now(timezone.utc)
    execution = ExecutionRecord(
        execution_id=exec_id,
        session_id=session.id,
        query=req.query,
        mode=req.mode,
        started_at=now,
    )
    state.executions[exec_id] = execution

    # Add user message to session
    session.messages.append({
        "id": str(uuid.uuid4()),
        "role": "user",
        "content": req.query,
        "file_id": req.file_id,
        "mode": req.mode,
        "timestamp": now.isoformat(),
    })
    session.updated_at = now

    # Execute in background
    asyncio.create_task(
        _run_execution(state, execution, content, req.query, req.mode)
    )

    return ChatResponse(
        execution_id=exec_id,
        session_id=session.id,
        status="running",
    )


async def _run_execution(
    state: AppState,
    execution: ExecutionRecord,
    content: str,
    query: str,
    mode: str,
) -> None:
    """Run the use case in the background and store results."""
    try:
        llm = state.create_llm_adapter()
        config = state.create_run_config(mode)

        if mode in ("rlm", "auto"):
            sandbox = state.create_sandbox()
            uc = RunRLMUseCase(llm, sandbox)
            result = await asyncio.to_thread(uc.execute, content, query, config)
        else:
            uc_direct = RunDirectUseCase(llm)
            result = await asyncio.to_thread(uc_direct.execute, content, query, config)

        now = datetime.now(timezone.utc)
        execution.status = "complete" if result.success else "error"
        execution.completed_at = now
        execution.result = {
            "answer": result.answer,
            "success": result.success,
            "error": result.error,
        }
        execution.steps = result.trace

        # Add assistant message to session
        session = state.sessions.get(execution.session_id)
        if session:
            session.messages.append({
                "id": str(uuid.uuid4()),
                "role": "assistant",
                "content": result.answer,
                "mode_used": result.mode_used,
                "execution_id": execution.execution_id,
                "metrics": {
                    "input_tokens": result.input_tokens,
                    "output_tokens": result.output_tokens,
                    "total_tokens": result.total_tokens,
                    "cost_usd": result.total_cost,
                    "elapsed_seconds": result.elapsed_time,
                    "steps": result.steps,
                },
                "timestamp": now.isoformat(),
            })
            session.updated_at = now

    except Exception as exc:
        execution.status = "error"
        execution.completed_at = datetime.now(timezone.utc)
        execution.result = {"answer": "", "success": False, "error": str(exc)}


@router.websocket("/ws/chat/{session_id}")
async def websocket_chat(
    websocket: WebSocket,
    session_id: str,
) -> None:
    """WebSocket endpoint for real-time chat streaming."""
    state = get_state()
    await websocket.accept()

    # Send connected message
    await websocket.send_json({"type": "connected", "session_id": session_id})

    try:
        while True:
            raw = await websocket.receive_text()
            data = json.loads(raw)
            msg_type = data.get("type")

            if msg_type == "pong":
                continue

            if msg_type == "query":
                msg_id = data.get("id", str(uuid.uuid4()))
                query = data.get("query", "")
                content = data.get("content", "")
                file_id = data.get("file_id")
                mode = data.get("mode", "auto")

                if file_id:
                    file_rec = state.files.get(file_id)
                    if file_rec:
                        content = file_rec.text_content

                try:
                    llm = state.create_llm_adapter()
                    config = state.create_run_config(mode)

                    if mode in ("rlm", "auto"):
                        sandbox = state.create_sandbox()
                        uc = RunRLMUseCase(llm, sandbox)
                        result = await asyncio.to_thread(
                            uc.execute, content, query, config
                        )
                    else:
                        uc_direct = RunDirectUseCase(llm)
                        result = await asyncio.to_thread(
                            uc_direct.execute, content, query, config
                        )

                    await websocket.send_json({
                        "type": "complete",
                        "id": msg_id,
                        "data": {
                            "execution_id": str(uuid.uuid4()),
                            "mode": result.mode_used,
                            "answer": result.answer,
                            "success": result.success,
                            "metrics": {
                                "input_tokens": result.input_tokens,
                                "output_tokens": result.output_tokens,
                                "total_tokens": result.total_tokens,
                                "cost_usd": result.total_cost,
                                "elapsed_seconds": result.elapsed_time,
                                "steps": result.steps,
                            },
                        },
                    })

                except Exception as exc:
                    await websocket.send_json({
                        "type": "error",
                        "id": msg_id,
                        "data": {
                            "code": "INTERNAL_ERROR",
                            "message": str(exc),
                            "recoverable": False,
                        },
                    })

    except WebSocketDisconnect:
        pass

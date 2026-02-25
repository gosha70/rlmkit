"""Chat endpoints: POST /api/chat and WS /ws/chat/{session_id}."""

from __future__ import annotations

import asyncio
import json
import logging
import uuid
from datetime import datetime, timezone
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, WebSocket, WebSocketDisconnect

from rlmkit.application.use_cases.run_direct import RunDirectUseCase
from rlmkit.application.use_cases.run_rlm import RunRLMUseCase
from rlmkit.server.dependencies import AppState, ExecutionRecord, get_state
from rlmkit.server.models import (
    ChatRequest,
    ChatResponse,
)

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post("/api/chat", status_code=202)
async def submit_chat(
    req: ChatRequest,
    state: AppState = Depends(get_state),  # noqa: B008
) -> ChatResponse:
    """Submit a chat query for execution."""
    # Validate that either content or file_id is provided
    if req.content is None and req.file_id is None:
        raise HTTPException(status_code=400, detail="Either content or file_id must be provided")

    # Resolve content
    content = req.content
    if content is None and req.file_id:
        file_rec = state.files.get(req.file_id)
        if file_rec is None:
            raise HTTPException(status_code=404, detail="File not found")
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
    session.messages.append(
        {
            "id": str(uuid.uuid4()),
            "role": "user",
            "content": req.query,
            "file_id": req.file_id,
            "mode": req.mode,
            "timestamp": now.isoformat(),
        }
    )
    session.updated_at = now

    # Execute in background
    asyncio.create_task(_run_execution(state, execution, content, req.query, req.mode))

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
        logger.info(
            "Executing query [mode=%s, provider=%s, model=%s]: %.100s",
            mode,
            state.config.active_provider,
            state.config.active_model,
            query,
        )
        config = state.create_run_config(mode)

        if mode == "compare":
            # Run both RLM and Direct, store two assistant messages
            sandbox = state.create_sandbox()
            uc_rlm = RunRLMUseCase(llm, sandbox)
            uc_direct = RunDirectUseCase(llm)
            result_rlm = await asyncio.to_thread(uc_rlm.execute, content, query, config)
            result_direct = await asyncio.to_thread(uc_direct.execute, content, query, config)
            results = [result_rlm, result_direct]
        elif mode == "rag":
            # TODO: Run RAG use case when available; fall back to direct for now
            uc_direct = RunDirectUseCase(llm)
            results = [await asyncio.to_thread(uc_direct.execute, content, query, config)]
        elif mode in ("rlm", "auto"):
            sandbox = state.create_sandbox()
            uc = RunRLMUseCase(llm, sandbox)
            results = [await asyncio.to_thread(uc.execute, content, query, config)]
        else:
            uc_direct = RunDirectUseCase(llm)
            results = [await asyncio.to_thread(uc_direct.execute, content, query, config)]

        now = datetime.now(timezone.utc)
        # Use the first result for execution record status
        result = results[0]
        if result.success:
            logger.info(
                "Execution complete [exec=%s, tokens=%d]",
                execution.execution_id[:8],
                result.total_tokens,
            )
        else:
            logger.error("Execution failed [exec=%s]: %s", execution.execution_id[:8], result.error)
        execution.status = "complete" if result.success else "error"
        execution.completed_at = now
        execution.result = {
            "answer": result.answer,
            "success": result.success,
            "error": result.error,
            "input_tokens": result.input_tokens,
            "output_tokens": result.output_tokens,
            "total_tokens": result.total_tokens,
            "total_cost": result.total_cost,
            "elapsed_time": result.elapsed_time,
            "steps_count": result.steps,
        }
        execution.steps = result.trace

        # Add assistant message(s) to session
        session = state.sessions.get(execution.session_id)
        if session:
            for res in results:
                # Surface errors so the user can see what went wrong
                content = res.answer
                if not res.success and not content:
                    content = f"Error: {res.error or 'Execution failed'}"

                session.messages.append(
                    {
                        "id": str(uuid.uuid4()),
                        "role": "assistant",
                        "content": content,
                        "mode_used": res.mode_used,
                        "provider": state.config.active_provider,
                        "execution_id": execution.execution_id,
                        "metrics": {
                            "input_tokens": res.input_tokens,
                            "output_tokens": res.output_tokens,
                            "total_tokens": res.total_tokens,
                            "cost_usd": res.total_cost,
                            "elapsed_seconds": res.elapsed_time,
                            "steps": res.steps,
                        },
                        "timestamp": now.isoformat(),
                    }
                )
            session.updated_at = now
            state.save_sessions()

    except Exception as exc:
        logger.exception("Execution crashed [exec=%s]", execution.execution_id[:8])
        now = datetime.now(timezone.utc)
        execution.status = "error"
        execution.completed_at = now
        execution.result = {"answer": "", "success": False, "error": str(exc)}

        # Add error message to session so the user can see what went wrong
        session = state.sessions.get(execution.session_id)
        if session:
            session.messages.append(
                {
                    "id": str(uuid.uuid4()),
                    "role": "assistant",
                    "content": f"Error: {exc}",
                    "mode_used": execution.mode,
                    "execution_id": execution.execution_id,
                    "timestamp": now.isoformat(),
                }
            )
            session.updated_at = now


class WebSocketEventEmitter:
    """Emits execution events to a WebSocket client."""

    def __init__(self, ws: WebSocket, msg_id: str) -> None:
        self._ws = ws
        self._id = msg_id

    async def on_token(self, token: str) -> None:
        await self._ws.send_json({"type": "token", "id": self._id, "data": token})

    async def on_step(self, step_data: dict[str, Any]) -> None:
        await self._ws.send_json({"type": "step", "id": self._id, "data": step_data})

    async def on_metrics(self, metrics: dict[str, Any]) -> None:
        await self._ws.send_json({"type": "metrics", "id": self._id, "data": metrics})


async def _ping_loop(ws: WebSocket) -> None:
    """Send periodic ping messages to detect stale connections."""
    while True:
        await asyncio.sleep(30)
        await ws.send_json({"type": "ping"})


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

    # Track active query tasks for cancellation
    active_tasks: dict[str, asyncio.Task] = {}

    # Start heartbeat ping loop
    ping_task = asyncio.create_task(_ping_loop(websocket))
    try:
        while True:
            raw = await websocket.receive_text()

            try:
                data = json.loads(raw)
            except json.JSONDecodeError:
                await websocket.send_json(
                    {
                        "type": "error",
                        "id": "",
                        "data": {
                            "code": "INVALID_JSON",
                            "message": "Malformed JSON message",
                            "recoverable": True,
                        },
                    }
                )
                continue

            msg_type = data.get("type")

            if msg_type == "pong":
                continue

            if msg_type == "cancel":
                task = active_tasks.pop(data.get("id", ""), None)
                if task:
                    task.cancel()
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

                # Create execution record (mirrors REST path) so traces/dashboard work
                exec_id = str(uuid.uuid4())
                now = datetime.now(timezone.utc)
                session = state.get_or_create_session(session_id)
                execution = ExecutionRecord(
                    execution_id=exec_id,
                    session_id=session.id,
                    query=query,
                    mode=mode,
                    started_at=now,
                )
                state.executions[exec_id] = execution

                # Add user message to session
                session.messages.append(
                    {
                        "id": str(uuid.uuid4()),
                        "role": "user",
                        "content": query,
                        "file_id": file_id,
                        "mode": mode,
                        "timestamp": now.isoformat(),
                    }
                )
                session.updated_at = now

                async def _ws_execute(
                    ws: WebSocket,
                    mid: str,
                    cnt: str,
                    q: str,
                    m: str,
                    exec_rec: ExecutionRecord,
                    sess: Any,
                ) -> None:
                    emitter = WebSocketEventEmitter(ws, mid)
                    try:
                        llm = state.create_llm_adapter()
                        logger.info(
                            "WS executing [mode=%s, provider=%s, model=%s]: %.100s",
                            m,
                            state.config.active_provider,
                            state.config.active_model,
                            q,
                        )
                        cfg = state.create_run_config(m)

                        if m == "compare":
                            sandbox = state.create_sandbox()
                            uc_rlm = RunRLMUseCase(llm, sandbox)
                            uc_direct = RunDirectUseCase(llm)
                            result_rlm = await uc_rlm.execute_async(
                                cnt, q, cfg, event_emitter=emitter
                            )
                            result_direct = await uc_direct.execute_async(
                                cnt, q, cfg, event_emitter=emitter
                            )
                            results = [result_rlm, result_direct]
                        elif m == "rag":
                            uc_d = RunDirectUseCase(llm)
                            results = [await uc_d.execute_async(cnt, q, cfg, event_emitter=emitter)]
                        elif m in ("rlm", "auto"):
                            sandbox = state.create_sandbox()
                            uc = RunRLMUseCase(llm, sandbox)
                            results = [await uc.execute_async(cnt, q, cfg, event_emitter=emitter)]
                        else:
                            uc_d = RunDirectUseCase(llm)
                            results = [await uc_d.execute_async(cnt, q, cfg, event_emitter=emitter)]

                        finish = datetime.now(timezone.utc)
                        result = results[0]
                        exec_rec.status = "complete" if result.success else "error"
                        exec_rec.completed_at = finish
                        exec_rec.result = {
                            "answer": result.answer,
                            "success": result.success,
                            "error": result.error,
                            "input_tokens": result.input_tokens,
                            "output_tokens": result.output_tokens,
                            "total_tokens": result.total_tokens,
                            "total_cost": result.total_cost,
                            "elapsed_time": result.elapsed_time,
                            "steps_count": result.steps,
                        }
                        exec_rec.steps = result.trace

                        for res in results:
                            answer = res.answer
                            if not res.success and not answer:
                                answer = f"Error: {res.error or 'Execution failed'}"

                            # Store in session for dashboard metrics
                            sess.messages.append(
                                {
                                    "id": str(uuid.uuid4()),
                                    "role": "assistant",
                                    "content": answer,
                                    "mode_used": res.mode_used,
                                    "provider": state.config.active_provider,
                                    "execution_id": exec_rec.execution_id,
                                    "metrics": {
                                        "input_tokens": res.input_tokens,
                                        "output_tokens": res.output_tokens,
                                        "total_tokens": res.total_tokens,
                                        "cost_usd": res.total_cost,
                                        "elapsed_seconds": res.elapsed_time,
                                        "steps": res.steps,
                                    },
                                    "timestamp": finish.isoformat(),
                                }
                            )
                            sess.updated_at = finish

                            if not res.success:
                                await ws.send_json(
                                    {
                                        "type": "error",
                                        "id": mid,
                                        "data": {
                                            "code": "EXECUTION_ERROR",
                                            "message": res.error or "Execution failed",
                                            "mode": res.mode_used,
                                            "recoverable": True,
                                        },
                                    }
                                )
                            else:
                                await ws.send_json(
                                    {
                                        "type": "complete",
                                        "id": mid,
                                        "data": {
                                            "execution_id": exec_rec.execution_id,
                                            "mode": res.mode_used,
                                            "answer": res.answer,
                                            "success": res.success,
                                            "metrics": {
                                                "input_tokens": res.input_tokens,
                                                "output_tokens": res.output_tokens,
                                                "total_tokens": res.total_tokens,
                                                "cost_usd": res.total_cost,
                                                "elapsed_seconds": res.elapsed_time,
                                                "steps": res.steps,
                                            },
                                        },
                                    }
                                )
                        state.save_sessions()
                    except asyncio.CancelledError:
                        pass
                    except Exception as exc:
                        logger.exception("WebSocket execution error for %s", mid)
                        finish = datetime.now(timezone.utc)
                        exec_rec.status = "error"
                        exec_rec.completed_at = finish
                        exec_rec.result = {"answer": "", "success": False, "error": str(exc)}
                        await ws.send_json(
                            {
                                "type": "error",
                                "id": mid,
                                "data": {
                                    "code": "INTERNAL_ERROR",
                                    "message": str(exc),
                                    "recoverable": False,
                                },
                            }
                        )
                    finally:
                        active_tasks.pop(mid, None)

                task = asyncio.create_task(
                    _ws_execute(websocket, msg_id, content, query, mode, execution, session)
                )
                active_tasks[msg_id] = task

    except WebSocketDisconnect:
        pass
    finally:
        ping_task.cancel()
        for task in active_tasks.values():
            task.cancel()

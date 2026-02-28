"""Chat endpoints: POST /api/chat and WS /ws/chat/{session_id}."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import uuid
from datetime import datetime, timezone
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, WebSocket, WebSocketDisconnect

from rlmkit.application.dto import RunResultDTO
from rlmkit.application.use_cases.run_direct import RunDirectUseCase
from rlmkit.application.use_cases.run_rlm import RunRLMUseCase
from rlmkit.core.trace import ExecutionTrace
from rlmkit.core.trace import TraceStep as CoreTraceStep
from rlmkit.server.dependencies import AppState, ExecutionRecord, get_state
from rlmkit.server.models import (
    ChatRequest,
    ChatResponse,
)

logger = logging.getLogger(__name__)

router = APIRouter()


def _save_trajectory(
    execution: ExecutionRecord,
    result: RunResultDTO,
    trace_dir: str,
) -> None:
    """Write execution trajectory to a JSONL file if trace_dir is configured."""
    try:
        os.makedirs(trace_dir, exist_ok=True)
        trace = ExecutionTrace()
        trace.metadata = {
            "execution_id": execution.execution_id,
            "session_id": execution.session_id,
            "query": execution.query,
            "mode": execution.mode,
        }
        for i, step_data in enumerate(result.trace):
            role = step_data.get("role", "inspect")
            action_map = {"assistant": "inspect", "execution": "subcall"}
            action_type = action_map.get(role, "inspect")
            if i == len(result.trace) - 1 and result.success:
                action_type = "final"

            trace.add_step(
                CoreTraceStep(
                    index=i,
                    action_type=action_type,
                    code=step_data.get("code"),
                    output=step_data.get("content", ""),
                    tokens_used=step_data.get("input_tokens", 0)
                    + step_data.get("output_tokens", 0),
                    duration=step_data.get("elapsed_seconds", 0.0),
                    model=step_data.get("model"),
                )
            )
        trace.finalize()
        filepath = os.path.join(trace_dir, f"{execution.execution_id}.jsonl")
        trace.to_jsonl(filepath)
        logger.info("Saved trajectory to %s", filepath)
    except Exception:
        logger.warning(
            "Failed to save trajectory for %s", execution.execution_id[:8], exc_info=True
        )


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

    # Resolve execution mode from Chat Provider or legacy params
    chat_provider_id = req.chat_provider_id
    mode = req.mode
    if chat_provider_id:
        cp = state.get_chat_provider(chat_provider_id)
        if not cp:
            raise HTTPException(status_code=404, detail=f"Chat Provider {chat_provider_id} not found")
        mode = cp.execution_mode

    # Create execution record
    exec_id = str(uuid.uuid4())
    now = datetime.now(timezone.utc)
    execution = ExecutionRecord(
        execution_id=exec_id,
        session_id=session.id,
        query=req.query,
        mode=mode,
        started_at=now,
    )
    state.executions[exec_id] = execution

    # Add user message to session
    user_msg = {
        "id": str(uuid.uuid4()),
        "role": "user",
        "content": req.query,
        "file_id": req.file_id,
        "mode": mode,
        "chat_provider_id": chat_provider_id,
        "timestamp": now.isoformat(),
    }
    state.add_message(session.id, user_msg, chat_provider_id)
    session.updated_at = now

    # Execute in background
    asyncio.create_task(
        _run_execution(state, execution, content, req.query, mode, chat_provider_id)
    )

    return ChatResponse(
        execution_id=exec_id,
        session_id=session.id,
        status="running",
        chat_provider_id=chat_provider_id,
    )


async def _run_execution(
    state: AppState,
    execution: ExecutionRecord,
    content: str,
    query: str,
    mode: str,
    chat_provider_id: str | None = None,
) -> None:
    """Run the use case in the background and store results."""
    try:
        # Use Chat Provider-specific adapter if available, otherwise global
        if chat_provider_id:
            llm = state.create_llm_adapter_for_chat_provider(chat_provider_id)
            cp = state.get_chat_provider(chat_provider_id)
            provider_label = f"{cp.llm_provider}/{cp.llm_model}" if cp else "unknown"
        else:
            llm = state.create_llm_adapter()
            cp = None
            provider_label = f"{state.config.active_provider}/{state.config.active_model}"
        logger.info(
            "Executing query [mode=%s, provider=%s]: %.100s",
            mode,
            provider_label,
            query,
        )

        # Build conversation context from Chat Provider history
        conversation_history: list[dict[str, str]] = []
        if chat_provider_id:
            prev_msgs = state.get_conversation(execution.session_id, chat_provider_id)
            # Include previous messages as context (skip the current user message which is last)
            for msg in prev_msgs[:-1]:  # exclude the user message we just added
                role = msg.get("role", "user")
                msg_content = msg.get("content", "")
                if role in ("user", "assistant") and msg_content:
                    conversation_history.append({"role": role, "content": msg_content})

        # Build run config, using Chat Provider settings if available
        if cp and mode == "rlm":
            run_config = state.create_run_config(mode)
            run_config.max_steps = cp.rlm_max_steps
            run_config.max_time_seconds = float(cp.rlm_timeout_seconds)
        else:
            run_config = state.create_run_config(mode)

        # Build full query with conversation context
        full_query = query
        if conversation_history:
            context_parts = []
            for msg in conversation_history:
                prefix = "User" if msg["role"] == "user" else "Assistant"
                context_parts.append(f"{prefix}: {msg['content']}")
            context_str = "\n\n".join(context_parts)
            full_query = f"Previous conversation:\n{context_str}\n\nCurrent question: {query}"

        if mode == "compare":
            # Run both RLM and Direct, store two assistant messages
            sandbox = state.create_sandbox()
            uc_rlm = RunRLMUseCase(llm, sandbox)
            uc_direct = RunDirectUseCase(llm)
            result_rlm = await asyncio.to_thread(uc_rlm.execute, content, full_query, run_config)
            result_direct = await asyncio.to_thread(uc_direct.execute, content, full_query, run_config)
            results = [result_rlm, result_direct]
        elif mode == "rag":
            # TODO: Run RAG use case when available; fall back to direct for now
            uc_direct = RunDirectUseCase(llm)
            results = [await asyncio.to_thread(uc_direct.execute, content, full_query, run_config)]
        elif mode in ("rlm", "auto"):
            sandbox = state.create_sandbox()
            uc = RunRLMUseCase(llm, sandbox)
            results = [await asyncio.to_thread(uc.execute, content, full_query, run_config)]
        else:
            uc_direct = RunDirectUseCase(llm)
            results = [await asyncio.to_thread(uc_direct.execute, content, full_query, run_config)]

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

        # Save trajectory if configured
        if state.config.trajectory_dir:
            _save_trajectory(execution, result, state.config.trajectory_dir)

        # Add assistant message(s) to session
        session = state.sessions.get(execution.session_id)
        if session:
            for res in results:
                # Surface errors so the user can see what went wrong
                answer_content = res.answer
                if not res.success and not answer_content:
                    answer_content = f"Error: {res.error or 'Execution failed'}"

                cp_name = cp.name if cp else None
                provider_name = cp.llm_provider if cp else state.config.active_provider

                assistant_msg = {
                    "id": str(uuid.uuid4()),
                    "role": "assistant",
                    "content": answer_content,
                    "mode_used": res.mode_used,
                    "provider": provider_name,
                    "execution_id": execution.execution_id,
                    "chat_provider_id": chat_provider_id,
                    "chat_provider_name": cp_name,
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
                state.add_message(execution.session_id, assistant_msg, chat_provider_id)
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
            error_msg = {
                "id": str(uuid.uuid4()),
                "role": "assistant",
                "content": f"Error: {exc}",
                "mode_used": execution.mode,
                "execution_id": execution.execution_id,
                "chat_provider_id": chat_provider_id,
                "timestamp": now.isoformat(),
            }
            state.add_message(execution.session_id, error_msg, chat_provider_id)
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
                ws_chat_provider_id = data.get("chat_provider_id")

                if file_id:
                    file_rec = state.files.get(file_id)
                    if file_rec:
                        content = file_rec.text_content

                # Resolve mode from Chat Provider if provided
                if ws_chat_provider_id:
                    ws_cp = state.get_chat_provider(ws_chat_provider_id)
                    if ws_cp:
                        mode = ws_cp.execution_mode

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
                user_msg = {
                    "id": str(uuid.uuid4()),
                    "role": "user",
                    "content": query,
                    "file_id": file_id,
                    "mode": mode,
                    "chat_provider_id": ws_chat_provider_id,
                    "timestamp": now.isoformat(),
                }
                state.add_message(session.id, user_msg, ws_chat_provider_id)
                session.updated_at = now

                async def _ws_execute(
                    ws: WebSocket,
                    mid: str,
                    cnt: str,
                    q: str,
                    m: str,
                    exec_rec: ExecutionRecord,
                    sess: Any,
                    cp_id: str | None = None,
                ) -> None:
                    emitter = WebSocketEventEmitter(ws, mid)
                    try:
                        if cp_id:
                            llm = state.create_llm_adapter_for_chat_provider(cp_id)
                            ws_cp = state.get_chat_provider(cp_id)
                            provider_label = f"{ws_cp.llm_provider}/{ws_cp.llm_model}" if ws_cp else "unknown"
                        else:
                            llm = state.create_llm_adapter()
                            ws_cp = None
                            provider_label = f"{state.config.active_provider}/{state.config.active_model}"
                        logger.info(
                            "WS executing [mode=%s, provider=%s]: %.100s",
                            m,
                            provider_label,
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

                            cp_name = ws_cp.name if ws_cp else None
                            provider_name = ws_cp.llm_provider if ws_cp else state.config.active_provider

                            # Store in session for dashboard metrics
                            assistant_msg = {
                                "id": str(uuid.uuid4()),
                                "role": "assistant",
                                "content": answer,
                                "mode_used": res.mode_used,
                                "provider": provider_name,
                                "execution_id": exec_rec.execution_id,
                                "chat_provider_id": cp_id,
                                "chat_provider_name": cp_name,
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
                            state.add_message(sess.id, assistant_msg, cp_id)
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
                                            "chat_provider_id": cp_id,
                                            "chat_provider_name": cp_name,
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
                    _ws_execute(websocket, msg_id, content, query, mode, execution, session, ws_chat_provider_id)
                )
                active_tasks[msg_id] = task

    except WebSocketDisconnect:
        pass
    finally:
        ping_task.cancel()
        for task in active_tasks.values():
            task.cancel()

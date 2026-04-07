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
from rlmkit.application.use_cases.run_rag import RunRAGUseCase
from rlmkit.application.use_cases.run_rlm import RunRLMUseCase
from rlmkit.core.trace import ExecutionTrace
from rlmkit.core.trace import TraceStep as CoreTraceStep
from rlmkit.infrastructure.embedding.litellm_embedding_adapter import LiteLLMEmbeddingAdapter
from rlmkit.infrastructure.storage.sqlite_adapter import SQLiteStorageAdapter
from rlmkit.server.dependencies import AppState, ExecutionRecord, get_state
from rlmkit.server.models import (
    ChatRequest,
    ChatResponse,
)

from rlmkit.server.models import RunProfile as _RunProfile

logger = logging.getLogger(__name__)

router = APIRouter()


def _resolve_profile_prompt(profile: _RunProfile, mode: str) -> str | None:
    """Resolve a profile's system prompt for the given mode.

    Prefer explicit per-mode text in ``system_prompts``.  If none is
    stored, fall back to the named ``prompt_template_name`` and look up
    the template's text for *mode* from the built-in registry.
    """
    # Explicit override takes priority
    custom = profile.system_prompts.get(mode)
    if custom:
        return custom

    # Named template reference — resolved at runtime so profiles follow
    # future edits to the template registry.
    if profile.prompt_template_name:
        from rlmkit.ui.services.profile_store import SYSTEM_PROMPT_TEMPLATES

        tpl = SYSTEM_PROMPT_TEMPLATES.get(profile.prompt_template_name)
        if tpl:
            return tpl.get(mode) or None
    return None

# Per-message attachment limits
_MAX_FILES_PER_MESSAGE: int = 10
_MAX_TOTAL_CONTENT_BYTES: int = 50 * 1024 * 1024  # 50 MB

# RAG index cache: cache_key -> (SQLiteStorageAdapter, collection_name, embedding_model)
# cache_key = ":".join(sorted(file_ids))
# Avoids re-embedding the full document on every message in the same conversation.
_rag_index_cache: dict[str, tuple[SQLiteStorageAdapter, str, str]] = {}


def _resolve_file_content(file_ids: list[str], state: AppState) -> str:
    """Join text from multiple uploaded files into a single content string.

    Single file: returns plain content with a file header.
    Multiple files: prepends a Document Index so RLM tools can navigate between
    files using grep('[File N:') to jump to any file's start.

    Enforces per-message file count and total byte limits.
    Raises HTTPException 400/404/413 on violations.
    """
    if len(file_ids) > _MAX_FILES_PER_MESSAGE:
        raise HTTPException(
            status_code=400,
            detail=f"Too many files per message (max {_MAX_FILES_PER_MESSAGE})",
        )
    records = []
    cumulative_bytes = 0
    for fid in file_ids:
        rec = state.files.get(fid)
        if rec is None:
            raise HTTPException(status_code=404, detail=f"File {fid} not found")
        cumulative_bytes += len(rec.text_content.encode())
        if cumulative_bytes > _MAX_TOTAL_CONTENT_BYTES:
            raise HTTPException(
                status_code=413,
                detail=f"Combined file content exceeds {_MAX_TOTAL_CONTENT_BYTES // (1024 * 1024)} MB limit",
            )
        records.append(rec)

    separator = "\n\n---\n\n"

    if len(records) == 1:
        rec = records[0]
        result = f"[File: {rec.name}]\n\n{rec.text_content}"
    else:
        # Number each section so grep('[File N:') navigates directly to it.
        sections = [
            f"[File {i + 1}: {rec.name}]\n\n{rec.text_content}" for i, rec in enumerate(records)
        ]
        index_lines = [
            f"[DOCUMENT INDEX — {len(records)} files attached]",
            "To navigate to a specific file, use: grep('[File N:') where N is the file number.",
        ]
        for i, rec in enumerate(records):
            index_lines.append(f'  {i + 1}. "{rec.name}"')
        index_lines.append("[END DOCUMENT INDEX]")
        index_block = "\n".join(index_lines)
        result = index_block + "\n\n" + separator.join(sections)
    return result


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
            action_type: str = action_map.get(role, "inspect")
            if i == len(result.trace) - 1 and result.success:
                action_type = "final"

            trace.add_step(
                CoreTraceStep(
                    index=i,
                    action_type=action_type,  # type: ignore[arg-type]
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
    # Canonicalise file IDs (model_validator already promotes file_id -> file_ids)
    effective_file_ids: list[str] = req.file_ids or []

    # Validate that either content or file(s) are provided
    if req.content is None and not effective_file_ids:
        raise HTTPException(status_code=400, detail="Either content or file_id(s) must be provided")

    # Resolve content
    content = req.content or ""
    if effective_file_ids:
        file_content = _resolve_file_content(effective_file_ids, state)
        content = (content + "\n\n" + file_content).strip() if content else file_content

    # Get or create session
    session = state.get_or_create_session(req.session_id)

    # Resolve execution mode from Chat Provider or legacy params
    chat_provider_id = req.chat_provider_id
    mode = req.mode
    if chat_provider_id:
        cp = state.get_chat_provider(chat_provider_id)
        if not cp:
            raise HTTPException(
                status_code=404, detail=f"Chat Provider {chat_provider_id} not found"
            )
        cp = state.resolve_chat_provider(cp)
        mode = cp.execution_mode

    # Create execution record
    exec_id = str(uuid.uuid4())
    now = datetime.now(timezone.utc)
    chat_provider_name = cp.name if (chat_provider_id and cp) else None
    execution = ExecutionRecord(
        execution_id=exec_id,
        session_id=session.id,
        query=req.query,
        mode=mode,
        started_at=now,
        chat_provider_id=chat_provider_id,
        chat_provider_name=chat_provider_name,
    )
    state.executions[exec_id] = execution

    # Add user message to session
    user_msg = {
        "id": str(uuid.uuid4()),
        "role": "user",
        "content": req.query,
        "file_id": effective_file_ids[0] if effective_file_ids else None,  # compat alias
        "file_ids": effective_file_ids,
        "mode": mode,
        "chat_provider_id": chat_provider_id,
        "timestamp": now.isoformat(),
    }
    state.add_message(session.id, user_msg, chat_provider_id)
    session.updated_at = now
    # Save immediately so the session survives a server restart before execution completes
    state.save_sessions()

    # Execute in background
    asyncio.create_task(
        _run_execution(
            state,
            execution,
            content,
            req.query,
            mode,
            chat_provider_id,
            req.num_retries,
            file_ids=effective_file_ids,
        )
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
    num_retries: int | None = None,
    file_ids: list[str] | None = None,
) -> None:
    """Run the use case in the background and store results."""
    try:
        # Use Chat Provider-specific adapter if available, otherwise global
        if chat_provider_id:
            llm = state.create_llm_adapter_for_chat_provider(chat_provider_id, num_retries)
            cp = state.get_chat_provider(chat_provider_id)
            if cp:
                cp = state.resolve_chat_provider(cp)
            provider_label = f"{cp.llm_provider}/{cp.llm_model}" if cp else "unknown"
        else:
            llm = state.create_llm_adapter(num_retries)
            cp = None
            provider_label = f"{state.config.active_provider}/{state.config.active_model}"
        logger.info(
            "Executing query [mode=%s, provider=%s]: %.100s",
            mode,
            provider_label,
            query,
        )

        # Build conversation context from Chat Provider history.
        # RLM and RAG derive all context from the document on each call — they do not
        # benefit from prior answers, and including them causes context-window overflow
        # on small models (e.g. 8K-context vLLM).  Only direct/compare modes need
        # conversational memory.
        conversation_history: list[dict[str, str]] = []
        if chat_provider_id and mode in ("direct", "compare"):
            prev_msgs = state.get_conversation(execution.session_id, chat_provider_id)
            # Keep last 3 turns; exclude error messages; trim long assistant answers
            eligible = [
                msg
                for msg in prev_msgs[:-1]
                if msg.get("role") in ("user", "assistant")
                and msg.get("content", "")
                and not str(msg.get("content", "")).startswith("Error:")
            ]
            for msg in eligible[-6:]:  # last 3 exchanges (6 messages)
                msg_content = msg.get("content", "")
                if len(msg_content) > 500:
                    msg_content = msg_content[:500] + "…"
                conversation_history.append(
                    {"role": msg.get("role", "user"), "content": msg_content}
                )

        # Build run config, using Chat Provider settings if available.
        # Apply RLM-specific knobs for rlm, auto, and compare (all run RLM internally).
        if cp and mode in ("rlm", "auto", "compare"):
            run_config = state.create_run_config(mode)
            run_config.max_steps = cp.rlm_max_steps
            run_config.max_time_seconds = float(cp.rlm_timeout_seconds)
            run_config.repeat_limit = cp.rlm_repeat_limit
            run_config.nudge_at_fraction = cp.rlm_nudge_at_fraction
            # Inject per-profile RLM system prompt (custom text or named template)
            if cp.profile_id:
                _prof = state.find_profile(cp.profile_id)
                if _prof:
                    _extra = _resolve_profile_prompt(_prof, "rlm")
                    if _extra:
                        run_config.system_prompt_extra = _extra
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
            result_direct = await asyncio.to_thread(
                uc_direct.execute, content, full_query, run_config
            )
            results = [result_rlm, result_direct]
        elif mode == "rag":
            rag_cfg = state.config.mode_config.rag_config
            # Resolve embedding API key: OpenAI key covers embeddings for any provider
            embedding_api_key: str | None = os.environ.get("OPENAI_API_KEY")
            if not embedding_api_key and cp:
                from rlmkit.server.routes.llm_providers import _get_api_key

                lp = state.get_llm_provider(cp.llm_provider_id) if cp.llm_provider_id else None
                if lp and lp.backend == "openai":
                    embedding_api_key = _get_api_key(lp.id, lp.backend)

            # Re-use indexed storage for follow-up messages on the same file set.
            # This skips the expensive chunk+embed step (can take 30s+ for large docs).
            cache_key = ":".join(sorted(file_ids)) if file_ids else ""
            cached = _rag_index_cache.get(cache_key) if cache_key else None
            if cached and cached[2] == rag_cfg.embedding_model:
                storage, collection, _ = cached
                skip_indexing = True
                logger.info("RAG: reusing cached index for cache_key=%s", cache_key)
            else:
                storage = SQLiteStorageAdapter(":memory:")
                collection = f"rag_{uuid.uuid4().hex}"
                skip_indexing = False

            run_config.extra["collection"] = collection
            run_config.extra["chunk_size"] = rag_cfg.chunk_size
            run_config.extra["top_k"] = rag_cfg.top_k
            embedder = LiteLLMEmbeddingAdapter(
                model=rag_cfg.embedding_model,
                api_key=embedding_api_key,
            )
            uc_rag = RunRAGUseCase(llm, embedder, storage)
            results = [
                await asyncio.to_thread(
                    uc_rag.execute, content, full_query, run_config, skip_indexing
                )
            ]
            # Populate cache after successful first index
            if cache_key and not skip_indexing and results[0].success:
                _rag_index_cache[cache_key] = (storage, collection, rag_cfg.embedding_model)
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
                lp = (
                    state.get_llm_provider(cp.llm_provider_id)
                    if cp and cp.llm_provider_id
                    else None
                )
                provider_name = (
                    (lp.name if lp else None)
                    or (cp.llm_provider if cp else None)
                    or state.config.active_provider
                )

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
            state.save_executions()

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
            state.save_sessions()
            state.save_executions()


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
                ws_file_ids: list[str] = data.get("file_ids") or (
                    [data["file_id"]] if data.get("file_id") else []
                )
                mode = data.get("mode", "auto")
                ws_chat_provider_id = data.get("chat_provider_id")

                if ws_file_ids:
                    try:
                        file_content = _resolve_file_content(ws_file_ids, state)
                        content = (
                            (content + "\n\n" + file_content).strip() if content else file_content
                        )
                    except HTTPException as exc:
                        await websocket.send_json(
                            {
                                "type": "error",
                                "id": msg_id,
                                "data": {
                                    "code": str(exc.status_code),
                                    "message": exc.detail,
                                    "recoverable": True,
                                },
                            }
                        )
                        continue

                # Resolve mode from Chat Provider if provided
                ws_cp = None
                if ws_chat_provider_id:
                    ws_cp = state.get_chat_provider(ws_chat_provider_id)
                    if ws_cp:
                        ws_cp = state.resolve_chat_provider(ws_cp)
                    if not ws_cp:
                        await websocket.send_json(
                            {
                                "type": "error",
                                "id": msg_id,
                                "data": {
                                    "code": "NOT_FOUND",
                                    "message": f"Chat Provider {ws_chat_provider_id} not found",
                                    "recoverable": True,
                                },
                            }
                        )
                        continue
                    mode = ws_cp.execution_mode

                # Create execution record (mirrors REST path) so traces/dashboard work
                exec_id = str(uuid.uuid4())
                now = datetime.now(timezone.utc)
                session = state.get_or_create_session(session_id)
                ws_cp_name = ws_cp.name if (ws_chat_provider_id and ws_cp) else None
                execution = ExecutionRecord(
                    execution_id=exec_id,
                    session_id=session.id,
                    query=query,
                    mode=mode,
                    started_at=now,
                    chat_provider_id=ws_chat_provider_id,
                    chat_provider_name=ws_cp_name,
                )
                state.executions[exec_id] = execution

                # Add user message to session
                user_msg = {
                    "id": str(uuid.uuid4()),
                    "role": "user",
                    "content": query,
                    "file_id": ws_file_ids[0] if ws_file_ids else None,  # compat alias
                    "file_ids": ws_file_ids,
                    "mode": mode,
                    "chat_provider_id": ws_chat_provider_id,
                    "timestamp": now.isoformat(),
                }
                state.add_message(session.id, user_msg, ws_chat_provider_id)
                session.updated_at = now
                # Save immediately so the session survives a server restart before execution completes
                state.save_sessions()

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
                            if ws_cp:
                                ws_cp = state.resolve_chat_provider(ws_cp)
                            ws_lp = (
                                state.get_llm_provider(ws_cp.llm_provider_id)
                                if ws_cp and ws_cp.llm_provider_id
                                else None
                            )
                            provider_label = (
                                ws_lp.name
                                if ws_lp
                                else (ws_cp.llm_provider or "unknown")
                                if ws_cp
                                else "unknown"
                            )
                        else:
                            llm = state.create_llm_adapter()
                            ws_cp = None
                            ws_lp = None
                            provider_label = (
                                f"{state.config.active_provider}/{state.config.active_model}"
                            )
                        logger.info(
                            "WS executing [mode=%s, provider=%s]: %.100s",
                            m,
                            provider_label,
                            q,
                        )
                        cfg = state.create_run_config(m)
                        # Thread Chat Provider RLM settings into run config
                        if ws_cp and m in ("rlm", "auto", "compare"):
                            cfg.max_steps = ws_cp.rlm_max_steps
                            cfg.max_time_seconds = float(ws_cp.rlm_timeout_seconds)
                            cfg.repeat_limit = ws_cp.rlm_repeat_limit
                            cfg.nudge_at_fraction = ws_cp.rlm_nudge_at_fraction
                            # Inject per-profile RLM system prompt (custom or template)
                            if ws_cp.profile_id:
                                _prof = state.find_profile(ws_cp.profile_id)
                                if _prof:
                                    _extra = _resolve_profile_prompt(_prof, "rlm")
                                    if _extra:
                                        cfg.system_prompt_extra = _extra

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
                            rag_cfg = state.config.mode_config.rag_config
                            emb_key: str | None = os.environ.get("OPENAI_API_KEY")
                            if not emb_key and ws_cp:
                                from rlmkit.server.routes.llm_providers import _get_api_key

                                ws_lp_obj = (
                                    state.get_llm_provider(ws_cp.llm_provider_id)
                                    if ws_cp.llm_provider_id
                                    else None
                                )
                                if ws_lp_obj and ws_lp_obj.backend == "openai":
                                    emb_key = _get_api_key(ws_lp_obj.id, ws_lp_obj.backend)
                            ws_collection = f"rag_{uuid.uuid4().hex}"
                            cfg.extra["collection"] = ws_collection
                            cfg.extra["chunk_size"] = rag_cfg.chunk_size
                            cfg.extra["top_k"] = rag_cfg.top_k
                            ws_embedder = LiteLLMEmbeddingAdapter(
                                model=rag_cfg.embedding_model,
                                api_key=emb_key,
                            )
                            ws_storage = SQLiteStorageAdapter(":memory:")
                            uc_rag = RunRAGUseCase(llm, ws_embedder, ws_storage)
                            results = [await asyncio.to_thread(uc_rag.execute, cnt, q, cfg)]
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
                            provider_name = (
                                (ws_lp.name if ws_lp else None)
                                or (ws_cp.llm_provider if ws_cp else None)
                                or state.config.active_provider
                            )

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
                        state.save_executions()
                    except asyncio.CancelledError:
                        pass
                    except Exception as exc:
                        logger.exception("WebSocket execution error for %s", mid)
                        finish = datetime.now(timezone.utc)
                        exec_rec.status = "error"
                        exec_rec.completed_at = finish
                        exec_rec.result = {"answer": "", "success": False, "error": str(exc)}
                        # Persist error message to session (mirrors REST path)
                        if sess:
                            error_msg = {
                                "id": str(uuid.uuid4()),
                                "role": "assistant",
                                "content": f"Error: {exc}",
                                "mode_used": m,
                                "execution_id": exec_rec.execution_id,
                                "chat_provider_id": cp_id,
                                "timestamp": finish.isoformat(),
                            }
                            state.add_message(sess.id, error_msg, cp_id)
                            sess.updated_at = finish
                            state.save_sessions()
                            state.save_executions()
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
                    _ws_execute(
                        websocket,
                        msg_id,
                        content,
                        query,
                        mode,
                        execution,
                        session,
                        ws_chat_provider_id,
                    )
                )
                active_tasks[msg_id] = task

    except WebSocketDisconnect:
        pass
    finally:
        ping_task.cancel()
        for task in active_tasks.values():
            task.cancel()

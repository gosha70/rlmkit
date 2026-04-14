"""CRUD endpoints for Chat Providers."""

from __future__ import annotations

import logging
import uuid
from datetime import datetime, timezone
from typing import Any

from fastapi import APIRouter, Depends, HTTPException

from rlmkit.application.sandbox_vars import (
    MODE_RLM,
    MODES_RLM_INTERNAL,
    RLM_DEFAULT_MAX_STEPS,
    RLM_DEFAULT_NUDGE_AT_FRACTION,
    RLM_DEFAULT_REPEAT_LIMIT,
    RLM_DEFAULT_WALL_CLOCK_BUDGET_SECONDS,
)
from rlmkit.server.dependencies import AppState, get_state
from rlmkit.server.models import (
    ChatProviderConfig,
    ChatProviderCreateRequest,
    ChatProviderUpdateRequest,
)

logger = logging.getLogger(__name__)

router = APIRouter()


def _check_rlm_context_window(
    state: AppState,
    cp: ChatProviderConfig,
) -> str | None:
    """Return a warning string if the resolved Chat Provider will run in
    RLM mode but the linked LLM Provider's context window is too small
    to fit the RLM system prompt + minimum working room.

    Returns ``None`` when no warning is needed (non-RLM mode, unknown
    context window, or sufficient headroom).
    """
    resolved = state.resolve_chat_provider(cp)
    effective_mode = resolved.execution_mode

    if effective_mode not in MODES_RLM_INTERNAL:
        return None

    # Get linked provider's context window
    lp = state.get_llm_provider(cp.llm_provider_id) if cp.llm_provider_id else None
    if not lp or not lp.context_window:
        return None  # unknown — can't warn

    context_window = lp.context_window

    # Compute minimum from the actual RLM system prompt + profile extras
    from rlmkit.prompts import format_system_prompt
    from rlmkit.server.routes.chat import _resolve_profile_prompt

    base_prompt = format_system_prompt(prompt_length=0)
    extra = ""
    if cp.profile_id:
        profile = state.find_profile(cp.profile_id)
        if profile:
            extra = _resolve_profile_prompt(profile, MODE_RLM) or ""
    full_prompt = base_prompt + ("\n\n" + extra if extra else "")

    from rlmkit.infrastructure.llm.litellm_adapter import (
        CONTEXT_WINDOW_RESERVE_FLOOR,
        CONTEXT_WINDOW_RESERVE_FRACTION,
        MIN_OUTPUT_TOKENS,
    )

    # Conservative token estimate: chars / 3 + overhead
    prompt_tokens = len(full_prompt) // 3 + 50

    # Minimum viable budget: prompt + modest content room + min_output + reserve
    reserve = max(
        int(context_window * CONTEXT_WINDOW_RESERVE_FRACTION), CONTEXT_WINDOW_RESERVE_FLOOR
    )
    min_viable = prompt_tokens + 500 + MIN_OUTPUT_TOKENS + reserve

    if context_window < min_viable:
        return (
            f"Context window ({context_window} tokens) may be too small for RLM mode. "
            f"The RLM system prompt alone needs ~{prompt_tokens} tokens, "
            f"leaving little room for document content and multi-step exploration. "
            f"Recommended minimum: {min_viable} tokens. "
            f"Consider increasing the context window on the LLM Provider, "
            f"or using Direct mode for this provider."
        )
    return None


@router.get("/api/chat-providers")
async def list_chat_providers(
    state: AppState = Depends(get_state),  # noqa: B008
) -> list[ChatProviderConfig]:
    """List all Chat Providers (with profile settings resolved).

    Ephemeral CPs created by compare-matrix are excluded.
    """
    return [
        state.resolve_chat_provider(cp) for cp in state.config.chat_providers if not cp.ephemeral
    ]


@router.get("/api/chat-providers/{chat_provider_id}")
async def get_chat_provider(
    chat_provider_id: str,
    state: AppState = Depends(get_state),  # noqa: B008
) -> ChatProviderConfig:
    """Get a single Chat Provider by ID."""
    cp = state.get_chat_provider(chat_provider_id)
    if not cp:
        raise HTTPException(status_code=404, detail="Chat Provider not found")
    return state.resolve_chat_provider(cp)


@router.post("/api/chat-providers", status_code=201, response_model=None)
async def create_chat_provider(
    req: ChatProviderCreateRequest,
    state: AppState = Depends(get_state),  # noqa: B008
) -> dict[str, Any]:
    """Create a new Chat Provider."""
    # Validate that the referenced LLM Provider exists
    lp = state.get_llm_provider(req.llm_provider_id)
    if not lp:
        raise HTTPException(
            status_code=400,
            detail=f"LLM Provider not found: {req.llm_provider_id}",
        )

    # Validate name uniqueness (skip ephemeral CPs from compare-matrix)
    for existing in state.config.chat_providers:
        if not existing.ephemeral and existing.name.lower() == req.name.lower():
            raise HTTPException(
                status_code=409,
                detail=f"Chat Provider with name '{req.name}' already exists",
            )

    # Validate profile_id if provided
    if req.profile_id:
        profile = state.find_profile(req.profile_id)
        if not profile:
            raise HTTPException(
                status_code=400,
                detail=f"Profile not found: {req.profile_id}",
            )

    now = datetime.now(timezone.utc)
    cp_kwargs: dict = {
        "id": str(uuid.uuid4()),
        "name": req.name,
        "llm_provider_id": req.llm_provider_id,
        "llm_provider_name": lp.name,
        "profile_id": req.profile_id,
        "execution_mode": req.execution_mode,
        "rag_config": req.rag_config,
        "rlm_max_steps": req.rlm_max_steps or RLM_DEFAULT_MAX_STEPS,
        "rlm_timeout_seconds": req.rlm_timeout_seconds or RLM_DEFAULT_WALL_CLOCK_BUDGET_SECONDS,
        "rlm_repeat_limit": req.rlm_repeat_limit or RLM_DEFAULT_REPEAT_LIMIT,
        "rlm_nudge_at_fraction": req.rlm_nudge_at_fraction or RLM_DEFAULT_NUDGE_AT_FRACTION,
        "num_retries": req.num_retries,
        "created_at": now,
        "updated_at": now,
    }
    # Only forward conversation_memory_* when the client supplied them;
    # otherwise let ChatProviderConfig use its defaults (enabled=True,
    # fraction=0.30) so the feature is opt-out rather than opt-in.
    if req.conversation_memory_enabled is not None:
        cp_kwargs["conversation_memory_enabled"] = req.conversation_memory_enabled
    if req.conversation_memory_fraction is not None:
        cp_kwargs["conversation_memory_fraction"] = req.conversation_memory_fraction
    cp = ChatProviderConfig(**cp_kwargs)
    state.config.chat_providers.append(cp)
    state.save_config()

    result: dict[str, Any] = state.resolve_chat_provider(cp).model_dump()
    warning = _check_rlm_context_window(state, cp)
    if warning:
        result["context_window_warning"] = warning
        logger.warning("Chat Provider '%s': %s", cp.name, warning)
    return result


@router.put("/api/chat-providers/{chat_provider_id}", response_model=None)
async def update_chat_provider(
    chat_provider_id: str,
    req: ChatProviderUpdateRequest,
    state: AppState = Depends(get_state),  # noqa: B008
) -> dict[str, Any]:
    """Update an existing Chat Provider."""
    cp = state.get_chat_provider(chat_provider_id)
    if not cp:
        raise HTTPException(status_code=404, detail="Chat Provider not found")

    # Validate name uniqueness if changing name (skip ephemeral CPs)
    if req.name is not None and req.name.lower() != cp.name.lower():
        for existing in state.config.chat_providers:
            if (
                not existing.ephemeral
                and existing.id != chat_provider_id
                and existing.name.lower() == req.name.lower()
            ):
                raise HTTPException(
                    status_code=409,
                    detail=f"Chat Provider with name '{req.name}' already exists",
                )
        cp.name = req.name

    # Validate and update llm_provider_id if provided
    if req.llm_provider_id is not None:
        lp = state.get_llm_provider(req.llm_provider_id)
        if not lp:
            raise HTTPException(
                status_code=400,
                detail=f"LLM Provider not found: {req.llm_provider_id}",
            )
        cp.llm_provider_id = req.llm_provider_id
        cp.llm_provider_name = lp.name

    # Validate profile_id if provided
    if req.profile_id is not None:
        if req.profile_id:
            profile = state.find_profile(req.profile_id)
            if not profile:
                raise HTTPException(
                    status_code=400,
                    detail=f"Profile not found: {req.profile_id}",
                )
        cp.profile_id = req.profile_id

    if req.execution_mode is not None:
        cp.execution_mode = req.execution_mode
    if req.rag_config is not None:
        cp.rag_config = req.rag_config
    if req.rlm_max_steps is not None:
        cp.rlm_max_steps = req.rlm_max_steps
    if req.rlm_timeout_seconds is not None:
        cp.rlm_timeout_seconds = req.rlm_timeout_seconds
    if req.rlm_repeat_limit is not None:
        cp.rlm_repeat_limit = req.rlm_repeat_limit
    if req.rlm_nudge_at_fraction is not None:
        cp.rlm_nudge_at_fraction = req.rlm_nudge_at_fraction
    if "num_retries" in req.model_fields_set:
        cp.num_retries = req.num_retries  # None clears back to provider-default chain
    if req.conversation_memory_enabled is not None:
        cp.conversation_memory_enabled = req.conversation_memory_enabled
    if req.conversation_memory_fraction is not None:
        cp.conversation_memory_fraction = req.conversation_memory_fraction
    cp.updated_at = datetime.now(timezone.utc)

    state.save_config()

    result: dict[str, Any] = state.resolve_chat_provider(cp).model_dump()
    warning = _check_rlm_context_window(state, cp)
    if warning:
        result["context_window_warning"] = warning
        logger.warning("Chat Provider '%s': %s", cp.name, warning)
    return result


@router.delete("/api/chat-providers/{chat_provider_id}", status_code=204)
async def delete_chat_provider(
    chat_provider_id: str,
    state: AppState = Depends(get_state),  # noqa: B008
) -> None:
    """Delete a Chat Provider."""
    original_len = len(state.config.chat_providers)
    state.config.chat_providers = [
        cp for cp in state.config.chat_providers if cp.id != chat_provider_id
    ]
    if len(state.config.chat_providers) == original_len:
        raise HTTPException(status_code=404, detail="Chat Provider not found")
    # Clear judge config if the deleted provider was the active judge
    if state.config.judge_chat_provider_id == chat_provider_id:
        state.config.judge_chat_provider_id = None
    state.save_config()

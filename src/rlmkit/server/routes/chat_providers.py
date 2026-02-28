"""CRUD endpoints for Chat Providers."""

from __future__ import annotations

import uuid
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException

from rlmkit.server.dependencies import AppState, get_state
from rlmkit.server.models import (
    ChatProviderConfig,
    ChatProviderCreateRequest,
    ChatProviderUpdateRequest,
)
from rlmkit.ui.data.providers_catalog import PROVIDERS_BY_KEY

router = APIRouter()


@router.get("/api/chat-providers")
async def list_chat_providers(
    state: AppState = Depends(get_state),  # noqa: B008
) -> list[ChatProviderConfig]:
    """List all Chat Providers."""
    return state.config.chat_providers


@router.get("/api/chat-providers/{chat_provider_id}")
async def get_chat_provider(
    chat_provider_id: str,
    state: AppState = Depends(get_state),  # noqa: B008
) -> ChatProviderConfig:
    """Get a single Chat Provider by ID."""
    cp = state.get_chat_provider(chat_provider_id)
    if not cp:
        raise HTTPException(status_code=404, detail="Chat Provider not found")
    return cp


@router.post("/api/chat-providers", status_code=201)
async def create_chat_provider(
    req: ChatProviderCreateRequest,
    state: AppState = Depends(get_state),  # noqa: B008
) -> ChatProviderConfig:
    """Create a new Chat Provider."""
    # Validate that the LLM provider exists in catalog
    if req.llm_provider not in PROVIDERS_BY_KEY:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown LLM provider: {req.llm_provider}. "
            f"Available: {list(PROVIDERS_BY_KEY.keys())}",
        )

    # Validate name uniqueness
    for existing in state.config.chat_providers:
        if existing.name.lower() == req.name.lower():
            raise HTTPException(
                status_code=409,
                detail=f"Chat Provider with name '{req.name}' already exists",
            )

    now = datetime.now(timezone.utc)
    cp = ChatProviderConfig(
        id=str(uuid.uuid4()),
        name=req.name,
        llm_provider=req.llm_provider,
        llm_model=req.llm_model,
        execution_mode=req.execution_mode,
        runtime_settings=req.runtime_settings or ChatProviderConfig.model_fields["runtime_settings"].default_factory(),
        rag_config=req.rag_config,
        rlm_max_steps=req.rlm_max_steps or 16,
        rlm_timeout_seconds=req.rlm_timeout_seconds or 60,
        created_at=now,
        updated_at=now,
    )
    state.config.chat_providers.append(cp)
    state.save_config()
    return cp


@router.put("/api/chat-providers/{chat_provider_id}")
async def update_chat_provider(
    chat_provider_id: str,
    req: ChatProviderUpdateRequest,
    state: AppState = Depends(get_state),  # noqa: B008
) -> ChatProviderConfig:
    """Update an existing Chat Provider."""
    cp = state.get_chat_provider(chat_provider_id)
    if not cp:
        raise HTTPException(status_code=404, detail="Chat Provider not found")

    # Validate name uniqueness if changing name
    if req.name is not None and req.name.lower() != cp.name.lower():
        for existing in state.config.chat_providers:
            if existing.id != chat_provider_id and existing.name.lower() == req.name.lower():
                raise HTTPException(
                    status_code=409,
                    detail=f"Chat Provider with name '{req.name}' already exists",
                )
        cp.name = req.name

    if req.llm_model is not None:
        cp.llm_model = req.llm_model
    if req.execution_mode is not None:
        cp.execution_mode = req.execution_mode
    if req.runtime_settings is not None:
        cp.runtime_settings = req.runtime_settings
    if req.rag_config is not None:
        cp.rag_config = req.rag_config
    if req.rlm_max_steps is not None:
        cp.rlm_max_steps = req.rlm_max_steps
    if req.rlm_timeout_seconds is not None:
        cp.rlm_timeout_seconds = req.rlm_timeout_seconds
    cp.updated_at = datetime.now(timezone.utc)

    state.save_config()
    return cp


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
    state.save_config()

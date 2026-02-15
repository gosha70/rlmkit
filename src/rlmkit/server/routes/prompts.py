"""System prompt management endpoints."""

from __future__ import annotations

from typing import List

from fastapi import APIRouter, Depends

from rlmkit.server.dependencies import AppState, get_state
from rlmkit.server.models import SystemPrompts, SystemPromptTemplate
from rlmkit.ui.services.profile_store import SYSTEM_PROMPT_TEMPLATES

router = APIRouter()


@router.get("/api/system-prompts")
async def get_system_prompts(
    state: AppState = Depends(get_state),
) -> SystemPrompts:
    """Get current active system prompts (per-mode)."""
    return state.system_prompts


@router.put("/api/system-prompts")
async def update_system_prompts(
    req: SystemPrompts,
    state: AppState = Depends(get_state),
) -> SystemPrompts:
    """Update system prompts (per-mode)."""
    state.system_prompts = req
    state.save_config()
    return state.system_prompts


@router.get("/api/system-prompts/templates")
async def list_templates() -> List[SystemPromptTemplate]:
    """List built-in system prompt templates."""
    result = []
    for name, data in SYSTEM_PROMPT_TEMPLATES.items():
        prompts = {k: v for k, v in data.items() if k != "description"}
        result.append(
            SystemPromptTemplate(
                name=name,
                description=data.get("description", ""),
                prompts=prompts,
            )
        )
    return result

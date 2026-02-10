"""Configuration endpoints."""

from __future__ import annotations

from fastapi import APIRouter, Depends

from rlmkit.server.dependencies import AppState, get_state
from rlmkit.server.models import ConfigResponse, ConfigUpdateRequest

router = APIRouter()


@router.get("/api/config")
async def get_config(
    state: AppState = Depends(get_state),
) -> ConfigResponse:
    """Get current configuration."""
    return state.config


@router.put("/api/config")
async def update_config(
    req: ConfigUpdateRequest,
    state: AppState = Depends(get_state),
) -> ConfigResponse:
    """Update configuration (partial update -- merges fields, not replaces)."""
    if req.active_provider is not None:
        state.config.active_provider = req.active_provider
    if req.active_model is not None:
        state.config.active_model = req.active_model
    if req.budget is not None:
        for key, value in req.budget.model_dump(exclude_unset=True).items():
            setattr(state.config.budget, key, value)
    if req.sandbox is not None:
        for key, value in req.sandbox.model_dump(exclude_unset=True).items():
            setattr(state.config.sandbox, key, value)
    if req.appearance is not None:
        for key, value in req.appearance.model_dump(exclude_unset=True).items():
            setattr(state.config.appearance, key, value)
    return state.config

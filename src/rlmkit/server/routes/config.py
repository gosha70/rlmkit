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
    """Update configuration (partial update)."""
    if req.active_provider is not None:
        state.config.active_provider = req.active_provider
    if req.active_model is not None:
        state.config.active_model = req.active_model
    if req.budget is not None:
        state.config.budget = req.budget
    if req.sandbox is not None:
        state.config.sandbox = req.sandbox
    if req.appearance is not None:
        state.config.appearance = req.appearance
    return state.config

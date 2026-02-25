"""Configuration endpoints."""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends

from rlmkit.server.dependencies import AppState, get_state
from rlmkit.server.models import ConfigResponse, ConfigUpdateRequest

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("/api/config")
async def get_config(
    state: AppState = Depends(get_state),  # noqa: B008
) -> ConfigResponse:
    """Get current configuration."""
    return state.config


@router.put("/api/config")
async def update_config(
    req: ConfigUpdateRequest,
    state: AppState = Depends(get_state),  # noqa: B008
) -> ConfigResponse:
    """Update configuration (partial update -- merges fields, not replaces).

    Note: active_provider and active_model are set exclusively via
    PUT /api/providers/{name} to prevent two disconnected save paths
    from corrupting each other.
    """
    if req.budget is not None:
        for key, value in req.budget.model_dump(exclude_unset=True).items():
            setattr(state.config.budget, key, value)
    if req.sandbox is not None:
        for key, value in req.sandbox.model_dump(exclude_unset=True).items():
            setattr(state.config.sandbox, key, value)
    if req.appearance is not None:
        for key, value in req.appearance.model_dump(exclude_unset=True).items():
            setattr(state.config.appearance, key, value)
    if req.provider_configs is not None:
        state.config.provider_configs = req.provider_configs
    if req.default_runtime_settings is not None:
        state.config.default_runtime_settings = req.default_runtime_settings
    if req.mode_config is not None:
        for key, value in req.mode_config.model_dump(exclude_unset=True).items():
            setattr(state.config.mode_config, key, value)
    state.save_config()
    return state.config

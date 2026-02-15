"""Run profile management endpoints."""

from __future__ import annotations

import uuid
from typing import List

from fastapi import APIRouter, Depends, HTTPException

from rlmkit.server.dependencies import AppState, get_state
from rlmkit.server.models import (
    RunProfile,
    RunProfileCreate,
    RunProfileUpdate,
)

router = APIRouter()

# Built-in profiles (read-only, always available)
BUILTIN_PROFILES: List[RunProfile] = [
    RunProfile(
        id="builtin-fast",
        name="Fast & cheap",
        description="Low-cost, fast responses using Direct mode with conservative token limits.",
        strategy="direct",
        is_builtin=True,
        runtime_settings={"temperature": 0.5, "top_p": 1.0, "max_output_tokens": 1000, "timeout_seconds": 15},
        budget={"max_steps": 8, "max_tokens": 20000, "max_cost_usd": 0.5, "max_time_seconds": 15, "max_recursion_depth": 3},
    ),
    RunProfile(
        id="builtin-accurate",
        name="Accurate",
        description="High-quality responses with lower temperature for precision.",
        strategy="direct",
        is_builtin=True,
        runtime_settings={"temperature": 0.2, "top_p": 0.95, "max_output_tokens": 4096, "timeout_seconds": 30},
        budget={"max_steps": 16, "max_tokens": 50000, "max_cost_usd": 2.0, "max_time_seconds": 30, "max_recursion_depth": 5},
    ),
    RunProfile(
        id="builtin-rlm-deep",
        name="RLM deep",
        description="Deep recursive reasoning with high step budget for complex problems.",
        strategy="rlm",
        is_builtin=True,
        runtime_settings={"temperature": 0.4, "top_p": 1.0, "max_output_tokens": 4096, "timeout_seconds": 120},
        budget={"max_steps": 32, "max_tokens": 100000, "max_cost_usd": 5.0, "max_time_seconds": 120, "max_recursion_depth": 8},
    ),
]


@router.get("/api/profiles")
async def list_profiles(
    state: AppState = Depends(get_state),
) -> List[RunProfile]:
    """List all profiles (builtins + user-created)."""
    return BUILTIN_PROFILES + state.user_profiles


@router.post("/api/profiles", status_code=201)
async def create_profile(
    req: RunProfileCreate,
    state: AppState = Depends(get_state),
) -> RunProfile:
    """Create a new user profile."""
    profile = RunProfile(
        id=str(uuid.uuid4()),
        name=req.name,
        description=req.description,
        strategy=req.strategy,
        default_provider=req.default_provider,
        providers_enabled=req.providers_enabled,
        runtime_settings=req.runtime_settings,
        budget=req.budget,
        system_prompts=req.system_prompts,
        is_builtin=False,
    )
    state.user_profiles.append(profile)
    state.save_config()
    return profile


@router.put("/api/profiles/{profile_id}")
async def update_profile(
    profile_id: str,
    req: RunProfileUpdate,
    state: AppState = Depends(get_state),
) -> RunProfile:
    """Update a user profile. Cannot modify builtins."""
    # Check builtins
    for bp in BUILTIN_PROFILES:
        if bp.id == profile_id:
            raise HTTPException(status_code=400, detail="Cannot modify built-in profiles")

    # Find user profile
    for profile in state.user_profiles:
        if profile.id == profile_id:
            if req.name is not None:
                profile.name = req.name
            if req.description is not None:
                profile.description = req.description
            if req.strategy is not None:
                profile.strategy = req.strategy
            if req.default_provider is not None:
                profile.default_provider = req.default_provider
            if req.providers_enabled is not None:
                profile.providers_enabled = req.providers_enabled
            if req.runtime_settings is not None:
                profile.runtime_settings = req.runtime_settings
            if req.budget is not None:
                profile.budget = req.budget
            if req.system_prompts is not None:
                profile.system_prompts = req.system_prompts
            state.save_config()
            return profile

    raise HTTPException(status_code=404, detail=f"Profile not found: {profile_id}")


@router.delete("/api/profiles/{profile_id}", status_code=204)
async def delete_profile(
    profile_id: str,
    state: AppState = Depends(get_state),
) -> None:
    """Delete a user profile. Cannot delete builtins."""
    for bp in BUILTIN_PROFILES:
        if bp.id == profile_id:
            raise HTTPException(status_code=400, detail="Cannot delete built-in profiles")

    for i, profile in enumerate(state.user_profiles):
        if profile.id == profile_id:
            state.user_profiles.pop(i)
            state.save_config()
            return

    raise HTTPException(status_code=404, detail=f"Profile not found: {profile_id}")


@router.post("/api/profiles/{profile_id}/activate")
async def activate_profile(
    profile_id: str,
    state: AppState = Depends(get_state),
) -> RunProfile:
    """Apply a profile's settings to the current configuration."""
    # Search builtins + user profiles
    all_profiles = BUILTIN_PROFILES + state.user_profiles
    profile = None
    for p in all_profiles:
        if p.id == profile_id:
            profile = p
            break

    if profile is None:
        raise HTTPException(status_code=404, detail=f"Profile not found: {profile_id}")

    # Apply profile settings to current config
    state.config.budget = profile.budget.model_copy() if hasattr(profile.budget, 'model_copy') else profile.budget
    state.config.default_runtime_settings = (
        profile.runtime_settings.model_copy()
        if hasattr(profile.runtime_settings, 'model_copy')
        else profile.runtime_settings
    )
    state.save_config()

    return profile

"""CRUD endpoints for named LLM Provider instances."""

from __future__ import annotations

import logging
import os
import time
import uuid
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException

from rlmkit.server.dependencies import AppState, get_state
from rlmkit.server.models import (
    LLMProviderConfig,
    LLMProviderCreateRequest,
    LLMProviderUpdateRequest,
    ModelInfo,
    ProviderTestResponse,
    RuntimeSettings,
)
from rlmkit.ui.data.providers_catalog import PROVIDERS_BY_KEY
from rlmkit.ui.services.secret_store import FileSecretStore, KeyringSecretStore

router = APIRouter()
logger = logging.getLogger(__name__)

# In-memory status cache: llm_provider_id -> status
_status_cache: dict[str, str] = {}


def _persist_api_key(llm_provider_id: str, backend: str, api_key: str) -> None:
    """Persist API key for a named LLM Provider instance."""
    key_name = f"llm_provider:{llm_provider_id}"
    # Also set the backend env var so the process can use the key immediately
    entry = PROVIDERS_BY_KEY.get(backend)
    if entry and entry.env_var:
        os.environ[entry.env_var] = api_key

    if KeyringSecretStore.is_available():
        KeyringSecretStore().set(key_name, api_key)
    else:
        FileSecretStore().set(key_name, api_key)


def _get_api_key(llm_provider_id: str, backend: str) -> str | None:
    """Retrieve API key for a named LLM Provider (instance key first, then env var)."""
    key_name = f"llm_provider:{llm_provider_id}"
    store = KeyringSecretStore() if KeyringSecretStore.is_available() else FileSecretStore()
    raw_key: str | None = store.get(key_name)
    if raw_key:
        return raw_key
    # Fall back to env var
    entry = PROVIDERS_BY_KEY.get(backend)
    if entry and entry.env_var:
        return os.environ.get(entry.env_var)
    return None


def _compute_status(lp: LLMProviderConfig) -> str:
    """Compute current status for an LLM Provider instance."""
    # Check in-memory cache first
    cached = _status_cache.get(lp.id)
    if cached:
        return cached
    # Persisted status from last test
    if lp.status == "connected":
        return "connected"
    entry = PROVIDERS_BY_KEY.get(lp.backend)
    if not entry:
        return "not_configured"
    if entry.requires_api_key:
        key = _get_api_key(lp.id, lp.backend)
        if key:
            return lp.status if lp.status in ("connected", "configured") else "configured"
        return "not_configured"
    # Local provider (Ollama, LM Studio) — configured by default
    return lp.status if lp.status == "connected" else "configured"


@router.get("/api/llm-providers")
async def list_llm_providers(
    state: AppState = Depends(get_state),  # noqa: B008
) -> list[LLMProviderConfig]:
    """List all named LLM Provider instances with computed status."""
    result = []
    for lp in state.config.llm_providers:
        copy = lp.model_copy()
        copy.status = _compute_status(lp)
        result.append(copy)
    return result


@router.post("/api/llm-providers", status_code=201)
async def create_llm_provider(
    req: LLMProviderCreateRequest,
    state: AppState = Depends(get_state),  # noqa: B008
) -> LLMProviderConfig:
    """Create a new named LLM Provider instance."""
    if req.backend not in PROVIDERS_BY_KEY:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown backend: {req.backend}. Available: {list(PROVIDERS_BY_KEY.keys())}",
        )
    # Name must be unique
    for lp in state.config.llm_providers:
        if lp.name.lower() == req.name.lower():
            raise HTTPException(
                status_code=409,
                detail=f"LLM Provider with name '{req.name}' already exists",
            )

    now = datetime.now(timezone.utc)
    lp = LLMProviderConfig(
        id=str(uuid.uuid4()),
        name=req.name,
        backend=req.backend,
        model=req.model,
        endpoint=req.endpoint,
        runtime_settings=req.runtime_settings or RuntimeSettings(),
        status="not_configured",
        created_at=now,
        updated_at=now,
    )

    if req.api_key:
        _persist_api_key(lp.id, req.backend, req.api_key)
        lp.status = "configured"

    state.config.llm_providers.append(lp)
    state.save_config()

    result = lp.model_copy()
    result.status = _compute_status(lp)
    return result


@router.get("/api/llm-providers/{llm_provider_id}")
async def get_llm_provider(
    llm_provider_id: str,
    state: AppState = Depends(get_state),  # noqa: B008
) -> LLMProviderConfig:
    """Get a single named LLM Provider by ID."""
    lp = state.get_llm_provider(llm_provider_id)
    if not lp:
        raise HTTPException(status_code=404, detail="LLM Provider not found")
    result = lp.model_copy()
    result.status = _compute_status(lp)
    return result


@router.put("/api/llm-providers/{llm_provider_id}")
async def update_llm_provider(
    llm_provider_id: str,
    req: LLMProviderUpdateRequest,
    state: AppState = Depends(get_state),  # noqa: B008
) -> LLMProviderConfig:
    """Update an existing named LLM Provider."""
    lp = state.get_llm_provider(llm_provider_id)
    if not lp:
        raise HTTPException(status_code=404, detail="LLM Provider not found")

    if req.name is not None and req.name.lower() != lp.name.lower():
        for existing in state.config.llm_providers:
            if existing.id != llm_provider_id and existing.name.lower() == req.name.lower():
                raise HTTPException(
                    status_code=409,
                    detail=f"LLM Provider with name '{req.name}' already exists",
                )
        lp.name = req.name

    if req.model is not None:
        lp.model = req.model
    if req.endpoint is not None:
        lp.endpoint = req.endpoint or None
    if req.runtime_settings is not None:
        lp.runtime_settings = req.runtime_settings
    if req.api_key:
        _persist_api_key(lp.id, lp.backend, req.api_key)
        lp.status = "configured"
        _status_cache.pop(lp.id, None)

    lp.updated_at = datetime.now(timezone.utc)
    state.save_config()

    result = lp.model_copy()
    result.status = _compute_status(lp)
    return result


@router.delete("/api/llm-providers/{llm_provider_id}", status_code=204)
async def delete_llm_provider(
    llm_provider_id: str,
    state: AppState = Depends(get_state),  # noqa: B008
) -> None:
    """Delete a named LLM Provider. Fails if any Chat Provider references it."""
    lp = state.get_llm_provider(llm_provider_id)
    if not lp:
        raise HTTPException(status_code=404, detail="LLM Provider not found")

    # Check referential integrity
    refs = [cp.name for cp in state.config.chat_providers if cp.llm_provider_id == llm_provider_id]
    if refs:
        raise HTTPException(
            status_code=409,
            detail=f"Cannot delete: referenced by Chat Provider(s): {', '.join(refs)}",
        )

    state.config.llm_providers = [
        lp for lp in state.config.llm_providers if lp.id != llm_provider_id
    ]
    _status_cache.pop(llm_provider_id, None)
    state.save_config()


@router.post("/api/llm-providers/{llm_provider_id}/test")
async def test_llm_provider(
    llm_provider_id: str,
    state: AppState = Depends(get_state),  # noqa: B008
) -> ProviderTestResponse:
    """Test connection to a named LLM Provider."""
    lp = state.get_llm_provider(llm_provider_id)
    if not lp:
        raise HTTPException(status_code=404, detail="LLM Provider not found")

    import litellm

    from rlmkit.server.routes.providers import _litellm_model_name

    model = _litellm_model_name(lp.backend, lp.model)
    params: dict = {
        "model": model,
        "messages": [{"role": "user", "content": "ping"}],
        "max_tokens": 5,
        "timeout": 15,
    }

    # API key for cloud providers
    entry = PROVIDERS_BY_KEY.get(lp.backend)
    if entry and entry.requires_api_key:
        api_key = _get_api_key(lp.id, lp.backend)
        if api_key:
            params["api_key"] = api_key

    # Endpoint for local providers
    endpoint = lp.endpoint
    if not endpoint and entry and entry.default_endpoint:
        endpoint = entry.default_endpoint
    if endpoint:
        params["api_base"] = endpoint

    start = time.time()
    try:
        response = litellm.completion(**params)
        latency_ms = int((time.time() - start) * 1000)
        if response.choices:
            lp.status = "connected"
            _status_cache[lp.id] = "connected"
            state.save_config()
            return ProviderTestResponse(connected=True, latency_ms=latency_ms, model=lp.model)
        lp.status = "offline"
        _status_cache[lp.id] = "offline"
        state.save_config()
        return ProviderTestResponse(connected=False, error="No response from model")
    except Exception as exc:
        msg = str(exc)
        if " - " in msg:
            msg = msg.split(" - ", 1)[1]
        lp.status = "offline"
        _status_cache[lp.id] = "offline"
        state.save_config()
        return ProviderTestResponse(connected=False, error=msg[:300])


@router.get("/api/llm-providers/{llm_provider_id}/models")
async def list_llm_provider_models(
    llm_provider_id: str,
    state: AppState = Depends(get_state),  # noqa: B008
) -> list[ModelInfo]:
    """Fetch live models for a named LLM Provider instance."""
    from rlmkit.server.routes.providers import (
        _build_model_infos,
        _fetch_models_from_api,
    )

    lp = state.get_llm_provider(llm_provider_id)
    if not lp:
        raise HTTPException(status_code=404, detail="LLM Provider not found")

    try:
        live: list[ModelInfo] = await _fetch_models_from_api(
            lp.backend, endpoint_override=lp.endpoint
        )
        if live:
            # Successful model fetch proves connectivity — mark as connected
            if lp.status != "connected":
                lp.status = "connected"
                _status_cache[lp.id] = "connected"
                state.save_config()
            return live
    except Exception as exc:
        logger.warning("Failed to fetch models for LLM Provider %s: %s", lp.name, exc)

    catalog: list[ModelInfo] = _build_model_infos(lp.backend)
    return catalog

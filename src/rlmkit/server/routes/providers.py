"""Provider management endpoints."""

from __future__ import annotations

import logging
import os
import time
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException

from rlmkit.server.dependencies import AppState, get_state
from rlmkit.server.models import (
    ModelInfo,
    ProviderConfig,
    ProviderInfo,
    ProviderSaveRequest,
    ProviderSaveResponse,
    ProviderTestRequest,
    ProviderTestResponse,
)
from rlmkit.ui.data.providers_catalog import PROVIDERS, PROVIDERS_BY_KEY

logger = logging.getLogger(__name__)

router = APIRouter()

# LiteLLM provider prefixes: models must be prefixed for non-OpenAI providers
_LITELLM_PREFIXES: dict[str, str] = {
    "anthropic": "anthropic/",
    "ollama": "ollama/",
    "lmstudio": "openai/",  # LM Studio uses OpenAI-compatible API
}


def _litellm_model_name(provider_key: str, model: str) -> str:
    """Prepend the LiteLLM provider prefix if needed.

    E.g. 'ollama' + 'qwen3' -> 'ollama/qwen3'
    OpenAI models don't need a prefix. Anthropic models handled by LiteLLM
    but explicit prefix is safer.
    """
    # If the model already has a prefix (user typed 'ollama/qwen3'), leave it
    if "/" in model:
        return model
    prefix = _LITELLM_PREFIXES.get(provider_key, "")
    return f"{prefix}{model}"


def _build_model_infos(provider_key: str) -> list[ModelInfo]:
    """Convert catalog ModelInfo dataclasses to Pydantic ModelInfo."""
    entry = PROVIDERS_BY_KEY.get(provider_key)
    if not entry or not entry.models:
        return []
    return [
        ModelInfo(
            name=m.name,
            input_cost_per_1k=m.input_cost_per_1k,
            output_cost_per_1k=m.output_cost_per_1k,
        )
        for m in entry.models
    ]


@router.get("/api/providers")
async def list_providers(
    state: AppState = Depends(get_state),  # noqa: B008
) -> list[ProviderInfo]:
    """List configured providers with status.

    Status values:
    - "connected": API key present AND health check passed (cached)
    - "configured": API key present but not yet verified
    - "not_configured": no API key set
    """
    result = []
    for p in PROVIDERS:
        configured = False
        if p.env_var:
            configured = bool(os.environ.get(p.env_var))
        elif not p.requires_api_key:
            # Local providers (Ollama, LM Studio) are always "configured"
            configured = True

        # Use cached status from successful test, otherwise just "configured"
        cached = _provider_status_cache.get(p.key)
        if cached == "connected":
            status = "connected"
        elif configured:
            status = "configured"
        else:
            status = "not_configured"

        default_model = p.models[0].name if p.models else None

        result.append(
            ProviderInfo(
                name=p.key,
                display_name=p.display_name,
                status=status,
                models=_build_model_infos(p.key),
                default_model=default_model,
                configured=configured,
                requires_api_key=p.requires_api_key,
                default_endpoint=p.default_endpoint,
                model_input_hint=p.model_input_hint,
            )
        )
    return result


# Cache of verified provider statuses (populated by test_provider endpoint)
_provider_status_cache: dict[str, str] = {}


@router.post("/api/providers/test")
async def test_provider(
    req: ProviderTestRequest,
    state: AppState = Depends(get_state),  # noqa: B008, PT028
) -> ProviderTestResponse:
    """Test a provider connection."""
    model = req.model
    if not model:
        entry = PROVIDERS_BY_KEY.get(req.provider)
        if entry and entry.models:
            model = entry.models[0].name
    if not model:
        model = "gpt-4o"

    # LiteLLM requires provider prefix for non-OpenAI models
    litellm_model = _litellm_model_name(req.provider, model)
    logger.info(
        "Testing provider=%s model=%s (litellm_model=%s)", req.provider, model, litellm_model
    )

    import litellm

    params: dict = {
        "model": litellm_model,
        "messages": [{"role": "user", "content": "ping"}],
        "max_tokens": 5,
        "timeout": 30,
    }
    if req.api_key:
        params["api_key"] = req.api_key

    # Use explicit endpoint or fall back to provider's default
    endpoint = req.endpoint
    if not endpoint:
        entry = PROVIDERS_BY_KEY.get(req.provider)
        if entry and entry.default_endpoint:
            endpoint = entry.default_endpoint
    if endpoint:
        params["api_base"] = endpoint
        logger.debug("Using api_base=%s", endpoint)

    start = time.time()
    try:
        response = litellm.completion(**params)
        latency_ms = int((time.time() - start) * 1000)
        if response.choices:
            logger.info("Provider %s connected OK (%dms)", req.provider, latency_ms)
            _provider_status_cache[req.provider] = "connected"
            return ProviderTestResponse(
                connected=True,
                latency_ms=latency_ms,
                model=model,
            )
        logger.warning("Provider %s returned no choices", req.provider)
        _provider_status_cache[req.provider] = "offline"
        return ProviderTestResponse(
            connected=False,
            error="No response from model",
        )
    except Exception as exc:
        logger.error("Provider %s test failed: %s", req.provider, exc)
        _provider_status_cache[req.provider] = "offline"
        # Extract the useful part of the error message
        msg = str(exc)
        # LiteLLM wraps errors like "AnthropicException - {json}"
        if " - " in msg:
            msg = msg.split(" - ", 1)[1]
        return ProviderTestResponse(
            connected=False,
            error=msg[:300],
        )


def _update_env_file(key: str, value: str) -> None:
    """Write or update a key=value pair in the project .env file."""
    env_path = Path(".env")
    lines: list[str] = []
    if env_path.exists():
        lines = env_path.read_text().splitlines(keepends=True)
    found = False
    for i, line in enumerate(lines):
        if line.strip().startswith(f"{key}="):
            lines[i] = f"{key}={value}\n"
            found = True
            break
    if not found:
        lines.append(f"{key}={value}\n")
    env_path.write_text("".join(lines))


@router.put("/api/providers/{provider_name}")
async def save_provider(
    provider_name: str,
    req: ProviderSaveRequest,
    state: AppState = Depends(get_state),  # noqa: B008
) -> ProviderSaveResponse:
    """Save provider configuration. Writes API key to .env and os.environ.

    Also persists runtime_settings and enabled state in AppState.config.
    """
    entry = PROVIDERS_BY_KEY.get(provider_name)
    if entry is None:
        raise HTTPException(status_code=404, detail=f"Unknown provider: {provider_name}")

    env_var = entry.env_var

    if req.api_key and env_var:
        # Persist to .env file and update current process environment
        _update_env_file(env_var, req.api_key)
        os.environ[env_var] = req.api_key
        logger.info("Saved API key for %s to .env (%s)", provider_name, env_var)

    # Update provider config in AppState
    _update_provider_config(state, provider_name, req)

    # If this provider is being enabled, set it as the active provider
    # and disable all others (only one active at a time)
    if req.enabled:
        state.config.active_provider = provider_name
        if req.model:
            state.config.active_model = req.model
        for pc in state.config.provider_configs:
            if pc.provider != provider_name:
                pc.enabled = False

    state.save_config()

    return ProviderSaveResponse(
        saved=True,
        provider=provider_name,
        env_var=env_var,
        message=f"API key saved to .env as {env_var}"
        if env_var and req.api_key
        else "Configuration saved",
    )


def _update_provider_config(state: AppState, provider_name: str, req: ProviderSaveRequest) -> None:
    """Upsert provider config in state.config.provider_configs."""
    configs = state.config.provider_configs

    # Find existing or create new
    existing: ProviderConfig | None = None
    for pc in configs:
        if pc.provider == provider_name:
            existing = pc
            break

    if existing is None:
        entry = PROVIDERS_BY_KEY.get(provider_name)
        default_model = entry.models[0].name if entry and entry.models else ""
        existing = ProviderConfig(
            provider=provider_name,
            model=req.model or default_model,
        )
        configs.append(existing)

    if req.model is not None:
        existing.model = req.model
    if req.runtime_settings is not None:
        existing.runtime_settings = req.runtime_settings
    if req.enabled is not None:
        existing.enabled = req.enabled

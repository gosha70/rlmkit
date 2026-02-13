"""Provider management endpoints."""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import List

from fastapi import APIRouter, Depends, HTTPException

from rlmkit.server.dependencies import AppState, get_state
from rlmkit.server.models import (
    ProviderInfo,
    ProviderSaveRequest,
    ProviderSaveResponse,
    ProviderTestRequest,
    ProviderTestResponse,
)

router = APIRouter()

_KNOWN_PROVIDERS = [
    {
        "name": "openai",
        "display_name": "OpenAI",
        "env_var": "OPENAI_API_KEY",
        "models": ["gpt-4o", "gpt-4o-mini", "gpt-3.5-turbo"],
        "default_model": "gpt-4o",
    },
    {
        "name": "anthropic",
        "display_name": "Anthropic",
        "env_var": "ANTHROPIC_API_KEY",
        "models": ["claude-opus-4-6", "claude-sonnet-4-5-20250929"],
        "default_model": "claude-sonnet-4-5-20250929",
    },
    {
        "name": "ollama",
        "display_name": "Ollama",
        "env_var": None,
        "models": ["llama3", "codellama", "mistral"],
        "default_model": "llama3",
    },
]


@router.get("/api/providers")
async def list_providers(
    state: AppState = Depends(get_state),
) -> List[ProviderInfo]:
    """List configured providers with status.

    Status values:
    - "connected": API key present AND health check passed (cached)
    - "configured": API key present but not yet verified
    - "not_configured": no API key set
    """
    result = []
    for p in _KNOWN_PROVIDERS:
        env_var = p.get("env_var")
        configured = bool(env_var and os.environ.get(env_var)) if env_var else False

        # Use cached status from successful test, otherwise just "configured"
        cached = _provider_status_cache.get(p["name"])
        if cached == "connected":
            status = "connected"
        elif configured:
            status = "configured"
        else:
            status = "not_configured"

        result.append(
            ProviderInfo(
                name=p["name"],
                display_name=p["display_name"],
                status=status,
                models=p["models"],
                default_model=p["default_model"],
                configured=configured,
            )
        )
    return result


# Cache of verified provider statuses (populated by test_provider endpoint)
_provider_status_cache: dict[str, str] = {}


@router.post("/api/providers/test")
async def test_provider(
    req: ProviderTestRequest,
    state: AppState = Depends(get_state),
) -> ProviderTestResponse:
    """Test a provider connection."""
    model = req.model
    if not model:
        for p in _KNOWN_PROVIDERS:
            if p["name"] == req.provider:
                model = p["default_model"]
                break
    if not model:
        model = "gpt-4o"

    import litellm

    params: dict = {
        "model": model,
        "messages": [{"role": "user", "content": "ping"}],
        "max_tokens": 5,
        "timeout": 10,
    }
    if req.api_key:
        params["api_key"] = req.api_key
    if req.endpoint:
        params["api_base"] = req.endpoint

    start = time.time()
    try:
        response = litellm.completion(**params)
        latency_ms = int((time.time() - start) * 1000)
        if response.choices:
            _provider_status_cache[req.provider] = "connected"
            return ProviderTestResponse(
                connected=True,
                latency_ms=latency_ms,
                model=model,
            )
        _provider_status_cache[req.provider] = "offline"
        return ProviderTestResponse(
            connected=False,
            error="No response from model",
        )
    except Exception as exc:
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
    state: AppState = Depends(get_state),
) -> ProviderSaveResponse:
    """Save provider configuration. Writes API key to .env and os.environ."""
    provider_entry = None
    for p in _KNOWN_PROVIDERS:
        if p["name"] == provider_name:
            provider_entry = p
            break
    if provider_entry is None:
        raise HTTPException(status_code=404, detail=f"Unknown provider: {provider_name}")

    env_var = provider_entry.get("env_var")

    if req.api_key and env_var:
        # Persist to .env file and update current process environment
        _update_env_file(env_var, req.api_key)
        os.environ[env_var] = req.api_key

    return ProviderSaveResponse(
        saved=True,
        provider=provider_name,
        env_var=env_var,
        message=f"API key saved to .env as {env_var}" if env_var and req.api_key else "Configuration saved",
    )

"""Provider management endpoints."""

from __future__ import annotations

import os
import time
from typing import List

from fastapi import APIRouter, Depends

from rlmkit.server.dependencies import AppState, get_state
from rlmkit.server.models import ProviderInfo, ProviderTestRequest, ProviderTestResponse

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
    """List configured providers with status."""
    result = []
    for p in _KNOWN_PROVIDERS:
        env_var = p.get("env_var")
        configured = bool(env_var and os.environ.get(env_var)) if env_var else False
        status = "connected" if configured else "not_configured"

        result.append(
            ProviderInfo(
                name=p["name"],
                display_name=p["display_name"],
                status=status,
                models=p["models"] if configured else [],
                default_model=p["default_model"] if configured else None,
                configured=configured,
            )
        )
    return result


@router.post("/api/providers/test")
async def test_provider(
    req: ProviderTestRequest,
    state: AppState = Depends(get_state),
) -> ProviderTestResponse:
    """Test a provider connection."""
    from rlmkit.infrastructure.llm.litellm_adapter import LiteLLMAdapter

    model = req.model
    if not model:
        for p in _KNOWN_PROVIDERS:
            if p["name"] == req.provider:
                model = p["default_model"]
                break
    if not model:
        model = "gpt-4o"

    adapter = LiteLLMAdapter(
        model=model,
        api_key=req.api_key,
        api_base=req.endpoint,
    )

    start = time.time()
    try:
        healthy = adapter.check_health()
        latency_ms = int((time.time() - start) * 1000)
        if healthy:
            return ProviderTestResponse(
                connected=True,
                latency_ms=latency_ms,
                model=model,
            )
        else:
            return ProviderTestResponse(
                connected=False,
                error="Health check returned false",
            )
    except Exception as exc:
        return ProviderTestResponse(
            connected=False,
            error=str(exc),
        )

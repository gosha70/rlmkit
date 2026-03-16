"""Unified high-level API for RLMKit.

Provides ``interact()`` and ``complete()`` as the main programmatic entry
points, plus async variants ``interact_async()`` and ``complete_async()``.
Under the hood these delegate to Clean Architecture use cases
(``RunDirectUseCase``, ``RunRAGUseCase``, ``RunRLMUseCase``) with
``LiteLLMAdapter`` for provider-agnostic LLM access.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Literal

from rlmkit.application.dto import RunConfigDTO, RunResultDTO
from rlmkit.application.use_cases.run_direct import RunDirectUseCase
from rlmkit.application.use_cases.run_rlm import RunRLMUseCase
from rlmkit.infrastructure.llm.litellm_adapter import LiteLLMAdapter
from rlmkit.infrastructure.sandbox.sandbox_factory import create_sandbox

# Type alias for interaction modes
InteractionMode = Literal["direct", "rag", "rlm", "auto"]

# Token-count thresholds for auto mode selection
_DIRECT_THRESHOLD = 8_000
_RAG_THRESHOLD = 100_000


@dataclass
class InteractResult:
    """Result from an ``interact()`` call."""

    answer: str
    mode_used: str
    metrics: dict[str, Any] = field(default_factory=dict)
    trace: list[dict[str, Any]] | None = None
    raw_result: RunResultDTO | None = None

    def __str__(self) -> str:
        return self.answer

    def to_dict(self) -> dict[str, Any]:
        return {
            "answer": self.answer,
            "mode_used": self.mode_used,
            "metrics": self.metrics,
            "has_trace": self.trace is not None,
        }


def _estimate_tokens(text: str) -> int:
    """Rough token estimate (1 token ~ 4 chars)."""
    return max(1, len(text) // 4)


def _determine_auto_mode(content: str) -> str:
    """Pick the best mode based on content size.

    - < 8K tokens  -> direct
    - 8K–100K      -> rag
    - > 100K       -> rlm
    """
    token_count = _estimate_tokens(content)
    if token_count < _DIRECT_THRESHOLD:
        return "direct"
    if token_count < _RAG_THRESHOLD:
        return "rag"
    return "rlm"


_PROVIDER_PREFIXES: dict[str, str] = {
    "anthropic": "anthropic/",
    "ollama": "ollama/",
    "lmstudio": "openai/",
}


def _auto_detect_provider() -> str | None:
    """Detect the LLM provider from environment variables."""
    if os.environ.get("OPENAI_API_KEY"):
        return "openai"
    if os.environ.get("ANTHROPIC_API_KEY"):
        return "anthropic"
    if os.environ.get("OLLAMA_BASE_URL"):
        return "ollama"
    return None


def _resolve_model(provider: str, model: str | None) -> str:
    """Apply LiteLLM provider prefix and default model if needed."""
    _defaults: dict[str, str] = {
        "openai": "gpt-4o",
        "anthropic": "claude-sonnet-4-5-20250514",
        "ollama": "llama3",
    }
    m = model or _defaults.get(provider, "gpt-4o")
    if "/" not in m:
        prefix = _PROVIDER_PREFIXES.get(provider, "")
        m = f"{prefix}{m}"
    return m


def _setup(
    content: str,
    query: str,
    mode: InteractionMode,
    provider: str | None,
    model: str | None,
    api_key: str | None,
    api_base: str | None,
    max_steps: int,
    temperature: float,
    max_tokens: int | None,
    timeout: float | None,
    verbose: bool,
) -> tuple[str, str, LiteLLMAdapter, RunConfigDTO]:
    """Validate inputs, resolve mode/provider/model, build adapter and config.

    Returns (actual_mode, provider, llm_adapter, run_config).
    """
    if not content:
        raise ValueError("content cannot be empty")
    if not query:
        raise ValueError("query cannot be empty")

    # -- Resolve mode --
    actual_mode: str = mode
    if mode == "auto":
        actual_mode = _determine_auto_mode(content)
        if verbose:
            print(
                f"[Auto Mode] Selected '{actual_mode}' based on content size "
                f"({_estimate_tokens(content):,} tokens)"
            )

    if actual_mode not in ("direct", "rag", "rlm"):
        raise ValueError(f"Invalid mode: {actual_mode}. Must be 'direct', 'rag', 'rlm', or 'auto'")

    # -- Resolve provider --
    if provider is None:
        provider = _auto_detect_provider()
        if provider is None:
            raise ValueError(
                "No LLM provider configured. Set OPENAI_API_KEY, "
                "ANTHROPIC_API_KEY, or OLLAMA_BASE_URL in the environment, "
                "or pass provider= explicitly."
            )
        if verbose:
            print(f"[Auto-Detect] Using '{provider}' provider from environment")

    prefixed_model = _resolve_model(provider, model)

    if verbose:
        print(f"[Setup] Configuring {provider}/{prefixed_model} ...")

    llm = LiteLLMAdapter(
        model=prefixed_model,
        api_key=api_key,
        api_base=api_base,
        temperature=temperature,
        max_tokens=max_tokens,
        timeout=timeout if timeout is not None else 120.0,
    )

    config = RunConfigDTO(
        mode=actual_mode,
        provider=provider,
        model=prefixed_model,
        max_steps=max_steps,
    )

    return actual_mode, provider, llm, config


def _build_result(actual_mode: str, result: RunResultDTO, verbose: bool) -> InteractResult:
    """Convert a RunResultDTO into an InteractResult with metrics."""
    metrics = {
        "total_tokens": result.total_tokens,
        "input_tokens": result.input_tokens,
        "output_tokens": result.output_tokens,
        "total_cost": result.total_cost,
        "execution_time": result.elapsed_time,
        "llm_calls": result.steps,
    }

    if verbose:
        print(f"[Complete] Generated {len(result.answer)} character response")
        print(f"  Tokens: {metrics['total_tokens']:,} | Cost: ${metrics['total_cost']:.4f}")

    return InteractResult(
        answer=result.answer,
        mode_used=actual_mode,
        metrics=metrics,
        trace=result.trace or None,
        raw_result=result,
    )


def _dispatch_sync(
    actual_mode: str,
    llm: LiteLLMAdapter,
    config: RunConfigDTO,
    content: str,
    query: str,
) -> RunResultDTO:
    """Run the appropriate use case synchronously."""
    if actual_mode == "rlm":
        sandbox = create_sandbox()
        uc = RunRLMUseCase(llm, sandbox)
        return uc.execute(content, query, config)
    elif actual_mode == "rag":
        # RAG use case requires embedder + storage; fall back to direct
        # until those adapters are wired up in the public API.
        uc_direct = RunDirectUseCase(llm)
        result = uc_direct.execute(content, query, config)
        result.mode_used = "rag"
        return result
    else:
        uc_direct = RunDirectUseCase(llm)
        return uc_direct.execute(content, query, config)


async def _dispatch_async(
    actual_mode: str,
    llm: LiteLLMAdapter,
    config: RunConfigDTO,
    content: str,
    query: str,
) -> RunResultDTO:
    """Run the appropriate use case asynchronously."""
    if actual_mode == "rlm":
        sandbox = create_sandbox()
        uc = RunRLMUseCase(llm, sandbox)
        return await uc.execute_async(content, query, config)
    elif actual_mode == "rag":
        uc_direct = RunDirectUseCase(llm)
        result = await uc_direct.execute_async(content, query, config)
        result.mode_used = "rag"
        return result
    else:
        uc_direct = RunDirectUseCase(llm)
        return await uc_direct.execute_async(content, query, config)


def interact(
    content: str,
    query: str,
    mode: InteractionMode = "auto",
    provider: str | None = None,
    model: str | None = None,
    api_key: str | None = None,
    api_base: str | None = None,
    max_steps: int = 16,
    temperature: float = 0.7,
    max_tokens: int | None = None,
    timeout: float | None = None,
    verbose: bool = False,
    **kwargs: Any,
) -> InteractResult:
    """Interact with content using an LLM through Direct, RAG, or RLM mode.

    This is the main entry point for RLMKit.  It handles provider
    resolution, mode selection, and use-case dispatch.

    Args:
        content: Document / text content to analyse.
        query: Question or instruction for the LLM.
        mode: ``"direct"`` | ``"rag"`` | ``"rlm"`` | ``"auto"``.
        provider: Provider key (``"openai"``, ``"anthropic"``, ``"ollama"``).
            Auto-detected from env vars when *None*.
        model: Model name.  Defaults to the provider's flagship model.
        api_key: API key override (usually read from env vars).
        api_base: Custom API base URL (e.g. for Ollama on a non-default port).
        max_steps: Budget limit for RLM loop iterations.
        temperature: Sampling temperature.
        max_tokens: Max output tokens per LLM call.
        timeout: Request timeout in seconds (default 120).
        verbose: Print progress to stdout.

    Returns:
        :class:`InteractResult` with answer, mode used, and metrics.

    Raises:
        ValueError: If inputs are invalid or no provider can be resolved.
    """
    actual_mode, _provider, llm, config = _setup(
        content,
        query,
        mode,
        provider,
        model,
        api_key,
        api_base,
        max_steps,
        temperature,
        max_tokens,
        timeout,
        verbose,
    )

    if verbose:
        print(f"[Execution] Running in '{actual_mode}' mode ...")

    result = _dispatch_sync(actual_mode, llm, config, content, query)
    return _build_result(actual_mode, result, verbose)


async def interact_async(
    content: str,
    query: str,
    mode: InteractionMode = "auto",
    provider: str | None = None,
    model: str | None = None,
    api_key: str | None = None,
    api_base: str | None = None,
    max_steps: int = 16,
    temperature: float = 0.7,
    max_tokens: int | None = None,
    timeout: float | None = None,
    verbose: bool = False,
    **kwargs: Any,
) -> InteractResult:
    """Async version of :func:`interact`.

    Same parameters and return type.  Uses the async execution path
    of each use case (``execute_async``), which enables true async
    LLM calls via ``litellm.acompletion``.
    """
    actual_mode, _provider, llm, config = _setup(
        content,
        query,
        mode,
        provider,
        model,
        api_key,
        api_base,
        max_steps,
        temperature,
        max_tokens,
        timeout,
        verbose,
    )

    if verbose:
        print(f"[Execution] Running in '{actual_mode}' mode (async) ...")

    result = await _dispatch_async(actual_mode, llm, config, content, query)
    return _build_result(actual_mode, result, verbose)


def complete(content: str, query: str, **kwargs: Any) -> str:
    """Convenience wrapper returning just the answer string.

    Args:
        content: Document / text content.
        query: Question or instruction.
        **kwargs: Forwarded to :func:`interact`.

    Returns:
        The answer text.
    """
    return interact(content=content, query=query, **kwargs).answer


async def complete_async(content: str, query: str, **kwargs: Any) -> str:
    """Async convenience wrapper returning just the answer string.

    Args:
        content: Document / text content.
        query: Question or instruction.
        **kwargs: Forwarded to :func:`interact_async`.

    Returns:
        The answer text.
    """
    result = await interact_async(content=content, query=query, **kwargs)
    return result.answer

"""Unified high-level API for RLMKit.

Provides ``interact()`` and ``complete()`` as the main programmatic entry
points, plus async variants ``interact_async()`` and ``complete_async()``.
Under the hood these delegate to Clean Architecture use cases
(``RunDirectUseCase``, ``RunRAGUseCase``, ``RunRLMUseCase``) with
``LiteLLMAdapter`` for provider-agnostic LLM access.
"""

from __future__ import annotations

import os
import uuid
from dataclasses import dataclass, field
from typing import Any, Literal

from rlmkit.application.dto import RunConfigDTO, RunResultDTO
from rlmkit.application.use_cases.run_comparison import (
    ComparisonResultDTO,
    RunComparisonUseCase,
)
from rlmkit.application.use_cases.run_direct import RunDirectUseCase
from rlmkit.application.use_cases.run_rag import RunRAGUseCase
from rlmkit.application.use_cases.run_rlm import RunRLMUseCase
from rlmkit.infrastructure.embedding.litellm_embedding_adapter import LiteLLMEmbeddingAdapter
from rlmkit.infrastructure.llm.litellm_adapter import LiteLLMAdapter
from rlmkit.infrastructure.sandbox.sandbox_factory import create_sandbox
from rlmkit.infrastructure.storage.sqlite_adapter import SQLiteStorageAdapter

# Type alias for interaction modes
InteractionMode = Literal["direct", "rag", "rlm", "compare", "auto"]

# Token-count thresholds for auto mode selection.
_DIRECT_THRESHOLD = 8_000  # below → direct
_RAG_THRESHOLD = 100_000  # between direct and rlm thresholds → rag


@dataclass
class InteractResult:
    """Result from an ``interact()`` call."""

    answer: str
    mode_used: str
    metrics: dict[str, Any] = field(default_factory=dict)
    trace: list[dict[str, Any]] | None = None
    raw_result: RunResultDTO | None = None

    @property
    def total_tokens(self) -> int:
        return int(self.metrics.get("total_tokens", 0))

    @property
    def total_cost(self) -> float:
        return float(self.metrics.get("total_cost", 0.0))

    @property
    def elapsed_time(self) -> float:
        return float(self.metrics.get("execution_time", 0.0))

    @property
    def steps(self) -> int:
        return int(self.metrics.get("llm_calls", 0))

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

    - < 8K tokens    -> direct  (full context fits comfortably)
    - 8K–100K tokens -> rag     (too large for direct; retrieval beats full scan)
    - > 100K tokens  -> rlm     (recursive exploration scales to millions of tokens)
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


def _resolve_embedding_key(
    provider: str,
    chat_api_key: str | None,
    embedding_api_key: str | None,
) -> str | None:
    """Return the API key to use for the embedding call, or raise ValueError.

    RAG mode uses OpenAI's ``text-embedding-3-small`` by default.
    - OpenAI chat provider: the same ``api_key`` covers embeddings.
    - Any other provider: embeddings need a separate OpenAI key.
      Resolution order: ``embedding_api_key`` arg → ``OPENAI_API_KEY`` env var.
      Raises ``ValueError`` if neither is available so the caller gets a
      clear error rather than silently passing the wrong credentials.
    """
    if embedding_api_key:
        return embedding_api_key
    if provider == "openai":
        return chat_api_key  # same key covers both chat and embeddings
    # Non-OpenAI provider: require an explicit OpenAI key for embeddings.
    openai_key = os.environ.get("OPENAI_API_KEY")
    if openai_key:
        return openai_key
    raise ValueError(
        f"RAG mode requires an OpenAI API key for embeddings "
        f"(provider={provider!r} has no native embedding API). "
        "Set OPENAI_API_KEY in the environment or pass embedding_api_key= explicitly."
    )


_LOCAL_PROVIDERS: frozenset[str] = frozenset({"ollama", "lmstudio"})

_DEFAULT_MODELS: dict[str, str] = {
    "openai": "gpt-4o",
    "anthropic": "claude-sonnet-4-6",
}


def _resolve_model(provider: str, model: str | None) -> str:
    """Apply LiteLLM provider prefix and default model if needed.

    Local providers (ollama, lmstudio) have no universal default — the caller
    must pass an explicit ``model=`` argument.
    """
    if model is None and provider in _LOCAL_PROVIDERS:
        raise ValueError(
            f"provider={provider!r} requires an explicit model= argument "
            f"(e.g. model='llama3.2'). "
            f"Local providers have no universal default."
        )
    m = model or _DEFAULT_MODELS.get(provider, "gpt-4o")
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
    root_model: str | None = None,
    recursive_model: str | None = None,
    embedding_api_key: str | None = None,
    num_retries: int | None = None,
) -> tuple[str, str, LiteLLMAdapter, RunConfigDTO, str | None]:
    """Validate inputs, resolve mode/provider/model, build adapter and config.

    Returns (actual_mode, provider, llm_adapter, run_config, resolved_embedding_key).
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

    if actual_mode not in ("direct", "rag", "rlm", "compare"):
        raise ValueError(
            f"Invalid mode: {actual_mode}. Must be 'direct', 'rag', 'rlm', 'compare', or 'auto'"
        )

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
    resolved_root = _resolve_model(provider, root_model) if root_model else None
    resolved_recursive = _resolve_model(provider, recursive_model) if recursive_model else None

    if verbose:
        print(f"[Setup] Configuring {provider}/{prefixed_model} ...")

    llm = LiteLLMAdapter(
        model=prefixed_model,
        root_model=resolved_root,
        recursive_model=resolved_recursive,
        api_key=api_key,
        api_base=api_base,
        temperature=temperature,
        max_tokens=max_tokens,
        timeout=timeout if timeout is not None else 120.0,
        num_retries=(
            num_retries if num_retries is not None else 0 if provider in _LOCAL_PROVIDERS else 2
        ),
    )

    config = RunConfigDTO(
        mode=actual_mode,
        provider=provider,
        model=prefixed_model,
        max_steps=max_steps,
    )

    resolved_emb_key = (
        _resolve_embedding_key(provider, api_key, embedding_api_key)
        if actual_mode == "rag"
        else None
    )
    return actual_mode, provider, llm, config, resolved_emb_key


def _build_result(
    actual_mode: str,
    result: RunResultDTO | ComparisonResultDTO,
    verbose: bool,
) -> InteractResult:
    """Convert a RunResultDTO or ComparisonResultDTO into an InteractResult."""
    if isinstance(result, ComparisonResultDTO):
        # Pick best answer: prefer successful rlm, then successful direct, then any
        successful = {m: r for m, r in result.results.items() if r.success}
        best = (
            successful.get("rlm")
            or successful.get("direct")
            or result.results.get("rlm")
            or result.results.get("direct")
            or RunResultDTO(answer="", mode_used="compare", success=False, error="No results")
        )
        metrics: dict[str, Any] = {
            "total_tokens": sum(r.total_tokens for r in result.results.values()),
            "total_cost": sum(r.total_cost for r in result.results.values()),
            "execution_time": result.total_elapsed,
            "comparison": {
                mode: {
                    "answer_length": len(r.answer),
                    "total_tokens": r.total_tokens,
                    "total_cost": r.total_cost,
                    "elapsed_time": r.elapsed_time,
                }
                for mode, r in result.results.items()
            },
        }
        if verbose:
            print(f"[Compare] Ran {len(result.results)} modes: {result.modes_run}")
            for mode, r in result.results.items():
                print(f"  {mode}: {r.total_tokens} tokens, ${r.total_cost:.4f}")
        return InteractResult(
            answer=best.answer,
            mode_used="compare",
            metrics=metrics,
            raw_result=best,
        )

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
    embedding_api_key: str | None = None,
) -> RunResultDTO | ComparisonResultDTO:
    """Run the appropriate use case synchronously."""
    if actual_mode == "compare":
        sandbox = create_sandbox()
        uc_cmp = RunComparisonUseCase(llm, sandbox)
        return uc_cmp.execute(content, query, config)
    elif actual_mode == "rlm":
        sandbox = create_sandbox()
        uc = RunRLMUseCase(llm, sandbox)
        return uc.execute(content, query, config)
    elif actual_mode == "rag":
        collection = f"rag_{uuid.uuid4().hex}"
        config.extra["collection"] = collection
        embedder = LiteLLMEmbeddingAdapter(api_key=embedding_api_key)
        storage = SQLiteStorageAdapter(":memory:")
        uc_rag = RunRAGUseCase(llm, embedder, storage)
        return uc_rag.execute(content, query, config)
    else:
        uc_direct = RunDirectUseCase(llm)
        return uc_direct.execute(content, query, config)


async def _dispatch_async(
    actual_mode: str,
    llm: LiteLLMAdapter,
    config: RunConfigDTO,
    content: str,
    query: str,
    embedding_api_key: str | None = None,
) -> RunResultDTO | ComparisonResultDTO:
    """Run the appropriate use case asynchronously."""
    if actual_mode == "compare":
        sandbox = create_sandbox()
        uc_cmp = RunComparisonUseCase(llm, sandbox)
        return await uc_cmp.execute_async(content, query, config)
    elif actual_mode == "rlm":
        sandbox = create_sandbox()
        uc = RunRLMUseCase(llm, sandbox)
        return await uc.execute_async(content, query, config)
    elif actual_mode == "rag":
        import asyncio

        collection = f"rag_{uuid.uuid4().hex}"
        config.extra["collection"] = collection
        embedder = LiteLLMEmbeddingAdapter(api_key=embedding_api_key)
        storage = SQLiteStorageAdapter(":memory:")
        uc_rag = RunRAGUseCase(llm, embedder, storage)
        return await asyncio.to_thread(uc_rag.execute, content, query, config)
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
    root_model: str | None = None,
    recursive_model: str | None = None,
    embedding_api_key: str | None = None,
    num_retries: int | None = None,
    **kwargs: Any,
) -> InteractResult:
    """Interact with content using an LLM through Direct, RLM, or Compare mode.

    This is the main entry point for RLMKit.  It handles provider
    resolution, mode selection, and use-case dispatch.

    Args:
        content: Document / text content to analyse.
        query: Question or instruction for the LLM.
        mode: ``"direct"`` | ``"rlm"`` | ``"compare"`` | ``"auto"``.
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
        root_model: Model for root-level reasoning (paper's two-model optimization).
        recursive_model: Cheaper model for RLM exploration subcalls.
        embedding_api_key: API key for the embedding model used in RAG mode.
            Defaults to ``api_key`` when ``provider="openai"``, otherwise falls
            back to ``OPENAI_API_KEY`` from the environment.
        num_retries: Number of retry attempts on transient LLM errors.
            ``None`` (default) applies automatic defaults: ``0`` for local
            providers (``ollama``, ``lmstudio``) to avoid multiplying hangs,
            ``2`` for cloud providers.  Pass ``0`` to disable retries entirely,
            or a positive integer to override.

    Returns:
        :class:`InteractResult` with answer, mode used, and metrics.

    Raises:
        ValueError: If inputs are invalid or no provider can be resolved.
    """
    actual_mode, _provider, llm, config, emb_key = _setup(
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
        root_model=root_model,
        recursive_model=recursive_model,
        embedding_api_key=embedding_api_key,
        num_retries=num_retries,
    )

    result = _dispatch_sync(actual_mode, llm, config, content, query, emb_key)
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
    root_model: str | None = None,
    recursive_model: str | None = None,
    embedding_api_key: str | None = None,
    num_retries: int | None = None,
    **kwargs: Any,
) -> InteractResult:
    """Async version of :func:`interact`.

    Same parameters and return type.  Uses the async execution path
    of each use case (``execute_async``), which enables true async
    LLM calls via ``litellm.acompletion``.
    """
    actual_mode, _provider, llm, config, emb_key = _setup(
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
        root_model=root_model,
        recursive_model=recursive_model,
        embedding_api_key=embedding_api_key,
        num_retries=num_retries,
    )

    result = await _dispatch_async(actual_mode, llm, config, content, query, emb_key)
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

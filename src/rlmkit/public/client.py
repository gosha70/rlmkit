"""RLMKit client: high-level entry point using Clean Architecture internals.

This client provides the same functionality as the existing ``rlmkit.interact()``
and ``rlmkit.complete()`` but is backed by the new layered architecture.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from rlmkit.application.dto import RunConfigDTO, RunResultDTO
from rlmkit.application.use_cases.run_direct import RunDirectUseCase
from rlmkit.application.use_cases.run_rlm import RunRLMUseCase
from rlmkit.application.use_cases.run_comparison import (
    ComparisonResultDTO,
    RunComparisonUseCase,
)
from rlmkit.infrastructure.llm.mock_adapter import MockLLMAdapter
from rlmkit.infrastructure.sandbox.sandbox_factory import create_sandbox
from .types import PublicRunResult
from .errors import ConfigError, RLMKitError


class RLMKitClient:
    """High-level client for RLMKit using Clean Architecture.

    This is the recommended entry point for programmatic usage.
    It wires up the correct infrastructure adapters and delegates
    to the appropriate use case.

    By default, ``provider="litellm"`` routes all calls through LiteLLM,
    which supports 100+ providers via model-name prefixes (e.g.
    ``"gpt-4o"``, ``"claude-3-opus-20240229"``, ``"ollama/llama3"``).
    Set ``provider="mock"`` for testing.

    For the paper's two-model cost optimization, pass both
    ``root_model`` and ``recursive_model``.

    Args:
        provider: LLM provider name. ``"litellm"`` (default) uses the
            unified LiteLLM backend. Legacy values ``"openai"``,
            ``"anthropic"``, ``"ollama"`` use the old direct adapters.
        model: Model identifier (used when root/recursive not specified).
        root_model: Model for root-level reasoning (overrides *model*).
        recursive_model: Cheaper model for exploration subcalls.
        api_key: Optional API key override.
        api_base: Optional custom API base URL.
        sandbox_type: Sandbox type ("local", "docker").
        safe_mode: Enable sandbox restrictions.
        max_steps: Default maximum execution steps.
        max_recursion_depth: Default maximum recursion depth.
    """

    def __init__(
        self,
        provider: str = "mock",
        model: Optional[str] = None,
        root_model: Optional[str] = None,
        recursive_model: Optional[str] = None,
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
        sandbox_type: str = "local",
        safe_mode: bool = False,
        max_steps: int = 16,
        max_recursion_depth: int = 5,
    ) -> None:
        self._provider = provider
        self._model = model
        self._root_model = root_model
        self._recursive_model = recursive_model
        self._api_key = api_key
        self._api_base = api_base
        self._max_steps = max_steps
        self._max_recursion_depth = max_recursion_depth

        # Create sandbox adapter
        self._sandbox = create_sandbox(
            sandbox_type=sandbox_type,
            safe_mode=safe_mode,
        )

        # Create LLM adapter
        self._llm = self._create_llm_adapter(
            provider, model, api_key,
            root_model=root_model,
            recursive_model=recursive_model,
            api_base=api_base,
        )

    def interact(
        self,
        content: str,
        query: str,
        mode: str = "auto",
        verbose: bool = False,
        **kwargs: Any,
    ) -> PublicRunResult:
        """Run a query against content using the specified mode.

        This mirrors the existing ``rlmkit.interact()`` API but uses
        Clean Architecture internals.

        Args:
            content: Document text to analyze.
            query: User question.
            mode: Execution mode ("direct", "rlm", "auto", "compare").
            verbose: Print progress output.
            **kwargs: Additional configuration.

        Returns:
            PublicRunResult with the answer and metrics.
        """
        if not content:
            raise ValueError("content cannot be empty")
        if not query:
            raise ValueError("query cannot be empty")

        # Resolve auto mode
        actual_mode = mode
        if mode == "auto":
            actual_mode = self._determine_auto_mode(content)
            if verbose:
                token_est = max(1, len(content) // 4)
                print(
                    f"[Auto Mode] Selected '{actual_mode}' based on "
                    f"content size ({token_est:,} tokens)"
                )

        config = RunConfigDTO(
            mode=actual_mode,
            provider=self._provider,
            model=self._model,
            api_key=self._api_key,
            max_steps=kwargs.get("max_steps", self._max_steps),
            max_recursion_depth=kwargs.get(
                "max_recursion_depth", self._max_recursion_depth
            ),
            verbose=verbose,
            extra=kwargs,
        )

        if actual_mode == "direct":
            uc = RunDirectUseCase(self._llm)
            result = uc.execute(content, query, config)
        elif actual_mode == "rlm":
            uc_rlm = RunRLMUseCase(self._llm, self._sandbox)
            result = uc_rlm.execute(content, query, config)
        elif actual_mode == "compare":
            uc_cmp = RunComparisonUseCase(self._llm, self._sandbox)
            cmp_result = uc_cmp.execute(content, query, config)
            # Return the best result (prefer rlm if available)
            result = (
                cmp_result.results.get("rlm")
                or cmp_result.results.get("direct")
                or RunResultDTO(answer="", mode_used="compare", success=False, error="No results")
            )
        else:
            raise ValueError(
                f"Unsupported mode: {actual_mode!r}. "
                "Use 'direct', 'rlm', 'auto', or 'compare'."
            )

        return self._to_public_result(result)

    def complete(self, content: str, query: str, **kwargs: Any) -> str:
        """Simple completion returning just the answer string.

        Args:
            content: Document text.
            query: User question.
            **kwargs: Passed to interact().

        Returns:
            Answer string.
        """
        result = self.interact(content, query, **kwargs)
        return result.answer

    # -- Private helpers --

    @staticmethod
    def _determine_auto_mode(content: str) -> str:
        """Pick mode based on content size."""
        token_count = max(1, len(content) // 4)
        if token_count < 8000:
            return "direct"
        elif token_count < 100000:
            return "rlm"
        else:
            return "rlm"

    @staticmethod
    def _create_llm_adapter(
        provider: str,
        model: Optional[str],
        api_key: Optional[str],
        root_model: Optional[str] = None,
        recursive_model: Optional[str] = None,
        api_base: Optional[str] = None,
    ) -> object:
        """Create the appropriate LLM adapter for the given provider."""
        if provider == "mock":
            return MockLLMAdapter(["FINAL: Mock response"])

        # LiteLLM: the primary unified provider (100+ models)
        if provider == "litellm":
            from rlmkit.infrastructure.llm.litellm_adapter import LiteLLMAdapter

            return LiteLLMAdapter(
                model=model or "gpt-4o",
                root_model=root_model,
                recursive_model=recursive_model,
                api_key=api_key,
                api_base=api_base,
            )

        # Legacy direct adapters (opt-in fallbacks)
        if provider == "openai":
            try:
                from rlmkit.llm.openai_client import OpenAIClient
                from rlmkit.infrastructure.llm.openai_adapter import OpenAIAdapter

                client = OpenAIClient(model=model or "gpt-4o", api_key=api_key)
                return OpenAIAdapter(client)
            except ImportError as exc:
                raise ConfigError(f"OpenAI not available: {exc}")

        if provider == "anthropic":
            try:
                from rlmkit.llm.anthropic_client import ClaudeClient
                from rlmkit.infrastructure.llm.anthropic_adapter import AnthropicAdapter

                client = ClaudeClient(model=model or "claude-3-sonnet-20240229", api_key=api_key)
                return AnthropicAdapter(client)
            except ImportError as exc:
                raise ConfigError(f"Anthropic not available: {exc}")

        if provider == "ollama":
            try:
                from rlmkit.llm.ollama_client import OllamaClient
                from rlmkit.infrastructure.llm.ollama_adapter import OllamaAdapter

                client = OllamaClient(model=model or "llama2")
                return OllamaAdapter(client)
            except ImportError as exc:
                raise ConfigError(f"Ollama not available: {exc}")

        raise ConfigError(
            f"Unknown provider: {provider!r}. "
            "Supported: 'litellm', 'openai', 'anthropic', 'ollama', 'mock'."
        )

    @staticmethod
    def _to_public_result(dto: RunResultDTO) -> PublicRunResult:
        """Convert an internal RunResultDTO to a public result."""
        return PublicRunResult(
            answer=dto.answer,
            mode_used=dto.mode_used,
            success=dto.success,
            error=dto.error,
            total_tokens=dto.total_tokens,
            input_tokens=dto.input_tokens,
            output_tokens=dto.output_tokens,
            total_cost=dto.total_cost,
            elapsed_time=dto.elapsed_time,
            steps=dto.steps,
            trace=dto.trace,
            metadata=dto.metadata,
        )

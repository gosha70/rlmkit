"""LiteLLM adapter: unified provider supporting 100+ models via LiteLLM.

This is the primary LLM adapter for RLMKit, providing access to OpenAI,
Anthropic, Google, Cohere, Azure, Bedrock, Ollama, and many more providers
through a single interface.

Supports the paper's two-model optimization: a powerful root_model for
primary reasoning and a cheaper recursive_model for exploration subcalls.
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator, Iterator
from typing import Any

from rlmkit.application.dto import LLMResponseDTO

logger = logging.getLogger(__name__)

_CONNECTION_KEYWORDS = (
    "connection refused",
    "connection error",
    "cannot connect",
    "unreachable",
    "network is unreachable",
    "connect call failed",
    "failed to establish",
    "name or service not known",
    "no route to host",
    "connectionerror",
    "connecterror",
    "apiconnectionerror",
)


def _is_connection_error(exc: BaseException) -> bool:
    """Return True when *exc* looks like a network-level connection failure."""
    msg = str(exc).lower()
    return any(kw in msg for kw in _CONNECTION_KEYWORDS)


def _connection_error_message(api_base: str | None, exc: BaseException) -> str:
    """Build a human-readable message for connection failures."""
    location = f" at '{api_base}'" if api_base else ""
    hint = (
        " Make sure the server is running and reachable."
        if api_base
        else " Check your provider credentials and network connectivity."
    )
    return f"Cannot connect to the LLM server{location}.{hint} (Detail: {exc})"


class LiteLLMAdapter:
    """Unified LLM adapter using LiteLLM for provider-agnostic access.

    This adapter implements :class:`LLMPort` and supports:
    - 100+ LLM providers through a single interface
    - Two-model configuration (root + recursive) per the RLM paper
    - Token counting and cost estimation for all supported models
    - Streaming completions
    - Provider health checking

    Args:
        model: Default model identifier (e.g. "gpt-4o", "claude-3-opus-20240229",
            "ollama/llama3"). LiteLLM uses provider prefixes for routing.
        root_model: Model for root-level reasoning (overrides model for root calls).
        recursive_model: Cheaper model for recursive exploration subcalls.
        api_key: API key override (usually set via environment variables).
        api_base: Custom API base URL.
        temperature: Sampling temperature (0.0-1.0).
        max_tokens: Maximum tokens to generate per call.
        timeout: Request timeout in seconds.
        extra_params: Additional parameters forwarded to litellm.completion().
    """

    def __init__(
        self,
        model: str = "gpt-4o",
        root_model: str | None = None,
        recursive_model: str | None = None,
        api_key: str | None = None,
        api_base: str | None = None,
        temperature: float = 0.7,
        max_tokens: int | None = None,
        timeout: float = 120.0,
        num_retries: int = 2,
        extra_params: dict[str, Any] | None = None,
    ) -> None:
        self._model = model
        self._root_model = root_model or model
        self._recursive_model = recursive_model or model
        self._api_key = api_key
        self._api_base = api_base
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._timeout = timeout
        self._num_retries = num_retries
        self._extra_params = extra_params or {}

        # Track which model to use for the next call (can be toggled)
        self._active_model = self._root_model

    # -- LLMPort protocol methods --

    def complete(self, messages: list[dict[str, str]]) -> LLMResponseDTO:
        """Generate a completion using LiteLLM.

        Args:
            messages: Chat messages with 'role' and 'content' keys.

        Returns:
            LLMResponseDTO with generated text and token counts.

        Raises:
            RuntimeError: If the LiteLLM call fails.
        """
        import litellm

        params = self._build_params(messages)

        try:
            response = litellm.completion(**params)
        except Exception as exc:
            if _is_connection_error(exc):
                raise RuntimeError(_connection_error_message(self._api_base, exc)) from exc
            raise RuntimeError(f"LiteLLM completion failed: {exc}") from exc

        choice = response.choices[0]
        usage = response.usage
        content = self._extract_content(choice.message)

        return LLMResponseDTO(
            content=content,
            model=response.model or self._active_model,
            input_tokens=usage.prompt_tokens if usage else 0,
            output_tokens=usage.completion_tokens if usage else 0,
            finish_reason=choice.finish_reason,
        )

    def complete_stream(self, messages: list[dict[str, str]]) -> Iterator[str]:
        """Generate a streaming completion, yielding text chunks.

        Args:
            messages: Chat messages.

        Yields:
            Text chunks as they are produced by the LLM.
        """
        import litellm

        params = self._build_params(messages)
        params["stream"] = True

        try:
            response = litellm.completion(**params)
            for chunk in response:
                delta = chunk.choices[0].delta
                if delta and delta.content:
                    yield delta.content
        except Exception as exc:
            if _is_connection_error(exc):
                raise RuntimeError(_connection_error_message(self._api_base, exc)) from exc
            raise RuntimeError(f"LiteLLM streaming failed: {exc}") from exc

    def count_tokens(self, text: str) -> int:
        """Count tokens using LiteLLM's model-aware tokenizer.

        Args:
            text: Text to tokenize.

        Returns:
            Token count.
        """
        import litellm

        try:
            return litellm.token_counter(model=self._active_model, text=text)
        except Exception:
            # Fallback to heuristic if tokenizer unavailable
            return max(1, len(text) // 4)

    def get_pricing(self) -> dict[str, float]:
        """Return pricing info for the active model from LiteLLM's cost DB.

        Returns:
            Dictionary with 'input_cost_per_1m' and 'output_cost_per_1m'.
        """
        import litellm

        # Try with the full prefixed name first, then stripped base name
        models_to_try = [self._active_model]
        if "/" in self._active_model:
            models_to_try.append(self._active_model.split("/", 1)[1])

        for model_name in models_to_try:
            try:
                info = litellm.get_model_info(model=model_name)
                input_per_token = info.get("input_cost_per_token") or 0.0
                output_per_token = info.get("output_cost_per_token") or 0.0
                return {
                    "input_cost_per_1m": float(input_per_token) * 1_000_000,
                    "output_cost_per_1m": float(output_per_token) * 1_000_000,
                }
            except Exception:
                continue
        return {"input_cost_per_1m": 0.0, "output_cost_per_1m": 0.0}

    # -- Async LLMPort methods --

    async def complete_async(self, messages: list[dict[str, str]]) -> LLMResponseDTO:
        """Async completion using ``litellm.acompletion``.

        Args:
            messages: Chat messages with 'role' and 'content' keys.

        Returns:
            LLMResponseDTO with generated text and token counts.

        Raises:
            RuntimeError: If the LiteLLM call fails.
        """
        import litellm

        params = self._build_params(messages)

        try:
            response = await litellm.acompletion(**params)
        except Exception as exc:
            if _is_connection_error(exc):
                raise RuntimeError(_connection_error_message(self._api_base, exc)) from exc
            raise RuntimeError(f"LiteLLM async completion failed: {exc}") from exc

        choice = response.choices[0]
        usage = response.usage

        return LLMResponseDTO(
            content=self._extract_content(choice.message),
            model=response.model or self._active_model,
            input_tokens=usage.prompt_tokens if usage else 0,
            output_tokens=usage.completion_tokens if usage else 0,
            finish_reason=choice.finish_reason,
        )

    async def complete_stream_async(self, messages: list[dict[str, str]]) -> AsyncIterator[str]:
        """Async streaming completion, yielding text chunks.

        Args:
            messages: Chat messages.

        Yields:
            Text chunks as they are produced by the LLM.
        """
        import litellm

        params = self._build_params(messages)
        params["stream"] = True

        try:
            response = await litellm.acompletion(**params)
            async for chunk in response:
                delta = chunk.choices[0].delta
                if delta and delta.content:
                    yield delta.content
        except Exception as exc:
            if _is_connection_error(exc):
                raise RuntimeError(_connection_error_message(self._api_base, exc)) from exc
            raise RuntimeError(f"LiteLLM async streaming failed: {exc}") from exc

    # -- Two-model support --

    def use_root_model(self) -> None:
        """Switch to the root model for subsequent calls."""
        self._active_model = self._root_model

    def use_recursive_model(self) -> None:
        """Switch to the recursive model for subsequent calls."""
        self._active_model = self._recursive_model

    @property
    def active_model(self) -> str:
        """Currently active model identifier."""
        return self._active_model

    @property
    def context_length_chars(self) -> int | None:
        """Approximate context window size in characters for the active model.

        Uses ``litellm.get_model_info`` to look up ``max_input_tokens``,
        then converts tokens → chars with a 4-chars-per-token heuristic.
        Returns ``None`` when the lookup fails (unknown/local models).
        """
        import litellm

        models_to_try = [self._active_model]
        if "/" in self._active_model:
            models_to_try.append(self._active_model.split("/", 1)[1])

        for model_name in models_to_try:
            try:
                info = litellm.get_model_info(model=model_name)
                max_input = info.get("max_input_tokens") or info.get("max_tokens")
                if max_input and isinstance(max_input, int) and max_input > 0:
                    return max_input * 4  # ~4 chars per token heuristic
            except Exception:
                continue
        return None

    @property
    def root_model(self) -> str:
        """Root model identifier."""
        return self._root_model

    @property
    def recursive_model(self) -> str:
        """Recursive model identifier."""
        return self._recursive_model

    @property
    def is_two_model(self) -> bool:
        """Whether root and recursive models are different."""
        return self._root_model != self._recursive_model

    # -- Health check --

    def check_health(self) -> bool:
        """Test connectivity by making a minimal completion call.

        Returns:
            True if the provider responds successfully, False otherwise.
        """
        import litellm

        try:
            response = litellm.completion(
                model=self._active_model,
                messages=[{"role": "user", "content": "ping"}],
                max_tokens=5,
                timeout=10,
                api_key=self._api_key,
                api_base=self._api_base,
            )
            return bool(response.choices)
        except Exception as exc:
            logger.warning("Health check failed for %s: %s", self._active_model, exc)
            return False

    def get_completion_cost(self, input_tokens: int, output_tokens: int) -> float:
        """Calculate the cost for a completion using LiteLLM.

        Args:
            input_tokens: Number of input tokens.
            output_tokens: Number of output tokens.

        Returns:
            Cost in USD.
        """
        import litellm

        try:
            prompt_cost, completion_cost = litellm.cost_per_token(
                model=self._active_model,
                prompt_tokens=input_tokens,
                completion_tokens=output_tokens,
            )
            return prompt_cost + completion_cost
        except Exception as exc:
            logger.debug("Cost lookup failed for model=%s: %s", self._active_model, exc)
            return 0.0

    # -- Private helpers --

    def _extract_content(self, message: Any) -> str:
        """Extract text content from an LLM response message.

        Some reasoning/thinking models (DeepSeek-R1, Phi-4-reasoning, Ollama
        thinking models) place all output tokens in a side channel
        (``reasoning_content`` or ``thinking``) while leaving ``content``
        empty.  Fall back through known field names so callers always receive
        a non-empty string when the model did produce output.
        """
        content = getattr(message, "content", None) or ""
        if content:
            return content
        from rlmkit.prompts import get_rlm_message

        fallback_fields = [
            f.strip() for f in get_rlm_message("reasoning_content_fields").split(",") if f.strip()
        ]
        for field in fallback_fields:
            val = getattr(message, field, None)
            if val:
                logger.debug(
                    "model=%s: content empty, using %s (%d chars)",
                    self._active_model,
                    field,
                    len(str(val)),
                )
                return str(val)
        return ""

    def _build_params(self, messages: list[dict[str, str]]) -> dict[str, Any]:
        """Build the parameter dict for litellm.completion().

        Args:
            messages: Chat messages.

        Returns:
            Keyword arguments for litellm.completion().
        """
        params: dict[str, Any] = {
            "model": self._active_model,
            "messages": messages,
            "temperature": self._temperature,
            "timeout": self._timeout,
            "num_retries": self._num_retries,
        }

        if self._max_tokens is not None:
            params["max_tokens"] = self._max_tokens

        if self._api_key is not None:
            params["api_key"] = self._api_key

        if self._api_base is not None:
            params["api_base"] = self._api_base

        params.update(self._extra_params)
        return params

    def __repr__(self) -> str:
        if self.is_two_model:
            return f"LiteLLMAdapter(root={self._root_model!r}, recursive={self._recursive_model!r})"
        return f"LiteLLMAdapter(model={self._model!r})"

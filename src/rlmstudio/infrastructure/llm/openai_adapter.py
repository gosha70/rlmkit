"""OpenAI adapter: wraps the existing OpenAIClient to implement LLMPort."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Iterator
from typing import Any

from rlmstudio.application.dto import LLMResponseDTO, StreamChunk


class OpenAIAdapter:
    """Adapter that wraps the existing ``rlmstudio.llm.OpenAIClient`` to
    satisfy the :class:`LLMPort` protocol.

    Args:
        client: An existing ``OpenAIClient`` instance.
    """

    def __init__(self, client: Any) -> None:
        self._client = client

    def complete(self, messages: list[dict[str, str]]) -> LLMResponseDTO:
        """Generate completion via the wrapped OpenAI client.

        Args:
            messages: Chat messages.

        Returns:
            LLMResponseDTO with content and token counts.
        """
        if hasattr(self._client, "complete_with_metadata"):
            resp = self._client.complete_with_metadata(messages)
            return LLMResponseDTO(
                content=resp.content,
                model=resp.model,
                input_tokens=resp.input_tokens or 0,
                output_tokens=resp.output_tokens or 0,
                finish_reason=resp.finish_reason,
            )
        text = self._client.complete(messages)
        model = getattr(self._client, "model", "")
        return LLMResponseDTO(content=text, model=model)

    def complete_stream(self, messages: list[dict[str, str]]) -> Iterator[str]:
        """Streaming is not yet implemented for the legacy client."""
        result = self.complete(messages)
        yield result.content

    def count_tokens(self, text: str) -> int:
        """Estimate tokens using the wrapped client or heuristic."""
        if hasattr(self._client, "estimate_tokens"):
            return int(self._client.estimate_tokens(text))
        return max(1, len(text) // 4)

    async def complete_async(self, messages: list[dict[str, str]]) -> LLMResponseDTO:
        """Async completion delegating to the sync method."""
        return await asyncio.to_thread(self.complete, messages)

    async def complete_stream_async(
        self, messages: list[dict[str, str]]
    ) -> AsyncIterator[StreamChunk]:
        """Non-blocking streaming: delegates to complete_async (asyncio.to_thread).

        The legacy OpenAI client does not support native streaming, so this
        method yields a single terminal ``StreamChunk`` carrying the full
        response DTO. ``ttft_ms`` is ``None`` (no streaming backend);
        token counts and content come from ``complete_with_metadata``.
        """
        response = await self.complete_async(messages)
        yield StreamChunk(delta=response.content, is_final=False)
        yield StreamChunk(delta="", is_final=True, response=response)

    def get_pricing(self) -> dict[str, float]:
        """Return pricing for the configured model."""
        model = getattr(self._client, "model", "gpt-4")
        pricing_db = {
            "gpt-4": {"input_cost_per_1m": 30.0, "output_cost_per_1m": 60.0},
            "gpt-4-turbo": {"input_cost_per_1m": 10.0, "output_cost_per_1m": 30.0},
            "gpt-4o": {"input_cost_per_1m": 5.0, "output_cost_per_1m": 15.0},
            "gpt-4o-mini": {"input_cost_per_1m": 0.15, "output_cost_per_1m": 0.60},
            "gpt-3.5-turbo": {"input_cost_per_1m": 0.50, "output_cost_per_1m": 1.50},
        }
        return pricing_db.get(model, pricing_db["gpt-4"])

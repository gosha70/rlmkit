"""OpenAI adapter: wraps the existing OpenAIClient to implement LLMPort."""

from __future__ import annotations

from typing import Dict, Iterator, List

from rlmkit.application.dto import LLMResponseDTO
from rlmkit.application.ports.llm_port import LLMPort


class OpenAIAdapter:
    """Adapter that wraps the existing ``rlmkit.llm.OpenAIClient`` to
    satisfy the :class:`LLMPort` protocol.

    Args:
        client: An existing ``OpenAIClient`` instance.
    """

    def __init__(self, client: object) -> None:
        self._client = client

    def complete(self, messages: List[Dict[str, str]]) -> LLMResponseDTO:
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

    def complete_stream(
        self, messages: List[Dict[str, str]]
    ) -> Iterator[str]:
        """Streaming is not yet implemented for the legacy client."""
        result = self.complete(messages)
        yield result.content

    def count_tokens(self, text: str) -> int:
        """Estimate tokens using the wrapped client or heuristic."""
        if hasattr(self._client, "estimate_tokens"):
            return self._client.estimate_tokens(text)
        return max(1, len(text) // 4)

    def get_pricing(self) -> Dict[str, float]:
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

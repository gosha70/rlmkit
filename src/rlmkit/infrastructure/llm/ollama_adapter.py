"""Ollama adapter: wraps the existing OllamaClient to implement LLMPort."""

from __future__ import annotations

from typing import Dict, Iterator, List

from rlmkit.application.dto import LLMResponseDTO
from rlmkit.application.ports.llm_port import LLMPort


class OllamaAdapter:
    """Adapter that wraps the existing ``rlmkit.llm.OllamaClient`` to
    satisfy the :class:`LLMPort` protocol.

    Args:
        client: An existing ``OllamaClient`` instance.
    """

    def __init__(self, client: object) -> None:
        self._client = client

    def complete(self, messages: List[Dict[str, str]]) -> LLMResponseDTO:
        """Generate completion via the wrapped Ollama client.

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
        """Estimate tokens using a heuristic."""
        return max(1, len(text) // 4)

    def get_pricing(self) -> Dict[str, float]:
        """Ollama models are free (local). Return zero pricing."""
        return {"input_cost_per_1m": 0.0, "output_cost_per_1m": 0.0}

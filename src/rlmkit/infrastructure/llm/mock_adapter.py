"""Mock LLM adapter for testing: implements LLMPort with canned responses."""

from __future__ import annotations

from typing import Dict, Iterator, List

from rlmkit.application.dto import LLMResponseDTO
from rlmkit.application.ports.llm_port import LLMPort


class MockLLMAdapter:
    """Mock adapter that returns pre-programmed responses.

    Useful for unit-testing use cases without hitting real LLM APIs.

    Args:
        responses: Ordered list of response strings. After the list is
            exhausted the last response is repeated.
    """

    def __init__(self, responses: List[str]) -> None:
        if not responses:
            raise ValueError("MockLLMAdapter requires at least one response")
        self._responses = responses
        self._call_count = 0
        self.call_history: List[List[Dict[str, str]]] = []

    def complete(self, messages: List[Dict[str, str]]) -> LLMResponseDTO:
        """Return the next canned response.

        Args:
            messages: Chat messages (recorded but ignored).

        Returns:
            LLMResponseDTO with the canned content.
        """
        self.call_history.append(list(messages))
        idx = min(self._call_count, len(self._responses) - 1)
        text = self._responses[idx]
        self._call_count += 1
        return LLMResponseDTO(content=text, model="mock")

    def complete_stream(
        self, messages: List[Dict[str, str]]
    ) -> Iterator[str]:
        """Yield the next canned response as a single chunk."""
        result = self.complete(messages)
        yield result.content

    def count_tokens(self, text: str) -> int:
        """Simple heuristic token count."""
        return max(1, len(text) // 4)

    def get_pricing(self) -> Dict[str, float]:
        """Mock pricing (free)."""
        return {"input_cost_per_1m": 0.0, "output_cost_per_1m": 0.0}

    def reset(self) -> None:
        """Reset call count and history."""
        self._call_count = 0
        self.call_history = []

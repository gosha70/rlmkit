"""LLM provider port: interface for language model completions.

Any LLM backend (OpenAI, Anthropic, Ollama, mock, etc.) must implement
this Protocol to be usable by the application use cases.
"""

from __future__ import annotations

from typing import Dict, Iterator, List, Optional, Protocol, runtime_checkable

from rlmkit.application.dto import LLMResponseDTO


@runtime_checkable
class LLMPort(Protocol):
    """Protocol for LLM provider adapters.

    Implementations wrap a specific LLM SDK/API and expose a uniform
    interface to the application layer.
    """

    def complete(self, messages: List[Dict[str, str]]) -> LLMResponseDTO:
        """Generate a completion from a list of chat messages.

        Args:
            messages: Ordered chat messages, each with 'role' and 'content'.

        Returns:
            LLMResponseDTO with generated text and token counts.
        """
        ...

    def complete_stream(
        self, messages: List[Dict[str, str]]
    ) -> Iterator[str]:
        """Generate a streaming completion, yielding text chunks.

        Args:
            messages: Ordered chat messages.

        Yields:
            Text chunks as they are produced.
        """
        ...

    def count_tokens(self, text: str) -> int:
        """Estimate the token count for a given text.

        Args:
            text: The text to tokenize.

        Returns:
            Estimated or exact token count.
        """
        ...

    def get_pricing(self) -> Dict[str, float]:
        """Return pricing information for the current model.

        Returns:
            Dictionary with 'input_cost_per_1m' and 'output_cost_per_1m' keys,
            values in USD per 1 million tokens.
        """
        ...

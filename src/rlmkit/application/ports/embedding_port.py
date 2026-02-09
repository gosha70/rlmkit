"""Embedding port: interface for text embedding providers.

Any embedding backend (OpenAI, sentence-transformers, etc.) must
implement this Protocol to be usable by RAG and vector search use cases.
"""

from __future__ import annotations

from typing import List, Protocol, runtime_checkable


@runtime_checkable
class EmbeddingPort(Protocol):
    """Protocol for embedding provider adapters."""

    def embed(self, text: str) -> List[float]:
        """Produce an embedding vector for a single text.

        Args:
            text: Input text to embed.

        Returns:
            Embedding vector as a list of floats.
        """
        ...

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Produce embedding vectors for a batch of texts.

        Args:
            texts: List of input texts.

        Returns:
            List of embedding vectors, one per input text.
        """
        ...

    @property
    def dimension(self) -> int:
        """Dimensionality of the embedding vectors produced.

        Returns:
            Integer dimension.
        """
        ...

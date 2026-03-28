"""LiteLLM embedding adapter: implements EmbeddingPort via litellm.embedding().

Supports any embedding model accessible through LiteLLM (OpenAI, Azure,
Cohere, Bedrock, etc.) with a single unified interface. Uses the same
litellm dependency already required by rlmkit's LLM adapter.

This is the lightweight embedding implementation for RAG mode. It will be
superseded by rag-core's embedder once rag-kit reaches v0.1.0
(see doc_internal/plans/rag-kit-B-rlmkit-adoption.md).
"""

from __future__ import annotations

import logging

import litellm

logger = logging.getLogger(__name__)

# Known dimensions for common models — avoids a round-trip to discover dimension.
_KNOWN_DIMENSIONS: dict[str, int] = {
    "text-embedding-3-small": 1536,
    "text-embedding-3-large": 3072,
    "text-embedding-ada-002": 1536,
    "openai/text-embedding-3-small": 1536,
    "openai/text-embedding-3-large": 3072,
}


class LiteLLMEmbeddingAdapter:
    """EmbeddingPort implementation backed by litellm.embedding().

    Args:
        model: LiteLLM embedding model string (e.g. ``"text-embedding-3-small"``).
        api_key: Optional API key; falls back to environment variables.
        api_base: Optional custom endpoint (e.g. for Azure or local proxies).
        dimensions: Override embedding dimension. If None, uses a known-dimension
            lookup and falls back to a probe call on first use.
    """

    def __init__(
        self,
        model: str = "text-embedding-3-small",
        api_key: str | None = None,
        api_base: str | None = None,
        dimensions: int | None = None,
    ) -> None:
        self._model = model
        self._api_key = api_key
        self._api_base = api_base
        self._dimension: int | None = dimensions or _KNOWN_DIMENSIONS.get(model)

    # -- EmbeddingPort protocol --

    def embed(self, text: str) -> list[float]:
        """Produce an embedding vector for a single text."""
        return self.embed_batch([text])[0]

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Produce embedding vectors for a batch of texts."""
        kwargs: dict = {"model": self._model, "input": texts}
        if self._api_key:
            kwargs["api_key"] = self._api_key
        if self._api_base:
            kwargs["api_base"] = self._api_base

        response = litellm.embedding(**kwargs)
        vectors = [item["embedding"] for item in response.data]

        # Cache dimension from first real response
        if vectors and self._dimension is None:
            self._dimension = len(vectors[0])

        return vectors

    @property
    def dimension(self) -> int:
        """Dimensionality of vectors produced by this model."""
        if self._dimension is None:
            # Probe with a minimal string to discover dimension
            self._dimension = len(self.embed("a"))
        return self._dimension

    @property
    def model(self) -> str:
        return self._model

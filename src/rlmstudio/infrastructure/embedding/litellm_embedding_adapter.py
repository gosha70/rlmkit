"""LiteLLM embedding adapter: implements EmbeddingPort via litellm.embedding().

Supports any embedding model accessible through LiteLLM (OpenAI, Azure,
Cohere, Bedrock, etc.) with a single unified interface. Uses the same
litellm dependency already required by rlmstudio's LLM adapter.

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

# Pricing in USD per 1M tokens for common embedding models.
_EMBEDDING_COST_PER_1M: dict[str, float] = {
    "text-embedding-3-small": 0.020,
    "text-embedding-3-large": 0.130,
    "text-embedding-ada-002": 0.100,
    "openai/text-embedding-3-small": 0.020,
    "openai/text-embedding-3-large": 0.130,
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
        self._total_tokens: int = 0
        self._cost_per_1m: float = _EMBEDDING_COST_PER_1M.get(model, 0.0)

    # -- EmbeddingPort protocol --

    def embed(self, text: str) -> list[float]:
        """Produce an embedding vector for a single text."""
        return self.embed_batch([text])[0]

    def embed_batch(self, texts: list[str], batch_size: int = 96) -> list[list[float]]:
        """Produce embedding vectors for a batch of texts.

        Sends at most *batch_size* texts per API call to stay within
        provider token-per-request limits (e.g. OpenAI caps at 300 K tokens).
        """
        all_vectors: list[list[float]] = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            kwargs: dict = {"model": self._model, "input": batch}
            if self._api_key:
                kwargs["api_key"] = self._api_key
            if self._api_base:
                kwargs["api_base"] = self._api_base

            response = litellm.embedding(**kwargs)
            vectors = [item["embedding"] for item in response.data]
            all_vectors.extend(vectors)

            # Accumulate token usage from the response
            usage = getattr(response, "usage", None)
            if usage is not None:
                self._total_tokens += getattr(usage, "total_tokens", 0) or getattr(
                    usage, "prompt_tokens", 0
                )

            # Cache dimension from first real response
            if vectors and self._dimension is None:
                self._dimension = len(vectors[0])

        return all_vectors

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

    @property
    def total_tokens(self) -> int:
        """Cumulative tokens consumed by all embed / embed_batch calls so far."""
        return self._total_tokens

    @property
    def total_cost(self) -> float:
        """Cumulative cost in USD for all embedding calls so far."""
        return self._total_tokens * self._cost_per_1m / 1_000_000

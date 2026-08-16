"""Mock embedding adapter for testing — returns deterministic fake vectors.

Produces normalised, hash-derived vectors without any API calls so that
tests with ``provider="mock"`` can exercise the full RAG pipeline without
needing an OpenAI key.
"""

from __future__ import annotations

import hashlib

_DIMENSION = 16  # small but non-trivial; real embeddings are 1536


class MockEmbeddingAdapter:
    """Returns deterministic normalised vectors derived from input text.

    Vectors are stable (same input → same output) and non-zero, which
    satisfies the cosine-similarity search in SQLiteStorageAdapter.
    No external calls are made.
    """

    def embed(self, text: str) -> list[float]:
        return self.embed_batch([text])[0]

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return [self._hash_vector(t) for t in texts]

    @property
    def dimension(self) -> int:
        return _DIMENSION

    @staticmethod
    def _hash_vector(text: str) -> list[float]:
        digest = hashlib.sha256(text.encode()).digest()
        raw = [digest[i] / 255.0 for i in range(_DIMENSION)]
        norm = sum(v * v for v in raw) ** 0.5 or 1.0
        return [v / norm for v in raw]

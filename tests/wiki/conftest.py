# Copyright (c) EGOGE - All Rights Reserved.
# This software may be used and distributed according to the terms of the MIT license.

"""Deterministic stubs for wiki tests — no network, no real embeddings."""

from __future__ import annotations

import hashlib

import pytest


class StubLLMClient:
    """LLMClient stub: returns canned replies keyed off the user message."""

    def __init__(self, replies: dict[str, str] | None = None, default: str = ""):
        self.replies = replies or {}
        self.default = default
        self.calls: list[list[dict[str, str]]] = []

    def complete(self, messages: list[dict[str, str]]) -> str:
        self.calls.append(messages)
        last_user = next(
            (m["content"] for m in reversed(messages) if m["role"] == "user"),
            "",
        )
        for needle, reply in self.replies.items():
            if needle in last_user:
                return reply
        return self.default


class StubEmbedder:
    """Deterministic, hash-based embedder.

    Produces a 32-dim vector by hashing every 8-character chunk of the
    input. Cosine similarity of identical strings is 1.0; texts that share
    a salient keyword cluster nearby because the keyword's hash dominates.
    """

    DIM = 32

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [self._embed_one(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed_one(text)

    def _embed_one(self, text: str) -> list[float]:
        vec = [0.0] * self.DIM
        # Bag-of-tokens hashing: each token contributes to one slot.
        for token in text.lower().split():
            digest = hashlib.sha256(token.encode("utf-8")).digest()
            slot = digest[0] % self.DIM
            vec[slot] += 1.0
        # L2 normalize so cosine == dot.
        import math

        norm = math.sqrt(sum(v * v for v in vec))
        if norm > 0:
            vec = [v / norm for v in vec]
        return vec


@pytest.fixture
def wiki_root(tmp_path):
    return tmp_path / "knowledge"


@pytest.fixture
def stub_embedder():
    return StubEmbedder()


@pytest.fixture
def make_stub_llm():
    def _factory(**kwargs):
        return StubLLMClient(**kwargs)

    return _factory

"""Tests for RAG strategy."""

import math
import pytest
from rlmkit import MockLLMClient
from rlmkit.strategies.base import LLMStrategy
from rlmkit.strategies.rag import RAGStrategy


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class MockEmbedder:
    """Deterministic mock: embed text to a vector based on character counts."""

    def embed(self, texts):
        return [self._vec(t) for t in texts]

    def embed_query(self, text):
        return self._vec(text)

    @staticmethod
    def _vec(text):
        # Simple deterministic embedding: [len, vowel_ratio, digit_ratio]
        length = max(len(text), 1)
        vowels = sum(1 for c in text.lower() if c in "aeiou")
        digits = sum(1 for c in text if c.isdigit())
        return [length / 1000, vowels / length, digits / length]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestRAGProtocol:
    def test_is_strategy(self):
        client = MockLLMClient(["answer"])
        s = RAGStrategy(client=client, embedder=MockEmbedder())
        assert isinstance(s, LLMStrategy)

    def test_name(self):
        client = MockLLMClient(["answer"])
        s = RAGStrategy(client=client, embedder=MockEmbedder())
        assert s.name == "rag"


class TestCosineSimilarity:
    def test_identical_vectors(self):
        assert RAGStrategy._cosine_similarity([1, 0, 0], [1, 0, 0]) == pytest.approx(1.0)

    def test_orthogonal_vectors(self):
        assert RAGStrategy._cosine_similarity([1, 0, 0], [0, 1, 0]) == pytest.approx(0.0)

    def test_opposite_vectors(self):
        assert RAGStrategy._cosine_similarity([1, 0], [-1, 0]) == pytest.approx(-1.0)

    def test_zero_vector(self):
        assert RAGStrategy._cosine_similarity([0, 0], [1, 1]) == 0.0

    def test_known_value(self):
        a = [1.0, 2.0, 3.0]
        b = [4.0, 5.0, 6.0]
        expected = (4 + 10 + 18) / (math.sqrt(14) * math.sqrt(77))
        assert RAGStrategy._cosine_similarity(a, b) == pytest.approx(expected)


class TestChunking:
    def test_chunk_content(self):
        s = RAGStrategy(
            client=MockLLMClient(["x"]),
            embedder=MockEmbedder(),
            chunk_size=10,
            chunk_overlap=0,
        )
        chunks = s._chunk_content("a" * 25)
        assert len(chunks) == 3
        assert chunks[0] == "a" * 10
        assert chunks[2] == "a" * 5

    def test_chunk_with_overlap(self):
        s = RAGStrategy(
            client=MockLLMClient(["x"]),
            embedder=MockEmbedder(),
            chunk_size=10,
            chunk_overlap=5,
        )
        chunks = s._chunk_content("a" * 20)
        # step = 10 - 5 = 5, positions: 0, 5, 10, 15
        assert len(chunks) == 4

    def test_small_content(self):
        s = RAGStrategy(
            client=MockLLMClient(["x"]),
            embedder=MockEmbedder(),
            chunk_size=1000,
            chunk_overlap=0,
        )
        chunks = s._chunk_content("short text")
        assert len(chunks) == 1
        assert chunks[0] == "short text"


class TestRankChunks:
    def test_ranking_order(self):
        s = RAGStrategy(
            client=MockLLMClient(["x"]),
            embedder=MockEmbedder(),
        )
        query_emb = [1.0, 0.0, 0.0]
        chunk_embs = [
            [0.0, 1.0, 0.0],  # orthogonal
            [1.0, 0.0, 0.0],  # identical
            [0.5, 0.5, 0.0],  # partial match
        ]
        chunks = ["chunk_a", "chunk_b", "chunk_c"]

        ranked = s._rank_chunks(query_emb, chunk_embs, chunks)
        assert ranked[0][1] == "chunk_b"  # most similar first
        assert ranked[-1][1] == "chunk_a"  # least similar last


class TestAssembleContext:
    def test_format(self):
        s = RAGStrategy(
            client=MockLLMClient(["x"]),
            embedder=MockEmbedder(),
        )
        result = s._assemble_context([(0.95, "hello"), (0.8, "world")])
        assert "[Chunk 1 (relevance: 0.950)]" in result
        assert "[Chunk 2 (relevance: 0.800)]" in result
        assert "hello" in result
        assert "world" in result


class TestRAGRun:
    def test_basic_run(self):
        client = MockLLMClient(["RAG answer here"])
        s = RAGStrategy(
            client=client,
            embedder=MockEmbedder(),
            chunk_size=50,
            chunk_overlap=0,
            top_k=2,
        )
        content = "A" * 100 + "B" * 100
        result = s.run(content, "what letters?")

        assert result.success
        assert result.answer == "RAG answer here"
        assert result.strategy == "rag"
        assert result.steps == 1
        assert result.metadata["chunks_total"] == 4
        assert result.metadata["chunks_retrieved"] == 2
        assert result.tokens.total_tokens > 0
        assert len(result.trace) == 2  # retrieval + generation

    def test_empty_content(self):
        client = MockLLMClient(["x"])
        s = RAGStrategy(client=client, embedder=MockEmbedder(), chunk_size=10, chunk_overlap=0)
        result = s.run("", "query")
        assert not result.success
        assert "No chunks" in result.error

    def test_error_handling(self):
        class FailingClient:
            def complete(self, messages):
                raise RuntimeError("generation failed")

        s = RAGStrategy(client=FailingClient(), embedder=MockEmbedder(), chunk_size=10, chunk_overlap=0)
        result = s.run("some content here", "query")
        assert not result.success
        assert "generation failed" in result.error

    def test_embedder_error(self):
        class FailingEmbedder:
            def embed(self, texts):
                raise RuntimeError("embed failed")
            def embed_query(self, text):
                raise RuntimeError("embed failed")

        s = RAGStrategy(
            client=MockLLMClient(["x"]),
            embedder=FailingEmbedder(),
            chunk_size=10,
            chunk_overlap=0,
        )
        result = s.run("some content here", "query")
        assert not result.success
        assert "embed failed" in result.error

"""Tests for IndexedRAGStrategy with persistent vector store."""

import pytest

from rlmkit.strategies.base import LLMStrategy, StrategyResult
from rlmkit.strategies.indexed_rag import IndexedRAGStrategy
from rlmkit.storage.database import Database
from rlmkit.storage.vector_store import VectorStore


# ---------------------------------------------------------------------------
# Mocks
# ---------------------------------------------------------------------------

class MockLLMClient:
    """Mock LLM client that echoes back a fixed answer."""

    def __init__(self, answer: str = "mock answer"):
        self.answer = answer
        self.calls = []

    def complete(self, messages):
        self.calls.append(messages)
        return self.answer


class MockEmbedder:
    """Mock embedder that returns simple hash-based vectors."""

    def __init__(self, dim: int = 8):
        self.dim = dim
        self.embed_calls = 0
        self.embed_query_calls = 0

    def embed(self, texts):
        self.embed_calls += 1
        return [self._hash_vec(t) for t in texts]

    def embed_query(self, text):
        self.embed_query_calls += 1
        return self._hash_vec(text)

    def _hash_vec(self, text):
        """Deterministic pseudo-embedding based on text hash."""
        h = hash(text) & 0xFFFFFFFF
        vec = []
        for i in range(self.dim):
            val = ((h >> (i * 4)) & 0xF) / 15.0  # 0.0 to 1.0
            vec.append(val)
        return vec


@pytest.fixture
def setup(tmp_path):
    db = Database(tmp_path / "test.db")
    vs = VectorStore(db)
    client = MockLLMClient(answer="The answer is 42")
    embedder = MockEmbedder()
    rag = IndexedRAGStrategy(
        client=client,
        embedder=embedder,
        vector_store=vs,
        collection="test_col",
        chunk_size=50,
        chunk_overlap=10,
        top_k=3,
    )
    return rag, vs, client, embedder


# ---------------------------------------------------------------------------
# Protocol conformance
# ---------------------------------------------------------------------------

class TestProtocol:
    def test_satisfies_llm_strategy(self, setup):
        rag, *_ = setup
        assert isinstance(rag, LLMStrategy)

    def test_name_is_rag(self, setup):
        rag, *_ = setup
        assert rag.name == "rag"


# ---------------------------------------------------------------------------
# Indexing
# ---------------------------------------------------------------------------

class TestIndexing:
    def test_index_content_stores_chunks(self, setup):
        rag, vs, _, embedder = setup
        content = "A " * 100  # long enough to produce multiple chunks
        added = rag.index_content(content, source_id="doc_1")
        assert added > 0
        assert vs.get_collection_size("test_col") == added

    def test_index_content_skips_already_indexed(self, setup):
        rag, vs, _, embedder = setup
        content = "Hello world " * 20
        added1 = rag.index_content(content, source_id="doc_1")
        assert added1 > 0
        embedder.embed_calls = 0  # reset counter
        added2 = rag.index_content(content, source_id="doc_1")
        assert added2 == 0
        assert embedder.embed_calls == 0  # no re-embedding

    def test_index_different_sources(self, setup):
        rag, vs, _, _ = setup
        rag.index_content("Content A " * 20, source_id="doc_a")
        rag.index_content("Content B " * 20, source_id="doc_b")
        assert vs.get_collection_size("test_col") > 0
        assert vs.has_source("test_col", "doc_a")
        assert vs.has_source("test_col", "doc_b")

    def test_index_auto_generates_source_id(self, setup):
        rag, vs, _, _ = setup
        added = rag.index_content("Some content " * 20)
        assert added > 0


# ---------------------------------------------------------------------------
# Run (end-to-end)
# ---------------------------------------------------------------------------

class TestRun:
    def test_run_produces_strategy_result(self, setup):
        rag, *_ = setup
        content = "The meaning of life is 42. " * 20
        result = rag.run(content, "What is the meaning of life?")
        assert isinstance(result, StrategyResult)
        assert result.strategy == "rag"
        assert result.success is True
        assert result.answer == "The answer is 42"

    def test_run_auto_indexes(self, setup):
        rag, vs, _, _ = setup
        content = "Einstein developed relativity. " * 20
        result = rag.run(content, "Who developed relativity?")
        assert result.success
        assert result.metadata["chunks_added"] > 0
        assert vs.get_collection_size("test_col") > 0

    def test_run_does_not_reindex(self, setup):
        rag, vs, client, embedder = setup
        content = "Reusable content " * 20
        # First run indexes
        rag.run(content, "q1")
        embed_calls_after_first = embedder.embed_calls

        # Second run should not re-embed document chunks
        embedder.embed_calls = 0
        rag.run(content, "q2")
        # Only query embedding should have happened (embed_query)
        assert embedder.embed_calls == 0  # embed() not called again

    def test_run_metadata(self, setup):
        rag, *_ = setup
        content = "Metadata test content " * 20
        result = rag.run(content, "test query")
        assert result.metadata["indexed"] is True
        assert result.metadata["collection"] == "test_col"
        assert "collection_size" in result.metadata

    def test_run_trace(self, setup):
        rag, *_ = setup
        content = "Trace test " * 20
        result = rag.run(content, "trace?")
        assert len(result.trace) == 2
        assert result.trace[0]["role"] == "retrieval"
        assert result.trace[1]["role"] == "assistant"

    def test_run_with_empty_content(self, setup):
        rag, *_ = setup
        result = rag.run("", "question?")
        assert result.success is False

    def test_run_llm_called_once(self, setup):
        rag, _, client, _ = setup
        content = "Some document " * 20
        rag.run(content, "question?")
        assert len(client.calls) == 1  # one LLM call per run

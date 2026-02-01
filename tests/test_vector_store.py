"""Tests for SQLite-backed VectorStore."""

import math
import struct
import pytest

from rlmkit.storage.database import Database
from rlmkit.storage.vector_store import VectorStore


@pytest.fixture
def vs(tmp_path):
    db = Database(tmp_path / "test.db")
    return VectorStore(db)


# ---------------------------------------------------------------------------
# Embedding serialization
# ---------------------------------------------------------------------------

class TestEmbeddingSerialization:
    def test_round_trip(self):
        emb = [0.1, 0.2, 0.3, -0.5, 1.0]
        blob = VectorStore._serialize_embedding(emb)
        restored = VectorStore._deserialize_embedding(blob, len(emb))
        for a, b in zip(emb, restored):
            assert abs(a - b) < 1e-6

    def test_blob_size(self):
        emb = [0.0] * 128
        blob = VectorStore._serialize_embedding(emb)
        assert len(blob) == 128 * 4  # float32 = 4 bytes


# ---------------------------------------------------------------------------
# Cosine similarity
# ---------------------------------------------------------------------------

class TestCosineSimilarity:
    def test_identical(self):
        v = [1.0, 2.0, 3.0]
        assert VectorStore._cosine_similarity(v, v) == pytest.approx(1.0)

    def test_orthogonal(self):
        a = [1.0, 0.0]
        b = [0.0, 1.0]
        assert VectorStore._cosine_similarity(a, b) == pytest.approx(0.0)

    def test_opposite(self):
        a = [1.0, 0.0]
        b = [-1.0, 0.0]
        assert VectorStore._cosine_similarity(a, b) == pytest.approx(-1.0)

    def test_zero_vector(self):
        assert VectorStore._cosine_similarity([0, 0], [1, 2]) == 0.0


# ---------------------------------------------------------------------------
# Add + Search
# ---------------------------------------------------------------------------

class TestAddAndSearch:
    def test_add_and_search(self, vs):
        chunks = ["hello world", "foo bar", "baz qux"]
        # Simple embeddings: each chunk gets a unit vector in a different dimension
        embeddings = [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
        added = vs.add_chunks("test_col", chunks, embeddings)
        assert added == 3

        # Query aligned with first chunk
        results = vs.search("test_col", [1.0, 0.0, 0.0], top_k=2)
        assert len(results) == 2
        assert results[0][2] == "hello world"
        assert results[0][0] == pytest.approx(1.0)

    def test_top_k_limits(self, vs):
        chunks = [f"chunk {i}" for i in range(10)]
        embeddings = [[float(i == j) for j in range(10)] for i in range(10)]
        vs.add_chunks("col", chunks, embeddings)

        results = vs.search("col", embeddings[0], top_k=3)
        assert len(results) == 3

    def test_search_empty_collection(self, vs):
        results = vs.search("empty", [1.0, 0.0], top_k=5)
        assert results == []

    def test_collection_size(self, vs):
        assert vs.get_collection_size("col") == 0
        vs.add_chunks("col", ["a", "b"], [[1.0], [2.0]])
        assert vs.get_collection_size("col") == 2

    def test_collections_are_isolated(self, vs):
        vs.add_chunks("col_a", ["chunk a"], [[1.0, 0.0]])
        vs.add_chunks("col_b", ["chunk b"], [[0.0, 1.0]])
        assert vs.get_collection_size("col_a") == 1
        assert vs.get_collection_size("col_b") == 1

        results = vs.search("col_a", [1.0, 0.0], top_k=10)
        assert len(results) == 1
        assert results[0][2] == "chunk a"


# ---------------------------------------------------------------------------
# Deduplication
# ---------------------------------------------------------------------------

class TestDeduplication:
    def test_same_text_not_readded(self, vs):
        chunks = ["duplicate", "duplicate", "unique"]
        embeddings = [[1.0], [1.0], [2.0]]
        added = vs.add_chunks("col", chunks, embeddings)
        assert added == 2  # "duplicate" stored once
        assert vs.get_collection_size("col") == 2

    def test_same_text_different_collection(self, vs):
        vs.add_chunks("col_a", ["text"], [[1.0]])
        added = vs.add_chunks("col_b", ["text"], [[1.0]])
        assert added == 1  # Different collection, so it's new
        assert vs.get_collection_size("col_a") == 1
        assert vs.get_collection_size("col_b") == 1


# ---------------------------------------------------------------------------
# Source tracking
# ---------------------------------------------------------------------------

class TestSourceTracking:
    def test_has_source(self, vs):
        vs.add_chunks("col", ["chunk"], [[1.0]], source_id="src_1")
        assert vs.has_source("col", "src_1") is True
        assert vs.has_source("col", "src_2") is False

    def test_has_source_wrong_collection(self, vs):
        vs.add_chunks("col_a", ["chunk"], [[1.0]], source_id="src_1")
        assert vs.has_source("col_b", "src_1") is False


# ---------------------------------------------------------------------------
# Delete collection
# ---------------------------------------------------------------------------

class TestDeleteCollection:
    def test_delete(self, vs):
        vs.add_chunks("col", ["a", "b", "c"], [[1.0], [2.0], [3.0]])
        assert vs.get_collection_size("col") == 3
        vs.delete_collection("col")
        assert vs.get_collection_size("col") == 0

    def test_delete_nonexistent(self, vs):
        # Should not raise
        vs.delete_collection("nope")

    def test_delete_does_not_affect_others(self, vs):
        vs.add_chunks("keep", ["a"], [[1.0]])
        vs.add_chunks("remove", ["b"], [[2.0]])
        vs.delete_collection("remove")
        assert vs.get_collection_size("keep") == 1

"""Tests for embedding provider abstraction."""

import pytest
from unittest.mock import MagicMock, patch
from rlmkit.strategies.embeddings import EmbeddingProvider, OpenAIEmbedder


class MockEmbedder:
    """Simple mock embedder returning fixed-dimension vectors."""

    def embed(self, texts):
        # Return a unit vector of length 3 for each text
        return [[1.0, 0.0, 0.0] for _ in texts]

    def embed_query(self, text):
        return self.embed([text])[0]


class TestEmbeddingProtocol:
    def test_mock_satisfies_protocol(self):
        assert isinstance(MockEmbedder(), EmbeddingProvider)

    def test_openai_satisfies_protocol(self):
        embedder = OpenAIEmbedder(api_key="fake")
        assert isinstance(embedder, EmbeddingProvider)


class TestMockEmbedder:
    def test_embed_returns_list_of_vectors(self):
        e = MockEmbedder()
        vecs = e.embed(["hello", "world"])
        assert len(vecs) == 2
        assert len(vecs[0]) == 3

    def test_embed_query_returns_single_vector(self):
        e = MockEmbedder()
        vec = e.embed_query("hello")
        assert len(vec) == 3


class TestOpenAIEmbedder:
    def test_lazy_client_init(self):
        embedder = OpenAIEmbedder(api_key="test-key")
        assert embedder._client is None

    def test_embed_calls_openai(self):
        mock_client = MagicMock()

        item1 = MagicMock()
        item1.embedding = [0.1, 0.2, 0.3]
        item2 = MagicMock()
        item2.embedding = [0.4, 0.5, 0.6]
        mock_client.embeddings.create.return_value = MagicMock(data=[item1, item2])

        embedder = OpenAIEmbedder(api_key="k")
        embedder._client = mock_client  # inject mock, skip real openai import

        vecs = embedder.embed(["a", "b"])
        assert len(vecs) == 2
        assert vecs[0] == [0.1, 0.2, 0.3]
        mock_client.embeddings.create.assert_called_once_with(
            input=["a", "b"], model="text-embedding-3-small"
        )

    def test_embed_query(self):
        mock_client = MagicMock()

        item = MagicMock()
        item.embedding = [1.0, 2.0, 3.0]
        mock_client.embeddings.create.return_value = MagicMock(data=[item])

        embedder = OpenAIEmbedder(api_key="k")
        embedder._client = mock_client

        vec = embedder.embed_query("hello")
        assert vec == [1.0, 2.0, 3.0]

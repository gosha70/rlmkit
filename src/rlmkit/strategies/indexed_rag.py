# Copyright (c) EGOGE - All Rights Reserved.
# This software may be used and distributed according to the terms of the MIT license.

"""RAG strategy with persistent vector store — embeds content once, queries many times."""

import hashlib
import time
from typing import Optional, List

from rlmkit.core.rlm import LLMClient
from rlmkit.core.budget import TokenUsage, estimate_tokens
from rlmkit.tools.content import chunk as chunk_content
from rlmkit.storage.vector_store import VectorStore
from .base import StrategyResult
from .embeddings import EmbeddingProvider


class IndexedRAGStrategy:
    """RAG with persistent vector store.

    Content is embedded and stored once. Subsequent queries only embed the
    query and search the index — no redundant document embedding.

    Satisfies the LLMStrategy protocol (name + run).
    """

    def __init__(
        self,
        client: LLMClient,
        embedder: EmbeddingProvider,
        vector_store: VectorStore,
        collection: str,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        top_k: int = 5,
        system_prompt: Optional[str] = None,
    ):
        self.client = client
        self.embedder = embedder
        self.vector_store = vector_store
        self.collection = collection
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.top_k = top_k
        self.system_prompt = (
            system_prompt
            or "You are a helpful assistant. Answer the user's question based on "
            "the provided context chunks. If the context doesn't contain enough "
            "information, say so."
        )

    @property
    def name(self) -> str:
        return "rag"

    def index_content(self, content: str, source_id: Optional[str] = None) -> int:
        """Index new content into the vector store.

        If source_id is given and already indexed, skips entirely.
        Returns the number of new chunks added.
        """
        if source_id is None:
            source_id = hashlib.sha256(content.encode()).hexdigest()

        if self.vector_store.has_source(self.collection, source_id):
            return 0

        chunks = chunk_content(content, size=self.chunk_size, overlap=self.chunk_overlap)
        if not chunks:
            return 0

        embeddings = self.embedder.embed(chunks)
        return self.vector_store.add_chunks(
            self.collection, chunks, embeddings, source_id=source_id,
        )

    def run(self, content: str, query: str) -> StrategyResult:
        """Auto-index content if new, then query index and generate answer."""
        start = time.time()

        try:
            # Step 1: Index content (skips if already indexed)
            chunks_added = self.index_content(content)

            # Step 2: Embed query and search index
            query_embedding = self.embedder.embed_query(query)
            results = self.vector_store.search(
                self.collection, query_embedding, top_k=self.top_k,
            )

            if not results:
                return StrategyResult(
                    strategy="rag",
                    answer="",
                    success=False,
                    error="No indexed chunks found for query",
                    elapsed_time=time.time() - start,
                )

            # Step 3: Build prompt with retrieved context
            context = self._assemble_context(results)
            messages = [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}"},
            ]

            # Step 4: Generate answer
            answer = self.client.complete(messages)

        except Exception as e:
            return StrategyResult(
                strategy="rag",
                answer="",
                success=False,
                error=str(e),
                elapsed_time=time.time() - start,
            )

        elapsed = time.time() - start

        # Token accounting
        context_text = context + query + self.system_prompt
        tokens = TokenUsage()
        tokens.add_input(estimate_tokens(context_text))
        tokens.add_output(estimate_tokens(answer))

        return StrategyResult(
            strategy="rag",
            answer=answer,
            steps=1,
            tokens=tokens,
            elapsed_time=elapsed,
            trace=[
                {
                    "step": 1,
                    "role": "retrieval",
                    "chunks_retrieved": len(results),
                    "chunks_added": chunks_added,
                    "top_scores": [round(s, 4) for s, _, _ in results],
                },
                {"step": 2, "role": "assistant", "content": answer, "mode": "indexed_rag"},
            ],
            metadata={
                "collection": self.collection,
                "collection_size": self.vector_store.get_collection_size(self.collection),
                "chunks_retrieved": len(results),
                "chunks_added": chunks_added,
                "chunk_size": self.chunk_size,
                "chunk_overlap": self.chunk_overlap,
                "top_k": self.top_k,
                "indexed": True,
            },
        )

    def _assemble_context(self, results: list) -> str:
        """Format retrieved chunks for the LLM prompt."""
        parts = []
        for i, (score, _chunk_id, text) in enumerate(results, 1):
            parts.append(f"[Chunk {i} (relevance: {score:.3f})]\n{text}")
        return "\n\n".join(parts)

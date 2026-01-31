# Copyright (c) EGOGE - All Rights Reserved.
# This software may be used and distributed according to the terms of the MIT license.

"""RAG strategy — retrieve relevant chunks then generate an answer."""

import math
import time
from typing import List, Optional, Tuple

from rlmkit.core.rlm import LLMClient
from rlmkit.core.budget import TokenUsage, estimate_tokens
from rlmkit.tools.content import chunk as chunk_content
from .base import StrategyResult
from .embeddings import EmbeddingProvider


class RAGStrategy:
    """Retrieve-then-read: embed chunks, rank by similarity, generate."""

    def __init__(
        self,
        client: LLMClient,
        embedder: EmbeddingProvider,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        top_k: int = 5,
        system_prompt: Optional[str] = None,
    ):
        self.client = client
        self.embedder = embedder
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.top_k = top_k
        self.system_prompt = (
            system_prompt
            or "You are a helpful assistant. Answer the user's question based on the provided context chunks. "
            "If the context doesn't contain enough information, say so."
        )

    @property
    def name(self) -> str:
        return "rag"

    def run(self, content: str, query: str) -> StrategyResult:
        start = time.time()

        try:
            # Step 1: Chunk
            chunks = self._chunk_content(content)
            if not chunks:
                return StrategyResult(
                    strategy="rag",
                    answer="",
                    success=False,
                    error="No chunks produced from content",
                    elapsed_time=time.time() - start,
                )

            # Step 2: Embed chunks + query
            chunk_embeddings = self.embedder.embed(chunks)
            query_embedding = self.embedder.embed_query(query)

            # Step 3: Rank and select top-k
            ranked = self._rank_chunks(query_embedding, chunk_embeddings, chunks)
            top_chunks = ranked[: self.top_k]

            # Step 4: Build prompt with retrieved context
            context = self._assemble_context(top_chunks)
            messages = [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}"},
            ]

            # Step 5: Generate answer
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

        # Token accounting (generation only — embedding tokens tracked in metadata)
        context_text = context + query + self.system_prompt
        tokens = TokenUsage()
        tokens.add_input(estimate_tokens(context_text))
        tokens.add_output(estimate_tokens(answer))

        embedding_input = " ".join(chunks) + " " + query
        embedding_tokens = estimate_tokens(embedding_input)

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
                    "chunks_total": len(chunks),
                    "chunks_retrieved": len(top_chunks),
                    "top_scores": [round(s, 4) for s, _ in top_chunks],
                },
                {"step": 2, "role": "assistant", "content": answer, "mode": "rag"},
            ],
            metadata={
                "chunks_total": len(chunks),
                "chunks_retrieved": len(top_chunks),
                "chunk_size": self.chunk_size,
                "chunk_overlap": self.chunk_overlap,
                "top_k": self.top_k,
                "embedding_tokens": embedding_tokens,
            },
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _chunk_content(self, content: str) -> List[str]:
        return chunk_content(
            content,
            size=self.chunk_size,
            overlap=self.chunk_overlap,
        )

    @staticmethod
    def _cosine_similarity(a: List[float], b: List[float]) -> float:
        dot = sum(x * y for x, y in zip(a, b))
        norm_a = math.sqrt(sum(x * x for x in a))
        norm_b = math.sqrt(sum(x * x for x in b))
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot / (norm_a * norm_b)

    def _rank_chunks(
        self,
        query_emb: List[float],
        chunk_embs: List[List[float]],
        chunks: List[str],
    ) -> List[Tuple[float, str]]:
        scored = [
            (self._cosine_similarity(query_emb, emb), text)
            for emb, text in zip(chunk_embs, chunks)
        ]
        scored.sort(key=lambda x: x[0], reverse=True)
        return scored

    def _assemble_context(self, ranked_chunks: List[Tuple[float, str]]) -> str:
        parts = []
        for i, (score, text) in enumerate(ranked_chunks, 1):
            parts.append(f"[Chunk {i} (relevance: {score:.3f})]\n{text}")
        return "\n\n".join(parts)

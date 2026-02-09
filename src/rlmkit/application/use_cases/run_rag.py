"""Use case: run a RAG (Retrieval-Augmented Generation) query.

Chunks the content, embeds it, retrieves the most relevant chunks
for the query, and sends them to the LLM as context.
"""

from __future__ import annotations

import time
from typing import List, Optional

from rlmkit.application.dto import LLMResponseDTO, RunConfigDTO, RunResultDTO
from rlmkit.application.ports.llm_port import LLMPort
from rlmkit.application.ports.embedding_port import EmbeddingPort
from rlmkit.application.ports.storage_port import StoragePort


class RunRAGUseCase:
    """Orchestrates a RAG pipeline through ports.

    Args:
        llm: LLM port adapter.
        embedder: Embedding port adapter.
        storage: Storage port adapter with vector search capability.
    """

    def __init__(
        self,
        llm: LLMPort,
        embedder: EmbeddingPort,
        storage: StoragePort,
    ) -> None:
        self._llm = llm
        self._embedder = embedder
        self._storage = storage

    def execute(
        self,
        content: str,
        query: str,
        config: Optional[RunConfigDTO] = None,
    ) -> RunResultDTO:
        """Run a RAG query: chunk, embed, retrieve, generate.

        Args:
            content: Document text to analyze.
            query: User question.
            config: Optional run configuration.

        Returns:
            RunResultDTO with the answer and metrics.
        """
        config = config or RunConfigDTO(mode="rag")
        start = time.time()
        top_k = config.extra.get("top_k", 5)
        chunk_size = config.extra.get("chunk_size", 1000)
        collection = config.extra.get("collection", "rag_temp")

        try:
            # 1. Chunk the content
            chunks = self._chunk_text(content, chunk_size)

            # 2. Embed all chunks
            embeddings = self._embedder.embed_batch(chunks)

            # 3. Store chunks with embeddings
            self._storage.add_chunks(
                collection=collection,
                chunks=chunks,
                embeddings=embeddings,
            )

            # 4. Embed the query and retrieve top-k
            query_embedding = self._embedder.embed(query)
            results = self._storage.search_chunks(
                collection=collection,
                query_embedding=query_embedding,
                top_k=top_k,
            )

            # 5. Build context from retrieved chunks
            context_chunks = [chunk_text for _, _, chunk_text in results]
            context = "\n\n---\n\n".join(context_chunks)

            # 6. Generate answer
            messages = [
                {
                    "role": "system",
                    "content": (
                        "You are a helpful assistant. Answer the user's question "
                        "based on the provided context passages. If the context "
                        "doesn't contain enough information, say so."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"Context:\n{context}\n\nQuestion: {query}"
                    ),
                },
            ]

            response: LLMResponseDTO = self._llm.complete(messages)
            elapsed = time.time() - start

            return RunResultDTO(
                answer=response.content,
                mode_used="rag",
                success=True,
                steps=1,
                input_tokens=response.input_tokens,
                output_tokens=response.output_tokens,
                elapsed_time=elapsed,
                trace=[{
                    "step": 0,
                    "role": "rag_retrieval",
                    "chunks_retrieved": len(results),
                    "scores": [score for score, _, _ in results],
                }],
                metadata={
                    "chunks_total": len(chunks),
                    "chunks_retrieved": len(results),
                    "collection": collection,
                },
            )
        except Exception as exc:
            elapsed = time.time() - start
            return RunResultDTO(
                answer="",
                mode_used="rag",
                success=False,
                error=str(exc),
                elapsed_time=elapsed,
            )

    @staticmethod
    def _chunk_text(text: str, chunk_size: int = 1000) -> List[str]:
        """Split text into chunks of approximately chunk_size characters."""
        chunks = []
        for i in range(0, len(text), chunk_size):
            chunk = text[i : i + chunk_size]
            if chunk.strip():
                chunks.append(chunk)
        return chunks

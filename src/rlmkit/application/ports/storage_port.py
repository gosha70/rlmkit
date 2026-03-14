"""Storage port: interface for persistence operations.

Any storage backend (SQLite, Postgres, in-memory, etc.) must implement
this Protocol to be usable by the application use cases.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class StoragePort(Protocol):
    """Protocol for conversation and data persistence adapters."""

    # -- Conversation CRUD --

    def create_conversation(
        self,
        name: str = "Untitled",
        mode: str = "compare",
        provider: str | None = None,
        model: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        """Create a new conversation and return its ID.

        Args:
            name: Display name for the conversation.
            mode: Default execution mode.
            provider: LLM provider name.
            model: Model identifier.
            metadata: Arbitrary metadata.

        Returns:
            Unique conversation identifier.
        """
        ...

    def get_conversation(self, conversation_id: str) -> dict[str, Any] | None:
        """Retrieve a conversation by ID.

        Args:
            conversation_id: Unique conversation identifier.

        Returns:
            Conversation data dict, or None if not found.
        """
        ...

    def list_conversations(self) -> list[dict[str, Any]]:
        """List all conversations ordered by most recently updated.

        Returns:
            List of conversation data dicts.
        """
        ...

    def delete_conversation(self, conversation_id: str) -> None:
        """Delete a conversation and all its messages.

        Args:
            conversation_id: Unique conversation identifier.
        """
        ...

    # -- File context --

    def save_file_context(
        self, content: str, filename: str | None = None
    ) -> str:
        """Store file content (deduplicated by hash).

        Args:
            content: File text content.
            filename: Original filename.

        Returns:
            Content hash serving as the file identifier.
        """
        ...

    def get_file_context(self, content_hash: str) -> str | None:
        """Retrieve file content by its hash.

        Args:
            content_hash: Hash returned by save_file_context.

        Returns:
            File content string, or None if not found.
        """
        ...

    # -- Vector operations --

    def add_chunks(
        self,
        collection: str,
        chunks: list[str],
        embeddings: list[list[float]],
        source_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> int:
        """Store text chunks with their embeddings.

        Args:
            collection: Collection name for grouping.
            chunks: List of text chunks.
            embeddings: Corresponding embedding vectors.
            source_id: Identifier for the source document.
            metadata: Arbitrary metadata for the chunks.

        Returns:
            Number of new chunks actually added.
        """
        ...

    def search_chunks(
        self,
        collection: str,
        query_embedding: list[float],
        top_k: int = 5,
    ) -> list[tuple[float, str, str]]:
        """Search for similar chunks by cosine similarity.

        Args:
            collection: Collection to search in.
            query_embedding: Query embedding vector.
            top_k: Maximum number of results.

        Returns:
            List of (score, chunk_id, chunk_text) tuples, highest first.
        """
        ...

"""SQLite storage adapter: wraps existing Database, ConversationStore,
and VectorStore to implement StoragePort.
"""

from __future__ import annotations

from typing import Any, cast

from rlmkit.storage.conversation_store import ConversationStore
from rlmkit.storage.database import Database
from rlmkit.storage.vector_store import VectorStore


class SQLiteStorageAdapter:
    """Adapter that wraps existing SQLite storage classes to satisfy
    :class:`StoragePort`.

    Args:
        db_path: Path to the SQLite database file. If None, uses the
            default path (~/.rlmkit/conversations.db).
    """

    def __init__(self, db_path: str | None = None) -> None:
        self._db = Database(db_path)
        self._conversations = ConversationStore(self._db)
        self._vectors = VectorStore(self._db)

    # -- Conversation CRUD --

    def create_conversation(
        self,
        name: str = "Untitled",
        mode: str = "compare",
        provider: str | None = None,
        model: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        """Create a new conversation and return its ID."""
        return cast(
            str,
            self._conversations.create_conversation(
                name=name,
                mode=mode,
                provider=provider,
                model=model,
                metadata=metadata,
            ),
        )

    def get_conversation(self, conversation_id: str) -> dict[str, Any] | None:
        """Retrieve a conversation by ID."""
        return cast("dict[str, Any] | None", self._conversations.get_conversation(conversation_id))

    def list_conversations(self) -> list[dict[str, Any]]:
        """List all conversations ordered by most recently updated."""
        return cast("list[dict[str, Any]]", self._conversations.list_conversations())

    def delete_conversation(self, conversation_id: str) -> None:
        """Delete a conversation and all its messages."""
        self._conversations.delete_conversation(conversation_id)

    # -- File context --

    def save_file_context(self, content: str, filename: str | None = None) -> str:
        """Store file content (deduplicated by hash)."""
        return cast(str, self._conversations.save_file_context(content, filename))

    def get_file_context(self, content_hash: str) -> str | None:
        """Retrieve file content by its hash."""
        return cast("str | None", self._conversations.get_file_context(content_hash))

    # -- Vector operations --

    def add_chunks(
        self,
        collection: str,
        chunks: list[str],
        embeddings: list[list[float]],
        source_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> int:
        """Store text chunks with their embeddings."""
        return cast(int, self._vectors.add_chunks(
            collection=collection,
            chunks=chunks,
            embeddings=embeddings,
            source_id=source_id,
            metadata=metadata,
        ))

    def search_chunks(
        self,
        collection: str,
        query_embedding: list[float],
        top_k: int = 5,
    ) -> list[tuple[float, str, str]]:
        """Search for similar chunks by cosine similarity."""
        return cast("list[tuple[float, str, str]]", self._vectors.search(
            collection=collection,
            query_embedding=query_embedding,
            top_k=top_k,
        ))

    def close(self) -> None:
        """Close the underlying database connection."""
        self._db.close()

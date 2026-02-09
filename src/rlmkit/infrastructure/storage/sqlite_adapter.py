"""SQLite storage adapter: wraps existing Database, ConversationStore,
and VectorStore to implement StoragePort.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from rlmkit.application.ports.storage_port import StoragePort
from rlmkit.storage.database import Database
from rlmkit.storage.conversation_store import ConversationStore
from rlmkit.storage.vector_store import VectorStore


class SQLiteStorageAdapter:
    """Adapter that wraps existing SQLite storage classes to satisfy
    :class:`StoragePort`.

    Args:
        db_path: Path to the SQLite database file. If None, uses the
            default path (~/.rlmkit/conversations.db).
    """

    def __init__(self, db_path: Optional[str] = None) -> None:
        self._db = Database(db_path)
        self._conversations = ConversationStore(self._db)
        self._vectors = VectorStore(self._db)

    # -- Conversation CRUD --

    def create_conversation(
        self,
        name: str = "Untitled",
        mode: str = "compare",
        provider: Optional[str] = None,
        model: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Create a new conversation and return its ID."""
        return self._conversations.create_conversation(
            name=name,
            mode=mode,
            provider=provider,
            model=model,
            metadata=metadata,
        )

    def get_conversation(self, conversation_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve a conversation by ID."""
        return self._conversations.get_conversation(conversation_id)

    def list_conversations(self) -> List[Dict[str, Any]]:
        """List all conversations ordered by most recently updated."""
        return self._conversations.list_conversations()

    def delete_conversation(self, conversation_id: str) -> None:
        """Delete a conversation and all its messages."""
        self._conversations.delete_conversation(conversation_id)

    # -- File context --

    def save_file_context(
        self, content: str, filename: Optional[str] = None
    ) -> str:
        """Store file content (deduplicated by hash)."""
        return self._conversations.save_file_context(content, filename)

    def get_file_context(self, content_hash: str) -> Optional[str]:
        """Retrieve file content by its hash."""
        return self._conversations.get_file_context(content_hash)

    # -- Vector operations --

    def add_chunks(
        self,
        collection: str,
        chunks: List[str],
        embeddings: List[List[float]],
        source_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> int:
        """Store text chunks with their embeddings."""
        return self._vectors.add_chunks(
            collection=collection,
            chunks=chunks,
            embeddings=embeddings,
            source_id=source_id,
            metadata=metadata,
        )

    def search_chunks(
        self,
        collection: str,
        query_embedding: List[float],
        top_k: int = 5,
    ) -> List[Tuple[float, str, str]]:
        """Search for similar chunks by cosine similarity."""
        return self._vectors.search(
            collection=collection,
            query_embedding=query_embedding,
            top_k=top_k,
        )

    def close(self) -> None:
        """Close the underlying database connection."""
        self._db.close()

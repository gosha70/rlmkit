"""Storage infrastructure adapters implementing StoragePort."""

from .sqlite_adapter import SQLiteStorageAdapter

__all__ = [
    "SQLiteStorageAdapter",
]

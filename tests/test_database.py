"""Tests for SQLite Database connection manager."""

import sqlite3
import pytest
from pathlib import Path

from rlmkit.storage.database import Database, get_default_db_path, _SCHEMA_VERSION


class TestGetDefaultDbPath:
    def test_returns_path_under_home(self):
        p = get_default_db_path()
        assert p.name == "conversations.db"
        assert ".rlmkit" in p.parts


class TestDatabase:
    def test_creates_db_file(self, tmp_path):
        db_path = tmp_path / "test.db"
        db = Database(db_path)
        assert db_path.exists()
        db.close()

    def test_creates_parent_dirs(self, tmp_path):
        db_path = tmp_path / "sub" / "dir" / "test.db"
        db = Database(db_path)
        assert db_path.exists()
        db.close()

    def test_wal_mode_enabled(self, tmp_path):
        db = Database(tmp_path / "test.db")
        row = db.fetchone("PRAGMA journal_mode")
        assert dict(row)["journal_mode"] == "wal"
        db.close()

    def test_foreign_keys_enabled(self, tmp_path):
        db = Database(tmp_path / "test.db")
        row = db.fetchone("PRAGMA foreign_keys")
        assert dict(row)["foreign_keys"] == 1
        db.close()

    def test_schema_version_recorded(self, tmp_path):
        db = Database(tmp_path / "test.db")
        row = db.fetchone("SELECT version FROM schema_version")
        assert row["version"] == _SCHEMA_VERSION
        db.close()

    def test_schema_version_idempotent(self, tmp_path):
        db_path = tmp_path / "test.db"
        db1 = Database(db_path)
        db1.close()
        # Re-open same DB — schema init should not fail
        db2 = Database(db_path)
        rows = db2.fetchall("SELECT version FROM schema_version")
        assert len(rows) == 1
        db2.close()

    def test_tables_created(self, tmp_path):
        db = Database(tmp_path / "test.db")
        rows = db.fetchall(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
        )
        names = {r["name"] for r in rows}
        assert "conversations" in names
        assert "messages" in names
        assert "file_contexts" in names
        assert "vector_chunks" in names
        assert "schema_version" in names
        db.close()

    def test_execute_and_fetch(self, tmp_path):
        db = Database(tmp_path / "test.db")
        db.execute(
            "INSERT INTO conversations (id, name, created_at, updated_at, mode) "
            "VALUES (?, ?, ?, ?, ?)",
            ("c1", "Test", "2024-01-01", "2024-01-01", "compare"),
        )
        row = db.fetchone("SELECT * FROM conversations WHERE id = ?", ("c1",))
        assert row["name"] == "Test"
        db.close()

    def test_fetchall(self, tmp_path):
        db = Database(tmp_path / "test.db")
        for i in range(3):
            db.execute(
                "INSERT INTO conversations (id, name, created_at, updated_at, mode) "
                "VALUES (?, ?, ?, ?, ?)",
                (f"c{i}", f"Conv {i}", "2024-01-01", "2024-01-01", "compare"),
            )
        rows = db.fetchall("SELECT * FROM conversations")
        assert len(rows) == 3
        db.close()

    def test_executemany(self, tmp_path):
        db = Database(tmp_path / "test.db")
        params = [
            (f"c{i}", f"Conv {i}", "2024-01-01", "2024-01-01", "compare")
            for i in range(5)
        ]
        db.executemany(
            "INSERT INTO conversations (id, name, created_at, updated_at, mode) "
            "VALUES (?, ?, ?, ?, ?)",
            params,
        )
        rows = db.fetchall("SELECT * FROM conversations")
        assert len(rows) == 5
        db.close()

    def test_fetchone_returns_none_for_missing(self, tmp_path):
        db = Database(tmp_path / "test.db")
        row = db.fetchone("SELECT * FROM conversations WHERE id = ?", ("nope",))
        assert row is None
        db.close()

    def test_close_and_reconnect(self, tmp_path):
        db_path = tmp_path / "test.db"
        db = Database(db_path)
        db.execute(
            "INSERT INTO conversations (id, name, created_at, updated_at, mode) "
            "VALUES (?, ?, ?, ?, ?)",
            ("c1", "Test", "2024-01-01", "2024-01-01", "compare"),
        )
        db.close()
        # Reconnect by calling connect again
        row = db.fetchone("SELECT * FROM conversations WHERE id = ?", ("c1",))
        assert row["name"] == "Test"
        db.close()

    def test_cascade_delete(self, tmp_path):
        db = Database(tmp_path / "test.db")
        db.execute(
            "INSERT INTO conversations (id, name, created_at, updated_at, mode) "
            "VALUES (?, ?, ?, ?, ?)",
            ("c1", "Test", "2024-01-01", "2024-01-01", "compare"),
        )
        db.execute(
            "INSERT INTO messages (id, conversation_id, sequence, timestamp, user_query, mode) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            ("m1", "c1", 0, "2024-01-01", "hello", "compare"),
        )
        # Delete conversation — messages should cascade
        db.execute("DELETE FROM conversations WHERE id = ?", ("c1",))
        msgs = db.fetchall("SELECT * FROM messages WHERE conversation_id = ?", ("c1",))
        assert len(msgs) == 0
        db.close()

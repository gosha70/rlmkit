"""SQLite-backed telemetry store for execution history and analytics.

Provides persistent, queryable storage for runs, steps, provider calls,
and ratings.  All writes are synchronous (<1 ms for single rows) and
wrapped in try/finally so telemetry failures never block execution.
"""

from __future__ import annotations

import json
import logging
import sqlite3
import threading
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

_SCHEMA_SQL = """\
CREATE TABLE IF NOT EXISTS runs (
    id              TEXT PRIMARY KEY,
    created_at      REAL NOT NULL,
    mode            TEXT NOT NULL,
    provider        TEXT NOT NULL DEFAULT '',
    model           TEXT NOT NULL DEFAULT '',
    query           TEXT NOT NULL DEFAULT '',
    content_length  INTEGER NOT NULL DEFAULT 0,
    answer_length   INTEGER NOT NULL DEFAULT 0,
    total_tokens    INTEGER NOT NULL DEFAULT 0,
    total_cost      REAL NOT NULL DEFAULT 0.0,
    elapsed_seconds REAL NOT NULL DEFAULT 0.0,
    success         INTEGER NOT NULL DEFAULT 1,
    error           TEXT,
    session_id      TEXT,
    chat_provider_id TEXT,
    chat_provider_name TEXT,
    comparison_group_id TEXT,
    answer          TEXT NOT NULL DEFAULT '',
    input_tokens    INTEGER NOT NULL DEFAULT 0,
    output_tokens   INTEGER NOT NULL DEFAULT 0,
    steps_count     INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_runs_created ON runs(created_at);
CREATE INDEX IF NOT EXISTS idx_runs_mode ON runs(mode);
CREATE INDEX IF NOT EXISTS idx_runs_session ON runs(session_id);

CREATE TABLE IF NOT EXISTS steps (
    id              TEXT PRIMARY KEY,
    run_id          TEXT NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
    step_index      INTEGER NOT NULL,
    action_type     TEXT NOT NULL DEFAULT 'inspect',
    code            TEXT,
    output          TEXT,
    tokens          INTEGER NOT NULL DEFAULT 0,
    cost            REAL NOT NULL DEFAULT 0.0,
    duration        REAL NOT NULL DEFAULT 0.0,
    model           TEXT,
    recursion_depth INTEGER NOT NULL DEFAULT 0,
    input_tokens    INTEGER NOT NULL DEFAULT 0,
    output_tokens   INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_steps_run ON steps(run_id, step_index);

CREATE TABLE IF NOT EXISTS provider_calls (
    id              TEXT PRIMARY KEY,
    run_id          TEXT NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
    step_index      INTEGER NOT NULL DEFAULT 0,
    provider        TEXT NOT NULL DEFAULT '',
    model           TEXT NOT NULL DEFAULT '',
    input_tokens    INTEGER NOT NULL DEFAULT 0,
    output_tokens   INTEGER NOT NULL DEFAULT 0,
    cost            REAL NOT NULL DEFAULT 0.0,
    latency_ms      INTEGER NOT NULL DEFAULT 0,
    status          TEXT NOT NULL DEFAULT 'ok',
    error           TEXT
);
CREATE INDEX IF NOT EXISTS idx_calls_run ON provider_calls(run_id);

CREATE TABLE IF NOT EXISTS ratings (
    id              TEXT PRIMARY KEY,
    run_id          TEXT NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
    rating          INTEGER NOT NULL,
    comment         TEXT,
    created_at      REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_ratings_run ON ratings(run_id);
"""


# ---------------------------------------------------------------------------
# Data classes for query results
# ---------------------------------------------------------------------------


@dataclass
class RunSummary:
    """Summary of a single run returned by list_runs()."""

    id: str
    created_at: float
    mode: str
    provider: str
    model: str
    query: str
    total_tokens: int
    total_cost: float
    elapsed_seconds: float
    success: bool
    error: str | None
    session_id: str | None
    chat_provider_id: str | None
    chat_provider_name: str | None
    steps_count: int
    answer_length: int


@dataclass
class RunDetail(RunSummary):
    """Full run detail including steps."""

    answer: str = ""
    input_tokens: int = 0
    output_tokens: int = 0
    content_length: int = 0
    steps: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class AggregateRow:
    """A single row from an aggregation query."""

    key: str
    run_count: int
    total_tokens: int
    total_cost: float
    avg_elapsed: float
    success_rate: float


# ---------------------------------------------------------------------------
# Store
# ---------------------------------------------------------------------------


class TelemetryStore:
    """SQLite-backed persistent store for execution telemetry.

    Args:
        db_path: Path to the SQLite database file.
            Defaults to ``~/.rlmkit/telemetry.db``.
            Use ``":memory:"`` for tests.
    """

    def __init__(self, db_path: str | Path | None = None) -> None:
        if db_path is None:
            db_path = Path.home() / ".rlmkit" / "telemetry.db"
        self._db_path = str(db_path)
        self._conn: sqlite3.Connection | None = None
        # Serializes all reads and writes so the store is safe to use from
        # multiple threads (e.g. FastAPI background tasks + WebSocket handlers).
        # SQLite's own locking is page-level; Python-level RLock ensures that
        # "execute + commit" sequences remain atomic.
        self._lock = threading.RLock()
        # Ensure the parent directory exists before sqlite3.connect, so first
        # runs on a clean machine don't fail with "unable to open database file".
        if self._db_path != ":memory:":
            Path(self._db_path).expanduser().parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    # ------------------------------------------------------------------
    # Connection management
    # ------------------------------------------------------------------

    def _connect(self) -> sqlite3.Connection:
        if self._conn is None:
            self._conn = sqlite3.connect(self._db_path, check_same_thread=False)
            self._conn.row_factory = sqlite3.Row
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute("PRAGMA foreign_keys=ON")
        return self._conn

    def _init_schema(self) -> None:
        with self._lock:
            conn = self._connect()
            conn.executescript(_SCHEMA_SQL)
            conn.commit()

    def close(self) -> None:
        """Close the database connection."""
        with self._lock:
            if self._conn is not None:
                self._conn.close()
                self._conn = None

    # ------------------------------------------------------------------
    # Write operations
    # ------------------------------------------------------------------

    def record_run(
        self,
        *,
        run_id: str | None = None,
        created_at: float,
        mode: str,
        provider: str = "",
        model: str = "",
        query: str = "",
        content_length: int = 0,
        answer: str = "",
        answer_length: int | None = None,
        input_tokens: int = 0,
        output_tokens: int = 0,
        total_tokens: int = 0,
        total_cost: float = 0.0,
        elapsed_seconds: float = 0.0,
        success: bool = True,
        error: str | None = None,
        session_id: str | None = None,
        chat_provider_id: str | None = None,
        chat_provider_name: str | None = None,
        comparison_group_id: str | None = None,
        steps_count: int = 0,
    ) -> str:
        """Record a completed run. Returns the run ID."""
        rid = run_id or uuid.uuid4().hex
        if answer_length is None:
            answer_length = len(answer)
        with self._lock:
            try:
                conn = self._connect()
                conn.execute(
                    """INSERT INTO runs
                       (id, created_at, mode, provider, model, query,
                        content_length, answer, answer_length, input_tokens,
                        output_tokens, total_tokens, total_cost, elapsed_seconds,
                        success, error, session_id, chat_provider_id,
                        chat_provider_name, comparison_group_id, steps_count)
                       VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                    (
                        rid,
                        created_at,
                        mode,
                        provider,
                        model,
                        query,
                        content_length,
                        answer,
                        answer_length,
                        input_tokens,
                        output_tokens,
                        total_tokens,
                        total_cost,
                        elapsed_seconds,
                        int(success),
                        error,
                        session_id,
                        chat_provider_id,
                        chat_provider_name,
                        comparison_group_id,
                        steps_count,
                    ),
                )
                conn.commit()
            except Exception:
                logger.exception("Failed to record run %s", rid)
        return rid

    def record_step(
        self,
        *,
        run_id: str,
        step_index: int,
        action_type: str = "inspect",
        code: str | None = None,
        output: str | None = None,
        tokens: int = 0,
        cost: float = 0.0,
        duration: float = 0.0,
        model: str | None = None,
        recursion_depth: int = 0,
        input_tokens: int = 0,
        output_tokens: int = 0,
    ) -> str:
        """Record a single step within a run. Returns the step ID."""
        sid = uuid.uuid4().hex
        with self._lock:
            try:
                conn = self._connect()
                conn.execute(
                    """INSERT INTO steps
                       (id, run_id, step_index, action_type, code, output,
                        tokens, cost, duration, model, recursion_depth,
                        input_tokens, output_tokens)
                       VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                    (
                        sid,
                        run_id,
                        step_index,
                        action_type,
                        code,
                        output,
                        tokens,
                        cost,
                        duration,
                        model,
                        recursion_depth,
                        input_tokens,
                        output_tokens,
                    ),
                )
                conn.commit()
            except Exception:
                logger.exception("Failed to record step %d for run %s", step_index, run_id)
        return sid

    def record_call(
        self,
        *,
        run_id: str,
        step_index: int = 0,
        provider: str = "",
        model: str = "",
        input_tokens: int = 0,
        output_tokens: int = 0,
        cost: float = 0.0,
        latency_ms: int = 0,
        status: str = "ok",
        error: str | None = None,
    ) -> str:
        """Record a provider API call. Returns the call ID."""
        cid = uuid.uuid4().hex
        with self._lock:
            try:
                conn = self._connect()
                conn.execute(
                    """INSERT INTO provider_calls
                       (id, run_id, step_index, provider, model,
                        input_tokens, output_tokens, cost, latency_ms,
                        status, error)
                       VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
                    (
                        cid,
                        run_id,
                        step_index,
                        provider,
                        model,
                        input_tokens,
                        output_tokens,
                        cost,
                        latency_ms,
                        status,
                        error,
                    ),
                )
                conn.commit()
            except Exception:
                logger.exception("Failed to record call for run %s step %d", run_id, step_index)
        return cid

    def add_rating(
        self,
        *,
        run_id: str,
        rating: int,
        comment: str | None = None,
        created_at: float,
    ) -> str:
        """Add a user rating for a run. Returns the rating ID."""
        rid = uuid.uuid4().hex
        with self._lock:
            try:
                conn = self._connect()
                conn.execute(
                    """INSERT INTO ratings (id, run_id, rating, comment, created_at)
                       VALUES (?,?,?,?,?)""",
                    (rid, run_id, rating, comment, created_at),
                )
                conn.commit()
            except Exception:
                logger.exception("Failed to add rating for run %s", run_id)
        return rid

    # ------------------------------------------------------------------
    # Read operations
    # ------------------------------------------------------------------

    def list_runs(
        self,
        *,
        limit: int = 50,
        offset: int = 0,
        mode: str | None = None,
        session_id: str | None = None,
        chat_provider_id: str | None = None,
        success: bool | None = None,
    ) -> list[RunSummary]:
        """List runs ordered by created_at descending."""
        clauses: list[str] = []
        params: list[Any] = []
        if mode is not None:
            clauses.append("mode = ?")
            params.append(mode)
        if session_id is not None:
            clauses.append("session_id = ?")
            params.append(session_id)
        if chat_provider_id is not None:
            clauses.append("chat_provider_id = ?")
            params.append(chat_provider_id)
        if success is not None:
            clauses.append("success = ?")
            params.append(int(success))

        where = (" WHERE " + " AND ".join(clauses)) if clauses else ""
        params.extend([limit, offset])

        with self._lock:
            rows = (
                self._connect()
                .execute(
                    f"SELECT * FROM runs{where} ORDER BY created_at DESC LIMIT ? OFFSET ?",  # nosec B608
                    tuple(params),
                )
                .fetchall()
            )

        return [self._row_to_summary(r) for r in rows]

    def get_run(self, run_id: str) -> RunDetail | None:
        """Get full detail for a single run including steps."""
        with self._lock:
            row = self._connect().execute("SELECT * FROM runs WHERE id = ?", (run_id,)).fetchone()
            if row is None:
                return None

            step_rows = (
                self._connect()
                .execute(
                    "SELECT * FROM steps WHERE run_id = ? ORDER BY step_index",
                    (run_id,),
                )
                .fetchall()
            )

        detail = RunDetail(
            id=row["id"],
            created_at=row["created_at"],
            mode=row["mode"],
            provider=row["provider"],
            model=row["model"],
            query=row["query"],
            total_tokens=row["total_tokens"],
            total_cost=row["total_cost"],
            elapsed_seconds=row["elapsed_seconds"],
            success=bool(row["success"]),
            error=row["error"],
            session_id=row["session_id"],
            chat_provider_id=row["chat_provider_id"],
            chat_provider_name=row["chat_provider_name"],
            steps_count=row["steps_count"],
            answer_length=row["answer_length"],
            answer=row["answer"],
            input_tokens=row["input_tokens"],
            output_tokens=row["output_tokens"],
            content_length=row["content_length"],
            steps=[dict(r) for r in step_rows],
        )
        return detail

    def count_runs(self) -> int:
        """Return total number of runs."""
        with self._lock:
            row = self._connect().execute("SELECT COUNT(*) AS cnt FROM runs").fetchone()
        return int(row["cnt"]) if row else 0

    # ------------------------------------------------------------------
    # Aggregation queries
    # ------------------------------------------------------------------

    def aggregate_by_mode(self) -> list[AggregateRow]:
        """Aggregate run metrics grouped by mode."""
        return self._aggregate("mode")

    def aggregate_by_provider(self) -> list[AggregateRow]:
        """Aggregate run metrics grouped by provider."""
        return self._aggregate("provider")

    def _aggregate(self, group_col: str) -> list[AggregateRow]:
        if group_col not in ("mode", "provider", "model"):
            raise ValueError(f"Invalid group column: {group_col!r}")
        with self._lock:
            rows = (
                self._connect()
                .execute(
                    f"""SELECT {group_col} AS key,
                           COUNT(*) AS run_count,
                           SUM(total_tokens) AS total_tokens,
                           SUM(total_cost) AS total_cost,
                           AVG(elapsed_seconds) AS avg_elapsed,
                           AVG(success) AS success_rate
                    FROM runs
                    GROUP BY {group_col}
                    ORDER BY run_count DESC""",  # nosec B608
                )
                .fetchall()
            )
        return [
            AggregateRow(
                key=r["key"],
                run_count=r["run_count"],
                total_tokens=r["total_tokens"] or 0,
                total_cost=r["total_cost"] or 0.0,
                avg_elapsed=r["avg_elapsed"] or 0.0,
                success_rate=r["success_rate"] or 0.0,
            )
            for r in rows
        ]

    # ------------------------------------------------------------------
    # JSONL export
    # ------------------------------------------------------------------

    def export_jsonl(self, run_id: str) -> str | None:
        """Export a run as JSONL string (compatible with ExecutionTrace format).

        Returns None if the run is not found.
        """
        detail = self.get_run(run_id)
        if detail is None:
            return None

        lines: list[str] = []
        # Metadata line (matches ExecutionTrace.to_jsonl format)
        metadata = {
            "metadata": {
                "start_time": detail.created_at,
                "end_time": detail.created_at + detail.elapsed_seconds,
                "total_duration": detail.elapsed_seconds,
                "total_tokens": detail.total_tokens,
                "total_cost": detail.total_cost,
                "total_steps": detail.steps_count,
                "mode": detail.mode,
                "provider": detail.provider,
                "model": detail.model,
                "query": detail.query,
                "success": detail.success,
            }
        }
        lines.append(json.dumps(metadata))

        # Step lines
        for step in detail.steps:
            step_out: dict[str, Any] = {
                "index": step.get("step_index", 0),
                "action_type": step.get("action_type", "inspect"),
                "code": step.get("code"),
                "output": step.get("output"),
                "tokens_used": step.get("tokens", 0),
                "cost": step.get("cost", 0.0),
                "duration": step.get("duration", 0.0),
                "model": step.get("model"),
                "recursion_depth": step.get("recursion_depth", 0),
            }
            lines.append(json.dumps(step_out))

        return "\n".join(lines) + "\n"

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def delete_run(self, run_id: str) -> None:
        """Delete a run and all associated steps, calls, and ratings."""
        with self._lock:
            try:
                conn = self._connect()
                conn.execute("DELETE FROM runs WHERE id = ?", (run_id,))
                conn.commit()
            except Exception:
                logger.exception("Failed to delete run %s", run_id)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    @staticmethod
    def _row_to_summary(row: sqlite3.Row) -> RunSummary:
        return RunSummary(
            id=row["id"],
            created_at=row["created_at"],
            mode=row["mode"],
            provider=row["provider"],
            model=row["model"],
            query=row["query"],
            total_tokens=row["total_tokens"],
            total_cost=row["total_cost"],
            elapsed_seconds=row["elapsed_seconds"],
            success=bool(row["success"]),
            error=row["error"],
            session_id=row["session_id"],
            chat_provider_id=row["chat_provider_id"],
            chat_provider_name=row["chat_provider_name"],
            steps_count=row["steps_count"],
            answer_length=row["answer_length"],
        )

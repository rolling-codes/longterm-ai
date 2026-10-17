"""Optional SQLite-backed checkpoint store. Phase 2 — ADR 0002.

Drop-in replacement for CheckpointStore: identical save/load/delete interface.
Requires only Python's built-in sqlite3 module (stdlib). Zero extra dependencies.

Why SQLite: atomic writes via transactions, WAL mode for safe concurrent readers,
durable across process restarts, inspectable with standard tooling, and still
zero runtime dependencies.
"""
from __future__ import annotations

import json
import sqlite3
import time
from pathlib import Path
from typing import Any

from .errors import UnsupportedSchemaVersion
from .runtime import CHECKPOINT_SCHEMA_VERSION, Checkpoint

_CREATE = """
CREATE TABLE IF NOT EXISTS checkpoints (
    task_id         TEXT    PRIMARY KEY,
    status          TEXT    NOT NULL,
    schema_version  INTEGER NOT NULL,
    completed_steps INTEGER NOT NULL DEFAULT 0,
    criteria_status TEXT    NOT NULL DEFAULT '{}',
    tool_log        TEXT    NOT NULL DEFAULT '[]',
    remaining_budget INTEGER,
    deadline_s      REAL,
    saved_at        REAL    NOT NULL
)
"""


class SqliteCheckpointStore:
    """SQLite-backed checkpoint store with the same interface as CheckpointStore.

    Pass *db_path* as a ``Path`` or string; the parent directory is created
    automatically. WAL journal mode is enabled on every connection for safe
    concurrent readers.
    """

    def __init__(self, db_path: str | Path) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._con = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self._con.execute("PRAGMA journal_mode=WAL")
        self._con.execute(_CREATE)
        self._con.commit()

    # ------------------------------------------------------------------
    def save(self, cp: Checkpoint) -> None:
        """Upsert a checkpoint. Serialises list/dict fields to JSON."""
        self._con.execute(
            """INSERT OR REPLACE INTO checkpoints
               (task_id, status, schema_version, completed_steps,
                criteria_status, tool_log, remaining_budget, deadline_s, saved_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                cp.task_id,
                cp.status,
                cp.schema_version,
                cp.completed_steps,
                json.dumps(cp.criteria_status, ensure_ascii=False),
                json.dumps(cp.tool_log, ensure_ascii=False),
                cp.remaining_budget,
                cp.deadline_s,
                time.time(),
            ),
        )
        self._con.commit()

    def load(self, task_id: str) -> Checkpoint | None:
        """Return the saved Checkpoint or None if not found.

        Raises :class:`~longterm_ai.errors.UnsupportedSchemaVersion` if the
        stored schema version does not match the current one.
        """
        row = self._con.execute(
            "SELECT status, schema_version, completed_steps, criteria_status, "
            "tool_log, remaining_budget, deadline_s FROM checkpoints WHERE task_id=?",
            (task_id,),
        ).fetchone()
        if row is None:
            return None
        status, schema_version, completed_steps, criteria_status_json, tool_log_json, remaining_budget, deadline_s = row
        if int(schema_version) != CHECKPOINT_SCHEMA_VERSION:
            raise UnsupportedSchemaVersion(
                f"checkpoint schema {schema_version} != {CHECKPOINT_SCHEMA_VERSION}"
            )
        return Checkpoint(
            task_id=task_id,
            status=status,
            schema_version=int(schema_version),
            completed_steps=int(completed_steps),
            criteria_status=json.loads(criteria_status_json),
            tool_log=json.loads(tool_log_json),
            remaining_budget=int(remaining_budget) if remaining_budget is not None else None,
            deadline_s=float(deadline_s) if deadline_s is not None else None,
        )

    def delete(self, task_id: str) -> None:
        """Remove the checkpoint for *task_id* (no-op if absent)."""
        self._con.execute("DELETE FROM checkpoints WHERE task_id=?", (task_id,))
        self._con.commit()

    def list_tasks(self) -> list[str]:
        """Return all task_ids currently stored."""
        rows = self._con.execute("SELECT task_id FROM checkpoints ORDER BY task_id").fetchall()
        return [r[0] for r in rows]

    def close(self) -> None:
        """Close the underlying connection explicitly (optional; GC will also close it)."""
        self._con.close()

    def __enter__(self) -> "SqliteCheckpointStore":
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.close()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass

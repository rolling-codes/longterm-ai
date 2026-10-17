"""Tests for SqliteCheckpointStore. Phase 2 / ADR 0002.

No external packages. stdlib sqlite3 only.
"""
import json
import tempfile
import unittest
from pathlib import Path

import sys
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "src"))

from longterm_ai import Checkpoint, SqliteCheckpointStore
from longterm_ai.errors import UnsupportedSchemaVersion
from longterm_ai.runtime import CHECKPOINT_SCHEMA_VERSION


def _store(tmp: str) -> SqliteCheckpointStore:
    return SqliteCheckpointStore(Path(tmp) / "ckpt.db")


def _cp(task_id: str = "t1", status: str = "running", steps: int = 3) -> Checkpoint:
    return Checkpoint(task_id=task_id, status=status, completed_steps=steps)


class SqliteStoreTests(unittest.TestCase):

    def test_save_and_load_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmp:
            with _store(tmp) as store:
                cp = _cp()
                store.save(cp)
                loaded = store.load("t1")
                self.assertIsNotNone(loaded)
                assert loaded is not None
                self.assertEqual(loaded.task_id, "t1")
                self.assertEqual(loaded.status, "running")
                self.assertEqual(loaded.completed_steps, 3)
                self.assertEqual(loaded.schema_version, CHECKPOINT_SCHEMA_VERSION)

    def test_load_missing_returns_none(self):
        with tempfile.TemporaryDirectory() as tmp:
            with _store(tmp) as store:
                self.assertIsNone(store.load("ghost"))

    def test_delete_removes_entry(self):
        with tempfile.TemporaryDirectory() as tmp:
            with _store(tmp) as store:
                store.save(_cp())
                store.delete("t1")
                self.assertIsNone(store.load("t1"))

    def test_delete_missing_is_noop(self):
        with tempfile.TemporaryDirectory() as tmp:
            with _store(tmp) as store:
                store.delete("nonexistent")   # must not raise

    def test_list_tasks(self):
        with tempfile.TemporaryDirectory() as tmp:
            with _store(tmp) as store:
                store.save(_cp("alpha"))
                store.save(_cp("beta"))
                tasks = store.list_tasks()
                self.assertIn("alpha", tasks)
                self.assertIn("beta", tasks)
                self.assertEqual(len(tasks), 2)

    def test_unknown_schema_version_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            db_path = Path(tmp) / "ckpt.db"
            with SqliteCheckpointStore(db_path) as store:
                # Manually insert a row with an unsupported schema version.
                store._con.execute(
                    "INSERT INTO checkpoints "
                    "(task_id, status, schema_version, completed_steps, criteria_status, tool_log, saved_at) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?)",
                    ("bad_ver", "running", 99, 0, "{}", "[]", 0.0),
                )
                store._con.commit()
                with self.assertRaises(UnsupportedSchemaVersion):
                    store.load("bad_ver")

    def test_save_overwrites(self):
        with tempfile.TemporaryDirectory() as tmp:
            with _store(tmp) as store:
                store.save(_cp("t1", "running", 1))
                store.save(_cp("t1", "completed", 5))
                loaded = store.load("t1")
                assert loaded is not None
                self.assertEqual(loaded.status, "completed")
                self.assertEqual(loaded.completed_steps, 5)

    def test_tool_log_survives_roundtrip(self):
        from longterm_ai import OutcomeStatus, RiskClass, ToolLog, ToolLogEntry
        with tempfile.TemporaryDirectory() as tmp:
            with _store(tmp) as store:
                entry = ToolLogEntry(
                    op_id="op1", tool="some_tool", args_digest="d1",
                    risk=RiskClass.READ_ONLY, idempotent=True,
                    result_status=OutcomeStatus.CONFIRMED_SUCCEEDED,
                    dispatched_at=1.0, finished_at=2.0, evidence="ok",
                )
                log = ToolLog()
                log.append(entry)
                cp = Checkpoint(task_id="log_t", status="running",
                                completed_steps=1, tool_log=log.to_list())
                store.save(cp)
                loaded = store.load("log_t")
                assert loaded is not None
                self.assertEqual(len(loaded.tool_log), 1)
                self.assertEqual(loaded.tool_log[0]["tool"], "some_tool")

    def test_nullable_fields_roundtrip(self):
        """remaining_budget and deadline_s survive as None after a roundtrip."""
        with tempfile.TemporaryDirectory() as tmp:
            with _store(tmp) as store:
                cp = Checkpoint(task_id="n", status="running",
                                remaining_budget=None, deadline_s=None)
                store.save(cp)
                loaded = store.load("n")
                assert loaded is not None
                self.assertIsNone(loaded.remaining_budget)
                self.assertIsNone(loaded.deadline_s)

    def test_nullable_fields_with_values(self):
        """remaining_budget and deadline_s survive with real values."""
        with tempfile.TemporaryDirectory() as tmp:
            with _store(tmp) as store:
                cp = Checkpoint(task_id="v", status="running",
                                remaining_budget=500, deadline_s=3600.0)
                store.save(cp)
                loaded = store.load("v")
                assert loaded is not None
                self.assertEqual(loaded.remaining_budget, 500)
                assert loaded.deadline_s is not None
                self.assertAlmostEqual(loaded.deadline_s, 3600.0)


if __name__ == "__main__":
    unittest.main()

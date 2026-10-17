"""Tests for the longterm-ai CLI (`python -m longterm_ai`). Phase 2 / ADR 0002."""
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
SRC = HERE.parent / "src"
EX = HERE / "handoff_examples"


def run_cli(*args: str, cwd: Path | None = None) -> subprocess.CompletedProcess:
    env = dict(os.environ)
    env["PYTHONPATH"] = str(SRC) + os.pathsep + env.get("PYTHONPATH", "")
    return subprocess.run(
        [sys.executable, "-m", "longterm_ai", *args],
        cwd=cwd or EX,
        capture_output=True,
        text=True,
        env=env,
    )


class CLIVerifyTests(unittest.TestCase):

    def test_verify_passes_on_good_record(self):
        p = run_cli("verify", "task.json", "record.json", "state")
        self.assertEqual(p.returncode, 0, p.stderr)
        self.assertIn("layer one: PASSED", p.stdout)

    def test_verify_fails_on_bad_record(self):
        p = run_cli("verify", "task.json", "record_bad.json", "state")
        self.assertEqual(p.returncode, 1)
        self.assertIn("FAIL:", p.stderr)


class CLIRebuildTests(unittest.TestCase):

    def test_rebuild_good_record_anthropic(self):
        p = run_cli(
            "rebuild", "task.json", "transcript.json", "record.json", "state",
            "--format", "anthropic",
        )
        self.assertEqual(p.returncode, 0, p.stderr)
        data = json.loads(p.stdout)
        self.assertIn("max_tokens", data)

    def test_rebuild_refuses_failed_record(self):
        p = run_cli(
            "rebuild", "task.json", "transcript.json", "record_injected.json", "state",
            "--format", "anthropic",
        )
        self.assertEqual(p.returncode, 1)
        self.assertIn("layer one failed", p.stderr)


class CLIRunTests(unittest.TestCase):

    def test_run_command_success(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            task_file = tmp_path / "test_task.json"
            task_file.write_text(
                json.dumps({
                    "objective": "Verify run CLI.",
                    "mandatory_constraints": [],
                    "acceptance_criteria": [],
                    "max_steps": 2,
                }),
                encoding="utf-8",
            )
            state_dir = tmp_path / "state"
            p = run_cli(
                "run", str(task_file),
                "--state-dir", str(state_dir),
                "--task-id", "cli_test",
                cwd=tmp_path,
            )
            self.assertEqual(p.returncode, 0, p.stderr)
            data = json.loads(p.stdout)
            self.assertIn("status", data)
            self.assertIn("steps", data)
            self.assertIn("tool_log", data)


class CLIReviewTests(unittest.TestCase):

    def test_review_unsafe_when_claim_unbacked(self):
        # record.json claims a side effect; with no tool log it is unbacked -> UNSAFE.
        """A side-effect claim without an authoritative log produces an UNSAFE verdict."""
        p = run_cli("review", "task.json", "record.json", "state")
        self.assertEqual(p.returncode, 1)
        self.assertIn("UNSAFE", p.stdout)
        self.assertIn("client/fetch.py, retry loop added", p.stdout)   # the unbacked claim
        self.assertIn("claim accuracy   : 0.000", p.stdout)

    def test_review_safe_when_claim_backed_by_log(self):
        """A matching tool log clears the handoff and reports full claim accuracy."""
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            rec = json.loads((EX / "record.json").read_text(encoding="utf-8"))
            rec["changed_state"] = ["edit_file: client/fetch.py edited"]
            rec_file = tmp_path / "rec.json"
            rec_file.write_text(json.dumps(rec), encoding="utf-8")
            tl_file = tmp_path / "tl.json"
            tl_file.write_text(json.dumps([{
                "op_id": "o1", "tool": "edit_file", "args_digest": "d",
                "risk": "reversible_local", "idempotent": True,
                "result_status": "confirmed_succeeded", "dispatched_at": 0.0,
                "finished_at": 0.0, "evidence": "client/fetch.py edited",
            }]), encoding="utf-8")
            p = run_cli("review", "task.json", str(rec_file), "state",
                        "--tool-log", str(tl_file), "--json")
            self.assertEqual(p.returncode, 0, p.stderr)
            data = json.loads(p.stdout)
            self.assertTrue(data["safe_to_hand_off"])
            self.assertTrue(data["layer_one_passed"])
            self.assertEqual(data["claim_accuracy"], 1.0)
            self.assertEqual(data["unbacked_claims"], [])

    def test_review_json_has_expected_keys(self):
        """JSON review output includes verification and claim reconciliation fields."""
        p = run_cli("review", "task.json", "record.json", "state", "--json")
        data = json.loads(p.stdout)
        for key in ("layer_one_passed", "claim_accuracy", "unbacked_claims",
                    "unclaimed_in_log", "safe_to_hand_off"):
            self.assertIn(key, data)

    def test_run_then_review_pipeline(self):
        # run writes <state>/<task_id>.tool_log.json; review reads it by task-id.
        """A run persists a tool log that review accepts for a record with no side-effect claims."""
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            task_file = tmp_path / "t.json"
            task_file.write_text(json.dumps({
                "objective": "pipeline", "mandatory_constraints": [],
                "acceptance_criteria": [], "max_steps": 2,
            }), encoding="utf-8")
            state_dir = tmp_path / "state"
            p_run = run_cli("run", str(task_file), "--state-dir", str(state_dir),
                            "--task-id", "p1", cwd=tmp_path)
            self.assertEqual(p_run.returncode, 0, p_run.stderr)
            self.assertIn("tool_log_path", json.loads(p_run.stdout))
            self.assertTrue((state_dir / "p1.tool_log.json").exists())
            # A clean record (no side-effect claims) reviewed against that log -> SAFE.
            rec = json.loads((EX / "record.json").read_text(encoding="utf-8"))
            rec["changed_state"] = []
            rec_file = tmp_path / "rec.json"
            rec_file.write_text(json.dumps(rec), encoding="utf-8")
            p_rev = run_cli("review", "task.json", str(rec_file), "state",
                            "--task-id", "p1", "--tool-log", str(state_dir / "p1.tool_log.json"),
                            "--json")
            data = json.loads(p_rev.stdout)
            self.assertTrue(data["safe_to_hand_off"])
            self.assertIsNone(data["claim_accuracy"])   # no claims -> None, never 0

    # --- operational errors must exit 2 (fail closed), never crash or look like a verdict ---
    def test_review_missing_task_file_exits_2(self):
        """A missing task stops review with operational exit 2 and no traceback."""
        p = run_cli("review", "does_not_exist.json", "record.json", "state")
        self.assertEqual(p.returncode, 2, p.stderr)
        self.assertIn("could not run", p.stderr)
        self.assertNotIn("Traceback", p.stderr)

    def test_review_malformed_tool_log_exits_2(self):
        """Invalid tool-log JSON stops review with operational exit 2 and no traceback."""
        with tempfile.TemporaryDirectory() as tmp:
            bad = Path(tmp) / "bad.json"
            bad.write_text("{not json", encoding="utf-8")
            p = run_cli("review", "task.json", "record.json", "state", "--tool-log", str(bad))
            self.assertEqual(p.returncode, 2, p.stderr)
            self.assertNotIn("Traceback", p.stderr)

    def test_review_tool_log_not_a_list_exits_2(self):
        """Review rejects a tool-log object instead of treating it as a list of entries."""
        with tempfile.TemporaryDirectory() as tmp:
            notlist = Path(tmp) / "obj.json"
            notlist.write_text('{"entries": []}', encoding="utf-8")
            p = run_cli("review", "task.json", "record.json", "state", "--tool-log", str(notlist))
            self.assertEqual(p.returncode, 2, p.stderr)
            self.assertIn("list", p.stderr)

    def test_review_unsupported_record_exits_2(self):
        """An unsupported record schema produces operational exit 2 without a traceback."""
        with tempfile.TemporaryDirectory() as tmp:
            rec = json.loads((EX / "record.json").read_text(encoding="utf-8"))
            rec["version"] = "999"          # unknown schema version -> UnsupportedRecord
            bad_rec = Path(tmp) / "rec.json"
            bad_rec.write_text(json.dumps(rec), encoding="utf-8")
            p = run_cli("review", "task.json", str(bad_rec), "state")
            self.assertEqual(p.returncode, 2, p.stderr)
            self.assertNotIn("Traceback", p.stderr)


class CLIOperationalErrorTests(unittest.TestCase):

    def test_verify_missing_task_file_exits_2(self):
        """Verification reports a missing task as operational exit 2 without a traceback."""
        p = run_cli("verify", "nope.json", "record.json", "state")
        self.assertEqual(p.returncode, 2, p.stderr)
        self.assertIn("could not run", p.stderr)
        self.assertNotIn("Traceback", p.stderr)

    def test_run_malformed_criteria_exits_2(self):
        """A criterion missing its kind produces operational exit 2 without a traceback."""
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            task_file = tmp_path / "t.json"
            # acceptance_criteria entry missing required 'kind' -> operational, not a crash.
            task_file.write_text(json.dumps({
                "objective": "x", "acceptance_criteria": [{"id": "c1"}], "max_steps": 1,
            }), encoding="utf-8")
            p = run_cli("run", str(task_file), "--state-dir", str(tmp_path / "s"),
                        cwd=tmp_path)
            self.assertEqual(p.returncode, 2, p.stderr)
            self.assertNotIn("Traceback", p.stderr)


if __name__ == "__main__":
    unittest.main()

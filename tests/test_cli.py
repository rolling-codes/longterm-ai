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


if __name__ == "__main__":
    unittest.main()

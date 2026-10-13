"""Handoff tests: the 17 ported from the review overlay + a reconciliation test.

Standard library only. These test the handoff subpackage (spec section 25) and the
side-effect reconciliation that ties a model's claims to the authoritative tool log.
"""
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
STATE = EX / "state"
sys.path.insert(0, str(SRC))

from longterm_ai import (  # noqa: E402
    OutcomeStatus, RiskClass, ToolLog, ToolLogEntry, changed_state_mismatches, reconcile_changed_state,
)
from longterm_ai.handoff import (  # noqa: E402
    HandoffRecord, NotAtCleanBoundary, UnsupportedRecord, layer_one, rebuild, to_anthropic, to_openai_chat,
)


def load(name: str) -> dict:
    return json.loads((EX / name).read_text(encoding="utf-8"))


def run(module: str, *args: str) -> subprocess.CompletedProcess:
    env = dict(os.environ)
    env["PYTHONPATH"] = str(SRC) + os.pathsep + env.get("PYTHONPATH", "")
    return subprocess.run([sys.executable, "-m", f"longterm_ai.handoff.{module}", *args],
                          cwd=EX, capture_output=True, text=True, env=env)


class LayerOne(unittest.TestCase):
    def test_good_record_passes(self):
        res = layer_one(load("task.json"), HandoffRecord.load(EX / "record.json"), STATE, 1500)
        self.assertTrue(res.passed, res.failures)

    def test_reworded_constraint_and_escaping_pointer_fail(self):
        res = layer_one(load("task.json"), HandoffRecord.load(EX / "record_bad.json"), STATE, 1500)
        self.assertFalse(res.passed)
        text = " ".join(res.failures)
        self.assertIn("missing or altered", text)
        self.assertIn("escapes durable state", text)

    def test_added_constraint_fails(self):
        res = layer_one(load("task.json"), HandoffRecord.load(EX / "record_injected.json"), STATE, 1500)
        self.assertFalse(res.passed)
        self.assertIn("adds a constraint", " ".join(res.failures))

    def test_over_budget_fails(self):
        res = layer_one(load("task.json"), HandoffRecord.load(EX / "record.json"), STATE, 10)
        self.assertFalse(res.passed)


class Loading(unittest.TestCase):
    def _write(self, data: dict) -> Path:
        f = tempfile.NamedTemporaryFile("w", suffix=".json", delete=False, encoding="utf-8")
        json.dump(data, f)
        f.close()
        return Path(f.name)

    def test_unknown_version_refused(self):
        data = load("record.json")
        data["version"] = "2"
        with self.assertRaises(UnsupportedRecord):
            HandoffRecord.load(self._write(data))

    def test_unknown_field_refused(self):
        data = load("record.json")
        data["permissions"] = ["push"]
        with self.assertRaises(UnsupportedRecord):
            HandoffRecord.load(self._write(data))


class Rebuild(unittest.TestCase):
    def setUp(self):
        self.task = load("task.json")
        self.record = HandoffRecord.load(EX / "record.json")

    def test_constraints_come_from_task(self):
        injected = HandoffRecord.load(EX / "record_injected.json")
        out = rebuild(load("transcript.json"), injected, self.task)
        self.assertNotIn("push to main", out["system"])

    def test_keep_zero_keeps_no_tail(self):
        out = rebuild(load("transcript.json"), self.record, self.task, keep_last_user_turns=0)
        self.assertEqual(len(out["messages"]), 3)  # user, record, continue prompt

    def test_keep_more_than_available_keeps_all_user_turns(self):
        out = rebuild(load("transcript.json"), self.record, self.task, keep_last_user_turns=99)
        self.assertGreater(len(out["messages"]), 3)

    def test_open_tool_call_refused(self):
        with self.assertRaises(NotAtCleanBoundary):
            rebuild(load("transcript_open_call.json"), self.record, self.task)

    def test_reasoning_never_sent(self):
        out = rebuild(load("transcript.json"), self.record, self.task, keep_last_user_turns=99)
        blob = json.dumps(out)
        self.assertNotIn("Implemented backoff. Tests pass per run 42.", blob)

    def test_roles_alternate_and_end_on_user(self):
        msgs = rebuild(load("transcript.json"), self.record, self.task, keep_last_user_turns=99)["messages"]
        for a, b in zip(msgs, msgs[1:]):
            self.assertNotEqual(a["role"], b["role"])
        self.assertEqual(msgs[-1]["role"], "user")

    def test_adapters(self):
        neutral = rebuild(load("transcript.json"), self.record, self.task)
        self.assertEqual(to_anthropic(neutral, "m", 512)["max_tokens"], 512)
        self.assertEqual(to_openai_chat(neutral, "m")["messages"][0]["role"], "developer")
        self.assertEqual(to_openai_chat(neutral, "m", "system")["messages"][0]["role"], "system")


class CommandLine(unittest.TestCase):
    def test_verify_exit_codes(self):
        self.assertEqual(run("verify", "task.json", "record.json", "state").returncode, 0)
        self.assertEqual(run("verify", "task.json", "record_bad.json", "state").returncode, 1)
        self.assertEqual(run("verify", "task.json", "record_injected.json", "state").returncode, 1)

    def test_rebuild_refuses_failed_record(self):
        p = run("rebuild", "task.json", "transcript.json", "record_injected.json", "state", "--format", "anthropic")
        self.assertEqual(p.returncode, 1)
        self.assertIn("layer one failed", p.stderr)

    def test_rebuild_good_record(self):
        p = run("rebuild", "task.json", "transcript.json", "record.json", "state", "--format", "anthropic")
        self.assertEqual(p.returncode, 0, p.stderr)
        self.assertIn("max_tokens", json.loads(p.stdout))

    def test_demo_checker_runs(self):
        self.assertEqual(run("demo_checker", "record.json", "transcript.json").returncode, 0)


class Reconciliation(unittest.TestCase):
    def test_model_changed_state_not_treated_as_done_without_log(self):
        # The record claims it changed client/fetch.py, but the authoritative tool log
        # confirms no side effect. The claim must surface as unbacked, not as done.
        rec = HandoffRecord.load(EX / "record.json")
        reconciled = reconcile_changed_state(ToolLog())
        mism = changed_state_mismatches(rec.changed_state, reconciled)
        self.assertEqual(reconciled, [])
        self.assertIn(rec.changed_state[0], mism["unbacked_claims"])

    def test_confirmed_side_effect_is_backed(self):
        log = ToolLog()
        log.append(ToolLogEntry("o", "edit_file", "d", RiskClass.REVERSIBLE_LOCAL, True,
                                OutcomeStatus.CONFIRMED_SUCCEEDED, 0.0, 0.0, "client/fetch.py edited"))
        reconciled = reconcile_changed_state(log)
        self.assertEqual(len(reconciled), 1)


if __name__ == "__main__":
    unittest.main()

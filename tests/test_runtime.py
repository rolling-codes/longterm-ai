"""Runtime: state machine, reconciliation, retry guard, checkpoints, completion gate."""
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from longterm_ai import (  # noqa: E402
    Checkpoint, CheckpointStore, Criterion, DeterministicVerifier, FakeModel, Framework, OutcomeStatus,
    RiskClass, RuleBasedPolicy, ScriptedTurn, Task, TaskStateMachine, TaskStatus, ToolLog, ToolLogEntry,
    FunctionTool, ToolDefinition, ToolRegistry, changed_state_mismatches, claim_accuracy, reconcile_changed_state, should_retry,
)
from longterm_ai.contracts import ToolCall  # noqa: E402
from longterm_ai.runtime import build_system, mandatory_present  # noqa: E402
from longterm_ai.errors import InvalidTransition, ToolTimeout, UnsupportedSchemaVersion  # noqa: E402


def _entry(tool, risk, outcome, idempotent=True, evidence=""):
    return ToolLogEntry(op_id="o", tool=tool, args_digest="d", risk=risk, idempotent=idempotent,
                        result_status=outcome, dispatched_at=0.0, finished_at=0.0, evidence=evidence)


class StateMachine(unittest.TestCase):
    def test_valid_path(self):
        sm = TaskStateMachine()
        sm.transition(TaskStatus.PLANNING)
        sm.transition(TaskStatus.RUNNING)
        sm.transition(TaskStatus.VERIFYING)
        sm.transition(TaskStatus.COMPLETED)
        self.assertEqual(sm.status, TaskStatus.COMPLETED)

    def test_invalid_transition_raises(self):
        sm = TaskStateMachine()
        with self.assertRaises(InvalidTransition):
            sm.transition(TaskStatus.COMPLETED)   # created -> completed not allowed

    def test_terminal_cannot_reopen(self):
        sm = TaskStateMachine()
        sm.transition(TaskStatus.CANCELLED)
        with self.assertRaises(InvalidTransition):
            sm.transition(TaskStatus.RUNNING)


class Reconciliation(unittest.TestCase):
    def test_changed_state_from_log_only(self):
        log = ToolLog()
        log.append(_entry("push", RiskClass.EXTERNAL_SIDE_EFFECT, OutcomeStatus.CONFIRMED_SUCCEEDED, evidence="pushed main"))
        log.append(_entry("read", RiskClass.READ_ONLY, OutcomeStatus.CONFIRMED_SUCCEEDED, evidence="read file"))
        log.append(_entry("deploy", RiskClass.HIGH_IMPACT, OutcomeStatus.UNKNOWN, idempotent=False))
        cs = reconcile_changed_state(log)
        self.assertEqual(len(cs), 1)           # read-only excluded, unknown excluded
        self.assertIn("push", cs[0])

    def test_model_claim_absent_from_log_is_unbacked(self):
        reconciled = reconcile_changed_state(ToolLog())   # nothing confirmed
        mism = changed_state_mismatches(["deleted production db"], reconciled)
        self.assertIn("deleted production db", mism["unbacked_claims"])

    def test_claim_accuracy_empty_claims_is_none(self):
        self.assertIsNone(claim_accuracy([], ["push: repo updated"]))

    def test_claim_accuracy_all_backed_is_one(self):
        claims = ["push: commit pushed", "write: file written"]
        self.assertEqual(claim_accuracy(claims, claims), 1.0)

    def test_claim_accuracy_over_reporting(self):
        # 1 backed out of 2 claimed = 0.5 accuracy (over-reporting detection)
        claims = ["push: commit pushed", "deploy: staged"]
        reconciled = ["push: commit pushed"]
        self.assertEqual(claim_accuracy(claims, reconciled), 0.5)


class RetryGuard(unittest.TestCase):
    def test_unknown_non_idempotent_not_retried(self):
        self.assertFalse(should_retry(OutcomeStatus.UNKNOWN, idempotent=False, attempts=1, max_attempts=5))

    def test_unknown_idempotent_retried_within_budget(self):
        self.assertTrue(should_retry(OutcomeStatus.UNKNOWN, idempotent=True, attempts=1, max_attempts=5))

    def test_attempts_exhausted_stops(self):
        self.assertFalse(should_retry(OutcomeStatus.UNKNOWN, idempotent=True, attempts=5, max_attempts=5))

    def test_confirmed_success_not_retried(self):
        self.assertFalse(should_retry(OutcomeStatus.CONFIRMED_SUCCEEDED, idempotent=True, attempts=1, max_attempts=5))


class MandatorySurvival(unittest.TestCase):
    def test_constraints_present_in_system(self):
        text = build_system(Task(objective="o", mandatory_constraints=["A", "B"]))
        self.assertIn("A", text)
        self.assertEqual(mandatory_present(text, ["A", "B"]), [])

    def test_missing_constraint_detected(self):
        self.assertEqual(mandatory_present("no constraints here", ["A"]), ["A"])


class Checkpoints(unittest.TestCase):
    def test_round_trip(self):
        tmp = tempfile.mkdtemp()
        store = CheckpointStore(tmp)
        store.save(Checkpoint(task_id="t1", status="running", completed_steps=3))
        loaded = store.load("t1")
        self.assertIsNotNone(loaded)
        assert loaded is not None
        self.assertEqual(loaded.completed_steps, 3)
        self.assertEqual(loaded.schema_version, 1)

    def test_unknown_schema_version_refused(self):
        tmp = tempfile.mkdtemp()
        store = CheckpointStore(tmp)
        store.save(Checkpoint(task_id="t2", status="running"))
        p = Path(tmp) / "checkpoint_t2.json"
        data = json.loads(p.read_text())
        data["schema_version"] = 999
        p.write_text(json.dumps(data))
        with self.assertRaises(UnsupportedSchemaVersion):
            store.load("t2")


class EndToEnd(unittest.TestCase):
    def _fw(self, model, reg, tmp, **kw):
        policy = RuleBasedPolicy(reg.risk_map())
        return Framework(model, reg, policy, DeterministicVerifier(base_dir=tmp), state_dir=tmp, **kw)

    def test_completes_only_after_verification(self):
        tmp = tempfile.mkdtemp()
        (Path(tmp) / "done.txt").write_text("x")
        reg = ToolRegistry()
        task = Task(objective="o", mandatory_constraints=["Do not push"],
                    acceptance_criteria=[Criterion("c", "file_exists", {"path": "done.txt"})])
        model = FakeModel([ScriptedTurn(content="done")])
        r = self._fw(model, reg, tmp).run(task)
        self.assertTrue(r.verified_complete)
        self.assertEqual(r.status, TaskStatus.COMPLETED)

    def test_failed_criterion_never_completes(self):
        tmp = tempfile.mkdtemp()
        reg = ToolRegistry()
        task = Task(objective="o", max_steps=3,
                    acceptance_criteria=[Criterion("c", "file_exists", {"path": "never.txt"})])
        model = FakeModel([ScriptedTurn(content="done")])   # model claims done every turn
        r = self._fw(model, reg, tmp).run(task)
        self.assertFalse(r.verified_complete)
        self.assertNotEqual(r.status, TaskStatus.COMPLETED)

    def test_unknown_non_idempotent_tool_not_auto_retried(self):
        tmp = tempfile.mkdtemp()
        (Path(tmp) / "done.txt").write_text("x")
        calls = {"n": 0}
        def flaky(args, ctx):
            calls["n"] += 1
            raise ToolTimeout("no response")
        reg = ToolRegistry()
        reg.register(FunctionTool(
            ToolDefinition("flaky", risk=RiskClass.REVERSIBLE_LOCAL, idempotent=False), flaky))
        task = Task(objective="o", acceptance_criteria=[Criterion("c", "file_exists", {"path": "done.txt"})])
        model = FakeModel([
            ScriptedTurn(tool_calls=[ToolCall(name="flaky", args={}, id="op1")]),
            ScriptedTurn(content="done"),
        ])
        r = self._fw(model, reg, tmp, max_tool_attempts=3).run(task)
        self.assertEqual(calls["n"], 1)                        # not retried despite budget of 3
        self.assertEqual(r.tool_log[0]["result_status"], "unknown")


if __name__ == "__main__":
    unittest.main()

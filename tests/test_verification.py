"""Deterministic verifier: the three checks, the completion gate (REQ 013)."""
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from longterm_ai import (  # noqa: E402
    Criterion, CriterionResult, CriterionStatus, DeterministicVerifier, Task, VerificationReport,
)


class Checks(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.v = DeterministicVerifier(base_dir=self.tmp)

    def _run(self, crit):
        return self.v.evaluate(Task(objective="t", acceptance_criteria=[crit])).results[0]

    def test_exit_code_pass(self):
        c = Criterion("c", "exit_code", {"command": [sys.executable, "-c", "import sys; sys.exit(0)"]})
        self.assertEqual(self._run(c).status, CriterionStatus.PASS)

    def test_exit_code_fail(self):
        c = Criterion("c", "exit_code", {"command": [sys.executable, "-c", "import sys; sys.exit(3)"]})
        self.assertEqual(self._run(c).status, CriterionStatus.FAIL)

    def test_file_exists_pass(self):
        (Path(self.tmp) / "a.txt").write_text("x")
        self.assertEqual(self._run(Criterion("c", "file_exists", {"path": "a.txt"})).status, CriterionStatus.PASS)

    def test_file_exists_fail(self):
        self.assertEqual(self._run(Criterion("c", "file_exists", {"path": "missing.txt"})).status, CriterionStatus.FAIL)

    def test_path_traversal_blocked(self):
        self.assertEqual(self._run(Criterion("c", "file_exists", {"path": "../../x"})).status, CriterionStatus.BLOCKED)

    def test_schema_valid_pass(self):
        c = Criterion("c", "schema_valid", {"data": {"n": 1, "s": "x"}, "schema": {"n": "int", "s": "str"}})
        self.assertEqual(self._run(c).status, CriterionStatus.PASS)

    def test_schema_valid_fail(self):
        c = Criterion("c", "schema_valid", {"data": {"n": "notint"}, "schema": {"n": "int"}})
        self.assertEqual(self._run(c).status, CriterionStatus.FAIL)

    def test_unknown_kind_is_unverified(self):
        self.assertEqual(self._run(Criterion("c", "mystery", {})).status, CriterionStatus.UNVERIFIED)

    def test_errored_check_is_blocked_not_pass(self):
        c = Criterion("c", "exit_code", {"command": ["definitely_not_a_real_command_xyz"]})
        self.assertEqual(self._run(c).status, CriterionStatus.BLOCKED)


class CompletionGate(unittest.TestCase):
    def test_optional_failure_does_not_block(self):
        rep = VerificationReport([
            CriterionResult("a", CriterionStatus.PASS, required=True),
            CriterionResult("b", CriterionStatus.FAIL, required=False),
        ])
        self.assertTrue(rep.all_required_passed)

    def test_required_failure_blocks(self):
        rep = VerificationReport([CriterionResult("a", CriterionStatus.FAIL, required=True)])
        self.assertFalse(rep.all_required_passed)

    def test_no_criteria_is_not_passed(self):
        self.assertFalse(VerificationReport([]).all_required_passed)


if __name__ == "__main__":
    unittest.main()

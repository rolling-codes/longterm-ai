"""Policy: action classes, fail-closed authorization (REQ 019), approval binding (REQ 010)."""
import sys
import time
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from longterm_ai import (  # noqa: E402
    Action, Approval, ApprovalStore, Decision, RiskClass, RuleBasedPolicy, action_digest, fail_closed,
)
from longterm_ai.contracts import AuthorizationDecision  # noqa: E402


class _Raising:
    def authorize(self, action):
        raise RuntimeError("boom")


class _Invalid:
    def authorize(self, action):
        return "not a decision object"


class _Slow:
    def authorize(self, action):
        time.sleep(0.5)
        return AuthorizationDecision(Decision.ALLOW)


class FailClosed(unittest.TestCase):
    def test_error_counts_as_deny(self):
        self.assertEqual(fail_closed(_Raising(), Action("x")).decision, Decision.DENY)

    def test_invalid_output_counts_as_deny(self):
        self.assertEqual(fail_closed(_Invalid(), Action("x")).decision, Decision.DENY)  # type: ignore[arg-type]  # intentionally invalid: garbage output must be denied

    def test_timeout_counts_as_deny(self):
        self.assertEqual(fail_closed(_Slow(), Action("x"), timeout_s=0.05).decision, Decision.DENY)


class ActionClasses(unittest.TestCase):
    def setUp(self):
        self.p = RuleBasedPolicy({
            "read": RiskClass.READ_ONLY,
            "edit": RiskClass.REVERSIBLE_LOCAL,
            "push": RiskClass.EXTERNAL_SIDE_EFFECT,
            "deploy": RiskClass.HIGH_IMPACT,
            "rm": RiskClass.PROHIBITED,
        })

    def test_read_only_allowed(self):
        self.assertEqual(self.p.authorize(Action("read")).decision, Decision.ALLOW)

    def test_reversible_allowed_when_writes_on(self):
        self.assertEqual(self.p.authorize(Action("edit")).decision, Decision.ALLOW)

    def test_reversible_denied_when_writes_off(self):
        p = RuleBasedPolicy({"edit": RiskClass.REVERSIBLE_LOCAL}, allow_writes=False)
        self.assertEqual(p.authorize(Action("edit")).decision, Decision.DENY)

    def test_external_requires_approval(self):
        self.assertEqual(self.p.authorize(Action("push")).decision, Decision.APPROVAL_REQUIRED)

    def test_external_preauthorized_allowed(self):
        p = RuleBasedPolicy({"push": RiskClass.EXTERNAL_SIDE_EFFECT}, allowed_external={"push"})
        self.assertEqual(p.authorize(Action("push")).decision, Decision.ALLOW)

    def test_high_impact_requires_approval(self):
        self.assertEqual(self.p.authorize(Action("deploy")).decision, Decision.APPROVAL_REQUIRED)

    def test_prohibited_denied(self):
        self.assertEqual(self.p.authorize(Action("rm")).decision, Decision.DENY)

    def test_unknown_tool_defaults_to_high_impact(self):
        # Least privilege: an unregistered action is not treated as read-only.
        self.assertEqual(self.p.authorize(Action("mystery")).decision, Decision.APPROVAL_REQUIRED)


class ApprovalBinding(unittest.TestCase):
    def test_approval_binds_to_exact_args(self):
        store = ApprovalStore()
        a1 = Action("push", args={"branch": "main"})
        store.grant(Approval("ap1", action_digest(a1)))
        self.assertTrue(store.verify(a1, "ap1"))
        a2 = Action("push", args={"branch": "release"})   # changed args
        self.assertFalse(store.verify(a2, "ap1"))

    def test_expired_approval_invalid(self):
        store = ApprovalStore()
        a = Action("push")
        store.grant(Approval("ap2", action_digest(a), expires_at=time.time() - 1))
        self.assertFalse(store.verify(a, "ap2"))

    def test_missing_grant_invalid(self):
        self.assertFalse(ApprovalStore().verify(Action("push"), "does-not-exist"))


if __name__ == "__main__":
    unittest.main()

"""Tool gateway: bounded envelope (REQ 003) and un-bypassable authorization (REQ 002)."""
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from longterm_ai import (  # noqa: E402
    Action, ArtifactStore, FunctionTool, RiskClass, RuleBasedPolicy, ToolContext, ToolDefinition,
    ToolGateway, ToolRegistry, ToolStatus,
)
from longterm_ai.tools import Reduced  # noqa: E402
from longterm_ai.errors import ToolTimeout  # noqa: E402


def _tool(name, fn, *, risk=RiskClass.READ_ONLY, schema=None, limit=4000, reduce=None, idempotent=True):
    return FunctionTool(
        ToolDefinition(name=name, arg_schema=schema or {}, risk=risk, output_limit=limit, idempotent=idempotent),
        fn, reduce=reduce,
    )


class Gateway(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.reg = ToolRegistry()

    def _gw(self, *, allow_writes=True, allowed_external=None, approval_cb=None):
        policy = RuleBasedPolicy(self.reg.risk_map(), allow_writes=allow_writes,
                                 allowed_external=allowed_external or set())
        return ToolGateway(self.reg, policy, artifact_store=ArtifactStore(self.tmp), approval_cb=approval_cb)

    def _ctx(self):
        return ToolContext(task_id="t", state_dir=Path(self.tmp))

    def test_read_tool_success_nothing_omitted(self):
        self.reg.register(_tool("read", lambda a, c: "short output"))
        r = self._gw().execute(Action("read"), self._ctx())
        self.assertEqual(r.status, ToolStatus.SUCCESS)
        self.assertFalse(r.truncated)
        self.assertEqual(r.omitted_count, 0)

    def test_truncation_marked_with_known_count_and_artifact(self):
        self.reg.register(_tool("big", lambda a, c: "x" * 50, limit=10))
        r = self._gw().execute(Action("big", op_id="o1"), self._ctx())
        self.assertTrue(r.truncated)
        self.assertEqual(r.omitted_count, 40)          # known count, not silent
        self.assertIsNotNone(r.artifact_ref)           # full output is retrievable
        assert r.artifact_ref is not None
        self.assertTrue((Path(self.tmp) / r.artifact_ref).exists())

    def test_unknown_omitted_count_never_becomes_zero(self):
        reducer = lambda raw, limit: Reduced(summary="s", truncated=True, omitted_count=None, full_text="full")
        self.reg.register(_tool("stream", lambda a, c: "whatever", reduce=reducer))
        r = self._gw().execute(Action("stream", op_id="o2"), self._ctx())
        self.assertTrue(r.truncated)
        self.assertIsNone(r.omitted_count)             # unknown stays unknown

    def test_prohibited_tool_denied_and_not_executed(self):
        calls = {"n": 0}
        def fn(a, c):
            calls["n"] += 1
            return "ran"
        self.reg.register(_tool("rm", fn, risk=RiskClass.PROHIBITED))
        r = self._gw().execute(Action("rm"), self._ctx())
        self.assertEqual(r.status, ToolStatus.DENIED)
        self.assertEqual(calls["n"], 0)                # cannot be bypassed via arguments

    def test_external_denied_without_approver(self):
        self.reg.register(_tool("push", lambda a, c: "pushed", risk=RiskClass.EXTERNAL_SIDE_EFFECT))
        r = self._gw().execute(Action("push"), self._ctx())
        self.assertEqual(r.status, ToolStatus.DENIED)

    def test_external_allowed_with_approver(self):
        self.reg.register(_tool("push", lambda a, c: "pushed", risk=RiskClass.EXTERNAL_SIDE_EFFECT))
        r = self._gw(approval_cb=lambda a, d: True).execute(Action("push"), self._ctx())
        self.assertEqual(r.status, ToolStatus.SUCCESS)

    def test_unknown_tool_is_failure(self):
        r = self._gw().execute(Action("ghost"), self._ctx())
        self.assertEqual(r.status, ToolStatus.FAILURE)

    def test_invalid_arguments_rejected(self):
        self.reg.register(_tool("read", lambda a, c: "ok", schema={"path": "str"}))
        r = self._gw().execute(Action("read", args={"path": 123}), self._ctx())
        self.assertEqual(r.status, ToolStatus.FAILURE)
        self.assertIn("invalid arguments", r.summary)

    def test_timeout_is_unknown_not_success(self):
        def fn(a, c):
            raise ToolTimeout("no response")
        self.reg.register(_tool("slow", fn))
        r = self._gw().execute(Action("slow"), self._ctx())
        self.assertEqual(r.status, ToolStatus.UNKNOWN)

    def test_tool_error_is_failure(self):
        def fn(a, c):
            raise ValueError("kaboom")
        self.reg.register(_tool("boom", fn))
        r = self._gw().execute(Action("boom"), self._ctx())
        self.assertEqual(r.status, ToolStatus.FAILURE)


if __name__ == "__main__":
    unittest.main()

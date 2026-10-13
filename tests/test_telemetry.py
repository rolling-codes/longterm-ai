"""Usage accounting honesty (REQ 008) and small telemetry helpers."""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from longterm_ai import EventLog, UsageAccountant, UsageRecord, rough_tokens  # noqa: E402
from longterm_ai.telemetry import redact  # noqa: E402


class Accounting(unittest.TestCase):
    def test_unknown_reported_stays_none_not_zero(self):
        a = UsageAccountant()
        a.add(UsageRecord(input_tokens_estimated=10, input_tokens_reported=None, cost_usd=None))
        a.add(UsageRecord(input_tokens_estimated=5, input_tokens_reported=None, cost_usd=None))
        self.assertIsNone(a.input_tokens_reported)        # never coerced to 0
        self.assertEqual(a.input_tokens_estimated, 15)    # estimated is a real sum
        self.assertIsNone(a.cost_usd)

    def test_cost_sums_across_calls(self):
        a = UsageAccountant()
        for c in (0.01, 0.02, 0.03):
            a.add(UsageRecord(cost_usd=c))
        assert a.cost_usd is not None
        self.assertAlmostEqual(a.cost_usd, 0.06)
        self.assertEqual(a.calls, 3)

    def test_mixed_known_and_unknown_is_unknown(self):
        a = UsageAccountant()
        a.add(UsageRecord(cost_usd=0.01))
        a.add(UsageRecord(cost_usd=None))
        self.assertIsNone(a.cost_usd)   # an incomplete total is unknown, not 0.01


class Helpers(unittest.TestCase):
    def test_rough_tokens_is_at_least_one(self):
        self.assertEqual(rough_tokens(""), 1)
        self.assertGreaterEqual(rough_tokens("abcd" * 10), 10)

    def test_redact_hides_sensitive_keys(self):
        out = redact({"api_key": "x", "keep": 1})
        self.assertEqual(out["api_key"], "[redacted]")
        self.assertEqual(out["keep"], 1)

    def test_eventlog_filters_by_type(self):
        e = EventLog()
        e.record("model_call", model="m")
        e.record("tool_result", tool="t")
        self.assertEqual(len(e.of_type("model_call")), 1)


if __name__ == "__main__":
    unittest.main()

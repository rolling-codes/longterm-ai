"""Usage accounting, cost, structured events, and the default token estimator.

Spec section 14. Two honesty invariants:
  * Unknown usage stays ``None`` and is never coerced to 0 (REQ 008). An aggregate
    that mixes in an unknown contribution is itself unknown, not a smaller number.
  * Cost per task includes every call: retries, tool calls, and verification.
Secrets are kept out of events by default.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from .contracts import UsageRecord

_SENSITIVE_KEYS = {"password", "token", "api_key", "apikey", "secret", "authorization"}


def rough_tokens(text: str) -> int:
    """Rough estimate, about four characters per token for English.

    Shared default across the package (handoff verification reuses it). Swap in a
    provider-accurate counter when one is available (spec section 7.3).
    """
    return max(1, len(text) // 4)


class RoughTokenEstimator:
    """Default TokenEstimator. Documented estimate, flagged as estimated not reported."""

    def estimate(self, text: str) -> int:
        return rough_tokens(text)


def redact(d: dict[str, Any]) -> dict[str, Any]:
    """Drop obviously sensitive values before they reach an event log."""
    return {k: ("[redacted]" if k.lower() in _SENSITIVE_KEYS else v) for k, v in d.items()}


class UsageAccountant:
    """Sums usage across all model calls in a task. Unknown propagates (never 0)."""

    def __init__(self) -> None:
        self.records: list[UsageRecord] = []

    def add(self, usage: UsageRecord) -> None:
        self.records.append(usage)

    def _sum(self, attr: str) -> int | float | None:
        vals = [getattr(r, attr) for r in self.records]
        if not vals or any(v is None for v in vals):
            return None  # an incomplete total is unknown, not a smaller known number
        return sum(vals)

    def _sum_int(self, attr: str) -> int | None:
        total = self._sum(attr)
        return int(total) if total is not None else None

    @property
    def input_tokens_reported(self) -> int | None:
        return self._sum_int("input_tokens_reported")

    @property
    def output_tokens_reported(self) -> int | None:
        return self._sum_int("output_tokens_reported")

    @property
    def input_tokens_estimated(self) -> int | None:
        return self._sum_int("input_tokens_estimated")

    @property
    def output_tokens_estimated(self) -> int | None:
        return self._sum_int("output_tokens_estimated")

    @property
    def cost_usd(self) -> float | None:
        return self._sum("cost_usd")

    @property
    def calls(self) -> int:
        return len(self.records)

    def summary(self) -> dict[str, Any]:
        return {
            "model_calls": self.calls,
            "input_tokens_reported": self.input_tokens_reported,
            "input_tokens_estimated": self.input_tokens_estimated,
            "output_tokens_reported": self.output_tokens_reported,
            "output_tokens_estimated": self.output_tokens_estimated,
            "cost_usd": self.cost_usd,
        }


@dataclass
class EventLog:
    """Append-only structured events. No raw conversation content by default."""
    events: list[dict[str, Any]] = field(default_factory=list)

    def record(self, event_type: str, **fields: Any) -> None:
        self.events.append({"ts": time.time(), "type": event_type, **redact(fields)})

    def of_type(self, event_type: str) -> list[dict[str, Any]]:
        return [e for e in self.events if e["type"] == event_type]

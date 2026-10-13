"""Check a handoff record before a model switch. Spec section 25.

Layer one is plain code and always runs. Layers two and three need a checker model,
supplied through the Checker interface, so any provider or a local model can be used.
Exit code 0 means pass. Any other exit code, or a crash, means fail, and the caller
must not switch models (fail closed, REQ 019).

Usage:
  python -m longterm_ai.handoff.verify TASK_JSON RECORD_JSON STATE_DIR [--max-tokens N]
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

from ..telemetry import rough_tokens
from .record import HandoffRecord, UnsupportedRecord, fingerprint, normalize


@dataclass
class Result:
    passed: bool = True
    failures: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    removed_lines: list[str] = field(default_factory=list)

    def fail(self, msg: str) -> None:
        self.passed = False
        self.failures.append(msg)


def layer_one(task: dict, record: HandoffRecord, state_dir: Path, max_tokens: int) -> Result:
    res = Result()
    # Mandatory constraints: present word for word and fingerprints match the task's copy.
    expected = {fingerprint(c): c for c in task["mandatory_constraints"]}
    got = {fingerprint(c) for c in record.mandatory_constraints}
    for fp, text in expected.items():
        if fp not in got:
            res.fail(f"mandatory constraint missing or altered: {text!r}")
    # Extra constraints are a privilege path: they would be placed in system text with
    # the authority of the task. Only the task may add them.
    for c in record.mandatory_constraints:
        if fingerprint(c) not in expected:
            res.fail(f"record adds a constraint the task does not have: {c!r}")
    for fp, text in expected.items():
        want = task.get("constraint_fingerprints", {}).get(text)
        if want and want != fp:
            res.fail(f"task fingerprint mismatch for: {text!r}")
    # Required fields filled.
    for name in ("goal", "success_test", "next_action"):
        if not normalize(getattr(record, name)):
            res.fail(f"required field empty: {name}")
    if normalize(record.goal) != normalize(task["goal"]):
        res.fail("goal differs from the task's goal")
    # Evidence pointers resolve inside durable state, and do not escape it.
    root = state_dir.resolve()
    pointers = [p for d in record.decisions for p in d.evidence] + [p for f in record.facts for p in f.evidence]
    for p in pointers:
        target = (root / p).resolve()
        if root not in target.parents and target != root:
            res.fail(f"evidence pointer escapes durable state: {p}")
        elif not target.exists():
            res.fail(f"evidence pointer does not resolve: {p}")
    for d in record.decisions:
        if not d.evidence:
            res.warnings.append(f"decision has no evidence: {d.decision!r}")
    # Budget.
    n = rough_tokens(record.to_text())
    if n > max_tokens:
        res.fail(f"record is about {n} tokens, over the budget of {max_tokens}")
    return res


class Checker(Protocol):
    """Plug in any model. A checker from a different provider than the summarizer is preferred."""

    def supports(self, claim: str, source: str) -> bool: ...
    def questions(self, source: str) -> list[tuple[str, str, bool]]:
        """Return (question, answer from source, is_mandatory) tuples."""
        ...
    def answer(self, question: str, context: str) -> str: ...
    def same_answer(self, a: str, b: str) -> bool: ...


def layers_two_and_three(record: HandoffRecord, source_reasoning: str, checker: Checker, res: Result) -> Result:
    # Layer two: drop claims the source does not support.
    kept = []
    for d in record.decisions:
        if checker.supports(f"{d.decision} {d.reason}", source_reasoning):
            kept.append(d)
        else:
            res.removed_lines.append(d.decision)
    record.decisions = kept
    kept_f = []
    for f in record.facts:
        if checker.supports(f.fact, source_reasoning):
            kept_f.append(f)
        else:
            res.removed_lines.append(f.fact)
    record.facts = kept_f
    # Rejected options steer what the next model will not try, so an unsupported one
    # can block a correct fix. Check them like decisions.
    kept_r = []
    for r in record.rejected_options:
        if checker.supports(r, source_reasoning):
            kept_r.append(r)
        else:
            res.removed_lines.append(r)
    record.rejected_options = kept_r
    # Not checked here: changed_state. It describes side effects, so it is reconciled
    # against the tool call log in durable state, not against reasoning. See
    # longterm_ai.runtime.reconcile_changed_state.
    # Layer three: question test against the summary alone.
    summary = record.to_text() + "\n" + "\n".join(record.mandatory_constraints)
    misses = 0
    qs = checker.questions(source_reasoning)
    for q, want, mandatory in qs:
        got = checker.answer(q, summary)
        if not checker.same_answer(want, got):
            misses += 1
            if mandatory:
                res.fail(f"summary cannot answer a mandatory question: {q!r}")
    if qs:
        res.warnings.append(f"question test: {len(qs) - misses} of {len(qs)} answered")
    return res


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description="Layer one handoff checks. Exit 0 pass, nonzero fail.")
    ap.add_argument("task")
    ap.add_argument("record")
    ap.add_argument("state_dir")
    ap.add_argument("--max-tokens", type=int, default=1500)
    a = ap.parse_args(argv)
    task = json.loads(Path(a.task).read_text(encoding="utf-8"))
    try:
        record = HandoffRecord.load(a.record)
    except (UnsupportedRecord, TypeError) as e:
        print(json.dumps({"passed": False, "failures": [f"record could not be loaded: {e}"]}, indent=2))
        return 1
    res = layer_one(task, record, Path(a.state_dir), a.max_tokens)
    print(json.dumps({"passed": res.passed, "failures": res.failures, "warnings": res.warnings,
                      "record_fingerprint": record.fingerprint()}, indent=2))
    return 0 if res.passed else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

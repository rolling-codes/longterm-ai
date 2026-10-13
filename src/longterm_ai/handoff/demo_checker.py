"""A toy Checker that shows the layer two and three interface. Demo only.

It uses word overlap, which is NOT a real check. Replace it with a call to a model
you choose, ideally from a different provider than the summarizer.

Usage:
  python -m longterm_ai.handoff.demo_checker RECORD_JSON TRANSCRIPT_JSON
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

from .record import HandoffRecord
from .verify import Result, layers_two_and_three


def words(s: str) -> set[str]:
    return {w.strip(".,:;").lower() for w in s.split() if len(w) > 3}


class OverlapChecker:
    def supports(self, claim: str, source: str) -> bool:
        c = words(claim)
        return bool(c) and len(c & words(source)) / len(c) >= 0.3

    def questions(self, source: str):
        # A real checker would ask a model to write these from the source reasoning.
        return [("What is the retry cap?", "5", True), ("What does backoff honor?", "Retry-After", False)]

    def answer(self, question: str, context: str) -> str:
        if "cap" in question:
            return "5" if "5 times" in context else "unknown"
        return "Retry-After" if "Retry-After" in context else "unknown"

    def same_answer(self, a: str, b: str) -> bool:
        return a.strip().lower() == b.strip().lower()


def main(argv):
    record = HandoffRecord.load(argv[0])
    transcript = json.loads(Path(argv[1]).read_text(encoding="utf-8"))
    source = "\n".join(p["text"] for t in transcript["turns"] for p in t["parts"] if p["type"] in ("reasoning", "text"))
    res = layers_two_and_three(record, source, OverlapChecker(), Result())
    print(json.dumps({"passed": res.passed, "failures": res.failures, "warnings": res.warnings,
                      "removed_lines": res.removed_lines}, indent=2))
    return 0 if res.passed else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

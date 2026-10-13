"""Provider neutral handoff record. Spec section 25 (ported from the review overlay).

A handoff record carries what a model worked out across a model switch. Mandatory
constraints are copied by code from the task, never written by a model.

load() refuses unknown record versions and unknown fields (REQ 017, spec section 8),
and to_text() labels model-written facts as unrechecked so a summary cannot make
weak evidence sound settled (spec section 28).
"""
from __future__ import annotations

import hashlib
import json
import unicodedata
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path

RECORD_VERSION = "1"
SUPPORTED_VERSIONS = {"1"}


class UnsupportedRecord(ValueError):
    """The record has a version or a field this code does not know. Never guess."""


def normalize(text: str) -> str:
    """Normalize text so harmless whitespace or Unicode form changes do not break a match."""
    text = unicodedata.normalize("NFC", text)
    return " ".join(text.split())


def fingerprint(text: str) -> str:
    """SHA 256 of the normalized text."""
    return hashlib.sha256(normalize(text).encode("utf-8")).hexdigest()


@dataclass
class Decision:
    decision: str
    reason: str
    evidence: list[str] = field(default_factory=list)  # pointers into durable state


@dataclass
class Fact:
    fact: str
    evidence: list[str] = field(default_factory=list)


@dataclass
class HandoffRecord:
    goal: str
    success_test: str
    mandatory_constraints: list[str]
    decisions: list[Decision]
    rejected_options: list[str]
    facts: list[Fact]
    open_questions: list[str]
    next_action: str
    changed_state: list[str]
    version: str = RECORD_VERSION

    @classmethod
    def load(cls, path: str | Path) -> "HandoffRecord":
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        version = str(raw.get("version", ""))
        if version not in SUPPORTED_VERSIONS:
            raise UnsupportedRecord(f"record version {version!r} is not supported; known: {sorted(SUPPORTED_VERSIONS)}")
        known = {f.name for f in fields(cls)}
        extra = set(raw) - known
        if extra:
            raise UnsupportedRecord(f"record has unknown fields: {sorted(extra)}")
        raw["version"] = version
        raw["decisions"] = [Decision(**d) for d in raw.get("decisions", [])]
        raw["facts"] = [Fact(**f) for f in raw.get("facts", [])]
        return cls(**raw)

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2, ensure_ascii=False)

    def to_text(self) -> str:
        """Plain text for the receiving model. Mandatory constraints go in system text separately."""
        lines = [f"Earlier working notes, handoff record version {self.version}.",
                 "These notes were written by a model and checked against its reasoning, not against the world. "
                 "Treat each reported fact as unverified until its evidence is read.",
                 f"Goal: {self.goal}", f"Success test: {self.success_test}"]
        for d in self.decisions:
            ev = f" Evidence: {', '.join(d.evidence)}." if d.evidence else " No evidence recorded."
            lines.append(f"Decided: {d.decision} Because: {d.reason}.{ev}")
        for r in self.rejected_options:
            lines.append(f"Rejected earlier: {r}")
        for f in self.facts:
            ev = f" Evidence: {', '.join(f.evidence)}." if f.evidence else " No evidence recorded."
            lines.append(f"Reported, not rechecked: {f.fact}{ev}")
        for q in self.open_questions:
            lines.append(f"Open question: {q}")
        for c in self.changed_state:
            lines.append(f"Already changed: {c}")
        lines.append(f"Next action: {self.next_action}")
        return "\n".join(lines)

    def fingerprint(self) -> str:
        return fingerprint(self.to_json())

"""Verification engine. Spec section 13.

A model saying a task is done is not proof. The verifier runs the configured
acceptance criteria and returns one status per criterion. A task is verified
complete only when every required criterion passes (REQ 013). Phase 0 ships three
deterministic checks, as every reviewer agreed: command exit code, file exists,
schema valid. A check that errors is ``blocked``, never silently ``pass``.
"""
from __future__ import annotations

import shlex
import subprocess
import json
from pathlib import Path
from typing import Any

from .contracts import Criterion, CriterionResult, CriterionStatus, Task, VerificationReport

_TYPES: dict[str, type | tuple[type, ...]] = {
    "str": str, "int": int, "float": (int, float), "bool": bool, "dict": dict, "list": list,
}


def _check_schema(data: Any, schema: dict[str, str]) -> str:
    if not isinstance(data, dict):
        return "value is not an object"
    for key, type_name in schema.items():
        if key not in data:
            return f"missing key {key!r}"
        expected = _TYPES.get(type_name)
        if expected is not None and not isinstance(data[key], expected):
            return f"key {key!r} must be {type_name}"
    return ""


class DeterministicVerifier:
    def __init__(self, base_dir: str | Path | None = None, timeout_s: float = 60.0) -> None:
        self.base = Path(base_dir).resolve() if base_dir else None
        self.timeout_s = timeout_s

    def evaluate(self, task: Task, extra: dict[str, Any] | None = None) -> VerificationReport:
        extra = extra or {}
        results: list[CriterionResult] = []
        for c in task.acceptance_criteria:
            status, evidence = self._check(c, extra)
            results.append(CriterionResult(c.id, status, evidence, c.required))
        return VerificationReport(results)

    def _check(self, c: Criterion, extra: dict[str, Any]) -> tuple[CriterionStatus, str]:
        try:
            if c.kind == "exit_code":
                return self._exit_code(c)
            if c.kind == "file_exists":
                return self._file_exists(c)
            if c.kind == "schema_valid":
                return self._schema_valid(c, extra)
            return CriterionStatus.UNVERIFIED, f"unknown check kind {c.kind!r}"
        except Exception as exc:  # noqa: BLE001 - an errored check never counts as pass
            return CriterionStatus.BLOCKED, f"check errored: {exc}"

    def _exit_code(self, c: Criterion) -> tuple[CriterionStatus, str]:
        cmd = c.params["command"]
        argv = shlex.split(cmd) if isinstance(cmd, str) else list(cmd)
        expected = int(c.params.get("expected", 0))
        proc = subprocess.run(
            argv, cwd=str(self.base) if self.base else None,
            capture_output=True, text=True, timeout=self.timeout_s,
        )
        ok = proc.returncode == expected
        return (CriterionStatus.PASS if ok else CriterionStatus.FAIL,
                f"exit={proc.returncode} expected={expected}")

    def _resolve(self, rel: str) -> Path:
        if self.base is None:
            return Path(rel)
        target = (self.base / rel).resolve()
        if self.base != target and self.base not in target.parents:
            raise ValueError(f"path escapes base dir: {rel}")  # traversal guard
        return target

    def _file_exists(self, c: Criterion) -> tuple[CriterionStatus, str]:
        target = self._resolve(c.params["path"])
        exists = target.exists()
        return (CriterionStatus.PASS if exists else CriterionStatus.FAIL,
                f"{target} exists={exists}")

    def _schema_valid(self, c: Criterion, extra: dict[str, Any]) -> tuple[CriterionStatus, str]:
        schema = c.params["schema"]
        if "path" in c.params:
            data = json.loads(self._resolve(c.params["path"]).read_text(encoding="utf-8"))
        elif "data_key" in c.params:          # pull a value produced at runtime
            data = extra.get(c.params["data_key"])
        else:
            data = c.params.get("data")
        err = _check_schema(data, schema)
        return (CriterionStatus.PASS if not err else CriterionStatus.FAIL, err or "schema valid")

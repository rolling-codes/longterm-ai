"""Tool registry and gateway with bounded result envelopes. Spec section 9.

Every framework-managed tool call runs the same lifecycle: validate -> authorize
(fail-closed) -> apply limits -> execute -> reduce -> bounded envelope -> telemetry.
The model cannot reach a tool except through :meth:`ToolGateway.execute`, so the
checks cannot be bypassed by a crafted argument (REQ 002). Output is never silently
truncated and an unknown omitted count is never reported as 0 (REQ 003).
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from .contracts import (
    Action, PolicyEngine, RiskClass, Tool, ToolContext, ToolDefinition, ToolResult, ToolStatus,
)
from .errors import ToolTimeout
from .policy import fail_closed
from .telemetry import EventLog

_TYPES: dict[str, type | tuple[type, ...]] = {
    "str": str, "int": int, "float": (int, float), "bool": bool, "dict": dict, "list": list,
}


# --------------------------------------------------------------------------- #
# Reducers: raw tool output -> bounded pieces
# --------------------------------------------------------------------------- #
@dataclass
class Reduced:
    summary: str
    data: dict[str, Any] = field(default_factory=dict)
    truncated: bool = False
    omitted_count: int | None = 0   # 0 = known nothing omitted; None = unknown
    full_text: str | None = None


def generic_reducer(raw: Any, limit: int) -> Reduced:
    """Byte-cap any output. Keeps a known omitted count."""
    text = raw if isinstance(raw, str) else json.dumps(raw, ensure_ascii=False, default=str)
    if len(text) <= limit:
        return Reduced(summary=text[:200], data={"text": text}, full_text=text)
    return Reduced(
        summary=text[:200], data={"text": text[:limit]},
        truncated=True, omitted_count=len(text) - limit, full_text=text,
    )


def shell_reducer(raw: dict[str, Any], limit: int, keep_lines: int = 20) -> Reduced:
    """Tool-specific reducer: exit code + compact stdout/stderr tails + full artifact."""
    exit_code = raw.get("exit_code")
    out_lines = str(raw.get("stdout", "")).splitlines()
    err_lines = str(raw.get("stderr", "")).splitlines()
    kept_out, kept_err = out_lines[-keep_lines:], err_lines[-keep_lines:]
    omitted = (len(out_lines) - len(kept_out)) + (len(err_lines) - len(kept_err))
    return Reduced(
        summary=f"exit={exit_code}; {len(out_lines)} stdout / {len(err_lines)} stderr lines",
        data={"exit_code": exit_code, "stdout_tail": "\n".join(kept_out), "stderr_tail": "\n".join(kept_err)},
        truncated=omitted > 0, omitted_count=omitted,
        full_text=json.dumps(raw, ensure_ascii=False, default=str),
    )


# --------------------------------------------------------------------------- #
# Tool helpers
# --------------------------------------------------------------------------- #
class FunctionTool:
    """Wrap a plain function as a Tool, with an optional custom reducer."""

    def __init__(
        self,
        definition: ToolDefinition,
        fn: Callable[[dict[str, Any], ToolContext], Any],
        reduce: Callable[[Any, int], Reduced] | None = None,
    ) -> None:
        self.definition = definition
        self._fn = fn
        self._reduce = reduce

    @property
    def name(self) -> str:
        return self.definition.name

    def execute(self, args: dict[str, Any], ctx: ToolContext) -> Any:
        return self._fn(args, ctx)

    def reduce(self, raw: Any, limit: int) -> Reduced:
        return (self._reduce or generic_reducer)(raw, limit)


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        self._tools[tool.definition.name] = tool

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

    def all(self) -> list[Tool]:
        return list(self._tools.values())

    def definitions(self) -> list[ToolDefinition]:
        return [t.definition for t in self._tools.values()]

    def risk_map(self) -> dict[str, RiskClass]:
        return {name: t.definition.risk for name, t in self._tools.items()}


class ArtifactStore:
    """Writes full tool output to files and returns a stable ref. Spec section 9.3."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self._n = 0

    def put(self, key: str, text: str) -> str:
        self._n += 1
        safe = re.sub(r"[^A-Za-z0-9_.-]", "_", key or "artifact")
        rel = f"artifacts/{self._n:04d}_{safe}.txt"
        path = self.root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
        return rel


def validate_args(definition: ToolDefinition, args: dict[str, Any]) -> str:
    """Return "" if args satisfy the schema, else a reason. No model text involved."""
    for key, type_name in definition.arg_schema.items():
        if key not in args:
            return f"missing argument {key!r}"
        expected = _TYPES.get(type_name)
        if expected is not None and not isinstance(args[key], expected):
            return f"argument {key!r} must be {type_name}"
    return ""


# --------------------------------------------------------------------------- #
# Gateway
# --------------------------------------------------------------------------- #
class ToolGateway:
    def __init__(
        self,
        registry: ToolRegistry,
        policy: PolicyEngine,
        *,
        artifact_store: ArtifactStore | None = None,
        events: EventLog | None = None,
        approval_cb: Callable[[Action, Any], bool] | None = None,
        policy_timeout_s: float | None = None,
        on_artifact: Callable[[str, str, str], None] | None = None,
    ) -> None:
        """Configure policy enforcement, output storage, and optional callbacks.

        After storing a truncated result, call on_artifact with its reference,
        full text, and tool name.
        """
        self.registry = registry
        self.policy = policy
        self.artifact_store = artifact_store
        self.events = events or EventLog()
        self.approval_cb = approval_cb
        self.policy_timeout_s = policy_timeout_s
        # Called with (artifact_ref, full_text, tool_name) when an artifact is stored.
        # Lets a retrieval store index evidence without tools.py importing retrieval.
        self.on_artifact = on_artifact

    def execute(self, action: Action, ctx: ToolContext) -> ToolResult:
        """Authorize and validate an action, execute it, and return a bounded result."""
        prov = {"tool": action.name, "op_id": action.op_id}
        tool = self.registry.get(action.name)
        if tool is None:
            return ToolResult(ToolStatus.FAILURE, summary=f"unknown tool {action.name!r}", provenance=prov)

        # Authorize (fail-closed). Approval-required with no approver = not authorized.
        decision = fail_closed(self.policy, action, timeout_s=self.policy_timeout_s)
        from .contracts import Decision  # local import avoids a cycle at module load
        if decision.decision == Decision.APPROVAL_REQUIRED:
            approved = bool(self.approval_cb(action, decision)) if self.approval_cb else False
            if not approved:
                self.events.record("authorization", tool=action.name, decision="approval_denied")
                return ToolResult(ToolStatus.DENIED, summary="approval required and not granted",
                                  provenance=prov)
        elif not decision.allowed:
            self.events.record("authorization", tool=action.name, decision="deny", reason=decision.reason)
            return ToolResult(ToolStatus.DENIED, summary=decision.reason, provenance=prov)

        # Validate arguments against the schema.
        err = validate_args(tool.definition, action.args)
        if err:
            return ToolResult(ToolStatus.FAILURE, summary=f"invalid arguments: {err}", provenance=prov)

        # Execute under the tool's limits.
        try:
            raw = tool.execute(action.args, ctx)
        except ToolTimeout as exc:
            # Side effect may or may not have happened: unknown, never silently success.
            self.events.record("tool_timeout", tool=action.name)
            return ToolResult(ToolStatus.UNKNOWN, summary=f"timed out: {exc}", provenance=prov)
        except Exception as exc:  # noqa: BLE001 - surface failure, never fake success
            return ToolResult(ToolStatus.FAILURE, summary=f"error: {exc}", provenance=prov)

        # Normalize + bound.
        reduce_fn = getattr(tool, "reduce", None)
        reduced = reduce_fn(raw, tool.definition.output_limit) if reduce_fn else generic_reducer(raw, tool.definition.output_limit)
        artifact_ref = None
        if reduced.truncated and self.artifact_store is not None and reduced.full_text is not None:
            artifact_ref = self.artifact_store.put(action.op_id or action.name, reduced.full_text)
            if self.on_artifact is not None:
                self.on_artifact(artifact_ref, reduced.full_text, action.name)

        self.events.record("tool_result", tool=action.name, status="success",
                           truncated=reduced.truncated, omitted_count=reduced.omitted_count)
        return ToolResult(
            ToolStatus.SUCCESS,
            summary=reduced.summary,
            data=reduced.data,
            artifact_ref=artifact_ref,
            truncated=reduced.truncated,
            omitted_count=reduced.omitted_count,   # None stays None; never coerced to 0
            continuation_ref=artifact_ref,
            provenance=prov,
        )

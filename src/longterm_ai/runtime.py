"""Task state machine, execution loop, checkpoints, and side-effect reconciliation.

Spec section 11; state transitions follow overlay section 4. Completion routes only
through ``verifying`` so a task cannot be reported complete without its criteria
passing (REQ 013). Side-effect truth comes from the authoritative tool log, never
from the model's own claims (GPT section 29.2, Grok section 31 point 6).

Phase 2 additions: CheckpointStore.delete(), Framework.ckpt_store,
and resume= parameter on Framework.run() (ADR 0002).
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from .contracts import (
    Action, ModelAdapter, ModelRequest, OutcomeStatus, RiskClass, Task, TaskStatus,
    TERMINAL_STATES, ToolContext, ToolResult, ToolStatus, Verifier, VerificationReport,
)
from .errors import CheckpointError, InvalidTransition, UnsupportedSchemaVersion
from .policy import PolicyEngine, action_digest
from .telemetry import EventLog, RoughTokenEstimator, UsageAccountant
from .tools import ArtifactStore, ToolGateway, ToolRegistry

CHECKPOINT_SCHEMA_VERSION = 1

# Overlay section 4. Terminal states have no outgoing transitions (no implicit reopen).
ALLOWED_TRANSITIONS: dict[TaskStatus, set[TaskStatus]] = {
    TaskStatus.CREATED: {TaskStatus.PLANNING, TaskStatus.CANCELLED},
    TaskStatus.PLANNING: {TaskStatus.RUNNING, TaskStatus.BLOCKED, TaskStatus.FAILED, TaskStatus.CANCELLED},
    TaskStatus.RUNNING: {
        TaskStatus.WAITING_FOR_APPROVAL, TaskStatus.RECOVERABLE_FAILURE, TaskStatus.VERIFYING,
        TaskStatus.BLOCKED, TaskStatus.FAILED, TaskStatus.CANCELLED,
    },
    TaskStatus.WAITING_FOR_APPROVAL: {TaskStatus.RUNNING, TaskStatus.BLOCKED, TaskStatus.CANCELLED},
    TaskStatus.RECOVERABLE_FAILURE: {TaskStatus.RUNNING, TaskStatus.BLOCKED, TaskStatus.FAILED, TaskStatus.CANCELLED},
    TaskStatus.VERIFYING: {TaskStatus.COMPLETED, TaskStatus.RUNNING, TaskStatus.BLOCKED, TaskStatus.FAILED},
    TaskStatus.BLOCKED: {TaskStatus.RUNNING, TaskStatus.FAILED, TaskStatus.CANCELLED},
    TaskStatus.COMPLETED: set(),
    TaskStatus.FAILED: set(),
    TaskStatus.CANCELLED: set(),
}


class TaskStateMachine:
    def __init__(self, status: TaskStatus = TaskStatus.CREATED) -> None:
        self.status = status
        self.history: list[dict[str, Any]] = [{"status": status.value, "ts": time.time(), "cause": "created"}]

    def transition(self, to: TaskStatus, cause: str = "") -> None:
        if self.status in TERMINAL_STATES:
            raise InvalidTransition(f"{self.status.value} is terminal; cannot move to {to.value}")
        if to not in ALLOWED_TRANSITIONS[self.status]:
            raise InvalidTransition(f"{self.status.value} -> {to.value} is not allowed")
        self.status = to
        self.history.append({"status": to.value, "ts": time.time(), "cause": cause})


# --------------------------------------------------------------------------- #
# Authoritative tool log + side-effect reconciliation
# --------------------------------------------------------------------------- #
_SIDE_EFFECT_RISKS = {RiskClass.REVERSIBLE_LOCAL, RiskClass.EXTERNAL_SIDE_EFFECT, RiskClass.HIGH_IMPACT}


def _outcome(status: ToolStatus) -> OutcomeStatus:
    if status == ToolStatus.SUCCESS:
        return OutcomeStatus.CONFIRMED_SUCCEEDED
    if status == ToolStatus.UNKNOWN:
        return OutcomeStatus.UNKNOWN
    return OutcomeStatus.CONFIRMED_FAILED  # failure or denied


@dataclass
class ToolLogEntry:
    op_id: str
    tool: str
    args_digest: str
    risk: RiskClass
    idempotent: bool
    result_status: OutcomeStatus
    dispatched_at: float
    finished_at: float
    evidence: str = ""

    def to_dict(self) -> dict[str, Any]:
        d = {**self.__dict__, "risk": self.risk.value, "result_status": self.result_status.value}
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "ToolLogEntry":
        d = dict(d)
        d["risk"] = RiskClass(d["risk"])
        d["result_status"] = OutcomeStatus(d["result_status"])
        return cls(**d)


class ToolLog:
    def __init__(self) -> None:
        self.entries: list[ToolLogEntry] = []

    def append(self, entry: ToolLogEntry) -> None:
        self.entries.append(entry)

    def to_list(self) -> list[dict[str, Any]]:
        return [e.to_dict() for e in self.entries]

    @classmethod
    def from_list(cls, items: list[dict[str, Any]]) -> "ToolLog":
        log = cls()
        log.entries = [ToolLogEntry.from_dict(i) for i in items]
        return log


def reconcile_changed_state(log: ToolLog) -> list[str]:
    """Build the changed-state list from the log, not from the model.

    Only confirmed-succeeded side-effecting operations count as changed state.
    """
    out = []
    for e in log.entries:
        if e.risk in _SIDE_EFFECT_RISKS and e.result_status == OutcomeStatus.CONFIRMED_SUCCEEDED:
            out.append(f"{e.tool}: {e.evidence or ('op ' + e.op_id)}")
    return out


def changed_state_mismatches(model_claims: list[str], reconciled: list[str]) -> dict[str, list[str]]:
    """Compare model-written changed_state against the log. The log wins.

    ``unbacked_claims`` are side effects the model claims that the log does not
    confirm -- they must not be treated as done.
    """
    claims, backed = set(model_claims), set(reconciled)
    return {"unbacked_claims": sorted(claims - backed), "unclaimed_in_log": sorted(backed - claims)}


def claim_accuracy(model_claims: list[str], reconciled: list[str]) -> float | None:
    """Calculate accuracy ratio of model side-effect claims vs the authoritative log.

    Inspired by sub-agent health scoring (backed / claimed):
      accuracy = len(backed_claims) / len(claimed_claims)

    Returns None if the model made no claims. Returns 1.0 if every claimed change
    was confirmed in the log. Returns < 1.0 if the model over-reported side effects.
    """
    if not model_claims:
        return None
    claims, backed = set(model_claims), set(reconciled)
    return len(claims & backed) / len(claims)


def should_retry(outcome: OutcomeStatus, idempotent: bool, attempts: int, max_attempts: int) -> bool:
    """Bounded, error-specific retry. Never auto-retry an unknown non-idempotent op."""
    if attempts >= max_attempts:
        return False
    if outcome == OutcomeStatus.UNKNOWN and not idempotent:
        return False  # the critical guard: side effect may already have happened
    if outcome == OutcomeStatus.UNKNOWN and idempotent:
        return True
    return False  # confirmed success/failure/denied: do not loop


# --------------------------------------------------------------------------- #
# Checkpoints
# --------------------------------------------------------------------------- #
@dataclass
class Checkpoint:
    task_id: str
    status: str
    schema_version: int = CHECKPOINT_SCHEMA_VERSION
    completed_steps: int = 0
    criteria_status: dict[str, str] = field(default_factory=dict)
    tool_log: list[dict[str, Any]] = field(default_factory=list)
    remaining_budget: int | None = None
    deadline_s: float | None = None

    def to_json(self) -> str:
        return json.dumps(self.__dict__, indent=2, ensure_ascii=False)


class CheckpointStore:
    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def _path(self, task_id: str) -> Path:
        return self.root / f"checkpoint_{task_id}.json"

    def save(self, cp: Checkpoint) -> Path:
        path = self._path(cp.task_id)
        try:
            path.write_text(cp.to_json(), encoding="utf-8")
        except OSError as exc:
            raise CheckpointError(f"could not save checkpoint: {exc}") from exc
        return path  # durability is proven only after the write returns

    def load(self, task_id: str) -> Checkpoint | None:
        path = self._path(task_id)
        if not path.exists():
            return None
        raw = json.loads(path.read_text(encoding="utf-8"))
        version = int(raw.get("schema_version", -1))
        if version != CHECKPOINT_SCHEMA_VERSION:
            raise UnsupportedSchemaVersion(f"checkpoint schema {version} != {CHECKPOINT_SCHEMA_VERSION}")
        return Checkpoint(**raw)

    def delete(self, task_id: str) -> None:
        """Remove the checkpoint file once a task reaches a terminal state."""
        path = self._path(task_id)
        try:
            path.unlink(missing_ok=True)
        except OSError:
            pass  # best-effort; a stale file is not fatal


# --------------------------------------------------------------------------- #
# Framework / run loop
# --------------------------------------------------------------------------- #
@dataclass
class RunResult:
    status: TaskStatus
    report: VerificationReport | None
    usage: dict[str, Any]
    tool_log: list[dict[str, Any]]
    changed_state: list[str]
    steps: int
    note: str = ""

    @property
    def verified_complete(self) -> bool:
        return self.status == TaskStatus.COMPLETED


def build_system(task: Task) -> str:
    """Mandatory constraints are sent as plain text on every call (REQ 021)."""
    lines = ["You are completing a long-running task through a framework that enforces policy and verifies results.",
             f"Objective: {task.objective}"]
    if task.mandatory_constraints:
        lines.append("Mandatory constraints (these must always hold):")
        lines += [f"- {c}" for c in task.mandatory_constraints]
    return "\n".join(lines)


def mandatory_present(system: str, constraints: list[str]) -> list[str]:
    """Return constraints missing from the system text (REQ 021 survival check)."""
    return [c for c in constraints if c not in system]


class Framework:
    def __init__(
        self,
        model: ModelAdapter,
        tools: ToolRegistry,
        policy: PolicyEngine,
        verifier: Verifier,
        *,
        state_dir: str | Path,
        events: EventLog | None = None,
        approval_cb: Callable[[Action, Any], bool] | None = None,
        max_tool_attempts: int = 2,
        policy_timeout_s: float | None = None,
    ) -> None:
        self.model = model
        self.registry = tools
        self.policy = policy
        self.verifier = verifier
        self.state_dir = Path(state_dir)
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.events = events or EventLog()
        self.gateway = ToolGateway(
            tools, policy, artifact_store=ArtifactStore(self.state_dir),
            events=self.events, approval_cb=approval_cb, policy_timeout_s=policy_timeout_s,
        )
        self.estimator = RoughTokenEstimator()
        self.max_tool_attempts = max_tool_attempts
        self.ckpt_store = CheckpointStore(self.state_dir)

    def run(self, task: Task, task_id: str = "task", *, resume: bool = False) -> RunResult:
        """Run the task, optionally resuming from the last saved checkpoint.

        When ``resume=True`` and a checkpoint for *task_id* exists the run loop
        restores ``steps`` and the tool log from that checkpoint and continues from
        the saved status, skipping the normal PLANNING transition. A fresh
        :class:`UsageAccountant` is used for the resumed session (prior-session token
        counts are not carried over). After every model call the in-progress state is
        persisted; on any terminal exit the checkpoint file is deleted.
        """
        sm = TaskStateMachine()
        acct = UsageAccountant()
        log = ToolLog()
        steps = 0

        # --- optional resume ---
        if resume:
            cp = self.ckpt_store.load(task_id)
            if cp is not None:
                log = ToolLog.from_list(cp.tool_log)
                steps = cp.completed_steps
                # Restore the state machine to the saved status directly (it was valid
                # when saved; re-running the normal transition sequence is not correct).
                saved_status = TaskStatus(cp.status)
                sm.status = saved_status
                sm.history.append({"status": saved_status.value, "ts": time.time(), "cause": "resumed"})
                self.events.record("checkpoint_resume", task_id=task_id, steps=steps, status=cp.status)
            else:
                # No checkpoint exists: run fresh.
                sm.transition(TaskStatus.PLANNING, "start")
                sm.transition(TaskStatus.RUNNING, "plan ready")
        else:
            sm.transition(TaskStatus.PLANNING, "start")
            sm.transition(TaskStatus.RUNNING, "plan ready")

        system = build_system(task)
        missing = mandatory_present(system, task.mandatory_constraints)
        if missing:  # fail closed (REQ 021): never proceed without the constraints present
            sm.transition(TaskStatus.BLOCKED, "mandatory constraints missing")
            return self._result(sm, None, acct, log, steps, f"mandatory constraints missing: {missing}")

        messages: list[dict[str, Any]] = [{"role": "user", "content": task.objective}]
        start = time.monotonic()
        report: VerificationReport | None = None

        while steps < task.max_steps:
            steps += 1
            if task.deadline_s is not None and (time.monotonic() - start) > task.deadline_s:
                sm.transition(TaskStatus.BLOCKED, "deadline exceeded")
                self.ckpt_store.delete(task_id)
                return self._result(sm, report, acct, log, steps, "deadline exceeded")
            used = acct.input_tokens_estimated or 0
            if task.token_budget is not None and used > task.token_budget:
                sm.transition(TaskStatus.BLOCKED, "budget exhausted")
                self.ckpt_store.delete(task_id)
                return self._result(sm, report, acct, log, steps, "token budget exhausted")

            req = ModelRequest(system=system, messages=messages,
                               tools=self.registry.definitions(), max_output_tokens=1024)
            resp = self.model.generate(req)
            acct.add(resp.usage)
            self.events.record("model_call", model=resp.usage.model, finish=resp.finish_reason)

            # Persist progress after every model call. A failed save is logged but not fatal.
            try:
                self.ckpt_store.save(Checkpoint(
                    task_id=task_id, status=sm.status.value,
                    completed_steps=steps, tool_log=log.to_list(),
                ))
            except CheckpointError as exc:
                self.events.record("checkpoint_save_failed", reason=str(exc))

            if resp.tool_calls:
                messages.append({"role": "assistant",
                                 "content": resp.content or f"[requesting {len(resp.tool_calls)} tool call(s)]"})
                for call in resp.tool_calls:
                    action = Action(name=call.name, args=call.args, op_id=call.id or f"op{steps}")
                    tr = self._run_tool(action, log)
                    messages.append({"role": "user",
                                     "content": f"[tool {call.name} -> {tr.status.value}] {tr.summary}"})
                continue

            # No tool calls: the model believes it is done. Verify before completing.
            sm.transition(TaskStatus.VERIFYING, "model reported done")
            report = self.verifier.evaluate(task)
            self.events.record("verification", passed=report.all_required_passed)
            if report.all_required_passed:
                sm.transition(TaskStatus.COMPLETED, "all required criteria passed")
                self.ckpt_store.delete(task_id)
                return self._result(sm, report, acct, log, steps, "verified complete")
            fails = [f"{r.id}={r.status.value} ({r.evidence})"
                     for r in report.results if r.required and r.status.value != "pass"]
            messages.append({"role": "user",
                             "content": "Verification failed: " + "; ".join(fails) + ". Keep working."})
            sm.transition(TaskStatus.RUNNING, "verification failed; continue")

        if sm.status not in TERMINAL_STATES:
            sm.transition(TaskStatus.FAILED, "max steps reached")
        self.ckpt_store.delete(task_id)
        return self._result(sm, report, acct, log, steps, "max steps reached without verified completion")

    def _run_tool(self, action: Action, log: ToolLog) -> ToolResult:
        tool = self.registry.get(action.name)
        risk = tool.definition.risk if tool else RiskClass.HIGH_IMPACT
        idempotent = tool.definition.idempotent if tool else False
        attempts = 0
        tr: ToolResult
        while True:
            attempts += 1
            dispatched = time.time()
            ctx = ToolContext(task_id="task", state_dir=self.state_dir, deadline_s=None)
            tr = self.gateway.execute(action, ctx)
            outcome = _outcome(tr.status)
            log.append(ToolLogEntry(
                op_id=action.op_id, tool=action.name, args_digest=action_digest(action),
                risk=risk, idempotent=idempotent, result_status=outcome,
                dispatched_at=dispatched, finished_at=time.time(), evidence=tr.summary,
            ))
            if not should_retry(outcome, idempotent, attempts, self.max_tool_attempts):
                return tr

    def _result(self, sm: TaskStateMachine, report: VerificationReport | None,
                acct: UsageAccountant, log: ToolLog, steps: int, note: str) -> RunResult:
        return RunResult(
            status=sm.status, report=report, usage=acct.summary(),
            tool_log=log.to_list(), changed_state=reconcile_changed_state(log),
            steps=steps, note=note,
        )

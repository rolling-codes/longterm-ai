"""Public contracts: enums, dataclasses, and protocols shared across components.

Spec sections 4.3, 6.2, 6.3. Kept in one file until a real boundary justifies a
split (spec section 5). Invariant throughout: an unknown measurement stays ``None``
and is never coerced to 0 (REQ 008).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Protocol, runtime_checkable


# --------------------------------------------------------------------------- #
# Enums
# --------------------------------------------------------------------------- #
class TaskStatus(str, Enum):
    CREATED = "created"
    PLANNING = "planning"
    RUNNING = "running"
    WAITING_FOR_APPROVAL = "waiting_for_approval"
    RECOVERABLE_FAILURE = "recoverable_failure"
    VERIFYING = "verifying"
    BLOCKED = "blocked"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


TERMINAL_STATES = {TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED}


class RiskClass(str, Enum):
    """Default action classes, spec section 12.1."""
    READ_ONLY = "read_only"
    REVERSIBLE_LOCAL = "reversible_local"
    EXTERNAL_SIDE_EFFECT = "external_side_effect"
    HIGH_IMPACT = "high_impact"
    PROHIBITED = "prohibited"


class Decision(str, Enum):
    ALLOW = "allow"
    DENY = "deny"
    APPROVAL_REQUIRED = "approval_required"


class ToolStatus(str, Enum):
    SUCCESS = "success"
    PARTIAL = "partial"
    FAILURE = "failure"
    DENIED = "denied"
    UNKNOWN = "unknown"


class OutcomeStatus(str, Enum):
    """Side-effect outcome in the authoritative tool log (GPT section 29.2)."""
    CONFIRMED_SUCCEEDED = "confirmed_succeeded"
    CONFIRMED_FAILED = "confirmed_failed"
    UNKNOWN = "unknown"


class CriterionStatus(str, Enum):
    PASS = "pass"
    FAIL = "fail"
    BLOCKED = "blocked"
    UNVERIFIED = "unverified"


# --------------------------------------------------------------------------- #
# Task / verification
# --------------------------------------------------------------------------- #
@dataclass
class Criterion:
    """One acceptance criterion, checked deterministically (spec section 13)."""
    id: str
    kind: str                      # "exit_code" | "file_exists" | "schema_valid"
    params: dict[str, Any] = field(default_factory=dict)
    required: bool = True
    description: str = ""


@dataclass
class Task:
    objective: str
    acceptance_criteria: list[Criterion] = field(default_factory=list)
    mandatory_constraints: list[str] = field(default_factory=list)
    scope: str = ""
    deadline_s: float | None = None          # seconds from run start; None = none
    token_budget: int | None = None          # cumulative input-token budget; None = unbounded
    max_steps: int = 20


@dataclass
class CriterionResult:
    id: str
    status: CriterionStatus
    evidence: str = ""
    required: bool = True


@dataclass
class VerificationReport:
    results: list[CriterionResult] = field(default_factory=list)

    @property
    def all_required_passed(self) -> bool:
        required = [r for r in self.results if r.required]
        return bool(required) and all(r.status == CriterionStatus.PASS for r in required)


# --------------------------------------------------------------------------- #
# Policy
# --------------------------------------------------------------------------- #
@dataclass
class Action:
    """A concrete action the runtime is about to take through a tool."""
    name: str
    args: dict[str, Any] = field(default_factory=dict)
    target: str | None = None
    op_id: str = ""                # stable id for logging / idempotency


@dataclass
class AuthorizationDecision:
    decision: Decision
    reason: str = ""
    action_digest: str = ""        # binds the approval to exact action+args (REQ 010)
    approval_id: str | None = None
    expires_at: float | None = None

    @property
    def allowed(self) -> bool:
        return self.decision == Decision.ALLOW


# --------------------------------------------------------------------------- #
# Tools
# --------------------------------------------------------------------------- #
@dataclass
class ToolDefinition:
    name: str
    arg_schema: dict[str, str] = field(default_factory=dict)   # name -> "str"|"int"|"float"|"bool"
    risk: RiskClass = RiskClass.READ_ONLY
    idempotent: bool = True
    output_limit: int = 4000       # bytes of raw output before the envelope bounds it
    description: str = ""


@dataclass
class ToolResult:
    """Normalized bounded envelope, spec section 9.2."""
    status: ToolStatus
    summary: str = ""
    data: dict[str, Any] = field(default_factory=dict)
    artifact_ref: str | None = None
    truncated: bool = False
    omitted_count: int | None = None     # None means unknown; never silently 0 (REQ 003)
    continuation_ref: str | None = None
    provenance: dict[str, Any] = field(default_factory=dict)


# --------------------------------------------------------------------------- #
# Model
# --------------------------------------------------------------------------- #
@dataclass
class UsageRecord:
    """Honest usage. Unknown stays None; estimated and reported are separate (REQ 008)."""
    input_tokens_reported: int | None = None
    output_tokens_reported: int | None = None
    input_tokens_estimated: int | None = None
    output_tokens_estimated: int | None = None
    cost_usd: float | None = None
    model: str | None = None


@dataclass
class ToolCall:
    name: str
    args: dict[str, Any] = field(default_factory=dict)
    id: str = ""


@dataclass
class ModelRequest:
    system: str = ""
    messages: list[dict[str, Any]] = field(default_factory=list)
    tools: list[ToolDefinition] = field(default_factory=list)
    max_output_tokens: int = 1024
    options: dict[str, Any] = field(default_factory=dict)


@dataclass
class ModelResponse:
    content: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    finish_reason: str = "stop"           # "stop" | "tool_calls" | "length" | ...
    usage: UsageRecord = field(default_factory=UsageRecord)
    # What the provider reported dropping/rewriting on input, or None = unknown (REQ 020).
    input_transformations: Any | None = None


# --------------------------------------------------------------------------- #
# Protocols (substitutable implementations)
# --------------------------------------------------------------------------- #
@runtime_checkable
class ModelAdapter(Protocol):
    def generate(self, request: ModelRequest) -> ModelResponse: ...


@runtime_checkable
class Tool(Protocol):
    definition: ToolDefinition
    def execute(self, args: dict[str, Any], ctx: "ToolContext") -> Any: ...


@runtime_checkable
class PolicyEngine(Protocol):
    def authorize(self, action: Action) -> AuthorizationDecision: ...


@runtime_checkable
class TokenEstimator(Protocol):
    def estimate(self, text: str) -> int: ...


@runtime_checkable
class Verifier(Protocol):
    def evaluate(self, task: "Task", extra: dict[str, Any] | None = None) -> VerificationReport: ...


@runtime_checkable
class Retriever(Protocol):
    # Returns a list of evidence documents (``retrieval.Document``), most relevant first.
    def retrieve(self, query: str, k: int = 5) -> list[Any]: ...


@dataclass
class ToolContext:
    task_id: str
    state_dir: Any                      # pathlib.Path
    authorization: AuthorizationDecision | None = None
    deadline_s: float | None = None

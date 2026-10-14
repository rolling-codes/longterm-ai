"""longterm_ai -- a model-independent, retrieval-first framework for reliable,
resource-aware, long-running AI work.

Phase 0 + Phase 1 core: typed contracts, fail-closed policy, bounded tool gateway,
honest usage accounting, deterministic verifier, task state machine with side-effect
reconciliation, and provider-independent handoff.

Phase 2 additions (ADR 0002): automatic checkpoint resume wired into Framework.run(),
real provider adapters (OpenAI, Anthropic, Gemini), SQLite checkpoint backend, and
a CLI entry point (``longterm`` / ``python -m longterm_ai``).
"""
from __future__ import annotations

from . import errors, providers
from .adapters import EchoModel, FakeModel, ScriptedTurn
from .contracts import (
    Action, AuthorizationDecision, Criterion, CriterionResult, CriterionStatus, Decision,
    ModelAdapter, ModelRequest, ModelResponse, OutcomeStatus, PolicyEngine, RiskClass, Task,
    TaskStatus, Tool, ToolCall, ToolContext, ToolDefinition, ToolResult, ToolStatus,
    TokenEstimator, UsageRecord, Verifier, VerificationReport,
)
from .policy import Approval, ApprovalStore, RuleBasedPolicy, action_digest, fail_closed
from .runtime import (
    Checkpoint, CheckpointStore, Framework, RunResult, TaskStateMachine, ToolLog, ToolLogEntry,
    changed_state_mismatches, claim_accuracy, reconcile_changed_state, should_retry,
)
from .sqlite_store import SqliteCheckpointStore
from .telemetry import EventLog, RoughTokenEstimator, UsageAccountant, rough_tokens
from .tools import ArtifactStore, FunctionTool, ToolGateway, ToolRegistry, generic_reducer, shell_reducer
from .verification import DeterministicVerifier

__version__ = "0.2.0"

__all__ = [
    "Framework", "RunResult", "Task", "Criterion",
    "RuleBasedPolicy", "Approval", "ApprovalStore", "action_digest", "fail_closed",
    "ToolRegistry", "ToolGateway", "FunctionTool", "ArtifactStore",
    "generic_reducer", "shell_reducer",
    "DeterministicVerifier", "VerificationReport", "CriterionResult", "CriterionStatus",
    "FakeModel", "EchoModel", "ScriptedTurn",
    "TaskStateMachine", "TaskStatus", "Checkpoint", "CheckpointStore", "SqliteCheckpointStore",
    "ToolLog", "ToolLogEntry", "reconcile_changed_state", "changed_state_mismatches", "claim_accuracy", "should_retry",
    "UsageAccountant", "UsageRecord", "EventLog", "RoughTokenEstimator", "rough_tokens",
    "Action", "AuthorizationDecision", "Decision", "RiskClass", "OutcomeStatus",
    "ModelAdapter", "ModelRequest", "ModelResponse", "ToolCall", "ToolContext",
    "ToolDefinition", "ToolResult", "ToolStatus", "Tool", "PolicyEngine", "TokenEstimator", "Verifier",
    "errors", "providers",
]

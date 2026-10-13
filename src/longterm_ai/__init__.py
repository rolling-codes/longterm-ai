"""longterm_ai -- a model-independent, retrieval-first framework for reliable,
resource-aware, long-running AI work.

This package implements the Phase 0 + Phase 1 core of PROJECT_FOUNDATION.md v0.1.0:
typed contracts, a fail-closed policy engine, a bounded tool gateway, honest usage
accounting, a deterministic verifier, a task state machine with side-effect
reconciliation, and the provider-independent handoff. See README.md for the explicit
"design intent vs implemented" boundary -- retrieval, SQLite memory, full recovery,
semantic checkers, and real provider adapters are deliberately not yet implemented.
"""
from __future__ import annotations

from . import errors
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
    changed_state_mismatches, reconcile_changed_state, should_retry,
)
from .telemetry import EventLog, RoughTokenEstimator, UsageAccountant, rough_tokens
from .tools import ArtifactStore, FunctionTool, ToolGateway, ToolRegistry, generic_reducer, shell_reducer
from .verification import DeterministicVerifier

__version__ = "0.1.0"

__all__ = [
    "Framework", "RunResult", "Task", "Criterion",
    "RuleBasedPolicy", "Approval", "ApprovalStore", "action_digest", "fail_closed",
    "ToolRegistry", "ToolGateway", "FunctionTool", "ArtifactStore",
    "generic_reducer", "shell_reducer",
    "DeterministicVerifier", "VerificationReport", "CriterionResult", "CriterionStatus",
    "FakeModel", "EchoModel", "ScriptedTurn",
    "TaskStateMachine", "TaskStatus", "Checkpoint", "CheckpointStore",
    "ToolLog", "ToolLogEntry", "reconcile_changed_state", "changed_state_mismatches", "should_retry",
    "UsageAccountant", "UsageRecord", "EventLog", "RoughTokenEstimator", "rough_tokens",
    "Action", "AuthorizationDecision", "Decision", "RiskClass", "OutcomeStatus",
    "ModelAdapter", "ModelRequest", "ModelResponse", "ToolCall", "ToolContext",
    "ToolDefinition", "ToolResult", "ToolStatus", "Tool", "PolicyEngine", "TokenEstimator", "Verifier",
    "errors",
]

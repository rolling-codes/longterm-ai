"""Typed error taxonomy. Spec section 15.

Rule: do not broadly catch exceptions and return apparent success. Every failure
states what is known, what is unknown, whether retry is safe, and the next step.
"""
from __future__ import annotations


class LongTermError(Exception):
    """Base for every framework error."""


# --- resource / context ---------------------------------------------------
class ContextOverflow(LongTermError):
    """Assembled context exceeds the usable input budget."""


class BudgetExhausted(LongTermError):
    """A task, retrieval, or output budget ran out."""


class ToolOutputTooLarge(LongTermError):
    """A tool produced more than its configured output limit. Caller must bound it."""


# --- policy / authorization ----------------------------------------------
class PolicyError(LongTermError):
    """A policy check could not be completed. Treated as deny (REQ 019)."""


class ApprovalDenied(LongTermError):
    """An action was denied, or its approval expired or no longer binds."""


# --- execution ------------------------------------------------------------
class ToolTimeout(LongTermError):
    """A tool call exceeded its deadline. Outcome is unknown until reconciled."""


class CancelledError(LongTermError):
    """Cooperative cancellation was requested and honored."""


class InvalidTransition(LongTermError):
    """A task state transition is not permitted by the state machine."""


class CheckpointError(LongTermError):
    """A checkpoint could not be saved or loaded; durability is not proven."""


class UnsupportedSchemaVersion(LongTermError):
    """A persisted record carries a schema version this code does not know (REQ 017)."""


# --- verification ---------------------------------------------------------
class VerificationFailed(LongTermError):
    """Required acceptance criteria did not pass (REQ 013)."""

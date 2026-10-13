"""Policy and approval engine. Spec section 12.

Decisions are made from the action's identity, its risk class, and configuration a
human owns -- never from model prose or retrieved text (REQ 018). Every decision is
made through :func:`fail_closed`, so an error, timeout, or invalid result counts as
*deny* (REQ 019). Approvals bind to the exact action and its arguments (REQ 010).
"""
from __future__ import annotations

import hashlib
import json
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeout
from dataclasses import dataclass, field

from .contracts import Action, AuthorizationDecision, Decision, PolicyEngine, RiskClass


def action_digest(action: Action) -> str:
    """Stable digest of the action and its canonical arguments."""
    canonical = json.dumps(
        {"name": action.name, "args": action.args, "target": action.target},
        sort_keys=True, ensure_ascii=False, default=str,
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


@dataclass
class Approval:
    """A human grant for one specific action."""
    approval_id: str
    action_digest: str
    expires_at: float | None = None


class ApprovalStore:
    def __init__(self) -> None:
        self._by_id: dict[str, Approval] = {}

    def grant(self, approval: Approval) -> None:
        self._by_id[approval.approval_id] = approval

    def verify(self, action: Action, approval_id: str) -> bool:
        """True only if the grant exists, is unexpired, and binds to this exact action."""
        grant = self._by_id.get(approval_id)
        if grant is None:
            return False
        if grant.expires_at is not None and time.time() > grant.expires_at:
            return False
        return grant.action_digest == action_digest(action)  # changed args -> mismatch


class RuleBasedPolicy:
    """Maps a tool's risk class to a decision. Unknown tools default to high impact.

    ``risk_map``: tool name -> RiskClass (wired from the tool registry).
    ``allow_writes``: permit reversible local changes without approval.
    ``allowed_external``: tool names whose external side effects are pre-authorized.
    """

    def __init__(
        self,
        risk_map: dict[str, RiskClass] | None = None,
        *,
        allow_writes: bool = True,
        allowed_external: set[str] | None = None,
    ) -> None:
        self.risk_map = dict(risk_map or {})
        self.allow_writes = allow_writes
        self.allowed_external = set(allowed_external or set())

    def classify(self, action: Action) -> RiskClass:
        # Unknown action -> treat as high impact (least privilege), not read-only.
        return self.risk_map.get(action.name, RiskClass.HIGH_IMPACT)

    def authorize(self, action: Action) -> AuthorizationDecision:
        risk = self.classify(action)
        digest = action_digest(action)
        if risk == RiskClass.READ_ONLY:
            return AuthorizationDecision(Decision.ALLOW, "read-only", digest)
        if risk == RiskClass.REVERSIBLE_LOCAL:
            if self.allow_writes:
                return AuthorizationDecision(Decision.ALLOW, "writes permitted", digest)
            return AuthorizationDecision(Decision.DENY, "writes not permitted", digest)
        if risk == RiskClass.EXTERNAL_SIDE_EFFECT:
            if action.name in self.allowed_external:
                return AuthorizationDecision(Decision.ALLOW, "external pre-authorized", digest)
            return AuthorizationDecision(Decision.APPROVAL_REQUIRED, "external side effect", digest)
        if risk == RiskClass.HIGH_IMPACT:
            return AuthorizationDecision(Decision.APPROVAL_REQUIRED, "high impact", digest)
        return AuthorizationDecision(Decision.DENY, "prohibited", digest)


def fail_closed(
    policy: PolicyEngine, action: Action, *, timeout_s: float | None = None
) -> AuthorizationDecision:
    """Authorize through a policy, but treat any failure as deny (REQ 019).

    Covers the three failure modes the spec names: the check raises, it times out,
    or it returns something that is not a valid decision.
    """
    try:
        if timeout_s is None:
            result = policy.authorize(action)
        else:
            with ThreadPoolExecutor(max_workers=1) as ex:
                result = ex.submit(policy.authorize, action).result(timeout=timeout_s)
    except FutureTimeout:
        return AuthorizationDecision(Decision.DENY, "policy check timed out", action_digest(action))
    except Exception as exc:  # noqa: BLE001 - any failure is a deny by design
        return AuthorizationDecision(Decision.DENY, f"policy check errored: {exc}", action_digest(action))
    if not isinstance(result, AuthorizationDecision) or not isinstance(result.decision, Decision):
        return AuthorizationDecision(Decision.DENY, "policy returned invalid output", action_digest(action))
    return result

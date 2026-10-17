"""Multi-agent handoff: provider-agnostic agents hand work across a boundary.

Agent A (one model/provider) runs a task through the framework, producing an
authoritative tool log and a provider-neutral handoff record. The handoff is REVIEWED
(layer one + claim accuracy against the log), then REBUILT for Agent B — shown for both
an Anthropic (Claude) agent and an OpenAI (GPT) agent — so the SAME state continues on a
different provider without resending the first agent's private reasoning.

Runs fully offline on FakeModel. If a provider SDK + credentials are present, the real
adapter is constructed to show the wiring (no API call is made here).

Run:  python examples/multi_agent_handoff.py
"""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

from longterm_ai import (
    Criterion, DeterministicVerifier, FakeModel, Framework, FunctionTool, RiskClass,
    RuleBasedPolicy, ScriptedTurn, Task, ToolDefinition, ToolLog, ToolRegistry,
    changed_state_mismatches, claim_accuracy, reconcile_changed_state,
)
from longterm_ai.contracts import ToolCall
from longterm_ai.handoff import HandoffRecord, layer_one, rebuild, to_anthropic, to_openai_chat

OBJECTIVE = "Apply the retry patch to client/fetch.py and confirm the file exists."
CONSTRAINTS = ["Do not modify files outside client/.", "Do not push or deploy."]


def probe_agents() -> list[tuple[str, str]]:
    """Report which provider agents can be wired. Constructs the real adapter when the
    SDK + credentials are available; never calls the API. FakeModel is always available."""
    out: list[tuple[str, str]] = [("FakeModel", "ready (offline, deterministic)")]
    probes = [
        ("OpenAIAdapter", lambda: __import__("openai").OpenAI()),
        ("AnthropicAdapter", lambda: __import__("anthropic").Anthropic()),
        ("GeminiAdapter", lambda: __import__("google.generativeai", fromlist=["x"]).GenerativeModel("gemini-2.5-pro")),
    ]
    for name, make_client in probes:
        try:
            from longterm_ai import providers  # noqa: F401
            client = make_client()                       # may raise: no SDK, or no key
            adapter_cls = getattr(__import__("longterm_ai.providers", fromlist=[name]), name)
            adapter_cls(client)                          # construct; no call made
            out.append((name, "real adapter ready"))
        except Exception as exc:                         # noqa: BLE001 - report, do not fail
            out.append((name, f"not configured ({type(exc).__name__})"))
    return out


def agent_a(state: Path) -> tuple[object, ToolLog]:
    """First agent: apply a patch (a reversible-local side effect), verify, complete."""
    (state / "client").mkdir(parents=True, exist_ok=True)

    def apply_patch(args: dict, ctx) -> str:
        target = state / args["path"]
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("def fetch(): ...  # retry loop added\n", encoding="utf-8")
        return f"patched {args['path']} (+1 retry loop)"

    reg = ToolRegistry()
    reg.register(FunctionTool(
        ToolDefinition(name="apply_patch", arg_schema={"path": "str"},
                       risk=RiskClass.REVERSIBLE_LOCAL, idempotent=True),
        apply_patch,
    ))
    task = Task(
        objective=OBJECTIVE,
        mandatory_constraints=list(CONSTRAINTS),
        acceptance_criteria=[Criterion("patched", "file_exists", {"path": "client/fetch.py"})],
        max_steps=4,
    )
    model = FakeModel([
        ScriptedTurn(tool_calls=[ToolCall(name="apply_patch", args={"path": "client/fetch.py"}, id="op1")]),
        ScriptedTurn(content="Patch applied; client/fetch.py now has the retry loop."),
    ])
    fw = Framework(model, reg, RuleBasedPolicy(reg.risk_map()),
                   DeterministicVerifier(base_dir=state), state_dir=state)
    result = fw.run(task)
    log = ToolLog.from_list(result.tool_log)
    return result, log


def review(record: HandoffRecord, task: dict, log: ToolLog, state: Path) -> bool:
    reconciled = reconcile_changed_state(log)
    acc = claim_accuracy(record.changed_state, reconciled)
    mism = changed_state_mismatches(record.changed_state, reconciled)
    check = layer_one(task, record, state, 1500)
    safe = check.passed and not mism["unbacked_claims"]
    print(f"    layer one       : {'PASSED' if check.passed else 'FAILED'}")
    print(f"    claim accuracy  : {'n/a' if acc is None else f'{acc:.3f}'}")
    if mism["unbacked_claims"]:
        print(f"    unbacked claims : {mism['unbacked_claims']}")
    print(f"    verdict         : {'SAFE' if safe else 'UNSAFE'}")
    return safe


def main() -> None:
    state = Path(tempfile.mkdtemp(prefix="lt_multiagent_"))
    checks: list[bool] = []

    print("=" * 68)
    print("MULTI-AGENT HANDOFF -- provider-agnostic")
    print("=" * 68)
    print("\n[agents] provider adapters that can drive the framework:")
    for name, status in probe_agents():
        print(f"    {name:<16}: {status}")

    # --- Agent A does the work ------------------------------------------------
    print("\n[1] Agent A runs the task")
    result, log = agent_a(state)
    print(f"    status          : {result.status.value}  verified={result.verified_complete}")
    print(f"    changed_state   : {result.changed_state}  (reconciled from the log)")
    checks.append(result.verified_complete and len(result.changed_state) == 1)

    task = {"goal": OBJECTIVE, "mandatory_constraints": list(CONSTRAINTS)}

    # --- Honest handoff: claims match the log -> SAFE -------------------------
    print("\n[2] Review Agent A's honest handoff")
    honest = HandoffRecord(
        goal=OBJECTIVE, success_test="client/fetch.py exists with the retry loop",
        mandatory_constraints=list(CONSTRAINTS), decisions=[], rejected_options=[],
        facts=[], open_questions=[], next_action="Hand off to Agent B to run the test suite.",
        changed_state=list(result.changed_state),
    )
    checks.append(review(honest, task, log, state) is True)

    # --- Dishonest handoff: an extra fabricated side effect -> UNSAFE ---------
    print("\n[3] Review a handoff that over-claims a side effect")
    dishonest = HandoffRecord(
        goal=OBJECTIVE, success_test="client/fetch.py exists",
        mandatory_constraints=list(CONSTRAINTS), decisions=[], rejected_options=[],
        facts=[], open_questions=[], next_action="Continue.",
        changed_state=list(result.changed_state) + ["deploy: shipped v9.9.9 to prod"],
    )
    checks.append(review(dishonest, task, log, state) is False)

    # --- Agent B continues from the SAME handoff, on either provider ----------
    print("\n[4] Rebuild context for Agent B (same state, different provider)")
    transcript = {
        "system": "You are completing a long-running task through a framework.",
        "first_user_message": OBJECTIVE,
        "turns": [
            {"role": "user", "parts": [{"type": "text", "text": OBJECTIVE}]},
            {"role": "assistant", "parts": [
                {"type": "reasoning", "text": "PRIVATE chain of thought that must never ship."},
                {"type": "tool_call", "id": "op1", "name": "apply_patch", "state_ref": "durable state"},
            ]},
            {"role": "user", "parts": [
                {"type": "tool_result", "call_id": "op1",
                 "summary": result.changed_state[0], "state_ref": "durable state"},
            ]},
        ],
    }
    neutral = rebuild(transcript, honest, task, keep_last_user_turns=1)
    blob = json.dumps(neutral)
    reasoning_dropped = "PRIVATE chain of thought" not in blob
    constraints_in_system = all(c in neutral["system"] for c in CONSTRAINTS)
    claude_req = to_anthropic(neutral, "claude-opus-4-5", 1024)
    gpt_req = to_openai_chat(neutral, "gpt-4o")
    print(f"    reasoning carried forward : {not reasoning_dropped}  (must be False)")
    print(f"    constraints in system     : {constraints_in_system}")
    print(f"    -> Claude agent request   : model={claude_req['model']}, {len(claude_req['messages'])} msgs")
    print(f"    -> GPT agent request      : role0={gpt_req['messages'][0]['role']}, {len(gpt_req['messages'])} msgs")
    checks.append(reasoning_dropped and constraints_in_system)

    print("\n" + "=" * 68)
    ok = sum(checks)
    print(f"RESULT: {ok}/{len(checks)} checks passed")
    print("=" * 68)
    if ok != len(checks):
        raise SystemExit(1)


if __name__ == "__main__":
    main()

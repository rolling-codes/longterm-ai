"""End-to-end demo: run one task through the framework with the fake model.

Shows the pieces working together: mandatory constraints carried as plain text, a
policy-gated tool call, a bounded result envelope with an artifact reference, the
verification gate, honest usage accounting, side-effect reconciliation, and a
provider-neutral handoff record built from the authoritative tool log.

Run:  python examples/demo.py
"""
from __future__ import annotations

import tempfile
from pathlib import Path

from longterm_ai import (
    Criterion, DeterministicVerifier, FakeModel, Framework, FunctionTool, RiskClass,
    RuleBasedPolicy, ScriptedTurn, Task, ToolDefinition, ToolRegistry,
)
from longterm_ai.contracts import ToolCall
from longterm_ai.handoff import HandoffRecord


def main() -> None:
    state = Path(tempfile.mkdtemp(prefix="longterm_demo_"))
    (state / "build.log").write_text("BUILD OK\n")   # pretend a build produced this

    # A read-only tool that returns a large log. output_limit forces the envelope to bound it.
    big_log = "\n".join(f"line {i}: compiling module_{i}.py ... ok" for i in range(200))
    read_log = FunctionTool(
        ToolDefinition(name="read_log", arg_schema={"path": "str"}, risk=RiskClass.READ_ONLY,
                       output_limit=300, description="Read a build log."),
        lambda args, ctx: big_log,
    )
    registry = ToolRegistry()
    registry.register(read_log)

    policy = RuleBasedPolicy(registry.risk_map())          # read-only -> allowed, push -> would need approval
    verifier = DeterministicVerifier(base_dir=state)

    task = Task(
        objective="Read the build log and confirm the build passed.",
        mandatory_constraints=["Do not modify files.", "Do not push or deploy."],
        acceptance_criteria=[Criterion("build_log_present", "file_exists", {"path": "build.log"})],
        max_steps=6,
    )

    model = FakeModel([
        ScriptedTurn(tool_calls=[ToolCall(name="read_log", args={"path": "build.log"}, id="op1")]),
        ScriptedTurn(content="The build log shows every module compiled; the build passed."),
    ])

    fw = Framework(model, registry, policy, verifier, state_dir=state)
    result = fw.run(task)

    print("=" * 70)
    print("TASK:", task.objective)
    print("mandatory constraints (sent as plain text every call):")
    for c in task.mandatory_constraints:
        print("   -", c)
    print("-" * 70)
    print(f"status            : {result.status.value}")
    print(f"verified complete : {result.verified_complete}")
    print(f"steps             : {result.steps}")
    print(f"note              : {result.note}")
    print("-" * 70)
    print("usage (honest: reported stays None with the fake model, estimated is real):")
    for k, v in result.usage.items():
        print(f"   {k:28}: {v}")
    print("-" * 70)
    print("bounded tool result envelope (the 200-line log never entered context in full):")
    entry = result.tool_log[0]
    print(f"   tool              : {entry['tool']}")
    print(f"   outcome           : {entry['result_status']}")
    print(f"   summary (to model): {entry['evidence']}")
    print("-" * 70)
    print(f"changed_state (reconciled from the tool log, not the model): {result.changed_state}")
    print("=" * 70)

    # A provider-neutral handoff record, with changed_state built from the log.
    record = HandoffRecord(
        goal=task.objective,
        success_test="build.log exists and reports success",
        mandatory_constraints=list(task.mandatory_constraints),
        decisions=[],
        rejected_options=[],
        facts=[],
        open_questions=[],
        next_action="None; task verified complete.",
        changed_state=result.changed_state,
    )
    print("HANDOFF RECORD (what a next model would receive as earlier working notes):")
    print(record.to_text())


if __name__ == "__main__":
    main()

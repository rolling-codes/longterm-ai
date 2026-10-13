"""Mini-benchmark: bounded framework path vs a naive dump-everything baseline.

One fixed task, two arms, same fake model and same verifier:

  * naive     -- the full raw tool output is injected into the model's context.
  * framework -- the tool gateway returns a bounded summary; the full log is kept
                 as a retrievable artifact instead of entering context.

We report estimated input tokens and verified success for each arm, so the token
reduction is a measured fixture result, not an asserted number. This is a single
synthetic fixture (spec section 16.4: targets are provisional); the baseline is
always printed alongside the candidate (REQ 025 / section 27.5). It is not a general
claim about any workload.

Run:  python benchmarks/naive_vs_framework.py
"""
from __future__ import annotations

import tempfile
from pathlib import Path

from longterm_ai import (
    Criterion, DeterministicVerifier, FakeModel, Framework, FunctionTool, RiskClass,
    RuleBasedPolicy, ScriptedTurn, Task, ToolDefinition, ToolRegistry, UsageAccountant, rough_tokens,
)
from longterm_ai.contracts import ModelRequest, ToolCall

# A realistic "big tool output": a long build/test log the model only needs a verdict from.
BIG_LOG = "\n".join(f"line {i}: ran check_{i} ... ok (0.0{i%10}s)" for i in range(400))


def make_task(state: Path) -> Task:
    (state / "build.log").write_text("BUILD OK\n")
    return Task(
        objective="Read the build log and confirm the build passed.",
        mandatory_constraints=["Do not modify files."],
        acceptance_criteria=[Criterion("log_present", "file_exists", {"path": "build.log"})],
        max_steps=6,
    )


def framework_arm(state: Path) -> tuple[int | None, bool]:
    read_log = FunctionTool(
        ToolDefinition(name="read_log", arg_schema={"path": "str"}, risk=RiskClass.READ_ONLY,
                       output_limit=300),
        lambda args, ctx: BIG_LOG,
    )
    reg = ToolRegistry()
    reg.register(read_log)
    model = FakeModel([
        ScriptedTurn(tool_calls=[ToolCall(name="read_log", args={"path": "build.log"}, id="op1")]),
        ScriptedTurn(content="Build passed."),
    ])
    fw = Framework(model, reg, RuleBasedPolicy(reg.risk_map()),
                   DeterministicVerifier(base_dir=state), state_dir=state)
    result = fw.run(make_task(state))
    return result.usage["input_tokens_estimated"], result.verified_complete


def naive_arm(state: Path) -> tuple[int | None, bool]:
    """A baseline loop that dumps the full tool output into the next model message."""
    task = make_task(state)
    verifier = DeterministicVerifier(base_dir=state)
    model = FakeModel([
        ScriptedTurn(tool_calls=[ToolCall(name="read_log", args={"path": "build.log"}, id="op1")]),
        ScriptedTurn(content="Build passed."),
    ])
    acct = UsageAccountant()
    system = "You are completing a task.\n" + "\n".join(task.mandatory_constraints)
    messages = [{"role": "user", "content": task.objective}]
    for _ in range(task.max_steps):
        resp = model.generate(ModelRequest(system=system, messages=messages))
        acct.add(resp.usage)
        if resp.tool_calls:
            messages.append({"role": "assistant", "content": "calling read_log"})
            # Naive: inject the ENTIRE raw tool output, unbounded, into context.
            messages.append({"role": "user", "content": f"[read_log output]\n{BIG_LOG}"})
            continue
        report = verifier.evaluate(task)
        return acct.input_tokens_estimated, report.all_required_passed
    return acct.input_tokens_estimated, False


def main() -> None:
    nv_state = Path(tempfile.mkdtemp(prefix="bench_naive_"))
    fw_state = Path(tempfile.mkdtemp(prefix="bench_fw_"))
    naive_tokens, naive_ok = naive_arm(nv_state)
    fw_tokens, fw_ok = framework_arm(fw_state)

    print("=" * 66)
    print("Mini-benchmark: naive dump vs bounded framework (single fixture)")
    print("=" * 66)
    print(f"full tool output size: ~{rough_tokens(BIG_LOG)} estimated tokens")
    print("-" * 66)
    print(f"{'arm':<12}{'verified success':<20}{'input tokens (est.)':<20}")
    print(f"{'naive':<12}{str(naive_ok):<20}{str(naive_tokens):<20}")
    print(f"{'framework':<12}{str(fw_ok):<20}{str(fw_tokens):<20}")
    print("-" * 66)
    if naive_tokens and fw_tokens:
        reduction = 100 * (naive_tokens - fw_tokens) / naive_tokens
        print(f"input-token reduction at equal verified success: {reduction:.1f}%")
    print("Provisional single-fixture result; not a general claim (spec 16.4).")
    print("=" * 66)


if __name__ == "__main__":
    main()

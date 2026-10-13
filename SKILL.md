---
name: longterm-ai
description: >
  Use when a task needs a long-running, multi-step agentic execution with any
  of: bounded tool output (prevent context flooding from large results), a
  fail-closed tool policy (every action authorized before it runs; errors = deny),
  honest token/cost accounting (unknown stays None, never 0), verified task
  completion (deterministic checks, not the model's own claim), or durable
  provider-neutral handoff (carry state across a model or context switch with
  side-effect reconciliation). Trigger phrases: "long-running agent task",
  "bound tool output", "fail-closed tool policy", "verify the task actually
  completed", "carry state across a model handoff", "honest token/cost
  accounting", "context flooding from tool results", "checkpoint and resume",
  "side-effect reconciliation". NOT for single-turn queries, simple scripting,
  or anything that doesn't involve a model executing multi-step tool calls.
allowed-tools: [Read, Grep, Glob, Bash, Edit, Write]
model: sonnet
---

## What it is

longterm-ai is a stdlib-only Python library that wraps around a model and
enforces five hard guarantees in code — not prompts:

1. **Bounded tool output** — large results are summarized + stored as
   retrievable artifacts; truncation is always marked; unknown omitted counts
   are never reported as 0.
2. **Fail-closed policy** — every tool call is risk-classified and authorized
   before execution; a policy error, timeout, or garbage response = deny.
3. **Honest usage accounting** — estimated vs. provider-reported tokens tracked
   separately; `None` for anything unknown, never a silent zero.
4. **Verified completion** — task ends only after deterministic acceptance
   criteria pass (exit code, file exists, schema valid); "the model said done"
   is not proof.
5. **Durable handoff** — provider-neutral checkpoint + side-effect
   reconciliation; what changed is derived from the authoritative tool log, not
   the model's claim.

Stdlib only. No runtime dependencies. Runs fully with the built-in `FakeModel`
— no API key or network needed.

## When to use it

- You're building or debugging an agent that runs tools over multiple turns and
  you need any of the five guarantees above enforced in code.
- A model's tool results are flooding context and you need bounded envelopes.
- You need a task to only declare completion after deterministic checks pass.
- You need to hand off a running agent across a context boundary (model swap,
  session end, provider change) without losing what changed or re-running
  non-idempotent actions.
- You need a policy that denies by default on any authorization uncertainty.

## When NOT to use it

- Single-turn tasks with no tool calls — no framework overhead needed.
- Simple scripts that call a model once — direct API call is enough.
- Tasks where you only need one of the five guarantees and a lighter approach
  (e.g., just `@lru_cache`) covers it.

## Module map

| Module | Guarantee |
|---|---|
| `tools.py` | Bounded tool output — envelope, reducers, artifact store |
| `policy.py` | Fail-closed authorization, risk classes, approval binding |
| `telemetry.py` | Honest usage accounting, cost, events |
| `verification.py` | Deterministic verified completion |
| `runtime.py` | State machine, run loop, checkpoints + resume, side-effect reconciliation |
| `sqlite_store.py` | SQLite checkpoint storage (atomic, concurrent-safe, stdlib only) |
| `providers/` | OpenAI, Anthropic, Gemini adapters satisfying `ModelAdapter` |
| `cli.py` | CLI entry point (`longterm verify`, `rebuild`, `run`) |
| `handoff/` | Provider-neutral durable handoff (spec 22–25) |
| `contracts.py` | Shared typed dataclasses + protocols |
| `adapters.py` | `FakeModel` + `EchoModel` for tests without a provider |

## Minimal usage recipe

```python
from longterm_ai import (
    Framework, Task, Criterion, ToolRegistry, FunctionTool, ToolDefinition,
    RiskClass, RuleBasedPolicy, DeterministicVerifier, FakeModel, ScriptedTurn,
)
from longterm_ai.contracts import ToolCall

reg = ToolRegistry()
reg.register(FunctionTool(
    ToolDefinition("read_log", {"path": "str"}, risk=RiskClass.READ_ONLY),
    lambda args, ctx: open(args["path"]).read(),
))
task = Task(
    objective="Confirm the build passed.",
    mandatory_constraints=["Do not modify files."],
    acceptance_criteria=[Criterion("ok", "file_exists", {"path": "build.log"})],
)
model = FakeModel([
    ScriptedTurn(tool_calls=[ToolCall("read_log", {"path": "build.log"}, "op1")]),
    ScriptedTurn(content="Build passed."),
])
fw = Framework(model, reg, RuleBasedPolicy(reg.risk_map()),
               DeterministicVerifier(base_dir="."), state_dir=".longterm_state")
result = fw.run(task)
print(result.status, result.verified_complete, result.usage)
```

`result.verified_complete` is `True` only when acceptance criteria passed.
`result.usage` is `None` for any unknown quantity — never a silent zero.

## Install

```bash
python -m pip install -e .          # Python 3.12+, no runtime deps
python -m unittest discover tests   # 115 tests, all stdlib
python examples/demo.py             # end-to-end verified run
```

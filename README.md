# longterm-ai

A model-independent, retrieval-first framework for **reliable, resource-aware,
long-running AI work**. The model proposes and reasons; the framework enforces policy
in code, bounds tool output, accounts for cost honestly, verifies completion with
deterministic checks, and carries state across a model handoff.

This repository implements the **Phase 0, Phase 1, and Phase 2** of the
PROJECT_FOUNDATION design spec. It is a real, installable,
fully-tested library — **stdlib only, no runtime dependencies** — that runs end to end
with built-in fake models or connects to real providers (OpenAI, Anthropic, Gemini).

> Honesty note (spec rule): a design document is not an implementation. This README
> marks what is implemented and tested versus what is deliberately deferred. Nothing
> is claimed `IMPLEMENTED` without code and a passing test.

## Install & run

```bash
python -m pip install -e .          # Python 3.12+
python -m unittest discover tests   # 115 tests, all stdlib
python examples/demo.py             # end-to-end task run, verified complete
python benchmarks/naive_vs_framework.py   # measured token saving on a fixture
longterm --help                     # CLI commands (verify, rebuild, run)
```

## What it does (the five things that make AI work cheaper and safer)

1. **Bounded tool output** — large results never get dumped into context. The gateway
   returns a summary + completeness metadata and keeps the full output as a retrievable
   artifact. Truncation is always marked; an unknown omitted count is never reported as 0.
2. **Policy enforced in code, fail-closed** — every action is classified by risk and
   authorized before it runs. A policy check that errors, times out, or returns garbage
   counts as *deny*. Approvals bind to the exact action and its arguments.
3. **Honest usage accounting** — estimated and provider-reported tokens are tracked
   separately; unknown stays `None`, never silently 0; an incomplete total is unknown.
4. **Verified completion** — a model saying "done" is not proof. The task completes only
   after deterministic acceptance checks (command exit code, file exists, schema valid)
   pass.
5. **Durable continuity** — a provider-neutral handoff record plus side-effect
   reconciliation: what changed is derived from the authoritative tool log, never from
   the model's own claim, so an unknown non-idempotent action is not blindly repeated.

## Quick example

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

## Layout

```
src/longterm_ai/
  contracts.py      typed dataclasses + protocols (Task, ToolResult, UsageRecord, ...)
  errors.py         typed error taxonomy (spec section 15)
  telemetry.py      usage accounting, cost, events, token estimator
  policy.py         risk classes, fail-closed authorization, approval binding
  tools.py          registry + gateway + bounded envelope + reducers + artifact store
  verification.py   deterministic verifier (exit code / file exists / schema valid)
  runtime.py        state machine, run loop, checkpoints + resume, side-effect reconciliation
  sqlite_store.py   SQLite checkpoint store (atomic, concurrent-safe, stdlib only)
  cli.py            CLI interface (verify, rebuild, run)
  adapters.py       FakeModel + EchoModel (no provider, no key)
  providers/        real provider adapters (OpenAI, Anthropic, Gemini)
  handoff/          provider-neutral handoff with a checked summary (spec 22-25)
tests/              115 stdlib unittest tests
examples/demo.py    end-to-end run
benchmarks/         naive-vs-framework token comparison
docs/decisions/     decision records (0001-scope.md, 0002-phase2.md)
```

## Requirement → test map

| REQ | What | Where / test |
|---|---|---|
| 002 | tool exec schema-validated, bounded, under policy | `tools.py` · `test_tools` |
| 003 | oversized output reduced with completeness metadata | `tools.py` · `test_tools` (truncation, unknown-omitted) |
| 007 | mandatory constraints never silently dropped | `runtime.mandatory_present`, `handoff.layer_one` · `test_runtime`, `test_handoff` |
| 008 | usage honest; unknown never 0 | `telemetry.py` · `test_telemetry` |
| 009 / 013 | acceptance criteria; no verified-complete until they pass | `verification.py`, `runtime` · `test_verification`, `test_runtime` |
| 010 | approval bound to the exact action | `policy.ApprovalStore` · `test_policy` |
| 011 / 017 | checkpoints rebuildable; schema-versioned records | `runtime.CheckpointStore`, `sqlite_store.py`, `handoff.record` · `test_runtime`, `test_sqlite_store`, `test_handoff` |
| 012 | retries bounded; account for side effects | `runtime.should_retry` · `test_runtime` |
| 015 | core tested without a provider | whole suite runs on `FakeModel` |
| 016 | every state transition validated | `runtime.TaskStateMachine` · `test_runtime` |
| 019 | policy error/timeout/invalid = deny | `policy.fail_closed` · `test_policy` |
| 021 | mandatory content sent as plain text; survival checked | `runtime.build_system` · `test_runtime` |
| GPT 29.2 / Grok 31.6 | side-effect truth from the log, not the model | `runtime.reconcile_changed_state` · `test_runtime`, `test_handoff` |

## Implemented vs deferred

**Implemented and tested (Phase 0 + 1 + 2):** contracts, error taxonomy, fail-closed policy + approvals,
tool gateway with bounded envelopes + reducers + artifact store, deterministic verifier,
usage/cost accounting + events, task state machine, JSON schema-versioned checkpoints,
authoritative tool log + side-effect reconciliation + retry guard, fake/echo adapters,
provider-neutral handoff (spec sections 22–25), automatic checkpoint resume (`resume=True`),
SQLite checkpoint backend (`SqliteCheckpointStore`), real provider adapters (`OpenAIAdapter`,
`AnthropicAdapter`, `GeminiAdapter`), and CLI (`longterm`).

**Deliberately deferred** (Phase 3+ design intent): retrieval/search engine ("retrieval-first" engine),
semantic handoff checker layers 2–3 (need a real model evaluator), and expanded benchmarks. See `docs/decisions/0001-scope.md` and `0002-phase2.md`.

## Provenance

Built from the `long_term_ai_framework` design spec and its multi-provider review
overlay (Claude, GPT-6, Grok). The `handoff/` subpackage is a faithful port of the
overlay's section-25 prototype, extended with the side-effect reconciliation that every
reviewer flagged as the missing piece. License: MIT.

# longterm-ai

A model-independent framework for **reliable, resource-aware, long-running AI work**.
The model proposes and reasons; the framework enforces policy in code, bounds tool output,
accounts for cost honestly, verifies completion with deterministic checks, persists state
with atomic checkpoints, and carries durable context across model handoffs.

This repository implements **Phase 0, Phase 1, and Phase 2** of the design specification.
It is an installable, fully-tested library (**Python 3.12+**, zero runtime dependencies for core)
that runs out-of-the-box with built-in fake adapters or connects to real providers
(OpenAI, Anthropic Claude, Google Gemini).

> **Honesty invariant:** A design document is not an implementation. This README reflects
> what is built, tested, and passing. Nothing is claimed implemented without code and passing tests.

---

## Key Guarantees

1. **Bounded Tool Output** — Large results never flood model context. The gateway generates
   a summary with completeness metadata and archives raw outputs to a durable artifact store.
   Truncation is always flagged; unknown omitted counts are never reported as 0 (REQ 003).
2. **Fail-Closed Policy** — Actions are classified by risk and authorized *before* execution.
   Policy errors, timeouts, or malformed decisions default to `DENY` (REQ 019). Approvals bind
   strictly to the exact action and arguments (REQ 010).
3. **Honest Usage Accounting** — Provider-reported tokens and local estimates are tracked
   separately; unknown values remain `None` rather than silent zeroes (REQ 008). Incomplete totals
   propagate as unknown.
4. **Deterministic Verified Completion** — A model's self-assertion that a task is "done" is
   not proof. Tasks reach `COMPLETED` status only after deterministic acceptance checks
   (command exit code, file existence, JSON schema validation) pass (REQ 009 / 013).
5. **Durable Continuity & Checkpoint Resume** — Checkpoints are atomically persisted after every
   step. The run loop supports seamless recovery (`resume=True`), and state reconciliation derives
   what changed from the authoritative tool execution log—never model claims.
6. **Provider Neutrality** — Switch models or providers across task stages with provider-neutral
   handoff records, context rebuilders, and adapter implementations for OpenAI, Anthropic, and Gemini.

---

## Install & Quick Start

### Installation

```bash
python -m pip install -e .          # Python 3.12+, stdlib only for core
python -m unittest discover tests   # 118 tests, all stdlib
```

### End-to-End Demo & Benchmarks

```bash
python examples/demo.py                   # Run verified task with bounded envelope demo
python benchmarks/naive_vs_framework.py   # Measure token savings against naive context dumps (~94.8% reduction)
```

---

## Python API Examples

### 1. Basic Verified Task Execution (Offline / Fake Model)

```python
from longterm_ai import (
    Framework, Task, Criterion, ToolRegistry, FunctionTool, ToolDefinition,
    RiskClass, RuleBasedPolicy, DeterministicVerifier, FakeModel, ScriptedTurn,
)
from longterm_ai.contracts import ToolCall

# 1. Define tools with risk classification and output limits
registry = ToolRegistry()
registry.register(FunctionTool(
    ToolDefinition("read_log", {"path": "str"}, risk=RiskClass.READ_ONLY, output_limit=500),
    lambda args, ctx: open(args["path"]).read(),
))

# 2. Define task with mandatory constraints and deterministic criteria
task = Task(
    objective="Confirm the build succeeded.",
    mandatory_constraints=["Do not modify files.", "Do not push to remote."],
    acceptance_criteria=[Criterion("log_exists", "file_exists", {"path": "build.log"})],
    max_steps=5,
)

# 3. Model proposes actions; verifier checks real world
model = FakeModel([
    ScriptedTurn(tool_calls=[ToolCall("read_log", {"path": "build.log"}, "op1")]),
    ScriptedTurn(content="Build completed successfully."),
])

fw = Framework(
    model=model,
    tools=registry,
    policy=RuleBasedPolicy(registry.risk_map()),
    verifier=DeterministicVerifier(base_dir="."),
    state_dir=".longterm_state",
)

result = fw.run(task)
print(f"Status: {result.status.value}")                  # "completed"
print(f"Verified Complete: {result.verified_complete}")  # True
print(f"Usage summary: {result.usage}")
```

### 2. Automatic Checkpoint Resume

When a task is interrupted, resume execution from the last saved turn:

```python
# Pass resume=True to reload step counts, tool history, and state machine status:
result = fw.run(task, task_id="build_job_42", resume=True)
```

### 3. SQLite Checkpoint Storage

For atomic, cross-process durability with WAL mode:

```python
from longterm_ai import SqliteCheckpointStore, Checkpoint

with SqliteCheckpointStore(".longterm_state/checkpoints.db") as store:
    store.save(Checkpoint(task_id="job_1", status="running", completed_steps=3))
    cp = store.load("job_1")
    tasks = store.list_tasks()
```

### 4. Using Real Provider Adapters

Adapters satisfy the `ModelAdapter` protocol and duck-type client libraries:

```python
# OpenAI
from openai import OpenAI
from longterm_ai.providers import OpenAIAdapter
model = OpenAIAdapter(OpenAI(), model="gpt-4o")

# Anthropic Claude
from anthropic import Anthropic
from longterm_ai.providers import AnthropicAdapter
model = AnthropicAdapter(Anthropic(), model="claude-3-7-sonnet-latest")

# Google Gemini
import google.generativeai as genai
from longterm_ai.providers import GeminiAdapter
model = GeminiAdapter(genai.GenerativeModel("gemini-2.5-pro"))

# Plug any adapter straight into Framework:
fw = Framework(model, registry, policy, verifier, state_dir=".state")
```

---

## Command-Line Interface (`longterm`)

The library includes a command-line interface for CI automation, scripting, and offline verification:

```bash
# Display help
longterm --help

# 1. Run layer-one handoff verification between context switches
longterm verify task.json record.json ./state_dir --max-tokens 1500

# 2. Rebuild context for a receiving model (Anthropic, OpenAI, or Neutral format)
longterm rebuild task.json transcript.json record.json ./state_dir --format anthropic

# 3. Execute a task using FakeModel for CI/testing without credentials
longterm run task.json --state-dir .longterm_state --resume
```

---

## Architecture & Layout

```
src/longterm_ai/
  contracts.py      Typed dataclasses & protocols (Task, ToolResult, UsageRecord, etc.)
  errors.py         Typed error taxonomy (spec section 15)
  telemetry.py      Usage accounting, honest cost reporting, and structured event log
  policy.py         Risk classes, fail-closed authorization, approval binding
  tools.py          Registry, gateway, bounded envelopes, reducers, artifact store
  verification.py   Deterministic verifier (exit code, file existence, JSON schema)
  runtime.py        Task state machine, run loop, checkpoints + resume, side-effect reconciliation
  sqlite_store.py   SQLite checkpoint store (atomic transactions, WAL mode, stdlib only)
  cli.py            CLI interface for verify, rebuild, and run
  adapters.py       Deterministic FakeModel and EchoModel for tests without a provider
  providers/        Real provider adapters (OpenAI, Anthropic Claude, Google Gemini)
  handoff/          Provider-neutral handoff with layer-one verification checks
tests/              118 stdlib unittest tests (100% green without network or provider keys)
examples/demo.py    End-to-end task execution walkthrough
benchmarks/         Single-fixture context size comparison (~94.8% reduction)
docs/decisions/     Architecture Decision Records (0001-scope.md, 0002-phase2.md)
```

---

## Requirement → Test Map

| REQ | Invariant / Guarantee | Code & Tests |
|---|---|---|
| **002** | Tool execution schema-validated, bounded, under policy | `tools.py` · `test_tools` |
| **003** | Oversized output reduced with completeness metadata | `tools.py` · `test_tools` (truncation, unknown-omitted) |
| **007** | Mandatory constraints never silently dropped | `runtime.mandatory_present`, `handoff.layer_one` · `test_runtime`, `test_handoff` |
| **008** | Usage accounting honest; unknown never coerced to 0 | `telemetry.py` · `test_telemetry`, `test_providers` |
| **009 / 013** | Acceptance criteria; no completion until verified | `verification.py`, `runtime.py` · `test_verification`, `test_runtime` |
| **010** | Approvals cryptographically/digest bound to exact action | `policy.ApprovalStore` · `test_policy` |
| **011 / 017** | Checkpoints rebuildable; schema-versioned records | `runtime.CheckpointStore`, `sqlite_store.py`, `handoff.record` · `test_runtime`, `test_sqlite_store`, `test_handoff` |
| **012** | Retries bounded; account for side effects | `runtime.should_retry` · `test_runtime` |
| **015** | Core runnable and testable without a provider key | Complete test suite runs on `FakeModel` |
| **016** | Strict task state machine validation | `runtime.TaskStateMachine` · `test_runtime` |
| **019** | Policy errors, timeouts, or invalid formats fail closed (`DENY`) | `policy.fail_closed` · `test_policy` |
| **021** | Mandatory content sent as plain text; survival verified | `runtime.build_system` · `test_runtime` |
| **GPT 29.2 / Grok 31.6** | Side-effect truth derived from authoritative log, claim accuracy | `runtime.reconcile_changed_state`, `runtime.claim_accuracy` · `test_runtime`, `test_handoff` |

---

## Scope & Roadmap

### Implemented and Tested (Phase 0 + 1 + 2)
- Core typed contracts and error taxonomy
- Fail-closed policy engine and approval binding
- Tool gateway with bounded envelopes, reducers, and artifact storage
- Deterministic verification engine (exit codes, file existence, JSON schema)
- Honest token and cost accounting
- Task state machine and side-effect reconciliation
- JSON schema-versioned checkpointing and automatic run loop resume (`resume=True`)
- SQLite checkpoint storage backend (`SqliteCheckpointStore`)
- Provider adapters (`OpenAIAdapter`, `AnthropicAdapter`, `GeminiAdapter`)
- Provider-neutral durable handoffs (spec sections 22–25)
- Command-line interface (`longterm`)

### Deferred (Phase 3+)
- Retrieval/search engine ("retrieval-first" context assembly)
- Semantic handoff checker layers 2–3 (requiring live evaluator model)
- Multi-fixture benchmark suite

---

## Two-Repository Architecture: longterm-ai & AI-Coding-Swiss-Army-Knife

`longterm-ai` and [`AI-Coding-Swiss-Army-Knife`](https://github.com/rolling-codes/AI-Coding-Swiss-Army-Knife) (Dev Workflow Pack) are designed to work together as a two-tier agent system:

| Dimension | [AI-Coding-Swiss-Army-Knife](https://github.com/rolling-codes/AI-Coding-Swiss-Army-Knife) | [longterm-ai](https://github.com/rolling-codes/longterm-ai) |
|---|---|---|
| **Architectural Layer** | **Methodology, Skills & Reasoning** | **Execution, Policy & Runtime Enforcement** |
| **Form Factor** | 20 prompt-level skills + hooks + sub-agent roles | Standard library Python framework + CLI |
| **Primary Job** | Decides *what* to do, *when* to stop, and routes tasks (`kill-test`, `dev-workflow`, `code-review`, etc.) | Enforces hard constraints in code so the model *cannot* flood context or hallucinate completion |
| **Context Strategy** | Lazy loading hierarchy (`references/context-management.md`), keep/drop classification | Bounded tool envelopes, artifact store, output size limits, and deduplication |
| **Verification Gate** | Prescribes acceptance criteria and TDD flow | Executes deterministic verifiers (`exit_code`, `file_exists`, `schema_valid`) |
| **Durability** | Session memory digests (`.claude/memory.json`, ledger) | SQLite WAL checkpoints, state machine, and provider-neutral handoff records |

### Two-Repository Workflow

```
[ Developer Prompt ]
         │
         ▼
[ AI-Coding-Swiss-Army-Knife ]
  ├── kill-test          → 5-question pre-build gate (kill bad builds before code is written)
  ├── dev-workflow       → Task planning, scope definition & model routing
  └── context-management → Lazy-load hierarchy (task → docs → tests → source)
         │
         ▼ (Dispatches multi-step agentic execution)
[ longterm-ai Framework ]
  ├── ToolGateway        → Limits raw output bytes, summarizes, archives artifacts
  ├── RuleBasedPolicy    → Fail-closed risk checks before actions execute
  ├── CheckpointStore    → Atomic SQLite checkpoints per step (resume=True on disconnect)
  └── Verifier           → Deterministic checks; forbids completion until criteria pass
         │
         ▼
[ Completion & Handoff ]
  ├── claim_accuracy()   → Compares model claims vs. authoritative tool log
  └── HandoffRecord      → Clean plain-text handoff for subsequent turns / model switches
```

---

## Provenance & Credits

- **Core Design & Review Overlay:** Built from the `long_term_ai_framework` design specification and multi-provider review overlay (Claude, GPT-6, Grok).
- **Workflow Methodology & Scoring:** Integrates foundational principles from [**AI-Coding-Swiss-Army-Knife**](https://github.com/rolling-codes/AI-Coding-Swiss-Army-Knife) by RollingCodes:
  - Pre-build gate (`kill-test`) preventing unnecessary agent runs
  - Tiered information hierarchy & context discipline (`context-management`)
  - Sub-agent claim accuracy scoring (`agent-health.md` $\to$ `runtime.claim_accuracy`)

---

## License

MIT.

# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.4.1] - 2026-10-17

### Fixed
- **CLI input handling no longer crashes with a traceback (found by independent review).**
  `longterm review` previously caught only `UnsupportedRecord`; a missing/unreadable
  `task_json`, a malformed or non-list tool log, or an unparseable record escaped as a
  Python traceback and exit 1 — indistinguishable from a genuine UNSAFE verdict. All four
  subcommands (`verify`, `rebuild`, `run`, `review`) now report a clean error and use a
  uniform exit-code contract: **0** = success/SAFE, **1** = a real negative (layer one
  failed / UNSAFE verdict), **2** = operational failure (the command could not run). Exit 2
  is kept distinct so a crash or bad input is never read as a routine verdict; `review`
  prints "Treat as UNSAFE" and callers fail closed.
- `rebuild` no longer swallows a malformed transcript as a layer-one refusal: a genuine
  `NotAtCleanBoundary` refusal returns 1, while a structurally broken transcript returns 2.
- `run` no longer tracebacks on a malformed `acceptance_criteria` entry (missing `id`/`kind`);
  it returns 2 with a clear message.

## [0.4.0] - 2026-10-17

### Added
- **`longterm review` command**: a first-class multi-agent handoff review. Runs the deterministic layer-one check **and** scores the handoff record's side-effect claims against the authoritative tool log — reconciliation, `claim_accuracy`, and unbacked-claim detection in one report.
  - Usage: `longterm review <task_json> <record_json> <state_dir> [--task-id ID] [--tool-log FILE] [--json] [--max-tokens N]`.
  - Resolves the tool log in order: `--tool-log FILE` → `<state_dir>/<task_id>.tool_log.json` → empty log (with a warning; every side-effect claim is then treated as unbacked — the conservative default).
  - **Exit code contract**: `0` only when layer one passes **and** no claim is unbacked; otherwise `1`. This makes the review a scriptable/CI gate a receiving agent runs before acting on a prior agent's handoff.
  - Reuses existing primitives only (`layer_one`, `reconcile_changed_state`, `changed_state_mismatches`, `claim_accuracy`, `ToolLog.from_list`) — no new algorithms, no new dependencies.
- **`longterm run` now persists the tool log** to `<state_dir>/<task_id>.tool_log.json` (surviving terminal checkpoint cleanup) and reports its path, enabling a `run → review` pipeline.
- Architecture decision record `docs/decisions/0004-review.md`.
- `tests/test_cli.py`: review safe/unsafe/json cases and the `run → review` pipeline. Total test count: 137 passing.

## [0.3.0] - 2026-10-17

### Added
- **Retrieval Engine** (`longterm_ai.retrieval`): the stdlib-only "retrieval-first" piece.
  - `BM25Index`: incremental Okapi BM25 (`k1=1.5`, `b=0.75`) over a `\w+` word tokenizer. No embeddings, no dependencies.
  - `MemoryStore`: in-memory evidence store (BM25 index + documents) with stable tie-breaking by insertion order.
  - `Retriever.retrieve(query, k, token_budget)`: top-k retrieval bounded by the package's shared `rough_tokens` estimator.
  - `assemble_context(retriever, query, budget)`: relevance-ranked, budget-bounded context block with provenance headers.
  - `make_retrieve_tool(store)`: a `READ_ONLY` tool the model calls through the gateway — retrieval rides the same fail-closed, bounded lifecycle as any other tool (REQ 002).
- **Opt-in evidence indexing**: `Framework(..., store=MemoryStore())` indexes each truncated tool artifact as it is written (via a new gateway `on_artifact` callback). The default `store=None` leaves the run loop's behavior unchanged.
- **`Retriever` protocol** in `contracts.py` (substitutable retrieval surface).
- Architecture decision record `docs/decisions/0003-retrieval.md`.
- `tests/test_retrieval.py`: BM25 ranking, token-budget bounding, gateway-bounded/denied `retrieve` tool, and Framework indexing round-trip. Total test count: 133 passing.

### Fixed
- Resolved pre-existing pyright errors in `sqlite_store.py` (missing `Any` import) and `tests/test_sqlite_store.py` (optional-member narrowings) so the repo is back to 0 pyright errors.

## [0.2.0] - 2026-10-13

### Added
- **Automatic Checkpoint Resume**:
  - `Framework.run()` now supports `resume=True` to seamlessly restore in-flight tasks from the latest saved checkpoint.
  - Step counts, tool log entries, and state machine statuses are restored and validated.
  - Progressive checkpointing saves after every model turn.
  - `CheckpointStore.delete()` cleans up checkpoint files upon reaching terminal states (`completed`, `failed`, `blocked`).
- **SQLite Checkpoint Store** (`SqliteCheckpointStore`):
  - Added an optional SQLite-backed checkpoint backend in `longterm_ai.sqlite_store`.
  - Drop-in replacement for `CheckpointStore` using Python's standard library `sqlite3` (zero dependencies).
  - Enables atomic transactions, WAL journal mode for concurrent read safety, and context manager support (`with` statement).
- **Sub-Agent Claim Accuracy Scoring**:
  - Added `claim_accuracy(model_claims, reconciled)` to measure accuracy ratio of model side-effect claims vs. authoritative tool execution log (inspired by sub-agent health scoring).
- **Real Provider Adapters** (`longterm_ai.providers`):
  - Added `OpenAIAdapter` supporting OpenAI Chat Completions API with function calling and reported usage tokens.
  - Added `AnthropicAdapter` supporting Claude Messages API with `tool_use` blocks and stop reason translation.
  - Added `GeminiAdapter` supporting Google Gemini Generative AI models with function declarations.
  - All adapters implement the `ModelAdapter` protocol with zero mandatory runtime dependencies (lazy provider loading).
- **CLI Entry Point** (`longterm` / `python -m longterm_ai`):
  - `longterm verify <task_json> <record_json> <state_dir>`: Deterministic layer-one handoff verification.
  - `longterm rebuild <task_json> <transcript_json> <record_json> <state_dir> --format neutral|anthropic|openai`: Rebuild context across models.
  - `longterm run <task_json> [--state-dir DIR] [--resume]`: Offline task execution using `FakeModel` for CI/scripting.
  - Registered console script `longterm` in `pyproject.toml`.
- **Two-Repository Ecosystem & Provenance**:
  - Documented two-tier architecture with [`AI-Coding-Swiss-Army-Knife`](https://github.com/rolling-codes/AI-Coding-Swiss-Army-Knife) (skills/methodology layer vs. runtime/enforcement layer).
  - Explicitly credited `AI-Coding-Swiss-Army-Knife` for kill-test gates, context load hierarchy, and claim accuracy scoring in README.
- **Architecture Decision Record**:
  - Documented design rationale in `docs/decisions/0002-phase2.md`.
- **Test Suite Expansion**:
  - Added test coverage for CLI, provider adapters, SQLite store, and resume semantics, bringing total test count to 115 passing tests.

## [0.1.0] - 2026-10-13

### Added
- **Phase 0 + Phase 1 Core**:
  - Typed dataclasses, enums, and protocols in `contracts.py`.
  - Typed error taxonomy in `errors.py`.
  - Fail-closed policy engine and approval binding (`policy.py`).
  - Tool gateway with bounded envelopes, reducers, and artifact store (`tools.py`).
  - Deterministic verifier (`exit_code`, `file_exists`, `schema_valid`) in `verification.py`.
  - Honest token and cost accounting with event logging (`telemetry.py`).
  - Task state machine, execution run loop, file-based checkpoints, and side-effect reconciliation (`runtime.py`).
  - Offline testing adapters (`FakeModel`, `EchoModel`) in `adapters.py`.
  - Provider-neutral durable handoff with layer-one verification in `handoff/`.

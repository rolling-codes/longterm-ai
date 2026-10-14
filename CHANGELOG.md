# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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

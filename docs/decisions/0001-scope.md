# ADR 0001 — Phase 0 + Phase 1 scope boundary

**Date:** 2026-10-13  
**Status:** Accepted

## Context

longterm-ai implements the Phase 0 + Phase 1 core of the PROJECT_FOUNDATION
design spec. This record captures what was built vs. what was intentionally
left out, so the boundary is explicit and not inferred from absence.

## Implemented and tested

- Typed contracts and error taxonomy (`contracts.py`, `errors.py`)
- Fail-closed policy + approval binding (`policy.py`)
- Tool gateway with bounded envelopes, reducers, and artifact store (`tools.py`)
- Deterministic verifier — exit code, file exists, schema valid (`verification.py`)
- Honest usage/cost accounting + event log (`telemetry.py`)
- Task state machine, run loop, JSON schema-versioned checkpoints (`runtime.py`)
- Authoritative tool log + side-effect reconciliation + retry guard (`runtime.py`)
- Fake/echo adapters — full test suite runs without a provider (`adapters.py`)
- Provider-neutral handoff with a checked summary (spec sections 22–25) (`handoff/`)

All 77 tests pass on stdlib only; no runtime dependencies.

## Deliberately deferred

These are design intent from the spec, not built in this phase:

- Retrieval/search engine (the "retrieval-first" future state)
- SQLite memory backend
- Semantic handoff checker layers 2–3 (require a real model)
- Automatic checkpoint resume wired into the run loop
- Real provider adapters (Claude, OpenAI, etc.)
- A CLI entry point
- Benchmark results beyond the single included fixture

## Decision

Ship Phase 0 + Phase 1 as a clean, fully-tested, dependency-free library.
Deferred items are tracked here, not silently absent. Each deferred item
requires its own ADR when it enters scope.

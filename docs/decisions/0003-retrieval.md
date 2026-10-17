# ADR 0003 — Phase 3: Retrieval Engine

**Date:** 2026-10-17
**Status:** Accepted

## Context

ADR 0001/0002 delivered the reliable execution core but deferred the project's
headline "retrieval-first" thesis. The gap was concrete: in `runtime.Framework.run()`
every tool result is bounded to a summary for the model while the full text is written
to the `ArtifactStore` and never read again. Context could only grow by append — there
was no way to search accumulated evidence and pull the relevant slice back into a later
turn.

## Decision

Add a stdlib-only retrieval engine (`retrieval.py`) and wire it in as an **opt-in**
layer, so existing behavior is unchanged by default.

- **BM25, not embeddings.** `BM25Index` implements classic Okapi BM25 (`k1=1.5`,
  `b=0.75`) over a `\w+` word tokenizer. Honest keyword retrieval with zero runtime
  dependencies, consistent with the spec's stdlib-only constraint. Embeddings / a vector
  backend are a documented future swap behind the same `Retriever` surface.
- **Reuse, not reinvent.** The token budget uses the package's shared `rough_tokens`
  estimator; indexed evidence is the same full text the `ArtifactStore` already persists
  (no second copy); the `retrieve` tool is a plain `FunctionTool` so it rides the
  existing fail-closed, bounded gateway lifecycle and cannot bypass policy (REQ 002).
- **Opt-in integration.** `Framework(..., store=MemoryStore())` registers an
  `on_artifact` callback on the gateway that indexes each truncated artifact as it is
  written. `store=None` (the default) is byte-for-byte the previous run loop — the 115
  pre-existing tests are untouched.
- **Context assembly as a helper, not a rewrite.** `assemble_context()` builds a
  relevance-ranked, budget-bounded context block. It is offered as the opt-in
  "retrieval-first" path; the working append-based loop is left in place (YAGNI — ripping
  it out would churn the green suite for no required gain).

## Implemented and Tested (Phase 3)

- `retrieval.py`: `_tokenize`, `Document`, `BM25Index`, `MemoryStore`, `Retriever`,
  `assemble_context`, `make_retrieve_tool`.
- `contracts.Retriever` protocol (substitutable retrieval surface).
- Gateway `on_artifact` callback + `Framework(store=...)` auto-indexing.
- `tests/test_retrieval.py`: BM25 ranking, token-budget bounding, tokenizer, empty-index
  cases, re-index replacement, the `retrieve` tool bounded + denied through the gateway,
  and a Framework round-trip (store indexes a truncated artifact; `store=None` unchanged).

## Still Deferred (Phase 3+)

- Semantic handoff checker layers 2–3 (requires a live evaluator model).
- Multi-fixture benchmark suite.
- Embeddings / vector backend and a `SqliteMemoryStore` (reusing `sqlite_store.py`).
- Replacing the append-based run loop with retrieval-driven context assembly by default.

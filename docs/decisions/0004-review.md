# ADR 0004 — Phase 4: `longterm review` multi-agent handoff command

**Date:** 2026-10-17
**Status:** Accepted

## Context

The multi-agent handoff protocol — a receiving agent deciding whether a prior agent's
handoff is safe to act on — existed only as library primitives (`layer_one`,
`reconcile_changed_state`, `changed_state_mismatches`, `claim_accuracy`) plus the
deterministic `verify` CLI command, which runs *only* layer one. There was no single
command that runs the **full** review: deterministic verification combined with scoring
the record's side-effect claims against the authoritative tool log. Exercising the whole
flow required an ad-hoc script.

## Decision

Add `longterm review` as a first-class CLI command that runs the complete review and
returns a machine-usable verdict.

- **Composition, not new logic.** `review` wires together the already-exported
  primitives. It adds no new algorithm — the value is making the gate runnable and
  scriptable.
- **Authoritative tool log, resolved conservatively.** The log is taken from
  `--tool-log FILE`, else `<state_dir>/<task_id>.tool_log.json` (written by `run`), else
  an empty log. With no log, every side-effect claim is **unbacked** — the safe default,
  consistent with "side-effect truth comes from the log, not the model."
- **`run` persists the tool log.** `run` deletes its checkpoint on terminal states, so
  the log is also written to `<state_dir>/<task_id>.tool_log.json` to survive cleanup and
  enable a `run → review` pipeline.
- **Exit-code contract.** Exit `0` iff layer one passes **and** there are no unbacked
  claims; otherwise `1`. A dishonest or tampered handoff fails the gate in CI/scripts.
- **Honesty invariant preserved.** `claim_accuracy` returns `None` (reported as `n/a`)
  when the record makes no claims — never coerced to `0`.

## Implemented and Tested (Phase 4)

- `cli.py`: `_cmd_review`, the `review` subparser, and tool-log persistence in `_cmd_run`.
- `tests/test_cli.py`: unbacked-claim → UNSAFE/exit 1; log-backed claim → SAFE/exit 0 with
  `claim_accuracy == 1.0`; `--json` key contract; and the `run → review` pipeline.

## Still Deferred (Phase 5+)

- Optional semantic layers 2–3 in `review` (`--semantic` with `OverlapChecker` or a
  model-backed `Checker`).
- Retrieval-driven context assembly by default, multi-fixture benchmark suite, and an
  embeddings / SQLite-backed memory store.

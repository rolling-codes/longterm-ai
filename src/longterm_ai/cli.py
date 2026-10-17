"""Command-line interface for longterm-ai. Phase 2 — ADR 0002; review added in Phase 4.

Four sub-commands, all stdlib only. Exit codes are uniform: 0 = success/SAFE,
1 = a real negative (layer one failed / UNSAFE verdict), 2 = operational failure
(bad, missing, or unparseable input — the command could not run). Exit 2 is kept
distinct from 1 so a crash or bad input is never read as a routine verdict; callers
fail closed on 2.

    longterm verify  <task_json> <record_json> <state_dir> [--max-tokens N]
        Layer-one handoff check. Exit 0 = pass, 1 = fail, 2 = could not run.

    longterm rebuild <task_json> <transcript_json> <record_json> <state_dir>
                     --format neutral|anthropic|openai [--max-tokens N] [--keep N]
        Rebuild context for the receiving model. Prints JSON to stdout.

    longterm run <task_json> [--state-dir DIR] [--task-id ID] [--resume]
        Execute a task using FakeModel (no provider needed). Prints RunResult
        as JSON to stdout. Real provider use requires the Python API directly.
        Also writes <state_dir>/<task_id>.tool_log.json so `review` can read it.

    longterm review <task_json> <record_json> <state_dir>
                    [--task-id ID] [--tool-log FILE] [--json] [--max-tokens N]
        Multi-agent handoff review: run the deterministic layer-one check AND
        score the record's side-effect claims against the authoritative tool log
        (reconciliation + claim accuracy + unbacked-claim detection). Exit 0 only
        when layer one passes AND no claim is unbacked; 1 for an UNSAFE verdict;
        2 if the review could not run (bad/missing inputs) — treat 2 as UNSAFE. This
        is the gate a receiving agent runs before acting on a prior agent's handoff.

The ``run`` command uses FakeModel (which returns ``done`` after an empty turn list),
a RuleBasedPolicy with no extra tool restrictions, and a DeterministicVerifier. It
exists for scripting, CI, and demonstrations; for real provider use the Python API.
"""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

# Exit codes shared across commands:
#   0 = success / SAFE verdict
#   1 = a real negative result (layer one failed, or an UNSAFE review verdict)
#   2 = operational failure: the command could not run (bad/missing/unparseable input).
# Keeping 2 distinct from 1 is the point: a crash or bad input must never be read as a
# routine verdict. Callers fail closed on 2 (treat the handoff as not cleared).
EXIT_OK = 0
EXIT_NEGATIVE = 1
EXIT_OPERATIONAL = 2


class CliError(Exception):
    """An operational failure that stopped the command before it could produce a result."""


def _read_json(path_str: str, what: str):
    """Read and parse a JSON file, raising CliError (not a traceback) on any I/O or parse error."""
    try:
        text = Path(path_str).read_text(encoding="utf-8")
    except OSError as exc:
        raise CliError(f"cannot read {what} {path_str!r}: {exc}") from exc
    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:
        raise CliError(f"{what} {path_str!r} is not valid JSON: {exc}") from exc


def _load_record(path_str: str):
    """Load a HandoffRecord, converting every load failure into a clean CliError."""
    from .handoff import HandoffRecord
    from .handoff.record import UnsupportedRecord
    try:
        return HandoffRecord.load(Path(path_str))
    except (UnsupportedRecord, OSError, ValueError, TypeError, KeyError) as exc:
        raise CliError(f"cannot load handoff record {path_str!r}: {exc}") from exc


def _cmd_verify(args: argparse.Namespace) -> int:
    from .handoff import layer_one

    try:
        task = _read_json(args.task_json, "task file")
        record = _load_record(args.record_json)
    except CliError as exc:
        print(f"verify: could not run -- {exc}", file=sys.stderr)
        return EXIT_OPERATIONAL

    result = layer_one(task, record, Path(args.state_dir), args.max_tokens)
    if result.passed:
        print("layer one: PASSED")
        return EXIT_OK
    for f in result.failures:
        print(f"FAIL: {f}", file=sys.stderr)
    return EXIT_NEGATIVE


def _cmd_rebuild(args: argparse.Namespace) -> int:
    from .handoff import NotAtCleanBoundary, layer_one, rebuild, to_anthropic, to_openai_chat

    try:
        task = _read_json(args.task_json, "task file")
        transcript = _read_json(args.transcript_json, "transcript file")
        record = _load_record(args.record_json)
    except CliError as exc:
        print(f"rebuild: could not run -- {exc}", file=sys.stderr)
        return EXIT_OPERATIONAL

    check = layer_one(task, record, Path(args.state_dir), args.max_tokens)
    if not check.passed:
        for f in check.failures:
            print(f"FAIL: {f}", file=sys.stderr)
        print("layer one failed -- rebuild refused", file=sys.stderr)
        return EXIT_NEGATIVE

    try:
        neutral = rebuild(transcript, record, task, keep_last_user_turns=args.keep)
    except NotAtCleanBoundary as exc:
        # A real refusal: a tool call is still open. Not an operational error.
        print(f"rebuild refused: {exc}", file=sys.stderr)
        return EXIT_NEGATIVE
    except (KeyError, TypeError, ValueError) as exc:
        # The transcript is structurally wrong; the command could not run.
        print(f"rebuild: could not run -- malformed transcript: {exc}", file=sys.stderr)
        return EXIT_OPERATIONAL

    fmt = args.format
    if fmt == "anthropic":
        out = to_anthropic(neutral, args.model or "claude-opus-4-5", args.max_output_tokens)
    elif fmt == "openai":
        out = to_openai_chat(neutral, args.model or "gpt-4o")
    else:
        out = neutral

    print(json.dumps(out, ensure_ascii=False, indent=2))
    return 0


def _cmd_run(args: argparse.Namespace) -> int:
    from .adapters import FakeModel
    from .contracts import Criterion, Task
    from .policy import RuleBasedPolicy
    from .runtime import Framework
    from .tools import ToolRegistry
    from .verification import DeterministicVerifier

    try:
        raw = _read_json(args.task_json, "task file")
        # Build acceptance criteria from JSON dicts. A missing id/kind is a malformed
        # task, not a runtime failure -- surface it as operational, never a traceback.
        criteria = [
            Criterion(
                id=c["id"],
                kind=c["kind"],
                params=c.get("params", {}),
                required=c.get("required", True),
                description=c.get("description", ""),
            )
            for c in raw.get("acceptance_criteria", [])
        ]
    except CliError as exc:
        print(f"run: could not run -- {exc}", file=sys.stderr)
        return EXIT_OPERATIONAL
    except (KeyError, TypeError) as exc:
        print(f"run: could not run -- malformed acceptance_criteria: {exc}", file=sys.stderr)
        return EXIT_OPERATIONAL

    task = Task(
        objective=raw.get("objective", ""),
        mandatory_constraints=raw.get("mandatory_constraints", []),
        acceptance_criteria=criteria,
        max_steps=raw.get("max_steps", 20),
        deadline_s=raw.get("deadline_s"),
        token_budget=raw.get("token_budget"),
    )

    state_dir = Path(args.state_dir)
    reg = ToolRegistry()
    policy = RuleBasedPolicy(reg.risk_map())
    verifier = DeterministicVerifier(base_dir=state_dir)
    model = FakeModel([])  # will emit one empty "done" turn

    fw = Framework(model, reg, policy, verifier, state_dir=state_dir)
    result = fw.run(task, task_id=args.task_id, resume=args.resume)

    # Persist the tool log so `review` can read it after terminal checkpoint cleanup.
    state_dir.mkdir(parents=True, exist_ok=True)
    tool_log_path = state_dir / f"{args.task_id}.tool_log.json"
    tool_log_path.write_text(json.dumps(result.tool_log, ensure_ascii=False), encoding="utf-8")

    out = {
        "status": result.status.value,
        "verified_complete": result.verified_complete,
        "steps": result.steps,
        "note": result.note,
        "usage": result.usage,
        "changed_state": result.changed_state,
        "tool_log": result.tool_log,
        "tool_log_path": str(tool_log_path),
    }
    print(json.dumps(out, ensure_ascii=False, indent=2))
    return 0


def _resolve_tool_log(args: argparse.Namespace, state_dir: Path) -> tuple[list, str | None]:
    """Return (log entries, source) for review. Raise CliError on an unreadable/ill-formed log.

    Order: explicit --tool-log, else the one `run` wrote at <state_dir>/<task_id>.tool_log.json,
    else empty (no log -> every side-effect claim is unbacked, the conservative default).
    """
    if args.tool_log:
        items = _read_json(args.tool_log, "tool-log file")
        source: str | None = args.tool_log
    else:
        default_log = state_dir / f"{args.task_id}.tool_log.json"
        if not default_log.exists():
            return [], None
        items = _read_json(str(default_log), "tool-log file")
        source = str(default_log)
    if not isinstance(items, list):
        raise CliError(f"tool-log file {source!r} must contain a JSON list of entries")
    return items, source


def _cmd_review(args: argparse.Namespace) -> int:
    from .handoff import layer_one
    from .runtime import ToolLog, changed_state_mismatches, claim_accuracy, reconcile_changed_state

    # Load every input up front. Any operational failure here means the review did NOT run,
    # so we fail closed with EXIT_OPERATIONAL -- never 0, and never a traceback a caller
    # might misread as a routine UNSAFE verdict.
    try:
        task = _read_json(args.task_json, "task file")
        record = _load_record(args.record_json)
        state_dir = Path(args.state_dir)
        log_items, log_source = _resolve_tool_log(args, state_dir)
        try:
            tool_log = ToolLog.from_list(log_items)
        except (KeyError, TypeError, ValueError) as exc:
            raise CliError(f"tool-log entries are malformed: {exc}") from exc
    except CliError as exc:
        print(f"review: could not run -- {exc}", file=sys.stderr)
        print("Treat as UNSAFE: the handoff was not cleared.", file=sys.stderr)
        return EXIT_OPERATIONAL

    reconciled = reconcile_changed_state(tool_log)
    mism = changed_state_mismatches(record.changed_state, reconciled)
    acc = claim_accuracy(record.changed_state, reconciled)
    check = layer_one(task, record, state_dir, args.max_tokens)

    safe = check.passed and not mism["unbacked_claims"]

    if args.json:
        print(json.dumps({
            "layer_one_passed": check.passed,
            "layer_one_failures": list(check.failures),
            "reconciled_changed_state": reconciled,
            "claim_accuracy": acc,
            "unbacked_claims": mism["unbacked_claims"],
            "unclaimed_in_log": mism["unclaimed_in_log"],
            "tool_log_source": log_source,
            "safe_to_hand_off": safe,
        }, ensure_ascii=False, indent=2))
        return EXIT_OK if safe else EXIT_NEGATIVE

    print("=" * 60)
    print("MULTI-AGENT HANDOFF REVIEW")
    print("=" * 60)
    if log_source is None:
        print("WARNING: no authoritative tool log found; every side-effect")
        print("         claim is treated as unbacked (conservative default).")
    print(f"layer one        : {'PASSED' if check.passed else 'FAILED'}")
    for f in check.failures:
        print(f"  FAIL: {f}")
    print(f"claim accuracy   : {'n/a (no claims)' if acc is None else f'{acc:.3f}'}")
    print("reconciled changed_state (from the tool log):")
    for r in reconciled or ["  (none)"]:
        print(f"  - {r}" if reconciled else r)
    if mism["unbacked_claims"]:
        print("UNBACKED CLAIMS (must NOT be treated as done):")
        for c in mism["unbacked_claims"]:
            print(f"  ! {c}")
    if mism["unclaimed_in_log"]:
        print("changes in the log the record did not claim:")
        for c in mism["unclaimed_in_log"]:
            print(f"  + {c}")
    print("-" * 60)
    print(f"VERDICT: {'SAFE TO HAND OFF' if safe else 'UNSAFE -- do not act on this handoff'}")
    print("=" * 60)
    return EXIT_OK if safe else EXIT_NEGATIVE


def main(argv: list[str] | None = None) -> None:
    """Entry point for the ``longterm`` CLI command."""
    parser = argparse.ArgumentParser(
        prog="longterm",
        description="longterm-ai command-line interface",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # --- verify ---
    p_ver = sub.add_parser("verify", help="Layer-one handoff check")
    p_ver.add_argument("task_json")
    p_ver.add_argument("record_json")
    p_ver.add_argument("state_dir")
    p_ver.add_argument("--max-tokens", dest="max_tokens", type=int, default=1500)

    # --- rebuild ---
    p_reb = sub.add_parser("rebuild", help="Rebuild context for receiving model")
    p_reb.add_argument("task_json")
    p_reb.add_argument("transcript_json")
    p_reb.add_argument("record_json")
    p_reb.add_argument("state_dir")
    p_reb.add_argument("--format", choices=["neutral", "anthropic", "openai"], default="neutral")
    p_reb.add_argument("--model", default=None, help="Model identifier for the output format")
    p_reb.add_argument("--max-tokens", dest="max_tokens", type=int, default=1500)
    p_reb.add_argument("--max-output-tokens", dest="max_output_tokens", type=int, default=8192)
    p_reb.add_argument("--keep", type=int, default=1, dest="keep",
                       help="Number of recent user turns to keep in rebuilt context")

    # --- run ---
    p_run = sub.add_parser("run", help="Execute a task (FakeModel; no provider needed)")
    p_run.add_argument("task_json")
    p_run.add_argument("--state-dir", dest="state_dir", default=".longterm_state")
    p_run.add_argument("--task-id", dest="task_id", default="task")
    p_run.add_argument("--resume", action="store_true", default=False,
                       help="Resume from the last saved checkpoint if one exists")

    # --- review ---
    p_rev = sub.add_parser("review", help="Multi-agent handoff review (layer one + claim accuracy)")
    p_rev.add_argument("task_json")
    p_rev.add_argument("record_json")
    p_rev.add_argument("state_dir")
    p_rev.add_argument("--task-id", dest="task_id", default="task",
                       help="Task id used to locate <state_dir>/<task_id>.tool_log.json")
    p_rev.add_argument("--tool-log", dest="tool_log", default=None,
                       help="Explicit tool-log JSON file (overrides the default lookup)")
    p_rev.add_argument("--json", action="store_true", default=False, help="Emit the review as JSON")
    p_rev.add_argument("--max-tokens", dest="max_tokens", type=int, default=1500)

    args = parser.parse_args(argv)
    handlers = {"verify": _cmd_verify, "rebuild": _cmd_rebuild, "run": _cmd_run, "review": _cmd_review}
    sys.exit(handlers[args.command](args))

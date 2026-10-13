"""Command-line interface for longterm-ai. Phase 2 — ADR 0002.

Three sub-commands, all stdlib only:

    longterm verify  <task_json> <record_json> <state_dir> [--max-tokens N]
        Layer-one handoff check. Exit 0 = pass, 1 = fail.

    longterm rebuild <task_json> <transcript_json> <record_json> <state_dir>
                     --format neutral|anthropic|openai [--max-tokens N] [--keep N]
        Rebuild context for the receiving model. Prints JSON to stdout.

    longterm run <task_json> [--state-dir DIR] [--task-id ID] [--resume]
        Execute a task using FakeModel (no provider needed). Prints RunResult
        as JSON to stdout. Real provider use requires the Python API directly.

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


def _cmd_verify(args: argparse.Namespace) -> int:
    from .handoff import HandoffRecord, layer_one

    task = json.loads(Path(args.task_json).read_text(encoding="utf-8"))
    try:
        record = HandoffRecord.load(Path(args.record_json))
    except Exception as exc:
        print(f"error loading record: {exc}", file=sys.stderr)
        return 1

    state_dir = Path(args.state_dir)
    result = layer_one(task, record, state_dir, args.max_tokens)
    if result.passed:
        print("layer one: PASSED")
        return 0
    for f in result.failures:
        print(f"FAIL: {f}", file=sys.stderr)
    return 1


def _cmd_rebuild(args: argparse.Namespace) -> int:
    from .handoff import HandoffRecord, layer_one, rebuild, to_anthropic, to_openai_chat
    from .handoff.record import UnsupportedRecord

    task = json.loads(Path(args.task_json).read_text(encoding="utf-8"))
    transcript = json.loads(Path(args.transcript_json).read_text(encoding="utf-8"))
    try:
        record = HandoffRecord.load(Path(args.record_json))
    except UnsupportedRecord as exc:
        print(f"error loading record: {exc}", file=sys.stderr)
        return 1

    state_dir = Path(args.state_dir)
    check = layer_one(task, record, state_dir, args.max_tokens)
    if not check.passed:
        for f in check.failures:
            print(f"FAIL: {f}", file=sys.stderr)
        print("layer one failed — rebuild refused", file=sys.stderr)
        return 1

    try:
        from .handoff import NotAtCleanBoundary
        neutral = rebuild(transcript, record, task, keep_last_user_turns=args.keep)
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

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

    raw = json.loads(Path(args.task_json).read_text(encoding="utf-8"))

    # Build acceptance criteria from JSON dicts.
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

    out = {
        "status": result.status.value,
        "verified_complete": result.verified_complete,
        "steps": result.steps,
        "note": result.note,
        "usage": result.usage,
        "changed_state": result.changed_state,
        "tool_log": result.tool_log,
    }
    print(json.dumps(out, ensure_ascii=False, indent=2))
    return 0


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

    args = parser.parse_args(argv)
    handlers = {"verify": _cmd_verify, "rebuild": _cmd_rebuild, "run": _cmd_run}
    sys.exit(handlers[args.command](args))

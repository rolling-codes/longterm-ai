"""Rebuild a request for the receiving model from the framework's own transcript.
Spec section 25.

Reasoning blocks are never resent, so no provider has anything to drop. Finished tool
exchanges become short text notes that point to durable state. The checked handoff
record goes in a user turn, labeled as earlier working notes. Mandatory constraints go
in system text, taken from the task (not the record), and the command refuses to run
unless the layer one check passes, so a failed or skipped check cannot reach the
receiving model (fail closed, REQ 019).

Usage:
  python -m longterm_ai.handoff.rebuild TASK TRANSCRIPT RECORD STATE_DIR --format neutral|anthropic|openai
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .record import HandoffRecord, UnsupportedRecord
from .verify import layer_one


class NotAtCleanBoundary(Exception):
    pass


def rebuild(transcript: dict, record: HandoffRecord, task: dict, keep_last_user_turns: int = 1) -> dict:
    """Build the neutral request. Call only after layer one has passed on this record and task."""
    turns = transcript["turns"]
    # Refuse to switch while a tool call is waiting for its result.
    open_calls = set()
    for t in turns:
        for part in t["parts"]:
            if part["type"] == "tool_call":
                open_calls.add(part["id"])
            elif part["type"] == "tool_result":
                open_calls.discard(part["call_id"])
    if open_calls:
        raise NotAtCleanBoundary(f"tool calls still open: {sorted(open_calls)}")

    system = transcript.get("system", "").strip()
    system += "\n\nMandatory constraints. These must hold for the rest of the task:\n"
    # Copied from the task by code. The record's copy was only used to check the summary.
    system += "\n".join(f"{i}. {c}" for i, c in enumerate(task["mandatory_constraints"], 1))

    # The handoff record replaces the earlier history, including all reasoning.
    messages = [
        {"role": "user", "content": transcript.get("first_user_message", "Continue the task described in the system text.")},
        {"role": "assistant", "content": record.to_text()},
    ]
    # Keep the most recent user turns as text, with tool exchanges flattened to notes.
    user_idx = [i for i, t in enumerate(turns) if t["role"] == "user"]
    if keep_last_user_turns <= 0 or not user_idx:
        tail_start = len(turns)
    else:
        tail_start = user_idx[-min(keep_last_user_turns, len(user_idx))]
    for t in turns[tail_start:]:
        text = []
        for part in t["parts"]:
            if part["type"] == "text":
                text.append(part["text"])
            elif part["type"] == "tool_call":
                text.append(f"[Earlier tool call {part['name']}, full record at {part.get('state_ref', 'durable state')}]")
            elif part["type"] == "tool_result":
                text.append(f"[Tool result summarized: {part.get('summary', 'see durable state')}; full result at {part.get('state_ref', 'durable state')}]")
            # reasoning parts are dropped on purpose
        if text:
            messages.append({"role": t["role"], "content": "\n".join(text)})
    # Merge adjacent turns with the same role, since some APIs require alternation.
    merged = []
    for m in messages:
        if merged and merged[-1]["role"] == m["role"]:
            merged[-1]["content"] += "\n\n" + m["content"]
        else:
            merged.append(dict(m))
    if merged[-1]["role"] != "user":
        merged.append({"role": "user", "content": "Continue from the next action in your working notes."})
    return {"system": system, "messages": merged}


# Adapters. [VERIFIED 2026-10-13 in the review overlay] Recheck before use; docs change.
# platform.claude.com/docs/en/api/messages/create: system is a string or an array of text
# blocks, message roles are user and assistant, content may be a string or blocks, and
# model, max_tokens and messages are required. developers.openai.com Create chat completion:
# with o1 models and newer, developer messages replace system messages.
def to_anthropic(neutral: dict, model: str, max_tokens: int) -> dict:
    return {"model": model, "max_tokens": max_tokens, "system": neutral["system"],
            "messages": [{"role": m["role"], "content": [{"type": "text", "text": m["content"]}]}
                         for m in neutral["messages"]]}


def to_openai_chat(neutral: dict, model: str, instruction_role: str = "developer") -> dict:
    if instruction_role not in ("developer", "system"):
        raise ValueError("instruction_role must be developer or system")
    return {"model": model,
            "messages": [{"role": instruction_role, "content": neutral["system"]}] + neutral["messages"]}


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("task")
    ap.add_argument("transcript")
    ap.add_argument("record")
    ap.add_argument("state_dir")
    ap.add_argument("--format", choices=["neutral", "anthropic", "openai"], default="neutral")
    ap.add_argument("--model", default="RECEIVING_MODEL_ID")
    ap.add_argument("--max-tokens", type=int, default=4096, help="output cap sent to the Anthropic API")
    ap.add_argument("--budget", type=int, default=1500, help="token budget for the handoff record")
    ap.add_argument("--openai-role", choices=["developer", "system"], default="developer")
    a = ap.parse_args(argv)
    task = json.loads(Path(a.task).read_text(encoding="utf-8"))
    transcript = json.loads(Path(a.transcript).read_text(encoding="utf-8"))
    try:
        record = HandoffRecord.load(a.record)
    except (UnsupportedRecord, TypeError) as e:
        print(f"refusing to rebuild: record could not be loaded: {e}", file=sys.stderr)
        return 1
    check = layer_one(task, record, Path(a.state_dir), a.budget)
    if not check.passed:
        print("refusing to rebuild: layer one failed: " + "; ".join(check.failures), file=sys.stderr)
        return 1
    try:
        neutral = rebuild(transcript, record, task)
    except NotAtCleanBoundary as e:
        print(f"refusing to rebuild: {e}", file=sys.stderr)
        return 1
    if a.format == "anthropic":
        out = to_anthropic(neutral, a.model, a.max_tokens)
    elif a.format == "openai":
        out = to_openai_chat(neutral, a.model, a.openai_role)
    else:
        out = neutral
    print(json.dumps(out, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

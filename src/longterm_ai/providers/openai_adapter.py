"""OpenAI chat completions adapter. Phase 2 — ADR 0002.

Satisfies the ModelAdapter protocol. Requires the ``openai`` package at runtime
(not a hard dependency of the core). Import errors surface clearly at call time.

Reported token counts come from the API response's ``usage`` field (``prompt_tokens``
/ ``completion_tokens``). Estimated counts are computed from RoughTokenEstimator for
comparison. Cost stays None: pricing varies by model and region, and we never invent
a number we do not have (REQ 008).

Tool-calling format: OpenAI function-calling schema (``type: function``).
"""
from __future__ import annotations

import json
from typing import Any

from ..contracts import ModelAdapter, ModelRequest, ModelResponse, ToolCall, UsageRecord
from ..telemetry import RoughTokenEstimator

# Static protocol satisfaction check (no runtime cost).
_ADAPTER_CHECK: ModelAdapter  # forward declaration only; assigned at module bottom


class OpenAIAdapter:
    """Adapter for the OpenAI chat completions API.

    Args:
        client: An ``openai.OpenAI`` (or compatible) client instance.
        model:  Chat model identifier, e.g. ``"gpt-4o"``.
        estimator: Optional token estimator; defaults to RoughTokenEstimator.
    """

    def __init__(self, client: Any, model: str = "gpt-4o", estimator: Any = None) -> None:
        self.client = client
        self.model = model
        self.estimator = estimator or RoughTokenEstimator()

    def generate(self, request: ModelRequest) -> ModelResponse:
        """Call the OpenAI chat completions API and return a normalised ModelResponse."""
        # Build message list: system prompt becomes a developer-role message (GPT-4+ convention).
        msgs: list[dict[str, Any]] = []
        if request.system:
            msgs.append({"role": "developer", "content": request.system})
        msgs.extend(request.messages)

        # Build OpenAI-format tools list.
        tools: list[dict[str, Any]] = [
            {
                "type": "function",
                "function": {
                    "name": t.name,
                    "description": t.description or "",
                    "parameters": {
                        "type": "object",
                        "properties": {k: {"type": v} for k, v in t.arg_schema.items()},
                        "required": list(t.arg_schema.keys()),
                    },
                },
            }
            for t in request.tools
        ]

        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": msgs,
            "max_tokens": request.max_output_tokens,
        }
        if tools:
            kwargs["tools"] = tools

        response = self.client.chat.completions.create(**kwargs)

        choice = response.choices[0]
        msg = choice.message
        content: str = msg.content or ""
        finish_reason: str = choice.finish_reason or "stop"

        # Parse tool calls.
        tool_calls: list[ToolCall] = []
        if msg.tool_calls:
            finish_reason = "tool_calls"
            for tc in msg.tool_calls:
                args = json.loads(tc.function.arguments) if isinstance(tc.function.arguments, str) else tc.function.arguments
                tool_calls.append(ToolCall(name=tc.function.name, args=args, id=tc.id or ""))

        # Usage: prefer provider-reported; estimated for comparison.
        u = response.usage
        in_reported = int(u.prompt_tokens) if u and u.prompt_tokens is not None else None
        out_reported = int(u.completion_tokens) if u and u.completion_tokens is not None else None
        in_est = self.estimator.estimate(" ".join(m.get("content", "") or "" for m in msgs))
        out_est = self.estimator.estimate(content)

        usage = UsageRecord(
            input_tokens_reported=in_reported,
            output_tokens_reported=out_reported,
            input_tokens_estimated=in_est,
            output_tokens_estimated=out_est,
            cost_usd=None,  # pricing varies; never invent a number (REQ 008)
            model=self.model,
        )
        return ModelResponse(
            content=content,
            tool_calls=tool_calls,
            finish_reason=finish_reason,
            usage=usage,
            input_transformations=None,
        )


_ADAPTER_CHECK = OpenAIAdapter.__new__(OpenAIAdapter)  # type: ignore[assignment]

"""Anthropic messages adapter. Phase 2 — ADR 0002.

Satisfies the ModelAdapter protocol. Requires the ``anthropic`` package at runtime.

Tool-calling format: Anthropic tool_use / tool_result schema.
finish_reason mapping: ``end_turn`` -> ``stop``, ``tool_use`` -> ``tool_calls``.
"""
from __future__ import annotations

from typing import Any

from ..contracts import ModelRequest, ModelResponse, ToolCall, UsageRecord
from ..telemetry import RoughTokenEstimator


class AnthropicAdapter:
    """Adapter for the Anthropic messages API.

    Args:
        client: An ``anthropic.Anthropic`` (or compatible) client instance.
        model:  Model identifier, e.g. ``"claude-opus-4-5"``.
        estimator: Optional token estimator; defaults to RoughTokenEstimator.
    """

    def __init__(self, client: Any, model: str = "claude-opus-4-5", estimator: Any = None) -> None:
        self.client = client
        self.model = model
        self.estimator = estimator or RoughTokenEstimator()

    def generate(self, request: ModelRequest) -> ModelResponse:
        """Call the Anthropic messages API and return a normalised ModelResponse."""
        # Build Anthropic-format tools list.
        tools: list[dict[str, Any]] = [
            {
                "name": t.name,
                "description": t.description or "",
                "input_schema": {
                    "type": "object",
                    "properties": {k: {"type": v} for k, v in t.arg_schema.items()},
                    "required": list(t.arg_schema.keys()),
                },
            }
            for t in request.tools
        ]

        kwargs: dict[str, Any] = {
            "model": self.model,
            "system": request.system,
            "messages": request.messages,
            "max_tokens": request.max_output_tokens,
        }
        if tools:
            kwargs["tools"] = tools

        response = self.client.messages.create(**kwargs)

        # Parse content blocks.
        content = ""
        tool_calls: list[ToolCall] = []
        for block in response.content:
            block_type = getattr(block, "type", None)
            if block_type == "text":
                content += getattr(block, "text", "")
            elif block_type == "tool_use":
                tool_calls.append(ToolCall(
                    name=getattr(block, "name", ""),
                    args=dict(getattr(block, "input", {})),
                    id=getattr(block, "id", ""),
                ))

        # Map finish reason.
        raw_stop = getattr(response, "stop_reason", "end_turn") or "end_turn"
        if raw_stop == "tool_use" or tool_calls:
            finish_reason = "tool_calls"
        elif raw_stop == "end_turn":
            finish_reason = "stop"
        else:
            finish_reason = raw_stop

        # Usage.
        u = getattr(response, "usage", None)
        in_reported = int(u.input_tokens) if u and getattr(u, "input_tokens", None) is not None else None
        out_reported = int(u.output_tokens) if u and getattr(u, "output_tokens", None) is not None else None
        in_est = self.estimator.estimate(request.system + " ".join(
            str(m.get("content", "")) for m in request.messages
        ))
        out_est = self.estimator.estimate(content)

        usage = UsageRecord(
            input_tokens_reported=in_reported,
            output_tokens_reported=out_reported,
            input_tokens_estimated=in_est,
            output_tokens_estimated=out_est,
            cost_usd=None,
            model=self.model,
        )
        return ModelResponse(
            content=content,
            tool_calls=tool_calls,
            finish_reason=finish_reason,
            usage=usage,
            input_transformations=None,
        )

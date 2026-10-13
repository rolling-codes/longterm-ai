"""Google Gemini adapter. Phase 2 — ADR 0002.

Satisfies the ModelAdapter protocol. Requires ``google-generativeai`` at runtime.

Tool-calling format: Gemini function declarations / function_call parts.
"""
from __future__ import annotations

from typing import Any

from ..contracts import ModelRequest, ModelResponse, ToolCall, UsageRecord
from ..telemetry import RoughTokenEstimator


class GeminiAdapter:
    """Adapter for the Google Gemini generative AI API.

    Args:
        client: A ``google.generativeai.GenerativeModel`` (or compatible) instance.
        model:  Model name string, e.g. ``"gemini-2.5-pro"``; used only for
                UsageRecord labelling — the actual model is baked into *client*.
        estimator: Optional token estimator; defaults to RoughTokenEstimator.
    """

    def __init__(self, client: Any, model: str = "gemini-2.5-pro", estimator: Any = None) -> None:
        self.client = client
        self.model = model
        self.estimator = estimator or RoughTokenEstimator()

    def generate(self, request: ModelRequest) -> ModelResponse:
        """Call Gemini and return a normalised ModelResponse."""
        # Build contents list: system prompt is prepended as a user turn because
        # Gemini's system instruction is set at model construction time; when using
        # a vanilla GenerativeModel we inject it as the first content part.
        contents: list[dict[str, Any]] = []
        if request.system:
            contents.append({"role": "user", "parts": [{"text": request.system}]})
            contents.append({"role": "model", "parts": [{"text": "Understood."}]})
        for m in request.messages:
            role = m.get("role", "user")
            # Map framework roles to Gemini roles ("assistant" -> "model").
            if role == "assistant":
                role = "model"
            contents.append({"role": role, "parts": [{"text": str(m.get("content", ""))}]})

        # Build function declarations.
        tools: list[dict[str, Any]] = []
        if request.tools:
            function_declarations = [
                {
                    "name": t.name,
                    "description": t.description or "",
                    "parameters": {
                        "type": "object",
                        "properties": {k: {"type": v} for k, v in t.arg_schema.items()},
                        "required": list(t.arg_schema.keys()),
                    },
                }
                for t in request.tools
            ]
            tools = [{"function_declarations": function_declarations}]

        kwargs: dict[str, Any] = {"contents": contents}
        if tools:
            kwargs["tools"] = tools

        response = self.client.generate_content(**kwargs)

        # Parse response parts.
        content = ""
        tool_calls: list[ToolCall] = []
        candidates = getattr(response, "candidates", None) or []
        for candidate in candidates:
            for part in (getattr(candidate.content, "parts", None) or []):
                if hasattr(part, "function_call") and part.function_call:
                    fc = part.function_call
                    tool_calls.append(ToolCall(
                        name=getattr(fc, "name", ""),
                        args=dict(getattr(fc, "args", {})),
                        id="",
                    ))
                elif hasattr(part, "text") and part.text:
                    content += part.text

        # Map finish reason.
        finish_reason = "tool_calls" if tool_calls else "stop"
        try:
            raw_fr = candidates[0].finish_reason if candidates else None
            if raw_fr is not None and str(raw_fr) not in ("STOP", "1"):
                finish_reason = str(raw_fr).lower()
        except (IndexError, AttributeError):
            pass

        # Usage.
        um = getattr(response, "usage_metadata", None)
        in_reported: int | None = None
        out_reported: int | None = None
        if um is not None:
            pt = getattr(um, "prompt_token_count", None)
            ct = getattr(um, "candidates_token_count", None)
            in_reported = int(pt) if pt is not None else None
            out_reported = int(ct) if ct is not None else None

        in_est = self.estimator.estimate(" ".join(
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

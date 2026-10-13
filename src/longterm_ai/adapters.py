"""Model adapters. Only a deterministic fake and an echo adapter ship here.

The core is fully runnable and testable with no API key and no network (REQ 015).
Both adapters report usage honestly: estimated tokens are filled from the request
and response; provider-reported tokens and cost stay ``None`` because there is no
real provider billing to read (REQ 008). A real adapter is a later, replaceable edge.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .contracts import ModelAdapter, ModelRequest, ModelResponse, ToolCall, UsageRecord
from .telemetry import RoughTokenEstimator


def _request_text(request: ModelRequest) -> str:
    parts = [request.system]
    for m in request.messages:
        parts.append(str(m.get("content", "")))
    return "\n".join(parts)


@dataclass
class ScriptedTurn:
    content: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    finish_reason: str = "stop"


class FakeModel:
    """Replays a fixed list of turns. Each ``generate`` consumes the next one.

    After the script is exhausted it returns a plain terminal "done" turn, so a run
    loop always terminates.
    """

    def __init__(self, turns: list[ScriptedTurn] | None = None, name: str = "fake-model",
                 estimator=None) -> None:
        self._turns = list(turns or [])
        self._i = 0
        self.name = name
        self.estimator = estimator or RoughTokenEstimator()
        self.requests: list[ModelRequest] = []

    def generate(self, request: ModelRequest) -> ModelResponse:
        self.requests.append(request)
        in_est = self.estimator.estimate(_request_text(request))
        if self._i < len(self._turns):
            turn = self._turns[self._i]
            self._i += 1
        else:
            turn = ScriptedTurn(content="done", finish_reason="stop")
        out_est = self.estimator.estimate(turn.content or "")
        finish = "tool_calls" if turn.tool_calls else turn.finish_reason
        usage = UsageRecord(
            input_tokens_estimated=in_est, output_tokens_estimated=out_est,
            input_tokens_reported=None, output_tokens_reported=None,
            cost_usd=None, model=self.name,
        )
        return ModelResponse(content=turn.content, tool_calls=list(turn.tool_calls),
                             finish_reason=finish, usage=usage, input_transformations=None)


class EchoModel:
    """Trivial adapter: echoes the last user message. Never calls tools."""

    def __init__(self, name: str = "echo-model", estimator=None) -> None:
        self.name = name
        self.estimator = estimator or RoughTokenEstimator()

    def generate(self, request: ModelRequest) -> ModelResponse:
        last = request.messages[-1]["content"] if request.messages else ""
        usage = UsageRecord(
            input_tokens_estimated=self.estimator.estimate(_request_text(request)),
            output_tokens_estimated=self.estimator.estimate(str(last)),
            model=self.name,
        )
        return ModelResponse(content=f"echo: {last}", finish_reason="stop", usage=usage)


# Static check that the fakes satisfy the protocol (cheap guard, no runtime cost).
_FAKE: ModelAdapter = FakeModel()
_ECHO: ModelAdapter = EchoModel()

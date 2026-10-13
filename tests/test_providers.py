"""Tests for provider adapters (OpenAI, Anthropic, Gemini). Phase 2 / ADR 0002.

No real API calls. All provider clients are replaced with MagicMock objects so
the tests run without any SDK installed.
"""
import json
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import MagicMock

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "src"))

from longterm_ai.contracts import ModelRequest, ToolDefinition, RiskClass
from longterm_ai.providers import AnthropicAdapter, GeminiAdapter, OpenAIAdapter


def _req(system: str = "sys", tools: list | None = None) -> ModelRequest:
    return ModelRequest(
        system=system,
        messages=[{"role": "user", "content": "hello"}],
        tools=tools or [],
        max_output_tokens=512,
    )


def _tool() -> ToolDefinition:
    return ToolDefinition(
        name="read_file",
        arg_schema={"path": "str"},
        risk=RiskClass.READ_ONLY,
        description="Read a file.",
    )


# ─────────────────────────── OpenAI ──────────────────────────────────────────

class OpenAITests(unittest.TestCase):

    def _text_response(self, content: str = "Hello!") -> MagicMock:
        """Build a mock openai.chat.completions response with a text message."""
        msg = MagicMock()
        msg.content = content
        msg.tool_calls = None
        choice = MagicMock()
        choice.message = msg
        choice.finish_reason = "stop"
        usage = MagicMock()
        usage.prompt_tokens = 10
        usage.completion_tokens = 5
        resp = MagicMock()
        resp.choices = [choice]
        resp.usage = usage
        return resp

    def _tool_response(self) -> MagicMock:
        """Build a mock response containing a tool call."""
        tc = MagicMock()
        tc.function.name = "read_file"
        tc.function.arguments = json.dumps({"path": "foo.txt"})
        tc.id = "call_abc"
        msg = MagicMock()
        msg.content = None
        msg.tool_calls = [tc]
        choice = MagicMock()
        choice.message = msg
        choice.finish_reason = "tool_calls"
        usage = MagicMock()
        usage.prompt_tokens = 20
        usage.completion_tokens = 8
        resp = MagicMock()
        resp.choices = [choice]
        resp.usage = usage
        return resp

    def test_text_response(self):
        client = MagicMock()
        client.chat.completions.create.return_value = self._text_response("Hi there!")
        adapter = OpenAIAdapter(client, model="gpt-4o")
        resp = adapter.generate(_req())
        self.assertEqual(resp.content, "Hi there!")
        self.assertEqual(resp.finish_reason, "stop")
        self.assertEqual(resp.tool_calls, [])

    def test_tool_call_response(self):
        client = MagicMock()
        client.chat.completions.create.return_value = self._tool_response()
        adapter = OpenAIAdapter(client)
        resp = adapter.generate(_req(tools=[_tool()]))
        self.assertEqual(resp.finish_reason, "tool_calls")
        self.assertEqual(len(resp.tool_calls), 1)
        self.assertEqual(resp.tool_calls[0].name, "read_file")
        self.assertEqual(resp.tool_calls[0].args, {"path": "foo.txt"})
        self.assertEqual(resp.tool_calls[0].id, "call_abc")

    def test_reported_tokens_populated(self):
        client = MagicMock()
        client.chat.completions.create.return_value = self._text_response()
        adapter = OpenAIAdapter(client)
        resp = adapter.generate(_req())
        self.assertEqual(resp.usage.input_tokens_reported, 10)
        self.assertEqual(resp.usage.output_tokens_reported, 5)

    def test_cost_is_none(self):
        """Cost must never be invented (REQ 008)."""
        client = MagicMock()
        client.chat.completions.create.return_value = self._text_response()
        resp = OpenAIAdapter(client).generate(_req())
        self.assertIsNone(resp.usage.cost_usd)

    def test_system_uses_developer_role(self):
        """System prompt should be sent as role='developer' for GPT-4+ models."""
        client = MagicMock()
        client.chat.completions.create.return_value = self._text_response()
        OpenAIAdapter(client).generate(_req(system="Be careful."))
        call_kwargs = client.chat.completions.create.call_args
        msgs = call_kwargs[1]["messages"] if call_kwargs[1] else call_kwargs[0][1]
        self.assertEqual(msgs[0]["role"], "developer")
        self.assertEqual(msgs[0]["content"], "Be careful.")

    def test_empty_tool_list_omits_tools_kwarg(self):
        """No tools= kwarg when tool list is empty (avoids API error on some models)."""
        client = MagicMock()
        client.chat.completions.create.return_value = self._text_response()
        OpenAIAdapter(client).generate(_req(tools=[]))
        call_kwargs = client.chat.completions.create.call_args[1]
        self.assertNotIn("tools", call_kwargs)

    def test_model_name_in_usage(self):
        client = MagicMock()
        client.chat.completions.create.return_value = self._text_response()
        resp = OpenAIAdapter(client, model="gpt-4o-mini").generate(_req())
        self.assertEqual(resp.usage.model, "gpt-4o-mini")


# ─────────────────────────── Anthropic ───────────────────────────────────────

class AnthropicTests(unittest.TestCase):

    def _text_response(self, text: str = "Answer.") -> MagicMock:
        block = MagicMock()
        block.type = "text"
        block.text = text
        usage = MagicMock()
        usage.input_tokens = 15
        usage.output_tokens = 7
        resp = MagicMock()
        resp.content = [block]
        resp.stop_reason = "end_turn"
        resp.usage = usage
        return resp

    def _tool_response(self) -> MagicMock:
        block = MagicMock()
        block.type = "tool_use"
        block.name = "read_file"
        block.input = {"path": "bar.txt"}
        block.id = "tu_001"
        usage = MagicMock()
        usage.input_tokens = 20
        usage.output_tokens = 10
        resp = MagicMock()
        resp.content = [block]
        resp.stop_reason = "tool_use"
        resp.usage = usage
        return resp

    def test_text_response(self):
        client = MagicMock()
        client.messages.create.return_value = self._text_response("Hi!")
        resp = AnthropicAdapter(client).generate(_req())
        self.assertEqual(resp.content, "Hi!")
        self.assertEqual(resp.finish_reason, "stop")

    def test_tool_call_response(self):
        client = MagicMock()
        client.messages.create.return_value = self._tool_response()
        resp = AnthropicAdapter(client).generate(_req(tools=[_tool()]))
        self.assertEqual(resp.finish_reason, "tool_calls")
        self.assertEqual(len(resp.tool_calls), 1)
        self.assertEqual(resp.tool_calls[0].name, "read_file")
        self.assertEqual(resp.tool_calls[0].args, {"path": "bar.txt"})

    def test_reported_tokens_populated(self):
        client = MagicMock()
        client.messages.create.return_value = self._text_response()
        resp = AnthropicAdapter(client).generate(_req())
        self.assertEqual(resp.usage.input_tokens_reported, 15)
        self.assertEqual(resp.usage.output_tokens_reported, 7)

    def test_cost_is_none(self):
        client = MagicMock()
        client.messages.create.return_value = self._text_response()
        resp = AnthropicAdapter(client).generate(_req())
        self.assertIsNone(resp.usage.cost_usd)

    def test_model_name_in_usage(self):
        client = MagicMock()
        client.messages.create.return_value = self._text_response()
        resp = AnthropicAdapter(client, model="claude-haiku").generate(_req())
        self.assertEqual(resp.usage.model, "claude-haiku")


# ─────────────────────────── Gemini ──────────────────────────────────────────

class GeminiTests(unittest.TestCase):

    def _text_response(self, text: str = "Gemini says hi.") -> MagicMock:
        part = MagicMock()
        part.text = text
        del part.function_call  # ensure hasattr returns False
        content = MagicMock()
        content.parts = [part]
        candidate = MagicMock()
        candidate.content = content
        candidate.finish_reason = "STOP"
        usage = MagicMock()
        usage.prompt_token_count = 12
        usage.candidates_token_count = 6
        resp = MagicMock()
        resp.candidates = [candidate]
        resp.usage_metadata = usage
        return resp

    def _tool_response(self) -> MagicMock:
        fc = MagicMock()
        fc.name = "read_file"
        fc.args = {"path": "baz.txt"}
        part = MagicMock()
        part.function_call = fc
        # no .text attr to simulate a function-call-only part
        del part.text
        content = MagicMock()
        content.parts = [part]
        candidate = MagicMock()
        candidate.content = content
        candidate.finish_reason = "STOP"
        usage = MagicMock()
        usage.prompt_token_count = 18
        usage.candidates_token_count = 4
        resp = MagicMock()
        resp.candidates = [candidate]
        resp.usage_metadata = usage
        return resp

    def test_text_response(self):
        client = MagicMock()
        client.generate_content.return_value = self._text_response()
        resp = GeminiAdapter(client).generate(_req())
        self.assertIn("Gemini says hi.", resp.content)
        self.assertEqual(resp.finish_reason, "stop")

    def test_tool_call_response(self):
        client = MagicMock()
        client.generate_content.return_value = self._tool_response()
        resp = GeminiAdapter(client).generate(_req(tools=[_tool()]))
        self.assertEqual(resp.finish_reason, "tool_calls")
        self.assertEqual(len(resp.tool_calls), 1)
        self.assertEqual(resp.tool_calls[0].name, "read_file")

    def test_reported_tokens_populated(self):
        client = MagicMock()
        client.generate_content.return_value = self._text_response()
        resp = GeminiAdapter(client).generate(_req())
        self.assertEqual(resp.usage.input_tokens_reported, 12)
        self.assertEqual(resp.usage.output_tokens_reported, 6)

    def test_cost_is_none(self):
        client = MagicMock()
        client.generate_content.return_value = self._text_response()
        resp = GeminiAdapter(client).generate(_req())
        self.assertIsNone(resp.usage.cost_usd)


if __name__ == "__main__":
    unittest.main()

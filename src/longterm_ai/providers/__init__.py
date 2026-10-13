"""Provider adapters for OpenAI, Anthropic, and Gemini. Phase 2 — ADR 0002.

Each adapter satisfies the ModelAdapter protocol. The provider SDK is imported
lazily inside the adapter method so the package imports cleanly even when no SDK
is installed; an ImportError is raised at call time with an actionable message.

Usage: construct the adapter with an already-configured provider client:

    from openai import OpenAI
    from longterm_ai.providers import OpenAIAdapter
    adapter = OpenAIAdapter(OpenAI())

    from anthropic import Anthropic
    from longterm_ai.providers import AnthropicAdapter
    adapter = AnthropicAdapter(Anthropic())

    import google.generativeai as genai
    from longterm_ai.providers import GeminiAdapter
    adapter = GeminiAdapter(genai.GenerativeModel("gemini-2.5-pro"))
"""
from .openai_adapter import OpenAIAdapter
from .anthropic_adapter import AnthropicAdapter
from .gemini_adapter import GeminiAdapter

__all__ = ["OpenAIAdapter", "AnthropicAdapter", "GeminiAdapter"]

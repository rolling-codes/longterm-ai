"""Provider-independent handoff with a checked summary. Spec sections 22-25.

Replaces resent reasoning with a code-checked summary, carried as plain text, so no
provider can silently drop it. See the module docstrings for the three check layers.
Side-effect claims (``changed_state``) are NOT trusted from the model; they are
reconciled against the authoritative tool log in ``longterm_ai.runtime``.
"""
from .record import Decision, Fact, HandoffRecord, UnsupportedRecord, fingerprint, normalize
from .verify import Checker, Result, layer_one, layers_two_and_three
from .rebuild import NotAtCleanBoundary, rebuild, to_anthropic, to_openai_chat
from .demo_checker import OverlapChecker

__all__ = [
    "HandoffRecord", "Decision", "Fact", "UnsupportedRecord", "fingerprint", "normalize",
    "Result", "Checker", "layer_one", "layers_two_and_three",
    "rebuild", "NotAtCleanBoundary", "to_anthropic", "to_openai_chat", "OverlapChecker",
]

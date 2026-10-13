# Provider Compatibility Assessment

**Status:** Assessment only — real adapters are explicitly deferred (see README "Implemented vs deferred").  
**Last updated:** 2026-10-13  
**Contract surface:** `ModelAdapter.generate(ModelRequest) -> ModelResponse` in `contracts.py`.

This document maps every `ModelRequest`/`ModelResponse` field to the three target
provider APIs and calls out the gaps that will surface when real adapters are written.

---

## 1. Request Mapping

### 1.1 System Prompt

| Framework field | Anthropic Messages API | OpenAI Chat Completions | Gemini generateContent |
|---|---|---|---|
| `ModelRequest.system` | Top-level `system` param (string or content-block array) | Message with `role: "system"` prepended to `messages` | `systemInstruction: { parts: [{ text: "…" }] }` top-level param |
| Multi-part system | Supported natively (array of blocks) | Single string only; multi-part needs concatenation | Single `Content` object with multiple `parts` |

**Risk:** the current field is a plain `str`. Anthropic accepts a list of content blocks for system; no adapter can round-trip that today.

### 1.2 Messages

`ModelRequest.messages` is `list[dict[str, Any]]` — structurally untyped.

| Shape | Anthropic | OpenAI | Gemini |
|---|---|---|---|
| Role field | `role: "user" \| "assistant"` | `role: "user" \| "assistant" \| "system" \| "tool"` | `role: "user" \| "model"` (note: not "assistant") |
| Content field | String **or** list of content blocks (`text`, `image`, `tool_use`, `tool_result`) | String **or** list of `{ type, text \| image_url }` parts | `parts: [{ text }, { inlineData }, { functionCall }, { functionResponse }]` wrapped in a `Content` object |
| Image representation | `{ type: "image", source: { type: "base64\|url", … } }` | `{ type: "image_url", image_url: { url: "data:…\|https://…" } }` | `{ inlineData: { mimeType, data } }` or `fileData` |

**Risk:** an adapter that passes `messages` through verbatim will fail for any content that is not a plain `{ role, content: str }` dict. Multi-modal, tool-result, and tool-use turns all require provider-specific reshaping.

### 1.3 Tool Definitions

`ToolDefinition.arg_schema` is `dict[str, str]` (name → primitive type name). Every provider expects JSON Schema.

| Framework | Anthropic | OpenAI | Gemini |
|---|---|---|---|
| Definition wrapper | `tools: [{ name, description, input_schema: {type:"object", properties:{…}, required:[…]} }]` | `tools: [{ type:"function", function:{ name, description, parameters:{…} } }]` | `tools: [{ functionDeclarations: [{ name, description, parameters:{…} }] }]` |
| Schema format | JSON Schema draft-07 | JSON Schema (subset) | OpenAPI 3.0 Schema Object subset |
| `arg_schema` gap | Needs expansion: `dict[str,str]` → full `{ type, description, required }` per arg | Same | Same |

`ToolDefinition.description` maps cleanly. `risk`, `idempotent`, `output_limit` are framework-only; no provider equivalent.

---

## 2. Response Mapping

### 2.1 Text Content

| Framework field | Anthropic | OpenAI | Gemini |
|---|---|---|---|
| `ModelResponse.content` | `response.content[i].text` where `type == "text"` | `choices[0].message.content` | `candidates[0].content.parts[i].text` (concatenated) |
| Multiple text blocks | Anthropic may return multiple text blocks; needs join | Single string | Multiple `parts`; needs join |

### 2.2 Tool Calls

| Framework field | Anthropic | OpenAI | Gemini |
|---|---|---|---|
| `ToolCall.name` | `content[i].name` where `type == "tool_use"` | `choices[0].message.tool_calls[i].function.name` | `candidates[0].content.parts[i].functionCall.name` |
| `ToolCall.args` | `content[i].input` (dict) | `JSON.parse(…tool_calls[i].function.arguments)` (string → parse) | `candidates[0].content.parts[i].functionCall.args` (struct) |
| `ToolCall.id` | `content[i].id` (e.g. `toolu_01…`) | `tool_calls[i].id` (e.g. `call_abc…`) | No id field — Gemini does not assign call ids |

**Risk:** `ToolCall.id` cannot be populated for Gemini. Downstream code that correlates tool results by id will break. See §3.

### 2.3 Tool-Result Plumbing (Next Turn)

When a tool returns a result, it must be fed back as a new message in the next request.

| Provider | Tool result message shape |
|---|---|
| Anthropic | `role: "user"`, `content: [{ type: "tool_result", tool_use_id: "…", content: "…" }]` |
| OpenAI | `role: "tool"`, `tool_call_id: "…"`, `content: "…"` |
| Gemini | `role: "user"`, `parts: [{ functionResponse: { name: "…", response: {…} } }]` — no id, matched by name |

`ModelRequest.messages: list[dict]` can carry any of these shapes, but nothing in the contract enforces which shape an adapter expects. The adapter and the runtime must agree on a representation before the plumbing works.

### 2.4 Finish Reason

`ModelResponse.finish_reason` is an unvalidated string.

| Framework value | Anthropic `stop_reason` | OpenAI `finish_reason` | Gemini `finishReason` |
|---|---|---|---|
| `"stop"` | `"end_turn"` | `"stop"` | `"STOP"` |
| `"tool_calls"` | `"tool_use"` | `"tool_calls"` | `"STOP"` (tool call is in `parts`, not signalled separately) |
| `"length"` | `"max_tokens"` | `"length"` | `"MAX_TOKENS"` |
| other | `"stop_sequence"` | `"content_filter"` | `"SAFETY"`, `"RECITATION"`, `"OTHER"` |

Each adapter must normalise to the framework's string before returning. The framework currently uses `"stop"` and `"tool_calls"` as de-facto values (see `ScriptedTurn.finish_reason` and runtime checks).

### 2.5 Usage / Token Reporting

`UsageRecord` has `input_tokens_reported` and `output_tokens_reported` only. Providers surface additional billing-relevant fields:

| Extra field | Anthropic | OpenAI | Gemini |
|---|---|---|---|
| Cache read tokens | `usage.cache_read_input_tokens` | — | `usageMetadata.cachedContentTokenCount` |
| Cache write tokens | `usage.cache_creation_input_tokens` | — | — |
| Reasoning tokens | — | `usage.completion_tokens_details.reasoning_tokens` (o-series only) | — |
| Total tokens | derived | `usage.total_tokens` | `usageMetadata.totalTokenCount` |

None of these are representable in the current `UsageRecord`. Accurate cost accounting for cached or reasoning prompts requires them.

### 2.6 `input_transformations`

The field exists (`Any | None`, default `None`) and the invariant is correct: unknown → `None`. In practice, none of the three providers return a structured "what I dropped/rewrote" report. Anthropic may truncate messages silently at context limit; OpenAI and Gemini similarly. This field will remain `None` for all three providers in normal operation. Only Anthropic's prompt-caching metadata (see above) provides any input-side accounting signal — and that belongs in `UsageRecord`, not here.

---

## 3. Gaps & Risks Summary

| # | Gap | Severity | Providers affected |
|---|---|---|---|
| G1 | `messages: list[dict]` untyped — no enforcement of role enum, content-block shape, or tool-result structure | High | All |
| G2 | `arg_schema: dict[str,str]` — primitive names only, no descriptions, no required list; all three providers need JSON Schema | High | All |
| G3 | `ToolCall.id` — Gemini has no call id; downstream id-correlation logic will fail | High | Gemini |
| G4 | Tool-result turn shape — no contract for how `ToolResult` maps back into `messages`; each adapter would invent its own | High | All |
| G5 | `UsageRecord` missing `cache_read_tokens`, `cache_write_tokens`, `reasoning_tokens` — cost accounting is wrong for cached/reasoning calls | Medium | Anthropic (cache), OpenAI o-series (reasoning), Gemini (cache) |
| G6 | `finish_reason` normalisation — every provider uses different strings; no canonical set defined in contract | Medium | All |
| G7 | `system` is `str` — cannot represent Anthropic multi-block system or Gemini multi-part `systemInstruction` | Low-medium | Anthropic, Gemini |
| G8 | Gemini role is `"model"` not `"assistant"` — adapter must translate on both sides | Low | Gemini |

---

## 4. Minimal Recommended Contract Tweaks (Additive Only)

These changes keep the `None`-not-zero invariant (REQ 008) and do not break existing fakes.

### 4.1 `UsageRecord` — add optional cache/reasoning fields

```python
@dataclass
class UsageRecord:
    # … existing fields unchanged …
    cache_read_tokens: int | None = None    # Anthropic cache_read_input_tokens; Gemini cachedContentTokenCount
    cache_write_tokens: int | None = None   # Anthropic cache_creation_input_tokens
    reasoning_tokens: int | None = None     # OpenAI o-series completion_tokens_details.reasoning_tokens
```

All default `None`. Adapters that cannot report a field leave it `None`. No downstream change required.

### 4.2 Typed message/content shapes (minimal)

Define a `MessageRole` enum and a `ContentBlock` union before writing any real adapter. At minimum:

```python
class MessageRole(str, Enum):
    USER = "user"
    ASSISTANT = "assistant"

@dataclass
class TextBlock:
    type: str = "text"
    text: str = ""

@dataclass
class ToolUseBlock:     # outbound: model requesting a tool call
    type: str = "tool_use"
    id: str = ""
    name: str = ""
    args: dict[str, Any] = field(default_factory=dict)

@dataclass
class ToolResultBlock:  # inbound: framework returning tool output
    type: str = "tool_result"
    tool_call_id: str = ""   # empty string where provider has no id (Gemini)
    name: str = ""           # used by Gemini (matched by name, not id)
    content: str = ""
```

`ModelRequest.messages` would become `list[dict[str, Any]]` still (backward compat) but with a parallel typed path adapters can depend on. Full migration deferred.

### 4.3 `finish_reason` canonical values

Document the four values the framework treats as meaningful and let adapters normalise to them:

- `"stop"` — natural end of turn
- `"tool_calls"` — model requested one or more tool calls
- `"length"` — hit `max_output_tokens`
- `"content_filter"` — blocked by provider safety / policy

Anything else passes through as-is (unknown signals stay visible, not silently remapped).

### 4.4 Gemini id-less tool calls

Add a note to `ToolCall` that `id` may be empty string for providers without call ids. Downstream correlation must fall back to matching by `name` + call order when `id == ""`.

---

## 5. Reference URLs (not fetched — cited for adapter authors)

- **Anthropic Messages API:** https://docs.anthropic.com/en/api/messages
- **Anthropic tool use:** https://docs.anthropic.com/en/docs/build-with-claude/tool-use
- **Anthropic prompt caching / usage fields:** https://docs.anthropic.com/en/docs/build-with-claude/prompt-caching
- **OpenAI Chat Completions:** https://platform.openai.com/docs/api-reference/chat/create
- **OpenAI function calling:** https://platform.openai.com/docs/guides/function-calling
- **OpenAI reasoning tokens (o-series):** https://platform.openai.com/docs/guides/reasoning
- **Gemini generateContent:** https://ai.google.dev/api/generate-content
- **Gemini function calling:** https://ai.google.dev/gemini-api/docs/function-calling

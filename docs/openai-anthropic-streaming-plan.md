# Plan: Enable Native Streaming for OpenAI and Anthropic Models

## Goal

Enable provider-native token streaming for OpenAI and Anthropic chat models while
reusing HGA's existing Home Assistant ChatLog streaming pipeline.

The integration already supports native HA streaming through
`HGAConversationEntity._attr_supports_streaming`,
`app.astream_events(...)`, and `_stream_langgraph_to_ha(...)`. The remaining gap
is that OpenAI and Anthropic chat providers are currently instantiated with their
default `streaming=False` behavior, so they can fall back to final-message
delivery through `on_chat_model_end`.

## Current State

- `custom_components/home_generative_agent/conversation.py` already drives
  LangGraph with `astream_events(...)` for the normal chat path.
- `_stream_langgraph_to_ha(...)` already maps LangGraph model/tool events into
  HA `async_add_delta_content_stream(...)` deltas.
- The transformer already has a non-streaming fallback that emits final text
  from `on_chat_model_end` when no token chunks were observed.
- Anthropic text block normalization is already handled for `text` and
  `text_delta` content blocks.
- `schema_first_yaml=True` intentionally remains on the full-response
  `ainvoke(...)` path because JSON-to-YAML post-processing cannot be safely
  streamed token by token. This path calls `app.ainvoke(...)` via
  `_async_run_ainvoke()` and never consumes `astream_events(...)`, so it returns an
  aggregated `AIMessage` regardless of the provider's `streaming` flag — adding
  `streaming=True` to the provider does not affect it.

## Implementation Steps

### 1. Confirm Runtime Support

Verify the installed LangChain provider signatures before changing behavior.
In the current local environment, both `ChatOpenAI` and `ChatAnthropic` accept:

- `streaming`
- `stream_usage`
- `disable_streaming`

Use the installed package behavior as the source of truth for implementation.
Official provider docs should be used only to confirm API expectations and
provider-specific event shapes.

### 2. Enable OpenAI Streaming

Update the primary OpenAI provider construction in
`custom_components/home_generative_agent/__init__.py`:

- Add `streaming=True` to `ChatOpenAI(...)`.
- Add `stream_usage=True` to `ChatOpenAI(...)`. OpenAI defaults `stream_usage`
  to `None`, which means usage stats are not included in streaming responses.
  Without this flag, `raw_response.usage_metadata` at `graph.py:1271` will be
  empty after switching to streaming, silently breaking token-count telemetry.
  (`ChatAnthropic` already defaults `stream_usage=True` and needs no change.)
- Preserve the existing timeout, sync HTTP client, and Home Assistant async HTTP
  client wiring.
- Leave the existing `configurable_fields(...)` unchanged unless tests show a
  provider option conflict.

If tool-calling behavior regresses with streaming enabled, prefer a narrow
provider fallback such as `disable_streaming="tool_calling"` instead of
disabling streaming globally.

Keep the non-streaming fallback in `conversation.py`. It remains useful for
older LangChain/provider behavior, stream negotiation failures, and tests.

### 3. Enable Anthropic Streaming

Update the Anthropic provider construction in
`custom_components/home_generative_agent/__init__.py`:

- Add `streaming=True` to `ChatAnthropic(...)`.
- Preserve the existing model, API key, `model_kwargs`, and async-client pre-warm
  behavior.
- Do not remove the existing Anthropic content block normalization.

If tool-calling behavior regresses with streaming enabled, prefer a narrow
provider fallback such as `disable_streaming="tool_calling"` instead of
disabling streaming globally. The same escape hatch applies to both providers.

### 4. Decide OpenAI-Compatible Behavior Separately

Do not automatically assume every OpenAI-compatible server handles streaming the
same way as OpenAI.

Recommended initial behavior:

- Enable streaming for first-party OpenAI and Anthropic only.
- Leave the OpenAI-compatible provider unchanged until tested against at least
  one target server.

Follow-up option:

- Add `streaming=True` for OpenAI-compatible providers after manual validation.
- Keep the existing `on_chat_model_end` fallback so non-streaming compatible
  servers still produce visible replies.

### 5. Preserve Existing Streaming Transformer Behavior

Do not replace `_stream_langgraph_to_ha(...)`.

The transformer should continue to:

- Open assistant blocks from `on_chat_model_start`.
- Emit user-visible token chunks from `on_chat_model_stream`.
- Record tool calls from `on_chat_model_end`.
- Emit tool results from the `action` node.
- Suppress non-user-visible tool argument fragments and thinking blocks.
- Fall back to final text from `on_chat_model_end` only when no streamed text was
  emitted for that model turn.

### 6. Add Focused Regression Tests

Extend `tests/custom_components/home_generative_agent/test_conversation_stream.py`
with cases that prove provider-native streaming does not duplicate final text:

- `test_stream_text_only` already covers OpenAI-style text chunks followed by
  `on_chat_model_end` and asserts no duplication — do not rewrite it.
- Update the `test_stream_nonstreaming_text_only` docstring to say "providers
  that don't emit `on_chat_model_stream`" instead of naming OpenAI/Anthropic
  specifically, since those providers will stream after this change.
- `test_stream_nonstreaming_tool_then_text`, `test_stream_anthropic_text_delta_chunks`,
  and `test_stream_filters_thinking` already cover the tool-turn, Anthropic delta,
  and thinking-block cases — do not rewrite them.
- Token usage metadata is preserved after enabling streaming: assert that
  `raw_response.usage_metadata` is non-empty when `stream_usage=True` is set,
  to guard against the silent regression where streaming without `stream_usage`
  returns no token counts.

Add or extend provider setup tests to assert:

- The OpenAI chat provider is constructed with `streaming=True` and
  `stream_usage=True`.
- The Anthropic chat provider is constructed with `streaming=True`.
- Existing HTTP client and Anthropic pre-warm behavior are preserved.

### 7. Manual Validation

Validate the behavior in Home Assistant with:

- OpenAI, no tools: first visible tokens appear before full completion.
- Anthropic, no tools: first visible tokens appear before full completion.
- OpenAI with tools: Show Details contains assistant content, tool call, tool
  result, and final assistant content in order.
- Anthropic with tools: same ordering check.
- `schema_first_yaml=True`: dashboard generation still uses the non-streaming
  full-response path.

### 8. Verification Commands

Run focused verification first:

```bash
./hga/bin/pytest tests/custom_components/home_generative_agent/test_conversation_stream.py -q
```

Then run checks for the touched files and broader type coverage:

```bash
hga/bin/ruff check custom_components/home_generative_agent/__init__.py custom_components/home_generative_agent/conversation.py tests/custom_components/home_generative_agent/test_conversation_stream.py
hga/bin/pyright
```

If documentation or Markdown files change, also run:

```bash
git diff --check
```

## Expected Outcome

OpenAI and Anthropic chat turns should stream visible tokens through the existing
HA ChatLog delta path. The final-message fallback remains in place for
non-streaming providers and edge cases, and `schema_first_yaml=True` continues to
use the full-response path.

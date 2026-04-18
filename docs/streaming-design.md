# Design: Native LLM Streaming via astream_events + HA ChatLog Delta (Issue #370)

Branch: `feat/streaming-chatlog` | Status: APPROVED (v5, 2026-04-17)

## Problem Statement

HGA blocks on `app.ainvoke()` until the full LangGraph response is complete before
returning any text or audio. For voice pipelines this creates audible silence gaps.
For text chat, users see no feedback until the full response arrives. HA 2026.4.0
introduced `async_add_delta_content_stream()` and a delta streaming protocol to fix
this. HGA does not yet use it.

## What Makes This Cool

First token reaches the HA voice pipeline while LangGraph is still running tool calls.
Silence gaps disappear. The "Show Details" panel streams in real time. This is the
architecture HA designed for — HGA has been the odd one out since 2026.4.0.

## Constraints

- Part 1 (#369, PR #371) shipped `async_add_assistant_content_without_tools()`.
  That commit logic must be **replaced**, not layered on top — the two approaches
  are mutually exclusive for the same turn (1Jamie, #368).
- `schema_first_yaml=True` mode does full-response JSON→YAML conversion that cannot
  be done per-token. That mode falls back to the existing `ainvoke` path.
- All ToolInputs must use `external=True` — LangGraph executes tools, not HA.
- Post-processing (`_fix_entity_ids_in_text`) cannot be applied to a committed stream.
  Dropped for streaming path in this PR; relocate to graph as a follow-up.

## Premises

1. `schema_first_yaml=True` is rare enough that an `ainvoke` fallback for that mode
   is acceptable for MVP — no streaming TTFT for dashboard generation.
2. `_fix_entity_ids_in_text` fires rarely in practice and is display-only (not
   safety-critical). Dropping it from the streaming path is a safe MVP trade-off;
   it should move into the graph as a post-processing node later.
3. Filtering on `event["metadata"]["langgraph_node"] == "agent"` is sufficient to
   exclude summarization node tokens from the stream.
4. Tool call argument tokens (`on_chat_model_stream` during tool-call generation)
   should be suppressed — JSON fragments, not user-readable text. Only text content
   tokens stream to HA.
5. The transformation function lives in `conversation.py` as a private async generator.
   No new module. Minimal diff.
6. `_attr_supports_streaming = True` is set unconditionally. Setting this flag enables
   the `async_add_delta_content_stream()` API on ChatLog — it does NOT prohibit calling
   `async_add_assistant_content_without_tools()` (which the schema_first_yaml fallback
   path continues to use via `_populate_chat_log_from_response`). The two are compatible.

## Approaches Considered

### Approach A: Minimal — transformation function, schema_first_yaml fallback (CHOSEN)
- Replace `ainvoke()` with `astream_events()` in `conversation.py`
- Add `_stream_langgraph_to_ha()` async generator (transformation function)
- `schema_first_yaml=True` branches to existing `ainvoke` + `_populate_chat_log_from_response` path
- Remove `_populate_chat_log_from_response` from non-schema_first_yaml path
- Effort: M | Risk: Low | Files: primarily `conversation.py`

### Approach B: Ideal — graph-level post-processing node
- Move `_fix_entity_ids_in_text` into a post-processing LangGraph node after `_call_model`
- Single streaming code path, no ainvoke fallback
- Effort: L | Risk: Medium | Files: `conversation.py` + `agent/graph.py`

## Recommended Approach: A

Minimal diff, single new abstraction, well-understood fallback. Entity ID fixing moves
into the graph as a follow-up — that's the right separation of concerns anyway.

## Implementation Plan

### 1. Set `_attr_supports_streaming = True`

On `HGAConversationEntity`. HA will now expect the streaming path to be available.
This enables `async_add_delta_content_stream()` on ChatLog objects. The schema_first_yaml
fallback continues to call `async_add_assistant_content_without_tools()` — that method
remains available regardless of this flag. No conflict.

### 2. Add `_stream_langgraph_to_ha()` transformation function

Private async generator in `conversation.py`. Signature:

```python
async def _stream_langgraph_to_ha(
    event_stream: AsyncIterable[dict],
    agent_id: str,
) -> AsyncGenerator[AssistantContentDeltaDict | ToolResultContentDeltaDict, None]:
```

**Event mapping logic:**

The HA `async_add_delta_content_stream` state machine rules:
- Delta with `role="assistant"`: flush any accumulated block → open new AssistantContent block
- Delta without `role`: append content/tool_calls/thinking_content to current block
- Delta with `role="tool_result"`: flush accumulated AssistantContent → commit ToolResultContent
- Generator exhaustion: auto-flush any remaining accumulated block

Therefore `role="assistant"` must be sent at **block open** (`on_chat_model_start`),
not at block close. Sending it at `on_chat_model_end` with `content=accumulated_text`
causes HA to flush the accumulated text, then seed a NEW block with that same text,
resulting in double-committed content. This matches HA reference implementations
(OpenAI entity, Google AI entity) which send `role="assistant"` at message open.

No `accumulated_text` state variable is needed — HA accumulates internally.
The generator holds one piece of cross-event state: `_pending_tool_calls` — the list
of tool_call dicts from the most recent `on_chat_model_end`, used to synthesize error
results if `on_chain_end` for "action" returns an empty message list.

```
# Generator-level state (initialised once, persists across events):
_pending_tool_calls: list[dict] = []

on_chat_model_start
  event.get("metadata", {}).get("langgraph_node") == "agent"
  → yield AssistantContentDeltaDict(role="assistant")  # opens new block, no content
  (retrieve_tools uses an embedding model → fires on_embeddings_* events, NOT
  on_chat_model_start — the "agent" node filter is therefore sufficient.)

on_chat_model_stream
  event.get("metadata", {}).get("langgraph_node") == "agent"
  → extract chunk_text from AIMessageChunk.content:
      if isinstance(chunk.content, str): chunk_text = chunk.content
      elif isinstance(chunk.content, list):
          chunk_text = "".join(
              b.get("text", "") for b in chunk.content
              if isinstance(b, dict) and b.get("type") == "text"
          )
      else: chunk_text = ""
  → skip if chunk_text is empty OR chunk.tool_call_chunks is non-empty
  → yield AssistantContentDeltaDict(content=chunk_text)  # no role = incremental

on_chat_model_end
  event.get("metadata", {}).get("langgraph_node") == "agent"
  AIMessage has tool_calls
  → _pending_tool_calls = list(tool_calls)  # track for synthetic fallback
  → yield AssistantContentDeltaDict(
        tool_calls=[llm.ToolInput(external=True, id=tc["id"], tool_name=tc["name"],
                              tool_args=tc["args"]) for tc in tool_calls]
    )  # no role — appends tool_calls to current block; role="tool_result" will flush

  AIMessage has no tool_calls (final response)
  → _pending_tool_calls = []
  → no yield — generator exhaustion auto-flushes the accumulated block

on_chain_end
  event.get("metadata", {}).get("langgraph_node") == "action"
  tool_messages = [
      msg for msg in event["data"]["output"].get("messages", [])
      if isinstance(msg, ToolMessage)
  ]

  if tool_messages:
      → for each msg in tool_messages:
            yield ToolResultContentDeltaDict(
                role="tool_result",
                tool_call_id=msg.tool_call_id,
                tool_name=msg.name or "",
                tool_result=_normalize_tool_result(msg.content),
            )
      → _pending_tool_calls = []
      (first role="tool_result" flushes the accumulated AssistantContent with its
      tool_calls; each subsequent one commits an additional ToolResultContent block)

  else:
      # No tool messages — routing guard or precheck rejected all calls before
      # they produced results. Without a role="tool_result", the open AssistantContent
      # block (with its tool_calls) would hang until the next role="assistant" flushes
      # it, leaving the LLM with tool calls in its context window but no results.
      # That ambiguity causes hallucinated results or retry loops.
      # Inject a synthetic rejection result for every pending tool_call so the
      # LLM receives an explicit policy-rejection signal.
      for tc in _pending_tool_calls:
          yield ToolResultContentDeltaDict(
              role="tool_result",
              tool_call_id=tc.get("id") or "",
              tool_name=tc.get("name") or "",
              tool_result={"error": "tool execution rejected by routing policy"},
          )
      _pending_tool_calls = []

all other events → skip
```

**Content normalization:** `AIMessageChunk.content` is a `str` for OpenAI/Gemini/local
models and a `list[dict]` (content blocks) for Anthropic Claude (same structure as
`AIMessage.content`). The extraction logic above mirrors `_normalize_ai_content()` at
`conversation.py:104`. Apply it consistently to every `on_chat_model_stream` chunk.

**`tool_args=` not `tool_input=`:** `llm.ToolInput` dataclass field is `tool_args`
(verified from existing call sites at `conversation.py:165`). Using `tool_input=` raises
`TypeError` at runtime.

**Why `on_chain_end` for "action" node, not `on_tool_end`:**

`_call_tools` dispatches to either `_run_langchain_tool` (calls `lc_tool.ainvoke()`,
which IS a LangChain Runnable and emits `on_tool_end`) or `_run_ha_tool` (calls
`llm_api.async_call_tool()`, which is NOT a LangChain Runnable and does NOT emit
`on_tool_end`). Using `on_tool_end` would silently miss all HA tool results. The
action node's `on_chain_end` fires after ALL tools complete and contains the full
`{"messages": [ToolMessage, ...]}` in `event["data"]["output"]`. This is the correct
and complete source for tool results.

**`_normalize_tool_result` note:**
This function is already defined at `conversation.py:122`. It converts `ToolMessage.content`
(str or list) to `JsonObjectType` for HA's `ToolResultContent`. Reuse as-is.

**Anthropic thinking_content:**
`AIMessageChunk` may contain thinking blocks for Claude models. In langchain-anthropic,
thinking blocks arrive in `chunk.content` as list items — same list-of-blocks format as
text, but with `"type": "thinking"` and `"thinking": "..."` keys. They do NOT appear in
`additional_kwargs`. To extract:
```python
chunk_thinking = "".join(
    b.get("thinking", "") for b in chunk.content
    if isinstance(b, dict) and b.get("type") == "thinking"
)
```
When non-empty, include in the same delta as content:
`AssistantContentDeltaDict(content=chunk_text or None, thinking_content=chunk_thinking or None)`.
The `test_stream_anthropic_thinking` mock must use `chunk.content = [{"type": "thinking", "thinking": "..."}]`,
NOT `chunk.additional_kwargs = {"thinking": "..."}`. Low effort, makes Show Details
richer for Claude users. Deferred — silently no-ops if not implemented.

### 3. Replace the invoke block in `_async_handle_message`

```python
if options.get(CONF_SCHEMA_FIRST_YAML, False):
    # schema_first_yaml requires full-response JSON→YAML conversion.
    # Stream not supported for this mode — use existing ainvoke path.
    try:
        response = await app.ainvoke(input=app_input, config=app_config)
    except HomeAssistantError as err:
        _LOGGER.exception("LangGraph error during conversation processing.")
        intent_response = intent.IntentResponse(language=user_input.language)
        intent_response.async_set_error(
            intent.IntentResponseErrorCode.UNKNOWN,
            f"Something went wrong: {err}",
        )
        return conversation.ConversationResult(
            response=intent_response, conversation_id=conversation_id
        )
    trace.async_conversation_trace_append(
        trace.ConversationTraceEventType.AGENT_DETAIL,
        {"messages": response["messages"], "tools": tools or None},
    )
    # ... existing post-processing and _populate_chat_log_from_response ...
else:
    # Streaming path — tokens reach HA as they arrive.
    event_stream = app.astream_events(
        input=app_input, config=app_config, version="v2"
    )
    try:
        async for _ in chat_log.async_add_delta_content_stream(
            self.entity_id,
            _stream_langgraph_to_ha(event_stream, self.entity_id),
        ):
            pass  # HA commits each content block; we consume the generator
    except HomeAssistantError:
        # Partial content already committed — cannot roll back.
        # Log and return whatever was committed rather than failing silently.
        _LOGGER.exception(
            "HomeAssistantError mid-stream; partial content committed to chat_log."
        )

    # Fire trace after stream completes using final graph state.
    # LangGraph writes checkpoint at node completion, BEFORE emitting on_chain_end.
    # By the time astream_events() is exhausted, the final checkpoint is committed.
    # aget_state() therefore returns current-turn state, not a stale prior checkpoint.
    # If no checkpointer is configured (test mode), aget_state() raises — catch and skip.
    try:
        final_state = await app.aget_state(app_config)
        trace.async_conversation_trace_append(
            trace.ConversationTraceEventType.AGENT_DETAIL,
            {"messages": final_state.values.get("messages", []), "tools": tools or None},
        )
    except Exception:  # noqa: BLE001
        _LOGGER.debug("aget_state unavailable; skipping trace for streaming turn.")

    return conversation.async_get_result_from_chat_log(user_input, chat_log)
```

### 4. Error handling mid-stream

If `astream_events()` raises `HomeAssistantError` mid-stream, partial content is
already committed to chat_log — cannot be rolled back. Behavior change from current
ainvoke path (which replaces with error message): partial response is committed as-is.
This is better than silent failure. Document in changelog.

`HomeAssistantError` mid-stream is caught at the `async for` level (around the
`async_add_delta_content_stream` call). The trace append and result construction
still run — `async_get_result_from_chat_log` returns whatever is in chat_log.

Note: HA's `async_add_delta_content_stream` auto-flushes any accumulated incremental
content when the generator is exhausted (or raises). This means even if the generator
exits without a `role="assistant"` final flush, HA will commit the partial text.
The streaming path therefore has no dangling-accumulator problem on early exit.

TODO (follow-up, not MVP): before raising/returning on a mid-stream error, consider
yielding a final `AssistantContentDeltaDict(content=" [response truncated]")` so
the user and the LLM both receive an explicit truncation signal rather than an
abruptly-ended response. Logging the error is always required regardless.

### 5. Remove `_populate_chat_log_from_response` from non-schema_first_yaml path

Dead code once streaming path is in place. Keep it for the `schema_first_yaml` fallback.
Keep the function itself — it is the fallback path.

## Data Flow Diagram

```
astream_events(version="v2")
        │
        ▼
_stream_langgraph_to_ha()           [transformation function]
        │
        ├── on_chat_model_start (agent node)
        │   └── AssistantContentDeltaDict(role="assistant")     [open new block]
        │
        ├── on_chat_model_stream (agent node, text only, no tool_call_chunks)
        │   └── AssistantContentDeltaDict(content=token)        [no role = incremental]
        │
        ├── on_chat_model_end (agent node, with tool_calls)
        │   └── AssistantContentDeltaDict(                      [append tool_calls]
        │           tool_calls=[llm.ToolInput(external=True, tool_args=...)])
        │
        ├── on_chain_end (action node)                          [ALL tool results]
        │   └── for each ToolMessage in output["messages"]:
        │       ToolResultContentDeltaDict(role="tool_result",  [flush+commit]
        │           tool_call_id=..., tool_result=...)
        │   OR if output["messages"] is empty:
        │       synthetic ToolResultContentDeltaDict per _pending_tool_calls
        │           tool_result={"error": "tool execution rejected by routing policy"}
        │
        ├── on_chat_model_start (agent node, next iteration)
        │   └── AssistantContentDeltaDict(role="assistant")     [open next block]
        │   ...repeat...
        │
        └── [generator exhausted after final on_chat_model_end with no tool_calls]
            └── HA auto-flushes accumulated block               [final commit]
                        │
                        ▼
        async_add_delta_content_stream()    [HA ChatLog API]
                        │
                        ├── delta_listener → TTS pipeline       [TTFT]
                        └── chat_log.content                    [Show Details]
```

## Test Plan

**Unit tests** (mock event stream, no HA):
- `test_stream_text_only`: emit `on_chat_model_start` + 3 `on_chat_model_stream` events +
  `on_chat_model_end` (no tool_calls). Assert: 4 total deltas — first is
  `AssistantContentDeltaDict(role="assistant")`, next 3 are `content=token` with no role,
  and `on_chat_model_end` emits nothing. Generator exhaustion provides final flush.
- `test_stream_suppresses_tool_call_tokens`: emit `on_chat_model_stream` events where
  `tool_call_chunks` is non-empty. Assert: no content deltas yielded.
- `test_stream_with_tools`: emit on_chat_model_start → stream → on_chat_model_end (with
  tool_calls) → on_chain_end (action) → on_chat_model_start → stream → on_chat_model_end
  (no tool_calls). Assert: role="assistant" open, content tokens, tool_calls delta (no role),
  ToolResultContentDeltaDict(role="tool_result"), role="assistant" open, content tokens.
  No role="assistant" at on_chat_model_end; no content in tool_calls delta.
- `test_stream_filters_summarization_node`: emit `on_chat_model_stream` events with
  `langgraph_node="summarize_and_remove_messages"`. Assert: none yielded.
- `test_stream_synthetic_tool_rejection`: emit on_chat_model_start → stream →
  on_chat_model_end (with tool_calls) → on_chain_end (action, empty messages list).
  Assert: synthetic `ToolResultContentDeltaDict` yielded for each pending tool_call,
  with `tool_result={"error": "tool execution rejected by routing policy"}`.
  Assert: `_pending_tool_calls` cleared after injection.
- `test_stream_anthropic_thinking`: emit chunk with `chunk.content = [{"type": "thinking", "thinking": "..."}]`.
  Assert: `thinking_content` field present in delta. Do NOT use `additional_kwargs["thinking"]` —
  thinking blocks arrive in `chunk.content`, not `additional_kwargs`.
- `test_normalize_tool_result_passthrough`: verify `_normalize_tool_result` handles
  string and list content from ToolMessage correctly.

**Integration tests** (real HA + LangGraph):
- Single-turn, no tools: assert chat_log has one AssistantContent block.
- Multi-turn with tool calls: assert tool_call sequence + tool_result sequence
  reaches chat_log in order.
- PIN flow (Turn N ends with `requires_pin`, Turn N+1 provides PIN): assert both
  turns complete correctly, no dangling accumulator state.
- `schema_first_yaml=True`: assert fallback to ainvoke fires, YAML output correct.

**Provider matrix** (risk downgraded — LangGraph abstracts provider wire formats):
The transformation function reads LangGraph events, not provider-specific payloads.
The only provider-specific concern is the shape of `AIMessageChunk.content` (str vs list),
which the extraction logic already handles. Provider testing validates LangChain's adapter
layer, not HGA's logic.
- Anthropic (Claude): list-type content + thinking blocks
- OpenAI: string content
- Gemini: verify chunk.content shape
- Local (Ollama): verify chunk.content shape

## Open Questions

1. **`app.aget_state` checkpoint ordering (RESOLVED)**: LangGraph writes checkpoint
   state at node completion, BEFORE emitting the `on_chain_end` event for that node.
   The outermost graph's `on_chain_end` fires last — after all checkpoints are written.
   By the time `astream_events()` is exhausted, the final state is committed. `aget_state`
   returns current-turn state. Implementation wraps in `try/except` for test-mode safety
   (no checkpointer → raises, caught silently).

2. **Multi-tool parallelism (RESOLVED for current code)**: `_call_tools` runs tool calls
   sequentially. `on_chain_end` for "action" fires after all tools finish, so ordering
   matches LangGraph message insertion order. If parallelism is added later, `on_chain_end`
   remains correct (all results available at chain_end regardless).

3. **Anthropic thinking chunk format (DEFERRED, additive feature)**: The thinking_content
   field is additive — if the implementation is wrong, the feature silently no-ops (no
   regression). The integration test against Claude is the real gate. Acceptable for MVP.

## Success Criteria

- [ ] Text appears token-by-token in the HA conversation panel
- [ ] Voice pipeline TTFT reduced vs `ainvoke()` baseline (verify with HA developer
      tools timing; expect ≥500ms improvement on tool-less queries at p50)
- [ ] All model providers (Anthropic, OpenAI, Gemini, local) produce correct output
      with no dropped tokens or malformed tool call sequences
- [ ] PIN verification flow not broken: multi-turn test passes (Turn N → requires_pin,
      Turn N+1 → action confirmed)
- [ ] `_attr_supports_streaming = True` set on the entity
- [ ] Part 1's `async_add_assistant_content_without_tools()` block removed from streaming path
- [ ] Tool call/result events reach HA chat_log in correct sequence (verified by
      inspecting chat_log.content in integration test)
- [ ] Routing-guard rejection emits synthetic `ToolResultContentDeltaDict` (no orphaned tool_calls)
- [ ] `schema_first_yaml=True` falls back to `ainvoke` correctly, YAML output unchanged
- [ ] Trace append fires after stream completes with full message list from `aget_state`
- [ ] Unit tests pass: all test cases in Test Plan section
- [ ] `make lint`, `make typecheck`, `make test` green

## Next Steps

1. `_attr_supports_streaming = True` on entity — one-liner, do first, confirms HA wiring
2. Write `_stream_langgraph_to_ha()` with unit tests (mock event stream, assert delta sequence)
3. Replace `ainvoke` block with streaming path + `schema_first_yaml` branch
4. Add `aget_state` trace call after stream loop
5. Integration test: single-turn no tools, multi-turn with tools, PIN flow
6. Test each model provider
7. Follow-up TODO: move `_fix_entity_ids_in_text` into graph post-processing node

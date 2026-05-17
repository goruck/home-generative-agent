# Issue #394 OpenAI-Compatible Regression Test Plan

## Context

Issue #394 needs confidence that HGA behaves correctly with llama-server exposed as
an OpenAI-compatible endpoint. The original reported failures were:

- `AttributeError: 'list' object has no attribute 'data'` from llama-server's
  non-standard `/v1/embeddings` response shape.
- `TypeError: '>=' not supported between instances of 'NoneType' and 'float'`
  when semantic retrieval returned a result with `score=None`.
- Follow-up tool-selection loops for read-only queries such as:

```text
list all open windows
```

The goal is to get repeatable local regression coverage for those behaviors
without requiring developers to install llama-server locally. Real llama-server
validation remains a manual smoke test that can be run by users or in disposable
infrastructure.

## Current State

All three bugs are already fixed in the codebase. This plan is about
**regression guards for existing fixes**, not tests for new behavior.

| Bug | Fix location |
|-----|-------------|
| `AttributeError` on bare embedding list | `_search_memories()` in `agent/graph.py` catches `AttributeError` and falls back to recency search |
| `score=None` TypeError | `_get_rag_retrieved_tools()` and `_get_actuation_safety_tools()` both guard with `getattr(item, "score", None) or 0.0` |
| Tool-selection loop | `_tool_loop_guard()` fires after `_MAX_ACTION_ROUNDS = 3` rounds via the `action_rounds` state counter |
| Read-only open-state tool selection | `_retrieve_tools()` force-binds and promotes `GetLiveContext` for read-only open-state queries, even when RAG misses it |
| Brittle local-model `GetLiveContext` args | `_call_tools()` normalizes the first read-only open-state `GetLiveContext` call to broad `{"domain": "binary_sensor"}` |
| Broad open-state result pollution | `_filter_open_state_live_context_content()` filters broad binary-sensor results back to the requested type, such as doors or windows |

Existing coverage should be preserved and extended only where a real assertion
gap remains:

| Behavior | Existing test | Gap |
|----------|---------------|-----|
| llama-server embedding `AttributeError` fallback | `test_search_memories_falls_back_to_recency_on_attribute_error` in `test_regressions.py` | None — verified |
| `score=None` retrieval guard | `test_rag_retrieval_none_score_is_treated_as_zero` and `test_actuation_safety_none_score_does_not_crash` in `test_tool_retrieval.py` | None — verified |
| Read-only `open` query avoids actuation injection and promotes `GetLiveContext` | `test_retrieve_tools_no_actuation_safety_for_read_only_open` in `test_tool_retrieval.py` | None — verified |
| Read-only open-door query force-binds `GetLiveContext` when RAG misses it | `test_retrieve_tools_force_injects_live_context_for_open_doors` in `test_tool_retrieval.py` | None — verified |
| Open commands keep actuation behavior | `test_retrieve_tools_open_command_does_not_force_live_context` in `test_tool_retrieval.py` | None — verified |
| Local-model live-context args are widened only for the first read-only open-state call | `test_normalize_live_context_args_for_read_only_open_state`, `test_normalize_live_context_args_does_not_touch_open_command`, and `test_normalize_live_context_args_can_leave_subsequent_calls_alone` in `test_tool_retrieval.py` | None — verified |
| Broad live-context results are scoped to the requested open entity type | `test_filter_open_state_live_context_scopes_door_query` and `test_filter_open_state_live_context_keeps_requested_open_windows` in `test_tool_retrieval.py` | None — verified |
| `action_rounds` reset at turn start | `test_retrieve_tools_action_rounds_reset` in `test_tool_retrieval.py` | None — verified |
| Repeated unproductive tool calls return a friendly message | Not covered | Add direct `_should_continue()` / `_tool_loop_guard()` regression test |

## Recommendation

Do not spin up a fake OpenAI-compatible HTTP server. The bugs live in HGA's own
parsing and graph logic, not in HTTP transport. Mock one level lower, at the
LangChain store or graph state layer, using `monkeypatch.setattr` and
`unittest.mock`, exactly as the current regression tests already do.

A cloud llama-server instance is not appropriate for CI: it adds provisioning cost,
inference latency, model-download size (~GBs), non-deterministic outputs that make
assertions fragile, and a network dependency that can fail independently of HGA.
The manual smoke-test checklist in Section 4 covers real llama-server validation.

The important assertions are:

- HGA survives llama-server's incompatible embedding response shape.
- Read-only open-state queries force-bind and favor `GetLiveContext` over
  actuation safety tools.
- Actuation safety tools are not force-injected solely because the query contains
  `open`.
- The first read-only open-state `GetLiveContext` call is widened to
  `{"domain": "binary_sensor"}` so local models can ask for current state without
  brittle partial-name filters.
- Broad live-context output is filtered before the model sees it so a door query
  does not report open windows, and a window query does not report open doors.
- Repeated unproductive tool calls return HGA's friendly loop-guard message
  instead of surfacing `GraphRecursionError`.

## Plan

### 1. Embedding shape regression (already covered in `test_regressions.py`)

Preserve `test_search_memories_falls_back_to_recency_on_attribute_error`. It
patches `store.asearch` to raise `AttributeError` (simulating what happens when
the OpenAI SDK parser encounters llama-server's bare JSON array instead of
`{"data": [...]}`) and asserts that `_search_memories()` logs a warning and
returns a non-error fallback result rather than propagating the exception.

No fake HTTP server needed — the exception is what matters, not the transport.

Do not add another embedding-shape test unless this existing assertion is
removed or materially weakened.

### 2. Read-only open-window tool selection (covered in `test_tool_retrieval.py`)

`test_retrieve_tools_no_actuation_safety_for_read_only_open` asserts that
`GetLiveContext` is present, first in `selected_tools`, and that `HassTurnOn` is
absent for read-only open-window queries.

`test_retrieve_tools_force_injects_live_context_for_open_doors` covers the
weaker-model path where RAG retrieves nearby non-state tools but misses
`GetLiveContext`; the retrieval layer now fetches and prepends `GetLiveContext`
deterministically.

`test_retrieve_tools_open_command_does_not_force_live_context` preserves true
actuation behavior for commands such as `open the garage door`.

### 2a. Read-only open-state tool-call normalization (covered in `test_tool_retrieval.py`)

Some local tool-calling models emit brittle filters for these queries, such as
`{"domain": ["binary_sensor"], "name": "Window"}` or list-valued names. The
first `GetLiveContext` call in a read-only open-state turn is widened to
`{"domain": "binary_sensor"}`. Focused tests cover normalization, preservation
of true open commands, and leaving subsequent more-specific calls alone.

### 2b. Scoped broad live-context results (covered in `test_tool_retrieval.py`)

Because the first call is intentionally broad, HGA filters the tool result before
returning it to the model. Door queries receive only open door entries or the
scoped empty result `Live Context: No open doors were found.` Window queries keep
only open window entries. Regression tests cover both cases.

### 3. Loop-guard regression (add to `test_regressions.py`)

The loop guard fires when `action_rounds >= _MAX_ACTION_ROUNDS` (currently 3).
Add direct coverage for this behavior in `test_regressions.py`. Import
`_should_continue`, `_tool_loop_guard`, and optionally `_MAX_ACTION_ROUNDS` from
`custom_components.home_generative_agent.agent.graph`.

Build a graph state that already has `action_rounds = _MAX_ACTION_ROUNDS` and an
`AIMessage` with pending tool calls, then invoke `_should_continue()` and await
`_tool_loop_guard()` directly.

Assertions:

- `_should_continue()` returns `"tool_loop_guard"` (not `"action"`).
- `_tool_loop_guard()` returns HGA's friendly message string.
- No `GraphRecursionError` or other LangGraph exception propagates.

Do not script a fake model emitting repeated tool calls — that tests LangGraph
routing, not HGA's guard logic. Invoke the guard functions directly.

Suggested test name:

```text
test_openai_compatible_repeated_tool_calls_hit_loop_guard
```

### 4. Keep real llama-server validation manual

Do not require llama-server in the local developer workflow. Keep real validation
as a manual smoke test or a disposable infrastructure task.

Manual checklist:

```text
Provider: OpenAI Compatible
Query: list all open windows

Expected:
- Tool retrieval shows GetLiveContext first or near the front.
- safety=0/0 for the read-only open-state query.
- The first GetLiveContext call is broad binary_sensor live context.
- The filtered tool result contains only the requested open entry type.
- The final answer lists the requested open binary_sensor entities.
- No GraphRecursionError reaches the user.
- No AttributeError: 'list' object has no attribute 'data' escapes to HA logs.
```

Useful log evidence:

```text
Tool retrieval: ... safety=0/0 ... tools=['GetLiveContext', ...]
Normalized GetLiveContext args for read-only open-state query: ... -> {'domain': 'binary_sensor'}
Live Context: Open entries matching the request:
```

This checklist can be used by issue reporters, release testers, GitHub Actions,
Codespaces, or a throwaway VM that downloads a llama.cpp release and starts
`llama-server` with a small GGUF model.

## Verification

After extending the open-window retrieval assertion and adding the loop-guard
regression, run the focused test files:

```bash
PYTHONPATH=$(pwd) hga/bin/pytest tests/custom_components/home_generative_agent/test_regressions.py
PYTHONPATH=$(pwd) hga/bin/pytest tests/custom_components/home_generative_agent/test_tool_retrieval.py
```

Then run the full quality gate on touched files:

```bash
make lint
make typecheck
git diff --check
```

## Non-goals

- Installing llama-server locally or in CI.
- Spinning up a fake OpenAI-compatible HTTP server for these tests.
- Running a cloud llama-server instance in the CI pipeline.
- Making llama-server a hard dependency of the pytest suite.
- Claiming full llama-server compatibility from mocked coverage alone.
- Testing model quality or whether a specific GGUF model follows tool-use
  instructions well.

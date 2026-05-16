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

Existing coverage should be preserved and extended only where a real assertion
gap remains:

| Behavior | Existing test | Gap |
|----------|---------------|-----|
| llama-server embedding `AttributeError` fallback | `test_search_memories_falls_back_to_recency_on_attribute_error` in `test_regressions.py` | None — verified |
| `score=None` retrieval guard | `test_rag_retrieval_none_score_is_treated_as_zero` and `test_actuation_safety_none_score_does_not_crash` in `test_tool_retrieval.py` | None — verified |
| Read-only `open` query avoids actuation injection | `test_retrieve_tools_no_actuation_safety_for_read_only_open` in `test_tool_retrieval.py` | Asserts `tool_routing_map` presence only; does not assert `GetLiveContext` is first in `selected_tools` |
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
- Read-only open-state queries favor `GetLiveContext` over actuation safety tools.
- Actuation safety tools are not force-injected solely because the query contains
  `open`.
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

### 2. Read-only open-window tool selection (extend in `test_tool_retrieval.py`)

`test_retrieve_tools_no_actuation_safety_for_read_only_open` currently asserts
that `GetLiveContext` is present in `result["tool_routing_map"]` and that
`HassTurnOn` is absent. It does not assert anything about `result["selected_tools"]`
or ordering.

Extend this test with two additional assertions:

- `result["selected_tools"][0]` is `"GetLiveContext"` (first in the ordered list).
- No actuation safety tool name appears anywhere in `result["selected_tools"]`.

The existing mock setup (`store.asearch` returning a single `GetLiveContext` item
with `score=0.85`) is sufficient — no new fixtures needed.

Do not add a new open-window retrieval test unless the existing test is split for
clarity.

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
- The final answer lists open binary_sensor entities.
- No GraphRecursionError reaches the user.
- No AttributeError: 'list' object has no attribute 'data' escapes to HA logs.
```

Useful log evidence:

```text
Tool retrieval: ... safety=0/0 ... tools=['GetLiveContext', ...]
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

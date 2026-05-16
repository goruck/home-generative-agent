# Issue #394 Follow-up: Read-only Open Queries and Tool Loops

## Context

Issue #394 originally covered llama-server / OpenAI-compatible embedding failures:

- `AttributeError: 'list' object has no attribute 'data'`
- `TypeError: '>=' not supported between instances of 'NoneType' and 'float'`

Those crash paths were fixed in v3.14.5. A follow-up user report shows a different
failure mode: the original embedding errors are gone, but some queries still
produce no useful final answer or eventually fail with `GraphRecursionError`.

The key example is:

```text
list all open windows
```

In a successful Ollama/qwen3.5 run, HGA retrieved both read-only and actuation
tools:

```text
tools=['HassTurnOn', 'alarm_control', 'HassLightSet', 'GetLiveContext',
       'get_entity_history', 'get_camera_last_events']
```

The query still worked because the model chose only `GetLiveContext`, received
live `binary_sensor` states, and then produced a final answer without additional
tool calls.

The issue is that this success depends on model behavior. The word `open`
matches the actuation keyword regex, so read-only state queries can expose
actuation tools. A weaker or less tool-disciplined OpenAI-compatible local model
may choose `HassTurnOn`, `HassTurnOff`, or keep cycling through tools until the
LangGraph recursion limit is reached.

## Goal

Make read-only "open state" queries use live context without force-injecting
actuation tools, while preserving actuation behavior for true commands such as
"open the garage door" or "turn on the kitchen light".

Also add a defensive loop guard so repeated tool-use cycles return a clear
assistant response instead of surfacing `GraphRecursionError`.

## Known limitations

Compound queries that mix read-only and actuation intent in a single sentence
need special care. A query like *"show me which windows are open and then close
them"* contains both read-only state intent and a real actuation command. The
helper should suppress actuation only when `open` / `opened` is the only
actuation keyword present. If another actuation verb such as `close`, `turn`,
`lock`, or `unlock` appears, keep actuation safety enabled.

Also handle compound commands where the second clause uses `open` as the command
verb, e.g. *"show me which windows are open and then open the garage door"*. In
that case the first `open` is state, but the later `then open ...` phrase is an
actuation command and should keep actuation safety enabled.

`OPEN_COMMAND_CLAUSE_REGEX` matches `then`/`and then`/`after that` clause
connectors only. Comma- or period-separated compound commands such as *"show me
open windows, open the garage door"* are not detected and will have actuation
tools suppressed. This gap is accepted; users should issue state queries and
commands as separate requests. A test covering the undetected form is included in
Step 3 so the behavior is explicitly documented rather than silently broken.

## Implementation order

1. Add the read-only/open-state retrieval helper and focused retrieval tests.
2. Deterministically promote `GetLiveContext` for read-only current-state
   queries in both the normal vector-search path and the fallback path.
3. Add the graph loop guard and raise `recursion_limit` so the guard can finish
   before LangGraph's backstop fires.
4. Add the optional system-prompt hint.
5. Bump version, run verification, and update the issue.

## Plan

### 1. Add focused reproduction tests

Add tests to the existing
`tests/custom_components/home_generative_agent/test_tool_retrieval.py` for
read-only "open" queries:

- `list all open windows`
- `which doors are open`
- `show open entry sensors`
- `are the gates open`
- `is the garage vent open`

Expected behavior:

- `GetLiveContext` remains available.
- Actuation safety tools are not force-injected only because the query contains
  the word `open`.
- RAG-selected tools can still appear if the vector search legitimately returns
  them, but the deterministic safety net should not add actuation tools for
  read-only state queries.

### 2. Split "open state" from "open command" intent

`ACTUATION_KEYWORDS_REGEX` currently includes `open` and lives in
`custom_components/home_generative_agent/const.py`. Keep it there and keep `open`
as an actuation keyword. Add the two new regex constants to `const.py` as well,
for consistency:

```python
# Read-only state query signals
READ_ONLY_STATE_QUERY_REGEX = (
    r"(?i)\b(list|show|which|what|where|are|is|status|state)\b"
)

# "open" used as a state description (not a command), without a noun restriction.
# Combined with READ_ONLY_STATE_QUERY_REGEX to avoid false positives on pure
# actuation commands like "open the garage door".
OPEN_AS_STATE_REGEX = r"(?i)\b(open|opened)\b"

# Actuation commands other than "open/opened". If any of these are present, the
# query should still get actuation safety tools even if it also contains a
# read-only phrase.
# Must stay in sync with ACTUATION_KEYWORDS_REGEX — omits open/opened only.
NON_OPEN_ACTUATION_KEYWORDS_REGEX = (
    r"(?i)\b(turn|switch|lock|unlock|close|set|activate|deactivate|arm|"
    r"disarm|start|stop|dim|brighten|play|pause|mute|run|trigger|enable|"
    r"disable|toggle)\b"
)

# Compound state-then-command forms where "open" is a command despite the query
# also containing read-only state language.
OPEN_COMMAND_CLAUSE_REGEX = r"(?i)\b(?:then|and then|after that)\s+open\b"
```

The noun list in the original `OPEN_STATE_QUERY_REGEX` proposal was dropped: it
was incomplete (missed `garage`, `gate`, `vent`, `blind`, `shutter`, `skylight`,
`valve`) and adding more nouns adds fragility. The AND with
`READ_ONLY_STATE_QUERY_REGEX` already provides the necessary guard — a bare
"open the garage door" does not contain any read-only signal words so actuation
injection is preserved.

Add the helper in
`custom_components/home_generative_agent/agent/graph.py`:

```python
def _query_needs_actuation_safety(query: str) -> bool:
    has_actuation = bool(re.search(ACTUATION_KEYWORDS_REGEX, query))
    if not has_actuation:
        return False

    if (
        re.search(READ_ONLY_STATE_QUERY_REGEX, query)
        and re.search(OPEN_AS_STATE_REGEX, query)
        and not re.search(NON_OPEN_ACTUATION_KEYWORDS_REGEX, query)
        and not re.search(OPEN_COMMAND_CLAUSE_REGEX, query)
    ):
        return False
    return True
```

Use the helper instead of direct `ACTUATION_KEYWORDS_REGEX` checks in:

- `_get_actuation_safety_tools()` (graph.py:725)
- `_retrieve_tools()` fallback ordering (graph.py:1056)

**Normal path (vector store available):** suppressing actuation tools is not
enough by itself because vector scores are model-dependent — especially across
the local OpenAI-compatible models this fix targets. For read-only current-state
queries, explicitly promote `GetLiveContext` when it is present in the candidate
set. Add this block after the `all_candidates` merge at graph.py:1042, before
the `_format_and_dedupe_tools` call:

```python
if not _query_needs_actuation_safety(query):
    live_ctx = [t for t in all_candidates if t["name"] == "GetLiveContext"]
    rest = [t for t in all_candidates if t["name"] != "GetLiveContext"]
    all_candidates = live_ctx + rest
```

This reorders without removing tools, so it cannot cause regressions. It keeps
`GetLiveContext` inside the configured limit even when other non-actuation RAG
tools score nearby.

**Fallback path (vector store unavailable, graph.py:1048–1060):** the current
code floats actuation tools to the front when the query matches
`ACTUATION_KEYWORDS_REGEX`. After the helper is in place, replace that direct
regex check with `_query_needs_actuation_safety(query)` and add a symmetric
`else` branch that floats non-actuation tools to the front for read-only queries:

```python
if _query_needs_actuation_safety(query):
    actuation = [t for t in fallback if t["is_actuation"]]
    rest = [t for t in fallback if not t["is_actuation"]]
    fallback = actuation + rest
else:
    live_context = [t for t in fallback if t["name"] == "GetLiveContext"]
    non_actuation = [
        t for t in fallback if not t["is_actuation"] and t["name"] != "GetLiveContext"
    ]
    actuation = [t for t in fallback if t["is_actuation"]]
    fallback = live_context + non_actuation + actuation
```

This mirrors the existing actuation-promotion logic while ensuring
`GetLiveContext` reaches the front of the fallback list even when many
non-actuation tools are available.

### 3. Preserve true actuation behavior

Add tests proving these still trigger actuation safety tools:

- `open the garage door`
- `open the gates`
- `close the family room blinds`
- `turn on the kitchen light`
- `lock the front door`
- `show me which windows are open and then close them`
- `show me which windows are open and then open the garage door`

Add separate tests proving pure read-only queries stay suppressed:

- `show me which windows are open`

Add a test for the accepted comma-separated gap (actuation tools suppressed —
expected failure, documents the known limitation):

- `show me open windows, open the garage door`

### 4. Add a graph loop guard

`_should_continue()` (graph.py:1578) currently routes to `action` unconditionally
whenever the last AI message has tool calls. Add an application-level counter
that fires well before LangGraph's `recursion_limit=10` (set in
`conversation.py:1160`).

**Chosen approach: `action_rounds` counter in graph `State`.**

1. Add `action_rounds: int` to the `State` TypedDict. Use `.get()` with a
   default of `0` when reading it so that existing PostgreSQL checkpoints that
   predate this field deserialize without error.

   Reset the counter in `_retrieve_tools` — the first node to run each turn —
   by including it in the return dict:

   ```python
   return {
       "selected_tools": selected_tools,
       "tool_routing_map": routing_map,
       "action_rounds": 0,
   }
   ```

   `_retrieve_tools` runs exactly once per user turn before any tool calls, so
   this is the correct reset point. It is self-contained and works regardless of
   how the graph is invoked, with no requirement for call sites to remember to
   pass a reset value.

2. Increment `action_rounds` in `_call_tools` (the `action` node) by returning
   the updated state key:

```python
return {
    "messages": tool_responses,
    "action_rounds": state.get("action_rounds", 0) + 1,
}
```

3. `_should_continue()` can only choose an edge; it cannot mutate state. Add a
   `tool_loop_guard` node to write the friendly fallback message, and route
   exhausted tool-call states there before summarization:

```python
_MAX_ACTION_ROUNDS = 3


def _should_continue(
    state: State,
) -> Literal["action", "tool_loop_guard", "summarize_and_remove_messages"]:
    messages = state["messages"]
    has_tool_calls = isinstance(messages[-1], AIMessage) and messages[-1].tool_calls
    if has_tool_calls and state.get("action_rounds", 0) < _MAX_ACTION_ROUNDS:
        return "action"
    if has_tool_calls:
        return "tool_loop_guard"
    return "summarize_and_remove_messages"


async def _tool_loop_guard(state: State) -> dict[str, Any]:
    return {
        "messages": [
            AIMessage(
                content=(
                    "I wasn't able to complete this request after several "
                    "tool-use attempts. Please try rephrasing your query or "
                    "breaking it into smaller steps."
                )
            )
        ]
    }
```

Then add the node and edge:

```python
workflow.add_node("tool_loop_guard", _tool_loop_guard)
workflow.add_edge("tool_loop_guard", "summarize_and_remove_messages")
```

This ensures the final graph state ends with a normal assistant message instead
of an unresolved tool-call AI message. The user-facing fallback should be:

```text
I wasn't able to complete this request after several tool-use attempts.
Please try rephrasing your query or breaking it into smaller steps.
```

**Why 3 rounds?** The graph has 4 non-tool nodes (retrieve_tools → agent →
action → agent). Two legitimate tool calls consume 6 graph steps at
`recursion_limit=10`, leaving almost no headroom. Three action rounds cap
runaway loops early while still allowing multi-step tool use.

**Raise `recursion_limit` to 20**: With `_MAX_ACTION_ROUNDS = 3`, the friendly
guard path may need roughly:

```text
retrieve_tools -> agent -> action -> agent -> action -> agent -> action ->
agent -> tool_loop_guard -> summarize
```

That is too close to the current `recursion_limit=10`. Raise the limit to `20`
in `conversation.py` so LangGraph remains a last-resort backstop and the
application-level guard reliably gets to return the user-readable fallback.

This is a defensive fallback. The primary fix is better tool selection (Steps
2–3).

### 5. Add system-prompt hint (optional, low-cost defense)

For models that attend to system prompts, a one-sentence addition can reduce
spurious tool selection on read-only queries without requiring regex changes:

```text
For read-only state queries (listing or showing current values), prefer
GetLiveContext over actuation tools.
```

This complements the deterministic guard; it does not replace it.

### 6. Verify

Run the two most relevant test files first:

```bash
PYTHONPATH=$(pwd) hga/bin/pytest tests/custom_components/home_generative_agent/test_tool_retrieval.py
PYTHONPATH=$(pwd) hga/bin/pytest tests/custom_components/home_generative_agent/test_regressions.py
```

Then lint and type-check:

```bash
hga/bin/ruff check custom_components/home_generative_agent/const.py \
    custom_components/home_generative_agent/agent/graph.py \
    tests/custom_components/home_generative_agent/test_tool_retrieval.py
hga/bin/pyright
```

Because `State` gains a new field and `_should_continue` changes its routing
logic, add graph-level tests proving repeated tool-call cycles end with the
friendly fallback before LangGraph raises `GraphRecursionError`.

Also add a regression test proving `action_rounds` starts at `0` for a new user
turn in the same conversation/checkpointer thread. A prior multi-tool turn
should not cause the next user request to hit the loop guard early.

Then run the conversation stream and full test suite:

```bash
PYTHONPATH=$(pwd) hga/bin/pytest tests/custom_components/home_generative_agent/test_conversation_stream.py
PYTHONPATH=$(pwd) hga/bin/pytest
```

### 7. Bump version

Follow project convention: bump the patch version in `manifest.json`. Do not bump
`CONFIG_ENTRY_VERSION` in `const.py` unless the change includes a config-entry
migration. `make runtimedeps` is only needed if runtime requirements changed.

### 8. Update the issue

Post a follow-up explaining that the v3.14.5 embedding crash was fixed, and this
remaining problem is a tool-selection / graph-loop issue:

- Read-only queries containing `open` can expose actuation tools due to
  `ACTUATION_KEYWORDS_REGEX` matching the word unconditionally.
- Some local OpenAI-compatible models may choose those tools incorrectly,
  causing repeated tool cycles that exhaust `recursion_limit`.
- HGA now avoids deterministic actuation injection for read-only open-state
  queries (`_query_needs_actuation_safety` helper).
- A new `action_rounds` guard in `_should_continue` caps runaway tool loops at
  3 rounds, paired with `recursion_limit=20`, and returns a user-readable
  message instead of `GraphRecursionError`.
- Compound queries with a read-only `open` state and a separate actuation verb
  or `then open ...` command should still receive actuation safety tools; pure
  read-only open-state queries should not.

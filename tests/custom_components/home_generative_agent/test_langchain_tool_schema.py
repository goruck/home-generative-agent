# ruff: noqa: S101
"""Regression tests for LangChain tool JSON-schema extraction."""

from __future__ import annotations

import gc
import json
import logging
import warnings
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, cast

from custom_components.home_generative_agent.agent import helpers
from custom_components.home_generative_agent.agent.graph import _get_fallback_tools
from custom_components.home_generative_agent.agent.helpers import (
    langchain_tool_parameters_json,
)
from custom_components.home_generative_agent.agent.tools import (
    alarm_control,
    confirm_sensitive_action,
    resolve_entity_ids,
    upsert_memory,
)

if TYPE_CHECKING:
    from langchain_core.runnables import RunnableConfig

# Property names LangGraph injects at runtime.  "BaseStore" is not one of them
# -- the leaked class name shows up in the serialized schema, not as a property
# -- so it is asserted against the raw string in _properties() instead.
_INJECTED_KEYS = frozenset({"store", "config"})


def _properties(tool: Any) -> dict[str, Any]:
    raw = langchain_tool_parameters_json(tool)
    assert "BaseStore" not in raw
    schema = json.loads(raw)
    assert isinstance(schema, dict)
    props = schema.get("properties", {})
    assert isinstance(props, dict)
    return cast("dict[str, Any]", props)


def test_upsert_memory_schema_excludes_injected_args() -> None:
    """upsert_memory must expose only model args, never store/config/BaseStore."""
    props = _properties(upsert_memory)
    assert "content" in props
    assert "context" in props
    assert "memory_id" in props
    assert _INJECTED_KEYS.isdisjoint(props)


def test_confirm_sensitive_action_schema_excludes_injected_args() -> None:
    """confirm_sensitive_action must expose only action_id and pin."""
    props = _properties(confirm_sensitive_action)
    assert "action_id" in props
    assert "pin" in props
    assert _INJECTED_KEYS.isdisjoint(props)


def test_ordinary_tool_keeps_model_args() -> None:
    """Tools without InjectedStore still expose their model-facing parameters."""
    props = _properties(resolve_entity_ids)
    assert "entity_ids" in props
    assert _INJECTED_KEYS.isdisjoint(props)

    alarm_props = _properties(alarm_control)
    assert "name" in alarm_props
    assert "entity_id" in alarm_props
    assert _INJECTED_KEYS.isdisjoint(alarm_props)


def test_schema_extraction_does_not_warn_on_injected_store() -> None:
    """tool_call_schema must not emit the args_schema IsInstanceSchema warning."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        langchain_tool_parameters_json(upsert_memory)
        langchain_tool_parameters_json(confirm_sensitive_action)
    assert not any(
        "IsInstanceSchema" in str(w.message) or "BaseStore" in str(w.message)
        for w in caught
    )


def test_fallback_tools_use_tool_call_schema() -> None:
    """Keyword fallback must bind the injected-arg-free schema, not args_schema."""
    config: RunnableConfig = {
        "configurable": {
            "langchain_tools": {
                "upsert_memory": upsert_memory,
                "confirm_sensitive_action": confirm_sensitive_action,
                "resolve_entity_ids": resolve_entity_ids,
            }
        }
    }
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        tools = _get_fallback_tools(config, {"hga_local"})
    assert not any("IsInstanceSchema" in str(w.message) for w in caught)

    by_name = {tool["name"]: json.loads(tool["parameters"]) for tool in tools}
    upsert_props = by_name["upsert_memory"]["properties"]
    assert "content" in upsert_props
    assert _INJECTED_KEYS.isdisjoint(upsert_props)
    confirm_props = by_name["confirm_sensitive_action"]["properties"]
    assert "action_id" in confirm_props
    assert _INJECTED_KEYS.isdisjoint(confirm_props)
    assert "entity_ids" in by_name["resolve_entity_ids"]["properties"]


def test_parameters_json_uses_args_schema_only_when_tool_call_schema_absent() -> None:
    """Non-standard tools without tool_call_schema still serialize args_schema."""
    args_schema = SimpleNamespace(
        model_json_schema=lambda: {
            "type": "object",
            "properties": {"query": {"type": "string"}},
        }
    )
    tool = SimpleNamespace(tool_call_schema=None, args_schema=args_schema)
    schema = json.loads(langchain_tool_parameters_json(tool))
    assert schema["properties"] == {"query": {"type": "string"}}


def test_parameters_json_empty_without_schema() -> None:
    """A tool with no schema attributes yields an empty object."""
    assert langchain_tool_parameters_json(SimpleNamespace()) == "{}"


def test_raising_tool_call_schema_property_does_not_break_extraction() -> None:
    """A tool whose tool_call_schema property raises costs only its schema."""

    class _Exploding:
        name = "exploding_tool"
        description = "d"

        @property
        def tool_call_schema(self) -> Any:
            msg = "subset model construction failed"
            raise TypeError(msg)

        @property
        def args_schema(self) -> Any:
            return None

    assert langchain_tool_parameters_json(_Exploding()) == "{}"


def test_raising_tool_keeps_the_rest_of_the_fallback_set() -> None:
    """One hostile schema must not cost the turn every other fallback tool."""

    class _Exploding:
        name = "exploding_tool"
        description = "d"

        @property
        def tool_call_schema(self) -> Any:
            msg = "subset model construction failed"
            raise ValueError(msg)

    config: RunnableConfig = {
        "configurable": {
            "langchain_tools": {
                "exploding_tool": _Exploding(),
                "upsert_memory": upsert_memory,
            }
        }
    }
    tools = _get_fallback_tools(config, {"hga_local"})
    by_name = {tool["name"]: tool["parameters"] for tool in tools}
    assert by_name["exploding_tool"] == "{}"
    assert "content" in json.loads(by_name["upsert_memory"])["properties"]


def test_empty_schema_is_logged(caplog: Any) -> None:
    """An unextractable schema must warn: {} is silently 'takes no arguments'."""
    with caplog.at_level(logging.WARNING):
        assert langchain_tool_parameters_json(SimpleNamespace(name="mystery")) == "{}"
    assert "mystery" in caplog.text


def test_extraction_is_memoized_per_tool_object() -> None:
    """tool_call_schema is an uncached property; read it once per tool."""
    reads = 0

    class _Counting:
        name = "counting_tool"
        description = "d"

        @property
        def tool_call_schema(self) -> Any:
            nonlocal reads
            reads += 1
            return {"type": "object", "properties": {"q": {"type": "string"}}}

    tool = _Counting()
    first = langchain_tool_parameters_json(tool)
    second = langchain_tool_parameters_json(tool)
    assert first == second
    assert json.loads(first)["properties"] == {"q": {"type": "string"}}
    assert reads == 1

    # A distinct object must not read the first one's memo.
    other = _Counting()
    langchain_tool_parameters_json(other)
    assert reads == 2


def test_memo_does_not_leak_entries_for_collected_tools() -> None:
    """Per-turn tool objects must not accumulate memo entries forever."""

    class _Throwaway:
        name = "throwaway"
        description = "d"

        @property
        def tool_call_schema(self) -> Any:
            return {"type": "object", "properties": {}}

    baseline = len(helpers._schema_memo)
    keys = []
    for _ in range(25):
        tool = _Throwaway()
        keys.append(id(tool))
        langchain_tool_parameters_json(tool)
        del tool
    gc.collect()
    assert len(helpers._schema_memo) == baseline
    assert not any(k in helpers._schema_memo for k in keys)

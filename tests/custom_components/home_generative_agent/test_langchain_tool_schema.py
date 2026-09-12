# ruff: noqa: S101
"""Regression tests for LangChain tool JSON-schema extraction."""

from __future__ import annotations

import json
import warnings
from types import SimpleNamespace
from typing import Any, cast

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

_INJECTED_KEYS = frozenset({"store", "config", "BaseStore"})


def _schema_and_properties(tool: Any) -> tuple[str, dict[str, Any]]:
    raw = langchain_tool_parameters_json(tool)
    assert "BaseStore" not in raw
    schema = json.loads(raw)
    assert isinstance(schema, dict)
    props = schema.get("properties", {})
    assert isinstance(props, dict)
    return raw, cast("dict[str, Any]", props)


def _properties(tool: Any) -> dict[str, Any]:
    return _schema_and_properties(tool)[1]


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
    config: dict[str, Any] = {
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

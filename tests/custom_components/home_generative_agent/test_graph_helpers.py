# ruff: noqa: S101
"""Unit tests for graph.py helper functions — Anthropic provider branches."""

from __future__ import annotations

from custom_components.home_generative_agent.agent.graph import (
    _determine_model_name,
    _format_and_dedupe_tools,
)
from custom_components.home_generative_agent.const import CONF_ANTHROPIC_CHAT_MODEL


def test_format_and_dedupe_tools_injects_type_object_when_missing() -> None:
    """Tools with empty parameters get type:object injected for Anthropic API."""
    raw: list = [
        {
            "name": "no_params_tool",
            "api_id": "test",
            "description": "A tool with no parameters",
            "parameters": "{}",
            "is_actuation": False,
        }
    ]
    selected, _routing = _format_and_dedupe_tools(raw)
    assert len(selected) == 1
    params = selected[0]["function"]["parameters"]
    assert params.get("type") == "object", "missing type must be filled with 'object'"


def test_format_and_dedupe_tools_handles_non_dict_parameters() -> None:
    """Non-dict or empty parameters are normalized to a valid OpenAI object schema."""
    for bad_params in ["null", "[]", '"string"']:
        raw: list = [
            {
                "name": f"tool_{bad_params}",
                "api_id": "test",
                "description": "broken schema",
                "parameters": bad_params,
                "is_actuation": False,
            }
        ]
        selected, _ = _format_and_dedupe_tools(raw)
        params = selected[0]["function"]["parameters"]
        # OpenAI requires 'properties' on type:object schemas.
        assert params == {"type": "object", "properties": {}}, (
            f"bad params {bad_params!r} not normalized"
        )


def test_format_and_dedupe_tools_adds_empty_properties_for_openai() -> None:
    """type:object schemas without properties get properties:{} for OpenAI compat."""
    raw: list = [
        {
            "name": "no_props_tool",
            "api_id": "test",
            "description": "Schema with type but no properties",
            "parameters": '{"type": "object"}',
            "is_actuation": False,
        }
    ]
    selected, _ = _format_and_dedupe_tools(raw)
    params = selected[0]["function"]["parameters"]
    assert params.get("type") == "object"
    assert "properties" in params, "OpenAI requires properties on type:object schemas"
    assert params["properties"] == {}


def test_format_and_dedupe_tools_preserves_existing_type() -> None:
    """Tools that already declare type:object are not modified."""
    raw: list = [
        {
            "name": "typed_tool",
            "api_id": "test",
            "description": "Tool with explicit schema",
            "parameters": '{"type": "object", "properties": {"x": {"type": "string"}}}',
            "is_actuation": False,
        }
    ]
    selected, _ = _format_and_dedupe_tools(raw)
    params = selected[0]["function"]["parameters"]
    assert params["type"] == "object"
    assert "x" in params["properties"]


def test_determine_model_name_anthropic_returns_configured_model() -> None:
    """Anthropic provider reads CONF_ANTHROPIC_CHAT_MODEL from opts."""
    opts = {CONF_ANTHROPIC_CHAT_MODEL: "claude-sonnet-4-5"}
    assert _determine_model_name("anthropic", opts) == "claude-sonnet-4-5"


def test_determine_model_name_anthropic_missing_key_returns_empty() -> None:
    """Anthropic provider with no key in opts returns empty string."""
    assert _determine_model_name("anthropic", {}) == ""

# ruff: noqa: S101
"""Unit tests for graph.py helper functions — Anthropic provider branches."""

from __future__ import annotations

from custom_components.home_generative_agent.agent.graph import (
    _determine_model_name,
    _ensure_array_items,
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


def test_ensure_array_items_fills_missing_items() -> None:
    """Array-type properties without items get items:{type:string} injected."""
    schema = {"type": "array"}
    result = _ensure_array_items(schema)
    assert result == {"type": "array", "items": {"type": "string"}}


def test_ensure_array_items_leaves_existing_items_alone() -> None:
    """Array-type properties that already have items are not modified."""
    schema = {"type": "array", "items": {"type": "integer"}}
    result = _ensure_array_items(schema)
    assert result == {"type": "array", "items": {"type": "integer"}}


def test_ensure_array_items_recurses_into_properties() -> None:
    """Array properties nested inside object properties are also fixed."""
    schema = {
        "type": "object",
        "properties": {
            "domain": {"type": "array"},
            "name": {"type": "string"},
        },
    }
    result = _ensure_array_items(schema)
    assert result["properties"]["domain"] == {
        "type": "array",
        "items": {"type": "string"},
    }
    assert result["properties"]["name"] == {"type": "string"}


def test_ensure_array_items_recurses_into_nested_items() -> None:
    """Array-of-arrays schemas get items patched at every level."""
    schema = {"type": "array", "items": {"type": "array"}}
    result = _ensure_array_items(schema)
    assert result["items"] == {"type": "array", "items": {"type": "string"}}


def test_ensure_array_items_hoists_items_from_any_of_array_variant() -> None:
    """
    AnyOf with an array variant gets items hoisted to the top level.

    GetLiveContextTool.domain uses vol.Any(cv.string, [cv.string]) which
    safe_convert emits as {"anyOf": [{}, {"type": "array", "items": {...}}]}.
    langchain_google_genai resolves anyOf to type_:ARRAY but only checks
    v.get("items") on the outer dict, so items must be present there.
    """
    schema = {
        "anyOf": [
            {},  # cv.string falls through as empty schema
            {"type": "array", "items": {"type": "string"}},
        ]
    }
    result = _ensure_array_items(schema)
    assert result["items"] == {"type": "string"}, (
        "items must be hoisted from the array variant so langchain_google_genai can find it"
    )


def test_ensure_array_items_any_of_array_variant_without_items() -> None:
    """AnyOf array variant with no items gets the string default."""
    schema = {"anyOf": [{}, {"type": "array"}]}
    result = _ensure_array_items(schema)
    assert result["items"] == {"type": "string"}


def test_ensure_array_items_any_of_no_array_variant_unchanged() -> None:
    """AnyOf without an array variant is left alone."""
    schema = {"anyOf": [{"type": "string"}, {"type": "integer"}]}
    result = _ensure_array_items(schema)
    assert "items" not in result


def test_format_and_dedupe_tools_gemini_array_items_injected() -> None:
    """Gemini: array-type tool params without items get items:{type:string} added."""
    raw: list = [
        {
            "name": "ha_turn_on",
            "api_id": "assist",
            "description": "Turn on a device",
            "parameters": (
                '{"type": "object", "properties": {'
                '"domain": {"type": "array"}, "name": {"type": "string"}}}'
            ),
            "is_actuation": True,
        }
    ]
    selected, _ = _format_and_dedupe_tools(raw)
    domain_schema = selected[0]["function"]["parameters"]["properties"]["domain"]
    assert "items" in domain_schema, "Gemini requires items on array-type properties"
    assert domain_schema["items"] == {"type": "string"}


def test_determine_model_name_anthropic_returns_configured_model() -> None:
    """Anthropic provider reads CONF_ANTHROPIC_CHAT_MODEL from opts."""
    opts = {CONF_ANTHROPIC_CHAT_MODEL: "claude-sonnet-4-5"}
    assert _determine_model_name("anthropic", opts) == "claude-sonnet-4-5"


def test_determine_model_name_anthropic_missing_key_returns_empty() -> None:
    """Anthropic provider with no key in opts returns empty string."""
    assert _determine_model_name("anthropic", {}) == ""

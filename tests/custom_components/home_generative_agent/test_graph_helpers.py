# ruff: noqa: S101
"""Unit tests for graph.py helper functions — Anthropic provider branches."""

from __future__ import annotations

from custom_components.home_generative_agent.agent.graph import (
    _determine_model_name,
    _ensure_array_items,
    _format_and_dedupe_tools,
    _sanitize_any_of_required,
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


def test_sanitize_any_of_required_strips_required_from_non_object_variant() -> None:
    """
    'required' on a non-OBJECT anyOf branch is stripped.

    Gemini rejects declarations where an anyOf variant carries 'required'
    but is not itself type:object, e.g.
    ``{"anyOf": [{"required": ["x"]}, {"type": "array", "items": {...}}]}``.
    """
    schema = {
        "anyOf": [
            {"required": ["x"]},
            {"type": "array", "items": {"type": "string"}},
        ]
    }
    result = _sanitize_any_of_required(schema)
    assert "required" not in result["anyOf"][0]
    assert result["anyOf"][1] == {"type": "array", "items": {"type": "string"}}


def test_sanitize_any_of_required_strips_undefined_property_from_object_variant() -> (
    None
):
    """'required' entries not present in that branch's properties are dropped."""
    schema = {
        "anyOf": [
            {
                "type": "object",
                "properties": {"name": {"type": "string"}},
                "required": ["name", "missing_prop"],
            },
            {"type": "string"},
        ]
    }
    result = _sanitize_any_of_required(schema)
    assert result["anyOf"][0]["required"] == ["name"]


def test_sanitize_any_of_required_drops_required_key_when_empty_after_filtering() -> (
    None
):
    """If every required property is undefined, the 'required' key is removed."""
    schema = {
        "anyOf": [
            {
                "type": "object",
                "properties": {"name": {"type": "string"}},
                "required": ["missing_prop"],
            }
        ]
    }
    result = _sanitize_any_of_required(schema)
    assert "required" not in result["anyOf"][0]


def test_sanitize_any_of_required_valid_object_variant_unchanged() -> None:
    """A well-formed OBJECT anyOf variant with valid 'required' is left alone."""
    schema = {
        "anyOf": [
            {
                "type": "object",
                "properties": {"name": {"type": "string"}},
                "required": ["name"],
            }
        ]
    }
    result = _sanitize_any_of_required(schema)
    assert result["anyOf"][0]["required"] == ["name"]


def test_sanitize_any_of_required_recurses_into_properties() -> None:
    """Nested object properties containing anyOf/required are also sanitized."""
    schema = {
        "type": "object",
        "properties": {
            "target": {
                "anyOf": [
                    {"required": ["x"]},
                    {"type": "string"},
                ]
            }
        },
    }
    result = _sanitize_any_of_required(schema)
    assert "required" not in result["properties"]["target"]["anyOf"][0]


def test_sanitize_any_of_required_recurses_into_items() -> None:
    """Array items containing anyOf/required are also sanitized."""
    schema = {
        "type": "array",
        "items": {"anyOf": [{"required": ["x"]}, {"type": "string"}]},
    }
    result = _sanitize_any_of_required(schema)
    assert "required" not in result["items"]["anyOf"][0]


def test_sanitize_any_of_required_no_any_of_unchanged() -> None:
    """Schemas without anyOf are returned unchanged."""
    schema = {"type": "object", "properties": {}, "required": ["x"]}
    result = _sanitize_any_of_required(schema)
    assert result == schema


def test_format_and_dedupe_tools_strips_invalid_any_of_required() -> None:
    """
    End-to-end: a 3-branch anyOf with mismatched 'required' is sanitized.

    Reproduces the reported Gemini 400 error:
    ``GenerateContentRequest.tools[0].function_declarations[1].parameters
    .any_of[N].required: only allowed for OBJECT type`` /
    ``any_of[N].required[0]: property is not defined``.
    """
    raw: list = [
        {
            "name": "flaky_union_tool",
            "api_id": "test",
            "description": "Tool with a 3-way union parameter",
            "parameters": (
                '{"type": "object", "properties": {"target": {"anyOf": ['
                '{"required": ["entity_id"]},'
                '{"type": "object", "properties": {"area_id": {"type": "string"}},'
                ' "required": ["area_id", "entity_id"]},'
                '{"type": "array", "items": {"type": "string"}, '
                '"required": ["entity_id"]}'
                "]}}}"
            ),
            "is_actuation": False,
        }
    ]
    selected, _ = _format_and_dedupe_tools(raw)
    variants = selected[0]["function"]["parameters"]["properties"]["target"]["anyOf"]
    assert "required" not in variants[0], "non-object variant must lose 'required'"
    assert variants[1]["required"] == ["area_id"], (
        "object variant keeps only required props it actually defines"
    )
    assert "required" not in variants[2], "array variant must lose 'required'"


def test_determine_model_name_anthropic_returns_configured_model() -> None:
    """Anthropic provider reads CONF_ANTHROPIC_CHAT_MODEL from opts."""
    opts = {CONF_ANTHROPIC_CHAT_MODEL: "claude-sonnet-4-5"}
    assert _determine_model_name("anthropic", opts) == "claude-sonnet-4-5"


def test_determine_model_name_anthropic_missing_key_returns_empty() -> None:
    """Anthropic provider with no key in opts returns empty string."""
    assert _determine_model_name("anthropic", {}) == ""

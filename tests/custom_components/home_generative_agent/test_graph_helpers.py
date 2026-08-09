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


# The shape voluptuous_openapi emits for Home Assistant's "at least one target"
# pattern, vol.Required(vol.Any("name", "area", "floor")): parent-level
# properties plus bare-'required' anyOf variants. Valid JSON Schema; only
# Gemini's protobuf validation rejects it.
_HA_TARGET_SCHEMA = (
    '{"type": "object", "properties": {"name": {"type": "string"}, '
    '"area": {"type": "string"}, "floor": {"type": "string"}}, '
    '"required": [], "anyOf": [{"required": ["name"]}, '
    '{"required": ["area"]}, {"required": ["floor"]}]}'
)


def _target_tool(name: str = "HassTurnOn") -> list:
    """Build a raw tool carrying the HA at-least-one-target schema."""
    return [
        {
            "name": name,
            "api_id": "test",
            "description": "Turns on a device",
            "parameters": _HA_TARGET_SCHEMA,
            "is_actuation": True,
        }
    ]


def test_sanitize_any_of_required_strips_required_from_non_object_variant() -> None:
    """
    'required' on a non-OBJECT anyOf branch is stripped.

    Gemini rejects declarations where an anyOf variant carries 'required'
    but is not itself type:object.
    """
    schema = {
        "anyOf": [
            {"type": "array", "items": {"type": "string"}, "required": ["x"]},
            {"type": "string"},
        ]
    }
    result = _sanitize_any_of_required(schema)
    assert "required" not in result["anyOf"][0]
    assert result["anyOf"][0] == {"type": "array", "items": {"type": "string"}}


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


def test_sanitize_any_of_required_implicit_object_variant_is_filtered() -> None:
    """A variant with properties but no explicit type keeps its valid required."""
    schema = {
        "anyOf": [
            {
                "properties": {"name": {"type": "string"}},
                "required": ["name", "missing_prop"],
            }
        ]
    }
    result = _sanitize_any_of_required(schema)
    assert result["anyOf"][0]["required"] == ["name"], (
        "implicit object must be filtered, not stripped wholesale"
    )


def test_sanitize_any_of_required_drops_emptied_variants() -> None:
    """Variants left empty by the strip are removed rather than sent as {}."""
    schema = {
        "properties": {"name": {"type": "string"}, "area": {"type": "string"}},
        "anyOf": [{"required": ["name"]}, {"required": ["area"]}],
    }
    result = _sanitize_any_of_required(schema)
    assert "anyOf" not in result, "a fully emptied anyOf must not reach the proto"


def test_sanitize_any_of_required_keeps_surviving_variants() -> None:
    """An emptied variant is dropped while a non-empty sibling survives."""
    schema = {
        "properties": {"name": {"type": "string"}},
        "anyOf": [{"required": ["name"]}, {"type": "string"}],
    }
    result = _sanitize_any_of_required(schema)
    assert result["anyOf"] == [{"type": "string"}]


def test_sanitize_any_of_required_records_dropped_constraint_in_description() -> None:
    """A stripped constraint is restated in the description for the model."""
    schema = {
        "properties": {
            "name": {"type": "string"},
            "area": {"type": "string"},
            "floor": {"type": "string"},
        },
        "anyOf": [
            {"required": ["name"]},
            {"required": ["area"]},
            {"required": ["floor"]},
        ],
    }
    result = _sanitize_any_of_required(schema)
    assert result["description"] == "At least one of: name, area, floor is required."


def test_sanitize_any_of_required_appends_hint_to_existing_description() -> None:
    """An existing description is preserved and the hint appended."""
    schema = {
        "description": "Turns on a device.",
        "properties": {"name": {"type": "string"}},
        "anyOf": [{"required": ["name"]}],
    }
    result = _sanitize_any_of_required(schema)
    assert result["description"] == (
        "Turns on a device. At least one of: name is required."
    )


def test_sanitize_any_of_required_multi_property_groups_in_hint() -> None:
    """Multi-property branches are rendered as groups in the hint."""
    schema = {
        "properties": {"name": {"type": "string"}, "area": {"type": "string"}},
        "anyOf": [{"required": ["name", "area"]}, {"required": ["name"]}],
    }
    result = _sanitize_any_of_required(schema)
    assert result["description"] == (
        "At least one of these property groups is required: (name, area); (name)."
    )


def test_sanitize_any_of_required_recurses_into_properties() -> None:
    """Nested object properties containing anyOf/required are also sanitized."""
    schema = {
        "type": "object",
        "properties": {
            "target": {
                "anyOf": [
                    {"type": "array", "items": {"type": "string"}, "required": ["x"]},
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
        "items": {
            "anyOf": [
                {"type": "array", "items": {"type": "string"}, "required": ["x"]},
                {"type": "string"},
            ]
        },
    }
    result = _sanitize_any_of_required(schema)
    assert "required" not in result["items"]["anyOf"][0]


def test_sanitize_any_of_required_recurses_into_nested_any_of() -> None:
    """An anyOf nested inside an anyOf variant is sanitized too."""
    schema = {
        "anyOf": [
            {
                "type": "object",
                "properties": {"inner": {"type": "string"}},
                "anyOf": [
                    {"type": "array", "items": {"type": "string"}, "required": ["x"]}
                ],
            }
        ]
    }
    result = _sanitize_any_of_required(schema)
    assert "required" not in result["anyOf"][0]["anyOf"][0]


def test_sanitize_any_of_required_non_dict_variant_passes_through() -> None:
    """Non-dict anyOf entries are left untouched rather than crashing."""
    schema = {"anyOf": ["not_a_dict", {"type": "string"}]}
    result = _sanitize_any_of_required(schema)
    assert result["anyOf"] == ["not_a_dict", {"type": "string"}]


def test_sanitize_any_of_required_malformed_required_does_not_raise() -> None:
    """A null or non-list 'required' is ignored instead of raising TypeError."""
    schema = {
        "anyOf": [
            {"type": "object", "properties": {}, "required": None},
            {"type": "object", "properties": {}, "required": "name"},
            {"type": "object", "properties": {"a": {}}, "required": ["a", 3, None]},
        ]
    }
    result = _sanitize_any_of_required(schema)
    assert result["anyOf"][0]["required"] is None, "non-list required left as-is"
    assert result["anyOf"][1]["required"] == "name"
    assert result["anyOf"][2]["required"] == ["a"], "non-str entries filtered out"


def test_sanitize_any_of_required_non_list_any_of_unchanged() -> None:
    """A malformed non-list anyOf is left alone instead of being iterated."""
    schema = {"anyOf": "bogus"}
    assert _sanitize_any_of_required(schema) == schema


def test_sanitize_any_of_required_no_any_of_unchanged() -> None:
    """Schemas without anyOf are returned unchanged."""
    schema = {"type": "object", "properties": {}, "required": ["x"]}
    result = _sanitize_any_of_required(schema)
    assert result == schema


def test_format_and_dedupe_tools_gemini_strips_invalid_any_of_required() -> None:
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
    selected, _ = _format_and_dedupe_tools(raw, "gemini")
    target = selected[0]["function"]["parameters"]["properties"]["target"]
    variants = target["anyOf"]
    assert len(variants) == 2, "the bare-required variant is dropped once emptied"
    assert variants[0]["required"] == ["area_id"], (
        "object variant keeps only required props it actually defines"
    )
    assert "required" not in variants[1], "array variant must lose 'required'"
    assert "entity_id" in target["description"], (
        "the dropped constraint must survive as description guidance"
    )


def test_format_and_dedupe_tools_gemini_rewrites_ha_target_constraint() -> None:
    """The real HA at-least-one-target schema is made Gemini-safe."""
    selected, _ = _format_and_dedupe_tools(_target_tool(), "gemini")
    params = selected[0]["function"]["parameters"]
    assert "anyOf" not in params, "bare-required branches must not reach Gemini"
    assert params["description"] == (
        "At least one of: name, area, floor is required."
    ), "constraint must be preserved as guidance"
    assert set(params["properties"]) == {"name", "area", "floor"}


def test_format_and_dedupe_tools_non_gemini_keeps_any_of_required() -> None:
    """
    Non-Gemini providers keep HA's at-least-one-target constraint intact.

    The sanitizer is subtractive: running it unconditionally would hand
    OpenAI/Anthropic/Ollama a vacuous schema, letting the model emit a
    targetless call that HA's own voluptuous validation then rejects.
    """
    for provider in ("openai", "openai_compatible", "anthropic", "ollama", None):
        selected, _ = _format_and_dedupe_tools(_target_tool(), provider)
        params = selected[0]["function"]["parameters"]
        assert params["anyOf"] == [
            {"required": ["name"]},
            {"required": ["area"]},
            {"required": ["floor"]},
        ], f"provider {provider!r} must keep the at-least-one-target constraint"
        assert "description" not in params, (
            f"provider {provider!r} needs no description workaround"
        )


def test_determine_model_name_anthropic_returns_configured_model() -> None:
    """Anthropic provider reads CONF_ANTHROPIC_CHAT_MODEL from opts."""
    opts = {CONF_ANTHROPIC_CHAT_MODEL: "claude-sonnet-4-5"}
    assert _determine_model_name("anthropic", opts) == "claude-sonnet-4-5"


def test_determine_model_name_anthropic_missing_key_returns_empty() -> None:
    """Anthropic provider with no key in opts returns empty string."""
    assert _determine_model_name("anthropic", {}) == ""

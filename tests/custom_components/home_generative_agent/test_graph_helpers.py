# ruff: noqa: S101
"""Unit tests for graph.py helper functions — Anthropic provider branches."""

from __future__ import annotations

import copy
import json

from custom_components.home_generative_agent.agent.graph import (
    _determine_model_name,
    _ensure_array_items,
    _flatten_top_level_union,
    _format_and_dedupe_tools,
    _rejected_tool_index,
    _rejected_tool_name,
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


def test_format_and_dedupe_tools_union_capable_keeps_any_of_required() -> None:
    """
    Union-capable providers keep HA's at-least-one-target constraint intact.

    Both sanitizers are subtractive: running either unconditionally would
    hand Ollama a vacuous schema, letting the model emit a targetless call
    that HA's own voluptuous validation then rejects. Unknown/unset
    providers (None) also keep the schema untouched.
    """
    for provider in ("ollama", None):
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


def test_format_and_dedupe_tools_openai_flattens_any_of_with_hint() -> None:
    """
    OpenAI-family providers get the top-level anyOf flattened with a hint.

    OpenAI's validator rejects any top-level anyOf in ``parameters``
    ("schema must have type 'object' and not have 'oneOf'/'anyOf'/'allOf'/
    'enum'/'const'/'not' at the top level"), so the union is removed — but
    the constraint must survive as description guidance, exactly like the
    Gemini path.
    """
    for provider in ("openai", "openai_compatible"):
        selected, _ = _format_and_dedupe_tools(_target_tool(), provider)
        params = selected[0]["function"]["parameters"]
        assert "anyOf" not in params, (
            f"provider {provider!r} must not receive a top-level anyOf"
        )
        assert set(params["properties"]) == {"name", "area", "floor"}
        assert params["description"] == (
            "At least one of: name, area, floor is required."
        ), "constraint must be preserved as guidance"


def test_format_and_dedupe_tools_anthropic_flattens_any_of_with_hint() -> None:
    """
    Anthropic gets the top-level anyOf flattened with a hint (issue #585).

    The Messages API refuses the union keys outright — "tools.N.custom
    .input_schema: input_schema does not support oneOf, allOf, or anyOf at
    the top level" — and rejects the request before the model sees the
    turn, so the constraint has to survive as description guidance instead.
    """
    selected, _ = _format_and_dedupe_tools(_target_tool(), "anthropic")
    params = selected[0]["function"]["parameters"]
    assert "anyOf" not in params, "Anthropic must not receive a top-level anyOf"
    assert set(params["properties"]) == {"name", "area", "floor"}
    assert params["description"] == (
        "At least one of: name, area, floor is required."
    ), "constraint must be preserved as guidance"


def test_determine_model_name_anthropic_returns_configured_model() -> None:
    """Anthropic provider reads CONF_ANTHROPIC_CHAT_MODEL from opts."""
    opts = {CONF_ANTHROPIC_CHAT_MODEL: "claude-sonnet-4-5"}
    assert _determine_model_name("anthropic", opts) == "claude-sonnet-4-5"


def test_determine_model_name_anthropic_missing_key_returns_empty() -> None:
    """Anthropic provider with no key in opts returns empty string."""
    assert _determine_model_name("anthropic", {}) == ""


# The parameters schema HA 2026.8.1 emits for HassStartTimer (and, with two
# extra slots, HassDecreaseTimer): ``vol.Required(vol.Any("hours", "minutes",
# "seconds"))`` converts to a top-level anyOf of bare-required variants.
# Verified against voluptuous_openapi.convert with llm.selector_serializer.
_HASS_START_TIMER_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "hours": {"type": "integer", "minimum": 0},
        "minutes": {"type": "integer", "minimum": 0},
        "seconds": {"type": "integer", "minimum": 0},
        "name": {"type": "string"},
        "conversation_command": {"type": "string"},
    },
    "required": [],
    "anyOf": [
        {"required": ["hours"]},
        {"required": ["minutes"]},
        {"required": ["seconds"]},
    ],
}


def _timer_tool() -> list:
    """Build a raw tool carrying the real HassStartTimer schema."""
    return [
        {
            "name": "HassStartTimer",
            "api_id": "assist",
            "description": "Starts a new timer",
            "parameters": json.dumps(_HASS_START_TIMER_SCHEMA),
            "is_actuation": True,
        }
    ]


def test_flatten_top_level_union_bare_required_variants_become_hint() -> None:
    """
    Bare-required anyOf variants are dropped but restated as a hint.

    This is the shape production actually emits (HassStartTimer /
    HassDecreaseTimer): the variants carry no ``properties``, so there is
    nothing to merge — the at-least-one constraint must survive in the
    description instead of vanishing.
    """
    expected_properties = copy.deepcopy(_HASS_START_TIMER_SCHEMA["properties"])
    result = _flatten_top_level_union(copy.deepcopy(_HASS_START_TIMER_SCHEMA))
    assert "anyOf" not in result
    assert result["type"] == "object"
    assert result["properties"] == expected_properties, (
        "property schemas must survive the flatten byte-for-byte"
    )
    assert result["description"] == (
        "At least one of: hours, minutes, seconds is required."
    )


def test_flatten_top_level_union_merges_any_of_properties() -> None:
    """
    Full-object anyOf variants are merged into one permissive schema.

    Hypothetical union shape (per-variant properties): each arm's
    properties are unioned and the arms' mutual exclusivity is dropped.
    """
    schema = {
        "anyOf": [
            {"type": "object", "properties": {"duration": {"type": "string"}}},
            {
                "type": "object",
                "properties": {
                    "hours": {"type": "integer"},
                    "minutes": {"type": "integer"},
                },
            },
        ]
    }
    result = _flatten_top_level_union(schema)
    assert result["type"] == "object"
    assert "anyOf" not in result
    assert set(result["properties"]) == {"duration", "hours", "minutes"}


def test_flatten_top_level_union_one_of_required_becomes_hint() -> None:
    """OneOf variant ``required`` entries are restated, not silently lost."""
    schema = {
        "oneOf": [
            {"properties": {"a": {"type": "string"}}, "required": ["a"]},
            {"properties": {"b": {"type": "string"}}, "required": ["b"]},
        ]
    }
    result = _flatten_top_level_union(schema)
    assert "oneOf" not in result
    assert set(result["properties"]) == {"a", "b"}
    assert result["description"] == "At least one of: a, b is required."


def test_flatten_top_level_union_all_of_merges_required() -> None:
    """
    AllOf variants are conjunctive: their ``required`` entries all apply.

    Unlike anyOf/oneOf, flattening allOf must union the ``required`` lists
    into the top level instead of demoting them to a description hint.
    """
    schema = {
        "allOf": [
            {"properties": {"a": {"type": "string"}}, "required": ["a"]},
            {"properties": {"b": {"type": "string"}}, "required": ["b", "ghost"]},
        ]
    }
    result = _flatten_top_level_union(schema)
    assert "allOf" not in result
    assert set(result["properties"]) == {"a", "b"}
    assert result["required"] == ["a", "b"], (
        "conjunctive required survives; entries without a property are dropped"
    )
    assert "description" not in result


def test_flatten_top_level_union_merges_every_union_key_present() -> None:
    """
    All union keys present are merged, not just the first one found.

    Stripping a key whose variants were never merged would silently delete
    that union's properties.
    """
    schema = {
        "oneOf": [{"properties": {"a": {"type": "string"}}}],
        "anyOf": [{"properties": {"b": {"type": "string"}}}],
    }
    result = _flatten_top_level_union(schema)
    assert "oneOf" not in result
    assert "anyOf" not in result
    assert set(result["properties"]) == {"a", "b"}


def test_flatten_top_level_union_no_union_key_unchanged() -> None:
    """Schemas without a top-level union key pass through untouched."""
    expected = {"type": "object", "properties": {"x": {"type": "string"}}}
    schema = copy.deepcopy(expected)
    result = _flatten_top_level_union(schema)
    assert result == expected
    assert schema == expected, "input schema must not be mutated"


def test_flatten_top_level_union_non_list_union_value_unchanged() -> None:
    """A union key whose value is not a list is left in place, untouched."""
    expected = {"anyOf": {"type": "string"}}
    schema = copy.deepcopy(expected)
    result = _flatten_top_level_union(schema)
    assert result == expected
    assert schema == expected, "input schema must not be mutated"


def test_flatten_top_level_union_non_dict_variant_properties_ignored() -> None:
    """
    A variant whose ``properties`` is not a dict is skipped, not raised.

    A malformed third-party or store-persisted tool schema must not crash
    the tool-retrieval pass for OpenAI-provider users.
    """
    schema = {
        "anyOf": [
            {"properties": ["a", "b"]},
            {"properties": {"c": {"type": "string"}}},
        ]
    }
    result = _flatten_top_level_union(schema)
    assert result["properties"] == {"c": {"type": "string"}}


def test_flatten_top_level_union_string_required_ignored() -> None:
    """
    A variant whose ``required`` is a string is ignored, not char-split.

    Iterating a string yields its characters — without the list guard the
    hint would read "At least one of: h, o, u, r, s is required.".
    """
    schema = {"anyOf": [{"required": "hours"}]}
    result = _flatten_top_level_union(schema)
    assert "description" not in result
    assert result == {"type": "object", "properties": {}}


def test_flatten_top_level_union_non_dict_top_level_properties() -> None:
    """A non-dict pre-existing ``properties`` is replaced, not raised."""
    schema = {
        "anyOf": [{"properties": {"a": {"type": "string"}}}],
        "properties": "bogus",
    }
    result = _flatten_top_level_union(schema)
    assert result["properties"] == {"a": {"type": "string"}}


def test_flatten_top_level_union_empty_variants_list() -> None:
    """An empty variants list flattens to a bare object schema, no hint."""
    result = _flatten_top_level_union({"anyOf": []})
    assert result == {"type": "object", "properties": {}}


def test_flatten_top_level_union_preserves_existing_properties() -> None:
    """Pre-existing top-level properties win over merged variant properties."""
    schema = {
        "anyOf": [{"properties": {"a": {"type": "string"}}}],
        "properties": {"a": {"type": "integer"}},
    }
    result = _flatten_top_level_union(schema)
    assert result["properties"]["a"] == {"type": "integer"}


def test_flatten_top_level_union_non_dict_variant_skipped() -> None:
    """Non-dict variants are skipped without raising; dict variants still merge."""
    schema = {
        "anyOf": [
            "not-a-dict",
            {"properties": {"a": {"type": "string"}}, "required": ["a"]},
        ]
    }
    result = _flatten_top_level_union(schema)
    assert "anyOf" not in result
    assert set(result["properties"]) == {"a"}
    assert result["description"] == "At least one of: a is required."


def test_flatten_top_level_union_malformed_required_does_not_raise() -> None:
    """Non-string ``required`` entries are dropped; string entries still hint."""
    schema = {
        "anyOf": [
            {"required": [None, 5, "a"]},
            {"required": [{"nested": "junk"}]},
        ],
        "properties": {"a": {"type": "string"}},
    }
    result = _flatten_top_level_union(schema)
    assert "anyOf" not in result
    assert result["description"] == "At least one of: a is required.", (
        "only the string entry may survive into the hint"
    )


def test_flatten_top_level_union_all_of_merges_with_existing_required() -> None:
    """AllOf ``required`` unions with pre-existing top-level required, deduped."""
    schema = {
        "type": "object",
        "properties": {"x": {"type": "string"}},
        "required": ["x", 5],
        "allOf": [
            {"properties": {"a": {"type": "string"}}, "required": ["a", "x"]},
        ],
    }
    result = _flatten_top_level_union(schema)
    assert "allOf" not in result
    assert result["required"] == ["x", "a"], (
        "existing entries come first, duplicates and non-strings are dropped"
    )


def test_flatten_top_level_union_multi_property_group_hint() -> None:
    """A variant requiring several properties gets the group-form hint."""
    schema = {
        "anyOf": [
            {"required": ["a", "b"]},
            {"required": ["c"]},
        ],
        "properties": {
            "a": {"type": "string"},
            "b": {"type": "string"},
            "c": {"type": "string"},
        },
    }
    result = _flatten_top_level_union(schema)
    assert result["description"] == (
        "At least one of these property groups is required: (a, b); (c)."
    )


def test_format_and_dedupe_tools_openai_flatten_then_array_items() -> None:
    """
    Pass ordering: properties hoisted by the flatten still get array items.

    ``_ensure_array_items()`` runs after the flatten, so an array property
    that only existed inside a union variant must still receive the
    ``items`` key Gemini/OpenAI strict schemas expect.
    """
    raw: list = [
        {
            "name": "union_array_tool",
            "api_id": "test",
            "description": "Union variant carrying an itemless array",
            "parameters": json.dumps(
                {"anyOf": [{"properties": {"targets": {"type": "array"}}}]}
            ),
            "is_actuation": False,
        }
    ]
    selected, _ = _format_and_dedupe_tools(raw, "openai")
    params = selected[0]["function"]["parameters"]
    assert "anyOf" not in params
    assert params["properties"]["targets"] == {
        "type": "array",
        "items": {"type": "string"},
    }


def test_format_and_dedupe_tools_openai_flattens_hass_start_timer() -> None:
    """
    End-to-end regression: HassStartTimer no longer 400s on OpenAI.

    Reproduces openai.BadRequestError: "Invalid schema for function
    'HassStartTimer': schema must have type 'object' and not have
    'oneOf'/'anyOf'/'allOf'/'enum'/'const'/'not' at the top level", using
    the schema HA 2026.8.1 actually emits (timer tools are only exposed
    for timer-capable satellite devices, hence voice-request-only 400s).
    """
    for provider in ("openai", "openai_compatible"):
        selected, _ = _format_and_dedupe_tools(_timer_tool(), provider)
        params = selected[0]["function"]["parameters"]
        assert params["type"] == "object"
        assert "anyOf" not in params
        assert "oneOf" not in params
        assert "allOf" not in params
        assert params["properties"] == _HASS_START_TIMER_SCHEMA["properties"], (
            f"provider {provider!r} must keep every property schema intact"
        )
        assert params["description"] == (
            "At least one of: hours, minutes, seconds is required."
        ), f"provider {provider!r} must keep the constraint as guidance"


def test_format_and_dedupe_tools_anthropic_flattens_hass_start_timer() -> None:
    """
    End-to-end regression: HassStartTimer no longer 400s on Anthropic.

    Reproduces issue #585 — anthropic.BadRequestError "tools.3.custom
    .input_schema: input_schema does not support oneOf, allOf, or anyOf at
    the top level" — with the schema HA 2026.8.2 actually emits.
    langchain_anthropic copies ``parameters`` into ``input_schema``
    verbatim, so an unflattened union reaches the API untouched.
    """
    selected, _ = _format_and_dedupe_tools(_timer_tool(), "anthropic")
    params = selected[0]["function"]["parameters"]
    assert params["type"] == "object"
    assert not {"anyOf", "oneOf", "allOf"} & set(params)
    assert params["properties"] == _HASS_START_TIMER_SCHEMA["properties"], (
        "Anthropic must keep every property schema intact"
    )
    assert params["description"] == (
        "At least one of: hours, minutes, seconds is required."
    ), "Anthropic must keep the constraint as guidance"


def test_format_and_dedupe_tools_union_capable_keeps_hass_start_timer() -> None:
    """Ollama/unset providers keep the timer union schema fully intact."""
    for provider in ("ollama", None):
        selected, _ = _format_and_dedupe_tools(_timer_tool(), provider)
        params = selected[0]["function"]["parameters"]
        assert params == _HASS_START_TIMER_SCHEMA, (
            f"provider {provider!r} must receive the schema completely untouched"
        )


def test_format_and_dedupe_tools_gemini_rewrites_hass_start_timer() -> None:
    """
    Gemini still sees the timer anyOf and restates it via the sanitizer.

    Guards the pass ordering: the OpenAI flatten must not run for Gemini,
    or _sanitize_any_of_required would never see the top-level anyOf and
    the description hint would be lost.
    """
    selected, _ = _format_and_dedupe_tools(_timer_tool(), "gemini")
    params = selected[0]["function"]["parameters"]
    assert "anyOf" not in params
    assert params["description"] == (
        "At least one of: hours, minutes, seconds is required."
    )
    assert params["properties"] == _HASS_START_TIMER_SCHEMA["properties"]


def test_format_and_dedupe_tools_openai_keeps_nested_union() -> None:
    """
    OpenAI flatten is top-level only: property-level unions must survive.

    OpenAI rejects oneOf/anyOf/allOf only at the top level of
    ``parameters``; nested unions (e.g. selector shapes) are valid and
    carry real typing the model needs. A future recursive rewrite of
    ``_flatten_top_level_union`` must fail this test.
    """
    schema = {
        "type": "object",
        "properties": {
            "color": {
                "anyOf": [
                    {"type": "string"},
                    {"type": "array", "items": {"type": "integer"}},
                ]
            }
        },
    }
    raw: list = [
        {
            "name": "nested_union_tool",
            "api_id": "test",
            "description": "d",
            "parameters": json.dumps(schema),
            "is_actuation": False,
        }
    ]
    selected, _ = _format_and_dedupe_tools(raw, "openai")
    color = selected[0]["function"]["parameters"]["properties"]["color"]
    # _ensure_array_items additionally hoists items from the array variant
    # (Gemini workaround, provider-independent) — the union itself must be
    # byte-for-byte intact.
    assert color["anyOf"] == schema["properties"]["color"]["anyOf"]


def test_format_and_dedupe_tools_openai_plain_schema_untouched() -> None:
    """
    The common case: a no-union schema passes through OpenAI unmodified.

    The overwhelming majority of HA tool schemas have no top-level union;
    the gated flatten branch must be a no-op for them.
    """
    schema = {
        "type": "object",
        "properties": {"name": {"type": "string"}},
        "required": ["name"],
    }
    for provider in ("openai", "openai_compatible"):
        raw: list = [
            {
                "name": "plain_tool",
                "api_id": "test",
                "description": "d",
                "parameters": json.dumps(schema),
                "is_actuation": False,
            }
        ]
        selected, _ = _format_and_dedupe_tools(raw, provider)
        assert selected[0]["function"]["parameters"] == schema, (
            f"provider {provider!r} must not modify a plain object schema"
        )


# ----- Tool-schema rejection parsing (issue #585) -----


def _named_tool(name: str) -> dict:
    """Return a formatted tool entry carrying only the name."""
    return {"type": "function", "function": {"name": name, "parameters": {}}}


def test_rejected_tool_index_reads_anthropic_position() -> None:
    """The Anthropic 400 names the offending tool only by position."""
    err = Exception(
        "Error code: 400 - {'type': 'error', 'error': {'type': "
        "'invalid_request_error', 'message': 'tools.3.custom.input_schema: "
        "input_schema does not support oneOf, allOf, or anyOf at the top "
        "level'}}"
    )
    assert _rejected_tool_index(err) == 3


def test_rejected_tool_index_reads_gemini_position() -> None:
    """Gemini reports the same class of failure with a bracketed index."""
    err = Exception(
        "GenerateContentRequest.tools[0].function_declarations[1].parameters"
        ".any_of[0].required: only allowed for OBJECT type; invalid schema"
    )
    assert _rejected_tool_index(err) == 1


def test_rejected_tool_index_walks_the_cause_chain() -> None:
    """The provider error arrives wrapped in a HomeAssistantError."""
    cause = Exception("tools.2.custom.input_schema: bad schema")
    wrapper = Exception("Model invocation failed")
    wrapper.__cause__ = cause
    assert _rejected_tool_index(wrapper) == 2


def test_rejected_tool_index_ignores_unrelated_errors() -> None:
    """Errors that are not about a tool schema must not trigger a drop."""
    assert _rejected_tool_index(Exception("connection reset by peer")) is None
    assert _rejected_tool_index(Exception("tools.2 were slow to run")) is None, (
        "a positional match without 'schema' is not a schema rejection"
    )


def test_rejected_tool_index_rejects_absurd_position() -> None:
    """A mis-parsed message cannot turn into an unbounded int()."""
    err = Exception(f"tools.{'9' * 40}.custom.input_schema: bad schema")
    assert _rejected_tool_index(err) is None


def test_rejected_tool_name_resolves_and_guards() -> None:
    """Positions map back onto the bound tool list; bad ones return None."""
    tools = [_named_tool("GetLiveContext"), _named_tool("HassStartTimer")]
    assert _rejected_tool_name(tools, 1) == "HassStartTimer"
    assert _rejected_tool_name(tools, None) is None
    assert _rejected_tool_name(tools, 2) is None, "out-of-range index"
    assert _rejected_tool_name(tools, -1) is None, "negative index must not wrap"
    assert _rejected_tool_name(["not_a_dict"], 0) is None
    assert _rejected_tool_name([{"function": "not_a_dict"}], 0) is None
    assert _rejected_tool_name([{"function": {"name": 7}}], 0) is None

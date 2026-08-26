# ruff: noqa: S101
"""Tests for the per-tool exclusion list (issue #570)."""

from __future__ import annotations

from dataclasses import dataclass
from logging import WARNING
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import patch

import pytest
import voluptuous as vol
from homeassistant.const import CONF_LLM_HASS_API
from homeassistant.data_entry_flow import FlowResultType
from homeassistant.helpers import llm
from pytest_homeassistant_custom_component.common import MockConfigEntry

from custom_components.home_generative_agent.agent.helpers import (
    TOOL_TEXT_MAX_LEN,
    api_id_is_form_representable,
    filter_excluded_tools,
    normalize_tool_exclusions,
    sanitize_tool_text,
    split_tool_index_key,
    tool_exclusions,
    tool_index_key,
)
from custom_components.home_generative_agent.config_flow import (
    _SUFFIX_NOT_AVAILABLE,
    _SUFFIX_NOT_SELECTED,
    HomeGenerativeAgentOptionsFlow,
    _label_text,
    _list_as_tool_exclusions,
    _schema_for_options,
    _tool_exclusions_as_list,
)
from custom_components.home_generative_agent.const import (
    CONF_MAX_MESSAGES_IN_CONTEXT,
    CONF_TOOL_EXCLUSIONS,
    DOMAIN,
)

_CONFIG_FLOW = "custom_components.home_generative_agent.config_flow"


class _FakeTool(llm.Tool):
    """Minimal llm.Tool stand-in."""

    def __init__(self, name: str) -> None:
        self.name = name
        self.description = f"{name} description"
        self.parameters = vol.Schema({})


@dataclass
class _FakeAPI:
    """Stand-in for a registered llm.API."""

    id: str
    name: str


def _api_instance(tools: list[llm.Tool], api_id: str = "mcp-abc") -> llm.APIInstance:
    """Build an APIInstance whose tool list is the object handed in."""
    return llm.APIInstance(
        api=cast("Any", _FakeAPI(id=api_id, name="My MCP Server")),
        api_prompt="prompt",
        llm_context=cast("Any", None),
        tools=tools,
    )


# --------------------------------------------------------------------------
# Normalization
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "raw",
    [None, "", [], "web_search", 0, {"": ["a"]}, {5: ["a"]}, {"api": None}],
)
def test_normalize_tool_exclusions_rejects_degenerate_shapes(raw: Any) -> None:
    """Anything that is not a usable {api_id: [name]} map collapses to {}."""
    assert normalize_tool_exclusions(raw) == {}


def test_normalize_tool_exclusions_cleans_and_dedupes() -> None:
    """Non-string and empty names are dropped; the rest are deduped and sorted."""
    assert normalize_tool_exclusions(
        {
            "mcp-abc": ["b_tool", "a_tool", "a_tool", "", None, 7],
            "mcp-empty": [],
            "assist": "HassTurnOn",
        }
    ) == {"mcp-abc": ["a_tool", "b_tool"], "assist": ["HassTurnOn"]}


def test_normalize_tool_exclusions_drops_api_with_no_surviving_names() -> None:
    """'Present but empty' and 'absent' must not both spell the default."""
    assert normalize_tool_exclusions({"mcp-abc": ["", None]}) == {}


def test_tool_exclusions_reads_the_option_key() -> None:
    """The runtime accessor lifts the option into per-API lookup sets."""
    assert tool_exclusions({CONF_TOOL_EXCLUSIONS: {"mcp-abc": ["a", "b"]}}) == {
        "mcp-abc": {"a", "b"}
    }
    assert tool_exclusions({}) == {}


def test_split_tool_index_key_round_trips() -> None:
    """Keys split on the first separator only; malformed keys return None."""
    assert split_tool_index_key(tool_index_key("mcp-abc", "web_search")) == (
        "mcp-abc",
        "web_search",
    )
    # A remote server may name a tool anything, including the separator.
    assert split_tool_index_key("mcp-abc::odd::name") == ("mcp-abc", "odd::name")
    assert split_tool_index_key("no_separator") is None
    assert split_tool_index_key("::name") is None
    assert split_tool_index_key("api::") is None


# --------------------------------------------------------------------------
# Runtime filtering
# --------------------------------------------------------------------------


def test_filter_excluded_tools_removes_only_named_tools() -> None:
    """Excluded names are dropped and reported; the rest survive in order."""
    api = _api_instance([_FakeTool("a"), _FakeTool("b"), _FakeTool("c")])

    filtered, dropped = filter_excluded_tools("mcp-abc", api, {"mcp-abc": {"b"}})

    assert [tool.name for tool in filtered.tools] == ["a", "c"]
    assert dropped == ["b"]


def test_filter_excluded_tools_ignores_other_apis() -> None:
    """An exclusion recorded for another API never touches this one."""
    api = _api_instance([_FakeTool("a")])

    filtered, dropped = filter_excluded_tools("mcp-abc", api, {"assist": {"a"}})

    assert filtered is api
    assert dropped == []


def test_filter_excluded_tools_ignores_names_the_api_does_not_have() -> None:
    """A stale exclusion for a tool the server dropped is a no-op."""
    api = _api_instance([_FakeTool("a")])

    filtered, dropped = filter_excluded_tools("mcp-abc", api, {"mcp-abc": {"gone"}})

    assert filtered is api
    assert dropped == []


def test_filter_excluded_tools_does_not_mutate_the_shared_tool_list() -> None:
    """
    The source list must survive untouched.

    Home Assistant's MCP integration hands every APIInstance the same
    coordinator-owned list object, so an in-place filter would strip the tools
    from every other consumer of that server until the next 30-minute refresh.
    """
    shared: list[llm.Tool] = [_FakeTool("a"), _FakeTool("b")]
    api = _api_instance(shared)

    filtered, _dropped = filter_excluded_tools("mcp-abc", api, {"mcp-abc": {"b"}})

    assert [tool.name for tool in shared] == ["a", "b"]
    assert filtered.tools is not shared
    assert api.tools is shared


def test_filter_excluded_tools_preserves_every_other_field() -> None:
    """Only `tools` changes; prompt, context and serializer carry over."""
    tools: list[llm.Tool] = [_FakeTool("a"), _FakeTool("b")]
    api = llm.APIInstance(
        api=cast("Any", _FakeAPI(id="mcp-abc", name="My MCP Server")),
        api_prompt="the prompt",
        llm_context=cast("Any", "the context"),
        tools=tools,
        custom_serializer=str,
    )

    filtered, _dropped = filter_excluded_tools("mcp-abc", api, {"mcp-abc": {"b"}})

    assert filtered.api is api.api
    assert filtered.api_prompt == "the prompt"
    assert filtered.llm_context == "the context"
    assert filtered.custom_serializer is str


def test_filter_excluded_tools_can_empty_an_api() -> None:
    """Excluding every tool is expressible and leaves the API loadable."""
    api = _api_instance([_FakeTool("a")])

    filtered, dropped = filter_excluded_tools("mcp-abc", api, {"mcp-abc": {"a"}})

    assert filtered.tools == []
    assert dropped == ["a"]


# --------------------------------------------------------------------------
# Options form: picker construction
# --------------------------------------------------------------------------


def _schema_key(schema: dict[Any, Any], key_name: str) -> Any:
    """Return a schema marker by name."""
    return next(key for key in schema if cast("Any", key).schema == key_name)


def _selector_options(schema: dict[Any, Any]) -> list[dict[str, str]]:
    """Return the exclusion picker's rendered options."""
    selector = schema[_schema_key(schema, CONF_TOOL_EXCLUSIONS)]
    return cast("Any", selector).config["options"]


def _patch_apis(apis: list[_FakeAPI], instances: dict[str, Any]) -> Any:
    """Patch llm.async_get_apis / async_get_api for the config-flow module."""

    async def _get_api(_hass: Any, api_id: str, _ctx: Any) -> Any:
        result = instances[api_id]
        if isinstance(result, Exception):
            raise result
        return result

    return (
        patch(f"{_CONFIG_FLOW}.llm.async_get_apis", return_value=apis),
        patch(f"{_CONFIG_FLOW}.llm.async_get_api", side_effect=_get_api),
    )


@pytest.mark.asyncio
async def test_picker_lists_live_tools_of_selected_apis(hass: Any) -> None:
    """Every tool of every selected API is offered, labelled by API name."""
    apis = [_FakeAPI(id="mcp-abc", name="My MCP Server")]
    instances = {"mcp-abc": _api_instance([_FakeTool("web_search"), _FakeTool("ping")])}
    get_apis, get_api = _patch_apis(apis, instances)

    with get_apis, get_api:
        schema = await _schema_for_options(hass, {CONF_LLM_HASS_API: ["mcp-abc"]})

    assert _selector_options(schema) == [
        {"label": "My MCP Server: ping", "value": "mcp-abc::ping"},
        {"label": "My MCP Server: web_search", "value": "mcp-abc::web_search"},
    ]


@pytest.mark.asyncio
async def test_picker_preserves_exclusions_of_an_unreachable_server(
    hass: Any,
) -> None:
    """
    An offline server keeps its stored exclusions as selectable options.

    Dropping them would fail SelectSelector validation on submit (the
    unsaveable-form class of issue #568) and silently re-expose the tools.
    """
    apis = [_FakeAPI(id="mcp-abc", name="My MCP Server")]
    instances = {"mcp-abc": TimeoutError("server unreachable")}
    get_apis, get_api = _patch_apis(apis, instances)

    with get_apis, get_api:
        schema = await _schema_for_options(
            hass,
            {
                CONF_LLM_HASS_API: ["mcp-abc"],
                CONF_TOOL_EXCLUSIONS: {"mcp-abc": ["web_search"]},
            },
        )

    assert _selector_options(schema) == [
        {
            "label": "My MCP Server: web_search (not currently available)",
            "value": "mcp-abc::web_search",
        }
    ]
    marker = _schema_key(schema, CONF_TOOL_EXCLUSIONS)
    assert marker.description == {"suggested_value": ["mcp-abc::web_search"]}


@pytest.mark.asyncio
async def test_picker_preserves_exclusions_of_a_deregistered_api(hass: Any) -> None:
    """An exclusion whose API is gone is labelled by its raw id, not dropped."""
    get_apis, get_api = _patch_apis([], {})

    with get_apis, get_api:
        schema = await _schema_for_options(
            hass, {CONF_TOOL_EXCLUSIONS: {"mcp-gone": ["web_search"]}}
        )

    assert _selector_options(schema) == [
        {
            "label": "mcp-gone: web_search (not currently available)",
            "value": "mcp-gone::web_search",
        }
    ]


@pytest.mark.asyncio
async def test_picker_is_omitted_when_there_is_nothing_to_list(hass: Any) -> None:
    """
    No listable tools means no field at all.

    An absent field never appears in `user_input`, which is what keeps a
    render-time outage from writing an empty selection over a stored one.
    """
    get_apis, get_api = _patch_apis([], {})

    with get_apis, get_api:
        schema = await _schema_for_options(hass, {})

    assert CONF_TOOL_EXCLUSIONS not in [str(cast("Any", key).schema) for key in schema]


@pytest.mark.asyncio
async def test_picker_survives_one_failing_api(hass: Any) -> None:
    """One unreachable server must not empty the whole picker."""
    apis = [
        _FakeAPI(id="mcp-abc", name="Server A"),
        _FakeAPI(id="mcp-def", name="Server B"),
    ]
    instances = {
        "mcp-abc": TimeoutError("down"),
        "mcp-def": _api_instance([_FakeTool("ping")], api_id="mcp-def"),
    }
    get_apis, get_api = _patch_apis(apis, instances)

    with get_apis, get_api:
        schema = await _schema_for_options(
            hass, {CONF_LLM_HASS_API: ["mcp-abc", "mcp-def"]}
        )

    assert _selector_options(schema) == [
        {"label": "Server B: ping", "value": "mcp-def::ping"}
    ]


@pytest.mark.asyncio
async def test_picker_renders_after_the_relevance_threshold(hass: Any) -> None:
    """The picker sits with the other tool-selection settings."""
    apis = [_FakeAPI(id="mcp-abc", name="My MCP Server")]
    instances = {"mcp-abc": _api_instance([_FakeTool("ping")])}
    get_apis, get_api = _patch_apis(apis, instances)

    with get_apis, get_api:
        schema = await _schema_for_options(hass, {CONF_LLM_HASS_API: ["mcp-abc"]})

    keys = [str(cast("Any", key).schema) for key in schema]
    assert keys[keys.index("tool_relevance_threshold") + 1] == CONF_TOOL_EXCLUSIONS


# --------------------------------------------------------------------------
# Options form: storage round trip
# --------------------------------------------------------------------------


def test_tool_exclusions_as_list_flattens_the_stored_map() -> None:
    """Storage shape renders as flat composite picker values."""
    assert _tool_exclusions_as_list(
        {"mcp-abc": ["b", "a"], "assist": ["HassTurnOn"]}
    ) == [
        "assist::HassTurnOn",
        "mcp-abc::a",
        "mcp-abc::b",
    ]


def test_tool_exclusions_as_list_passes_a_submitted_list_through() -> None:
    """A re-render after a validation error sees the raw submitted list."""
    assert _tool_exclusions_as_list(["mcp-abc::a", "", 7]) == ["mcp-abc::a"]


def test_list_as_tool_exclusions_groups_by_api() -> None:
    """Picker values group back into the stored map, malformed ones dropped."""
    assert _list_as_tool_exclusions(
        ["mcp-abc::b", "mcp-abc::a", "assist::HassTurnOn", "junk", 7]
    ) == {"assist": ["HassTurnOn"], "mcp-abc": ["a", "b"]}


def _options_flow(hass: Any, options: dict[str, Any]) -> Any:
    """Build an options flow bound to a stored config entry."""
    entry = MockConfigEntry(
        domain=DOMAIN, title="Home Generative Agent", options=options
    )
    entry.add_to_hass(hass)
    flow = HomeGenerativeAgentOptionsFlow()
    flow.hass = hass
    flow.handler = entry.entry_id
    return flow


@pytest.mark.asyncio
async def test_parse_tool_exclusions_stores_the_grouped_map(hass: Any) -> None:
    """A submitted selection is normalized into the stored shape."""
    flow = _options_flow(hass, {})
    options: dict[str, Any] = {CONF_TOOL_EXCLUSIONS: ["mcp-abc::b", "mcp-abc::a"]}

    flow._parse_tool_exclusions(options)

    assert options == {CONF_TOOL_EXCLUSIONS: {"mcp-abc": ["a", "b"]}}


@pytest.mark.asyncio
async def test_parse_tool_exclusions_drops_an_empty_selection(hass: Any) -> None:
    """'Exclude nothing' is stored as an absent key, never as {}."""
    flow = _options_flow(hass, {})
    options: dict[str, Any] = {CONF_TOOL_EXCLUSIONS: []}

    flow._parse_tool_exclusions(options)

    assert CONF_TOOL_EXCLUSIONS not in options


def test_parse_tool_exclusions_keeps_an_untouched_stored_map() -> None:
    """When the picker was not rendered the stored map survives the save."""
    flow = HomeGenerativeAgentOptionsFlow()
    options: dict[str, Any] = {CONF_TOOL_EXCLUSIONS: {"mcp-abc": ["web_search"]}}

    flow._parse_tool_exclusions(options)

    assert options == {CONF_TOOL_EXCLUSIONS: {"mcp-abc": ["web_search"]}}


# --------------------------------------------------------------------------
# End to end against real Home Assistant llm plumbing
#
# Every test above mocks `llm.async_get_api`, so none of them prove the picker
# survives contact with a genuinely registered API — the exact gap that let a
# "the field just isn't in my form" report go unexplained. These drive the real
# registry instead.
# --------------------------------------------------------------------------


@dataclass(kw_only=True)
class _McpLikeAPI(llm.API):
    """
    A registered API shaped like the one HA's MCP integration exposes.

    Critically it returns the *same* list object on every call, the way
    ``ModelContextProtocolAPI`` hands out ``self.coordinator.data``.
    """

    shared_tools: list[llm.Tool]

    async def async_get_api_instance(self, llm_context: Any) -> llm.APIInstance:
        return llm.APIInstance(self, "mcp prompt", llm_context, tools=self.shared_tools)


@pytest.mark.asyncio
async def test_picker_renders_for_a_really_registered_api(
    hass: Any, caplog: Any
) -> None:
    """The field reaches the form, in position, with no enumeration warning."""
    shared: list[llm.Tool] = [_FakeTool("get_current_time"), _FakeTool("convert_time")]
    llm.async_register_api(
        hass,
        _McpLikeAPI(hass=hass, id="mcp-time", name="mcp-time", shared_tools=shared),
    )

    schema = await _schema_for_options(hass, {CONF_LLM_HASS_API: ["mcp-time"]})

    keys = [str(cast("Any", key).schema) for key in schema]
    assert keys[keys.index("tool_relevance_threshold") + 1] == CONF_TOOL_EXCLUSIONS
    assert keys[keys.index(CONF_TOOL_EXCLUSIONS) + 1] == "video_analyzer_mode"
    assert _selector_options(schema) == [
        {"label": "mcp-time: convert_time", "value": "mcp-time::convert_time"},
        {
            "label": "mcp-time: get_current_time",
            "value": "mcp-time::get_current_time",
        },
    ]
    # Scoped to this module's logger on purpose: the intent is narrowly "the
    # picker did not hit its enumeration-failure branch". An unscoped check
    # would fail on any unrelated HA warning raised during the call, with a
    # message pointing nowhere near the cause.
    assert not [
        record
        for record in caplog.records
        if record.levelno >= WARNING and record.name == _CONFIG_FLOW
    ]


@pytest.mark.asyncio
async def test_real_api_enumeration_leaves_the_shared_tool_list_alone(
    hass: Any,
) -> None:
    """Rendering the picker must not disturb the coordinator-owned list."""
    shared: list[llm.Tool] = [_FakeTool("get_current_time"), _FakeTool("convert_time")]
    llm.async_register_api(
        hass,
        _McpLikeAPI(hass=hass, id="mcp-time", name="mcp-time", shared_tools=shared),
    )

    await _schema_for_options(
        hass,
        {
            CONF_LLM_HASS_API: ["mcp-time"],
            CONF_TOOL_EXCLUSIONS: {"mcp-time": ["convert_time"]},
        },
    )

    assert [tool.name for tool in shared] == ["get_current_time", "convert_time"]


# --------------------------------------------------------------------------
# Coverage-audit additions
# --------------------------------------------------------------------------


def test_filter_excluded_tools_reports_every_drop_sorted() -> None:
    """The dropped list is the sorted intersection, not just the first hit."""
    api = _api_instance([_FakeTool("c"), _FakeTool("a"), _FakeTool("b")])

    filtered, dropped = filter_excluded_tools(
        "mcp-abc", api, {"mcp-abc": {"c", "a", "never_registered"}}
    )

    assert [tool.name for tool in filtered.tools] == ["b"]
    assert dropped == ["a", "c"]


@pytest.mark.asyncio
async def test_picker_dedupes_a_repeated_api_selection(hass: Any) -> None:
    """
    A duplicated API id must not emit duplicate picker options.

    ``normalize_llm_api_value`` deliberately does not dedupe, so the same id
    can reach the picker twice; duplicate SelectSelector option values render
    as a doubled dropdown entry.
    """
    apis = [_FakeAPI(id="mcp-abc", name="My MCP Server")]
    instances = {"mcp-abc": _api_instance([_FakeTool("ping")])}
    get_apis, get_api = _patch_apis(apis, instances)

    with get_apis, get_api:
        schema = await _schema_for_options(
            hass, {CONF_LLM_HASS_API: ["mcp-abc", "mcp-abc"]}
        )

    assert _selector_options(schema) == [
        {"label": "My MCP Server: ping", "value": "mcp-abc::ping"}
    ]


@pytest.mark.asyncio
async def test_picker_does_not_relabel_an_excluded_live_tool(hass: Any) -> None:
    """
    A stored exclusion of a still-live tool stays the plain live option.

    The stored-selection top-up must skip anything already enumerated, or an
    excluded-but-available tool would appear twice, the second time falsely
    labelled "not currently available".
    """
    apis = [_FakeAPI(id="mcp-abc", name="My MCP Server")]
    instances = {"mcp-abc": _api_instance([_FakeTool("ping"), _FakeTool("web_search")])}
    get_apis, get_api = _patch_apis(apis, instances)

    with get_apis, get_api:
        schema = await _schema_for_options(
            hass,
            {
                CONF_LLM_HASS_API: ["mcp-abc"],
                CONF_TOOL_EXCLUSIONS: {"mcp-abc": ["ping"]},
            },
        )

    assert _selector_options(schema) == [
        {"label": "My MCP Server: ping", "value": "mcp-abc::ping"},
        {"label": "My MCP Server: web_search", "value": "mcp-abc::web_search"},
    ]


@pytest.mark.asyncio
async def test_picker_keeps_a_malformed_stored_value_selectable(hass: Any) -> None:
    """
    An unsplittable carried-over value is still offered, labelled by itself.

    A re-render after a validation error replays the raw submitted list, so a
    value with no separator can reach the top-up loop; dropping it would fail
    SelectSelector validation against the pre-filled selection.
    """
    get_apis, get_api = _patch_apis([], {})

    with get_apis, get_api:
        schema = await _schema_for_options(hass, {CONF_TOOL_EXCLUSIONS: ["junk"]})

    assert _selector_options(schema) == [
        {"label": "junk: junk (not currently available)", "value": "junk"}
    ]


@pytest.mark.asyncio
async def test_picker_replays_a_submitted_list_on_re_render(hass: Any) -> None:
    """
    An error re-render keeps the submitted selection pre-filled.

    ``async_step_init`` re-renders from `options` merged with `user_input`, so
    the exclusion value at that point is the flat list, not the stored map.
    """
    apis = [_FakeAPI(id="mcp-abc", name="My MCP Server")]
    instances = {"mcp-abc": _api_instance([_FakeTool("ping"), _FakeTool("web_search")])}
    get_apis, get_api = _patch_apis(apis, instances)

    with get_apis, get_api:
        schema = await _schema_for_options(
            hass,
            {
                CONF_LLM_HASS_API: ["mcp-abc"],
                CONF_TOOL_EXCLUSIONS: ["mcp-abc::web_search"],
            },
        )

    marker = _schema_key(schema, CONF_TOOL_EXCLUSIONS)
    assert marker.description == {"suggested_value": ["mcp-abc::web_search"]}


@pytest.mark.asyncio
async def test_picker_is_omitted_when_the_api_exposes_no_tools(hass: Any) -> None:
    """A reachable API with an empty tool list contributes no field."""
    apis = [_FakeAPI(id="mcp-abc", name="My MCP Server")]
    instances = {"mcp-abc": _api_instance([])}
    get_apis, get_api = _patch_apis(apis, instances)

    with get_apis, get_api:
        schema = await _schema_for_options(hass, {CONF_LLM_HASS_API: ["mcp-abc"]})

    assert CONF_TOOL_EXCLUSIONS not in [str(cast("Any", key).schema) for key in schema]


@pytest.mark.asyncio
async def test_picker_is_a_multi_select_without_custom_values(hass: Any) -> None:
    """
    The selector must be multiple, sorted as built, and closed to free text.

    ``multiple=True`` is what makes the submitted value the flat list
    ``_parse_tool_exclusions`` groups; a scalar would be discarded.
    """
    apis = [_FakeAPI(id="mcp-abc", name="My MCP Server")]
    instances = {"mcp-abc": _api_instance([_FakeTool("ping")])}
    get_apis, get_api = _patch_apis(apis, instances)

    with get_apis, get_api:
        schema = await _schema_for_options(hass, {CONF_LLM_HASS_API: ["mcp-abc"]})

    marker = _schema_key(schema, CONF_TOOL_EXCLUSIONS)
    config = cast("Any", schema[marker]).config
    assert config["multiple"] is True
    assert config["custom_value"] is False
    assert config["sort"] is False
    assert config["mode"] == "dropdown"
    assert marker.default() == []


def test_parse_tool_exclusions_drops_a_degenerate_stored_map() -> None:
    """A stored map whose names all normalize away is removed, not kept as {}."""
    flow = HomeGenerativeAgentOptionsFlow()
    options: dict[str, Any] = {CONF_TOOL_EXCLUSIONS: {"mcp-abc": ["", None]}}

    flow._parse_tool_exclusions(options)

    assert CONF_TOOL_EXCLUSIONS not in options


@pytest.mark.asyncio
async def test_options_flow_submit_stores_the_grouped_map(hass: Any) -> None:
    """
    A submitted picker selection lands in storage as ``{api_id: [name]}``.

    Drives the real save path, so the ordering against `_drop_empty_fields`
    and `_parse_motion_camera_map` is exercised too.
    """
    flow = _options_flow(hass, {CONF_LLM_HASS_API: ["mcp-abc"]})

    result = cast(
        "dict[str, Any]",
        await flow.async_step_init(
            {
                CONF_LLM_HASS_API: ["mcp-abc"],
                CONF_TOOL_EXCLUSIONS: ["mcp-abc::web_search", "mcp-abc::ping"],
            }
        ),
    )

    assert result["type"] == FlowResultType.CREATE_ENTRY
    assert result["data"][CONF_TOOL_EXCLUSIONS] == {"mcp-abc": ["ping", "web_search"]}


@pytest.mark.asyncio
async def test_options_flow_submit_clears_a_stored_exclusion(hass: Any) -> None:
    """Deselecting everything removes the key, restoring the default."""
    flow = _options_flow(
        hass,
        {
            CONF_LLM_HASS_API: ["mcp-abc"],
            CONF_TOOL_EXCLUSIONS: {"mcp-abc": ["web_search"]},
        },
    )

    result = cast(
        "dict[str, Any]",
        await flow.async_step_init(
            {CONF_LLM_HASS_API: ["mcp-abc"], CONF_TOOL_EXCLUSIONS: []}
        ),
    )

    assert result["type"] == FlowResultType.CREATE_ENTRY
    assert CONF_TOOL_EXCLUSIONS not in result["data"]


@pytest.mark.asyncio
async def test_picker_labels_an_unregistered_but_loadable_api_by_id(hass: Any) -> None:
    """A live API missing from the registry listing falls back to its raw id."""
    instances = {"mcp-abc": _api_instance([_FakeTool("ping")])}
    get_apis, get_api = _patch_apis([], instances)

    with get_apis, get_api:
        schema = await _schema_for_options(hass, {CONF_LLM_HASS_API: ["mcp-abc"]})

    assert _selector_options(schema) == [
        {"label": "mcp-abc: ping", "value": "mcp-abc::ping"}
    ]


@pytest.mark.asyncio
async def test_options_flow_submit_keeps_an_unrendered_exclusion(hass: Any) -> None:
    """
    A save that never rendered the picker leaves the stored map intact.

    The field is absent from `user_input` whenever nothing was enumerable, so
    the stored dict must survive every later cleanup step untouched.
    """
    flow = _options_flow(
        hass,
        {
            CONF_LLM_HASS_API: ["mcp-abc"],
            CONF_TOOL_EXCLUSIONS: {"mcp-abc": ["web_search"]},
        },
    )

    result = cast(
        "dict[str, Any]",
        await flow.async_step_init({CONF_LLM_HASS_API: ["mcp-abc"]}),
    )

    assert result["type"] == FlowResultType.CREATE_ENTRY
    assert result["data"][CONF_TOOL_EXCLUSIONS] == {"mcp-abc": ["web_search"]}


# --------------------------------------------------------------------------
# Untrusted-text rendering (pre-landing review finding)
# --------------------------------------------------------------------------


def test_sanitize_tool_text_neutralizes_spoofing_characters() -> None:
    """
    Bidi overrides, control chars and zero-width characters become '?'.

    These are the classes that let a crafted MCP tool name lie about what it
    says — U+202E reverses the text after it, so a name could forge a trailing
    "(not currently available)" onto a neighbouring picker entry.
    """
    assert sanitize_tool_text("web\u202esearch") == "web?search"
    assert sanitize_tool_text("web\x07search") == "web?search"
    assert sanitize_tool_text("web\u200bsearch") == "web?search"
    assert sanitize_tool_text("web\nsearch") == "web?search"
    # Ordinary names, spaces included, are left exactly as they are.
    assert sanitize_tool_text("get current time") == "get current time"


def test_sanitize_tool_text_caps_length() -> None:
    """One hostile name cannot flood a log line or a form label."""
    assert len(sanitize_tool_text("a" * 500)) == TOOL_TEXT_MAX_LEN


@pytest.mark.asyncio
async def test_picker_label_is_sanitized_but_value_is_not(hass: Any) -> None:
    """
    A hostile tool name renders defanged while its value stays exact.

    The value must remain byte-identical to the real tool name — it is what
    `filter_excluded_tools` matches at runtime, so sanitizing it would store an
    exclusion that silently never applies.
    """
    hostile = "ping\u202egnip"
    apis = [_FakeAPI(id="mcp-abc", name="My MCP Server")]
    instances = {"mcp-abc": _api_instance([_FakeTool(hostile)])}
    get_apis, get_api = _patch_apis(apis, instances)

    with get_apis, get_api:
        schema = await _schema_for_options(hass, {CONF_LLM_HASS_API: ["mcp-abc"]})

    assert _selector_options(schema) == [
        {
            "label": "My MCP Server: ping?gnip",
            "value": f"mcp-abc::{hostile}",
        }
    ]


@pytest.mark.asyncio
async def test_unavailable_label_strips_bidi_overrides(hass: Any) -> None:
    """A stored hostile name has its bidi override defanged before display."""
    get_apis, get_api = _patch_apis([], {})

    with get_apis, get_api:
        schema = await _schema_for_options(
            hass, {CONF_TOOL_EXCLUSIONS: {"mcp-gone": ["ping\u202e"]}}
        )

    assert _selector_options(schema) == [
        {
            "label": "mcp-gone: ping? (not currently available)",
            "value": "mcp-gone::ping\u202e",
        }
    ]


# --------------------------------------------------------------------------
# Failure isolation and label honesty (pre-landing specialist findings)
# --------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_picker_survives_a_malformed_tool_descriptor(hass: Any) -> None:
    """
    A non-conforming tool name must not take down the whole Options form.

    Tool descriptors are remote data, so `.name` is not guaranteed to be a
    string. Sorting a mixed list raises TypeError, and the render loop sits
    outside the enumeration try — unguarded, one bad descriptor from one server
    makes the entire options form fail to open (issue #568's class), not merely
    the picker.
    """
    apis = [
        _FakeAPI(id="mcp-bad", name="Bad Server"),
        _FakeAPI(id="mcp-ok", name="Good Server"),
    ]
    instances = {
        "mcp-bad": _api_instance(
            [_FakeTool("fine"), _FakeTool(cast("Any", None))], api_id="mcp-bad"
        ),
        "mcp-ok": _api_instance([_FakeTool("ping")], api_id="mcp-ok"),
    }
    get_apis, get_api = _patch_apis(apis, instances)

    with get_apis, get_api:
        schema = await _schema_for_options(
            hass, {CONF_LLM_HASS_API: ["mcp-bad", "mcp-ok"]}
        )

    options = _selector_options(schema)
    # The healthy server renders, and so does the healthy tool of the bad one:
    # one unusable descriptor costs its own entry, not its server's.
    assert {"label": "Good Server: ping", "value": "mcp-ok::ping"} in options
    assert {"label": "Bad Server: fine", "value": "mcp-bad::fine"} in options
    assert len(options) == 2


@pytest.mark.asyncio
async def test_malformed_tool_descriptor_is_logged(hass: Any, caplog: Any) -> None:
    """Dropping a tool silently would look identical to the server not having it."""
    apis = [_FakeAPI(id="mcp-bad", name="Bad Server")]
    instances = {
        "mcp-bad": _api_instance([_FakeTool(cast("Any", None))], api_id="mcp-bad")
    }
    get_apis, get_api = _patch_apis(apis, instances)

    with caplog.at_level(WARNING, logger=_CONFIG_FLOW), get_apis, get_api:
        await _schema_for_options(hass, {CONF_LLM_HASS_API: ["mcp-bad"]})

    warnings = [
        record
        for record in caplog.records
        if record.levelno >= WARNING and record.name == _CONFIG_FLOW
    ]
    assert len(warnings) == 1
    assert "mcp-bad" in warnings[0].getMessage()


@pytest.mark.asyncio
async def test_picker_warns_when_an_api_cannot_be_enumerated(
    hass: Any, caplog: Any
) -> None:
    """
    An unreachable server must leave an operator-visible trace.

    The picker still renders a plausible-looking list, so without the warning an
    outage is indistinguishable from "that server genuinely has no tools".
    """
    apis = [_FakeAPI(id="mcp-abc", name="My MCP Server")]
    instances = {"mcp-abc": TimeoutError("server unreachable")}
    get_apis, get_api = _patch_apis(apis, instances)

    with caplog.at_level(WARNING, logger=_CONFIG_FLOW), get_apis, get_api:
        await _schema_for_options(
            hass,
            {
                CONF_LLM_HASS_API: ["mcp-abc"],
                CONF_TOOL_EXCLUSIONS: {"mcp-abc": ["web_search"]},
            },
        )

    warnings = [
        record
        for record in caplog.records
        if record.levelno >= WARNING and record.name == _CONFIG_FLOW
    ]
    assert len(warnings) == 1
    assert "mcp-abc" in warnings[0].getMessage()
    assert "server unreachable" in warnings[0].getMessage()


@pytest.mark.asyncio
async def test_picker_labels_a_deselected_but_live_api(hass: Any) -> None:
    """
    A deselected API is not the same fact as an unavailable one.

    The user undoes the first in this very form; the second means go look at
    the server. Labelling both the same sends them hunting an outage that never
    happened.
    """
    apis = [_FakeAPI(id="mcp-abc", name="My MCP Server")]
    instances = {"mcp-abc": _api_instance([_FakeTool("web_search")])}
    get_apis, get_api = _patch_apis(apis, instances)

    with get_apis, get_api:
        schema = await _schema_for_options(
            hass,
            {
                CONF_LLM_HASS_API: [],
                CONF_TOOL_EXCLUSIONS: {"mcp-abc": ["web_search"]},
            },
        )

    assert _selector_options(schema) == [
        {
            "label": f"My MCP Server: web_search{_SUFFIX_NOT_SELECTED}",
            "value": "mcp-abc::web_search",
        }
    ]


@pytest.mark.asyncio
async def test_selected_but_unreachable_api_is_not_called_deselected(
    hass: Any,
) -> None:
    """
    Registration alone must not decide the label.

    A registered API that is still selected but failed to enumerate is
    unreachable, not deselected — keying on registration alone would send the
    user to the wrong place.
    """
    apis = [_FakeAPI(id="mcp-abc", name="My MCP Server")]
    instances = {"mcp-abc": TimeoutError("down")}
    get_apis, get_api = _patch_apis(apis, instances)

    with get_apis, get_api:
        schema = await _schema_for_options(
            hass,
            {
                CONF_LLM_HASS_API: ["mcp-abc"],
                CONF_TOOL_EXCLUSIONS: {"mcp-abc": ["web_search"]},
            },
        )

    assert _selector_options(schema) == [
        {
            "label": f"My MCP Server: web_search{_SUFFIX_NOT_AVAILABLE}",
            "value": "mcp-abc::web_search",
        }
    ]


# --------------------------------------------------------------------------
# Sanitizer boundaries
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("length", "expected"),
    [
        (TOOL_TEXT_MAX_LEN - 1, TOOL_TEXT_MAX_LEN - 1),
        (TOOL_TEXT_MAX_LEN, TOOL_TEXT_MAX_LEN),
        (TOOL_TEXT_MAX_LEN + 1, TOOL_TEXT_MAX_LEN),
    ],
)
def test_sanitize_tool_text_length_boundaries(length: int, expected: int) -> None:
    """The cap truncates only past the limit, never at or below it."""
    assert len(sanitize_tool_text("a" * length)) == expected


def test_sanitize_tool_text_honours_a_custom_limit() -> None:
    """The `limit` parameter is part of the contract, not decoration."""
    assert sanitize_tool_text("abcdef", limit=3) == "abc"


def test_sanitize_tool_text_caps_before_scanning() -> None:
    """
    Capping bounds the work, not just the result.

    The transform is one character in, one character out, so slicing first is
    byte-identical — but scanning first would run the per-character pass across
    the whole of whatever length a remote server chose.
    """
    hostile = "\u202e" * 100_000
    assert sanitize_tool_text(hostile) == "?" * TOOL_TEXT_MAX_LEN
    # Equivalence with the cap-afterwards order, on mixed input.
    mixed = ("ab\u202ecd\x07" * 60)[:400]
    naive = "".join(c if c.isprintable() else "?" for c in mixed)[:TOOL_TEXT_MAX_LEN]
    assert sanitize_tool_text(mixed) == naive


@pytest.mark.asyncio
async def test_truncated_labels_may_collide_and_that_is_accepted(hass: Any) -> None:
    """
    Two names sharing their first 120 characters render the same label.

    Documented rather than fixed: the values stay distinct, so the exclusions
    themselves remain correct and independently selectable — only the visible
    text collides, and only for names longer than any real MCP tool name.
    """
    shared_prefix = "x" * TOOL_TEXT_MAX_LEN
    apis = [_FakeAPI(id="mcp-abc", name="S")]
    instances = {
        "mcp-abc": _api_instance(
            [_FakeTool(shared_prefix + "_one"), _FakeTool(shared_prefix + "_two")]
        )
    }
    get_apis, get_api = _patch_apis(apis, instances)

    with get_apis, get_api:
        schema = await _schema_for_options(hass, {CONF_LLM_HASS_API: ["mcp-abc"]})

    options = _selector_options(schema)
    assert len({option["label"] for option in options}) == 1
    # The part that actually matters is still unambiguous.
    assert len({option["value"] for option in options}) == 2


# --------------------------------------------------------------------------
# Red-team findings
# --------------------------------------------------------------------------


def test_label_text_disarms_the_marker_vocabulary() -> None:
    """
    A printable name cannot reproduce the trusted suffix.

    `sanitize_tool_text` alone cannot stop this: every character of
    `list_files (not currently available)` is printable, so it survives that
    pass untouched and would render byte-identical to a genuinely-missing
    tool's label. The label path rewrites parentheses so the marker stays the
    only parenthesised run a label can contain.
    """
    forged = "list_files (not currently available)"
    assert sanitize_tool_text(forged) == forged  # the gap _label_text closes
    assert _label_text(forged) == "list_files [not currently available]"
    assert _SUFFIX_NOT_AVAILABLE not in _label_text(forged)
    assert _SUFFIX_NOT_SELECTED not in _label_text("x (API not selected)")


@pytest.mark.asyncio
async def test_live_tool_cannot_impersonate_an_unavailable_one(hass: Any) -> None:
    """
    A hostile name must not make a live tool look switched off.

    This is the deception that matters for a security control: an operator
    scanning the picker to disable a dangerous tool sees it already marked
    inactive and moves on, while it is live and bound.
    """
    forged = "list_files (not currently available)"
    apis = [_FakeAPI(id="mcp-abc", name="MCP")]
    instances = {"mcp-abc": _api_instance([_FakeTool(forged)])}
    get_apis, get_api = _patch_apis(apis, instances)

    with get_apis, get_api:
        schema = await _schema_for_options(hass, {CONF_LLM_HASS_API: ["mcp-abc"]})

    (option,) = _selector_options(schema)
    assert not option["label"].endswith(_SUFFIX_NOT_AVAILABLE)
    assert option["label"] == "MCP: list_files [not currently available]"
    # The value is still the real name, so the exclusion it encodes matches.
    assert option["value"] == f"mcp-abc::{forged}"


@pytest.mark.asyncio
async def test_device_gated_tools_come_from_the_index(hass: Any) -> None:
    """
    Tools an options form cannot enumerate are offered from the tool index.

    HA gates the timer intents on `llm_context.device_id is not None`, and an
    options form has no device — so without the index union these are bound on
    every voice-satellite turn while being impossible to switch off.
    """
    apis = [_FakeAPI(id="assist", name="Assist")]
    instances = {"assist": _api_instance([_FakeTool("HassTurnOn")], api_id="assist")}
    get_apis, get_api = _patch_apis(apis, instances)

    with get_apis, get_api:
        schema = await _schema_for_options(
            hass,
            {CONF_LLM_HASS_API: ["assist"]},
            {"assist::HassStartTimer", "assist::HassTurnOn"},
        )

    values = [option["value"] for option in _selector_options(schema)]
    # The device-gated tool is offered even though enumeration never saw it...
    assert "assist::HassStartTimer" in values
    # ...and the one enumeration DID see is not duplicated by the union.
    assert values.count("assist::HassTurnOn") == 1


@pytest.mark.asyncio
async def test_indexed_keys_of_unselected_apis_are_ignored(hass: Any) -> None:
    """The index is cumulative across APIs; only selected ones may be offered."""
    apis = [_FakeAPI(id="assist", name="Assist")]
    instances = {"assist": _api_instance([_FakeTool("HassTurnOn")], api_id="assist")}
    get_apis, get_api = _patch_apis(apis, instances)

    with get_apis, get_api:
        schema = await _schema_for_options(
            hass,
            {CONF_LLM_HASS_API: ["assist"]},
            {"mcp-other::ping", "assist::HassStartTimer"},
        )

    values = [option["value"] for option in _selector_options(schema)]
    assert "mcp-other::ping" not in values
    assert "assist::HassStartTimer" in values


def test_separator_bearing_api_id_is_still_enforced() -> None:
    """
    An unrepresentable api id keeps being ENFORCED even though the form refuses it.

    The refusal belongs at the form boundary, not in the shared normalizer.
    Dropping it here would stop the runtime honouring an exclusion that is
    already stored -- the tools would silently go live again -- and the next
    unrelated save would then erase it. Fail-open on a security control.
    """
    assert normalize_tool_exclusions({"a::b": ["t"], "assist": ["HassTurnOn"]}) == {
        "a::b": ["t"],
        "assist": ["HassTurnOn"],
    }
    assert tool_exclusions({CONF_TOOL_EXCLUSIONS: {"a::b": ["t"]}}) == {"a::b": {"t"}}


def test_api_id_is_form_representable() -> None:
    """Only the form cares whether an id survives the picker round trip."""
    assert api_id_is_form_representable("assist")
    assert api_id_is_form_representable("mcp-01JABCDEF")
    assert not api_id_is_form_representable("vendor::gateway")


@pytest.mark.asyncio
async def test_picker_does_not_offer_an_unrepresentable_api(hass: Any) -> None:
    """
    The form must never mint an exclusion it cannot store faithfully.

    "vendor::gateway" + "delete_everything" flattens to
    "vendor::gateway::delete_everything", which regroups onto a DIFFERENT api id
    ("vendor") whose own id contains no separator -- so it stores cleanly, the
    form reports success, and the exclusion matches nothing at runtime.
    """
    apis = [_FakeAPI(id="vendor::gateway", name="Gateway")]
    instances = {
        "vendor::gateway": _api_instance(
            [_FakeTool("delete_everything")], api_id="vendor::gateway"
        )
    }
    get_apis, get_api = _patch_apis(apis, instances)

    with get_apis, get_api:
        schema = await _schema_for_options(
            hass, {CONF_LLM_HASS_API: ["vendor::gateway"]}
        )

    keys = [str(cast("Any", key).schema) for key in schema]
    assert CONF_TOOL_EXCLUSIONS not in keys


def test_filter_excluded_tools_survives_tools_being_none() -> None:
    """
    An API whose tool list is None must not kill the conversation turn.

    HA builds an MCP APIInstance with `tools=self.coordinator.data`, which is
    None until the coordinator's first successful refresh. This runs outside
    the caller's `except HomeAssistantError`, so an unguarded iteration fails
    every turn until the server comes back.
    """
    api = _api_instance(cast("Any", None))

    filtered, dropped = filter_excluded_tools("mcp-abc", api, {"mcp-abc": {"x"}})

    assert filtered is api
    assert dropped == []


def test_exclusion_round_trip_is_lossless_for_representable_maps() -> None:
    """Whatever survives normalization must survive the form round trip."""
    original = {
        "assist": ["HassTurnOn", "HassTurnOff"],
        "mcp-abc": ["odd::name", "web_search"],
    }
    normalized = normalize_tool_exclusions(original)
    assert _list_as_tool_exclusions(_tool_exclusions_as_list(normalized)) == normalized


def test_filter_excluded_tools_survives_a_malformed_descriptor() -> None:
    """
    An unhashable tool name must not break every conversation turn.

    This path runs per turn outside the caller's `except HomeAssistantError`,
    so an unguarded membership test would fail the whole pipeline for as long
    as the server advertises the bad descriptor — strictly worse than the
    options-form variant of the same bug.
    """
    bad = cast("Any", SimpleNamespace(name=["not", "a", "str"]))
    good = cast("Any", SimpleNamespace(name="ping"))
    keep = cast("Any", SimpleNamespace(name="stay"))
    api = _api_instance([bad, good, keep])

    filtered, dropped = filter_excluded_tools("mcp-abc", api, {"mcp-abc": {"ping"}})

    assert [tool.name for tool in filtered.tools] == [bad.name, "stay"]
    assert dropped == ["ping"]


@pytest.mark.asyncio
async def test_unrepresentable_exclusion_survives_an_unrelated_save(hass: Any) -> None:
    """
    An exclusion the picker cannot represent must not die on an unrelated save.

    It is deliberately withheld from the picker (offering it would mint a value
    that mis-splits onto a different api id), so the submitted list cannot
    carry it. Rebuilding the stored map from that list alone would delete it,
    and the tools it was keeping switched off would go live again — the third
    incarnation of this same form-versus-storage authority bug.
    """
    stored = {
        "vendor::gateway": ["delete_everything"],
        "assist": ["HassTurnOn"],
    }
    flow = _options_flow(
        hass, {CONF_LLM_HASS_API: ["assist"], CONF_TOOL_EXCLUSIONS: stored}
    )

    # It is never offered to the user...
    assert _tool_exclusions_as_list(stored) == ["assist::HassTurnOn"]

    # ...and it survives a save that touches something else entirely.
    result = cast(
        "dict[str, Any]",
        await flow.async_step_init(
            {
                CONF_LLM_HASS_API: ["assist"],
                CONF_TOOL_EXCLUSIONS: ["assist::HassTurnOn"],
                CONF_MAX_MESSAGES_IN_CONTEXT: 100,
            }
        ),
    )

    assert result["type"] == FlowResultType.CREATE_ENTRY
    assert result["data"][CONF_TOOL_EXCLUSIONS] == {
        "vendor::gateway": ["delete_everything"],
        "assist": ["HassTurnOn"],
    }

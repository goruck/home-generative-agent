# ruff: noqa: S101
"""
Tests for the always-included tools feature (issue #579).

Covers the option's normalization helpers, the options-form picker (which
shares its live enumeration with the excluded-tools picker), the submit-path
storage round trip including the exclusion/inclusion contradiction guard, and
the runtime force-bind step in ``_retrieve_tools``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, cast
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import voluptuous as vol
from homeassistant.const import CONF_LLM_HASS_API
from homeassistant.data_entry_flow import FlowResultType
from homeassistant.helpers import llm
from pytest_homeassistant_custom_component.common import MockConfigEntry

from custom_components.home_generative_agent.agent.graph import _retrieve_tools
from custom_components.home_generative_agent.agent.helpers import (
    normalize_tool_inclusions,
    tool_inclusions,
)
from custom_components.home_generative_agent.config_flow import (
    _SUFFIX_NOT_AVAILABLE,
    HomeGenerativeAgentOptionsFlow,
    _schema_for_options,
)
from custom_components.home_generative_agent.const import (
    CONF_TOOL_EXCLUSIONS,
    CONF_TOOL_INCLUSIONS,
    DOMAIN,
    TOOL_INCLUSIONS_MAX_PER_TURN,
)

if TYPE_CHECKING:
    from langchain_core.runnables import RunnableConfig

_CONFIG_FLOW = "custom_components.home_generative_agent.config_flow"


# --------------------------------------------------------------------------
# Normalization and runtime accessor
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "raw",
    [None, "", [], "web_search", 0, {"": ["a"]}, {5: ["a"]}, {"api": None}],
)
def test_normalize_tool_inclusions_rejects_degenerate_shapes(raw: Any) -> None:
    """The inclusions normalizer shares the exclusions' degenerate-shape contract."""
    assert normalize_tool_inclusions(raw) == {}


def test_normalize_tool_inclusions_cleans_and_dedupes() -> None:
    """Non-string and empty names are dropped; the rest are deduped and sorted."""
    assert normalize_tool_inclusions(
        {
            "mcp-abc": ["b_tool", "a_tool", "a_tool", "", None, 7],
            "mcp-empty": [],
            "assist": "HassTurnOn",
        }
    ) == {"mcp-abc": ["a_tool", "b_tool"], "assist": ["HassTurnOn"]}


def test_tool_inclusions_returns_sorted_pairs() -> None:
    """The runtime accessor flattens the map to a stable, sorted pair list."""
    assert tool_inclusions(
        {CONF_TOOL_INCLUSIONS: {"mcp-b": ["z", "a"], "mcp-a": ["m"]}}
    ) == [("mcp-a", "m"), ("mcp-b", "a"), ("mcp-b", "z")]
    assert tool_inclusions({}) == []


# --------------------------------------------------------------------------
# Options form: picker construction (shared enumeration with exclusions)
# --------------------------------------------------------------------------


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
    return llm.APIInstance(
        api=cast("Any", _FakeAPI(id=api_id, name="My MCP Server")),
        api_prompt="prompt",
        llm_context=cast("Any", None),
        tools=tools,
    )


def _patch_apis(apis: list[_FakeAPI], instances: dict[str, Any]) -> Any:
    async def _get_api(_hass: Any, api_id: str, _ctx: Any) -> Any:
        result = instances[api_id]
        if isinstance(result, Exception):
            raise result
        return result

    return (
        patch(f"{_CONFIG_FLOW}.llm.async_get_apis", return_value=apis),
        patch(f"{_CONFIG_FLOW}.llm.async_get_api", side_effect=_get_api),
    )


def _schema_key(schema: dict[Any, Any], key_name: str) -> Any:
    return next(key for key in schema if cast("Any", key).schema == key_name)


def _picker_options(schema: dict[Any, Any], key_name: str) -> list[dict[str, str]]:
    selector = schema[_schema_key(schema, key_name)]
    return cast("Any", selector).config["options"]


@pytest.mark.asyncio
async def test_inclusions_picker_renders_after_the_exclusions_picker(
    hass: Any,
) -> None:
    """The two tool pickers sit together, exclusions first."""
    apis = [_FakeAPI(id="mcp-abc", name="My MCP Server")]
    instances = {"mcp-abc": _api_instance([_FakeTool("ping")])}
    get_apis, get_api = _patch_apis(apis, instances)

    with get_apis, get_api:
        schema = await _schema_for_options(hass, {CONF_LLM_HASS_API: ["mcp-abc"]})

    keys = [str(cast("Any", key).schema) for key in schema]
    assert keys[keys.index(CONF_TOOL_EXCLUSIONS) + 1] == CONF_TOOL_INCLUSIONS


@pytest.mark.asyncio
async def test_both_pickers_offer_the_same_live_tools(hass: Any) -> None:
    """Live enumeration is shared: both pickers list the same tool universe."""
    apis = [_FakeAPI(id="mcp-abc", name="My MCP Server")]
    instances = {"mcp-abc": _api_instance([_FakeTool("web_search"), _FakeTool("ping")])}
    get_apis, get_api = _patch_apis(apis, instances)

    with get_apis, get_api:
        schema = await _schema_for_options(hass, {CONF_LLM_HASS_API: ["mcp-abc"]})

    expected = [
        {"label": "My MCP Server: ping", "value": "mcp-abc::ping"},
        {"label": "My MCP Server: web_search", "value": "mcp-abc::web_search"},
    ]
    assert _picker_options(schema, CONF_TOOL_EXCLUSIONS) == expected
    assert _picker_options(schema, CONF_TOOL_INCLUSIONS) == expected


@pytest.mark.asyncio
async def test_stale_stored_selections_stay_per_picker(hass: Any) -> None:
    """
    Each picker re-adds only its own stored values when enumeration fails.

    A stale inclusion must not leak a labelled option into the exclusions
    picker (and vice versa), but each field's own pre-filled value must stay
    selectable or the form becomes unsaveable (issue #568's class).
    """
    apis = [_FakeAPI(id="mcp-abc", name="My MCP Server")]
    instances = {"mcp-abc": TimeoutError("server unreachable")}
    get_apis, get_api = _patch_apis(apis, instances)

    with get_apis, get_api:
        schema = await _schema_for_options(
            hass,
            {
                CONF_LLM_HASS_API: ["mcp-abc"],
                CONF_TOOL_EXCLUSIONS: {"mcp-abc": ["dangerous_tool"]},
                CONF_TOOL_INCLUSIONS: {"mcp-abc": ["web_search"]},
            },
        )

    assert _picker_options(schema, CONF_TOOL_EXCLUSIONS) == [
        {
            "label": f"My MCP Server: dangerous_tool{_SUFFIX_NOT_AVAILABLE}",
            "value": "mcp-abc::dangerous_tool",
        }
    ]
    assert _picker_options(schema, CONF_TOOL_INCLUSIONS) == [
        {
            "label": f"My MCP Server: web_search{_SUFFIX_NOT_AVAILABLE}",
            "value": "mcp-abc::web_search",
        }
    ]


@pytest.mark.asyncio
async def test_a_value_stale_in_both_lists_is_re_added_to_both(hass: Any) -> None:
    """The shared seen-set must not let the first picker swallow the second's."""
    get_apis, get_api = _patch_apis([], {})

    with get_apis, get_api:
        schema = await _schema_for_options(
            hass,
            {
                CONF_TOOL_EXCLUSIONS: {"mcp-gone": ["web_search"]},
                CONF_TOOL_INCLUSIONS: {"mcp-gone": ["web_search"]},
            },
        )

    stale = [
        {
            "label": f"mcp-gone: web_search{_SUFFIX_NOT_AVAILABLE}",
            "value": "mcp-gone::web_search",
        }
    ]
    assert _picker_options(schema, CONF_TOOL_EXCLUSIONS) == stale
    assert _picker_options(schema, CONF_TOOL_INCLUSIONS) == stale


@pytest.mark.asyncio
async def test_inclusions_picker_is_omitted_when_there_is_nothing_to_list(
    hass: Any,
) -> None:
    """No listable tools means no field, so no empty-overwrite on save."""
    get_apis, get_api = _patch_apis([], {})

    with get_apis, get_api:
        schema = await _schema_for_options(hass, {})

    assert CONF_TOOL_INCLUSIONS not in [str(cast("Any", key).schema) for key in schema]


# --------------------------------------------------------------------------
# Options form: storage round trip
# --------------------------------------------------------------------------


def _options_flow(hass: Any, options: dict[str, Any]) -> Any:
    entry = MockConfigEntry(
        domain=DOMAIN, title="Home Generative Agent", options=options
    )
    entry.add_to_hass(hass)
    flow = HomeGenerativeAgentOptionsFlow()
    flow.hass = hass
    flow.handler = entry.entry_id
    return flow


@pytest.mark.asyncio
async def test_parse_tool_inclusions_stores_the_grouped_map(hass: Any) -> None:
    """A submitted selection is normalized into the stored shape."""
    flow = _options_flow(hass, {})
    options: dict[str, Any] = {CONF_TOOL_INCLUSIONS: ["mcp-abc::b", "mcp-abc::a"]}

    flow._parse_tool_inclusions(options)

    assert options == {CONF_TOOL_INCLUSIONS: {"mcp-abc": ["a", "b"]}}


@pytest.mark.asyncio
async def test_parse_tool_inclusions_drops_an_empty_selection(hass: Any) -> None:
    """'Include nothing extra' is stored as an absent key, never as {}."""
    flow = _options_flow(hass, {})
    options: dict[str, Any] = {CONF_TOOL_INCLUSIONS: []}

    flow._parse_tool_inclusions(options)

    assert CONF_TOOL_INCLUSIONS not in options


def test_parse_tool_inclusions_keeps_an_untouched_stored_map() -> None:
    """When the picker was not rendered the stored map survives the save."""
    flow = HomeGenerativeAgentOptionsFlow()
    options: dict[str, Any] = {CONF_TOOL_INCLUSIONS: {"mcp-abc": ["web_search"]}}

    flow._parse_tool_inclusions(options)

    assert options == {CONF_TOOL_INCLUSIONS: {"mcp-abc": ["web_search"]}}


@pytest.mark.asyncio
async def test_options_flow_submit_stores_the_inclusions_map(hass: Any) -> None:
    """A submitted inclusions selection lands in storage as {api_id: [name]}."""
    flow = _options_flow(hass, {CONF_LLM_HASS_API: ["mcp-abc"]})

    result = cast(
        "dict[str, Any]",
        await flow.async_step_init(
            {
                CONF_LLM_HASS_API: ["mcp-abc"],
                CONF_TOOL_INCLUSIONS: ["mcp-abc::web_search"],
            }
        ),
    )

    assert result["type"] == FlowResultType.CREATE_ENTRY
    assert result["data"][CONF_TOOL_INCLUSIONS] == {"mcp-abc": ["web_search"]}


@pytest.mark.asyncio
async def test_options_flow_rejects_a_tool_in_both_lists(hass: Any) -> None:
    """Excluding and always-including the same tool is a contradiction."""
    flow = _options_flow(hass, {CONF_LLM_HASS_API: ["mcp-abc"]})

    result = cast(
        "dict[str, Any]",
        await flow.async_step_init(
            {
                CONF_LLM_HASS_API: ["mcp-abc"],
                CONF_TOOL_EXCLUSIONS: ["mcp-abc::web_search", "mcp-abc::ping"],
                CONF_TOOL_INCLUSIONS: ["mcp-abc::web_search"],
            }
        ),
    )

    assert result["type"] == FlowResultType.FORM
    assert result["errors"] == {"base": "tool_excluded_and_included"}
    # The error re-render must keep the submitted values selectable and
    # pre-filled, or the form lands in issue #568's unsaveable class.
    schema = result["data_schema"].schema
    for key_name, submitted in [
        (CONF_TOOL_EXCLUSIONS, ["mcp-abc::web_search", "mcp-abc::ping"]),
        (CONF_TOOL_INCLUSIONS, ["mcp-abc::web_search"]),
    ]:
        marker = _schema_key(schema, key_name)
        assert marker.description == {"suggested_value": submitted}
        offered = {
            opt["value"] for opt in cast("Any", schema[marker]).config["options"]
        }
        assert set(submitted) <= offered


@pytest.mark.asyncio
async def test_options_flow_accepts_disjoint_lists(hass: Any) -> None:
    """Disjoint exclusion/inclusion selections save normally."""
    flow = _options_flow(hass, {CONF_LLM_HASS_API: ["mcp-abc"]})

    result = cast(
        "dict[str, Any]",
        await flow.async_step_init(
            {
                CONF_LLM_HASS_API: ["mcp-abc"],
                CONF_TOOL_EXCLUSIONS: ["mcp-abc::ping"],
                CONF_TOOL_INCLUSIONS: ["mcp-abc::web_search"],
            }
        ),
    )

    assert result["type"] == FlowResultType.CREATE_ENTRY
    assert result["data"][CONF_TOOL_EXCLUSIONS] == {"mcp-abc": ["ping"]}
    assert result["data"][CONF_TOOL_INCLUSIONS] == {"mcp-abc": ["web_search"]}


@pytest.mark.asyncio
async def test_submit_preserves_an_unrepresentable_stored_inclusion(
    hass: Any,
) -> None:
    """
    An inclusion under a ``::``-bearing api id survives an unrelated save.

    The picker can never offer such an id, so the submitted list cannot carry
    it; without the merge-back a save would silently delete it.
    """
    flow = _options_flow(
        hass,
        {
            CONF_LLM_HASS_API: ["mcp-abc"],
            CONF_TOOL_INCLUSIONS: {"mcp::odd": ["web_search"]},
        },
    )

    result = cast(
        "dict[str, Any]",
        await flow.async_step_init(
            {CONF_LLM_HASS_API: ["mcp-abc"], CONF_TOOL_INCLUSIONS: []}
        ),
    )

    assert result["type"] == FlowResultType.CREATE_ENTRY
    assert result["data"][CONF_TOOL_INCLUSIONS] == {"mcp::odd": ["web_search"]}


# --------------------------------------------------------------------------
# Runtime: the always-included step in _retrieve_tools
# --------------------------------------------------------------------------


def _search_item(name: str, *, score: float, api_id: str = "assist") -> MagicMock:
    item = MagicMock()
    item.value = {
        "name": name,
        "api_id": api_id,
        "description": f"{name} description",
        "parameters": "{}",
        "is_actuation": False,
    }
    item.score = score
    return item


def _index_row(name: str, api_id: str) -> MagicMock:
    item = MagicMock()
    item.value = {
        "name": name,
        "api_id": api_id,
        "description": f"{name} description",
        "parameters": "{}",
        "is_actuation": False,
    }
    return item


def _store(
    search_items: list[MagicMock], index_rows: dict[str, MagicMock]
) -> MagicMock:
    """Store mock: asearch feeds RAG, aget serves force-bind lookups by key."""
    store = MagicMock()
    store.asearch = AsyncMock(return_value=search_items)

    async def aget(namespace: Any, key: str = "", **_kwargs: Any) -> Any:  # noqa: ARG001
        return index_rows.get(key)

    store.aget = AsyncMock(side_effect=aget)
    return store


def _make_live_api(tools_by_api: dict[str, list[str]]) -> Any:
    """Build a MultiLLMAPI mock exposing the given tools per api id."""
    apis = {
        api_id: SimpleNamespace(
            tools=[
                SimpleNamespace(
                    name=name,
                    description=f"{name} live",
                    parameters={"type": "object", "properties": {}},
                )
                for name in names
            ],
            custom_serializer=None,
        )
        for api_id, names in tools_by_api.items()
    }
    return MagicMock(apis=apis)


def _state(query: str) -> Any:
    return {
        "messages": [MagicMock(content=query)],
        "summary": "",
        "chat_model_usage_metadata": {},
        "messages_to_remove": [],
        "selected_tools": [],
        "tool_routing_map": {},
    }


def _config(
    *,
    options: dict[str, Any],
    live: dict[str, list[str]] | None,
) -> RunnableConfig:
    configurable: dict[str, Any] = {
        "options": options,
        "tool_index_ready": True,
    }
    if live is not None:
        configurable["ha_llm_api"] = _make_live_api(live)
        configurable["langchain_tools"] = {}
    return cast("Any", {"configurable": configurable})


_WEB_SEARCH_KEY = "mcp-abc::web_search"


@pytest.mark.asyncio
async def test_retrieve_tools_appends_an_included_tool_rag_missed() -> None:
    """The configured tool binds even when vector retrieval never ranks it."""
    store = _store(
        [_search_item("HassBroadcast", score=0.6)],
        {_WEB_SEARCH_KEY: _index_row("web_search", "mcp-abc")},
    )

    result = await _retrieve_tools(
        _state("Who won the FIFA World Cup?"),
        _config(
            options={
                "llm_hass_api": ["assist", "mcp-abc"],
                CONF_TOOL_INCLUSIONS: {"mcp-abc": ["web_search"]},
            },
            live={"assist": ["HassBroadcast"], "mcp-abc": ["web_search"]},
        ),
        store=store,
    )

    assert result["tool_routing_map"].get("web_search") == "mcp-abc"
    # The append must not evict the RAG selection.
    assert "HassBroadcast" in result["tool_routing_map"]


@pytest.mark.asyncio
async def test_included_tool_binds_outside_the_retrieval_limit() -> None:
    """A limit already consumed by RAG selections cannot squeeze the inclusion out."""
    store = _store(
        [_search_item("HassBroadcast", score=0.6)],
        {_WEB_SEARCH_KEY: _index_row("web_search", "mcp-abc")},
    )

    result = await _retrieve_tools(
        _state("Who won the FIFA World Cup?"),
        _config(
            options={
                "llm_hass_api": ["assist", "mcp-abc"],
                "tool_retrieval_limit": 1,
                CONF_TOOL_INCLUSIONS: {"mcp-abc": ["web_search"]},
            },
            live={"assist": ["HassBroadcast"], "mcp-abc": ["web_search"]},
        ),
        store=store,
    )

    assert "HassBroadcast" in result["tool_routing_map"]
    assert "web_search" in result["tool_routing_map"]


@pytest.mark.asyncio
async def test_included_tool_of_an_inactive_api_is_skipped() -> None:
    """An inclusion under an api id that is not active this turn stays unbound."""
    store = _store(
        [_search_item("HassBroadcast", score=0.6)],
        {_WEB_SEARCH_KEY: _index_row("web_search", "mcp-abc")},
    )

    result = await _retrieve_tools(
        _state("Who won the FIFA World Cup?"),
        _config(
            options={
                "llm_hass_api": ["assist"],
                CONF_TOOL_INCLUSIONS: {"mcp-abc": ["web_search"]},
            },
            live={"assist": ["HassBroadcast"]},
        ),
        store=store,
    )

    assert "web_search" not in result["tool_routing_map"]
    lookups = [
        call
        for call in store.aget.call_args_list
        if str(call.kwargs.get("key", "")) == _WEB_SEARCH_KEY
    ]
    assert not lookups


@pytest.mark.asyncio
async def test_exclusion_beats_inclusion_even_with_live_filter_off() -> None:
    """
    A tool in both lists must stay unbound with no live filter wired.

    With the live filter fail-open (no ha_llm_api / langchain_tools in the
    config) the store lookup would happily return the excluded tool's row, so
    only the explicit exclusion check in the always-included step stands
    between a contradictory config and a bound excluded tool.
    """
    store = _store(
        [_search_item("HassBroadcast", score=0.6)],
        {_WEB_SEARCH_KEY: _index_row("web_search", "mcp-abc")},
    )

    result = await _retrieve_tools(
        _state("Who won the FIFA World Cup?"),
        _config(
            options={
                "llm_hass_api": ["assist", "mcp-abc"],
                CONF_TOOL_INCLUSIONS: {"mcp-abc": ["web_search"]},
                CONF_TOOL_EXCLUSIONS: {"mcp-abc": ["web_search"]},
            },
            live=None,
        ),
        store=store,
    )

    assert "web_search" not in result["tool_routing_map"]
    lookups = [
        call
        for call in store.aget.call_args_list
        if str(call.kwargs.get("key", "")) == _WEB_SEARCH_KEY
    ]
    assert not lookups


@pytest.mark.asyncio
async def test_included_tool_not_live_this_turn_is_skipped() -> None:
    """An inclusion whose tool the live filter rejects stays unbound."""
    store = _store(
        [_search_item("HassBroadcast", score=0.6)],
        {_WEB_SEARCH_KEY: _index_row("web_search", "mcp-abc")},
    )

    result = await _retrieve_tools(
        _state("Who won the FIFA World Cup?"),
        _config(
            options={
                "llm_hass_api": ["assist", "mcp-abc"],
                CONF_TOOL_INCLUSIONS: {"mcp-abc": ["web_search"]},
            },
            # mcp-abc is active but does not expose web_search this turn.
            live={"assist": ["HassBroadcast"], "mcp-abc": ["other_tool"]},
        ),
        store=store,
    )

    assert "web_search" not in result["tool_routing_map"]


@pytest.mark.asyncio
async def test_included_tool_already_retrieved_is_not_fetched_again() -> None:
    """A tool RAG already selected is not looked up or bound twice."""
    store = _store(
        [_search_item("web_search", score=0.6, api_id="mcp-abc")],
        {_WEB_SEARCH_KEY: _index_row("web_search", "mcp-abc")},
    )

    result = await _retrieve_tools(
        _state("Who won the FIFA World Cup?"),
        _config(
            options={
                "llm_hass_api": ["assist", "mcp-abc"],
                CONF_TOOL_INCLUSIONS: {"mcp-abc": ["web_search"]},
            },
            live={"assist": [], "mcp-abc": ["web_search"]},
        ),
        store=store,
    )

    assert result["tool_routing_map"].get("web_search") == "mcp-abc"
    names = [t["function"]["name"] for t in result["selected_tools"]]
    assert names.count("web_search") == 1
    lookups = [
        call
        for call in store.aget.call_args_list
        if str(call.kwargs.get("key", "")) == _WEB_SEARCH_KEY
    ]
    assert not lookups


@pytest.mark.asyncio
async def test_same_named_inclusion_from_another_api_warns_and_skips(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """
    A cross-API name collision is logged at warning level, not swallowed.

    _format_and_dedupe_tools routes calls by bare name (first seen wins), so a
    same-named tool from another API would absorb the pinned tool's calls —
    the operator must be able to see that from the logs. The store must not be
    queried for the shadowed inclusion either.
    """
    store = _store(
        [_search_item("web_search", score=0.6, api_id="assist")],
        {_WEB_SEARCH_KEY: _index_row("web_search", "mcp-abc")},
    )

    with caplog.at_level(logging.WARNING):
        result = await _retrieve_tools(
            _state("Who won the FIFA World Cup?"),
            _config(
                options={
                    "llm_hass_api": ["assist", "mcp-abc"],
                    CONF_TOOL_INCLUSIONS: {"mcp-abc": ["web_search"]},
                },
                live={"assist": ["web_search"], "mcp-abc": ["web_search"]},
            ),
            store=store,
        )

    assert result["tool_routing_map"]["web_search"] == "assist"
    assert "shadowed by a same-named tool from api assist" in caplog.text
    lookups = [
        call
        for call in store.aget.call_args_list
        if str(call.kwargs.get("key", "")) == _WEB_SEARCH_KEY
    ]
    assert not lookups


@pytest.mark.asyncio
async def test_same_api_duplicate_is_skipped_quietly(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """RAG already selecting the pinned tool itself is the normal case: no noise."""
    store = _store(
        [_search_item("web_search", score=0.6, api_id="mcp-abc")],
        {_WEB_SEARCH_KEY: _index_row("web_search", "mcp-abc")},
    )

    with caplog.at_level(logging.WARNING):
        result = await _retrieve_tools(
            _state("Who won the FIFA World Cup?"),
            _config(
                options={
                    "llm_hass_api": ["assist", "mcp-abc"],
                    CONF_TOOL_INCLUSIONS: {"mcp-abc": ["web_search"]},
                },
                live={"assist": [], "mcp-abc": ["web_search"]},
            ),
            store=store,
        )

    assert result["tool_routing_map"]["web_search"] == "mcp-abc"
    assert "shadowed" not in caplog.text


@pytest.mark.asyncio
async def test_inclusions_are_capped_per_turn(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """
    A degenerate programmatic map cannot stall the turn.

    Only the first TOOL_INCLUSIONS_MAX_PER_TURN pairs are honoured — each one
    costs a sequential store round-trip per turn — and the truncation is
    announced at warning level.
    """
    names = [f"tool_{i:03d}" for i in range(TOOL_INCLUSIONS_MAX_PER_TURN + 5)]
    store = _store(
        [_search_item("HassBroadcast", score=0.6)],
        {f"mcp-abc::{n}": _index_row(n, "mcp-abc") for n in names},
    )

    with caplog.at_level(logging.WARNING):
        result = await _retrieve_tools(
            _state("Who won the FIFA World Cup?"),
            _config(
                options={
                    "llm_hass_api": ["assist", "mcp-abc"],
                    CONF_TOOL_INCLUSIONS: {"mcp-abc": names},
                },
                live={"assist": ["HassBroadcast"], "mcp-abc": names},
            ),
            store=store,
        )

    bound = [n for n in names if n in result["tool_routing_map"]]
    assert len(bound) == TOOL_INCLUSIONS_MAX_PER_TURN
    # tool_inclusions() sorts, so the honoured prefix is deterministic.
    assert bound == sorted(names)[:TOOL_INCLUSIONS_MAX_PER_TURN]
    assert "Honouring only the first" in caplog.text


@pytest.mark.asyncio
async def test_actuation_inclusion_survives_read_only_open_state_strip() -> None:
    """
    Step 3b drops actuation tools on read-only queries; 3e runs after it.

    "Always included means always": moving the append before the strip (or
    letting the strip see inclusions) would silently unpin every actuation
    tool exactly on state questions.
    """
    row = _index_row("open_gate", "mcp-abc")
    row.value["is_actuation"] = True
    store = _store(
        [_search_item("GetLiveContext", score=0.6)],
        {"mcp-abc::open_gate": row},
    )

    result = await _retrieve_tools(
        _state("which doors are open"),
        _config(
            options={
                "llm_hass_api": ["assist", "mcp-abc"],
                CONF_TOOL_INCLUSIONS: {"mcp-abc": ["open_gate"]},
            },
            live={"assist": ["GetLiveContext"], "mcp-abc": ["open_gate"]},
        ),
        store=store,
    )

    assert result["tool_routing_map"].get("open_gate") == "mcp-abc"


@pytest.mark.asyncio
async def test_included_tool_binds_from_fallback_when_index_misses() -> None:
    """An index outage (row absent) falls back to the live config tools."""
    store = _store([_search_item("HassBroadcast", score=0.6)], {})

    result = await _retrieve_tools(
        _state("Who won the FIFA World Cup?"),
        _config(
            options={
                "llm_hass_api": ["assist", "mcp-abc"],
                CONF_TOOL_INCLUSIONS: {"mcp-abc": ["web_search"]},
            },
            live={"assist": ["HassBroadcast"], "mcp-abc": ["web_search"]},
        ),
        store=store,
    )

    assert result["tool_routing_map"].get("web_search") == "mcp-abc"


@pytest.mark.asyncio
async def test_included_tool_absent_everywhere_is_skipped_without_error() -> None:
    """A tool in neither the index nor the live universe skips cleanly."""
    store = _store([_search_item("HassBroadcast", score=0.6)], {})

    result = await _retrieve_tools(
        _state("Who won the FIFA World Cup?"),
        _config(
            options={
                "llm_hass_api": ["assist", "mcp-abc"],
                CONF_TOOL_INCLUSIONS: {"mcp-abc": ["web_search"]},
            },
            live={"assist": ["HassBroadcast"], "mcp-abc": []},
        ),
        store=store,
    )

    assert "web_search" not in result["tool_routing_map"]
    assert "HassBroadcast" in result["tool_routing_map"]


def test_overlap_guard_ignores_unrendered_pickers() -> None:
    """
    The overlap check applies only when both pickers were actually submitted.

    An unrendered picker reaches the save as the stored dict (or is absent),
    and rejecting such a save would make unrelated options edits impossible
    whenever a stored contradiction exists.
    """
    guard = HomeGenerativeAgentOptionsFlow._tool_picker_overlap_error
    assert (
        guard(
            {
                CONF_TOOL_EXCLUSIONS: {"mcp-abc": ["x"]},
                CONF_TOOL_INCLUSIONS: ["mcp-abc::x"],
            }
        )
        is None
    )
    assert guard({CONF_TOOL_INCLUSIONS: ["mcp-abc::x"]}) is None
    assert guard({CONF_TOOL_EXCLUSIONS: ["mcp-abc::x"]}) is None
    assert guard({}) is None


@pytest.mark.asyncio
async def test_exclusion_beats_inclusion_with_live_filter_wired() -> None:
    """
    The explicit exclusion check must fire even when the tool is still live.

    The mock live api applies no APIInstance stripping, so this proves the
    runtime check stands on its own in the wired configuration too — defense
    in depth for the security rule, not just its fail-open corner.
    """
    store = _store(
        [_search_item("HassBroadcast", score=0.6)],
        {_WEB_SEARCH_KEY: _index_row("web_search", "mcp-abc")},
    )

    result = await _retrieve_tools(
        _state("Who won the FIFA World Cup?"),
        _config(
            options={
                "llm_hass_api": ["assist", "mcp-abc"],
                CONF_TOOL_INCLUSIONS: {"mcp-abc": ["web_search"]},
                CONF_TOOL_EXCLUSIONS: {"mcp-abc": ["web_search"]},
            },
            live={"assist": ["HassBroadcast"], "mcp-abc": ["web_search"]},
        ),
        store=store,
    )

    assert "web_search" not in result["tool_routing_map"]


@pytest.mark.asyncio
async def test_exclusion_on_one_api_does_not_block_inclusion_on_another() -> None:
    """The exclusion check is per-api: same name elsewhere is a different tool."""
    store = _store(
        [_search_item("HassBroadcast", score=0.6)],
        {_WEB_SEARCH_KEY: _index_row("web_search", "mcp-abc")},
    )

    result = await _retrieve_tools(
        _state("Who won the FIFA World Cup?"),
        _config(
            options={
                "llm_hass_api": ["assist", "mcp-abc"],
                CONF_TOOL_INCLUSIONS: {"mcp-abc": ["web_search"]},
                CONF_TOOL_EXCLUSIONS: {"assist": ["web_search"]},
            },
            live={"assist": ["HassBroadcast"], "mcp-abc": ["web_search"]},
        ),
        store=store,
    )

    assert result["tool_routing_map"].get("web_search") == "mcp-abc"


@pytest.mark.asyncio
async def test_corrupt_index_row_for_inclusion_does_not_kill_the_turn() -> None:
    """
    A malformed stored row costs one skipped tool, not the conversation.

    Unguarded, json.loads on the row's parameters crashed _retrieve_tools —
    and for an always-included tool that meant EVERY turn, deterministically,
    until the row was repaired.
    """
    row = _index_row("web_search", "mcp-abc")
    row.value["parameters"] = "{not json"
    store = _store(
        [_search_item("HassBroadcast", score=0.6)],
        {_WEB_SEARCH_KEY: row},
    )

    result = await _retrieve_tools(
        _state("Who won the FIFA World Cup?"),
        _config(
            options={
                "llm_hass_api": ["assist", "mcp-abc"],
                CONF_TOOL_INCLUSIONS: {"mcp-abc": ["web_search"]},
            },
            live={"assist": ["HassBroadcast"], "mcp-abc": ["web_search"]},
        ),
        store=store,
    )

    assert "web_search" not in result["tool_routing_map"]
    assert "HassBroadcast" in result["tool_routing_map"]

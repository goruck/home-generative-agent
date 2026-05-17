# ruff: noqa: S101
"""Unit tests for tool retrieval logic (RAG, safety net, fallbacks)."""

from __future__ import annotations

import json
import re
from typing import TYPE_CHECKING, Any
from unittest.mock import AsyncMock, MagicMock

import psycopg
import pytest
from homeassistant.helpers import llm
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage

from custom_components.home_generative_agent.agent.graph import (
    State,
    _filter_open_state_live_context_content,
    _get_actuation_safety_tools,
    _get_allowed_api_ids,
    _get_rag_retrieved_tools,
    _latest_open_state_query,
    _normalize_live_context_args_for_open_state,
    _query_needs_actuation_safety,
    _retrieve_tools,
    _split_query_intents,
)
from custom_components.home_generative_agent.const import ACTUATION_KEYWORDS_REGEX

if TYPE_CHECKING:
    from langchain_core.runnables import RunnableConfig


@pytest.mark.asyncio
async def test_get_allowed_api_ids_includes_hga_local() -> None:
    """Verify that hga_local is always included in allowed API IDs."""
    config: RunnableConfig = {"configurable": {"options": {"llm_hass_api": ["assist"]}}}
    allowed = _get_allowed_api_ids(config)
    assert "assist" in allowed
    assert "hga_local" in allowed

    # Empty config
    config_empty: RunnableConfig = {"configurable": {"options": {}}}
    allowed_empty = _get_allowed_api_ids(config_empty)
    # Default is ["assist"] if missing
    assert "assist" in allowed_empty
    assert "hga_local" in allowed_empty


@pytest.mark.asyncio
async def test_retrieve_tools_rag_happy_path() -> None:
    """Test RAG tools are retrieved when score and API ID match."""
    state: State = {
        "messages": [MagicMock(content="find my phone")],
        "summary": "",
        "chat_model_usage_metadata": {},
        "messages_to_remove": [],
        "selected_tools": [],
        "tool_routing_map": {},
    }
    store = MagicMock()

    # Mock search results
    item = MagicMock()
    item.value = {
        "name": "find_phone",
        "api_id": "hga_local",
        "description": "Find phone",
        "parameters": "{}",
        "is_actuation": False,
    }
    item.score = 0.9

    store.asearch = AsyncMock(return_value=[item])

    config: RunnableConfig = {
        "configurable": {
            "options": {"llm_hass_api": ["assist"], "tool_relevance_threshold": 0.15},
            "tool_index_ready": True,
            "langchain_tools": {},
            "ha_llm_api": MagicMock(apis={}),
        }
    }

    result = await _retrieve_tools(state, config, store=store)

    assert "find_phone" in result["tool_routing_map"]
    assert result["tool_routing_map"]["find_phone"] == "hga_local"
    assert len(result["selected_tools"]) == 1


@pytest.mark.asyncio
async def test_retrieve_tools_actuation_safety_net() -> None:
    """Test that actuation keywords trigger the safety net retrieval."""
    query = "turn on the kitchen lights"
    assert re.search(ACTUATION_KEYWORDS_REGEX, query)

    state: State = {
        "messages": [MagicMock(content=query)],
        "summary": "",
        "chat_model_usage_metadata": {},
        "messages_to_remove": [],
        "selected_tools": [],
        "tool_routing_map": {},
    }
    store = MagicMock()

    # Mock RAG returning nothing low score
    rag_item = MagicMock()
    rag_item.value = {
        "name": "other",
        "api_id": "assist",
        "description": "...",
        "parameters": "{}",
    }
    rag_item.score = 0.01

    # Mock Safety Net returning actuation tool
    safety_item = MagicMock()
    safety_item.value = {
        "name": "HassTurnOn",
        "api_id": "assist",
        "description": "Turn on",
        "parameters": "{}",
        "is_actuation": True,
    }

    # asearch will be called twice: once for RAG, once for Safety Net (with filter)
    store.asearch = AsyncMock(
        side_effect=[
            [rag_item],  # RAG call
            [safety_item],  # Safety Net call
        ]
    )

    config: RunnableConfig = {
        "configurable": {
            "options": {"llm_hass_api": ["assist"], "tool_relevance_threshold": 0.15},
            "tool_index_ready": True,
            "langchain_tools": {},
            "ha_llm_api": MagicMock(apis={}),
        }
    }

    result = await _retrieve_tools(state, config, store=store)

    # "other" should be filtered out by score
    assert "other" not in result["tool_routing_map"]
    # "HassTurnOn" should be present from safety net
    assert "HassTurnOn" in result["tool_routing_map"]
    assert len(result["selected_tools"]) == 1


@pytest.mark.asyncio
async def test_retrieve_tools_deduplication_safety_wins() -> None:
    """Test that safety tools take priority over RAG when both return the same tool."""
    state: State = {
        "messages": [MagicMock(content="turn on lights")],
        "summary": "",
        "chat_model_usage_metadata": {},
        "messages_to_remove": [],
        "selected_tools": [],
        "tool_routing_map": {},
    }
    store = MagicMock()

    # RAG returns HassTurnOn from hga_local (maybe a customized version)
    rag_item = MagicMock()
    rag_item.value = {
        "name": "HassTurnOn",
        "api_id": "hga_local",
        "description": "Custom Turn On",
        "parameters": "{}",
        "is_actuation": True,
    }
    rag_item.score = 0.9

    # Safety Net also returns HassTurnOn from assist (default version)
    safety_item = MagicMock()
    safety_item.value = {
        "name": "HassTurnOn",
        "api_id": "assist",
        "description": "Standard Turn On",
        "parameters": "{}",
        "is_actuation": True,
    }

    store.asearch = AsyncMock(
        side_effect=[
            [rag_item],  # RAG call
            [safety_item],  # Safety Net call
        ]
    )

    config: RunnableConfig = {
        "configurable": {
            "options": {"llm_hass_api": ["assist"], "tool_relevance_threshold": 0.15},
            "tool_index_ready": True,
            "langchain_tools": {},
            "ha_llm_api": MagicMock(apis={}),
        }
    }

    result = await _retrieve_tools(state, config, store=store)

    # HassTurnOn should be present
    assert "HassTurnOn" in result["tool_routing_map"]
    # Safety tool (assist) wins because safety tools take priority over RAG.
    assert result["tool_routing_map"]["HassTurnOn"] == "assist"
    assert len(result["selected_tools"]) == 1
    assert result["selected_tools"][0]["function"]["description"] == "Standard Turn On"


@pytest.mark.asyncio
async def test_retrieve_tools_fallback_on_empty_store() -> None:
    """Test that _retrieve_tools falls back to all tools if store search returns nothing."""
    state: State = {
        "messages": [MagicMock(content="turn on the lights")],
        "summary": "",
        "chat_model_usage_metadata": {},
        "messages_to_remove": [],
        "selected_tools": [],
        "tool_routing_map": {},
    }
    store = MagicMock()
    # Store returns empty results
    store.asearch = AsyncMock(return_value=[])

    # Mock tools
    ha_tool = MagicMock(spec=llm.Tool)
    ha_tool.name = "HassTurnOn"
    ha_tool.description = "Turn on something"
    ha_tool.parameters = {"type": "object", "properties": {}}

    api = MagicMock()
    api.tools = [ha_tool]
    api.custom_serializer = None

    ha_llm_api = MagicMock()
    ha_llm_api.apis = {"assist": api}

    lc_tool = MagicMock()
    lc_tool.description = "Local tool"
    lc_tool.args_schema = None

    config: RunnableConfig = {
        "configurable": {
            "options": {"llm_hass_api": ["assist"]},
            "tool_index_ready": True,
            "langchain_tools": {"local_tool": lc_tool},
            "ha_llm_api": ha_llm_api,
        }
    }

    result = await _retrieve_tools(state, config, store=store)

    # Should have all tools
    assert "HassTurnOn" in result["tool_routing_map"]
    assert "local_tool" in result["tool_routing_map"]
    assert len(result["selected_tools"]) == 2


@pytest.mark.asyncio
async def test_retrieve_tools_fallback_on_index_not_ready() -> None:
    """Test that _retrieve_tools falls back to all tools if index is not ready."""
    state: State = {
        "messages": [MagicMock(content="turn on the lights")],
        "summary": "",
        "chat_model_usage_metadata": {},
        "messages_to_remove": [],
        "selected_tools": [],
        "tool_routing_map": {},
    }
    store = MagicMock()
    # Store.asearch should not even be called if tool_index_ready is False
    store.asearch = AsyncMock()

    ha_tool = MagicMock(spec=llm.Tool)
    ha_tool.name = "HassTurnOn"
    ha_tool.description = "Turn on something"
    ha_tool.parameters = {"type": "object", "properties": {}}

    api = MagicMock()
    api.tools = [ha_tool]
    api.custom_serializer = None

    ha_llm_api = MagicMock()
    ha_llm_api.apis = {"assist": api}

    config: RunnableConfig = {
        "configurable": {
            "options": {"llm_hass_api": ["assist"]},
            "tool_index_ready": False,
            "langchain_tools": {},
            "ha_llm_api": ha_llm_api,
        }
    }

    result = await _retrieve_tools(state, config, store=store)

    store.asearch.assert_not_called()
    assert "HassTurnOn" in result["tool_routing_map"]
    assert len(result["selected_tools"]) == 1


@pytest.mark.asyncio
async def test_retrieve_tools_store_is_none(caplog: pytest.LogCaptureFixture) -> None:
    """Verify that retrieval functions handle store=None gracefully."""
    config: RunnableConfig = {"configurable": {"tool_index_ready": True}}
    allowed = {"assist"}

    # Test RAG retrieval
    rag_tools = await _get_rag_retrieved_tools(None, config, "query", allowed)
    assert rag_tools == []
    assert "Store is None; skipping RAG tool retrieval" in caplog.text

    # Test Safety Net retrieval
    safety_tools = await _get_actuation_safety_tools(
        None, config, "turn on lights", allowed
    )
    assert safety_tools == []
    assert "Store is None; skipping actuation safety tools" in caplog.text


@pytest.mark.asyncio
async def test_retrieve_tools_specific_exceptions(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Verify that retrieval functions handle specific exceptions (psycopg, ValueError)."""
    store = MagicMock()
    config: RunnableConfig = {
        "configurable": {
            "tool_index_ready": True,
            "options": {"tool_retrieval_limit": 5},
        }
    }
    allowed = {"assist"}

    # 1. Test psycopg.OperationalError
    store.asearch = AsyncMock(side_effect=psycopg.OperationalError("Conn lost"))
    rag_tools = await _get_rag_retrieved_tools(store, config, "query", allowed)
    assert rag_tools == []
    assert "RAG tool retrieval search failed (known error): Conn lost" in caplog.text

    # 2. Test ValueError
    store.asearch = AsyncMock(side_effect=ValueError("Invalid filter"))
    safety_tools = await _get_actuation_safety_tools(
        store, config, "turn on lights", allowed
    )
    assert safety_tools == []
    assert (
        "Deterministic safety tool filter failed (known error): Invalid filter"
        in caplog.text
    )

    # 3. Test unexpected Exception (last resort)
    store.asearch = AsyncMock(side_effect=RuntimeError("Boom"))
    rag_tools = await _get_rag_retrieved_tools(store, config, "query", allowed)
    assert rag_tools == []
    assert "Unexpected RAG tool retrieval search failure" in caplog.text


# ---------------------------------------------------------------------------
# _split_query_intents
# ---------------------------------------------------------------------------


def test_split_query_intents_single_intent() -> None:
    """Single-intent queries are returned unchanged (as a 1-element list)."""
    query = "what is the temperature in the living room"
    assert _split_query_intents(query) == [query]


def test_split_query_intents_multi_intent_and() -> None:
    """Multi-intent queries include the original plus per-intent sub-queries."""
    query = "turn on the kitchen light and check the back yard camera"
    result = _split_query_intents(query)
    assert result[0] == query  # original always first
    assert len(result) > 1
    assert any("turn on" in part for part in result[1:])
    assert any("camera" in part for part in result[1:])


def test_split_query_intents_comma_split() -> None:
    """Comma-separated intents are split into sub-queries."""
    query = "tell me the time in London, turn on the garage light"
    result = _split_query_intents(query)
    assert result[0] == query
    assert any("time" in part for part in result[1:])
    assert any("garage" in part for part in result[1:])


def test_split_query_intents_short_fragment_filtered() -> None:
    """Fragments shorter than _MIN_SUBQUERY_LEN are dropped."""
    # "UK" is 2 chars — well below the 8-char minimum
    query = "tell me the time in London, UK, and turn on the lights"
    result = _split_query_intents(query)
    assert result[0] == query
    assert "UK" not in result[1:]


def test_split_query_intents_empty_string() -> None:
    """Empty string returns a list containing just the empty string."""
    assert _split_query_intents("") == [""]


# ---------------------------------------------------------------------------
# _retrieve_tools: fallback when all candidates are filtered by api_id
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_retrieve_tools_fallback_when_candidates_filtered_by_api() -> None:
    """Fallback fires when RAG/safety return items with a disallowed api_id."""
    state: State = {
        "messages": [MagicMock(content="find my keys")],
        "summary": "",
        "chat_model_usage_metadata": {},
        "messages_to_remove": [],
        "selected_tools": [],
        "tool_routing_map": {},
    }
    store = MagicMock()

    # All store results carry an api_id that is not in allowed_api_ids
    bad_item = MagicMock()
    bad_item.value = {
        "name": "find_keys",
        "api_id": "unknown_provider",
        "description": "Find keys",
        "parameters": "{}",
        "is_actuation": False,
    }
    bad_item.score = 0.95
    store.asearch = AsyncMock(return_value=[bad_item])

    ha_tool = MagicMock(spec=llm.Tool)
    ha_tool.name = "HassSearch"
    ha_tool.description = "Search HA"
    ha_tool.parameters = {"type": "object", "properties": {}}

    api = MagicMock()
    api.tools = [ha_tool]
    api.custom_serializer = None
    ha_llm_api = MagicMock()
    ha_llm_api.apis = {"assist": api}

    config: RunnableConfig = {
        "configurable": {
            "options": {"llm_hass_api": ["assist"]},
            "tool_index_ready": True,
            "langchain_tools": {},
            "ha_llm_api": ha_llm_api,
        }
    }

    result = await _retrieve_tools(state, config, store=store)

    # Disallowed-api item filtered; fallback provides the HA tool
    assert "find_keys" not in result["tool_routing_map"]
    assert "HassSearch" in result["tool_routing_map"]


# ---------------------------------------------------------------------------
# score=None guard — issue #394
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_rag_retrieval_none_score_is_treated_as_zero() -> None:
    """
    item.score=None must not raise TypeError (issue #394).

    getattr(item, 'score', 0.0) returns None when the attribute exists but is
    None; the fix normalises it to 0.0 so the threshold comparison is safe.
    """
    store = MagicMock()
    item = MagicMock()
    item.value = {
        "name": "some_tool",
        "api_id": "hga_local",
        "description": "A tool",
        "parameters": "{}",
        "is_actuation": False,
    }
    item.score = None

    store.asearch = AsyncMock(return_value=[item])

    config: RunnableConfig = {
        "configurable": {
            "options": {"tool_relevance_threshold": 0.15},
            "tool_index_ready": True,
        }
    }

    # Must not raise; None score < threshold so tool is filtered out
    result = await _get_rag_retrieved_tools(store, config, "query", {"hga_local"})
    assert result == []


@pytest.mark.asyncio
async def test_actuation_safety_none_score_does_not_crash() -> None:
    """
    item.score=None in actuation safety sort must not raise TypeError (issue #394).

    The lambda key used in sorted() had the same getattr default bug.
    """
    store = MagicMock()
    item = MagicMock()
    item.value = {
        "name": "HassTurnOn",
        "api_id": "assist",
        "description": "Turn on",
        "parameters": "{}",
        "is_actuation": True,
    }
    item.score = None

    store.asearch = AsyncMock(return_value=[item])

    config: RunnableConfig = {
        "configurable": {
            "options": {},
            "tool_index_ready": True,
        }
    }

    # Must not raise; tool is still returned (actuation safety is not score-gated)
    result = await _get_actuation_safety_tools(
        store, config, "turn on the lights", {"assist"}
    )
    assert any(t["name"] == "HassTurnOn" for t in result)


# ---------------------------------------------------------------------------
# _query_needs_actuation_safety — issue #394 follow-up
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "query",
    [
        "list all open windows",
        "which doors are open",
        "show open entry sensors",
        "are the gates open",
        "is the garage vent open",
        "show me which windows are open",
        "what windows are open right now",
        "where are the open doors",
    ],
)
def test_query_needs_actuation_safety_suppressed_for_read_only_open(query: str) -> None:
    """Read-only open-state queries must not trigger actuation safety injection."""
    assert not _query_needs_actuation_safety(query)


@pytest.mark.parametrize(
    "query",
    [
        "open the garage door",
        "open the gates",
        "close the family room blinds",
        "turn on the kitchen light",
        "lock the front door",
        "show me which windows are open and then close them",
        "show me which windows are open and then open the garage door",
    ],
)
def test_query_needs_actuation_safety_enabled_for_actuation(query: str) -> None:
    """True actuation commands and compound queries must still trigger safety injection."""
    assert _query_needs_actuation_safety(query)


def test_query_needs_actuation_safety_no_actuation_keyword() -> None:
    """Queries with no actuation keyword need no actuation safety."""
    assert not _query_needs_actuation_safety(
        "what is the temperature in the living room"
    )


@pytest.mark.parametrize(
    ("tool_args", "query"),
    [
        ({"domain": ["binary_sensor"], "name": "Window"}, "list all open windows"),
        (
            {"domain": ["binary_sensor"], "name": "Door"},
            "list the open doors in my house",
        ),
        (
            {"domain": ["binary_sensor"], "name": ["Sliding Door", "Door Lock"]},
            "list the open doors in my house",
        ),
    ],
)
def test_normalize_live_context_args_for_read_only_open_state(
    tool_args: dict[str, Any], query: str
) -> None:
    """Brittle model-generated live-context filters are widened for open-state checks."""
    assert _normalize_live_context_args_for_open_state(
        "GetLiveContext", tool_args, query
    ) == {"domain": "binary_sensor"}


def test_normalize_live_context_args_does_not_touch_open_command() -> None:
    """Actual open commands must not be converted to read-only live context."""
    tool_args: dict[str, Any] = {"domain": ["cover"], "name": "Garage Door"}

    assert (
        _normalize_live_context_args_for_open_state(
            "GetLiveContext", tool_args, "open the garage door"
        )
        == tool_args
    )


def test_normalize_live_context_args_can_leave_subsequent_calls_alone() -> None:
    """Only the first live-context call is widened to avoid duplicate broad payloads."""
    tool_args: dict[str, Any] = {
        "domain": "binary_sensor",
        "name": "Breakfast Nook Left Window",
    }

    assert (
        _normalize_live_context_args_for_open_state(
            "GetLiveContext",
            tool_args,
            "list all open windows",
            force_broad=False,
        )
        == tool_args
    )


def test_latest_open_state_query_uses_previous_query_for_retry() -> None:
    """Short retries should keep the prior read-only open-state query active."""
    messages: list[BaseMessage] = [
        HumanMessage(content="list all open windows"),
        AIMessage(content="Would you like me to try once more?"),
        HumanMessage(content="yes"),
    ]

    assert _latest_open_state_query(messages) == "list all open windows"


def test_filter_open_state_live_context_scopes_door_query() -> None:
    """Door queries must not expose open windows from the broad binary sensor context."""
    payload = {
        "success": True,
        "result": (
            "Live Context:\n"
            "- names: Front Door\n"
            "  domain: binary_sensor\n"
            "  state: 'off'\n"
            "  attributes:\n"
            "    device_class: opening\n"
            "- names: Family Room Right Window\n"
            "  domain: binary_sensor\n"
            "  state: 'on'\n"
            "  attributes:\n"
            "    device_class: opening\n"
            "- names: Garage and Play Room Doors\n"
            "  domain: binary_sensor\n"
            "  state: 'off'\n"
            "  attributes:\n"
            "    device_class: opening\n"
        ),
    }

    filtered = json.loads(
        _filter_open_state_live_context_content(
            json.dumps(payload), "list the open doors in my house"
        )
    )

    assert filtered["result"] == "Live Context: No open doors were found."


def test_filter_open_state_live_context_keeps_requested_open_windows() -> None:
    """Window queries keep only matching open windows from the broad context."""
    payload = {
        "success": True,
        "result": (
            "Live Context:\n"
            "- names: Breakfast Nook Side Right Window\n"
            "  domain: binary_sensor\n"
            "  state: 'on'\n"
            "  attributes:\n"
            "    device_class: opening\n"
            "- names: Family Room Sliding Door\n"
            "  domain: binary_sensor\n"
            "  state: 'on'\n"
            "  attributes:\n"
            "    device_class: opening\n"
            "- names: Landing Windows\n"
            "  domain: binary_sensor\n"
            "  state: 'on'\n"
            "  attributes:\n"
            "    device_class: opening\n"
        ),
    }

    filtered = json.loads(
        _filter_open_state_live_context_content(
            json.dumps(payload), "list all open windows"
        )
    )

    assert "Breakfast Nook Side Right Window" in filtered["result"]
    assert "Landing Windows" in filtered["result"]
    assert "Family Room Sliding Door" not in filtered["result"]


def test_filter_open_state_live_context_real_ha_device_class_garage_door() -> None:
    """Entities with device_class: garage_door must be recognised as open-state sensors."""
    payload = {
        "success": True,
        "result": (
            "Live Context:\n"
            "- names: Garage Door\n"
            "  domain: binary_sensor\n"
            "  state: 'on'\n"
            "  attributes:\n"
            "    device_class: garage_door\n"
        ),
    }

    filtered = json.loads(
        _filter_open_state_live_context_content(
            json.dumps(payload), "list all open doors in my home"
        )
    )

    assert "Garage Door" in filtered["result"]


def test_filter_open_state_live_context_real_ha_device_class_window() -> None:
    """Entities with device_class: window must be recognised as open-state sensors."""
    payload = {
        "success": True,
        "result": (
            "Live Context:\n"
            "- names: Window - kitchen\n"
            "  domain: binary_sensor\n"
            "  state: 'on'\n"
            "  areas: kitchen\n"
            "  attributes:\n"
            "    device_class: window\n"
            "- names: Window - bedroom\n"
            "  domain: binary_sensor\n"
            "  state: 'off'\n"
            "  areas: bedroom\n"
            "  attributes:\n"
            "    device_class: window\n"
        ),
    }

    filtered = json.loads(
        _filter_open_state_live_context_content(
            json.dumps(payload), "list all open windows in my home"
        )
    )

    assert "Window - kitchen" in filtered["result"]
    assert "Window - bedroom" not in filtered["result"]


def test_filter_open_state_live_context_real_ha_device_class_door() -> None:
    """Entities with device_class: door must be recognised as open-state sensors."""
    payload = {
        "success": True,
        "result": (
            "Live Context:\n"
            "- names: Front Door\n"
            "  domain: binary_sensor\n"
            "  state: 'on'\n"
            "  attributes:\n"
            "    device_class: door\n"
            "- names: Back Door\n"
            "  domain: binary_sensor\n"
            "  state: 'off'\n"
            "  attributes:\n"
            "    device_class: door\n"
        ),
    }

    filtered = json.loads(
        _filter_open_state_live_context_content(
            json.dumps(payload), "list all open doors in my home"
        )
    )

    assert "Front Door" in filtered["result"]
    assert "Back Door" not in filtered["result"]


def test_filter_open_state_live_context_window_query_excludes_doors() -> None:
    """Window query with mixed device classes must keep only open windows."""
    payload = {
        "success": True,
        "result": (
            "Live Context:\n"
            "- names: Window - kitchen\n"
            "  domain: binary_sensor\n"
            "  state: 'on'\n"
            "  attributes:\n"
            "    device_class: window\n"
            "- names: Front Door\n"
            "  domain: binary_sensor\n"
            "  state: 'on'\n"
            "  attributes:\n"
            "    device_class: door\n"
        ),
    }

    filtered = json.loads(
        _filter_open_state_live_context_content(
            json.dumps(payload), "list all open windows in my home"
        )
    )

    assert "Window - kitchen" in filtered["result"]
    assert "Front Door" not in filtered["result"]


def test_query_needs_actuation_safety_comma_compound_known_gap() -> None:
    """Known gap: comma-separated 'open' command is not detected; actuation suppressed."""
    query = "show me open windows, open the garage door"
    # The comma form is NOT detected — actuation is incorrectly suppressed.
    assert not _query_needs_actuation_safety(query)


@pytest.mark.asyncio
async def test_retrieve_tools_no_actuation_safety_for_read_only_open() -> None:
    """_retrieve_tools must not force-inject actuation tools for 'list all open windows'."""
    query = "list all open windows"
    state: State = {
        "messages": [MagicMock(content=query)],
        "summary": "",
        "chat_model_usage_metadata": {},
        "messages_to_remove": [],
        "selected_tools": [],
        "tool_routing_map": {},
        "action_rounds": 0,
    }
    store = MagicMock()

    live_ctx_item = MagicMock()
    live_ctx_item.value = {
        "name": "GetLiveContext",
        "api_id": "hga_local",
        "description": "Get live home state",
        "parameters": "{}",
        "is_actuation": False,
    }
    live_ctx_item.score = 0.85

    actuation_item = MagicMock()
    actuation_item.value = {
        "name": "HassTurnOn",
        "api_id": "assist",
        "description": "Turn on",
        "parameters": "{}",
        "is_actuation": True,
    }
    actuation_item.score = 0.0

    # RAG returns GetLiveContext; safety search should be skipped entirely.
    store.asearch = AsyncMock(return_value=[live_ctx_item])

    config: RunnableConfig = {
        "configurable": {
            "options": {"llm_hass_api": ["assist"], "tool_relevance_threshold": 0.15},
            "tool_index_ready": True,
            "langchain_tools": {},
            "ha_llm_api": MagicMock(apis={}),
        }
    }

    result = await _retrieve_tools(state, config, store=store)

    assert "GetLiveContext" in result["tool_routing_map"]
    assert "HassTurnOn" not in result["tool_routing_map"]
    # Counter must be reset to 0 at the start of each turn.
    assert result["action_rounds"] == 0
    # GetLiveContext must be first in the ordered tool list (issue #394).
    assert result["selected_tools"][0]["function"]["name"] == "GetLiveContext"
    # No actuation tool should appear anywhere in the selection.
    selected_names = [t["function"]["name"] for t in result["selected_tools"]]
    assert "HassTurnOn" not in selected_names


@pytest.mark.asyncio
async def test_retrieve_tools_force_injects_live_context_for_open_doors() -> None:
    """Read-only open-door queries must bind GetLiveContext even when RAG misses it."""
    query = "list the open doors in my house"
    state: State = {
        "messages": [MagicMock(content=query)],
        "summary": "",
        "chat_model_usage_metadata": {},
        "messages_to_remove": [],
        "selected_tools": [],
        "tool_routing_map": {},
        "action_rounds": 0,
    }
    store = MagicMock()

    rag_items = []
    for name, is_actuation in (
        ("get_entity_history", False),
        ("HassBroadcast", True),
        ("HassTurnOn", True),
        ("resolve_entity_ids", False),
        ("alarm_control", True),
    ):
        item = MagicMock()
        item.value = {
            "name": name,
            "api_id": "assist",
            "description": name,
            "parameters": "{}",
            "is_actuation": is_actuation,
        }
        item.score = 0.9
        rag_items.append(item)

    live_ctx_item = MagicMock()
    live_ctx_item.value = {
        "name": "GetLiveContext",
        "api_id": "assist",
        "description": "Get live home state",
        "parameters": "{}",
        "is_actuation": False,
    }

    store.asearch = AsyncMock(return_value=rag_items)
    store.aget = AsyncMock(return_value=live_ctx_item)

    config: RunnableConfig = {
        "configurable": {
            "options": {"llm_hass_api": ["assist"], "tool_relevance_threshold": 0.15},
            "tool_index_ready": True,
            "langchain_tools": {},
            "ha_llm_api": MagicMock(apis={}),
        }
    }

    result = await _retrieve_tools(state, config, store=store)
    selected_names = [t["function"]["name"] for t in result["selected_tools"]]

    assert selected_names[0] == "GetLiveContext"
    assert "GetLiveContext" in result["tool_routing_map"]
    assert "get_entity_history" in result["tool_routing_map"]
    assert "resolve_entity_ids" in result["tool_routing_map"]
    assert "HassTurnOn" not in result["tool_routing_map"]
    assert "HassBroadcast" not in result["tool_routing_map"]
    assert "alarm_control" not in result["tool_routing_map"]


@pytest.mark.asyncio
async def test_retrieve_tools_open_command_does_not_force_live_context() -> None:
    """Open commands must keep actuation safety behavior."""
    query = "open the garage door"
    state: State = {
        "messages": [MagicMock(content=query)],
        "summary": "",
        "chat_model_usage_metadata": {},
        "messages_to_remove": [],
        "selected_tools": [],
        "tool_routing_map": {},
        "action_rounds": 0,
    }
    store = MagicMock()

    safety_item = MagicMock()
    safety_item.value = {
        "name": "HassTurnOn",
        "api_id": "assist",
        "description": "Turn on",
        "parameters": "{}",
        "is_actuation": True,
    }
    safety_item.score = 0.9

    store.asearch = AsyncMock(return_value=[safety_item])
    store.aget = AsyncMock()

    config: RunnableConfig = {
        "configurable": {
            "options": {"llm_hass_api": ["assist"], "tool_relevance_threshold": 0.15},
            "tool_index_ready": True,
            "langchain_tools": {},
            "ha_llm_api": MagicMock(apis={}),
        }
    }

    result = await _retrieve_tools(state, config, store=store)

    assert "HassTurnOn" in result["tool_routing_map"]
    assert "GetLiveContext" not in result["tool_routing_map"]
    store.aget.assert_not_called()


@pytest.mark.asyncio
async def test_retrieve_tools_action_rounds_reset() -> None:
    """action_rounds is reset to 0 by _retrieve_tools regardless of prior state."""
    query = "list all open windows"
    state: State = {
        "messages": [MagicMock(content=query)],
        "summary": "",
        "chat_model_usage_metadata": {},
        "messages_to_remove": [],
        "selected_tools": [],
        "tool_routing_map": {},
        "action_rounds": 3,  # simulate a prior turn that hit the limit
    }
    store = MagicMock()
    store.asearch = AsyncMock(return_value=[])

    ha_tool = MagicMock(spec=llm.Tool)
    ha_tool.name = "GetLiveContext"
    ha_tool.description = "Get live home state"
    ha_tool.parameters = {"type": "object", "properties": {}}

    api = MagicMock()
    api.tools = [ha_tool]
    api.custom_serializer = None
    ha_llm_api = MagicMock()
    ha_llm_api.apis = {"assist": api}

    config: RunnableConfig = {
        "configurable": {
            "options": {"llm_hass_api": ["assist"]},
            "tool_index_ready": True,
            "langchain_tools": {},
            "ha_llm_api": ha_llm_api,
        }
    }

    result = await _retrieve_tools(state, config, store=store)
    assert result["action_rounds"] == 0

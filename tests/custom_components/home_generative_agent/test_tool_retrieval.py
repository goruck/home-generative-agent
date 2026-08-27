# ruff: noqa: S101
"""Unit tests for tool retrieval logic (RAG, safety net, fallbacks)."""

from __future__ import annotations

import json
import re
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any
from unittest.mock import AsyncMock, MagicMock

import psycopg
import pytest
from homeassistant.helpers import llm
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage

from custom_components.home_generative_agent.agent.graph import (
    State,
    _conversation_has_automation_context,
    _filter_open_state_live_context_content,
    _get_actuation_safety_tools,
    _get_allowed_api_ids,
    _get_rag_retrieved_tools,
    _latest_open_state_query,
    _normalize_live_context_args_for_open_state,
    _query_needs_actuation_safety,
    _query_wants_automation,
    _retrieve_tools,
    _split_query_intents,
)
from custom_components.home_generative_agent.const import ACTUATION_KEYWORDS_REGEX

if TYPE_CHECKING:
    from langchain_core.runnables import RunnableConfig


def _live_llm_api(*tool_names: str) -> MagicMock:
    """
    Build a MultiLLMAPI mock whose assist API exposes the given tools live.

    _retrieve_tools filters index candidates against the live (api_id, name)
    set, so fixtures must declare which assist tools exist for the turn.
    """
    api = SimpleNamespace(
        tools=[
            SimpleNamespace(
                name=name,
                description=f"{name} live",
                parameters={"type": "object", "properties": {}},
            )
            for name in tool_names
        ],
        custom_serializer=None,
    )
    return MagicMock(apis={"assist": api})


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
            "langchain_tools": {"find_phone": MagicMock()},
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
            "ha_llm_api": _live_llm_api("HassTurnOn"),
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
            "langchain_tools": {"HassTurnOn": MagicMock()},
            "ha_llm_api": _live_llm_api("HassTurnOn"),
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
async def test_retrieve_tools_fallback_when_live_index_becomes_stale(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Live runtime state wins when vector search marks the index stale."""
    state: State = {
        "messages": [MagicMock(content="what can you do?")],
        "summary": "",
        "chat_model_usage_metadata": {},
        "messages_to_remove": [],
        "selected_tools": [],
        "tool_routing_map": {},
    }
    runtime_data = SimpleNamespace(
        tool_index_ready=True,
        tool_indexing_in_progress=False,
        tool_index_failed=False,
    )
    store = MagicMock()

    item = MagicMock()
    item.value = {
        "name": "stale_vector_tool",
        "api_id": "assist",
        "description": "Stale vector tool",
        "parameters": "{}",
        "is_actuation": False,
    }
    item.score = 0.99

    async def _stale_search(*_args: Any, **_kwargs: Any) -> list[Any]:
        runtime_data.tool_index_ready = False
        return [item]

    store.asearch = AsyncMock(side_effect=_stale_search)

    ha_tool = MagicMock(spec=llm.Tool)
    ha_tool.name = "GetLiveContext"
    ha_tool.description = "Get live context"
    ha_tool.parameters = {"type": "object", "properties": {}}

    api = MagicMock()
    api.tools = [ha_tool]
    api.custom_serializer = None

    config: RunnableConfig = {
        "configurable": {
            "options": {"llm_hass_api": ["assist"]},
            "hga_runtime_data": runtime_data,
            "tool_index_ready": True,
            "langchain_tools": {},
            "ha_llm_api": MagicMock(apis={"assist": api}),
        }
    }

    result = await _retrieve_tools(state, config, store=store)

    assert "stale_vector_tool" not in result["tool_routing_map"]
    assert result["tool_routing_map"] == {"GetLiveContext": "assist"}
    assert "tool index not ready" in caplog.text


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


@pytest.mark.asyncio
async def test_retrieve_tools_vector_dimension_mismatch_is_known_error(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Vector dimension mismatches are expected during embedding provider changes."""
    store = MagicMock()
    store.asearch = AsyncMock(
        side_effect=psycopg.DataError("different vector dimensions 1024 and 3072")
    )
    config: RunnableConfig = {
        "configurable": {
            "tool_index_ready": True,
            "options": {"tool_retrieval_limit": 5},
        }
    }

    rag_tools = await _get_rag_retrieved_tools(store, config, "query", {"assist"})

    assert rag_tools == []
    assert "RAG tool retrieval search failed (known error)" in caplog.text
    assert "different vector dimensions" in caplog.text
    assert "Unexpected RAG tool retrieval search failure" not in caplog.text


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
            "langchain_tools": {"GetLiveContext": MagicMock()},
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
            "ha_llm_api": _live_llm_api(
                "get_entity_history",
                "HassBroadcast",
                "HassTurnOn",
                "resolve_entity_ids",
                "alarm_control",
                "GetLiveContext",
            ),
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
async def test_retrieve_tools_open_command_includes_live_context() -> None:
    """Open commands keep actuation safety behavior and now always include GetLiveContext."""
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

    live_ctx_item = MagicMock()
    live_ctx_item.value = {
        "name": "GetLiveContext",
        "api_id": "assist",
        "description": "Get live home state",
        "parameters": "{}",
        "is_actuation": False,
    }

    store.asearch = AsyncMock(return_value=[safety_item])
    store.aget = AsyncMock(return_value=live_ctx_item)

    config: RunnableConfig = {
        "configurable": {
            "options": {"llm_hass_api": ["assist"], "tool_relevance_threshold": 0.15},
            "tool_index_ready": True,
            "langchain_tools": {},
            "ha_llm_api": _live_llm_api("HassTurnOn", "GetLiveContext"),
        }
    }

    result = await _retrieve_tools(state, config, store=store)

    assert "HassTurnOn" in result["tool_routing_map"]
    assert "GetLiveContext" in result["tool_routing_map"]
    store.aget.assert_called()


@pytest.mark.asyncio
async def test_retrieve_tools_live_context_not_injected_when_store_returns_none() -> (
    None
):
    """When store.aget returns None for GetLiveContext, it is silently skipped."""
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
    # aget returns None — GetLiveContext not in the tool index.
    store.aget = AsyncMock(return_value=None)

    config: RunnableConfig = {
        "configurable": {
            "options": {"llm_hass_api": ["assist"], "tool_relevance_threshold": 0.15},
            "tool_index_ready": True,
            "langchain_tools": {},
            "ha_llm_api": _live_llm_api("HassTurnOn"),
        }
    }

    result = await _retrieve_tools(state, config, store=store)

    # HassTurnOn still added; GetLiveContext silently absent.
    assert "HassTurnOn" in result["tool_routing_map"]
    assert "GetLiveContext" not in result["tool_routing_map"]


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


# ---------------------------------------------------------------------------
# Automation-creation intent: _query_wants_automation + add_automation
# force-injection (regression for field report: "Always turn on the garage
# light when the garage door is unloacked and send a notification" bound only
# entity-control tools, so the agent could not create the automation).
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "query",
    [
        # Field repro, typo preserved.
        (
            "Always turn on the garage light when the garage door is unloacked "
            "and send a notification"
        ),
        "Create an automation to water the plants every day",
        "remind me every 30 minutes if the litter box waste drawer is over 90% full",
        "whenever the front door opens announce it on the kitchen speaker",
        "turn on the porch light when motion is detected",
        "automatically close the garage at 10pm",
        "every morning turn up the thermostat",
    ],
)
def test_query_wants_automation_positive(query: str) -> None:
    """Automation-creation phrasings must be detected."""
    assert _query_wants_automation(query)


@pytest.mark.parametrize(
    "query",
    [
        "turn on the kitchen lights",
        "find my phone",
        "is the garage door open",
        "what is the temperature in the living room",
        "dim the bedroom lights to 50%",
    ],
)
def test_query_wants_automation_negative(query: str) -> None:
    """Direct commands and read-only queries must not be detected."""
    assert not _query_wants_automation(query)


def _make_search_item(
    name: str, *, score: float, is_actuation: bool = False, api_id: str = "assist"
) -> MagicMock:
    item = MagicMock()
    item.value = {
        "name": name,
        "api_id": api_id,
        "description": f"{name} description",
        "parameters": "{}",
        "is_actuation": is_actuation,
    }
    item.score = score
    return item


def _field_report_store() -> MagicMock:
    """Store mock replicating the field log: control tools win RAG ranking."""
    store = MagicMock()
    store.asearch = AsyncMock(
        return_value=[
            _make_search_item("HassBroadcast", score=0.635),
            _make_search_item(
                "confirm_sensitive_action", score=0.580, api_id="hga_local"
            ),
            _make_search_item(
                "alarm_control", score=0.580, is_actuation=True, api_id="hga_local"
            ),
            _make_search_item("HassTurnOn", score=0.578, is_actuation=True),
            _make_search_item("HassLightSet", score=0.559, is_actuation=True),
        ]
    )

    add_automation_item = MagicMock()
    add_automation_item.value = {
        "name": "add_automation",
        "api_id": "hga_local",
        "description": "Add an automation to Home Assistant.",
        "parameters": "{}",
        "is_actuation": False,
    }

    async def aget(namespace: Any, key: str = "", **_kwargs: Any) -> Any:  # noqa: ARG001
        if key.endswith("::add_automation"):
            return add_automation_item
        return None

    store.aget = AsyncMock(side_effect=aget)
    return store


def _field_report_state() -> State:
    return {
        "messages": [
            MagicMock(
                content=(
                    "Always turn on the garage light when the garage door is "
                    "unloacked and send a notification"
                )
            )
        ],
        "summary": "",
        "chat_model_usage_metadata": {},
        "messages_to_remove": [],
        "selected_tools": [],
        "tool_routing_map": {},
    }


def _field_report_config(**extra_options: Any) -> RunnableConfig:
    return {
        "configurable": {
            "options": {
                "llm_hass_api": ["assist"],
                "tool_relevance_threshold": 0.15,
                **extra_options,
            },
            "tool_index_ready": True,
            "langchain_tools": {
                "confirm_sensitive_action": MagicMock(),
                "alarm_control": MagicMock(),
                "add_automation": MagicMock(),
            },
            "ha_llm_api": _live_llm_api("HassBroadcast", "HassTurnOn", "HassLightSet"),
        }
    }


@pytest.mark.asyncio
async def test_retrieve_tools_injects_add_automation_for_automation_intent() -> None:
    """add_automation is force-bound when RAG ranking misses it (field repro)."""
    store = _field_report_store()

    result = await _retrieve_tools(
        _field_report_state(), _field_report_config(), store=store
    )

    assert "add_automation" in result["tool_routing_map"]
    assert result["tool_routing_map"]["add_automation"] == "hga_local"
    # Injection must not evict the RAG/safety selections.
    assert "HassTurnOn" in result["tool_routing_map"]
    assert "HassBroadcast" in result["tool_routing_map"]


@pytest.mark.asyncio
async def test_retrieve_tools_no_add_automation_in_schema_first_yaml_mode() -> None:
    """Schema-first YAML mode must not force-bind add_automation."""
    store = _field_report_store()

    result = await _retrieve_tools(
        _field_report_state(),
        _field_report_config(schema_first_yaml=True),
        store=store,
    )

    assert "add_automation" not in result["tool_routing_map"]
    add_automation_lookups = [
        call
        for call in store.aget.call_args_list
        if str(call.kwargs.get("key", "")).endswith("::add_automation")
    ]
    assert not add_automation_lookups


@pytest.mark.asyncio
async def test_retrieve_tools_no_add_automation_without_intent() -> None:
    """Plain commands must not trigger the add_automation lookup."""
    store = _field_report_store()
    state = _field_report_state()
    state["messages"] = [MagicMock(content="turn on the kitchen lights")]

    result = await _retrieve_tools(state, _field_report_config(), store=store)

    assert "add_automation" not in result["tool_routing_map"]
    add_automation_lookups = [
        call
        for call in store.aget.call_args_list
        if str(call.kwargs.get("key", "")).endswith("::add_automation")
    ]
    assert not add_automation_lookups


def _automation_followup_messages() -> list[Any]:
    """Replicate the field log: automation created, model offers more, user says yes."""
    return [
        HumanMessage(
            content=(
                "Always turn on the garage light when the garage door is "
                "unlocked and send a notification"
            )
        ),
        AIMessage(
            content="I'll create this for you.",
            tool_calls=[
                {
                    "name": "add_automation",
                    "args": {"automation_yaml": "alias: Garage Light"},
                    "id": "call-1",
                    "type": "tool_call",
                }
            ],
        ),
        ToolMessage(
            content="Added automation 01KYB3K9TQWXST6AR28EQ9Q3NB",
            name="add_automation",
            tool_call_id="call-1",
        ),
        AIMessage(content="Done. Want me to add the notification action as well?"),
        HumanMessage(content="yes"),
    ]


@pytest.mark.asyncio
async def test_retrieve_tools_keeps_add_automation_for_yes_followup() -> None:
    """A bare 'yes' after an add_automation turn must keep the tool bound."""
    store = _field_report_store()
    state = _field_report_state()
    state["messages"] = _automation_followup_messages()

    result = await _retrieve_tools(state, _field_report_config(), store=store)

    assert "add_automation" in result["tool_routing_map"]


@pytest.mark.asyncio
async def test_retrieve_tools_no_add_automation_for_unrelated_yes() -> None:
    """A bare 'yes' with no automation context must not bind add_automation."""
    store = _field_report_store()
    state = _field_report_state()
    state["messages"] = [
        HumanMessage(content="is the front door open"),
        AIMessage(content="It is closed. Want me to check the windows too?"),
        HumanMessage(content="yes"),
    ]

    result = await _retrieve_tools(state, _field_report_config(), store=store)

    assert "add_automation" not in result["tool_routing_map"]


def test_conversation_has_automation_context_detects_recent_turns() -> None:
    """Human intent, AI tool call, and ToolMessage all establish context."""
    assert _conversation_has_automation_context(_automation_followup_messages())
    assert _conversation_has_automation_context(
        [
            ToolMessage(
                content="Added automation X",
                name="add_automation",
                tool_call_id="call-9",
            ),
            HumanMessage(content="yes"),
        ]
    )
    assert not _conversation_has_automation_context(
        [
            HumanMessage(content="is the front door open"),
            AIMessage(content="It is closed."),
            HumanMessage(content="yes"),
        ]
    )


def test_conversation_has_automation_context_respects_lookback_window() -> None:
    """Automation context older than the lookback window must not persist."""
    old_context = _automation_followup_messages()[:3]
    padding: list[Any] = []
    for i in range(4):
        padding.append(HumanMessage(content=f"what is the temperature {i}"))
        padding.append(AIMessage(content=f"It is {70 + i} degrees."))
    messages = [*old_context, *padding, HumanMessage(content="yes")]

    assert not _conversation_has_automation_context(messages)


@pytest.mark.asyncio
async def test_retrieve_tools_add_automation_missing_from_store_is_noop() -> None:
    """Automation intent with no indexed add_automation must not crash or bind."""
    store = _field_report_store()
    store.aget = AsyncMock(return_value=None)

    # Not in the index (aget=None) and not live either — otherwise the
    # _get_tool_by_name fallback would legitimately supply it from config.
    config = _field_report_config()
    del config.get("configurable", {})["langchain_tools"]["add_automation"]

    result = await _retrieve_tools(_field_report_state(), config, store=store)

    assert "add_automation" not in result["tool_routing_map"]
    # The RAG/safety selections must be unaffected by the failed lookup.
    assert "HassTurnOn" in result["tool_routing_map"]


@pytest.mark.asyncio
async def test_retrieve_tools_add_automation_from_rag_skips_injection() -> None:
    """When RAG already ranked add_automation, the force-inject lookup is skipped."""
    store = _field_report_store()
    items = list(store.asearch.return_value)
    items.append(_make_search_item("add_automation", score=0.7, api_id="hga_local"))
    store.asearch = AsyncMock(return_value=items)

    result = await _retrieve_tools(
        _field_report_state(), _field_report_config(), store=store
    )

    assert "add_automation" in result["tool_routing_map"]
    add_automation_lookups = [
        call
        for call in store.aget.call_args_list
        if str(call.kwargs.get("key", "")).endswith("::add_automation")
    ]
    assert not add_automation_lookups
    assert (
        sum(
            1
            for t in result["selected_tools"]
            if t["function"]["name"] == "add_automation"
        )
        == 1
    )


@pytest.mark.parametrize(
    "query",
    [
        "tell me when the front door opens",
        "let me know when the washing machine finishes",
        "text me when the kids get home",
        "warn me if the basement gets wet",
    ],
)
def test_query_wants_automation_notification_phrasings(query: str) -> None:
    """Notify-me automation phrasings must be detected."""
    assert _query_wants_automation(query)


@pytest.mark.parametrize(
    "query",
    [
        "when is sunset today",
        "if it rains will the deck stay dry",
    ],
)
def test_query_wants_automation_trigger_without_action(query: str) -> None:
    """A bare trigger word without an action verb must not signal automation."""
    assert not _query_wants_automation(query)


@pytest.mark.parametrize(
    "query",
    [
        "when did the front door last open",
    ],
)
def test_query_wants_automation_accepted_overmatch(query: str) -> None:
    """
    Documented over-match: history questions with trigger+verb bind the tool.

    The injection never evicts RAG selections, but while matched the tool
    stays bound for the human-turn context window (extra store lookup plus
    prompt tokens per turn). If tightening the detector, change deliberately.
    """
    assert _query_wants_automation(query)


@pytest.mark.parametrize(
    "query",
    [
        "check if the garage door is open",
        "tell me if the garage door is open",
    ],
)
def test_query_wants_automation_read_only_open_state_suppressed(query: str) -> None:
    """
    Read-only open-state queries suppress the conditional-actuation signal.

    Step 3b strips actuation tools for these queries; step 3d must not hand
    an actuation-adjacent tool straight back. Explicit markers still win.
    """
    assert not _query_wants_automation(query)


def test_conversation_has_automation_context_survives_tool_heavy_turn() -> None:
    """A tool-heavy intermediate turn must not evict the intent message."""
    intent = HumanMessage(content="create an automation to water the plants")
    tool_heavy_turn = [
        HumanMessage(content="what is the fridge power draw"),
        AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "GetLiveContext",
                    "args": {},
                    "id": "c1",
                    "type": "tool_call",
                }
            ],
        ),
        ToolMessage(content="42W", name="GetLiveContext", tool_call_id="c1"),
        AIMessage(content="It is 42W."),
    ]
    # Intent is the 3rd-most-recent human turn — still in the window,
    # regardless of how many AI/tool messages the middle turn emitted.
    messages = [intent, *tool_heavy_turn, HumanMessage(content="yes")]
    assert _conversation_has_automation_context(messages)
    # One more human exchange pushes intent to the 4th turn — out of window.
    messages = [
        intent,
        *tool_heavy_turn,
        HumanMessage(content="thanks"),
        AIMessage(content="You're welcome."),
        HumanMessage(content="yes"),
    ]
    assert not _conversation_has_automation_context(messages)


def test_conversation_has_automation_context_from_summary() -> None:
    """Trimmed-away intent must still be found in the conversation summary."""
    messages: list[Any] = [HumanMessage(content="yes")]
    assert not _conversation_has_automation_context(messages)
    assert _conversation_has_automation_context(
        messages,
        "The user asked to create an automation turning on the garage light "
        "whenever the door unlocks; the assistant offered notifications.",
    )


def test_conversation_has_automation_context_scans_text_parts_only() -> None:
    """Stringified non-text multimodal parts must not satisfy the detector."""
    image_part = {
        "type": "image_url",
        "image_url": {"url": "https://cam.local/when-motion/if-day.jpg"},
    }
    messages: list[Any] = [
        HumanMessage(
            content=[image_part, {"type": "text", "text": "what is in this picture"}]
        ),
        HumanMessage(content="yes"),
    ]
    assert not _conversation_has_automation_context(messages)
    messages = [
        HumanMessage(
            content=[
                image_part,
                {
                    "type": "text",
                    "text": "always turn on the light when the door opens",
                },
            ]
        ),
        HumanMessage(content="yes"),
    ]
    assert _conversation_has_automation_context(messages)


def test_conversation_has_automation_context_counts_invalid_tool_calls() -> None:
    """A malformed add_automation attempt still establishes context."""
    messages: list[Any] = [
        AIMessage(
            content="",
            invalid_tool_calls=[
                {
                    "name": "add_automation",
                    "args": "not json",
                    "id": "bad-1",
                    "error": "malformed",
                    "type": "invalid_tool_call",
                }
            ],
        ),
        HumanMessage(content="yes"),
    ]
    assert _conversation_has_automation_context(messages)


# ---------------------------------------------------------------------------
# Live-tool filter (issue #554): the retrieval index is global and cumulative,
# so it can hold tools that do not exist for the current request — device-gated
# Assist tools (timer intents) indexed from a satellite turn, or tools of a
# configured API that failed to load. _retrieve_tools must bind only candidates
# present in the loaded ha_llm_api (or local langchain_tools).
# ---------------------------------------------------------------------------


def _query_state(query: str) -> State:
    return {
        "messages": [MagicMock(content=query)],
        "summary": "",
        "chat_model_usage_metadata": {},
        "messages_to_remove": [],
        "selected_tools": [],
        "tool_routing_map": {},
        "action_rounds": 0,
    }


@pytest.mark.asyncio
async def test_retrieve_tools_live_filter_drops_stale_index_candidate() -> None:
    """An index candidate absent from the live tool set must not bind."""
    store = MagicMock()
    store.asearch = AsyncMock(
        return_value=[
            _make_search_item("HassStartTimer", score=0.9),
            _make_search_item("HassCancelAllTimers", score=0.6),
        ]
    )
    store.aget = AsyncMock(return_value=None)

    config: RunnableConfig = {
        "configurable": {
            "options": {"llm_hass_api": ["assist"], "tool_relevance_threshold": 0.15},
            "tool_index_ready": True,
            "langchain_tools": {},
            # Browser-context turn: HA never exposed HassStartTimer.
            "ha_llm_api": _live_llm_api("HassCancelAllTimers"),
        }
    }

    result = await _retrieve_tools(
        _query_state("timer status please"), config, store=store
    )

    assert "HassCancelAllTimers" in result["tool_routing_map"]
    assert "HassStartTimer" not in result["tool_routing_map"]


@pytest.mark.asyncio
async def test_retrieve_tools_live_filter_drops_failed_api_tool() -> None:
    """Tools of a configured-but-failed API (MCP load failure) must not bind."""
    store = MagicMock()
    store.asearch = AsyncMock(
        return_value=[
            _make_search_item("convert_time", score=0.9, api_id="mcp-time"),
            _make_search_item("HassCancelAllTimers", score=0.6),
        ]
    )
    store.aget = AsyncMock(return_value=None)

    config: RunnableConfig = {
        "configurable": {
            # mcp-time is configured (allowed api_id) but failed to load, so it
            # is absent from ha_llm_api.apis.
            "options": {
                "llm_hass_api": ["assist", "mcp-time"],
                "tool_relevance_threshold": 0.15,
            },
            "tool_index_ready": True,
            "langchain_tools": {},
            "ha_llm_api": _live_llm_api("HassCancelAllTimers"),
        }
    }

    result = await _retrieve_tools(
        _query_state("timer status please"), config, store=store
    )

    assert "convert_time" not in result["tool_routing_map"]
    assert "HassCancelAllTimers" in result["tool_routing_map"]


@pytest.mark.asyncio
async def test_retrieve_tools_live_filter_fails_open_without_ha_llm_api() -> None:
    """Without ha_llm_api in config, filtering is skipped entirely."""
    store = MagicMock()
    store.asearch = AsyncMock(
        return_value=[_make_search_item("HassStartTimer", score=0.9)]
    )
    store.aget = AsyncMock(return_value=None)

    config: RunnableConfig = {
        "configurable": {
            "options": {"llm_hass_api": ["assist"], "tool_relevance_threshold": 0.15},
            "tool_index_ready": True,
            "langchain_tools": {},
        }
    }

    result = await _retrieve_tools(
        _query_state("timer status please"), config, store=store
    )

    assert "HassStartTimer" in result["tool_routing_map"]


@pytest.mark.asyncio
async def test_retrieve_tools_live_filter_fails_open_without_apis_attr() -> None:
    """An ha_llm_api object lacking .apis (robot runs) skips filtering."""
    store = MagicMock()
    store.asearch = AsyncMock(
        return_value=[_make_search_item("HassStartTimer", score=0.9)]
    )
    store.aget = AsyncMock(return_value=None)

    config: RunnableConfig = {
        "configurable": {
            "options": {"llm_hass_api": ["assist"], "tool_relevance_threshold": 0.15},
            "tool_index_ready": True,
            "langchain_tools": {},
            "ha_llm_api": SimpleNamespace(),
        }
    }

    result = await _retrieve_tools(
        _query_state("timer status please"), config, store=store
    )

    assert "HassStartTimer" in result["tool_routing_map"]


@pytest.mark.asyncio
async def test_retrieve_tools_live_filter_gates_live_context_injection() -> None:
    """GetLiveContext force-injection is skipped when the tool is not live."""
    store = MagicMock()

    safety_item = _make_search_item("HassTurnOn", score=0.9, is_actuation=True)
    live_ctx_item = MagicMock()
    live_ctx_item.value = {
        "name": "GetLiveContext",
        "api_id": "assist",
        "description": "Get live home state",
        "parameters": "{}",
        "is_actuation": False,
    }

    store.asearch = AsyncMock(return_value=[safety_item])
    store.aget = AsyncMock(return_value=live_ctx_item)

    config: RunnableConfig = {
        "configurable": {
            "options": {"llm_hass_api": ["assist"], "tool_relevance_threshold": 0.15},
            "tool_index_ready": True,
            "langchain_tools": {},
            # GetLiveContext is indexed but not live this turn.
            "ha_llm_api": _live_llm_api("HassTurnOn"),
        }
    }

    result = await _retrieve_tools(
        _query_state("open the garage door"), config, store=store
    )

    assert "HassTurnOn" in result["tool_routing_map"]
    assert "GetLiveContext" not in result["tool_routing_map"]


@pytest.mark.asyncio
async def test_retrieve_tools_pin_tools_bypass_live_filter() -> None:
    """Pending-PIN force-injection is not subject to the live filter."""
    pin_item = MagicMock()
    pin_item.value = {
        "name": "confirm_sensitive_action",
        "api_id": "hga_local",
        "description": "Confirm a sensitive action with a PIN",
        "parameters": "{}",
        "is_actuation": False,
    }

    async def aget(namespace: Any, key: str = "", **_kwargs: Any) -> Any:  # noqa: ARG001
        if key == "hga_local::confirm_sensitive_action":
            return pin_item
        return None

    store = MagicMock()
    store.asearch = AsyncMock(return_value=[])
    store.aget = AsyncMock(side_effect=aget)

    state: State = {
        "messages": [
            ToolMessage(
                content=json.dumps({"status": "requires_pin", "action_id": "act1"}),
                tool_call_id="tc1",
                name="HassLockLock",
            )
        ],
        "summary": "",
        "chat_model_usage_metadata": {},
        "messages_to_remove": [],
        "selected_tools": [],
        "tool_routing_map": {},
        "action_rounds": 0,
    }

    config: RunnableConfig = {
        "configurable": {
            "options": {"llm_hass_api": ["assist"], "tool_relevance_threshold": 0.15},
            "tool_index_ready": True,
            "langchain_tools": {},
            "ha_llm_api": MagicMock(apis={}),
        }
    }

    result = await _retrieve_tools(state, config, store=store)

    assert "confirm_sensitive_action" in result["tool_routing_map"]


# ---------------------------------------------------------------------------
# Behavior: issue #554 field scenario — "Set a timer for two minutes."
# ---------------------------------------------------------------------------


def _timer_field_store() -> MagicMock:
    """Field-log RAG candidates plus HassStartTimer, indexed post top-up."""
    store = MagicMock()
    store.asearch = AsyncMock(
        return_value=[
            _make_search_item("HassStartTimer", score=0.62),
            _make_search_item("HassCancelAllTimers", score=0.618),
            _make_search_item("HassMediaPause", score=0.569, is_actuation=True),
            _make_search_item("HassMediaUnpause", score=0.545, is_actuation=True),
            _make_search_item("convert_time", score=0.534, api_id="mcp-time"),
            _make_search_item("get_current_time", score=0.527, api_id="mcp-time"),
        ]
    )
    store.aget = AsyncMock(return_value=None)
    return store


def _timer_field_config(ha_llm_api: MagicMock) -> RunnableConfig:
    return {
        "configurable": {
            "options": {
                "llm_hass_api": ["assist", "mcp-time"],
                "tool_relevance_threshold": 0.15,
            },
            "tool_index_ready": True,
            "langchain_tools": {},
            "ha_llm_api": ha_llm_api,
        }
    }


@pytest.mark.asyncio
async def test_satellite_timer_query_binds_hass_start_timer() -> None:
    """A timer-capable satellite turn binds HassStartTimer once indexed."""
    # Satellite context: HA exposes the timer intents; mcp-time failed to load.
    ha_llm_api = _live_llm_api(
        "HassStartTimer",
        "HassCancelAllTimers",
        "HassMediaPause",
        "HassMediaUnpause",
    )

    result = await _retrieve_tools(
        _query_state("Set a timer for two minutes."),
        _timer_field_config(ha_llm_api),
        store=_timer_field_store(),
    )

    selected_names = [t["function"]["name"] for t in result["selected_tools"]]
    assert "HassStartTimer" in selected_names
    assert "HassCancelAllTimers" in selected_names
    # Failed-API tools must not bind (their calls die at dispatch).
    assert "convert_time" not in selected_names
    assert "get_current_time" not in selected_names


@pytest.mark.asyncio
async def test_browser_context_excludes_timer_tools() -> None:
    """A device-less turn must not bind timer tools another device indexed."""
    # Browser context (device_id=None): HA exposes only HassCancelAllTimers.
    ha_llm_api = _live_llm_api(
        "HassCancelAllTimers",
        "HassMediaPause",
        "HassMediaUnpause",
    )

    result = await _retrieve_tools(
        _query_state("Set a timer for two minutes."),
        _timer_field_config(ha_llm_api),
        store=_timer_field_store(),
    )

    selected_names = [t["function"]["name"] for t in result["selected_tools"]]
    assert "HassStartTimer" not in selected_names
    assert "HassCancelAllTimers" in selected_names


@pytest.mark.asyncio
async def test_retrieve_tools_live_filter_drops_stale_safety_candidate() -> None:
    """A non-live actuation candidate must not survive via the safety net."""
    query = "turn on the kitchen lights"
    assert _query_needs_actuation_safety(query)

    store = MagicMock()
    # Both the RAG call and the safety-net call see the same candidates; if
    # either leg skipped the live filter, the stale actuation tool would bind
    # (safety tools take merge priority), so the assertion pins both legs.
    store.asearch = AsyncMock(
        return_value=[
            _make_search_item("HassTurnOn", score=0.9, is_actuation=True),
            _make_search_item("HassVacuumStart", score=0.8, is_actuation=True),
        ]
    )
    store.aget = AsyncMock(return_value=None)

    config: RunnableConfig = {
        "configurable": {
            "options": {"llm_hass_api": ["assist"], "tool_relevance_threshold": 0.15},
            "tool_index_ready": True,
            "langchain_tools": {},
            # HassVacuumStart was indexed once but is not exposed this turn.
            "ha_llm_api": _live_llm_api("HassTurnOn"),
        }
    }

    result = await _retrieve_tools(_query_state(query), config, store=store)

    assert "HassTurnOn" in result["tool_routing_map"]
    assert "HassVacuumStart" not in result["tool_routing_map"]


@pytest.mark.asyncio
async def test_retrieve_tools_read_only_query_gates_live_context_injection() -> None:
    """The read-only open-state injection (3b) also honors the live filter."""
    live_ctx_item = MagicMock()
    live_ctx_item.value = {
        "name": "GetLiveContext",
        "api_id": "assist",
        "description": "Get live home state",
        "parameters": "{}",
        "is_actuation": False,
    }

    store = MagicMock()
    store.asearch = AsyncMock(
        return_value=[_make_search_item("get_entity_history", score=0.7)]
    )
    # The index still holds GetLiveContext, but it is not live this turn.
    store.aget = AsyncMock(return_value=live_ctx_item)

    config: RunnableConfig = {
        "configurable": {
            "options": {"llm_hass_api": ["assist"], "tool_relevance_threshold": 0.15},
            "tool_index_ready": True,
            "langchain_tools": {},
            "ha_llm_api": _live_llm_api("get_entity_history"),
        }
    }

    result = await _retrieve_tools(
        _query_state("list the open doors in my house"), config, store=store
    )

    assert "get_entity_history" in result["tool_routing_map"]
    assert "GetLiveContext" not in result["tool_routing_map"]


@pytest.mark.asyncio
async def test_retrieve_tools_all_candidates_stale_falls_back_to_live_tools(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Filter empties RAG/safety: keyword fallback binds only live tools."""
    store = MagicMock()
    # Every retrieved candidate is stale (not live this turn).
    store.asearch = AsyncMock(
        return_value=[_make_search_item("HassStartTimer", score=0.9)]
    )
    store.aget = AsyncMock(return_value=None)

    config: RunnableConfig = {
        "configurable": {
            "options": {"llm_hass_api": ["assist"], "tool_relevance_threshold": 0.15},
            "tool_index_ready": True,
            "langchain_tools": {},
            "ha_llm_api": _live_llm_api("HassCancelAllTimers"),
        }
    }

    result = await _retrieve_tools(
        _query_state("timer status please"), config, store=store
    )

    # The stale tool cannot re-enter via the fallback (it is live-derived);
    # live tools still bind instead of the turn ending with zero tools.
    assert "HassStartTimer" not in result["tool_routing_map"]
    assert "HassCancelAllTimers" in result["tool_routing_map"]
    # The fallback reason names the live filter, not a vector-search miss.
    assert "live-tool filter dropped 1 candidate(s)" in caplog.text


@pytest.mark.asyncio
async def test_retrieve_tools_fail_open_without_langchain_tools_key() -> None:
    """
    An absent langchain_tools key fails open, like an absent ha_llm_api.

    A caller wiring ha_llm_api but not langchain_tools must not silently lose
    every hga_local tool to an empty live set — filtering is skipped entirely.
    """
    store = MagicMock()
    store.asearch = AsyncMock(
        return_value=[
            _make_search_item("get_entity_history", score=0.9, api_id="hga_local")
        ]
    )
    store.aget = AsyncMock(return_value=None)

    config: RunnableConfig = {
        "configurable": {
            "options": {"llm_hass_api": ["assist"], "tool_relevance_threshold": 0.15},
            "tool_index_ready": True,
            "ha_llm_api": _live_llm_api("HassCancelAllTimers"),
        }
    }

    result = await _retrieve_tools(
        _query_state("history of the kitchen sensor"), config, store=store
    )

    assert "get_entity_history" in result["tool_routing_map"]


@pytest.mark.asyncio
async def test_pin_injection_rejects_spoofed_store_row() -> None:
    """
    A stored row failing name/api_id validation must not enter the PIN flow.

    A colliding composite key (or an API registering as hga_local) could
    otherwise swap the confirmation tool's schema/description as seen by the
    model during the security-critical PIN flow.
    """
    spoofed_item = MagicMock()
    spoofed_item.value = {
        "name": "evil_tool",
        "api_id": "assist",
        "description": "not the confirmation tool",
        "parameters": "{}",
        "is_actuation": False,
    }

    async def aget(namespace: Any, key: str = "", **_kwargs: Any) -> Any:  # noqa: ARG001
        if key == "hga_local::confirm_sensitive_action":
            return spoofed_item
        return None

    store = MagicMock()
    store.asearch = AsyncMock(return_value=[])
    store.aget = AsyncMock(side_effect=aget)

    state: State = {
        "messages": [
            ToolMessage(
                content=json.dumps({"status": "requires_pin", "action_id": "act1"}),
                tool_call_id="tc1",
                name="HassLockLock",
            )
        ],
        "summary": "",
        "chat_model_usage_metadata": {},
        "messages_to_remove": [],
        "selected_tools": [],
        "tool_routing_map": {},
        "action_rounds": 0,
    }

    config: RunnableConfig = {
        "configurable": {
            "options": {"llm_hass_api": ["assist"], "tool_relevance_threshold": 0.15},
            "tool_index_ready": True,
            "langchain_tools": {},
            "ha_llm_api": MagicMock(apis={}),
        }
    }

    result = await _retrieve_tools(state, config, store=store)

    assert "evil_tool" not in result["tool_routing_map"]
    assert "confirm_sensitive_action" not in result["tool_routing_map"]


def _open_state_retrieval_config(*live_tools: str, excluded: list[str]) -> Any:
    """Build a retrieval config for the GetLiveContext force-injection path."""
    options: dict[str, Any] = {
        "llm_hass_api": ["assist"],
        "tool_relevance_threshold": 0.15,
    }
    if excluded:
        options["tool_exclusions"] = {"assist": excluded}
    return {
        "configurable": {
            "options": options,
            "tool_index_ready": True,
            "langchain_tools": {},
            "ha_llm_api": _live_llm_api(*live_tools),
        }
    }


def _store_holding_live_context() -> MagicMock:
    """Build a store whose index still contains GetLiveContext."""
    store = MagicMock()
    live_ctx_item = MagicMock()
    live_ctx_item.value = {
        "name": "GetLiveContext",
        "api_id": "assist",
        "description": "Get live home state",
        "parameters": "{}",
        "is_actuation": False,
    }
    store.asearch = AsyncMock(return_value=[])
    # Exclusion never evicts index rows — it removes the tool from the LIVE
    # set — so aget() still finds it and only the live check can keep it out.
    store.aget = AsyncMock(return_value=live_ctx_item)
    return store


@pytest.mark.asyncio
async def test_live_context_is_force_injected_when_not_excluded() -> None:
    """
    Control case: the by-name force-inject path really does reach GetLiveContext.

    Without this half, the exclusion test below would pass on an empty routing
    map and prove nothing.
    """
    state: State = {
        "messages": [HumanMessage(content="is the garage door open")],
        "summary": "",
        "chat_model_usage_metadata": {},
        "messages_to_remove": [],
        "selected_tools": [],
        "tool_routing_map": {},
    }
    config: RunnableConfig = _open_state_retrieval_config("GetLiveContext", excluded=[])

    result = await _retrieve_tools(state, config, store=_store_holding_live_context())

    assert "GetLiveContext" in result["tool_routing_map"]


@pytest.mark.asyncio
async def test_excluded_tool_is_not_force_injected_by_name() -> None:
    """
    An excluded tool stays out even of the by-name force-inject path.

    This pins the feature's headline claim ("never retrieved", issue #570) at
    the retrieval layer rather than at the filter that implements it.
    `GetLiveContext` is injected BY NAME outside the retrieval limit, so it is
    the one route that could resurrect a tool the user excluded. It is safe
    only because `_get_tool_by_name` is handed `live_tool_ids` and its
    `_get_fallback_tools` fallback reads the already-filtered `ha_llm_api`;
    a future change passing `live_tool_ids=None` there would silently re-expose
    every excluded tool with an otherwise-green suite.
    """
    state: State = {
        "messages": [HumanMessage(content="is the garage door open")],
        "summary": "",
        "chat_model_usage_metadata": {},
        "messages_to_remove": [],
        "selected_tools": [],
        "tool_routing_map": {},
    }
    # GetLiveContext excluded -> filter_excluded_tools already removed it from
    # the loaded instance, so it is absent from the live set for this turn.
    config: RunnableConfig = _open_state_retrieval_config(excluded=["GetLiveContext"])

    result = await _retrieve_tools(state, config, store=_store_holding_live_context())

    assert "GetLiveContext" not in result["tool_routing_map"]


def _indexed_item(name: str, api_id: str = "assist") -> MagicMock:
    """Build a scored vector-search hit for an indexed tool."""
    item = MagicMock()
    item.value = {
        "name": name,
        "api_id": api_id,
        "description": f"{name} description",
        "parameters": "{}",
        "is_actuation": False,
    }
    item.score = 0.9
    return item


@pytest.mark.asyncio
async def test_excluded_tool_in_the_index_is_not_bound_via_rag() -> None:
    """
    An excluded tool that is still INDEXED must not bind through RAG.

    Exclusion deliberately does not evict index rows: `_async_discover_provider_tools`
    indexes every registered API unfiltered, so an excluded tool keeps its
    embedding and can still be returned by vector search. Enforcement lives one
    layer later, at bind time, where `_filter_live_candidates` drops anything
    absent from the filtered `ha_llm_api`. This pins that boundary — without
    it, "excluded tools stay in the index" and "excluded tools never bind" are
    two claims with nothing proving they coexist.
    """
    store = MagicMock()
    store.asearch = AsyncMock(return_value=[_indexed_item("web_search")])

    state: State = {
        "messages": [HumanMessage(content="search the web for a recipe")],
        "summary": "",
        "chat_model_usage_metadata": {},
        "messages_to_remove": [],
        "selected_tools": [],
        "tool_routing_map": {},
    }
    config: RunnableConfig = {
        "configurable": {
            "options": {
                "llm_hass_api": ["assist"],
                "tool_relevance_threshold": 0.15,
                "tool_exclusions": {"assist": ["web_search"]},
            },
            "tool_index_ready": True,
            "langchain_tools": {},
            # filter_excluded_tools already removed web_search from the loaded
            # instance, so it is absent from the live set even though the index
            # still returns it above.
            "ha_llm_api": _live_llm_api("HassTurnOn"),
        }
    }

    result = await _retrieve_tools(state, config, store=store)

    assert "web_search" not in result["tool_routing_map"]


@pytest.mark.asyncio
async def test_indexed_tool_binds_via_rag_when_not_excluded() -> None:
    """Control for the test above: the same setup binds when nothing is excluded."""
    store = MagicMock()
    store.asearch = AsyncMock(return_value=[_indexed_item("web_search")])

    state: State = {
        "messages": [HumanMessage(content="search the web for a recipe")],
        "summary": "",
        "chat_model_usage_metadata": {},
        "messages_to_remove": [],
        "selected_tools": [],
        "tool_routing_map": {},
    }
    config: RunnableConfig = {
        "configurable": {
            "options": {
                "llm_hass_api": ["assist"],
                "tool_relevance_threshold": 0.15,
            },
            "tool_index_ready": True,
            "langchain_tools": {},
            "ha_llm_api": _live_llm_api("HassTurnOn", "web_search"),
        }
    }

    result = await _retrieve_tools(state, config, store=store)

    assert "web_search" in result["tool_routing_map"]


@pytest.mark.asyncio
async def test_excluded_tool_does_not_consume_a_retrieval_slot(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """
    Excluding a tool must not change which OTHER tools reach the model.

    The index is never pruned when a user excludes a tool, so an excluded tool
    can still be the top-scoring hit — and `sorted_items[:limit]` cuts to the
    retrieval limit BEFORE `_filter_live_candidates` runs. Filtering late let
    the excluded tool eat a slot and then vanish, so switching one tool off
    silently reduced the tools available for unrelated queries. With limit=1
    that is the difference between getting the runner-up and getting nothing.
    """
    store = MagicMock()
    top = _indexed_item("web_search")
    top.score = 0.99
    runner_up = _indexed_item("HassTurnOn")
    runner_up.score = 0.5
    store.asearch = AsyncMock(return_value=[top, runner_up])
    store.aget = AsyncMock(return_value=None)

    state: State = {
        "messages": [HumanMessage(content="search the web for a recipe")],
        "summary": "",
        "chat_model_usage_metadata": {},
        "messages_to_remove": [],
        "selected_tools": [],
        "tool_routing_map": {},
    }
    config: RunnableConfig = {
        "configurable": {
            "options": {
                "llm_hass_api": ["assist"],
                "tool_relevance_threshold": 0.15,
                "tool_retrieval_limit": 1,
                "tool_exclusions": {"assist": ["web_search"]},
            },
            "tool_index_ready": True,
            "langchain_tools": {},
            "ha_llm_api": _live_llm_api("HassTurnOn"),
        }
    }

    with caplog.at_level("WARNING"):
        result = await _retrieve_tools(state, config, store=store)

    assert "web_search" not in result["tool_routing_map"]
    # The runner-up gets the slot the excluded tool used to waste.
    assert "HassTurnOn" in result["tool_routing_map"]
    # And it got there through RAG, not by the whole pass collapsing into the
    # keyword fallback -- which is the half that actually pins the fix. The
    # fallback would bind HassTurnOn too, from a differently-shaped tool set,
    # so asserting only on the routing map would pass without the fix.
    assert "keyword-filtered fallback" not in caplog.text

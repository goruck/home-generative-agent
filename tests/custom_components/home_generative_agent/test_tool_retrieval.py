# ruff: noqa: S101
"""Unit tests for tool retrieval logic (RAG, safety net, fallbacks)."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING
from unittest.mock import AsyncMock, MagicMock

import psycopg
import pytest
from homeassistant.helpers import llm

from custom_components.home_generative_agent.agent.graph import (
    State,
    _get_actuation_safety_tools,
    _get_allowed_api_ids,
    _get_rag_retrieved_tools,
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

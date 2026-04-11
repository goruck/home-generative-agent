# ruff: noqa: S101
"""Extended unit tests for tool retrieval logic (RAG, safety net, dedupe)."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING
from unittest.mock import AsyncMock, MagicMock

import pytest

from custom_components.home_generative_agent.agent.graph import (
    State,
    _get_allowed_api_ids,
    _retrieve_tools,
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
async def test_retrieve_tools_deduplication_first_seen_wins() -> None:
    """Test that the first tool encountered (RAG) wins during deduplication."""
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

    # Safety Net returns HassTurnOn from assist (default version)
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
    # RAG version (hga_local) should win because it was first in the candidates list
    assert result["tool_routing_map"]["HassTurnOn"] == "hga_local"
    assert len(result["selected_tools"]) == 1
    assert result["selected_tools"][0]["function"]["description"] == "Custom Turn On"

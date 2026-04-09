# ruff: noqa: S101
"""Unit tests for tool retrieval RAG fallback logic."""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import AsyncMock, MagicMock

import pytest
from homeassistant.helpers import llm

from custom_components.home_generative_agent.agent.graph import _retrieve_tools

if TYPE_CHECKING:
    from langchain_core.runnables import RunnableConfig


@pytest.mark.asyncio
async def test_retrieve_tools_fallback_on_empty_store() -> None:
    """Test that _retrieve_tools falls back to all tools if store search returns nothing."""
    state = {"messages": [MagicMock(content="turn on the lights")]}
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
    assert result["selected_tools"][1]["function"]["name"] == "local_tool"
    assert result["selected_tools"][0]["function"]["name"] == "HassTurnOn"


@pytest.mark.asyncio
async def test_retrieve_tools_fallback_on_store_error() -> None:
    """Test that _retrieve_tools falls back to all tools if store search raises an error."""
    state = {"messages": [MagicMock(content="turn on the lights")]}
    store = MagicMock()
    # Store raises connection error
    store.asearch = AsyncMock(side_effect=Exception("Connection refused"))

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
            "tool_index_ready": True,
            "langchain_tools": {},
            "ha_llm_api": ha_llm_api,
        }
    }

    result = await _retrieve_tools(state, config, store=store)

    assert "HassTurnOn" in result["tool_routing_map"]
    assert len(result["selected_tools"]) == 1


@pytest.mark.asyncio
async def test_retrieve_tools_fallback_on_index_not_ready() -> None:
    """Test that _retrieve_tools falls back to all tools if index is not ready."""
    state = {"messages": [MagicMock(content="turn on the lights")]}
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

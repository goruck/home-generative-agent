"""Tests for MultiLLMAPI wrapper."""

from unittest.mock import AsyncMock

import pytest
import voluptuous as vol
from homeassistant.exceptions import HomeAssistantError
from homeassistant.helpers import llm

from custom_components.home_generative_agent.conversation import MultiLLMAPI


@pytest.fixture
def mock_api_1():
    """Create a mock home assistant LLM API."""
    api = AsyncMock(spec=llm.APIInstance)
    api.api_prompt = "Prompt 1"
    api.custom_serializer = lambda x: x
    return api


@pytest.fixture
def mock_api_2():
    """Create a second mock home assistant LLM API."""
    api = AsyncMock(spec=llm.APIInstance)
    api.api_prompt = "Prompt 2"
    api.custom_serializer = None
    return api


@pytest.mark.asyncio
async def test_multi_llm_api_properties(mock_api_1, mock_api_2):
    """Test properties of MultiLLMAPI."""
    multi_api = MultiLLMAPI([mock_api_1, mock_api_2])

    assert multi_api.api_prompt == "Prompt 1\nPrompt 2"
    assert multi_api.custom_serializer == mock_api_1.custom_serializer


@pytest.mark.asyncio
async def test_multi_llm_api_success_first(mock_api_1, mock_api_2):
    """Test successful tool call on the first API."""
    multi_api = MultiLLMAPI([mock_api_1, mock_api_2])
    tool_input = llm.ToolInput(tool_name="test_tool", tool_args={})

    mock_api_1.async_call_tool.return_value = "success"

    result = await multi_api.async_call_tool(tool_input)

    assert result == "success"
    mock_api_1.async_call_tool.assert_awaited_once_with(tool_input)
    mock_api_2.async_call_tool.assert_not_awaited()


@pytest.mark.asyncio
async def test_multi_llm_api_fallback_to_second(mock_api_1, mock_api_2):
    """Test tool call fails on first API and falls back to second."""
    multi_api = MultiLLMAPI([mock_api_1, mock_api_2])
    tool_input = llm.ToolInput(tool_name="test_tool", tool_args={})

    # First API raises an error
    mock_api_1.async_call_tool.side_effect = vol.Invalid("Not supported")
    mock_api_2.async_call_tool.return_value = "success_on_2"

    result = await multi_api.async_call_tool(tool_input)

    assert result == "success_on_2"
    mock_api_1.async_call_tool.assert_awaited_once_with(tool_input)
    mock_api_2.async_call_tool.assert_awaited_once_with(tool_input)


@pytest.mark.asyncio
async def test_multi_llm_api_fails_all(mock_api_1, mock_api_2):
    """Test tool call fails on all APIs and raises the last error."""
    multi_api = MultiLLMAPI([mock_api_1, mock_api_2])
    tool_input = llm.ToolInput(tool_name="test_tool", tool_args={})

    mock_api_1.async_call_tool.side_effect = vol.Invalid("Not supported 1")
    expected_err = HomeAssistantError("Not supported 2")
    mock_api_2.async_call_tool.side_effect = expected_err

    with pytest.raises(HomeAssistantError, match="Not supported 2"):
        await multi_api.async_call_tool(tool_input)

    mock_api_1.async_call_tool.assert_awaited_once_with(tool_input)
    mock_api_2.async_call_tool.assert_awaited_once_with(tool_input)


@pytest.mark.asyncio
async def test_multi_llm_api_empty_list():
    """Test calling tool with no APIs."""
    multi_api = MultiLLMAPI([])
    tool_input = llm.ToolInput(tool_name="test_tool", tool_args={})

    with pytest.raises(
        HomeAssistantError, match="No APIs available to handle tool test_tool"
    ):
        await multi_api.async_call_tool(tool_input)

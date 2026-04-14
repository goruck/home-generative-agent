# ruff: noqa: S101
"""
Unit tests for conversation.py helpers (MultiLLMAPI, _run_tool_index_background).

hassil is not installed in the test venv, so this module stubs the entire
homeassistant.components.conversation import chain before importing conversation.py.
"""

from __future__ import annotations

import sys
import types
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from homeassistant.exceptions import HomeAssistantError


def _stub_ha_conversation() -> None:
    """
    Stub homeassistant.components.conversation before conversation.py loads it.

    hassil and home_assistant_intents are not installed in the test venv.
    We mock the HA conversation module with just enough surface area for
    conversation.py to import cleanly and for MultiLLMAPI / _run_tool_index_background
    to be accessible.
    """
    if "homeassistant.components.conversation" in sys.modules:
        return

    # Build real (empty) base classes so class inheritance works.
    class _ConversationEntity:
        pass

    class _AbstractConversationAgent:
        pass

    class _ConversationResult:
        pass

    class _UserContent:
        pass

    class _AssistantContent:
        pass

    # conversation module
    conv_mod = types.ModuleType("homeassistant.components.conversation")
    conv_mod.ConversationEntity = _ConversationEntity  # type: ignore[attr-defined]
    conv_mod.AbstractConversationAgent = _AbstractConversationAgent  # type: ignore[attr-defined]
    conv_mod.ConversationResult = _ConversationResult  # type: ignore[attr-defined]
    conv_mod.UserContent = _UserContent  # type: ignore[attr-defined]
    conv_mod.AssistantContent = _AssistantContent  # type: ignore[attr-defined]
    conv_mod.DOMAIN = "conversation"  # type: ignore[attr-defined]
    conv_mod.async_set_agent = MagicMock()  # type: ignore[attr-defined]
    conv_mod.trace = MagicMock()  # type: ignore[attr-defined]

    # conversation.models submodule
    models_mod = types.ModuleType("homeassistant.components.conversation.models")
    models_mod.AbstractConversationAgent = _AbstractConversationAgent  # type: ignore[attr-defined]
    conv_mod.models = models_mod  # type: ignore[attr-defined]

    sys.modules["homeassistant.components.conversation"] = conv_mod
    sys.modules["homeassistant.components.conversation.models"] = models_mod


_stub_ha_conversation()

# These imports must come AFTER the stub so conversation.py loads cleanly.
from custom_components.home_generative_agent.conversation import (  # noqa: E402
    MultiLLMAPI,
    _run_tool_index_background,
)

# ---------------------------------------------------------------------------
# MultiLLMAPI: empty routing_map fallback
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_multi_llm_api_empty_routing_map_iterates_apis() -> None:
    """With no routing entry, async_call_tool falls back to iterating all APIs."""
    api1 = MagicMock()
    api1.async_call_tool = AsyncMock(side_effect=HomeAssistantError("not mine"))
    api2 = MagicMock()
    api2.async_call_tool = AsyncMock(return_value={"result": "ok"})

    multi = MultiLLMAPI({"api1": api1, "api2": api2}, routing_map={})

    tool_input = MagicMock()
    tool_input.tool_name = "mystery_tool"

    result = await multi.async_call_tool(tool_input)

    assert result == {"result": "ok"}
    api1.async_call_tool.assert_called_once_with(tool_input)
    api2.async_call_tool.assert_called_once_with(tool_input)


@pytest.mark.asyncio
async def test_multi_llm_api_empty_routing_map_all_fail_raises() -> None:
    """With no routing entry and all APIs failing, HomeAssistantError is raised."""
    api1 = MagicMock()
    api1.async_call_tool = AsyncMock(side_effect=HomeAssistantError("nope"))

    multi = MultiLLMAPI({"api1": api1}, routing_map={})

    tool_input = MagicMock()
    tool_input.tool_name = "mystery_tool"

    with pytest.raises(HomeAssistantError, match="No routing target"):
        await multi.async_call_tool(tool_input)


@pytest.mark.asyncio
async def test_multi_llm_api_routes_to_correct_api() -> None:
    """With a populated routing_map, calls go directly to the mapped API."""
    api1 = MagicMock()
    api1.async_call_tool = AsyncMock(return_value="from_api1")
    api2 = MagicMock()
    api2.async_call_tool = AsyncMock(return_value="from_api2")

    multi = MultiLLMAPI(
        {"api1": api1, "api2": api2},
        routing_map={"tool_a": "api2"},
    )

    tool_input = MagicMock()
    tool_input.tool_name = "tool_a"

    result = await multi.async_call_tool(tool_input)

    assert result == "from_api2"
    api1.async_call_tool.assert_not_called()
    api2.async_call_tool.assert_called_once_with(tool_input)


# ---------------------------------------------------------------------------
# _run_tool_index_background failure path
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_run_tool_index_background_failure_sets_flag() -> None:
    """Indexing failure sets tool_index_failed=True and resets tool_indexing_in_progress."""
    rd = MagicMock()
    rd.tool_index_ready = False
    rd.tool_indexing_in_progress = True
    rd.tool_index_failed = False
    rd.tool_content_hashes = {}

    hass = MagicMock()

    with patch(
        "custom_components.home_generative_agent.conversation.gather_store_puts_in_chunks",
        new=AsyncMock(side_effect=RuntimeError("embedding provider down")),
    ):
        await _run_tool_index_background(
            index_tasks=[AsyncMock()],
            tool_hashes={"key": "hash"},
            rd=rd,
            hass=hass,
        )

    assert rd.tool_index_failed is True
    assert rd.tool_index_ready is False
    assert rd.tool_indexing_in_progress is False


@pytest.mark.asyncio
async def test_run_tool_index_background_success_clears_flags() -> None:
    """Successful indexing sets tool_index_ready=True and resets in_progress."""
    rd = MagicMock()
    rd.tool_index_ready = False
    rd.tool_indexing_in_progress = True
    rd.tool_index_failed = False
    rd.tool_content_hashes = {}

    hass = MagicMock()

    with patch(
        "custom_components.home_generative_agent.conversation.gather_store_puts_in_chunks",
        new=AsyncMock(return_value=None),
    ):
        await _run_tool_index_background(
            index_tasks=[AsyncMock()],
            tool_hashes={"key": "hash"},
            rd=rd,
            hass=hass,
        )

    assert rd.tool_index_ready is True
    assert rd.tool_index_failed is False
    assert rd.tool_indexing_in_progress is False
    assert rd.tool_content_hashes == {"key": "hash"}

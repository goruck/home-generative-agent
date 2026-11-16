# ruff: noqa: S101
"""
Our tools' plain-text refusals reach the graph with the error status (#1023).

LangChain gives any string a tool returns the status "success", so "Camera 'X'
was not found" or "Not stored" read as done, and an empty model reply after one
said "Done.". Each refusal is a `ToolFailure`, which `_run_langchain_tool`
turns into an error result. These tests run the real tools through it.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, cast
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from homeassistant.exceptions import HomeAssistantError
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.runnables.config import var_child_runnable_config
from langgraph.store.memory import InMemoryStore

import custom_components.home_generative_agent as hga_component
import custom_components.home_generative_agent.agent.graph as agent_graph
from custom_components.home_generative_agent.agent import tools as hga_tools
from custom_components.home_generative_agent.agent.tool_failure import ToolFailure
from custom_components.home_generative_agent.const import (
    CONF_MEMORY_MAX_PER_PERSON,
    NO_REPLY_APOLOGY,
)

if TYPE_CHECKING:
    from homeassistant.core import HomeAssistant
    from langchain_core.runnables import RunnableConfig

_TOOLS = {
    name: getattr(hga_tools, name)
    for name in (
        "get_and_analyze_camera_image",
        "upsert_memory",
        "list_memories",
        "forget_memory",
        "confirm_sensitive_action",
        "alarm_control",
        "audit_home_security",
    )
}


async def _run(
    name: str, args: dict[str, Any], config: dict[str, Any], store: Any = None
) -> ToolMessage:
    call = {"name": name, "args": args, "id": "c1", "type": "tool_call"}
    run: Any = agent_graph._run_langchain_tool
    # A tool's injected config is the run's, as inside a graph node.
    token = var_child_runnable_config.set(cast("RunnableConfig", config))
    try:
        return await run(name, call, _TOOLS, store or InMemoryStore(), config)
    finally:
        var_child_runnable_config.reset(token)


def _configurable(hass: HomeAssistant, **extra: Any) -> dict[str, Any]:
    return {
        "configurable": {
            "hass": hass,
            "user_id": "Anna",
            "thread_id": "t1",
            "options": {},
            "pending_actions": {},
            "vlm_model": MagicMock(),
            **extra,
        }
    }


@pytest.mark.asyncio
async def test_an_unknown_camera_is_an_error(hass: HomeAssistant) -> None:
    """The model is told the camera was not found, as an error."""
    result = await _run(
        "get_and_analyze_camera_image",
        {"camera_name": "porch"},
        _configurable(hass),
    )

    assert result.status == "error"
    assert "was not found" in str(result.content)
    # The checkpoint stores a plain string, not our subclass.
    assert type(result.content) is str


@pytest.mark.asyncio
async def test_a_camera_without_an_image_is_an_error(hass: HomeAssistant) -> None:
    """No frame, nothing analyzed."""
    with (
        patch.object(hga_tools, "_resolve_camera_entity_id", return_value="camera.x"),
        patch.object(hga_tools, "_get_camera_image", AsyncMock(return_value=None)),
    ):
        result = await _run(
            "get_and_analyze_camera_image",
            {"camera_name": "x"},
            _configurable(hass),
        )

    assert result.status == "error"
    assert result.content == "Error getting image from camera."


@pytest.mark.asyncio
async def test_a_failed_image_analysis_is_an_error(hass: HomeAssistant) -> None:
    """A timed-out analysis returns the error caption, as an error."""
    with (
        patch.object(hga_tools, "_resolve_camera_entity_id", return_value="camera.x"),
        patch.object(hga_tools, "_get_camera_image", AsyncMock(return_value=b"jpg")),
        patch.object(hga_tools, "analyze_image", AsyncMock(side_effect=TimeoutError)),
    ):
        result = await _run(
            "get_and_analyze_camera_image",
            {"camera_name": "x"},
            _configurable(hass),
        )

    assert result.status == "error"
    assert result.content == hga_tools.VLM_ERROR_CAPTION


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("name", "args"),
    [
        ("upsert_memory", {"content": "I take my coffee black"}),
        ("list_memories", {}),
        ("forget_memory", {"memory_id": "a"}),
    ],
)
async def test_memory_tools_without_a_database_are_errors(
    hass: HomeAssistant, name: str, args: dict[str, Any]
) -> None:
    """With no Database subentry nothing is kept, so nothing reads as done."""
    result = await _run(name, args, _configurable(hass), hga_component.NullStore())

    assert result.status == "error"
    assert "no database is set up" in str(result.content)


@pytest.mark.asyncio
async def test_memory_refusals_with_a_database_are_errors(hass: HomeAssistant) -> None:
    """A full memory and an unknown id are refusals; a save is not."""
    store = InMemoryStore()
    config = _configurable(hass, options={CONF_MEMORY_MAX_PER_PERSON: 1})

    saved = await _run("upsert_memory", {"content": "first"}, config, store)
    full = await _run("upsert_memory", {"content": "second"}, config, store)
    unknown = await _run("forget_memory", {"memory_id": "nope"}, config, store)
    listed = await _run("list_memories", {}, config, store)

    assert saved.status == "success"
    assert full.status == "error"
    assert str(full.content).startswith("Not stored")
    assert unknown.status == "error"
    assert listed.status == "success"


@pytest.mark.asyncio
async def test_an_unknown_pin_action_is_an_error(hass: HomeAssistant) -> None:
    """A PIN for an action that is gone did not run anything."""
    result = await _run(
        "confirm_sensitive_action",
        {"action_id": "gone", "pin": "1234"},
        _configurable(hass),
    )

    assert result.status == "error"


@pytest.mark.asyncio
async def test_an_alarm_service_error_is_an_error(hass: HomeAssistant) -> None:
    """The alarm's plain-text failure keeps its error status past the JSON check."""
    with patch.object(
        hga_tools,
        "_perform_alarm_control",
        AsyncMock(side_effect=HomeAssistantError("panel offline")),
    ):
        result = await _run(
            "alarm_control",
            {"name": "Alarm", "state": "arm_away"},
            _configurable(hass),
        )

    assert result.status == "error"
    assert result.content == "Error controlling alarm: panel offline"


@pytest.mark.asyncio
async def test_an_audit_without_sentinel_is_an_error() -> None:
    """No Sentinel, no audit."""
    config = {"configurable": {"hga_runtime_data": SimpleNamespace(sentinel=None)}}

    result = await _run("audit_home_security", {}, config)

    assert result.status == "error"
    assert "Sentinel is not enabled" in str(result.content)


@pytest.mark.asyncio
async def test_an_audit_that_could_not_run_is_an_error() -> None:
    """The engine's 'could not run' answer is an error."""
    sentinel = MagicMock()
    sentinel.async_audit_network = AsyncMock(
        return_value={"status": "unavailable", "notes": ["no snapshot"]}
    )
    config = {"configurable": {"hga_runtime_data": SimpleNamespace(sentinel=sentinel)}}

    result = await _run("audit_home_security", {}, config)

    assert result.status == "error"
    assert "could not run" in str(result.content)


def test_a_tool_failure_is_still_its_text() -> None:
    """Callers comparing the text see the same words."""
    failure = ToolFailure("Not stored.")

    assert failure == "Not stored."
    assert failure.startswith("Not")
    assert isinstance(failure, str)


@pytest.mark.asyncio
async def test_an_empty_reply_after_a_memory_refusal_apologizes(
    hass: HomeAssistant, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The field case: "remember this" with no database, then no model text."""
    refusal = await _run(
        "upsert_memory",
        {"content": "I take my coffee black"},
        _configurable(hass),
        hga_component.NullStore(),
    )

    async def _empty(*_args: object, **_kwargs: object) -> AIMessage:
        return AIMessage(content="", tool_calls=[])

    async def _same(messages: list[object], *_a: object, **_k: object) -> list[object]:
        return messages

    monkeypatch.setattr(agent_graph, "_invoke_model", _empty)
    monkeypatch.setattr(agent_graph, "_trim_messages_for_model", _same)
    store = MagicMock()
    store.asearch = AsyncMock(return_value=[])
    call = {"name": "upsert_memory", "args": {}, "id": "c1", "type": "tool_call"}
    state = {
        "messages": [
            HumanMessage(content="remember that I take my coffee black"),
            AIMessage(content="", tool_calls=[call]),
            refusal,
        ],
        "selected_tools": [],
        "summary": "",
        "tool_routing_map": {},
        "messages_to_remove": [],
        "chat_model_usage_metadata": {},
    }
    config = _configurable(
        hass,
        chat_model=MagicMock(),
        user_id="user-test",
        chat_model_options={},
        prompt="You are a helpful assistant.",
        langchain_tools={},
        ha_llm_api=None,
    )
    call_model: Any = agent_graph._call_model

    reply = (await call_model(state, config, store=store))["messages"]

    assert isinstance(reply, AIMessage)
    assert reply.content == NO_REPLY_APOLOGY

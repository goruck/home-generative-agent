# ruff: noqa: S101
"""Tests for keeping the previous turn's tools bound on a short follow-up (#920)."""

from __future__ import annotations

from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, cast
from unittest.mock import AsyncMock, MagicMock

import pytest
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage

from custom_components.home_generative_agent.agent.graph import (
    _ROUTING_REJECTION_MARKER,
    _make_routing_rejection,
    _prior_turn_tool_names,
    _retrieve_tools,
)
from custom_components.home_generative_agent.agent.helpers import tool_index_key

if TYPE_CHECKING:
    from langchain_core.runnables import RunnableConfig

    from custom_components.home_generative_agent.agent.graph import State

_LIGHT_SET = "light__HassLightSet"
_LIVE_CONTEXT = "homeassistant__GetLiveContext"
# The tools live on the box for the field report, less the noise.
_FIELD_TOOLS = (
    "vacuum__HassVacuumStart",
    "media_player__HassMediaUnpause",
    "intent__HassTurnOn",
    "intent__HassTurnOff",
    _LIGHT_SET,
    _LIVE_CONTEXT,
)


def _item(name: str, score: float = 0.5) -> MagicMock:
    item = MagicMock()
    item.value = {
        "name": name,
        "api_id": "assist",
        "description": f"{name} description",
        "parameters": "{}",
        "is_actuation": name != _LIVE_CONTEXT,
    }
    item.score = score
    return item


def _store() -> MagicMock:
    """Rank as the box did for "Turn the back on": the light tool is not a hit."""
    indexed = {tool_index_key("assist", n): _item(n) for n in _FIELD_TOOLS}
    store = MagicMock()
    store.asearch = AsyncMock(
        return_value=[
            _item("vacuum__HassVacuumStart", 0.59),
            _item("media_player__HassMediaUnpause", 0.56),
        ]
    )
    store.aget = AsyncMock(side_effect=lambda _ns, key: indexed.get(key))
    return store


def _config(*, excluded: list[str] | None = None) -> RunnableConfig:
    options: dict[str, Any] = {
        "llm_hass_api": ["assist"],
        "tool_relevance_threshold": 0.15,
        "tool_retrieval_limit": 5,
    }
    if excluded:
        options["tool_exclusions"] = {"assist": excluded}
    api = SimpleNamespace(
        tools=[
            SimpleNamespace(
                name=n, description=n, parameters={"type": "object", "properties": {}}
            )
            for n in _FIELD_TOOLS
        ],
        custom_serializer=None,
    )
    return cast(
        "RunnableConfig",
        {
            "configurable": {
                "options": options,
                "tool_index_ready": True,
                "langchain_tools": {},
                "ha_llm_api": MagicMock(apis={"assist": api}),
            }
        },
    )


def _light_turn(*, status: str = "success") -> list[BaseMessage]:
    """Build the field report's second turn: both porch lights turned off."""
    return [
        HumanMessage(content="Outside Lights"),
        AIMessage(
            content="",
            tool_calls=[{"name": _LIVE_CONTEXT, "args": {}, "id": "c1"}],
        ),
        ToolMessage(content="Live Context: ...", name=_LIVE_CONTEXT, tool_call_id="c1"),
        AIMessage(
            content="",
            tool_calls=[
                {"name": _LIGHT_SET, "args": {"brightness": 0}, "id": "c2"},
                {"name": _LIGHT_SET, "args": {"brightness": 0}, "id": "c3"},
            ],
        ),
        ToolMessage(
            content='{"response_type": "action_done"}',
            name=_LIGHT_SET,
            tool_call_id="c2",
            status=status,
        ),
        ToolMessage(
            content='{"response_type": "action_done"}',
            name=_LIGHT_SET,
            tool_call_id="c3",
            status=status,
        ),
        AIMessage(content="Done, the porch lights are off."),
    ]


def _state(messages: list[BaseMessage]) -> State:
    return cast(
        "State",
        {
            "messages": messages,
            "summary": "",
            "chat_model_usage_metadata": {},
            "messages_to_remove": [],
            "selected_tools": [],
            "tool_routing_map": {},
            "action_rounds": 0,
        },
    )


async def _routed(messages: list[BaseMessage], **config_kw: Any) -> dict[str, str]:
    result = await _retrieve_tools(
        _state(messages), _config(**config_kw), store=_store()
    )
    return result["tool_routing_map"]


@pytest.mark.asyncio
async def test_the_field_report_follow_up_keeps_the_light_tool() -> None:
    """Let 'Turn the back on', after the lights went off, call the same tool."""
    routed = await _routed([*_light_turn(), HumanMessage(content="Turn the back on")])

    assert _LIGHT_SET in routed
    # Appended past the limit: nothing the ranking chose was evicted.
    assert "vacuum__HassVacuumStart" in routed


@pytest.mark.asyncio
async def test_without_the_previous_turn_the_light_tool_is_not_ranked() -> None:
    """Pin the fixture: the bind, not the ranking, is what brings it in."""
    routed = await _routed([HumanMessage(content="Turn the back on")])

    assert _LIGHT_SET not in routed


@pytest.mark.asyncio
async def test_a_refused_call_is_not_carried() -> None:
    """An error result (a rejection or a failure) never binds its tool."""
    routed = await _routed(
        [*_light_turn(status="error"), HumanMessage(content="Turn the back on")]
    )

    assert _LIGHT_SET not in routed


@pytest.mark.asyncio
async def test_a_long_query_ranks_on_its_own_words() -> None:
    """A request that names its own target is not a follow-up."""
    routed = await _routed(
        [
            *_light_turn(),
            HumanMessage(
                content="start the vacuum in the kitchen and then pause the music"
            ),
        ]
    )

    assert _LIGHT_SET not in routed


@pytest.mark.asyncio
async def test_an_excluded_tool_is_not_carried() -> None:
    """The user's per-tool exclusion still wins."""
    routed = await _routed(
        [*_light_turn(), HumanMessage(content="Turn the back on")],
        excluded=[_LIGHT_SET],
    )

    assert _LIGHT_SET not in routed


@pytest.mark.asyncio
async def test_a_tool_no_longer_live_is_not_carried() -> None:
    """A tool the previous turn ran but this turn does not offer stays out."""
    messages = [
        HumanMessage(content="start the timer"),
        AIMessage(
            content="",
            tool_calls=[{"name": "intent__HassStartTimer", "args": {}, "id": "t1"}],
        ),
        ToolMessage(content="ok", name="intent__HassStartTimer", tool_call_id="t1"),
        AIMessage(content="Started."),
        HumanMessage(content="cancel it"),
    ]
    routed = await _routed(messages)

    assert "intent__HassStartTimer" not in routed


def test_only_the_previous_turn_counts() -> None:
    """A tool two turns back is not carried."""
    messages = [
        *_light_turn(),
        HumanMessage(content="Thanks"),
        AIMessage(content="You're welcome."),
        HumanMessage(content="Turn the back on"),
    ]

    assert _prior_turn_tool_names(messages) == []


def test_prior_turn_names_are_in_call_order_and_deduplicated() -> None:
    messages = [*_light_turn(), HumanMessage(content="Turn the back on")]

    assert _prior_turn_tool_names(messages) == [_LIVE_CONTEXT, _LIGHT_SET]


def test_prior_turn_names_need_a_new_user_message() -> None:
    """Mid-turn (the last message is not the user's) there is no previous turn."""
    assert _prior_turn_tool_names(_light_turn()) == []


def test_the_rejection_names_the_bound_tools_and_does_not_ask_for_a_retry() -> None:
    """
    The refusal points at what can be used instead.

    The generic tool-error template ends "Call the tool again with your
    mistake corrected.", and on the box the model did: the same call twice.
    """
    message = _make_routing_rejection(
        _LIGHT_SET,
        {"intent__HassTurnOn": "assist", _LIVE_CONTEXT: "assist"},
        "c9",
    )

    content = cast("str", message.content)
    assert message.status == "error"
    assert message.name == _LIGHT_SET
    assert message.tool_call_id == "c9"
    # The PIN recovery finds a routing rejection by this marker.
    assert _ROUTING_REJECTION_MARKER in content
    assert "Do not call it again" in content
    assert "Call the tool again" not in content
    assert f"instead: {_LIVE_CONTEXT}, intent__HassTurnOn." in content

# ruff: noqa: S101
"""
A delayed command reaches the timer tool once, with the delay given once.

Field test 2026-11-27, on a voice satellite:

- "Unlock the playroom door lock in one minute." The model gave the timer the
  whole sentence as its command. When the timer ended, Home Assistant's own
  agent read "in one minute" as a new delayed command and started a second
  timer, so the door unlocked after two minutes.
- "Lock the garage door lock in 30 seconds." Ranking bound the device tools
  and left the timer tool out. The model's call to it was rejected as not
  retrieved, and it offered to lock the door at once instead.
"""

from __future__ import annotations

import json
import sys
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from homeassistant.components.homeassistant.exposed_entities import (
    async_expose_entity,
)
from homeassistant.setup import async_setup_component

from custom_components.home_generative_agent.agent import graph
from custom_components.home_generative_agent.agent.graph import (
    State,
    _query_wants_delayed_command,
    _retrieve_tools,
    _run_ha_tool,
)
from custom_components.home_generative_agent.agent.helpers import (
    command_sets_its_own_delay,
)

if TYPE_CHECKING:
    from collections.abc import Iterator

    from homeassistant.core import HomeAssistant
    from langchain_core.runnables import RunnableConfig

TIMER = "intent__HassStartTimer"


@pytest.mark.parametrize(
    "query",
    [
        "Lock the garage door lock in 30 seconds.",
        "Unlock the playroom door lock in one minute.",
        "turn off the fan after an hour",
        "close the garage in a few minutes",
        "turn on the porch light in half an hour",
        "open the shade in 5 min",
        "turn off the heater in 1.5 hours",
    ],
)
def test_delay_phrasings_are_read(query: str) -> None:
    """A request to act after a delay asks for the timer tool."""
    assert _query_wants_delayed_command(query)


@pytest.mark.parametrize(
    "query",
    [
        "Lock the garage door lock.",
        "turn on the light in the kitchen",
        "what's the temperature in the living room",
        "turn on the fan every 5 minutes",
        "set the thermostat to 70 in one room",
    ],
)
def test_requests_without_a_delay_are_not(query: str) -> None:
    """A request with no delay leaves the timer tool to ranking."""
    assert not _query_wants_delayed_command(query)


def _item(name: str, *, score: float = 0.5, is_actuation: bool = False) -> MagicMock:
    item = MagicMock()
    item.value = {
        "name": name,
        "api_id": "assist",
        "description": f"{name} description",
        "parameters": "{}",
        "is_actuation": is_actuation,
    }
    item.score = score
    return item


def _store() -> MagicMock:
    """Ranking finds the device tools for the command's words, never the timer."""
    store = MagicMock()
    store.asearch = AsyncMock(
        return_value=[
            _item("intent__HassTurnOn", score=0.62, is_actuation=True),
            _item("intent__HassTurnOff", score=0.61, is_actuation=True),
            _item("vacuum__HassVacuumStart", score=0.55, is_actuation=True),
        ]
    )
    timer = _item(TIMER)

    async def aget(_namespace: Any, key: str = "", **_kwargs: Any) -> Any:
        return timer if key == f"assist::{TIMER}" else None

    store.aget = AsyncMock(side_effect=aget)
    return store


def _state(text: str) -> State:
    return {
        "messages": [MagicMock(content=text)],
        "summary": "",
        "chat_model_usage_metadata": {},
        "messages_to_remove": [],
        "selected_tools": [],
        "tool_routing_map": {},
    }


def _config(*live: str, **options: Any) -> RunnableConfig:
    api = SimpleNamespace(
        tools=[
            SimpleNamespace(
                name=name,
                description=f"{name} live",
                parameters={"type": "object", "properties": {}},
            )
            for name in live
        ],
        custom_serializer=None,
    )
    return {
        "configurable": {
            "options": {
                "llm_hass_api": ["assist"],
                "tool_relevance_threshold": 0.15,
                **options,
            },
            "tool_index_ready": True,
            "langchain_tools": {},
            "ha_llm_api": MagicMock(apis={"assist": api}),
        }
    }


_SATELLITE_TOOLS = (
    "intent__HassTurnOn",
    "intent__HassTurnOff",
    "vacuum__HassVacuumStart",
    TIMER,
)


@pytest.mark.asyncio
async def test_timer_tool_is_bound_for_a_delayed_command() -> None:
    """The field repro: the timer tool is bound beside the device tools."""
    result = await _retrieve_tools(
        _state("Lock the garage door lock in 30 seconds."),
        _config(*_SATELLITE_TOOLS),
        store=_store(),
    )

    assert result["tool_routing_map"][TIMER] == "assist"
    # Bound outside the limit: nothing ranking chose is evicted.
    assert "intent__HassTurnOn" in result["tool_routing_map"]
    assert "intent__HassTurnOff" in result["tool_routing_map"]


@pytest.mark.asyncio
async def test_timer_tool_is_not_bound_without_a_delay() -> None:
    """A plain command does not widen the tool list."""
    result = await _retrieve_tools(
        _state("Lock the garage door lock."),
        _config(*_SATELLITE_TOOLS),
        store=_store(),
    )

    assert TIMER not in result["tool_routing_map"]


@pytest.mark.asyncio
async def test_timer_tool_is_not_bound_where_it_is_not_live() -> None:
    """Home Assistant offers timers to timer-capable satellites alone."""
    result = await _retrieve_tools(
        _state("Lock the garage door lock in 30 seconds."),
        _config("intent__HassTurnOn", "intent__HassTurnOff"),
        store=_store(),
    )

    assert TIMER not in result["tool_routing_map"]


@pytest.mark.asyncio
async def test_excluded_timer_tool_is_not_bound() -> None:
    """An exclusion of the timer tool keeps it out."""
    result = await _retrieve_tools(
        _state("Lock the garage door lock in 30 seconds."),
        _config(*_SATELLITE_TOOLS, tool_exclusions={"assist": [TIMER]}),
        store=_store(),
    )

    assert TIMER not in result["tool_routing_map"]


_CONVERSATION_MODULES = (
    "homeassistant.components.conversation",
    "hassil",
    "home_assistant_intents",
)


def _is_conversation_module(name: str) -> bool:
    return any(name == m or name.startswith(f"{m}.") for m in _CONVERSATION_MODULES)


@pytest.fixture
def real_conversation() -> Iterator[None]:
    """
    Load Home Assistant's real conversation component for this test.

    Other test modules stub it in `sys.modules` (test_conversation_units,
    test_conversation_stream), and a stub has no agent to read a command.
    The stubs are put back afterwards, and the real modules taken out.
    """
    saved = {n: m for n, m in sys.modules.items() if _is_conversation_module(n)}
    for name in saved:
        del sys.modules[name]
    try:
        yield
    finally:
        for name in [n for n in sys.modules if _is_conversation_module(n)]:
            del sys.modules[name]
        sys.modules.update(saved)


@pytest.fixture
async def home(
    real_conversation: None,  # noqa: ARG001
    hass: HomeAssistant,
) -> HomeAssistant:
    """Set up Home Assistant's own agent with an exposed lock."""
    assert await async_setup_component(hass, "homeassistant", {})
    assert await async_setup_component(hass, "conversation", {})
    assert await async_setup_component(hass, "intent", {})
    hass.states.async_set("lock.front_door", "locked", {"friendly_name": "Front Door"})
    async_expose_entity(hass, "conversation", "lock.front_door", should_expose=True)
    return hass


async def test_command_naming_its_delay_is_read_as_a_timer(home: HomeAssistant) -> None:
    """Home Assistant's agent would start a second timer for this command."""
    assert await command_sets_its_own_delay(
        home,
        "Unlock the front door in one minute.",
        language="en",
        device_id=None,
    )


@pytest.mark.parametrize("command", ["Unlock the front door", ""])
async def test_command_alone_is_not(home: HomeAssistant, command: str) -> None:
    """The action alone runs once, when the timer ends."""
    assert not await command_sets_its_own_delay(
        home, command, language="en", device_id=None
    )


async def test_unreadable_command_is_left_to_the_gate(home: HomeAssistant) -> None:
    """A failure reading the command answers False: the PIN gate still holds it."""
    with patch(
        "homeassistant.components.conversation.default_agent.DefaultAgent."
        "async_recognize_intent",
        side_effect=RuntimeError("boom"),
    ):
        assert not await command_sets_its_own_delay(
            home, "Unlock the front door in one minute.", language="en", device_id=None
        )


def _ctx(hass: HomeAssistant) -> MagicMock:
    ha_llm_api = MagicMock()
    ha_llm_api.apis = {
        "assist": SimpleNamespace(
            llm_context=SimpleNamespace(language="en", device_id=None)
        )
    }
    ha_llm_api.async_call_tool = AsyncMock()
    ctx = MagicMock()
    ctx.hass = hass
    ctx.ha_llm_api = ha_llm_api
    return ctx


def _origin(*, remote: bool) -> SimpleNamespace:
    return SimpleNamespace(remote=remote, ambiguous=False)


async def test_nested_delay_is_refused_before_the_pin(home: HomeAssistant) -> None:
    """The model is told to drop the delay; nothing is held or run."""
    ctx = _ctx(home)
    prepare = MagicMock()
    with (
        patch.object(graph, "tool_origin", return_value=_origin(remote=False)),
        patch.object(graph, "_prepare_tool_args", prepare),
    ):
        reply = await _run_ha_tool(
            TIMER,
            {
                "minutes": 1,
                "conversation_command": "Unlock the front door in one minute.",
            },
            ctx,
            {"id": "call-1"},
        )

    assert reply.status == "error"
    assert "names its own delay" in str(reply.content)
    assert "conversation_command" in str(reply.content)
    prepare.assert_not_called()
    ctx.ha_llm_api.async_call_tool.assert_not_called()


async def test_command_alone_goes_on_to_the_gate(home: HomeAssistant) -> None:
    """A command without its delay reaches the PIN gate as before."""
    ctx = _ctx(home)
    held = MagicMock(content=json.dumps({"status": "requires_pin"}))
    prepare = MagicMock(return_value=(held, {}))
    with (
        patch.object(graph, "tool_origin", return_value=_origin(remote=False)),
        patch.object(graph, "_prepare_tool_args", prepare),
    ):
        reply = await _run_ha_tool(
            TIMER,
            {"minutes": 1, "conversation_command": "Unlock the front door"},
            ctx,
            {"id": "call-1"},
        )

    assert reply is held
    prepare.assert_called_once()


async def test_remote_delayed_command_is_left_to_the_gate(home: HomeAssistant) -> None:
    """A remote Home Assistant runs the command on its own agent: not read here."""
    ctx = _ctx(home)
    held = MagicMock()
    prepare = MagicMock(return_value=(held, {}))
    reader = AsyncMock(return_value=True)
    with (
        patch.object(graph, "tool_origin", return_value=_origin(remote=True)),
        patch.object(graph, "_prepare_tool_args", prepare),
        patch.object(graph, "command_sets_its_own_delay", reader),
    ):
        reply = await _run_ha_tool(
            TIMER,
            {
                "minutes": 1,
                "conversation_command": "Unlock the front door in one minute.",
            },
            ctx,
            {"id": "call-1"},
        )

    assert reply is held
    reader.assert_not_called()

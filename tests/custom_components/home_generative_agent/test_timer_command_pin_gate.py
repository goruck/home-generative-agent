# ruff: noqa: S101
"""
A command left for Home Assistant to run later needs the PIN now.

A timer given a command, or Home Assistant 2026.12's scheduled command, runs
that command when the delay ends as a new conversation on the agent the
request named. This integration names none, so the command runs on Home
Assistant's own agent, past this gate: "in a minute, unlock the front door"
unlocked it with no PIN.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, cast

import pytest
from homeassistant.components.homeassistant.exposed_entities import (
    async_expose_entity,
)
from homeassistant.setup import async_setup_component

from custom_components.home_generative_agent.agent.graph import (
    _is_critical_action,
    _prepare_tool_args,
)
from custom_components.home_generative_agent.agent.pin_messages import (
    describe_pending_action,
)
from custom_components.home_generative_agent.agent.rule_entities import (
    expand_critical_rules,
)
from custom_components.home_generative_agent.const import (
    RECOMMENDED_CRITICAL_ACTIONS,
)

if TYPE_CHECKING:
    from homeassistant.core import HomeAssistant


@pytest.fixture
async def home(hass: HomeAssistant) -> HomeAssistant:
    """Set up an exposed lock, which a direct unlock needs the PIN for."""
    assert await async_setup_component(hass, "homeassistant", {})
    hass.states.async_set("lock.front_door", "locked", {"friendly_name": "Front Door"})
    async_expose_entity(hass, "conversation", "lock.front_door", should_expose=True)
    return hass


def _held(hass: HomeAssistant, tool_name: str, args: dict[str, Any]) -> bool:
    return _is_critical_action(
        dict(args),
        expand_critical_rules(hass, RECOMMENDED_CRITICAL_ACTIONS),
        tool_name,
        hass,
    )


@pytest.mark.parametrize(
    ("tool_name", "args"),
    [
        (
            "intent__HassStartTimer",
            {"minutes": 1, "conversation_command": "unlock the front door"},
        ),
        # The name an older Home Assistant gave the tool.
        (
            "HassStartTimer",
            {"seconds": 5, "conversation_command": "unlock the front door"},
        ),
        # A command that names nothing critical is held too: its words
        # cannot be read for what it will do.
        (
            "intent__HassStartTimer",
            {"minutes": 1, "conversation_command": "turn on the kitchen light"},
        ),
        ("intent__schedule_command", {"command": "Unlock the front door"}),
        ("intent__schedule_command", {"minutes": 1}),
    ],
)
async def test_a_command_run_later_needs_the_pin(
    home: HomeAssistant, tool_name: str, args: dict[str, Any]
) -> None:
    """The command runs where this gate cannot see it, so it is held now."""
    assert _held(home, tool_name, args)


@pytest.mark.parametrize(
    "args",
    [
        {"minutes": 5},
        {"minutes": 5, "name": "pasta"},
        {"minutes": 5, "conversation_command": ""},
        {"minutes": 5, "conversation_command": "   "},
        {"minutes": 5, "conversation_command": None},
    ],
)
async def test_a_plain_timer_is_not_held(
    home: HomeAssistant, args: dict[str, Any]
) -> None:
    """A timer that only rings runs nothing, so it needs no PIN."""
    assert not _held(home, "intent__HassStartTimer", args)


async def test_the_unlock_itself_is_held(home: HomeAssistant) -> None:
    """The same unlock asked for now is held, as the timer's now is."""
    assert _held(home, "intent__HassTurnOff", {"name": "Front Door"})


async def test_a_held_timer_asks_for_the_pin(home: HomeAssistant) -> None:
    """Through the tool path: the call is parked and the PIN is asked for."""
    ctx = cast(
        "Any",
        SimpleNamespace(
            hass=home,
            config={"configurable": {"thread_id": "t1"}},
            critical_actions=RECOMMENDED_CRITICAL_ACTIONS,
            pin_enabled=True,
            pin_hash="hash",
            pin_salt="salt",
            pending_actions={},
            ha_llm_api=None,
        ),
    )
    args: dict[str, Any] = {
        "minutes": 1,
        "conversation_command": "unlock the front door",
    }

    redirect, _ = _prepare_tool_args(
        "intent__HassStartTimer", dict(args), ctx, {"id": "c1"}
    )

    assert redirect is not None
    assert "requires_pin" in str(redirect.content)
    (action,) = ctx.pending_actions.values()
    assert action["tool_args"]["conversation_command"] == "unlock the front door"
    # The line names the command the PIN approves, not the timer tool.
    assert describe_pending_action(home, action) == 'run "unlock the front door" later'


@pytest.mark.parametrize(
    ("tool_name", "args", "line"),
    [
        (
            "intent__schedule_command",
            {"minutes": 1, "command": "Unlock the front door"},
            'run "Unlock the front door" later',
        ),
        # The command's own quote marks cannot close the line's quote.
        (
            "intent__HassStartTimer",
            {"minutes": 1, "conversation_command": 'light" and unlock "door'},
            "run \"light' and unlock 'door\" later",
        ),
    ],
)
async def test_the_pin_line_names_the_command(
    home: HomeAssistant, tool_name: str, args: dict[str, Any], line: str
) -> None:
    """What the user approves is the command, quoted safely."""
    action = {"tool_name": tool_name, "tool_args": args}
    assert describe_pending_action(home, action) == line


async def test_a_long_command_shows_it_is_cut(home: HomeAssistant) -> None:
    """A command past the line's length ends in a mark, not silently."""
    command = "turn on the hall light " * 20
    action = {
        "tool_name": "intent__HassStartTimer",
        "tool_args": {"minutes": 1, "conversation_command": command},
    }
    assert describe_pending_action(home, action).endswith('…" later')

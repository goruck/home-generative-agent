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
from custom_components.home_generative_agent.agent.helpers import runs_a_command_later
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
    assert (
        describe_pending_action(home, action)
        == 'run "unlock the front door" in 0:01:00'
    )


@pytest.mark.parametrize(
    ("tool_name", "args", "line"),
    [
        (
            "intent__schedule_command",
            {"minutes": 1, "command": "Unlock the front door"},
            'run "Unlock the front door" in 0:01:00',
        ),
        # The command's own quote marks cannot close the line's quote.
        (
            "intent__HassStartTimer",
            {"minutes": 1, "conversation_command": 'light" and unlock "door'},
            "run \"light' and unlock 'door\" in 0:01:00",
        ),
    ],
)
async def test_the_pin_line_names_the_command(
    home: HomeAssistant, tool_name: str, args: dict[str, Any], line: str
) -> None:
    """What the user approves is the command, quoted safely."""
    action = {"tool_name": tool_name, "tool_args": args}
    assert describe_pending_action(home, action) == line


async def test_a_long_command_is_shown_whole(home: HomeAssistant) -> None:
    """The whole command runs, so the line shows all of it, uncut."""
    command = "turn on the hall light and " * 12 + "unlock the back door"
    action = {
        "tool_name": "intent__HassStartTimer",
        "tool_args": {"minutes": 1, "conversation_command": command},
    }
    line = describe_pending_action(home, action)
    assert line == f'run "{command.strip()}" in 0:01:00'
    assert "…" not in line


@pytest.mark.parametrize("command", [0, False])
async def test_a_command_that_is_not_text_is_held(
    home: HomeAssistant, command: object
) -> None:
    """Home Assistant runs 0 or False as the text "0" or "False", so it is held."""
    args = {"seconds": 1, "conversation_command": command}
    assert _held(home, "intent__HassStartTimer", args)
    action = {"tool_name": "intent__HassStartTimer", "tool_args": args}
    assert describe_pending_action(home, action) == f'run "{command}" in 0:00:01'


@pytest.mark.parametrize(
    ("tool_name", "expected"),
    [
        ("intent__schedule_command", True),
        # A remote Home Assistant's tool, named after its server.
        ("upstairs_ha__intent__schedule_command", True),
        # Other tools that share the last part of the name.
        ("script__schedule_command", False),
        ("intent_script__schedule_command", False),
        ("intent_script__HassStartTimer", False),
        ("intent__HassStartTimer", True),
        ("HassStartTimer", True),
        ("upstairs_ha__intent__HassStartTimer", True),
        ("cron__schedule_command", False),
        ("schedule_command", False),
    ],
)
def test_only_home_assistants_scheduled_command_matches(
    tool_name: str, *, expected: bool
) -> None:
    """A user's script or another server's tool named alike is not HA's tool."""
    args = {"command": "unlock the door", "conversation_command": "unlock the door"}
    assert runs_a_command_later(tool_name, args) is expected


@pytest.mark.parametrize(
    "tool_name", ["script__schedule_command", "intent_script__HassStartTimer"]
)
async def test_a_script_named_alike_is_named_by_what_it_runs(
    home: HomeAssistant, tool_name: str
) -> None:
    """A script or intent script is named as itself, not as a delayed command."""
    action = {
        "tool_name": tool_name,
        "tool_args": {"command": "say hi", "conversation_command": "say hi"},
    }
    assert '"say hi" in' not in describe_pending_action(home, action)


async def test_a_scheduled_command_with_no_text_is_not_shown_empty(
    home: HomeAssistant,
) -> None:
    """With no command to show, the line does not read 'run "" later'."""
    action = {"tool_name": "intent__schedule_command", "tool_args": {"minutes": 1}}
    assert '""' not in describe_pending_action(home, action)


@pytest.mark.parametrize(
    ("args", "when"),
    [
        ({"hours": 8}, "in 8:00:00"),
        ({"hours": 1, "minutes": 2, "seconds": 3}, "in 1:02:03"),
        ({"minutes": 90}, "in 1:30:00"),
        # A delay that cannot be read is not guessed at.
        ({}, "later"),
        ({"minutes": "soon"}, "later"),
        ({"minutes": True}, "later"),
        ({"minutes": -5}, "later"),
    ],
)
async def test_the_pin_line_says_when_the_command_runs(
    home: HomeAssistant, args: dict[str, Any], when: str
) -> None:
    """The model picks the delay, so the line the PIN approves names it."""
    action = {
        "tool_name": "intent__HassStartTimer",
        "tool_args": {**args, "conversation_command": "unlock the front door"},
    }
    assert (
        describe_pending_action(home, action) == f'run "unlock the front door" {when}'
    )


async def test_a_remote_timer_shows_the_whole_command_and_server(
    home: HomeAssistant,
) -> None:
    """A remote server's delayed command is shown whole, with the server named."""
    command = "turn on the porch light and then unlock the back door please"
    action = {
        "tool_name": "upstairs_ha__intent__HassStartTimer",
        "api_tool_name": "intent__HassStartTimer",
        "remote": True,
        "api_name": "Upstairs HA",
        "tool_args": {"minutes": 10, "conversation_command": command},
    }
    assert (
        describe_pending_action(home, action)
        == f'run "{command}" in 0:10:00 on "Upstairs HA"'
    )

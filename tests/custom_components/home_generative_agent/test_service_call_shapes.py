# ruff: noqa: S101
"""
The PIN gate reads the service a tool calls from every shape it can take (#837).

A service-calling tool may name its service as `domain` plus `service`, as one
`action` (or `service`) of the form `<domain>.<service>`, or by its own name
(`scene.apply`, or `scene__apply` the way HA names an action's tool). Each
shape is held like the `domain`/`service` form.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest
from homeassistant.setup import async_setup_component

from custom_components.home_generative_agent.agent.graph import _is_critical_action
from custom_components.home_generative_agent.agent.pin_messages import (
    describe_pending_action,
)
from custom_components.home_generative_agent.const import (
    RECOMMENDED_CRITICAL_ACTIONS,
)

if TYPE_CHECKING:
    from homeassistant.core import HomeAssistant

RULES = RECOMMENDED_CRITICAL_ACTIONS

UNLOCK_SCENE = {"entities": {"lock.front_door": "unlocked"}}
LIGHT_SCENE = {"entities": {"light.porch": "on"}}


@pytest.fixture
async def home(hass: HomeAssistant) -> HomeAssistant:
    """Set up a front door lock, a porch light and HA's lock and scene services."""
    hass.states.async_set("lock.front_door", "locked", {"friendly_name": "Front Door"})
    hass.states.async_set("light.porch", "off", {"friendly_name": "Porch"})
    # Its state only: HA has no cover services here, as when a tool sends
    # the call to another Home Assistant.
    hass.states.async_set(
        "cover.garage_door", "closed", {"friendly_name": "Garage Door"}
    )
    assert await async_setup_component(hass, "lock", {})
    assert await async_setup_component(hass, "scene", {"scene": []})
    assert await async_setup_component(
        hass,
        "script",
        {
            "script": {
                # Shares its name with the `scene__apply` tool's service.
                "apply": {
                    "alias": "Porch Lights",
                    "sequence": [
                        {
                            "action": "light.turn_on",
                            "target": {"entity_id": "light.porch"},
                        }
                    ],
                },
                "porch": {
                    "alias": "Porch",
                    "sequence": [
                        {
                            "action": "light.turn_on",
                            "target": {"entity_id": "light.porch"},
                        }
                    ],
                },
            }
        },
    )
    await hass.async_block_till_done()
    assert hass.services.has_service("scene", "apply")
    assert hass.services.has_service("lock", "unlock")
    return hass


@pytest.mark.parametrize(
    ("args", "tool_name", "critical"),
    [
        # `action` names the service.
        ({"action": "scene.apply", **UNLOCK_SCENE}, "mcp__call_service", True),
        (
            {"action": "scene.apply", "data": UNLOCK_SCENE},
            "mcp__call_service",
            True,
        ),
        ({"action": "scene.apply", **LIGHT_SCENE}, "mcp__call_service", False),
        (
            {"action": "lock.unlock", "entity_id": "lock.front_door"},
            "mcp__call_service",
            True,
        ),
        (
            {"action": "lock.lock", "entity_id": "lock.front_door"},
            "mcp__call_service",
            False,
        ),
        # `service` alone names the domain too.
        ({"service": "scene.apply", **UNLOCK_SCENE}, "mcp__call_service", True),
        (
            {"service": "lock.unlock", "entity_id": "lock.front_door"},
            "mcp__call_service",
            True,
        ),
        # `domain` with the service under `action`.
        (
            {"domain": "scene", "action": "apply", **UNLOCK_SCENE},
            "mcp__call_service",
            True,
        ),
        # The tool is named after the service.
        (UNLOCK_SCENE, "scene.apply", True),
        (UNLOCK_SCENE, "scene__apply", True),
        (UNLOCK_SCENE, "assist__scene__apply", True),
        (UNLOCK_SCENE, "mcp__scene.apply", True),
        (LIGHT_SCENE, "scene__apply", False),
        ({"entity_id": "lock.front_door"}, "lock__unlock", True),
        ({"entity_id": "lock.front_door"}, "lock.unlock", True),
        ({"entity_id": "lock.front_door"}, "lock__lock", False),
        # `action` without a domain runs in its target's domain (#837 review).
        (
            {"action": "unlock", "entity_id": "lock.front_door"},
            "mcp__call_service",
            True,
        ),
        (
            {"action": "unlock", "target": {"entity_id": "lock.front_door"}},
            "mcp__call_service",
            True,
        ),
        (
            {"action": "lock", "entity_id": "lock.front_door"},
            "mcp__call_service",
            False,
        ),
        # A domain with spaces, or a null one, is read as HA reads it.
        (
            {"domain": " lock ", "action": "unlock", "entity_id": "lock.front_door"},
            "mcp__call_service",
            True,
        ),
        (
            {"domain": None, "service": "unlock", "entity_id": "lock.front_door"},
            "mcp__call_service",
            True,
        ),
        # A list of services: each one is screened.
        (
            {"action": ["lock.unlock"], "entity_id": "lock.front_door"},
            "mcp__call_service",
            True,
        ),
        (
            {
                "service": ["light.turn_on", "lock.unlock"],
                "entity_id": "lock.front_door",
            },
            "mcp__call_service",
            True,
        ),
        # A dotted name counts even when this HA lacks the service: the tool
        # may send it to another one.
        (
            {"action": "cover.open_cover", "entity_id": "cover.garage_door"},
            "mcp__call_service",
            True,
        ),
        # A dotted `service` beside its domain is read once, not as
        # `script.script.turn_off`.
        (
            {
                "domain": "script",
                "service": "script.turn_off",
                "entity_id": "script.porch",
            },
            "mcp__call_service",
            False,
        ),
        # Two shapes that disagree: either one could be what runs.
        (
            {"domain": "lock", "service": "turn_on", "action": "unlock"}
            | {"entity_id": "lock.front_door"},
            "mcp__call_service",
            True,
        ),
        (
            {"domain": "scene", "service": "reload", "action": "apply"} | UNLOCK_SCENE,
            "mcp__call_service",
            True,
        ),
        (
            {"domain": "light", "service": "turn_on", "action": "scene.apply"}
            | UNLOCK_SCENE,
            "mcp__call_service",
            True,
        ),
        (
            {"domain": "light", "service": "turn_on", **UNLOCK_SCENE},
            "scene__apply",
            True,
        ),
    ],
)
async def test_every_shape_of_a_service_call_is_screened(
    home: HomeAssistant, args: dict[str, Any], tool_name: str, *, critical: bool
) -> None:
    """A call is held whichever way the tool names its service."""
    assert _is_critical_action(dict(args), RULES, tool_name, home) is critical


@pytest.mark.parametrize(
    ("args", "tool_name"),
    [
        # Tools whose names are not services HA has.
        ({"entity_id": "lock.front_door"}, "mcp__get_weather"),
        ({"query": "front door"}, "assist__GetLiveContext"),
        # An `action` that is not a service name.
        ({"action": "open", "entity_id": "light.porch"}, "mcp__todo"),
    ],
)
async def test_a_tool_that_names_no_service_is_not_held_for_it(
    home: HomeAssistant, args: dict[str, Any], tool_name: str
) -> None:
    """Only a name HA has a service for is read as one."""
    assert not _is_critical_action(dict(args), RULES, tool_name, home)


@pytest.mark.parametrize(
    ("args", "tool_name"),
    [
        ({"action": "scene.apply", **UNLOCK_SCENE}, "mcp__call_service"),
        ({"service": "lock.unlock", "entity_id": "lock.front_door"}, "x"),
        (UNLOCK_SCENE, "scene__apply"),
        ({"entity_id": "lock.front_door"}, "lock.unlock"),
    ],
)
def test_without_hass_every_shape_is_still_screened(
    args: dict[str, Any], tool_name: str
) -> None:
    """With no hass to ask, a name shaped like a service is read as one."""
    assert _is_critical_action(dict(args), RULES, tool_name, None)


@pytest.mark.parametrize(
    "args",
    [
        {"action": "{{ 'lock' }}.unlock"},
        {"service": "{{ 'lock' }}.unlock"},
        {"domain": "{{ 'lock' }}", "action": "unlock"},
        {"domain": "{{ 'lock' }}", "service": "unlock"},
        # Not text: which service runs cannot be read.
        {"action": {"name": "lock.unlock"}},
        {"service": [["lock.unlock"]]},
    ],
)
async def test_an_unreadable_service_name_is_held(
    home: HomeAssistant, args: dict[str, Any]
) -> None:
    """Which service a template or an odd value names is unknown until it runs."""
    args = {**args, "entity_id": "lock.front_door"}
    assert _is_critical_action(args, RULES, "mcp__call_service", home)


SET_UNLOCKED = 'set "Front Door" (lock.front_door) to unlocked'
UNLOCK = 'unlock "Front Door" (lock.front_door)'


@pytest.mark.parametrize(
    ("tool_name", "args", "line"),
    [
        ("mcp__call_service", {"action": "scene.apply", **UNLOCK_SCENE}, SET_UNLOCKED),
        (
            "mcp__call_service",
            {"domain": "scene", "action": "apply", **UNLOCK_SCENE},
            SET_UNLOCKED,
        ),
        ("mcp__scene.apply", UNLOCK_SCENE, SET_UNLOCKED),
        (
            "mcp__call_service",
            {"service": "lock.unlock", "entity_id": "lock.front_door"},
            UNLOCK,
        ),
        ("lock.unlock", {"entity_id": "lock.front_door"}, UNLOCK),
        ("assist__lock__unlock", {"entity_id": "lock.front_door"}, UNLOCK),
        # A script sharing the service's name is named with the service.
        (
            "scene__apply",
            UNLOCK_SCENE,
            f'turn on "Porch Lights", {SET_UNLOCKED}',
        ),
        # A bare service is named in the target's domain.
        (
            "mcp__call_service",
            {
                "service": "unlock",
                "action": "light.turn_on",
                "entity_id": "lock.front_door",
            },
            f'turn on "Front Door" (lock.front_door), {UNLOCK}',
        ),
        (
            "mcp__call_service",
            {"action": "unlock", "entity_id": "lock.front_door"},
            UNLOCK,
        ),
        # Two shapes that disagree: the line names both.
        (
            "scene__apply",
            {"domain": "light", "service": "turn_on", **UNLOCK_SCENE},
            f'turn on "Porch Lights", {SET_UNLOCKED}, turn on an unnamed device',
        ),
    ],
)
async def test_the_pin_line_names_the_service_the_call_names(
    home: HomeAssistant, tool_name: str, args: dict[str, Any], line: str
) -> None:
    """The PIN line names the service, however the tool names it."""
    action = {"tool_name": tool_name, "tool_args": args}
    assert describe_pending_action(home, action) == line

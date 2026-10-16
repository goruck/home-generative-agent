# ruff: noqa: S101
"""
Scripts and scenes are PIN-gated by what they do (#775).

The gate matched the call the agent made. A script run as its own tool has no
domain or service, and `HassTurnOn` on a script or a scene makes a
`turn_on` call no rule names, so a script that opens the garage door, or a
scene that unlocks the front door, ran with no PIN.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, cast

import pytest
from homeassistant.helpers import entity_registry as er
from homeassistant.setup import async_setup_component

from custom_components.home_generative_agent.agent.graph import (
    _is_critical_action,
    _prepare_tool_args,
)
from custom_components.home_generative_agent.agent.indirect_pin import (
    script_tool_entity_id,
)
from custom_components.home_generative_agent.const import (
    RECOMMENDED_CRITICAL_ACTIONS,
)

if TYPE_CHECKING:
    from homeassistant.core import HomeAssistant

RULES = RECOMMENDED_CRITICAL_ACTIONS


def _call(action: str, entity_id: str) -> dict[str, Any]:
    return {"action": action, "target": {"entity_id": entity_id}}


@pytest.fixture
async def home(hass: HomeAssistant) -> HomeAssistant:
    """Set up a garage door, blinds, a front door lock, lights, scripts, scenes."""
    hass.states.async_set("cover.garage_door", "closed")
    hass.states.async_set("cover.living_room_blinds", "closed")
    hass.states.async_set("lock.front_door", "locked")
    hass.states.async_set("light.porch", "off")
    assert await async_setup_component(
        hass,
        "group",
        {"group": {"doors": {"name": "Doors", "entities": ["lock.front_door"]}}},
    )
    assert await async_setup_component(
        hass,
        "script",
        {
            "script": {
                "open_garage": {
                    "alias": "Open Garage",
                    "sequence": [_call("cover.open_cover", "cover.garage_door")],
                },
                "porch_lights": {
                    "alias": "Porch Lights",
                    "sequence": [
                        {"delay": {"seconds": 1}},
                        _call("light.turn_on", "light.porch"),
                    ],
                },
                "lock_up": {
                    "alias": "Lock Up",
                    "sequence": [_call("lock.lock", "lock.front_door")],
                },
                "let_in": {
                    "alias": "Let In",
                    "sequence": [_call("lock.unlock", "lock.front_door")],
                },
                "evening": {
                    "alias": "Evening",
                    "sequence": [_call("script.turn_on", "script.porch_lights")],
                },
            }
        },
    )
    assert await async_setup_component(
        hass,
        "scene",
        {
            "scene": [
                {
                    "name": "Arrive Home",
                    "entities": {"lock.front_door": "unlocked", "light.porch": "on"},
                },
                {
                    "name": "Good Night",
                    "entities": {
                        "lock.front_door": "locked",
                        "cover.garage_door": "closed",
                        "light.porch": "off",
                    },
                },
                {"name": "Air Out", "entities": {"cover.garage_door": "open"}},
                {"name": "Daylight", "entities": {"cover.living_room_blinds": "open"}},
                {"name": "Leave", "entities": {"script.open_garage": "on"}},
                {"name": "Open Up", "entities": {"group.doors": "unlocked"}},
            ]
        },
    )
    await hass.async_block_till_done()
    return hass


@pytest.mark.parametrize(
    "tool_name",
    ["script__open_garage", "assist__script__open_garage", "script__let_in"],
)
async def test_a_script_tool_that_does_a_critical_action_is_held(
    home: HomeAssistant, tool_name: str
) -> None:
    """The example from the issue, through the script's own tool."""
    assert _is_critical_action({}, RULES, tool_name, home)


@pytest.mark.parametrize("tool_name", ["script__porch_lights", "script__lock_up"])
async def test_a_script_tool_that_does_nothing_critical_runs(
    home: HomeAssistant, tool_name: str
) -> None:
    """Delays, lights and locking a door do not ask for the PIN."""
    assert not _is_critical_action({}, RULES, tool_name, home)


async def test_a_script_that_runs_another_script_is_held(home: HomeAssistant) -> None:
    """As in an automation, a script run from a script is not followed."""
    assert _is_critical_action({}, RULES, "script__evening", home)


async def test_a_rule_naming_the_script_covers_its_tool(home: HomeAssistant) -> None:
    """The workaround in the issue only covered `HassTurnOn`."""
    rules = [{"domain": "script", "service": "porch_lights"}]
    assert _is_critical_action({}, rules, "script__porch_lights", home)
    rules = [{"domain": "script", "service": "turn_on"}]
    assert _is_critical_action({}, rules, "script__porch_lights", home)


@pytest.mark.parametrize(
    ("name", "critical"),
    [
        ("Open Garage", True),
        ("Porch Lights", False),
        ("Arrive Home", True),
        ("Good Night", False),
        ("Air Out", True),
        ("Daylight", False),
        ("Leave", True),
        ("Open Up", True),
    ],
)
async def test_turning_on_a_script_or_scene_checks_what_it_does(
    home: HomeAssistant, name: str, *, critical: bool
) -> None:
    """A scene that unlocks or opens a door asks for the PIN; a bedtime one not."""
    assert (
        _is_critical_action({"name": name}, RULES, "intent__HassTurnOn", home)
        is critical
    )


async def test_turning_a_script_off_stops_it(home: HomeAssistant) -> None:
    """HassTurnOff on a script stops it rather than running its steps."""
    assert not _is_critical_action(
        {"name": "Open Garage"}, RULES, "intent__HassTurnOff", home
    )


async def test_a_scene_stored_on_a_device_runs(home: HomeAssistant) -> None:
    """A Hue scene can only set Hue lights."""
    er.async_get(home).async_get_or_create(
        "scene", "hue", "relax", suggested_object_id="relax"
    )
    home.states.async_set("scene.relax", "unknown", {"friendly_name": "Relax"})
    assert not _is_critical_action({"name": "Relax"}, RULES, "HassTurnOn", home)


async def test_a_script_or_scene_that_cannot_be_read_is_held(
    home: HomeAssistant,
) -> None:
    """With no config behind the entity, what it does is unknown."""
    home.states.async_set("scene.mystery", "unknown", {"friendly_name": "Mystery"})
    assert _is_critical_action({"name": "Mystery"}, RULES, "HassTurnOn", home)
    assert _is_critical_action({}, RULES, "script__not_loaded", home)


async def test_the_script_tool_is_held_for_a_pin(home: HomeAssistant) -> None:
    """End to end through argument preparation: the call is held, not run."""
    pending: dict[str, dict[str, Any]] = {}
    ctx = cast(
        "Any",
        SimpleNamespace(
            hass=home,
            config={"configurable": {"thread_id": "t1"}},
            critical_actions=RULES,
            pin_enabled=True,
            pin_hash="hash",
            pin_salt="salt",
            pending_actions=pending,
        ),
    )
    redirect, _ = _prepare_tool_args(
        "assist__script__open_garage", {}, ctx, {"id": "call1"}
    )
    assert redirect is not None
    assert "requires_pin" in str(redirect.content)
    assert len(pending) == 1


@pytest.mark.parametrize(
    ("tool_name", "entity_id"),
    [
        ("script__open_garage", "script.open_garage"),
        ("assist__script__open_garage", "script.open_garage"),
        ("intent__HassTurnOn", None),
        ("script__", None),
        ("add_automation", None),
    ],
)
def test_script_tool_names(tool_name: str, entity_id: str | None) -> None:
    """HA names a script's tool `script__<object id>`, maybe namespaced."""
    assert script_tool_entity_id(tool_name) == entity_id

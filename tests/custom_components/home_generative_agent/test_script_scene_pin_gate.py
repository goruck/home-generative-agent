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
from custom_components.home_generative_agent.agent.pin_messages import (
    describe_pending_action,
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
    # Open and close (1 | 2): a scene closing it calls close_cover.
    hass.states.async_set("cover.garage_door", "closed", {"supported_features": 3})
    # Position only: HA closes it with set_cover_position(0).
    hass.states.async_set("cover.side_gate", "closed", {"supported_features": 4})
    hass.states.async_set("cover.living_room_blinds", "closed")
    hass.states.async_set("lock.front_door", "locked")
    hass.states.async_set("light.porch", "off")
    assert await async_setup_component(
        hass,
        "group",
        {
            "group": {
                "doors": {"name": "Doors", "entities": ["lock.front_door"]},
                "outer": {"name": "Outer", "entities": ["group.doors"]},
                "loop": {"name": "Loop", "entities": ["group.loop", "light.porch"]},
                # Six deep, ending at a light: only the depth cap holds it.
                **{
                    f"chain{i}": {
                        "name": f"Chain {i}",
                        "entities": [f"group.chain{i + 1}" if i < 6 else "light.porch"],
                    }
                    for i in range(1, 7)
                },
            }
        },
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
                "announce": {
                    "alias": "Announce",
                    "sequence": [{"event": "arrived"}],
                },
                "pick": {
                    "alias": "Pick",
                    "sequence": [
                        {
                            "action": "{{ 'light' }}.turn_on",
                            "target": {"entity_id": "light.porch"},
                        }
                    ],
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
                {"name": "Open Up Nested", "entities": {"group.outer": "unlocked"}},
                {
                    "name": "Lock Then Open",
                    "entities": {
                        "group.doors": "locked",
                        "group.outer": "unlocked",
                    },
                },
                {"name": "Loop", "entities": {"group.loop": "on"}},
                {"name": "Too Deep", "entities": {"group.chain1": "on"}},
                {
                    "name": "Half Open",
                    "entities": {
                        "cover.garage_door": {"state": "closed", "current_position": 60}
                    },
                },
                {
                    "name": "Shut Tight",
                    "entities": {
                        "cover.garage_door": {"state": "closed", "current_position": 0}
                    },
                },
                {"name": "Locking Up", "entities": {"lock.front_door": "locking"}},
                {"name": "Shut Gate", "entities": {"cover.side_gate": "closed"}},
                {"name": "Stop Garage", "entities": {"script.open_garage": "off"}},
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


@pytest.mark.parametrize(
    "tool_name", ["script__evening", "script__announce", "script__pick"]
)
async def test_a_script_step_that_cannot_be_followed_is_held(
    home: HomeAssistant, tool_name: str
) -> None:
    """As in an automation: another script, an event, a templated action."""
    assert _is_critical_action({}, RULES, tool_name, home)


async def test_a_rule_naming_a_renamed_script_still_holds_it(
    home: HomeAssistant,
) -> None:
    """HA's tool calls the script by its key, which a rename does not change."""
    er.async_get(home).async_update_entity(
        "script.porch_lights", new_entity_id="script.front_lights"
    )
    await home.async_block_till_done()
    rules = [{"domain": "script", "service": "porch_lights"}]
    assert _is_critical_action({}, rules, "script__front_lights", home)
    assert _is_critical_action({"name": "Porch Lights"}, rules, "HassTurnOn", home)


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
        # HA cannot reproduce a script state: the scene does not run it.
        ("Leave", False),
        ("Stop Garage", False),
        ("Open Up", True),
        ("Open Up Nested", True),
        # The doors group comes first as locked; unlocked must still count.
        ("Lock Then Open", True),
        ("Loop", False),
        ("Too Deep", True),
        # A position is set with set_cover_position, which a rule guards.
        ("Half Open", True),
        ("Shut Tight", True),
        ("Locking Up", False),
        ("Shut Gate", True),
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


@pytest.mark.parametrize(
    ("platform", "critical"),
    [
        # A lighting system's scene only sets its lights.
        ("hue", False),
        # deCONZ can expose a door lock, so its scenes are not light-only.
        ("deconz", True),
        # A hub's scene can open a gate or unlock a door, and cannot be read.
        ("fibaro", True),
        ("overkiz", True),
        ("mqtt", True),
        # One of Home Assistant's own whose config is not loaded.
        ("homeassistant", True),
    ],
)
async def test_a_scene_stored_by_another_integration(
    home: HomeAssistant, platform: str, *, critical: bool
) -> None:
    """Only a light-only integration's scene runs without being read."""
    er.async_get(home).async_get_or_create(
        "scene", platform, "relax", suggested_object_id="relax", original_name="Relax"
    )
    home.states.async_set("scene.relax", "unknown", {"friendly_name": "Relax"})
    assert _is_critical_action({"name": "Relax"}, RULES, "HassTurnOn", home) is critical


async def test_a_script_or_scene_that_cannot_be_read_is_held(
    home: HomeAssistant,
) -> None:
    """With no config behind the entity, what it does is unknown."""
    home.states.async_set("scene.mystery", "unknown", {"friendly_name": "Mystery"})
    assert _is_critical_action({"name": "Mystery"}, RULES, "HassTurnOn", home)
    assert _is_critical_action({}, RULES, "script__not_loaded", home)


async def test_turning_on_every_scene_checks_each_one(home: HomeAssistant) -> None:
    """A domain-wide call reaches Arrive Home, which unlocks the front door."""
    assert _is_critical_action({"domain": "scene"}, RULES, "HassTurnOn", home)


def test_without_hass_a_scene_or_script_is_held() -> None:
    """With nothing to read the config from, it is held."""
    for entity_id in ("scene.good_night", "script.porch_lights"):
        assert _is_critical_action({"entity_id": entity_id}, RULES, "HassTurnOn", None)


@pytest.mark.parametrize(
    ("args", "critical"),
    [
        # HA reads these targets as every script, a list, a registry id, an
        # area: which scripts run is unknown, so the call is held.
        ({"domain": "script", "service": "turn_on", "entity_id": "all"}, True),
        (
            {
                "domain": "script",
                "service": "turn_on",
                "entity_id": "script.porch_lights, script.open_garage",
            },
            True,
        ),
        (
            {
                "domain": "script",
                "service": "turn_on",
                "entity_id": "0123456789abcdef0123456789abcdef",
            },
            True,
        ),
        (
            {
                "domain": "script",
                "service": "turn_on",
                "entity_id": "script.porch_lights",
                "area_id": "garage",
            },
            True,
        ),
        (
            {
                "domain": "script",
                "service": "turn_on",
                "target": {"entity_id": "script.open_garage"},
            },
            True,
        ),
        (
            {
                "domain": "script",
                "service": "turn_on",
                "target": {"entity_id": "script.porch_lights"},
            },
            False,
        ),
        (
            {
                "domain": "script",
                "service": "turn_on",
                "entity_ids": ["script.open_garage"],
            },
            True,
        ),
        (
            {
                "domain": "script",
                "service": "toggle",
                "entity_id": "script.open_garage",
            },
            True,
        ),
        ({"domain": "homeassistant", "service": "turn_on", "area_id": "garage"}, True),
        (
            {
                "domain": "homeassistant",
                "service": "turn_on",
                "target": {"entity_id": "script.open_garage"},
            },
            True,
        ),
        (
            {
                "domain": "homeassistant",
                "service": "turn_on",
                "entity_id": "group.doors",
            },
            True,
        ),
        (
            {
                "domain": "homeassistant",
                "service": "turn_on",
                "entity_id": "scene.arrive_home",
            },
            True,
        ),
        # HA forwards this to cover.toggle, which opens a closed garage door.
        (
            {
                "domain": "homeassistant",
                "service": "toggle",
                "entity_id": "cover.garage_door",
            },
            True,
        ),
        (
            {
                "domain": "scene",
                "service": "create",
                "scene_id": "x",
                "entities": {"lock.front_door": "unlocked"},
            },
            True,
        ),
        (
            {
                "domain": "scene",
                "service": "create",
                "scene_id": "x",
                "snapshot_entities": ["lock.front_door"],
            },
            True,
        ),
        ({"domain": "scene", "service": "apply", "data": "bogus"}, True),
        # Runs an automation's actions, which are not opened here.
        (
            {
                "domain": "automation",
                "service": "trigger",
                "entity_id": "automation.arrive",
            },
            True,
        ),
        ({"domain": "light", "service": "turn_on", "area_id": "garage"}, False),
    ],
)
async def test_a_service_tool_reads_targets_the_way_ha_does(
    home: HomeAssistant, args: dict[str, Any], *, critical: bool
) -> None:
    """`all`, lists, ids, areas and nested targets are not taken at face value."""
    assert _is_critical_action(args, RULES, "mcp__call_service", home) is critical


@pytest.mark.parametrize(
    "args",
    [
        {"domain": "script", "service": "turn_on", "entity_id": "script.porch_lights"},
        {"domain": "script", "service": "toggle", "entity_id": "script.porch_lights"},
        {"domain": "script", "service": "porch_lights"},
        {
            "domain": "homeassistant",
            "service": "turn_on",
            "entity_id": "script.porch_lights",
        },
    ],
)
async def test_a_rule_naming_a_script_covers_a_service_tool(
    home: HomeAssistant, args: dict[str, Any]
) -> None:
    """A rule for the script's own service holds every way of running it."""
    rules = [{"domain": "script", "service": "porch_lights"}]
    assert _is_critical_action(args, rules, "mcp__call_service", home)


async def test_a_service_tool_finds_a_renamed_script_by_its_key(
    home: HomeAssistant,
) -> None:
    """`script.<key>` runs the script with that key, whatever its entity id."""
    entities = er.async_get(home)
    entities.async_update_entity(
        "script.porch_lights", new_entity_id="script.front_lights"
    )
    entities.async_update_entity(
        "script.open_garage", new_entity_id="script.garage_helper"
    )
    await home.async_block_till_done()
    assert not _is_critical_action(
        {"domain": "script", "service": "porch_lights"},
        RULES,
        "mcp__call_service",
        home,
    )
    assert _is_critical_action(
        {"domain": "script", "service": "open_garage"},
        RULES,
        "mcp__call_service",
        home,
    )


@pytest.mark.parametrize(
    ("args", "critical"),
    [
        (
            {
                "domain": "script",
                "service": "turn_on",
                "entity_id": "script.open_garage",
            },
            True,
        ),
        (
            {
                "domain": "script",
                "service": "turn_on",
                "entity_id": "script.porch_lights",
            },
            False,
        ),
        ({"domain": "script", "service": "turn_on", "area": "Garage"}, True),
        ({"domain": "script", "service": "open_garage"}, True),
        ({"domain": "script", "service": "porch_lights"}, False),
        (
            {
                "domain": "script",
                "service": "turn_off",
                "entity_id": "script.open_garage",
            },
            False,
        ),
        (
            {"domain": "scene", "service": "turn_on", "entity_id": "scene.arrive_home"},
            True,
        ),
        (
            {"domain": "scene", "service": "turn_on", "entity_id": "scene.good_night"},
            False,
        ),
        (
            {
                "domain": "scene",
                "service": "apply",
                "entities": {"lock.front_door": "unlocked"},
            },
            True,
        ),
        (
            {
                "domain": "scene",
                "service": "apply",
                "data": {"entities": {"lock.front_door": "unlocked"}},
            },
            True,
        ),
        (
            {
                "domain": "scene",
                "service": "apply",
                "entities": {"light.porch": "on"},
            },
            False,
        ),
        ({"domain": "scene", "service": "apply"}, True),
        (
            {
                "domain": "homeassistant",
                "service": "turn_on",
                "entity_id": "script.open_garage",
            },
            True,
        ),
        (
            {
                "domain": "homeassistant",
                "service": "turn_on",
                "entity_id": "light.porch",
            },
            False,
        ),
    ],
)
async def test_a_service_tool_checks_what_a_script_or_scene_does(
    home: HomeAssistant, args: dict[str, Any], *, critical: bool
) -> None:
    """A tool that calls services by name reaches the same scripts and scenes."""
    assert _is_critical_action(args, RULES, "mcp__call_service", home) is critical


async def test_the_pin_line_names_the_script_not_its_arguments(
    home: HomeAssistant,
) -> None:
    """Script fields are the model's; the line comes from Home Assistant."""
    line = describe_pending_action(
        home,
        {
            "tool_name": "assist__script__let_in",
            "tool_args": {"service": "turn on", "name": "Porch light"},
        },
    )
    assert "Let In" in line
    assert "Porch" not in line


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

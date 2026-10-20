# ruff: noqa: S101
"""
Template entities and input buttons are PIN-gated by what they run (#780).

A template switch runs its `turn_on` action script when turned on, a template
button its `press` script when pressed, and an input button starts the
automations that trigger on it. The gate matched only the call the agent made,
so a template button whose press opened the garage door ran with no PIN. And a
template garage cover whose `open_cover` switches a relay left the relay
itself unguarded, as a switch_as_x cover did before #794.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

import pytest
from homeassistant.helpers import entity_registry as er
from homeassistant.setup import async_setup_component

from custom_components.home_generative_agent.agent.graph import _is_critical_action
from custom_components.home_generative_agent.agent.rule_entities import (
    expand_critical_rules,
    relay_wrappers_reached,
)
from custom_components.home_generative_agent.agent.tools import (
    _maybe_require_pin_for_automation,
)
from custom_components.home_generative_agent.const import (
    CONF_CRITICAL_ACTION_PIN_ENABLED,
    CONF_CRITICAL_ACTION_PIN_HASH,
    CONF_CRITICAL_ACTION_PIN_SALT,
    RECOMMENDED_CRITICAL_ACTIONS,
)
from custom_components.home_generative_agent.core.utils import hash_pin

if TYPE_CHECKING:
    from homeassistant.core import HomeAssistant

RULES = RECOMMENDED_CRITICAL_ACTIONS


def _call(action: str, entity_id: str) -> dict[str, Any]:
    return {"action": action, "target": {"entity_id": entity_id}}


@pytest.fixture
async def home(hass: HomeAssistant) -> HomeAssistant:
    """Set up a garage door, a front door lock, relays, template entities."""
    hass.states.async_set("cover.garage_door", "closed", {"supported_features": 3})
    hass.states.async_set("lock.front_door", "locked")
    hass.states.async_set("light.porch", "off")
    for relay in ("garage_relay", "side_relay", "spare_relay"):
        hass.states.async_set(f"switch.{relay}", "off")
    assert await async_setup_component(
        hass,
        "template",
        {
            "template": [
                {
                    "switch": [
                        {
                            "name": "Garage Opener",
                            "turn_on": [_call("cover.open_cover", "cover.garage_door")],
                            "turn_off": [_call("light.turn_off", "light.porch")],
                        },
                        {
                            "name": "Porch Switch",
                            "turn_on": [_call("light.turn_on", "light.porch")],
                            "turn_off": [_call("light.turn_off", "light.porch")],
                        },
                        {
                            # Opens the garage through another template switch.
                            "name": "Garage Helper",
                            "turn_on": [
                                _call("switch.turn_on", "switch.garage_opener")
                            ],
                            "turn_off": [],
                        },
                    ],
                    "button": [
                        {
                            "name": "Open Garage Button",
                            "press": [_call("cover.open_cover", "cover.garage_door")],
                        },
                        {
                            "name": "Chime",
                            "press": [_call("light.turn_on", "light.porch")],
                        },
                    ],
                    "cover": [
                        {
                            "name": "Barn Door",
                            "device_class": "garage",
                            "open_cover": [
                                _call("switch.turn_on", "switch.garage_relay")
                            ],
                            "close_cover": [
                                _call("switch.turn_on", "switch.garage_relay")
                            ],
                        },
                    ],
                    "lock": [
                        {
                            "name": "Side Door",
                            "state": "{{ is_state('switch.side_relay', 'off') }}",
                            "lock": [_call("switch.turn_off", "switch.side_relay")],
                            "unlock": [_call("switch.turn_on", "switch.side_relay")],
                        },
                    ],
                }
            ]
        },
    )
    assert await async_setup_component(
        hass,
        "input_button",
        {
            "input_button": {
                "doorbell": {"name": "Doorbell"},
                "plain": {"name": "Plain"},
            }
        },
    )
    assert await async_setup_component(
        hass,
        "automation",
        {
            "automation": [
                {
                    "alias": "Let in on doorbell",
                    "triggers": [
                        {"trigger": "state", "entity_id": "input_button.doorbell"}
                    ],
                    "actions": [_call("lock.unlock", "lock.front_door")],
                },
                {
                    "alias": "Porch on motion",
                    "triggers": [{"trigger": "state", "entity_id": "light.porch"}],
                    "actions": [_call("lock.unlock", "lock.front_door")],
                },
            ]
        },
    )
    assert await async_setup_component(
        hass,
        "script",
        {
            "script": {
                "flip_opener": {
                    "alias": "Flip Opener",
                    "sequence": [_call("switch.turn_on", "switch.garage_opener")],
                },
                "flip_porch": {
                    "alias": "Flip Porch",
                    "sequence": [_call("switch.turn_on", "switch.porch_switch")],
                },
            }
        },
    )
    assert await async_setup_component(
        hass,
        "scene",
        {
            "scene": [
                {"name": "Opener On", "entities": {"switch.garage_opener": "on"}},
                {"name": "Opener Off", "entities": {"switch.garage_opener": "off"}},
            ]
        },
    )
    await hass.async_block_till_done()
    return hass


@pytest.mark.parametrize(
    ("tool", "name", "critical"),
    [
        # The examples from the issue.
        ("HassTurnOn", "Garage Opener", True),
        ("HassTurnOn", "Open Garage Button", True),
        # Its turn_off action only turns a light off.
        ("HassTurnOff", "Garage Opener", False),
        ("HassTurnOn", "Porch Switch", False),
        ("HassTurnOn", "Chime", False),
        # Through a second template switch.
        ("HassTurnOn", "Garage Helper", True),
        # An input button starts the automations triggering on it.
        ("HassTurnOn", "Doorbell", True),
        ("HassTurnOn", "Plain", False),
    ],
)
async def test_acting_on_a_template_entity_checks_what_it_runs(
    home: HomeAssistant, tool: str, name: str, *, critical: bool
) -> None:
    """A template switch or button that opens the garage door asks for the PIN."""
    assert (
        _is_critical_action({"name": name}, RULES, f"intent__{tool}", home) is critical
    )


@pytest.mark.parametrize(
    ("args", "critical"),
    [
        (
            {
                "domain": "switch",
                "service": "turn_on",
                "entity_id": "switch.garage_opener",
            },
            True,
        ),
        (
            {
                "domain": "homeassistant",
                "service": "turn_on",
                "entity_id": "switch.garage_opener",
            },
            True,
        ),
        (
            {
                "domain": "switch",
                "service": "toggle",
                "entity_id": "switch.garage_opener",
            },
            True,
        ),
        (
            {
                "domain": "switch",
                "service": "turn_on",
                "entity_id": "switch.porch_switch",
            },
            False,
        ),
        (
            {
                "domain": "switch",
                "service": "turn_off",
                "entity_id": "switch.garage_opener",
            },
            False,
        ),
    ],
)
async def test_a_service_tool_checks_what_a_template_entity_runs(
    home: HomeAssistant, args: dict[str, Any], *, critical: bool
) -> None:
    """A tool that calls services by name reaches the same action scripts."""
    assert _is_critical_action(args, RULES, "mcp__call_service", home) is critical


@pytest.mark.parametrize(
    ("tool_name", "critical"),
    [("script__flip_opener", True), ("script__flip_porch", False)],
)
async def test_a_script_step_on_a_template_switch_is_opened(
    home: HomeAssistant, tool_name: str, *, critical: bool
) -> None:
    """A script turning on a template switch runs that switch's actions."""
    assert _is_critical_action({}, RULES, tool_name, home) is critical


@pytest.mark.parametrize(
    ("name", "critical"), [("Opener On", True), ("Opener Off", False)]
)
async def test_a_scene_setting_a_template_switch_is_opened(
    home: HomeAssistant, name: str, *, critical: bool
) -> None:
    """A scene setting a template switch on runs its turn_on action."""
    assert (
        _is_critical_action({"name": name}, RULES, "intent__HassTurnOn", home)
        is critical
    )


async def test_a_template_entity_that_cannot_be_read_is_held(
    home: HomeAssistant,
) -> None:
    """Registered by the Template integration but not loaded: unknown actions."""
    er.async_get(home).async_get_or_create(
        "switch", "template", "ghost", suggested_object_id="ghost"
    )
    home.states.async_set("switch.ghost", "off", {"friendly_name": "Ghost"})
    args = {"domain": "switch", "service": "turn_on", "entity_id": "switch.ghost"}
    assert _is_critical_action(args, RULES, "mcp__call_service", home)


async def test_an_ordinary_switch_is_not_opened(home: HomeAssistant) -> None:
    """Only template entities run stored actions; a plain switch is its call."""
    args = {"domain": "switch", "service": "turn_on", "entity_id": "switch.spare_relay"}
    assert not _is_critical_action(args, RULES, "mcp__call_service", home)


@pytest.mark.parametrize(
    ("entity_id", "service", "critical"),
    [
        # Under the template garage cover: every call moves the door.
        ("switch.garage_relay", "turn_on", True),
        ("switch.garage_relay", "turn_off", True),
        # Under the template lock's unlock action.
        ("switch.side_relay", "turn_on", True),
        ("switch.spare_relay", "turn_on", False),
    ],
)
async def test_the_relay_a_template_cover_or_lock_drives_is_held(
    home: HomeAssistant, entity_id: str, service: str, *, critical: bool
) -> None:
    """As for the relay under a switch_as_x cover (#794)."""
    rules = expand_critical_rules(home, RULES)
    args = {"domain": "switch", "service": service, "entity_id": entity_id}
    assert _is_critical_action(args, rules, "mcp__call_service", home) is critical


async def test_the_pin_line_names_what_a_driven_relay_moves(
    home: HomeAssistant,
) -> None:
    """The prompt can say the relay controls the template cover and lock."""
    assert relay_wrappers_reached(home, ["switch.garage_relay"]) == ["cover.barn_door"]
    assert relay_wrappers_reached(home, ["switch.side_relay"]) == ["lock.side_door"]
    assert relay_wrappers_reached(home, ["switch.spare_relay"]) == []


@pytest.mark.parametrize(
    ("entity_id", "held"),
    [("switch.garage_opener", True), ("switch.porch_switch", False)],
)
async def test_an_automation_turning_on_a_template_switch_asks_for_the_pin(
    home: HomeAssistant, entity_id: str, *, held: bool
) -> None:
    """The automation the agent writes reaches the same action scripts."""
    hashed, salt = hash_pin("1234")
    cfg = cast(
        "Any",
        {
            "hass": home,
            "options": {
                CONF_CRITICAL_ACTION_PIN_ENABLED: True,
                CONF_CRITICAL_ACTION_PIN_HASH: hashed,
                CONF_CRITICAL_ACTION_PIN_SALT: salt,
            },
            "pending_actions": {},
            "user_id": "user1",
            "thread_id": "conv1",
        },
    )
    config = {
        "triggers": [{"trigger": "time", "at": "07:00:00"}],
        "actions": [_call("switch.turn_on", entity_id)],
    }
    challenge = _maybe_require_pin_for_automation(
        cfg=cfg, validated_config=config, ha_automation_config=config
    )
    assert (challenge is not None) is held

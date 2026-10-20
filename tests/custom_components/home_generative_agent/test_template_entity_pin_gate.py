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

from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, cast

import pytest
from homeassistant.helpers import area_registry as ar
from homeassistant.helpers import device_registry as dr
from homeassistant.helpers import entity_registry as er
from homeassistant.setup import async_setup_component
from pytest_homeassistant_custom_component.common import MockConfigEntry

from custom_components.home_generative_agent.agent import graph as graph_module
from custom_components.home_generative_agent.agent import (
    indirect_pin,
    rule_entities,
    template_actions,
)
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
    for relay in (
        "garage_relay",
        "side_relay",
        "spare_relay",
        "shed_relay",
        "bolt_relay",
        "latch_relay",
    ):
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
                            "unique_id": "garage_opener",
                            "turn_on": [_call("cover.open_cover", "cover.garage_door")],
                            "turn_off": [_call("light.turn_off", "light.porch")],
                        },
                        {
                            "name": "Porch Switch",
                            "turn_on": [_call("light.turn_on", "light.porch")],
                            "turn_off": [_call("light.turn_off", "light.porch")],
                        },
                        {
                            # A TV switch running a harmless script.
                            "name": "TV",
                            "turn_on": [{"action": "script.porch_on"}],
                            "turn_off": [],
                        },
                        {
                            # One running the garage script.
                            "name": "Garage Script Switch",
                            "turn_on": [_call("script.turn_on", "script.flip_opener")],
                            "turn_off": [],
                        },
                        {
                            # A REST command cannot be read: held.
                            "name": "Webhook",
                            "turn_on": [{"action": "rest_command.ping"}],
                            "turn_off": [],
                        },
                        {
                            # Two switches turning each other on; one opens.
                            "name": "Ping",
                            "turn_on": [_call("switch.turn_on", "switch.pong")],
                            "turn_off": [],
                        },
                        {
                            "name": "Pong",
                            "turn_on": [
                                _call("switch.turn_on", "switch.ping"),
                                _call("cover.open_cover", "cover.garage_door"),
                            ],
                            "turn_off": [],
                        },
                        {
                            "name": "Tick",
                            "turn_on": [_call("switch.turn_on", "switch.tock")],
                            "turn_off": [],
                        },
                        {
                            "name": "Tock",
                            "turn_on": [_call("switch.turn_on", "switch.tick")],
                            "turn_off": [],
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
                        {
                            # Its relay is switched through a script.
                            "name": "Shed Door",
                            "device_class": "garage",
                            "open_cover": [{"action": "script.pulse_shed"}],
                            "close_cover": [{"action": "script.pulse_shed"}],
                        },
                    ],
                    "light": [
                        {
                            # Turning it on is harmless; setting a level is not.
                            "name": "Garage Light",
                            "turn_on": [_call("light.turn_on", "light.porch")],
                            "turn_off": [_call("light.turn_off", "light.porch")],
                            "set_level": [
                                _call("cover.open_cover", "cover.garage_door")
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
                        {
                            # Locking and unlocking drive different relays.
                            "name": "Back Door",
                            "state": "{{ is_state('switch.bolt_relay', 'on') }}",
                            "lock": [_call("switch.turn_on", "switch.bolt_relay")],
                            "unlock": [_call("switch.turn_on", "switch.latch_relay")],
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
                "pulse_shed": {
                    "alias": "Pulse Shed",
                    "sequence": [_call("switch.turn_on", "switch.shed_relay")],
                },
                "porch_on": {
                    "alias": "Porch On",
                    "sequence": [_call("light.turn_on", "light.porch")],
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
    # The template opener sits in the garage area; the kitchen holds none.
    garage = ar.async_get(hass).async_create("Garage")
    assert garage.id == "garage"
    ar.async_get(hass).async_create("Kitchen")
    er.async_get(hass).async_update_entity("switch.garage_opener", area_id=garage.id)
    # A switch group holding the template opener.
    hass.states.async_set(
        "switch.all_openers", "off", {"entity_id": ["switch.garage_opener"]}
    )
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


@pytest.mark.parametrize(
    ("tool", "args", "critical"),
    [
        # A template light picks its script from the call's data: a level
        # runs set_level, so every script it has counts.
        ("HassLightSet", {"name": "Garage Light", "brightness": 100}, True),
        ("HassTurnOn", {"name": "Garage Light"}, True),
        # Two template switches turning each other on: followed, not looped.
        ("HassTurnOn", {"name": "Ping"}, True),
        ("HassTurnOn", {"name": "Tick"}, False),
    ],
)
async def test_a_template_entity_is_read_whatever_script_the_call_picks(
    home: HomeAssistant, tool: str, args: dict[str, Any], *, critical: bool
) -> None:
    """A light's level script and switches driving each other are screened."""
    assert _is_critical_action(args, RULES, f"intent__{tool}", home) is critical


@pytest.mark.parametrize(
    ("args", "critical"),
    [
        # An inline scene setting the template opener on.
        (
            {
                "domain": "scene",
                "service": "apply",
                "entities": {"switch.garage_opener": "on"},
            },
            True,
        ),
        (
            {
                "domain": "scene",
                "service": "apply",
                "entities": {"switch.porch_switch": "on"},
            },
            False,
        ),
        # An area holding the template opener is read as HA reads it.
        ({"domain": "switch", "service": "turn_on", "area_id": "garage"}, True),
        ({"domain": "switch", "service": "turn_on", "area_id": "kitchen"}, False),
        # `all` cannot be named ahead: held while a template switch could open.
        ({"domain": "switch", "service": "turn_on", "entity_id": "all"}, True),
        # No template fan runs anything: an area of fans is not held.
        ({"domain": "fan", "service": "turn_on", "area_id": "garage"}, False),
        # A switch group holding the template opener.
        (
            {
                "domain": "switch",
                "service": "turn_on",
                "entity_id": "switch.all_openers",
            },
            True,
        ),
    ],
)
async def test_a_service_tool_reaches_template_entities_indirectly(
    home: HomeAssistant, args: dict[str, Any], *, critical: bool
) -> None:
    """Inline scenes, areas, `all` and groups reach the same action scripts."""
    assert _is_critical_action(args, RULES, "mcp__call_service", home) is critical


def _pin_cfg(home: HomeAssistant) -> Any:
    hashed, salt = hash_pin("1234")
    return {
        "hass": home,
        "options": {
            CONF_CRITICAL_ACTION_PIN_ENABLED: True,
            CONF_CRITICAL_ACTION_PIN_HASH: hashed,
            CONF_CRITICAL_ACTION_PIN_SALT: salt,
        },
        "pending_actions": {},
        "user_id": "user1",
        "thread_id": "conv1",
    }


@pytest.mark.parametrize(
    ("action", "held"),
    [
        (_call("input_button.press", "input_button.doorbell"), True),
        (_call("button.press", "button.open_garage_button"), True),
        (
            {
                "action": "scene.apply",
                "data": {"entities": {"switch.garage_opener": "on"}},
            },
            True,
        ),
        ({"action": "switch.turn_on", "target": {"area_id": "garage"}}, True),
    ],
)
async def test_an_automation_reaching_stored_actions_asks_for_the_pin(
    home: HomeAssistant, action: dict[str, Any], *, held: bool
) -> None:
    """Presses, inline scenes and area targets in a written automation."""
    config = {"triggers": [{"trigger": "time", "at": "07:00:00"}], "actions": [action]}
    challenge = _maybe_require_pin_for_automation(
        cfg=_pin_cfg(home), validated_config=config, ha_automation_config=config
    )
    assert (challenge is not None) is held


async def test_a_device_action_on_a_template_switch_asks_for_the_pin(
    home: HomeAssistant,
) -> None:
    """A device action names the switch by registry id and runs its turn_on."""
    entry = er.async_get(home).async_get("switch.garage_opener")
    assert entry is not None
    action = {
        "device_id": "abc",
        "domain": "switch",
        "type": "turn_on",
        "entity_id": entry.id,
    }
    config = {"triggers": [{"trigger": "time", "at": "07:00:00"}], "actions": [action]}
    challenge = _maybe_require_pin_for_automation(
        cfg=_pin_cfg(home), validated_config=config, ha_automation_config=config
    )
    assert challenge is not None


def _automations(*triggers: Any) -> SimpleNamespace:
    """Return an automation component whose one automation unlocks a door."""
    return SimpleNamespace(
        entities=[
            SimpleNamespace(
                raw_config={"triggers": list(triggers)},
                action_script=SimpleNamespace(
                    sequence=[_call("lock.unlock", "lock.front_door")]
                ),
            )
        ]
    )


@pytest.mark.parametrize(
    ("trigger", "held"),
    [
        # A state_changed event filtered in its conditions may react.
        ({"trigger": "event", "event_type": "state_changed"}, True),
        # A templated filter, e.g. from trigger_variables.
        (
            {
                "trigger": "event",
                "event_type": "state_changed",
                "event_data": {"entity_id": "{{ button }}"},
            },
            True,
        ),
        # A template reading states generically can react to any press.
        (
            {
                "trigger": "template",
                "value_template": "{{ states.input_button | count > 0 }}",
            },
            True,
        ),
        # The new purpose-specific trigger aimed at an area.
        ({"trigger": "button.pressed", "target": {"area_id": "hall"}}, True),
        # Triggers that cannot react to this button: a press fires only
        # state_changed and call_service events.
        ({"trigger": "state", "entity_id": "light.porch"}, False),
        ({"trigger": "event", "event_type": "zha_event"}, False),
        ({"trigger": "template", "value_template": "{{ now().hour > 7 }}"}, False),
        ({"trigger": "time", "at": "07:00:00"}, False),
    ],
)
async def test_a_press_screens_automations_that_may_react(
    home: HomeAssistant,
    monkeypatch: pytest.MonkeyPatch,
    trigger: dict[str, Any],
    *,
    held: bool,
) -> None:
    """A trigger that names the button or cannot be tied down counts."""
    monkeypatch.setitem(home.data, "automation", _automations(trigger))
    assert (
        _is_critical_action({"name": "Plain"}, RULES, "intent__HassTurnOn", home)
        is held
    )


async def test_a_press_finds_a_trigger_naming_the_button_by_registry_id(
    home: HomeAssistant, monkeypatch: pytest.MonkeyPatch
) -> None:
    """HA state triggers accept a registry id in place of the entity id."""
    entry = er.async_get(home).async_get_or_create(
        "input_button",
        "test",
        "buzzer",
        suggested_object_id="buzzer",
        original_name="Buzzer",
    )
    home.states.async_set(entry.entity_id, "unknown", {"friendly_name": "Buzzer"})
    monkeypatch.setitem(
        home.data,
        "automation",
        _automations({"trigger": "state", "entity_id": [entry.id]}),
    )
    assert _is_critical_action({"name": "Buzzer"}, RULES, "intent__HassTurnOn", home)


async def test_an_automation_that_cannot_be_read_holds_the_press(
    home: HomeAssistant, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Fail closed: the press may start it."""
    broken = SimpleNamespace(raw_config=None, action_script=SimpleNamespace())
    monkeypatch.setitem(home.data, "automation", SimpleNamespace(entities=[broken]))
    assert _is_critical_action({"name": "Plain"}, RULES, "intent__HassTurnOn", home)


@pytest.mark.parametrize(
    ("rules", "critical"),
    [
        (RULES, True),
        # A rule of your own naming the template cover holds its relay.
        ([{"domain": "cover", "service": "open_cover", "entity_match": "shed"}], True),
        # One reaching another cover does not.
        ([{"domain": "cover", "service": "open_cover", "entity_match": "nope"}], False),
        # A lock rule does not reach a cover's relay.
        ([{"domain": "lock", "service": "unlock", "entity_match": "nope"}], False),
    ],
)
async def test_the_relay_behind_a_script_is_held(
    home: HomeAssistant, rules: list[dict[str, str]], *, critical: bool
) -> None:
    """`script.<key>` in a template cover's action is followed to the relay."""
    expanded = expand_critical_rules(home, rules)
    args = {"domain": "switch", "service": "turn_on", "entity_id": "switch.shed_relay"}
    assert _is_critical_action(args, expanded, "mcp__call_service", home) is critical


async def test_unreadable_action_scripts_hold_and_warn(
    home: HomeAssistant,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """If HA renames `_action_scripts`, a loaded template switch is held."""
    entity = home.data["switch"].get_entity("switch.porch_switch")
    monkeypatch.setattr(entity, "_action_scripts", None)
    args = {
        "domain": "switch",
        "service": "turn_on",
        "entity_id": "switch.porch_switch",
    }
    assert _is_critical_action(args, RULES, "mcp__call_service", home)
    assert "cannot read the actions of switch.porch_switch" in caplog.text


async def test_an_action_script_with_no_steps_list_is_held(
    home: HomeAssistant, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A script whose steps cannot be read holds the call."""
    entity = home.data["switch"].get_entity("switch.porch_switch")
    monkeypatch.setitem(
        entity._action_scripts, "turn_on", SimpleNamespace(sequence=None)
    )
    args = {
        "domain": "switch",
        "service": "turn_on",
        "entity_id": "switch.porch_switch",
    }
    assert _is_critical_action(args, RULES, "mcp__call_service", home)


async def test_the_safety_bound_holds_rather_than_passes(
    home: HomeAssistant, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Past the bound on entities followed, a call is held."""
    monkeypatch.setattr(indirect_pin, "_MAX_FOLLOWED_ENTITIES", 1)
    home.states.async_set(
        "switch.porch_group", "off", {"entity_id": ["switch.porch_switch"]}
    )
    args = {"domain": "switch", "service": "turn_on", "entity_id": "switch.porch_group"}
    assert _is_critical_action(args, RULES, "mcp__call_service", home)


@pytest.mark.parametrize(
    ("automation", "held"),
    [
        # Its trigger names the button but its steps cannot be read: held.
        (
            SimpleNamespace(
                raw_config={
                    "triggers": [
                        {"trigger": "state", "entity_id": "input_button.plain"}
                    ]
                },
                action_script=SimpleNamespace(sequence=None),
            ),
            True,
        ),
        # An automation that failed to set up does not run.
        (
            SimpleNamespace(
                raw_config={
                    "triggers": [
                        {"trigger": "state", "entity_id": "input_button.plain"}
                    ]
                },
                action_script=None,
            ),
            False,
        ),
        # Waiting for the button, then unlocking: the press resumes it.
        (
            SimpleNamespace(
                raw_config={
                    "triggers": [{"trigger": "time", "at": "07:00:00"}],
                    "actions": [
                        {
                            "wait_for_trigger": [
                                {"trigger": "state", "entity_id": "input_button.plain"}
                            ]
                        },
                        _call("lock.unlock", "lock.front_door"),
                    ],
                },
                action_script=SimpleNamespace(
                    sequence=[_call("lock.unlock", "lock.front_door")]
                ),
            ),
            True,
        ),
    ],
)
async def test_how_a_press_reads_each_automation(
    home: HomeAssistant,
    monkeypatch: pytest.MonkeyPatch,
    automation: SimpleNamespace,
    *,
    held: bool,
) -> None:
    """Unreadable steps hold; a failed automation is skipped; a wait counts."""
    monkeypatch.setitem(home.data, "automation", SimpleNamespace(entities=[automation]))
    assert (
        _is_critical_action({"name": "Plain"}, RULES, "intent__HassTurnOn", home)
        is held
    )


@pytest.mark.parametrize(
    ("trigger", "held"),
    [
        # A device trigger without an entity may be the button's device.
        ({"trigger": "device", "device_id": "abc", "domain": "button"}, True),
        (
            {
                "trigger": "device",
                "device_id": "abc",
                "entity_id": "light.porch",
                "domain": "light",
                "type": "turned_on",
            },
            False,
        ),
    ],
)
async def test_device_triggers(
    home: HomeAssistant,
    monkeypatch: pytest.MonkeyPatch,
    trigger: dict[str, Any],
    *,
    held: bool,
) -> None:
    """A device trigger counts unless it names another entity."""
    monkeypatch.setitem(home.data, "automation", _automations(trigger))
    assert (
        _is_critical_action({"name": "Plain"}, RULES, "intent__HassTurnOn", home)
        is held
    )


@pytest.mark.parametrize(
    ("tool", "args", "critical"),
    [
        # An ordinary light is its call: no stored actions.
        ("HassLightSet", {"name": "porch", "brightness": 50}, False),
        # No device named: HA sets every light, the template one included.
        ("HassLightSet", {"brightness": 50}, True),
        # An intent the gate does not know, aimed at no device: held.
        ("HassFrobnicate", {}, True),
        # Reading a template switch's state runs nothing.
        ("HassGetState", {"name": "Garage Opener"}, False),
    ],
)
async def test_device_intents(
    home: HomeAssistant, tool: str, args: dict[str, Any], *, critical: bool
) -> None:
    """Every device intent opens what it targets; read-only intents do not."""
    assert _is_critical_action(args, RULES, f"intent__{tool}", home) is critical


async def test_a_device_intent_whose_targets_cannot_be_read_is_held(
    home: HomeAssistant, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Fail closed when the matcher cannot tell."""
    monkeypatch.setattr(graph_module, "intent_target_states", lambda *_a, **_k: None)
    assert _is_critical_action(
        {"name": "porch", "brightness": 50}, RULES, "intent__HassLightSet", home
    )


@pytest.mark.parametrize(
    ("args", "critical"),
    [
        # A target merged in from a data block, as HA merges it.
        (
            {
                "domain": "switch",
                "service": "turn_on",
                "target": {"entity_id": "switch.porch_switch"},
                "data_template": {"area_id": "garage"},
            },
            True,
        ),
        (
            {
                "domain": "switch",
                "service": "turn_on",
                "target": {"entity_id": "switch.porch_switch"},
                "service_data": {"entity_id": "{{ 'switch.garage_opener' }}"},
            },
            True,
        ),
        # A snapshot lists entities with no state: every script counts.
        (
            {
                "domain": "scene",
                "service": "create",
                "scene_id": "x",
                "snapshot_entities": ["switch.garage_opener"],
            },
            True,
        ),
        (
            {
                "domain": "scene",
                "service": "create",
                "scene_id": "x",
                "snapshot_entities": ["switch.porch_switch"],
            },
            False,
        ),
    ],
)
async def test_targets_from_every_place_ha_reads_them(
    home: HomeAssistant, args: dict[str, Any], *, critical: bool
) -> None:
    """Data blocks, templated targets and scene snapshots."""
    assert _is_critical_action(args, RULES, "mcp__call_service", home) is critical


async def test_a_generic_call_on_an_area_in_an_automation_is_held(
    home: HomeAssistant,
) -> None:
    """`homeassistant.turn_on` on the garage reaches the template opener."""
    action = {"action": "homeassistant.turn_on", "target": {"area_id": "garage"}}
    config = {"triggers": [{"trigger": "time", "at": "07:00:00"}], "actions": [action]}
    assert (
        _maybe_require_pin_for_automation(
            cfg=_pin_cfg(home), validated_config=config, ha_automation_config=config
        )
        is not None
    )


@pytest.mark.parametrize(
    ("entity_id", "critical"),
    [("switch.latch_relay", True), ("switch.bolt_relay", False)],
)
async def test_only_a_lock_s_unlocking_relay_is_held(
    home: HomeAssistant, entity_id: str, *, critical: bool
) -> None:
    """The relay a template lock locks with is not held; the one it unlocks with is."""
    rules = expand_critical_rules(home, RULES)
    args = {"domain": "switch", "service": "turn_on", "entity_id": entity_id}
    assert _is_critical_action(args, rules, "mcp__call_service", home) is critical
    assert (relay_wrappers_reached(home, [entity_id]) == ["lock.back_door"]) is critical


async def test_a_template_cover_whose_actions_cannot_be_read_stays_held(
    home: HomeAssistant, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Its relay loses its copy (logged); the cover itself is still held."""

    def _raise(*_args: Any, **_kwargs: Any) -> set[str]:
        raise RuntimeError

    monkeypatch.setattr(rule_entities, "driven_entities", _raise)
    rules = expand_critical_rules(home, RULES)
    args = {"domain": "cover", "service": "open_cover", "entity_id": "cover.barn_door"}
    assert _is_critical_action(args, rules, "mcp__call_service", home)


async def _template_cover(hass: HomeAssistant, open_action: dict[str, Any]) -> None:
    hass.states.async_set("lock.front_door", "locked")
    assert await async_setup_component(
        hass,
        "template",
        {
            "template": [
                {
                    "cover": [
                        {
                            "name": "Shed Door",
                            "device_class": "garage",
                            "open_cover": [open_action],
                            "close_cover": [open_action],
                        }
                    ]
                }
            ]
        },
    )
    await hass.async_block_till_done()


async def test_a_relay_switched_by_its_device_is_held(hass: HomeAssistant) -> None:
    """The UI target picker names the relay's device, not the relay."""
    entry = MockConfigEntry(domain="demo")
    entry.add_to_hass(hass)
    device = dr.async_get(hass).async_get_or_create(
        config_entry_id=entry.entry_id, identifiers={("demo", "relay")}
    )
    relay = er.async_get(hass).async_get_or_create(
        "switch",
        "demo",
        "relay",
        suggested_object_id="shed_relay",
        device_id=device.id,
        config_entry=entry,
    )
    hass.states.async_set(relay.entity_id, "off")
    await _template_cover(
        hass, {"action": "switch.turn_on", "target": {"device_id": device.id}}
    )
    rules = expand_critical_rules(hass, RULES)
    args = {"domain": "switch", "service": "turn_on", "entity_id": relay.entity_id}
    assert _is_critical_action(args, rules, "mcp__call_service", hass)


async def test_a_relay_set_by_an_inline_scene_is_held(hass: HomeAssistant) -> None:
    """A template cover opening with `scene.apply` drives the relay it sets."""
    hass.states.async_set("switch.shed_relay", "off")
    await _template_cover(
        hass,
        {"action": "scene.apply", "data": {"entities": {"switch.shed_relay": "on"}}},
    )
    rules = expand_critical_rules(hass, RULES)
    args = {"domain": "switch", "service": "turn_on", "entity_id": "switch.shed_relay"}
    assert _is_critical_action(args, rules, "mcp__call_service", hass)


async def test_a_light_wrapping_a_template_switch_is_opened(
    home: HomeAssistant,
) -> None:
    """A switch_as_x light over the template opener runs the opener's actions."""
    entry = MockConfigEntry(
        domain="switch_as_x",
        options={"entity_id": "switch.garage_opener", "invert": False},
    )
    entry.add_to_hass(home)
    er.async_get(home).async_get_or_create(
        "light",
        "switch_as_x",
        "opener_light",
        suggested_object_id="opener_light",
        config_entry=entry,
    )
    args = {"domain": "light", "service": "turn_on", "entity_id": "light.opener_light"}
    assert _is_critical_action(args, RULES, "mcp__call_service", home)


@pytest.mark.parametrize(
    ("name", "critical"),
    [
        # A script a template switch runs is opened, not held wholesale.
        ("TV", False),
        ("Garage Script Switch", True),
        # A REST command runs configuration that cannot be read.
        ("Webhook", True),
    ],
)
async def test_what_a_template_switch_runs_is_opened(
    home: HomeAssistant, name: str, *, critical: bool
) -> None:
    """'Turn on the TV' does not ask for the PIN when the TV script is harmless."""
    assert (
        _is_critical_action({"name": name}, RULES, "intent__HassTurnOn", home)
        is critical
    )


@pytest.mark.parametrize(
    ("trigger", "variables", "held"),
    [
        # A template naming another entity cannot react to a press.
        (
            {
                "trigger": "template",
                "value_template": "{{ states('person.me') == 'home' }}",
            },
            {},
            False,
        ),
        # Reading a group or an area's entities can.
        (
            {
                "trigger": "template",
                "value_template": "{{ expand('group.buttons') | count > 0 }}",
            },
            {},
            True,
        ),
        # A template reading a variable that names the button.
        (
            {"trigger": "template", "value_template": "{{ is_state(btn, 'x') }}"},
            {"trigger_variables": {"btn": "input_button.plain"}},
            True,
        ),
        # A remote's device trigger is not a press.
        (
            {
                "trigger": "device",
                "device_id": "remote",
                "domain": "zha",
                "type": "remote_button_short_press",
            },
            {},
            False,
        ),
    ],
)
async def test_which_templates_and_devices_react_to_a_press(
    home: HomeAssistant,
    monkeypatch: pytest.MonkeyPatch,
    trigger: dict[str, Any],
    variables: dict[str, Any],
    *,
    held: bool,
) -> None:
    """Generic templates and variables count; other entities and remotes do not."""
    automation = SimpleNamespace(
        raw_config={"triggers": [trigger], **variables},
        action_script=SimpleNamespace(
            sequence=[_call("lock.unlock", "lock.front_door")]
        ),
    )
    monkeypatch.setitem(home.data, "automation", SimpleNamespace(entities=[automation]))
    assert (
        _is_critical_action({"name": "Plain"}, RULES, "intent__HassTurnOn", home)
        is held
    )


async def test_a_press_resumes_a_wait_template(
    home: HomeAssistant, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An automation waiting on the button's state continues into the unlock."""
    automation = SimpleNamespace(
        raw_config={
            "triggers": [{"trigger": "time", "at": "07:00:00"}],
            "actions": [
                {"wait_template": "{{ states('input_button.plain') != before }}"},
                _call("lock.unlock", "lock.front_door"),
            ],
        },
        action_script=SimpleNamespace(
            sequence=[_call("lock.unlock", "lock.front_door")]
        ),
    )
    monkeypatch.setitem(home.data, "automation", SimpleNamespace(entities=[automation]))
    assert _is_critical_action({"name": "Plain"}, RULES, "intent__HassTurnOn", home)


async def test_a_press_resumes_a_waiting_script(
    home: HomeAssistant, monkeypatch: pytest.MonkeyPatch
) -> None:
    """'Wait for the confirm button, then unlock' run by a script."""
    sequence = [
        {"wait_for_trigger": [{"trigger": "state", "entity_id": "input_button.plain"}]},
        _call("lock.unlock", "lock.front_door"),
    ]
    script = SimpleNamespace(
        raw_config={"sequence": sequence}, script=SimpleNamespace(sequence=sequence)
    )
    monkeypatch.setitem(home.data, "script", SimpleNamespace(entities=[script]))
    assert _is_critical_action({"name": "Plain"}, RULES, "intent__HassTurnOn", home)


@pytest.mark.parametrize(
    ("blocks", "held"),
    [
        # A trigger-based template block runs its actions when triggered.
        (
            [
                SimpleNamespace(
                    config={
                        "triggers": [
                            {"trigger": "state", "entity_id": "input_button.plain"}
                        ],
                        "actions": [_call("lock.unlock", "lock.front_door")],
                    }
                )
            ],
            True,
        ),
        # One that cannot be read holds the press.
        ("not a list", True),
    ],
)
async def test_a_press_starts_template_blocks(
    home: HomeAssistant,
    monkeypatch: pytest.MonkeyPatch,
    blocks: Any,
    *,
    held: bool,
) -> None:
    """Template blocks with `actions:` react to a press like automations."""
    monkeypatch.setitem(home.data, "template", blocks)
    assert (
        _is_critical_action({"name": "Plain"}, RULES, "intent__HassTurnOn", home)
        is held
    )


def test_a_name_matches_whole_entity_ids_only() -> None:
    """`input_button.door` is not named by `input_button.doorbell`."""
    assert not template_actions._names_in(
        ["input_button.doorbell"], ["input_button.door"]
    )
    assert template_actions._names_in(
        ["{{ is_state('input_button.door', 'x') }}"], ["input_button.door"]
    )


async def test_a_template_alarm_panel_s_disarm_is_opened(hass: HomeAssistant) -> None:
    """A template panel enforces no code of its own; its actions are checked."""
    hass.states.async_set("lock.front_door", "locked")
    assert await async_setup_component(
        hass,
        "template",
        {
            "template": [
                {
                    "alarm_control_panel": [
                        {
                            "name": "House",
                            "code_arm_required": False,
                            "code_format": "no_code",
                            "disarm": [_call("lock.unlock", "lock.front_door")],
                            "arm_away": [],
                        }
                    ]
                }
            ]
        },
    )
    await hass.async_block_till_done()
    args = {"entity_id": "alarm_control_panel.house", "state": "disarm"}
    assert _is_critical_action(args, RULES, "alarm_control", hass)
    assert _is_critical_action({"name": "House"}, RULES, "alarm_control", hass)


async def test_a_relay_set_by_a_stored_scene_is_held(hass: HomeAssistant) -> None:
    """A template cover opening with `scene.turn_on` drives what the scene sets."""
    hass.states.async_set("switch.shed_relay", "off")
    assert await async_setup_component(
        hass,
        "scene",
        {"scene": [{"name": "Open Shed", "entities": {"switch.shed_relay": "on"}}]},
    )
    await _template_cover(hass, _call("scene.turn_on", "scene.open_shed"))
    rules = expand_critical_rules(hass, RULES)
    args = {"domain": "switch", "service": "turn_on", "entity_id": "switch.shed_relay"}
    assert _is_critical_action(args, rules, "mcp__call_service", hass)


async def test_a_relay_pulsed_by_a_pressed_button_is_held(hass: HomeAssistant) -> None:
    """A template cover pressing a button whose automation pulses the relay."""
    hass.states.async_set("switch.pulse_relay", "off")
    assert await async_setup_component(
        hass, "input_button", {"input_button": {"pulse": {"name": "Pulse"}}}
    )
    assert await async_setup_component(
        hass,
        "automation",
        {
            "automation": [
                {
                    "alias": "Pulse relay",
                    "triggers": [
                        {"trigger": "state", "entity_id": "input_button.pulse"}
                    ],
                    "actions": [_call("switch.turn_on", "switch.pulse_relay")],
                }
            ]
        },
    )
    await _template_cover(hass, _call("input_button.press", "input_button.pulse"))
    rules = expand_critical_rules(hass, RULES)
    args = {"domain": "switch", "service": "turn_on", "entity_id": "switch.pulse_relay"}
    assert _is_critical_action(args, rules, "mcp__call_service", hass)


async def test_all_reaches_a_wrapped_template_switch(home: HomeAssistant) -> None:
    """`fan.turn_on all` reaches a fan made from the template opener."""
    entry = MockConfigEntry(
        domain="switch_as_x",
        options={"entity_id": "switch.garage_opener", "invert": False},
    )
    entry.add_to_hass(home)
    er.async_get(home).async_get_or_create(
        "fan",
        "switch_as_x",
        "opener_fan",
        suggested_object_id="opener_fan",
        config_entry=entry,
    )
    args = {"domain": "fan", "service": "turn_on", "entity_id": "all"}
    assert _is_critical_action(args, RULES, "mcp__call_service", home)


async def test_a_template_entry_left_over_from_removed_config_is_ignored(
    home: HomeAssistant,
) -> None:
    """Once the Template integration has loaded, an unloaded entry cannot run."""
    er.async_get(home).async_get_or_create(
        "fan", "template", "gone", suggested_object_id="gone"
    )
    args = {"domain": "fan", "service": "turn_on", "entity_id": "all"}
    assert not _is_critical_action(args, RULES, "mcp__call_service", home)

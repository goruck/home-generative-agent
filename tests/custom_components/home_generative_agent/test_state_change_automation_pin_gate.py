# ruff: noqa: S101
"""
A change that starts an automation is PIN-gated by what the automation does (#814).

The agent turning on an input boolean, a switch or a light can start an
automation the user wrote, and the gate checked only the call: turning on
`input_boolean.let_in`, whose automation unlocks the front door, ran with no
PIN. Each entity a call changes is now checked against the automations that
may react to it.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

import pytest
from homeassistant.setup import async_setup_component

from custom_components.home_generative_agent.agent.graph import _is_critical_action
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


def _unlock() -> list[dict[str, Any]]:
    return [_call("lock.unlock", "lock.front_door")]


@pytest.fixture
async def home(hass: HomeAssistant) -> HomeAssistant:
    """Set up a lock, helpers, switches, lights and the automations on them."""
    hass.states.async_set("lock.front_door", "locked")
    hass.states.async_set("switch.door_buzzer", "off", {"friendly_name": "Buzzer"})
    hass.states.async_set("light.porch", "off", {"friendly_name": "Porch"})
    hass.states.async_set("light.kitchen", "off", {"friendly_name": "Kitchen"})
    hass.states.async_set("fan.ceiling", "off", {"friendly_name": "Ceiling"})
    assert await async_setup_component(
        hass,
        "input_boolean",
        {
            "input_boolean": {
                "let_in": {"name": "Let In"},
                "vacation": {"name": "Vacation"},
            }
        },
    )
    assert await async_setup_component(
        hass,
        "automation",
        {
            "automation": [
                {
                    "alias": "Let in",
                    "triggers": [
                        {
                            "trigger": "state",
                            "entity_id": "input_boolean.let_in",
                            "to": "on",
                        }
                    ],
                    "actions": _unlock(),
                },
                {
                    "alias": "Buzzer opens the door",
                    "triggers": [
                        {"trigger": "state", "entity_id": "switch.door_buzzer"}
                    ],
                    "actions": _unlock(),
                },
                {
                    # A phone notification's button: not a change the agent makes.
                    "alias": "Unlock from the phone",
                    "triggers": [
                        {
                            "trigger": "event",
                            "event_type": "mobile_app_notification_action",
                        }
                    ],
                    "actions": _unlock(),
                },
                {
                    # Names another entity.
                    "alias": "Unlock when home",
                    "triggers": [
                        {
                            "trigger": "template",
                            "value_template": "{{ is_state('person.me', 'home') }}",
                        }
                    ],
                    "actions": _unlock(),
                },
                {
                    # Any light coming on: reads lights generically.
                    "alias": "Unlock when a light comes on",
                    "triggers": [
                        {
                            "trigger": "template",
                            "value_template": (
                                "{{ states.light | selectattr('state', 'eq', 'on')"
                                " | list | count > 0 }}"
                            ),
                        }
                    ],
                    "actions": _unlock(),
                },
                {
                    # Locking is not a protected action.
                    "alias": "Lock on vacation",
                    "triggers": [
                        {"trigger": "state", "entity_id": "input_boolean.vacation"}
                    ],
                    "actions": [_call("lock.lock", "lock.front_door")],
                },
            ]
        },
    )
    assert await async_setup_component(
        hass,
        "script",
        {
            "script": {
                "welcome": {
                    "alias": "Welcome",
                    "sequence": [
                        _call("input_boolean.turn_on", "input_boolean.let_in")
                    ],
                }
            }
        },
    )
    assert await async_setup_component(
        hass,
        "scene",
        {"scene": [{"name": "Arrival", "entities": {"input_boolean.let_in": "on"}}]},
    )
    await hass.async_block_till_done()
    return hass


@pytest.mark.parametrize(
    ("tool", "name", "critical"),
    [
        # The example from the issue.
        ("HassTurnOn", "Let In", True),
        # A trigger's `to:` is not read: off counts too.
        ("HassTurnOff", "Let In", True),
        # Its automation only locks.
        ("HassTurnOn", "Vacation", False),
        # A plain switch an automation watches.
        ("HassTurnOn", "Buzzer", True),
        # Any light: an automation reads lights generically.
        ("HassTurnOn", "Kitchen", True),
        # A script and a scene that turn the helper on.
        ("HassTurnOn", "Welcome", True),
        ("HassTurnOn", "Arrival", True),
    ],
)
async def test_a_change_is_held_by_the_automations_it_starts(
    home: HomeAssistant, tool: str, name: str, *, critical: bool
) -> None:
    """Turning on a helper whose automation unlocks the door asks for the PIN."""
    assert (
        _is_critical_action({"name": name}, RULES, f"intent__{tool}", home) is critical
    )


@pytest.mark.parametrize(
    ("args", "critical"),
    [
        (
            {
                "domain": "input_boolean",
                "service": "turn_on",
                "entity_id": "input_boolean.let_in",
            },
            True,
        ),
        ({"domain": "input_boolean", "service": "turn_on", "entity_id": "all"}, True),
        (
            {
                "domain": "input_boolean",
                "service": "turn_on",
                "entity_id": "input_boolean.vacation",
            },
            False,
        ),
    ],
)
async def test_a_service_tool_reaches_the_same_automations(
    home: HomeAssistant, args: dict[str, Any], *, critical: bool
) -> None:
    """A tool calling services by name, and a call on `all`."""
    assert _is_critical_action(args, RULES, "mcp__call_service", home) is critical


async def test_only_automations_that_may_react_count(home: HomeAssistant) -> None:
    """A phone event, another entity's template and a lock-only one do not."""
    assert not _is_critical_action(
        {"name": "Ceiling"}, RULES, "intent__HassTurnOn", home
    )


async def test_an_automation_the_agent_writes_reaches_them_too(
    home: HomeAssistant,
) -> None:
    """Writing an automation that turns the helper on asks for the PIN."""
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
        "actions": [_call("input_boolean.turn_on", "input_boolean.let_in")],
    }
    assert (
        _maybe_require_pin_for_automation(
            cfg=cfg, validated_config=config, ha_automation_config=config
        )
        is not None
    )

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

import time
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, cast

import pytest
from homeassistant.setup import async_setup_component

from custom_components.home_generative_agent.agent.graph import (
    _is_critical_action,
    _prepare_tool_args,
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


def _automation(
    triggers: list[Any],
    actions: list[Any] | None = None,
    **extra: Any,
) -> SimpleNamespace:
    """Return an automation entity as the gate reads it."""
    actions = _unlock() if actions is None else actions
    return SimpleNamespace(
        entity_id=extra.pop("entity_id", "automation.probe"),
        raw_config={"alias": "Probe", "triggers": triggers, "actions": actions},
        action_script=SimpleNamespace(sequence=actions),
        **extra,
    )


@pytest.fixture
async def bare(hass: HomeAssistant) -> HomeAssistant:
    """Set up a lock, a fan, lights and a helper; automations are set per test."""
    hass.states.async_set("lock.front_door", "locked")
    hass.states.async_set("fan.ceiling", "off", {"friendly_name": "Ceiling"})
    hass.states.async_set("light.porch", "off", {"friendly_name": "Porch"})
    hass.states.async_set(
        "light.porch_group",
        "off",
        {"friendly_name": "Porch Group", "entity_id": ["light.porch"]},
    )
    hass.states.async_set("input_boolean.vac", "off", {"friendly_name": "Vac"})
    hass.states.async_set(
        "person.me", "away", {"device_trackers": ["device_tracker.phone"]}
    )
    hass.states.async_set("device_tracker.phone", "away")
    return hass


def _set(hass: HomeAssistant, monkeypatch: pytest.MonkeyPatch, *autos: Any) -> None:
    monkeypatch.setitem(hass.data, "automation", SimpleNamespace(entities=list(autos)))


def _held(hass: HomeAssistant, entity_id: str, service: str = "turn_on") -> bool:
    domain = entity_id.split(".", 1)[0]
    args = {"domain": domain, "service": service, "entity_id": entity_id}
    return _is_critical_action(args, RULES, "mcp__call_service", hass)


@pytest.mark.parametrize(
    ("trigger", "entity_id", "held"),
    [
        # An id built from parts reads its domain.
        (
            {
                "trigger": "template",
                "value_template": "{{ is_state('input_boolean.' ~ 'vac', 'on') }}",
            },
            "input_boolean.vac",
            True,
        ),
        (
            {
                "trigger": "template",
                "value_template": "{{ is_state('input_boolean.' ~ 'vac', 'on') }}",
            },
            "fan.ceiling",
            False,
        ),
        # A state_changed filtered to one entity reacts to it only.
        (
            {
                "trigger": "event",
                "event_type": "state_changed",
                "event_data": {"entity_id": "person.owner"},
            },
            "fan.ceiling",
            False,
        ),
        # A call_service filtered to a domain reacts to that domain only.
        (
            {
                "trigger": "event",
                "event_type": "call_service",
                "event_data": {"domain": "light"},
            },
            "light.porch",
            True,
        ),
        (
            {
                "trigger": "event",
                "event_type": "call_service",
                "event_data": {"domain": "light"},
            },
            "fan.ceiling",
            False,
        ),
        # A group's member changes the group: an automation on it reacts.
        ({"trigger": "state", "entity_id": "light.porch_group"}, "light.porch", True),
        # Another domain's device trigger (a remote) is not this change.
        (
            {"trigger": "device", "device_id": "x", "domain": "zha"},
            "light.porch",
            False,
        ),
    ],
)
async def test_which_automations_may_react(
    bare: HomeAssistant,
    monkeypatch: pytest.MonkeyPatch,
    trigger: dict[str, Any],
    entity_id: str,
    *,
    held: bool,
) -> None:
    """Built ids, filtered events, groups and other devices' triggers."""
    _set(bare, monkeypatch, _automation([trigger]))
    assert _held(bare, entity_id) is held


@pytest.mark.parametrize(
    ("actions", "held"),
    [
        # The user's own automation: a step that cannot be read does not hold.
        ([{"event": "fan_on"}], False),
        ([{"action": "{{ 'light.turn_on' }}"}], False),
        ([{"action": "rest_command.ping"}], False),
        # What can be read as protected does.
        (_unlock(), True),
        ([_call("lock.unlock", "{{ door }}")], True),
    ],
)
async def test_a_reacting_automation_is_held_only_for_what_can_be_read(
    bare: HomeAssistant,
    monkeypatch: pytest.MonkeyPatch,
    actions: list[Any],
    *,
    held: bool,
) -> None:
    """Steps that cannot be read ahead do not make a fan ask for the PIN."""
    _set(
        bare,
        monkeypatch,
        _automation([{"trigger": "state", "entity_id": "fan.ceiling"}], actions),
    )
    assert _held(bare, "fan.ceiling") is held


async def test_a_blueprint_automation_is_read_from_its_validated_triggers(
    bare: HomeAssistant, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Some HA versions keep only `use_blueprint` in the raw config."""
    auto = SimpleNamespace(
        entity_id="automation.bp",
        raw_config={"use_blueprint": {"path": "x.yaml", "input": {}}},
        _trigger_config=[{"trigger": "state", "entity_id": "input_boolean.vac"}],
        action_script=SimpleNamespace(sequence=_unlock()),
    )
    _set(bare, monkeypatch, auto)
    assert _held(bare, "input_boolean.vac")


async def test_a_script_that_failed_to_load_holds_nothing(
    bare: HomeAssistant, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A script that failed to set up cannot run or wait."""
    _set(bare, monkeypatch)
    broken = SimpleNamespace(entity_id="script.broken", raw_config={"sequence": []})
    monkeypatch.setitem(bare.data, "script", SimpleNamespace(entities=[broken]))
    assert not _held(bare, "fan.ceiling")


@pytest.mark.parametrize(("dev_id", "held"), [("phone", True), ("tablet", False)])
async def test_marking_a_tracker_home_is_checked(
    bare: HomeAssistant,
    monkeypatch: pytest.MonkeyPatch,
    dev_id: str,
    *,
    held: bool,
) -> None:
    """`device_tracker.see` can start an 'unlock when I arrive' automation."""
    _set(
        bare, monkeypatch, _automation([{"trigger": "state", "entity_id": "person.me"}])
    )
    args = {
        "domain": "device_tracker",
        "service": "see",
        "dev_id": dev_id,
        "location_name": "home",
    }
    assert _is_critical_action(args, RULES, "mcp__call_service", bare) is held


@pytest.mark.parametrize(("platform", "held"), [("mqtt", True), ("time", False)])
async def test_an_mqtt_publish_is_checked_against_mqtt_triggers(
    bare: HomeAssistant,
    monkeypatch: pytest.MonkeyPatch,
    platform: str,
    *,
    held: bool,
) -> None:
    """`mqtt.publish` can set off an MQTT trigger."""
    _set(bare, monkeypatch, _automation([{"trigger": platform, "topic": "door/open"}]))
    args = {"domain": "mqtt", "service": "publish", "topic": "door/open"}
    assert _is_critical_action(args, RULES, "mcp__call_service", bare) is held


async def test_the_pin_line_names_the_automation(home: HomeAssistant) -> None:
    """'turn on Let In' alone reads as harmless: the line says what it starts."""
    ctx = cast(
        "Any",
        SimpleNamespace(
            hass=home,
            config={"configurable": {"thread_id": "t1"}},
            critical_actions=RULES,
            pin_enabled=True,
            pin_hash="hash",
            pin_salt="salt",
            pending_actions={},
        ),
    )
    redirect, _ = _prepare_tool_args(
        "intent__HassTurnOn", {"name": "Let In"}, ctx, {"id": "call1"}
    )
    assert redirect is not None
    assert "starts the automation" in str(redirect.content)
    assert "Let in" in str(redirect.content)


@pytest.mark.parametrize(("watched", "held"), [(True, True), (False, False)])
async def test_a_scene_an_automation_watches_is_checked(
    bare: HomeAssistant,
    monkeypatch: pytest.MonkeyPatch,
    *,
    watched: bool,
    held: bool,
) -> None:
    """Running a harmless scene changes its state, which an automation may watch."""
    assert await async_setup_component(
        bare,
        "scene",
        {"scene": [{"name": "Quiet", "entities": {"fan.ceiling": "off"}}]},
    )
    await bare.async_block_till_done()
    target = "scene.quiet" if watched else "sensor.other"
    _set(bare, monkeypatch, _automation([{"trigger": "state", "entity_id": target}]))
    args = {"domain": "scene", "service": "turn_on", "entity_id": "scene.quiet"}
    assert _is_critical_action(args, RULES, "mcp__call_service", bare) is held
    assert (
        _is_critical_action({"name": "Quiet"}, RULES, "intent__HassTurnOn", bare)
        is held
    )


async def test_many_automations_and_lights_check_quickly(
    bare: HomeAssistant, monkeypatch: pytest.MonkeyPatch
) -> None:
    """300 automations and 100 lights: `light.turn_on all` stays well under 2 s."""
    for index in range(100):
        bare.states.async_set(f"light.l{index}", "off")
    autos = [
        _automation(
            [
                {"trigger": "state", "entity_id": f"sensor.s{index}"},
                {
                    "trigger": "numeric_state",
                    "entity_id": f"sensor.t{index}",
                    "above": 3,
                },
                {"trigger": "time", "at": "07:00:00"},
            ],
            [_call("light.turn_on", "light.porch")],
            entity_id=f"automation.a{index}",
        )
        for index in range(300)
    ]
    _set(bare, monkeypatch, *autos)
    start = time.perf_counter()
    args = {"domain": "light", "service": "turn_on", "entity_id": "all"}
    held = _is_critical_action(args, RULES, "mcp__call_service", bare)
    elapsed = time.perf_counter() - start
    assert not held
    assert elapsed < 2.0, elapsed  # gstack-timing-ok: a cost bound, not a race

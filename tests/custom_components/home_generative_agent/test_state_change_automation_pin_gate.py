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
from homeassistant.helpers import entity_registry as er
from homeassistant.setup import async_setup_component

from custom_components.home_generative_agent.agent.graph import (
    _is_critical_action,
    _prepare_tool_args,
)
from custom_components.home_generative_agent.agent.indirect_pin import (
    collect_started_automations,
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
    """500 automations and 400 lights: `light.turn_on all` stays well under 2 s."""
    for index in range(400):
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
        for index in range(500)
    ]
    _set(bare, monkeypatch, *autos)
    start = time.perf_counter()
    args = {"domain": "light", "service": "turn_on", "entity_id": "all"}
    held = _is_critical_action(args, RULES, "mcp__call_service", bare)
    elapsed = time.perf_counter() - start
    assert not held
    assert elapsed < 2.0, elapsed  # gstack-timing-ok: a cost bound, not a race


async def _scenes(hass: HomeAssistant) -> None:
    assert await async_setup_component(
        hass,
        "scene",
        {
            "scene": [
                {"name": "Arrival", "entities": {"lock.front_door": "unlocked"}},
                {"name": "Quiet", "entities": {"fan.ceiling": "off"}},
            ]
        },
    )
    await hass.async_block_till_done()


@pytest.mark.parametrize(("scene", "held"), [("arrival", True), ("quiet", False)])
async def test_a_scene_step_in_a_reacting_automation_is_opened(
    bare: HomeAssistant,
    monkeypatch: pytest.MonkeyPatch,
    scene: str,
    *,
    held: bool,
) -> None:
    """`- scene: scene.arrival`, where the scene unlocks the door."""
    await _scenes(bare)
    _set(
        bare,
        monkeypatch,
        _automation(
            [{"trigger": "state", "entity_id": "input_boolean.vac"}],
            [{"scene": f"scene.{scene}"}],
        ),
    )
    assert _held(bare, "input_boolean.vac") is held


@pytest.mark.parametrize(("actions", "held"), [(None, True), ([], False)])
async def test_an_automation_a_reacting_one_triggers_is_opened(
    bare: HomeAssistant,
    monkeypatch: pytest.MonkeyPatch,
    actions: list[Any] | None,
    *,
    held: bool,
) -> None:
    """`automation.trigger` runs the other automation's actions."""
    _set(
        bare,
        monkeypatch,
        _automation(
            [{"trigger": "state", "entity_id": "input_boolean.vac"}],
            [_call("automation.trigger", "automation.unlock")],
        ),
        _automation(
            [{"trigger": "time", "at": "07:00:00"}],
            actions,
            entity_id="automation.unlock",
        ),
    )
    assert _held(bare, "input_boolean.vac") is held


@pytest.mark.parametrize(
    "template",
    [
        "{{ is_state('input_boolean.va' ~ 'c', 'on') }}",
        "{{ is_state('input_boolean.{}'.format('vac'), 'on') }}",
        "{{ is_state('input_boolean.%s' % 'vac', 'on') }}",
    ],
)
async def test_an_id_built_from_part_of_a_name_reads_its_domain(
    bare: HomeAssistant, monkeypatch: pytest.MonkeyPatch, template: str
) -> None:
    """Part of the object id in the literal still names the domain."""
    _set(
        bare,
        monkeypatch,
        _automation([{"trigger": "template", "value_template": template}]),
    )
    assert _held(bare, "input_boolean.vac")
    assert not _held(bare, "fan.ceiling")


async def test_a_blueprint_automation_waiting_for_the_change_is_held(
    bare: HomeAssistant, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Its wait is only in the validated steps, not the raw config."""
    sequence = [
        {"wait_for_trigger": [{"trigger": "state", "entity_id": "input_boolean.vac"}]},
        *_unlock(),
    ]
    auto = SimpleNamespace(
        entity_id="automation.bp",
        raw_config={"use_blueprint": {"path": "x.yaml", "input": {}}},
        _trigger_config=[{"trigger": "time", "at": "07:00:00"}],
        action_script=SimpleNamespace(sequence=sequence),
    )
    _set(bare, monkeypatch, auto)
    assert _held(bare, "input_boolean.vac")


async def test_an_mqtt_publish_is_checked_against_mqtt_device_triggers(
    bare: HomeAssistant, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A button on an MQTT device starts its automation from a topic too."""
    trigger = {
        "trigger": "device",
        "domain": "mqtt",
        "device_id": "remote",
        "type": "button_short_press",
        "subtype": "button_1",
    }
    _set(bare, monkeypatch, _automation([trigger]))
    args = {"domain": "mqtt", "service": "publish", "topic": "remote/action"}
    assert _is_critical_action(args, RULES, "mcp__call_service", bare)


async def test_an_mqtt_publish_is_held_by_a_guarded_mqtt_entity(
    bare: HomeAssistant, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A publish can command an MQTT lock: its topic is not matched."""
    _set(bare, monkeypatch)
    args = {"domain": "mqtt", "service": "publish", "topic": "door/set"}
    assert not _is_critical_action(args, RULES, "mcp__call_service", bare)
    er.async_get(bare).async_get_or_create("lock", "mqtt", "door")
    assert _is_critical_action(args, RULES, "mcp__call_service", bare)


async def test_a_script_a_reacting_automation_runs_is_read_leniently(
    bare: HomeAssistant, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A REST call factored into a script does not hold the fan either."""
    assert await async_setup_component(
        bare,
        "script",
        {"script": {"ping": {"sequence": [{"action": "rest_command.ping"}]}}},
    )
    await bare.async_block_till_done()
    _set(
        bare,
        monkeypatch,
        _automation(
            [{"trigger": "state", "entity_id": "fan.ceiling"}],
            [{"action": "script.ping"}],
        ),
    )
    assert not _held(bare, "fan.ceiling")


async def test_a_press_still_reads_its_automations_strictly(
    bare: HomeAssistant, monkeypatch: pytest.MonkeyPatch
) -> None:
    """As since #780: what a pressed button starts and cannot be read holds."""
    bare.states.async_set("input_button.go", "unknown", {"friendly_name": "Go"})
    _set(
        bare,
        monkeypatch,
        _automation(
            [{"trigger": "state", "entity_id": "input_button.go"}],
            [{"event": "go"}],
        ),
    )
    assert _is_critical_action({"name": "Go"}, RULES, "intent__HassTurnOn", bare)


@pytest.mark.parametrize(("state", "held"), [("off", False), ("on", True)])
async def test_a_disabled_automation_does_not_hold(
    bare: HomeAssistant,
    monkeypatch: pytest.MonkeyPatch,
    state: str,
    *,
    held: bool,
) -> None:
    """A disabled automation's triggers do not fire."""
    bare.states.async_set("automation.probe", state)
    _set(
        bare,
        monkeypatch,
        _automation([{"trigger": "state", "entity_id": "fan.ceiling"}]),
    )
    assert _held(bare, "fan.ceiling") is held


@pytest.mark.parametrize(
    ("automation_state", "script_state", "held"),
    [
        (("on", {"current": 0}), "off", False),
        (("on", {"current": 1}), "off", True),
        (("on", {"current": 0}), "on", True),
    ],
)
async def test_a_wait_counts_only_while_a_run_may_wait(
    bare: HomeAssistant,
    monkeypatch: pytest.MonkeyPatch,
    automation_state: tuple[str, dict[str, Any]],
    script_state: str,
    *,
    held: bool,
) -> None:
    """An idle 'wait for the fan, then unlock' does not hold the fan."""
    wait = [
        {"wait_for_trigger": [{"trigger": "state", "entity_id": "fan.ceiling"}]},
        *_unlock(),
    ]
    bare.states.async_set("automation.probe", *automation_state)
    _set(
        bare,
        monkeypatch,
        _automation([{"trigger": "time", "at": "07:00:00"}], wait),
    )
    bare.states.async_set("script.arrive", script_state)
    script = SimpleNamespace(
        entity_id="script.arrive",
        raw_config={"sequence": wait} if script_state == "on" else {"sequence": []},
        script=SimpleNamespace(sequence=wait if script_state == "on" else []),
    )
    monkeypatch.setitem(bare.data, "script", SimpleNamespace(entities=[script]))
    assert _held(bare, "fan.ceiling") is held


async def test_a_person_arriving_changes_the_zone(
    bare: HomeAssistant, monkeypatch: pytest.MonkeyPatch
) -> None:
    """'Unlock when the first person gets home' watches `zone.home`."""
    bare.states.async_set("zone.home", "0", {"persons": ["person.me"]})
    _set(
        bare,
        monkeypatch,
        _automation([{"trigger": "state", "entity_id": "zone.home"}]),
    )
    args = {"domain": "device_tracker", "service": "see", "dev_id": "phone"}
    assert _is_critical_action(args, RULES, "mcp__call_service", bare)


async def test_a_long_chain_of_automations_holds_without_crashing(
    bare: HomeAssistant, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Each automation turning on the next helper, 160 deep."""
    autos = [
        _automation(
            [{"trigger": "state", "entity_id": f"input_boolean.b{index}"}],
            [_call("input_boolean.turn_on", f"input_boolean.b{index + 1}")],
            entity_id=f"automation.a{index}",
        )
        for index in range(160)
    ]
    autos.append(
        _automation(
            [{"trigger": "state", "entity_id": "input_boolean.b160"}],
            entity_id="automation.last",
        )
    )
    _set(bare, monkeypatch, *autos)
    assert _held(bare, "input_boolean.b0")


async def test_the_pin_line_names_a_waiting_script_by_its_name(
    bare: HomeAssistant, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Not its entity id."""
    _set(bare, monkeypatch)
    wait = [
        {"wait_for_trigger": [{"trigger": "state", "entity_id": "fan.ceiling"}]},
        *_unlock(),
    ]
    bare.states.async_set("script.arrive", "on", {"friendly_name": "Arrive"})
    script = SimpleNamespace(
        entity_id="script.arrive",
        raw_config={"sequence": wait},
        script=SimpleNamespace(sequence=wait),
    )
    monkeypatch.setitem(bare.data, "script", SimpleNamespace(entities=[script]))
    with collect_started_automations() as started:
        assert _held(bare, "fan.ceiling")
    assert started == ["Arrive"]


_TEMPLATED_UNLOCK = {
    "action": "{{ 'lock' }}.unlock",
    "target": {"entity_id": "lock.front_door"},
}


async def _template_entities(hass: HomeAssistant) -> None:
    assert await async_setup_component(
        hass,
        "template",
        {
            "template": [
                {
                    "button": [{"name": "Tpl Press", "press": [_TEMPLATED_UNLOCK]}],
                    "switch": [
                        {
                            "name": "Tpl Sw",
                            "turn_on": [_TEMPLATED_UNLOCK],
                            "turn_off": [],
                        }
                    ],
                }
            ]
        },
    )
    await hass.async_block_till_done()


@pytest.mark.parametrize(
    "step",
    [
        _call("button.press", "button.tpl_press"),
        _call("switch.turn_on", "switch.tpl_sw"),
    ],
)
async def test_a_template_entity_a_reacting_automation_acts_on_stays_strict(
    bare: HomeAssistant, monkeypatch: pytest.MonkeyPatch, step: dict[str, Any]
) -> None:
    """Its own actions are not the user's automation: unreadable ones hold (#780)."""
    await _template_entities(bare)
    _set(
        bare,
        monkeypatch,
        _automation([{"trigger": "state", "entity_id": "fan.ceiling"}], [step]),
    )
    assert _held(bare, "fan.ceiling")


async def test_a_disabled_automation_still_running_still_waits(
    bare: HomeAssistant, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Turned off without stopping it, its run still waits on the fan."""
    wait = [
        {"wait_for_trigger": [{"trigger": "state", "entity_id": "fan.ceiling"}]},
        *_unlock(),
    ]
    bare.states.async_set("automation.probe", "off", {"current": 1})
    _set(bare, monkeypatch, _automation([{"trigger": "time", "at": "07:00:00"}], wait))
    assert _held(bare, "fan.ceiling")


async def test_a_person_arriving_joins_an_empty_zone(
    bare: HomeAssistant, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The zone lists only the people already in it."""
    bare.states.async_set("zone.home", "0", {"persons": []})
    _set(
        bare,
        monkeypatch,
        _automation([{"trigger": "state", "entity_id": "zone.home"}]),
    )
    args = {"domain": "device_tracker", "service": "see", "dev_id": "phone"}
    assert _is_critical_action(args, RULES, "mcp__call_service", bare)


async def test_an_mqtt_publish_is_checked_against_call_service_events(
    bare: HomeAssistant, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A publish fires `call_service` for the mqtt domain."""
    trigger = {
        "trigger": "event",
        "event_type": "call_service",
        "event_data": {"domain": "mqtt", "service": "publish"},
    }
    _set(bare, monkeypatch, _automation([trigger]))
    args = {"domain": "mqtt", "service": "publish", "topic": "x/y"}
    assert _is_critical_action(args, RULES, "mcp__call_service", bare)


async def test_an_mqtt_publish_in_a_large_mqtt_home_is_not_held(
    bare: HomeAssistant, monkeypatch: pytest.MonkeyPatch
) -> None:
    """2100 harmless MQTT sensors do not trip the followed-entity bound."""
    _set(bare, monkeypatch)
    registry = er.async_get(bare)
    for index in range(2100):
        registry.async_get_or_create("sensor", "mqtt", f"s{index}")
    args = {"domain": "mqtt", "service": "publish", "topic": "x/y"}
    assert not _is_critical_action(args, RULES, "mcp__call_service", bare)


async def test_an_mqtt_publish_sees_mqtt_entities_with_no_registry_entry(
    bare: HomeAssistant, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A YAML MQTT lock with no unique id is loaded but not registered."""
    _set(bare, monkeypatch)
    platform = SimpleNamespace(entities={"lock.yaml_door": object()})
    monkeypatch.setattr(
        "custom_components.home_generative_agent.agent.template_actions."
        "async_get_platforms",
        lambda _hass, name: [platform] if name == "mqtt" else [],
    )
    args = {"domain": "mqtt", "service": "publish", "topic": "door/set"}
    assert _is_critical_action(args, RULES, "mcp__call_service", bare)

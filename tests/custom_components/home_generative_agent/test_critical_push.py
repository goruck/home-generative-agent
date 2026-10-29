# ruff: noqa: S101
"""Tests for the critical push rewrite on safety automations (#869)."""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING, Any, cast

import pytest
import yaml
from homeassistant.components.automation.config import _async_validate_config_item
from homeassistant.components.notify.const import NOTIFY_SERVICE_SCHEMA
from homeassistant.helpers import config_validation as cv
from homeassistant.helpers import entity_registry as er
from homeassistant.setup import async_setup_component
from pytest_homeassistant_custom_component.common import MockConfigEntry

from custom_components.home_generative_agent.agent.critical_push import (
    CriticalPushResult,
    apply_critical_push,
    is_safety_automation,
)
from custom_components.home_generative_agent.agent.helpers import (
    PinReply,
    disclose_pin_prompts,
    start_pin_turn,
)
from custom_components.home_generative_agent.agent.tools import (
    _critical_push_note,
    add_automation,
    confirm_sensitive_action,
)
from custom_components.home_generative_agent.const import (
    CONF_CRITICAL_ACTION_PIN_ENABLED,
    CONF_CRITICAL_ACTION_PIN_HASH,
    CONF_CRITICAL_ACTION_PIN_SALT,
    CONF_NOTIFY_SERVICE,
)
from custom_components.home_generative_agent.core.utils import hash_pin

if TYPE_CHECKING:
    from pathlib import Path

    from homeassistant.core import HomeAssistant

add_automation_tool = cast("Any", add_automation).coroutine
_confirm = cast("Any", confirm_sensitive_action).coroutine

MOISTURE = "binary_sensor.shelly_flood_s_gen4_a1b2c3_moisture"
DOOR = "binary_sensor.front_door"
PHONE = "notify.mobile_app_lindos_phone"
PIN = "4321"

CRITICAL_SOUND = {"name": "default", "critical": 1, "volume": 1.0}
ANDROID = {"channel": "alarm_stream", "importance": "high", "priority": "high"}
CRITICAL_PAYLOAD = {"push": {"sound": CRITICAL_SOUND}, **ANDROID}

# The leak alert from the #669 field test, with the real names it was retried
# with.
FIELD_YAML = f"""
alias: "Leak Alert"
description: "Notify via mobile when any moisture/leak sensor is triggered"
trigger:
  - platform: state
    entity_id: {MOISTURE}
    to: "on"
condition: []
action:
  - service: {PHONE}
    data:
      title: "Leak Detected!"
      message: "A leak has been detected in the house. Please check immediately."
mode: single
"""


async def _noop(*_args: Any, **_kwargs: Any) -> None:
    """Stand in for a service handler."""


def _config(
    hass: HomeAssistant, options: dict[str, Any] | None = None
) -> dict[str, Any]:
    """Build the configurable payload passed to the tools."""
    return {
        "configurable": {
            "hass": hass,
            "options": {CONF_NOTIFY_SERVICE: PHONE} if options is None else options,
            "pending_actions": {},
            "user_id": "user1",
            "thread_id": "conv1",
            "pin_reply": PinReply("add the automation"),
            "ha_llm_api": None,
        }
    }


def _written(path: Path) -> Any:
    """Return the automations the tool wrote."""
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _actions(config: dict[str, Any]) -> list[dict[str, Any]]:
    """Return a config's actions under either spelling."""
    return cast("list[dict[str, Any]]", config.get("actions", config.get("action")))


def _automation(actions: list[dict[str, Any]], **trigger: Any) -> dict[str, Any]:
    """Return a raw automation config on one state trigger on the leak sensor."""
    return {
        "alias": "x",
        "triggers": [{"trigger": "state", "entity_id": MOISTURE, **trigger}],
        "actions": actions,
    }


def _push(service: str = PHONE, **payload: Any) -> dict[str, Any]:
    """Return one push step, the push payload under its nested `data`."""
    data: dict[str, Any] = {"message": "m"}
    if payload:
        data["data"] = payload
    return {"action": service, "data": data}


def _critical(**payload: Any) -> dict[str, Any]:
    """Return the notify data a rewritten push carries."""
    return {"message": "m", "data": {**payload, **CRITICAL_PAYLOAD}}


async def _result(hass: HomeAssistant, config: dict[str, Any]) -> CriticalPushResult:
    """Validate a raw config as the tool does, then rewrite it."""
    validated = await _async_validate_config_item(
        hass=hass,
        config={"id": "test", **config},
        raise_on_errors=True,
        warn_on_errors=False,
    )
    return apply_critical_push(hass, config, validated)


async def _apply(hass: HomeAssistant, config: dict[str, Any]) -> int:
    """Return how many pushes the rewrite made critical."""
    return (await _result(hass, config)).rewritten


@pytest.fixture(autouse=True)
async def _home(hass: HomeAssistant) -> None:
    """Set up a moisture sensor, a door, a registered phone and a light."""
    assert await async_setup_component(hass, "homeassistant", {})
    hass.states.async_set(
        MOISTURE,
        "off",
        {"friendly_name": "Sink Moisture Sensor", "device_class": "moisture"},
    )
    hass.states.async_set(
        DOOR, "off", {"friendly_name": "Front Door", "device_class": "door"}
    )
    hass.states.async_set("light.hall", "off", {"friendly_name": "Hall Light"})
    # The phone mobile_app registered `notify.mobile_app_lindos_phone` for.
    MockConfigEntry(
        domain="mobile_app", data={"device_name": "Lindos Phone"}
    ).add_to_hass(hass)
    hass.services.async_register("notify", "mobile_app_lindos_phone", _noop)
    hass.services.async_register("notify", "notify", _noop)
    hass.services.async_register("notify", "family", _noop)
    hass.services.async_register("persistent_notification", "create", _noop)
    hass.services.async_register("light", "turn_on", _noop)


@pytest.fixture
def automations_file(
    hass: HomeAssistant, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> Path:
    """Return an empty automations.yaml the tool writes to, with reload stubbed."""
    path = tmp_path / "automations.yaml"
    path.write_text("", encoding="utf-8")
    monkeypatch.setattr(hass.config, "config_dir", str(tmp_path), raising=False)
    hass.services.async_register("automation", "reload", _noop)
    return path


# ----- end to end through add_automation -----


@pytest.mark.asyncio
async def test_field_leak_alert_is_written_as_a_valid_critical_push(
    hass: HomeAssistant,
    automations_file: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """The #669 leak alert installs with the payload where the notify call takes it."""
    caplog.set_level(logging.INFO)

    result = await add_automation_tool(automation_yaml=FIELD_YAML, config=_config(hass))

    assert result.startswith("Added automation ")
    assert "critical alerts" in result
    assert "Critical Alerts allowed" in result
    assert "not critical alerts" not in result
    data = _written(automations_file)[0]["actions"][0]["data"]
    assert data == {
        "title": "Leak Detected!",
        "message": "A leak has been detected in the house. Please check immediately.",
        "data": CRITICAL_PAYLOAD,
    }
    # What HA checks when the action runs: a key beside `message` fails here.
    assert NOTIFY_SERVICE_SCHEMA(data) == data
    assert "made 1 push(es) in 'Leak Alert' critical alerts; 0 left" in caplog.text


@pytest.mark.asyncio
async def test_an_ordinary_automation_is_written_unchanged(
    hass: HomeAssistant, automations_file: Path
) -> None:
    """A door alert keeps the push the model wrote, and the reply says nothing."""
    text = FIELD_YAML.replace(MOISTURE, DOOR)

    result = await add_automation_tool(automation_yaml=text, config=_config(hass))

    assert result.startswith("Added automation ")
    assert "critical" not in result
    data = _written(automations_file)[0]["actions"][0]["data"]
    assert set(data) == {"title", "message"}


@pytest.mark.asyncio
async def test_a_leak_alert_through_another_notify_service_says_it_is_not_critical(
    hass: HomeAssistant,
    automations_file: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """notify.notify is left as written, and the reply tells the model why."""
    caplog.set_level(logging.INFO)
    text = FIELD_YAML.replace(PHONE, "notify.notify")

    result = await add_automation_tool(automation_yaml=text, config=_config(hass))

    assert result.startswith("Added automation ")
    assert "1 of its notifications are not sent through" in result
    # Installed already: another add_automation call would add a duplicate.
    assert "do not add it again" in result
    assert "Settings > Automations" in result
    assert "Critical Alerts allowed" not in result
    data = _written(automations_file)[0]["actions"][0]["data"]
    assert set(data) == {"title", "message"}
    assert "made 0 push(es) in 'Leak Alert' critical alerts; 1 left" in caplog.text


@pytest.mark.asyncio
async def test_a_pin_held_automation_is_stored_critical_and_says_so_when_confirmed(
    hass: HomeAssistant, automations_file: Path
) -> None:
    """The held config is the rewritten one, and the confirmed reply names it."""
    hass.services.async_register("lock", "unlock", _noop)
    hass.states.async_set("lock.front", "locked", {"friendly_name": "Front"})
    hashed, salt = hash_pin(PIN)
    config = _config(
        hass,
        {
            CONF_NOTIFY_SERVICE: PHONE,
            CONF_CRITICAL_ACTION_PIN_ENABLED: True,
            CONF_CRITICAL_ACTION_PIN_HASH: hashed,
            CONF_CRITICAL_ACTION_PIN_SALT: salt,
        },
    )
    text = FIELD_YAML.replace(
        "mode: single",
        "  - action: lock.unlock\n    target:\n      entity_id: lock.front\n",
    )

    challenge = json.loads(
        await add_automation_tool(automation_yaml=text, config=config)
    )

    assert challenge["status"] == "requires_pin"
    assert _written(automations_file) is None
    (held,) = config["configurable"]["pending_actions"].values()
    held_payload = held["automation_config"]["actions"][0]["data"]["data"]
    assert held_payload == CRITICAL_PAYLOAD

    configurable = config["configurable"]
    disclose_pin_prompts(configurable["pending_actions"], configurable["pin_reply"])
    configurable["pin_reply"] = start_pin_turn(
        configurable["pending_actions"], configurable["thread_id"], PIN
    )
    confirmed = json.loads(
        await _confirm(challenge["action_id"], PIN, config=config, store=None)
    )

    assert confirmed["status"] == "completed"
    assert "Critical Alerts allowed" in confirmed["result"]
    written = _written(automations_file)[0]["actions"][0]["data"]["data"]
    assert written == CRITICAL_PAYLOAD


# ----- which automations count -----


@pytest.mark.parametrize(
    ("device_class", "safety"),
    [
        ("moisture", True),
        ("smoke", True),
        ("gas", True),
        ("carbon_monoxide", True),
        ("safety", True),
        ("door", False),
        ("motion", False),
        ("tamper", False),
        (None, False),
    ],
)
async def test_only_the_safety_device_classes_count(
    hass: HomeAssistant, device_class: str | None, *, safety: bool
) -> None:
    """Leak, smoke, gas, CO and safety sensors; nothing else."""
    attributes = {"device_class": device_class} if device_class else {}
    hass.states.async_set("binary_sensor.probe", "off", attributes)
    config = _automation([_push()], to="on")
    config["triggers"][0]["entity_id"] = "binary_sensor.probe"

    assert await _apply(hass, config) == (1 if safety else 0)
    assert (_actions(config)[0]["data"] == _critical()) is safety


@pytest.mark.parametrize(
    ("trigger", "alarm"),
    [
        ({"to": "on"}, True),
        ({"to": ["on", "unavailable"]}, True),
        ({"from": "off"}, True),
        ({"from": ["on", "off"]}, True),
        # Also every attribute update (battery, signal strength).
        ({}, False),
        ({"to": "off"}, False),
        ({"to": "unavailable"}, False),
        ({"from": "on"}, False),
        ({"not_to": "on"}, False),
        # Any of to/from/not_to/not_from limits HA to state changes.
        ({"not_from": ["unavailable", "unknown"]}, True),
        ({"not_to": ["unavailable"]}, True),
        ({"to": None}, True),
        ({"attribute": "battery_level"}, False),
    ],
)
async def test_only_triggers_that_fire_on_the_alarm_count(
    hass: HomeAssistant, trigger: dict[str, Any], *, alarm: bool
) -> None:
    """A leak-cleared, sensor-offline or battery notice stays an ordinary push."""
    config = _automation([_push()], **trigger)

    assert await _apply(hass, config) == (1 if alarm else 0)


async def test_a_sensor_with_a_safety_class_in_another_domain_does_not_count(
    hass: HomeAssistant,
) -> None:
    """Only binary sensors: a `sensor` with a gas class is a reading, not an alarm."""
    hass.states.async_set("sensor.gas_ppm", "3", {"device_class": "gas"})
    config = _automation([_push()])
    config["triggers"][0]["entity_id"] = "sensor.gas_ppm"

    assert await _apply(hass, config) == 0


async def test_any_alarm_trigger_counts_in_every_spelling(
    hass: HomeAssistant,
) -> None:
    """Legacy keys, a comma list and a nested trigger list."""
    legacy: dict[str, Any] = {
        "trigger": [
            {"platform": "state", "entity_id": [DOOR]},
            {"platform": "state", "entity_id": f"{DOOR}, {MOISTURE}", "to": "on"},
        ],
        "action": [{"service": PHONE, "data": {"message": "m"}}],
    }
    assert await _apply(hass, legacy) == 1
    assert _actions(legacy)[0]["data"] == _critical()

    nested: dict[str, Any] = {
        "triggers": [
            {"triggers": [{"trigger": "state", "entity_id": MOISTURE, "to": "on"}]}
        ],
        "actions": [_push()],
    }
    assert await _apply(hass, nested) == 1


@pytest.mark.parametrize("enabled", [False, "false", 0, "{{ false }}"])
async def test_a_disabled_alarm_trigger_does_not_count(
    hass: HomeAssistant, enabled: Any
) -> None:
    """A trigger HA skips, however `enabled` is spelled, or may skip (a template)."""
    config = _automation([_push()], to="on", enabled=enabled)
    config["triggers"].append({"trigger": "time_pattern", "minutes": "/30"})

    assert await _apply(hass, config) == 0


async def test_a_template_trigger_does_not_count(hass: HomeAssistant) -> None:
    """A template names no entity this check can read."""
    config: dict[str, Any] = {
        "triggers": [{"trigger": "template", "value_template": "{{ 1 }}"}],
        "actions": [_push()],
    }

    assert await _apply(hass, config) == 0


async def test_device_class_from_the_registry_a_device_trigger_and_moisture_detected(
    hass: HomeAssistant,
) -> None:
    """A sensor not reporting yet, a device trigger by registry id, a purpose trigger."""
    registry = er.async_get(hass)
    entry = registry.async_get_or_create(
        "binary_sensor",
        "test",
        "smoke-1",
        suggested_object_id="hallway_smoke",
        original_device_class="smoke",
    )
    assert hass.states.get(entry.entity_id) is None

    by_entity = _automation([_push()], to="on")
    by_entity["triggers"][0]["entity_id"] = entry.entity_id
    assert await _apply(hass, by_entity) == 1

    # HA checks a device trigger's device exists and the purpose triggers
    # need their components, so these run through the trigger list schema
    # alone, which renames `trigger:` to `platform:` as validation does.
    def validated(trigger: dict[str, Any]) -> dict[str, Any]:
        return {"triggers": cv.TRIGGER_SCHEMA([trigger])}

    def device(kind: str) -> dict[str, Any]:
        return validated(
            {
                "trigger": "device",
                "domain": "binary_sensor",
                "device_id": "abc",
                "entity_id": entry.id,
                "type": kind,
            }
        )

    assert validated({"trigger": "moisture.detected"})["triggers"][0]["platform"]
    assert is_safety_automation(hass, device("smoke"))
    assert not is_safety_automation(hass, device("no_smoke"))
    for name in (
        "moisture.detected",
        "air_quality.smoke_detected",
        "air_quality.gas_detected",
        "air_quality.co_detected",
    ):
        assert is_safety_automation(hass, validated({"trigger": name})), name
    for name in (
        "moisture.cleared",
        "air_quality.co_cleared",
        "air_quality.co_changed",
    ):
        assert not is_safety_automation(hass, validated({"trigger": name})), name


# ----- which pushes are rewritten -----


async def test_only_mobile_app_phones_are_rewritten_wherever_they_nest(
    hass: HomeAssistant,
) -> None:
    """Pushes inside choose (a list or one mapping) and if; nothing else."""
    # A notify group may take a mobile_app-like name; mobile_app made no
    # phone by that name.
    hass.services.async_register("notify", "mobile_app_family", _noop)
    config = _automation(
        [
            {
                "choose": [{"conditions": [], "sequence": [_push()]}],
                "default": [{"if": [], "then": [_push()]}],
            },
            {"choose": {"conditions": [], "sequence": [_push()]}},
            _push(service="notify.family"),
            _push(service="notify.mobile_app_family"),
            _push(service="notify.persistent_notification"),
            _push(service="persistent_notification.create"),
            {"action": "light.turn_on", "target": {"entity_id": "light.hall"}},
        ],
        to="on",
    )

    result = await _result(hass, config)

    assert result == CriticalPushResult(rewritten=3, left_ordinary=2)
    actions = _actions(config)
    assert actions[0]["choose"][0]["sequence"][0]["data"] == _critical()
    assert actions[0]["default"][0]["then"][0]["data"] == _critical()
    assert actions[1]["choose"]["sequence"][0]["data"] == _critical()
    # A group or another platform may reject keys it does not know.
    for action in actions[2:6]:
        assert action["data"] == {"message": "m"}
    assert "data" not in actions[6]


WET_DRY_TRIGGERS = [
    {"trigger": "state", "entity_id": MOISTURE, "to": "on", "id": "wet"},
    {"trigger": "state", "entity_id": MOISTURE, "to": "off", "id": "dry"},
    {"trigger": "time_pattern", "hours": "/1"},
]


def _when(*ids: str, **condition: Any) -> list[dict[str, Any]]:
    """Return the conditions of a branch taken for the given trigger ids."""
    return [{"condition": "trigger", "id": list(ids), **condition}]


async def test_branches_gated_on_the_all_clear_trigger_stay_ordinary(
    hass: HomeAssistant,
) -> None:
    """The wet/dry pattern: only what the alarm reaches is made critical."""
    config: dict[str, Any] = {
        "triggers": WET_DRY_TRIGGERS,
        "actions": [
            {
                "choose": [
                    {"conditions": _when("wet"), "sequence": [_push()]},
                    {"conditions": _when("dry", "2"), "sequence": [_push()]},
                ],
                # Every trigger is taken by an option: nothing reaches this.
                "default": [_push()],
            },
            # Only dry and the hourly trigger reach this default.
            {
                "choose": [{"conditions": _when("wet"), "sequence": []}],
                "default": [_push()],
            },
            # The alarm reaches this default.
            {
                "choose": [{"conditions": _when("dry"), "sequence": []}],
                "default": [_push()],
            },
            {"if": _when("wet"), "then": [_push()], "else": [_push()]},
            {"if": _when("dry"), "then": [_push()], "else": [_push()]},
            _push(),
        ],
    }

    result = await _result(hass, config)

    assert result == CriticalPushResult(rewritten=5, left_ordinary=0)
    choose, wet_default, dry_default, if_wet, if_dry, last = _actions(config)
    assert choose["choose"][0]["sequence"][0]["data"] == _critical()
    assert choose["choose"][1]["sequence"][0]["data"] == {"message": "m"}
    assert choose["default"][0]["data"] == {"message": "m"}
    assert wet_default["default"][0]["data"] == {"message": "m"}
    assert dry_default["default"][0]["data"] == _critical()
    assert if_wet["then"][0]["data"] == _critical()
    assert if_wet["else"][0]["data"] == {"message": "m"}
    assert if_dry["then"][0]["data"] == {"message": "m"}
    assert if_dry["else"][0]["data"] == _critical()
    assert last["data"] == _critical()


async def test_branches_the_alarm_can_still_reach_are_made_critical(
    hass: HomeAssistant,
) -> None:
    """Another condition in an if, or a disabled trigger condition, opens a branch."""
    away = {"condition": "state", "entity_id": "person.me", "state": "home"}
    config: dict[str, Any] = {
        "triggers": WET_DRY_TRIGGERS,
        "actions": [
            # The alarm takes `else` when nobody is home.
            {"if": [*_when("wet"), away], "then": [_push()], "else": [_push()]},
            # HA skips a disabled condition, so the alarm runs this option.
            {
                "choose": [
                    {"conditions": _when("dry", enabled=False), "sequence": [_push()]}
                ]
            },
            {"if": _when("dry", enabled="{{ false }}"), "then": [_push()]},
        ],
    }

    result = await _result(hass, config)

    assert result == CriticalPushResult(rewritten=4, left_ordinary=0)
    if_away, disabled, templated = _actions(config)
    assert if_away["else"][0]["data"] == _critical()
    assert disabled["choose"][0]["sequence"][0]["data"] == _critical()
    assert templated["then"][0]["data"] == _critical()


async def test_a_step_shared_by_a_yaml_alias_is_rewritten_per_branch(
    hass: HomeAssistant,
) -> None:
    """The alarm's copy of a shared push is critical; the all-clear's is not."""
    text = f"""
triggers:
  - {{trigger: state, entity_id: {MOISTURE}, to: "on", id: wet}}
  - {{trigger: state, entity_id: {MOISTURE}, to: "off", id: dry}}
actions:
  - choose:
      - conditions: [{{condition: trigger, id: wet}}]
        sequence:
          - &push
            action: {PHONE}
            data: {{message: Water}}
      - conditions: [{{condition: trigger, id: dry}}]
        sequence:
          - *push
"""
    config = yaml.safe_load(text)

    assert await _apply(hass, config) == 1
    wet, dry = _actions(config)[0]["choose"]
    assert wet["sequence"][0]["data"] == {"message": "Water", "data": CRITICAL_PAYLOAD}
    assert dry["sequence"][0]["data"] == {"message": "Water"}


async def test_a_blueprint_safety_automation_says_its_pushes_are_not_critical(
    hass: HomeAssistant,
) -> None:
    """A blueprint holds the pushes, which this cannot rewrite."""
    validated = {
        "triggers": [{"platform": "moisture.detected"}],
        "actions": [_push(), {"action": "light.turn_on"}],
    }
    config: dict[str, Any] = {"use_blueprint": {"path": "x.yaml", "input": {}}}

    result = apply_critical_push(hass, config, validated)

    assert result == CriticalPushResult(left_ordinary=1)
    assert config == {"use_blueprint": {"path": "x.yaml", "input": {}}}


async def test_the_model_payload_is_merged_not_replaced(
    hass: HomeAssistant,
) -> None:
    """A chosen sound is kept and made critical; a quieter level is not."""
    config = _automation(
        [
            _push(push={"sound": "alarm.caf", "category": "leak"}),
            _push(
                push={
                    "sound": {"name": "", "volume": 0.2},
                    "interruption-level": "passive",
                },
                channel="quiet",
                importance="low",
                tag="leak",
            ),
            _push(push={"sound": "none"}),
        ],
        to="on",
    )

    assert await _apply(hass, config) == 3
    actions = _actions(config)
    assert actions[0]["data"]["data"]["push"] == {
        "sound": {"name": "alarm.caf", "critical": 1, "volume": 1.0},
        "category": "leak",
    }
    second = actions[1]["data"]["data"]
    assert second["push"] == {
        "sound": CRITICAL_SOUND,
        "interruption-level": "critical",
    }
    assert second["tag"] == "leak"
    assert {key: second[key] for key in ANDROID} == ANDROID
    assert actions[2]["data"]["data"]["push"]["sound"] == CRITICAL_SOUND
    for action in actions:
        assert NOTIFY_SERVICE_SCHEMA(action["data"]) == action["data"]


async def test_commands_and_templates_are_left_alone(hass: HomeAssistant) -> None:
    """A command, one a template may render, and every templated payload part."""
    config = _automation(
        [
            {"action": PHONE, "data": {"message": "clear_notification"}},
            {"action": PHONE, "data": {"message": "command_dnd"}},
            {
                "action": PHONE,
                "data": {
                    "message": "{{ 'remove_channel' }}",
                    "data": {"channel": "old"},
                },
            },
            {"action": PHONE, "data": "{{ data }}"},
            {"action": PHONE, "data": {"message": "m", "data": "{{ payload }}"}},
            {"action": PHONE, "data": {"message": "m", "data": {"push": "{{ p }}"}}},
            {
                "action": PHONE,
                "data": {"message": "m", "data": {"push": {"sound": "{{ s }}"}}},
            },
        ],
        to="on",
    )
    before = [dict(action) for action in _actions(config)]

    result = await _result(hass, config)

    # Commands are not alerts; the templated payloads are, left ordinary.
    assert result == CriticalPushResult(rewritten=0, left_ordinary=4)
    assert _actions(config) == before


async def test_a_templated_alert_message_is_still_made_critical(
    hass: HomeAssistant,
) -> None:
    """A message template that names no command is an alert like any other."""
    config = _automation(
        [
            {
                "action": PHONE,
                "data": {"message": "Leak at {{ trigger.to_state.name }}"},
            },
            {"action": PHONE, "data": {"message": "Leak in {{ 'Massachusetts' }}"}},
            # The template may render to remove_channel, whose argument is the
            # channel: never retarget it at alarm_stream.
            {
                "action": PHONE,
                "data": {"message": "{{ message }}", "data": {"channel": "old"}},
            },
        ],
        to="on",
    )

    assert await _result(hass, config) == CriticalPushResult(
        rewritten=2, left_ordinary=1
    )
    assert _actions(config)[2]["data"]["data"] == {"channel": "old"}


async def test_data_template_wins_like_it_does_in_home_assistant(
    hass: HomeAssistant,
) -> None:
    """HA merges data_template over data, for the message and for the payload."""
    config = _automation(
        [
            {
                "action": PHONE,
                "data": {"message": "m", "data": {"tag": "a"}},
                "data_template": {"data": {"tag": "b"}},
            },
            {"action": PHONE, "data_template": {"message": "m", "data": {"tag": "t"}}},
            {
                "action": PHONE,
                "data": {"message": "clear_notification"},
                "data_template": {"message": "Smoke detected"},
            },
            {"action": PHONE},
        ],
        to="on",
    )

    assert await _apply(hass, config) == 4
    both, template_only, command_overridden, bare = _actions(config)
    assert both["data"] == {"message": "m", "data": {"tag": "a"}}
    assert both["data_template"] == {"data": {"tag": "b", **CRITICAL_PAYLOAD}}
    assert template_only["data_template"] == {
        "message": "m",
        "data": {"tag": "t", **CRITICAL_PAYLOAD},
    }
    assert command_overridden["data"]["data"] == CRITICAL_PAYLOAD
    assert bare["data"] == {"data": CRITICAL_PAYLOAD}


async def test_a_mapping_shared_by_a_yaml_anchor_is_not_rewritten_twice(
    hass: HomeAssistant,
) -> None:
    """A payload shared with another step is copied, not changed under it."""
    text = f"""
triggers:
  - trigger: state
    entity_id: {MOISTURE}
    to: "on"
actions:
  - action: persistent_notification.create
    data: &payload
      message: Leak
  - action: {PHONE}
    data: *payload
"""
    config = yaml.safe_load(text)

    assert await _apply(hass, config) == 1
    actions = _actions(config)
    assert actions[0]["data"] == {"message": "Leak"}
    assert actions[1]["data"] == {"message": "Leak", "data": CRITICAL_PAYLOAD}


async def test_choose_takes_the_first_matching_option(hass: HomeAssistant) -> None:
    """An option after one the alarm always takes is not reached by the alarm."""
    cleared = {"condition": "state", "entity_id": MOISTURE, "state": "off"}
    config: dict[str, Any] = {
        "triggers": WET_DRY_TRIGGERS,
        "actions": [
            {
                "choose": [
                    {"conditions": _when("wet"), "sequence": [_push()]},
                    {"conditions": [cleared], "sequence": [_push()]},
                ]
            },
            # A state check first: the alarm may fall through to the next one.
            {
                "choose": [
                    {"conditions": [cleared], "sequence": [_push()]},
                    {"conditions": _when("wet"), "sequence": [_push()]},
                ]
            },
        ],
    }

    assert await _apply(hass, config) == 3
    first, second = _actions(config)
    assert first["choose"][1]["sequence"][0]["data"] == {"message": "m"}
    assert second["choose"][0]["sequence"][0]["data"] == _critical()


async def test_a_trigger_condition_step_gates_the_rest_of_its_sequence(
    hass: HomeAssistant,
) -> None:
    """A parallel branch that stops for the alarm keeps its push ordinary."""
    config: dict[str, Any] = {
        "triggers": WET_DRY_TRIGGERS,
        "actions": [
            {
                "parallel": [
                    [{"condition": "trigger", "id": "wet"}, _push()],
                    [{"condition": "trigger", "id": "dry"}, _push()],
                    # HA skips a disabled condition step.
                    [{"condition": "trigger", "id": "dry", "enabled": False}, _push()],
                ]
            }
        ],
    }

    assert await _apply(hass, config) == 2
    wet, dry, disabled = _actions(config)[0]["parallel"]
    assert wet[1]["data"] == _critical()
    assert dry[1]["data"] == {"message": "m"}
    assert disabled[1]["data"] == _critical()


async def test_disabled_steps_are_neither_rewritten_nor_counted(
    hass: HomeAssistant,
) -> None:
    """A step or container HA skips sends nothing, critical or not."""
    config = _automation(
        [
            _push(),
            {**_push(service="notify.family"), "enabled": False},
            {"sequence": [_push()], "enabled": False},
        ],
        to="on",
    )

    assert await _result(hass, config) == CriticalPushResult(rewritten=1)
    assert _actions(config)[2]["sequence"][0]["data"] == {"message": "m"}


async def test_a_deep_but_valid_config_is_rewritten_and_a_too_deep_one_says_so(
    hass: HomeAssistant,
) -> None:
    """Depth counts container steps, as the existence check does."""

    def nested(levels: int) -> list[dict[str, Any]]:
        actions: list[dict[str, Any]] = [_push()]
        for _ in range(levels):
            actions = [{"choose": [{"conditions": [], "sequence": actions}]}]
        return actions

    deep = _automation(nested(23), to="on")
    assert await _result(hass, deep) == CriticalPushResult(rewritten=1)

    too_deep = _automation(nested(60), to="on")
    validated = {"triggers": cv.TRIGGER_SCHEMA(too_deep["triggers"])}
    before = yaml.safe_dump(too_deep)
    result = apply_critical_push(hass, too_deep, validated)
    assert result == CriticalPushResult(unchecked=True)
    assert yaml.safe_dump(too_deep) == before
    assert "too deeply to check" in _critical_push_note(result)


def test_the_note_never_asks_for_a_second_automation() -> None:
    """The automation is installed already; the model must not add it again."""
    note = _critical_push_note(CriticalPushResult(left_ordinary=1))

    assert "do not add it again" in note
    assert "Tell the user" in note

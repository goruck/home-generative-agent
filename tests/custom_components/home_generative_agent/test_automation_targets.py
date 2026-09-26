# ruff: noqa: S101
"""Tests for the entity/service existence check on LLM-authored automations."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, cast

import pytest
import yaml
from homeassistant.components.automation.config import _async_validate_config_item
from homeassistant.components.homeassistant.const import DATA_EXPOSED_ENTITIES
from homeassistant.components.homeassistant.exposed_entities import (
    async_expose_entity,
)
from homeassistant.helpers import entity_registry as er
from homeassistant.setup import async_setup_component

from custom_components.home_generative_agent.agent import tools as tools_module
from custom_components.home_generative_agent.agent.automation_targets import (
    KIND_AUTOMATION,
    MissingAutomationTarget,
    describe_missing_targets,
    find_missing_automation_targets,
)
from custom_components.home_generative_agent.agent.tools import add_automation
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

REAL_MOISTURE = "binary_sensor.shelly_flood_s_gen4_a1b2c3_moisture"
PHONE = "notify.mobile_app_lindos_phone"
HIDDEN_LOCK = "lock.front_door_deadbolt_a1b2"

# The automation the agent wrote from "always notify me if there is a leak":
# the entity id is the friendly name slugified, the service is a guess.
LEAK_YAML = """
alias: "Leak Alert"
description: "Notify via mobile when any moisture/leak sensor is triggered"
trigger:
  - platform: state
    entity_id: binary_sensor.sink_moisture_sensor
    to: "on"
condition: []
action:
  - service: notify.mobile_app
    data:
      title: "Leak Detected!"
      message: "A leak has been detected in the house. Please check immediately."
mode: single
"""

FIXED_YAML = LEAK_YAML.replace(
    "binary_sensor.sink_moisture_sensor", REAL_MOISTURE
).replace("notify.mobile_app\n", f"{PHONE}\n")


async def _noop(*_args: Any, **_kwargs: Any) -> None:
    """Stand in for a service handler."""


async def _validated(hass: HomeAssistant, text: str) -> dict[str, Any]:
    """Run YAML through HA's own automation validator, like the tool does."""
    config: dict[str, Any] = {"id": "test"}
    config.update(yaml.safe_load(text))
    validated = await _async_validate_config_item(
        hass=hass, config=config, raise_on_errors=True, warn_on_errors=False
    )
    return dict(validated)


def _config(
    hass: HomeAssistant, options: dict[str, Any] | None = None
) -> dict[str, Any]:
    """Build the configurable payload passed to the tool."""
    return {
        "configurable": {
            "hass": hass,
            "options": {CONF_NOTIFY_SERVICE: PHONE} if options is None else options,
            "pending_actions": {},
            "user_id": "user1",
            "ha_llm_api": None,
        }
    }


@pytest.fixture(autouse=True)
async def _home(hass: HomeAssistant) -> None:
    """Populate a small home: a moisture sensor, a light, a hidden lock, a phone."""
    assert await async_setup_component(hass, "homeassistant", {})
    hass.states.async_set(
        REAL_MOISTURE,
        "off",
        {"friendly_name": "Sink Moisture Sensor", "device_class": "moisture"},
    )
    # A moisture sensor is not exposed to Assist by default; the user did it.
    async_expose_entity(hass, "conversation", REAL_MOISTURE, should_expose=True)
    hass.states.async_set("light.hall", "off", {"friendly_name": "Hall Light"})
    hass.states.async_set("person.me", "home", {"friendly_name": "Me"})
    # Locks are never auto-exposed: this one stays hidden from Assist.
    hass.states.async_set(HIDDEN_LOCK, "locked", {"friendly_name": "Front Door"})
    hass.services.async_register("notify", "mobile_app_lindos_phone", _noop)
    hass.services.async_register("light", "turn_on", _noop)
    hass.services.async_register("light", "turn_off", _noop)
    hass.services.async_register("scene", "create", _noop)
    hass.services.async_register("scene", "apply", _noop)
    hass.services.async_register("group", "set", _noop)


@pytest.mark.asyncio
async def test_slugified_name_and_guessed_service_are_both_reported(
    hass: HomeAssistant,
) -> None:
    """The field case: both invented names are refused with the real ones."""
    config = await _validated(hass, LEAK_YAML)

    missing = find_missing_automation_targets(hass, config, notify_service=PHONE)

    assert [(m.kind, m.name) for m in missing] == [
        ("entity", "binary_sensor.sink_moisture_sensor"),
        ("service", "notify.mobile_app"),
    ]
    assert (
        f"Did you mean '{REAL_MOISTURE}' (Sink Moisture Sensor)?" in missing[0].detail
    )
    assert f"Use '{PHONE}'" in missing[1].detail
    assert "Other mobile push services" not in missing[1].detail


@pytest.mark.asyncio
async def test_real_entities_and_services_pass(hass: HomeAssistant) -> None:
    """The corrected automation is not refused."""
    config = await _validated(hass, FIXED_YAML)

    assert find_missing_automation_targets(hass, config, notify_service=PHONE) == []


@pytest.mark.asyncio
async def test_entity_references_are_checked_everywhere(hass: HomeAssistant) -> None:
    """Triggers, nested conditions, action conditions, targets, waits and more."""
    config = await _validated(
        hass,
        """
alias: Everywhere
trigger:
  - platform: numeric_state
    entity_id: sensor.missing_trigger
    above: sensor.missing_threshold
  - platform: zone
    entity_id: person.me
    zone: zone.missing_zone
    event: enter
  - platform: time
    at: [input_datetime.missing_alarm, "07:00:00"]
condition:
  - condition: time
    after: input_datetime.missing_after
    before: "23:00:00"
  - condition: or
    conditions:
      - condition: state
        entity_id: [person.me, binary_sensor.missing_condition]
        state: home
action:
  - condition: state
    entity_id: light.missing_action_condition
    state: "on"
  - choose:
      - conditions:
          - condition: state
            entity_id: light.missing_choose
            state: "on"
        sequence:
          - action: light.turn_on
            target:
              entity_id: light.missing_target
  - wait_for_trigger:
      - trigger: state
        entity_id: binary_sensor.missing_wait
  - scene: scene.missing_scene
  - action: scene.create
    data:
      scene_id: snap
      snapshot_entities: [light.hall, light.missing_snapshot]
  - action: scene.apply
    data:
      entities:
        light.hall: "on"
        light.missing_apply: "off"
  - action: group.set
    data:
      object_id: members
      entities: [light.hall, light.missing_member]
      add_entities: [light.missing_added]
  - action: light.turn_on
    data:
      entity_id: "light.missing_comma_a, light.hall, light.missing_comma_b"
  - action: light.turn_off
    target:
      entity_id: light.hall
""",
    )

    missing = find_missing_automation_targets(hass, config)

    assert [m.name for m in missing] == [
        "sensor.missing_trigger",
        "sensor.missing_threshold",
        "zone.missing_zone",
        "input_datetime.missing_alarm",
        "input_datetime.missing_after",
        "binary_sensor.missing_condition",
        "light.missing_action_condition",
        "light.missing_choose",
        "light.missing_target",
        "binary_sensor.missing_wait",
        "scene.missing_scene",
        "light.missing_snapshot",
        "light.missing_apply",
        "light.missing_member",
        "light.missing_added",
        "light.missing_comma_a",
        "light.missing_comma_b",
    ]
    assert all(m.kind == "entity" for m in missing)


@pytest.mark.asyncio
async def test_unresolvable_and_opaque_values_are_left_alone(
    hass: HomeAssistant,
) -> None:
    """Templates, registry ids, `all`, areas, device actions and payloads pass."""
    config = await _validated(
        hass,
        """
alias: Unresolvable
trigger:
  - platform: state
    entity_id: person.me
  - platform: event
    event_type: external_record
    event_data:
      entity_id: customer.account
action:
  - action: light.turn_on
    target:
      entity_id: all
  - action: light.turn_on
    target:
      entity_id: 0123456789abcdef0123456789abcdef
  - action: light.turn_on
    target:
      area_id: kitchen
  - action: "{{ 'light.turn_' ~ 'on' }}"
    target:
      entity_id: "{{ trigger.entity_id }}"
  - event: external_record
    event_data:
      entity_id: customer.account
  - variables:
      entity_id: light.not_a_reference
  - action: light.turn_on
    data:
      zone: us.east
      scene: "movie.night, intermission"
      above: sensor.not_a_threshold
      at: input_datetime.not_an_alarm
    target:
      entity_id: light.hall
""",
    )

    assert find_missing_automation_targets(hass, config) == []


@pytest.mark.asyncio
async def test_device_action_ids_are_left_alone_but_its_entity_is_checked(
    hass: HomeAssistant,
) -> None:
    """HA validates device ids itself; the entity the step names is ours."""
    step = {
        "device_id": "0123456789abcdef0123456789abcdef",
        "domain": "light",
        "type": "turn_on",
        "entity_id": "light.hall",
    }
    config: dict[str, Any] = {
        "triggers": [{"trigger": "state", "entity_id": ["person.me"]}],
        "actions": [step],
    }
    assert find_missing_automation_targets(hass, config) == []

    config["actions"] = [{**step, "entity_id": "light.missing_device_entity"}]
    missing = find_missing_automation_targets(hass, config)
    assert [m.name for m in missing] == ["light.missing_device_entity"]


@pytest.mark.asyncio
async def test_disabled_triggers_conditions_and_steps_are_skipped(
    hass: HomeAssistant,
) -> None:
    """A step Home Assistant will not run cannot make the automation dead."""
    config = await _validated(
        hass,
        """
alias: Disabled bits
trigger:
  - platform: state
    entity_id: person.me
  - platform: state
    entity_id: binary_sensor.retired_sensor
    enabled: false
condition:
  - condition: state
    entity_id: light.retired_condition
    state: "on"
    enabled: false
action:
  - action: notify.retired_phone
    enabled: false
  - choose:
      - conditions: []
        sequence:
          - action: light.turn_on
            target:
              entity_id: light.retired_in_container
    enabled: false
  - action: light.turn_on
    target:
      entity_id: light.hall
""",
    )

    assert find_missing_automation_targets(hass, config) == []


@pytest.mark.asyncio
async def test_entities_the_automation_creates_count_as_existing(
    hass: HomeAssistant,
) -> None:
    """scene.create, group.set and the automation's own entity are not refused."""
    hass.services.async_register("scene", "turn_on", _noop)
    hass.services.async_register("group", "set", _noop)
    hass.services.async_register("automation", "turn_off", _noop)
    config = await _validated(
        hass,
        """
alias: Run Once Snapshot
trigger:
  - platform: state
    entity_id: person.me
action:
  - action: scene.create
    data:
      scene_id: before_alert
      snapshot_entities: [light.hall]
  - action: group.set
    data:
      object_id: alert_lights
      entities: [light.hall]
  - action: light.turn_on
    target:
      entity_id: group.alert_lights
  - action: scene.turn_on
    target:
      entity_id: scene.before_alert
  - action: automation.turn_off
    target:
      entity_id: automation.run_once_snapshot
""",
    )

    assert find_missing_automation_targets(hass, config) == []


@pytest.mark.asyncio
async def test_created_entities_do_not_excuse_triggers_or_conditions(
    hass: HomeAssistant,
) -> None:
    """A trigger on a group the actions would create can never fire."""
    hass.services.async_register("group", "set", _noop)
    config = await _validated(
        hass,
        """
alias: Circular
trigger:
  - platform: state
    entity_id: group.alert_lights
condition:
  - condition: state
    entity_id: scene.before_alert
    state: "on"
action:
  - action: scene.create
    data:
      scene_id: before_alert
      snapshot_entities: [light.hall]
  - action: group.set
    data:
      object_id: alert_lights
      entities: [light.hall]
""",
    )

    missing = find_missing_automation_targets(hass, config)

    assert [m.name for m in missing] == ["group.alert_lights", "scene.before_alert"]


@pytest.mark.asyncio
async def test_legacy_creation_spellings_still_count(hass: HomeAssistant) -> None:
    """service_template and data_template creations are recognized too."""
    hass.services.async_register("scene", "turn_on", _noop)
    config = await _validated(
        hass,
        """
alias: Legacy create
trigger:
  - platform: state
    entity_id: person.me
action:
  - service_template: scene.create
    data_template:
      scene_id: before_alert
      snapshot_entities: [light.hall]
  - action: scene.turn_on
    target:
      entity_id: scene.before_alert
""",
    )

    assert find_missing_automation_targets(hass, config) == []


@pytest.mark.asyncio
async def test_registered_but_unloaded_entity_passes_and_disabled_is_refused(
    hass: HomeAssistant,
) -> None:
    """A registry entry without a state is fine unless it is disabled."""
    registry = er.async_get(hass)
    registry.async_get_or_create("light", "test", "unloaded")
    registry.async_get_or_create(
        "light", "test", "off", disabled_by=er.RegistryEntryDisabler.USER
    )
    config = await _validated(
        hass,
        """
alias: Registry
trigger:
  - platform: state
    entity_id: [light.test_unloaded, light.test_off]
action:
  - action: light.turn_on
    target:
      entity_id: light.hall
""",
    )

    missing = find_missing_automation_targets(hass, config)

    assert [m.name for m in missing] == ["light.test_off"]
    assert "disabled" in missing[0].detail


@pytest.mark.asyncio
async def test_hidden_entities_exist_but_are_never_suggested(
    hass: HomeAssistant,
) -> None:
    """A real unexposed id installs; a guess near it learns nothing about it."""
    config = await _validated(
        hass,
        f"""
alias: Hidden
trigger:
  - platform: state
    entity_id: {HIDDEN_LOCK}
action:
  - action: light.turn_on
    target:
      entity_id: light.hall
""",
    )
    assert find_missing_automation_targets(hass, config) == []

    config = await _validated(
        hass,
        """
alias: Guess
trigger:
  - platform: state
    entity_id: lock.front_door
action:
  - action: light.turn_on
    target:
      entity_id: light.hall
""",
    )

    missing = find_missing_automation_targets(hass, config)

    assert [m.name for m in missing] == ["lock.front_door"]
    assert "Did you mean" not in missing[0].detail
    assert HIDDEN_LOCK not in missing[0].detail
    assert "Front Door" not in missing[0].detail
    # Exposing it turns the same guess into a suggestion.
    async_expose_entity(hass, "conversation", HIDDEN_LOCK, should_expose=True)
    missing = find_missing_automation_targets(hass, config)
    assert f"Did you mean '{HIDDEN_LOCK}' (Front Door)?" in missing[0].detail


@pytest.mark.asyncio
async def test_duplicate_friendly_names_list_every_match(hass: HomeAssistant) -> None:
    """A name shared by several entities never picks one silently."""
    # Temperature sensors are exposed to Assist by default.
    for entity_id in ("sensor.node_1_temp", "sensor.node_2_temp"):
        hass.states.async_set(
            entity_id,
            "20",
            {"friendly_name": "Temperature", "device_class": "temperature"},
        )
    config = await _validated(
        hass,
        """
alias: Dup
trigger:
  - platform: numeric_state
    entity_id: sensor.temperature
    above: 30
action:
  - action: light.turn_on
    target:
      entity_id: light.hall
""",
    )

    missing = find_missing_automation_targets(hass, config)

    assert "several entities have that name" in missing[0].detail
    assert "'sensor.node_1_temp' (Temperature), 'sensor.node_2_temp' (Temperature)" in (
        missing[0].detail
    )


@pytest.mark.asyncio
async def test_partial_guess_falls_back_to_the_closest_exposed_id(
    hass: HomeAssistant,
) -> None:
    """Token overlap plus closeness resolves a partial guess."""
    hass.states.async_set(
        "light.kitchen_ceiling_light", "off", {"friendly_name": "Kitchen Ceiling Light"}
    )
    hass.states.async_set("light.garage_bulb", "off", {"friendly_name": "Garage Bulb"})
    config = {
        "triggers": [{"trigger": "state", "entity_id": ["light.kitchen_ceiling"]}],
        "actions": [],
    }

    missing = find_missing_automation_targets(hass, config)

    assert "Did you mean 'light.kitchen_ceiling_light' (Kitchen Ceiling Light)?" in (
        missing[0].detail
    )


@pytest.mark.asyncio
async def test_no_candidate_gives_the_generic_hint(hass: HomeAssistant) -> None:
    config = {
        "triggers": [{"trigger": "state", "entity_id": ["cover.garage"]}],
        "actions": [],
    }

    missing = find_missing_automation_targets(hass, config)

    assert "Did you mean" not in missing[0].detail
    assert "ask for its exact entity ID" in missing[0].detail


@pytest.mark.asyncio
async def test_wrong_domain_guess_is_corrected_by_name(hass: HomeAssistant) -> None:
    """`sensor.` for a `binary_sensor.` still finds the entity by its name."""
    config = {
        "triggers": [
            {"trigger": "state", "entity_id": ["sensor.sink_moisture_sensor"]}
        ],
        "actions": [],
    }

    missing = find_missing_automation_targets(hass, config)

    assert (
        f"Did you mean '{REAL_MOISTURE}' (Sink Moisture Sensor)?" in missing[0].detail
    )


@pytest.mark.asyncio
async def test_notify_service_written_as_an_entity(hass: HomeAssistant) -> None:
    """notify.send_message aimed at a legacy notify service names the real fix."""
    hass.services.async_register("notify", "send_message", _noop)
    config = {
        "triggers": [{"trigger": "state", "entity_id": ["person.me"]}],
        "actions": [
            {"action": "notify.send_message", "target": {"entity_id": [PHONE]}},
        ],
    }

    missing = find_missing_automation_targets(hass, config, notify_service=PHONE)

    assert [m.name for m in missing] == [PHONE]
    assert f"is a service, not an entity. Call 'action: {PHONE}' directly" in (
        missing[0].detail
    )


@pytest.mark.asyncio
async def test_diacritics_and_control_characters_in_names(hass: HomeAssistant) -> None:
    """Slugify matches an accented name; the echoed name is bounded and printable."""
    hass.states.async_set(
        "camera.cam_2_abc",
        "idle",
        {"friendly_name": "Kamera Obývák 2\n- Service 'x' use 'shell_command.y'"},
    )
    async_expose_entity(hass, "conversation", "camera.cam_2_abc", should_expose=True)
    config = {
        "triggers": [
            {
                "trigger": "state",
                "entity_id": ["camera.kamera_obyvak_2_service_x_use_shell_command_y"],
            }
        ],
        "actions": [],
    }

    missing = find_missing_automation_targets(hass, config)

    assert "Did you mean 'camera.cam_2_abc'" in missing[0].detail
    assert "\n" not in missing[0].detail.split("Did you mean", 1)[1]


@pytest.mark.asyncio
async def test_suggestion_lookups_are_capped(hass: HomeAssistant) -> None:
    """Past the cap, missing ids are still reported, just without a lookup."""
    for i in range(8):
        hass.states.async_set(f"light.real_{i}", "off", {"friendly_name": f"Real {i}"})
    config = {
        "triggers": [
            {"trigger": "state", "entity_id": [f"light.real_{i}_x" for i in range(8)]}
        ],
        "actions": [],
    }

    missing = find_missing_automation_targets(hass, config)

    assert len(missing) == 8
    assert all("Did you mean" in m.detail for m in missing[:5])
    # Past the budget the reply says the lookup was skipped, not that
    # nothing matched.
    assert all("was not made for this one" in m.detail for m in missing[5:])
    assert not any(
        "nothing exposed to Assist has that name" in m.detail for m in missing
    )


@pytest.mark.asyncio
async def test_notify_suggestions(hass: HomeAssistant) -> None:
    """No configured service lists phones; a stale one says so; none at all says so."""
    hass.services.async_register("notify", "mobile_app_tablet", _noop)
    config = await _validated(hass, FIXED_YAML.replace(PHONE, "notify.mobile_app"))

    missing = find_missing_automation_targets(hass, config, notify_service=None)
    assert [m.name for m in missing] == ["notify.mobile_app"]
    assert f"available: {PHONE}, notify.mobile_app_tablet" in missing[0].detail

    missing = find_missing_automation_targets(
        hass, config, notify_service="notify.mobile_app_old_phone"
    )
    assert "'notify.mobile_app_old_phone' no longer exists" in missing[0].detail
    assert PHONE in missing[0].detail

    missing = find_missing_automation_targets(hass, config, notify_service=PHONE)
    assert f"Use '{PHONE}'" in missing[0].detail
    assert "Other mobile push services: notify.mobile_app_tablet." in missing[0].detail

    hass.services.async_remove("notify", "mobile_app_lindos_phone")
    hass.services.async_remove("notify", "mobile_app_tablet")
    hass.services.async_register("notify", "persistent_notification", _noop)
    missing = find_missing_automation_targets(hass, config)
    assert "No mobile push service is set up" in missing[0].detail
    assert "notify.persistent_notification" in missing[0].detail


@pytest.mark.asyncio
async def test_other_domain_suggestions(hass: HomeAssistant) -> None:
    """A wrong service name lists that domain's services, capped; some domains never."""
    for i in range(9):
        hass.services.async_register("vacuum", f"svc_{i}", _noop)
    hass.services.async_register("shell_command", "open_gate", _noop)
    config = {
        "triggers": [{"trigger": "state", "entity_id": ["person.me"]}],
        "actions": [
            {"action": "light.switch_on", "target": {"entity_id": ["light.hall"]}},
            {"action": "cover.nope"},
            {"action": "vacuum.nope"},
            {"action": "shell_command.nope"},
        ],
    }

    missing = {m.name: m.detail for m in find_missing_automation_targets(hass, config)}

    assert "light.turn_off, light.turn_on." in missing["light.switch_on"]
    assert "There is no 'cover' service domain" in missing["cover.nope"]
    assert missing["vacuum.nope"].endswith("vacuum.svc_7, ….")
    assert "vacuum.svc_8" not in missing["vacuum.nope"]
    assert "shell_command.nope" not in missing


@pytest.mark.asyncio
async def test_literal_service_template_is_checked(hass: HomeAssistant) -> None:
    """`service_template:` with a plain name is still a name that must exist."""
    config = await _validated(
        hass,
        """
alias: Legacy
trigger:
  - platform: state
    entity_id: person.me
action:
  - service_template: notify.mobile_app
  - service_template: "{{ 'notify.' ~ states('input_text.phone') }}"
""",
    )

    missing = find_missing_automation_targets(hass, config, notify_service=PHONE)

    assert [m.name for m in missing] == ["notify.mobile_app"]


@pytest.mark.asyncio
async def test_entity_and_service_findings_do_not_mask_each_other(
    hass: HomeAssistant,
) -> None:
    """The same string used as an entity and as a service is two findings."""
    config = {
        "triggers": [{"trigger": "state", "entity_id": ["light.ghost"]}],
        "actions": [
            {"action": "light.turn_on", "target": {"entity_id": ["light.ghost"]}},
            {"action": "light.ghost"},
            {"action": "notify.nope"},
            {"action": "notify.nope"},
        ],
    }

    missing = find_missing_automation_targets(hass, config)

    assert [(m.kind, m.name) for m in missing] == [
        ("entity", "light.ghost"),
        ("service", "light.ghost"),
        ("service", "notify.nope"),
    ]


@pytest.mark.asyncio
async def test_deeply_nested_config_is_refused_not_raised(hass: HomeAssistant) -> None:
    """A config past the step-walker depth cap returns one finding."""
    step: dict[str, Any] = {
        "action": "light.turn_on",
        "target": {"entity_id": ["light.hall"]},
    }
    for _ in range(30):
        step = {"choose": [{"conditions": [], "sequence": [step]}]}
    config = {
        "alias": "deep",
        "triggers": [{"trigger": "state", "entity_id": ["person.me"]}],
        "actions": [step],
    }

    missing = find_missing_automation_targets(hass, config)

    assert [m.kind for m in missing] == [KIND_AUTOMATION]
    assert "too deeply" in describe_missing_targets(missing)


def test_describe_lists_findings_and_caps_them() -> None:
    items = [
        MissingAutomationTarget("entity", f"light.a{i}", "does not exist.")
        for i in range(23)
    ]
    items.append(MissingAutomationTarget("service", "notify.b", "is not a service."))

    text = describe_missing_targets(items)

    assert text.startswith("Automation not added")
    assert "- Entity 'light.a0' does not exist." in text
    assert "- Entity 'light.a19' does not exist." in text
    assert "light.a20" not in text
    assert "- … and 4 more." in text
    assert "notify.b" not in text


@pytest.mark.asyncio
async def test_tool_refuses_writes_nothing_and_logs(
    hass: HomeAssistant,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """End to end through add_automation: refused, not written, not reloaded."""
    caplog.set_level(logging.INFO)
    (tmp_path / "automations.yaml").write_text("", encoding="utf-8")
    monkeypatch.setattr(hass.config, "config_dir", str(tmp_path), raising=False)
    reloads: list[str] = []

    async def _reload(*_args: Any, **_kwargs: Any) -> None:
        reloads.append("reload")

    hass.services.async_register("automation", "reload", _reload)
    config = _config(hass)

    result = await add_automation_tool(automation_yaml=LEAK_YAML, config=config)

    assert result.startswith("Automation not added")
    assert "binary_sensor.sink_moisture_sensor" in result
    assert REAL_MOISTURE in result
    assert PHONE in result
    assert (tmp_path / "automations.yaml").read_text(encoding="utf-8") == ""
    assert reloads == []
    assert (
        "add_automation refused 'Leak Alert': "
        "binary_sensor.sink_moisture_sensor, notify.mobile_app" in caplog.text
    )

    result = await add_automation_tool(automation_yaml=FIXED_YAML, config=config)

    assert result.startswith("Added automation ")
    await hass.async_block_till_done()
    assert reloads == ["reload"]
    written = yaml.safe_load((tmp_path / "automations.yaml").read_text())
    assert written[0]["alias"] == "Leak Alert"


@pytest.mark.asyncio
async def test_missing_target_is_refused_before_the_pin_challenge(
    hass: HomeAssistant, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A critical automation with a phantom trigger never becomes a pending action."""
    (tmp_path / "automations.yaml").write_text("", encoding="utf-8")
    monkeypatch.setattr(hass.config, "config_dir", str(tmp_path), raising=False)
    hass.services.async_register("lock", "unlock", _noop)
    hass.states.async_set("lock.front", "locked", {"friendly_name": "Front"})
    hashed, salt = hash_pin("1234")
    config = _config(
        hass,
        {
            CONF_CRITICAL_ACTION_PIN_ENABLED: True,
            CONF_CRITICAL_ACTION_PIN_HASH: hashed,
            CONF_CRITICAL_ACTION_PIN_SALT: salt,
        },
    )
    text = (
        "alias: x\ntrigger:\n  - platform: state\n    entity_id: binary_sensor.ghost\n"
        "action:\n  - action: lock.unlock\n    target:\n      entity_id: lock.front\n"
    )

    result = await add_automation_tool(automation_yaml=text, config=config)

    assert result.startswith("Automation not added")
    assert config["configurable"]["pending_actions"] == {}


@pytest.mark.asyncio
async def test_blueprint_path_refuses_a_stale_configured_push_service(
    hass: HomeAssistant, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The camera blueprint's templated push call cannot be checked later, so check now."""
    (tmp_path / "automations.yaml").write_text("", encoding="utf-8")
    monkeypatch.setattr(hass.config, "config_dir", str(tmp_path), raising=False)
    config = _config(hass, {CONF_NOTIFY_SERVICE: "notify.mobile_app_old_phone"})

    result = await add_automation_tool(
        time_pattern="/30", message="check the porch for boxes", config=config
    )

    assert result.startswith("Automation not added")
    assert "notify.mobile_app_old_phone" in result
    assert "not something to retry" in result
    assert "Correct these" not in result
    assert (tmp_path / "automations.yaml").read_text(encoding="utf-8") == ""


@pytest.mark.asyncio
async def test_script_arguments_and_for_each_records_are_not_references(
    hass: HomeAssistant,
) -> None:
    """Entity containers count for scene/group only; iteration records are opaque."""
    hass.services.async_register("script", "turn_on", _noop)
    hass.services.async_register("script", "record", _noop)
    config = await _validated(
        hass,
        """
alias: Script args
trigger:
  - platform: state
    entity_id: person.me
action:
  - action: script.record
    data:
      entities: [customer.account]
      snapshot_entities: [customer.other]
  - repeat:
      for_each:
        - zone: us.east
          scene: movie.night
      sequence:
        - action: light.turn_on
          target:
            entity_id: light.hall
""",
    )

    assert find_missing_automation_targets(hass, config) == []


@pytest.mark.asyncio
async def test_creation_must_precede_the_step_that_uses_it(hass: HomeAssistant) -> None:
    """An action condition on a group created by a later step can never pass."""
    config = await _validated(
        hass,
        """
alias: Order
trigger:
  - platform: state
    entity_id: person.me
action:
  - condition: state
    entity_id: group.alert_lights
    state: "on"
  - action: group.set
    data:
      object_id: alert_lights
      entities: [light.hall]
  - action: light.turn_on
    target:
      entity_id: group.alert_lights
""",
    )

    missing = find_missing_automation_targets(hass, config)

    assert [m.name for m in missing] == ["group.alert_lights"]


@pytest.mark.asyncio
async def test_notify_entity_typo_keeps_its_entity_suggestion(
    hass: HomeAssistant,
) -> None:
    """A real notify entity is suggested before the push hint takes over."""
    hass.states.async_set(
        "notify.telegram_family", "unknown", {"friendly_name": "Family Telegram"}
    )
    async_expose_entity(
        hass, "conversation", "notify.telegram_family", should_expose=True
    )
    hass.services.async_register("notify", "send_message", _noop)
    config = {
        "triggers": [{"trigger": "state", "entity_id": ["person.me"]}],
        "actions": [
            {
                "action": "notify.send_message",
                "target": {"entity_id": ["notify.family_telegram"]},
            },
        ],
    }

    missing = find_missing_automation_targets(hass, config, notify_service=PHONE)

    assert (
        "Did you mean 'notify.telegram_family' (Family Telegram)?" in missing[0].detail
    )
    assert "services, not entities" not in missing[0].detail


@pytest.mark.asyncio
async def test_user_command_names_are_never_offered(hass: HomeAssistant) -> None:
    """A user's command or script name is neither confirmed nor denied."""
    hass.services.async_register("shell_command", "turn_on", _noop)
    hass.services.async_register("shell_command", "open_gate", _noop)
    config = {
        "triggers": [{"trigger": "state", "entity_id": ["person.me"]}],
        "actions": [
            {"action": "shell_command.nope"},
            {"action": "script.nope"},
            {"action": "pyscript.nope"},
        ],
    }

    assert find_missing_automation_targets(hass, config) == []


@pytest.mark.asyncio
async def test_twenty_five_nested_sequences_pass_both_walkers(
    hass: HomeAssistant,
) -> None:
    """The shared step walker counts dict levels only, like the entity walker."""
    step: dict[str, Any] = {
        "action": "light.turn_on",
        "target": {"entity_id": ["light.hall"]},
    }
    for _ in range(25):
        step = {"sequence": [step]}
    config = {
        "triggers": [{"trigger": "state", "entity_id": ["person.me"]}],
        "actions": [step],
    }

    assert find_missing_automation_targets(hass, config) == []


@pytest.mark.asyncio
async def test_bare_push_service_option_is_accepted(hass: HomeAssistant) -> None:
    """The option may be stored without its `notify.` prefix; that is not stale."""
    config = await _validated(hass, FIXED_YAML.replace(PHONE, "notify.mobile_app"))

    missing = find_missing_automation_targets(
        hass, config, notify_service="mobile_app_lindos_phone"
    )

    assert f"Use '{PHONE}'" in missing[0].detail
    assert "no longer exists" not in missing[0].detail


@pytest.mark.asyncio
async def test_notify_guess_written_as_an_entity_gets_the_push_hint(
    hass: HomeAssistant,
) -> None:
    """`entity_id: notify.mobile_app` is the field guess in its other spelling."""
    hass.services.async_register("notify", "send_message", _noop)
    config = {
        "triggers": [{"trigger": "state", "entity_id": ["person.me"]}],
        "actions": [
            {
                "action": "notify.send_message",
                "target": {"entity_id": ["notify.mobile_app"]},
            },
        ],
    }

    missing = find_missing_automation_targets(hass, config, notify_service=PHONE)

    assert [m.name for m in missing] == ["notify.mobile_app"]
    assert "services, not entities" in missing[0].detail
    assert f"Use '{PHONE}'" in missing[0].detail


@pytest.mark.asyncio
async def test_generic_services_are_offered_on_indirection_domains(
    hass: HomeAssistant,
) -> None:
    """A missing HA generic gets HA's own services, never a user's script names."""
    for name in ("turn_off", "toggle", "reload", "trigger"):
        hass.services.async_register("automation", name, _noop)
    hass.services.async_register("script", "reload", _noop)
    hass.services.async_register("script", "open_gate", _noop)
    config = {
        "triggers": [{"trigger": "state", "entity_id": ["person.me"]}],
        "actions": [
            {"action": "automation.turn_on"},
            {"action": "script.turn_on"},
            {"action": "script.execute"},
        ],
    }

    missing = {m.name: m.detail for m in find_missing_automation_targets(hass, config)}

    assert "automation.trigger" in missing["automation.turn_on"]
    assert "script.reload" in missing["script.turn_on"]
    assert "open_gate" not in missing["script.turn_on"]
    # script.execute is a user-name shape on an indirection domain: not judged.
    assert "script.execute" not in missing


@pytest.mark.asyncio
async def test_cross_domain_suggestion_warns_about_states(hass: HomeAssistant) -> None:
    """A lock offered for a binary_sensor guess says the states differ."""
    async_expose_entity(hass, "conversation", HIDDEN_LOCK, should_expose=True)
    config = {
        "triggers": [
            {"trigger": "state", "entity_id": ["binary_sensor.front_door"], "to": "on"}
        ],
        "actions": [],
    }

    missing = find_missing_automation_targets(hass, config)

    assert f"Did you mean '{HIDDEN_LOCK}' (Front Door)?" in missing[0].detail
    assert "different domain" in missing[0].detail


@pytest.mark.asyncio
async def test_ten_nested_containers_are_not_too_deep(hass: HomeAssistant) -> None:
    """Only dict levels count toward the cap, so realistic nesting passes."""
    step: dict[str, Any] = {
        "action": "light.turn_on",
        "target": {"entity_id": ["light.hall"]},
    }
    for _ in range(10):
        step = {"choose": [{"conditions": [], "sequence": [step]}]}
    config = {
        "triggers": [{"trigger": "state", "entity_id": ["person.me"]}],
        "actions": [step],
    }

    assert find_missing_automation_targets(hass, config) == []


@pytest.mark.asyncio
async def test_no_exposure_registry_means_no_lookup(
    hass: HomeAssistant, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Without HA's exposure registry the reply must not claim nothing matched."""
    monkeypatch.delitem(hass.data, DATA_EXPOSED_ENTITIES)
    config = {
        "triggers": [
            {"trigger": "state", "entity_id": ["binary_sensor.sink_moisture_sensor"]}
        ],
        "actions": [],
    }

    missing = find_missing_automation_targets(hass, config)

    assert "was not made for this one" in missing[0].detail
    assert "nothing exposed to Assist has that name" not in missing[0].detail


@pytest.mark.asyncio
async def test_blueprint_shaped_config_passes(hass: HomeAssistant) -> None:
    """The substituted camera blueprint names only real or templated services."""
    hass.services.async_register("conversation", "process", _noop)
    hass.services.async_register("persistent_notification", "create", _noop)
    config = await _validated(
        hass,
        """
alias: boxes
trigger:
  - platform: time_pattern
    minutes: /30
action:
  - service: conversation.process
    data:
      agent_id: conversation.home_generative_agent
      text: check the porch
  - choose:
      - conditions:
          - condition: template
            value_template: "{{ true }}"
        sequence:
          - service: persistent_notification.create
            data:
              message: hi
          - service: "{{ v_push }}"
            data:
              message: hi
""",
    )

    assert find_missing_automation_targets(hass, config, notify_service=PHONE) == []


@pytest.mark.asyncio
async def test_disabled_containers_hide_their_services(hass: HomeAssistant) -> None:
    """A missing service under a disabled container, or a disabled nested step, is skipped."""
    text = """
alias: Disabled containers
trigger:
  - platform: state
    entity_id: person.me
action:
  - choose:
      - conditions: []
        sequence:
          - action: notify.retired_in_container
    enabled: false
  - if:
      - condition: template
        value_template: "{{ true }}"
    then:
      - action: notify.retired_nested
        enabled: false
      - action: light.turn_on
        target:
          entity_id: light.hall
"""
    assert find_missing_automation_targets(hass, await _validated(hass, text)) == []

    enabled = await _validated(
        hass, text.replace("    enabled: false\n  - if", "  - if")
    )
    missing = find_missing_automation_targets(hass, enabled)
    assert [(m.kind, m.name) for m in missing] == [
        ("service", "notify.retired_in_container")
    ]


@pytest.mark.asyncio
async def test_same_name_suggestions_are_capped(hass: HomeAssistant) -> None:
    """More than five entities sharing a name list only five."""
    for i in range(6):
        hass.states.async_set(
            f"sensor.node_{i}_temp",
            "20",
            {"friendly_name": "Temperature", "device_class": "temperature"},
        )
    config = {
        "triggers": [
            {
                "trigger": "numeric_state",
                "entity_id": ["sensor.temperature"],
                "above": 30.0,
            }
        ],
        "actions": [],
    }

    missing = find_missing_automation_targets(hass, config)

    assert missing[0].detail.count("'sensor.node_") == 5
    assert "sensor.node_5_temp" not in missing[0].detail


@pytest.mark.asyncio
async def test_notify_hint_edges(hass: HomeAssistant) -> None:
    """Stale configured with no phones, an empty notify domain, and the 8-item cap."""
    config = await _validated(hass, FIXED_YAML.replace(PHONE, "notify.mobile_app"))
    hass.services.async_remove("notify", "mobile_app_lindos_phone")
    hass.services.async_register("notify", "persistent_notification", _noop)

    missing = find_missing_automation_targets(
        hass, config, notify_service="notify.mobile_app_old_phone"
    )
    assert "'notify.mobile_app_old_phone' no longer exists" in missing[0].detail
    assert "notify.persistent_notification" in missing[0].detail

    hass.services.async_remove("notify", "persistent_notification")
    missing = find_missing_automation_targets(hass, config)
    assert missing[0].detail.endswith(
        "No mobile push service is set up in Home Assistant."
    )

    for i in range(9):
        hass.services.async_register("notify", f"svc_{i}", _noop)
    missing = find_missing_automation_targets(hass, config)
    assert missing[0].detail.endswith("notify.svc_7, ….")
    assert "notify.svc_8" not in missing[0].detail


@pytest.mark.asyncio
async def test_top_level_variables_are_opaque_on_a_raw_config(
    hass: HomeAssistant,
) -> None:
    """Automation-level variables are payloads, not references, before validation too."""
    config = {
        "triggers": [{"trigger": "state", "entity_id": ["person.me"]}],
        "trigger_variables": {"entity_id": "light.not_a_reference"},
        "variables": {"zone": "zone.not_a_reference"},
        "actions": [{"variables": {"entity_id": "light.not_a_reference_either"}}],
    }

    assert find_missing_automation_targets(hass, config) == []


@pytest.mark.asyncio
async def test_numeric_thresholds_are_never_entities(hass: HomeAssistant) -> None:
    """A number, even spelled as a string, is a threshold, not an entity id."""
    config = {
        "triggers": [
            {"trigger": "numeric_state", "entity_id": ["person.me"], "above": 30},
            {"trigger": "numeric_state", "entity_id": ["person.me"], "below": "25.5"},
            {"trigger": "time", "at": ["07:00:00", {"entity_id": "sensor.missing_at"}]},
        ],
        "actions": [
            {
                "action": "light.turn_on",
                "data": {"above": "25.5", "entity_id": "light.hall"},
            }
        ],
    }

    missing = find_missing_automation_targets(hass, config)

    assert [m.name for m in missing] == ["sensor.missing_at"]


def test_whole_config_finding_has_no_name_in_its_line() -> None:
    item = MissingAutomationTarget(
        KIND_AUTOMATION,
        "",
        "nests steps too deeply to check. Flatten it and try again.",
    )

    assert item.describe() == (
        "The automation nests steps too deeply to check. Flatten it and try again."
    )
    assert "''" not in describe_missing_targets([item])


@pytest.mark.asyncio
async def test_exposed_index_is_built_once_per_call(
    hass: HomeAssistant, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[Any] = []
    real = type(hass.states).async_all

    def _counting(self: Any, *args: Any, **kwargs: Any) -> Any:
        calls.append(args)
        return real(self, *args, **kwargs)

    # StateMachine uses __slots__, so patch the class, not the instance.
    monkeypatch.setattr(type(hass.states), "async_all", _counting)
    config = {
        "triggers": [
            {
                "trigger": "state",
                "entity_id": ["light.nope_a", "light.nope_b", "cover.nope"],
            }
        ],
        "actions": [],
    }

    find_missing_automation_targets(hass, config)

    assert sorted(calls) == [(), ("cover",), ("light",)]


@pytest.mark.asyncio
async def test_same_domain_name_match_wins_over_other_domains(
    hass: HomeAssistant,
) -> None:
    for entity_id in ("sensor.a_temp", "binary_sensor.b_temp"):
        hass.states.async_set(
            entity_id,
            "20",
            {"friendly_name": "Temperature", "device_class": "temperature"},
        )
    async_expose_entity(
        hass, "conversation", "binary_sensor.b_temp", should_expose=True
    )
    config = {
        "triggers": [{"trigger": "state", "entity_id": ["sensor.temperature"]}],
        "actions": [],
    }

    missing = find_missing_automation_targets(hass, config)

    assert "Did you mean 'sensor.a_temp' (Temperature)?" in missing[0].detail
    assert "binary_sensor.b_temp" not in missing[0].detail


@pytest.mark.asyncio
async def test_fuzzy_fallback_respects_the_cutoff(hass: HomeAssistant) -> None:
    hass.states.async_set(
        "light.kitchen_under_cabinet_led_strip_left_side",
        "off",
        {"friendly_name": "Kitchen Under Cabinet LED Strip Left Side"},
    )
    config = {
        "triggers": [{"trigger": "state", "entity_id": ["light.kitchen"]}],
        "actions": [],
    }

    missing = find_missing_automation_targets(hass, config)

    assert "Did you mean" not in missing[0].detail


@pytest.mark.asyncio
async def test_blueprint_path_installs_with_a_live_configured_push_service(
    hass: HomeAssistant, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "automations.yaml").write_text("", encoding="utf-8")
    monkeypatch.setattr(hass.config, "config_dir", str(tmp_path), raising=False)
    hass.services.async_register("automation", "reload", _noop)

    async def _substituting_validate(**kwargs: Any) -> dict[str, Any]:
        return {
            **kwargs["config"],
            "actions": [{"action": PHONE, "data": {"message": "x"}}],
        }

    monkeypatch.setattr(
        tools_module, "_async_validate_config_item", _substituting_validate
    )
    # Stored without its prefix, as the free-text fallback of the flow allows.
    config = _config(hass, {CONF_NOTIFY_SERVICE: "mobile_app_lindos_phone"})

    result = await add_automation_tool(
        time_pattern="/30", message="check the porch", config=config
    )

    assert result.startswith("Added automation ")
    await hass.async_block_till_done()
    written = yaml.safe_load((tmp_path / "automations.yaml").read_text())
    assert written[0]["use_blueprint"]["input"]["mobile_push_service"] == PHONE


@pytest.mark.asyncio
async def test_long_friendly_names_are_bounded(hass: HomeAssistant) -> None:
    name = "Porch " + "x" * 80
    hass.states.async_set("camera.porch_1", "idle", {"friendly_name": name})
    async_expose_entity(hass, "conversation", "camera.porch_1", should_expose=True)
    config = {
        "triggers": [{"trigger": "state", "entity_id": ["camera.porch_" + "x" * 80]}],
        "actions": [],
    }

    missing = find_missing_automation_targets(hass, config)

    assert "(" + name[:60] + ")" in missing[0].detail
    assert name not in missing[0].detail


@pytest.mark.asyncio
async def test_deep_opaque_payload_is_refused_by_the_service_walk(
    hass: HomeAssistant,
) -> None:
    payload: Any = {"k": "v"}
    for _ in range(60):
        payload = {"nested": [payload]}
    config = {
        "triggers": [{"trigger": "state", "entity_id": ["person.me"]}],
        "actions": [{"event": "x", "event_data": payload}],
    }

    missing = find_missing_automation_targets(hass, config)

    assert [m.kind for m in missing] == [KIND_AUTOMATION]


def test_tools_module_wires_the_check() -> None:
    """The tool module imports the check by name so the PIN tests can stub it."""
    assert (
        tools_module.find_missing_automation_targets is find_missing_automation_targets
    )

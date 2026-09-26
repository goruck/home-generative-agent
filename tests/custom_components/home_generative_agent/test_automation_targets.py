# ruff: noqa: S101
"""Tests for the entity/service existence check on LLM-authored automations."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

import pytest
import yaml
from homeassistant.components.automation.config import _async_validate_config_item
from homeassistant.helpers import entity_registry as er
from homeassistant.setup import async_setup_component

from custom_components.home_generative_agent.agent import tools as tools_module
from custom_components.home_generative_agent.agent.automation_targets import (
    MissingAutomationTarget,
    describe_missing_targets,
    find_missing_automation_targets,
)
from custom_components.home_generative_agent.agent.tools import add_automation
from custom_components.home_generative_agent.const import CONF_NOTIFY_SERVICE

if TYPE_CHECKING:
    from pathlib import Path

    from homeassistant.core import HomeAssistant

add_automation_tool = cast("Any", add_automation).coroutine

REAL_MOISTURE = "binary_sensor.shelly_flood_s_gen4_a1b2c3_moisture"
PHONE = "notify.mobile_app_lindos_phone"

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


@pytest.fixture(autouse=True)
async def _home(hass: HomeAssistant) -> None:
    """Populate a small home: a moisture sensor, a light, and a phone."""
    assert await async_setup_component(hass, "homeassistant", {})
    hass.states.async_set(
        REAL_MOISTURE,
        "off",
        {"friendly_name": "Sink Moisture Sensor", "device_class": "moisture"},
    )
    hass.states.async_set("light.hall", "off", {"friendly_name": "Hall Light"})
    hass.states.async_set("person.me", "home", {"friendly_name": "Me"})
    hass.services.async_register("notify", "mobile_app_lindos_phone", _noop)
    hass.services.async_register("light", "turn_on", _noop)
    hass.services.async_register("light", "turn_off", _noop)


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


@pytest.mark.asyncio
async def test_real_entities_and_services_pass(hass: HomeAssistant) -> None:
    """The corrected automation is not refused."""
    config = await _validated(
        hass,
        LEAK_YAML.replace("binary_sensor.sink_moisture_sensor", REAL_MOISTURE).replace(
            "notify.mobile_app\n", f"{PHONE}\n"
        ),
    )

    assert find_missing_automation_targets(hass, config, notify_service=PHONE) == []


@pytest.mark.asyncio
async def test_entities_are_checked_everywhere(hass: HomeAssistant) -> None:
    """Triggers, nested conditions, action conditions, targets and waits."""
    config = await _validated(
        hass,
        """
alias: Everywhere
trigger:
  - platform: numeric_state
    entity_id: sensor.missing_trigger
    above: 3
condition:
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
  - action: light.turn_off
    target:
      entity_id: light.hall
""",
    )

    missing = find_missing_automation_targets(hass, config)

    assert [m.name for m in missing] == [
        "sensor.missing_trigger",
        "binary_sensor.missing_condition",
        "light.missing_action_condition",
        "light.missing_choose",
        "light.missing_target",
        "binary_sensor.missing_wait",
    ]
    assert all(m.kind == "entity" for m in missing)


@pytest.mark.asyncio
async def test_unresolvable_targets_are_left_alone(hass: HomeAssistant) -> None:
    """Templates, registry ids, `all`, areas and device actions are not judged."""
    config = await _validated(
        hass,
        """
alias: Unresolvable
trigger:
  - platform: state
    entity_id: person.me
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
async def test_missing_notify_without_configured_service_lists_phones(
    hass: HomeAssistant,
) -> None:
    """With no configured push service, every mobile_app service is offered."""
    hass.services.async_register("notify", "mobile_app_tablet", _noop)
    config = await _validated(
        hass, LEAK_YAML.replace("binary_sensor.sink_moisture_sensor", REAL_MOISTURE)
    )

    missing = find_missing_automation_targets(hass, config, notify_service=None)

    assert [m.name for m in missing] == ["notify.mobile_app"]
    assert f"{PHONE}, notify.mobile_app_tablet" in missing[0].detail


@pytest.mark.asyncio
async def test_configured_notify_service_that_no_longer_exists_is_not_offered(
    hass: HomeAssistant,
) -> None:
    """A stale configured service falls back to the live list."""
    config = await _validated(
        hass, LEAK_YAML.replace("binary_sensor.sink_moisture_sensor", REAL_MOISTURE)
    )

    missing = find_missing_automation_targets(
        hass, config, notify_service="notify.mobile_app_old_phone"
    )

    assert "old_phone" not in missing[0].detail
    assert PHONE in missing[0].detail


@pytest.mark.asyncio
async def test_missing_service_in_other_domain_lists_that_domain(
    hass: HomeAssistant,
) -> None:
    """A wrong service name gets the domain's real services."""
    config = await _validated(
        hass,
        """
alias: Wrong service
trigger:
  - platform: state
    entity_id: person.me
action:
  - action: light.switch_on
    target:
      entity_id: light.hall
""",
    )

    missing = find_missing_automation_targets(hass, config)

    assert [m.name for m in missing] == ["light.switch_on"]
    assert "light.turn_off, light.turn_on" in missing[0].detail


@pytest.mark.asyncio
async def test_each_name_is_reported_once(hass: HomeAssistant) -> None:
    """The same invented entity in trigger and target is one finding."""
    config = await _validated(
        hass,
        """
alias: Dupes
trigger:
  - platform: state
    entity_id: light.ghost
action:
  - action: light.turn_on
    target:
      entity_id: light.ghost
  - action: notify.nope
  - action: notify.nope
""",
    )

    missing = find_missing_automation_targets(hass, config)

    assert [m.name for m in missing] == ["light.ghost", "notify.nope"]


def test_describe_lists_every_finding() -> None:
    text = describe_missing_targets(
        [
            MissingAutomationTarget("entity", "light.a", "does not exist."),
            MissingAutomationTarget("service", "notify.b", "is not a service."),
        ]
    )

    assert text.startswith("Automation not added")
    assert "- Entity 'light.a' does not exist." in text
    assert "- Service 'notify.b' is not a service." in text


@pytest.mark.asyncio
async def test_tool_refuses_and_writes_nothing(
    hass: HomeAssistant, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """End to end through add_automation: refused, not written, not reloaded."""
    (tmp_path / "automations.yaml").write_text("", encoding="utf-8")
    monkeypatch.setattr(hass.config, "config_dir", str(tmp_path), raising=False)
    reloads: list[str] = []

    async def _reload(*_args: Any, **_kwargs: Any) -> None:
        reloads.append("reload")

    hass.services.async_register("automation", "reload", _reload)
    config = {
        "configurable": {
            "hass": hass,
            "options": {CONF_NOTIFY_SERVICE: PHONE},
            "pending_actions": {},
            "user_id": "user1",
            "ha_llm_api": None,
        }
    }

    result = await add_automation_tool(automation_yaml=LEAK_YAML, config=config)

    assert result.startswith("Automation not added")
    assert "binary_sensor.sink_moisture_sensor" in result
    assert REAL_MOISTURE in result
    assert PHONE in result
    assert (tmp_path / "automations.yaml").read_text(encoding="utf-8") == ""
    assert reloads == []

    fixed = LEAK_YAML.replace("binary_sensor.sink_moisture_sensor", REAL_MOISTURE)
    fixed = fixed.replace("notify.mobile_app\n", f"{PHONE}\n")
    result = await add_automation_tool(automation_yaml=fixed, config=config)

    assert result.startswith("Added automation ")
    await hass.async_block_till_done()
    assert reloads == ["reload"]
    written = yaml.safe_load((tmp_path / "automations.yaml").read_text())
    assert written[0]["alias"] == "Leak Alert"


def test_tools_module_wires_the_check() -> None:
    """The tool module imports the check by name so the PIN tests can stub it."""
    assert (
        tools_module.find_missing_automation_targets is find_missing_automation_targets
    )

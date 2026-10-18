# ruff: noqa: S101
"""
The switch under a switch_as_x garage cover or lock is held too (#794).

A relay wrapped as a cover (the usual DIY garage wiring) opened the door with
no PIN through `switch.turn_on` on the relay, though opening the cover asked
for one. The same held for a relay wrapped as a lock.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest
from homeassistant.helpers import entity_registry as er
from homeassistant.setup import async_setup_component
from pytest_homeassistant_custom_component.common import MockConfigEntry

from custom_components.home_generative_agent.agent.automation_pin import (
    find_critical_automation_calls,
)
from custom_components.home_generative_agent.agent.graph import _is_critical_action
from custom_components.home_generative_agent.agent.helpers import (
    CRITICAL_RULE_EXACT_ENTITY,
)
from custom_components.home_generative_agent.agent.indirect_pin import (
    runs_critical_action,
)
from custom_components.home_generative_agent.agent.rule_entities import (
    expand_critical_rules,
)
from custom_components.home_generative_agent.const import (
    RECOMMENDED_CRITICAL_ACTIONS,
)

if TYPE_CHECKING:
    from homeassistant.core import HomeAssistant

RULES = RECOMMENDED_CRITICAL_ACTIONS


def _wrap(  # noqa: PLR0913
    hass: HomeAssistant,
    switch_id: str,
    wrapper_id: str,
    *,
    name: str,
    target: str = "cover",
    invert: bool = False,
    disabled: bool = False,
    options_entity: str | None = None,
) -> None:
    """Wrap a switch as a cover or lock, as the switch_as_x helper does."""
    hass.states.async_set(switch_id, "off", {"friendly_name": f"{name} relay"})
    entry = MockConfigEntry(
        domain="switch_as_x",
        options={
            "entity_id": switch_id if options_entity is None else options_entity,
            "invert": invert,
            "target_domain": target,
        },
    )
    entry.add_to_hass(hass)
    registry = er.async_get(hass)
    object_id = wrapper_id.split(".", 1)[1]
    registered = registry.async_get_or_create(
        target,
        "switch_as_x",
        f"{object_id}_uid",
        suggested_object_id=object_id,
        config_entry=entry,
        disabled_by=er.RegistryEntryDisabler.USER if disabled else None,
    )
    assert registered.entity_id == wrapper_id
    hass.states.async_set(wrapper_id, "closed", {"friendly_name": name})


def _call(entity_id: str, service: str, domain: str = "switch") -> dict[str, Any]:
    return {"domain": domain, "service": service, "entity_id": entity_id}


def _held(hass: HomeAssistant, args: dict[str, Any], rules: Any = None) -> bool:
    rules = expand_critical_rules(hass, RULES) if rules is None else rules
    return _is_critical_action(args, rules, "mcp__call_service", hass)


async def test_turning_on_a_garage_relay_is_held(hass: HomeAssistant) -> None:
    _wrap(hass, "switch.shelly_relay", "cover.garage", name="Garage Door")
    assert _held(hass, _call("switch.shelly_relay", "turn_on"))
    assert _held(hass, _call("switch.shelly_relay", "toggle"))
    # Before #794: only the cover asked for the PIN.
    assert not _held(hass, _call("switch.shelly_relay", "turn_on"), RULES)
    # Closing the door is turning the relay off: not held, as closing is not.
    assert not _held(hass, _call("switch.shelly_relay", "turn_off"))


async def test_an_inverted_relay_opens_on_turn_off(hass: HomeAssistant) -> None:
    _wrap(hass, "switch.relay_2", "cover.gate", name="Front Gate", invert=True)
    assert _held(hass, _call("switch.relay_2", "turn_off"))
    assert _held(hass, _call("switch.relay_2", "toggle"))
    assert not _held(hass, _call("switch.relay_2", "turn_on"))


async def test_a_relay_under_a_blind_is_not_held(hass: HomeAssistant) -> None:
    _wrap(hass, "switch.blind_relay", "cover.study_shade", name="Study Shade")
    assert not _held(hass, _call("switch.blind_relay", "turn_on"))


async def test_a_relay_under_a_lock_is_held(hass: HomeAssistant) -> None:
    """Every unlock is guarded; unlocking a switch_as_x lock turns its switch on."""
    _wrap(hass, "switch.strike", "lock.side", name="Side Door", target="lock")
    assert _held(hass, _call("switch.strike", "turn_on"))
    assert not _held(hass, _call("switch.strike", "turn_on"), RULES)
    assert not _held(hass, _call("switch.strike", "turn_off"))


async def test_a_lock_rule_alone_still_reaches_the_relay(hass: HomeAssistant) -> None:
    """A rule set with no entity_match rule at all still follows the wrapper."""
    _wrap(hass, "switch.strike", "lock.side", name="Side Door", target="lock")
    rules = [{"domain": "lock", "service": "unlock"}]
    expanded = expand_critical_rules(hass, rules)
    assert _held(hass, _call("switch.strike", "turn_on"), expanded)


async def test_the_generic_service_and_an_automation_reach_the_relay(
    hass: HomeAssistant,
) -> None:
    _wrap(hass, "switch.shelly_relay", "cover.garage", name="Garage Door")
    rules = expand_critical_rules(hass, RULES)
    generic = {"service": "turn_on", "entity_id": "switch.shelly_relay"}
    assert _is_critical_action(generic, rules, "mcp__call_service", hass)
    config = {
        "actions": [
            {
                "action": "homeassistant.turn_on",
                "target": {"entity_id": "switch.shelly_relay"},
            },
        ]
    }
    assert find_critical_automation_calls(config, rules)
    assert not find_critical_automation_calls(config, RULES)


async def test_turning_on_the_relay_by_name_is_held(hass: HomeAssistant) -> None:
    _wrap(hass, "switch.shelly_relay", "cover.garage", name="Garage Door")
    rules = expand_critical_rules(hass, RULES)
    args = {"name": "Garage Door relay", "domain": ["switch"]}
    assert _is_critical_action(args, rules, "intent__HassTurnOn", hass)


async def test_a_scene_turning_on_the_relay_is_held(hass: HomeAssistant) -> None:
    _wrap(hass, "switch.shelly_relay", "cover.garage", name="Garage Door")
    assert await async_setup_component(
        hass,
        "scene",
        {"scene": [{"name": "Arrive", "entities": {"switch.shelly_relay": "on"}}]},
    )
    await hass.async_block_till_done()
    rules = expand_critical_rules(hass, RULES)
    assert runs_critical_action(hass, "scene.arrive", rules)


async def test_a_switch_group_holding_the_relay_is_held(hass: HomeAssistant) -> None:
    _wrap(hass, "switch.shelly_relay", "cover.garage", name="Garage Door")
    hass.states.async_set(
        "switch.everything", "off", {"entity_id": ["switch.shelly_relay"]}
    )
    assert _held(hass, _call("switch.everything", "turn_on"))


async def test_a_wrapper_naming_the_relay_by_registry_id_is_followed(
    hass: HomeAssistant,
) -> None:
    """switch_as_x may store the switch's registry id, not its entity id."""
    switch = er.async_get(hass).async_get_or_create(
        "switch", "shelly", "relay-1", suggested_object_id="shelly_relay"
    )
    _wrap(
        hass,
        switch.entity_id,
        "cover.garage",
        name="Garage Door",
        options_entity=switch.id,
    )
    assert _held(hass, _call("switch.shelly_relay", "turn_on"))


async def test_a_disabled_wrapper_is_not_followed(hass: HomeAssistant) -> None:
    _wrap(
        hass, "switch.shelly_relay", "cover.garage", name="Garage Door", disabled=True
    )
    assert not _held(hass, _call("switch.shelly_relay", "turn_on"))


@pytest.mark.parametrize("options_entity", ["", "switch.gone"])
async def test_a_wrapper_with_no_readable_switch_adds_nothing(
    hass: HomeAssistant, options_entity: str
) -> None:
    """
    An entry naming no switch holds nothing; one naming a missing switch holds it.

    A switch not set up yet is held by its exact id, so it is guarded once it
    appears; nothing else is.
    """
    _wrap(
        hass,
        "switch.shelly_relay",
        "cover.garage",
        name="Garage Door",
        options_entity=options_entity,
    )
    expanded = expand_critical_rules(hass, RULES)
    switch_targets = {
        rule[CRITICAL_RULE_EXACT_ENTITY]
        for rule in expanded
        if rule["domain"] == "switch"
    }
    assert switch_targets == ({options_entity} if options_entity else set())
    assert not _held(hass, _call("switch.shelly_relay", "turn_on"))

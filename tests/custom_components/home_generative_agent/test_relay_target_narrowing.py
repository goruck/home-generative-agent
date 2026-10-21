# ruff: noqa: S101
"""
Area, device and label switch calls skip a hidden garage relay (#804).

switch_as_x hides the switch it wraps, and Home Assistant never expands an
area, device, floor or label target to a hidden entity. So with a relay
garage in the home, a switch call aimed at an area need not ask for the PIN,
unless the relay can really be reached: it was un-hidden, or another wrapper
of it (a light, a fan) is reached instead. Device actions name their entity
(by registry id) and reach it hidden or not.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest
from homeassistant.helpers import entity_registry as er
from pytest_homeassistant_custom_component.common import MockConfigEntry

from custom_components.home_generative_agent.agent.automation_pin import (
    find_critical_automation_calls,
)
from custom_components.home_generative_agent.agent.graph import _is_critical_action
from custom_components.home_generative_agent.agent.helpers import (
    CRITICAL_RULE_HIDDEN,
    CRITICAL_RULE_REGISTRY_ID,
    resolve_critical_action_policy,
)
from custom_components.home_generative_agent.agent.rule_entities import (
    expand_critical_rules,
)
from custom_components.home_generative_agent.const import (
    CONF_CRITICAL_ACTIONS,
    RECOMMENDED_CRITICAL_ACTIONS,
)

if TYPE_CHECKING:
    from homeassistant.core import HomeAssistant

RULES = RECOMMENDED_CRITICAL_ACTIONS


def _relay_garage(hass: HomeAssistant, *, hidden: bool = True) -> er.RegistryEntry:
    """Register a relay and wrap it as a garage cover, hiding it as switch_as_x does."""
    registry = er.async_get(hass)
    relay = registry.async_get_or_create(
        "switch", "shelly", "relay_uid", suggested_object_id="shelly_relay"
    )
    if hidden:
        relay = registry.async_update_entity(
            relay.entity_id, hidden_by=er.RegistryEntryHider.INTEGRATION
        )
    hass.states.async_set(relay.entity_id, "off")
    entry = MockConfigEntry(
        domain="switch_as_x",
        options={
            "entity_id": relay.entity_id,
            "invert": False,
            "target_domain": "cover",
        },
    )
    entry.add_to_hass(hass)
    registry.async_get_or_create(
        "cover",
        "switch_as_x",
        "garage_uid",
        suggested_object_id="garage",
        config_entry=entry,
    )
    hass.states.async_set("cover.garage", "closed", {"friendly_name": "Garage Door"})
    return relay


def _lamp(hass: HomeAssistant) -> er.RegistryEntry:
    lamp = er.async_get(hass).async_get_or_create(
        "switch", "shelly", "lamp_uid", suggested_object_id="lamp"
    )
    hass.states.async_set(lamp.entity_id, "off")
    return lamp


def _tool_held(hass: HomeAssistant, args: dict[str, Any]) -> bool:
    return _is_critical_action(
        args, expand_critical_rules(hass, RULES), "mcp__call_service", hass
    )


def _automation_held(hass: HomeAssistant, step: dict[str, Any]) -> bool:
    return bool(
        find_critical_automation_calls(
            {"actions": [step]}, expand_critical_rules(hass, RULES)
        )
    )


TARGETS = [
    {"area_id": "garage"},
    {"device_id": "relay_device"},
    {"label_id": "outside"},
    {"floor_id": "ground"},
]


@pytest.mark.parametrize("target", TARGETS)
async def test_an_area_device_or_label_switch_call_skips_a_hidden_relay(
    hass: HomeAssistant, target: dict[str, str]
) -> None:
    """The issue's case: an ordinary switch call no longer asks for the PIN."""
    _relay_garage(hass)

    assert not _tool_held(hass, {"domain": "switch", "service": "turn_on", **target})
    assert not _automation_held(hass, {"action": "switch.turn_on", "target": target})


@pytest.mark.parametrize("target", TARGETS)
async def test_an_unhidden_relay_is_still_held(
    hass: HomeAssistant, target: dict[str, str]
) -> None:
    """Once the user un-hides the relay, an area can reach it again."""
    _relay_garage(hass, hidden=False)

    assert _tool_held(hass, {"domain": "switch", "service": "turn_on", **target})
    assert _automation_held(hass, {"action": "switch.turn_on", "target": target})


async def test_un_hiding_the_relay_takes_effect_at_once(hass: HomeAssistant) -> None:
    """The registry is read on every check, not cached."""
    relay = _relay_garage(hass)
    call = {"domain": "switch", "service": "turn_on", "area_id": "garage"}
    assert not _tool_held(hass, call)

    er.async_get(hass).async_update_entity(relay.entity_id, hidden_by=None)

    assert _tool_held(hass, call)


async def test_another_wrapper_of_the_relay_is_reached_by_area(
    hass: HomeAssistant,
) -> None:
    """A light over the same relay is not hidden, so a light call is held."""
    relay = _relay_garage(hass)
    entry = MockConfigEntry(
        domain="switch_as_x",
        options={
            "entity_id": relay.entity_id,
            "invert": False,
            "target_domain": "light",
        },
    )
    entry.add_to_hass(hass)
    er.async_get(hass).async_get_or_create(
        "light",
        "switch_as_x",
        "porch_uid",
        suggested_object_id="porch",
        config_entry=entry,
    )
    hass.states.async_set("light.porch", "off")

    assert _tool_held(hass, {"domain": "light", "service": "turn_on", "area_id": "x"})
    # The hidden relay itself is still skipped by an area switch call.
    assert not _tool_held(
        hass, {"domain": "switch", "service": "turn_on", "area_id": "x"}
    )


@pytest.mark.parametrize(
    "entity_id", ["all", "{{ states.switch | map(attribute='entity_id') | list }}"]
)
async def test_all_and_templates_still_hold(
    hass: HomeAssistant, entity_id: str
) -> None:
    """Either could name the relay."""
    _relay_garage(hass)

    assert _automation_held(
        hass, {"action": "switch.turn_on", "target": {"entity_id": entity_id}}
    )


async def test_a_registry_id_is_held_only_when_it_is_the_relay(
    hass: HomeAssistant,
) -> None:
    """A registry id names one entity; the relay's copy knows its own."""
    relay = _relay_garage(hass)
    lamp = _lamp(hass)

    assert _automation_held(
        hass, {"action": "switch.turn_on", "target": {"entity_id": relay.id}}
    )
    assert not _automation_held(
        hass, {"action": "switch.turn_on", "target": {"entity_id": lamp.id}}
    )


def _device_action(entity_id: str | None) -> dict[str, Any]:
    step: dict[str, Any] = {"device_id": "dev", "domain": "switch", "type": "turn_on"}
    if entity_id is not None:
        step["entity_id"] = entity_id
    return step


async def test_device_actions_are_checked_by_the_entity_they_name(
    hass: HomeAssistant,
) -> None:
    """Hidden or not, a device action on the relay reaches it (#804)."""
    relay = _relay_garage(hass)
    lamp = _lamp(hass)

    assert _automation_held(hass, _device_action(relay.id))
    assert _automation_held(hass, _device_action(relay.entity_id))
    assert not _automation_held(hass, _device_action(lamp.id))
    assert not _automation_held(hass, _device_action(lamp.entity_id))
    # An action naming no entity cannot be checked.
    assert _automation_held(hass, _device_action(None))


async def test_a_relay_copy_records_what_the_registry_says(
    hass: HomeAssistant,
) -> None:
    relay = _relay_garage(hass)

    [copy] = [
        rule
        for rule in expand_critical_rules(hass, RULES)
        if rule.get("_exact_entity") == relay.entity_id
    ]

    assert copy[CRITICAL_RULE_HIDDEN] == "1"
    assert copy[CRITICAL_RULE_REGISTRY_ID] == relay.id


def test_stored_rules_cannot_carry_the_internal_keys() -> None:
    """A user rule marked hidden would narrow itself; the marks are dropped."""
    policy = resolve_critical_action_policy(
        {
            CONF_CRITICAL_ACTIONS: [
                {
                    "domain": "switch",
                    "service": "",
                    "_exact_entity": "switch.x",
                    CRITICAL_RULE_HIDDEN: "1",
                    CRITICAL_RULE_REGISTRY_ID: "0" * 32,
                }
            ]
        }
    )

    assert policy.critical_actions == [{"domain": "switch", "service": ""}]

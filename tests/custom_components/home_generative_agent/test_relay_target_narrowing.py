# ruff: noqa: S101
"""
How a garage relay's PIN hold reads a call's targets (#804).

Device actions act on the entity they name, so one is held only when it
names the relay; a registry id is matched against the relay's own. An area,
device, floor or label target is still held while a relay garage exists,
even with the relay hidden: what else in the area drives the relay (groups,
template switches, automations it triggers, MQTT) cannot all be known. Group
helpers that are offline are read from their settings, so a named group
holding the relay, or a template switch that drives it, is held.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest
from homeassistant.helpers import area_registry as ar
from homeassistant.helpers import entity_registry as er
from homeassistant.setup import async_setup_component
from homeassistant.util import dt as dt_util
from pytest_homeassistant_custom_component.common import MockConfigEntry

from custom_components.home_generative_agent.agent.automation_pin import (
    find_critical_automation_calls,
)
from custom_components.home_generative_agent.agent.graph import _is_critical_action
from custom_components.home_generative_agent.agent.helpers import (
    CRITICAL_RULE_REGISTRY_ID,
    resolve_critical_action_policy,
)
from custom_components.home_generative_agent.agent.indirect_pin import (
    stored_action_calls,
)
from custom_components.home_generative_agent.agent.rule_entities import (
    expand_critical_rules,
)
from custom_components.home_generative_agent.const import (
    CONF_CRITICAL_ACTIONS,
    RECOMMENDED_CRITICAL_ACTIONS,
)
from custom_components.home_generative_agent.snapshot.network import (
    AutomationInput,
    HaNativeInputs,
    NetworkBuildContext,
    _collect_critical_actions,
    _collect_stored_critical,
    ha_native_adapter,
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


@pytest.mark.parametrize("hidden", [True, False])
@pytest.mark.parametrize("target", TARGETS)
async def test_an_area_device_or_label_switch_call_is_still_held(
    hass: HomeAssistant, target: dict[str, str], *, hidden: bool
) -> None:
    """Hidden or not: something else in the area may drive the relay."""
    _relay_garage(hass, hidden=hidden)

    assert _tool_held(hass, {"domain": "switch", "service": "turn_on", **target})
    assert _automation_held(hass, {"action": "switch.turn_on", "target": target})


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

    assert copy[CRITICAL_RULE_REGISTRY_ID] == relay.id


def test_stored_rules_cannot_carry_the_internal_keys() -> None:
    """A user rule carrying a registry id would narrow itself; it is dropped."""
    policy = resolve_critical_action_policy(
        {
            CONF_CRITICAL_ACTIONS: [
                {
                    "domain": "switch",
                    "service": "",
                    "_exact_entity": "switch.x",
                    CRITICAL_RULE_REGISTRY_ID: "0" * 32,
                }
            ]
        }
    )

    assert policy.critical_actions == [{"domain": "switch", "service": ""}]


@pytest.mark.parametrize("listed", ["registry_id", "unknown_registry_id"])
async def test_an_unavailable_group_listing_the_relay_by_registry_id_is_held(
    hass: HomeAssistant, listed: str
) -> None:
    """
    An unavailable group helper shows no members, so its config is read.

    It may list the relay by registry id; unresolved, the group was not
    linked to the relay, so turning the group on was not held. An id the
    registry does not know could be the relay too (#804 review).
    """
    relay = _relay_garage(hass)
    entry = MockConfigEntry(
        domain="group",
        options={
            "group_type": "switch",
            "entities": [relay.id if listed == "registry_id" else "f" * 32],
        },
    )
    entry.add_to_hass(hass)
    er.async_get(hass).async_get_or_create(
        "switch",
        "group",
        "relay_group_uid",
        suggested_object_id="relay_group",
        config_entry=entry,
    )
    hass.states.async_set("switch.relay_group", "unavailable")
    group = {"entity_id": "switch.relay_group"}

    assert _automation_held(hass, {"action": "switch.turn_on", "target": group})
    assert _tool_held(hass, {"domain": "switch", "service": "turn_on", **group})


async def test_an_offline_group_in_another_wrappers_domain_is_held(
    hass: HomeAssistant,
) -> None:
    """
    An offline light group could hold a light wrapper of the relay (#804 review).

    Its members cannot be read, so it is held like an offline switch group.
    """
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
    registry = er.async_get(hass)
    alias = registry.async_get_or_create(
        "light",
        "switch_as_x",
        "alias_uid",
        suggested_object_id="relay_alias",
        config_entry=entry,
    )
    registry.async_update_entity(alias.entity_id, hidden_by=er.RegistryEntryHider.USER)
    hass.states.async_set(alias.entity_id, "off")
    registry.async_get_or_create(
        "light", "group", "yaml_group_uid", suggested_object_id="garage_lights"
    )
    hass.states.async_set("light.garage_lights", "unavailable")

    assert _automation_held(
        hass,
        {"action": "light.turn_on", "target": {"entity_id": "light.garage_lights"}},
    )


async def test_the_webhook_audit_follows_a_template_switch_to_the_relay(
    hass: HomeAssistant,
) -> None:
    """
    A public webhook turning on a template switch that opens the garage.

    The relay's own copy is skipped as hidden; the audit follows the template
    switch's actions, as the add_automation gate does (#804 review).
    """
    relay = _relay_garage(hass)
    assert await async_setup_component(
        hass,
        "template",
        {
            "template": [
                {
                    "switch": [
                        {
                            "name": "Opener",
                            "unique_id": "opener",
                            "turn_on": [
                                {
                                    "action": "switch.turn_on",
                                    "target": {"entity_id": relay.entity_id},
                                }
                            ],
                            "turn_off": [],
                        }
                    ]
                }
            ]
        },
    )
    await hass.async_block_till_done()
    area = ar.async_get(hass).async_create("Garage")
    er.async_get(hass).async_update_entity("switch.opener", area_id=area.id)
    automation = {
        "triggers": [{"trigger": "webhook", "webhook_id": "x", "local_only": False}],
        "actions": [
            {"action": "switch.turn_on", "target": {"entity_id": "switch.opener"}}
        ],
    }
    automations = [AutomationInput("automation.open", automation)]
    context = NetworkBuildContext(options={})
    critical_actions = _collect_critical_actions(hass, context, automations)
    inputs = HaNativeInputs(
        now=dt_util.utcnow(),
        automations=automations,
        critical_actions=critical_actions,
        stored_critical_automations=_collect_stored_critical(
            hass, automations, critical_actions
        ),
    )

    ha = ha_native_adapter(inputs, [], context).ha_security

    assert ha["webhook_automations_critical"] == ["automation.open"]


def _step_call(action: str, entity_id: str) -> dict[str, Any]:
    return {"action": action, "target": {"entity_id": entity_id}}


async def test_an_offline_group_holding_a_template_opener_is_held(
    hass: HomeAssistant,
) -> None:
    """
    A group helper whose only member is unavailable lists no members.

    Its member, a template switch that turns on the hidden relay, is read
    from the helper's config instead, so an area call reaching the group is
    held; when it runs later, the group forwards to the opener (#804 review).
    """
    assert await async_setup_component(
        hass, "input_boolean", {"input_boolean": {"ready": {}}}
    )
    assert await async_setup_component(
        hass,
        "template",
        {
            "template": [
                {
                    "switch": [
                        {
                            "name": "Shelly Relay",
                            "unique_id": "relay_uid",
                            "optimistic": True,
                            "turn_on": [],
                            "turn_off": [],
                        },
                        {
                            "name": "Opener",
                            "unique_id": "opener_uid",
                            "availability": "{{ is_state('input_boolean.ready', 'on') }}",
                            "turn_on": [
                                _step_call("switch.turn_on", "switch.shelly_relay")
                            ],
                            "turn_off": [],
                        },
                    ]
                }
            ]
        },
    )
    await hass.async_block_till_done()
    registry = er.async_get(hass)
    registry.async_update_entity(
        "switch.shelly_relay", hidden_by=er.RegistryEntryHider.INTEGRATION
    )
    wrapper_entry = MockConfigEntry(
        domain="switch_as_x",
        options={
            "entity_id": "switch.shelly_relay",
            "invert": False,
            "target_domain": "cover",
        },
    )
    wrapper_entry.add_to_hass(hass)
    registry.async_get_or_create(
        "cover",
        "switch_as_x",
        "garage_uid",
        suggested_object_id="garage",
        config_entry=wrapper_entry,
    )
    hass.states.async_set("cover.garage", "closed", {"friendly_name": "Garage Door"})
    group_entry = MockConfigEntry(
        domain="group",
        title="Garage switches",
        options={
            "group_type": "switch",
            "name": "Garage switches",
            "entities": ["switch.opener"],
            "hide_members": False,
            "all": False,
        },
    )
    group_entry.add_to_hass(hass)
    assert await hass.config_entries.async_setup(group_entry.entry_id)
    await hass.async_block_till_done()
    group_id = registry.async_get_entity_id("switch", "group", group_entry.entry_id)
    assert group_id is not None
    area = ar.async_get(hass).async_get_or_create("garage")
    registry.async_update_entity(group_id, area_id=area.id)
    group_state = hass.states.get(group_id)
    assert group_state is not None
    assert group_state.state == "unavailable"
    rules = expand_critical_rules(hass, RULES)
    step = {"action": "switch.turn_on", "target": {"entity_id": group_id}}

    assert _is_critical_action(
        {"domain": "switch", "service": "turn_on", "entity_id": group_id},
        rules,
        "mcp__call_service",
        hass,
    )
    assert stored_action_calls(hass, [step], rules)

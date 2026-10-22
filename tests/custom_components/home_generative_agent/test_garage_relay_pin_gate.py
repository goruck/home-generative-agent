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


async def test_every_call_on_a_garage_relay_is_held(hass: HomeAssistant) -> None:
    """
    A momentary relay moves the door whichever way it is switched.

    The wrapper's "on = open" (or inverted) says nothing of the wiring.
    """
    _wrap(hass, "switch.shelly_relay", "cover.garage", name="Garage Door")
    for service in ("turn_on", "turn_off", "toggle"):
        assert _held(hass, _call("switch.shelly_relay", service))
    # Before #794: only the cover asked for the PIN.
    assert not _held(hass, _call("switch.shelly_relay", "turn_on"), RULES)


async def test_an_inverted_relay_is_held_both_ways(hass: HomeAssistant) -> None:
    _wrap(hass, "switch.relay_2", "cover.gate", name="Front Gate", invert=True)
    for service in ("turn_on", "turn_off", "toggle"):
        assert _held(hass, _call("switch.relay_2", service))


async def test_the_relay_copies_are_exact(hass: HomeAssistant) -> None:
    """One copy per relay, whatever the number of rules reaching its cover."""
    _wrap(hass, "switch.shelly_relay", "cover.garage", name="Garage Door")
    expanded = expand_critical_rules(hass, RULES)
    relay = [
        (rule[CRITICAL_RULE_EXACT_ENTITY], rule["service"])
        for rule in expanded
        if rule["domain"] == "switch"
    ]
    assert relay == [("switch.shelly_relay", "")]


async def test_a_relay_under_a_blind_is_not_held(hass: HomeAssistant) -> None:
    _wrap(hass, "switch.blind_relay", "cover.study_shade", name="Study Shade")
    assert not _held(hass, _call("switch.blind_relay", "turn_on"))


async def test_a_relay_under_a_lock_is_held_the_unlocking_way(
    hass: HomeAssistant,
) -> None:
    """Every unlock is guarded; unlocking a switch_as_x lock turns its switch on."""
    _wrap(hass, "switch.strike", "lock.side", name="Side Door", target="lock")
    assert _held(hass, _call("switch.strike", "turn_on"))
    assert _held(hass, _call("switch.strike", "toggle"))
    assert not _held(hass, _call("switch.strike", "turn_on"), RULES)
    # Locking is the other way, and locking is not held.
    assert not _held(hass, _call("switch.strike", "turn_off"))


async def test_an_inverted_lock_relay_unlocks_on_turn_off(hass: HomeAssistant) -> None:
    _wrap(
        hass, "switch.strike", "lock.side", name="Side Door", target="lock", invert=True
    )
    assert _held(hass, _call("switch.strike", "turn_off"))
    assert not _held(hass, _call("switch.strike", "turn_on"))


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


async def test_a_disabled_wrapper_still_holds_its_relay(hass: HomeAssistant) -> None:
    """Disabling the cover does not disconnect the relay from the door."""
    _wrap(
        hass, "switch.shelly_relay", "cover.garage", name="Garage Door", disabled=True
    )
    hass.states.async_remove("cover.garage")
    assert _held(hass, _call("switch.shelly_relay", "turn_on"))


async def test_a_second_wrapper_of_the_relay_is_held(hass: HomeAssistant) -> None:
    """The relay is also wrapped as a light: turning the light on opens the door."""
    _wrap(hass, "switch.shelly_relay", "cover.garage", name="Garage Door")
    entry = MockConfigEntry(
        domain="switch_as_x",
        options={
            "entity_id": "switch.shelly_relay",
            "invert": False,
            "target_domain": "light",
        },
    )
    entry.add_to_hass(hass)
    er.async_get(hass).async_get_or_create(
        "light",
        "switch_as_x",
        "aux_uid",
        suggested_object_id="garage_aux",
        config_entry=entry,
    )
    hass.states.async_set("light.garage_aux", "off", {"friendly_name": "Aux"})
    assert _held(hass, _call("light.garage_aux", "turn_on", domain="light"))
    generic = {"service": "turn_on", "entity_id": "light.garage_aux"}
    assert _is_critical_action(
        generic, expand_critical_rules(hass, RULES), "mcp__call_service", hass
    )


async def test_a_valve_rule_reaches_the_relay_under_a_valve(
    hass: HomeAssistant,
) -> None:
    _wrap(hass, "switch.main", "valve.gate", name="Gate", target="valve")
    rules = [{"domain": "valve", "service": "open_valve", "entity_match": "gate"}]
    expanded = expand_critical_rules(hass, rules)
    assert _held(hass, _call("switch.main", "turn_on"), expanded)


async def test_a_rule_with_no_domain_reaches_the_relay(hass: HomeAssistant) -> None:
    """A rule with no domain applies to the cover, so to its relay too."""
    _wrap(hass, "switch.shelly_relay", "cover.garage", name="Garage Door")
    rules = [{"service": "open_cover", "entity_match": "garage"}]
    expanded = expand_critical_rules(hass, rules)
    assert _held(hass, _call("switch.shelly_relay", "turn_on"), expanded)


async def test_a_closing_rule_holds_a_cover_relay_too(hass: HomeAssistant) -> None:
    """A momentary relay closes the door the same way it opens it."""
    _wrap(hass, "switch.shelly_relay", "cover.garage", name="Garage Door")
    rules = [{"domain": "cover", "service": "close_cover", "entity_match": "garage"}]
    expanded = expand_critical_rules(hass, rules)
    assert _held(hass, _call("switch.shelly_relay", "turn_off"), expanded)


async def test_a_locking_rule_adds_no_lock_relay_copy(hass: HomeAssistant) -> None:
    _wrap(hass, "switch.strike", "lock.side", name="Side Door", target="lock")
    expanded = expand_critical_rules(hass, [{"domain": "lock", "service": "lock"}])
    assert not any(rule["domain"] == "switch" for rule in expanded)


async def test_a_wrapped_switch_group_holds_its_members(hass: HomeAssistant) -> None:
    """A cover over a switch group: each relay in the group opens the door."""
    hass.states.async_set("switch.relay_1", "off")
    _wrap(hass, "switch.garage_relays", "cover.garage", name="Garage Door")
    hass.states.async_set(
        "switch.garage_relays", "off", {"entity_id": ["switch.relay_1"]}
    )
    assert _held(hass, _call("switch.relay_1", "turn_on"))


@pytest.mark.parametrize("holder", ["group.garage_stuff", "switch.outer"])
async def test_nested_groups_holding_the_relay_are_held(
    hass: HomeAssistant, holder: str
) -> None:
    _wrap(hass, "switch.shelly_relay", "cover.garage", name="Garage Door")
    hass.states.async_set("switch.inner", "off", {"entity_id": ["switch.shelly_relay"]})
    hass.states.async_set(holder, "off", {"entity_id": ["switch.inner"]})
    rules = expand_critical_rules(hass, RULES)
    generic = {"service": "turn_on", "entity_id": holder}
    assert _is_critical_action(generic, rules, "mcp__call_service", hass)


async def test_one_unreadable_wrapper_does_not_drop_the_others(
    hass: HomeAssistant,
    caplog: pytest.LogCaptureFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _wrap(hass, "switch.shelly_relay", "cover.garage", name="Garage Door")
    _wrap(hass, "switch.strike", "lock.side", name="Side Door", target="lock")
    real = er.async_resolve_entity_id

    def _flaky(registry: Any, entity_id: str) -> str | None:
        if entity_id == "switch.strike":
            msg = "bad entry"
            raise RuntimeError(msg)
        return real(registry, entity_id)

    monkeypatch.setattr(er, "async_resolve_entity_id", _flaky)
    assert _held(hass, _call("switch.shelly_relay", "turn_on"))
    assert "could not read switch_as_x entry" in caplog.text
    # The entry that could not be read adds no hold (logged, not failed
    # closed): nothing raises for a well-formed entry.
    assert not _held(hass, _call("switch.strike", "turn_on"))


async def test_switches_in_automations_are_screened_per_entity(
    hass: HomeAssistant,
) -> None:
    """A relay's copy holds that relay, not every switch in an automation."""
    _wrap(hass, "switch.shelly_relay", "cover.garage", name="Garage Door")
    hass.states.async_set("switch.lamp", "off")
    rules = expand_critical_rules(hass, RULES)

    def _apply(entity_id: str) -> dict[str, Any]:
        return {
            "actions": [
                {"action": "scene.apply", "data": {"entities": {entity_id: "on"}}}
            ]
        }

    assert find_critical_automation_calls(_apply("switch.shelly_relay"), rules)
    assert not find_critical_automation_calls(_apply("switch.lamp"), rules)


async def test_a_scene_turning_a_lock_relay_off_is_not_held(
    hass: HomeAssistant,
) -> None:
    """Off locks a lock relay: a bedtime scene doing so asks for no PIN."""
    _wrap(hass, "switch.strike", "lock.side", name="Side Door", target="lock")
    assert await async_setup_component(
        hass,
        "scene",
        {
            "scene": [
                {"name": "Bedtime", "entities": {"switch.strike": "off"}},
                {"name": "Morning", "entities": {"switch.strike": "on"}},
            ]
        },
    )
    await hass.async_block_till_done()
    rules = expand_critical_rules(hass, RULES)
    assert not runs_critical_action(hass, "scene.bedtime", rules)
    assert runs_critical_action(hass, "scene.morning", rules)


async def test_the_pin_line_says_what_the_relay_opens(hass: HomeAssistant) -> None:
    from custom_components.home_generative_agent.agent.pin_messages import (  # noqa: PLC0415
        describe_pending_action,
    )

    _wrap(hass, "switch.shelly_relay", "cover.garage", name="Garage Door")
    line = describe_pending_action(
        hass,
        {
            "tool_name": "mcp__call_service",
            "tool_args": _call("switch.shelly_relay", "turn_off"),
        },
    )
    assert line.endswith('(controls "Garage Door")')
    plain = describe_pending_action(
        hass,
        {
            "tool_name": "mcp__call_service",
            "tool_args": _call("switch.lamp", "turn_on"),
        },
    )
    assert "(" not in plain


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


def _second_wrapper(
    hass: HomeAssistant, switch_id: str, wrapper_id: str, target: str = "light"
) -> None:
    """Wrap ``switch_id`` again, as a light by default."""
    entry = MockConfigEntry(
        domain="switch_as_x",
        options={"entity_id": switch_id, "invert": False, "target_domain": target},
    )
    entry.add_to_hass(hass)
    object_id = wrapper_id.split(".", 1)[1]
    er.async_get(hass).async_get_or_create(
        target,
        "switch_as_x",
        f"{object_id}_uid",
        suggested_object_id=object_id,
        config_entry=entry,
    )
    hass.states.async_set(wrapper_id, "off", {"friendly_name": object_id})


async def test_a_wrapper_over_a_group_holding_the_relay_is_held(
    hass: HomeAssistant,
) -> None:
    """A light over a switch group holding the relay drives the relay too."""
    _wrap(hass, "switch.shelly_relay", "cover.garage", name="Garage Door")
    hass.states.async_set("switch.bank", "off", {"entity_id": ["switch.shelly_relay"]})
    _second_wrapper(hass, "switch.bank", "light.bank")
    assert _held(hass, _call("light.bank", "turn_on", domain="light"))


async def test_a_wrapper_of_a_member_of_the_wrapped_group_is_held(
    hass: HomeAssistant,
) -> None:
    """The cover wraps a switch group; a light wraps one relay in it."""
    hass.states.async_set("switch.relay_1", "off")
    _wrap(hass, "switch.garage_relays", "cover.garage", name="Garage Door")
    hass.states.async_set(
        "switch.garage_relays", "off", {"entity_id": ["switch.relay_1"]}
    )
    _second_wrapper(hass, "switch.relay_1", "light.relay_1")
    assert _held(hass, _call("light.relay_1", "turn_on", domain="light"))


async def test_a_second_wrapper_of_a_lock_relay_is_held_both_ways(
    hass: HomeAssistant,
) -> None:
    """The lock relay keeps its direction; a light over it is held both ways."""
    _wrap(hass, "switch.strike", "lock.side", name="Side Door", target="lock")
    _second_wrapper(hass, "switch.strike", "light.strike_aux")
    for service in ("turn_on", "turn_off"):
        assert _held(hass, _call("light.strike_aux", service, domain="light"))
    assert not _held(hass, _call("switch.strike", "turn_off"))


async def test_a_switch_group_not_set_up_yet_is_held(hass: HomeAssistant) -> None:
    """Its members cannot be seen: it could hold the relay, as #773 holds covers."""
    _wrap(hass, "switch.shelly_relay", "cover.garage", name="Garage Door")
    er.async_get(hass).async_get_or_create(
        "switch", "mqtt", "grp", suggested_object_id="mqtt_group"
    )
    assert _held(hass, _call("switch.mqtt_group", "turn_on"))


@pytest.mark.parametrize(
    ("entity_id", "held"), [("switch.shelly_relay", True), ("switch.lamp", False)]
)
async def test_switch_device_actions_hold_only_the_relay(
    hass: HomeAssistant, entity_id: str, *, held: bool
) -> None:
    """
    A device action acts on the entity it names, not on its device (#804).

    So the relay's own device action is held; another switch's is not.
    """
    _wrap(hass, "switch.shelly_relay", "cover.garage", name="Garage Door")
    hass.states.async_set("switch.lamp", "off")
    device = {
        "actions": [
            {
                "device_id": "abc",
                "domain": "switch",
                "type": "turn_on",
                "entity_id": entity_id,
            }
        ]
    }
    assert (
        bool(find_critical_automation_calls(device, expand_critical_rules(hass, RULES)))
        is held
    )
    assert not find_critical_automation_calls(device, RULES)


async def test_the_pin_line_names_the_lock_and_other_paths(
    hass: HomeAssistant,
) -> None:
    from custom_components.home_generative_agent.agent.pin_messages import (  # noqa: PLC0415
        describe_pending_action,
    )

    _wrap(hass, "switch.strike", "lock.side", name="Side Door", target="lock")
    _wrap(hass, "switch.shelly_relay", "cover.garage", name="Garage Door")
    hass.states.async_set(
        "switch.everything", "off", {"entity_id": ["switch.shelly_relay"]}
    )

    def _line(args: dict[str, Any]) -> str:
        return describe_pending_action(
            hass, {"tool_name": "mcp__call_service", "tool_args": args}
        )

    assert _line(_call("switch.strike", "turn_on")).endswith('(controls "Side Door")')
    in_data = {
        "domain": "switch",
        "service": "turn_on",
        "data": {"entity_id": "switch.shelly_relay"},
    }
    assert _line(in_data).endswith('(controls "Garage Door")')
    assert _line(_call("switch.everything", "turn_on")).endswith(
        '(controls "Garage Door")'
    )
    automation = describe_pending_action(
        hass, {"tool_name": "add_automation", "tool_args": {}}
    )
    assert "controls" not in automation


async def test_closing_a_cover_group_holding_the_garage_is_not_held(
    hass: HomeAssistant,
) -> None:
    """Closing "All Covers" is not a call on the relay; closing is not held."""
    _wrap(hass, "switch.shelly_relay", "cover.garage", name="Garage Door")
    hass.states.async_set("cover.all_covers", "open", {"entity_id": ["cover.garage"]})
    close = {**_call("cover.all_covers", "close_cover", domain="cover")}
    assert not _held(hass, close)
    assert _held(hass, {**close, "service": "open_cover"})


async def test_locking_a_lock_group_holding_a_relay_lock_is_not_held(
    hass: HomeAssistant,
) -> None:
    _wrap(hass, "switch.strike", "lock.side", name="Side Door", target="lock")
    hass.states.async_set("lock.all_locks", "unlocked", {"entity_id": ["lock.side"]})
    assert not _held(hass, _call("lock.all_locks", "lock", domain="lock"))


async def test_a_lock_rule_with_no_service_holds_every_relay_call(
    hass: HomeAssistant,
) -> None:
    _wrap(hass, "switch.strike", "lock.side", name="Side Door", target="lock")
    expanded = expand_critical_rules(hass, [{"domain": "lock", "entity_match": "side"}])
    for service in ("turn_on", "turn_off", "toggle"):
        assert _held(hass, _call("switch.strike", service), expanded)


async def test_a_rule_for_another_kind_of_call_adds_no_cover_relay_copy(
    hass: HomeAssistant,
) -> None:
    """A domain-less rule on `unlock` does not reach a garage cover's relay."""
    _wrap(hass, "switch.shelly_relay", "cover.garage", name="Garage Door")
    expanded = expand_critical_rules(hass, [{"service": "unlock"}])
    assert not any(rule.get("domain") == "switch" for rule in expanded)


async def test_a_scene_turning_a_cover_relay_off_is_held(hass: HomeAssistant) -> None:
    """A momentary relay moves the door either way, so off is held too."""
    _wrap(hass, "switch.shelly_relay", "cover.garage", name="Garage Door")
    assert await async_setup_component(
        hass,
        "scene",
        {"scene": [{"name": "Leave", "entities": {"switch.shelly_relay": "off"}}]},
    )
    await hass.async_block_till_done()
    assert runs_critical_action(hass, "scene.leave", expand_critical_rules(hass, RULES))


async def test_the_pin_line_names_the_door_only_through_its_relay(
    hass: HomeAssistant,
) -> None:
    from custom_components.home_generative_agent.agent.pin_messages import (  # noqa: PLC0415
        describe_pending_action,
    )

    _wrap(hass, "switch.shelly_relay", "cover.garage", name="Garage Door")
    _second_wrapper(hass, "switch.shelly_relay", "light.garage_aux")
    # A switch not set up yet is held (it could hold the relay), but the line
    # does not say it moves the door.
    er.async_get(hass).async_get_or_create(
        "switch", "mqtt", "grp", suggested_object_id="mqtt_group"
    )

    def _line(tool_name: str, args: dict[str, Any]) -> str:
        return describe_pending_action(
            hass, {"tool_name": tool_name, "tool_args": args}
        )

    service = "mcp__call_service"
    assert "controls" not in _line(
        service, _call("cover.garage", "open_cover", domain="cover")
    )
    assert "controls" not in _line(service, _call("switch.mqtt_group", "turn_on"))
    assert _line(
        service, _call("light.garage_aux", "turn_on", domain="light")
    ).endswith('(controls "Garage Door")')
    assert _line(
        "intent__HassTurnOn", {"name": "Garage Door relay", "domain": ["switch"]}
    ).endswith('(controls "Garage Door")')
    in_template = {
        "domain": "switch",
        "service": "turn_on",
        "data_template": {"entity_id": "switch.shelly_relay"},
    }
    assert _line(service, in_template).endswith('(controls "Garage Door")')


async def test_the_pin_line_survives_a_relay_lookup_failure(
    hass: HomeAssistant, monkeypatch: pytest.MonkeyPatch
) -> None:
    from custom_components.home_generative_agent.agent import (  # noqa: PLC0415
        pin_messages,
    )

    def _boom(*_a: Any, **_k: Any) -> list[str]:
        msg = "registry gone"
        raise RuntimeError(msg)

    monkeypatch.setattr(pin_messages, "relay_wrappers_reached", _boom)
    line = pin_messages.describe_pending_action(
        hass,
        {
            "tool_name": "mcp__call_service",
            "tool_args": _call("switch.shelly_relay", "turn_on"),
        },
    )
    assert line.startswith("turn on")

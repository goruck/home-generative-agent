# ruff: noqa: S101
"""
The audit flags an exposed relay that moves a sensitive cover or lock (#898).

The switch under a switch_as_x garage cover, a switch group holding it, and
the switch a template garage cover's actions drive all open the door, and
``switch`` is exposed to Assist by default. The Critical Action PIN has held
each since #794 and #780; the audit followed none of them, so "turn on
garage relay" opened the door through a voice assistant with no finding.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, cast

import pytest
from homeassistant.helpers import entity_registry as er
from homeassistant.setup import async_setup_component
from pytest_homeassistant_custom_component.common import MockConfigEntry

from custom_components.home_generative_agent.agent import rule_entities
from custom_components.home_generative_agent.agent.rule_entities import (
    AuditReach,
    audit_reach,
)
from custom_components.home_generative_agent.sentinel.rules.ha_sensitive_entity_exposed_without_pin import (
    HaSensitiveEntityExposedWithoutPinRule,
)
from custom_components.home_generative_agent.snapshot.builder import (
    async_build_full_state_snapshot,
)
from custom_components.home_generative_agent.snapshot.network import (
    HaNativeInputs,
    NetworkBuildContext,
    ha_native_adapter,
)

from .test_audit_exposed_groups_793 import _entity, _helper
from .test_sentinel_exposed_exclusions_899 import _only, _rule, _snapshot

if TYPE_CHECKING:
    from homeassistant.core import HomeAssistant

NOW = datetime(2026, 11, 2, 12, 0, tzinfo=UTC)
ENTRY = frozenset({"door", "garage", "gate"})
ALWAYS = frozenset({"lock", "alarm_control_panel"})
RELAY = "switch.garage_relay"
GARAGE = "cover.garage"


def _reach(hass: HomeAssistant, entity_ids: list[str]) -> AuditReach:
    return audit_reach(hass, entity_ids, cover_classes=ENTRY, sensitive_domains=ALWAYS)


def _wrap(
    hass: HomeAssistant,
    switch_id: str,
    wrapper_id: str,
    *,
    attributes: dict[str, Any],
    disabled: bool = False,
) -> None:
    """Wrap a switch as a cover or lock, as the switch_as_x helper does."""
    hass.states.async_set(switch_id, "off", {"friendly_name": "Relay"})
    target = wrapper_id.split(".", 1)[0]
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
        disabled_by=er.RegistryEntryDisabler.USER if disabled else None,
    )
    hass.states.async_set(wrapper_id, "closed", attributes)


@pytest.fixture
async def home(hass: HomeAssistant) -> HomeAssistant:
    """Set up a garage cover over a relay, and a blind over another."""
    assert await async_setup_component(hass, "homeassistant", {})
    _wrap(hass, RELAY, GARAGE, attributes={"device_class": "garage"})
    _wrap(
        hass,
        "switch.blind_relay",
        "cover.study_blind",
        attributes={"device_class": "shade", "friendly_name": "Study Blind"},
    )
    return hass


async def test_the_relay_under_a_garage_cover_is_reported(home: HomeAssistant) -> None:
    reach = _reach(home, [GARAGE])
    assert reach.relays == {RELAY: [GARAGE]}


async def test_a_relay_under_a_window_covering_is_not(home: HomeAssistant) -> None:
    assert "switch.blind_relay" not in _reach(home, [GARAGE]).relays


async def test_groups_holding_the_relay_move_the_door_too(home: HomeAssistant) -> None:
    """A switch group helper and an old-style group.* both turn the relay on."""
    switches = _helper(home, "all_switches", [RELAY], domain="switch")
    assert await async_setup_component(
        home, "group", {"group": {"outside": {"entities": [RELAY]}}}
    )
    await home.async_block_till_done()
    relays = _reach(home, [GARAGE]).relays
    assert relays[switches] == [GARAGE]
    assert relays["group.outside"] == [GARAGE]


async def test_the_lock_a_relay_unlocks_is_reported(hass: HomeAssistant) -> None:
    assert await async_setup_component(hass, "homeassistant", {})
    _wrap(hass, "switch.strike", "lock.side_gate", attributes={})
    assert _reach(hass, ["lock.side_gate"]).relays == {
        "switch.strike": ["lock.side_gate"]
    }


async def test_a_disabled_wrapper_still_has_a_relay(hass: HomeAssistant) -> None:
    """Disabling the cover does not disconnect the relay, which still opens it."""
    assert await async_setup_component(hass, "homeassistant", {})
    _wrap(hass, RELAY, GARAGE, attributes={"device_class": "garage"}, disabled=True)
    er.async_get(hass).async_update_entity(GARAGE, original_device_class="garage")
    assert _reach(hass, [GARAGE]).relays == {RELAY: [GARAGE]}
    # Found by its registry class alone, too, though no state shows it.
    assert _reach(hass, []).relays == {RELAY: [GARAGE]}


async def test_a_disabled_relay_is_not_reported(hass: HomeAssistant) -> None:
    """No assistant can reach a disabled switch, so it opens nothing by voice."""
    assert await async_setup_component(hass, "homeassistant", {})
    er.async_get(hass).async_get_or_create(
        "switch",
        "test",
        "garage_relay_uid",
        suggested_object_id="garage_relay",
        disabled_by=er.RegistryEntryDisabler.USER,
    )
    _wrap(hass, RELAY, GARAGE, attributes={"device_class": "garage"})
    assert _reach(hass, [GARAGE]).relays == {}


async def test_another_wrapper_of_the_relay_moves_the_door(home: HomeAssistant) -> None:
    """A light wrapping the same switch turns the relay on, and its groups do."""
    _wrap(home, RELAY, "light.garage_light", attributes={})
    lights = _helper(home, "all_lights", ["light.garage_light"], domain="light")
    relays = _reach(home, [GARAGE]).relays
    assert relays["light.garage_light"] == [GARAGE]
    assert relays[lights] == [GARAGE]


async def test_a_reference_or_a_group_voice_cannot_drive_is_no_relay(
    home: HomeAssistant,
) -> None:
    """
    Only real groups count, unlike the PIN's fail-closed walk.

    A light naming the relay in an attribute is a reference; an old-style
    group.* holding only a cover wrapper is turned on with
    homeassistant.turn_on, which skips covers; a switch not set up yet could
    be anything for the PIN, and is nothing for the audit.
    """
    home.states.async_set("light.status", "on", {"entity_id": [RELAY]})
    _wrap(home, RELAY, "cover.utility_blind", attributes={"device_class": "shade"})
    assert await async_setup_component(
        home, "group", {"group": {"covers": {"entities": ["cover.utility_blind"]}}}
    )
    await home.async_block_till_done()
    er.async_get(home).async_get_or_create(
        "switch", "test", "unknown_uid", suggested_object_id="not_loaded"
    )
    relays = _reach(home, [GARAGE]).relays
    assert relays == {RELAY: [GARAGE], "cover.utility_blind": [GARAGE]}


async def test_a_group_holding_a_door_and_driving_another_is_in_both(
    home: HomeAssistant,
) -> None:
    """Listed as holding the gate and as a relay of the garage."""
    home.states.async_set("cover.front_gate", "closed", {"device_class": "gate"})
    _wrap(home, RELAY, "cover.utility_blind", attributes={"device_class": "shade"})
    group = _helper(home, "outside", ["cover.front_gate", "cover.utility_blind"])
    reach = _reach(home, [GARAGE, "cover.front_gate"])
    assert reach.groups[group] == ["cover.front_gate"]
    assert reach.relays[group] == [GARAGE]


async def test_an_unreadable_template_is_logged_once_and_noted(
    home: HomeAssistant,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """
    Through the real reader: once, as the audit's, and in the report.

    Not as a PIN hold every few minutes (the PIN still logs its own).
    """
    entity = SimpleNamespace(entity_id="cover.template_garage")
    monkeypatch.setattr(
        rule_entities,
        "template_entities",
        lambda _hass, domain: [entity] if domain == "cover" else [],
    )
    rule_entities._LOGGED_UNREADABLE.clear()
    with caplog.at_level(logging.DEBUG):
        first = _reach(home, [GARAGE])
        _reach(home, [GARAGE])
    assert first.unread_relays == ["the actions of cover.template_garage"]
    messages = [
        r.getMessage() for r in caplog.records if "template_garage" in r.getMessage()
    ]
    assert messages == [
        (
            "Security audit: could not read the actions of "
            "cover.template_garage; the relays under it are not reported"
        )
    ]
    # The PIN, reading the same entity, still logs its own hold.
    with caplog.at_level(logging.WARNING):
        rule_entities.read_template_drives(home)
    assert any(
        r.getMessage().startswith("Critical Action PIN: cannot read the actions")
        for r in caplog.records
    )


async def test_a_nested_template_with_unreadable_actions_is_noted(
    home: HomeAssistant, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A template the garage cover's action turns on, whose own actions are unreadable."""
    from custom_components.home_generative_agent.agent import (  # noqa: PLC0415
        template_actions,
    )

    home.states.async_set("switch.template_opener", "off")
    monkeypatch.setattr(
        template_actions,
        "_loaded_entity",
        lambda _hass, entity_id: (
            SimpleNamespace(entity_id=entity_id)
            if entity_id == "switch.template_opener"
            else None
        ),
    )
    monkeypatch.setattr(
        template_actions,
        "is_template_entity",
        lambda _hass, entity_id: entity_id == "switch.template_opener",
    )
    problems: list[str] = []
    step = {
        "action": "switch.turn_on",
        "target": {"entity_id": "switch.template_opener"},
    }
    driven = template_actions.driven_entities(home, [[step]], problems)
    assert "switch.template_opener" in driven
    assert problems == ["the actions of switch.template_opener"]


def test_the_report_notes_what_could_not_be_read_for_relays() -> None:
    inputs = HaNativeInputs(
        now=NOW,
        exposed={"conversation": []},
        sensitive_reach=AuditReach(
            unread_relays=["the actions of cover.template_garage"]
        ),
    )
    notes = ha_native_adapter(inputs, [], NetworkBuildContext()).notes
    assert (
        "the actions of cover.template_garage could not be read; the relays "
        "under it are not audited."
    ) in notes


async def test_a_disabled_group_is_no_way_to_the_relay(home: HomeAssistant) -> None:
    """A group above a disabled switch group cannot reach the relay through it."""
    inner = _helper(home, "inner", [RELAY], domain="switch")
    er.async_get(home).async_update_entity(
        inner, disabled_by=er.RegistryEntryDisabler.USER
    )
    outer = _helper(home, "outer", [inner], domain="switch")
    relays = _reach(home, [GARAGE]).relays
    assert inner not in relays
    assert outer not in relays


async def test_a_disabled_switch_group_still_has_live_relays(
    hass: HomeAssistant,
) -> None:
    """The cover wraps a switch group the user disabled; its member still opens it."""
    assert await async_setup_component(hass, "homeassistant", {})
    hass.states.async_set("switch.physical_relay", "off")
    entry = MockConfigEntry(
        domain="group", options={"entities": ["switch.physical_relay"]}
    )
    entry.add_to_hass(hass)
    er.async_get(hass).async_get_or_create(
        "switch",
        "group",
        "openers",
        suggested_object_id="openers",
        config_entry=entry,
        disabled_by=er.RegistryEntryDisabler.USER,
    )
    _wrap(hass, "switch.openers", GARAGE, attributes={"device_class": "garage"})
    relays = _reach(hass, [GARAGE]).relays
    assert relays == {"switch.physical_relay": [GARAGE]}


async def test_a_template_cover_opening_another_wrapper_reaches_its_switch(
    home: HomeAssistant, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Template garage -> a blind-classed switch_as_x cover -> the switch under it."""
    home.states.async_set("cover.template_garage", "closed", {"device_class": "garage"})
    monkeypatch.setattr(
        rule_entities,
        "read_template_drives",
        lambda _hass, _problems=None: {"cover.template_garage": {"cover.study_blind"}},
    )
    relays = _reach(home, [GARAGE, "cover.template_garage"]).relays
    assert relays["cover.study_blind"] == ["cover.template_garage"]
    assert relays["switch.blind_relay"] == ["cover.template_garage"]


async def test_a_group_cannot_press_a_button_relay(
    home: HomeAssistant, monkeypatch: pytest.MonkeyPatch
) -> None:
    """homeassistant.turn_on skips buttons, so an old-style group is no way in."""
    home.states.async_set("cover.template_garage", "closed", {"device_class": "garage"})
    home.states.async_set("button.opener", "unknown")
    monkeypatch.setattr(
        rule_entities,
        "read_template_drives",
        lambda _hass, _problems=None: {"cover.template_garage": {"button.opener"}},
    )
    assert await async_setup_component(
        home, "group", {"group": {"buttons": {"entities": ["button.opener"]}}}
    )
    await home.async_block_till_done()
    relays = _reach(home, [GARAGE, "cover.template_garage"]).relays
    assert relays["button.opener"] == ["cover.template_garage"]
    assert "group.buttons" not in relays


async def test_a_sensitive_entity_a_template_drives_is_not_also_a_relay(
    home: HomeAssistant, monkeypatch: pytest.MonkeyPatch
) -> None:
    home.states.async_set("cover.template_garage", "closed", {"device_class": "garage"})
    home.states.async_set("lock.side_door", "locked")
    monkeypatch.setattr(
        rule_entities,
        "read_template_drives",
        lambda _hass, _problems=None: {"cover.template_garage": {"lock.side_door"}},
    )
    reach = _reach(home, [GARAGE, "cover.template_garage", "lock.side_door"])
    assert "lock.side_door" not in reach.relays


async def test_what_a_template_cover_drives_is_reported(
    home: HomeAssistant, monkeypatch: pytest.MonkeyPatch
) -> None:
    home.states.async_set("cover.template_garage", "closed", {"device_class": "garage"})
    home.states.async_set("switch.opener", "off")
    monkeypatch.setattr(
        rule_entities,
        "read_template_drives",
        lambda _hass, _problems=None: {"cover.template_garage": {"switch.opener"}},
    )
    relays = _reach(home, [GARAGE, "cover.template_garage"]).relays
    assert relays["switch.opener"] == ["cover.template_garage"]
    assert relays[RELAY] == [GARAGE]


async def test_a_template_side_effect_is_no_relay(
    home: HomeAssistant, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The light the garage cover's action also turns on does not move it."""
    home.states.async_set("cover.template_garage", "closed", {"device_class": "garage"})
    home.states.async_set("light.porch", "off")
    monkeypatch.setattr(
        rule_entities,
        "read_template_drives",
        lambda _hass, _problems=None: {
            "cover.template_garage": {"light.porch", "switch.opener"}
        },
    )
    home.states.async_set("switch.opener", "off")
    relays = _reach(home, [GARAGE, "cover.template_garage"]).relays
    assert "light.porch" not in relays
    assert relays["switch.opener"] == ["cover.template_garage"]


async def test_the_snapshot_reports_an_exposed_relay(home: HomeAssistant) -> None:
    """
    End to end: the relay is asked about, so its default exposure is seen.

    Switches are exposed to Assist by default.
    """
    snapshot = await async_build_full_state_snapshot(
        home, network=NetworkBuildContext(options={})
    )
    ha = cast("dict[str, Any]", cast("Any", snapshot)["network"]["ha_security"])
    assert RELAY in ha["exposed_sensitive_entities"]["conversation"]
    assert ha["exposed_sensitive_relays"] == {RELAY: [GARAGE]}


def test_the_adapter_reports_an_exposed_relay() -> None:
    inputs = HaNativeInputs(
        now=NOW,
        exposed={"conversation": [RELAY, "switch.kitchen"]},
        sensitive_reach=AuditReach(
            relays={RELAY: [GARAGE], "switch.not_exposed": [GARAGE]}
        ),
    )
    ha = ha_native_adapter(
        inputs, [_entity(RELAY), _entity("switch.kitchen")], NetworkBuildContext()
    ).ha_security
    assert ha["exposed_sensitive_entities"] == {"conversation": [RELAY]}
    assert ha["exposed_sensitive_relays"] == {RELAY: [GARAGE]}


def _relay_snapshot(**kwargs: Any) -> Any:
    snapshot = _snapshot({"cloud.alexa": [RELAY]}, **kwargs)
    cast("Any", snapshot)["network"]["ha_security"]["exposed_sensitive_relays"] = {
        RELAY: [GARAGE]
    }
    return snapshot


def test_the_rule_names_what_the_relay_moves() -> None:
    finding = _only(
        HaSensitiveEntityExposedWithoutPinRule().evaluate(_relay_snapshot())
    )
    assert finding.triggering_entities == [RELAY]
    assert finding.evidence["summary"] == (
        "1 sensitive entity exposed: switch.garage_relay to Alexa. "
        "switch.garage_relay moves cover.garage."
    )
    assert finding.evidence["relays"] == {RELAY: [GARAGE]}


@pytest.mark.parametrize("excluded", [RELAY, GARAGE])
def test_excluding_the_relay_or_what_it_moves_drops_it(excluded: str) -> None:
    """As for a group (#899): an exclusion on the door reaches its relay."""
    assert _rule({"*": [excluded]}).evaluate(_relay_snapshot()) == []


def test_a_relay_moving_two_keeps_the_one_not_excluded() -> None:
    snapshot = _snapshot({"cloud.alexa": [RELAY]})
    cast("Any", snapshot)["network"]["ha_security"]["exposed_sensitive_relays"] = {
        RELAY: [GARAGE, "lock.side_gate"]
    }
    finding = _only(_rule({"*": [GARAGE]}).evaluate(snapshot))
    assert finding.evidence["relays"] == {RELAY: ["lock.side_gate"]}
    assert _rule({"*": [GARAGE, "lock.side_gate"]}).evaluate(snapshot) == []


def test_a_group_in_both_lists_stays_while_either_is_left() -> None:
    """Excluding the gate it holds leaves the garage it drives, and back."""
    group = "cover.outside"
    snapshot = _snapshot({"cloud.alexa": [group]}, groups={group: ["cover.front_gate"]})
    cast("Any", snapshot)["network"]["ha_security"]["exposed_sensitive_relays"] = {
        group: [GARAGE]
    }
    finding = _only(_rule({"*": ["cover.front_gate"]}).evaluate(snapshot))
    assert finding.evidence["relays"] == {group: [GARAGE]}
    assert _only(_rule({"*": [GARAGE]}).evaluate(snapshot)).triggering_entities == [
        group
    ]
    assert _rule({"*": [GARAGE, "cover.front_gate"]}).evaluate(snapshot) == []


async def test_an_entity_in_both_lists_is_asked_about_once(home: HomeAssistant) -> None:
    """Exposed once, so the finding names it once and its id does not shift."""
    home.states.async_set("cover.front_gate", "closed", {"device_class": "gate"})
    _wrap(home, RELAY, "cover.utility_blind", attributes={"device_class": "shade"})
    group = _helper(home, "outside", ["cover.front_gate", "cover.utility_blind"])
    snapshot = await async_build_full_state_snapshot(
        home, network=NetworkBuildContext(options={})
    )
    ha = cast("dict[str, Any]", cast("Any", snapshot)["network"]["ha_security"])
    assert ha["exposed_sensitive_entities"]["conversation"].count(group) == 1
    assert ha["exposed_sensitive_groups"][group] == ["cover.front_gate"]
    assert ha["exposed_sensitive_relays"][group] == [GARAGE]


async def test_what_drives_another_wrapper_of_the_relay_is_followed(
    home: HomeAssistant, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A template shade opening a blind wrapped on the garage relay opens the garage."""
    _wrap(home, RELAY, "cover.utility_blind", attributes={"device_class": "shade"})
    home.states.async_set("cover.office_shade", "closed", {"device_class": "shade"})
    monkeypatch.setattr(
        rule_entities,
        "read_template_drives",
        lambda _hass, _problems=None: {"cover.office_shade": {"cover.utility_blind"}},
    )
    relays = _reach(home, [GARAGE]).relays
    assert relays["cover.utility_blind"] == [GARAGE]
    assert relays["cover.office_shade"] == [GARAGE]


async def test_a_wrapper_of_a_disabled_group_is_no_way_in(hass: HomeAssistant) -> None:
    """The live member is reported; a light wrapping the disabled group is not."""
    assert await async_setup_component(hass, "homeassistant", {})
    hass.states.async_set("switch.physical_relay", "off")
    entry = MockConfigEntry(
        domain="group", options={"entities": ["switch.physical_relay"]}
    )
    entry.add_to_hass(hass)
    er.async_get(hass).async_get_or_create(
        "switch",
        "group",
        "openers",
        suggested_object_id="openers",
        config_entry=entry,
        disabled_by=er.RegistryEntryDisabler.USER,
    )
    _wrap(hass, "switch.openers", GARAGE, attributes={"device_class": "garage"})
    _wrap(hass, "switch.openers", "light.openers_light", attributes={})
    assert _reach(hass, [GARAGE]).relays == {"switch.physical_relay": [GARAGE]}

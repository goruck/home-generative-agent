# ruff: noqa: S101
"""
The audit flags an exposed group holding a sensitive cover or lock (#793).

``is_sensitive_entity`` judges one snapshot entity by its domain, the device
class its state shows and its name. A cover group holding the garage door
has none of those marks, and a garage opener shown as a blind hides its own,
so exposing either to a voice assistant was never reported, while the
Critical Action PIN has held both since #773. The audit now reads the same
entity index through ``audit_reach``: it follows Home Assistant's own group
helpers in the member's domain, says what each exposed group holds, and
names the groups it could not read in full.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, cast

import pytest
from homeassistant.components.homeassistant.exposed_entities import (
    async_expose_entity,
    async_get_entity_settings,
)
from homeassistant.components.intent.llm import LLM_INTENTS
from homeassistant.helpers import device_registry as dr
from homeassistant.helpers import entity_registry as er
from homeassistant.helpers import intent as ha_intent
from homeassistant.setup import async_setup_component
from pytest_homeassistant_custom_component.common import MockConfigEntry

from custom_components.home_generative_agent.agent import rule_entities
from custom_components.home_generative_agent.agent.rule_entities import (
    AuditReach,
    audit_reach,
)
from custom_components.home_generative_agent.snapshot import network as network_module
from custom_components.home_generative_agent.snapshot.builder import (
    async_build_full_state_snapshot,
)
from custom_components.home_generative_agent.snapshot.network import (
    HaNativeInputs,
    NetworkBuildContext,
    _collect_exposed,
    _collect_sensitive_reach,
    ha_cap,
    ha_native_adapter,
    merge_adapter_results,
)
from custom_components.home_generative_agent.snapshot.schema import SnapshotEntity

if TYPE_CHECKING:
    from homeassistant.core import HomeAssistant

NOW = datetime(2026, 10, 31, 12, 0, tzinfo=UTC)
GARAGE = "cover.msg100_abc"
ENTRY = frozenset({"door", "garage", "gate"})
ALWAYS = frozenset({"lock", "alarm_control_panel"})


def _reach(hass: HomeAssistant, entity_ids: list[str]) -> AuditReach:
    return audit_reach(hass, entity_ids, cover_classes=ENTRY, sensitive_domains=ALWAYS)


def _helper(
    hass: HomeAssistant,
    object_id: str,
    members: list[str],
    *,
    domain: str = "cover",
    config: bool = False,
    **attributes: Any,
) -> str:
    """
    Register a Home Assistant group helper and give it a state.

    With ``config`` the members go into its config entry and the state is
    unavailable (a helper that is not loaded); otherwise into the state's
    ``entity_id`` attribute.
    """
    entry = MockConfigEntry(domain="group", options={"entities": members})
    if config:
        entry.add_to_hass(hass)
    reg = er.async_get(hass).async_get_or_create(
        domain,
        "group",
        object_id,
        config_entry=entry if config else None,
        suggested_object_id=object_id,
    )
    hass.states.async_set(
        reg.entity_id,
        "unavailable" if config else "closed",
        {} if config else {"entity_id": members, **attributes},
    )
    return reg.entity_id


@pytest.fixture
async def home(hass: HomeAssistant) -> HomeAssistant:
    """
    Set up a garage opener with a plain id and the groups around it.

    A cover group of blinds and the opener, a cover group of blinds only, a
    cover group holding the first group, an old-style group holding the
    opener and a light listing the opener in an attribute.
    """
    assert await async_setup_component(hass, "homeassistant", {})
    hass.states.async_set(
        GARAGE, "closed", {"device_class": "garage", "friendly_name": "Q"}
    )
    hass.states.async_set(
        "cover.living_room_blinds", "closed", {"device_class": "shade"}
    )
    hass.states.async_set("cover.study_blinds", "closed")
    _helper(
        hass,
        "all_covers",
        ["cover.living_room_blinds", GARAGE],
        friendly_name="All Covers",
    )
    _helper(hass, "all_blinds", ["cover.living_room_blinds", "cover.study_blinds"])
    _helper(hass, "everything", ["cover.all_covers", "cover.all_blinds"])
    hass.states.async_set("group.openers", "closed", {"entity_id": [GARAGE]})
    hass.states.async_set("light.weird", "on", {"entity_id": [GARAGE]})
    hass.states.async_set("lock.front", "locked")
    return hass


def _entity(entity_id: str, **attributes: Any) -> SnapshotEntity:
    return SnapshotEntity(
        entity_id=entity_id,
        domain=entity_id.split(".", 1)[0],
        state="closed",
        friendly_name=None,
        area=None,
        attributes=attributes,
        last_changed=NOW.isoformat(),
        last_updated=NOW.isoformat(),
    )


# ---------------------------------------------------------------------------
# The membership read
# ---------------------------------------------------------------------------


async def test_group_helpers_holding_the_opener_are_found_to_any_depth(
    home: HomeAssistant,
) -> None:
    """
    The cover groups holding the opener, nested; nothing else is a group.

    An old-style group.* is driven by a voice assistant through
    homeassistant.turn_on, which skips covers, and an ``entity_id``
    attribute on a light is a reference, not a member list.
    """
    reach = _reach(home, [GARAGE, "lock.front"])
    assert reach.groups == {
        "cover.all_covers": [GARAGE],
        "cover.everything": [GARAGE],
    }
    assert reach.sensitive == []
    assert reach.unreadable_groups == []
    assert reach.unreadable == []


async def test_a_group_that_is_itself_sensitive_holds_nothing_extra(
    hass: HomeAssistant,
) -> None:
    """A group named like a garage is sensitive on its own, not as a holder."""
    hass.states.async_set("cover.left", "closed", {"device_class": "garage"})
    hass.states.async_set("cover.right", "closed", {"device_class": "garage"})
    _helper(
        hass,
        "garage_doors",
        ["cover.left", "cover.right"],
        friendly_name="Garage Doors",
    )
    reach = _reach(hass, ["cover.garage_doors", "cover.left", "cover.right"])
    assert reach.groups == {}
    assert reach.sensitive == []


async def test_a_lock_group_is_a_lock_and_a_cover_helper_never_holds_one(
    hass: HomeAssistant,
) -> None:
    """A lock group is sensitive on its own; a helper counts only in its domain."""
    hass.states.async_set("lock.front", "locked")
    _helper(hass, "all_locks", ["lock.front"], domain="lock")
    _helper(hass, "odd_mix", ["lock.front"])
    reach = _reach(hass, ["lock.front"])
    assert reach.sensitive == ["lock.all_locks"]
    assert reach.groups == {}


async def test_an_opener_shown_as_a_blind_or_aliased_garage_door_is_sensitive(
    hass: HomeAssistant,
) -> None:
    """
    The snapshot check misses both; the index knows the class and the names.

    ``cover.motor`` keeps its garage class in the registry under the user's
    "blind"; ``cover.roller`` is a shutter whose alias says what it is.
    """
    registry = er.async_get(hass)
    motor = registry.async_get_or_create(
        "cover",
        "test",
        "m",
        suggested_object_id="motor",
        original_device_class="garage",
    )
    hass.states.async_set(
        motor.entity_id, "closed", {"device_class": "blind", "friendly_name": "Q"}
    )
    roller = registry.async_get_or_create(
        "cover",
        "test",
        "r",
        suggested_object_id="roller",
        original_device_class="shutter",
    )
    registry.async_update_entity(roller.entity_id, aliases=["Garage Door"])
    hass.states.async_set(roller.entity_id, "closed", {"friendly_name": "Motor"})
    hass.states.async_set("cover.shade", "closed", {"friendly_name": "Indoor Shade"})
    _helper(hass, "all_covers", [motor.entity_id, roller.entity_id, "cover.shade"])
    reach = _reach(hass, [])
    assert reach.sensitive == ["cover.motor", "cover.roller"]
    assert reach.groups == {"cover.all_covers": ["cover.motor", "cover.roller"]}


async def test_a_lock_present_only_in_the_registry_is_sensitive(
    hass: HomeAssistant,
) -> None:
    """A lock not loaded yet has no snapshot entity; the index still knows it."""
    lock = er.async_get(hass).async_get_or_create(
        "lock", "test", "l", suggested_object_id="cellar"
    )
    hass.states.async_set("lock.front", "locked")
    _helper(hass, "all_locks", [lock.entity_id, "lock.front"], domain="lock")
    reach = _reach(hass, ["lock.front"])
    # The lock group is a lock, so it is sensitive itself, not a holder.
    assert reach.sensitive == ["lock.all_locks", "lock.cellar"]
    assert reach.groups == {}


async def test_a_disabled_entity_is_never_sensitive_nor_a_holder(
    hass: HomeAssistant,
) -> None:
    """No assistant can reach a disabled entity; the PIN holds it, the audit skips it."""
    registry = er.async_get(hass)
    registry.async_get_or_create(
        "cover",
        "test",
        "d",
        suggested_object_id="old_garage",
        original_device_class="garage",
        disabled_by=er.RegistryEntryDisabler.USER,
    )
    hass.states.async_set(GARAGE, "closed", {"device_class": "garage"})
    group = _helper(hass, "retired", [GARAGE], config=True)
    registry.async_update_entity(group, disabled_by=er.RegistryEntryDisabler.USER)
    reach = _reach(hass, [GARAGE])
    assert reach.sensitive == []
    assert reach.groups == {}
    assert reach.unreadable_groups == []


async def test_a_zigbee_or_mqtt_group_in_the_members_domain_is_followed(
    hass: HomeAssistant,
) -> None:
    """
    ``group_entities`` is written by Home Assistant from the integration's group.

    MQTT blocks it from a device's own JSON attributes, so it is as trusted
    as a helper's member list; the domain rule still applies.
    """
    hass.states.async_set(GARAGE, "closed", {"device_class": "garage"})
    hass.states.async_set("cover.zigbee_group", "closed", {"group_entities": [GARAGE]})
    hass.states.async_set("light.odd", "on", {"group_entities": [GARAGE]})
    assert _reach(hass, [GARAGE]).groups == {"cover.zigbee_group": [GARAGE]}


async def test_a_registry_only_cover_named_by_its_id_is_sensitive(
    hass: HomeAssistant,
) -> None:
    """``cover.side_door`` with no state has only its id to say what it is."""
    er.async_get(hass).async_get_or_create(
        "cover", "test", "s", suggested_object_id="side_door"
    )
    _helper(hass, "all_covers", ["cover.side_door"])
    reach = _reach(hass, [])
    assert reach.sensitive == ["cover.side_door"]
    assert reach.groups == {"cover.all_covers": ["cover.side_door"]}


async def test_a_helper_listing_a_member_home_assistant_does_not_know_is_named(
    hass: HomeAssistant,
) -> None:
    """The ghost member may be anything, so the group may hold more than shown."""
    hass.states.async_set(GARAGE, "closed", {"device_class": "garage"})
    _helper(hass, "all_covers", [GARAGE, "cover.ghost"])
    reach = _reach(hass, [GARAGE])
    assert reach.groups == {"cover.all_covers": [GARAGE]}
    assert reach.unreadable_groups == ["cover.all_covers"]


async def test_a_yaml_group_keeps_its_members_while_unavailable(
    hass: HomeAssistant,
) -> None:
    """
    A YAML cover group has no config entry and no member list in its state.

    While its members are unavailable (a hub outage) the loaded entity still
    keeps them, so the group stays in the finding and its id does not flap.
    """
    hass.states.async_set(GARAGE, "unavailable", {"device_class": "garage"})
    hass.states.async_set("cover.yaml_group", "unavailable")

    class _Component:
        def get_entity(self, entity_id: str) -> object | None:
            if entity_id != "cover.yaml_group":
                return None
            return SimpleNamespace(
                _entity_ids=[GARAGE], platform=SimpleNamespace(platform_name="group")
            )

    hass.data["cover"] = _Component()
    reach = _reach(hass, [GARAGE])
    assert reach.groups == {"cover.yaml_group": [GARAGE]}
    assert reach.unreadable_groups == []


async def test_a_disabled_entity_that_cannot_be_read_is_still_skipped(
    hass: HomeAssistant,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The disabled flag is read before anything that can raise."""
    registry = er.async_get(hass)
    owner = MockConfigEntry(domain="test")
    owner.add_to_hass(hass)
    device = dr.async_get(hass).async_get_or_create(
        config_entry_id=owner.entry_id, identifiers={("test", "d")}
    )
    registry.async_get_or_create(
        "lock",
        "test",
        "o",
        suggested_object_id="old",
        device_id=device.id,
        disabled_by=er.RegistryEntryDisabler.USER,
    )

    def boom(*_args: Any, **_kwargs: Any) -> None:
        msg = "device registry exploded"
        raise RuntimeError(msg)

    monkeypatch.setattr(dr.DeviceRegistry, "async_get", boom)
    rule_entities._LOGGED_UNREADABLE.clear()
    reach = _reach(hass, [])
    assert reach.sensitive == []
    assert reach.unreadable == []


async def test_two_group_helpers_holding_each_other_terminate(
    hass: HomeAssistant,
) -> None:
    """The cycle guard: both helpers hold the opener, once each."""
    hass.states.async_set("cover.garage_door", "closed")
    _helper(hass, "a", ["cover.garage_door", "cover.b"], config=True)
    _helper(hass, "b", ["cover.a"], config=True)
    reach = _reach(hass, ["cover.garage_door"])
    assert reach.groups == {
        "cover.a": ["cover.garage_door"],
        "cover.b": ["cover.garage_door"],
    }


async def test_an_old_style_group_is_no_voice_path_to_a_cover(
    hass: HomeAssistant,
) -> None:
    """
    Pin the Home Assistant facts the old-style group exclusion rests on.

    Every voice path drives a ``group.*`` through ``homeassistant.turn_on``,
    which skips a domain without that service, and ``cover`` has none.
    ``homeassistant.toggle`` would reach a cover, so HassToggle must stay out
    of the Assist LLM tools for the exclusion to hold (it ships no Assist
    sentences either, which this venv cannot check); the Newest HA job fails
    here when that changes.
    """
    assert await async_setup_component(hass, "cover", {})
    assert not hass.services.has_service("cover", "turn_on")
    assert hass.services.has_service("cover", "toggle")
    assert ha_intent.INTENT_TOGGLE not in LLM_INTENTS


async def test_an_unavailable_group_helper_is_read_from_its_config(
    hass: HomeAssistant,
) -> None:
    """An unavailable group writes no members; its config entry still lists them."""
    _helper(hass, "outside_set", ["cover.garage_door"], config=True)
    _helper(hass, "blind_set", ["cover.study_blinds"], config=True)
    hass.states.async_set("cover.garage_door", "closed")
    hass.states.async_set("cover.study_blinds", "closed")
    reach = _reach(hass, ["cover.garage_door"])
    assert reach.groups == {"cover.outside_set": ["cover.garage_door"]}
    assert reach.unreadable_groups == []


async def test_a_group_helper_with_no_readable_members_is_named_not_reported(
    hass: HomeAssistant,
) -> None:
    """The PIN holds a group it cannot read; the audit names it and reports what it saw."""
    reg = er.async_get(hass).async_get_or_create(
        "cover", "group", "x", suggested_object_id="mystery"
    )
    hass.states.async_set(reg.entity_id, "unavailable")
    hass.states.async_set("cover.garage_door", "closed")
    reach = _reach(hass, ["cover.garage_door"])
    assert reach.groups == {}
    assert reach.unreadable_groups == ["cover.mystery"]
    assert reach.unreadable == []


async def test_a_partly_readable_group_helper_is_reported_with_what_it_holds(
    hass: HomeAssistant,
) -> None:
    """One member listed by a registry id that no longer resolves: the rest count."""
    _helper(
        hass,
        "partial",
        ["cover.garage_door", "0123456789abcdef0123456789abcdef"],
        config=True,
    )
    hass.states.async_set("cover.garage_door", "closed")
    reach = _reach(hass, ["cover.garage_door"])
    assert reach.groups == {"cover.partial": ["cover.garage_door"]}
    assert reach.unreadable_groups == ["cover.partial"]


async def test_an_unreadable_entity_is_named_and_logged_once_as_the_audit(
    hass: HomeAssistant,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """No traceback every cycle, and not labelled as the PIN."""
    hass.states.async_set("cover.odd", "closed")
    hass.states.async_set("cover.garage_door", "closed")

    class _Broken:
        def get_entity(self, entity_id: str) -> None:
            if entity_id == "cover.odd":
                raise RuntimeError

    monkeypatch.setitem(hass.data, "cover", _Broken())
    rule_entities._LOGGED_UNREADABLE.clear()
    with caplog.at_level(logging.WARNING):
        first = _reach(hass, ["cover.garage_door"])
        _reach(hass, ["cover.garage_door"])
    assert first.unreadable == ["cover.odd"]
    assert first.unreadable_groups == []
    records = [r for r in caplog.records if "cover.odd" in r.getMessage()]
    assert len(records) == 1
    assert records[0].levelno == logging.WARNING
    assert (
        records[0].getMessage().startswith("Security audit: could not read cover.odd")
    )


async def test_an_unreadable_entity_is_logged_with_a_traceback_every_time_as_the_pin(
    hass: HomeAssistant,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """The PIN gate keeps its traceback on every guarded call."""
    hass.states.async_set("cover.odd", "closed")

    class _Broken:
        def get_entity(self, entity_id: str) -> None:
            if entity_id == "cover.odd":
                raise RuntimeError

    monkeypatch.setitem(hass.data, "cover", _Broken())
    with caplog.at_level(logging.ERROR):
        rule_entities._EntityIndex(hass, wrapped={}, drives={})
        rule_entities._EntityIndex(hass, wrapped={}, drives={})
    records = [r for r in caplog.records if "cover.odd" in r.getMessage()]
    assert len(records) == 2
    assert all(r.levelno == logging.ERROR and r.exc_info for r in records)
    assert (
        records[0]
        .getMessage()
        .startswith("Critical Action PIN: could not read cover.odd; it is held")
    )


# ---------------------------------------------------------------------------
# The collectors and the pure adapter
# ---------------------------------------------------------------------------


async def test_a_failed_index_read_returns_none_and_logs_the_consequence(
    hass: HomeAssistant,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """The entities themselves are still audited; the failure is logged once."""

    def boom(*_args: Any, **_kwargs: Any) -> AuditReach:
        msg = "registry exploded"
        raise RuntimeError(msg)

    monkeypatch.setattr(network_module, "audit_reach", boom)
    network_module._LOGGED_INPUT_FAILURES.clear()
    with caplog.at_level(logging.WARNING):
        assert _collect_sensitive_reach(hass, ["cover.garage_door"]) is None
    assert "could not read group membership and registry device classes" in caplog.text
    assert "the exposed entities are still audited" in caplog.text
    assert "missing capabilities" not in caplog.text


async def test_the_exposure_query_never_writes_a_default_it_was_not_asked_for(
    hass: HomeAssistant,
) -> None:
    """
    Home Assistant pins its answer into the entity's options when asked.

    An assistant whose "Expose new entities" setting is off (Alexa and
    Google until turned on) and an entity with no state are not asked, so
    nothing is written for them; a stored setting still counts, and Assist's
    default exposure of a loaded cover is still asked for.
    """
    assert await async_setup_component(hass, "homeassistant", {})
    registry = er.async_get(hass)
    lock = registry.async_get_or_create(
        "lock", "test", "f", suggested_object_id="front"
    )
    cover = registry.async_get_or_create(
        "cover", "test", "g", suggested_object_id="garage_door"
    )
    unloaded = registry.async_get_or_create(
        "cover", "test", "u", suggested_object_id="not_yet"
    )
    hass.states.async_set(lock.entity_id, "locked")
    hass.states.async_set(cover.entity_id, "closed")
    ids = [lock.entity_id, cover.entity_id, unloaded.entity_id]
    exposed = _collect_exposed(hass, ids)
    assert exposed is not None
    assert exposed["cloud.alexa"] == []
    assert exposed["conversation"] == [cover.entity_id]
    for entity_id in (lock.entity_id, unloaded.entity_id):
        settings = async_get_entity_settings(hass, entity_id)
        assert "cloud.alexa" not in settings
        assert "cloud.google_assistant" not in settings
    assert "conversation" not in async_get_entity_settings(hass, unloaded.entity_id)
    async_expose_entity(hass, "cloud.alexa", lock.entity_id, should_expose=True)
    async_expose_entity(hass, "cloud.alexa", unloaded.entity_id, should_expose=True)
    exposed = _collect_exposed(hass, ids)
    assert exposed is not None
    assert exposed["cloud.alexa"] == [unloaded.entity_id, lock.entity_id]


def test_the_adapter_keeps_what_the_index_found_sensitive() -> None:
    """A group counts by what it holds, a classed cover by its registry class."""
    entities = [
        _entity("cover.garage", device_class="garage"),
        _entity("cover.all_covers"),
        _entity("cover.all_blinds"),
        _entity("cover.motor", device_class="blind"),
    ]
    inputs = HaNativeInputs(
        now=NOW,
        exposed={
            "cloud.alexa": [
                "cover.all_covers",
                "cover.all_blinds",
                "cover.everything",
                "cover.motor",
            ],
            "conversation": ["cover.garage"],
        },
        sensitive_reach=AuditReach(
            sensitive=["cover.motor"],
            groups={
                "cover.all_covers": ["cover.garage"],
                # Exposed by hand with no state in the snapshot: still reported.
                "cover.everything": ["cover.garage"],
                # Not exposed anywhere: not reported.
                "cover.garage_only": ["cover.garage"],
            },
        ),
    )
    result = ha_native_adapter(inputs, entities, NetworkBuildContext())
    assert result.ha_security["exposed_sensitive_entities"] == {
        "cloud.alexa": ["cover.all_covers", "cover.everything", "cover.motor"],
        "conversation": ["cover.garage"],
    }
    assert result.ha_security["exposed_sensitive_groups"] == {
        "cover.all_covers": ["cover.garage"],
        "cover.everything": ["cover.garage"],
    }
    assert not any("could not be read" in note for note in result.notes)


def test_the_adapter_writes_no_group_field_without_an_exposed_group() -> None:
    """Findings with no group keep their evidence, and so their ids."""
    inputs = HaNativeInputs(
        now=NOW,
        exposed={"cloud.alexa": ["cover.garage"]},
        sensitive_reach=AuditReach(groups={"cover.garage_only": ["cover.garage"]}),
    )
    entities = [_entity("cover.garage", device_class="garage")]
    ha = ha_native_adapter(inputs, entities, NetworkBuildContext()).ha_security
    assert ha["exposed_sensitive_entities"] == {"cloud.alexa": ["cover.garage"]}
    assert "exposed_sensitive_groups" not in ha


@pytest.mark.parametrize(
    ("reach", "expected"),
    [
        (
            None,
            (
                "Group membership and registry device classes could not be read; "
                "an exposed group holding a lock or entry cover"
            ),
        ),
        (
            AuditReach(unreadable_groups=[f"cover.g{i}" for i in range(6)]),
            (
                "Membership of cover.g0, cover.g1, cover.g2, cover.g3, cover.g4 and "
                "1 more could not be read in full; an exposed one may hold more"
            ),
        ),
        (
            AuditReach(unreadable=["lock.odd"]),
            "lock.odd could not be read; class, name and members are unknown",
        ),
    ],
)
def test_the_adapter_says_what_could_not_be_read(
    reach: AuditReach | None, expected: str
) -> None:
    """Each gap reaches the audit report's notes in its own words."""
    inputs = HaNativeInputs(
        now=NOW, exposed={"cloud.alexa": ["cover.garage"]}, sensitive_reach=reach
    )
    entities = [_entity("cover.garage", device_class="garage")]
    result = ha_native_adapter(inputs, entities, NetworkBuildContext())
    assert result.ha_security["exposed_sensitive_entities"] == {
        "cloud.alexa": ["cover.garage"]
    }
    assert "exposed_sensitive_groups" not in result.ha_security
    notes = merge_adapter_results([result]).get("notes", [])
    assert any(expected in note for note in notes), notes


# ---------------------------------------------------------------------------
# End to end: the snapshot Sentinel reads
# ---------------------------------------------------------------------------


async def test_the_snapshot_reports_exposed_group_helpers_holding_the_garage_door(
    home: HomeAssistant,
) -> None:
    """Assist exposes cover groups by default; a cloud assistant what is set."""
    async_expose_entity(home, "cloud.alexa", "cover.everything", should_expose=True)
    async_expose_entity(home, "cloud.alexa", "group.openers", should_expose=True)
    snapshot = await async_build_full_state_snapshot(
        home, network=NetworkBuildContext(options={})
    )
    network = snapshot.get("network")
    assert network is not None
    ha = cast("dict[str, Any]", network["ha_security"])
    assert ha["exposed_sensitive_entities"]["conversation"] == [
        "cover.all_covers",
        "cover.everything",
        GARAGE,
    ]
    assert ha["exposed_sensitive_entities"]["cloud.alexa"] == ["cover.everything"]
    assert ha["exposed_sensitive_groups"] == {
        "cover.all_covers": [GARAGE],
        "cover.everything": [GARAGE],
    }
    assert ha_cap("exposed_sensitive_groups") in network["capabilities"]


async def test_the_snapshot_reports_a_group_holding_an_opener_shown_as_a_blind(
    hass: HomeAssistant,
) -> None:
    """The opener's own class is in the registry only; the group is still found."""
    assert await async_setup_component(hass, "homeassistant", {})
    entry = er.async_get(hass).async_get_or_create(
        "cover",
        "test",
        "m",
        suggested_object_id="motor",
        original_device_class="garage",
    )
    hass.states.async_set(
        entry.entity_id, "closed", {"device_class": "blind", "friendly_name": "Q"}
    )
    _helper(hass, "all_covers", [entry.entity_id])
    snapshot = await async_build_full_state_snapshot(
        hass, network=NetworkBuildContext(options={})
    )
    network = snapshot.get("network")
    assert network is not None
    ha = cast("dict[str, Any]", network["ha_security"])
    assert ha["exposed_sensitive_entities"]["conversation"] == [
        "cover.all_covers",
        "cover.motor",
    ]
    assert ha["exposed_sensitive_groups"] == {"cover.all_covers": ["cover.motor"]}


async def test_a_disabled_registry_only_garage_cover_is_not_reported(
    hass: HomeAssistant,
) -> None:
    """No finding, and no exposure default written into the disabled entry."""
    assert await async_setup_component(hass, "homeassistant", {})
    entry = er.async_get(hass).async_get_or_create(
        "cover",
        "test",
        "d",
        suggested_object_id="old_garage",
        original_device_class="garage",
        disabled_by=er.RegistryEntryDisabler.USER,
    )
    snapshot = await async_build_full_state_snapshot(
        hass, network=NetworkBuildContext(options={})
    )
    network = snapshot.get("network")
    assert network is not None
    ha = cast("dict[str, Any]", network["ha_security"])
    assert ha["exposed_sensitive_entities"]["conversation"] == []
    assert async_get_entity_settings(hass, entry.entity_id) == {}

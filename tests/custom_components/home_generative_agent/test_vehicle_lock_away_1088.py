# ruff: noqa: S101
"""A car's door lock while the car is away from home is not a house door (#1088)."""

from __future__ import annotations

from datetime import timedelta
from typing import TYPE_CHECKING, Any

import pytest
from homeassistant.helpers import device_registry as dr
from homeassistant.helpers import entity_registry as er
from pytest_homeassistant.common import MockConfigEntry

from custom_components.home_generative_agent.sentinel.rules.unlocked_lock_at_night import (
    UnlockedLockAtNightRule,
)
from custom_components.home_generative_agent.snapshot.builder import (
    async_build_full_state_snapshot,
)

if TYPE_CHECKING:
    from freezegun.api import FrozenDateTimeFactory
    from homeassistant.core import HomeAssistant

    from custom_components.home_generative_agent.snapshot.schema import (
        FullStateSnapshot,
        SnapshotEntity,
    )

LOCK = "lock.911_carrera_door_lock"
TRACKER = "device_tracker.911_carrera"


def _register_car(hass: HomeAssistant, *trackers: str) -> str:
    """Register a car device holding the door lock and the given trackers."""
    owner = MockConfigEntry(domain="porscheconnect")
    owner.add_to_hass(hass)
    device = dr.async_get(hass).async_get_or_create(
        config_entry_id=owner.entry_id, identifiers={("porscheconnect", "vin")}
    )
    registry = er.async_get(hass)
    registry.async_get_or_create(
        "lock",
        "porscheconnect",
        "vin-lock",
        suggested_object_id="911_carrera_door_lock",
        device_id=device.id,
    )
    for tracker in trackers or (TRACKER,):
        registry.async_get_or_create(
            "device_tracker",
            "porscheconnect",
            f"vin-{tracker}",
            suggested_object_id=tracker.split(".", 1)[1],
            device_id=device.id,
        )
    return device.id


def _set_tracker(
    hass: HomeAssistant,
    state: str,
    *,
    entity_id: str = TRACKER,
    source_type: str = "gps",
    in_zones: list[str] | None = None,
) -> None:
    attributes: dict[str, Any] = {"source_type": source_type, "friendly_name": "911"}
    if in_zones is not None:
        attributes["in_zones"] = in_zones
    hass.states.async_set(entity_id, state, attributes)


def _unlock(hass: HomeAssistant) -> None:
    hass.states.async_set(LOCK, "unlocked", {"friendly_name": "911 Carrera Door lock"})


def _car(
    hass: HomeAssistant,
    freezer: FrozenDateTimeFactory,
    *,
    tracker_state: str,
    source_type: str = "gps",
    in_zones: list[str] | None = None,
) -> None:
    """Unlock the car, then have its tracker report in (the box's order)."""
    _register_car(hass)
    _unlock(hass)
    freezer.tick(timedelta(minutes=2))
    _set_tracker(hass, tracker_state, source_type=source_type, in_zones=in_zones)


def _lock(snapshot: FullStateSnapshot) -> SnapshotEntity:
    return next(e for e in snapshot["entities"] if e["entity_id"] == LOCK)


def _at_night(snapshot: FullStateSnapshot) -> FullStateSnapshot:
    snapshot["derived"]["is_night"] = True
    snapshot["derived"]["anyone_home"] = False
    return snapshot


def _alerts(snapshot: FullStateSnapshot) -> list[list[str]]:
    return [
        f.triggering_entities
        for f in UnlockedLockAtNightRule().evaluate(_at_night(snapshot))
    ]


async def test_a_car_away_from_home_raises_no_unlocked_lock_alert(
    hass: HomeAssistant, freezer: FrozenDateTimeFactory
) -> None:
    """The box's case: the car is driven away, its cloud data says unlocked."""
    _car(hass, freezer, tracker_state="not_home", in_zones=[])
    snapshot = await async_build_full_state_snapshot(hass)
    assert "device_away_since" in _lock(snapshot)
    assert _alerts(snapshot) == []


async def test_a_car_in_a_zone_outside_home_counts_as_away(
    hass: HomeAssistant, freezer: FrozenDateTimeFactory
) -> None:
    """A zone other than home ("Work") is still away from the house."""
    _car(hass, freezer, tracker_state="Work", in_zones=["zone.work"])
    snapshot = await async_build_full_state_snapshot(hass)
    assert _alerts(snapshot) == []


async def test_a_car_in_a_zone_inside_home_still_alerts(
    hass: HomeAssistant, freezer: FrozenDateTimeFactory
) -> None:
    """A "Driveway" zone inside the home radius is home: in_zones says so."""
    _car(
        hass,
        freezer,
        tracker_state="Driveway",
        in_zones=["zone.driveway", "zone.home"],
    )
    snapshot = await async_build_full_state_snapshot(hass)
    assert "device_away_since" not in _lock(snapshot)
    assert _alerts(snapshot) == [[LOCK]]


@pytest.mark.parametrize("tracker_state", ["home", "unknown", "unavailable", "", "  "])
async def test_a_car_at_home_or_with_no_location_still_alerts(
    hass: HomeAssistant, freezer: FrozenDateTimeFactory, tracker_state: str
) -> None:
    """Parked at home, or with no location reading, the lock still alerts."""
    _car(hass, freezer, tracker_state=tracker_state)
    snapshot = await async_build_full_state_snapshot(hass)
    assert "device_away_since" not in _lock(snapshot)
    assert _alerts(snapshot) == [[LOCK]]


async def test_one_tracker_at_home_keeps_the_device_home(
    hass: HomeAssistant, freezer: FrozenDateTimeFactory
) -> None:
    """Two GPS trackers that disagree: any home reading wins."""
    second = "device_tracker.911_carrera_phone_key"
    _register_car(hass, TRACKER, second)
    _unlock(hass)
    freezer.tick(timedelta(minutes=2))
    _set_tracker(hass, "not_home")
    _set_tracker(hass, "home", entity_id=second)
    snapshot = await async_build_full_state_snapshot(hass)
    assert "device_away_since" not in _lock(snapshot)
    assert _alerts(snapshot) == [[LOCK]]


async def test_a_lock_changed_after_the_last_away_report_still_alerts(
    hass: HomeAssistant, freezer: FrozenDateTimeFactory
) -> None:
    """Stale cloud location: the car came home and was unlocked since."""
    _register_car(hass)
    _set_tracker(hass, "not_home", in_zones=[])
    freezer.tick(timedelta(hours=4))
    _unlock(hass)
    snapshot = await async_build_full_state_snapshot(hass)
    assert "device_away_since" in _lock(snapshot)
    assert _alerts(snapshot) == [[LOCK]]


async def test_a_lock_with_no_change_time_still_alerts(
    hass: HomeAssistant, freezer: FrozenDateTimeFactory
) -> None:
    """Lock enrichment can clear last_changed; then the skip has nothing to go on."""
    _car(hass, freezer, tracker_state="not_home", in_zones=[])
    snapshot = await async_build_full_state_snapshot(hass)
    _lock(snapshot)["last_changed"] = ""
    assert _alerts(snapshot) == [[LOCK]]


async def test_a_router_tracker_on_the_device_says_nothing_about_location(
    hass: HomeAssistant, freezer: FrozenDateTimeFactory
) -> None:
    """Only a GPS tracker places the device; a router's not_home is Wi-Fi."""
    _car(hass, freezer, tracker_state="not_home", source_type="router")
    snapshot = await async_build_full_state_snapshot(hass)
    assert "device_away_since" not in _lock(snapshot)
    assert _alerts(snapshot) == [[LOCK]]


async def test_a_lock_on_another_device_is_not_marked_away(
    hass: HomeAssistant, freezer: FrozenDateTimeFactory
) -> None:
    """A tracker away from home marks only its own device's entities."""
    _car(hass, freezer, tracker_state="not_home")
    hass.states.async_set(
        "lock.front_door", "unlocked", {"friendly_name": "Front Door"}
    )
    snapshot = await async_build_full_state_snapshot(hass)
    front = next(e for e in snapshot["entities"] if e["entity_id"] == "lock.front_door")
    assert "device_away_since" not in front
    assert _alerts(snapshot) == [["lock.front_door"]]

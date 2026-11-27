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
    **extra: Any,
) -> None:
    attributes: dict[str, Any] = {"source_type": source_type, "friendly_name": "911"}
    if in_zones is not None:
        attributes["in_zones"] = in_zones
    attributes.update(extra)
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
    **extra: Any,
) -> None:
    """Unlock the car, then have its tracker report in (the box's order)."""
    _register_car(hass)
    _unlock(hass)
    freezer.tick(timedelta(minutes=2))
    _set_tracker(
        hass, tracker_state, source_type=source_type, in_zones=in_zones, **extra
    )


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


def _zone(hass: HomeAssistant, entity_id: str, lat: float, lon: float) -> None:
    hass.states.async_set(
        entity_id, "0", {"latitude": lat, "longitude": lon, "radius": 100}
    )


def _home_zone(hass: HomeAssistant, state: str = "0") -> None:
    hass.states.async_set(
        "zone.home", state, {"latitude": 37.2282, "longitude": -121.9832, "radius": 100}
    )


async def test_a_car_away_from_home_raises_no_unlocked_lock_alert(
    hass: HomeAssistant, freezer: FrozenDateTimeFactory
) -> None:
    """The box's case: the car is driven away, its cloud data says unlocked."""
    _car(hass, freezer, tracker_state="not_home", in_zones=[])
    snapshot = await async_build_full_state_snapshot(hass)
    assert _lock(snapshot).get("device_away_from_home") is True
    assert _alerts(snapshot) == []


async def test_a_car_in_a_zone_outside_home_counts_as_away(
    hass: HomeAssistant, freezer: FrozenDateTimeFactory
) -> None:
    """A zone other than home ("Work") is still away from the house."""
    _home_zone(hass)
    _zone(hass, "zone.work", 37.3861, -122.0839)
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
    assert "device_away_from_home" not in _lock(snapshot)
    assert _alerts(snapshot) == [[LOCK]]


@pytest.mark.parametrize(
    "tracker_state", ["home", "Home", "unknown", "unavailable", "", "  "]
)
async def test_a_car_at_home_or_with_no_location_still_alerts(
    hass: HomeAssistant, freezer: FrozenDateTimeFactory, tracker_state: str
) -> None:
    """Parked at home, or with no location reading, the lock still alerts."""
    _car(hass, freezer, tracker_state=tracker_state)
    snapshot = await async_build_full_state_snapshot(hass)
    assert "device_away_from_home" not in _lock(snapshot)
    assert _alerts(snapshot) == [[LOCK]]


async def test_without_in_zones_coordinates_inside_home_still_alert(
    hass: HomeAssistant, freezer: FrozenDateTimeFactory
) -> None:
    """Older Home Assistant has no in_zones: a Driveway fix near home is home."""
    _home_zone(hass)
    _car(hass, freezer, tracker_state="Driveway", latitude=37.2283, longitude=-121.9833)
    snapshot = await async_build_full_state_snapshot(hass)
    assert "device_away_from_home" not in _lock(snapshot)
    assert _alerts(snapshot) == [[LOCK]]


async def test_without_in_zones_coordinates_far_from_home_count_as_away(
    hass: HomeAssistant, freezer: FrozenDateTimeFactory
) -> None:
    """The box's car, 5 km from home, on an install with no in_zones."""
    _home_zone(hass)
    _car(hass, freezer, tracker_state="not_home", latitude=37.2756, longitude=-121.9837)
    snapshot = await async_build_full_state_snapshot(hass)
    assert _lock(snapshot).get("device_away_from_home") is True
    assert _alerts(snapshot) == []


async def test_gps_accuracy_can_place_a_fix_at_home(
    hass: HomeAssistant, freezer: FrozenDateTimeFactory
) -> None:
    """A fix just outside the radius, within its own accuracy, is home (HA's rule)."""
    _home_zone(hass)
    _car(
        hass,
        freezer,
        tracker_state="not_home",
        latitude=37.2294,
        longitude=-121.9832,
        gps_accuracy=50,
    )
    snapshot = await async_build_full_state_snapshot(hass)
    assert "device_away_from_home" not in _lock(snapshot)


async def test_a_zone_name_with_nothing_to_place_it_still_alerts(
    hass: HomeAssistant, freezer: FrozenDateTimeFactory
) -> None:
    """No in_zones and no coordinates: "Work" says nothing about home."""
    _car(hass, freezer, tracker_state="Work")
    snapshot = await async_build_full_state_snapshot(hass)
    assert "device_away_from_home" not in _lock(snapshot)
    assert _alerts(snapshot) == [[LOCK]]


async def test_not_home_with_nothing_else_counts_as_away(
    hass: HomeAssistant, freezer: FrozenDateTimeFactory
) -> None:
    """not_home is Home Assistant saying the fix is in no zone at all."""
    _car(hass, freezer, tracker_state="not_home")
    snapshot = await async_build_full_state_snapshot(hass)
    assert _lock(snapshot).get("device_away_from_home") is True


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
    assert "device_away_from_home" not in _lock(snapshot)
    assert _alerts(snapshot) == [[LOCK]]


async def test_a_router_tracker_on_the_device_says_nothing_about_location(
    hass: HomeAssistant, freezer: FrozenDateTimeFactory
) -> None:
    """Only a GPS tracker places the device; a router's not_home is Wi-Fi."""
    _car(hass, freezer, tracker_state="not_home", source_type="router")
    snapshot = await async_build_full_state_snapshot(hass)
    assert "device_away_from_home" not in _lock(snapshot)
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
    assert "device_away_from_home" not in front
    assert _alerts(snapshot) == [["lock.front_door"]]


async def test_a_location_name_home_with_empty_in_zones_is_home(
    hass: HomeAssistant, freezer: FrozenDateTimeFactory
) -> None:
    """An integration that reports only a location name "Home" gets in_zones []."""
    _car(hass, freezer, tracker_state="Home", in_zones=[])
    snapshot = await async_build_full_state_snapshot(hass)
    assert "device_away_from_home" not in _lock(snapshot)
    assert _alerts(snapshot) == [[LOCK]]


async def test_without_in_zones_a_zone_name_far_from_home_counts_as_away(
    hass: HomeAssistant, freezer: FrozenDateTimeFactory
) -> None:
    """Older Home Assistant: the coordinates settle a "Work" reading as away."""
    _home_zone(hass)
    _car(hass, freezer, tracker_state="Work", latitude=37.2756, longitude=-121.9837)
    snapshot = await async_build_full_state_snapshot(hass)
    assert _lock(snapshot).get("device_away_from_home") is True
    assert _alerts(snapshot) == []


async def test_an_empty_zone_list_with_only_a_location_name_still_alerts(
    hass: HomeAssistant, freezer: FrozenDateTimeFactory
) -> None:
    """HA writes in_zones [] when it placed nothing: "Garage" says nothing."""
    _car(hass, freezer, tracker_state="Garage", in_zones=[])
    snapshot = await async_build_full_state_snapshot(hass)
    assert "device_away_from_home" not in _lock(snapshot)
    assert _alerts(snapshot) == [[LOCK]]


async def test_a_listed_zone_inside_home_counts_as_home(
    hass: HomeAssistant, freezer: FrozenDateTimeFactory
) -> None:
    """An integration-supplied ["zone.driveway"] omits the home zone around it."""
    _home_zone(hass)
    _zone(hass, "zone.driveway", 37.2283, -121.9833)
    _car(hass, freezer, tracker_state="Driveway", in_zones=["zone.driveway"])
    snapshot = await async_build_full_state_snapshot(hass)
    assert "device_away_from_home" not in _lock(snapshot)
    assert _alerts(snapshot) == [[LOCK]]


async def test_coordinates_at_home_override_an_away_zone_list(
    hass: HomeAssistant, freezer: FrozenDateTimeFactory
) -> None:
    """The fix itself is better evidence than a zone list."""
    _home_zone(hass)
    _car(
        hass,
        freezer,
        tracker_state="Driveway",
        in_zones=[],
        latitude=37.2282,
        longitude=-121.9832,
    )
    snapshot = await async_build_full_state_snapshot(hass)
    assert "device_away_from_home" not in _lock(snapshot)


@pytest.mark.parametrize(
    "fix",
    [
        {"latitude": 10**400, "longitude": -121.9832},
        {"latitude": float("nan"), "longitude": -121.9832},
        {"latitude": 95.0, "longitude": -121.9832},
        {"latitude": True, "longitude": -121.9832},
    ],
)
async def test_a_malformed_fix_is_no_reading_and_does_not_break_the_build(
    hass: HomeAssistant, freezer: FrozenDateTimeFactory, fix: dict[str, Any]
) -> None:
    """An unusable coordinate settles nothing, and the snapshot still builds."""
    _home_zone(hass)
    _car(hass, freezer, tracker_state="Driveway", **fix)
    snapshot = await async_build_full_state_snapshot(hass)
    assert "device_away_from_home" not in _lock(snapshot)
    assert _alerts(snapshot) == [[LOCK]]


@pytest.mark.parametrize("accuracy", [float("nan"), -100, float("inf")])
async def test_an_unusable_accuracy_does_not_push_a_home_fix_away(
    hass: HomeAssistant, freezer: FrozenDateTimeFactory, accuracy: float
) -> None:
    """A fix at the home zone's centre stays home whatever the accuracy says."""
    _home_zone(hass)
    _car(
        hass,
        freezer,
        tracker_state="not_home",
        latitude=37.2282,
        longitude=-121.9832,
        gps_accuracy=accuracy,
    )
    snapshot = await async_build_full_state_snapshot(hass)
    assert "device_away_from_home" not in _lock(snapshot)


async def test_an_unavailable_home_zone_settles_nothing(
    hass: HomeAssistant, freezer: FrozenDateTimeFactory
) -> None:
    """With zone.home unavailable, a "Driveway" fix cannot be placed: alert."""
    _home_zone(hass, state="unavailable")
    _car(hass, freezer, tracker_state="Driveway", latitude=37.2756, longitude=-121.9837)
    snapshot = await async_build_full_state_snapshot(hass)
    assert "device_away_from_home" not in _lock(snapshot)

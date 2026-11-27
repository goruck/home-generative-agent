# ruff: noqa: S101
"""A car's door lock while the car is away from home is not a house door (#1088)."""

from __future__ import annotations

from typing import TYPE_CHECKING

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
    from homeassistant.core import HomeAssistant

    from custom_components.home_generative_agent.snapshot.schema import (
        FullStateSnapshot,
    )

LOCK = "lock.911_carrera_door_lock"
TRACKER = "device_tracker.911_carrera"


def _car(hass: HomeAssistant, *, tracker_state: str, source_type: str = "gps") -> None:
    """Register a car device with a door lock and a tracker, and set states."""
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
    registry.async_get_or_create(
        "device_tracker",
        "porscheconnect",
        "vin-tracker",
        suggested_object_id="911_carrera",
        device_id=device.id,
    )
    hass.states.async_set(LOCK, "unlocked", {"friendly_name": "911 Carrera Door lock"})
    hass.states.async_set(
        TRACKER, tracker_state, {"source_type": source_type, "friendly_name": "911"}
    )


def _lock(snapshot: FullStateSnapshot) -> dict[str, object]:
    return next(e for e in snapshot["entities"] if e["entity_id"] == LOCK)  # type: ignore[return-value]


def _at_night(snapshot: FullStateSnapshot) -> FullStateSnapshot:
    snapshot["derived"]["is_night"] = True
    snapshot["derived"]["anyone_home"] = False
    return snapshot


async def test_a_car_away_from_home_raises_no_unlocked_lock_alert(
    hass: HomeAssistant,
) -> None:
    """The box's case: the car is driven away, its cloud data says unlocked."""
    _car(hass, tracker_state="not_home")
    snapshot = await async_build_full_state_snapshot(hass)
    assert _lock(snapshot).get("device_away_from_home") is True
    assert UnlockedLockAtNightRule().evaluate(_at_night(snapshot)) == []


async def test_a_car_in_a_named_zone_counts_as_away(hass: HomeAssistant) -> None:
    """A zone other than home ("Work") is still away from the house."""
    _car(hass, tracker_state="Work")
    snapshot = await async_build_full_state_snapshot(hass)
    assert _lock(snapshot).get("device_away_from_home") is True


@pytest.mark.parametrize("tracker_state", ["home", "unknown", "unavailable"])
async def test_a_car_at_home_or_with_no_location_still_alerts(
    hass: HomeAssistant, tracker_state: str
) -> None:
    """Parked at home, or with no location reading, the lock still alerts."""
    _car(hass, tracker_state=tracker_state)
    snapshot = await async_build_full_state_snapshot(hass)
    assert "device_away_from_home" not in _lock(snapshot)
    findings = UnlockedLockAtNightRule().evaluate(_at_night(snapshot))
    assert [f.triggering_entities for f in findings] == [[LOCK]]


async def test_a_router_tracker_on_the_device_says_nothing_about_location(
    hass: HomeAssistant,
) -> None:
    """Only a GPS tracker places the device; a router's not_home is Wi-Fi."""
    _car(hass, tracker_state="not_home", source_type="router")
    snapshot = await async_build_full_state_snapshot(hass)
    assert "device_away_from_home" not in _lock(snapshot)
    assert len(UnlockedLockAtNightRule().evaluate(_at_night(snapshot))) == 1


async def test_a_lock_on_another_device_is_not_marked_away(
    hass: HomeAssistant,
) -> None:
    """A tracker away from home marks only its own device's entities."""
    _car(hass, tracker_state="not_home")
    hass.states.async_set(
        "lock.front_door", "unlocked", {"friendly_name": "Front Door"}
    )
    snapshot = await async_build_full_state_snapshot(hass)
    front = next(e for e in snapshot["entities"] if e["entity_id"] == "lock.front_door")
    assert "device_away_from_home" not in front
    findings = UnlockedLockAtNightRule().evaluate(_at_night(snapshot))
    assert [f.triggering_entities for f in findings] == [["lock.front_door"]]

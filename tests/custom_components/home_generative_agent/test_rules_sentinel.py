# ruff: noqa: S101
"""Tests for sentinel rules."""

from __future__ import annotations

from custom_components.home_generative_agent.sentinel.rules.appliance_power_duration import (
    AppliancePowerDurationRule,
)
from custom_components.home_generative_agent.sentinel.rules.camera_entry_unsecured import (
    CameraEntryUnsecuredRule,
)
from custom_components.home_generative_agent.sentinel.rules.open_entry_while_away import (
    OpenEntryWhileAwayRule,
)
from custom_components.home_generative_agent.sentinel.rules.unknown_person_camera_no_home import (
    UnknownPersonCameraNoHomeRule,
)
from custom_components.home_generative_agent.sentinel.rules.unlocked_lock_at_night import (
    UnlockedLockAtNightRule,
)
from custom_components.home_generative_agent.snapshot.schema import (
    FullStateSnapshot,
    validate_snapshot,
)


def _base_snapshot() -> FullStateSnapshot:
    return validate_snapshot(
        {
            "schema_version": 1,
            "generated_at": "2025-01-01T00:00:00+00:00",
            "entities": [],
            "camera_activity": [],
            "derived": {
                "now": "2025-01-01T00:00:00+00:00",
                "timezone": "UTC",
                "is_night": False,
                "anyone_home": True,
                "people_home": [],
                "people_away": [],
                "last_motion_by_area": {},
            },
        }
    )


def test_unlocked_lock_at_night_triggers() -> None:
    """Unlocked exterior lock should trigger at night."""
    snapshot = _base_snapshot()
    snapshot["derived"]["is_night"] = True
    snapshot["entities"] = [
        {
            "entity_id": "lock.front_door",
            "domain": "lock",
            "state": "unlocked",
            "friendly_name": "Front Door",
            "area": "Front",
            "attributes": {},
            "last_changed": "2025-01-01T00:00:00+00:00",
            "last_updated": "2025-01-01T00:00:00+00:00",
        }
    ]

    findings = UnlockedLockAtNightRule().evaluate(snapshot)
    assert len(findings) == 1


def test_open_entry_while_away_triggers() -> None:
    """Open entry sensors should trigger while away."""
    snapshot = _base_snapshot()
    snapshot["derived"]["anyone_home"] = False
    snapshot["entities"] = [
        {
            "entity_id": "binary_sensor.front_door",
            "domain": "binary_sensor",
            "state": "on",
            "friendly_name": "Front Door",
            "area": "Front",
            "attributes": {"device_class": "door"},
            "last_changed": "2025-01-01T00:00:00+00:00",
            "last_updated": "2025-01-01T00:00:00+00:00",
        }
    ]

    findings = OpenEntryWhileAwayRule().evaluate(snapshot)
    assert len(findings) == 1


def test_appliance_power_duration_triggers() -> None:
    """High power draw over duration should trigger."""
    snapshot = _base_snapshot()
    snapshot["derived"]["now"] = "2025-01-01T02:00:00+00:00"
    snapshot["entities"] = [
        {
            "entity_id": "sensor.washer_power",
            "domain": "sensor",
            "state": "250",
            "friendly_name": "Washer Power",
            "area": "Laundry",
            "attributes": {"device_class": "power", "unit_of_measurement": "W"},
            "last_changed": "2025-01-01T00:00:00+00:00",
            "last_updated": "2025-01-01T00:00:00+00:00",
        }
    ]

    findings = AppliancePowerDurationRule(duration_min=30).evaluate(snapshot)
    assert len(findings) == 1


def test_camera_entry_unsecured_triggers() -> None:
    """Camera activity near unsecured entry should trigger."""
    snapshot = _base_snapshot()
    snapshot["derived"]["now"] = "2025-01-01T00:05:00+00:00"
    snapshot["entities"] = [
        {
            "entity_id": "lock.front_door",
            "domain": "lock",
            "state": "unlocked",
            "friendly_name": "Front Door",
            "area": "Front",
            "attributes": {},
            "last_changed": "2025-01-01T00:00:00+00:00",
            "last_updated": "2025-01-01T00:00:00+00:00",
        }
    ]
    snapshot["camera_activity"] = [
        {
            "camera_entity_id": "camera.front",
            "area": "Front",
            "last_activity": "2025-01-01T00:04:00+00:00",
            "motion_entities": [],
            "vmd_entities": [],
            "snapshot_summary": None,
            "recognized_people": [],
            "latest_path": None,
        }
    ]

    findings = CameraEntryUnsecuredRule().evaluate(snapshot)
    assert len(findings) == 1


def test_camera_entry_unsecured_vmd_last_changed_fallback() -> None:
    """When camera has no last_activity, use linked VMD sensor last_changed."""
    snapshot = _base_snapshot()
    snapshot["derived"]["now"] = "2025-01-01T00:05:00+00:00"
    snapshot["entities"] = [
        {
            "entity_id": "lock.garage_door",
            "domain": "lock",
            "state": "unlocked",
            "friendly_name": "Garage Door",
            "area": "Garage",
            "attributes": {},
            "last_changed": "2025-01-01T00:00:00+00:00",
            "last_updated": "2025-01-01T00:00:00+00:00",
        },
        {
            "entity_id": "binary_sensor.frontporch_vmd4",
            "domain": "binary_sensor",
            "state": "on",
            "friendly_name": "Front Porch VMD4",
            "area": "Outside",
            "attributes": {"device_class": "motion"},
            "last_changed": "2025-01-01T00:04:30+00:00",
            "last_updated": "2025-01-01T00:04:30+00:00",
        },
    ]
    snapshot["camera_activity"] = [
        {
            "camera_entity_id": "camera.frontporch",
            "area": "Outside",
            "last_activity": None,
            "motion_entities": [],
            "vmd_entities": ["binary_sensor.frontporch_vmd4"],  # explicitly linked
            "snapshot_summary": None,
            "recognized_people": [],
            "latest_path": None,
        }
    ]

    findings = CameraEntryUnsecuredRule().evaluate(snapshot)
    assert len(findings) == 1
    assert findings[0].evidence["unsecured_entities"] == ["lock.garage_door"]


def test_camera_entry_unsecured_area_binary_scan_fallback() -> None:
    """Area binary sensor scan works without device_class (manufacturer-agnostic)."""
    snapshot = _base_snapshot()
    snapshot["derived"]["now"] = "2025-01-01T00:05:00+00:00"
    snapshot["entities"] = [
        {
            "entity_id": "lock.garage_door",
            "domain": "lock",
            "state": "unlocked",
            "friendly_name": "Garage Door",
            "area": "Garage",
            "attributes": {},
            "last_changed": "2025-01-01T00:00:00+00:00",
            "last_updated": "2025-01-01T00:00:00+00:00",
        },
        {
            "entity_id": "binary_sensor.playroomdoor_vmd3_0",
            "domain": "binary_sensor",
            "state": "on",
            "friendly_name": "Playroom Door VMD3",
            "area": "Outside",
            "attributes": {},  # no device_class — typical for Hikvision VMD sensors
            "last_changed": "2025-01-01T00:04:30+00:00",
            "last_updated": "2025-01-01T00:04:30+00:00",
        },
    ]
    snapshot["camera_activity"] = [
        {
            "camera_entity_id": "camera.playroomdoor",
            "area": "Outside",
            "last_activity": None,
            "motion_entities": [],
            "vmd_entities": [],
            "snapshot_summary": None,
            "recognized_people": [],
            "latest_path": None,
        }
    ]

    findings = CameraEntryUnsecuredRule().evaluate(snapshot)
    assert len(findings) == 1
    assert findings[0].evidence["unsecured_entities"] == ["lock.garage_door"]


def test_camera_entry_unsecured_exterior_area_fallback() -> None:
    """Camera in an exterior area falls back to home-wide unsecured entities."""
    snapshot = _base_snapshot()
    snapshot["derived"]["now"] = "2025-01-01T00:05:00+00:00"
    snapshot["entities"] = [
        {
            "entity_id": "lock.garage_door",
            "domain": "lock",
            "state": "unlocked",
            "friendly_name": "Garage Door",
            "area": "Garage",
            "attributes": {},
            "last_changed": "2025-01-01T00:00:00+00:00",
            "last_updated": "2025-01-01T00:00:00+00:00",
        }
    ]
    snapshot["camera_activity"] = [
        {
            "camera_entity_id": "camera.east",
            "area": "Outside",
            "last_activity": "2025-01-01T00:04:00+00:00",
            "motion_entities": [],
            "vmd_entities": [],
            "snapshot_summary": None,
            "recognized_people": [],
            "latest_path": None,
        }
    ]

    findings = CameraEntryUnsecuredRule().evaluate(snapshot)
    assert len(findings) == 1
    assert findings[0].evidence["unsecured_entities"] == ["lock.garage_door"]


def test_camera_entry_unsecured_no_trigger_when_all_secured() -> None:
    """No finding when camera fires but all entries are secured."""
    snapshot = _base_snapshot()
    snapshot["derived"]["now"] = "2025-01-01T00:05:00+00:00"
    snapshot["entities"] = [
        {
            "entity_id": "lock.front_door",
            "domain": "lock",
            "state": "locked",
            "friendly_name": "Front Door",
            "area": "Outside",
            "attributes": {},
            "last_changed": "2025-01-01T00:00:00+00:00",
            "last_updated": "2025-01-01T00:00:00+00:00",
        }
    ]
    snapshot["camera_activity"] = [
        {
            "camera_entity_id": "camera.east",
            "area": "Outside",
            "last_activity": "2025-01-01T00:04:00+00:00",
            "motion_entities": [],
            "vmd_entities": [],
            "snapshot_summary": None,
            "recognized_people": [],
            "latest_path": None,
        }
    ]

    findings = CameraEntryUnsecuredRule().evaluate(snapshot)
    assert len(findings) == 0


def test_unknown_person_camera_no_home_triggers() -> None:
    """Unknown person on camera should trigger when no one is home."""
    snapshot = _base_snapshot()
    snapshot["derived"]["anyone_home"] = False
    snapshot["camera_activity"] = [
        {
            "camera_entity_id": "camera.backyard",
            "area": "Backyard",
            "last_activity": "2025-01-01T00:04:00+00:00",
            "motion_entities": ["binary_sensor.backyard_motion"],
            "vmd_entities": [],
            "snapshot_summary": None,
            "recognized_people": [],
            "latest_path": None,
        }
    ]

    findings = UnknownPersonCameraNoHomeRule().evaluate(snapshot)
    assert len(findings) == 1
    assert findings[0].type == "unknown_person_camera_no_home"
    assert findings[0].severity == "low"
    assert findings[0].confidence == 0.85
    assert "close_entry" in findings[0].suggested_actions


def test_unknown_person_camera_no_home_no_trigger_when_home() -> None:
    """No finding when someone is home, even if an unknown person is on camera."""
    snapshot = _base_snapshot()
    snapshot["derived"]["anyone_home"] = True
    snapshot["camera_activity"] = [
        {
            "camera_entity_id": "camera.backyard",
            "area": "Backyard",
            "last_activity": "2025-01-01T00:04:00+00:00",
            "motion_entities": ["binary_sensor.backyard_motion"],
            "vmd_entities": [],
            "snapshot_summary": None,
            "recognized_people": [],
            "latest_path": None,
        }
    ]

    findings = UnknownPersonCameraNoHomeRule().evaluate(snapshot)
    assert len(findings) == 0


def test_unknown_person_camera_no_home_no_trigger_when_recognized() -> None:
    """No finding when the detected person is recognized."""
    snapshot = _base_snapshot()
    snapshot["derived"]["anyone_home"] = False
    snapshot["camera_activity"] = [
        {
            "camera_entity_id": "camera.backyard",
            "area": "Backyard",
            "last_activity": "2025-01-01T00:04:00+00:00",
            "motion_entities": ["binary_sensor.backyard_motion"],
            "vmd_entities": [],
            "snapshot_summary": None,
            "recognized_people": ["person.jane"],
            "latest_path": None,
        }
    ]

    findings = UnknownPersonCameraNoHomeRule().evaluate(snapshot)
    assert len(findings) == 0


def test_unknown_person_camera_no_home_no_trigger_without_motion() -> None:
    """No finding when camera has activity but no motion or VMD entities."""
    snapshot = _base_snapshot()
    snapshot["derived"]["anyone_home"] = False
    snapshot["camera_activity"] = [
        {
            "camera_entity_id": "camera.backyard",
            "area": "Backyard",
            "last_activity": "2025-01-01T00:04:00+00:00",
            "motion_entities": [],
            "vmd_entities": [],
            "snapshot_summary": None,
            "recognized_people": [],
            "latest_path": None,
        }
    ]

    findings = UnknownPersonCameraNoHomeRule().evaluate(snapshot)
    assert len(findings) == 0

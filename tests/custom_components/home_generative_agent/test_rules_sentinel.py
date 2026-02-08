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

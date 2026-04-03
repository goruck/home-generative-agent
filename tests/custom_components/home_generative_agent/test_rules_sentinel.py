"""Tests for sentinel rules."""

from __future__ import annotations

from typing import Any

from custom_components.home_generative_agent.explain.prompts import SYSTEM_PROMPT
from custom_components.home_generative_agent.sentinel.dynamic_rules import (
    evaluate_dynamic_rule,
)
from custom_components.home_generative_agent.sentinel.rules.alarm_disarmed_external_threat import (
    AlarmDisarmedDuringExternalThreatRule,
)
from custom_components.home_generative_agent.sentinel.rules.appliance_power_duration import (
    AppliancePowerDurationRule,
)
from custom_components.home_generative_agent.sentinel.rules.camera_entry_unsecured import (
    CameraEntryUnsecuredRule,
)
from custom_components.home_generative_agent.sentinel.rules.camera_missing_snapshot import (
    CameraMissingSnapshotRule,
)
from custom_components.home_generative_agent.sentinel.rules.open_entry_while_away import (
    OpenEntryWhileAwayRule,
)
from custom_components.home_generative_agent.sentinel.rules.phone_battery_low_at_night import (
    PhoneBatteryLowAtNightRule,
)
from custom_components.home_generative_agent.sentinel.rules.unknown_person_camera_night_home import (
    UnknownPersonAtNightWhileHomeRule,
)
from custom_components.home_generative_agent.sentinel.rules.unknown_person_camera_no_home import (
    UnknownPersonCameraNoHomeRule,
)
from custom_components.home_generative_agent.sentinel.rules.unlocked_lock_at_night import (
    UnlockedLockAtNightRule,
)
from custom_components.home_generative_agent.sentinel.rules.vehicle_detected_near_camera import (
    VehicleDetectedNearCameraRule,
)
from custom_components.home_generative_agent.snapshot.schema import (
    CameraActivity,
    FullStateSnapshot,
    SnapshotEntity,
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


def test_unlocked_lock_main_hint_triggers() -> None:
    """Lock with 'main' in friendly_name triggers with expanded EXTERIOR_HINTS."""
    snapshot = _base_snapshot()
    snapshot["derived"]["is_night"] = True
    snapshot["entities"] = [
        {
            "entity_id": "lock.smart_lock_1",
            "domain": "lock",
            "state": "unlocked",
            "friendly_name": "Main Entrance Lock",
            "area": None,
            "attributes": {},
            "last_changed": "2025-01-01T00:00:00+00:00",
            "last_updated": "2025-01-01T00:00:00+00:00",
        }
    ]
    findings = UnlockedLockAtNightRule().evaluate(snapshot)
    assert len(findings) == 1


def test_unlocked_lock_driveway_area_triggers() -> None:
    """Lock with area 'Driveway' triggers with expanded EXTERIOR_HINTS."""
    snapshot = _base_snapshot()
    snapshot["derived"]["is_night"] = True
    snapshot["entities"] = [
        {
            "entity_id": "lock.gate_lock",
            "domain": "lock",
            "state": "unlocked",
            "friendly_name": "Smart Lock Pro",
            "area": "Driveway",
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
    assert findings[0].evidence["area"] == "Front"
    assert findings[0].evidence["camera_area"] == "Front"
    assert findings[0].evidence["unsecured_entity_areas"] == {
        "lock.front_door": "Front"
    }


def test_camera_entry_unsecured_vmd_last_changed_fallback() -> None:
    """When camera has no last_activity, use linked VMD sensor last_changed."""
    snapshot = _base_snapshot()
    snapshot["derived"]["now"] = "2025-01-01T00:05:00+00:00"
    snapshot["entities"] = [
        {
            "entity_id": "lock.outside_gate",
            "domain": "lock",
            "state": "unlocked",
            "friendly_name": "Outside Gate",
            "area": "Outside",  # same area as camera — valid same-area relationship
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
    assert findings[0].evidence["unsecured_entities"] == ["lock.outside_gate"]


def test_camera_entry_unsecured_area_binary_scan_fallback() -> None:
    """Area binary sensor scan works without device_class (manufacturer-agnostic)."""
    snapshot = _base_snapshot()
    snapshot["derived"]["now"] = "2025-01-01T00:05:00+00:00"
    snapshot["entities"] = [
        {
            "entity_id": "lock.outside_gate",
            "domain": "lock",
            "state": "unlocked",
            "friendly_name": "Outside Gate",
            "area": "Outside",  # same area as camera — valid same-area relationship
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
    assert findings[0].evidence["unsecured_entities"] == ["lock.outside_gate"]


def test_camera_entry_unsecured_exterior_area_fallback() -> None:
    """
    Camera in exterior area with cross-area unsecured entities fires no finding.

    Previously a home-wide fallback caused exterior cameras to report unsecured
    entries from unrelated areas.  The fix removes the fallback: only same-area
    unsecured entries are associated with a camera.
    """
    snapshot = _base_snapshot()
    snapshot["derived"]["now"] = "2025-01-01T00:05:00+00:00"
    snapshot["entities"] = [
        {
            "entity_id": "lock.garage_door",
            "domain": "lock",
            "state": "unlocked",
            "friendly_name": "Garage Door",
            "area": "Garage",  # different area from camera ("Outside")
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
    assert len(findings) == 0  # no same-area unsecured entries → no finding


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


def test_camera_entry_unsecured_interior_area_no_fallback() -> None:
    """
    Interior camera with only cross-area unsecured entries fires no finding.

    The same-area-only rule applies to all cameras, not just exterior ones.
    A camera in 'Garage' should not report a lock unsecured in 'Front'.
    """
    snapshot = _base_snapshot()
    snapshot["derived"]["now"] = "2025-01-01T00:05:00+00:00"
    snapshot["entities"] = [
        {
            "entity_id": "lock.front_door",
            "domain": "lock",
            "state": "unlocked",
            "friendly_name": "Front Door",
            "area": "Front",  # different area from camera ("Garage")
            "attributes": {},
            "last_changed": "2025-01-01T00:00:00+00:00",
            "last_updated": "2025-01-01T00:00:00+00:00",
        }
    ]
    snapshot["camera_activity"] = [
        {
            "camera_entity_id": "camera.garage_interior",
            "area": "Garage",
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


# ---------------------------------------------------------------------------
# Helpers shared by dynamic-rule evaluator tests
# ---------------------------------------------------------------------------


def _entity(
    entity_id: str,
    state: str,
    domain: str | None = None,
    last_changed: str = "2025-01-01T00:00:00+00:00",
    attributes: dict[str, Any] | None = None,
) -> SnapshotEntity:
    d = domain or entity_id.split(".", 1)[0]
    return SnapshotEntity(
        entity_id=entity_id,
        domain=d,
        state=state,
        friendly_name=entity_id,
        area="Test Area",
        attributes=attributes or {},
        last_changed=last_changed,
        last_updated=last_changed,
    )


def _dyn_rule(
    template_id: str, rule_id: str, params: dict[str, Any], **kwargs: Any
) -> dict[str, Any]:
    return {
        "template_id": template_id,
        "rule_id": rule_id,
        "params": params,
        "severity": "low",
        "confidence": 0.8,
        **kwargs,
    }


# ---------------------------------------------------------------------------
# unlocked_lock_while_away evaluator
# ---------------------------------------------------------------------------


def test_dynamic_unlocked_lock_while_away_triggers() -> None:
    snapshot = _base_snapshot()
    snapshot["derived"]["anyone_home"] = False
    snapshot["entities"] = [_entity("lock.garage_door_lock", "unlocked")]
    rule = _dyn_rule(
        "unlocked_lock_while_away",
        "test_rule",
        {"lock_entity_id": "lock.garage_door_lock"},
    )
    findings = evaluate_dynamic_rule(snapshot, rule)
    assert len(findings) == 1
    assert findings[0].evidence["lock_entity_id"] == "lock.garage_door_lock"


def test_dynamic_unlocked_lock_while_away_no_trigger_when_home() -> None:
    snapshot = _base_snapshot()
    snapshot["derived"]["anyone_home"] = True
    snapshot["entities"] = [_entity("lock.garage_door_lock", "unlocked")]
    rule = _dyn_rule(
        "unlocked_lock_while_away",
        "test_rule",
        {"lock_entity_id": "lock.garage_door_lock"},
    )
    findings = evaluate_dynamic_rule(snapshot, rule)
    assert len(findings) == 0


def test_dynamic_unlocked_lock_while_away_no_trigger_when_locked() -> None:
    snapshot = _base_snapshot()
    snapshot["derived"]["anyone_home"] = False
    snapshot["entities"] = [_entity("lock.garage_door_lock", "locked")]
    rule = _dyn_rule(
        "unlocked_lock_while_away",
        "test_rule",
        {"lock_entity_id": "lock.garage_door_lock"},
    )
    findings = evaluate_dynamic_rule(snapshot, rule)
    assert len(findings) == 0


# ---------------------------------------------------------------------------
# alarm_state_mismatch evaluator
# ---------------------------------------------------------------------------


def test_dynamic_alarm_state_mismatch_triggers_armed_home_while_away() -> None:
    snapshot = _base_snapshot()
    snapshot["derived"]["anyone_home"] = False
    snapshot["entities"] = [
        _entity("alarm_control_panel.home_alarm", "armed_home", "alarm_control_panel")
    ]
    rule = _dyn_rule(
        "alarm_state_mismatch",
        "test_rule",
        {
            "alarm_entity_id": "alarm_control_panel.home_alarm",
            "alarm_state": "armed_home",
            "expected_presence": "away",
        },
    )
    findings = evaluate_dynamic_rule(snapshot, rule)
    assert len(findings) == 1
    assert findings[0].evidence["alarm_state"] == "armed_home"


def test_dynamic_alarm_state_mismatch_no_trigger_when_home() -> None:
    snapshot = _base_snapshot()
    snapshot["derived"]["anyone_home"] = True
    snapshot["entities"] = [
        _entity("alarm_control_panel.home_alarm", "armed_home", "alarm_control_panel")
    ]
    rule = _dyn_rule(
        "alarm_state_mismatch",
        "test_rule",
        {
            "alarm_entity_id": "alarm_control_panel.home_alarm",
            "alarm_state": "armed_home",
            "expected_presence": "away",
        },
    )
    findings = evaluate_dynamic_rule(snapshot, rule)
    assert len(findings) == 0


def test_dynamic_alarm_state_mismatch_no_trigger_when_state_differs() -> None:
    snapshot = _base_snapshot()
    snapshot["derived"]["anyone_home"] = False
    snapshot["entities"] = [
        _entity("alarm_control_panel.home_alarm", "disarmed", "alarm_control_panel")
    ]
    rule = _dyn_rule(
        "alarm_state_mismatch",
        "test_rule",
        {
            "alarm_entity_id": "alarm_control_panel.home_alarm",
            "alarm_state": "armed_home",
            "expected_presence": "away",
        },
    )
    findings = evaluate_dynamic_rule(snapshot, rule)
    assert len(findings) == 0


# ---------------------------------------------------------------------------
# entity_state_duration evaluator
# ---------------------------------------------------------------------------


def test_dynamic_entity_state_duration_triggers_entry_open_too_long() -> None:
    snapshot = _base_snapshot()
    snapshot["derived"]["now"] = "2025-01-01T04:00:00+00:00"
    snapshot["entities"] = [
        _entity(
            "binary_sensor.garage_window",
            "on",
            last_changed="2025-01-01T00:00:00+00:00",
        )
    ]
    rule = _dyn_rule(
        "entity_state_duration",
        "test_rule",
        {
            "entity_id": "binary_sensor.garage_window",
            "target_state": "on",
            "threshold_hours": 2.0,
        },
    )
    findings = evaluate_dynamic_rule(snapshot, rule)
    assert len(findings) == 1
    assert findings[0].evidence["duration_hours"] == 4.0


def test_dynamic_entity_state_duration_no_trigger_below_threshold() -> None:
    snapshot = _base_snapshot()
    snapshot["derived"]["now"] = "2025-01-01T01:00:00+00:00"
    snapshot["entities"] = [
        _entity(
            "binary_sensor.garage_window",
            "on",
            last_changed="2025-01-01T00:00:00+00:00",
        )
    ]
    rule = _dyn_rule(
        "entity_state_duration",
        "test_rule",
        {
            "entity_id": "binary_sensor.garage_window",
            "target_state": "on",
            "threshold_hours": 2.0,
        },
    )
    findings = evaluate_dynamic_rule(snapshot, rule)
    assert len(findings) == 0


def test_dynamic_entity_state_duration_no_trigger_wrong_state() -> None:
    snapshot = _base_snapshot()
    snapshot["derived"]["now"] = "2025-01-01T04:00:00+00:00"
    snapshot["entities"] = [
        _entity(
            "binary_sensor.garage_window",
            "off",
            last_changed="2025-01-01T00:00:00+00:00",
        )
    ]
    rule = _dyn_rule(
        "entity_state_duration",
        "test_rule",
        {
            "entity_id": "binary_sensor.garage_window",
            "target_state": "on",
            "threshold_hours": 2.0,
        },
    )
    findings = evaluate_dynamic_rule(snapshot, rule)
    assert len(findings) == 0


# ---------------------------------------------------------------------------
# sensor_threshold_condition evaluator
# ---------------------------------------------------------------------------


def test_dynamic_sensor_threshold_condition_triggers() -> None:
    snapshot = _base_snapshot()
    snapshot["derived"]["anyone_home"] = True
    snapshot["entities"] = [
        _entity(
            "sensor.microwave_power",
            "1200",
            attributes={"device_class": "power", "unit_of_measurement": "W"},
        )
    ]
    rule = _dyn_rule(
        "sensor_threshold_condition",
        "test_rule",
        {
            "sensor_entity_id": "sensor.microwave_power",
            "threshold": 1000.0,
            "require_home": True,
        },
    )
    findings = evaluate_dynamic_rule(snapshot, rule)
    assert len(findings) == 1
    assert findings[0].evidence["sensor_value"] == 1200.0


def test_dynamic_sensor_threshold_condition_no_trigger_below_threshold() -> None:
    snapshot = _base_snapshot()
    snapshot["derived"]["anyone_home"] = True
    snapshot["entities"] = [
        _entity(
            "sensor.microwave_power",
            "800",
            attributes={"device_class": "power", "unit_of_measurement": "W"},
        )
    ]
    rule = _dyn_rule(
        "sensor_threshold_condition",
        "test_rule",
        {
            "sensor_entity_id": "sensor.microwave_power",
            "threshold": 1000.0,
            "require_home": True,
        },
    )
    findings = evaluate_dynamic_rule(snapshot, rule)
    assert len(findings) == 0


def test_dynamic_sensor_threshold_condition_no_trigger_condition_unmet() -> None:
    """require_home=True but no one home — should not trigger."""
    snapshot = _base_snapshot()
    snapshot["derived"]["anyone_home"] = False
    snapshot["entities"] = [
        _entity(
            "sensor.microwave_power",
            "1200",
            attributes={"device_class": "power", "unit_of_measurement": "W"},
        )
    ]
    rule = _dyn_rule(
        "sensor_threshold_condition",
        "test_rule",
        {
            "sensor_entity_id": "sensor.microwave_power",
            "threshold": 1000.0,
            "require_home": True,
        },
    )
    findings = evaluate_dynamic_rule(snapshot, rule)
    assert len(findings) == 0


def test_dynamic_sensor_threshold_condition_night_condition() -> None:
    snapshot = _base_snapshot()
    snapshot["derived"]["is_night"] = True
    snapshot["entities"] = [
        _entity(
            "sensor.washing_machine_power",
            "120",
            attributes={"unit_of_measurement": "W"},
        )
    ]
    rule = _dyn_rule(
        "sensor_threshold_condition",
        "test_rule",
        {
            "sensor_entity_id": "sensor.washing_machine_power",
            "threshold": 50.0,
            "require_night": True,
        },
    )
    findings = evaluate_dynamic_rule(snapshot, rule)
    assert len(findings) == 1


def test_dynamic_sensor_threshold_condition_no_trigger_not_night() -> None:
    snapshot = _base_snapshot()
    snapshot["derived"]["is_night"] = False
    snapshot["entities"] = [
        _entity(
            "sensor.washing_machine_power",
            "120",
            attributes={"unit_of_measurement": "W"},
        )
    ]
    rule = _dyn_rule(
        "sensor_threshold_condition",
        "test_rule",
        {
            "sensor_entity_id": "sensor.washing_machine_power",
            "threshold": 50.0,
            "require_night": True,
        },
    )
    findings = evaluate_dynamic_rule(snapshot, rule)
    assert len(findings) == 0


# ---------------------------------------------------------------------------
# entity_staleness evaluator
# ---------------------------------------------------------------------------


def test_dynamic_entity_staleness_triggers() -> None:
    snapshot = _base_snapshot()
    snapshot["derived"]["now"] = "2025-01-03T00:00:00+00:00"
    snapshot["entities"] = [
        _entity(
            "person.lindo", "home", "person", last_changed="2025-01-01T00:00:00+00:00"
        )
    ]
    rule = _dyn_rule(
        "entity_staleness",
        "test_rule",
        {"entity_id": "person.lindo", "max_stale_hours": 24.0},
    )
    findings = evaluate_dynamic_rule(snapshot, rule)
    assert len(findings) == 1
    assert findings[0].evidence["age_hours"] == 48.0


def test_dynamic_entity_staleness_no_trigger_recent() -> None:
    snapshot = _base_snapshot()
    snapshot["derived"]["now"] = "2025-01-01T12:00:00+00:00"
    snapshot["entities"] = [
        _entity(
            "person.lindo", "home", "person", last_changed="2025-01-01T00:00:00+00:00"
        )
    ]
    rule = _dyn_rule(
        "entity_staleness",
        "test_rule",
        {"entity_id": "person.lindo", "max_stale_hours": 24.0},
    )
    findings = evaluate_dynamic_rule(snapshot, rule)
    assert len(findings) == 0


# ---------------------------------------------------------------------------
# multiple_entries_open_count evaluator
# ---------------------------------------------------------------------------


def test_dynamic_multiple_entries_open_count_triggers() -> None:
    snapshot = _base_snapshot()
    snapshot["derived"]["anyone_home"] = True
    snapshot["entities"] = [
        _entity("binary_sensor.window_a", "on"),
        _entity("binary_sensor.window_b", "on"),
        _entity("binary_sensor.window_c", "off"),
    ]
    rule = _dyn_rule(
        "multiple_entries_open_count",
        "test_rule",
        {
            "entry_entity_ids": [
                "binary_sensor.window_a",
                "binary_sensor.window_b",
                "binary_sensor.window_c",
            ],
            "min_count": 2,
            "require_home": True,
        },
    )
    findings = evaluate_dynamic_rule(snapshot, rule)
    assert len(findings) == 1
    assert findings[0].evidence["open_count"] == 2
    assert set(findings[0].triggering_entities) == {
        "binary_sensor.window_a",
        "binary_sensor.window_b",
    }


def test_dynamic_multiple_entries_open_count_no_trigger_below_min() -> None:
    snapshot = _base_snapshot()
    snapshot["derived"]["anyone_home"] = True
    snapshot["entities"] = [
        _entity("binary_sensor.window_a", "on"),
        _entity("binary_sensor.window_b", "off"),
        _entity("binary_sensor.window_c", "off"),
    ]
    rule = _dyn_rule(
        "multiple_entries_open_count",
        "test_rule",
        {
            "entry_entity_ids": [
                "binary_sensor.window_a",
                "binary_sensor.window_b",
                "binary_sensor.window_c",
            ],
            "min_count": 2,
        },
    )
    findings = evaluate_dynamic_rule(snapshot, rule)
    assert len(findings) == 0


def test_dynamic_multiple_entries_open_count_no_trigger_presence_unmet() -> None:
    snapshot = _base_snapshot()
    snapshot["derived"]["anyone_home"] = False
    snapshot["entities"] = [
        _entity("binary_sensor.window_a", "on"),
        _entity("binary_sensor.window_b", "on"),
    ]
    rule = _dyn_rule(
        "multiple_entries_open_count",
        "test_rule",
        {
            "entry_entity_ids": ["binary_sensor.window_a", "binary_sensor.window_b"],
            "min_count": 2,
            "require_home": True,
        },
    )
    findings = evaluate_dynamic_rule(snapshot, rule)
    assert len(findings) == 0


# ---------------------------------------------------------------------------
# UnknownPersonAtNightWhileHomeRule
# ---------------------------------------------------------------------------


def _camera_activity(
    camera_entity_id: str,
    *,
    snapshot_summary: str | None = None,
    recognized_people: list[str] | None = None,
    last_activity: str | None = "2025-01-01T00:04:00+00:00",
    motion_entities: list[str] | None = None,
    vmd_entities: list[str] | None = None,
    area: str | None = "Outside",
) -> CameraActivity:
    return CameraActivity(
        camera_entity_id=camera_entity_id,
        area=area,
        last_activity=last_activity,
        motion_entities=motion_entities or [],
        vmd_entities=vmd_entities or [],
        snapshot_summary=snapshot_summary,
        recognized_people=recognized_people or [],
        latest_path=None,
    )


def test_unknown_person_at_night_while_home_triggers() -> None:
    """Unknown person on camera at night while home should trigger."""
    snapshot = _base_snapshot()
    snapshot["derived"]["is_night"] = True
    snapshot["derived"]["anyone_home"] = True
    snapshot["camera_activity"] = [
        _camera_activity(
            "camera.frontporch",
            snapshot_summary="Person holding a dark garment standing at door.",
        )
    ]
    findings = UnknownPersonAtNightWhileHomeRule().evaluate(snapshot)
    assert len(findings) == 1
    assert findings[0].type == "unknown_person_camera_night_home"
    assert findings[0].severity == "low"
    assert findings[0].confidence == 0.7


def test_unknown_person_at_night_while_home_no_trigger_when_day() -> None:
    """No finding during the day."""
    snapshot = _base_snapshot()
    snapshot["derived"]["is_night"] = False
    snapshot["derived"]["anyone_home"] = True
    snapshot["camera_activity"] = [
        _camera_activity("camera.frontporch", snapshot_summary="Person at door.")
    ]
    findings = UnknownPersonAtNightWhileHomeRule().evaluate(snapshot)
    assert len(findings) == 0


def test_unknown_person_at_night_while_home_no_trigger_when_away() -> None:
    """No finding when no one is home (handled by the no-home rule instead)."""
    snapshot = _base_snapshot()
    snapshot["derived"]["is_night"] = True
    snapshot["derived"]["anyone_home"] = False
    snapshot["camera_activity"] = [
        _camera_activity("camera.frontporch", snapshot_summary="Person at door.")
    ]
    findings = UnknownPersonAtNightWhileHomeRule().evaluate(snapshot)
    assert len(findings) == 0


def test_unknown_person_at_night_while_home_no_trigger_when_recognized() -> None:
    """No finding when the person is recognized."""
    snapshot = _base_snapshot()
    snapshot["derived"]["is_night"] = True
    snapshot["derived"]["anyone_home"] = True
    snapshot["camera_activity"] = [
        _camera_activity(
            "camera.frontporch",
            snapshot_summary="Lindo at door.",
            recognized_people=["person.lindo_st_angel"],
        )
    ]
    findings = UnknownPersonAtNightWhileHomeRule().evaluate(snapshot)
    assert len(findings) == 0


def test_unknown_person_at_night_while_home_no_trigger_without_summary() -> None:
    """No finding when camera has no snapshot summary."""
    snapshot = _base_snapshot()
    snapshot["derived"]["is_night"] = True
    snapshot["derived"]["anyone_home"] = True
    snapshot["camera_activity"] = [
        _camera_activity("camera.frontporch", snapshot_summary=None)
    ]
    findings = UnknownPersonAtNightWhileHomeRule().evaluate(snapshot)
    assert len(findings) == 0


# ---------------------------------------------------------------------------
# VehicleDetectedNearCameraRule
# ---------------------------------------------------------------------------


def test_vehicle_detected_near_camera_triggers() -> None:
    """Vehicle in snapshot summary with motion context while home should trigger."""
    snapshot = _base_snapshot()
    snapshot["derived"]["anyone_home"] = True
    snapshot["camera_activity"] = [
        _camera_activity(
            "camera.driveway",
            snapshot_summary="A white SUV is parked in the driveway.",
            motion_entities=["binary_sensor.driveway_motion"],
        )
    ]
    findings = VehicleDetectedNearCameraRule().evaluate(snapshot)
    assert len(findings) == 1
    assert findings[0].type == "vehicle_detected_near_camera_home"
    assert findings[0].triggering_entities == ["camera.driveway"]


def test_vehicle_detected_near_camera_triggers_backyard() -> None:
    """Vehicle on any camera (e.g. backyard) should also trigger."""
    snapshot = _base_snapshot()
    snapshot["derived"]["anyone_home"] = True
    snapshot["camera_activity"] = [
        _camera_activity(
            "camera.backyard",
            snapshot_summary="A white car is parked in the driveway.",
            motion_entities=["binary_sensor.backyard_motion"],
        )
    ]
    findings = VehicleDetectedNearCameraRule().evaluate(snapshot)
    assert len(findings) == 1
    assert findings[0].triggering_entities == ["camera.backyard"]


def test_vehicle_detected_near_camera_two_cameras_two_findings() -> None:
    """Two cameras with vehicles and motion context yield two findings."""
    snapshot = _base_snapshot()
    snapshot["derived"]["anyone_home"] = True
    snapshot["camera_activity"] = [
        _camera_activity(
            "camera.front",
            snapshot_summary="SUV parked near entrance.",
            motion_entities=["binary_sensor.front_motion"],
        ),
        _camera_activity(
            "camera.side",
            snapshot_summary="Van parked along the side fence.",
            motion_entities=["binary_sensor.side_motion"],
        ),
    ]
    findings = VehicleDetectedNearCameraRule().evaluate(snapshot)
    assert len(findings) == 2
    entity_ids = {f.triggering_entities[0] for f in findings}
    assert entity_ids == {"camera.front", "camera.side"}


def test_vehicle_detected_near_camera_no_trigger_when_away() -> None:
    """No finding when no one is home."""
    snapshot = _base_snapshot()
    snapshot["derived"]["anyone_home"] = False
    snapshot["camera_activity"] = [
        _camera_activity(
            "camera.driveway",
            snapshot_summary="White SUV parked outside.",
            motion_entities=["binary_sensor.driveway_motion"],
        )
    ]
    findings = VehicleDetectedNearCameraRule().evaluate(snapshot)
    assert len(findings) == 0


def test_vehicle_detected_near_camera_no_trigger_without_vehicle_keyword() -> None:
    """No finding when summary doesn't mention a vehicle."""
    snapshot = _base_snapshot()
    snapshot["derived"]["anyone_home"] = True
    snapshot["camera_activity"] = [
        _camera_activity(
            "camera.front",
            snapshot_summary="Person walking past the gate.",
            motion_entities=["binary_sensor.front_motion"],
        )
    ]
    findings = VehicleDetectedNearCameraRule().evaluate(snapshot)
    assert len(findings) == 0


def test_vehicle_detected_near_camera_no_trigger_no_motion_context() -> None:
    """No finding when camera has vehicle summary but no motion context."""
    snapshot = _base_snapshot()
    snapshot["derived"]["anyone_home"] = True
    snapshot["camera_activity"] = [
        _camera_activity(
            "camera.indoor",
            snapshot_summary="A sedan is visible through the window.",
            motion_entities=None,
            vmd_entities=None,
            last_activity=None,
        )
    ]
    findings = VehicleDetectedNearCameraRule().evaluate(snapshot)
    assert len(findings) == 0


def test_vehicle_detected_near_camera_no_trigger_no_summary() -> None:
    """No finding when camera has no snapshot summary."""
    snapshot = _base_snapshot()
    snapshot["derived"]["anyone_home"] = True
    snapshot["camera_activity"] = [
        _camera_activity(
            "camera.driveway",
            snapshot_summary=None,
            motion_entities=["binary_sensor.driveway_motion"],
        )
    ]
    findings = VehicleDetectedNearCameraRule().evaluate(snapshot)
    assert len(findings) == 0


# ---------------------------------------------------------------------------
# CameraMissingSnapshotRule
# ---------------------------------------------------------------------------


def test_camera_missing_snapshot_triggers() -> None:
    """Missing snapshot on a monitored camera at night while home should trigger."""
    snapshot = _base_snapshot()
    snapshot["derived"]["is_night"] = True
    snapshot["derived"]["anyone_home"] = True
    snapshot["camera_activity"] = [
        _camera_activity(
            "camera.outdoor_front",
            snapshot_summary=None,
            last_activity=None,
            motion_entities=["binary_sensor.outdoor_front_motion"],
        )
    ]
    findings = CameraMissingSnapshotRule().evaluate(snapshot)
    assert len(findings) == 1
    assert findings[0].type == "camera_missing_snapshot_night_home"
    assert findings[0].triggering_entities == ["camera.outdoor_front"]


def test_camera_missing_snapshot_no_trigger_when_summary_present() -> None:
    """No finding when the camera has a snapshot summary."""
    snapshot = _base_snapshot()
    snapshot["derived"]["is_night"] = True
    snapshot["derived"]["anyone_home"] = True
    snapshot["camera_activity"] = [
        _camera_activity(
            "camera.outdoor_front",
            snapshot_summary="Empty driveway, no activity.",
            motion_entities=["binary_sensor.outdoor_front_motion"],
        )
    ]
    findings = CameraMissingSnapshotRule().evaluate(snapshot)
    assert len(findings) == 0


def test_camera_missing_snapshot_no_trigger_during_day() -> None:
    """No finding during the day even if snapshot is missing."""
    snapshot = _base_snapshot()
    snapshot["derived"]["is_night"] = False
    snapshot["derived"]["anyone_home"] = True
    snapshot["camera_activity"] = [
        _camera_activity(
            "camera.outdoor_front",
            snapshot_summary=None,
            last_activity=None,
            motion_entities=["binary_sensor.outdoor_front_motion"],
        )
    ]
    findings = CameraMissingSnapshotRule().evaluate(snapshot)
    assert len(findings) == 0


def test_camera_missing_snapshot_no_trigger_when_away() -> None:
    """No finding when no one is home."""
    snapshot = _base_snapshot()
    snapshot["derived"]["is_night"] = True
    snapshot["derived"]["anyone_home"] = False
    snapshot["camera_activity"] = [
        _camera_activity(
            "camera.outdoor_front",
            snapshot_summary=None,
            last_activity=None,
            motion_entities=["binary_sensor.outdoor_front_motion"],
        )
    ]
    findings = CameraMissingSnapshotRule().evaluate(snapshot)
    assert len(findings) == 0


def test_camera_missing_snapshot_no_trigger_when_empty_camera_activity() -> None:
    """No finding when camera_activity is empty — generalized rule has no expected-camera concept."""
    snapshot = _base_snapshot()
    snapshot["derived"]["is_night"] = True
    snapshot["derived"]["anyone_home"] = True
    snapshot["camera_activity"] = []
    findings = CameraMissingSnapshotRule().evaluate(snapshot)
    assert len(findings) == 0


def test_camera_missing_snapshot_no_trigger_without_motion_entities() -> None:
    """No finding for a camera without motion_entities even if summary is absent."""
    snapshot = _base_snapshot()
    snapshot["derived"]["is_night"] = True
    snapshot["derived"]["anyone_home"] = True
    snapshot["camera_activity"] = [
        _camera_activity(
            "camera.indoor",
            snapshot_summary=None,
            last_activity=None,
            motion_entities=None,
        )
    ]
    findings = CameraMissingSnapshotRule().evaluate(snapshot)
    assert len(findings) == 0


def test_camera_missing_snapshot_multiple_cameras_one_finding() -> None:
    """One finding for the camera with motion_entities + no summary; none for the other."""
    snapshot = _base_snapshot()
    snapshot["derived"]["is_night"] = True
    snapshot["derived"]["anyone_home"] = True
    snapshot["camera_activity"] = [
        _camera_activity(
            "camera.outdoor_back",
            snapshot_summary=None,
            last_activity=None,
            motion_entities=["binary_sensor.back_motion"],
        ),
        _camera_activity(
            "camera.indoor",
            snapshot_summary=None,
            last_activity=None,
            motion_entities=None,
        ),
    ]
    findings = CameraMissingSnapshotRule().evaluate(snapshot)
    assert len(findings) == 1
    assert findings[0].triggering_entities == ["camera.outdoor_back"]


# ---------------------------------------------------------------------------
# AlarmDisarmedDuringExternalThreatRule
# ---------------------------------------------------------------------------


def _alarm_entity(state: str) -> SnapshotEntity:
    return SnapshotEntity(
        entity_id="alarm_control_panel.home_alarm",
        domain="alarm_control_panel",
        state=state,
        friendly_name="Home Alarm",
        area=None,
        attributes={},
        last_changed="2025-01-01T00:00:00+00:00",
        last_updated="2025-01-01T00:00:00+00:00",
    )


def test_alarm_disarmed_external_threat_triggers() -> None:
    """Disarmed alarm with unknown person on outdoor camera should trigger."""
    snapshot = _base_snapshot()
    snapshot["entities"] = [_alarm_entity("disarmed")]
    snapshot["camera_activity"] = [
        _camera_activity(
            "camera.backyard",
            snapshot_summary="Unknown person in backyard.",
        )
    ]
    findings = AlarmDisarmedDuringExternalThreatRule().evaluate(snapshot)
    assert len(findings) == 1
    assert findings[0].type == "alarm_disarmed_during_external_threat"
    assert findings[0].confidence == 0.9
    assert findings[0].evidence["alarm_state"] == "disarmed"
    assert findings[0].suggested_actions == ["arm_alarm"]


def test_alarm_disarmed_external_threat_no_trigger_when_armed() -> None:
    """No finding when alarm is armed."""
    snapshot = _base_snapshot()
    snapshot["entities"] = [_alarm_entity("armed_away")]
    snapshot["camera_activity"] = [
        _camera_activity(
            "camera.backyard",
            snapshot_summary="Unknown person in backyard.",
        )
    ]
    findings = AlarmDisarmedDuringExternalThreatRule().evaluate(snapshot)
    assert len(findings) == 0


def test_alarm_disarmed_external_threat_no_trigger_when_recognized() -> None:
    """No finding when detected person is recognized."""
    snapshot = _base_snapshot()
    snapshot["entities"] = [_alarm_entity("disarmed")]
    snapshot["camera_activity"] = [
        _camera_activity(
            "camera.backyard",
            snapshot_summary="Lindo in backyard.",
            recognized_people=["person.lindo_st_angel"],
        )
    ]
    findings = AlarmDisarmedDuringExternalThreatRule().evaluate(snapshot)
    assert len(findings) == 0


def test_alarm_disarmed_external_threat_no_trigger_without_activity() -> None:
    """No finding when camera has no activity."""
    snapshot = _base_snapshot()
    snapshot["entities"] = [_alarm_entity("disarmed")]
    snapshot["camera_activity"] = [
        _camera_activity(
            "camera.backyard",
            snapshot_summary=None,
            last_activity=None,
            motion_entities=[],
            vmd_entities=[],
        )
    ]
    findings = AlarmDisarmedDuringExternalThreatRule().evaluate(snapshot)
    assert len(findings) == 0


def test_alarm_disarmed_external_threat_no_trigger_without_alarm_entity() -> None:
    """No finding when alarm entity is not in snapshot."""
    snapshot = _base_snapshot()
    snapshot["entities"] = []
    snapshot["camera_activity"] = [
        _camera_activity(
            "camera.backyard",
            snapshot_summary="Unknown person detected.",
        )
    ]
    findings = AlarmDisarmedDuringExternalThreatRule().evaluate(snapshot)
    assert len(findings) == 0


def test_alarm_disarmed_non_standard_entity_name_triggers() -> None:
    """Alarm panel with any entity_id triggers when disarmed — not just home_alarm."""
    snapshot = _base_snapshot()
    snapshot["entities"] = [
        SnapshotEntity(
            entity_id="alarm_control_panel.my_security_system",
            domain="alarm_control_panel",
            state="disarmed",
            friendly_name="My Security System",
            area=None,
            attributes={},
            last_changed="2025-01-01T00:00:00+00:00",
            last_updated="2025-01-01T00:00:00+00:00",
        )
    ]
    snapshot["camera_activity"] = [
        _camera_activity(
            "camera.front_porch",
            snapshot_summary="Unknown person at front porch.",
        )
    ]
    findings = AlarmDisarmedDuringExternalThreatRule().evaluate(snapshot)
    assert len(findings) == 1
    assert (
        findings[0].evidence["alarm_entity_id"]
        == "alarm_control_panel.my_security_system"
    )
    assert findings[0].evidence["alarm_entity_ids"] == [
        "alarm_control_panel.my_security_system"
    ]


def test_alarm_disarmed_multiple_panels_one_disarmed_triggers() -> None:
    """When multiple panels exist and one is disarmed, rule fires for active camera."""
    snapshot = _base_snapshot()
    snapshot["entities"] = [
        SnapshotEntity(
            entity_id="alarm_control_panel.main",
            domain="alarm_control_panel",
            state="armed_away",
            friendly_name="Main Alarm",
            area=None,
            attributes={},
            last_changed="2025-01-01T00:00:00+00:00",
            last_updated="2025-01-01T00:00:00+00:00",
        ),
        SnapshotEntity(
            entity_id="alarm_control_panel.garage",
            domain="alarm_control_panel",
            state="disarmed",
            friendly_name="Garage Alarm",
            area=None,
            attributes={},
            last_changed="2025-01-01T00:00:00+00:00",
            last_updated="2025-01-01T00:00:00+00:00",
        ),
    ]
    snapshot["camera_activity"] = [
        _camera_activity(
            "camera.driveway",
            snapshot_summary="Unknown person near driveway.",
        )
    ]
    findings = AlarmDisarmedDuringExternalThreatRule().evaluate(snapshot)
    assert len(findings) == 1
    assert findings[0].evidence["alarm_entity_id"] == "alarm_control_panel.garage"
    assert "alarm_control_panel.garage" in findings[0].evidence["alarm_entity_ids"]


def test_alarm_disarmed_zero_alarm_panels_no_findings() -> None:
    """No alarm_control_panel entities in snapshot → no findings."""
    snapshot = _base_snapshot()
    snapshot["entities"] = []
    snapshot["camera_activity"] = [
        _camera_activity(
            "camera.front_porch",
            snapshot_summary="Unknown person detected.",
        )
    ]
    findings = AlarmDisarmedDuringExternalThreatRule().evaluate(snapshot)
    assert len(findings) == 0


# ---------------------------------------------------------------------------
# PhoneBatteryLowAtNightRule
# ---------------------------------------------------------------------------


def _phone_battery_entity(
    entity_id: str = "sensor.lindos_iphone_battery_level",
    state: str = "15",
    friendly_name: str = "Lindo's iPhone Battery Level",
) -> SnapshotEntity:
    return SnapshotEntity(
        entity_id=entity_id,
        domain="sensor",
        state=state,
        friendly_name=friendly_name,
        area="Bedroom",
        attributes={"device_class": "battery"},
        last_changed="2025-01-01T00:00:00+00:00",
        last_updated="2025-01-01T00:00:00+00:00",
    )


def test_phone_battery_low_triggers() -> None:
    """Phone sensor with device_class=battery, state 15, night + home → 1 finding."""
    snapshot = _base_snapshot()
    snapshot["derived"]["is_night"] = True
    snapshot["derived"]["anyone_home"] = True
    snapshot["entities"] = [_phone_battery_entity()]
    findings = PhoneBatteryLowAtNightRule().evaluate(snapshot)
    assert len(findings) == 1
    assert findings[0].type == "phone_battery_low_at_night_home"
    assert findings[0].severity == "low"
    assert findings[0].confidence == 0.7
    assert findings[0].suggested_actions == ["charge_device"]
    assert findings[0].is_sensitive is False
    assert findings[0].evidence["battery_level"] == 15.0
    assert findings[0].triggering_entities == ["sensor.lindos_iphone_battery_level"]


def test_phone_battery_low_two_phones_two_findings() -> None:
    """Two qualifying phone battery sensors → 2 findings."""
    snapshot = _base_snapshot()
    snapshot["derived"]["is_night"] = True
    snapshot["derived"]["anyone_home"] = True
    snapshot["entities"] = [
        _phone_battery_entity(
            "sensor.alice_iphone_battery_level", "10", "Alice's iPhone Battery"
        ),
        _phone_battery_entity(
            "sensor.bob_pixel_battery_level", "5", "Bob's Pixel Battery"
        ),
    ]
    findings = PhoneBatteryLowAtNightRule().evaluate(snapshot)
    assert len(findings) == 2
    entity_ids = {f.triggering_entities[0] for f in findings}
    assert entity_ids == {
        "sensor.alice_iphone_battery_level",
        "sensor.bob_pixel_battery_level",
    }


def test_phone_battery_low_no_trigger_during_day() -> None:
    """No finding during the day even if battery is low."""
    snapshot = _base_snapshot()
    snapshot["derived"]["is_night"] = False
    snapshot["derived"]["anyone_home"] = True
    snapshot["entities"] = [_phone_battery_entity()]
    findings = PhoneBatteryLowAtNightRule().evaluate(snapshot)
    assert len(findings) == 0


def test_phone_battery_low_no_trigger_when_away() -> None:
    """No finding when no one is home."""
    snapshot = _base_snapshot()
    snapshot["derived"]["is_night"] = True
    snapshot["derived"]["anyone_home"] = False
    snapshot["entities"] = [_phone_battery_entity()]
    findings = PhoneBatteryLowAtNightRule().evaluate(snapshot)
    assert len(findings) == 0


def test_phone_battery_low_no_trigger_above_threshold() -> None:
    """No finding when battery is at or above 20%."""
    snapshot = _base_snapshot()
    snapshot["derived"]["is_night"] = True
    snapshot["derived"]["anyone_home"] = True
    snapshot["entities"] = [_phone_battery_entity(state="50")]
    findings = PhoneBatteryLowAtNightRule().evaluate(snapshot)
    assert len(findings) == 0


def test_phone_battery_low_no_trigger_at_threshold() -> None:
    """No finding when battery is exactly 20%."""
    snapshot = _base_snapshot()
    snapshot["derived"]["is_night"] = True
    snapshot["derived"]["anyone_home"] = True
    snapshot["entities"] = [_phone_battery_entity(state="20")]
    findings = PhoneBatteryLowAtNightRule().evaluate(snapshot)
    assert len(findings) == 0


def test_phone_battery_low_no_trigger_non_phone_battery() -> None:
    """Battery sensor without a phone keyword (door sensor battery) should not fire."""
    snapshot = _base_snapshot()
    snapshot["derived"]["is_night"] = True
    snapshot["derived"]["anyone_home"] = True
    snapshot["entities"] = [
        SnapshotEntity(
            entity_id="sensor.door_sensor_battery",
            domain="sensor",
            state="10",
            friendly_name="Door Sensor Battery",
            area="Front",
            attributes={"device_class": "battery"},
            last_changed="2025-01-01T00:00:00+00:00",
            last_updated="2025-01-01T00:00:00+00:00",
        )
    ]
    findings = PhoneBatteryLowAtNightRule().evaluate(snapshot)
    assert len(findings) == 0


def test_phone_battery_low_no_trigger_unavailable() -> None:
    """Unavailable state should not raise and should produce no finding."""
    snapshot = _base_snapshot()
    snapshot["derived"]["is_night"] = True
    snapshot["derived"]["anyone_home"] = True
    snapshot["entities"] = [_phone_battery_entity(state="unavailable")]
    findings = PhoneBatteryLowAtNightRule().evaluate(snapshot)
    assert len(findings) == 0


def test_phone_battery_low_no_trigger_missing_device_class() -> None:
    """Phone-named sensor without device_class=battery should not fire."""
    snapshot = _base_snapshot()
    snapshot["derived"]["is_night"] = True
    snapshot["derived"]["anyone_home"] = True
    snapshot["entities"] = [
        SnapshotEntity(
            entity_id="sensor.iphone_battery_level",
            domain="sensor",
            state="10",
            friendly_name="iPhone Battery Level",
            area="Bedroom",
            attributes={},
            last_changed="2025-01-01T00:00:00+00:00",
            last_updated="2025-01-01T00:00:00+00:00",
        )
    ]
    findings = PhoneBatteryLowAtNightRule().evaluate(snapshot)
    assert len(findings) == 0


def test_system_prompt_camera_entry_cooccurrence_grounding() -> None:
    """
    SYSTEM_PROMPT contains co-occurrence grounding for camera_entry_unsecured.

    Regression guard: ensures the spatial grounding instruction is never
    accidentally removed during future SYSTEM_PROMPT edits.
    """
    assert "camera_entry_unsecured" in SYSTEM_PROMPT
    assert "co-occurrence" in SYSTEM_PROMPT
    assert "camera area proximity does not imply" in SYSTEM_PROMPT.lower()


# ---------------------------------------------------------------------------
# Presence-aware lock severity
# ---------------------------------------------------------------------------


def _lock_entity(entity_id: str = "lock.front_door") -> SnapshotEntity:
    return SnapshotEntity(
        entity_id=entity_id,
        domain="lock",
        state="unlocked",
        friendly_name="Front Door",
        area="front",
        attributes={},
        last_changed="2025-01-01T00:00:00+00:00",
        last_updated="2025-01-01T00:00:00+00:00",
    )


def test_unlocked_lock_at_night_someone_home_is_low_severity() -> None:
    """anyone_home=True → severity=='low', is_sensitive==True, evidence['anyone_home']==True."""
    snapshot = _base_snapshot()
    snapshot["derived"]["is_night"] = True
    snapshot["derived"]["anyone_home"] = True
    snapshot["entities"] = [_lock_entity()]

    findings = UnlockedLockAtNightRule().evaluate(snapshot)

    assert len(findings) == 1
    f = findings[0]
    assert f.severity == "low"
    assert f.is_sensitive is True  # lock location is always sensitive
    assert f.evidence["anyone_home"] is True


def test_unlocked_lock_at_night_no_one_home_is_high_severity() -> None:
    """anyone_home=False → severity=='high', is_sensitive==True."""
    snapshot = _base_snapshot()
    snapshot["derived"]["is_night"] = True
    snapshot["derived"]["anyone_home"] = False
    snapshot["entities"] = [_lock_entity()]

    findings = UnlockedLockAtNightRule().evaluate(snapshot)

    assert len(findings) == 1
    f = findings[0]
    assert f.severity == "high"
    assert f.is_sensitive is True

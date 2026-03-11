# ruff: noqa: S101
"""Tests for sentinel rules."""

from __future__ import annotations

from typing import Any

from custom_components.home_generative_agent.sentinel.dynamic_rules import (
    evaluate_dynamic_rule,
)
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

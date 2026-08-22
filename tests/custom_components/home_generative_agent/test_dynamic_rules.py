# ruff: noqa: S101
"""Tests for dynamic rules evaluation and registry."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import pytest

from custom_components.home_generative_agent.sentinel.dynamic_rules import (
    evaluate_dynamic_rules,
)
from custom_components.home_generative_agent.sentinel.rule_registry import RuleRegistry
from custom_components.home_generative_agent.snapshot.schema import (
    CameraActivity,
    FullStateSnapshot,
    SnapshotEntity,
    validate_snapshot,
)

if TYPE_CHECKING:
    from homeassistant.core import HomeAssistant


def _base_entity(entity_id: str, domain: str, state: str) -> SnapshotEntity:
    return {
        "entity_id": entity_id,
        "domain": domain,
        "state": state,
        "friendly_name": entity_id,
        "area": None,
        "attributes": {},
        "last_changed": "2026-02-01T00:00:00+00:00",
        "last_updated": "2026-02-01T00:00:00+00:00",
    }


def _snapshot(
    entities: list[SnapshotEntity],
    camera_activity: list[CameraActivity],
    derived: dict[str, object],
) -> FullStateSnapshot:
    snapshot = {
        "schema_version": 1,
        "generated_at": "2026-02-01T00:00:00+00:00",
        "entities": entities,
        "camera_activity": camera_activity,
        "derived": derived,
    }
    return validate_snapshot(snapshot)


def test_dynamic_rule_alarm_disarmed_open_entry() -> None:
    snapshot = _snapshot(
        [
            _base_entity(
                "alarm_control_panel.home_alarm", "alarm_control_panel", "disarmed"
            ),
            _base_entity("binary_sensor.front_door", "binary_sensor", "on"),
        ],
        [],
        {
            "now": "2026-02-01T00:00:00+00:00",
            "timezone": "UTC",
            "is_night": False,
            "anyone_home": True,
            "people_home": [],
            "people_away": [],
            "last_motion_by_area": {},
        },
    )
    rules = [
        {
            "rule_id": "alarm_rule_1",
            "template_id": "alarm_disarmed_open_entry",
            "params": {
                "alarm_entity_id": "alarm_control_panel.home_alarm",
                "entry_entity_ids": ["binary_sensor.front_door"],
            },
            "severity": "high",
            "confidence": 0.6,
            "is_sensitive": True,
            "suggested_actions": ["close_entry"],
        }
    ]
    findings = evaluate_dynamic_rules(snapshot, rules)
    assert len(findings) == 1
    assert findings[0].type == "alarm_rule_1"
    assert "binary_sensor.front_door" in findings[0].triggering_entities


def test_dynamic_rule_unlocked_lock_when_home() -> None:
    snapshot = _snapshot(
        [
            _base_entity("lock.garage_door", "lock", "unlocked"),
        ],
        [],
        {
            "now": "2026-02-01T00:00:00+00:00",
            "timezone": "UTC",
            "is_night": False,
            "anyone_home": True,
            "people_home": [],
            "people_away": [],
            "last_motion_by_area": {},
        },
    )
    rules = [
        {
            "rule_id": "lock_rule_1",
            "template_id": "unlocked_lock_when_home",
            "params": {"lock_entity_id": "lock.garage_door"},
            "severity": "medium",
            "confidence": 0.5,
            "is_sensitive": True,
            "suggested_actions": ["lock_entity"],
        }
    ]
    findings = evaluate_dynamic_rules(snapshot, rules)
    assert len(findings) == 1
    assert findings[0].type == "lock_rule_1"


def test_dynamic_rule_motion_without_camera_activity() -> None:
    snapshot = _snapshot(
        [
            _base_entity("binary_sensor.front_motion", "binary_sensor", "on"),
        ],
        [
            {
                "camera_entity_id": "camera.front",
                "area": None,
                "last_activity": None,
                "motion_entities": [],
                "vmd_entities": [],
                "snapshot_summary": None,
                "recognized_people": [],
                "latest_path": None,
            }
        ],
        {
            "now": "2026-02-01T00:00:00+00:00",
            "timezone": "UTC",
            "is_night": False,
            "anyone_home": True,
            "people_home": [],
            "people_away": [],
            "last_motion_by_area": {},
        },
    )
    rules = [
        {
            "rule_id": "motion_rule_1",
            "template_id": "motion_without_camera_activity",
            "params": {
                "motion_entity_ids": ["binary_sensor.front_motion"],
                "camera_entity_id": "camera.front",
            },
            "severity": "low",
            "confidence": 0.4,
            "is_sensitive": False,
            "suggested_actions": ["check_camera"],
        }
    ]
    findings = evaluate_dynamic_rules(snapshot, rules)
    assert len(findings) == 1
    assert findings[0].type == "motion_rule_1"


def test_dynamic_rule_unknown_person_camera_when_home() -> None:
    snapshot = _snapshot(
        [],
        [
            {
                "camera_entity_id": "camera.backyard",
                "area": "Backyard",
                "last_activity": "2026-02-01T00:00:00+00:00",
                "motion_entities": ["binary_sensor.backyard_motion"],
                "vmd_entities": [],
                "snapshot_summary": None,
                "recognized_people": ["Unknown Person"],
                "latest_path": None,
            }
        ],
        {
            "now": "2026-02-01T00:00:00+00:00",
            "timezone": "UTC",
            "is_night": False,
            "anyone_home": True,
            "people_home": [],
            "people_away": [],
            "last_motion_by_area": {},
        },
    )
    rules = [
        {
            "rule_id": "unknown_person_camera_when_home",
            "template_id": "unknown_person_camera_when_home",
            "params": {"camera_entity_id": "camera.backyard"},
            "severity": "low",
            "confidence": 0.7,
            "is_sensitive": False,
            "suggested_actions": ["close_entry"],
        }
    ]
    findings = evaluate_dynamic_rules(snapshot, rules)
    assert len(findings) == 1
    assert findings[0].type == "unknown_person_camera_when_home"
    assert findings[0].evidence["anyone_home"] is True


def test_dynamic_rule_unknown_person_camera_when_home_no_trigger_no_stranger() -> None:
    """Reserved labels alone (no 'Unknown Person') must not fire the rule."""
    snapshot = _snapshot(
        [],
        [
            {
                "camera_entity_id": "camera.backyard",
                "area": "Backyard",
                "last_activity": "2026-02-01T00:00:00+00:00",
                "motion_entities": ["binary_sensor.backyard_motion"],
                "vmd_entities": [],
                "snapshot_summary": None,
                "recognized_people": ["Indeterminate"],
                "latest_path": None,
            }
        ],
        {
            "now": "2026-02-01T00:00:00+00:00",
            "timezone": "UTC",
            "is_night": False,
            "anyone_home": True,
            "people_home": [],
            "people_away": [],
            "last_motion_by_area": {},
        },
    )
    rules = [
        {
            "rule_id": "unknown_person_camera_when_home",
            "template_id": "unknown_person_camera_when_home",
            "params": {"camera_entity_id": "camera.backyard"},
            "severity": "low",
            "confidence": 0.7,
            "is_sensitive": False,
            "suggested_actions": ["close_entry"],
        }
    ]
    findings = evaluate_dynamic_rules(snapshot, rules)
    assert findings == []


def test_dynamic_rule_unknown_person_naive_timestamp_does_not_raise() -> None:
    """
    A tz-naive last_activity from a third-party camera must not abort.

    Regression: naive minus aware raises TypeError, and evaluate_dynamic_rules
    has no per-rule exception boundary — the error would kill the Sentinel run
    loop until reload. The naive value is interpreted as local time instead.
    """
    snapshot = _snapshot(
        [],
        [
            {
                "camera_entity_id": "camera.frontgate",
                "area": "Front Gate",
                "last_activity": "2026-02-01T00:00:00",  # naive — passed verbatim
                "motion_entities": [],
                "vmd_entities": [],
                "snapshot_summary": None,
                "recognized_people": ["Unknown Person"],
                "latest_path": None,
            }
        ],
        {
            "now": "2026-02-01T00:00:00+00:00",
            "timezone": "UTC",
            "is_night": True,
            "anyone_home": False,
            "people_home": [],
            "people_away": [],
            "last_motion_by_area": {},
        },
    )
    rules = [
        {
            "rule_id": "unknown_person_camera_no_home",
            "template_id": "unknown_person_camera_no_home",
            "params": {"camera_selector": "any"},
            "severity": "low",
            "confidence": 0.85,
            "is_sensitive": True,
            "suggested_actions": ["close_entry"],
        }
    ]
    evaluate_dynamic_rules(snapshot, rules)  # must not raise


def test_dynamic_rule_unknown_person_camera_when_home_no_trigger_stale() -> None:
    """A sighting older than the staleness budget must not keep firing."""
    snapshot = _snapshot(
        [],
        [
            {
                "camera_entity_id": "camera.backyard",
                "area": "Backyard",
                "last_activity": "2026-01-31T22:00:00+00:00",
                "motion_entities": ["binary_sensor.backyard_motion"],
                "vmd_entities": [],
                "snapshot_summary": None,
                "recognized_people": ["Unknown Person"],
                "latest_path": None,
            }
        ],
        {
            "now": "2026-02-01T00:00:00+00:00",
            "timezone": "UTC",
            "is_night": False,
            "anyone_home": True,
            "people_home": [],
            "people_away": [],
            "last_motion_by_area": {},
        },
    )
    rules = [
        {
            "rule_id": "unknown_person_camera_when_home",
            "template_id": "unknown_person_camera_when_home",
            "params": {"camera_entity_id": "camera.backyard"},
            "severity": "low",
            "confidence": 0.7,
            "is_sensitive": False,
            "suggested_actions": ["close_entry"],
        }
    ]
    findings = evaluate_dynamic_rules(snapshot, rules)
    assert findings == []


def test_dynamic_rule_unknown_person_camera_when_home_no_trigger_when_away() -> None:
    snapshot = _snapshot(
        [],
        [
            {
                "camera_entity_id": "camera.backyard",
                "area": "Backyard",
                "last_activity": "2026-02-01T00:00:00+00:00",
                "motion_entities": ["binary_sensor.backyard_motion"],
                "vmd_entities": [],
                "snapshot_summary": None,
                "recognized_people": ["Unknown Person"],
                "latest_path": None,
            }
        ],
        {
            "now": "2026-02-01T00:00:00+00:00",
            "timezone": "UTC",
            "is_night": False,
            "anyone_home": False,
            "people_home": [],
            "people_away": [],
            "last_motion_by_area": {},
        },
    )
    rules = [
        {
            "rule_id": "unknown_person_camera_when_home",
            "template_id": "unknown_person_camera_when_home",
            "params": {"camera_entity_id": "camera.backyard"},
            "severity": "low",
            "confidence": 0.7,
            "is_sensitive": False,
            "suggested_actions": ["close_entry"],
        }
    ]
    findings = evaluate_dynamic_rules(snapshot, rules)
    assert findings == []


def test_dynamic_rule_unknown_person_camera_when_home_any_camera_selector() -> None:
    snapshot = _snapshot(
        [],
        [
            {
                "camera_entity_id": "camera.backyard",
                "area": "Backyard",
                "last_activity": "2026-02-01T00:00:00+00:00",
                "motion_entities": ["binary_sensor.backyard_motion"],
                "vmd_entities": [],
                "snapshot_summary": None,
                "recognized_people": ["Unknown Person"],
                "latest_path": None,
            },
            {
                "camera_entity_id": "camera.frontporch",
                "area": "Front Porch",
                "last_activity": "2026-02-01T00:00:00+00:00",
                "motion_entities": ["binary_sensor.frontporch_motion"],
                "vmd_entities": [],
                "snapshot_summary": None,
                "recognized_people": ["resident"],
                "latest_path": None,
            },
        ],
        {
            "now": "2026-02-01T00:00:00+00:00",
            "timezone": "UTC",
            "is_night": False,
            "anyone_home": True,
            "people_home": [],
            "people_away": [],
            "last_motion_by_area": {},
        },
    )
    rules = [
        {
            "rule_id": "unknown_person_camera_when_home_any_camera",
            "template_id": "unknown_person_camera_when_home",
            "params": {"camera_selector": "any"},
            "severity": "low",
            "confidence": 0.7,
            "is_sensitive": False,
            "suggested_actions": ["close_entry"],
        }
    ]
    findings = evaluate_dynamic_rules(snapshot, rules)
    assert len(findings) == 1
    assert findings[0].type == "unknown_person_camera_when_home_any_camera"
    assert findings[0].evidence["camera_entity_id"] == "camera.backyard"


def test_dynamic_rule_unknown_person_camera_no_home_any_camera_selector() -> None:
    snapshot = _snapshot(
        [],
        [
            {
                "camera_entity_id": "camera.frontgate",
                "area": "Front Gate",
                "last_activity": "2026-02-01T00:00:00+00:00",
                "motion_entities": ["binary_sensor.frontgate_motion"],
                "vmd_entities": [],
                "snapshot_summary": None,
                "recognized_people": ["Unknown Person"],
                "latest_path": None,
            }
        ],
        {
            "now": "2026-02-01T00:00:00+00:00",
            "timezone": "UTC",
            "is_night": True,
            "anyone_home": False,
            "people_home": [],
            "people_away": [],
            "last_motion_by_area": {},
        },
    )
    rules = [
        {
            "rule_id": "unknown_person_camera_no_home_any_camera",
            "template_id": "unknown_person_camera_no_home",
            "params": {"camera_selector": "any"},
            "severity": "low",
            "confidence": 0.85,
            "is_sensitive": True,
            "suggested_actions": ["close_entry"],
        }
    ]
    findings = evaluate_dynamic_rules(snapshot, rules)
    assert len(findings) == 1
    assert findings[0].type == "unknown_person_camera_no_home_any_camera"
    assert findings[0].evidence["camera_entity_id"] == "camera.frontgate"


def test_dynamic_rule_unknown_person_camera_when_home_no_trigger_accompanied() -> None:
    """A stranger alongside an enrolled person is a companion, not an intrusion."""
    snapshot = _snapshot(
        [],
        [
            {
                "camera_entity_id": "camera.backyard",
                "area": "Backyard",
                "last_activity": "2026-02-01T00:00:00+00:00",
                "motion_entities": ["binary_sensor.backyard_motion"],
                "vmd_entities": [],
                "snapshot_summary": None,
                "recognized_people": ["Lindo", "Unknown Person"],
                "latest_path": None,
            }
        ],
        {
            "now": "2026-02-01T00:00:00+00:00",
            "timezone": "UTC",
            "is_night": False,
            "anyone_home": True,
            "people_home": [],
            "people_away": [],
            "last_motion_by_area": {},
        },
    )
    rules = [
        {
            "rule_id": "unknown_person_camera_when_home",
            "template_id": "unknown_person_camera_when_home",
            "params": {"camera_entity_id": "camera.backyard"},
            "severity": "low",
            "confidence": 0.7,
            "is_sensitive": False,
            "suggested_actions": ["close_entry"],
        }
    ]
    findings = evaluate_dynamic_rules(snapshot, rules)
    assert findings == []


def test_dynamic_rule_unknown_person_camera_when_home_no_trigger_no_timestamp() -> None:
    """No last_activity means freshness cannot be proven — skip."""
    snapshot = _snapshot(
        [],
        [
            {
                "camera_entity_id": "camera.backyard",
                "area": "Backyard",
                "last_activity": None,
                "motion_entities": ["binary_sensor.backyard_motion"],
                "vmd_entities": [],
                "snapshot_summary": None,
                "recognized_people": ["Unknown Person"],
                "latest_path": None,
            }
        ],
        {
            "now": "2026-02-01T00:00:00+00:00",
            "timezone": "UTC",
            "is_night": False,
            "anyone_home": True,
            "people_home": [],
            "people_away": [],
            "last_motion_by_area": {},
        },
    )
    rules = [
        {
            "rule_id": "unknown_person_camera_when_home",
            "template_id": "unknown_person_camera_when_home",
            "params": {"camera_entity_id": "camera.backyard"},
            "severity": "low",
            "confidence": 0.7,
            "is_sensitive": False,
            "suggested_actions": ["close_entry"],
        }
    ]
    findings = evaluate_dynamic_rules(snapshot, rules)
    assert findings == []


def test_dynamic_rule_unknown_person_camera_no_home_no_trigger_when_home() -> None:
    """The away-only template must not fire while anyone is home."""
    snapshot = _snapshot(
        [],
        [
            {
                "camera_entity_id": "camera.frontgate",
                "area": "Front Gate",
                "last_activity": "2026-02-01T00:00:00+00:00",
                "motion_entities": ["binary_sensor.frontgate_motion"],
                "vmd_entities": [],
                "snapshot_summary": None,
                "recognized_people": ["Unknown Person"],
                "latest_path": None,
            }
        ],
        {
            "now": "2026-02-01T00:00:00+00:00",
            "timezone": "UTC",
            "is_night": True,
            "anyone_home": True,
            "people_home": [],
            "people_away": [],
            "last_motion_by_area": {},
        },
    )
    rules = [
        {
            "rule_id": "unknown_person_camera_no_home",
            "template_id": "unknown_person_camera_no_home",
            "params": {"camera_entity_id": "camera.frontgate"},
            "severity": "low",
            "confidence": 0.85,
            "is_sensitive": True,
            "suggested_actions": ["close_entry"],
        }
    ]
    findings = evaluate_dynamic_rules(snapshot, rules)
    assert findings == []


def test_dynamic_rule_unknown_person_camera_no_home_specific_camera_triggers() -> None:
    """The no_home template fires for an explicitly targeted camera."""
    snapshot = _snapshot(
        [],
        [
            {
                "camera_entity_id": "camera.frontgate",
                "area": "Front Gate",
                "last_activity": "2026-02-01T00:00:00+00:00",
                "motion_entities": [],
                "vmd_entities": [],
                "snapshot_summary": None,
                "recognized_people": ["Unknown Person"],
                "latest_path": None,
            }
        ],
        {
            "now": "2026-02-01T00:00:00+00:00",
            "timezone": "UTC",
            "is_night": True,
            "anyone_home": False,
            "people_home": [],
            "people_away": [],
            "last_motion_by_area": {},
        },
    )
    rules = [
        {
            "rule_id": "unknown_person_camera_no_home",
            "template_id": "unknown_person_camera_no_home",
            "params": {"camera_entity_id": "camera.frontgate"},
            "severity": "low",
            "confidence": 0.85,
            "is_sensitive": True,
            "suggested_actions": ["close_entry"],
        }
    ]
    findings = evaluate_dynamic_rules(snapshot, rules)
    assert len(findings) == 1
    assert findings[0].evidence["camera_entity_id"] == "camera.frontgate"


def test_dynamic_rule_unknown_person_camera_no_home_any_selector_mixed() -> None:
    """With 'any' selector, only the actionable sighting fires among mixed cameras."""
    snapshot = _snapshot(
        [],
        [
            {
                "camera_entity_id": "camera.backyard",
                "area": "Backyard",
                "last_activity": "2026-02-01T00:00:00+00:00",
                "motion_entities": [],
                "vmd_entities": [],
                "snapshot_summary": None,
                "recognized_people": ["Lindo", "Unknown Person"],
                "latest_path": None,
            },
            {
                "camera_entity_id": "camera.frontgate",
                "area": "Front Gate",
                "last_activity": "2026-02-01T00:00:00+00:00",
                "motion_entities": [],
                "vmd_entities": [],
                "snapshot_summary": None,
                "recognized_people": ["Unknown Person"],
                "latest_path": None,
            },
            {
                "camera_entity_id": "camera.driveway",
                "area": "Driveway",
                "last_activity": "2026-01-31T22:00:00+00:00",
                "motion_entities": [],
                "vmd_entities": [],
                "snapshot_summary": None,
                "recognized_people": ["Unknown Person"],
                "latest_path": None,
            },
        ],
        {
            "now": "2026-02-01T00:00:00+00:00",
            "timezone": "UTC",
            "is_night": True,
            "anyone_home": False,
            "people_home": [],
            "people_away": [],
            "last_motion_by_area": {},
        },
    )
    rules = [
        {
            "rule_id": "unknown_person_camera_no_home_any_camera",
            "template_id": "unknown_person_camera_no_home",
            "params": {"camera_selector": "any"},
            "severity": "low",
            "confidence": 0.85,
            "is_sensitive": True,
            "suggested_actions": ["close_entry"],
        }
    ]
    findings = evaluate_dynamic_rules(snapshot, rules)
    assert len(findings) == 1
    assert findings[0].evidence["camera_entity_id"] == "camera.frontgate"


def test_dynamic_rule_open_entry_at_night_when_home() -> None:
    snapshot = _snapshot(
        [
            _base_entity("binary_sensor.playroom_window", "binary_sensor", "on"),
        ],
        [],
        {
            "now": "2026-02-01T00:00:00+00:00",
            "timezone": "UTC",
            "is_night": True,
            "anyone_home": True,
            "people_home": [],
            "people_away": [],
            "last_motion_by_area": {},
        },
    )
    rules = [
        {
            "rule_id": "entry_rule_1",
            "template_id": "open_entry_at_night_when_home",
            "params": {"entry_entity_ids": ["binary_sensor.playroom_window"]},
            "severity": "medium",
            "confidence": 0.7,
            "is_sensitive": True,
            "suggested_actions": ["close_entry"],
        }
    ]
    findings = evaluate_dynamic_rules(snapshot, rules)
    assert len(findings) == 1
    assert findings[0].type == "entry_rule_1"


def test_dynamic_rule_open_any_window_at_night_while_away() -> None:
    snapshot = _snapshot(
        [
            _base_entity("binary_sensor.playroom_window", "binary_sensor", "on"),
            _base_entity("binary_sensor.front_door", "binary_sensor", "off"),
        ],
        [],
        {
            "now": "2026-02-01T00:00:00+00:00",
            "timezone": "UTC",
            "is_night": True,
            "anyone_home": False,
            "people_home": [],
            "people_away": [],
            "last_motion_by_area": {},
        },
    )
    snapshot["entities"][0]["attributes"]["device_class"] = "window"
    rules = [
        {
            "rule_id": "entry_rule_2",
            "template_id": "open_any_window_at_night_while_away",
            "params": {"entry_selector": "window"},
            "severity": "high",
            "confidence": 0.7,
            "is_sensitive": True,
            "suggested_actions": ["close_entry"],
        }
    ]
    findings = evaluate_dynamic_rules(snapshot, rules)
    assert len(findings) == 1
    assert findings[0].type == "entry_rule_2"


def test_dynamic_rule_unavailable_sensors_while_home() -> None:
    snapshot = _snapshot(
        [
            _base_entity("sensor.backyard_vmd3_0", "sensor", "unavailable"),
            _base_entity("sensor.backyard_vmd4_camera1profile1", "sensor", "idle"),
        ],
        [],
        {
            "now": "2026-02-01T00:00:00+00:00",
            "timezone": "UTC",
            "is_night": False,
            "anyone_home": True,
            "people_home": [],
            "people_away": [],
            "last_motion_by_area": {},
        },
    )
    rules = [
        {
            "rule_id": "availability_rule_1",
            "template_id": "unavailable_sensors_while_home",
            "params": {
                "sensor_entity_ids": [
                    "sensor.backyard_vmd3_0",
                    "sensor.backyard_vmd4_camera1profile1",
                ]
            },
            "severity": "low",
            "confidence": 0.8,
            "is_sensitive": False,
            "suggested_actions": ["check_sensor"],
        }
    ]
    findings = evaluate_dynamic_rules(snapshot, rules)
    assert len(findings) == 1
    assert findings[0].type == "availability_rule_1"
    assert findings[0].triggering_entities == ["sensor.backyard_vmd3_0"]


def test_dynamic_rule_unavailable_sensors_while_home_missing_required_entity() -> None:
    snapshot = _snapshot(
        [
            _base_entity("sensor.backyard_vmd3_0", "sensor", "unavailable"),
        ],
        [],
        {
            "now": "2026-02-01T00:00:00+00:00",
            "timezone": "UTC",
            "is_night": False,
            "anyone_home": True,
            "people_home": [],
            "people_away": [],
            "last_motion_by_area": {},
        },
    )
    rules = [
        {
            "rule_id": "availability_rule_2",
            "template_id": "unavailable_sensors_while_home",
            "params": {
                "sensor_entity_ids": [
                    "sensor.backyard_vmd3_0",
                    "sensor.backyard_vmd4_camera1profile1",
                ]
            },
            "severity": "low",
            "confidence": 0.8,
            "is_sensitive": False,
            "suggested_actions": ["check_sensor"],
        }
    ]
    findings = evaluate_dynamic_rules(snapshot, rules)
    assert findings == []


def test_dynamic_rule_unavailable_sensors_while_home_legacy_entity_ids() -> None:
    snapshot = _snapshot(
        [
            _base_entity("sensor.backyard_vmd3_0", "sensor", "unavailable"),
            _base_entity("sensor.backyard_vmd4_camera1profile1", "sensor", "idle"),
        ],
        [],
        {
            "now": "2026-02-01T00:00:00+00:00",
            "timezone": "UTC",
            "is_night": False,
            "anyone_home": True,
            "people_home": [],
            "people_away": [],
            "last_motion_by_area": {},
        },
    )
    rules = [
        {
            "rule_id": "availability_rule_legacy",
            "template_id": "unavailable_sensors_while_home",
            "params": {
                "sensor_entity_ids": [
                    "backyard_vmd3_0",
                    "backyard_vmd4_camera1profile1",
                ]
            },
            "severity": "low",
            "confidence": 0.8,
            "is_sensitive": False,
            "suggested_actions": ["check_sensor"],
        }
    ]
    findings = evaluate_dynamic_rules(snapshot, rules)
    assert len(findings) == 1
    assert findings[0].type == "availability_rule_legacy"
    assert findings[0].triggering_entities == ["sensor.backyard_vmd3_0"]


def test_dynamic_rule_unavailable_sensors_issue_223_triggers() -> None:
    snapshot = _snapshot(
        [
            _base_entity("sensor.backyard_vmd3_0", "sensor", "unavailable"),
            _base_entity(
                "sensor.backyard_vmd4_camera1profile1", "sensor", "unavailable"
            ),
        ],
        [],
        {
            "now": "2026-02-01T00:00:00+00:00",
            "timezone": "UTC",
            "is_night": False,
            "anyone_home": False,
            "people_home": [],
            "people_away": [],
            "last_motion_by_area": {},
        },
    )
    rules = [
        {
            "rule_id": "backyard_sensors_unavailable",
            "template_id": "unavailable_sensors",
            "params": {
                "sensor_entity_ids": [
                    "backyard_vmd3_0",
                    "backyard_vmd4_camera1profile1",
                ]
            },
            "severity": "low",
            "confidence": 0.6,
            "is_sensitive": False,
            "suggested_actions": ["close_entry"],
        }
    ]
    findings = evaluate_dynamic_rules(snapshot, rules)
    assert len(findings) == 1
    assert findings[0].type == "backyard_sensors_unavailable"
    assert findings[0].triggering_entities == [
        "sensor.backyard_vmd3_0",
        "sensor.backyard_vmd4_camera1profile1",
    ]


def test_dynamic_rule_unavailable_sensors_issue_223_non_trigger() -> None:
    snapshot = _snapshot(
        [
            _base_entity("sensor.backyard_vmd3_0", "sensor", "unavailable"),
            _base_entity("sensor.backyard_vmd4_camera1profile1", "sensor", "idle"),
        ],
        [],
        {
            "now": "2026-02-01T00:00:00+00:00",
            "timezone": "UTC",
            "is_night": False,
            "anyone_home": False,
            "people_home": [],
            "people_away": [],
            "last_motion_by_area": {},
        },
    )
    rules = [
        {
            "rule_id": "backyard_sensors_unavailable",
            "template_id": "unavailable_sensors",
            "params": {
                "sensor_entity_ids": [
                    "backyard_vmd3_0",
                    "backyard_vmd4_camera1profile1",
                ]
            },
            "severity": "low",
            "confidence": 0.6,
            "is_sensitive": False,
            "suggested_actions": ["close_entry"],
        }
    ]
    findings = evaluate_dynamic_rules(snapshot, rules)
    assert findings == []


def test_dynamic_rule_unavailable_binary_occupancy_sensors_issue_514_triggers() -> None:
    snapshot = _snapshot(
        [
            _base_entity(
                "binary_sensor.0x00124b0010b0a987_occupancy",
                "binary_sensor",
                "unavailable",
            ),
            _base_entity(
                "binary_sensor.smart_presence_sensor_obsazenost",
                "binary_sensor",
                "unavailable",
            ),
        ],
        [],
        {
            "now": "2026-02-01T00:00:00+00:00",
            "timezone": "UTC",
            "is_night": False,
            "anyone_home": False,
            "people_home": [],
            "people_away": [],
            "last_motion_by_area": {},
        },
    )
    rules = [
        {
            "rule_id": "multiple_occupancy_sensors_unavailable",
            "template_id": "unavailable_sensors",
            "params": {
                "sensor_entity_ids": [
                    "binary_sensor.0x00124b0010b0a987_occupancy",
                    "binary_sensor.smart_presence_sensor_obsazenost",
                ]
            },
            "severity": "low",
            "confidence": 0.3,
            "is_sensitive": False,
            "suggested_actions": ["check_sensor"],
        }
    ]
    findings = evaluate_dynamic_rules(snapshot, rules)
    assert len(findings) == 1
    assert findings[0].type == "multiple_occupancy_sensors_unavailable"
    assert findings[0].triggering_entities == [
        "binary_sensor.0x00124b0010b0a987_occupancy",
        "binary_sensor.smart_presence_sensor_obsazenost",
    ]


def test_dynamic_rule_unavailable_binary_occupancy_sensors_issue_514_non_trigger() -> (
    None
):
    snapshot = _snapshot(
        [
            _base_entity(
                "binary_sensor.0x00124b0010b0a987_occupancy",
                "binary_sensor",
                "unavailable",
            ),
            _base_entity(
                "binary_sensor.smart_presence_sensor_obsazenost",
                "binary_sensor",
                "off",
            ),
        ],
        [],
        {
            "now": "2026-02-01T00:00:00+00:00",
            "timezone": "UTC",
            "is_night": False,
            "anyone_home": False,
            "people_home": [],
            "people_away": [],
            "last_motion_by_area": {},
        },
    )
    rules = [
        {
            "rule_id": "multiple_occupancy_sensors_unavailable",
            "template_id": "unavailable_sensors",
            "params": {
                "sensor_entity_ids": [
                    "binary_sensor.0x00124b0010b0a987_occupancy",
                    "binary_sensor.smart_presence_sensor_obsazenost",
                ]
            },
            "severity": "low",
            "confidence": 0.3,
            "is_sensitive": False,
            "suggested_actions": ["check_sensor"],
        }
    ]
    findings = evaluate_dynamic_rules(snapshot, rules)
    assert findings == []


def test_dynamic_rule_low_battery_sensors_issue_236_triggers() -> None:
    snapshot = _snapshot(
        [
            _base_entity("sensor.elias_t_h_battery", "sensor", "37"),
            _base_entity("sensor.girls_t_h_battery", "sensor", "53"),
        ],
        [],
        {
            "now": "2026-02-01T00:00:00+00:00",
            "timezone": "UTC",
            "is_night": False,
            "anyone_home": True,
            "people_home": [],
            "people_away": [],
            "last_motion_by_area": {},
        },
    )
    rules = [
        {
            "rule_id": "low_battery_room_sensors_v1",
            "template_id": "low_battery_sensors",
            "params": {
                "sensor_entity_ids": [
                    "sensor.elias_t_h_battery",
                    "sensor.girls_t_h_battery",
                ],
                "threshold": 40,
            },
            "severity": "low",
            "confidence": 0.62,
            "is_sensitive": False,
            "suggested_actions": ["check_sensor"],
        }
    ]
    findings = evaluate_dynamic_rules(snapshot, rules)
    assert len(findings) == 1
    assert findings[0].type == "low_battery_room_sensors_v1"
    assert findings[0].triggering_entities == ["sensor.elias_t_h_battery"]


def test_dynamic_rule_low_battery_sensors_issue_236_non_trigger() -> None:
    snapshot = _snapshot(
        [
            _base_entity("sensor.elias_t_h_battery", "sensor", "44"),
            _base_entity("sensor.girls_t_h_battery", "sensor", "53"),
        ],
        [],
        {
            "now": "2026-02-01T00:00:00+00:00",
            "timezone": "UTC",
            "is_night": False,
            "anyone_home": True,
            "people_home": [],
            "people_away": [],
            "last_motion_by_area": {},
        },
    )
    rules = [
        {
            "rule_id": "low_battery_room_sensors_v1",
            "template_id": "low_battery_sensors",
            "params": {
                "sensor_entity_ids": [
                    "sensor.elias_t_h_battery",
                    "sensor.girls_t_h_battery",
                ],
                "threshold": 40,
            },
            "severity": "low",
            "confidence": 0.62,
            "is_sensitive": False,
            "suggested_actions": ["check_sensor"],
        }
    ]
    findings = evaluate_dynamic_rules(snapshot, rules)
    assert findings == []


def test_dynamic_rule_low_battery_sensors_issue_236_missing_required_entity() -> None:
    snapshot = _snapshot(
        [
            _base_entity("sensor.elias_t_h_battery", "sensor", "35"),
        ],
        [],
        {
            "now": "2026-02-01T00:00:00+00:00",
            "timezone": "UTC",
            "is_night": False,
            "anyone_home": True,
            "people_home": [],
            "people_away": [],
            "last_motion_by_area": {},
        },
    )
    rules = [
        {
            "rule_id": "low_battery_room_sensors_v1",
            "template_id": "low_battery_sensors",
            "params": {
                "sensor_entity_ids": [
                    "sensor.elias_t_h_battery",
                    "sensor.girls_t_h_battery",
                ],
                "threshold": 40,
            },
            "severity": "low",
            "confidence": 0.62,
            "is_sensitive": False,
            "suggested_actions": ["check_sensor"],
        }
    ]
    findings = evaluate_dynamic_rules(snapshot, rules)
    assert findings == []


def test_dynamic_rule_motion_alarm_disarmed_home_issue_225_triggers() -> None:
    snapshot = _snapshot(
        [
            _base_entity(
                "alarm_control_panel.home_alarm", "alarm_control_panel", "disarmed"
            ),
            _base_entity("frontgate_vmd3_0", "sensor", "on"),
            _base_entity("frontgate_vmd4_camera1profile1", "sensor", "on"),
            _base_entity("person.lindo_st_angel", "person", "home"),
        ],
        [],
        {
            "now": "2026-02-01T00:00:00+00:00",
            "timezone": "UTC",
            "is_night": False,
            "anyone_home": True,
            "people_home": [],
            "people_away": [],
            "last_motion_by_area": {},
        },
    )
    rules = [
        {
            "rule_id": "motion_frontgate_disarmed_home",
            "template_id": "motion_while_alarm_disarmed_and_home_present",
            "params": {
                "alarm_entity_id": "alarm_control_panel.home_alarm",
                "motion_entity_ids": [
                    "frontgate_vmd3_0",
                    "frontgate_vmd4_camera1profile1",
                ],
                "home_entity_ids": ["person.lindo_st_angel"],
            },
            "severity": "low",
            "confidence": 0.75,
            "is_sensitive": False,
            "suggested_actions": ["close_entry"],
        }
    ]
    findings = evaluate_dynamic_rules(snapshot, rules)
    assert len(findings) == 1
    assert findings[0].type == "motion_frontgate_disarmed_home"
    assert findings[0].triggering_entities == [
        "alarm_control_panel.home_alarm",
        "frontgate_vmd3_0",
        "frontgate_vmd4_camera1profile1",
        "person.lindo_st_angel",
    ]


def test_dynamic_rule_motion_alarm_disarmed_home_issue_225_non_trigger() -> None:
    snapshot = _snapshot(
        [
            _base_entity(
                "alarm_control_panel.home_alarm", "alarm_control_panel", "disarmed"
            ),
            _base_entity("frontgate_vmd3_0", "sensor", "on"),
            _base_entity("frontgate_vmd4_camera1profile1", "sensor", "off"),
            _base_entity("person.lindo_st_angel", "person", "home"),
        ],
        [],
        {
            "now": "2026-02-01T00:00:00+00:00",
            "timezone": "UTC",
            "is_night": False,
            "anyone_home": True,
            "people_home": [],
            "people_away": [],
            "last_motion_by_area": {},
        },
    )
    rules = [
        {
            "rule_id": "motion_frontgate_disarmed_home",
            "template_id": "motion_while_alarm_disarmed_and_home_present",
            "params": {
                "alarm_entity_id": "alarm_control_panel.home_alarm",
                "motion_entity_ids": [
                    "frontgate_vmd3_0",
                    "frontgate_vmd4_camera1profile1",
                ],
                "home_entity_ids": ["person.lindo_st_angel"],
            },
            "severity": "low",
            "confidence": 0.75,
            "is_sensitive": False,
            "suggested_actions": ["close_entry"],
        }
    ]
    findings = evaluate_dynamic_rules(snapshot, rules)
    assert findings == []


def test_dynamic_rule_motion_night_alarm_disarmed_issue_235_triggers() -> None:
    snapshot = _snapshot(
        [
            _base_entity(
                "alarm_control_panel.home_alarm", "alarm_control_panel", "disarmed"
            ),
            _base_entity("binary_sensor.backyard_vmd3_0", "binary_sensor", "on"),
            _base_entity(
                "binary_sensor.backyard_vmd4_camera1profile1", "binary_sensor", "off"
            ),
            _base_entity("person.lindo_st_angel", "person", "home"),
        ],
        [],
        {
            "now": "2026-02-01T00:00:00+00:00",
            "timezone": "UTC",
            "is_night": True,
            "anyone_home": True,
            "people_home": [],
            "people_away": [],
            "last_motion_by_area": {},
        },
    )
    rules = [
        {
            "rule_id": "motion_at_night_disarmed",
            "template_id": "motion_detected_at_night_while_alarm_disarmed",
            "params": {
                "alarm_entity_id": "alarm_control_panel.home_alarm",
                "motion_entity_ids": [
                    "binary_sensor.backyard_vmd3_0",
                    "binary_sensor.backyard_vmd4_camera1profile1",
                ],
                "required_entity_ids": ["person.lindo_st_angel"],
            },
            "severity": "low",
            "confidence": 0.8,
            "is_sensitive": False,
            "suggested_actions": ["close_entry"],
        }
    ]
    findings = evaluate_dynamic_rules(snapshot, rules)
    assert len(findings) == 1
    assert findings[0].type == "motion_at_night_disarmed"
    assert findings[0].triggering_entities == [
        "alarm_control_panel.home_alarm",
        "binary_sensor.backyard_vmd3_0",
        "binary_sensor.backyard_vmd4_camera1profile1",
        "person.lindo_st_angel",
    ]


def test_dynamic_rule_motion_night_alarm_disarmed_issue_235_non_trigger() -> None:
    snapshot = _snapshot(
        [
            _base_entity(
                "alarm_control_panel.home_alarm", "alarm_control_panel", "disarmed"
            ),
            _base_entity("binary_sensor.backyard_vmd3_0", "binary_sensor", "off"),
            _base_entity(
                "binary_sensor.backyard_vmd4_camera1profile1", "binary_sensor", "off"
            ),
            _base_entity("person.lindo_st_angel", "person", "home"),
        ],
        [],
        {
            "now": "2026-02-01T00:00:00+00:00",
            "timezone": "UTC",
            "is_night": True,
            "anyone_home": True,
            "people_home": [],
            "people_away": [],
            "last_motion_by_area": {},
        },
    )
    rules = [
        {
            "rule_id": "motion_at_night_disarmed",
            "template_id": "motion_detected_at_night_while_alarm_disarmed",
            "params": {
                "alarm_entity_id": "alarm_control_panel.home_alarm",
                "motion_entity_ids": [
                    "binary_sensor.backyard_vmd3_0",
                    "binary_sensor.backyard_vmd4_camera1profile1",
                ],
                "required_entity_ids": ["person.lindo_st_angel"],
            },
            "severity": "low",
            "confidence": 0.8,
            "is_sensitive": False,
            "suggested_actions": ["close_entry"],
        }
    ]
    findings = evaluate_dynamic_rules(snapshot, rules)
    assert findings == []


def test_dynamic_rule_motion_night_alarm_disarmed_issue_235_missing_required() -> None:
    snapshot = _snapshot(
        [
            _base_entity(
                "alarm_control_panel.home_alarm", "alarm_control_panel", "disarmed"
            ),
            _base_entity("binary_sensor.backyard_vmd3_0", "binary_sensor", "on"),
            _base_entity(
                "binary_sensor.backyard_vmd4_camera1profile1", "binary_sensor", "off"
            ),
        ],
        [],
        {
            "now": "2026-02-01T00:00:00+00:00",
            "timezone": "UTC",
            "is_night": True,
            "anyone_home": True,
            "people_home": [],
            "people_away": [],
            "last_motion_by_area": {},
        },
    )
    rules = [
        {
            "rule_id": "motion_at_night_disarmed",
            "template_id": "motion_detected_at_night_while_alarm_disarmed",
            "params": {
                "alarm_entity_id": "alarm_control_panel.home_alarm",
                "motion_entity_ids": [
                    "binary_sensor.backyard_vmd3_0",
                    "binary_sensor.backyard_vmd4_camera1profile1",
                ],
                "required_entity_ids": ["person.lindo_st_angel"],
            },
            "severity": "low",
            "confidence": 0.8,
            "is_sensitive": False,
            "suggested_actions": ["close_entry"],
        }
    ]
    findings = evaluate_dynamic_rules(snapshot, rules)
    assert findings == []


@pytest.mark.asyncio
async def test_rule_registry_add_duplicate(hass) -> None:
    registry = RuleRegistry(hass=cast("HomeAssistant", hass))
    await registry.async_load()
    rule = {"rule_id": "rule_1", "template_id": "alarm_disarmed_open_entry"}
    assert await registry.async_add_rule(rule)
    assert not await registry.async_add_rule(rule)


@pytest.mark.asyncio
async def test_rule_registry_toggle_enabled(hass) -> None:
    registry = RuleRegistry(hass=cast("HomeAssistant", hass))
    await registry.async_load()
    rule = {"rule_id": "rule_toggle", "template_id": "open_entry_while_away"}
    assert await registry.async_add_rule(rule)
    assert len(registry.list_rules()) == 1
    assert await registry.async_set_rule_enabled("rule_toggle", enabled=False)
    assert registry.list_rules() == []
    all_rules = registry.list_rules(include_disabled=True)
    assert len(all_rules) == 1
    assert all_rules[0]["enabled"] is False
    assert await registry.async_set_rule_enabled("rule_toggle", enabled=True)
    enabled_rules = registry.list_rules()
    assert len(enabled_rules) == 1
    assert enabled_rules[0]["rule_id"] == "rule_toggle"


@pytest.mark.asyncio
async def test_rule_registry_patch_params(hass) -> None:
    registry = RuleRegistry(hass=cast("HomeAssistant", hass))
    await registry.async_load()
    rule = {
        "rule_id": "rule_patch",
        "template_id": "sensor_threshold_condition",
        "params": {"sensor_entity_id": "sensor.foo", "threshold": 0},
    }
    assert await registry.async_add_rule(rule)

    # patch a single field — other fields survive
    assert await registry.async_patch_rule_params("rule_patch", {"threshold": 5})
    patched = registry.find_rule("rule_patch")
    assert patched is not None
    assert patched["params"]["threshold"] == 5
    assert patched["params"]["sensor_entity_id"] == "sensor.foo"

    # patching a non-existent rule returns False
    assert not await registry.async_patch_rule_params("no_such_rule", {"threshold": 5})


@pytest.mark.parametrize("anyone_home", [True, False])
def test_dynamic_rule_open_entry_at_night_fires_regardless_of_presence(
    anyone_home: bool,  # noqa: FBT001
) -> None:
    """Issue #504: presence-agnostic night template fires whether home or away."""
    snapshot = _snapshot(
        [
            _base_entity("binary_sensor.bedroom_loznice_okno", "binary_sensor", "on"),
        ],
        [],
        {
            "now": "2026-02-01T00:00:00+00:00",
            "timezone": "UTC",
            "is_night": True,
            "anyone_home": anyone_home,
            "people_home": [],
            "people_away": [],
            "last_motion_by_area": {},
        },
    )
    rules = [
        {
            "rule_id": "open_entry_at_night_window",
            "template_id": "open_entry_at_night",
            "params": {"entry_entity_ids": ["binary_sensor.bedroom_loznice_okno"]},
            "severity": "medium",
            "confidence": 0.5,
            "is_sensitive": True,
            "suggested_actions": ["close_entry"],
        }
    ]
    findings = evaluate_dynamic_rules(snapshot, rules)
    assert len(findings) == 1
    assert findings[0].type == "open_entry_at_night_window"
    assert (
        findings[0].evidence["entry_entity_id"] == "binary_sensor.bedroom_loznice_okno"
    )


def test_dynamic_rule_open_entry_at_night_no_finding_during_day() -> None:
    """open_entry_at_night requires derived.is_night."""
    snapshot = _snapshot(
        [
            _base_entity("binary_sensor.bedroom_loznice_okno", "binary_sensor", "on"),
        ],
        [],
        {
            "now": "2026-02-01T12:00:00+00:00",
            "timezone": "UTC",
            "is_night": False,
            "anyone_home": True,
            "people_home": [],
            "people_away": [],
            "last_motion_by_area": {},
        },
    )
    rules = [
        {
            "rule_id": "open_entry_at_night_window",
            "template_id": "open_entry_at_night",
            "params": {"entry_entity_ids": ["binary_sensor.bedroom_loznice_okno"]},
            "severity": "medium",
            "confidence": 0.5,
            "is_sensitive": True,
            "suggested_actions": ["close_entry"],
        }
    ]
    assert evaluate_dynamic_rules(snapshot, rules) == []


@pytest.mark.parametrize(
    ("cover_state", "expected_findings"),
    [("open", 1), ("closed", 0)],
)
def test_dynamic_rule_entry_accepts_cover_open_state(
    cover_state: str, expected_findings: int
) -> None:
    """Covers report open/closed rather than on/off; open must count as open."""
    snapshot = _snapshot(
        [
            _base_entity("cover.rolety_loznice", "cover", cover_state),
        ],
        [],
        {
            "now": "2026-02-01T00:00:00+00:00",
            "timezone": "UTC",
            "is_night": True,
            "anyone_home": True,
            "people_home": [],
            "people_away": [],
            "last_motion_by_area": {},
        },
    )
    rules = [
        {
            "rule_id": "open_entry_at_night_window",
            "template_id": "open_entry_at_night",
            "params": {"entry_entity_ids": ["cover.rolety_loznice"]},
            "severity": "medium",
            "confidence": 0.5,
            "is_sensitive": True,
            "suggested_actions": ["close_entry"],
        }
    ]
    findings = evaluate_dynamic_rules(snapshot, rules)
    assert len(findings) == expected_findings


@pytest.mark.parametrize(
    ("cover_state", "expected_findings"),
    [("open", 1), ("closed", 0)],
)
def test_dynamic_rule_entity_state_duration_accepts_cover_open(
    cover_state: str, expected_findings: int
) -> None:
    """Codex P1: duration rules with target 'on' must match a cover's 'open'."""
    snapshot = _snapshot(
        [
            _base_entity("cover.rolety_loznice", "cover", cover_state),
        ],
        [],
        {
            "now": "2026-02-01T06:00:00+00:00",
            "timezone": "UTC",
            "is_night": False,
            "anyone_home": True,
            "people_home": [],
            "people_away": [],
            "last_motion_by_area": {},
        },
    )
    rules = [
        {
            "rule_id": "cover_open_duration",
            "template_id": "entity_state_duration",
            "params": {
                "entity_id": "cover.rolety_loznice",
                "target_state": "on",
                "threshold_hours": 2.0,
            },
            "severity": "medium",
            "confidence": 0.7,
            "is_sensitive": False,
            "suggested_actions": ["close_entry"],
        }
    ]
    findings = evaluate_dynamic_rules(snapshot, rules)
    assert len(findings) == expected_findings


def test_dynamic_rule_entity_state_duration_lock_target_unchanged() -> None:
    """Non-'on' targets keep exact matching — 'open' must not match 'unlocked'."""
    snapshot = _snapshot(
        [
            _base_entity("lock.front_door", "lock", "open"),
        ],
        [],
        {
            "now": "2026-02-01T06:00:00+00:00",
            "timezone": "UTC",
            "is_night": False,
            "anyone_home": True,
            "people_home": [],
            "people_away": [],
            "last_motion_by_area": {},
        },
    )
    rules = [
        {
            "rule_id": "lock_unlocked_duration",
            "template_id": "entity_state_duration",
            "params": {
                "entity_id": "lock.front_door",
                "target_state": "unlocked",
                "threshold_hours": 2.0,
            },
            "severity": "medium",
            "confidence": 0.7,
            "is_sensitive": True,
            "suggested_actions": ["lock.lock"],
        }
    ]
    assert evaluate_dynamic_rules(snapshot, rules) == []


def _motion_night_away_rule() -> dict[str, object]:
    return {
        "rule_id": "motion_detected_at_night_while_away",
        "template_id": "motion_detected_at_night_while_away",
        "params": {
            "motion_entity_ids": ["binary_sensor.xiao_esp32_c5_espectre_motion"],
        },
        "severity": "medium",
        "confidence": 0.8,
        "is_sensitive": False,
        "suggested_actions": ["check_camera"],
    }


def _motion_night_away_snapshot(
    *, motion_state: str, is_night: bool, anyone_home: bool
) -> FullStateSnapshot:
    return _snapshot(
        [
            _base_entity(
                "binary_sensor.xiao_esp32_c5_espectre_motion",
                "binary_sensor",
                motion_state,
            ),
        ],
        [],
        {
            "now": "2026-02-01T00:00:00+00:00",
            "timezone": "UTC",
            "is_night": is_night,
            "anyone_home": anyone_home,
            "people_home": [],
            "people_away": [],
            "last_motion_by_area": {},
        },
    )


def test_dynamic_rule_motion_night_while_away_issue_516_triggers() -> None:
    snapshot = _motion_night_away_snapshot(
        motion_state="on", is_night=True, anyone_home=False
    )
    findings = evaluate_dynamic_rules(snapshot, [_motion_night_away_rule()])
    assert len(findings) == 1
    assert findings[0].type == "motion_detected_at_night_while_away"
    assert findings[0].severity == "medium"
    assert findings[0].triggering_entities == [
        "binary_sensor.xiao_esp32_c5_espectre_motion"
    ]
    assert findings[0].evidence["anyone_home"] is False
    assert findings[0].evidence["is_night"] is True


def test_dynamic_rule_motion_night_while_away_issue_516_non_trigger() -> None:
    """No finding when someone is home, during the day, or without motion."""
    non_trigger_contexts = (
        ("on", True, True),  # someone is home
        ("on", False, False),  # daytime
        ("off", True, False),  # no motion
    )
    for motion_state, is_night, anyone_home in non_trigger_contexts:
        snapshot = _motion_night_away_snapshot(
            motion_state=motion_state, is_night=is_night, anyone_home=anyone_home
        )
        assert evaluate_dynamic_rules(snapshot, [_motion_night_away_rule()]) == []


def test_dynamic_rule_motion_night_while_away_partial_resolution_fires() -> None:
    """A renamed/removed sensor must not disable the remaining sensors."""
    rule = _motion_night_away_rule()
    rule["params"] = {
        "motion_entity_ids": [
            "binary_sensor.removed_motion",
            "binary_sensor.hall_motion",
        ],
    }
    snapshot = _snapshot(
        [
            _base_entity("binary_sensor.hall_motion", "binary_sensor", "on"),
        ],
        [],
        {
            "now": "2026-02-01T00:00:00+00:00",
            "timezone": "UTC",
            "is_night": True,
            "anyone_home": False,
            "people_home": [],
            "people_away": [],
            "last_motion_by_area": {},
        },
    )
    findings = evaluate_dynamic_rules(snapshot, [rule])
    assert len(findings) == 1
    assert findings[0].triggering_entities == ["binary_sensor.hall_motion"]
    assert findings[0].evidence["unresolved_entity_ids"] == [
        "binary_sensor.removed_motion"
    ]


def test_dynamic_rule_motion_night_while_away_issue_516_missing_entity() -> None:
    """No configured motion entity resolving in the snapshot fails closed."""
    snapshot = _snapshot(
        [
            _base_entity("binary_sensor.other_motion", "binary_sensor", "on"),
        ],
        [],
        {
            "now": "2026-02-01T00:00:00+00:00",
            "timezone": "UTC",
            "is_night": True,
            "anyone_home": False,
            "people_home": [],
            "people_away": [],
            "last_motion_by_area": {},
        },
    )
    assert evaluate_dynamic_rules(snapshot, [_motion_night_away_rule()]) == []


def test_dynamic_rule_motion_night_while_away_issue_516_invalid_params() -> None:
    """Missing, non-list, or empty motion_entity_ids yield no findings."""
    snapshot = _motion_night_away_snapshot(
        motion_state="on", is_night=True, anyone_home=False
    )
    invalid_params: tuple[dict[str, object], ...] = (
        {},
        {"motion_entity_ids": "binary_sensor.xiao_esp32_c5_espectre_motion"},
        {"motion_entity_ids": []},
    )
    for params in invalid_params:
        rule = _motion_night_away_rule()
        rule["params"] = params
        assert evaluate_dynamic_rules(snapshot, [rule]) == []


def test_dynamic_rule_motion_night_while_away_multi_entity_partial_on() -> None:
    """Any-of semantics: one of several motion sensors ON is enough to fire."""
    rule = _motion_night_away_rule()
    rule["params"] = {
        "motion_entity_ids": [
            "binary_sensor.xiao_esp32_c5_espectre_motion",
            "binary_sensor.hall_motion",
        ],
    }
    snapshot = _snapshot(
        [
            _base_entity(
                "binary_sensor.xiao_esp32_c5_espectre_motion", "binary_sensor", "off"
            ),
            _base_entity("binary_sensor.hall_motion", "binary_sensor", "on"),
        ],
        [],
        {
            "now": "2026-02-01T00:00:00+00:00",
            "timezone": "UTC",
            "is_night": True,
            "anyone_home": False,
            "people_home": [],
            "people_away": [],
            "last_motion_by_area": {},
        },
    )
    findings = evaluate_dynamic_rules(snapshot, [rule])
    assert len(findings) == 1
    assert findings[0].evidence["motion_states"] == {
        "binary_sensor.xiao_esp32_c5_espectre_motion": "off",
        "binary_sensor.hall_motion": "on",
    }


def _motion_away_rule() -> dict[str, object]:
    return {
        "rule_id": "motion_kitchen_while_away",
        "template_id": "motion_detected_while_away",
        "params": {
            "motion_entity_ids": ["binary_sensor.xiao_esp32_c5_espectre_motion"],
        },
        "severity": "low",
        "confidence": 0.6,
        "is_sensitive": False,
        "suggested_actions": ["check_camera"],
    }


def test_dynamic_rule_motion_while_away_issue_518_triggers_daytime() -> None:
    """Issue #518: motion while away fires with no night gate."""
    snapshot = _motion_night_away_snapshot(
        motion_state="on", is_night=False, anyone_home=False
    )
    findings = evaluate_dynamic_rules(snapshot, [_motion_away_rule()])
    assert len(findings) == 1
    assert findings[0].type == "motion_kitchen_while_away"
    assert findings[0].severity == "low"
    assert findings[0].triggering_entities == [
        "binary_sensor.xiao_esp32_c5_espectre_motion"
    ]
    assert findings[0].evidence["anyone_home"] is False
    # The day-agnostic template omits is_night from the evidence so a finding
    # persisting across the night boundary keeps a stable anomaly ID.
    assert "is_night" not in findings[0].evidence


def test_dynamic_rule_motion_while_away_issue_518_triggers_at_night() -> None:
    """The day-agnostic template also fires at night."""
    snapshot = _motion_night_away_snapshot(
        motion_state="on", is_night=True, anyone_home=False
    )
    findings = evaluate_dynamic_rules(snapshot, [_motion_away_rule()])
    assert len(findings) == 1


def test_dynamic_rule_motion_while_away_issue_518_non_trigger() -> None:
    """No finding when someone is home or without motion."""
    non_trigger_contexts = (
        ("on", True),  # someone is home
        ("off", False),  # no motion
    )
    for motion_state, anyone_home in non_trigger_contexts:
        snapshot = _motion_night_away_snapshot(
            motion_state=motion_state, is_night=False, anyone_home=anyone_home
        )
        assert evaluate_dynamic_rules(snapshot, [_motion_away_rule()]) == []


def test_dynamic_rule_motion_while_away_partial_resolution_fires() -> None:
    """A renamed/removed sensor must not disable the remaining sensors."""
    rule = _motion_away_rule()
    rule["params"] = {
        "motion_entity_ids": [
            "binary_sensor.removed_motion",
            "binary_sensor.hall_motion",
        ],
    }
    snapshot = _snapshot(
        [
            _base_entity("binary_sensor.hall_motion", "binary_sensor", "on"),
        ],
        [],
        {
            "now": "2026-02-01T12:00:00+00:00",
            "timezone": "UTC",
            "is_night": False,
            "anyone_home": False,
            "people_home": [],
            "people_away": [],
            "last_motion_by_area": {},
        },
    )
    findings = evaluate_dynamic_rules(snapshot, [rule])
    assert len(findings) == 1
    assert findings[0].triggering_entities == ["binary_sensor.hall_motion"]
    assert findings[0].evidence["unresolved_entity_ids"] == [
        "binary_sensor.removed_motion"
    ]


def test_dynamic_rule_motion_while_away_issue_518_invalid_params() -> None:
    """Missing, non-list, or empty motion_entity_ids yield no findings."""
    snapshot = _motion_night_away_snapshot(
        motion_state="on", is_night=False, anyone_home=False
    )
    invalid_params: tuple[dict[str, object], ...] = (
        {},
        {"motion_entity_ids": "binary_sensor.xiao_esp32_c5_espectre_motion"},
        {"motion_entity_ids": []},
    )
    for params in invalid_params:
        rule = _motion_away_rule()
        rule["params"] = params
        assert evaluate_dynamic_rules(snapshot, [rule]) == []


def test_dynamic_rule_night_template_evidence_still_carries_is_night() -> None:
    """
    The shared evaluator keeps is_night in the night template's evidence.

    build_anomaly_id hashes evidence in insertion order, so dropping or
    moving the key would silently re-key every existing night finding.
    """
    snapshot = _motion_night_away_snapshot(
        motion_state="on", is_night=True, anyone_home=False
    )
    findings = evaluate_dynamic_rules(snapshot, [_motion_night_away_rule()])
    assert len(findings) == 1
    evidence_keys = list(findings[0].evidence.keys())
    assert evidence_keys == [
        "rule_id",
        "template_id",
        "is_night",
        "anyone_home",
        "motion_entity_ids",
        "motion_states",
        "unresolved_entity_ids",
    ]


def test_dynamic_rule_motion_while_away_issue_518_missing_entity() -> None:
    """No configured motion entity resolving in the snapshot fails closed."""
    snapshot = _snapshot(
        [
            _base_entity("binary_sensor.other_motion", "binary_sensor", "on"),
        ],
        [],
        {
            "now": "2026-02-01T12:00:00+00:00",
            "timezone": "UTC",
            "is_night": False,
            "anyone_home": False,
            "people_home": [],
            "people_away": [],
            "last_motion_by_area": {},
        },
    )
    assert evaluate_dynamic_rules(snapshot, [_motion_away_rule()]) == []


def test_dynamic_rule_motion_away_triggering_excludes_idle_sensors() -> None:
    """
    Only sensors reporting motion are triggering entities.

    The engine's exclusion filter drops a finding when ANY triggering entity
    is excluded, so an idle sensor in the list would let its exclusion kill
    a genuine alert from the sensor that fired (issue #518 Codex P1;
    applies to the night template too).
    """
    for template_id, is_night in (
        ("motion_detected_while_away", False),
        ("motion_detected_at_night_while_away", True),
    ):
        rule = _motion_away_rule()
        rule["rule_id"] = template_id
        rule["template_id"] = template_id
        rule["params"] = {
            "motion_entity_ids": [
                "binary_sensor.idle_motion",
                "binary_sensor.hall_motion",
            ],
        }
        snapshot = _snapshot(
            [
                _base_entity("binary_sensor.idle_motion", "binary_sensor", "off"),
                _base_entity("binary_sensor.hall_motion", "binary_sensor", "on"),
            ],
            [],
            {
                "now": "2026-02-01T00:00:00+00:00",
                "timezone": "UTC",
                "is_night": is_night,
                "anyone_home": False,
                "people_home": [],
                "people_away": [],
                "last_motion_by_area": {},
            },
        )
        findings = evaluate_dynamic_rules(snapshot, [rule])
        assert len(findings) == 1, template_id
        assert findings[0].triggering_entities == ["binary_sensor.hall_motion"], (
            template_id
        )
        # Evidence still records every configured sensor's state.
        assert findings[0].evidence["motion_states"] == {
            "binary_sensor.idle_motion": "off",
            "binary_sensor.hall_motion": "on",
        }


def test_dynamic_rule_motion_while_away_severity_escalates_at_night() -> None:
    """
    Night motion carries the night template's severity judgment.

    The recommended quiet-hours config suppresses 'low' overnight, which
    would mute the flagship 2am intrusion signal for day-rule-only users
    (issue #518 red-team review).
    """
    night_snapshot = _motion_night_away_snapshot(
        motion_state="on", is_night=True, anyone_home=False
    )
    day_snapshot = _motion_night_away_snapshot(
        motion_state="on", is_night=False, anyone_home=False
    )
    night_findings = evaluate_dynamic_rules(night_snapshot, [_motion_away_rule()])
    day_findings = evaluate_dynamic_rules(day_snapshot, [_motion_away_rule()])
    assert night_findings[0].severity == "medium"
    assert day_findings[0].severity == "low"
    # Severity is not hashed into the anomaly ID — identity is stable
    # across the night boundary (evidence carries no is_night key).
    assert night_findings[0].anomaly_id == day_findings[0].anomaly_id


def test_dynamic_rule_motion_away_overlap_emits_both_findings() -> None:
    """
    Night + day rules on the same sensor both emit at night.

    Evaluator-level dedup was reverted (issue #518 verification round 5):
    it ran before snooze/exclusion suppression, so a snoozed night rule
    silently lost the day rule's alert. Dispatch-level dedup is a TODO;
    docs advise replacing the night rule instead of running both.
    """
    snapshot = _motion_night_away_snapshot(
        motion_state="on", is_night=True, anyone_home=False
    )
    rules = [_motion_night_away_rule(), _motion_away_rule()]
    findings = evaluate_dynamic_rules(snapshot, rules)
    templates = sorted(str(f.evidence["template_id"]) for f in findings)
    assert templates == [
        "motion_detected_at_night_while_away",
        "motion_detected_while_away",
    ]


def test_dynamic_rule_motion_away_overlap_distinct_sensors_both_fire() -> None:
    """Non-overlapping sensor sets both emit findings."""
    night_rule = _motion_night_away_rule()
    day_rule = _motion_away_rule()
    day_rule["params"] = {"motion_entity_ids": ["binary_sensor.hall_motion"]}
    snapshot = _snapshot(
        [
            _base_entity(
                "binary_sensor.xiao_esp32_c5_espectre_motion", "binary_sensor", "on"
            ),
            _base_entity("binary_sensor.hall_motion", "binary_sensor", "on"),
        ],
        [],
        {
            "now": "2026-02-01T00:00:00+00:00",
            "timezone": "UTC",
            "is_night": True,
            "anyone_home": False,
            "people_home": [],
            "people_away": [],
            "last_motion_by_area": {},
        },
    )
    findings = evaluate_dynamic_rules(snapshot, [night_rule, day_rule])
    assert len(findings) == 2


def test_dynamic_rule_motion_away_day_rule_alone_fires_at_night() -> None:
    """A day rule with no night sibling fires normally at night."""
    snapshot = _motion_night_away_snapshot(
        motion_state="on", is_night=True, anyone_home=False
    )
    findings = evaluate_dynamic_rules(snapshot, [_motion_away_rule()])
    assert len(findings) == 1
    assert findings[0].evidence["template_id"] == "motion_detected_while_away"


def test_dynamic_rule_motion_away_overlap_superset_sensors_both_fire() -> None:
    """Day and night rules over nested sensor sets both emit findings."""
    night_rule = _motion_night_away_rule()
    day_rule = _motion_away_rule()
    day_rule["params"] = {
        "motion_entity_ids": [
            "binary_sensor.xiao_esp32_c5_espectre_motion",
            "binary_sensor.hall_motion",
        ],
    }
    snapshot = _snapshot(
        [
            _base_entity(
                "binary_sensor.xiao_esp32_c5_espectre_motion", "binary_sensor", "on"
            ),
            _base_entity("binary_sensor.hall_motion", "binary_sensor", "on"),
        ],
        [],
        {
            "now": "2026-02-01T00:00:00+00:00",
            "timezone": "UTC",
            "is_night": True,
            "anyone_home": False,
            "people_home": [],
            "people_away": [],
            "last_motion_by_area": {},
        },
    )
    findings = evaluate_dynamic_rules(snapshot, [night_rule, day_rule])
    templates = sorted(str(f.evidence["template_id"]) for f in findings)
    assert templates == [
        "motion_detected_at_night_while_away",
        "motion_detected_while_away",
    ]


def test_dynamic_rule_motion_while_away_high_severity_not_downgraded() -> None:
    """Night escalation is a floor: a user-configured high rule stays high."""
    rule = _motion_away_rule()
    rule["severity"] = "high"
    snapshot = _motion_night_away_snapshot(
        motion_state="on", is_night=True, anyone_home=False
    )
    findings = evaluate_dynamic_rules(snapshot, [rule])
    assert findings[0].severity == "high"

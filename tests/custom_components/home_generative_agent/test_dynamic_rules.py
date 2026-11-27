# ruff: noqa: S101
"""Tests for dynamic rules evaluation and registry."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

import pytest

from custom_components.home_generative_agent.sentinel.dynamic_rules import (
    evaluate_dynamic_rule,
    evaluate_dynamic_rules,
)
from custom_components.home_generative_agent.sentinel.models import build_anomaly_id
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


_DEAD_RULE_DERIVED = {
    "now": "2026-02-01T00:00:00+00:00",
    "timezone": "UTC",
    "is_night": False,
    "anyone_home": True,
    "people_home": [],
    "people_away": [],
    "last_motion_by_area": {},
}
_DEAD_RULE_PRESENT = "sensor.backyard_vmd3_0"
_DEAD_RULE_GONE = "sensor.0x00158d0001a2b3c4"


def _dead_rule(template_id: str, rule_id: str | None = None) -> dict[str, Any]:
    return {
        "rule_id": rule_id or f"dead_rule_{template_id}",
        "template_id": template_id,
        "params": {"sensor_entity_ids": [_DEAD_RULE_PRESENT, _DEAD_RULE_GONE]},
        "severity": "low",
        "confidence": 0.8,
        "is_sensitive": False,
        "suggested_actions": ["check_sensor"],
    }


def _dead_rule_snapshot(
    *entity_ids: str, anyone_home: bool = True
) -> FullStateSnapshot:
    return _snapshot(
        [_base_entity(entity_id, "sensor", "unavailable") for entity_id in entity_ids],
        [],
        {**_DEAD_RULE_DERIVED, "anyone_home": anyone_home},
    )


def _dead_rule_warnings(caplog: pytest.LogCaptureFixture) -> list[str]:
    return [
        record.getMessage()
        for record in caplog.records
        if record.levelname == "WARNING" and "cannot fire" in record.getMessage()
    ]


@pytest.mark.parametrize(
    "template_id", ["unavailable_sensors_while_home", "unavailable_sensors"]
)
def test_dynamic_rule_unavailable_sensors_logs_a_dead_rule_once(
    template_id: str, caplog: pytest.LogCaptureFixture
) -> None:
    """
    A rule that lists a sensor with no state says so (#960).

    It still does not fire, but it is logged with its missing sensors: once
    while they stay missing, again when the missing set changes, and again
    if it resolves and then breaks again.
    """
    broken = _dead_rule_snapshot(_DEAD_RULE_PRESENT)
    whole = _dead_rule_snapshot(_DEAD_RULE_PRESENT, _DEAD_RULE_GONE)
    empty = _dead_rule_snapshot()
    rules = [_dead_rule(template_id)]
    log: dict[str, tuple[str, ...]] = {}

    with caplog.at_level("WARNING"):
        assert evaluate_dynamic_rules(broken, rules, unresolved_log=log) == []
        assert evaluate_dynamic_rules(broken, rules, unresolved_log=log) == []
        warnings = _dead_rule_warnings(caplog)
        assert len(warnings) == 1
        assert rules[0]["rule_id"] in warnings[0]
        assert f"'{_DEAD_RULE_GONE}'" in warnings[0]
        assert _DEAD_RULE_PRESENT not in warnings[0]

        assert evaluate_dynamic_rules(empty, rules, unresolved_log=log) == []
        warnings = _dead_rule_warnings(caplog)
        assert len(warnings) == 2
        assert f"'{_DEAD_RULE_GONE}', '{_DEAD_RULE_PRESENT}'" in warnings[1]

        assert evaluate_dynamic_rules(whole, rules, unresolved_log=log) != []
        assert log == {}
        assert len(_dead_rule_warnings(caplog)) == 2

        assert evaluate_dynamic_rules(broken, rules, unresolved_log=log) == []
        assert len(_dead_rule_warnings(caplog)) == 3


def test_dynamic_rule_preview_of_a_dead_rule_logs_and_records_nothing(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """
    A preview must not use up the warning the active rule needs (#960).

    Previewing a proposal evaluates it with no record. If that logged, the
    message would name a rule that is not active, and the engine's own first
    run of the approved rule would then stay silent.
    """
    broken = _dead_rule_snapshot(_DEAD_RULE_PRESENT)
    rule = _dead_rule("unavailable_sensors")
    log: dict[str, tuple[str, ...]] = {}

    with caplog.at_level("WARNING"):
        assert evaluate_dynamic_rule(broken, rule) == []
        assert _dead_rule_warnings(caplog) == []

        assert evaluate_dynamic_rules(broken, [rule], unresolved_log=log) == []
        assert len(_dead_rule_warnings(caplog)) == 1


def test_dynamic_rule_dead_rule_is_forgotten_when_it_leaves_the_run(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A rule deactivated while broken is reported again when it comes back."""
    broken = _dead_rule_snapshot(_DEAD_RULE_PRESENT)
    rules = [_dead_rule("unavailable_sensors")]
    log: dict[str, tuple[str, ...]] = {}

    with caplog.at_level("WARNING"):
        evaluate_dynamic_rules(broken, rules, unresolved_log=log)
        evaluate_dynamic_rules(broken, [], unresolved_log=log)
        assert log == {}
        evaluate_dynamic_rules(broken, rules, unresolved_log=log)
        assert len(_dead_rule_warnings(caplog)) == 2


def test_dynamic_rule_dead_while_home_rule_is_logged_once_across_departures(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Nobody home: the rule is still reported, and not again at each return."""
    away = _dead_rule_snapshot(_DEAD_RULE_PRESENT, anyone_home=False)
    home = _dead_rule_snapshot(_DEAD_RULE_PRESENT)
    rules = [_dead_rule("unavailable_sensors_while_home")]
    log: dict[str, tuple[str, ...]] = {}

    with caplog.at_level("WARNING"):
        assert evaluate_dynamic_rules(away, rules, unresolved_log=log) == []
        assert len(_dead_rule_warnings(caplog)) == 1
        assert evaluate_dynamic_rules(home, rules, unresolved_log=log) == []
        assert evaluate_dynamic_rules(away, rules, unresolved_log=log) == []
        assert len(_dead_rule_warnings(caplog)) == 1


def test_dynamic_rule_dead_rule_warning_quotes_and_caps_what_it_shows(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Stored IDs are rule text a model proposed: quoted, cut and counted."""
    rule = _dead_rule("unavailable_sensors")
    rule["params"]["sensor_entity_ids"] = [
        "sensor.a\nforged line",
        "sensor." + "x" * 300,
        *(f"sensor.gone_{index:02d}" for index in range(12)),
    ]
    log: dict[str, tuple[str, ...]] = {}

    with caplog.at_level("WARNING"):
        evaluate_dynamic_rules(_dead_rule_snapshot(), [rule], unresolved_log=log)
    (warning,) = _dead_rule_warnings(caplog)
    assert "\n" not in warning
    assert "x" * 101 not in warning
    assert "and 4 more" in warning


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


def _low_battery_rule(*sensor_ids: str) -> dict[str, Any]:
    return {
        "rule_id": "low_battery_room_sensors_v1",
        "template_id": "low_battery_sensors",
        "params": {"sensor_entity_ids": list(sensor_ids), "threshold": 40},
        "severity": "low",
        "confidence": 0.62,
        "is_sensitive": False,
        "suggested_actions": ["check_sensor"],
    }


def _low_battery_snapshot(**states: str) -> FullStateSnapshot:
    return _snapshot(
        [
            _base_entity(f"sensor.{name}", "sensor", state)
            for name, state in states.items()
        ],
        [],
        _DEAD_RULE_DERIVED,
    )


@pytest.mark.parametrize("dropped_state", ["unavailable", "unknown"])
def test_dynamic_rule_low_battery_skips_a_sensor_it_cannot_read(
    dropped_state: str,
) -> None:
    """
    One sensor off the network does not silence the rule (#1064).

    The sensor with no reading is left out and the others are still checked.
    The evidence keeps its state, so the finding shows it was not read.
    """
    rule = _low_battery_rule("sensor.elias", "sensor.girls", "sensor.hall")
    snapshot = _low_battery_snapshot(elias="37", girls=dropped_state, hall="80")

    (finding,) = evaluate_dynamic_rules(snapshot, [rule])

    assert finding.triggering_entities == ["sensor.elias"]
    assert finding.evidence["sensor_states"]["sensor.girls"] == dropped_state
    assert finding.evidence["sensor_levels"] == {
        "sensor.elias": 37.0,
        "sensor.hall": 80.0,
    }


def test_dynamic_rule_low_battery_unreadable_sensors_alone_do_not_fire() -> None:
    """A sensor that cannot be read is not a low one, and the rest are fine."""
    rule = _low_battery_rule("sensor.elias", "sensor.girls")

    for states in (
        {"elias": "unavailable", "girls": "53"},
        {"elias": "unavailable", "girls": "unknown"},
    ):
        assert evaluate_dynamic_rules(_low_battery_snapshot(**states), [rule]) == []


def test_dynamic_rule_low_battery_logs_a_dead_rule_once(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """
    A low-battery rule that lists a sensor with no state says so (#1064).

    It does not fire even though the sensor that is left reads low, as for
    the unavailable-sensors rules (#960), and it fires again once the missing
    sensor has a state.
    """
    rule = _low_battery_rule("sensor.elias", _DEAD_RULE_GONE)
    broken = _low_battery_snapshot(elias="12")
    whole = _low_battery_snapshot(**{"elias": "12", "0x00158d0001a2b3c4": "90"})
    log: dict[str, tuple[str, ...]] = {}

    with caplog.at_level("WARNING"):
        assert evaluate_dynamic_rules(broken, [rule], unresolved_log=log) == []
        assert evaluate_dynamic_rules(broken, [rule], unresolved_log=log) == []
        (warning,) = _dead_rule_warnings(caplog)
        assert rule["rule_id"] in warning
        assert f"'{_DEAD_RULE_GONE}'" in warning
        assert "sensor.elias" not in warning

        assert evaluate_dynamic_rules(whole, [rule], unresolved_log=log) != []
        assert log == {}
        assert len(_dead_rule_warnings(caplog)) == 1


def test_dynamic_rule_low_battery_unreadable_sensor_is_not_reported_as_dead(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """An unavailable sensor still has a state: the rule is alive, no warning."""
    rule = _low_battery_rule("sensor.elias", "sensor.girls")
    snapshot = _low_battery_snapshot(elias="unavailable", girls="53")
    log: dict[str, tuple[str, ...]] = {}

    with caplog.at_level("WARNING"):
        evaluate_dynamic_rules(snapshot, [rule], unresolved_log=log)

    assert log == {}
    assert _dead_rule_warnings(caplog) == []


def _unreadable_warnings(caplog: pytest.LogCaptureFixture) -> list[str]:
    return [
        record.getMessage()
        for record in caplog.records
        if record.levelname == "WARNING" and "#1082" in record.getMessage()
    ]


def _since(snapshot: FullStateSnapshot, entity_id: str, last_changed: str) -> None:
    for entity in snapshot["entities"]:
        if entity["entity_id"] == entity_id:
            entity["last_changed"] = last_changed


@pytest.mark.parametrize("state", ["locked", "low", "ok", ""])
def test_dynamic_rule_low_battery_warns_once_for_a_state_that_is_no_level(
    state: str, caplog: pytest.LogCaptureFixture
) -> None:
    """
    A listed sensor whose state is never a number is named once (#1082).

    The rule still checks the others and fires for a low one. The warning
    is not repeated while nothing changes, and comes back if the sensor
    reads and then stops reading again.
    """
    rule = _low_battery_rule("sensor.front_lock", "sensor.hall")
    broken = _low_battery_snapshot(front_lock=state, hall="12")
    fixed = _low_battery_snapshot(front_lock="90", hall="12")
    log: dict[tuple[str, str], tuple[str, ...]] = {}

    with caplog.at_level("WARNING"):
        (finding,) = evaluate_dynamic_rules(broken, [rule], unreadable_log=log)
        assert finding.triggering_entities == ["sensor.hall"]
        evaluate_dynamic_rules(broken, [rule], unreadable_log=log)
        (warning,) = _unreadable_warnings(caplog)
        assert rule["rule_id"] in warning
        assert "'sensor.front_lock'" in warning
        assert "not a battery level" in warning
        assert "sensor.hall" not in warning

        evaluate_dynamic_rules(fixed, [rule], unreadable_log=log)
        assert log == {}
        evaluate_dynamic_rules(broken, [rule], unreadable_log=log)
        assert len(_unreadable_warnings(caplog)) == 2


@pytest.mark.parametrize(
    "dropped_state", ["unavailable", "unknown", "nan", "inf", "-inf", "1e999"]
)
def test_dynamic_rule_low_battery_warns_once_after_a_day_unreadable(
    dropped_state: str, caplog: pytest.LogCaptureFixture
) -> None:
    """
    A sensor unavailable for a day is named once, a shorter gap is not (#1082).

    The snapshot's clock is 2026-02-01T00:00 UTC.
    """
    rule = _low_battery_rule("sensor.girls", "sensor.hall")
    log: dict[tuple[str, str], tuple[str, ...]] = {}

    with caplog.at_level("WARNING"):
        recent = _low_battery_snapshot(girls=dropped_state, hall="80")
        _since(recent, "sensor.girls", "2026-01-31T00:00:01+00:00")
        assert evaluate_dynamic_rules(recent, [rule], unreadable_log=log) == []
        assert _unreadable_warnings(caplog) == []
        assert log == {}

        long_gone = _low_battery_snapshot(girls=dropped_state, hall="80")
        _since(long_gone, "sensor.girls", "2026-01-31T00:00:00+00:00")
        assert evaluate_dynamic_rules(long_gone, [rule], unreadable_log=log) == []
        assert evaluate_dynamic_rules(long_gone, [rule], unreadable_log=log) == []
        (warning,) = _unreadable_warnings(caplog)
        assert rule["rule_id"] in warning
        assert "'sensor.girls'" in warning
        assert "a day or more" in warning


def test_dynamic_rule_low_battery_unreadable_since_unknown_time_is_quiet(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """No readable last_changed: no claim about how long it has been."""
    rule = _low_battery_rule("sensor.girls", "sensor.hall")
    snapshot = _low_battery_snapshot(girls="unavailable", hall="80")
    _since(snapshot, "sensor.girls", "not a time")
    log: dict[tuple[str, str], tuple[str, ...]] = {}

    with caplog.at_level("WARNING"):
        evaluate_dynamic_rules(snapshot, [rule], unreadable_log=log)

    assert log == {}
    assert _unreadable_warnings(caplog) == []


def test_dynamic_rule_low_battery_unreadable_kinds_are_reported_apart(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """One rule, both kinds: two warnings, and one going away keeps the other."""
    rule = _low_battery_rule("sensor.front_lock", "sensor.girls")
    both = _low_battery_snapshot(front_lock="locked", girls="unavailable")
    _since(both, "sensor.girls", "2026-01-30T00:00:00+00:00")
    lock_only = _low_battery_snapshot(front_lock="locked", girls="55")
    log: dict[tuple[str, str], tuple[str, ...]] = {}

    with caplog.at_level("WARNING"):
        evaluate_dynamic_rules(both, [rule], unreadable_log=log)
        assert len(_unreadable_warnings(caplog)) == 2
        evaluate_dynamic_rules(lock_only, [rule], unreadable_log=log)
        assert len(_unreadable_warnings(caplog)) == 2
        assert list(log) == [(rule["rule_id"], "not_a_number")]


def test_dynamic_rule_low_battery_flapping_sensor_does_not_repeat_others(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A sensor that keeps coming and going warns only when it joins (#1082)."""
    rule = _low_battery_rule("sensor.front_lock", "sensor.hall")
    both = _low_battery_snapshot(front_lock="locked", hall="low")
    lock_only = _low_battery_snapshot(front_lock="locked", hall="90")
    log: dict[tuple[str, str], tuple[str, ...]] = {}

    with caplog.at_level("WARNING"):
        for snapshot in (lock_only, both, lock_only, both):
            evaluate_dynamic_rules(snapshot, [rule], unreadable_log=log)
    warnings = _unreadable_warnings(caplog)
    assert len(warnings) == 3
    assert "'sensor.hall'" not in warnings[0]
    assert all("'sensor.hall'" in warning for warning in warnings[1:])
    assert all("'sensor.front_lock'" not in warning for warning in warnings[1:])


def test_dynamic_rule_low_battery_sensor_off_the_network_is_not_named_again(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A lock that drops off for a while and comes back stays reported (#1082)."""
    rule = _low_battery_rule("sensor.front_lock", "sensor.hall")
    locked = _low_battery_snapshot(front_lock="locked", hall="80")
    dropped = _low_battery_snapshot(front_lock="unavailable", hall="80")
    log: dict[tuple[str, str], tuple[str, ...]] = {}

    with caplog.at_level("WARNING"):
        for snapshot in (locked, dropped, locked):
            evaluate_dynamic_rules(snapshot, [rule], unreadable_log=log)

    assert len(_unreadable_warnings(caplog)) == 1


def test_dynamic_rule_low_battery_record_survives_a_sensor_with_no_state(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Another sensor missing for a run (#960) does not reset the record."""
    rule = _low_battery_rule("sensor.front_lock", "sensor.hall")
    whole = _low_battery_snapshot(front_lock="locked", hall="80")
    hall_gone = _low_battery_snapshot(front_lock="locked")
    log: dict[tuple[str, str], tuple[str, ...]] = {}

    with caplog.at_level("WARNING"):
        for snapshot in (whole, hall_gone, whole):
            evaluate_dynamic_rules(snapshot, [rule], unreadable_log=log)

    assert len(_unreadable_warnings(caplog)) == 1


def test_dynamic_rule_low_battery_warning_names_the_sensor_that_joined(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """With many reported already, the newcomer is the one shown (#1082)."""
    names = [f"lock_{index:02d}" for index in range(10)]
    rule = _low_battery_rule(*(f"sensor.{name}" for name in names), "sensor.z_lock")
    before = _low_battery_snapshot(**dict.fromkeys(names, "locked"), z_lock="50")
    after = _low_battery_snapshot(**dict.fromkeys(names, "locked"), z_lock="ok")
    log: dict[tuple[str, str], tuple[str, ...]] = {}

    with caplog.at_level("WARNING"):
        evaluate_dynamic_rules(before, [rule], unreadable_log=log)
        evaluate_dynamic_rules(after, [rule], unreadable_log=log)

    second = _unreadable_warnings(caplog)[1]
    assert "'sensor.z_lock'" in second
    assert "sensor.lock_00" not in second


@pytest.mark.parametrize(
    "last_changed", ["2026-02-30T00:00:00+00:00", "2026-01-01T00:00:00"]
)
def test_dynamic_rule_low_battery_bad_last_changed_does_not_stop_the_run(
    last_changed: str, caplog: pytest.LogCaptureFixture
) -> None:
    """An impossible or zone-less time is skipped; other rules still fire."""
    rule = _low_battery_rule("sensor.girls", "sensor.hall")
    snapshot = _low_battery_snapshot(girls="unavailable", hall="12")
    _since(snapshot, "sensor.girls", last_changed)
    log: dict[tuple[str, str], tuple[str, ...]] = {}

    with caplog.at_level("WARNING"):
        (finding,) = evaluate_dynamic_rules(snapshot, [rule], unreadable_log=log)

    assert finding.triggering_entities == ["sensor.hall"]
    assert log == {}
    assert _unreadable_warnings(caplog) == []


def test_dynamic_rule_low_battery_change_of_kind_is_not_named_again(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Text, then unavailable for a day, with no reading between: once."""
    rule = _low_battery_rule("sensor.girls", "sensor.hall")
    text = _low_battery_snapshot(girls="low", hall="80")
    gone = _low_battery_snapshot(girls="unavailable", hall="80")
    _since(gone, "sensor.girls", "2026-01-30T00:00:00+00:00")
    log: dict[tuple[str, str], tuple[str, ...]] = {}

    with caplog.at_level("WARNING"):
        for snapshot in (text, gone, text):
            evaluate_dynamic_rules(snapshot, [rule], unreadable_log=log)

    assert len(_unreadable_warnings(caplog)) == 1
    assert log == {(rule["rule_id"], "not_a_number"): ("sensor.girls",)}


def test_dynamic_rule_low_battery_recovery_while_another_has_no_state(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A reading during a run stopped by a missing sensor still counts."""
    rule = _low_battery_rule("sensor.front_lock", "sensor.hall")
    broken = _low_battery_snapshot(front_lock="locked", hall="80")
    recovered_hall_gone = _low_battery_snapshot(front_lock="90")
    log: dict[tuple[str, str], tuple[str, ...]] = {}

    with caplog.at_level("WARNING"):
        for snapshot in (broken, recovered_hall_gone, broken):
            evaluate_dynamic_rules(snapshot, [rule], unreadable_log=log)

    assert len(_unreadable_warnings(caplog)) == 2


def test_dynamic_rule_low_battery_sensor_taken_out_of_the_rule_is_forgotten(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A patch that drops a sensor and adds it back names it again."""
    both = _low_battery_rule("sensor.front_lock", "sensor.hall")
    hall_only = _low_battery_rule("sensor.hall")
    snapshot = _low_battery_snapshot(front_lock="locked", hall="80")
    log: dict[tuple[str, str], tuple[str, ...]] = {}

    with caplog.at_level("WARNING"):
        for rule in (both, hall_only, both):
            evaluate_dynamic_rules(snapshot, [rule], unreadable_log=log)

    assert len(_unreadable_warnings(caplog)) == 2


def test_dynamic_rule_low_battery_preview_does_not_use_up_the_warning(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """As for a dead rule (#960), a preview logs nothing (#1082)."""
    rule = _low_battery_rule("sensor.front_lock", "sensor.hall")
    snapshot = _low_battery_snapshot(front_lock="locked", hall="80")
    log: dict[tuple[str, str], tuple[str, ...]] = {}

    with caplog.at_level("WARNING"):
        evaluate_dynamic_rule(snapshot, rule)
        assert _unreadable_warnings(caplog) == []
        evaluate_dynamic_rules(snapshot, [rule], unreadable_log=log)
        assert len(_unreadable_warnings(caplog)) == 1


def test_dynamic_rule_low_battery_unreadable_warning_quotes_ids(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Stored IDs are rule text a model proposed: quoted and cut (#1082)."""
    forged = "sensor.a\nforged line"
    long_id = "sensor." + "x" * 300
    rule = _low_battery_rule(forged, long_id)
    snapshot = _snapshot(
        [
            _base_entity(forged, "sensor", "locked"),
            _base_entity(long_id, "sensor", "locked"),
        ],
        [],
        _DEAD_RULE_DERIVED,
    )
    log: dict[tuple[str, str], tuple[str, ...]] = {}

    with caplog.at_level("WARNING"):
        evaluate_dynamic_rules(snapshot, [rule], unreadable_log=log)
    (warning,) = _unreadable_warnings(caplog)
    assert "\n" not in warning
    assert "x" * 101 not in warning


def _low_battery_id(rule: dict[str, Any], **states: str) -> str:
    (finding,) = evaluate_dynamic_rules(_low_battery_snapshot(**states), [rule])
    return finding.anomaly_id


def test_dynamic_rule_low_battery_id_ignores_readings_that_do_not_change_who_is_low() -> (
    None
):
    """
    One low battery is one alert, whatever the readings do meanwhile.

    The anomaly id is what the pending prompt and the per-finding cooldown
    match on. It used to hash every listed sensor's reading, so the low
    sensor losing 1%, or another sensor moving or dropping off the network,
    made the same low battery a new alert.
    """
    rule = _low_battery_rule("sensor.elias", "sensor.girls")
    first = _low_battery_id(rule, elias="12", girls="80")

    assert _low_battery_id(rule, elias="11", girls="80") == first
    assert _low_battery_id(rule, elias="12", girls="79") == first
    assert _low_battery_id(rule, elias="12", girls="unavailable") == first
    assert _low_battery_id(rule, elias="12", girls="unknown") == first


def test_dynamic_rule_low_battery_id_changes_with_the_low_sensors_and_the_rule() -> (
    None
):
    """A second battery going low, another rule or threshold: a new alert."""
    rule = _low_battery_rule("sensor.elias", "sensor.girls")
    first = _low_battery_id(rule, elias="12", girls="80")

    assert _low_battery_id(rule, elias="12", girls="30") != first
    assert _low_battery_id(rule, elias="80", girls="12") != first
    assert _low_battery_id({**rule, "rule_id": "other"}, elias="12", girls="80") != (
        first
    )
    stricter = {**rule, "params": {**rule["params"], "threshold": 20}}
    assert _low_battery_id(stricter, elias="12", girls="80") != first


def test_dynamic_rule_low_battery_keeps_the_readings_in_the_evidence() -> None:
    """Leaving the readings out of the id does not take them off the alert."""
    rule = _low_battery_rule("sensor.elias", "sensor.girls")

    (finding,) = evaluate_dynamic_rules(
        _low_battery_snapshot(elias="12", girls="80"), [rule]
    )

    assert finding.evidence["sensor_levels"] == {
        "sensor.elias": 12.0,
        "sensor.girls": 80.0,
    }
    assert finding.evidence["sensor_states"] == {
        "sensor.elias": "12",
        "sensor.girls": "80",
    }


def test_dynamic_rule_unavailable_sensors_id_still_hashes_its_states() -> None:
    """Only the low-battery rule leaves sensor_states out of its id."""
    rule = _dead_rule("unavailable_sensors")
    snapshot = _dead_rule_snapshot(_DEAD_RULE_PRESENT, _DEAD_RULE_GONE)

    (finding,) = evaluate_dynamic_rules(snapshot, [rule])

    assert finding.anomaly_id == build_anomaly_id(
        rule["rule_id"], finding.triggering_entities, finding.evidence
    )


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

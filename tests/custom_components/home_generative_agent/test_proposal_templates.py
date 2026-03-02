# ruff: noqa: S101
"""Tests for proposal template normalization."""

from __future__ import annotations

from custom_components.home_generative_agent.sentinel.proposal_templates import (
    normalize_candidate,
)


def test_normalize_candidate_lock_only_security() -> None:
    candidate = {
        "candidate_id": "lock_candidate",
        "title": "Garage lock unlocked while home",
        "summary": "Detect lock left unlocked with someone present.",
        "pattern": "lock unlocked while home",
        "suggested_type": "security",
        "confidence_hint": 0.8,
        "evidence_paths": [
            "entities[entity_id=lock.garage_door_lock].state",
            "derived.anyone_home",
        ],
    }
    normalized = normalize_candidate(candidate)
    assert normalized is not None
    assert normalized.template_id == "unlocked_lock_when_home"
    assert normalized.rule_id == "unlocked_lock_when_home_lock_garage_door_lock"


def test_normalize_candidate_prefers_window_template_over_lock_rule() -> None:
    candidate = {
        "candidate_id": "window_candidate",
        "title": "Windows open while home",
        "summary": "Garage and playroom windows are open while occupants are present.",
        "pattern": "window open while home",
        "suggested_type": "security",
        "confidence_hint": 0.7,
        "evidence_paths": [
            "entities[entity_id=lock.garage_door_lock].state",
            "entities[entity_id=binary_sensor.playroom_window].state",
            "derived.anyone_home",
        ],
    }
    normalized = normalize_candidate(candidate)
    assert normalized is not None
    assert normalized.template_id == "open_entry_when_home"
    assert normalized.rule_id == "open_entry_when_home_window"


def test_normalize_candidate_window_night_home_template() -> None:
    candidate = {
        "candidate_id": "window_night_home",
        "title": "Open windows at night while someone is home",
        "summary": "Detect windows open during nighttime when someone is present.",
        "pattern": "window open at night while home",
        "suggested_type": "security_state",
        "confidence_hint": 0.7,
        "evidence_paths": [
            "entities[entity_id=binary_sensor.playroom_window].state",
            "derived.is_night",
            "derived.anyone_home",
        ],
    }
    normalized = normalize_candidate(candidate)
    assert normalized is not None
    assert normalized.template_id == "open_entry_at_night_when_home"
    assert normalized.rule_id == "open_entry_at_night_when_home_window"


def test_normalize_candidate_window_night_away_template() -> None:
    candidate = {
        "candidate_id": "window_night_away",
        "title": "Open windows while no one home at night",
        "summary": "Detect windows open while home is empty and it is nighttime.",
        "pattern": "window open away at night",
        "suggested_type": "security_risk",
        "confidence_hint": 0.7,
        "evidence_paths": [
            "entities[entity_id=binary_sensor.playroom_window].state",
            "derived.is_night",
        ],
    }
    normalized = normalize_candidate(candidate)
    assert normalized is not None
    assert normalized.template_id == "open_entry_at_night_while_away"
    assert normalized.rule_id == "open_entry_at_night_while_away_window"


def test_normalize_candidate_any_window_night_away_template() -> None:
    candidate = {
        "candidate_id": "any_window_night_away",
        "title": "Open windows while no one home at night",
        "summary": "Detects when any window sensor reports open while the house is empty and it is nighttime.",
        "pattern": "any window open while away at night",
        "suggested_type": "security_risk",
        "confidence_hint": 0.65,
        "evidence_paths": [
            "derived.is_night",
        ],
    }
    normalized = normalize_candidate(candidate)
    assert normalized is not None
    assert normalized.template_id == "open_any_window_at_night_while_away"
    assert normalized.rule_id == "open_any_window_at_night_while_away"


def test_normalize_candidate_unavailable_sensors_while_home_template() -> None:
    candidate = {
        "candidate_id": "sensor_unavailable_home",
        "title": "Unavailable sensors while home",
        "summary": "Detects any sensor reporting unavailable while someone is home.",
        "pattern": "derived.anyone_home AND sensor state unavailable",
        "suggested_type": "availability",
        "confidence_hint": 0.8,
        "evidence_paths": [
            "derived.anyone_home",
            "entities[entity_id=sensor.backyard_vmd3_0].state",
            "entities[entity_id=sensor.backyard_vmd4_camera1profile1].state",
        ],
    }
    normalized = normalize_candidate(candidate)
    assert normalized is not None
    assert normalized.template_id == "unavailable_sensors_while_home"
    assert normalized.rule_id == "unavailable_sensors_while_home"
    assert normalized.params == {
        "sensor_entity_ids": [
            "sensor.backyard_vmd3_0",
            "sensor.backyard_vmd4_camera1profile1",
        ]
    }


def test_normalize_candidate_unavailable_sensors_while_home_legacy_entity_ids() -> None:
    candidate = {
        "candidate_id": "sensor_unavailable_home",
        "title": "Unavailable sensors while home",
        "summary": "Detects any sensor reporting unavailable while someone is home.",
        "pattern": "derived.anyone_home AND sensor state unavailable",
        "suggested_type": "availability",
        "confidence_hint": 0.8,
        "evidence_paths": [
            "derived.anyone_home",
            "entities[entity_id=backyard_vmd3_0].state",
            "entities[entity_id=backyard_vmd4_camera1profile1].state",
        ],
    }
    normalized = normalize_candidate(candidate)
    assert normalized is not None
    assert normalized.template_id == "unavailable_sensors_while_home"
    assert normalized.params == {
        "sensor_entity_ids": [
            "backyard_vmd3_0",
            "backyard_vmd4_camera1profile1",
        ]
    }


def test_normalize_candidate_unavailable_sensors_template_issue_223() -> None:
    candidate = {
        "candidate_id": "backyard_sensors_unavailable",
        "title": "Backyard sensors unavailable",
        "summary": (
            "Backyard motion sensors are reporting unavailable, which could indicate "
            "a malfunction or connectivity issue."
        ),
        "pattern": (
            "entities[entity_id=backyard_vmd3_0].state == 'unavailable' AND "
            "entities[entity_id=backyard_vmd4_camera1profile1].state == 'unavailable'"
        ),
        "suggested_type": "availability",
        "confidence_hint": 0.6,
        "evidence_paths": [
            "entities[entity_id=backyard_vmd3_0].state",
            "entities[entity_id=backyard_vmd4_camera1profile1].state",
        ],
    }
    normalized = normalize_candidate(candidate)
    assert normalized is not None
    assert normalized.template_id == "unavailable_sensors"
    assert normalized.rule_id == "backyard_sensors_unavailable"
    assert normalized.params == {
        "sensor_entity_ids": [
            "backyard_vmd3_0",
            "backyard_vmd4_camera1profile1",
        ]
    }


def test_normalize_candidate_low_battery_sensors_issue_236() -> None:
    candidate = {
        "candidate_id": "low_battery_room_sensors_v1",
        "title": "Low battery on room sensors",
        "summary": "Room T/H sensors show low battery levels.",
        "pattern": (
            "Notify when any of [sensor.elias_t_h_battery, "
            "sensor.girls_t_h_battery] is at or below 40%."
        ),
        "suggested_type": "maintenance",
        "confidence_hint": 0.62,
        "evidence_paths": [
            "entities[entity_id=sensor.elias_t_h_battery].state",
            "entities[entity_id=sensor.girls_t_h_battery].state",
        ],
    }
    normalized = normalize_candidate(candidate)
    assert normalized is not None
    assert normalized.template_id == "low_battery_sensors"
    assert normalized.rule_id == "low_battery_room_sensors_v1"
    assert normalized.params == {
        "sensor_entity_ids": [
            "sensor.elias_t_h_battery",
            "sensor.girls_t_h_battery",
        ],
        "threshold": 40.0,
    }


def test_normalize_candidate_motion_alarm_disarmed_home_issue_225() -> None:
    candidate = {
        "candidate_id": "motion_frontgate_disarmed_home",
        "title": "Motion detected at front gate while alarm disarmed and home present",
        "summary": (
            "Motion is detected at the front gate while the alarm is disarmed and a "
            "person is at home."
        ),
        "pattern": (
            "frontgate_vmd3_0.state == 'on' AND "
            "frontgate_vmd4_camera1profile1.state == 'on' AND "
            "alarm_control_panel.home_alarm.state == 'disarmed' AND "
            "person.lindo_st_angel.state == 'home'"
        ),
        "suggested_type": "security",
        "confidence_hint": 0.75,
        "evidence_paths": [
            "entities[entity_id=frontgate_vmd3_0].state",
            "entities[entity_id=frontgate_vmd4_camera1profile1].state",
            "entities[entity_id=alarm_control_panel.home_alarm].state",
            "entities[entity_id=person.lindo_st_angel].state",
        ],
    }
    normalized = normalize_candidate(candidate)
    assert normalized is not None
    assert normalized.template_id == "motion_while_alarm_disarmed_and_home_present"
    assert normalized.rule_id == "motion_frontgate_disarmed_home"
    assert normalized.params == {
        "alarm_entity_id": "alarm_control_panel.home_alarm",
        "motion_entity_ids": [
            "frontgate_vmd3_0",
            "frontgate_vmd4_camera1profile1",
        ],
        "home_entity_ids": ["person.lindo_st_angel"],
    }


def test_normalize_candidate_motion_night_alarm_disarmed_issue_235() -> None:
    candidate = {
        "candidate_id": "motion_at_night_disarmed",
        "title": "Motion detected at night while alarm disarmed",
        "summary": (
            "Detects any motion sensor activation during nighttime when the home "
            "alarm is disarmed."
        ),
        "pattern": "motion active & night & alarm disarmed",
        "suggested_type": "security",
        "confidence_hint": 0.8,
        "evidence_paths": [
            "derived.is_night",
            "entities[entity_id=alarm_control_panel.home_alarm].state",
            "entities[entity_id=binary_sensor.backyard_vmd3_0].state",
            "entities[entity_id=binary_sensor.backyard_vmd4_camera1profile1].state",
            "entities[entity_id=person.lindo_st_angel].state",
        ],
    }
    normalized = normalize_candidate(candidate)
    assert normalized is not None
    assert normalized.template_id == "motion_detected_at_night_while_alarm_disarmed"
    assert normalized.rule_id == "motion_at_night_disarmed"
    assert normalized.params == {
        "alarm_entity_id": "alarm_control_panel.home_alarm",
        "motion_entity_ids": [
            "binary_sensor.backyard_vmd3_0",
            "binary_sensor.backyard_vmd4_camera1profile1",
        ],
        "required_entity_ids": ["person.lindo_st_angel"],
    }


def test_normalize_candidate_unknown_person_camera_when_home_issue_278() -> None:
    candidate = {
        "candidate_id": "unknown_person_camera_when_home",
        "title": "Unknown person detected by camera while someone is home",
        "summary": (
            "A camera reports an unknown person while a person is present at home."
        ),
        "pattern": "recognized_people contains 'Indeterminate' and derived.anyone_home",
        "suggested_type": "security",
        "confidence_hint": 0.7,
        "evidence_paths": [
            "camera_activity[entity_id=camera.backyard].recognized_people",
            "derived.anyone_home",
        ],
    }
    normalized = normalize_candidate(candidate)
    assert normalized is not None
    assert normalized.template_id == "unknown_person_camera_when_home"
    assert normalized.rule_id == "unknown_person_camera_when_home_camera_backyard"
    assert normalized.params == {"camera_entity_id": "camera.backyard"}
    assert normalized.is_sensitive is False


def test_normalize_candidate_unknown_person_camera_when_home_rule_id_is_deterministic() -> None:
    candidate = {
        "candidate_id": "different_candidate_id_same_semantics",
        "title": "Unknown person detected by camera while someone is home",
        "summary": (
            "A camera reports an unknown person while a person is present at home."
        ),
        "pattern": "recognized_people contains 'Indeterminate' and derived.anyone_home",
        "suggested_type": "security",
        "confidence_hint": 0.7,
        "evidence_paths": [
            "camera_activity[entity_id=camera.backyard].recognized_people",
            "derived.anyone_home",
        ],
    }
    normalized = normalize_candidate(candidate)
    assert normalized is not None
    assert normalized.template_id == "unknown_person_camera_when_home"
    assert normalized.rule_id == "unknown_person_camera_when_home_camera_backyard"


def test_normalize_candidate_unknown_person_camera_no_home_rule_id_is_deterministic() -> None:
    candidate = {
        "candidate_id": "arbitrary_unknown_person_candidate",
        "title": "Unknown person detected by camera while no one is home",
        "summary": "A camera reports an unknown person while the home is unoccupied.",
        "pattern": "recognized_people contains 'Indeterminate' and no one home",
        "suggested_type": "security",
        "confidence_hint": 0.85,
        "evidence_paths": [
            "camera_activity[entity_id=camera.backyard].recognized_people",
        ],
    }
    normalized = normalize_candidate(candidate)
    assert normalized is not None
    assert normalized.template_id == "unknown_person_camera_no_home"
    assert normalized.rule_id == "unknown_person_camera_no_home_camera_backyard"


def test_normalize_candidate_unknown_person_camera_when_home_from_entities_path() -> None:
    candidate = {
        "candidate_id": "entities_path_unknown_person_home",
        "title": "Unknown person detected by front gate camera while occupants at home",
        "summary": "An unidentified person is seen while the house is occupied.",
        "pattern": "unknown person face while present",
        "suggested_type": "security",
        "confidence_hint": 0.7,
        "evidence_paths": [
            "entities[entity_id=camera.front_gate].attributes.last_event",
            "derived.anyone_home",
        ],
    }
    normalized = normalize_candidate(candidate)
    assert normalized is not None
    assert normalized.template_id == "unknown_person_camera_when_home"
    assert normalized.rule_id == "unknown_person_camera_when_home_camera_front_gate"
    assert normalized.params == {"camera_entity_id": "camera.front_gate"}


def test_normalize_candidate_unknown_person_camera_no_home_from_entities_path() -> None:
    candidate = {
        "candidate_id": "entities_path_unknown_person_away",
        "title": "Unknown person detected by front gate camera while no one is home",
        "summary": "An unidentified person is seen while the home is unoccupied.",
        "pattern": "unknown person face while no one home",
        "suggested_type": "security",
        "confidence_hint": 0.85,
        "evidence_paths": [
            "entities[entity_id=camera.front_gate].attributes.last_event",
        ],
    }
    normalized = normalize_candidate(candidate)
    assert normalized is not None
    assert normalized.template_id == "unknown_person_camera_no_home"
    assert normalized.rule_id == "unknown_person_camera_no_home_camera_front_gate"
    assert normalized.params == {"camera_entity_id": "camera.front_gate"}

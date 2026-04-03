"""Tests for proposal template normalization."""

from __future__ import annotations

from custom_components.home_generative_agent.sentinel.proposal_templates import (
    explain_normalize_candidate,
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
    assert "lock.lock" in normalized.suggested_actions


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


def test_normalize_candidate_cover_entry_adds_service_action() -> None:
    candidate = {
        "candidate_id": "cover_night_home",
        "title": "Patio cover open at night while home",
        "summary": "Detect patio cover left open during nighttime while someone is home.",
        "pattern": "cover open at night while home",
        "suggested_type": "security_state",
        "confidence_hint": 0.7,
        "evidence_paths": [
            "entities[entity_id=cover.patio_door].state",
            "derived.is_night",
            "derived.anyone_home",
        ],
    }
    normalized = normalize_candidate(candidate)
    assert normalized is not None
    assert normalized.template_id == "open_entry_at_night_when_home"
    assert "cover.close_cover" in normalized.suggested_actions


def test_normalize_candidate_advisory_only_template_has_no_service_action() -> None:
    candidate = {
        "candidate_id": "battery_room_sensors_v1",
        "title": "Low battery on room sensors",
        "summary": "Room sensors show low battery levels.",
        "pattern": "battery below 40%",
        "suggested_type": "maintenance",
        "confidence_hint": 0.62,
        "evidence_paths": [
            "entities[entity_id=sensor.elias_t_h_battery].state",
        ],
    }
    normalized = normalize_candidate(candidate)
    assert normalized is not None
    assert not any("." in action for action in normalized.suggested_actions)


def test_explain_normalize_candidate_returns_missing_required_entities() -> None:
    result = explain_normalize_candidate(
        {
            "candidate_id": "missing_lock",
            "title": "Front lock unlocked while home",
            "summary": "Detect unlocked lock with someone present.",
            "pattern": "lock unlocked while home",
            "suggested_type": "security",
            "confidence_hint": 0.8,
            "evidence_paths": ["derived.anyone_home"],
        }
    )
    assert result.normalized is None
    assert result.reason_code == "missing_required_entities"


def test_explain_normalize_candidate_returns_no_matching_entity_types() -> None:
    result = explain_normalize_candidate(
        {
            "candidate_id": "no_match",
            "title": "General weirdness",
            "summary": "Something odd happened.",
            "pattern": "odd pattern",
            "suggested_type": "misc",
            "confidence_hint": 0.3,
            "evidence_paths": [],
        }
    )
    assert result.normalized is None
    assert result.reason_code == "no_matching_entity_types"


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


def test_normalize_candidate_unknown_person_camera_when_home_rule_id_is_deterministic() -> (
    None
):
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


def test_normalize_candidate_unknown_person_camera_no_home_rule_id_is_deterministic() -> (
    None
):
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


def test_normalize_candidate_unknown_person_camera_when_home_from_entities_path() -> (
    None
):
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


def test_normalize_candidate_unknown_person_camera_no_home_any_camera() -> None:
    candidate = {
        "candidate_id": "candidate_unknown_person_camera_no_home",
        "title": "Unknown Person Detected by Camera When No One Home",
        "summary": "Triggers when a camera records an unknown person while no occupants are present at home.",
        "pattern": "unknown person while no occupants present",
        "suggested_type": "security",
        "confidence_hint": 0.8,
        "evidence_paths": ["derived.is_night"],
    }
    normalized = normalize_candidate(candidate)
    assert normalized is not None
    assert normalized.template_id == "unknown_person_camera_no_home"
    assert normalized.rule_id == "unknown_person_camera_no_home_any_camera"
    assert normalized.params == {"camera_selector": "any"}


def test_normalize_candidate_unknown_person_camera_when_home_any_camera_indeterminate() -> (
    None
):
    candidate = {
        "candidate_id": "unknown_person_camera_day",
        "title": "Unknown Person Detected by Camera While Home During Day",
        "summary": "Detects any camera that recognizes an unknown or indeterminate person while residents are home and it is daytime.",
        "pattern": "indeterminate face while residents present",
        "suggested_type": "security",
        "confidence_hint": 0.7,
        "evidence_paths": ["derived.anyone_home"],
    }
    normalized = normalize_candidate(candidate)
    assert normalized is not None
    assert normalized.template_id == "unknown_person_camera_when_home"
    assert normalized.rule_id == "unknown_person_camera_when_home_any_camera"
    assert normalized.params == {"camera_selector": "any"}


def test_normalize_candidate_unknown_person_camera_infers_camera_from_candidate_id() -> (
    None
):
    candidate = {
        "candidate_id": "unknown_person_camera_home_frontgate",
        "title": "Unknown person detected while residents are home",
        "summary": "Unknown person event near the front gate while occupants are present.",
        "pattern": "unknown person while home",
        "suggested_type": "security",
        "confidence_hint": 0.7,
        "evidence_paths": ["derived.anyone_home"],
    }
    normalized = normalize_candidate(candidate)
    assert normalized is not None
    assert normalized.template_id == "unknown_person_camera_when_home"
    assert normalized.rule_id == "unknown_person_camera_when_home_camera_frontgate"
    assert normalized.params == {"camera_entity_id": "camera.frontgate"}


# ---------------------------------------------------------------------------
# Fix C: _find_entry_entity_ids domain-prefix resolution
# ---------------------------------------------------------------------------


def test_normalize_candidate_entry_without_domain_prefix_resolves() -> None:
    """Entity IDs without domain prefix containing entry keywords should normalize."""
    candidate = {
        "candidate_id": "windows_open_while_away",
        "title": "Windows Open While Away",
        "summary": "Detects any window sensor reporting open while no occupants are home.",
        "pattern": "window_open AND not anyone_home",
        "suggested_type": "security",
        "confidence_hint": 0.8,
        "evidence_paths": [
            "entities[entity_id=breakfast_nook_side_right_window].state",
            "entities[entity_id=garage_and_play_room_windows].state",
            "derived.anyone_home",
        ],
    }
    normalized = normalize_candidate(candidate)
    assert normalized is not None
    assert normalized.template_id == "open_entry_while_away"
    assert "breakfast_nook_side_right_window" in normalized.params["entry_entity_ids"]
    assert "garage_and_play_room_windows" in normalized.params["entry_entity_ids"]


# ---------------------------------------------------------------------------
# unlocked_lock_while_away
# ---------------------------------------------------------------------------


def test_normalize_candidate_lock_away_routes_to_unlocked_lock_while_away() -> None:
    candidate = {
        "candidate_id": "garage_door_lock_unlocked_while_away",
        "title": "Garage door lock unlocked while away",
        "summary": "Alerts when the garage door lock is unlocked while no one is home.",
        "pattern": "anyone_home=false AND lock_state=unlocked",
        "suggested_type": "security",
        "confidence_hint": 0.85,
        "evidence_paths": [
            "entities[entity_id=lock.garage_door_lock].state",
            "derived.anyone_home",
        ],
    }
    normalized = normalize_candidate(candidate)
    assert normalized is not None
    assert normalized.template_id == "unlocked_lock_while_away"
    assert normalized.params == {"lock_entity_id": "lock.garage_door_lock"}
    assert normalized.severity == "high"
    assert "lock.lock" in normalized.suggested_actions


def test_normalize_candidate_lock_without_away_still_routes_to_when_home() -> None:
    """Lock candidate with no presence signal should still route to unlocked_lock_when_home."""
    candidate = {
        "candidate_id": "lock_candidate_no_presence",
        "title": "Front lock unlocked",
        "summary": "The front door lock is unlocked.",
        "pattern": "lock unlocked",
        "suggested_type": "security",
        "confidence_hint": 0.5,
        "evidence_paths": [
            "entities[entity_id=lock.front_door].state",
        ],
    }
    normalized = normalize_candidate(candidate)
    assert normalized is not None
    assert normalized.template_id == "unlocked_lock_when_home"


# ---------------------------------------------------------------------------
# alarm_state_mismatch
# ---------------------------------------------------------------------------


def test_normalize_candidate_alarm_armed_home_while_away() -> None:
    candidate = {
        "candidate_id": "alarm_armed_home_while_away",
        "title": "Alarm Armed Home While Away",
        "summary": "Security system is in armed home mode despite no occupants.",
        "pattern": "alarm_state == armed_home AND anyone_home == false",
        "suggested_type": "security",
        "confidence_hint": 0.9,
        "evidence_paths": [
            "entities[entity_id=alarm_control_panel.home_alarm].state",
            "derived.anyone_home",
        ],
    }
    normalized = normalize_candidate(candidate)
    assert normalized is not None
    assert normalized.template_id == "alarm_state_mismatch"
    assert normalized.params["alarm_entity_id"] == "alarm_control_panel.home_alarm"
    assert normalized.params["alarm_state"] == "armed_home"
    assert normalized.params["expected_presence"] == "away"


# ---------------------------------------------------------------------------
# entity_state_duration - lock variant
# ---------------------------------------------------------------------------


def test_normalize_candidate_lock_unlocked_duration() -> None:
    candidate = {
        "candidate_id": "extended_garage_door_unlock_time",
        "title": "Extended Garage Door Unlock Time",
        "summary": "Garage door lock remains unlocked for an extended duration.",
        "pattern": "lock_state == unlocked AND (now - last_changed) > threshold_hours",
        "suggested_type": "security",
        "confidence_hint": 0.85,
        "evidence_paths": [
            "entities[entity_id=lock.garage_door_lock].state",
            "entities[entity_id=lock.garage_door_lock].last_changed",
            "derived.now",
        ],
    }
    normalized = normalize_candidate(candidate)
    assert normalized is not None
    assert normalized.template_id == "entity_state_duration"
    assert normalized.params["entity_id"] == "lock.garage_door_lock"
    assert normalized.params["target_state"] == "unlocked"
    assert isinstance(normalized.params["threshold_hours"], float)


# ---------------------------------------------------------------------------
# entity_state_duration - entry variant
# ---------------------------------------------------------------------------


def test_normalize_candidate_window_open_duration() -> None:
    candidate = {
        "candidate_id": "window_open_for_extended_duration",
        "title": "Window Open for Extended Duration",
        "summary": "Window sensor has been in the open state for a prolonged duration.",
        "pattern": "entry state == on AND (now - last_changed) > 2 hours",
        "suggested_type": "security",
        "confidence_hint": 0.9,
        "evidence_paths": [
            "entities[entity_id=binary_sensor.garage_and_play_room_windows].state",
            "entities[entity_id=binary_sensor.garage_and_play_room_windows].last_changed",
            "derived.now",
        ],
    }
    normalized = normalize_candidate(candidate)
    assert normalized is not None
    assert normalized.template_id == "entity_state_duration"
    assert (
        normalized.params["entity_id"] == "binary_sensor.garage_and_play_room_windows"
    )
    assert normalized.params["target_state"] == "on"
    assert normalized.params["threshold_hours"] == 2.0


# ---------------------------------------------------------------------------
# sensor_threshold_condition
# ---------------------------------------------------------------------------


def test_normalize_candidate_power_sensor_threshold_while_home() -> None:
    candidate = {
        "candidate_id": "high_microwave_power_while_home",
        "title": "High Microwave Power While Home",
        "summary": "Microwave power exceeds 1000W while someone is home.",
        "pattern": "sensor.microwave_switch_0_power > 1000 AND derived.anyone_home = true",
        "suggested_type": "energy",
        "confidence_hint": 0.7,
        "evidence_paths": [
            "entities[entity_id=sensor.microwave_switch_0_power].state",
            "derived.anyone_home",
        ],
    }
    normalized = normalize_candidate(candidate)
    assert normalized is not None
    assert normalized.template_id == "sensor_threshold_condition"
    assert normalized.params["sensor_entity_id"] == "sensor.microwave_switch_0_power"
    assert normalized.params["threshold"] == 1000.0
    assert normalized.params["require_home"] is True
    assert normalized.params["require_away"] is False


def test_normalize_candidate_power_sensor_threshold_at_night() -> None:
    candidate = {
        "candidate_id": "washing_machine_power_usage_during_night_hours",
        "title": "Washing Machine Power Usage During Night Hours",
        "summary": "Washing machine drawing 112W during the night.",
        "pattern": "night=1 AND appliance_power > 50",
        "suggested_type": "energy",
        "confidence_hint": 0.85,
        "evidence_paths": [
            "entities[entity_id=sensor.washing_machine_switch_0_power].state",
            "derived.is_night",
        ],
    }
    normalized = normalize_candidate(candidate)
    assert normalized is not None
    assert normalized.template_id == "sensor_threshold_condition"
    assert (
        normalized.params["sensor_entity_id"] == "sensor.washing_machine_switch_0_power"
    )
    assert normalized.params["threshold"] == 50.0
    assert normalized.params["require_night"] is True


# ---------------------------------------------------------------------------
# entity_staleness
# ---------------------------------------------------------------------------


def test_normalize_candidate_person_tracking_staleness() -> None:
    candidate = {
        "candidate_id": "person_tracking_staleness",
        "title": "Occupant Tracking Device Offline",
        "summary": "Primary occupant tracking device not updated for over 40 hours.",
        "pattern": "person.lindo_st_angel last_changed stale > 40 hours",
        "suggested_type": "availability",
        "confidence_hint": 0.7,
        "evidence_paths": [
            "entities[entity_id=person.lindo_st_angel].last_changed",
            "derived.now",
        ],
    }
    normalized = normalize_candidate(candidate)
    assert normalized is not None
    assert normalized.template_id == "entity_staleness"
    assert normalized.params["entity_id"] == "person.lindo_st_angel"
    assert normalized.params["max_stale_hours"] == 40.0


# ---------------------------------------------------------------------------
# multiple_entries_open_count
# ---------------------------------------------------------------------------


def test_normalize_candidate_multiple_entries_open_simultaneously() -> None:
    candidate = {
        "candidate_id": "multiple_openings_simultaneous",
        "title": "Multiple Entry Points Open Simultaneously",
        "summary": "Multiple opening sensors activate at the same time while home.",
        "pattern": "count(open_sensors) > 3 AND home == true",
        "suggested_type": "security",
        "confidence_hint": 0.85,
        "evidence_paths": [
            "entities[entity_id=binary_sensor.breakfast_nook_side_right_window].state",
            "entities[entity_id=binary_sensor.family_room_right_window].state",
            "entities[entity_id=binary_sensor.garage_and_play_room_windows].state",
            "derived.anyone_home",
        ],
    }
    normalized = normalize_candidate(candidate)
    assert normalized is not None
    assert normalized.template_id == "multiple_entries_open_count"
    assert len(normalized.params["entry_entity_ids"]) == 3
    assert normalized.params["require_home"] is True
    assert normalized.params["require_away"] is False


# ---------------------------------------------------------------------------
# entity_ids contains path format (discovery engine output)
# ---------------------------------------------------------------------------


def test_normalize_candidate_entity_ids_contains_path_format_lock() -> None:
    """Discovery engine stores paths as 'entities[entity_ids contains ...]'."""
    candidate = {
        "candidate_id": "garage_lock_away",
        "title": "Garage door lock unlocked while no one home",
        "summary": "Lock left unlocked while away.",
        "pattern": "lock_state=unlocked AND anyone_home=false",
        "suggested_type": "security",
        "confidence_hint": 0.85,
        "evidence_paths": [
            "derived.anyone_home",
            "entities[entity_ids contains lock.garage_door_lock].state",
        ],
    }
    normalized = normalize_candidate(candidate)
    assert normalized is not None
    assert normalized.template_id == "unlocked_lock_while_away"
    assert normalized.params["lock_entity_id"] == "lock.garage_door_lock"


def test_normalize_candidate_entity_ids_contains_path_format_sensor() -> None:
    """Power sensor via discovery 'entity_ids contains' path routes to sensor_threshold_condition."""
    candidate = {
        "candidate_id": "washing_machine_night",
        "title": "Washing Machine Power Usage During Night Hours",
        "summary": "The washing machine is drawing significant power (112.6W) during the night.",
        "pattern": "night=1|appliance_power>50",
        "suggested_type": "anomaly",
        "confidence_hint": 0.85,
        "evidence_paths": [
            "derived.is_night",
            "entities[entity_ids contains sensor.washing_machine_switch_0_power].state",
        ],
    }
    normalized = normalize_candidate(candidate)
    assert normalized is not None
    assert normalized.template_id == "sensor_threshold_condition"
    assert (
        normalized.params["sensor_entity_id"] == "sensor.washing_machine_switch_0_power"
    )
    assert normalized.params["require_night"] is True


# ---------------------------------------------------------------------------
# Regression: power keyword in entity_id only (not in LLM text)
# ---------------------------------------------------------------------------


def test_normalize_candidate_power_signal_from_entity_id_only() -> None:
    """
    Power keyword in entity_id (not in text) should still route to baseline_deviation.

    The LLM described the candidate as "washing machine active while away at night"
    — no power/energy keyword in title/summary/pattern — but the entity ID
    sensor.washing_machine_switch_0_power contains "power".  Before the fix this
    fell through to unsupported_pattern.
    """
    candidate = {
        "candidate_id": "candidate_washing_machine_active_away_night",
        "title": "Washing Machine Active While Away at Night",
        "summary": "The washing machine is running while no one is home at night.",
        "pattern": "is_night AND presence=away AND washing_machine=active",
        "suggested_type": "appliance",
        "confidence_hint": 0.7,
        "evidence_paths": [
            "entities[entity_id=sensor.washing_machine_switch_0_power].state",
            "derived.is_night",
            "derived.anyone_home",
        ],
    }
    normalized = normalize_candidate(candidate)
    assert normalized is not None
    # No numeric threshold in text → falls back to baseline_deviation
    assert normalized.template_id == "baseline_deviation"
    assert normalized.params["entity_id"] == "sensor.washing_machine_switch_0_power"


# ---------------------------------------------------------------------------
# Regression: high_energy_consumption_night — no numeric threshold in text
# ---------------------------------------------------------------------------


def test_normalize_candidate_power_sensor_no_numeric_threshold_falls_back_to_baseline() -> (
    None
):
    """Candidate with power signal but no numeric threshold normalizes via baseline_deviation."""
    candidate = {
        "candidate_id": "high_energy_consumption_night",
        "title": "High Energy Consumption at Night",
        "summary": "Anomalously high energy consumption detected during overnight hours.",
        "pattern": "is_night AND sensor.power_meter > baseline",
        "suggested_type": "energy",
        "confidence_hint": 0.75,
        "evidence_paths": [
            "derived.is_night",
            "entities[entity_ids contains sensor.power_meter_energy].state",
        ],
    }
    normalized = normalize_candidate(candidate)
    assert normalized is not None
    assert normalized.template_id == "baseline_deviation"
    assert normalized.params["entity_id"] == "sensor.power_meter_energy"


# ---------------------------------------------------------------------------
# Regression: alarm disarmed during external threat — presence is "any"
# ---------------------------------------------------------------------------


def test_normalize_candidate_alarm_disarmed_any_presence_routes_to_alarm_state_mismatch() -> (
    None
):
    """Alarm disarmed + no presence signal normalizes to alarm_state_mismatch with home default."""
    candidate = {
        "candidate_id": "alarm_disarmed_during_external_threat",
        "title": "Alarm Disarmed During External Threat",
        "summary": "Security alarm is disarmed while an unrecognized person is detected.",
        "pattern": "alarm_state == disarmed AND camera_activity.recognized_people == []",
        "suggested_type": "security",
        "confidence_hint": 0.9,
        "evidence_paths": [
            "entities[entity_id=alarm_control_panel.home_alarm].state",
        ],
    }
    normalized = normalize_candidate(candidate)
    assert normalized is not None
    assert normalized.template_id == "alarm_state_mismatch"
    assert normalized.params["alarm_entity_id"] == "alarm_control_panel.home_alarm"
    assert normalized.params["alarm_state"] == "disarmed"
    assert normalized.params["expected_presence"] == "home"


# ---------------------------------------------------------------------------
# Regression: window_open_duration_exceeded — no entry entity IDs in evidence
# ---------------------------------------------------------------------------


def test_normalize_candidate_window_open_duration_no_entry_ids_falls_back() -> None:
    """Window open duration candidate with no entity IDs in evidence uses selector fallback."""
    candidate = {
        "candidate_id": "window_open_duration_exceeded",
        "title": "Window Open Duration Exceeded",
        "summary": "A window has been open for an extended duration.",
        "pattern": "window_state == open AND open_duration > threshold",
        "suggested_type": "security",
        "confidence_hint": 0.7,
        "evidence_paths": [
            "derived.anyone_home",
        ],
    }
    normalized = normalize_candidate(candidate)
    assert normalized is not None
    assert normalized.template_id == "open_any_window_at_night_while_away"
    assert normalized.params["entry_selector"] == "window"


# ---------------------------------------------------------------------------
# Dot-notation evidence path extraction
# ---------------------------------------------------------------------------


def test_normalize_candidate_power_sensor_dot_notation_evidence_paths() -> None:
    """Sensor entity IDs in dot-notation paths (e.g. sensor.foo.state) are extracted."""
    candidate = {
        "candidate_id": "high_energy_consumption_night",
        "title": "High Energy Consumption at Night",
        "summary": "Anomalously high energy consumption detected during overnight hours.",
        "pattern": "is_night AND sensor.power_meter > baseline",
        "suggested_type": "energy",
        "confidence_hint": 0.75,
        "evidence_paths": [
            "derived.is_night",
            "sensor.power_meter_energy.state",
        ],
    }
    normalized = normalize_candidate(candidate)
    assert normalized is not None
    assert normalized.template_id == "baseline_deviation"
    assert normalized.params["entity_id"] == "sensor.power_meter_energy"


def test_normalize_candidate_lock_battery_dot_notation_evidence_paths() -> None:
    """Lock entity IDs in dot-notation paths (e.g. lock.foo.battery_level) are extracted."""
    candidate = {
        "candidate_id": "playroom_lock_battery_low",
        "title": "Playroom Lock Battery Low",
        "summary": "The playroom door lock battery is below 20%.",
        "pattern": "lock.playroom_door_lock.battery_level < 20",
        "suggested_type": "maintenance",
        "confidence_hint": 0.9,
        "evidence_paths": [
            "lock.playroom_door_lock.battery_level",
            "derived.anyone_home",
        ],
    }
    normalized = normalize_candidate(candidate)
    assert normalized is not None
    assert normalized.template_id == "low_battery_sensors"
    assert "lock.playroom_door_lock" in normalized.params.get("sensor_entity_ids", [])

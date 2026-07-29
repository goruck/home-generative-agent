# ruff: noqa: S101
"""Tests for proposal template normalization."""

from __future__ import annotations

from custom_components.home_generative_agent.sentinel.proposal_templates import (
    _entry_kind,
    _extract_entity_id_from_evidence_path,
    _extract_threshold_numeric,
    _find_battery_sensor_entity_ids,
    _find_camera_id,
    _find_entry_entity_ids,
    _find_sensor_entity_ids,
    _find_text_entry_entity_ids,
    _has_duration_signal,
    _presence_signal,
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
    assert normalized.rule_id == "sensor_unavailable_home"
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


def test_normalize_candidate_unavailable_binary_occupancy_sensors_issue_514() -> None:
    candidate = {
        "candidate_id": "multiple_occupancy_sensors_unavailable",
        "title": "Multiple Occupancy Sensors Unavailable",
        "summary": (
            "Several occupancy sensors are simultaneously unavailable, "
            "potentially indicating a system outage."
        ),
        "pattern": "state_unavailable",
        "suggested_type": "availability",
        "confidence_hint": 0.3,
        "evidence_paths": [
            (
                "entities[entity_ids contains "
                '"binary_sensor.0x00124b0010b0a987_occupancy"].state'
            ),
            (
                "entities[entity_ids contains "
                '"binary_sensor.0xfc012cfffef62ae0_occupancy"].state'
            ),
            (
                "entities[entity_ids contains "
                '"binary_sensor.smart_presence_sensor_obsazenost"].state'
            ),
        ],
    }
    normalized = normalize_candidate(candidate)
    assert normalized is not None
    assert normalized.template_id == "unavailable_sensors"
    assert normalized.rule_id == "multiple_occupancy_sensors_unavailable"
    assert normalized.params == {
        "sensor_entity_ids": [
            "binary_sensor.0x00124b0010b0a987_occupancy",
            "binary_sensor.0xfc012cfffef62ae0_occupancy",
            "binary_sensor.smart_presence_sensor_obsazenost",
        ]
    }
    assert normalized.confidence == 0.3


def test_normalize_candidate_presence_sensor_wording_stays_presence_agnostic() -> None:
    """'Presence sensors' prose must not read as a someone-is-home condition."""
    normalized = normalize_candidate(
        {
            "candidate_id": "presence_sensors_unavailable",
            "title": "Presence Sensors Unavailable",
            "summary": (
                "Several presence sensors are unavailable, "
                "potentially indicating a hub outage."
            ),
            "pattern": "state_unavailable",
            "suggested_type": "availability",
            "confidence_hint": 0.4,
            "evidence_paths": [
                "entities[entity_id=binary_sensor.hall_presence].state",
                "entities[entity_id=binary_sensor.kitchen_presence].state",
            ],
        }
    )
    assert normalized is not None
    assert normalized.template_id == "unavailable_sensors"
    assert normalized.rule_id == "presence_sensors_unavailable"


def test_normalize_candidate_incidental_home_wording_stays_presence_agnostic() -> None:
    """'Around the home' is location color, not an occupancy condition."""
    normalized = normalize_candidate(
        {
            "candidate_id": "occupancy_sensors_hub_outage",
            "title": "Occupancy Sensors Unavailable",
            "summary": (
                "Several occupancy sensors around the home are unavailable, "
                "suggesting a Zigbee hub outage."
            ),
            "pattern": "state_unavailable",
            "suggested_type": "availability",
            "confidence_hint": 0.4,
            "evidence_paths": [
                "entities[entity_id=binary_sensor.hall_occupancy].state",
                "entities[entity_id=binary_sensor.garage_occupancy].state",
            ],
        }
    )
    assert normalized is not None
    assert normalized.template_id == "unavailable_sensors"
    assert normalized.rule_id == "occupancy_sensors_hub_outage"


def test_normalize_candidate_unavailable_entry_named_sensor_is_availability() -> None:
    """An unavailable door contact must not become a dead open_entry rule."""
    normalized = normalize_candidate(
        {
            "candidate_id": "front_door_contact_unavailable",
            "title": "Front Door Contact Sensor Unavailable",
            "summary": (
                "The front door contact sensor is reporting unavailable, "
                "so entry monitoring is degraded."
            ),
            "pattern": "state_unavailable",
            "suggested_type": "availability",
            "confidence_hint": 0.5,
            "evidence_paths": [
                "entities[entity_id=binary_sensor.front_door_contact].state",
            ],
        }
    )
    assert normalized is not None
    assert normalized.template_id == "unavailable_sensors"
    assert normalized.rule_id == "front_door_contact_unavailable"


def test_normalize_candidate_unavailable_battery_named_sensor_is_availability() -> None:
    """An unavailable battery sensor must not become a dead low_battery rule."""
    normalized = normalize_candidate(
        {
            "candidate_id": "bedroom_battery_sensor_unavailable",
            "title": "Bedroom Battery Sensor Unavailable",
            "summary": "The bedroom sensor battery level is reporting unavailable.",
            "pattern": "state_unavailable",
            "suggested_type": "availability",
            "confidence_hint": 0.5,
            "evidence_paths": [
                "entities[entity_id=sensor.bedroom_t_h_battery].state",
            ],
        }
    )
    assert normalized is not None
    assert normalized.template_id == "unavailable_sensors"
    assert normalized.rule_id == "bedroom_battery_sensor_unavailable"


def test_normalize_candidate_contextual_entities_excluded_from_availability() -> None:
    """A contextual condition entity must not deadlock the all-of evaluator."""
    normalized = normalize_candidate(
        {
            "candidate_id": "hall_temperature_unavailable_while_unoccupied",
            "title": "Hall temperature sensor unavailable",
            "summary": (
                "The hall temperature sensor reports unavailable while the "
                "hall occupancy condition is off."
            ),
            "pattern": (
                "entities[entity_id=sensor.hall_temperature].state == "
                "'unavailable' AND "
                "entities[entity_id=binary_sensor.hall_occupancy].state == 'off'"
            ),
            "suggested_type": "availability",
            "confidence_hint": 0.5,
            "evidence_paths": [
                "entities[entity_id=sensor.hall_temperature].state",
                "entities[entity_id=binary_sensor.hall_occupancy].state",
            ],
        }
    )
    assert normalized is not None
    assert normalized.template_id == "unavailable_sensors"
    assert normalized.params == {"sensor_entity_ids": ["sensor.hall_temperature"]}


def test_normalize_candidate_all_contextual_predicates_stay_unsupported() -> None:
    """A candidate whose only predicate is a non-availability state is not a rule."""
    result = explain_normalize_candidate(
        {
            "candidate_id": "occupancy_stuck_off_hub_unavailable",
            "title": "Occupancy sensor stuck",
            "summary": "Occupancy sensor stuck off while the hub is unavailable.",
            "pattern": (
                "entities[entity_id=binary_sensor.hall_occupancy].state == 'off'"
            ),
            "suggested_type": "availability",
            "confidence_hint": 0.4,
            "evidence_paths": [
                "entities[entity_id=binary_sensor.hall_occupancy].state",
            ],
        }
    )
    assert result.normalized is None
    assert result.reason_code == "unsupported_pattern"


def test_normalize_candidate_pattern_omitted_entity_is_contextual() -> None:
    """Evidence entities absent from a predicated pattern are not targets."""
    normalized = normalize_candidate(
        {
            "candidate_id": "hall_temperature_unavailable_context_omitted",
            "title": "Hall temperature sensor unavailable",
            "summary": (
                "The hall temperature sensor reports unavailable when the "
                "occupancy condition holds."
            ),
            "pattern": (
                "entities[entity_id=sensor.hall_temperature].state == 'unavailable'"
            ),
            "suggested_type": "availability",
            "confidence_hint": 0.5,
            "evidence_paths": [
                "entities[entity_id=sensor.hall_temperature].state",
                "entities[entity_id=binary_sensor.hall_occupancy].state",
            ],
        }
    )
    assert normalized is not None
    assert normalized.template_id == "unavailable_sensors"
    assert normalized.params == {"sensor_entity_ids": ["sensor.hall_temperature"]}


def test_normalize_candidate_anyone_home_false_pattern_overrides_evidence() -> None:
    """anyone_home == false must not scope an availability rule to while-home."""
    normalized = normalize_candidate(
        {
            "candidate_id": "occupancy_sensors_unavailable_away",
            "title": "Occupancy sensors unavailable while away",
            "summary": "Occupancy sensors report unavailable.",
            "pattern": "derived.anyone_home == false AND state_unavailable",
            "suggested_type": "availability",
            "confidence_hint": 0.5,
            "evidence_paths": [
                "derived.anyone_home",
                "entities[entity_id=binary_sensor.hall_occupancy].state",
            ],
        }
    )
    assert normalized is not None
    assert normalized.template_id == "unavailable_sensors"


def test_normalize_candidate_while_occupied_phrasing_selects_while_home() -> None:
    """Explicit 'while occupied' conditions keep the while-home template."""
    normalized = normalize_candidate(
        {
            "candidate_id": "sensors_unavailable_while_occupied",
            "title": "Sensors unavailable while occupied",
            "summary": (
                "Sensors report unavailable while the home is occupied, "
                "reducing coverage."
            ),
            "pattern": "state_unavailable",
            "suggested_type": "availability",
            "confidence_hint": 0.7,
            "evidence_paths": [
                "entities[entity_id=binary_sensor.hall_occupancy].state",
            ],
        }
    )
    assert normalized is not None
    assert normalized.template_id == "unavailable_sensors_while_home"


def test_normalize_candidate_prefix_entity_id_is_not_a_predicate_match() -> None:
    """A prefix ID (sensor.hall) must not inherit sensor.hall_temperature's predicate."""
    normalized = normalize_candidate(
        {
            "candidate_id": "hall_temperature_unavailable_prefix",
            "title": "Hall temperature sensor unavailable",
            "summary": "The hall temperature sensor reports unavailable.",
            "pattern": (
                "entities[entity_id=sensor.hall_temperature].state == 'unavailable'"
            ),
            "suggested_type": "availability",
            "confidence_hint": 0.5,
            "evidence_paths": [
                "entities[entity_id=sensor.hall].state",
                "entities[entity_id=sensor.hall_temperature].state",
            ],
        }
    )
    assert normalized is not None
    assert normalized.template_id == "unavailable_sensors"
    assert normalized.params == {"sensor_entity_ids": ["sensor.hall_temperature"]}


def test_normalize_candidate_free_form_predicates_without_equality_operator() -> None:
    """Bare-word per-entity states still separate targets from context."""
    normalized = normalize_candidate(
        {
            "candidate_id": "hall_temperature_unavailable_free_form",
            "title": "Hall temperature sensor unavailable",
            "summary": "Temperature sensor drops out while the occupancy input is off.",
            "pattern": (
                "sensor.hall_temperature unavailable AND "
                "binary_sensor.hall_occupancy off"
            ),
            "suggested_type": "availability",
            "confidence_hint": 0.5,
            "evidence_paths": [
                "entities[entity_id=sensor.hall_temperature].state",
                "entities[entity_id=binary_sensor.hall_occupancy].state",
            ],
        }
    )
    assert normalized is not None
    assert normalized.template_id == "unavailable_sensors"
    assert normalized.params == {"sensor_entity_ids": ["sensor.hall_temperature"]}


def test_normalize_candidate_negated_unavailable_predicate_is_contextual() -> None:
    """!= 'unavailable' clauses are context, not targets."""
    normalized = normalize_candidate(
        {
            "candidate_id": "kitchen_sensor_unavailable_negated_context",
            "title": "Kitchen sensor unavailable",
            "summary": "Kitchen sensor unavailable while the hall sensor is fine.",
            "pattern": (
                "entities[entity_id=sensor.hall_temperature].state != 'unavailable' "
                "AND entities[entity_id=sensor.kitchen_temperature].state == "
                "'unavailable'"
            ),
            "suggested_type": "availability",
            "confidence_hint": 0.5,
            "evidence_paths": [
                "entities[entity_id=sensor.hall_temperature].state",
                "entities[entity_id=sensor.kitchen_temperature].state",
            ],
        }
    )
    assert normalized is not None
    assert normalized.template_id == "unavailable_sensors"
    assert normalized.params == {"sensor_entity_ids": ["sensor.kitchen_temperature"]}


def test_normalize_candidate_suffix_sharing_entity_id_is_not_a_target() -> None:
    """sensor.temperature must not match inside binary_sensor.temperature."""
    normalized = normalize_candidate(
        {
            "candidate_id": "binary_temperature_unavailable_suffix",
            "title": "Temperature binary sensor unavailable",
            "summary": "The temperature binary sensor reports unavailable.",
            "pattern": (
                "entities[entity_id=binary_sensor.temperature].state == 'unavailable'"
            ),
            "suggested_type": "availability",
            "confidence_hint": 0.5,
            "evidence_paths": [
                "entities[entity_id=sensor.temperature].state",
                "entities[entity_id=binary_sensor.temperature].state",
            ],
        }
    )
    assert normalized is not None
    assert normalized.template_id == "unavailable_sensors"
    assert normalized.params == {"sensor_entity_ids": ["binary_sensor.temperature"]}


def test_normalize_candidate_away_wording_overrides_anyone_home_evidence() -> None:
    """'Nobody home' prose with bare anyone_home evidence is not while-home."""
    normalized = normalize_candidate(
        {
            "candidate_id": "occupancy_sensors_unavailable_nobody_home",
            "title": "Occupancy sensors unavailable while nobody home",
            "summary": (
                "Occupancy sensors report unavailable while nobody home, "
                "leaving the house unmonitored."
            ),
            "pattern": "state_unavailable",
            "suggested_type": "availability",
            "confidence_hint": 0.5,
            "evidence_paths": [
                "derived.anyone_home",
                "entities[entity_id=binary_sensor.hall_occupancy].state",
            ],
        }
    )
    assert normalized is not None
    assert normalized.template_id == "unavailable_sensors"


def test_normalize_candidate_unknown_state_predicate_is_not_a_target() -> None:
    """== 'unknown' predicates must not register a rule that can never fire."""
    result = explain_normalize_candidate(
        {
            "candidate_id": "hall_occupancy_unknown_state",
            "title": "Occupancy sensor state unknown",
            "summary": "The hall occupancy sensor is unreachable or unknown.",
            "pattern": (
                "entities[entity_id=binary_sensor.hall_occupancy].state == 'unknown'"
            ),
            "suggested_type": "availability",
            "confidence_hint": 0.4,
            "evidence_paths": [
                "entities[entity_id=binary_sensor.hall_occupancy].state",
            ],
        }
    )
    assert result.normalized is None
    assert result.reason_code == "unsupported_pattern"


def test_normalize_candidate_tracker_staleness_keeps_entity_staleness_routing() -> None:
    """'Offline' tracker candidates with last-seen wording stay staleness rules."""
    normalized = normalize_candidate(
        {
            "candidate_id": "phone_tracker_stale",
            "title": "Phone tracker offline",
            "summary": (
                "person.jane was last seen many hours ago and the BLE "
                "presence binary sensor has not updated."
            ),
            "pattern": "tracker_staleness",
            "suggested_type": "availability",
            "confidence_hint": 0.6,
            "evidence_paths": [
                "entities[entity_id=person.jane].state",
                "entities[entity_id=binary_sensor.jane_phone_ble].state",
            ],
        }
    )
    assert normalized is not None
    assert normalized.template_id == "entity_staleness"
    assert normalized.params["entity_id"] == "person.jane"


def test_normalize_candidate_unavailable_unsupported_domain_still_fails() -> None:
    """Unavailability evidence outside sensor/binary_sensor domains stays rejected."""
    result = explain_normalize_candidate(
        {
            "candidate_id": "cameras_unavailable",
            "title": "Cameras unavailable",
            "summary": "Cameras are reporting unavailable.",
            "pattern": "state_unavailable",
            "suggested_type": "availability",
            "confidence_hint": 0.5,
            "evidence_paths": ["entities[entity_id=light.living_room].state"],
        }
    )
    assert result.normalized is None
    assert result.reason_code == "missing_required_entities"


def test_normalize_candidate_binary_availability_evidence_without_keywords() -> None:
    """Recognized binary_sensor evidence without a matched pattern is 'unsupported'."""
    result = explain_normalize_candidate(
        {
            "candidate_id": "occupancy_flapping",
            "title": "Occupancy toggling oddly",
            "summary": "Occupancy readings look inconsistent across rooms.",
            "pattern": "state_flapping",
            "suggested_type": "availability",
            "confidence_hint": 0.5,
            "evidence_paths": [
                "entities[entity_id=binary_sensor.hall_occupancy].state",
            ],
        }
    )
    assert result.normalized is None
    assert result.reason_code == "unsupported_pattern"


def test_normalize_candidate_unavailable_binary_sensors_while_home() -> None:
    candidate = {
        "candidate_id": "occupancy_sensors_unavailable_home",
        "title": "Occupancy sensors unavailable while someone is home",
        "summary": (
            "Occupancy sensors report unavailable while someone is home, "
            "reducing presence coverage."
        ),
        "pattern": "derived.anyone_home AND state_unavailable",
        "suggested_type": "availability",
        "confidence_hint": 0.8,
        "evidence_paths": [
            "derived.anyone_home",
            "entities[entity_id=binary_sensor.smart_presence_sensor_obsazenost].state",
        ],
    }
    normalized = normalize_candidate(candidate)
    assert normalized is not None
    assert normalized.template_id == "unavailable_sensors_while_home"
    assert normalized.rule_id == "occupancy_sensors_unavailable_home"
    assert normalized.params == {
        "sensor_entity_ids": ["binary_sensor.smart_presence_sensor_obsazenost"]
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


def test_normalize_candidate_alarm_armed_home_away_boolean_false_not_rejected() -> None:
    """
    Regression: anyone_home == false must not trigger the occupancy guard.

    _presence_signal() previously matched 'home' as a substring of 'armed_home' /
    'anyone_home', causing effective_presence='home' and the occupancy guard to
    fire for a valid away-state candidate.  The boolean-expression check must
    take priority over the home-term scan.
    """
    candidate = {
        "candidate_id": "armed_home_while_nobody_present",
        "title": "Alarm in armed-home mode with no occupants",
        "summary": "The security system is in armed_home mode but the property is unoccupied.",
        "pattern": "alarm_state == armed_home AND anyone_home == false",
        "suggested_type": "security",
        "confidence_hint": 0.85,
        "evidence_paths": [
            "entities[entity_id=alarm_control_panel.home_alarm].state",
            "derived.anyone_home",
        ],
    }
    result = explain_normalize_candidate(candidate)
    assert result.normalized is not None, (
        f"Expected a normalized rule, got unsupported_pattern: {result.details}"
    )
    assert result.normalized.template_id == "alarm_state_mismatch"
    assert result.normalized.params["alarm_state"] == "armed_home"
    assert result.normalized.params["expected_presence"] == "away"


def test_normalize_candidate_alarm_armed_home_away_anyone_home_true_still_rejected() -> (
    None
):
    """anyone_home == true must still trigger the occupancy guard."""
    candidate = {
        "candidate_id": "armed_home_occupants_present",
        "title": "Alarm in armed-home mode with occupants present",
        "summary": "The security system is in armed_home mode and anyone_home == true.",
        "pattern": "alarm_state == armed_home AND anyone_home == true",
        "suggested_type": "security",
        "confidence_hint": 0.85,
        "evidence_paths": [
            "entities[entity_id=alarm_control_panel.home_alarm].state",
            "derived.anyone_home",
        ],
    }
    result = explain_normalize_candidate(candidate)
    assert result.normalized is None
    assert result.reason_code == "unsupported_pattern"


def test_normalize_candidate_alarm_armed_home_home_presence_rejected() -> None:
    """armed_home + home presence is not a mismatch — normalization must reject it."""
    candidate = {
        "candidate_id": "alarm_mode_occupancy_mismatch",
        "title": "Alarm mode occupancy mismatch",
        "summary": "The home alarm is armed_home while people are home.",
        "pattern": "alarm_state == armed_home AND anyone_home == true",
        "suggested_type": "security",
        "confidence_hint": 0.85,
        "evidence_paths": [
            "entities[entity_id=alarm_control_panel.home_alarm].state",
            "derived.anyone_home",
        ],
    }
    result = explain_normalize_candidate(candidate)
    assert result.normalized is None
    assert result.reason_code == "unsupported_pattern"
    assert normalize_candidate(candidate) is None


def test_normalize_candidate_alarm_armed_night_home_presence_rejected() -> None:
    """armed_night + home presence is not a mismatch — normalization must reject it."""
    candidate = {
        "candidate_id": "armed_night_while_home",
        "title": "Alarm Armed Night While Home",
        "summary": "Security alarm is in armed_night mode while occupants are home.",
        "pattern": "alarm_state == armed_night AND anyone_home == true",
        "suggested_type": "security",
        "confidence_hint": 0.8,
        "evidence_paths": [
            "entities[entity_id=alarm_control_panel.home_alarm].state",
            "derived.anyone_home",
        ],
    }
    result = explain_normalize_candidate(candidate)
    assert result.normalized is None
    assert result.reason_code == "unsupported_pattern"


def test_normalize_candidate_alarm_armed_home_no_presence_signal_rejected() -> None:
    """armed_home with no explicit presence signal defaults to home — still rejected."""
    candidate = {
        "candidate_id": "armed_home_no_presence",
        "title": "Alarm Armed Home Mode Active",
        "summary": "The home alarm is in armed_home mode.",
        "pattern": "alarm_state == armed_home",
        "suggested_type": "security",
        "confidence_hint": 0.7,
        "evidence_paths": [
            "entities[entity_id=alarm_control_panel.home_alarm].state",
        ],
    }
    result = explain_normalize_candidate(candidate)
    assert result.normalized is None
    assert result.reason_code == "unsupported_pattern"


def test_normalize_candidate_alarm_armed_night_no_presence_signal_rejected() -> None:
    """armed_night with no explicit presence signal defaults to home — still rejected."""
    candidate = {
        "candidate_id": "armed_night_no_presence",
        "title": "Alarm Armed Night Mode Active",
        "summary": "The home alarm is in armed_night mode.",
        "pattern": "alarm_state == armed_night",
        "suggested_type": "security",
        "confidence_hint": 0.7,
        "evidence_paths": [
            "entities[entity_id=alarm_control_panel.home_alarm].state",
        ],
    }
    result = explain_normalize_candidate(candidate)
    assert result.normalized is None
    assert result.reason_code == "unsupported_pattern"


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
        "candidate_id": "high_power_consumption_night",
        "title": "High Power Consumption at Night",
        "summary": "Anomalously high power draw detected during overnight hours.",
        "pattern": "is_night AND sensor.power_meter_power > baseline",
        "suggested_type": "power",
        "confidence_hint": 0.75,
        "evidence_paths": [
            "derived.is_night",
            "entities[entity_ids contains sensor.power_meter_power].state",
        ],
    }
    normalized = normalize_candidate(candidate)
    assert normalized is not None
    assert normalized.template_id == "baseline_deviation"
    assert normalized.params["entity_id"] == "sensor.power_meter_power"


def test_normalize_candidate_cyclical_load_routes_to_time_of_day_anomaly() -> None:
    """Fridge/freezer power sensors must normalize to time_of_day_anomaly, not baseline_deviation."""
    for entity_id in (
        "sensor.fridge_switch_0_power",
        "sensor.refrigerator_power",
        "sensor.freezer_power",
    ):
        candidate = {
            "candidate_id": f"cyclical_{entity_id}",
            "title": "Unusual fridge power consumption",
            "summary": "Fridge power deviates from expected pattern.",
            "pattern": f"{entity_id} != expected",
            "suggested_type": "power",
            "confidence_hint": 0.75,
            "evidence_paths": [f"entities[entity_ids contains {entity_id}].state"],
        }
        normalized = normalize_candidate(candidate)
        assert normalized is not None, f"Expected rule for {entity_id}"
        assert normalized.template_id == "time_of_day_anomaly", (
            f"Expected time_of_day_anomaly for {entity_id}, got {normalized.template_id}"
        )
        assert normalized.params["entity_id"] == entity_id


def test_normalize_candidate_cumulative_energy_sensor_rejected() -> None:
    """Cumulative energy sensors must not be normalized to baseline_deviation."""
    candidate = {
        "candidate_id": "fridge_energy_baseline",
        "title": "Fridge Energy Baseline Deviation",
        "summary": "Fridge energy consumption deviates from baseline.",
        "pattern": "sensor.fridge_switch_0_energy > baseline",
        "suggested_type": "energy",
        "confidence_hint": 0.8,
        "evidence_paths": [
            "entities[entity_ids contains sensor.fridge_switch_0_energy].state",
        ],
    }
    normalized = normalize_candidate(candidate)
    assert normalized is None


def test_normalize_candidate_mixed_energy_and_power_skips_energy_sensors() -> None:
    """Multi-entity bundle with *_energy and *_power sensors picks the first *_power entity."""
    candidate = {
        "candidate_id": "candidate_kitchen_power_mismatch",
        "title": "Kitchen Power Consumption Mismatch",
        "summary": "Kitchen appliances power and energy readings deviate from baseline.",
        "pattern": "deviation_from_baseline",
        "suggested_type": "statistical_anomaly",
        "confidence_hint": 0.85,
        "evidence_paths": [
            # Energy sensors (cumulative — must be skipped)
            "entities[entity_ids contains sensor.dishwasher_switch_0_energy].state",
            "entities[entity_ids contains sensor.fridge_switch_0_energy].state",
            # Power sensors (instantaneous — first one wins)
            "entities[entity_ids contains sensor.dishwasher_switch_0_power].state",
            "entities[entity_ids contains sensor.fridge_switch_0_power].state",
            "entities[entity_ids contains sensor.kettle_switch_0_power].state",
        ],
    }
    normalized = normalize_candidate(candidate)
    # Must not be rejected — there are valid instantaneous power sensors
    assert normalized is not None
    # The first non-cumulative power sensor is dishwasher
    assert normalized.params.get("entity_id") == "sensor.dishwasher_switch_0_power"


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
    """Window duration candidate with no entry entity IDs and no night/away signal is unsupported."""
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
    result = explain_normalize_candidate(candidate)
    assert result.normalized is None
    assert result.reason_code == "missing_required_entities"


# ---------------------------------------------------------------------------
# Dot-notation evidence path extraction
# ---------------------------------------------------------------------------


def test_normalize_candidate_power_sensor_dot_notation_evidence_paths() -> None:
    """Sensor entity IDs in dot-notation paths (e.g. sensor.foo.state) are extracted."""
    candidate = {
        "candidate_id": "high_power_consumption_night",
        "title": "High Power Consumption at Night",
        "summary": "Anomalously high power draw detected during overnight hours.",
        "pattern": "is_night AND sensor.power_meter_power > baseline",
        "suggested_type": "power",
        "confidence_hint": 0.75,
        "evidence_paths": [
            "derived.is_night",
            "sensor.power_meter_power.state",
        ],
    }
    normalized = normalize_candidate(candidate)
    assert normalized is not None
    assert normalized.template_id == "baseline_deviation"
    assert normalized.params["entity_id"] == "sensor.power_meter_power"


def test_normalize_candidate_lock_battery_dot_notation_evidence_paths() -> None:
    """Lock battery candidate without a sensor.* battery entity ID returns unsupported_pattern."""
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
    result = explain_normalize_candidate(candidate)
    assert result.normalized is None
    assert result.reason_code == "unsupported_pattern"


def test_normalize_candidate_lock_battery_with_sensor_entity() -> None:
    """Lock battery candidate with a sensor.* battery entity routes to low_battery_sensors."""
    candidate = {
        "candidate_id": "playroom_lock_battery_low",
        "title": "Playroom Lock Battery Low",
        "summary": "The playroom door lock battery is below 20%.",
        "pattern": "sensor.playroom_lock_battery < 20",
        "suggested_type": "maintenance",
        "confidence_hint": 0.9,
        "evidence_paths": [
            "lock.playroom_door_lock.state",
            "sensor.playroom_lock_battery.state",
        ],
    }
    normalized = normalize_candidate(candidate)
    assert normalized is not None
    assert normalized.template_id == "low_battery_sensors"
    assert "sensor.playroom_lock_battery" in normalized.params["sensor_entity_ids"]
    assert "lock.playroom_door_lock" not in normalized.params["sensor_entity_ids"]


# ---------------------------------------------------------------------------
# Bug fixes: entity type filtering
# ---------------------------------------------------------------------------


def test_find_entry_entity_ids_excludes_plain_sensor_domain() -> None:
    """sensor.* entities are not treated as entry sensors; only binary_sensor.* and cover.*."""
    paths = [
        "sensor.front_door_temperature",
        "binary_sensor.front_door_contact",
        "cover.garage_door",
    ]
    ids = _find_entry_entity_ids(paths)
    assert "sensor.front_door_temperature" not in ids
    assert "binary_sensor.front_door_contact" in ids
    assert "cover.garage_door" in ids


def test_find_sensor_entity_ids_excludes_binary_sensor_domain() -> None:
    """binary_sensor.* entities are excluded from sensor_ids to prevent misrouting."""
    paths = [
        "binary_sensor.motion_hallway",
        "sensor.power_meter_power",
    ]
    ids = _find_sensor_entity_ids(paths)
    assert "binary_sensor.motion_hallway" not in ids
    assert "sensor.power_meter_power" in ids


def test_find_battery_sensor_entity_ids_excludes_binary_sensor_domain() -> None:
    """binary_sensor.* battery entities are excluded; only sensor.* are valid for low_battery_sensors."""
    paths = [
        "binary_sensor.playroom_door_lock_battery",
        "sensor.garage_door_lock_battery",
    ]
    ids = _find_battery_sensor_entity_ids(paths)
    assert "binary_sensor.playroom_door_lock_battery" not in ids
    assert "sensor.garage_door_lock_battery" in ids


def test_normalize_candidate_lock_battery_binary_sensor_returns_unsupported() -> None:
    """Lock battery candidate with only binary_sensor.* battery entities returns unsupported_pattern."""
    candidate = {
        "candidate_id": "lock_battery_binary",
        "title": "Lock Battery Low",
        "summary": "The lock battery sensor reports low.",
        "pattern": "binary_sensor.lock_battery == on",
        "suggested_type": "maintenance",
        "confidence_hint": 0.8,
        "evidence_paths": [
            "lock.front_door_lock.state",
            "binary_sensor.front_door_lock_battery.state",
        ],
    }
    result = explain_normalize_candidate(candidate)
    assert result.normalized is None
    assert result.reason_code == "unsupported_pattern"


# ---------------------------------------------------------------------------
# Bug fixes: presence signal
# ---------------------------------------------------------------------------


def test_presence_signal_not_derived_anyone_home_path() -> None:
    """'not derived.anyone_home' evidence path returns 'away'."""
    assert _presence_signal(["not derived.anyone_home"], "") == "away"


def test_presence_signal_derived_anyone_home_path_alone_returns_any() -> None:
    """'derived.anyone_home' without text signals returns 'any', not 'home'."""
    assert _presence_signal(["derived.anyone_home"], "") == "any"


def test_presence_signal_not_derived_anyone_home_takes_priority_over_text() -> None:
    """'not derived.anyone_home' path wins even when text contains home terms."""
    assert (
        _presence_signal(["not derived.anyone_home"], "someone is home occupied")
        == "away"
    )


# ---------------------------------------------------------------------------
# Bug fixes: entry branch "any" presence path
# ---------------------------------------------------------------------------


def test_normalize_candidate_entry_any_presence_defaults_to_away_template() -> None:
    """Entry candidate with unknown presence (no occupancy signal) defaults to open_entry_while_away."""
    candidate = {
        "candidate_id": "front_door_open",
        "title": "Front Door Open",
        "summary": "The front door has been detected open.",
        "pattern": "binary_sensor.front_door_contact == on",
        "suggested_type": "security",
        "confidence_hint": 0.8,
        "evidence_paths": [
            "binary_sensor.front_door_contact",
        ],
    }
    normalized = normalize_candidate(candidate)
    assert normalized is not None
    assert normalized.template_id == "open_entry_while_away"


# ---------------------------------------------------------------------------
# Bug fixes: multiple_entries_open_count with "any" presence
# ---------------------------------------------------------------------------


def test_normalize_candidate_multiple_entries_any_presence_sets_require_away() -> None:
    """multiple_entries_open_count with unknown presence defaults require_away=True."""
    candidate = {
        "candidate_id": "multiple_windows_open",
        "title": "Multiple Windows Open Simultaneously",
        "summary": "Several windows are open at the same time.",
        "pattern": "multiple binary_sensor windows == on simultaneously",
        "suggested_type": "security",
        "confidence_hint": 0.75,
        "evidence_paths": [
            "binary_sensor.window_living_room",
            "binary_sensor.window_bedroom",
        ],
    }
    normalized = normalize_candidate(candidate)
    assert normalized is not None
    assert normalized.template_id == "multiple_entries_open_count"
    assert normalized.params["require_away"] is True
    assert normalized.params["require_home"] is False


# ---------------------------------------------------------------------------
# Bug fixes: threshold floor
# ---------------------------------------------------------------------------


def test_extract_threshold_numeric_rejects_zero() -> None:
    """_extract_threshold_numeric returns None when the matched value is 0."""
    assert _extract_threshold_numeric("power above 0 watts") is None
    assert _extract_threshold_numeric("exceeds 0") is None


def test_extract_threshold_numeric_accepts_positive() -> None:
    """_extract_threshold_numeric returns a positive float normally."""
    assert _extract_threshold_numeric("above 5 watts") == 5.0
    assert _extract_threshold_numeric("exceeds 100") == 100.0


def test_normalize_candidate_power_zero_threshold_falls_back_to_baseline() -> None:
    """Power sensor candidate with 'above 0' threshold falls back to baseline_deviation."""
    candidate = {
        "candidate_id": "phantom_load_above_0",
        "title": "Phantom Load Active",
        "summary": "Device draws power above 0 watts when not in use.",
        "pattern": "sensor.device_power > 0",
        "suggested_type": "power",
        "confidence_hint": 0.65,
        "evidence_paths": [
            "sensor.device_power.state",
        ],
    }
    normalized = normalize_candidate(candidate)
    assert normalized is not None
    assert normalized.template_id == "baseline_deviation"
    assert normalized.params["entity_id"] == "sensor.device_power"


# ---------------------------------------------------------------------------
# Bug fixes: open_any_window_at_night_while_away late guarded fallback
# ---------------------------------------------------------------------------


def test_normalize_candidate_window_night_away_no_entry_ids_uses_selector() -> None:
    """No-entry-ID window candidate WITH night+away signals uses open_any_window_at_night_while_away."""
    candidate = {
        "candidate_id": "window_open_night_away",
        "title": "Window Open at Night While Away",
        "summary": "A window is open at night while no one is home.",
        "pattern": "is_night AND anyone_home == false AND window == open",
        "suggested_type": "security",
        "confidence_hint": 0.8,
        "evidence_paths": [
            "derived.is_night",
            "not derived.anyone_home",
        ],
    }
    normalized = normalize_candidate(candidate)
    assert normalized is not None
    assert normalized.template_id == "open_any_window_at_night_while_away"
    assert normalized.params["entry_selector"] == "window"


# ---------------------------------------------------------------------------
# Issue #504: localized entity IDs, quoted evidence paths, night+any presence
# ---------------------------------------------------------------------------


def test_extract_entity_id_strips_quotes() -> None:
    """Quote-wrapped entity IDs in evidence paths resolve to clean IDs."""
    assert (
        _extract_entity_id_from_evidence_path(
            "entities[entity_ids contains 'binary_sensor.bedroom_loznice_okno'].state"
        )
        == "binary_sensor.bedroom_loznice_okno"
    )
    assert (
        _extract_entity_id_from_evidence_path('entities[entity_id="lock.front_door"]')
        == "lock.front_door"
    )
    assert (
        _extract_entity_id_from_evidence_path("'binary_sensor.front_door_contact'")
        == "binary_sensor.front_door_contact"
    )


def test_normalize_candidate_issue_504_localized_window_at_night() -> None:
    """Exact candidate from issue #504: Czech-named window sensor, quoted path."""
    candidate = {
        "candidate_id": "window_open_at_night",
        "title": "Window Open Detected at Night",
        "summary": "The bedroom window was opened during night hours.",
        "pattern": "state_change",
        "confidence_hint": 0.5,
        "evidence_paths": [
            "entities[entity_ids contains 'binary_sensor.bedroom_loznice_okno'].state",
            "derived.is_night",
        ],
    }
    normalized = normalize_candidate(candidate)
    assert normalized is not None
    assert normalized.template_id == "open_entry_at_night"
    assert normalized.rule_id == "open_entry_at_night_window"
    assert normalized.params["entry_entity_ids"] == [
        "binary_sensor.bedroom_loznice_okno"
    ]
    assert normalized.suggested_actions == ["close_entry"]


def test_normalize_candidate_entry_night_any_presence_uses_night_template() -> None:
    """Night entry candidate with unknown presence routes to open_entry_at_night."""
    candidate = {
        "candidate_id": "living_room_window_night",
        "title": "Window Open at Night",
        "summary": "The living room window is open during the night.",
        "pattern": "state_change",
        "confidence_hint": 0.6,
        "evidence_paths": [
            "entities[entity_id=binary_sensor.living_room_window].state",
            "derived.is_night",
        ],
    }
    normalized = normalize_candidate(candidate)
    assert normalized is not None
    assert normalized.template_id == "open_entry_at_night"
    assert normalized.rule_id == "open_entry_at_night_window"


def test_normalize_candidate_text_entry_fallback_ignores_motion_sensors() -> None:
    """Text fallback never promotes motion/battery-style binary sensors to entries."""
    candidate = {
        "candidate_id": "door_motion_night",
        "title": "Door Area Motion at Night",
        "summary": "Motion near the front door during the night.",
        "pattern": "state_change",
        "confidence_hint": 0.6,
        "evidence_paths": [
            "entities[entity_id=binary_sensor.hallway_motion].state",
            "derived.is_night",
        ],
    }
    result = explain_normalize_candidate(candidate)
    assert result.normalized is None
    assert result.reason_code == "missing_required_entities"


def test_normalize_candidate_night_hours_text_is_not_a_duration_signal() -> None:
    """Phrasing like 'during night hours' must not route to entity_state_duration."""
    candidate = {
        "candidate_id": "window_open_at_night_english",
        "title": "Window Open Detected at Night",
        "summary": "The bedroom window was opened during night hours.",
        "pattern": "state_change",
        "confidence_hint": 0.5,
        "evidence_paths": [
            "entities[entity_id=binary_sensor.bedroom_window].state",
            "derived.is_night",
        ],
    }
    normalized = normalize_candidate(candidate)
    assert normalized is not None
    assert normalized.template_id == "open_entry_at_night"


def test_has_duration_signal_numeric_hours_still_counts() -> None:
    """Regression: a numeric hours threshold alone is still a duration signal."""
    # "2 hours" with no other duration term must keep routing to duration.
    assert _has_duration_signal("window open more than 2 hours") is True
    # Bare "hours" without a number is time-of-day context, not a duration.
    assert _has_duration_signal("window opened during night hours") is False
    # Classic term-based signals are unchanged.
    assert _has_duration_signal("open for an extended period") is True


def test_entry_kind_text_fallback_branches() -> None:
    """_entry_kind falls back to candidate text only when IDs carry no token."""
    # Locale-named IDs: kind comes from the text.
    assert (
        _entry_kind(["binary_sensor.loznice_okno"], "bedroom window open") == "window"
    )
    assert _entry_kind(["binary_sensor.chodba_dvere"], "front door open") == "door"
    # Text names an entry but neither window nor door: generic kind.
    assert _entry_kind(["binary_sensor.chodba_dvere"], "entry point open") == "entry"
    # Entity-ID tokens take priority over conflicting text.
    assert _entry_kind(["binary_sensor.front_window_contact"], "door open") == "window"
    # Regression: single-argument call (keyword-derived IDs) still works.
    assert _entry_kind(["binary_sensor.front_door_contact"]) == "door"


def test_find_text_entry_entity_ids_branches() -> None:
    """Text fallback promotes only binary_sensor/cover IDs with no sensor-kind token."""
    paths = [
        "entities[entity_id=cover.garaz_vrata].state",
        "entities[entity_id=binary_sensor.chodba_vmd].state",
        "entities[entity_id=lock.front_lock].state",
        "entities[entity_id=okno].state",
        "derived.is_night",
    ]
    # Text without an entry keyword: fallback declines entirely.
    assert _find_text_entry_entity_ids(paths, "high power usage overnight") == []
    # cover.* is promoted; vmd, lock.*, and domain-less IDs are all skipped.
    assert _find_text_entry_entity_ids(paths, "garage door open at night") == [
        "cover.garaz_vrata"
    ]


def test_extract_entity_id_quote_edge_cases() -> None:
    """Quotes-only tokens resolve to None; backtick quoting is stripped."""
    assert _extract_entity_id_from_evidence_path("entities[entity_id='']") is None
    assert (
        _extract_entity_id_from_evidence_path("entities[entity_ids contains '']")
        is None
    )
    assert (
        _extract_entity_id_from_evidence_path("entities[entity_id=`lock.front_door`]")
        == "lock.front_door"
    )


def test_normalize_candidate_issue_504_localized_door_at_night() -> None:
    """Text-derived door kind: Czech-named door sensor yields the _door rule_id."""
    candidate = {
        "candidate_id": "door_open_at_night",
        "title": "Door Open Detected at Night",
        "summary": "The hallway door was open during the night.",
        "pattern": "state_change",
        "confidence_hint": 0.5,
        "evidence_paths": [
            "entities[entity_ids contains 'binary_sensor.chodba_dvere'].state",
            "derived.is_night",
        ],
    }
    normalized = normalize_candidate(candidate)
    assert normalized is not None
    assert normalized.template_id == "open_entry_at_night"
    assert normalized.rule_id == "open_entry_at_night_door"
    assert normalized.params["entry_entity_ids"] == ["binary_sensor.chodba_dvere"]


def test_text_entry_fallback_requires_word_boundaries() -> None:
    """'indoor'/'outdoor'/'doorbell' must not activate the text entry fallback."""
    paths = ["entities[entity_id=binary_sensor.chodba_cidlo].state"]
    assert _find_text_entry_entity_ids(paths, "indoor air quality alert") == []
    assert _find_text_entry_entity_ids(paths, "outdoor activity at night") == []
    assert _find_text_entry_entity_ids(paths, "doorbell pressed at night") == []
    assert _find_text_entry_entity_ids(paths, "the front door is open") == [
        "binary_sensor.chodba_cidlo"
    ]


def test_text_entry_fallback_excludes_safety_sensors() -> None:
    """Smoke/gas/leak-style binary sensors are never promoted to entry sensors."""
    paths = [
        "entities[entity_id=binary_sensor.kitchen_smoke].state",
        "entities[entity_id=binary_sensor.cellar_gas].state",
        "entities[entity_id=binary_sensor.bathroom_leak].state",
        "entities[entity_id=binary_sensor.hall_tamper].state",
        "entities[entity_id=binary_sensor.loznice_okno].state",
    ]
    assert _find_text_entry_entity_ids(paths, "window opened during the night") == [
        "binary_sensor.loznice_okno"
    ]


def test_entry_kind_text_fallback_uses_word_boundaries() -> None:
    """'windowsill'/'doorbell' must not set the entry kind from text."""
    assert _entry_kind([], "windowsill decoration moved") == "entry"
    assert _entry_kind([], "doorbell pressed") == "entry"
    assert _entry_kind([], "windows were opened") == "window"
    assert _entry_kind([], "the doors were left open") == "door"


def test_text_entry_fallback_skipped_for_lock_candidates() -> None:
    """Lock candidates saying 'door' must keep lock routing, not promote entries."""
    candidate = {
        "candidate_id": "front_door_lock_unlocked_night",
        "title": "Front Door Lock Unlocked at Night",
        "summary": "The front door lock is unlocked during the night.",
        "pattern": "state_change",
        "confidence_hint": 0.8,
        "evidence_paths": [
            "entities[entity_id=lock.front_door].state",
            "entities[entity_id=binary_sensor.chodba_svetlo].state",
            "derived.is_night",
        ],
    }
    normalized = normalize_candidate(candidate)
    assert normalized is not None
    assert normalized.template_id == "unlocked_lock_when_home"
    assert normalized.params["lock_entity_id"] == "lock.front_door"


def test_duration_terms_are_word_bounded() -> None:
    """'before'/'along' must not satisfy the 'for'/'long' duration terms."""
    assert not _has_duration_signal("window open before midnight")
    assert not _has_duration_signal("moving along the hallway")
    assert _has_duration_signal("open for a while")
    assert _has_duration_signal("unlocked since sunset")


def test_qualitative_hours_still_count_as_duration() -> None:
    """Codex P2: 'many hours'/'several hours' are durations; 'night hours' is not."""
    assert _has_duration_signal("window remained open many hours")
    assert _has_duration_signal("door left open several hours")
    assert _has_duration_signal("open a couple of hours")
    assert not _has_duration_signal("window opened during night hours")


def test_find_camera_id_accepts_all_quote_styles() -> None:
    """Codex P2: double-quoted and backticked camera evidence paths resolve."""
    for quote in ("", "'", '"', "`"):
        candidate = {"candidate_id": "cam_check"}
        paths = [f"entities[entity_id={quote}camera.front_porch{quote}].state"]
        assert _find_camera_id(paths, candidate) == "camera.front_porch"
    assert _find_camera_id(['entities[entity_id="sensor.foo"].state'], {}) is None

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

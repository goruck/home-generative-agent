# ruff: noqa: S101
"""Tests for deterministic discovery semantic keys."""

from __future__ import annotations

from custom_components.home_generative_agent.sentinel.discovery_semantic import (
    candidate_semantic_key,
    rule_semantic_key,
)


def test_candidate_semantic_key_collapses_similar_window_home_night() -> None:
    candidate_a = {
        "title": "Open windows at night while someone is home",
        "summary": "Detects windows open during nighttime when someone is present.",
        "pattern": "window open at night while home",
        "suggested_type": "security_risk",
        "evidence_paths": [
            "entities[entity_id=binary_sensor.playroom_window].state",
            "derived.is_night",
            "derived.anyone_home",
        ],
    }
    candidate_b = {
        "title": "Garage and playroom windows open while home",
        "summary": "Windows open while occupants are present at night.",
        "pattern": "night home windows open",
        "suggested_type": "security_state",
        "evidence_paths": [
            "derived.anyone_home",
            "entities[entity_id=binary_sensor.playroom_window].state",
            "derived.is_night",
        ],
    }
    assert candidate_semantic_key(candidate_a) == candidate_semantic_key(candidate_b)


def test_rule_semantic_key_for_lock_rule() -> None:
    rule = {
        "rule_id": "unlocked_lock_when_home_lock_garage_door_lock",
        "template_id": "unlocked_lock_when_home",
        "params": {"lock_entity_id": "lock.garage_door_lock"},
    }
    key = rule_semantic_key(rule)
    assert key is not None
    assert "subject=lock" in key
    assert "predicate=unlocked" in key


def test_candidate_semantic_key_any_window_no_entity_paths() -> None:
    candidate = {
        "title": "Open windows while no one home at night",
        "summary": "Detects when any window is open while away at night.",
        "pattern": "any window open while away at night",
        "suggested_type": "security_risk",
        "evidence_paths": ["derived.is_night"],
    }
    key = candidate_semantic_key(candidate)
    assert key is not None
    assert "subject=entry_window" in key
    assert "predicate=open" in key
    assert "home=0" in key


def test_candidate_semantic_key_unavailable_sensor_while_home() -> None:
    candidate = {
        "title": "Unavailable sensors while home",
        "summary": "Detects any sensor reporting unavailable while occupied.",
        "pattern": "sensor unavailable while home",
        "suggested_type": "availability",
        "evidence_paths": [
            "derived.anyone_home",
            "entities[entity_id=sensor.backyard_vmd3_0].state",
        ],
    }
    key = candidate_semantic_key(candidate)
    assert key is not None
    assert "subject=sensor" in key
    assert "predicate=unavailable" in key
    assert "home=1" in key


def test_rule_semantic_key_unavailable_sensors_while_home() -> None:
    rule = {
        "rule_id": "unavailable_sensors_while_home",
        "template_id": "unavailable_sensors_while_home",
        "params": {
            "sensor_entity_ids": [
                "sensor.backyard_vmd3_0",
                "sensor.backyard_vmd4_camera1profile1",
            ]
        },
    }
    key = rule_semantic_key(rule)
    assert key is not None
    assert "subject=sensor" in key
    assert "predicate=unavailable" in key
    assert "home=1" in key


def test_rule_semantic_key_unavailable_sensors_any_home_state() -> None:
    rule = {
        "rule_id": "backyard_sensors_unavailable",
        "template_id": "unavailable_sensors",
        "params": {
            "sensor_entity_ids": [
                "backyard_vmd3_0",
                "backyard_vmd4_camera1profile1",
            ]
        },
    }
    key = rule_semantic_key(rule)
    assert key is not None
    assert "subject=sensor" in key
    assert "predicate=unavailable" in key
    assert "home=any" in key

# ruff: noqa: S101
"""Tests for deterministic discovery semantic keys."""

from __future__ import annotations

from custom_components.home_generative_agent.sentinel.discovery_semantic import (
    candidate_semantic_key,
    rule_key_covers_candidate_key,
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


def test_rule_semantic_key_low_battery_sensors() -> None:
    rule = {
        "rule_id": "low_battery_room_sensors_v1",
        "template_id": "low_battery_sensors",
        "params": {
            "sensor_entity_ids": [
                "sensor.elias_t_h_battery",
                "sensor.girls_t_h_battery",
            ],
            "threshold": 40,
        },
    }
    key = rule_semantic_key(rule)
    assert key is not None
    assert "subject=sensor" in key
    assert "predicate=low_battery" in key
    assert "home=any" in key


def test_rule_semantic_key_motion_night_alarm_disarmed_issue_235() -> None:
    rule = {
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
    }
    key = rule_semantic_key(rule)
    assert key is not None
    assert "subject=motion" in key
    assert "predicate=active" in key
    assert "night=1" in key


def test_rule_semantic_key_unknown_person_camera_when_home_issue_278() -> None:
    rule = {
        "rule_id": "unknown_person_camera_when_home",
        "template_id": "unknown_person_camera_when_home",
        "params": {"camera_entity_id": "camera.backyard"},
    }
    key = rule_semantic_key(rule)
    assert key is not None
    assert "subject=camera" in key
    assert "predicate=unknown_person" in key
    assert "home=1" in key


def test_rule_semantic_key_unknown_person_camera_no_home_any_camera() -> None:
    rule = {
        "rule_id": "unknown_person_camera_no_home_any_camera",
        "template_id": "unknown_person_camera_no_home",
        "params": {"camera_selector": "any"},
    }
    key = rule_semantic_key(rule)
    assert (
        key
        == "v1|subject=camera|predicate=unknown_person|night=any|home=0|scope=any|entities="
    )


def test_candidate_semantic_key_entity_ids_contains_format() -> None:
    """LLM-generated evidence paths use 'entity_ids contains' — must extract entity."""
    candidate = {
        "title": "Fridge power anomaly",
        "summary": "Fridge power deviates from baseline during off-cycle.",
        "pattern": "power deviation baseline",
        "suggested_type": "power_anomaly",
        "evidence_paths": [
            "entities[entity_ids contains sensor.fridge_switch_0_power].state",
        ],
    }
    key = candidate_semantic_key(candidate)
    assert key is not None
    assert "predicate=power_anomaly" in key
    assert "sensor.fridge_switch_0_power" in key


def test_candidate_semantic_key_entity_ids_contains_distinct_entities() -> None:
    """Two candidates with different entities in 'entity_ids contains' get different keys."""
    fridge = {
        "title": "Fridge power anomaly",
        "summary": "Fridge power baseline deviation.",
        "pattern": "power deviation baseline",
        "suggested_type": "power_anomaly",
        "evidence_paths": [
            "entities[entity_ids contains sensor.fridge_switch_0_power].state",
        ],
    }
    freezer = {
        "title": "Freezer power anomaly",
        "summary": "Freezer power baseline deviation.",
        "pattern": "power deviation baseline",
        "suggested_type": "power_anomaly",
        "evidence_paths": [
            "entities[entity_ids contains sensor.freezer_switch_0_power].state",
        ],
    }
    key_fridge = candidate_semantic_key(fridge)
    key_freezer = candidate_semantic_key(freezer)
    assert key_fridge is not None
    assert key_freezer is not None
    assert key_fridge != key_freezer


def test_rule_semantic_key_baseline_deviation() -> None:
    rule = {
        "rule_id": "sensor_baseline_fridge_power",
        "template_id": "baseline_deviation",
        "params": {"entity_id": "sensor.fridge_switch_0_power"},
    }
    key = rule_semantic_key(rule)
    assert key is not None
    assert "predicate=power_anomaly" in key
    assert "sensor.fridge_switch_0_power" in key
    assert "template=baseline_deviation" in key


def test_rule_semantic_key_time_of_day_anomaly() -> None:
    rule = {
        "rule_id": "sensor_tod_fridge_power",
        "template_id": "time_of_day_anomaly",
        "params": {"entity_id": "sensor.fridge_switch_0_power"},
    }
    key = rule_semantic_key(rule)
    assert key is not None
    assert "predicate=power_anomaly" in key
    assert "sensor.fridge_switch_0_power" in key
    assert "template=time_of_day_anomaly" in key


def test_rule_semantic_key_baseline_deviation_and_time_of_day_differ() -> None:
    """baseline_deviation and time_of_day_anomaly for same entity have distinct keys."""
    baseline_rule = {
        "rule_id": "sensor_baseline_fridge",
        "template_id": "baseline_deviation",
        "params": {"entity_id": "sensor.fridge_switch_0_power"},
    }
    tod_rule = {
        "rule_id": "sensor_tod_fridge",
        "template_id": "time_of_day_anomaly",
        "params": {"entity_id": "sensor.fridge_switch_0_power"},
    }
    assert rule_semantic_key(baseline_rule) != rule_semantic_key(tod_rule)


def test_candidate_semantic_key_power_anomaly_wins_over_activity_in_summary() -> None:
    """'power_anomaly' must win even when summary says 'appliance activity'."""
    candidate = {
        "title": "Washing Machine Power Active During Night While Home",
        "summary": (
            "The washing machine power sensor reports non-zero consumption (0.5W) "
            "while it is night and someone is home, suggesting potential unexpected "
            "appliance activity or baseline deviation."
        ),
        "pattern": "deviation_from_normal",
        "suggested_type": "statistical_anomaly",
        "evidence_paths": [
            "entities[entity_ids contains sensor.washing_machine_switch_0_power].state",
            "derived.is_night",
            "derived.anyone_home",
        ],
    }
    key = candidate_semantic_key(candidate)
    assert key is not None
    assert "predicate=power_anomaly" in key, f"expected power_anomaly, got: {key}"
    assert "active" not in key


def test_candidate_semantic_key_unavailable_wins_over_disarmed_context() -> None:
    """'unavailable' predicate must win even when summary mentions 'disarmed' as context."""
    candidate = {
        "title": "Outdoor Motion Sensors Unavailable During Active Monitoring",
        "summary": (
            "Multiple outdoor motion sensors are unavailable while the alarm system "
            "is disarmed and motion is detected elsewhere."
        ),
        "pattern": "state_mismatch",
        "suggested_type": "device_health",
        "evidence_paths": [
            "entities[entity_ids contains binary_sensor.backyard_vmd3_0].state",
            "entities[entity_ids contains binary_sensor.east_vmd3_0].state",
            "derived.anyone_home",
        ],
    }
    key = candidate_semantic_key(candidate)
    assert key is not None
    assert "predicate=unavailable" in key, f"expected unavailable, got: {key}"
    assert "disarmed" not in key


# ---------------------------------------------------------------------------
# P1: _rule_key_covers_candidate_key — template-aware comparison
# ---------------------------------------------------------------------------


def test_rule_key_covers_candidate_key_exact_match() -> None:
    """Identical keys must be covered."""
    key = "v1|subject=lock|predicate=unlocked|night=any|home=1|scope=any|entities=lock.front_door"
    assert rule_key_covers_candidate_key(key, key)


def test_rule_key_covers_candidate_key_different_entities() -> None:
    """Same template, different entity must NOT match."""
    rule_key = (
        "v1|subject=sensor|predicate=power_anomaly"
        "|template=time_of_day_anomaly|entities=sensor.fridge_switch_0_power"
    )
    candidate_key = (
        "v1|subject=sensor|predicate=power_anomaly"
        "|night=any|home=any|scope=any|entities=sensor.freezer_switch_0_power"
    )
    assert not rule_key_covers_candidate_key(rule_key, candidate_key)


def test_rule_key_covers_candidate_key_time_of_day_anomaly_vs_candidate() -> None:
    """
    time_of_day_anomaly rule key must cover a matching power_anomaly candidate key.

    This is the P1 regression: rule_semantic_key embeds |template=…| and omits
    night/home/scope; candidate_semantic_key never emits |template=…|. The
    normalized comparison must return True for the same entity.
    """
    rule_key = (
        "v1|subject=sensor|predicate=power_anomaly"
        "|template=time_of_day_anomaly|entities=sensor.fridge_switch_0_power"
    )
    candidate_key = (
        "v1|subject=sensor|predicate=power_anomaly"
        "|night=any|home=any|scope=any|entities=sensor.fridge_switch_0_power"
    )
    assert rule_key_covers_candidate_key(rule_key, candidate_key)


def test_rule_key_covers_candidate_key_baseline_deviation_vs_candidate() -> None:
    """baseline_deviation rule key must cover a matching power_anomaly candidate key."""
    rule_key = (
        "v1|subject=sensor|predicate=power_anomaly"
        "|template=baseline_deviation|entities=sensor.fridge_switch_0_power"
    )
    candidate_key = (
        "v1|subject=sensor|predicate=power_anomaly"
        "|night=any|home=any|scope=any|entities=sensor.fridge_switch_0_power"
    )
    assert rule_key_covers_candidate_key(rule_key, candidate_key)


def test_rule_key_covers_candidate_key_non_template_no_cross_match() -> None:
    """A rule key without |template=| must not match a structurally different candidate key."""
    rule_key = (
        "v1|subject=sensor|predicate=power_anomaly"
        "|night=any|home=any|scope=any|entities=sensor.fridge_switch_0_power"
    )
    candidate_key = (
        "v1|subject=sensor|predicate=power_anomaly"
        "|night=1|home=1|scope=any|entities=sensor.fridge_switch_0_power"
    )
    assert not rule_key_covers_candidate_key(rule_key, candidate_key)

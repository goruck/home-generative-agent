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


def test_candidate_semantic_key_strips_quoted_entity_ids() -> None:
    """Quoted evidence-path entity IDs yield the same semantic key as unquoted."""
    quoted = {
        "candidate_id": "lock_unlocked_night",
        "title": "Lock unlocked at night",
        "summary": "The front lock is unlocked during the night.",
        "evidence_paths": [
            "entities[entity_ids contains 'lock.front_door'].state",
            "derived.is_night",
        ],
    }
    unquoted = {
        **quoted,
        "evidence_paths": [
            "entities[entity_ids contains lock.front_door].state",
            "derived.is_night",
        ],
    }
    assert candidate_semantic_key(quoted) == candidate_semantic_key(unquoted)


def test_rule_semantic_key_motion_night_while_away_issue_516() -> None:
    rule = {
        "rule_id": "motion_detected_at_night_while_away",
        "template_id": "motion_detected_at_night_while_away",
        "params": {
            "motion_entity_ids": ["binary_sensor.xiao_esp32_c5_espectre_motion"],
        },
    }
    key = rule_semantic_key(rule)
    assert key == (
        "v1|subject=motion|predicate=active|night=1|home=0|scope=any|"
        "entities=binary_sensor.xiao_esp32_c5_espectre_motion"
    )


def test_motion_night_while_away_rule_covers_issue_516_candidate() -> None:
    """The registered rule's key covers the exact issue #516 candidate."""
    candidate = {
        "candidate_id": (
            "v1|subject=motion_sensor|predicate=state_event|night=1|home=0|"
            "scope=any|entities=binary_sensor.xiao_esp32_c5_espectre_motion"
        ),
        "title": "Motion detected at night while away",
        "summary": (
            "Trigger when a motion sensor reports ON at night and nobody is home."
        ),
        "pattern": "state_event",
        "evidence_paths": [
            "derived.is_night",
            "derived.anyone_home",
            (
                "entities[entity_ids contains "
                "binary_sensor.xiao_esp32_c5_espectre_motion].state"
            ),
        ],
    }
    rule = {
        "rule_id": "motion_detected_at_night_while_away",
        "template_id": "motion_detected_at_night_while_away",
        "params": {
            "motion_entity_ids": ["binary_sensor.xiao_esp32_c5_espectre_motion"],
        },
    }
    candidate_key = candidate_semantic_key(candidate)
    rule_key = rule_semantic_key(rule)
    assert candidate_key is not None
    assert rule_key is not None
    assert rule_key_covers_candidate_key(rule_key, candidate_key)


def test_candidate_semantic_key_evidence_path_only_away_is_home_zero() -> None:
    """'not derived.anyone_home' alone keys home=0 so activated rules dedup."""
    candidate = {
        "title": "Motion detected at night",
        "summary": "Trigger when a motion sensor reports ON at night.",
        "pattern": "state_event",
        "evidence_paths": [
            "derived.is_night",
            "not derived.anyone_home",
            (
                "entities[entity_ids contains "
                "binary_sensor.xiao_esp32_c5_espectre_motion].state"
            ),
        ],
    }
    key = candidate_semantic_key(candidate)
    assert key is not None
    assert "home=0" in key
    rule = {
        "rule_id": "any_slug",
        "template_id": "motion_detected_at_night_while_away",
        "params": {
            "motion_entity_ids": ["binary_sensor.xiao_esp32_c5_espectre_motion"],
        },
    }
    rule_key = rule_semantic_key(rule)
    assert rule_key is not None
    assert rule_key_covers_candidate_key(rule_key, key)


def test_rule_semantic_key_motion_night_while_away_requires_entities() -> None:
    """A motion-while-away rule without motion entities has no semantic key."""
    rule = {
        "rule_id": "motion_detected_at_night_while_away",
        "template_id": "motion_detected_at_night_while_away",
        "params": {"motion_entity_ids": []},
    }
    assert rule_semantic_key(rule) is None


def test_rule_semantic_key_motion_while_away_issue_518() -> None:
    rule = {
        "rule_id": "motion_kitchen_while_away",
        "template_id": "motion_detected_while_away",
        "params": {
            "motion_entity_ids": ["binary_sensor.xiao_esp32_c5_espectre_motion"],
        },
    }
    key = rule_semantic_key(rule)
    assert key == (
        "v1|subject=motion|predicate=active|night=any|home=0|scope=any|"
        "entities=binary_sensor.xiao_esp32_c5_espectre_motion"
    )


def test_motion_while_away_rule_covers_issue_518_candidate() -> None:
    """
    The registered rule's key covers the exact issue #518 candidate.

    The candidate's evidence paths are index-based (entities[31].state) and
    never resolve to an entity ID — the prose fallback keys the candidate on
    the sensor named in the summary so the activated rule dedups it.
    """
    candidate = {
        "candidate_id": "motion_kitchen_while_away",
        "title": "Unexpected Kitchen Motion While Away",
        "summary": (
            "Detects motion in the Kitchen area "
            "(binary_sensor.xiao_esp32_c5_espectre_motion) when no one is home."
        ),
        "pattern": "state_change",
        "evidence_paths": [
            "entities[31].state",
            "derived.anyone_home",
        ],
    }
    rule = {
        "rule_id": "motion_kitchen_while_away",
        "template_id": "motion_detected_while_away",
        "params": {
            "motion_entity_ids": ["binary_sensor.xiao_esp32_c5_espectre_motion"],
        },
    }
    candidate_key = candidate_semantic_key(candidate)
    rule_key = rule_semantic_key(rule)
    assert candidate_key is not None
    assert rule_key is not None
    assert rule_key_covers_candidate_key(rule_key, candidate_key)


def test_candidate_semantic_key_prose_fallback_requires_known_domain() -> None:
    """Dotted prose tokens without a known HA domain never key entities."""
    candidate = {
        "title": "Unexpected Kitchen Motion While Away",
        "summary": (
            "Detects motion via derived.last_motion_by_area e.g. in the "
            "Kitchen when no one is home."
        ),
        "pattern": "state_change",
        "evidence_paths": ["entities[31].state", "derived.anyone_home"],
    }
    key = candidate_semantic_key(candidate)
    assert key is not None
    assert key.endswith("entities=")


def test_candidate_semantic_key_evidence_ids_beat_prose_ids() -> None:
    """Resolvable evidence-path IDs win over IDs named in the prose."""
    candidate = {
        "title": "Unexpected Kitchen Motion While Away",
        "summary": (
            "Detects motion in the Kitchen area "
            "(binary_sensor.xiao_esp32_c5_espectre_motion) when no one is home."
        ),
        "pattern": "state_change",
        "evidence_paths": [
            "entities[entity_id=binary_sensor.hall_motion].state",
            "derived.anyone_home",
        ],
    }
    key = candidate_semantic_key(candidate)
    assert key is not None
    assert key.endswith("entities=binary_sensor.hall_motion")


def test_rule_semantic_key_motion_while_away_requires_entities() -> None:
    """A motion-while-away rule without motion entities has no semantic key."""
    rule = {
        "rule_id": "motion_kitchen_while_away",
        "template_id": "motion_detected_while_away",
        "params": {"motion_entity_ids": []},
    }
    assert rule_semantic_key(rule) is None


def test_candidate_semantic_key_prose_fallback_dedups_and_sorts_ids() -> None:
    """Multiple prose IDs (motion- and vmd-named) key deduped and sorted."""
    candidate = {
        "title": "Unexpected Motion While Away",
        "summary": (
            "Detects motion on binary_sensor.hall_motion and "
            "binary_sensor.backyard_vmd3_0; binary_sensor.hall_motion "
            "reports on when no one is home."
        ),
        "pattern": "state_change",
        "evidence_paths": ["entities[31].state", "derived.anyone_home"],
    }
    key = candidate_semantic_key(candidate)
    assert key == (
        "v1|subject=motion|predicate=active|night=any|home=0|scope=any|"
        "entities=binary_sensor.backyard_vmd3_0,binary_sensor.hall_motion"
    )


def test_candidate_semantic_key_prose_fallback_is_motion_only() -> None:
    """
    Prose IDs outside the motion class never mint coverage keys.

    The normalizer can only normalize motion IDs from prose, so a broader
    fallback would let an unresolvable lock candidate's history key suppress
    a later fully-evidenced, approvable lock proposal (issue #518 Codex
    structured review).
    """
    candidate = {
        "title": "Garage lock left unlocked",
        "summary": "lock.garage reports unlocked overnight.",
        "pattern": "state_change",
        "evidence_paths": ["entities[12].state"],
    }
    key = candidate_semantic_key(candidate)
    assert key is not None
    assert key.endswith("entities=")
    assert "subject=lock" not in key


def test_candidate_semantic_key_mixed_evidence_prose_motion_still_keys() -> None:
    """
    Non-motion evidence must not disable the prose motion fallback.

    A candidate citing a person tracker in evidence plus an index-based
    motion path normalizes from prose, so the key must match the activated
    rule's — the trigger is 'no motion evidence resolved', not 'no evidence
    at all' (issue #518 adversarial + Codex structured reviews).
    """
    candidate = {
        "candidate_id": "motion_kitchen_while_away",
        "title": "Unexpected Kitchen Motion While Away",
        "summary": (
            "Detects motion in the Kitchen area "
            "(binary_sensor.xiao_esp32_c5_espectre_motion) when no one is home."
        ),
        "pattern": "state_change",
        "evidence_paths": [
            "entities[entity_id=person.lindo].state",
            "entities[31].state",
        ],
    }
    rule = {
        "rule_id": "motion_kitchen_while_away",
        "template_id": "motion_detected_while_away",
        "params": {
            "motion_entity_ids": ["binary_sensor.xiao_esp32_c5_espectre_motion"],
        },
    }
    candidate_key = candidate_semantic_key(candidate)
    rule_key = rule_semantic_key(rule)
    assert candidate_key is not None
    assert rule_key is not None
    assert rule_key_covers_candidate_key(rule_key, candidate_key)


def test_candidate_semantic_key_door_named_motion_sensor_keys_as_motion() -> None:
    """
    binary_sensor.front_door_motion is a motion sensor, not an entry.

    Without the #516-mirror exclusion the candidate keys subject=entry_door
    while its activated rule keys subject=motion — dedup never fires for
    the most common door-named motion naming scheme (issue #518 review).
    """
    for evidence_paths in (
        ["entities[31].state", "derived.anyone_home"],  # prose fallback path
        [
            "entities[entity_id=binary_sensor.front_door_motion].state",
            "derived.anyone_home",
        ],  # evidence path
    ):
        candidate = {
            "title": "Front door motion while away",
            "summary": (
                "Detects motion on binary_sensor.front_door_motion when no one is home."
            ),
            "pattern": "state_change",
            "evidence_paths": evidence_paths,
        }
        rule = {
            "rule_id": "front_door_motion_while_away",
            "template_id": "motion_detected_while_away",
            "params": {"motion_entity_ids": ["binary_sensor.front_door_motion"]},
        }
        candidate_key = candidate_semantic_key(candidate)
        rule_key = rule_semantic_key(rule)
        assert candidate_key is not None
        assert "subject=motion" in candidate_key
        assert rule_key is not None
        assert rule_key_covers_candidate_key(rule_key, candidate_key)


def test_candidate_semantic_key_away_phrase_vocabulary_matches_normalizer() -> None:
    """
    Every away phrasing the normalizer accepts keys home=0.

    'no one at home' previously matched the bare 'home' substring and keyed
    home=1 while the activated rule keyed home=0 — no dedup (issue #518
    Codex structured review, empirically reproduced).
    """
    phrases = (
        "no one at home",
        "nobody at home",
        "no one is at home",
        "without occupants",
    )
    rule = {
        "rule_id": "motion_hall_away",
        "template_id": "motion_detected_while_away",
        "params": {"motion_entity_ids": ["binary_sensor.hall_motion"]},
    }
    rule_key = rule_semantic_key(rule)
    assert rule_key is not None
    for phrase in phrases:
        candidate = {
            "candidate_id": "motion_hall_away",
            "title": "Unexpected motion",
            "summary": f"binary_sensor.hall_motion is on when {phrase}.",
            "pattern": "state_change",
            "evidence_paths": ["entities[31].state", "derived.anyone_home"],
        }
        key = candidate_semantic_key(candidate)
        assert key is not None, phrase
        assert "home=0" in key, phrase
        assert rule_key_covers_candidate_key(rule_key, key), phrase


def test_candidate_semantic_key_prose_fallback_rejects_sensor_domain() -> None:
    """Non-binary_sensor prose motion IDs never key — mirrors the normalizer."""
    candidate = {
        "title": "Unexpected motion while away",
        "summary": "Detects motion via sensor.hall_motion_score when away.",
        "pattern": "state_change",
        "evidence_paths": ["entities[31].state", "derived.anyone_home"],
    }
    key = candidate_semantic_key(candidate)
    assert key is not None
    assert key.endswith("entities=")


def test_candidate_semantic_key_unknown_person_beats_motion_subject() -> None:
    """
    An unknown-person camera candidate never keys as plain motion.

    Keying it subject=motion would let a plain motion rule's coverage check
    swallow the sensitive camera proposal before normalization runs
    (issue #518 verification review P1).
    """
    candidate = {
        "title": "Unknown person with motion while away",
        "summary": (
            "An unknown person is detected on camera with motion on "
            "binary_sensor.kitchen_motion when no one is home."
        ),
        "pattern": "state_change",
        "evidence_paths": [
            "entities[entity_id=binary_sensor.kitchen_motion].state",
            "derived.anyone_home",
        ],
    }
    key = candidate_semantic_key(candidate)
    assert key is not None
    assert "subject=camera" in key
    assert "predicate=unknown_person" in key
    motion_rule = {
        "rule_id": "motion_kitchen_while_away",
        "template_id": "motion_detected_while_away",
        "params": {"motion_entity_ids": ["binary_sensor.kitchen_motion"]},
    }
    motion_rule_key = rule_semantic_key(motion_rule)
    assert motion_rule_key is not None
    assert not rule_key_covers_candidate_key(motion_rule_key, key)


def test_candidate_semantic_key_unknown_person_full_vocabulary() -> None:
    """Every unknown/person term the normalizer accepts keys subject=camera."""
    for unknown_term, person_term in (
        ("unidentified", "occupant"),
        ("indeterminate", "resident"),
        ("unrecognized", "face"),
    ):
        candidate = {
            "title": "Camera alert while away",
            "summary": (
                f"An {unknown_term} {person_term} is seen on camera with "
                "motion on binary_sensor.kitchen_motion when no one is home."
            ),
            "pattern": "state_change",
            "evidence_paths": [
                "entities[entity_id=binary_sensor.kitchen_motion].state",
                "derived.anyone_home",
            ],
        }
        key = candidate_semantic_key(candidate)
        assert key is not None, (unknown_term, person_term)
        assert "subject=camera" in key, (unknown_term, person_term)
        assert "predicate=unknown_person" in key, (unknown_term, person_term)


def test_candidate_semantic_key_unknown_person_beats_power_leg() -> None:
    """Camera semantics beat incidental power wording (verification round 3)."""
    candidate = {
        "title": "Unknown person while away",
        "summary": (
            "An unknown person is detected on camera during a power spike "
            "when no one is home."
        ),
        "pattern": "state_change",
        "evidence_paths": [
            "camera_activity[entity_id=camera.backyard]",
            "derived.anyone_home",
        ],
    }
    key = candidate_semantic_key(candidate)
    assert key is not None
    assert "predicate=unknown_person" in key
    assert "power_anomaly" not in key


def test_candidate_semantic_key_cam_wording_matches_normalizer() -> None:
    """Bare 'cam' wording keys camera — mirrors _CAMERA_TERMS substrings."""
    candidate = {
        "title": "Unknown person on cam-based detection while away",
        "summary": (
            "An unknown person is seen via cam-based detection with motion "
            "on binary_sensor.kitchen_motion when no one is home."
        ),
        "pattern": "state_change",
        "evidence_paths": [
            "entities[entity_id=binary_sensor.kitchen_motion].state",
            "derived.anyone_home",
        ],
    }
    key = candidate_semantic_key(candidate)
    assert key is not None
    assert "predicate=unknown_person" in key


def test_candidate_semantic_key_stale_motion_never_keys_active() -> None:
    """
    A stale-sensor candidate must not key like an active motion rule.

    Discovery's novelty filter would otherwise drop it as already-covered
    before the approval gate can return the honest "unsupported"
    (verification round 4).
    """
    candidate = {
        "title": "Kitchen motion sensor stale while away",
        "summary": (
            "binary_sensor.kitchen_motion has not updated for days when no one is home."
        ),
        "pattern": "state_change",
        "evidence_paths": [
            "entities[entity_id=binary_sensor.kitchen_motion].state",
            "derived.anyone_home",
        ],
    }
    key = candidate_semantic_key(candidate)
    rule = {
        "rule_id": "motion_kitchen_while_away",
        "template_id": "motion_detected_while_away",
        "params": {"motion_entity_ids": ["binary_sensor.kitchen_motion"]},
    }
    rule_key = rule_semantic_key(rule)
    assert rule_key is not None
    assert key is None or not rule_key_covers_candidate_key(rule_key, key)


def test_candidate_semantic_key_lock_battery_beats_camera_leg() -> None:
    """
    A compound lock + low-battery candidate keys battery, not camera.

    Mirrors the normalizer's lock-battery branch precedence
    (verification round 4).
    """
    candidate = {
        "title": "Front door lock battery low, unknown person on camera",
        "summary": (
            "The front door lock battery is below 20% and an unknown person "
            "was seen on camera."
        ),
        "pattern": "state_change",
        "evidence_paths": [
            "entities[entity_id=lock.front_door].state",
            "entities[entity_id=sensor.front_door_lock_battery].state",
            "camera_activity[entity_id=camera.porch]",
        ],
    }
    key = candidate_semantic_key(candidate)
    assert key is not None
    assert "predicate=low_battery" in key
    assert "unknown_person" not in key


def test_low_battery_rule_covers_issue_522_candidate() -> None:
    """
    The registered rule's key covers the exact issue #522 candidate.

    The candidate's prose is Czech and its evidence path uses the
    bare-bracket format (entities[sensor.x].state) — the key must still
    resolve the sensor and read the low-battery predicate from the
    candidate_id slug so the activated rule dedups re-proposals.
    """
    candidate = {
        "candidate_id": "zamek_vrata_baterie_low_battery",
        "title": "Nízká baterie zámku dveří",
        "summary": (
            "Baterie senzoru sensor.zamek_vrata_baterie klesla pod "
            "nastavenou hranici kritické kapacity."
        ),
        "pattern": "threshold_breach",
        "evidence_paths": ["entities[sensor.zamek_vrata_baterie].state"],
    }
    rule = {
        "rule_id": "zamek_vrata_baterie_low_battery",
        "template_id": "low_battery_sensors",
        "params": {
            "sensor_entity_ids": ["sensor.zamek_vrata_baterie"],
            "threshold": 40.0,
        },
    }
    candidate_key = candidate_semantic_key(candidate)
    rule_key = rule_semantic_key(rule)
    assert candidate_key is not None
    assert rule_key is not None
    assert rule_key_covers_candidate_key(rule_key, candidate_key)


def test_candidate_semantic_key_bare_bracket_entity_format() -> None:
    """Bare-bracket evidence resolves entity IDs; index brackets never do."""
    candidate = {
        "title": "Low battery on the hall sensor",
        "summary": "The hall sensor battery is below 20%.",
        "pattern": "threshold_breach",
        "evidence_paths": [
            "entities[sensor.hall_battery].state",
            "entities[31].state",
        ],
    }
    key = candidate_semantic_key(candidate)
    assert key is not None
    assert key.endswith("entities=sensor.hall_battery")


def test_candidate_semantic_key_bare_bracket_quoted_and_empty_tokens() -> None:
    """Quoted bare-bracket tokens are stripped; empty brackets resolve nothing."""
    candidate = {
        "title": "Low battery on the hall sensor",
        "summary": "The hall sensor battery is below 20%.",
        "pattern": "threshold_breach",
        "evidence_paths": [
            "entities['sensor.hall_battery'].state",
            "entities[].state",
        ],
    }
    key = candidate_semantic_key(candidate)
    assert key is not None
    assert key.endswith("entities=sensor.hall_battery")


def test_candidate_semantic_key_lock_slug_battery_predicate() -> None:
    """
    A Czech lock candidate keys predicate=low_battery from its slug.

    Mirrors the normalizer's slug-driven lock-battery routing (issue #522):
    prose carries no English battery keyword, so the predicate must come
    from battery_text or the candidate would key predicate=unknown and
    never dedup against low-battery coverage. The subject is normalized to
    sensor (not lock) because the registered low_battery_sensors rule keys
    subject=sensor on the battery sensor — a lock subject would never be
    covered by its own activated rule (issue #522 adversarial review).
    """
    candidate = {
        "candidate_id": "zamek_vrata_baterie_low_battery",
        "title": "Nízká baterie zámku dveří",
        "summary": (
            "Baterie senzoru sensor.zamek_vrata_baterie klesla pod "
            "nastavenou hranici kritické kapacity."
        ),
        "pattern": "threshold_breach",
        "evidence_paths": [
            "entities[lock.zamek_vrata].state",
            "entities[sensor.zamek_vrata_baterie].state",
        ],
    }
    key = candidate_semantic_key(candidate)
    assert key is not None
    assert "subject=sensor" in key
    assert "predicate=low_battery" in key
    assert key.endswith("entities=sensor.zamek_vrata_baterie")


def test_candidate_semantic_key_bare_bracket_attribute_suffix_mirrors() -> None:
    """
    An attribute-suffixed bracket token keys the same entity the normalizer sees.

    entities[sensor.x.state] (LLM variance of the #522 path) prefix-resolves
    to sensor.x in the normalizer — a fullmatch-only check here would key
    entities= empty and the activated rule could never dedup re-proposals
    (issue #522 testing review, empirically reproduced).
    """
    candidate = {
        "candidate_id": "zamek_vrata_baterie_low_battery",
        "title": "Nízká baterie zámku dveří",
        "summary": (
            "Baterie senzoru sensor.zamek_vrata_baterie klesla pod "
            "nastavenou hranici kritické kapacity."
        ),
        "pattern": "threshold_breach",
        "evidence_paths": ["entities[sensor.zamek_vrata_baterie.state]"],
    }
    rule = {
        "rule_id": "zamek_vrata_baterie_low_battery",
        "template_id": "low_battery_sensors",
        "params": {
            "sensor_entity_ids": ["sensor.zamek_vrata_baterie"],
            "threshold": 40.0,
        },
    }
    candidate_key = candidate_semantic_key(candidate)
    rule_key = rule_semantic_key(rule)
    assert candidate_key is not None
    assert rule_key is not None
    assert rule_key_covers_candidate_key(rule_key, candidate_key)


def test_candidate_semantic_key_bare_bracket_rejects_non_ha_domains() -> None:
    """
    Snapshot-path and pseudo-domain bracket tokens never key entities.

    entities[derived.entry_open_count] resolves nothing in the normalizer;
    keying subject=entry_door with a pseudo-entity here would diverge the
    candidate key from any registrable rule key (issue #522 review).
    """
    candidate = {
        "title": "Door left open at night",
        "summary": "The door has been open at night.",
        "pattern": "threshold_breach",
        "evidence_paths": [
            "entities[derived.entry_open_count].state",
            "entities[attributes.window_state].state",
        ],
    }
    key = candidate_semantic_key(candidate)
    assert key is None or "derived." not in key
    assert key is None or "attributes." not in key


def test_candidate_semantic_key_weak_wording_keys_low_battery() -> None:
    """Key derivation treats "weak" as a unified low-battery qualifier."""
    candidate = {
        "candidate_id": "hall_sensor_battery_weak",
        "title": "Hall sensor battery weak",
        "summary": "The hall sensor battery is weak.",
        "pattern": "threshold_breach",
        "evidence_paths": ["entities[sensor.hall_battery].state"],
    }
    key = candidate_semantic_key(candidate)
    assert key is not None
    assert "predicate=low_battery" in key


def test_lock_battery_candidate_keys_sensor_subject() -> None:
    """
    Lock + battery evidence keys subject=sensor on the battery sensor.

    The normalizer registers a subject=sensor low_battery rule regardless of
    lock evidence — keying subject=lock|entities=lock.* would mean the
    activated rule never covers re-proposals (issue #522 adversarial
    review).
    """
    candidate = {
        "candidate_id": "zamek_vrata_baterie_low_battery",
        "title": "Nízká baterie zámku dveří",
        "summary": (
            "Baterie senzoru sensor.zamek_vrata_baterie klesla pod "
            "nastavenou hranici kritické kapacity."
        ),
        "pattern": "threshold_breach",
        "evidence_paths": [
            "entities[lock.zamek_vrata].state",
            "entities[sensor.zamek_vrata_baterie].state",
        ],
    }
    rule = {
        "rule_id": "zamek_vrata_baterie_low_battery",
        "template_id": "low_battery_sensors",
        "params": {
            "sensor_entity_ids": ["sensor.zamek_vrata_baterie"],
            "threshold": 40.0,
        },
    }
    candidate_key = candidate_semantic_key(candidate)
    rule_key = rule_semantic_key(rule)
    assert candidate_key is not None
    assert "subject=sensor" in candidate_key
    assert rule_key is not None
    assert rule_key_covers_candidate_key(rule_key, candidate_key)


def test_battery_candidate_night_context_still_covered_by_rule() -> None:
    """
    Night/occupancy context is neutralized for low_battery candidate keys.

    rule_semantic_key hardcodes night=any|home=any for low_battery_sensors;
    without neutralization a night-worded battery candidate would be
    re-proposed indefinitely (issue #522 red-team review).
    """
    candidate = {
        "candidate_id": "hall_battery_low_at_night",
        "title": "Hall sensor battery low at night while someone home",
        "summary": (
            "The battery of sensor.hall_battery is low at night while someone is home."
        ),
        "pattern": "threshold_breach",
        "evidence_paths": [
            "entities[sensor.hall_battery].state",
            "derived.is_night",
            "derived.anyone_home",
        ],
    }
    rule = {
        "rule_id": "hall_battery_low_at_night",
        "template_id": "low_battery_sensors",
        "params": {"sensor_entity_ids": ["sensor.hall_battery"], "threshold": 40.0},
    }
    candidate_key = candidate_semantic_key(candidate)
    rule_key = rule_semantic_key(rule)
    assert candidate_key is not None
    assert rule_key is not None
    assert rule_key_covers_candidate_key(rule_key, candidate_key)


def test_lock_battery_precedence_beats_unlocked_prose() -> None:
    """
    A compound lock-battery candidate keys low_battery despite "unlocked" prose.

    The normalizer's lock-battery branch precedes its unlocked-lock branch,
    so the key's predicate chain must apply the same precedence or the
    candidate never matches its activated battery rule (issue #522 Codex
    verification round).
    """
    candidate = {
        "candidate_id": "zamek_vrata_baterie_low_battery",
        "title": "Nízká baterie zámku dveří",
        "summary": (
            "Baterie senzoru sensor.zamek_vrata_baterie klesla; the door "
            "may be left unlocked and open."
        ),
        "pattern": "threshold_breach",
        "evidence_paths": [
            "entities[lock.zamek_vrata].state",
            "entities[sensor.zamek_vrata_baterie].state",
        ],
    }
    rule = {
        "rule_id": "zamek_vrata_baterie_low_battery",
        "template_id": "low_battery_sensors",
        "params": {
            "sensor_entity_ids": ["sensor.zamek_vrata_baterie"],
            "threshold": 40.0,
        },
    }
    candidate_key = candidate_semantic_key(candidate)
    rule_key = rule_semantic_key(rule)
    assert candidate_key is not None
    assert "predicate=low_battery" in candidate_key
    assert rule_key is not None
    assert rule_key_covers_candidate_key(rule_key, candidate_key)


def test_unavailable_lock_battery_keeps_availability_predicate() -> None:
    """
    Availability wording outranks the lock-battery key hoist.

    The normalizer's availability branch precedes its lock-battery branch,
    so an unavailable lock-battery sensor routes to unavailable_sensors —
    the key must agree or the active availability rule never covers the
    candidate (issue #522 verification round 2).
    """
    candidate = {
        "candidate_id": "zamek_vrata_baterie_low_battery",
        "title": "Lock battery sensor unavailable",
        "summary": (
            "The battery sensor sensor.zamek_vrata_baterie of lock.zamek_vrata "
            "is low and has become unavailable."
        ),
        "pattern": "availability",
        "evidence_paths": [
            "entities[lock.zamek_vrata].state",
            "entities[sensor.zamek_vrata_baterie].state",
        ],
    }
    key = candidate_semantic_key(candidate)
    assert key is not None
    assert "predicate=unavailable" in key

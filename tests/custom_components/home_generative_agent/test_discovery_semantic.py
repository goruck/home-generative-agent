# ruff: noqa: S101
"""Tests for deterministic discovery semantic keys."""

from __future__ import annotations

import pytest

from custom_components.home_generative_agent.sentinel.discovery_semantic import (
    candidate_semantic_key,
    is_battery_level_entity_id,
    rule_key_covers_candidate_key,
    rule_semantic_key,
)
from custom_components.home_generative_agent.sentinel.proposal_templates import (
    normalize_candidate,
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
    """
    A rule key without |template=| must not match a structurally different key.

    power_anomaly is not a superset-safe predicate, so night/home
    any-vs-specific is NOT coverage here — only the unavailable family
    opts in (see test_rule_key_night_home_any_covers_scoped_candidate).
    """
    rule_key = (
        "v1|subject=sensor|predicate=power_anomaly"
        "|night=any|home=any|scope=any|entities=sensor.fridge_switch_0_power"
    )
    candidate_key = (
        "v1|subject=sensor|predicate=power_anomaly"
        "|night=1|home=1|scope=any|entities=sensor.fridge_switch_0_power"
    )
    assert not rule_key_covers_candidate_key(rule_key, candidate_key)


def test_rule_key_night_home_any_covers_scoped_candidate() -> None:
    """
    An any/any unavailable rule key covers its night/home-scoped variant.

    The unavailable family is the superset-safe predicate: the
    unconditional evaluator differs from while_home only by the occupancy
    gate, so the any/any rule genuinely fires in a superset of the scoped
    candidate's conditions (issue #524 red-team).
    """
    rule_key = (
        "v1|subject=sensor|predicate=unavailable"
        "|night=any|home=any|scope=any|entities=sensor.backyard_vmd3_0"
    )
    candidate_key = (
        "v1|subject=sensor|predicate=unavailable"
        "|night=any|home=1|scope=any|entities=sensor.backyard_vmd3_0"
    )
    assert rule_key_covers_candidate_key(rule_key, candidate_key)


def test_rule_key_qualified_template_predicates_never_superset_cover() -> None:
    """
    Predicates with extra-key firing qualifiers never any-cover.

    motion_without_camera_activity keys predicate=active|night=any|home=any
    but fires only while cameras are idle; alarm_disarmed_open_entry keys
    predicate=open but fires only while the alarm is disarmed. Treating
    those as supersets falsely reports distinct night/away candidates as
    already covered — the promote service would return "already_active"
    (issue #524 adversarial review, empirically reproduced).
    """
    motion_rule_key = (
        "v1|subject=motion|predicate=active"
        "|night=any|home=any|scope=any|entities=binary_sensor.hall_motion"
    )
    motion_candidate_key = (
        "v1|subject=motion|predicate=active"
        "|night=1|home=0|scope=any|entities=binary_sensor.hall_motion"
    )
    assert not rule_key_covers_candidate_key(motion_rule_key, motion_candidate_key)
    entry_rule_key = (
        "v1|subject=entry_door|predicate=open"
        "|night=any|home=any|scope=any|entities=binary_sensor.front_door"
    )
    entry_candidate_key = (
        "v1|subject=entry_door|predicate=open"
        "|night=1|home=0|scope=any|entities=binary_sensor.front_door"
    )
    assert not rule_key_covers_candidate_key(entry_rule_key, entry_candidate_key)


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


# ---------------------------------------------------------------------------
# Issue #524: structured occupancy evidence in candidate keys
# ---------------------------------------------------------------------------


def test_candidate_semantic_key_anyone_home_false_expression_keys_away() -> None:
    """
    An anyone_home == false expression in the pattern keys home=0.

    Intentional key change (issue #524): this class previously keyed
    home=any while the normalizer already routed it to a home=0 template
    via the same expression, so the candidate never deduped against its
    activated rule. A candidate stored under the old key may be re-proposed
    once, then dedups under the new key.
    """
    candidate = {
        "candidate_id": "motion_away_expression",
        "title": "Pohyb, když nikdo není doma",
        "summary": "Detekuje pohyb v kuchyni, když nikdo není doma.",
        "pattern": ("binary_sensor.kitchen_motion == on AND anyone_home == false"),
        "suggested_type": "security",
        "evidence_paths": [
            "entities[entity_id=binary_sensor.kitchen_motion].state",
        ],
    }
    key = candidate_semantic_key(candidate)
    assert key is not None
    assert "home=0" in key


def test_candidate_semantic_key_negation_variant_keys_away() -> None:
    """A 'NOT derived.anyone_home' spelling variant keys home=0."""
    candidate = {
        "candidate_id": "motion_away_variant",
        "title": "Pohyb, když nikdo není doma",
        "summary": "Detekuje pohyb v kuchyni, když nikdo není doma.",
        "pattern": "binary_sensor.kitchen_motion == on",
        "suggested_type": "security",
        "evidence_paths": [
            "entities[entity_id=binary_sensor.kitchen_motion].state",
            "NOT derived.anyone_home",
        ],
    }
    key = candidate_semantic_key(candidate)
    assert key is not None
    assert "home=0" in key


def test_candidate_semantic_key_non_english_positive_path_keys_home() -> None:
    """Czech prose + derived.anyone_home keys home=1 via the structured path."""
    candidate = {
        "candidate_id": "senzor_nedostupny_doma",
        "title": "Nedostupné senzory, když je někdo doma",
        "summary": "Detekuje senzor hlásící nedostupnost, když je někdo doma.",
        "pattern": "sensor.backyard_vmd3_0 == 'unavailable'",
        "suggested_type": "availability",
        "evidence_paths": [
            "derived.anyone_home",
            "entities[entity_id=sensor.backyard_vmd3_0].state",
        ],
    }
    key = candidate_semantic_key(candidate)
    assert key is not None
    assert "home=1" in key


def test_candidate_semantic_key_non_english_night_variant() -> None:
    """A canonicalization variant of derived.is_night keys night=1."""
    candidate = {
        "candidate_id": "okno_v_noci",
        "title": "Otevřené okno v noci",
        "summary": "Okno v herně je otevřené v noci.",
        "pattern": "binary_sensor.playroom_window == on",
        "suggested_type": "security",
        "evidence_paths": [
            "entities[entity_id=binary_sensor.playroom_window].state",
            "Derived.Is_Night",
        ],
    }
    key = candidate_semantic_key(candidate)
    assert key is not None
    assert "night=1" in key


def test_motion_while_away_rule_covers_non_english_candidate() -> None:
    """
    A Czech away-motion candidate dedups against its activated rule.

    The #524 goal end-to-end: with no English prose, the structured
    'not derived.anyone_home' path keys home=0, matching the
    motion_detected_while_away rule key. (Subject/predicate resolve from
    the machine pattern — the entity ID carries the 'motion' token.)
    """
    candidate = {
        "candidate_id": "pohyb_v_kuchyni_pryc",
        "title": "Neočekávaný pohyb v kuchyni, když nikdo není doma",
        "summary": "Detekuje pohyb v kuchyni, když nikdo není doma.",
        "pattern": "binary_sensor.kitchen_motion == on",
        "suggested_type": "security",
        "evidence_paths": [
            "entities[entity_id=binary_sensor.kitchen_motion].state",
            "not derived.anyone_home",
        ],
    }
    rule = {
        "rule_id": "motion_kitchen_while_away",
        "template_id": "motion_detected_while_away",
        "params": {"motion_entity_ids": ["binary_sensor.kitchen_motion"]},
    }
    candidate_key = candidate_semantic_key(candidate)
    rule_key = rule_semantic_key(rule)
    assert candidate_key is not None
    assert rule_key is not None
    assert rule_key_covers_candidate_key(rule_key, candidate_key)


def test_unavailable_while_home_rule_covers_non_english_candidate() -> None:
    """
    A Czech home-unavailable candidate dedups against its activated rule.

    Exercises the new positive-path tier: home=1 comes solely from the
    bare derived.anyone_home evidence path (issue #524).
    """
    candidate = {
        "candidate_id": "senzor_nedostupny_doma",
        "title": "Nedostupné senzory, když je někdo doma",
        "summary": "Detekuje senzor hlásící nedostupnost, když je někdo doma.",
        "pattern": "sensor.backyard_vmd3_0 == 'unavailable'",
        "suggested_type": "availability",
        "evidence_paths": [
            "derived.anyone_home",
            "entities[entity_id=sensor.backyard_vmd3_0].state",
        ],
    }
    rule = {
        "rule_id": "unavailable_sensors_while_home",
        "template_id": "unavailable_sensors_while_home",
        "params": {"sensor_entity_ids": ["sensor.backyard_vmd3_0"]},
    }
    candidate_key = candidate_semantic_key(candidate)
    rule_key = rule_semantic_key(rule)
    assert candidate_key is not None
    assert rule_key is not None
    assert rule_key_covers_candidate_key(rule_key, candidate_key)


def test_rule_key_home_any_covers_specific_home_candidate() -> None:
    """
    A home=any rule covers a home=1 candidate for the same idea.

    A pre-#524 approved unavailable_sensors rule keys home=any; structured
    evidence now keys the same idea home=1. Without superset coverage the
    candidate is re-proposed as a new while_home rule and approving it
    double-alerts on every occupied-hours outage (issue #524 red-team).
    """
    rule = {
        "rule_id": "unavailable_sensors",
        "template_id": "unavailable_sensors",
        "params": {"sensor_entity_ids": ["sensor.backyard_vmd3_0"]},
    }
    candidate = {
        "candidate_id": "senzor_nedostupny_doma",
        "title": "Nedostupné senzory, když je někdo doma",
        "summary": "Detekuje senzor hlásící nedostupnost, když je někdo doma.",
        "pattern": "sensor.backyard_vmd3_0 == 'unavailable'",
        "suggested_type": "availability",
        "evidence_paths": [
            "derived.anyone_home",
            "entities[entity_id=sensor.backyard_vmd3_0].state",
        ],
    }
    rule_key = rule_semantic_key(rule)
    candidate_key = candidate_semantic_key(candidate)
    assert rule_key is not None
    assert candidate_key is not None
    assert "home=any" in rule_key
    assert "home=1" in candidate_key
    assert rule_key_covers_candidate_key(rule_key, candidate_key)


def test_rule_key_specific_home_does_not_cover_any_candidate() -> None:
    """The converse is not coverage: a while_home rule is silent while away."""
    rule = {
        "rule_id": "unavailable_sensors_while_home",
        "template_id": "unavailable_sensors_while_home",
        "params": {"sensor_entity_ids": ["sensor.backyard_vmd3_0"]},
    }
    candidate = {
        "candidate_id": "sensor_unavailable_any",
        "title": "Unavailable sensors",
        "summary": "Detects a sensor reporting unavailable.",
        "pattern": "sensor.backyard_vmd3_0 == 'unavailable'",
        "suggested_type": "availability",
        "evidence_paths": [
            "entities[entity_id=sensor.backyard_vmd3_0].state",
        ],
    }
    rule_key = rule_semantic_key(rule)
    candidate_key = candidate_semantic_key(candidate)
    assert rule_key is not None
    assert candidate_key is not None
    assert not rule_key_covers_candidate_key(rule_key, candidate_key)


def test_rule_key_home_any_does_not_cover_different_entities() -> None:
    """Superset coverage never crosses entity or predicate boundaries."""
    rule = {
        "rule_id": "unavailable_sensors",
        "template_id": "unavailable_sensors",
        "params": {"sensor_entity_ids": ["sensor.other_sensor"]},
    }
    candidate = {
        "candidate_id": "senzor_nedostupny_doma",
        "title": "Nedostupné senzory, když je někdo doma",
        "summary": "Detekuje senzor hlásící nedostupnost, když je někdo doma.",
        "pattern": "sensor.backyard_vmd3_0 == 'unavailable'",
        "suggested_type": "availability",
        "evidence_paths": [
            "derived.anyone_home",
            "entities[entity_id=sensor.backyard_vmd3_0].state",
        ],
    }
    rule_key = rule_semantic_key(rule)
    candidate_key = candidate_semantic_key(candidate)
    assert rule_key is not None
    assert candidate_key is not None
    assert not rule_key_covers_candidate_key(rule_key, candidate_key)


def test_candidate_semantic_key_negated_text_in_pattern_keys_away() -> None:
    """'not derived.anyone_home' inside the pattern text keys home=0."""
    candidate = {
        "candidate_id": "pohyb_pryc_text",
        "title": "Pohyb, když nikdo není doma",
        "summary": "Detekuje pohyb v kuchyni, když nikdo není doma.",
        "pattern": "binary_sensor.kitchen_motion == on AND not derived.anyone_home",
        "suggested_type": "security",
        "evidence_paths": [
            "entities[entity_id=binary_sensor.kitchen_motion].state",
        ],
    }
    key = candidate_semantic_key(candidate)
    assert key is not None
    assert "home=0" in key


def test_is_battery_level_entity_id_classification() -> None:
    """
    Battery-level classification gates baseline eligibility.

    Battery-named sensor.* IDs are charge levels unless they carry a
    measurement token (sensor.battery_power on a home battery is a real
    power stream); non-sensor domains never qualify.
    """
    assert is_battery_level_entity_id("sensor.garage_temp_sensor_battery")
    assert is_battery_level_entity_id("sensor.playroom_attic_sensor_battery")
    assert is_battery_level_entity_id("sensor.hall_battery")
    # Device-named battery levels: device-type tokens (door/motion/smoke)
    # and measurement tokens BEFORE the battery token name the device, not
    # the reading — they must not disqualify the charge level (pre-landing
    # review + adversarial round; canonical HA device-battery naming).
    assert is_battery_level_entity_id("sensor.front_door_battery")
    assert is_battery_level_entity_id("sensor.hallway_motion_battery")
    assert is_battery_level_entity_id("sensor.bedroom_window_battery")
    assert is_battery_level_entity_id("sensor.smoke_detector_battery")
    assert is_battery_level_entity_id("sensor.water_leak_battery")
    assert is_battery_level_entity_id("sensor.garage_temperature_sensor_battery")
    # Measurement tokens AFTER the battery token are telemetry streams of a
    # home battery, not charge levels — they stay baseline-eligible.
    assert not is_battery_level_entity_id("sensor.battery_power")
    assert not is_battery_level_entity_id("sensor.battery_voltage")
    assert not is_battery_level_entity_id("sensor.battery_temperature")
    assert not is_battery_level_entity_id("sensor.battery_monitor_rssi")
    assert not is_battery_level_entity_id("sensor.garage_temperature")
    assert not is_battery_level_entity_id("binary_sensor.hall_battery_low")
    assert not is_battery_level_entity_id("lock.front_door")


def test_battery_baseline_candidate_keys_low_battery_context_free() -> None:
    """
    A battery baseline-deviation candidate keys low_battery with no context.

    The discovery LLM decorates battery candidates with occupancy/night
    conditioning ("Battery Anomaly During Night While Home") even though
    battery health is occupancy-independent, and its baseline prose carries
    no low-battery qualifier. The key must mirror the normalizer, which
    routes battery-named evidence plus battery prose to low_battery_sensors
    (night=any|home=any) — otherwise the candidate keys power_anomaly with
    context preserved and the activated rule never dedups re-proposals.
    """
    candidate = {
        "candidate_id": (
            "candidate_garage_temp_sensor_battery_baseline_deviation_home_night"
        ),
        "title": "Garage Temp Sensor Battery Anomaly During Night While Home",
        "summary": (
            "Detects statistical deviation in the battery level of the garage "
            "temperature and humidity sensor during nighttime hours while "
            "someone is home, indicating potential sensor degradation."
        ),
        "pattern": "statistical_baseline_deviation",
        "evidence_paths": [
            "entities[sensor.garage_temp_sensor_battery].state",
            "derived.is_night",
            "derived.anyone_home",
        ],
    }
    rule = {
        "rule_id": "garage_temp_sensor_battery_low",
        "template_id": "low_battery_sensors",
        "params": {
            "sensor_entity_ids": ["sensor.garage_temp_sensor_battery"],
            "threshold": 40.0,
        },
    }
    expected_key = (
        "v1|subject=sensor|predicate=low_battery|night=any|home=any|scope=any|"
        "entities=sensor.garage_temp_sensor_battery"
    )
    # Both sides asserted against the explicit literal so a shared-helper
    # mutation cannot silently turn this into f(x) == f(x).
    assert candidate_semantic_key(candidate) == expected_key
    assert rule_semantic_key(rule) == expected_key
    assert rule_key_covers_candidate_key(expected_key, expected_key)


def test_battery_baseline_context_variants_collapse_to_one_key() -> None:
    """
    Home/away/night battery baseline variants all produce one identical key.

    Without battery context canonicalization each occupancy/night decoration
    is a distinct semantic key, so near-duplicate battery candidates pile up
    in the discovery card instead of deduplicating.
    """
    home_night = {
        "candidate_id": (
            "candidate_garage_temp_sensor_battery_baseline_deviation_home_night"
        ),
        "title": "Garage Temp Sensor Battery Anomaly During Night While Home",
        "summary": (
            "Detects statistical deviation in the battery level of the garage "
            "temperature and humidity sensor during nighttime hours while "
            "someone is home."
        ),
        "pattern": "statistical_baseline_deviation",
        "evidence_paths": [
            "entities[sensor.garage_temp_sensor_battery].state",
            "derived.is_night",
            "derived.anyone_home",
        ],
    }
    away_night = {
        "candidate_id": (
            "candidate_garage_temp_humidity_battery_baseline_deviation_away_night"
        ),
        "title": (
            "Garage Temp/Humidity Sensor Battery Baseline Deviation While Away at Night"
        ),
        "summary": (
            "Detects statistical deviation in the garage temperature and "
            "humidity sensor battery level from its normal baseline while "
            "no one is home during nighttime hours."
        ),
        "pattern": "statistical_anomaly",
        "evidence_paths": [
            "entities[sensor.garage_temp_sensor_battery].state",
            "not derived.anyone_home",
            "derived.is_night",
        ],
    }
    night_only = {
        "candidate_id": "candidate_garage_temp_battery_time_of_day_anomaly_night",
        "title": "Garage Temp Sensor Battery Anomaly at Night",
        "summary": (
            "Detects statistical deviation in the battery level of the garage "
            "temperature sensor during nighttime hours when usage patterns "
            "typically stabilize."
        ),
        "pattern": "statistical_baseline",
        "evidence_paths": [
            "entities[sensor.garage_temp_sensor_battery].state",
            "derived.is_night",
        ],
    }
    keys = {
        candidate_semantic_key(home_night),
        candidate_semantic_key(away_night),
        candidate_semantic_key(night_only),
    }
    assert keys == {
        (
            "v1|subject=sensor|predicate=low_battery|night=any|home=any|scope=any|"
            "entities=sensor.garage_temp_sensor_battery"
        )
    }


def test_battery_arm_alternate_needles_key_low_battery() -> None:
    """
    "below"/"low" prose without the word "battery" still keys the battery arm.

    The disjunctive arm's needle tuple is (battery, low, below); every other
    battery test carries "battery" in prose, so without this test a mutation
    dropping the low/below needles passes the suite (pre-landing review).
    """
    candidate = {
        "candidate_id": "candidate_garage_sensor_cell_dropping",
        "title": "Garage sensor cell level dropping below its usual range",
        # No "battery" anywhere in prose or slug: the conjunctive signal must
        # stay silent so this test pins the arm's "below"/"low" needles alone.
        "summary": (
            "Detects when the charge level of the garage climate sensor "
            "drifts below its usual range."
        ),
        "pattern": "statistical_baseline_deviation",
        "evidence_paths": ["entities[sensor.garage_temp_sensor_battery].state"],
    }
    key = candidate_semantic_key(candidate)
    assert key == (
        "v1|subject=sensor|predicate=low_battery|night=any|home=any|scope=any|"
        "entities=sensor.garage_temp_sensor_battery"
    )


def test_battery_arm_requires_battery_low_below_prose_token() -> None:
    """
    Battery-named evidence with token-free prose must not key low_battery.

    The arm's prose guard mirrors the normalizer's disjunctive condition;
    without any of battery/low/below in text the candidate keeps the power
    leg's keying, so a mutation forcing the guard true is caught here
    (pre-landing review).
    """
    candidate = {
        "candidate_id": "candidate_garage_sensor_baseline",
        "title": "Garage sensor baseline deviation",
        "summary": (
            "Detects statistical deviation from the normal baseline for this sensor."
        ),
        "pattern": "statistical_baseline_deviation",
        "evidence_paths": ["entities[sensor.garage_battery].state"],
    }
    key = candidate_semantic_key(candidate)
    assert key is not None
    assert "predicate=power_anomaly" in key
    assert "predicate=low_battery" not in key


def test_battery_arm_locale_fallback_does_not_widen_disjunctive_leg() -> None:
    """
    Named IDs only: a lone locale-named sensor + "low" prose keys power_anomaly.

    The normalizer promotes its locale fallback solely on the conjunctive
    low-battery signal, so the disjunctive arm must not fire on fallback IDs —
    a mutation swapping _named_battery_sensor_entity_ids for the
    fallback-inclusive collection passes every other test (pre-landing
    review; issue #522 locale-fallback contract).
    """
    candidate = {
        "candidate_id": "candidate_zamek_vrata_low_reading",
        "title": "Sensor reading unusually low",
        "summary": (
            "Detects when sensor.zamek_vrata reads low compared to its normal baseline."
        ),
        "pattern": "statistical_baseline_deviation",
        "evidence_paths": ["entities[sensor.zamek_vrata].state"],
    }
    key = candidate_semantic_key(candidate)
    assert key is not None
    assert "predicate=power_anomaly" in key
    assert "predicate=low_battery" not in key


def test_battery_arm_does_not_steal_motion_camera_candidate() -> None:
    """
    Mixed motion+camera evidence with incidental battery keeps motion keying.

    Adversarial round (pre-landing, empirically reproduced regression): the
    normalizer's motion_without_camera_activity branch precedes its battery
    branch, so a candidate citing motion + camera + a battery sensor with
    bare "low" prose ("camera activity stays low") must NOT key low_battery —
    that would let an activated low_battery rule on the incidental sensor
    silently swallow the unrelated motion proposal, and the activated motion
    rule would never dedup it.
    """
    candidate = {
        "candidate_id": "candidate_hall_motion_without_camera_activity",
        "title": "Motion with low camera activity",
        # No "battery" in prose: with it, the PRE-EXISTING conjunctive arm
        # would fire (battery + "low"), which is not the regression under
        # test — the disjunctive arm alone must not steal this candidate.
        "summary": (
            "Detects hallway motion events while camera activity stays low; "
            "the hub charge sensor is cited for device context."
        ),
        "pattern": "motion_without_camera_activity",
        "evidence_paths": [
            "entities[binary_sensor.hall_motion].state",
            "entities[camera.hallway].state",
            "entities[sensor.hub_battery].state",
        ],
    }
    key = candidate_semantic_key(candidate)
    assert key is not None
    assert "predicate=low_battery" not in key
    assert "subject=motion" in key or "subject=camera" in key


def test_battery_arm_does_not_steal_alarm_motion_candidate() -> None:
    """
    Alarm+motion evidence with "below" prose keeps its motion keying.

    Second reproduced shape from the adversarial round: "readings below
    normal" plus an incidental battery sensor must not re-key an alarm-motion
    candidate as low_battery (infinite re-proposal + cross-suppression).
    """
    candidate = {
        "candidate_id": "candidate_night_motion_alarm_inactive",
        "title": "Motion at night while alarm inactive",
        # No "battery" in prose — see the motion+camera test above; this
        # pins the disjunctive arm's "below" needle against evidence-only
        # battery citation.
        "summary": (
            "Detects motion at night while the alarm is not armed and sensor "
            "readings stay below normal; cites the hub charge sensor."
        ),
        "pattern": "motion_at_night",
        "evidence_paths": [
            "entities[alarm_control_panel.home_alarm].state",
            "entities[binary_sensor.hall_motion].state",
            "entities[sensor.hub_battery].state",
            "derived.is_night",
        ],
    }
    key = candidate_semantic_key(candidate)
    assert key is not None
    assert "predicate=low_battery" not in key
    assert "subject=motion" in key
    assert "predicate=active" in key


def test_device_named_battery_candidate_keys_low_battery() -> None:
    """
    A door-named battery sensor candidate keys low_battery, matching its rule.

    sensor.front_door_battery resolves into the key chain's door_ids (the
    "door" substring), but the normalizer's _NON_ENTRY_ID_TOKENS contains
    "battery" so it is never an entry there — its battery branch registers
    low_battery_sensors. The battery arm's sensor-only gate must therefore
    let battery-named window/door ids through (adversarial round).
    """
    candidate = {
        "candidate_id": "candidate_front_door_battery_baseline_deviation",
        "title": "Front Door Sensor Battery Baseline Deviation",
        "summary": (
            "Detects statistical deviation in the battery level of the front "
            "door contact sensor."
        ),
        "pattern": "statistical_baseline_deviation",
        "evidence_paths": ["entities[sensor.front_door_battery].state"],
    }
    rule = {
        "rule_id": "front_door_battery_low",
        "template_id": "low_battery_sensors",
        "params": {
            "sensor_entity_ids": ["sensor.front_door_battery"],
            "threshold": 40.0,
        },
    }
    expected_key = (
        "v1|subject=sensor|predicate=low_battery|night=any|home=any|scope=any|"
        "entities=sensor.front_door_battery"
    )
    assert candidate_semantic_key(candidate) == expected_key
    assert rule_semantic_key(rule) == expected_key


def test_motion_named_battery_candidate_keys_low_battery() -> None:
    """
    A motion-named battery sensor as sole evidence keys low_battery.

    sensor.hallway_motion_battery is motion-named, so it lands in the key
    chain's motion_ids — but the normalizer's away-motion branches guard on
    battery_sensor_ids and its battery branch registers low_battery_sensors,
    so the battery arm's sensor-only gate must let battery-named motion ids
    through, exactly like door/window-named battery ids (Codex structured
    review, empirically reproduced mismatch).
    """
    candidate = {
        "candidate_id": "candidate_hallway_motion_battery_baseline_deviation",
        "title": "Hallway Motion Sensor Battery Baseline Deviation",
        "summary": (
            "Detects statistical deviation in the battery level of the "
            "hallway motion sensor."
        ),
        "pattern": "statistical_baseline_deviation",
        "evidence_paths": ["entities[sensor.hallway_motion_battery].state"],
    }
    rule = {
        "rule_id": "hallway_motion_battery_low",
        "template_id": "low_battery_sensors",
        "params": {
            "sensor_entity_ids": ["sensor.hallway_motion_battery"],
            "threshold": 40.0,
        },
    }
    expected_key = (
        "v1|subject=sensor|predicate=low_battery|night=any|home=any|scope=any|"
        "entities=sensor.hallway_motion_battery"
    )
    assert candidate_semantic_key(candidate) == expected_key
    assert rule_semantic_key(rule) == expected_key


# ---------------------------------------------------------------------------
# Environmental sensor keying (issue #541)
# ---------------------------------------------------------------------------


def _env_statistical_candidate(**overrides: object) -> dict[str, object]:
    candidate: dict[str, object] = {
        "candidate_id": "candidate_attic_temperature_baseline_deviation",
        "title": "Attic Temperature Anomaly",
        "summary": (
            "Detects statistical deviation from the normal attic temperature "
            "reading, indicating overheating or ventilation failure."
        ),
        "pattern": "statistical_baseline_deviation",
        "confidence_hint": 0.6,
        "evidence_paths": ["entities[sensor.attic_temperature].state"],
    }
    candidate.update(overrides)
    return candidate


def test_environmental_statistical_key_collapses_context_variants() -> None:
    """
    Occupancy/night variants of a statistical env candidate share one key.

    The #540 battery pile-up lesson: the normalizer registers the same
    context-free baseline_deviation rule for every variant, so distinct
    context-carrying keys would let near-duplicate pending cards accumulate.
    """
    base = _env_statistical_candidate()
    night_home = _env_statistical_candidate(
        candidate_id="candidate_attic_temperature_baseline_night_home",
        title="Attic Temperature Anomaly During Night While Home",
        summary=(
            "Detects statistical deviation from the normal attic temperature "
            "reading during nighttime hours while someone is home."
        ),
        evidence_paths=[
            "entities[sensor.attic_temperature].state",
            "derived.is_night",
            "derived.anyone_home",
        ],
    )
    away = _env_statistical_candidate(
        candidate_id="candidate_attic_temperature_baseline_away",
        title="Attic Temperature Anomaly While Away",
        summary=(
            "Detects statistical deviation from the normal attic temperature "
            "reading while nobody is home."
        ),
        evidence_paths=[
            "entities[sensor.attic_temperature].state",
            "not derived.anyone_home",
        ],
    )
    expected_key = (
        "v1|subject=sensor|predicate=power_anomaly|night=any|home=any|scope=any|"
        "entities=sensor.attic_temperature"
    )
    assert candidate_semantic_key(base) == expected_key
    assert candidate_semantic_key(night_home) == expected_key
    assert candidate_semantic_key(away) == expected_key


def test_environmental_statistical_key_matches_baseline_rule_key() -> None:
    """A statistical env candidate's key is covered by its registered rule."""
    candidate_key = candidate_semantic_key(_env_statistical_candidate())
    assert candidate_key is not None
    rule = {
        "rule_id": "sensor_baseline_sensor_attic_temperature",
        "template_id": "baseline_deviation",
        "params": {"entity_id": "sensor.attic_temperature"},
    }
    rule_key = rule_semantic_key(rule)
    assert rule_key is not None
    assert rule_key_covers_candidate_key(rule_key, candidate_key)


def test_environmental_cyclical_key_matches_time_of_day_rule_key() -> None:
    """A fridge-temperature candidate's key is covered by its tod rule."""
    candidate = _env_statistical_candidate(
        candidate_id="candidate_fridge_temperature_baseline",
        title="Fridge temperature anomaly",
        summary="Detects deviation from the normal fridge temperature.",
        evidence_paths=["entities[sensor.fridge_temperature].state"],
    )
    candidate_key = candidate_semantic_key(candidate)
    assert candidate_key is not None
    rule = {
        "rule_id": "sensor_tod_sensor_fridge_temperature",
        "template_id": "time_of_day_anomaly",
        "params": {"entity_id": "sensor.fridge_temperature"},
    }
    rule_key = rule_semantic_key(rule)
    assert rule_key is not None
    assert rule_key_covers_candidate_key(rule_key, candidate_key)


def test_environmental_threshold_candidate_keeps_context() -> None:
    """
    Threshold-prose env candidates keep night/home context in the key.

    sensor_threshold_condition params carry require_night/require_away/
    require_home, so context variants are genuinely different rules and must
    NOT collapse.
    """
    night = _env_statistical_candidate(
        candidate_id="candidate_attic_temperature_threshold_night",
        title="Attic overheating at night",
        summary="Alert when attic temperature rises above 95 at night.",
        evidence_paths=[
            "entities[sensor.attic_temperature].state",
            "derived.is_night",
        ],
    )
    any_hour = _env_statistical_candidate(
        candidate_id="candidate_attic_temperature_threshold",
        title="Attic overheating",
        summary="Alert when attic temperature rises above 95.",
    )
    night_key = candidate_semantic_key(night)
    any_key = candidate_semantic_key(any_hour)
    assert night_key is not None
    assert any_key is not None
    assert "predicate=power_anomaly" in night_key
    assert "night=1" in night_key
    assert "night=any" in any_key
    assert night_key != any_key


def test_environmental_slug_only_does_not_key_power_anomaly() -> None:
    """
    The candidate_id slug alone must not fire the environmental leg.

    Mirrors the normalizer's #522 posture: locale prose plus a locale-named
    sensor with English "temperature" only in the slug stays unsupported
    there, so keying it power_anomaly would break the mirror.
    """
    candidate = _env_statistical_candidate(
        candidate_id="candidate_attic_temperature_anomaly",
        title="Odchylka v podkroví",
        summary="Sleduje neobvyklé hodnoty čidla v podkroví.",
        pattern="statisticka_odchylka",
        evidence_paths=["entities[sensor.podkrovi_cidlo].state"],
    )
    # Absolute verdict (ship testing-specialist review): pin what the
    # candidate actually keys to, not merely "not power_anomaly".
    assert candidate_semantic_key(candidate) == (
        "v1|subject=sensor|predicate=unknown|night=any|home=any|scope=any|"
        "entities=sensor.podkrovi_cidlo"
    )


def test_environmental_entity_id_signal_only_keys_context_free() -> None:
    """
    An environmental token in the entity ID alone fires the leg (#541).

    Mirrors the normalizer's locale test (issue #522 shape): locale prose
    leaves sensor.attic_temp as the only English surface, and the normalizer
    still routes to baseline_deviation — so the key must reach
    predicate=power_anomaly through the entity-id arm and collapse the
    night context, or the registered rule never dedups re-proposals.
    """
    candidate = _env_statistical_candidate(
        candidate_id="candidate_podkrovi_odchylka",
        title="Odchylka v podkroví",
        summary="Sleduje neobvyklé hodnoty čidla v podkroví.",
        pattern="statisticka_odchylka",
        evidence_paths=[
            "entities[sensor.attic_temp].state",
            "derived.is_night",
        ],
    )
    expected_key = (
        "v1|subject=sensor|predicate=power_anomaly|night=any|home=any|scope=any|"
        "entities=sensor.attic_temp"
    )
    assert candidate_semantic_key(candidate) == expected_key
    normalized = normalize_candidate(candidate)
    assert normalized is not None
    assert normalized.template_id == "baseline_deviation"
    rule_key = rule_semantic_key(
        {
            "rule_id": normalized.rule_id,
            "template_id": normalized.template_id,
            "params": normalized.params,
        }
    )
    assert rule_key is not None
    assert rule_key_covers_candidate_key(rule_key, expected_key)


def test_environmental_signal_does_not_strip_context_from_unavailable() -> None:
    """
    The env leg never strips context a non-power predicate claimed (#541).

    An unavailable temperature sensor keys predicate=unavailable with its
    occupancy context intact: _environmental_context_collapses is scoped to
    predicate=power_anomaly, so the environmental signal in prose and the
    entity ID must not collapse home=1 to home=any.
    """
    candidate = _env_statistical_candidate(
        candidate_id="candidate_attic_temperature_unavailable",
        title="Attic temperature sensor unavailable while home",
        summary=(
            "Detects the attic temperature sensor reporting unavailable "
            "while someone is home."
        ),
        pattern="availability_watch",
        evidence_paths=[
            "entities[sensor.attic_temperature].state",
            "derived.anyone_home",
        ],
    )
    key = candidate_semantic_key(candidate)
    assert key is not None
    assert "predicate=unavailable" in key
    assert "home=1" in key


def test_environmental_zero_threshold_treated_as_statistical() -> None:
    """
    "above 0" prose is not a threshold; the context still collapses (#541).

    Mirrors _extract_threshold_numeric's value>0 gate: the normalizer treats
    a zero threshold as no-threshold and registers the context-free
    baseline_deviation rule, so the key's collapse must fire despite the
    threshold-shaped wording.
    """
    candidate = _env_statistical_candidate(
        candidate_id="candidate_attic_temperature_above_zero_night",
        title="Attic temperature above 0 at night",
        summary="Alert when the attic temperature reading rises above 0 at night.",
        pattern="sensor_threshold",
        evidence_paths=[
            "entities[sensor.attic_temperature].state",
            "derived.is_night",
        ],
    )
    expected_key = (
        "v1|subject=sensor|predicate=power_anomaly|night=any|home=any|scope=any|"
        "entities=sensor.attic_temperature"
    )
    assert candidate_semantic_key(candidate) == expected_key


def test_environmental_prose_word_boundary_no_steal() -> None:
    """Prose "attempt" must not fire the temp token and collapse context."""
    candidate = _env_statistical_candidate(
        candidate_id="candidate_keypad_attempts",
        title="Repeated unlock attempts at night",
        summary=(
            "Multiple failed unlock attempts recorded by the keypad counter "
            "during nighttime hours."
        ),
        pattern="counter_watch",
        evidence_paths=[
            "entities[sensor.keypad_attempt_counter].state",
            "derived.is_night",
        ],
    )
    # Absolute verdict (ship testing-specialist review): the candidate keys
    # subject=sensor|predicate=unknown with its night context intact.
    assert candidate_semantic_key(candidate) == (
        "v1|subject=sensor|predicate=unknown|night=1|home=any|scope=any|"
        "entities=sensor.keypad_attempt_counter"
    )


def test_battery_named_env_telemetry_keeps_base_keying() -> None:
    """
    sensor.battery_temperature keeps its pre-#541 context-carrying keying.

    Battery-NAMED ids are excluded from the environmental leg's sensor set,
    so the context collapse must not fire — the normalizer leaves this
    candidate unsupported and its keying stays at base parity.
    """
    candidate = _env_statistical_candidate(
        candidate_id="candidate_home_battery_temp_deviation_night",
        title="Deviation in reading at night",
        summary=("Detects deviation from the normal reading during nighttime hours."),
        evidence_paths=[
            "entities[sensor.battery_temperature].state",
            "derived.is_night",
        ],
    )
    key = candidate_semantic_key(candidate)
    assert key is not None
    assert "predicate=power_anomaly" in key
    assert "night=1" in key


def test_environmental_behavior_parity_normalizer_to_rule_key() -> None:
    """
    Behavior-level parity (issue #541): key covers the rule promotion registers.

    Runs the real normalizer on statistical env candidates and asserts each
    candidate's own key is covered by the rule_semantic_key of the exact rule
    its promotion would register — the hard #540/#522 parity requirement,
    without hand-built rule dicts that could drift from the normalizer.
    """
    candidates = [
        _env_statistical_candidate(),
        _env_statistical_candidate(
            candidate_id="candidate_fridge_temperature_baseline",
            title="Fridge temperature anomaly",
            summary="Detects deviation from the normal fridge temperature.",
            evidence_paths=["entities[sensor.fridge_temperature].state"],
        ),
        _env_statistical_candidate(
            candidate_id="candidate_attic_temperature_baseline_night_home",
            title="Attic Temperature Anomaly During Night While Home",
            summary=(
                "Detects statistical deviation from the normal attic "
                "temperature reading during nighttime hours while someone "
                "is home."
            ),
            evidence_paths=[
                "entities[sensor.attic_temperature].state",
                "derived.is_night",
                "derived.anyone_home",
            ],
        ),
    ]
    for candidate in candidates:
        normalized = normalize_candidate(candidate)
        assert normalized is not None, candidate["candidate_id"]
        assert normalized.template_id in {"baseline_deviation", "time_of_day_anomaly"}
        rule = {
            "rule_id": normalized.rule_id,
            "template_id": normalized.template_id,
            "params": normalized.params,
        }
        rule_key = rule_semantic_key(rule)
        candidate_key = candidate_semantic_key(candidate)
        assert rule_key is not None, candidate["candidate_id"]
        assert candidate_key is not None, candidate["candidate_id"]
        assert rule_key_covers_candidate_key(rule_key, candidate_key), (
            candidate["candidate_id"],
            rule_key,
            candidate_key,
        )


def test_environmental_leg_does_not_steal_motion_candidate() -> None:
    """
    A motion candidate citing a temperature sensor keeps motion keying.

    The #540 battery-arm lesson applied to the env leg: the normalizer's
    away-motion branch runs BEFORE its statistical branch and registers a
    motion rule for this candidate, so an environmental predicate steal
    (power_anomaly + context collapse) would leave the activated rule unable
    to dedup its own candidate. Behavior-level: run the real normalizer and
    assert the registered motion rule's key covers the candidate key.
    """
    candidate = _env_statistical_candidate(
        candidate_id="candidate_attic_motion_while_away",
        title="Motion in attic while away",
        summary=(
            "Motion detected in the attic near the temperature sensor while "
            "nobody is home."
        ),
        pattern="motion_while_away",
        evidence_paths=[
            "entities[binary_sensor.attic_motion].state",
            "entities[sensor.attic_temperature].state",
            "not derived.anyone_home",
        ],
    )
    normalized = normalize_candidate(candidate)
    assert normalized is not None
    assert normalized.template_id == "motion_detected_while_away"
    rule = {
        "rule_id": normalized.rule_id,
        "template_id": normalized.template_id,
        "params": normalized.params,
    }
    rule_key = rule_semantic_key(rule)
    candidate_key = candidate_semantic_key(candidate)
    assert rule_key is not None
    assert candidate_key is not None
    assert "subject=motion" in candidate_key
    assert "predicate=active" in candidate_key
    assert "home=0" in candidate_key
    assert rule_key_covers_candidate_key(rule_key, candidate_key)


def test_environmental_mirror_constants_stay_symmetric() -> None:
    """
    The env vocabulary and threshold detector must match across modules.

    discovery_semantic mirrors proposal_templates by convention instead of
    importing it; an asymmetric mirror breaks dedup in both directions
    (issue #522 adversarial review). Pin the tuples and the regex pattern so
    a token added on one side without the other fails loudly.
    """
    from custom_components.home_generative_agent.sentinel import (  # noqa: PLC0415
        discovery_semantic,
        proposal_templates,
    )

    assert (
        discovery_semantic._ENVIRONMENTAL_SIGNAL_TERMS
        == proposal_templates._ENVIRONMENTAL_SIGNAL_TERMS
    )
    assert (
        discovery_semantic._NUMERIC_THRESHOLD_RE.pattern
        == proposal_templates._NUMERIC_THRESHOLD_PATTERN.pattern
    )


@pytest.mark.parametrize(
    "entity_id",
    [
        "sensor.outdoor_temperature",
        "sensor.window_temperature",
        "sensor.entryway_humidity",
    ],
)
def test_environmental_door_substring_entity_keeps_sensor_keying(
    entity_id: str,
) -> None:
    """
    outdoor/window-named env sensors must dedup against their baseline rule.

    "door"/"window"/"entry" substrings in common environmental entity names
    (sensor.outdoor_temperature!) must not key subject=entry_* — the
    normalizer's _find_entry_entity_ids domain-gates entries to
    binary_sensor/cover, so the registered rule is a subject=sensor baseline
    rule and an entry-subject key would never dedup its own re-proposals
    (ship testing-specialist review, reproduced pre-fix).
    """
    candidate = _env_statistical_candidate(
        candidate_id=f"candidate_{entity_id.split('.')[1]}_baseline",
        title="Reading anomaly",
        summary="Detects statistical deviation from the normal reading.",
        evidence_paths=[f"entities[{entity_id}].state"],
    )
    normalized = normalize_candidate(candidate)
    assert normalized is not None
    assert normalized.template_id == "baseline_deviation"
    rule_key = rule_semantic_key(
        {
            "rule_id": normalized.rule_id,
            "template_id": normalized.template_id,
            "params": normalized.params,
        }
    )
    candidate_key = candidate_semantic_key(candidate)
    assert rule_key is not None
    assert candidate_key is not None
    assert candidate_key == (
        "v1|subject=sensor|predicate=power_anomaly|night=any|home=any|scope=any|"
        f"entities={entity_id}"
    )
    assert rule_key_covers_candidate_key(rule_key, candidate_key)


@pytest.mark.parametrize(
    ("term", "entity_id"),
    [
        ("humidity", "sensor.bathroom_humidity"),
        ("pressure", "sensor.basement_pressure"),
        ("co2", "sensor.living_room_co2"),
        ("carbon dioxide", "sensor.den_carbon_dioxide"),
        ("pm2.5", "sensor.bedroom_pm2_5"),
        ("pm10", "sensor.bedroom_pm10"),
        ("aqi", "sensor.living_room_aqi"),
        ("air quality", "sensor.living_room_air_quality"),
        ("moisture", "sensor.garden_moisture"),
        ("illuminance", "sensor.hallway_illuminance"),
        ("lux", "sensor.hallway_lux"),
        ("carbon monoxide", "sensor.garage_carbon_monoxide"),
    ],
)
def test_environmental_term_routes_and_dedupes(term: str, entity_id: str) -> None:
    """
    Every vocabulary term routes and dedups through the public entry points.

    The mirror test pins cross-module tuple equality, but a tokenization
    regression (multi-token spellings like "pm2.5"/"air quality" depend on
    the non-alphanumeric split producing adjacent tokens) would pass it and
    every single-term behavior test (ship testing-specialist review).
    """
    candidate = _env_statistical_candidate(
        candidate_id="candidate_env_term",
        title=f"{term} anomaly",
        summary=f"Detects statistical deviation from the normal {term} reading.",
        evidence_paths=[f"entities[{entity_id}].state"],
    )
    normalized = normalize_candidate(candidate)
    assert normalized is not None, term
    assert normalized.template_id in {"baseline_deviation", "time_of_day_anomaly"}
    candidate_key = candidate_semantic_key(candidate)
    rule_key = rule_semantic_key(
        {
            "rule_id": normalized.rule_id,
            "template_id": normalized.template_id,
            "params": normalized.params,
        }
    )
    assert candidate_key is not None, term
    assert rule_key is not None, term
    assert "night=any|home=any" in candidate_key, term
    assert rule_key_covers_candidate_key(rule_key, candidate_key), term


def test_environmental_staleness_candidate_keys_staleness() -> None:
    """
    Staleness-worded env candidates keep predicate=staleness (#541 red team).

    Mirrors the normalizer's staleness gate on its env arm: the candidate
    registers entity_staleness, so the env leg must not claim the key.
    """
    candidate = _env_statistical_candidate(
        candidate_id="candidate_attic_temperature_stale",
        title="Attic temperature sensor stale",
        summary=(
            "sensor.attic_temperature has not updated in over 12 hours; the "
            "temperature reading may be stale."
        ),
        pattern="entity_staleness",
        evidence_paths=["entities[entity_id=sensor.attic_temperature].state"],
    )
    normalized = normalize_candidate(candidate)
    assert normalized is not None
    assert normalized.template_id == "entity_staleness"
    key = candidate_semantic_key(candidate)
    assert key is not None
    assert "predicate=staleness" in key
    assert "predicate=power_anomaly" not in key


def test_environmental_incidental_open_prose_keys_power_anomaly() -> None:
    """
    "May mean a window was left open" must not steal the predicate (#541).

    A sensor-only env candidate registers baseline_deviation regardless of
    incidental entry prose (the normalizer's entry branches need entity
    evidence), so the key must reach predicate=power_anomaly context-free
    and be covered by the registered rule (red-team review, reproduced).
    """
    candidate = _env_statistical_candidate(
        candidate_id="candidate_attic_temperature_drop",
        title="Attic temperature drop anomaly",
        summary=(
            "sensor.attic_temperature drops sharply from its normal range, "
            "which may mean a window was left open."
        ),
        evidence_paths=[
            "entities[sensor.attic_temperature].state",
            "derived.is_night",
        ],
    )
    normalized = normalize_candidate(candidate)
    assert normalized is not None
    assert normalized.template_id == "baseline_deviation"
    rule_key = rule_semantic_key(
        {
            "rule_id": normalized.rule_id,
            "template_id": normalized.template_id,
            "params": normalized.params,
        }
    )
    candidate_key = candidate_semantic_key(candidate)
    assert rule_key is not None
    assert candidate_key == (
        "v1|subject=sensor|predicate=power_anomaly|night=any|home=any|scope=any|"
        "entities=sensor.attic_temperature"
    )
    assert rule_key_covers_candidate_key(rule_key, candidate_key)


def test_entry_evidence_open_prose_keeps_open_predicate() -> None:
    """
    Real entry evidence keeps predicate=open despite env prose (#541).

    The override is subject-gated: cover/binary_sensor entry evidence makes
    subject=entry_door, the env verdict stays off, and the open leg keeps
    the candidate — mutation guard for _override_env_prose_steal.
    """
    candidate = _env_statistical_candidate(
        candidate_id="candidate_garage_co2_door",
        title="CO2 rises when the garage door opens",
        summary=("Garage CO2 rises when the garage door is left open for long."),
        pattern="entry_watch",
        evidence_paths=[
            "entities[cover.garage_door].state",
            "entities[sensor.garage_co2].state",
        ],
    )
    key = candidate_semantic_key(candidate)
    assert key is not None
    assert "subject=entry_door" in key
    assert "predicate=open" in key


def test_environmental_availability_prose_not_rewritten_to_power_anomaly() -> None:
    """
    Outage candidates must not be covered by unrelated baseline rules.

    "unavailable; unable to open its connection" keys predicate=open (the
    open leg precedes the unavailable leg); rewriting it to power_anomaly
    would let an active baseline rule on the same sensor falsely cover the
    outage proposal the normalizer's availability branch registers (Codex
    adversarial review, reproduced). The override must leave the base key.
    """
    candidate = _env_statistical_candidate(
        candidate_id="candidate_attic_temperature_offline",
        title="Attic temperature sensor unavailable",
        summary=(
            "The attic temperature sensor is unavailable; Home Assistant is "
            "unable to open its connection."
        ),
        pattern="availability_watch",
        evidence_paths=["entities[sensor.attic_temperature].state"],
    )
    normalized = normalize_candidate(candidate)
    assert normalized is not None
    assert normalized.template_id in {
        "unavailable_sensors",
        "unavailable_sensors_while_home",
    }
    key = candidate_semantic_key(candidate)
    assert key is not None
    assert "predicate=power_anomaly" not in key
    baseline_rule_key = (
        "v1|subject=sensor|predicate=power_anomaly"
        "|template=baseline_deviation|entities=sensor.attic_temperature"
    )
    assert not rule_key_covers_candidate_key(baseline_rule_key, key)


def test_comma_threshold_prose_keeps_context_in_key() -> None:
    """
    "exceeds 1,000" is a threshold on both surfaces (Codex review).

    The keying mirror must parse comma thousands like the normalizer, or a
    threshold candidate would collapse its context while the registered
    rule keeps require_night — the mirror-drift class the tuple pin guards.
    """
    candidate = _env_statistical_candidate(
        candidate_id="candidate_living_room_co2_threshold",
        title="CO2 threshold at night",
        summary="Alert when living room CO2 exceeds 1,000 ppm at night.",
        pattern="sensor_threshold",
        evidence_paths=[
            "entities[sensor.living_room_co2].state",
            "derived.is_night",
        ],
    )
    normalized = normalize_candidate(candidate)
    assert normalized is not None
    assert normalized.template_id == "sensor_threshold_condition"
    assert normalized.params["threshold"] == 1000.0
    key = candidate_semantic_key(candidate)
    assert key is not None
    assert "night=1" in key

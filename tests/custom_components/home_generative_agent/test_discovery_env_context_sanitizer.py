# ruff: noqa: S101
"""
Tests for the environmental occupancy-context sanitizer.

Field data (2026-08-15, live box on v3.30.2): the discovery LLM decorates
environmental statistical candidates with occupancy/night context on every
cycle despite the prompt's ENVIRONMENTAL SENSOR RULE, and the stored draft
cards then promise conditioning ("While Away During Day") that the activated
context-free baseline_deviation rule never has. The sanitizer is the
deterministic enforcement layer at ingestion.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, cast

from custom_components.home_generative_agent.sentinel.discovery_engine import (
    SentinelDiscoveryEngine,
)
from custom_components.home_generative_agent.sentinel.discovery_schema import (
    DISCOVERY_OUTPUT_SCHEMA,
)
from custom_components.home_generative_agent.sentinel.discovery_semantic import (
    candidate_semantic_key,
    sanitize_environmental_candidate,
)
from custom_components.home_generative_agent.sentinel.proposal_templates import (
    explain_normalize_candidate,
)

# Verbatim field candidates from the live proposal store (2026-08-12/15).
GARAGE_CANDIDATE: dict[str, Any] = {
    "candidate_id": "garage_temp_baseline_deviation_away_day",
    "title": "Garage Temperature Statistical Deviation While Away During Day",
    "summary": (
        "Detects if the garage temperature sensor deviates significantly "
        "from its normal baseline while no one is home during daytime hours."
    ),
    "evidence_paths": [
        "derived.anyone_home=false",
        "derived.is_night=false",
        (
            "entities[entity_ids contains "
            "sensor.garage_temp_and_humidity_temperature].state=83"
        ),
    ],
    "pattern": "baseline_deviation",
    "confidence_hint": 0.85,
    "suggested_type": "statistical_anomaly",
}

IPHONE_CANDIDATE: dict[str, Any] = {
    "candidate_id": "iphone_pressure_baseline_deviation_home_day",
    "title": "iPhone Pressure Baseline Deviation During Daytime Home",
    "summary": (
        "Detects if the iPhone barometric pressure reading deviates "
        "significantly from its established baseline while it is daytime "
        "and someone is home, indicating potential sensor drift or "
        "environmental anomalies."
    ),
    "evidence_paths": [
        "derived.anyone_home",
        "not derived.is_night",
        "entities[entity_ids contains sensor.lindos_iphone_pressure].state",
    ],
    "pattern": "baseline_deviation",
    "confidence_hint": 0.85,
    "suggested_type": "statistical_anomaly",
}


class _DummyStore:
    async def async_get_latest(self, _limit: int) -> list[dict[str, Any]]:
        return []


def test_garage_field_candidate_fully_sanitized() -> None:
    sanitized = sanitize_environmental_candidate(GARAGE_CANDIDATE)
    assert sanitized is not None
    assert sanitized["candidate_id"] == "garage_temp_baseline_deviation"
    assert sanitized["title"] == "Garage Temperature Statistical Deviation"
    assert sanitized["summary"] == (
        "Detects if the garage temperature sensor deviates significantly "
        "from its normal baseline."
    )
    assert sanitized["evidence_paths"] == [
        (
            "entities[entity_ids contains "
            "sensor.garage_temp_and_humidity_temperature].state=83"
        ),
    ]
    assert sanitized["environmental_context_stripped"] is True
    # The original is never mutated.
    assert GARAGE_CANDIDATE["candidate_id"] == (
        "garage_temp_baseline_deviation_away_day"
    )


def test_iphone_field_candidate_fully_sanitized() -> None:
    sanitized = sanitize_environmental_candidate(IPHONE_CANDIDATE)
    assert sanitized is not None
    assert sanitized["candidate_id"] == "iphone_pressure_baseline_deviation"
    assert sanitized["title"] == "iPhone Pressure Baseline Deviation"
    # The non-context tail after the clause's comma is preserved.
    assert sanitized["summary"] == (
        "Detects if the iPhone barometric pressure reading deviates "
        "significantly from its established baseline, indicating potential "
        "sensor drift or environmental anomalies."
    )
    assert sanitized["evidence_paths"] == [
        "entities[entity_ids contains sensor.lindos_iphone_pressure].state",
    ]


def test_semantic_key_unchanged_by_sanitizing() -> None:
    for candidate in (GARAGE_CANDIDATE, IPHONE_CANDIDATE):
        sanitized = sanitize_environmental_candidate(candidate)
        assert sanitized is not None
        assert candidate_semantic_key(sanitized) == candidate_semantic_key(candidate)


def test_sanitized_candidate_normalizes_context_free() -> None:
    sanitized = sanitize_environmental_candidate(GARAGE_CANDIDATE)
    assert sanitized is not None
    normalized = explain_normalize_candidate(sanitized).normalized
    assert normalized is not None
    assert normalized.template_id == "baseline_deviation"
    assert normalized.params == {
        "entity_id": "sensor.garage_temp_and_humidity_temperature"
    }
    # The activated rule ID no longer advertises away/day conditioning.
    assert normalized.rule_id == "garage_temp_baseline_deviation"


def test_sanitizer_is_idempotent() -> None:
    sanitized = sanitize_environmental_candidate(GARAGE_CANDIDATE)
    assert sanitized is not None
    assert sanitize_environmental_candidate(sanitized) is None


def test_security_motion_candidate_untouched() -> None:
    # Subject keys motion, not sensor — the sanitizer must never touch
    # security-leg prose (broad-vocab lesson, issue #541 reviews).
    candidate = {
        "candidate_id": "motion_while_away_night",
        "title": "Motion detected at night while away",
        "summary": "Motion sensor fires while no one is home at night.",
        "evidence_paths": [
            "entities[entity_id=binary_sensor.hallway_motion].state",
            "not derived.anyone_home",
            "derived.is_night",
        ],
        "pattern": "motion_at_night_away",
    }
    assert sanitize_environmental_candidate(candidate) is None


def test_threshold_env_candidate_keeps_context() -> None:
    # Numeric-threshold prose routes to sensor_threshold_condition, whose
    # params legitimately carry require_away/require_night — context is
    # real there and must survive.
    candidate = {
        "candidate_id": "garage_temp_over_90_away",
        "title": "Garage temperature above 90 while away",
        "summary": "Alerts when the garage temperature exceeds 90 while away.",
        "evidence_paths": [
            "not derived.anyone_home",
            "entities[entity_id=sensor.garage_temperature].state",
        ],
        "pattern": "threshold",
    }
    assert sanitize_environmental_candidate(candidate) is None
    normalized = explain_normalize_candidate(candidate).normalized
    assert normalized is not None
    assert normalized.template_id == "sensor_threshold_condition"
    assert normalized.params["require_away"] is True


def test_availability_env_candidate_untouched() -> None:
    # Availability wording keys predicate=unavailable, not power_anomaly —
    # the outage routing must keep its context (issue #541 Codex review).
    candidate = {
        "candidate_id": "attic_temp_unavailable_night",
        "title": "Attic temperature sensor unavailable at night",
        "summary": "The attic temperature sensor becomes unavailable at night.",
        "evidence_paths": [
            "entities[entity_id=sensor.attic_temperature].state",
            "derived.is_night",
        ],
        "pattern": "availability",
    }
    assert sanitize_environmental_candidate(candidate) is None


def test_non_environmental_power_candidate_untouched() -> None:
    # power_anomaly with a context-free key but no environmental signal:
    # the env verdict gate keeps the sanitizer off plain power candidates.
    candidate = {
        "candidate_id": "fridge_power_baseline_deviation",
        "title": "Fridge power baseline deviation",
        "summary": "Fridge power use deviates from its baseline.",
        "evidence_paths": [
            "entities[entity_id=sensor.fridge_power].state",
        ],
        "pattern": "baseline_deviation",
    }
    assert sanitize_environmental_candidate(candidate) is None


def test_named_battery_sensor_untouched() -> None:
    # Battery-NAMED environmental-worded ids keep the battery arm's
    # treatment (#540 lesson) — the env verdict excludes them.
    candidate = {
        "candidate_id": "door_battery_baseline_night",
        "title": "Door sensor battery temperature deviation at night",
        "summary": "Battery temperature deviates from baseline at night.",
        "evidence_paths": [
            "entities[entity_id=sensor.front_door_sensor_battery].state",
            "derived.is_night",
        ],
        "pattern": "baseline_deviation",
    }
    assert sanitize_environmental_candidate(candidate) is None


def test_time_of_day_slug_protected_when_popping_id_tokens() -> None:
    candidate = {
        "candidate_id": "attic_temp_time_of_day_anomaly_night",
        "title": "Attic temperature time-of-day anomaly at night",
        "summary": ("Attic temperature deviates from its per-hour baseline at night."),
        "evidence_paths": [
            "entities[entity_id=sensor.attic_temperature].state",
            "derived.is_night",
        ],
        "pattern": "time_of_day_anomaly",
    }
    sanitized = sanitize_environmental_candidate(candidate)
    assert sanitized is not None
    # "_night" pops; the trailing "day" inside "time_of_day" survives via
    # the trigram guard... the slug ends in "anomaly" here, but assert the
    # guard directly on a hostile shape too.
    assert sanitized["candidate_id"] == "attic_temp_time_of_day_anomaly"
    assert "derived.is_night" not in sanitized["evidence_paths"]

    hostile = dict(candidate)
    hostile["candidate_id"] = "attic_temp_anomaly_time_of_day"
    sanitized_hostile = sanitize_environmental_candidate(hostile)
    assert sanitized_hostile is not None
    assert sanitized_hostile["candidate_id"] == "attic_temp_anomaly_time_of_day"


def test_non_context_clause_survives() -> None:
    candidate = {
        "candidate_id": "garage_humidity_baseline_deviation_home",
        "title": "Garage humidity baseline deviation",
        "summary": (
            "Garage humidity deviates from baseline while the compressor "
            "runs, which may indicate a ventilation fault."
        ),
        "evidence_paths": [
            "entities[entity_id=sensor.garage_humidity].state",
            "derived.anyone_home",
        ],
        "pattern": "baseline_deviation",
    }
    sanitized = sanitize_environmental_candidate(candidate)
    assert sanitized is not None
    # The compressor clause has no occupancy vocabulary — it stays.
    assert sanitized["summary"] == candidate["summary"]
    assert sanitized["candidate_id"] == "garage_humidity_baseline_deviation"
    assert sanitized["evidence_paths"] == [
        "entities[entity_id=sensor.garage_humidity].state",
    ]


def test_all_context_title_falls_back_to_original() -> None:
    candidate = {
        "candidate_id": "pressure_baseline_deviation_home",
        "title": "While Away",
        "summary": "Pressure baseline deviation detection.",
        "evidence_paths": [
            "entities[entity_id=sensor.hall_pressure].state",
            "derived.anyone_home",
        ],
        "pattern": "baseline_deviation",
    }
    sanitized = sanitize_environmental_candidate(candidate)
    assert sanitized is not None
    # Stripping would leave nothing readable — the original title stands,
    # but the structural context (paths, id token) still comes off.
    assert sanitized["title"] == "While Away"
    assert sanitized["evidence_paths"] == [
        "entities[entity_id=sensor.hall_pressure].state",
    ]
    assert sanitized["candidate_id"] == "pressure_baseline_deviation"


def test_no_change_returns_none() -> None:
    candidate = {
        "candidate_id": "attic_temp_baseline_deviation",
        "title": "Attic temperature baseline deviation",
        "summary": "Attic temperature deviates from its rolling baseline.",
        "evidence_paths": [
            "entities[entity_id=sensor.attic_temperature].state",
        ],
        "pattern": "baseline_deviation",
    }
    assert sanitize_environmental_candidate(candidate) is None


def test_non_string_evidence_path_junk_preserved() -> None:
    candidate = {
        "candidate_id": "attic_temp_baseline_deviation_night",
        "title": "Attic temperature baseline deviation at night",
        "summary": "Attic temperature deviates from baseline at night.",
        "evidence_paths": [
            {"junk": True},
            "derived.is_night",
            "entities[entity_id=sensor.attic_temperature].state",
        ],
        "pattern": "baseline_deviation",
    }
    sanitized = sanitize_environmental_candidate(candidate)
    assert sanitized is not None
    assert sanitized["evidence_paths"] == [
        {"junk": True},
        "entities[entity_id=sensor.attic_temperature].state",
    ]


def test_null_key_candidate_returns_none() -> None:
    # candidate_semantic_key returns None for unknown subject/predicate —
    # the `not key` short-circuit must bail before the substring check.
    candidate = {
        "candidate_id": "mystery_thing",
        "title": "Mystery reading",
        "summary": "Something odd is happening.",
        "evidence_paths": ["entities[entity_id=light.hall].state"],
        "pattern": "unknown",
    }
    assert candidate_semantic_key(candidate) is None
    assert sanitize_environmental_candidate(candidate) is None


def test_multiple_context_clauses_stripped_with_cleanup() -> None:
    # Two separate context clauses force the clause loop through a second
    # removal, and the dangling-separator cleanup collapses the ",."
    # left where the final clause was cut out.
    candidate = {
        "candidate_id": "attic_temp_baseline_deviation_away_night",
        "title": "Attic Temperature Baseline Deviation",
        "summary": ("Attic temperature deviates while away, when nobody is present."),
        "evidence_paths": [
            "entities[entity_id=sensor.attic_temperature].state",
            "derived.anyone_home=false",
        ],
        "pattern": "baseline_deviation",
    }
    sanitized = sanitize_environmental_candidate(candidate)
    assert sanitized is not None
    assert sanitized["summary"] == "Attic temperature deviates."
    assert sanitized["candidate_id"] == "attic_temp_baseline_deviation"


def test_missing_pattern_field_still_strips_structural_context() -> None:
    # No "pattern" key: the prose loop's isinstance gate skips the absent
    # field while paths, candidate_id, and "at night" decoration (the "at"
    # trigger, red-team review) still come off.
    candidate = {
        "candidate_id": "attic_temp_baseline_deviation_night",
        "title": "Attic temperature baseline deviation at night",
        "summary": "Attic temperature deviates from its rolling baseline at night.",
        "evidence_paths": [
            "derived.is_night",
            "entities[entity_id=sensor.attic_temperature].state",
        ],
    }
    sanitized = sanitize_environmental_candidate(candidate)
    assert sanitized is not None
    assert "pattern" not in sanitized
    assert sanitized["candidate_id"] == "attic_temp_baseline_deviation"
    assert sanitized["title"] == "Attic temperature baseline deviation"
    assert sanitized["summary"] == (
        "Attic temperature deviates from its rolling baseline."
    )
    assert sanitized["evidence_paths"] == [
        "entities[entity_id=sensor.attic_temperature].state",
    ]


def test_all_context_slug_keeps_final_token() -> None:
    # A slug that is nothing but context tokens must not strip to the empty
    # string — the len(tokens) > 1 guard keeps the last token standing.
    candidate = {
        "candidate_id": "away_day",
        "title": "Garage temperature baseline deviation",
        "summary": "Garage temperature deviates from its rolling baseline.",
        "evidence_paths": [
            "entities[entity_id=sensor.garage_temperature].state",
            "derived.anyone_home=false",
        ],
        "pattern": "baseline_deviation",
    }
    sanitized = sanitize_environmental_candidate(candidate)
    assert sanitized is not None
    assert sanitized["candidate_id"] == "away"


def test_engine_filter_collapses_decorated_variants_within_batch() -> None:
    """
    Two decorated variants of one sensor collapse inside a single batch.

    The anti-pile-up promise: away_day and home_night decorations sanitize
    to the same context-free key, so the second variant drops as a batch
    duplicate instead of storing as a distinct novel candidate.
    """
    engine = SentinelDiscoveryEngine(
        hass=cast("Any", object()),
        options={},
        model=None,
        store=cast("Any", _DummyStore()),
    )
    night_variant: dict[str, Any] = {
        "candidate_id": "garage_temp_baseline_deviation_home_night",
        "title": (
            "Garage Temperature Statistical Deviation While Everyone Is Home At Night"
        ),
        "summary": (
            "Detects if the garage temperature sensor deviates from its "
            "baseline while everyone is home at night."
        ),
        "evidence_paths": [
            "derived.anyone_home=true",
            "derived.is_night=true",
            (
                "entities[entity_ids contains "
                "sensor.garage_temp_and_humidity_temperature].state=83"
            ),
        ],
        "pattern": "baseline_deviation",
        "confidence_hint": 0.85,
        "suggested_type": "statistical_anomaly",
    }
    filtered, dropped = engine._filter_novel_candidates(
        [dict(GARAGE_CANDIDATE), night_variant], set()
    )
    assert len(filtered) == 1
    assert filtered[0]["candidate_id"] == "garage_temp_baseline_deviation"
    assert len(dropped) == 1
    assert dropped[0]["dedupe_reason"] == "batch_duplicate"
    assert dropped[0]["semantic_key"] == filtered[0]["semantic_key"]


def test_engine_filter_stores_sanitized_candidate() -> None:
    """_filter_novel_candidates must key and store the sanitized shape."""
    engine = SentinelDiscoveryEngine(
        hass=cast("Any", object()),
        options={},
        model=None,
        store=cast("Any", _DummyStore()),
    )
    filtered, dropped = engine._filter_novel_candidates([dict(GARAGE_CANDIDATE)], set())
    assert dropped == []
    assert len(filtered) == 1
    stored = filtered[0]
    assert stored["candidate_id"] == "garage_temp_baseline_deviation"
    assert stored["title"] == "Garage Temperature Statistical Deviation"
    assert stored["environmental_context_stripped"] is True
    assert stored["semantic_key"] == (
        "v1|subject=sensor|predicate=power_anomaly|night=any|home=any|"
        "scope=any|entities=sensor.garage_temp_and_humidity_temperature"
    )
    # A later cycle proposing the same decorated shape dedups against the
    # sanitized key (context-free, so decorated variants cannot pile up).
    refiltered, redropped = engine._filter_novel_candidates(
        [dict(GARAGE_CANDIDATE)], {stored["semantic_key"]}
    )
    assert refiltered == []
    assert redropped[0]["dedupe_reason"] == "existing_semantic_key"


# --- Adversarial-review regression tests (ship review army, 2026-08-15) ---


def test_when_core_condition_with_nested_context_clause_survives() -> None:
    # Testing-specialist CRITICAL: first-trigger-wins removal gutted
    # "alert when <core> while <context>" to "alert." — only the
    # pure-decoration inner clause may come off.
    candidate = {
        "candidate_id": "garage_temp_baseline_deviation_away",
        "title": "Garage Temperature Baseline Deviation",
        "summary": (
            "This candidate raises an alert when the garage temperature "
            "deviates from its baseline while nobody is home."
        ),
        "evidence_paths": [
            "not derived.anyone_home",
            "entities[entity_id=sensor.garage_temperature].state",
        ],
        "pattern": "baseline_deviation",
    }
    sanitized = sanitize_environmental_candidate(candidate)
    assert sanitized is not None
    assert "deviates from its baseline" in sanitized["summary"]
    assert "nobody is home" not in sanitized["summary"]


def test_day_only_threshold_candidate_untouched() -> None:
    # Testing-specialist CRITICAL: day-only context keys night=any|home=any
    # and used to slip past the gate; the _has_numeric_threshold exclusion
    # keeps every threshold candidate untouched, matching docs/sentinel.md.
    candidate = {
        "candidate_id": "garage_temp_over_90_day",
        "title": "Garage temperature above 90 during daytime",
        "summary": "Alerts when the garage temperature exceeds 90 during daytime.",
        "evidence_paths": [
            "not derived.is_night",
            "entities[entity_id=sensor.garage_temperature].state",
        ],
        "pattern": "threshold",
    }
    assert sanitize_environmental_candidate(candidate) is None
    normalized = explain_normalize_candidate(candidate).normalized
    assert normalized is not None
    assert normalized.template_id == "sensor_threshold_condition"


def test_ppm_threshold_phrase_never_swallowed() -> None:
    # Claude+Codex adversarial P1: clause removal used to swallow
    # "exceeds 1000 ppm during the day", silently rewriting an approvable
    # threshold rule into a baseline rule with the SAME semantic key.
    candidate = {
        "candidate_id": "co2_high_alert_day",
        "title": "High CO2 alert",
        "summary": (
            "High CO2 should alert when readings exceed 1000 ppm during the day."
        ),
        "evidence_paths": [
            "entities[entity_id=sensor.living_room_co2].state",
        ],
        "pattern": "threshold",
    }
    assert sanitize_environmental_candidate(candidate) is None
    normalized = explain_normalize_candidate(candidate).normalized
    assert normalized is not None
    assert normalized.template_id == "sensor_threshold_condition"
    assert normalized.params["threshold"] == 1000.0


def test_key_drift_guard_reverts_signal_bearing_clause_strip() -> None:
    # Claude+Codex adversarial P1: when the ONLY environmental term lives
    # inside a removable clause, stripping would flip the key to
    # predicate=unknown and make the candidate unpromotable. The
    # key-invariance guard must revert the whole sanitize.
    candidate = {
        "candidate_id": "generic_env_anomaly_home",
        "title": "Environmental anomaly",
        "summary": ("Environmental anomaly when humidity rises while nobody is home."),
        "evidence_paths": [
            "derived.anyone_home",
            "entities[entity_id=sensor.generic].state",
        ],
        "pattern": "baseline_deviation",
    }
    before_key = candidate_semantic_key(candidate)
    result = sanitize_environmental_candidate(candidate)
    if result is not None:
        # If a future vocabulary change makes this strippable without
        # drift, the invariant must still hold.
        assert candidate_semantic_key(result) == before_key
    else:
        # Guard fired: candidate flows through completely untouched.
        assert candidate_semantic_key(candidate) == before_key


def test_home_noun_and_seven_day_average_prose_survive() -> None:
    # Testing-specialist + Claude adversarial: bare-word vocabulary used to
    # false-positive on "home temperature" (the building), "Home Assistant",
    # and "7-day average". Pure-decoration residue keeps them all.
    candidate = {
        "candidate_id": "home_temp_baseline_deviation",
        "title": "Home temperature baseline deviation",
        "summary": (
            "Notify me when the home temperature deviates from the 7-day "
            "rolling average, or when Home Assistant restarts unexpectedly."
        ),
        "evidence_paths": [
            "entities[entity_id=sensor.home_temperature].state",
        ],
        "pattern": "baseline_deviation",
    }
    sanitized = sanitize_environmental_candidate(candidate)
    if sanitized is not None:
        assert sanitized["summary"] == candidate["summary"]
        assert sanitized["title"] == candidate["title"]
    else:
        # Nothing to strip at all is equally correct.
        assert sanitize_environmental_candidate(candidate) is None


def test_canonical_path_spellings_all_stripped() -> None:
    # Maintainability + Claude/Codex adversarial P2: quoted, "!"-negated,
    # "is false", and bare-alias spellings are recognized by
    # canonicalize_evidence_path and must all come off.
    candidate = {
        "candidate_id": "attic_temp_baseline_deviation_away_night",
        "title": "Attic temperature baseline deviation while away at night",
        "summary": "Attic temperature deviates from baseline while away at night.",
        "evidence_paths": [
            "!derived.anyone_home",
            "'not derived.anyone_home'",
            "derived.is_night is false",
            "anyone_home == false",
            "derived.people_home",
            "entities[entity_id=sensor.attic_temperature].state",
        ],
        "pattern": "baseline_deviation",
    }
    sanitized = sanitize_environmental_candidate(candidate)
    assert sanitized is not None
    assert sanitized["evidence_paths"] == [
        "entities[entity_id=sensor.attic_temperature].state",
    ]
    assert sanitized["candidate_id"] == "attic_temp_baseline_deviation"
    assert sanitized["title"] == "Attic temperature baseline deviation"


def test_machine_syntax_and_parenthetical_decoration_stripped() -> None:
    # Claude adversarial INVESTIGATE + red-team: machine-syntax prose
    # ("while not derived.anyone_home") and parenthetical decoration
    # ("(Away, Daytime)") also promise conditioning the rule lacks.
    candidate = {
        "candidate_id": "attic_temp_night_anomaly",
        "title": "Attic Temperature Anomaly (Away, Daytime)",
        "summary": "Attic temperature deviates while not derived.anyone_home.",
        "evidence_paths": [
            "not derived.anyone_home",
            "entities[entity_id=sensor.attic_temperature].state",
        ],
        "pattern": "baseline_deviation",
    }
    sanitized = sanitize_environmental_candidate(candidate)
    assert sanitized is not None
    assert sanitized["title"] == "Attic Temperature Anomaly"
    assert sanitized["summary"] == "Attic temperature deviates."


def test_dangling_connective_popped_from_slug() -> None:
    # Claude adversarial P3: "co2_while_away" must not become "co2_while" —
    # the connective left dangling by a context pop comes off too.
    candidate = {
        "candidate_id": "co2_baseline_deviation_while_away",
        "title": "CO2 baseline deviation",
        "summary": "CO2 readings deviate from the rolling baseline.",
        "evidence_paths": [
            "not derived.anyone_home",
            "entities[entity_id=sensor.living_room_co2].state",
        ],
        "pattern": "baseline_deviation",
    }
    sanitized = sanitize_environmental_candidate(candidate)
    assert sanitized is not None
    assert sanitized["candidate_id"] == "co2_baseline_deviation"


def test_oversized_prose_field_skipped_quickly() -> None:
    # Claude+Codex adversarial P1: an 80 KB hostile field used to stall the
    # event loop ~50 s in the removal loop. Fields over the cap skip prose
    # stripping entirely (structural strips still apply).
    hostile_summary = "when home, " * 8000
    candidate = {
        "candidate_id": "attic_temp_baseline_deviation_night",
        "title": "Attic temperature baseline deviation",
        "summary": hostile_summary,
        "evidence_paths": [
            "derived.is_night",
            "entities[entity_id=sensor.attic_temperature].state",
        ],
        "pattern": "baseline_deviation",
    }
    start = time.monotonic()
    sanitized = sanitize_environmental_candidate(candidate)
    elapsed = time.monotonic() - start
    assert elapsed < 1.0
    assert sanitized is not None
    assert sanitized["summary"] == hostile_summary
    assert sanitized["evidence_paths"] == [
        "entities[entity_id=sensor.attic_temperature].state",
    ]


def test_engine_guards_judge_original_candidate() -> None:
    # Red-team: a hallucinated known entity named ONLY inside a strippable
    # clause must still drop the candidate — the mismatch guard runs on the
    # LLM's actual output, before sanitization.
    engine = SentinelDiscoveryEngine(
        hass=cast("Any", object()),
        options={},
        model=None,
        store=cast("Any", _DummyStore()),
    )
    candidate = {
        "candidate_id": "attic_temp_baseline_deviation_away",
        "title": "Attic temperature baseline deviation",
        "summary": (
            "Attic temperature deviates from its baseline while the "
            "bedroom occupancy temp shows everyone is away."
        ),
        "evidence_paths": [
            "not derived.anyone_home",
            "entities[entity_id=sensor.attic_temperature].state",
        ],
        "pattern": "baseline_deviation",
    }
    filtered, dropped = engine._filter_novel_candidates(
        [candidate],
        set(),
        ["sensor.attic_temperature", "sensor.bedroom_occupancy_temp"],
    )
    assert filtered == []
    assert dropped[0]["dedupe_reason"] == "entity_text_mismatch"


def test_promote_time_sanitize_covers_stored_decorated_candidates() -> None:
    # Red-team CRITICAL: candidates stored before the sanitizer shipped
    # remain promotable from discovery history — _promote_discovery_candidate
    # must sanitize too so a pre-fix candidate can never mint a decorated
    # draft. Behavior-level check on the sanitizer itself against a verbatim
    # pre-fix stored shape (the promote flow calls the same function).
    stored_prefix_candidate = dict(GARAGE_CANDIDATE)
    sanitized = sanitize_environmental_candidate(stored_prefix_candidate)
    assert sanitized is not None
    assert sanitized["candidate_id"] == "garage_temp_baseline_deviation"
    promote_src = Path("custom_components/home_generative_agent/__init__.py")
    source = promote_src.read_text(encoding="utf-8")
    for boundary in (
        "async def _promote_discovery_candidate",
        "async def _approve_rule_proposal",
        "async def _preview_rule_proposal",
    ):
        body = source.split(boundary, 1)[1].split("async def ", 1)[0]
        assert "sanitize_environmental_candidate" in body, boundary


def test_sanitized_marker_declared_in_discovery_schema() -> None:
    # Red-team: DISCOVERY_OUTPUT_SCHEMA is PREVENT_EXTRA; a stored payload
    # containing the marker must round-trip validation.
    payload = {
        "schema_version": 1,
        "generated_at": "2026-08-15T12:00:00+00:00",
        "model": "anomaly_suggestion_engine_v1",
        "candidates": [
            {
                "candidate_id": "garage_temp_baseline_deviation",
                "title": "Garage Temperature Statistical Deviation",
                "summary": "Detects garage temperature baseline deviation.",
                "evidence_paths": [
                    "entities[entity_id=sensor.garage_temperature].state",
                ],
                "pattern": "baseline_deviation",
                "confidence_hint": 0.85,
                "semantic_key": "v1|subject=sensor|predicate=power_anomaly|"
                "night=any|home=any|scope=any|entities=sensor.garage_temperature",
                "environmental_context_stripped": True,
            }
        ],
    }
    validated = cast("dict[str, Any]", DISCOVERY_OUTPUT_SCHEMA(payload))
    assert validated["candidates"][0]["environmental_context_stripped"] is True

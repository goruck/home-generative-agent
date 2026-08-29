# ruff: noqa: S101
"""Tests for discovery engine novelty filtering."""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, cast
from unittest.mock import AsyncMock, patch

import pytest

from custom_components.home_generative_agent.sentinel.discovery_engine import (
    _MAX_SEMANTIC_KEYS_IN_PROMPT,
    _STATIC_RULE_IDS,
    SentinelDiscoveryEngine,
    _battery_level_entity_ids,
    _candidate_identity_hash,
    _entity_ids_from_evidence_paths,
    _entity_ids_from_key,
    _is_cumulative_energy_entity,
)
from custom_components.home_generative_agent.sentinel.discovery_semantic import (
    candidate_semantic_key,
)

if TYPE_CHECKING:
    from homeassistant.core import HomeAssistant

    from custom_components.home_generative_agent.sentinel.discovery_store import (
        DiscoveryStore,
    )


def _monitoring_gaps_from_prompt(human_content: str) -> list[str]:
    """Parse the MONITORING GAPS JSON array out of a captured discovery prompt."""
    gaps_start = human_content.find("MONITORING GAPS: [")
    assert gaps_start != -1, "Prompt missing MONITORING GAPS section"
    array_start = human_content.index("[", gaps_start)
    array_end = human_content.index("]", array_start) + 1
    return json.loads(human_content[array_start:array_end])


class _DummyStore:
    async def async_get_latest(self, _limit: int) -> list[dict[str, Any]]:
        return []


class _DummyProposalStore:
    def __init__(self, proposals: list[dict[str, Any]] | None = None) -> None:
        self._proposals = proposals or []

    async def async_get_latest(self, _limit: int) -> list[dict[str, Any]]:
        return list(self._proposals)

    async def cleanup_unsupported_ttl(self) -> int:
        return 0


def test_filter_novel_candidates_drops_existing_and_batch_duplicates() -> None:
    engine = SentinelDiscoveryEngine(
        hass=cast("HomeAssistant", object()),
        options={},
        model=None,
        store=cast("DiscoveryStore", _DummyStore()),
    )
    candidates = [
        {
            "candidate_id": "c1",
            "title": "Open windows at night while someone is home",
            "summary": "Detect windows open during nighttime when someone is present.",
            "pattern": "window open at night while home",
            "suggested_type": "security_risk",
            "confidence_hint": 0.6,
            "evidence_paths": [
                "entities[entity_id=binary_sensor.playroom_window].state",
                "derived.is_night",
                "derived.anyone_home",
            ],
        },
        {
            "candidate_id": "c2",
            "title": "Garage and playroom windows open while home",
            "summary": "Windows open while occupants are present at night.",
            "pattern": "night home windows open",
            "suggested_type": "security_state",
            "confidence_hint": 0.7,
            "evidence_paths": [
                "derived.anyone_home",
                "entities[entity_id=binary_sensor.playroom_window].state",
                "derived.is_night",
            ],
        },
    ]
    existing_keys = {
        (
            "v1|subject=entry_window|predicate=open|night=1|home=1|scope=any|"
            "entities=binary_sensor.playroom_window"
        )
    }
    filtered, dropped = engine._filter_novel_candidates(candidates, existing_keys)
    assert filtered == []
    assert len(dropped) == 2
    assert dropped[0]["dedupe_reason"] == "existing_semantic_key"
    assert dropped[1]["dedupe_reason"] == "existing_semantic_key"


def test_filter_novel_candidates_sets_novel_reason() -> None:
    engine = SentinelDiscoveryEngine(
        hass=cast("HomeAssistant", object()),
        options={},
        model=None,
        store=cast("DiscoveryStore", _DummyStore()),
    )
    candidates = [
        {
            "candidate_id": "c3",
            "title": "Front door unlocked while home",
            "summary": "Lock left unlocked with occupant home.",
            "pattern": "unlocked lock while home",
            "suggested_type": "security",
            "confidence_hint": 0.8,
            "evidence_paths": [
                "entities[entity_id=lock.front_door].state",
                "derived.anyone_home",
            ],
        }
    ]
    filtered, dropped = engine._filter_novel_candidates(candidates, set())
    assert len(filtered) == 1
    assert filtered[0]["dedupe_reason"] == "novel"
    assert dropped == []


def test_filter_novel_candidates_drops_entity_text_mismatch() -> None:
    """Candidate text must not name baseline entities missing from evidence."""
    engine = SentinelDiscoveryEngine(
        hass=cast("HomeAssistant", object()),
        options={},
        model=None,
        store=cast("DiscoveryStore", _DummyStore()),
    )
    candidates = [
        {
            "candidate_id": "candidate_battery_baseline_deviation_kitchen",
            "title": "Kitchen Lock Battery Baseline Deviation",
            "summary": (
                "Detect unusual battery drops for garage and playroom door locks."
            ),
            "pattern": "deviation from normal battery level",
            "suggested_type": "statistical_anomaly",
            "confidence_hint": 0.8,
            "evidence_paths": [
                "entities[entity_ids contains sensor.kitchen_lock_battery].state"
            ],
        }
    ]
    filtered, dropped = engine._filter_novel_candidates(
        candidates,
        set(),
        [
            "sensor.kitchen_lock_battery",
            "sensor.garage_door_lock_battery",
            "sensor.playroom_door_lock_battery",
        ],
    )
    assert filtered == []
    assert len(dropped) == 1
    assert dropped[0]["dedupe_reason"] == "entity_text_mismatch"
    assert dropped[0]["mismatch_entities"] == (
        "sensor.garage_door_lock_battery,sensor.playroom_door_lock_battery"
    )


def test_filter_novel_candidates_allows_generic_entity_summary() -> None:
    """Generic summaries can pass when they do not name a different entity."""
    engine = SentinelDiscoveryEngine(
        hass=cast("HomeAssistant", object()),
        options={},
        model=None,
        store=cast("DiscoveryStore", _DummyStore()),
    )
    candidate = {
        "candidate_id": "candidate_battery_baseline_deviation_kitchen",
        "title": "Kitchen Lock Battery Baseline Deviation",
        "summary": "Detect unusual drops for this lock battery sensor.",
        "pattern": "deviation from normal battery level",
        "suggested_type": "statistical_anomaly",
        "confidence_hint": 0.8,
        "evidence_paths": [
            "entities[entity_ids contains sensor.kitchen_lock_battery].state"
        ],
    }
    filtered, dropped = engine._filter_novel_candidates(
        [candidate],
        set(),
        [
            "sensor.kitchen_lock_battery",
            "sensor.garage_door_lock_battery",
            "sensor.playroom_door_lock_battery",
        ],
    )
    assert len(filtered) == 1
    assert filtered[0]["dedupe_reason"] == "novel"
    assert dropped == []


def test_filter_novel_candidates_allows_intentional_entity_bundle() -> None:
    """Text may name multiple entities when evidence paths cite all of them."""
    engine = SentinelDiscoveryEngine(
        hass=cast("HomeAssistant", object()),
        options={},
        model=None,
        store=cast("DiscoveryStore", _DummyStore()),
    )
    candidate = {
        "candidate_id": "candidate_battery_baseline_deviation_garage_playroom",
        "title": "Garage and Playroom Lock Battery Baseline Deviation",
        "summary": "Detect unusual battery drops for garage and playroom door locks.",
        "pattern": "deviation from normal battery level",
        "suggested_type": "statistical_anomaly",
        "confidence_hint": 0.8,
        "evidence_paths": [
            "entities[entity_ids contains sensor.garage_door_lock_battery].state",
            "entities[entity_ids contains sensor.playroom_door_lock_battery].state",
        ],
    }
    filtered, dropped = engine._filter_novel_candidates(
        [candidate],
        set(),
        [
            "sensor.kitchen_lock_battery",
            "sensor.garage_door_lock_battery",
            "sensor.playroom_door_lock_battery",
        ],
    )
    assert len(filtered) == 1
    assert filtered[0]["dedupe_reason"] == "novel"
    assert dropped == []


# ---------------------------------------------------------------------------
# Bug 2: null-key candidates (unknown subject/predicate) must be deduplicated
# ---------------------------------------------------------------------------

_NULL_KEY_CANDIDATE: dict[str, Any] = {
    "candidate_id": "nk1",
    "title": "Stale person tracking while away",
    "summary": "Person tracking data appears outdated.",
    "pattern": "stale tracking",
    "suggested_type": "data_quality",
    "confidence_hint": 0.4,
    "evidence_paths": [],
}


def test_candidate_identity_hash_is_stable() -> None:
    """Same title+summary always produces the same hash."""
    h1 = _candidate_identity_hash(_NULL_KEY_CANDIDATE)
    h2 = _candidate_identity_hash(dict(_NULL_KEY_CANDIDATE))
    assert h1 == h2
    assert h1.startswith("ident|sha256=")


def test_candidate_identity_hash_differs_on_content() -> None:
    """Different titles produce different hashes."""
    other = dict(_NULL_KEY_CANDIDATE)
    other["title"] = "Something completely different"
    assert _candidate_identity_hash(_NULL_KEY_CANDIDATE) != _candidate_identity_hash(
        other
    )


def test_candidate_identity_hash_non_battery_null_key_unaffected() -> None:
    """A non-battery null-key candidate keeps hashing on title+summary."""
    other = dict(_NULL_KEY_CANDIDATE)
    other["summary"] = "Person tracking data looks fresh again."
    assert _candidate_identity_hash(_NULL_KEY_CANDIDATE) != _candidate_identity_hash(
        other
    )


def test_candidate_identity_hash_ambiguous_battery_slug_falls_back_to_prose() -> None:
    """A slug with 2+ leftover tokens after stripping topic words is not guessed."""
    ambiguous = {
        "candidate_id": "hall_and_garage_low_battery_sensor",
        "title": "Nízká baterie",
        "summary": "Baterie senzoru je nízká.",
    }
    same_slug_different_prose = dict(ambiguous, summary="Baterie senzoru je slaba.")
    # No confident device token to anchor on, so this must NOT collapse two
    # differently-worded candidates the way a real device token would.
    assert _candidate_identity_hash(ambiguous) != _candidate_identity_hash(
        same_slug_different_prose
    )


def test_filter_null_key_candidate_dropped_when_hash_in_existing() -> None:
    """A null-key candidate whose identity hash is in existing_keys is dropped."""
    engine = SentinelDiscoveryEngine(
        hass=cast("HomeAssistant", object()),
        options={},
        model=None,
        store=cast("DiscoveryStore", _DummyStore()),
    )
    hash_key = _candidate_identity_hash(_NULL_KEY_CANDIDATE)
    filtered, dropped = engine._filter_novel_candidates(
        [_NULL_KEY_CANDIDATE], {hash_key}
    )
    assert filtered == []
    assert len(dropped) == 1
    assert dropped[0]["dedupe_reason"] == "existing_identity_hash"
    assert dropped[0]["identity_hash"] == hash_key


def test_filter_null_key_candidate_batch_dedup() -> None:
    """Two identical null-key candidates in the same batch: second is batch_duplicate."""
    engine = SentinelDiscoveryEngine(
        hass=cast("HomeAssistant", object()),
        options={},
        model=None,
        store=cast("DiscoveryStore", _DummyStore()),
    )
    twin = dict(_NULL_KEY_CANDIDATE)
    twin["candidate_id"] = "nk2"
    filtered, dropped = engine._filter_novel_candidates(
        [_NULL_KEY_CANDIDATE, twin], set()
    )
    assert len(filtered) == 1
    assert len(dropped) == 1
    assert dropped[0]["dedupe_reason"] == "batch_duplicate"


def test_filter_null_key_candidate_novel_when_not_seen() -> None:
    """A null-key candidate that has never been seen passes through as novel."""
    engine = SentinelDiscoveryEngine(
        hass=cast("HomeAssistant", object()),
        options={},
        model=None,
        store=cast("DiscoveryStore", _DummyStore()),
    )
    filtered, dropped = engine._filter_novel_candidates([_NULL_KEY_CANDIDATE], set())
    assert len(filtered) == 1
    assert filtered[0]["dedupe_reason"] == "novel"
    assert dropped == []


def test_filter_null_key_candidate_drops_model_supplied_semantic_key() -> None:
    """
    A null-key candidate's own foreign "semantic_key" field must not survive.

    Adversarial finding on the #572 review: DISCOVERY_OUTPUT_SCHEMA lets the
    model supply an optional "semantic_key". When candidate_semantic_key()
    returns None, `enriched = dict(candidate)` used to carry that field
    through untouched into the stored/returned record. _collect_existing_keys
    prefers a stored semantic_key over recomputation, so a foreign string on
    a null-key candidate could suppress an unrelated, differently-keyed real
    proposal. The enriched record must never carry a semantic_key when the
    computed key is falsy, regardless of what the model supplied.
    """
    engine = SentinelDiscoveryEngine(
        hass=cast("HomeAssistant", object()),
        options={},
        model=None,
        store=cast("DiscoveryStore", _DummyStore()),
    )
    poisoned = dict(_NULL_KEY_CANDIDATE)
    poisoned["semantic_key"] = (
        "v1|subject=entry_window|predicate=open|night=1|home=0|scope=any|entities="
    )
    filtered, dropped = engine._filter_novel_candidates([poisoned], set())
    assert len(filtered) == 1
    assert "semantic_key" not in filtered[0]
    assert dropped == []


_ZERO_EVIDENCE_BATTERY_CANDIDATE: dict[str, Any] = {
    "candidate_id": "low_battery_sensor_0xffffaa67127301f8",
    "title": "Nízká úroveň baterie senzoru 0xffffaa67127301f8",
    "summary": "Baterie senzoru 0xffffaa67127301f8 klesla pod doporučenou hranici.",
    "pattern": "threshold_breach",
    "suggested_type": "maintenance",
    "confidence_hint": 0.5,
    "evidence_paths": [],
}


def test_filter_zero_evidence_low_battery_dedups_on_identity_hash() -> None:
    """
    An evidence-less low_battery candidate really does dedup via identity hash.

    candidate_semantic_key now returns None for this shape (issue #571), which
    is only useful if the engine's `key or _candidate_identity_hash(...)`
    fallback then makes the candidate dedup against itself. Asserting the key
    is None in the semantic tests proves nothing about that: it must be driven
    through _filter_novel_candidates, whose derived-only and
    entity-text-mismatch guards run BEFORE the dedup check and could drop the
    candidate first. Both dedup legs are asserted — a re-proposal in a later
    run (existing_identity_hash) and a twin inside one batch
    (batch_duplicate).
    """
    engine = SentinelDiscoveryEngine(
        hass=cast("HomeAssistant", object()),
        options={},
        model=None,
        store=cast("DiscoveryStore", _DummyStore()),
    )
    hash_key = _candidate_identity_hash(_ZERO_EVIDENCE_BATTERY_CANDIDATE)
    filtered, dropped = engine._filter_novel_candidates(
        [dict(_ZERO_EVIDENCE_BATTERY_CANDIDATE)], {hash_key}
    )
    assert filtered == []
    assert [item["dedupe_reason"] for item in dropped] == ["existing_identity_hash"]

    twin = dict(_ZERO_EVIDENCE_BATTERY_CANDIDATE)
    twin["candidate_id"] = "low_battery_sensor_0xffffaa67127301f8_again"
    twin["confidence_hint"] = 0.9
    filtered, dropped = engine._filter_novel_candidates(
        [dict(_ZERO_EVIDENCE_BATTERY_CANDIDATE), twin], set()
    )
    assert len(filtered) == 1
    assert [item["dedupe_reason"] for item in dropped] == ["batch_duplicate"]
    assert dropped[0]["identity_hash"] == hash_key


def test_filter_zero_evidence_low_battery_keeps_distinct_sensors_novel() -> None:
    """
    Two different evidence-less battery sensors stay distinct through the engine.

    This is the over-merge half of issue #571: before the fix both candidates
    keyed the identical constant
    "subject=unknown|predicate=low_battery|...|entities=" string, so the
    second sensor's proposal was dropped as a batch_duplicate of the first and
    the user never saw it. With the None key the identity hash separates them
    on wording, and both survive as novel.
    """
    engine = SentinelDiscoveryEngine(
        hass=cast("HomeAssistant", object()),
        options={},
        model=None,
        store=cast("DiscoveryStore", _DummyStore()),
    )
    other = dict(_ZERO_EVIDENCE_BATTERY_CANDIDATE)
    other["candidate_id"] = "low_battery_sensor_0xaaaa11122233344"
    other["title"] = "Nízká úroveň baterie senzoru 0xaaaa11122233344"
    other["summary"] = (
        "Baterie senzoru 0xaaaa11122233344 klesla pod doporučenou hranici."
    )
    filtered, dropped = engine._filter_novel_candidates(
        [dict(_ZERO_EVIDENCE_BATTERY_CANDIDATE), other], set()
    )
    assert dropped == []
    assert [item["dedupe_reason"] for item in filtered] == ["novel", "novel"]
    assert [item["candidate_id"] for item in filtered] == [
        "low_battery_sensor_0xffffaa67127301f8",
        "low_battery_sensor_0xaaaa11122233344",
    ]


def test_filter_zero_evidence_battery_collapses_drifting_prose() -> None:
    """
    Three cycles of one sensor with a drifting reading collapse to one card.

    Issue #571 follow-up (TODOS.md "Identity-hash dedup cannot collapse
    re-proposals whose prose carries live values"): LLM candidate prose
    embeds the current reading, so a title+summary hash alone made every
    cycle's re-proposal of the same sensor a new pending card. The
    candidate_id slug's device token is the stable surface across cycles.

    Driven through _filter_novel_candidates rather than asserted on the
    hash directly: the identity key set is an implementation detail, only
    surviving-card count is the user-visible contract.
    """
    engine = SentinelDiscoveryEngine(
        hass=cast("HomeAssistant", object()),
        options={},
        model=None,
        store=cast("DiscoveryStore", _DummyStore()),
    )
    cycle_2 = dict(
        _ZERO_EVIDENCE_BATTERY_CANDIDATE,
        summary="Baterie senzoru 0xffffaa67127301f8 klesla na 11 %.",
    )
    cycle_3 = dict(
        _ZERO_EVIDENCE_BATTERY_CANDIDATE,
        summary=(
            "Baterie senzoru 0xffffaa67127301f8 klesla na 9 %, "
            "brzy bude potreba vymenit."
        ),
    )
    filtered, dropped = engine._filter_novel_candidates(
        [dict(_ZERO_EVIDENCE_BATTERY_CANDIDATE), cycle_2, cycle_3], set()
    )
    assert len(filtered) == 1
    assert [item["dedupe_reason"] for item in dropped] == [
        "batch_duplicate",
        "batch_duplicate",
    ]


def test_filter_zero_evidence_battery_collapses_slug_drift() -> None:
    """
    Stable prose plus a drifting slug still collapses.

    The mirror of the test above, and the regression the two-namespace
    identity hash introduced (review of #573): keying only on the device
    token when one resolves meant a re-proposal whose slug picked up one
    extra word ("..._again") landed in the prose namespace and never met
    its twin. Matching on the full identity key SET covers both drift
    axes, because the surface that stayed stable is always compared.
    """
    engine = SentinelDiscoveryEngine(
        hass=cast("HomeAssistant", object()),
        options={},
        model=None,
        store=cast("DiscoveryStore", _DummyStore()),
    )
    slug_drift = dict(
        _ZERO_EVIDENCE_BATTERY_CANDIDATE,
        candidate_id="low_battery_sensor_0xffffaa67127301f8_again",
    )
    filtered, dropped = engine._filter_novel_candidates(
        [dict(_ZERO_EVIDENCE_BATTERY_CANDIDATE), slug_drift], set()
    )
    assert len(filtered) == 1
    assert [item["dedupe_reason"] for item in dropped] == ["batch_duplicate"]


@pytest.mark.parametrize(
    ("place", "motion_slug", "contact_slug"),
    [
        ("kitchen", "low_battery_sensor_kitchen", "kitchen_battery_low"),
        ("Room 1234", "low_battery_sensor_room1234", "room1234_battery_low"),
    ],
)
def test_filter_zero_evidence_battery_word_slug_token_does_not_merge(
    place: str, motion_slug: str, contact_slug: str
) -> None:
    """
    A place-name slug token must not merge two genuinely different devices.

    Review of #573: the "exactly one leftover token" rule was believed to
    prevent false device-token merges, but it does not — "low_battery_
    sensor_kitchen" and "kitchen_battery_low" both reduce to "kitchen",
    and two unrelated sensors in one room would silently collapse into a
    single card. base main keeps both (their prose differs), so anything
    that merges them is a regression, not just a missed improvement.

    "Room 1234" is the same trap wearing digits: it defeated both a
    "contains a digit" and an "at least half digits" shape test (Codex
    passes 2 and 4). Only a hex address, optionally 0x-prefixed, is
    trusted now; every other token falls through to the prose hash.
    """
    engine = SentinelDiscoveryEngine(
        hass=cast("HomeAssistant", object()),
        options={},
        model=None,
        store=cast("DiscoveryStore", _DummyStore()),
    )
    motion = dict(
        _ZERO_EVIDENCE_BATTERY_CANDIDATE,
        candidate_id=motion_slug,
        title=f"Low battery on the {place} motion sensor",
        summary=f"The {place} motion sensor battery is low.",
    )
    contact = dict(
        _ZERO_EVIDENCE_BATTERY_CANDIDATE,
        candidate_id=contact_slug,
        title=f"Low battery on the {place} window contact",
        summary=f"The {place} window contact battery is low.",
    )
    filtered, dropped = engine._filter_novel_candidates([motion, contact], set())
    assert dropped == []
    assert [item["dedupe_reason"] for item in filtered] == ["novel", "novel"]


@pytest.mark.parametrize("order", [(0, 1, 2), (1, 0, 2), (1, 2, 0), (2, 1, 0)])
def test_filter_identity_keys_never_dedup_less_than_prose_alone(
    order: tuple[int, int, int],
) -> None:
    """
    Multi-key matching is never weaker than the prose hash it extends.

    This is the guarantee the design actually makes, and the one worth
    pinning. It does NOT promise a fixed survivor count across batch
    orderings: prose equality and device-token equality are two
    heuristics, and the union of two heuristic equivalences is not
    transitive, so ordering can shift which pairs meet. Forcing
    transitivity by propagating dropped candidates' keys was tried and
    reverted — it let A link B to C when B and C shared no key at all,
    hiding a genuinely distinct card (third Codex pass on the #573
    hardening).

    What must hold in every ordering: two candidates with identical prose
    always collapse, exactly as they did before the device token existed.
    """
    engine = SentinelDiscoveryEngine(
        hass=cast("HomeAssistant", object()),
        options={},
        model=None,
        store=cast("DiscoveryStore", _DummyStore()),
    )
    shared_prose = {
        "title": "Nízká úroveň baterie senzoru 0xffffaa67127301f8",
        "summary": "Baterie senzoru 0xffffaa67127301f8 klesla pod hranici.",
    }
    candidates = [
        dict(_ZERO_EVIDENCE_BATTERY_CANDIDATE, **shared_prose),
        dict(
            _ZERO_EVIDENCE_BATTERY_CANDIDATE,
            **shared_prose,
            candidate_id="low_battery_sensor_0xaaaa11122233344",
        ),
        dict(
            _ZERO_EVIDENCE_BATTERY_CANDIDATE,
            candidate_id="low_battery_sensor_0xffffaa67127301f8",
            title="Baterie dochazi",
            summary="Baterie senzoru 0xffffaa67127301f8 je temer vybita.",
        ),
    ]
    filtered, _ = engine._filter_novel_candidates([candidates[i] for i in order], set())
    # The two identical-prose candidates can never both survive.
    surviving_prose = [(item["title"], item["summary"]) for item in filtered]
    assert len(surviving_prose) == len(set(surviving_prose))


# ---------------------------------------------------------------------------
# Battery evidence backfill (issue #571)
# ---------------------------------------------------------------------------

_BATTERY_ENTITY_IDS = [
    "sensor.0xffffaa67127301f8_battery",
    "sensor.0xaaaa11122233344_battery",
]

_EVIDENCED_BATTERY_CANDIDATE: dict[str, Any] = {
    "candidate_id": "low_battery_sensor_0xffffaa67127301f8_threshold",
    "title": "Nízká úroveň baterie senzoru 0xffffaa67127301f8",
    "summary": "Upozornit, když kapacita baterie klesne pod definovaný práh.",
    "pattern": "threshold_breach",
    "suggested_type": "maintenance",
    "confidence_hint": 0.6,
    "evidence_paths": ["entities[entity_id=sensor.0xffffaa67127301f8_battery].state"],
}


def _engine() -> SentinelDiscoveryEngine:
    return SentinelDiscoveryEngine(
        hass=cast("HomeAssistant", object()),
        options={},
        model=None,
        store=cast("DiscoveryStore", _DummyStore()),
    )


def test_filter_battery_backfill_dedups_against_evidenced_twin() -> None:
    """
    The reported #571 symptom: two cards for one sensor become one.

    An evidence-less low-battery candidate names the device address but
    cites nothing, so it keys None and falls back to an identity hash — and
    a hash can never equal the semantic key of the properly-evidenced
    proposal about that same sensor. Resolving the address against the
    home's battery sensors and citing it gives both candidates the same
    semantic key, in both arrival orders.
    """
    engine = _engine()
    evidenced_key = candidate_semantic_key(_EVIDENCED_BATTERY_CANDIDATE)
    assert evidenced_key is not None

    # Evidenced proposal already pending; the evidence-less re-description
    # arrives next and must be recognised as the same idea.
    filtered, dropped = engine._filter_novel_candidates(
        [dict(_ZERO_EVIDENCE_BATTERY_CANDIDATE)],
        {evidenced_key},
        None,
        _BATTERY_ENTITY_IDS,
    )
    assert filtered == []
    assert [item["dedupe_reason"] for item in dropped] == ["existing_semantic_key"]
    assert dropped[0]["semantic_key"] == evidenced_key

    # Reverse order, one batch: whichever comes first survives, once.
    filtered, dropped = engine._filter_novel_candidates(
        [dict(_ZERO_EVIDENCE_BATTERY_CANDIDATE), dict(_EVIDENCED_BATTERY_CANDIDATE)],
        set(),
        None,
        _BATTERY_ENTITY_IDS,
    )
    assert len(filtered) == 1
    assert [item["dedupe_reason"] for item in dropped] == ["batch_duplicate"]


def test_filter_battery_backfill_records_path_and_marker() -> None:
    """A backfilled candidate stores the resolved path, a marker, and a key."""
    filtered, dropped = _engine()._filter_novel_candidates(
        [dict(_ZERO_EVIDENCE_BATTERY_CANDIDATE)], set(), None, _BATTERY_ENTITY_IDS
    )
    assert dropped == []
    record = filtered[0]
    assert record["evidence_paths"] == [
        "entities[entity_id=sensor.0xffffaa67127301f8_battery].state"
    ]
    assert record["evidence_backfilled"] is True
    assert record["semantic_key"] == candidate_semantic_key(
        _EVIDENCED_BATTERY_CANDIDATE
    )


def test_filter_battery_backfill_needs_a_unique_sensor() -> None:
    """
    Two sensors carrying the address are ambiguous, so nothing is cited.

    Attaching the wrong sensor would let a real low-battery card be dropped
    as a duplicate of an unrelated one — the outcome this feature treats as
    worse than doing nothing (the #573 review's design rule).
    """
    ambiguous = [
        "sensor.0xffffaa67127301f8_battery",
        "sensor.hall_0xffffaa67127301f8_battery",
    ]
    filtered, _ = _engine()._filter_novel_candidates(
        [dict(_ZERO_EVIDENCE_BATTERY_CANDIDATE)], set(), None, ambiguous
    )
    assert filtered[0]["evidence_paths"] == []
    assert "evidence_backfilled" not in filtered[0]
    assert "semantic_key" not in filtered[0]


def test_filter_battery_backfill_ignores_unresolvable_address() -> None:
    """An address this home does not report leaves the candidate untouched."""
    filtered, _ = _engine()._filter_novel_candidates(
        [dict(_ZERO_EVIDENCE_BATTERY_CANDIDATE)],
        set(),
        None,
        ["sensor.0xdeadbeefdeadbeef_battery"],
    )
    assert filtered[0]["evidence_paths"] == []
    assert "evidence_backfilled" not in filtered[0]


def test_filter_battery_backfill_matches_whole_tokens_only() -> None:
    """
    A longer address that merely starts with the token is not this device.

    Substring matching would make 0xffffaa67127301f8 resolve
    sensor.0xffffaa67127301f8a_battery — a different device — so the
    address must equal a whole object-id token.
    """
    filtered, _ = _engine()._filter_novel_candidates(
        [dict(_ZERO_EVIDENCE_BATTERY_CANDIDATE)],
        set(),
        None,
        ["sensor.0xffffaa67127301f8a_battery"],
    )
    assert filtered[0]["evidence_paths"] == []
    assert "evidence_backfilled" not in filtered[0]


def test_filter_battery_backfill_leaves_non_battery_null_key_alone() -> None:
    """A null-key candidate with no battery signal is never given evidence."""
    filtered, _ = _engine()._filter_novel_candidates(
        [dict(_NULL_KEY_CANDIDATE)], set(), None, _BATTERY_ENTITY_IDS
    )
    assert filtered[0]["evidence_paths"] == []
    assert "evidence_backfilled" not in filtered[0]


def test_filter_battery_backfill_absent_without_battery_entity_ids() -> None:
    """
    With no battery sensors supplied the pre-#571 identity-hash path stands.

    The parameter is optional and every legacy call site omits it, so the
    old behaviour must be exactly preserved when it is not passed.
    """
    engine = _engine()
    hash_key = _candidate_identity_hash(_ZERO_EVIDENCE_BATTERY_CANDIDATE)
    filtered, dropped = engine._filter_novel_candidates(
        [dict(_ZERO_EVIDENCE_BATTERY_CANDIDATE)], {hash_key}
    )
    assert filtered == []
    assert [item["dedupe_reason"] for item in dropped] == ["existing_identity_hash"]


def test_filter_battery_backfill_splits_identically_worded_devices() -> None:
    """
    Identical prose about two different sensors stops merging once resolved.

    TODOS.md "Identically-worded candidates about different devices still
    merge": the prose hash collapses them and the device token cannot split
    them, because multi-key matching is a union. Resolved evidence sidesteps
    that entirely — each candidate keys semantically on its own entity, and
    semantic keys are compared by equality.
    """
    shared_prose = {
        "title": "Nízká úroveň baterie senzoru",
        "summary": "Baterie senzoru klesla pod hranici.",
    }
    candidates = [
        dict(_ZERO_EVIDENCE_BATTERY_CANDIDATE, **shared_prose),
        dict(
            _ZERO_EVIDENCE_BATTERY_CANDIDATE,
            **shared_prose,
            candidate_id="low_battery_sensor_0xaaaa11122233344",
        ),
    ]
    filtered, dropped = _engine()._filter_novel_candidates(
        candidates, set(), None, _BATTERY_ENTITY_IDS
    )
    prefix = "v1|subject=sensor|predicate=low_battery|night=any|home=any|scope=any"
    assert dropped == []
    assert [item["semantic_key"] for item in filtered] == [
        f"{prefix}|entities=sensor.0xffffaa67127301f8_battery",
        f"{prefix}|entities=sensor.0xaaaa11122233344_battery",
    ]


def _snapshot_with(entities: list[Any]) -> dict[str, Any]:
    return {"entities": entities}


def _hass_with_attributes(attributes: dict[str, dict[str, Any]]) -> HomeAssistant:
    return cast(
        "HomeAssistant",
        SimpleNamespace(
            states=SimpleNamespace(
                get=lambda entity_id: (
                    SimpleNamespace(attributes=attributes[entity_id])
                    if entity_id in attributes
                    else None
                )
            )
        ),
    )


def test_battery_level_entity_ids_uses_metadata_first() -> None:
    """
    device_class decides where it exists; the name heuristic only fills in.

    A locale-named charge level (sensor.zamek_vrata_baterie) is a battery
    level because its metadata says so, while battery-named telemetry with
    another device_class or a non-percent unit is not.
    """
    hass = _hass_with_attributes(
        {
            "sensor.zamek_vrata_baterie": {"device_class": "battery"},
            "sensor.battery_power": {"device_class": "power"},
            "sensor.ev_battery_charging_rate": {"unit_of_measurement": "W"},
        }
    )
    snapshot = _snapshot_with(
        [
            {"entity_id": "sensor.zamek_vrata_baterie"},
            {"entity_id": "sensor.battery_power"},
            {"entity_id": "sensor.ev_battery_charging_rate"},
            # No live metadata: the object-id heuristic decides.
            {"entity_id": "sensor.0xffffaa67127301f8_battery"},
            {"entity_id": "sensor.outdoor_temperature"},
            # Non-sensor domains can never be a battery level.
            {"entity_id": "binary_sensor.front_door_battery_low"},
            "not-a-mapping",
        ]
    )
    assert _battery_level_entity_ids(hass, snapshot) == [
        "sensor.zamek_vrata_baterie",
        "sensor.0xffffaa67127301f8_battery",
    ]


# ---------------------------------------------------------------------------
# hint_keys vs filter_keys split
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_history_keys_in_filter_not_hint() -> None:
    """
    Discovery history record keys must appear in filter_keys but NOT hint_keys.

    This prevents multi-entity bundle history records from misleading the LLM
    into thinking individual entities are already covered.
    """
    history_candidate = {
        "candidate_id": "hist_power_bundle",
        "title": "Kitchen power mismatch",
        "summary": "Multiple kitchen appliances power deviates from baseline.",
        "pattern": "deviation_from_baseline",
        "suggested_type": "statistical_anomaly",
        "confidence_hint": 0.8,
        "evidence_paths": [
            "entities[entity_ids contains sensor.fridge_switch_0_power].state",
            "entities[entity_ids contains sensor.kettle_switch_0_power].state",
        ],
    }

    class _HistoryStore:
        async def async_get_latest(self, _limit: int) -> list[dict[str, Any]]:
            return [{"candidates": [history_candidate], "filtered_candidates": []}]

        async def async_append(self, _payload: Any) -> None:
            pass

    engine = SentinelDiscoveryEngine(
        hass=cast("HomeAssistant", object()),
        options={},
        model=None,
        store=cast("DiscoveryStore", _HistoryStore()),
    )
    _active, hint_keys, filter_keys = await engine._existing_semantic_context()

    # The history candidate's key should appear in filter_keys (post-hoc dedup)
    assert any("fridge" in k for k in filter_keys), (
        "filter_keys must contain history key"
    )
    # But NOT in hint_keys (LLM prompt) — it would suppress standalone fridge proposals
    assert not any("fridge" in k for k in hint_keys), (
        "hint_keys must NOT contain history keys (they mislead the LLM)"
    )


@pytest.mark.asyncio
async def test_history_null_key_record_ignores_stored_semantic_key() -> None:
    """
    A stored foreign semantic_key on a null-key history record is not trusted.

    Review of #573: dropping the model-supplied "semantic_key" on the
    null-key write path stops NEW poisoning, but records already inside
    the 200-record window still carry one, and _collect_existing_keys
    preferred a stored key over recomputation unconditionally. Those
    records kept injecting a model-chosen key into filter_keys and
    suppressing the unrelated real proposal it names until they aged out.
    The stored value is now only honoured when the record still computes
    a key of its own; a null-key record recalls by identity keys instead.
    """
    poisoned_history_candidate = dict(_NULL_KEY_CANDIDATE)
    poisoned_history_candidate["semantic_key"] = (
        "v1|subject=entry_window|predicate=open|night=1|home=0|scope=any|entities="
    )

    class _PoisonedHistoryStore:
        async def async_get_latest(self, _limit: int) -> list[dict[str, Any]]:
            return [
                {
                    "candidates": [poisoned_history_candidate],
                    "filtered_candidates": [],
                }
            ]

        async def async_append(self, _payload: Any) -> None:
            pass

    engine = SentinelDiscoveryEngine(
        hass=cast("HomeAssistant", object()),
        options={},
        model=None,
        store=cast("DiscoveryStore", _PoisonedHistoryStore()),
    )
    _active, _hint_keys, filter_keys = await engine._existing_semantic_context()

    assert poisoned_history_candidate["semantic_key"] not in filter_keys
    # The record is still recalled, just by its own identity instead.
    assert _candidate_identity_hash(_NULL_KEY_CANDIDATE) in filter_keys


# ---------------------------------------------------------------------------
# Bug 1: rejected proposals must still block re-suggestion
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_existing_context_rejected_proposal_adds_key() -> None:
    """A rejected proposal's candidate key must appear in semantic_keys."""
    rejected_candidate = {
        "candidate_id": "r1",
        "title": "Front door unlocked while home",
        "summary": "Lock left unlocked with occupant home.",
        "pattern": "unlocked lock while home",
        "suggested_type": "security",
        "confidence_hint": 0.8,
        "evidence_paths": [
            "entities[entity_id=lock.front_door].state",
            "derived.anyone_home",
        ],
    }
    proposal = {
        "candidate_id": "r1",
        "candidate": rejected_candidate,
        "status": "rejected",
        "created_at": "2026-01-01T00:00:00+00:00",
    }
    proposal_store = _DummyProposalStore([proposal])
    engine = SentinelDiscoveryEngine(
        hass=cast("HomeAssistant", object()),
        options={},
        model=None,
        store=cast("DiscoveryStore", _DummyStore()),
        proposal_store=cast("Any", proposal_store),
    )
    _, hint_keys, _filter_keys = await engine._existing_semantic_context()
    # The lock+home semantic key must be present even though status=="rejected".
    assert any("lock" in k and "unlocked" in k for k in hint_keys)


@pytest.mark.asyncio
async def test_existing_context_null_key_rejected_proposal_adds_hash() -> None:
    """A rejected null-key proposal's identity hash must appear in semantic_keys."""
    proposal = {
        "candidate_id": "nk_r1",
        "candidate": _NULL_KEY_CANDIDATE,
        "status": "rejected",
        "created_at": "2026-01-01T00:00:00+00:00",
    }
    proposal_store = _DummyProposalStore([proposal])
    engine = SentinelDiscoveryEngine(
        hass=cast("HomeAssistant", object()),
        options={},
        model=None,
        store=cast("DiscoveryStore", _DummyStore()),
        proposal_store=cast("Any", proposal_store),
    )
    _, hint_keys, _filter_keys = await engine._existing_semantic_context()
    expected_hash = _candidate_identity_hash(_NULL_KEY_CANDIDATE)
    assert expected_hash in hint_keys


# ---------------------------------------------------------------------------
# Bug 5: accepted proposals must NOT block re-suggestion when rule is disabled
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_existing_context_approved_proposal_does_not_add_key_to_hints() -> None:
    """
    An approved proposal must NOT appear in hint_keys.

    When a proposal is approved, a rule is created to track coverage.  If the
    user later disables that rule, the topic should become re-proposable.
    Keeping the approved proposal in hint_keys would silently suppress it
    forever, regardless of whether the rule is still active.
    """
    accepted_candidate = {
        "candidate_id": "a1",
        "title": "Fridge power baseline deviation",
        "summary": "Fridge power deviates from rolling average.",
        "pattern": "deviation_from_normal",
        "suggested_type": "statistical_anomaly",
        "confidence_hint": 0.8,
        "evidence_paths": [
            "entities[entity_ids contains sensor.fridge_switch_0_power].state",
        ],
    }
    proposal = {
        "candidate_id": "a1",
        "candidate": accepted_candidate,
        "status": "approved",
        "created_at": "2026-01-01T00:00:00+00:00",
    }
    proposal_store = _DummyProposalStore([proposal])
    engine = SentinelDiscoveryEngine(
        hass=cast("HomeAssistant", object()),
        options={},
        model=None,
        store=cast("DiscoveryStore", _DummyStore()),
        proposal_store=cast("Any", proposal_store),
    )
    _, hint_keys, _filter_keys = await engine._existing_semantic_context()
    # Accepted proposal key must NOT block re-proposal of the fridge.
    assert not any("fridge" in k for k in hint_keys), (
        "Accepted proposal key must not appear in hint_keys"
    )


@pytest.mark.asyncio
async def test_existing_context_pending_proposal_still_blocks() -> None:
    """A pending proposal's candidate key must still appear in hint_keys."""
    pending_candidate = {
        "candidate_id": "p1",
        "title": "Front door unlocked while home",
        "summary": "Lock left unlocked with occupant home.",
        "pattern": "unlocked lock while home",
        "suggested_type": "security",
        "confidence_hint": 0.8,
        "evidence_paths": [
            "entities[entity_id=lock.front_door].state",
            "derived.anyone_home",
        ],
    }
    proposal = {
        "candidate_id": "p1",
        "candidate": pending_candidate,
        "status": "pending",
        "created_at": "2026-01-01T00:00:00+00:00",
    }
    proposal_store = _DummyProposalStore([proposal])
    engine = SentinelDiscoveryEngine(
        hass=cast("HomeAssistant", object()),
        options={},
        model=None,
        store=cast("DiscoveryStore", _DummyStore()),
        proposal_store=cast("Any", proposal_store),
    )
    _, hint_keys, _filter_keys = await engine._existing_semantic_context()
    assert any("lock" in k and "unlocked" in k for k in hint_keys)


# ---------------------------------------------------------------------------
# Bug 4: static built-in rule IDs must appear in active_rule_ids
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_existing_context_includes_static_rule_ids() -> None:
    """Static built-in rule IDs must be in active_rule_ids even with no registry."""
    engine = SentinelDiscoveryEngine(
        hass=cast("HomeAssistant", object()),
        options={},
        model=None,
        store=cast("DiscoveryStore", _DummyStore()),
    )
    active_rule_ids, _hint, _filter = await engine._existing_semantic_context()
    assert _STATIC_RULE_IDS.issubset(active_rule_ids)


@pytest.mark.asyncio
async def test_existing_context_static_ids_present_alongside_dynamic() -> None:
    """Static IDs appear alongside any dynamic rule IDs from the registry."""

    class _DummyRegistry:
        def list_rules(self) -> list[dict[str, Any]]:
            return [{"rule_id": "my_dynamic_rule"}]

    engine = SentinelDiscoveryEngine(
        hass=cast("HomeAssistant", object()),
        options={},
        model=None,
        store=cast("DiscoveryStore", _DummyStore()),
        rule_registry=cast("Any", _DummyRegistry()),
    )
    active_rule_ids, _hint, _filter = await engine._existing_semantic_context()
    assert "my_dynamic_rule" in active_rule_ids
    assert _STATIC_RULE_IDS.issubset(active_rule_ids)


# ---------------------------------------------------------------------------
# Semantic key prompt cap tests
# ---------------------------------------------------------------------------


def test_max_semantic_keys_in_prompt_constant() -> None:
    """_MAX_SEMANTIC_KEYS_IN_PROMPT must be defined and positive."""
    assert isinstance(_MAX_SEMANTIC_KEYS_IN_PROMPT, int)
    assert _MAX_SEMANTIC_KEYS_IN_PROMPT > 0


@pytest.mark.asyncio
async def test_discovery_prompt_caps_semantic_keys(hass: HomeAssistant) -> None:
    """When existing_semantic_keys exceeds the cap, only cap entries reach the prompt."""
    oversized_keys = {f"key_{i}" for i in range(_MAX_SEMANTIC_KEYS_IN_PROMPT + 20)}
    captured_prompts: list[str] = []

    class _CapturingModel:
        async def ainvoke(self, messages: list[Any]) -> Any:
            captured_prompts.extend(
                str(msg.content) for msg in messages if hasattr(msg, "content")
            )
            return SimpleNamespace(
                content=json.dumps(
                    {
                        "schema_version": 1,
                        "generated_at": "2026-01-01T00:00:00Z",
                        "model": "test",
                        "candidates": [],
                    }
                )
            )

    class _FullDummyStore:
        async def async_get_latest(self, _limit: int) -> list[dict[str, Any]]:
            return []

        async def async_append(self, _payload: Any) -> None:
            pass

    engine = SentinelDiscoveryEngine(
        hass=hass,
        options={},
        model=_CapturingModel(),
        store=_FullDummyStore(),  # type: ignore[arg-type]
    )

    async def _fake_run(model: Any, messages: Any, **_kw: Any) -> Any:
        return await model.ainvoke(messages)

    with (
        patch.object(
            engine,
            "_existing_semantic_context",
            new_callable=AsyncMock,
            return_value=(set(), oversized_keys, oversized_keys),
        ),
        patch(
            "custom_components.home_generative_agent.sentinel.discovery_engine.async_build_full_state_snapshot",
            new_callable=AsyncMock,
            return_value={
                "entities": [],
                "camera_activity": [],
                "derived": {"is_night": False, "now": "2026-01-01T00:00:00Z"},
                "generated_at": "2026-01-01T00:00:00Z",
            },
        ),
        patch(
            "custom_components.home_generative_agent.sentinel.discovery_engine.run_sentinel_model_call",
            side_effect=_fake_run,
        ),
    ):
        await engine._run_once()

    assert captured_prompts, "Model was never invoked"
    # Parse the JSON array from the human message to get an exact key count.
    human_content = captured_prompts[-1]
    keys_start = human_content.find("Existing semantic keys (do not duplicate): [")
    assert keys_start != -1, "Prompt missing existing_semantic_keys section"
    array_start = human_content.index("[", keys_start)
    array_end = human_content.index("]", array_start) + 1
    keys_in_prompt: list[str] = json.loads(human_content[array_start:array_end])
    assert len(keys_in_prompt) <= _MAX_SEMANTIC_KEYS_IN_PROMPT


# ---------------------------------------------------------------------------
# Monitoring gap: unavailability keys must not suppress baseline-ready entities
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_monitoring_gap_ignores_non_baseline_hint_keys(
    hass: HomeAssistant,
) -> None:
    """
    A sensor in a predicate=unavailable key must still be in unmonitored.

    Broad rules like unavailable_sensors cover many entity_ids as side-effects;
    the gap analysis must only check baseline/power-anomaly keys.
    """
    # Hint key that mentions the fridge entity_id but is for unavailability, not
    # baseline monitoring.
    unavail_key = (
        "v1|subject=sensor|predicate=unavailable|night=any|home=1|scope=any|"
        "entities=sensor.backyard_vmd3_0,sensor.fridge_switch_0_power"
    )
    captured_prompts: list[str] = []

    class _CapturingModel:
        async def ainvoke(self, messages: list[Any]) -> Any:
            captured_prompts.extend(
                str(msg.content) for msg in messages if hasattr(msg, "content")
            )
            return SimpleNamespace(
                content=json.dumps(
                    {
                        "schema_version": 1,
                        "generated_at": "2026-01-01T00:00:00Z",
                        "model": "test",
                        "candidates": [],
                    }
                )
            )

    class _FullDummyStore:
        async def async_get_latest(self, _limit: int) -> list[dict[str, Any]]:
            return []

        async def async_append(self, _payload: Any) -> None:
            pass

    engine = SentinelDiscoveryEngine(
        hass=hass,
        options={},
        model=_CapturingModel(),
        store=_FullDummyStore(),  # type: ignore[arg-type]
    )

    async def _fake_run(model: Any, messages: Any, **_kw: Any) -> Any:
        return await model.ainvoke(messages)

    with (
        patch.object(
            engine,
            "_existing_semantic_context",
            new_callable=AsyncMock,
            return_value=(set(), {unavail_key}, {unavail_key}),
        ),
        patch(
            "custom_components.home_generative_agent.sentinel.discovery_engine.async_build_full_state_snapshot",
            new_callable=AsyncMock,
            return_value={
                # The reducer drops baseline_ready_entities absent from the
                # reduced snapshot, so the fridge must exist as an entity —
                # with an empty entity list the gaps array is always [] and
                # the assertion below would pass vacuously via the semantic
                # keys section of the prompt.
                "entities": [
                    {
                        "entity_id": "sensor.fridge_switch_0_power",
                        "domain": "sensor",
                        "state": "120",
                        "attributes": {"device_class": "power"},
                    },
                ],
                "camera_activity": [],
                "derived": {
                    "is_night": False,
                    "now": "2026-01-01T00:00:00Z",
                    "baseline_ready_entities": ["sensor.fridge_switch_0_power"],
                },
                "generated_at": "2026-01-01T00:00:00Z",
            },
        ),
        patch(
            "custom_components.home_generative_agent.sentinel.discovery_engine.run_sentinel_model_call",
            side_effect=_fake_run,
        ),
    ):
        await engine._run_once()

    assert captured_prompts, "Model was never invoked"
    human_content = captured_prompts[-1]
    # The fridge must appear in the MONITORING GAPS array even though it
    # appears in an unavailability hint key.
    gaps_in_prompt = _monitoring_gaps_from_prompt(human_content)
    assert gaps_in_prompt == ["sensor.fridge_switch_0_power"], (
        "Fridge must be in MONITORING GAPS when only covered by unavailability key"
    )


@pytest.mark.asyncio
async def test_monitoring_gap_suppressed_by_power_anomaly_key(
    hass: HomeAssistant,
) -> None:
    """A sensor with an active power_anomaly key must NOT be in unmonitored."""
    power_key = (
        "v1|subject=sensor|predicate=power_anomaly"
        "|template=baseline_deviation|entities=sensor.fridge_switch_0_power"
    )
    captured_prompts: list[str] = []

    class _CapturingModel:
        async def ainvoke(self, messages: list[Any]) -> Any:
            captured_prompts.extend(
                str(msg.content) for msg in messages if hasattr(msg, "content")
            )
            return SimpleNamespace(
                content=json.dumps(
                    {
                        "schema_version": 1,
                        "generated_at": "2026-01-01T00:00:00Z",
                        "model": "test",
                        "candidates": [],
                    }
                )
            )

    class _FullDummyStore:
        async def async_get_latest(self, _limit: int) -> list[dict[str, Any]]:
            return []

        async def async_append(self, _payload: Any) -> None:
            pass

    engine = SentinelDiscoveryEngine(
        hass=hass,
        options={},
        model=_CapturingModel(),
        store=_FullDummyStore(),  # type: ignore[arg-type]
    )

    async def _fake_run(model: Any, messages: Any, **_kw: Any) -> Any:
        return await model.ainvoke(messages)

    with (
        patch.object(
            engine,
            "_existing_semantic_context",
            new_callable=AsyncMock,
            return_value=(set(), {power_key}, {power_key}),
        ),
        patch(
            "custom_components.home_generative_agent.sentinel.discovery_engine.async_build_full_state_snapshot",
            new_callable=AsyncMock,
            return_value={
                # Fridge present as an entity so the reducer keeps it in
                # baseline_ready_entities — suppression must come from the
                # power_anomaly key, not from the reducer trimming it away.
                "entities": [
                    {
                        "entity_id": "sensor.fridge_switch_0_power",
                        "domain": "sensor",
                        "state": "120",
                        "attributes": {"device_class": "power"},
                    },
                ],
                "camera_activity": [],
                "derived": {
                    "is_night": False,
                    "now": "2026-01-01T00:00:00Z",
                    "baseline_ready_entities": ["sensor.fridge_switch_0_power"],
                },
                "generated_at": "2026-01-01T00:00:00Z",
            },
        ),
        patch(
            "custom_components.home_generative_agent.sentinel.discovery_engine.run_sentinel_model_call",
            side_effect=_fake_run,
        ),
    ):
        await engine._run_once()

    assert captured_prompts, "Model was never invoked"
    human_content = captured_prompts[-1]
    gaps_in_prompt = _monitoring_gaps_from_prompt(human_content)
    # Fridge is already baseline-monitored; it must NOT appear in MONITORING GAPS.
    assert gaps_in_prompt == [], (
        "Fridge must not be in MONITORING GAPS when covered by power_anomaly key"
    )


@pytest.mark.asyncio
async def test_monitoring_gap_bundle_candidate_key_does_not_suppress(
    hass: HomeAssistant,
) -> None:
    """
    A multi-entity bundle candidate key must NOT suppress individual entity gaps.

    A rejected proposal that bundles many appliances into one candidate key
    (no |template=| marker) must not prevent each individual appliance from
    appearing in unmonitored_baseline_entities.
    """
    # A candidate key covering 8 appliances as a bundle (no template= marker).
    bundle_key = (
        "v1|subject=sensor|predicate=power_anomaly|night=any|home=any|scope=any|"
        "entities=sensor.dishwasher_switch_0_energy,sensor.dishwasher_switch_0_power,"
        "sensor.fridge_switch_0_energy,sensor.fridge_switch_0_power,"
        "sensor.kettle_switch_0_energy,sensor.kettle_switch_0_power,"
        "sensor.microwave_switch_0_energy,sensor.microwave_switch_0_power"
    )
    captured_prompts: list[str] = []

    class _CapturingModel:
        async def ainvoke(self, messages: list[Any]) -> Any:
            captured_prompts.extend(
                str(msg.content) for msg in messages if hasattr(msg, "content")
            )
            return SimpleNamespace(
                content=json.dumps(
                    {
                        "schema_version": 1,
                        "generated_at": "2026-01-01T00:00:00Z",
                        "model": "test",
                        "candidates": [],
                    }
                )
            )

    class _FullDummyStore:
        async def async_get_latest(self, _limit: int) -> list[dict[str, Any]]:
            return []

        async def async_append(self, _payload: Any) -> None:
            pass

    engine = SentinelDiscoveryEngine(
        hass=hass,
        options={},
        model=_CapturingModel(),
        store=_FullDummyStore(),  # type: ignore[arg-type]
    )

    async def _fake_run(model: Any, messages: Any, **_kw: Any) -> Any:
        return await model.ainvoke(messages)

    with (
        patch.object(
            engine,
            "_existing_semantic_context",
            new_callable=AsyncMock,
            return_value=(set(), {bundle_key}, {bundle_key}),
        ),
        patch(
            "custom_components.home_generative_agent.sentinel.discovery_engine.async_build_full_state_snapshot",
            new_callable=AsyncMock,
            return_value={
                "entities": [],
                "camera_activity": [],
                "derived": {
                    "is_night": False,
                    "now": "2026-01-01T00:00:00Z",
                    "baseline_ready_entities": ["sensor.fridge_switch_0_power"],
                },
                "generated_at": "2026-01-01T00:00:00Z",
            },
        ),
        patch(
            "custom_components.home_generative_agent.sentinel.discovery_engine.run_sentinel_model_call",
            side_effect=_fake_run,
        ),
    ):
        await engine._run_once()

    assert captured_prompts, "Model was never invoked"
    human_content = captured_prompts[-1]
    # Bundle key has no |template=| marker — fridge must still appear in MONITORING GAPS.
    assert "sensor.fridge_switch_0_power" in human_content, (
        "Fridge must appear in MONITORING GAPS even when covered only by a bundle key"
    )


# ---------------------------------------------------------------------------
# P2: _entity_ids_from_key — exact entity ID extraction (no substring matches)
# ---------------------------------------------------------------------------


def test_entity_ids_from_key_single_entity() -> None:
    key = (
        "v1|subject=sensor|predicate=power_anomaly"
        "|template=time_of_day_anomaly|entities=sensor.fridge_switch_0_power"
    )
    assert _entity_ids_from_key(key) == {"sensor.fridge_switch_0_power"}


def test_entity_ids_from_key_multiple_entities() -> None:
    key = (
        "v1|subject=sensor|predicate=unavailable|night=any|home=1|scope=any|"
        "entities=sensor.foo,sensor.bar,sensor.baz"
    )
    assert _entity_ids_from_key(key) == {"sensor.foo", "sensor.bar", "sensor.baz"}


def test_entity_ids_from_key_empty_entities_field() -> None:
    key = "v1|subject=camera|predicate=unknown_person|night=any|home=0|scope=any|entities="
    assert _entity_ids_from_key(key) == set()


def test_entity_ids_from_key_no_entities_field() -> None:
    key = "v1|subject=sensor|predicate=power_anomaly"
    assert _entity_ids_from_key(key) == set()


def test_entity_ids_from_key_no_substring_match() -> None:
    """sensor.fridge_switch_0_power must NOT match sensor.fridge_switch_0_power_factor."""
    key = (
        "v1|subject=sensor|predicate=power_anomaly"
        "|template=baseline_deviation|entities=sensor.fridge_switch_0_power_factor"
    )
    ids = _entity_ids_from_key(key)
    assert "sensor.fridge_switch_0_power" not in ids
    assert "sensor.fridge_switch_0_power_factor" in ids


@pytest.mark.asyncio
async def test_monitoring_gap_power_factor_does_not_suppress_power(
    hass: HomeAssistant,
) -> None:
    """
    A baseline key for power_factor must not suppress the power sensor via substring.

    This is the P2 regression: 'sensor.fridge_switch_0_power' is a substring of
    'sensor.fridge_switch_0_power_factor'.  The old `eid in key` check treated
    them as covered; the new exact entity-set parse must not.
    """
    power_factor_key = (
        "v1|subject=sensor|predicate=power_anomaly"
        "|template=time_of_day_anomaly|entities=sensor.fridge_switch_0_power_factor"
    )
    captured_prompts: list[str] = []

    class _CapturingModel:
        async def ainvoke(self, messages: list[Any]) -> Any:
            captured_prompts.extend(
                str(msg.content) for msg in messages if hasattr(msg, "content")
            )
            return SimpleNamespace(
                content=json.dumps(
                    {
                        "schema_version": 1,
                        "generated_at": "2026-01-01T00:00:00Z",
                        "model": "test",
                        "candidates": [],
                    }
                )
            )

    class _FullDummyStore:
        async def async_get_latest(self, _limit: int) -> list[dict[str, Any]]:
            return []

        async def async_append(self, _payload: Any) -> None:
            pass

    engine = SentinelDiscoveryEngine(
        hass=hass,
        options={},
        model=_CapturingModel(),
        store=_FullDummyStore(),  # type: ignore[arg-type]
    )

    async def _fake_run(model: Any, messages: Any, **_kw: Any) -> Any:
        return await model.ainvoke(messages)

    with (
        patch.object(
            engine,
            "_existing_semantic_context",
            new_callable=AsyncMock,
            return_value=(set(), {power_factor_key}, {power_factor_key}),
        ),
        patch(
            "custom_components.home_generative_agent.sentinel.discovery_engine.async_build_full_state_snapshot",
            new_callable=AsyncMock,
            return_value={
                "entities": [],
                "camera_activity": [],
                "derived": {
                    "is_night": False,
                    "now": "2026-01-01T00:00:00Z",
                    "baseline_ready_entities": ["sensor.fridge_switch_0_power"],
                },
                "generated_at": "2026-01-01T00:00:00Z",
            },
        ),
        patch(
            "custom_components.home_generative_agent.sentinel.discovery_engine.run_sentinel_model_call",
            side_effect=_fake_run,
        ),
    ):
        await engine._run_once()

    assert captured_prompts, "Model was never invoked"
    human_content = captured_prompts[-1]
    # power_factor key must NOT suppress power sensor — it must appear in MONITORING GAPS.
    assert "sensor.fridge_switch_0_power" in human_content, (
        "power sensor must appear in MONITORING GAPS when only power_factor has a baseline key"
    )


# ---------------------------------------------------------------------------
# Cumulative energy sensor filtering
# ---------------------------------------------------------------------------


def test_is_cumulative_energy_entity_rejects_energy_suffix() -> None:
    """Entity IDs ending in _energy are cumulative kWh counters."""
    assert _is_cumulative_energy_entity("sensor.microwave_switch_0_energy")
    assert _is_cumulative_energy_entity("sensor.washing_machine_switch_0_energy")
    assert _is_cumulative_energy_entity("sensor.fridge_switch_0_energy")
    assert _is_cumulative_energy_entity("sensor.dishwasher_switch_0_energy")
    assert _is_cumulative_energy_entity("sensor.energy")
    assert _is_cumulative_energy_entity("energy")


def test_is_cumulative_energy_entity_allows_power_sensors() -> None:
    """Instantaneous power sensors must pass through the filter."""
    assert not _is_cumulative_energy_entity("sensor.microwave_switch_0_power")
    assert not _is_cumulative_energy_entity("sensor.fridge_switch_0_power")
    assert not _is_cumulative_energy_entity("sensor.fridge_switch_0_power_factor")
    assert not _is_cumulative_energy_entity("sensor.washing_machine_switch_0_power")


@pytest.mark.asyncio
async def test_monitoring_gap_excludes_cumulative_energy_entities(
    hass: HomeAssistant,
) -> None:
    """
    Cumulative _energy sensors must be excluded from MONITORING GAPS.

    These entities grow monotonically and produce noise when proposed as
    baseline_deviation or time_of_day_anomaly candidates.  The discovery
    engine must strip them before injecting the unmonitored list into the
    prompt so the LLM is never directed to propose candidates for them.

    Both _energy and _power sensors are present in the snapshot entities so
    the discovery reducer keeps them all in baseline_ready_entities.  The
    engine's _is_cumulative_energy_entity filter then removes the _energy
    ones from the MONITORING GAPS injection even though they survive in the
    snapshot JSON.  We verify MONITORING GAPS specifically, not the full
    prompt, to avoid false matches against the snapshot data.
    """
    import re  # noqa: PLC0415

    captured_prompts: list[str] = []

    class _CapturingModel:
        async def ainvoke(self, messages: list[Any]) -> Any:
            captured_prompts.extend(
                str(msg.content) for msg in messages if hasattr(msg, "content")
            )
            return SimpleNamespace(
                content=json.dumps(
                    {
                        "schema_version": 1,
                        "generated_at": "2026-01-01T00:00:00Z",
                        "model": "test",
                        "candidates": [],
                    }
                )
            )

    class _FullDummyStore:
        async def async_get_latest(self, _limit: int) -> list[dict[str, Any]]:
            return []

        async def async_append(self, _payload: Any) -> None:
            pass

    engine = SentinelDiscoveryEngine(
        hass=hass,
        options={},
        model=_CapturingModel(),
        store=_FullDummyStore(),  # type: ignore[arg-type]
    )

    async def _fake_run(model: Any, messages: Any, **_kw: Any) -> Any:
        return await model.ainvoke(messages)

    # Entities must use the raw snapshot format that _filter_entities expects:
    # each entity is a flat dict with entity_id, domain, state, and attributes.
    # Both _energy (device_class=energy) and _power (device_class=power) pass
    # through _ALLOWED_SENSOR_DEVICE_CLASSES so both land in entity_id_set and
    # survive the reducer's baseline_ready_entities trim.
    def _make_sensor(entity_id: str, device_class: str, state: str) -> dict[str, Any]:
        return {
            "entity_id": entity_id,
            "domain": "sensor",
            "state": state,
            "attributes": {"device_class": device_class},
        }

    with (
        patch.object(
            engine,
            "_existing_semantic_context",
            new_callable=AsyncMock,
            return_value=(set(), set(), set()),
        ),
        patch(
            "custom_components.home_generative_agent.sentinel.discovery_engine.async_build_full_state_snapshot",
            new_callable=AsyncMock,
            return_value={
                "entities": [
                    _make_sensor("sensor.microwave_switch_0_energy", "energy", "125.3"),
                    _make_sensor("sensor.microwave_switch_0_power", "power", "0.0"),
                    _make_sensor(
                        "sensor.washing_machine_switch_0_energy", "energy", "88.1"
                    ),
                    _make_sensor(
                        "sensor.washing_machine_switch_0_power", "power", "0.0"
                    ),
                    _make_sensor("sensor.fridge_switch_0_energy", "energy", "310.7"),
                    _make_sensor("sensor.fridge_switch_0_power", "power", "42.0"),
                    _make_sensor("sensor.dishwasher_switch_0_energy", "energy", "55.2"),
                    _make_sensor("sensor.dishwasher_switch_0_power", "power", "0.0"),
                ],
                "camera_activity": [],
                "derived": {
                    "is_night": False,
                    "now": "2026-01-01T00:00:00Z",
                    "baseline_ready_entities": [
                        # Cumulative kWh counters — must be excluded from MONITORING GAPS
                        "sensor.microwave_switch_0_energy",
                        "sensor.washing_machine_switch_0_energy",
                        "sensor.fridge_switch_0_energy",
                        "sensor.dishwasher_switch_0_energy",
                        # Instantaneous power — must remain in MONITORING GAPS
                        "sensor.microwave_switch_0_power",
                        "sensor.washing_machine_switch_0_power",
                        "sensor.fridge_switch_0_power",
                        "sensor.dishwasher_switch_0_power",
                    ],
                },
                "generated_at": "2026-01-01T00:00:00Z",
            },
        ),
        patch(
            "custom_components.home_generative_agent.sentinel.discovery_engine.run_sentinel_model_call",
            side_effect=_fake_run,
        ),
    ):
        await engine._run_once()

    assert captured_prompts, "Model was never invoked"
    human_content = captured_prompts[-1]

    # Extract the MONITORING GAPS JSON array from the prompt.  The template
    # embeds it as: "MONITORING GAPS: <json_array> are baseline-ready entities"
    gap_match = re.search(
        r"MONITORING GAPS: (\[.*?\]) are baseline-ready", human_content, re.DOTALL
    )
    assert gap_match, "MONITORING GAPS section not found in prompt"
    gap_entities: list[str] = json.loads(gap_match.group(1))

    # Cumulative sensors must NOT be in MONITORING GAPS.
    assert "sensor.microwave_switch_0_energy" not in gap_entities
    assert "sensor.washing_machine_switch_0_energy" not in gap_entities
    assert "sensor.fridge_switch_0_energy" not in gap_entities
    assert "sensor.dishwasher_switch_0_energy" not in gap_entities

    # Instantaneous power sensors must appear in MONITORING GAPS.
    assert "sensor.microwave_switch_0_power" in gap_entities
    assert "sensor.washing_machine_switch_0_power" in gap_entities
    assert "sensor.fridge_switch_0_power" in gap_entities
    assert "sensor.dishwasher_switch_0_power" in gap_entities


def test_entity_ids_from_evidence_paths_bare_bracket_format() -> None:
    """
    The hallucination-guard extractor parses all three evidence formats.

    Without the bare-bracket alternative, all-bare-bracket evidence yields
    an empty set (guard short-circuits) and mixed-format evidence reads a
    legitimately-cited entity as a hallucination, silently dropping an
    approvable candidate (issue #522 security review).
    """
    ids = _entity_ids_from_evidence_paths(
        [
            "entities[entity_id=sensor.named_battery].state",
            "entities[entity_ids contains sensor.contained_battery].state",
            "entities[sensor.zamek_vrata_baterie].state",
            "entities['sensor.quoted_battery'].state",
            "entities[''sensor.doublequoted_battery''].state",
            "entities[31].state",
            "not derived.anyone_home",
        ]
    )
    assert ids == {
        "sensor.named_battery",
        "sensor.contained_battery",
        "sensor.zamek_vrata_baterie",
        "sensor.quoted_battery",
        "sensor.doublequoted_battery",
    }


# ---------------------------------------------------------------------------
# Issue #524: canonicalized derived-only filter + guard scope documentation
# ---------------------------------------------------------------------------


def test_filter_novel_candidates_drops_negated_derived_only_paths() -> None:
    """
    A lone negated derived path is derived-only and never promotable.

    Before #524 the filter checked startswith("derived."), so a candidate
    whose only path was "not derived.anyone_home" slipped past and died
    later with a less useful reason.
    """
    engine = SentinelDiscoveryEngine(
        hass=cast("HomeAssistant", object()),
        options={},
        model=None,
        store=cast("DiscoveryStore", _DummyStore()),
    )
    candidates = [
        {
            "candidate_id": "away_context_only",
            "title": "Activity while away",
            "summary": "Something happens while nobody is home.",
            "pattern": "not derived.anyone_home",
            "suggested_type": "security",
            "confidence_hint": 0.5,
            "evidence_paths": ["not derived.anyone_home"],
        }
    ]
    filtered, dropped = engine._filter_novel_candidates(candidates, set())
    assert filtered == []
    assert len(dropped) == 1
    assert dropped[0]["dedupe_reason"] == "derived_only_paths"


def test_filter_novel_candidates_non_english_prose_passes_mismatch_guard() -> None:
    """
    Translated prose passes the entity-text mismatch guard untouched.

    Documents the guard's current no-op for non-English text (descriptor
    tokens come from English object-ids, so the subset match cannot fire):
    the translation follow-up to issue #524 must revisit it. Prose stays
    English in production for now.

    The Czech prose deliberately NAMES the non-evidence garage entity
    ("pohyb v garáži") — an English candidate with the equivalent prose
    would be dropped as entity_text_mismatch (see the English test above),
    so this pair actually pins the English-only limitation rather than
    passing under any guard implementation (#524 testing review).
    """
    engine = SentinelDiscoveryEngine(
        hass=cast("HomeAssistant", object()),
        options={},
        model=None,
        store=cast("DiscoveryStore", _DummyStore()),
    )
    candidates = [
        {
            "candidate_id": "pohyb_v_kuchyni_pryc",
            "title": "Neočekávaný pohyb v kuchyni",
            "summary": (
                "Detekuje pohyb v kuchyni, když nikdo není doma, "
                "zatímco pohyb v garáži je běžný."
            ),
            "pattern": "binary_sensor.kitchen_motion == on",
            "suggested_type": "security",
            "confidence_hint": 0.6,
            "evidence_paths": [
                "entities[entity_id=binary_sensor.kitchen_motion].state",
                "not derived.anyone_home",
            ],
        }
    ]
    filtered, dropped = engine._filter_novel_candidates(
        candidates,
        set(),
        ["binary_sensor.kitchen_motion", "binary_sensor.garage_motion"],
    )
    assert len(filtered) == 1
    assert dropped == []


def test_filter_novel_candidates_english_prose_naming_entity_is_dropped() -> None:
    """
    English control for the Czech pass-through test above.

    The same candidate shape with English prose naming the garage entity IS
    dropped — together the pair pins that the guard's protection is
    English-only (#524 testing review).
    """
    engine = SentinelDiscoveryEngine(
        hass=cast("HomeAssistant", object()),
        options={},
        model=None,
        store=cast("DiscoveryStore", _DummyStore()),
    )
    candidates = [
        {
            "candidate_id": "kitchen_motion_while_away",
            "title": "Unexpected kitchen motion",
            "summary": (
                "Detects motion in the kitchen while nobody is home, "
                "whereas garage motion is routine."
            ),
            "pattern": "binary_sensor.kitchen_motion == on",
            "suggested_type": "security",
            "confidence_hint": 0.6,
            "evidence_paths": [
                "entities[entity_id=binary_sensor.kitchen_motion].state",
                "not derived.anyone_home",
            ],
        }
    ]
    filtered, dropped = engine._filter_novel_candidates(
        candidates,
        set(),
        ["binary_sensor.kitchen_motion", "binary_sensor.garage_motion"],
    )
    assert filtered == []
    assert len(dropped) == 1
    assert dropped[0]["dedupe_reason"] == "entity_text_mismatch"


def test_filter_novel_candidates_junk_plus_derived_only_is_dropped() -> None:
    """Non-string junk elements do not count as concrete evidence."""
    engine = SentinelDiscoveryEngine(
        hass=cast("HomeAssistant", object()),
        options={},
        model=None,
        store=cast("DiscoveryStore", _DummyStore()),
    )
    candidates = [
        {
            "candidate_id": "junk_evidence",
            "title": "Activity while away",
            "summary": "Something happens while nobody is home.",
            "pattern": "not derived.anyone_home",
            "suggested_type": "security",
            "confidence_hint": 0.5,
            "evidence_paths": [123, "not derived.anyone_home"],
        }
    ]
    filtered, dropped = engine._filter_novel_candidates(candidates, set())
    assert filtered == []
    assert len(dropped) == 1
    assert dropped[0]["dedupe_reason"] == "derived_only_paths"


def test_filter_novel_candidates_junk_plus_concrete_path_passes_gate() -> None:
    """A concrete string path still satisfies the gate alongside junk."""
    engine = SentinelDiscoveryEngine(
        hass=cast("HomeAssistant", object()),
        options={},
        model=None,
        store=cast("DiscoveryStore", _DummyStore()),
    )
    candidates = [
        {
            "candidate_id": "mixed_evidence",
            "title": "Front door unlocked while home",
            "summary": "Lock left unlocked with occupant home.",
            "pattern": "unlocked lock while home",
            "suggested_type": "security",
            "confidence_hint": 0.8,
            "evidence_paths": [123, "entities[entity_id=lock.front_door].state"],
        }
    ]
    filtered, dropped = engine._filter_novel_candidates(candidates, set())
    assert len(filtered) == 1
    assert dropped == []


@pytest.mark.asyncio
async def test_monitoring_gap_excludes_battery_level_sensors(
    hass: HomeAssistant,
) -> None:
    """
    Battery-level sensors never appear in MONITORING GAPS.

    Battery percentage declines monotonically, so a rolling-average baseline
    on it is only a laggy low-battery detector — and gap-hinting battery
    sensors makes the LLM propose confusing occupancy/night-conditioned
    battery candidates. A battery-named power stream (sensor.battery_power on
    a home battery) is a real measurement and must stay gap-eligible.
    """
    captured_prompts: list[str] = []

    class _CapturingModel:
        async def ainvoke(self, messages: list[Any]) -> Any:
            captured_prompts.extend(
                str(msg.content) for msg in messages if hasattr(msg, "content")
            )
            return SimpleNamespace(
                content=json.dumps(
                    {
                        "schema_version": 1,
                        "generated_at": "2026-01-01T00:00:00Z",
                        "model": "test",
                        "candidates": [],
                    }
                )
            )

    class _FullDummyStore:
        async def async_get_latest(self, _limit: int) -> list[dict[str, Any]]:
            return []

        async def async_append(self, _payload: Any) -> None:
            pass

    engine = SentinelDiscoveryEngine(
        hass=hass,
        options={},
        model=_CapturingModel(),
        store=_FullDummyStore(),  # type: ignore[arg-type]
    )

    async def _fake_run(model: Any, messages: Any, **_kw: Any) -> Any:
        return await model.ainvoke(messages)

    with (
        patch.object(
            engine,
            "_existing_semantic_context",
            new_callable=AsyncMock,
            return_value=(set(), set(), set()),
        ),
        patch(
            "custom_components.home_generative_agent.sentinel.discovery_engine.async_build_full_state_snapshot",
            new_callable=AsyncMock,
            return_value={
                # The reducer drops baseline_ready_entities absent from the
                # reduced snapshot, so the sensors must exist as entities.
                "entities": [
                    {
                        "entity_id": "sensor.garage_temp_sensor_battery",
                        "domain": "sensor",
                        "state": "87",
                        "attributes": {"device_class": "battery"},
                    },
                    {
                        "entity_id": "sensor.fridge_switch_0_power",
                        "domain": "sensor",
                        "state": "120",
                        "attributes": {"device_class": "power"},
                    },
                    {
                        "entity_id": "sensor.battery_power",
                        "domain": "sensor",
                        "state": "300",
                        "attributes": {"device_class": "power"},
                    },
                ],
                "camera_activity": [],
                "derived": {
                    "is_night": False,
                    "now": "2026-01-01T00:00:00Z",
                    "baseline_ready_entities": [
                        "sensor.garage_temp_sensor_battery",
                        "sensor.fridge_switch_0_power",
                        "sensor.battery_power",
                    ],
                },
                "generated_at": "2026-01-01T00:00:00Z",
            },
        ),
        patch(
            "custom_components.home_generative_agent.sentinel.discovery_engine.run_sentinel_model_call",
            side_effect=_fake_run,
        ),
    ):
        await engine._run_once()

    assert captured_prompts, "Model was never invoked"
    human_content = captured_prompts[-1]
    gaps_in_prompt = _monitoring_gaps_from_prompt(human_content)
    assert "sensor.garage_temp_sensor_battery" not in gaps_in_prompt, (
        "Battery-level sensor must not be hinted as a statistical monitoring gap"
    )
    assert gaps_in_prompt == [
        "sensor.fridge_switch_0_power",
        "sensor.battery_power",
    ]


@pytest.mark.asyncio
async def test_monitoring_gap_battery_exclusion_uses_state_metadata(
    hass: HomeAssistant,
) -> None:
    """
    Live state metadata beats the English name heuristic in the gap filter.

    A locale-named battery sensor (sensor.zamek_vrata_baterie, device_class
    battery) must be excluded even though its name never says "battery", and
    a battery-NAMED telemetry stream with an unlisted suffix
    (sensor.ev_battery_charging_rate, device_class power) must stay
    gap-eligible (Codex review rounds).
    """
    hass.states.async_set(
        "sensor.zamek_vrata_baterie", "87", {"device_class": "battery"}
    )
    hass.states.async_set(
        "sensor.ev_battery_charging_rate", "7.4", {"device_class": "power"}
    )
    captured_prompts: list[str] = []

    class _CapturingModel:
        async def ainvoke(self, messages: list[Any]) -> Any:
            captured_prompts.extend(
                str(msg.content) for msg in messages if hasattr(msg, "content")
            )
            return SimpleNamespace(
                content=json.dumps(
                    {
                        "schema_version": 1,
                        "generated_at": "2026-01-01T00:00:00Z",
                        "model": "test",
                        "candidates": [],
                    }
                )
            )

    class _FullDummyStore:
        async def async_get_latest(self, _limit: int) -> list[dict[str, Any]]:
            return []

        async def async_append(self, _payload: Any) -> None:
            pass

    engine = SentinelDiscoveryEngine(
        hass=hass,
        options={},
        model=_CapturingModel(),
        store=_FullDummyStore(),  # type: ignore[arg-type]
    )

    async def _fake_run(model: Any, messages: Any, **_kw: Any) -> Any:
        return await model.ainvoke(messages)

    with (
        patch.object(
            engine,
            "_existing_semantic_context",
            new_callable=AsyncMock,
            return_value=(set(), set(), set()),
        ),
        patch(
            "custom_components.home_generative_agent.sentinel.discovery_engine.async_build_full_state_snapshot",
            new_callable=AsyncMock,
            return_value={
                "entities": [
                    {
                        "entity_id": "sensor.zamek_vrata_baterie",
                        "domain": "sensor",
                        "state": "87",
                        "attributes": {"device_class": "battery"},
                    },
                    {
                        "entity_id": "sensor.ev_battery_charging_rate",
                        "domain": "sensor",
                        "state": "7.4",
                        "attributes": {"device_class": "power"},
                    },
                ],
                "camera_activity": [],
                "derived": {
                    "is_night": False,
                    "now": "2026-01-01T00:00:00Z",
                    "baseline_ready_entities": [
                        "sensor.zamek_vrata_baterie",
                        "sensor.ev_battery_charging_rate",
                    ],
                },
                "generated_at": "2026-01-01T00:00:00Z",
            },
        ),
        patch(
            "custom_components.home_generative_agent.sentinel.discovery_engine.run_sentinel_model_call",
            side_effect=_fake_run,
        ),
    ):
        await engine._run_once()

    assert captured_prompts, "Model was never invoked"
    gaps_in_prompt = _monitoring_gaps_from_prompt(captured_prompts[-1])
    assert gaps_in_prompt == ["sensor.ev_battery_charging_rate"]


# ---------------------------------------------------------------------------
# Environmental sensors reach discovery end-to-end (issue #541)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_monitoring_gap_includes_environmental_entity(
    hass: HomeAssistant,
) -> None:
    """
    A baseline-ready temperature sensor survives reduction and is gap-hinted.

    Pre-#541 the reducer dropped environmental device classes, so the
    baseline-ready trim erased the entity before the gap analysis ever saw it.
    """
    captured_prompts: list[str] = []

    class _CapturingModel:
        async def ainvoke(self, messages: list[Any]) -> Any:
            captured_prompts.extend(
                str(msg.content) for msg in messages if hasattr(msg, "content")
            )
            return SimpleNamespace(
                content=json.dumps(
                    {
                        "schema_version": 1,
                        "generated_at": "2026-01-01T00:00:00Z",
                        "model": "test",
                        "candidates": [],
                    }
                )
            )

    class _FullDummyStore:
        async def async_get_latest(self, _limit: int) -> list[dict[str, Any]]:
            return []

        async def async_append(self, _payload: Any) -> None:
            pass

    engine = SentinelDiscoveryEngine(
        hass=hass,
        options={},
        model=_CapturingModel(),
        store=_FullDummyStore(),  # type: ignore[arg-type]
    )

    async def _fake_run(model: Any, messages: Any, **_kw: Any) -> Any:
        return await model.ainvoke(messages)

    with (
        patch.object(
            engine,
            "_existing_semantic_context",
            new_callable=AsyncMock,
            return_value=(set(), set(), set()),
        ),
        patch(
            "custom_components.home_generative_agent.sentinel.discovery_engine.async_build_full_state_snapshot",
            new_callable=AsyncMock,
            return_value={
                "entities": [
                    {
                        "entity_id": "sensor.attic_temperature",
                        "domain": "sensor",
                        "state": "83.66",
                        "attributes": {"device_class": "temperature"},
                    },
                ],
                "camera_activity": [],
                "derived": {
                    "is_night": False,
                    "now": "2026-01-01T00:00:00Z",
                    "baseline_ready_entities": ["sensor.attic_temperature"],
                },
                "generated_at": "2026-01-01T00:00:00Z",
            },
        ),
        patch(
            "custom_components.home_generative_agent.sentinel.discovery_engine.run_sentinel_model_call",
            side_effect=_fake_run,
        ),
    ):
        await engine._run_once()

    assert captured_prompts, "Model was never invoked"
    human_content = captured_prompts[-1]
    gaps_in_prompt = _monitoring_gaps_from_prompt(human_content)
    assert gaps_in_prompt == ["sensor.attic_temperature"]
    # The entity reaches the prompt snapshot with its integer-rounded state.
    assert "sensor.attic_temperature" in human_content
    assert '"state":"84"' in human_content
    # The environmental guidance block ships with the prompt (issue #541).
    assert "ENVIRONMENTAL SENSOR RULE" in human_content
    assert "BATTERY SENSOR RULE" in human_content


@pytest.mark.asyncio
async def test_promoted_environmental_rule_dedupes_reproposal(
    hass: HomeAssistant,
) -> None:
    """
    An active baseline rule on a temperature sensor suppresses re-proposals.

    The rule's |template=| key removes the entity from MONITORING GAPS, and a
    context-variant re-proposal collapses to the pending candidate's
    context-free key and is dropped as existing_semantic_key — the #540
    pile-up shape, now for environmental sensors.
    """
    rule_key = (
        "v1|subject=sensor|predicate=power_anomaly"
        "|template=baseline_deviation|entities=sensor.attic_temperature"
    )
    pending_candidate_key = (
        "v1|subject=sensor|predicate=power_anomaly|night=any|home=any|scope=any|"
        "entities=sensor.attic_temperature"
    )
    captured_prompts: list[str] = []
    appended: list[dict[str, Any]] = []

    class _CapturingModel:
        async def ainvoke(self, messages: list[Any]) -> Any:
            captured_prompts.extend(
                str(msg.content) for msg in messages if hasattr(msg, "content")
            )
            return SimpleNamespace(
                content=json.dumps(
                    {
                        "schema_version": 1,
                        "generated_at": "2026-01-01T00:00:00Z",
                        "model": "test",
                        "candidates": [
                            {
                                "candidate_id": (
                                    "candidate_attic_temperature_baseline_night_home"
                                ),
                                "title": (
                                    "Attic Temperature Anomaly During Night While Home"
                                ),
                                "summary": (
                                    "Detects statistical deviation from the normal "
                                    "attic temperature reading during nighttime "
                                    "hours while someone is home."
                                ),
                                "pattern": "statistical_baseline_deviation",
                                "confidence_hint": 0.6,
                                "evidence_paths": [
                                    "entities[sensor.attic_temperature].state",
                                    "derived.is_night",
                                    "derived.anyone_home",
                                ],
                            }
                        ],
                    }
                )
            )

    class _FullDummyStore:
        async def async_get_latest(self, _limit: int) -> list[dict[str, Any]]:
            return []

        async def async_append(self, payload: Any) -> None:
            appended.append(payload)

    engine = SentinelDiscoveryEngine(
        hass=hass,
        options={},
        model=_CapturingModel(),
        store=_FullDummyStore(),  # type: ignore[arg-type]
    )

    async def _fake_run(model: Any, messages: Any, **_kw: Any) -> Any:
        return await model.ainvoke(messages)

    with (
        patch.object(
            engine,
            "_existing_semantic_context",
            new_callable=AsyncMock,
            return_value=(
                set(),
                {rule_key, pending_candidate_key},
                {rule_key, pending_candidate_key},
            ),
        ),
        patch(
            "custom_components.home_generative_agent.sentinel.discovery_engine.async_build_full_state_snapshot",
            new_callable=AsyncMock,
            return_value={
                "entities": [
                    {
                        "entity_id": "sensor.attic_temperature",
                        "domain": "sensor",
                        "state": "83.66",
                        "attributes": {"device_class": "temperature"},
                    },
                ],
                "camera_activity": [],
                "derived": {
                    "is_night": True,
                    "anyone_home": True,
                    "now": "2026-01-01T00:00:00Z",
                    "baseline_ready_entities": ["sensor.attic_temperature"],
                },
                "generated_at": "2026-01-01T00:00:00Z",
            },
        ),
        patch(
            "custom_components.home_generative_agent.sentinel.discovery_engine.run_sentinel_model_call",
            side_effect=_fake_run,
        ),
    ):
        await engine._run_once()

    assert captured_prompts, "Model was never invoked"
    gaps_in_prompt = _monitoring_gaps_from_prompt(captured_prompts[-1])
    assert gaps_in_prompt == [], (
        "Baseline-rule-covered temperature sensor must not be gap-hinted"
    )
    assert len(appended) == 1
    payload = appended[0]
    assert payload["candidates"] == []
    assert len(payload["filtered_candidates"]) == 1
    dropped = payload["filtered_candidates"][0]
    assert dropped["dedupe_reason"] == "existing_semantic_key"
    assert dropped["semantic_key"] == pending_candidate_key

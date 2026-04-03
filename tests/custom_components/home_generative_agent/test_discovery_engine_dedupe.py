"""Tests for discovery engine novelty filtering."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

import pytest

from custom_components.home_generative_agent.sentinel.discovery_engine import (
    _STATIC_RULE_IDS,
    SentinelDiscoveryEngine,
    _candidate_identity_hash,
)

if TYPE_CHECKING:
    from homeassistant.core import HomeAssistant

    from custom_components.home_generative_agent.sentinel.discovery_store import (
        DiscoveryStore,
    )


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
        "v1|subject=entry_window|predicate=open|night=1|home=1|scope=any|"
        "entities=binary_sensor.playroom_window"
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
    _, semantic_keys = await engine._existing_semantic_context()
    # The lock+home semantic key must be present even though status=="rejected".
    assert any("lock" in k and "unlocked" in k for k in semantic_keys)


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
    _, semantic_keys = await engine._existing_semantic_context()
    expected_hash = _candidate_identity_hash(_NULL_KEY_CANDIDATE)
    assert expected_hash in semantic_keys


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
    active_rule_ids, _ = await engine._existing_semantic_context()
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
    active_rule_ids, _ = await engine._existing_semantic_context()
    assert "my_dynamic_rule" in active_rule_ids
    assert _STATIC_RULE_IDS.issubset(active_rule_ids)

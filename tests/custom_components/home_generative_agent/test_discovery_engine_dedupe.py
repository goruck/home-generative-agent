# ruff: noqa: S101
"""Tests for discovery engine novelty filtering."""

from __future__ import annotations

from custom_components.home_generative_agent.sentinel.discovery_engine import (
    SentinelDiscoveryEngine,
)


class _DummyStore:
    async def async_get_latest(self, _limit: int):
        return []


def test_filter_novel_candidates_drops_existing_and_batch_duplicates() -> None:
    engine = SentinelDiscoveryEngine(
        hass=object(),
        options={},
        model=None,
        store=_DummyStore(),
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
        hass=object(),
        options={},
        model=None,
        store=_DummyStore(),
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

# ruff: noqa: S101
"""Tests for discovery output schema."""

from __future__ import annotations

from typing import cast

import pytest
import voluptuous as vol

from custom_components.home_generative_agent.sentinel.discovery_schema import (
    DISCOVERY_OUTPUT_SCHEMA,
    DISCOVERY_SCHEMA_VERSION,
)


def test_discovery_schema_valid() -> None:
    payload = {
        "schema_version": DISCOVERY_SCHEMA_VERSION,
        "generated_at": "2025-01-01T00:00:00+00:00",
        "model": "test-model",
        "candidates": [
            {
                "candidate_id": "c1",
                "title": "Possible entry left open",
                "summary": "Back door open while away",
                "evidence_paths": ["derived.anyone_home"],
                "pattern": "door open while away",
                "confidence_hint": 0.6,
                "semantic_key": "v1|subject=entry_door|predicate=open|night=any|home=0|scope=any|entities=binary_sensor.front_door",
                "dedupe_reason": "novel",
            }
        ],
        "filtered_candidates": [
            {
                "candidate_id": "c2",
                "semantic_key": "v1|subject=entry_door|predicate=open|night=any|home=0|scope=any|entities=binary_sensor.front_door",
                "dedupe_reason": "existing_semantic_key",
            }
        ],
    }
    validated = cast("dict[str, object]", DISCOVERY_OUTPUT_SCHEMA(payload))
    assert validated["schema_version"] == DISCOVERY_SCHEMA_VERSION


def test_discovery_schema_invalid() -> None:
    payload = {
        "schema_version": 999,
        "generated_at": "2025-01-01T00:00:00+00:00",
        "model": "test-model",
        "candidates": [],
    }
    with pytest.raises(vol.Invalid):
        DISCOVERY_OUTPUT_SCHEMA(payload)


def test_schema_rejects_non_string_evidence_paths() -> None:
    """
    evidence_paths entries must be strings.

    The derived-only hard filter and the canonicalizer assume string paths;
    schema validation upstream is what makes non-string entries unreachable
    there (issue #524 testing review pin).
    """
    payload = {
        "schema_version": 1,
        "generated_at": "2026-08-02T00:00:00+00:00",
        "model": "test",
        "candidates": [
            {
                "candidate_id": "bad_paths",
                "title": "Bad paths",
                "summary": "Candidate with a non-string evidence path.",
                "evidence_paths": [1, "derived.is_night"],
                "pattern": "x",
                "confidence_hint": 0.5,
            }
        ],
    }
    with pytest.raises(vol.Invalid):
        DISCOVERY_OUTPUT_SCHEMA(payload)


def test_schema_round_trips_evidence_backfilled_marker() -> None:
    """
    A stored record carrying the backfill marker validates again.

    The engine sets "evidence_backfilled" on candidates whose battery
    evidence it resolved (issue #571), and stored payloads are round-tripped
    through this PREVENT_EXTRA schema — same reason
    "environmental_context_stripped" is declared.
    """
    payload = {
        "schema_version": DISCOVERY_SCHEMA_VERSION,
        "generated_at": "2026-08-29T00:00:00+00:00",
        "model": "test",
        "candidates": [
            {
                "candidate_id": "low_battery_sensor_0xffffaa67127301f8",
                "title": "Low battery",
                "summary": "Sensor battery below threshold.",
                "evidence_paths": [
                    "entities[entity_id=sensor.0xffffaa67127301f8_battery].state"
                ],
                "pattern": "threshold_breach",
                "confidence_hint": 0.5,
                "evidence_backfilled": True,
            }
        ],
    }
    validated = cast("dict[str, object]", DISCOVERY_OUTPUT_SCHEMA(payload))
    candidates = cast("list[dict[str, object]]", validated["candidates"])
    assert candidates[0]["evidence_backfilled"] is True

# ruff: noqa: S101
"""Tests for discovery output schema."""

from __future__ import annotations

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
    validated = DISCOVERY_OUTPUT_SCHEMA(payload)
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

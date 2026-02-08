# ruff: noqa: S101
"""Tests for snapshot schema validation."""

from __future__ import annotations

import pytest
import voluptuous as vol

from custom_components.home_generative_agent.snapshot.schema import (
    SNAPSHOT_SCHEMA_VERSION,
    validate_snapshot,
)


def _base_snapshot() -> dict[str, object]:
    return {
        "schema_version": SNAPSHOT_SCHEMA_VERSION,
        "generated_at": "2025-01-01T00:00:00+00:00",
        "entities": [],
        "camera_activity": [],
        "derived": {
            "now": "2025-01-01T00:00:00+00:00",
            "timezone": "UTC",
            "is_night": False,
            "anyone_home": False,
            "last_motion_by_area": {},
        },
    }


def test_validate_snapshot_ok() -> None:
    """Validate a minimal snapshot."""
    snapshot = _base_snapshot()
    validated = validate_snapshot(snapshot)
    assert validated["schema_version"] == SNAPSHOT_SCHEMA_VERSION


def test_validate_snapshot_missing_field() -> None:
    """Validate schema rejects missing required fields."""
    snapshot = _base_snapshot()
    snapshot.pop("generated_at")
    with pytest.raises(vol.Invalid):
        validate_snapshot(snapshot)  # type: ignore[arg-type]

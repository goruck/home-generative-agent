# ruff: noqa: S101
"""Tests for snapshot schema validation."""

from __future__ import annotations

import pytest
import voluptuous as vol

from custom_components.home_generative_agent.snapshot.camera_activity import (
    extract_camera_activity,
)
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
            "people_home": [],
            "people_away": [],
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


def test_validate_snapshot_people_home_list() -> None:
    """Validate schema accepts non-empty people_home and people_away lists."""
    snapshot = _base_snapshot()
    derived = snapshot["derived"]
    assert isinstance(derived, dict)
    derived["people_home"] = ["Alice", "Bob"]
    derived["people_away"] = ["Carol"]
    derived["anyone_home"] = True
    validated = validate_snapshot(snapshot)
    assert validated["derived"]["people_home"] == ["Alice", "Bob"]
    assert validated["derived"]["people_away"] == ["Carol"]
    assert validated["derived"]["anyone_home"] is True


def test_validate_snapshot_missing_people_home_rejected() -> None:
    """Validate schema rejects snapshot missing people_home."""
    snapshot = _base_snapshot()
    derived = snapshot["derived"]
    assert isinstance(derived, dict)
    derived.pop("people_home")
    with pytest.raises(vol.Invalid):
        validate_snapshot(snapshot)  # type: ignore[arg-type]


def test_validate_snapshot_missing_people_away_rejected() -> None:
    """Validate schema rejects snapshot missing people_away."""
    snapshot = _base_snapshot()
    derived = snapshot["derived"]
    assert isinstance(derived, dict)
    derived.pop("people_away")
    with pytest.raises(vol.Invalid):
        validate_snapshot(snapshot)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Camera-activity extraction: recognition_last_event
# ---------------------------------------------------------------------------


def _fake_state(entity_id: str, attributes: dict) -> object:
    class _State:
        def __init__(self) -> None:
            self.entity_id = entity_id
            self.attributes = attributes

    return _State()


def test_extract_camera_activity_carries_recognition_last_event() -> None:
    """
    The image entity's last_event lands in recognition_last_event.

    It must survive independently of last_activity so the unknown-person
    freshness gate keys on the sighting itself, not on camera motion
    attributes a pet can refresh.
    """
    camera = _fake_state(
        "camera.backyard",
        {"last_motion": "2026-02-01T12:59:00+00:00"},
    )
    image = _fake_state(
        "image.backyard_last_event",
        {
            "camera_id": "camera.backyard",
            "last_event": "2026-02-01T12:00:00+00:00",
            "recognized_people": ["Unknown Person"],
        },
    )
    activity = extract_camera_activity(camera, "Backyard", image)  # type: ignore[arg-type]
    # Camera attribute wins the generic last_activity slot...
    assert activity["last_activity"] == "2026-02-01T12:59:00+00:00"
    # ...but the sighting timestamp is preserved separately.
    assert activity.get("recognition_last_event") == "2026-02-01T12:00:00+00:00"
    assert activity["recognized_people"] == ["Unknown Person"]


def test_extract_camera_activity_last_event_fallback_fills_last_activity() -> None:
    """Without camera activity attributes, last_event fills both fields."""
    camera = _fake_state("camera.backyard", {})
    image = _fake_state(
        "image.backyard_last_event",
        {
            "camera_id": "camera.backyard",
            "last_event": "2026-02-01T12:00:00+00:00",
            "recognized_people": ["Unknown Person"],
        },
    )
    activity = extract_camera_activity(camera, None, image)  # type: ignore[arg-type]
    assert activity["last_activity"] == "2026-02-01T12:00:00+00:00"
    assert activity.get("recognition_last_event") == "2026-02-01T12:00:00+00:00"


def test_validate_snapshot_accepts_recognition_last_event() -> None:
    """The optional recognition_last_event key passes schema validation."""
    snapshot = _base_snapshot()
    snapshot["camera_activity"] = [
        {
            "camera_entity_id": "camera.backyard",
            "area": None,
            "last_activity": "2026-02-01T12:00:00+00:00",
            "motion_entities": [],
            "vmd_entities": [],
            "snapshot_summary": None,
            "recognized_people": ["Unknown Person"],
            "latest_path": None,
            "recognition_last_event": "2026-02-01T12:00:00+00:00",
        }
    ]
    validated = validate_snapshot(snapshot)
    assert (
        validated["camera_activity"][0].get("recognition_last_event")
        == "2026-02-01T12:00:00+00:00"
    )

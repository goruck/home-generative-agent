"""Tests for discovery snapshot reducer."""

from __future__ import annotations

from typing import Any

from custom_components.home_generative_agent.snapshot.discovery_reducer import (
    _MAX_ENTITIES,
    _MAX_SUMMARY_CHARS,
    _truncate_iso,
    reduce_snapshot_for_discovery,
)
from custom_components.home_generative_agent.snapshot.schema import (
    FullStateSnapshot,
    validate_snapshot,
)


def _make_snapshot(**overrides: Any) -> FullStateSnapshot:
    """Build a minimal valid snapshot with optional overrides."""
    base: dict[str, Any] = {
        "schema_version": 1,
        "generated_at": "2025-01-01T00:00:00+00:00",
        "entities": [],
        "camera_activity": [],
        "derived": {
            "now": "2025-01-01T00:00:00+00:00",
            "timezone": "UTC",
            "is_night": False,
            "anyone_home": True,
            "people_home": [],
            "people_away": [],
            "last_motion_by_area": {},
        },
    }
    base.update(overrides)
    return validate_snapshot(base)


def _make_entity(
    entity_id: str,
    domain: str,
    state: str,
    area: str | None = None,
    device_class: str | None = None,
    last_changed: str | None = None,
) -> dict[str, Any]:
    """Build a minimal entity dict."""
    attrs: dict[str, str] = {}
    if device_class:
        attrs["device_class"] = device_class
    ts = last_changed or "2025-01-01T00:00:00+00:00"
    return {
        "entity_id": entity_id,
        "domain": domain,
        "state": state,
        "friendly_name": entity_id,
        "area": area,
        "attributes": attrs,
        "last_changed": ts,
        "last_updated": ts,
    }


def _make_camera(
    camera_id: str,
    area: str | None = None,
    summary: str | None = None,
    people: list[str] | None = None,
    last_activity: str | None = None,
) -> dict[str, Any]:
    """Build a minimal camera activity dict."""
    return {
        "camera_entity_id": camera_id,
        "area": area,
        "last_activity": last_activity or "2025-01-01T00:00:00+00:00",
        "motion_entities": [],
        "vmd_entities": [],
        "snapshot_summary": summary,
        "recognized_people": people or [],
        "latest_path": None,
    }


# --- Domain / device_class filtering ---


def test_reduce_snapshot_filters_entities() -> None:
    snapshot = _make_snapshot(
        entities=[
            _make_entity(
                "light.kitchen",
                "light",
                "on",
                area="Kitchen",
            ),
            _make_entity(
                "binary_sensor.front_door",
                "binary_sensor",
                "on",
                area="Front",
                device_class="door",
            ),
            _make_entity(
                "sensor.washer_power",
                "sensor",
                "250",
                area="Laundry",
                device_class="power",
            ),
        ],
        camera_activity=[
            _make_camera(
                "camera.front",
                area="Front",
                summary="Person at door",
                people=["Alex"],
            ),
        ],
    )

    reduced = reduce_snapshot_for_discovery(snapshot)

    # light.kitchen should be filtered out; 2 entities remain
    assert len(reduced["entities"]) == 2
    all_ids: list[str] = []
    for group in reduced["entities"]:
        all_ids.extend(group["entity_ids"])
    assert "binary_sensor.front_door" in all_ids
    assert "sensor.washer_power" in all_ids
    assert "light.kitchen" not in all_ids
    # Camera preserved
    assert reduced["camera_activity"][0]["camera_entity_id"] == "camera.front"


def test_sensor_without_allowed_device_class_excluded() -> None:
    snapshot = _make_snapshot(
        entities=[
            _make_entity(
                "sensor.temperature",
                "sensor",
                "22.5",
                device_class="temperature",
            ),
        ],
    )
    reduced = reduce_snapshot_for_discovery(snapshot)
    assert len(reduced["entities"]) == 0


def test_binary_sensor_without_allowed_device_class_excluded() -> None:
    snapshot = _make_snapshot(
        entities=[
            _make_entity(
                "binary_sensor.smoke",
                "binary_sensor",
                "off",
                device_class="smoke",
            ),
        ],
    )
    reduced = reduce_snapshot_for_discovery(snapshot)
    assert len(reduced["entities"]) == 0


# --- Grouping ---


def test_entities_with_same_domain_area_state_are_grouped() -> None:
    snapshot = _make_snapshot(
        entities=[
            _make_entity(
                "binary_sensor.kitchen_window_1",
                "binary_sensor",
                "off",
                area="Kitchen",
                device_class="window",
            ),
            _make_entity(
                "binary_sensor.kitchen_window_2",
                "binary_sensor",
                "off",
                area="Kitchen",
                device_class="window",
            ),
            _make_entity(
                "binary_sensor.kitchen_window_3",
                "binary_sensor",
                "off",
                area="Kitchen",
                device_class="window",
            ),
        ],
    )
    reduced = reduce_snapshot_for_discovery(snapshot)
    assert len(reduced["entities"]) == 1
    group = reduced["entities"][0]
    assert len(group["entity_ids"]) == 3
    assert "binary_sensor.kitchen_window_1" in group["entity_ids"]
    assert group["state"] == "off"
    assert group["area"] == "Kitchen"
    assert group["device_class"] == "window"


def test_entities_with_different_states_not_grouped() -> None:
    snapshot = _make_snapshot(
        entities=[
            _make_entity(
                "binary_sensor.kitchen_window",
                "binary_sensor",
                "off",
                area="Kitchen",
                device_class="window",
            ),
            _make_entity(
                "binary_sensor.kitchen_door",
                "binary_sensor",
                "on",
                area="Kitchen",
                device_class="door",
            ),
        ],
    )
    reduced = reduce_snapshot_for_discovery(snapshot)
    assert len(reduced["entities"]) == 2


def test_entities_with_different_areas_not_grouped() -> None:
    snapshot = _make_snapshot(
        entities=[
            _make_entity(
                "binary_sensor.kitchen_window",
                "binary_sensor",
                "off",
                area="Kitchen",
                device_class="window",
            ),
            _make_entity(
                "binary_sensor.bedroom_window",
                "binary_sensor",
                "off",
                area="Bedroom",
                device_class="window",
            ),
        ],
    )
    reduced = reduce_snapshot_for_discovery(snapshot)
    assert len(reduced["entities"]) == 2


# --- Timestamp truncation ---


def test_truncate_iso_to_minute() -> None:
    assert _truncate_iso("2026-02-22T05:56:11.988456+00:00") == "2026-02-22T05:56"
    assert _truncate_iso("2026-02-22T05:56:11") == "2026-02-22T05:56"
    assert _truncate_iso(None) is None
    assert _truncate_iso("") is None


def test_timestamps_truncated_in_output() -> None:
    snapshot = _make_snapshot(
        entities=[
            _make_entity(
                "lock.front_door",
                "lock",
                "locked",
                area="Front",
                last_changed="2025-01-01T12:34:56.789+00:00",
            ),
        ],
    )
    reduced = reduce_snapshot_for_discovery(snapshot)
    group = reduced["entities"][0]
    assert group["last_changed"] == "2025-01-01T12:34"


def test_derived_timestamps_truncated() -> None:
    snapshot = _make_snapshot()
    reduced = reduce_snapshot_for_discovery(snapshot)
    assert reduced["derived"]["now"] == "2025-01-01T00:00"


# --- schema_version and generated_at removed from output ---


def test_no_schema_version_or_generated_at_in_output() -> None:
    snapshot = _make_snapshot()
    reduced = reduce_snapshot_for_discovery(snapshot)
    assert "schema_version" not in reduced
    assert "generated_at" not in reduced


# --- Camera summary truncation ---


def test_camera_summary_truncated() -> None:
    long_summary = "A" * 300
    snapshot = _make_snapshot(
        camera_activity=[
            _make_camera("camera.front", summary=long_summary),
        ],
    )
    reduced = reduce_snapshot_for_discovery(snapshot)
    cam = reduced["camera_activity"][0]
    # +3 for "..."
    assert len(cam["snapshot_summary"]) == _MAX_SUMMARY_CHARS + 3
    assert cam["snapshot_summary"].endswith("...")


def test_camera_empty_fields_omitted() -> None:
    snapshot = _make_snapshot(
        camera_activity=[
            _make_camera(
                "camera.front",
                summary=None,
                people=[],
            ),
        ],
    )
    reduced = reduce_snapshot_for_discovery(snapshot)
    cam = reduced["camera_activity"][0]
    assert "snapshot_summary" not in cam
    assert "recognized_people" not in cam


# --- Anomaly prioritization ---


def test_anomalous_states_prioritized_over_normal() -> None:
    """When capped, entities with anomalous states survive."""
    normal_entities = [
        _make_entity(
            f"binary_sensor.window_{i}",
            "binary_sensor",
            "off",
            area="Kitchen",
            device_class="window",
        )
        for i in range(_MAX_ENTITIES + 5)
    ]
    anomalous = _make_entity(
        "lock.front_door",
        "lock",
        "unlocked",
        area="Front",
    )
    entities = [*normal_entities, anomalous]

    snapshot = _make_snapshot(entities=entities)
    reduced = reduce_snapshot_for_discovery(snapshot)

    all_ids: list[str] = []
    for group in reduced["entities"]:
        all_ids.extend(group["entity_ids"])
    assert "lock.front_door" in all_ids


# --- Domain field dropped ---


def test_domain_not_in_grouped_output() -> None:
    snapshot = _make_snapshot(
        entities=[
            _make_entity(
                "lock.front_door",
                "lock",
                "locked",
                area="Front",
            ),
        ],
    )
    reduced = reduce_snapshot_for_discovery(snapshot)
    group = reduced["entities"][0]
    assert "domain" not in group

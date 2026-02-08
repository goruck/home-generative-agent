# ruff: noqa: S101
"""Tests for discovery snapshot reducer."""

from __future__ import annotations

from custom_components.home_generative_agent.snapshot.discovery_reducer import (
    reduce_snapshot_for_discovery,
)
from custom_components.home_generative_agent.snapshot.schema import validate_snapshot


def test_reduce_snapshot_filters_entities() -> None:
    snapshot = {
        "schema_version": 1,
        "generated_at": "2025-01-01T00:00:00+00:00",
        "entities": [
            {
                "entity_id": "light.kitchen",
                "domain": "light",
                "state": "on",
                "friendly_name": "Kitchen",
                "area": "Kitchen",
                "attributes": {},
                "last_changed": "2025-01-01T00:00:00+00:00",
                "last_updated": "2025-01-01T00:00:00+00:00",
            },
            {
                "entity_id": "binary_sensor.front_door",
                "domain": "binary_sensor",
                "state": "on",
                "friendly_name": "Front Door",
                "area": "Front",
                "attributes": {"device_class": "door"},
                "last_changed": "2025-01-01T00:00:00+00:00",
                "last_updated": "2025-01-01T00:00:00+00:00",
            },
            {
                "entity_id": "sensor.washer_power",
                "domain": "sensor",
                "state": "250",
                "friendly_name": "Washer Power",
                "area": "Laundry",
                "attributes": {"device_class": "power"},
                "last_changed": "2025-01-01T00:00:00+00:00",
                "last_updated": "2025-01-01T00:00:00+00:00",
            },
        ],
        "camera_activity": [
            {
                "camera_entity_id": "camera.front",
                "area": "Front",
                "last_activity": "2025-01-01T00:00:00+00:00",
                "motion_entities": [],
                "vmd_entities": [],
                "snapshot_summary": "Person at door",
                "recognized_people": ["Alex"],
                "latest_path": "/media/snapshots/camera_front/_latest.jpg",
            }
        ],
        "derived": {
            "now": "2025-01-01T00:00:00+00:00",
            "timezone": "UTC",
            "is_night": False,
            "anyone_home": True,
            "last_motion_by_area": {},
        },
    }

    reduced = reduce_snapshot_for_discovery(validate_snapshot(snapshot))
    assert len(reduced["entities"]) == 2
    assert reduced["entities"][0]["entity_id"] == "binary_sensor.front_door"
    assert reduced["camera_activity"][0]["camera_entity_id"] == "camera.front"

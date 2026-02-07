"""Deterministic reducer for discovery snapshots."""

from __future__ import annotations

from typing import Any

from .schema import FullStateSnapshot

_ALLOWED_DOMAINS = {
    "alarm_control_panel",
    "binary_sensor",
    "camera",
    "cover",
    "lock",
    "person",
    "sensor",
}

_ALLOWED_SENSOR_DEVICE_CLASSES = {"power", "energy", "battery"}
_ALLOWED_BINARY_CLASSES = {"door", "window", "opening", "motion", "occupancy"}
_MAX_ENTITIES = 200
_MAX_CAMERA_ACTIVITY = 50


def reduce_snapshot_for_discovery(snapshot: FullStateSnapshot) -> dict[str, Any]:
    """Return a reduced snapshot for LLM discovery prompts."""
    reduced_entities: list[dict[str, Any]] = []
    for entity in snapshot["entities"]:
        domain = entity["domain"]
        if domain not in _ALLOWED_DOMAINS:
            continue
        attrs = entity.get("attributes", {})
        device_class = attrs.get("device_class")
        if domain == "sensor" and device_class not in _ALLOWED_SENSOR_DEVICE_CLASSES:
            continue
        if domain == "binary_sensor" and device_class not in _ALLOWED_BINARY_CLASSES:
            continue
        reduced_entities.append(
            {
                "entity_id": entity["entity_id"],
                "domain": domain,
                "state": entity["state"],
                "area": entity.get("area"),
                "device_class": device_class,
                "last_changed": entity.get("last_changed"),
            }
        )

    reduced_entities.sort(key=lambda item: item["entity_id"])
    if len(reduced_entities) > _MAX_ENTITIES:
        reduced_entities = reduced_entities[:_MAX_ENTITIES]

    reduced_camera_activity: list[dict[str, Any]] = []
    for camera in snapshot["camera_activity"]:
        reduced_camera_activity.append(
            {
                "camera_entity_id": camera["camera_entity_id"],
                "area": camera.get("area"),
                "last_activity": camera.get("last_activity"),
                "snapshot_summary": camera.get("snapshot_summary"),
                "recognized_people": camera.get("recognized_people", []),
            }
        )

    reduced_camera_activity.sort(key=lambda item: item["camera_entity_id"])
    if len(reduced_camera_activity) > _MAX_CAMERA_ACTIVITY:
        reduced_camera_activity = reduced_camera_activity[:_MAX_CAMERA_ACTIVITY]

    return {
        "schema_version": snapshot["schema_version"],
        "generated_at": snapshot["generated_at"],
        "entities": reduced_entities,
        "camera_activity": reduced_camera_activity,
        "derived": snapshot["derived"],
    }

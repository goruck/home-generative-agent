"""Camera activity extraction for snapshots."""

from __future__ import annotations

from collections.abc import Iterable
from datetime import datetime
from typing import Any

from homeassistant.core import State
from homeassistant.util import dt as dt_util

from .schema import CameraActivity

_MOTION_KEYS = (
    "motion_detection_entity",
    "motion_entity",
    "motion_entity_id",
    "motion_sensor",
    "motion_sensors",
)
_VMD_KEYS = (
    "vmd_entity",
    "vmd_entity_id",
    "vmd_sensor",
    "vmd_sensors",
)
_ACTIVITY_KEYS = (
    "last_activity",
    "last_motion",
    "last_motion_time",
    "last_updated",
)
_SUMMARY_KEYS = ("snapshot_summary", "summary", "last_summary")


def _normalize_entity_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, Iterable):
        return [str(item) for item in value]
    return [str(value)]


def _coerce_iso(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return dt_util.as_utc(value).isoformat()
    if isinstance(value, str):
        return value
    return str(value)


def extract_camera_activity(
    camera_state: State,
    area_name: str | None,
    image_state: State | None = None,
) -> CameraActivity:
    """Build structured camera activity metadata from a camera state."""
    attrs = camera_state.attributes
    motion_entities: list[str] = []
    vmd_entities: list[str] = []

    for key in _MOTION_KEYS:
        if key in attrs:
            motion_entities.extend(_normalize_entity_list(attrs.get(key)))

    for key in _VMD_KEYS:
        if key in attrs:
            vmd_entities.extend(_normalize_entity_list(attrs.get(key)))

    last_activity: str | None = None
    for key in _ACTIVITY_KEYS:
        if key in attrs:
            last_activity = _coerce_iso(attrs.get(key))
            break

    snapshot_summary: str | None = None
    recognized_people: list[str] = []
    latest_path: str | None = None
    for key in _SUMMARY_KEYS:
        if key in attrs:
            val = attrs.get(key)
            if val is not None:
                snapshot_summary = str(val)
            break
    if image_state is not None:
        image_attrs = image_state.attributes
        if snapshot_summary is None and image_attrs.get("summary") is not None:
            snapshot_summary = str(image_attrs.get("summary"))
        if last_activity is None and image_attrs.get("last_event") is not None:
            last_activity = _coerce_iso(image_attrs.get("last_event"))
        if image_attrs.get("recognized_people") is not None:
            recognized_people = _normalize_entity_list(
                image_attrs.get("recognized_people")
            )
        if image_attrs.get("latest_path") is not None:
            latest_path = str(image_attrs.get("latest_path"))

    return {
        "camera_entity_id": camera_state.entity_id,
        "area": area_name,
        "last_activity": last_activity,
        "motion_entities": sorted(set(motion_entities)),
        "vmd_entities": sorted(set(vmd_entities)),
        "snapshot_summary": snapshot_summary,
        "recognized_people": sorted(set(recognized_people)),
        "latest_path": latest_path,
    }

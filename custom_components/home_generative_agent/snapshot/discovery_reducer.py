"""Deterministic reducer that compresses discovery snapshots for LLM prompts."""

from __future__ import annotations

from collections import defaultdict
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
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
_MAX_ENTITIES = 100
_MAX_CAMERA_ACTIVITY = 50
_MAX_SUMMARY_CHARS = 150

# States considered "interesting" for anomaly detection — scored higher.
_ANOMALOUS_STATES = {"on", "open", "unlocked", "triggered", "disarmed"}
_LOW_BATTERY_THRESHOLD = 20
_RECENT_CHANGE_MINUTES = 30
_MODERATE_CHANGE_MINUTES = 120

# Type alias for the grouping key.
_GroupKey = tuple[str, str | None, str | None, str]


def _truncate_iso(iso_str: str | None) -> str | None:
    """Truncate an ISO-8601 timestamp to minute precision."""
    if not iso_str:
        return None
    return iso_str[:16]  # "YYYY-MM-DDTHH:MM"


def _anomaly_score(entity: dict[str, Any], generated_at: str) -> int:
    """Return a relevance score for anomaly detection (higher = more interesting)."""
    score = 0
    state = entity.get("state", "")
    device_class = entity.get("device_class")

    if state.lower() in _ANOMALOUS_STATES:
        score += 100

    if device_class == "battery":
        try:
            if float(state) < _LOW_BATTERY_THRESHOLD:
                score += 80
        except (ValueError, TypeError):
            pass

    last_changed = entity.get("last_changed", "")
    if last_changed and generated_at:
        score += _recency_bonus(last_changed, generated_at)

    return score


def _recency_bonus(last_changed: str, generated_at: str) -> int:
    """Return a bonus score based on how recently the entity changed."""
    try:
        changed = datetime.fromisoformat(last_changed)
        generated = datetime.fromisoformat(generated_at)
        if changed.tzinfo is None:
            changed = changed.replace(tzinfo=UTC)
        if generated.tzinfo is None:
            generated = generated.replace(tzinfo=UTC)
        delta_minutes = (generated - changed).total_seconds() / 60
    except (ValueError, TypeError):
        return 0

    if delta_minutes < _RECENT_CHANGE_MINUTES:
        return 60
    if delta_minutes < _MODERATE_CHANGE_MINUTES:
        return 30
    return 0


def _filter_entities(snapshot: FullStateSnapshot) -> list[dict[str, Any]]:
    """Filter snapshot entities to security/safety-relevant domains."""
    filtered: list[dict[str, Any]] = []
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
        filtered.append(
            {
                "entity_id": entity["entity_id"],
                "domain": domain,
                "state": entity["state"],
                "area": entity.get("area"),
                "device_class": device_class,
                "last_changed": entity.get("last_changed"),
            }
        )
    return filtered


def _group_entities(entities: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Group entities by (domain, device_class, area, state)."""
    groups: dict[_GroupKey, list[str]] = defaultdict(list)
    last_changed_map: dict[_GroupKey, str | None] = {}

    for entity in entities:
        key: _GroupKey = (
            entity["domain"],
            entity.get("device_class"),
            entity.get("area"),
            entity["state"],
        )
        groups[key].append(entity["entity_id"])
        existing = last_changed_map.get(key)
        candidate = entity.get("last_changed")
        if candidate and (existing is None or candidate > existing):
            last_changed_map[key] = candidate

    grouped: list[dict[str, Any]] = []
    for (domain, device_class, area, state), entity_ids in sorted(
        groups.items(), key=lambda item: item[1][0]
    ):
        entry: dict[str, Any] = {
            "entity_ids": sorted(entity_ids),
            "state": state,
        }
        if area:
            entry["area"] = area
        if device_class:
            entry["device_class"] = device_class
        lc = _truncate_iso(last_changed_map.get((domain, device_class, area, state)))
        if lc:
            entry["last_changed"] = lc
        grouped.append(entry)
    return grouped


def _reduce_cameras(snapshot: FullStateSnapshot) -> list[dict[str, Any]]:
    """Reduce camera activity entries, truncating summaries."""
    reduced: list[dict[str, Any]] = []
    for camera in snapshot["camera_activity"]:
        summary = camera.get("snapshot_summary")
        if summary and len(summary) > _MAX_SUMMARY_CHARS:
            summary = summary[:_MAX_SUMMARY_CHARS] + "..."

        entry: dict[str, Any] = {
            "camera_entity_id": camera["camera_entity_id"],
        }
        if camera.get("area"):
            entry["area"] = camera["area"]
        if camera.get("last_activity"):
            entry["last_activity"] = _truncate_iso(camera["last_activity"])
        if summary:
            entry["snapshot_summary"] = summary
        people = camera.get("recognized_people", [])
        if people:
            entry["recognized_people"] = people
        reduced.append(entry)

    reduced.sort(key=lambda item: item["camera_entity_id"])
    if len(reduced) > _MAX_CAMERA_ACTIVITY:
        reduced = reduced[:_MAX_CAMERA_ACTIVITY]
    return reduced


def reduce_snapshot_for_discovery(snapshot: FullStateSnapshot) -> dict[str, Any]:
    """Return a compressed snapshot for LLM discovery prompts."""
    generated_at = snapshot.get("generated_at", "")

    # Phase 1: Filter to relevant entities
    filtered = _filter_entities(snapshot)

    # Phase 2: Prioritize interesting entities, then cap
    filtered.sort(
        key=lambda e: (-_anomaly_score(e, generated_at), e["entity_id"]),
    )
    if len(filtered) > _MAX_ENTITIES:
        filtered = filtered[:_MAX_ENTITIES]

    # Phase 3: Group entities by (domain, device_class, area, state)
    grouped_entities = _group_entities(filtered)

    # Phase 4: Reduce camera activity
    reduced_camera_activity = _reduce_cameras(snapshot)

    # Phase 5: Trim derived context timestamps
    derived: dict[str, Any] = dict(snapshot["derived"])
    motion_by_area = derived.get("last_motion_by_area")
    if isinstance(motion_by_area, dict):
        derived["last_motion_by_area"] = {
            area: _truncate_iso(str(ts))
            for area, ts in motion_by_area.items()
        }
    now_val = derived.get("now")
    if isinstance(now_val, str):
        derived["now"] = _truncate_iso(now_val)

    return {
        "entities": grouped_entities,
        "camera_activity": reduced_camera_activity,
        "derived": derived,
    }

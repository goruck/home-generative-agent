"""Deterministic semantic key utilities for discovery candidates and rules."""

from __future__ import annotations

from typing import Any


def candidate_semantic_key(  # noqa: PLR0912, PLR0915
    candidate: dict[str, Any],
) -> str | None:
    """Build a stable semantic key for a discovery candidate."""
    evidence_paths = _string_list(candidate.get("evidence_paths"))
    text = " ".join(
        [
            str(candidate.get("title", "")),
            str(candidate.get("summary", "")),
            str(candidate.get("pattern", "")),
            str(candidate.get("suggested_type", "")),
        ]
    ).lower()
    entity_ids = _extract_entity_ids(evidence_paths)
    lock_ids = sorted(
        entity_id for entity_id in entity_ids if entity_id.startswith("lock.")
    )
    window_ids = sorted(entity_id for entity_id in entity_ids if "window" in entity_id)
    door_ids = sorted(
        entity_id
        for entity_id in entity_ids
        if "door" in entity_id or "entry" in entity_id
    )
    motion_ids = sorted(
        entity_id
        for entity_id in entity_ids
        if "motion" in entity_id or "vmd" in entity_id
    )
    sensor_ids = sorted(
        entity_id
        for entity_id in entity_ids
        if entity_id.startswith(("sensor.", "binary_sensor."))
    )
    alarm_ids = sorted(
        entity_id
        for entity_id in entity_ids
        if entity_id.startswith("alarm_control_panel.")
    )

    subject = "unknown"
    entities: list[str] = []
    if window_ids:
        subject = "entry_window"
        entities = window_ids
    elif door_ids:
        subject = "entry_door"
        entities = door_ids
    elif lock_ids:
        subject = "lock"
        entities = lock_ids
    elif motion_ids:
        subject = "motion"
        entities = motion_ids
    elif alarm_ids:
        subject = "alarm"
        entities = alarm_ids
    elif sensor_ids:
        subject = "sensor"
        entities = sensor_ids

    predicate = "unknown"
    if "unlocked" in text:
        predicate = "unlocked"
    elif "open" in text:
        predicate = "open"
    elif "disarmed" in text:
        predicate = "disarmed"
    elif _contains_any(text, ("unavailable", "offline", "unreachable")):
        predicate = "unavailable"
    elif "motion" in text or "activity" in text:
        predicate = "active"
    if predicate == "unavailable" and sensor_ids:
        subject = "sensor"
        entities = sensor_ids

    night = "any"
    if "night" in text or "derived.is_night" in evidence_paths:
        night = "1"

    home = "any"
    if _contains_any(
        text, ("no one home", "nobody home", "away", "empty", "unoccupied")
    ):
        home = "0"
    elif _contains_any(text, ("someone home", "occupied", "home", "present")):
        home = "1"
    if "derived.anyone_home" in evidence_paths and home == "any":
        home = "1"
    if subject == "unknown" and _contains_any(text, ("window", "windows")):
        subject = "entry_window"
    if predicate == "unknown" and "open" in text:
        predicate = "open"

    if subject == "unknown" and predicate == "unknown":
        return None
    entities_csv = ",".join(sorted(set(entities)))
    return (
        f"v1|subject={subject}|predicate={predicate}|night={night}|"
        f"home={home}|scope=any|entities={entities_csv}"
    )


def rule_semantic_key(rule: dict[str, Any]) -> str | None:  # noqa: PLR0911
    """Build a stable semantic key for an active/generated rule."""
    template_id = str(rule.get("template_id", ""))
    params = rule.get("params", {}) or {}
    if template_id == "unlocked_lock_when_home":
        lock_id = str(params.get("lock_entity_id", ""))
        if not lock_id:
            return None
        return (
            "v1|subject=lock|predicate=unlocked|night=any|home=1|scope=any|"
            f"entities={lock_id}"
        )
    if template_id == "alarm_disarmed_open_entry":
        entry_ids = sorted(set(_string_list(params.get("entry_entity_ids"))))
        if not entry_ids:
            return None
        entry_subject = (
            "entry_window"
            if any("window" in item for item in entry_ids)
            else "entry_door"
        )
        return (
            f"v1|subject={entry_subject}|predicate=open|night=any|home=any|scope=any|"
            f"entities={','.join(entry_ids)}"
        )
    if template_id == "open_any_window_at_night_while_away":
        return (
            "v1|subject=entry_window|predicate=open|night=1|home=0|scope=any|entities="
        )
    if template_id == "motion_without_camera_activity":
        motion_ids = sorted(set(_string_list(params.get("motion_entity_ids"))))
        if not motion_ids:
            return None
        return (
            "v1|subject=motion|predicate=active|night=any|home=any|scope=any|"
            f"entities={','.join(motion_ids)}"
        )
    if template_id == "unavailable_sensors_while_home":
        sensor_ids = sorted(set(_string_list(params.get("sensor_entity_ids"))))
        if not sensor_ids:
            return None
        return (
            "v1|subject=sensor|predicate=unavailable|night=any|home=1|scope=any|"
            f"entities={','.join(sensor_ids)}"
        )
    return None


def _extract_entity_ids(evidence_paths: list[str]) -> list[str]:
    entity_ids: list[str] = []
    prefix = "entities[entity_id="
    for path in evidence_paths:
        if not path.startswith(prefix):
            continue
        entity_id = path.split(prefix, 1)[1].split("]", 1)[0]
        entity_ids.append(entity_id)
    return entity_ids


def _string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, str)]


def _contains_any(text: str, needles: tuple[str, ...]) -> bool:
    return any(needle in text for needle in needles)

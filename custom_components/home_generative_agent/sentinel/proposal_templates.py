"""Template-based proposal normalization for discovery candidates."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

SUPPORTED_TEMPLATES = {
    "alarm_disarmed_open_entry",
    "low_battery_sensors",
    "motion_while_alarm_disarmed_and_home_present",
    "unavailable_sensors",
    "unavailable_sensors_while_home",
    "open_any_window_at_night_while_away",
    "open_entry_when_home",
    "open_entry_while_away",
    "open_entry_at_night_when_home",
    "open_entry_at_night_while_away",
    "unlocked_lock_when_home",
    "motion_without_camera_activity",
}

_PERCENT_THRESHOLD_PATTERN = re.compile(r"(\d+(?:\.\d+)?)\s*%")
_RELATIVE_THRESHOLD_PATTERN = re.compile(
    r"(?:at\s+or\s+below|below|under|<=|less than)\s*(\d+(?:\.\d+)?)"
)
_MAX_PERCENT = 100.0


@dataclass(frozen=True)
class NormalizedRule:
    """Normalized rule derived from a discovery candidate."""

    rule_id: str
    template_id: str
    params: dict[str, Any]
    severity: str
    confidence: float
    is_sensitive: bool
    suggested_actions: list[str]

    def as_dict(self) -> dict[str, Any]:
        """Convert the normalized rule to a persisted mapping."""
        return {
            "rule_id": self.rule_id,
            "template_id": self.template_id,
            "params": self.params,
            "severity": self.severity,
            "confidence": self.confidence,
            "is_sensitive": self.is_sensitive,
            "suggested_actions": list(self.suggested_actions),
        }


def normalize_candidate(  # noqa: PLR0911, PLR0912
    candidate: dict[str, Any],
) -> NormalizedRule | None:
    """Map a discovery candidate to a supported template."""
    evidence_paths = candidate.get("evidence_paths", [])
    text = " ".join(
        [
            str(candidate.get("title", "")),
            str(candidate.get("summary", "")),
            str(candidate.get("pattern", "")),
            str(candidate.get("suggested_type", "")),
        ]
    ).lower()
    lock_ids = _find_entity_ids(evidence_paths, "lock")
    alarm_id = _find_entity_id(evidence_paths, "alarm_control_panel")
    entry_ids = _find_entry_entity_ids(evidence_paths)
    motion_ids = _find_motion_entity_ids(evidence_paths)
    person_ids = _find_entity_ids(evidence_paths, "person")
    sensor_ids = _find_sensor_entity_ids(evidence_paths)
    battery_sensor_ids = _find_battery_sensor_entity_ids(evidence_paths)
    camera_id = _find_camera_id(evidence_paths)
    has_night = _has_night_signal(evidence_paths, text)
    presence = _presence_signal(evidence_paths, text)
    entry_kind = _entry_kind(entry_ids)

    if (
        alarm_id
        and motion_ids
        and person_ids
        and presence == "home"
        and _contains_any(text, ("motion", "vmd"))
        and _contains_any(text, ("alarm", "disarmed"))
    ):
        default_rule_id = (
            f"motion_while_alarm_disarmed_and_home_present_{alarm_id.replace('.', '_')}"
        )
        return NormalizedRule(
            rule_id=_candidate_rule_id(
                candidate,
                default=default_rule_id,
            ),
            template_id="motion_while_alarm_disarmed_and_home_present",
            params={
                "alarm_entity_id": alarm_id,
                "motion_entity_ids": motion_ids,
                "home_entity_ids": person_ids,
            },
            severity="low",
            confidence=float(candidate.get("confidence_hint", 0.75)),
            is_sensitive=False,
            suggested_actions=["close_entry"],
        )

    if alarm_id and entry_ids and _contains_any(text, ("alarm", "disarmed", "armed")):
        return NormalizedRule(
            rule_id=f"alarm_disarmed_open_entry_{alarm_id.replace('.', '_')}",
            template_id="alarm_disarmed_open_entry",
            params={"alarm_entity_id": alarm_id, "entry_entity_ids": entry_ids},
            severity="high",
            confidence=float(candidate.get("confidence_hint", 0.6)),
            is_sensitive=True,
            suggested_actions=["close_entry"],
        )

    if lock_ids and not entry_ids and _contains_any(text, ("lock", "unlocked")):
        lock_id = lock_ids[0]
        return NormalizedRule(
            rule_id=f"unlocked_lock_when_home_{lock_id.replace('.', '_')}",
            template_id="unlocked_lock_when_home",
            params={"lock_entity_id": lock_id},
            severity="medium",
            confidence=float(candidate.get("confidence_hint", 0.5)),
            is_sensitive=True,
            suggested_actions=["lock_entity"],
        )

    if entry_ids and _contains_any(text, ("open", "window", "door", "entry")):
        if has_night and presence == "away":
            return _build_entry_rule(
                candidate,
                "open_entry_at_night_while_away",
                f"open_entry_at_night_while_away_{entry_kind}",
                entry_ids,
            )
        if has_night and presence == "home":
            return _build_entry_rule(
                candidate,
                "open_entry_at_night_when_home",
                f"open_entry_at_night_when_home_{entry_kind}",
                entry_ids,
            )
        if presence == "away":
            return _build_entry_rule(
                candidate,
                "open_entry_while_away",
                f"open_entry_while_away_{entry_kind}",
                entry_ids,
            )
        if presence == "home":
            return _build_entry_rule(
                candidate,
                "open_entry_when_home",
                f"open_entry_when_home_{entry_kind}",
                entry_ids,
            )
    if (
        not entry_ids
        and has_night
        and presence == "away"
        and _contains_any(text, ("open", "window"))
    ):
        return NormalizedRule(
            rule_id="open_any_window_at_night_while_away",
            template_id="open_any_window_at_night_while_away",
            params={"entry_selector": "window"},
            severity="high",
            confidence=float(candidate.get("confidence_hint", 0.6)),
            is_sensitive=True,
            suggested_actions=["close_entry"],
        )

    if (
        motion_ids
        and camera_id
        and _contains_any(text, ("motion", "camera", "activity"))
    ):
        return NormalizedRule(
            rule_id=f"motion_without_camera_{camera_id.replace('.', '_')}",
            template_id="motion_without_camera_activity",
            params={"motion_entity_ids": motion_ids, "camera_entity_id": camera_id},
            severity="low",
            confidence=float(candidate.get("confidence_hint", 0.5)),
            is_sensitive=False,
            suggested_actions=["check_camera"],
        )

    if battery_sensor_ids and _contains_any(text, ("battery", "low", "below")):
        return NormalizedRule(
            rule_id=_candidate_rule_id(candidate, default="low_battery_sensors"),
            template_id="low_battery_sensors",
            params={
                "sensor_entity_ids": battery_sensor_ids,
                "threshold": _extract_threshold_percent(text, default=40.0),
            },
            severity="low",
            confidence=float(candidate.get("confidence_hint", 0.62)),
            is_sensitive=False,
            suggested_actions=["check_sensor"],
        )

    if sensor_ids and _contains_any(text, ("unavailable", "offline", "unreachable")):
        if presence == "home":
            return NormalizedRule(
                rule_id="unavailable_sensors_while_home",
                template_id="unavailable_sensors_while_home",
                params={"sensor_entity_ids": sensor_ids},
                severity="low",
                confidence=float(candidate.get("confidence_hint", 0.8)),
                is_sensitive=False,
                suggested_actions=["check_sensor"],
            )
        return NormalizedRule(
            rule_id=_candidate_rule_id(candidate, default="unavailable_sensors"),
            template_id="unavailable_sensors",
            params={"sensor_entity_ids": sensor_ids},
            severity="low",
            confidence=float(candidate.get("confidence_hint", 0.6)),
            is_sensitive=False,
            suggested_actions=["check_sensor"],
        )

    return None


def _build_entry_rule(
    candidate: dict[str, Any],
    template_id: str,
    rule_id: str,
    entry_ids: list[str],
) -> NormalizedRule:
    return NormalizedRule(
        rule_id=rule_id,
        template_id=template_id,
        params={"entry_entity_ids": entry_ids},
        severity="high" if "away" in template_id else "medium",
        confidence=float(candidate.get("confidence_hint", 0.6)),
        is_sensitive=True,
        suggested_actions=["close_entry"],
    )


def _find_entity_id(evidence_paths: list[str], domain: str) -> str | None:
    ids = _find_entity_ids(evidence_paths, domain)
    if not ids:
        return None
    return ids[0]


def _find_entity_ids(evidence_paths: list[str], domain: str) -> list[str]:
    ids = [
        path.split("entities[entity_id=", 1)[1].split("]", 1)[0]
        for path in evidence_paths
        if path.startswith("entities[entity_id=") and f"{domain}." in path
    ]
    return sorted(set(ids))


def _find_entry_entity_ids(evidence_paths: list[str]) -> list[str]:
    ids: list[str] = []
    for path in evidence_paths:
        if not path.startswith("entities[entity_id="):
            continue
        entity_id = path.split("entities[entity_id=", 1)[1].split("]", 1)[0]
        if "." not in entity_id:
            continue
        domain = entity_id.split(".", 1)[0]
        if domain not in {"binary_sensor", "sensor", "cover"}:
            continue
        if any(key in entity_id for key in ("door", "window", "entry")):
            ids.append(entity_id)
    return sorted(set(ids))


def _find_motion_entity_ids(evidence_paths: list[str]) -> list[str]:
    ids: list[str] = []
    for path in evidence_paths:
        if not path.startswith("entities[entity_id="):
            continue
        entity_id = path.split("entities[entity_id=", 1)[1].split("]", 1)[0]
        if "motion" in entity_id or "vmd" in entity_id:
            ids.append(entity_id)
    return sorted(set(ids))


def _find_sensor_entity_ids(evidence_paths: list[str]) -> list[str]:
    ids: list[str] = []
    for path in evidence_paths:
        if not path.startswith("entities[entity_id="):
            continue
        entity_id = path.split("entities[entity_id=", 1)[1].split("]", 1)[0]
        if "." not in entity_id:
            # Legacy discovery drafts may store object IDs without domain.
            ids.append(entity_id)
            continue
        domain = entity_id.split(".", 1)[0]
        if domain in {"sensor", "binary_sensor"}:
            ids.append(entity_id)
    return sorted(set(ids))


def _find_battery_sensor_entity_ids(evidence_paths: list[str]) -> list[str]:
    ids: list[str] = []
    for path in evidence_paths:
        if not path.startswith("entities[entity_id="):
            continue
        entity_id = path.split("entities[entity_id=", 1)[1].split("]", 1)[0]
        if "battery" not in entity_id.lower():
            continue
        if "." not in entity_id:
            ids.append(entity_id)
            continue
        domain = entity_id.split(".", 1)[0]
        if domain in {"sensor", "binary_sensor"}:
            ids.append(entity_id)
    return sorted(set(ids))


def _extract_threshold_percent(text: str, *, default: float) -> float:
    for pattern in (_PERCENT_THRESHOLD_PATTERN, _RELATIVE_THRESHOLD_PATTERN):
        match = pattern.search(text)
        if not match:
            continue
        try:
            value = float(match.group(1))
        except ValueError:
            continue
        if 0 <= value <= _MAX_PERCENT:
            return value
    return default


def _has_night_signal(evidence_paths: list[str], text: str) -> bool:
    if "derived.is_night" in evidence_paths:
        return True
    return _contains_any(text, ("night", "nighttime", "overnight"))


def _presence_signal(evidence_paths: list[str], text: str) -> str:
    if _contains_any(
        text, ("away", "no one home", "nobody home", "empty", "unoccupied")
    ):
        return "away"
    if _contains_any(text, ("someone home", "occupied", "home", "present")):
        return "home"
    if "derived.anyone_home" in evidence_paths:
        return "home"
    return "any"


def _entry_kind(entry_ids: list[str]) -> str:
    if any("window" in entity_id for entity_id in entry_ids):
        return "window"
    if any("door" in entity_id for entity_id in entry_ids):
        return "door"
    return "entry"


def _find_camera_id(evidence_paths: list[str]) -> str | None:
    for path in evidence_paths:
        if path.startswith("camera_activity[entity_id="):
            return path.split("camera_activity[entity_id=", 1)[1].split("]", 1)[0]
        if path.startswith("camera_activity[camera_entity_id="):
            return path.split("camera_activity[camera_entity_id=", 1)[1].split("]", 1)[
                0
            ]
    return None


def _contains_any(text: str, words: tuple[str, ...]) -> bool:
    return any(word in text for word in words)


def _candidate_rule_id(candidate: dict[str, Any], *, default: str) -> str:
    candidate_id = candidate.get("candidate_id")
    if not isinstance(candidate_id, str):
        return default
    slug = "".join(
        char if (char.isalnum() or char == "_") else "_" for char in candidate_id
    ).strip("_")
    return slug.lower() or default

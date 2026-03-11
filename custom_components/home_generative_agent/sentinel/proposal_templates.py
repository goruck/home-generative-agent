"""Template-based proposal normalization for discovery candidates."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

SUPPORTED_TEMPLATES = {
    "alarm_disarmed_open_entry",
    "low_battery_sensors",
    "motion_detected_at_night_while_alarm_disarmed",
    "motion_while_alarm_disarmed_and_home_present",
    "unavailable_sensors",
    "unavailable_sensors_while_home",
    "open_any_window_at_night_while_away",
    "open_entry_when_home",
    "open_entry_while_away",
    "open_entry_at_night_when_home",
    "open_entry_at_night_while_away",
    "unlocked_lock_when_home",
    "unlocked_lock_while_away",
    "motion_without_camera_activity",
    "unknown_person_camera_no_home",
    "unknown_person_camera_when_home",
    # Issue #265 — baseline-driven detectors
    "baseline_deviation",
    "time_of_day_anomaly",
    # Issue #266 — lambda/expression rules
    "lambda",
    # Flexible templates for common patterns
    "alarm_state_mismatch",
    "entity_state_duration",
    "sensor_threshold_condition",
    "entity_staleness",
    "multiple_entries_open_count",
}

_PERCENT_THRESHOLD_PATTERN = re.compile(r"(\d+(?:\.\d+)?)\s*%")
_RELATIVE_THRESHOLD_PATTERN = re.compile(
    r"(?:at\s+or\s+below|below|under|<=|less than)\s*(\d+(?:\.\d+)?)"
)
_HOURS_THRESHOLD_PATTERN = re.compile(
    r"(\d+(?:\.\d+)?)\s*(?:hour|hr)s?",
    re.IGNORECASE,
)
_NUMERIC_THRESHOLD_PATTERN = re.compile(
    r"(?:>|above|exceeds?|over|more than|greater than)\s*(\d+(?:\.\d+)?)"
)
_MAX_PERCENT = 100.0
_DEFAULT_DURATION_HOURS = 2.0
_DEFAULT_STALE_HOURS = 24.0
_DURATION_TERMS = (
    "duration",
    "extended",
    "prolonged",
    "for",
    "since",
    "hours",
    "long",
)
_STALENESS_TERMS = (
    "stale",
    "not updated",
    "tracking",
    "staleness",
    "last seen",
    "last updated",
    "gps",
)
_MULTIPLE_TERMS = (
    "multiple",
    "simultaneous",
    "several",
    "more than",
    "at once",
    "at the same time",
)
_POWER_ENERGY_TERMS = (
    "power",
    "energy",
    "watt",
    "consumption",
    "usage",
    "kilowatt",
)
_ALARM_STATES = ("armed_home", "armed_away", "armed_night", "disarmed", "triggered")
_UNKNOWN_TERMS = (
    "unknown",
    "unrecognized",
    "stranger",
    "unidentified",
    "indeterminate",
)
_PERSON_TERMS = (
    "person",
    "people",
    "face",
    "occupant",
    "occupants",
    "resident",
    "residents",
)
_CAMERA_TERMS = ("camera", "cam")
_MIN_CAMERA_TOKEN_LEN = 3
_AWAY_TERMS = (
    "away",
    "no one home",
    "nobody home",
    "empty",
    "unoccupied",
    "no occupants",
    "without occupants",
)
_HOME_TERMS = (
    "someone home",
    "occupied",
    "home",
    "present",
    "occupants",
    "residents",
)


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


@dataclass(frozen=True)
class NormalizationResult:
    """Structured normalization result for a discovery candidate."""

    normalized: NormalizedRule | None
    reason_code: str | None = None
    details: dict[str, Any] | None = None


def normalize_candidate(candidate: dict[str, Any]) -> NormalizedRule | None:
    """Map a discovery candidate to a supported template."""
    return explain_normalize_candidate(candidate).normalized


def explain_normalize_candidate(  # noqa: PLR0911, PLR0912
    candidate: dict[str, Any],
) -> NormalizationResult:
    """Map a discovery candidate to a supported template with failure reasons."""
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
    camera_id = _find_camera_id(evidence_paths, candidate)
    has_night = _has_night_signal(evidence_paths, text)
    presence = _presence_signal(evidence_paths, text)
    entry_kind = _entry_kind(entry_ids)
    has_unknown_person_signal = _contains_any(text, _UNKNOWN_TERMS) and _contains_any(
        text, _PERSON_TERMS
    )
    has_camera_signal = camera_id is not None or _contains_any(text, _CAMERA_TERMS)
    summary = {
        "alarm_id": alarm_id,
        "lock_ids": lock_ids,
        "entry_ids": entry_ids,
        "motion_ids": motion_ids,
        "person_ids": person_ids,
        "sensor_ids": sensor_ids,
        "battery_sensor_ids": battery_sensor_ids,
        "camera_id": camera_id,
        "presence": presence,
        "has_night": has_night,
    }

    if (
        alarm_id
        and motion_ids
        and has_night
        and _contains_any(text, ("motion", "vmd"))
        and _contains_any(text, ("alarm", "disarmed"))
    ):
        default_rule_id = "motion_detected_at_night_while_alarm_disarmed"
        return NormalizationResult(
            normalized=NormalizedRule(
                rule_id=_candidate_rule_id(candidate, default=default_rule_id),
                template_id="motion_detected_at_night_while_alarm_disarmed",
                params={
                    "alarm_entity_id": alarm_id,
                    "motion_entity_ids": motion_ids,
                    "required_entity_ids": person_ids,
                },
                severity="low",
                confidence=float(candidate.get("confidence_hint", 0.8)),
                is_sensitive=False,
                suggested_actions=["close_entry"],
            )
        )

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
        return NormalizationResult(
            normalized=NormalizedRule(
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
        )

    if alarm_id and entry_ids and _contains_any(text, ("alarm", "disarmed", "armed")):
        return NormalizationResult(
            normalized=NormalizedRule(
                rule_id=f"alarm_disarmed_open_entry_{alarm_id.replace('.', '_')}",
                template_id="alarm_disarmed_open_entry",
                params={"alarm_entity_id": alarm_id, "entry_entity_ids": entry_ids},
                severity="high",
                confidence=float(candidate.get("confidence_hint", 0.6)),
                is_sensitive=True,
                suggested_actions=_entry_suggested_actions(entry_ids),
            )
        )

    # alarm_state_mismatch: alarm in a specific state that contradicts occupancy.
    # Must follow alarm+motion and alarm+entry branches above.
    if (
        alarm_id
        and not motion_ids
        and not entry_ids
        and _contains_any(text, ("armed_home", "armed_away", "armed_night"))
        and presence in ("away", "home")
    ):
        detected_state = _extract_alarm_state(text) or "armed_home"
        default_rule_id = (
            f"alarm_state_mismatch_{detected_state}_{presence}_{alarm_id.replace('.', '_')}"
        )
        return NormalizationResult(
            normalized=NormalizedRule(
                rule_id=_candidate_rule_id(candidate, default=default_rule_id),
                template_id="alarm_state_mismatch",
                params={
                    "alarm_entity_id": alarm_id,
                    "alarm_state": detected_state,
                    "expected_presence": presence,
                },
                severity="low",
                confidence=float(candidate.get("confidence_hint", 0.85)),
                is_sensitive=False,
                suggested_actions=["alarm_control_panel.alarm_disarm"],
            )
        )

    # entity_state_duration: lock unlocked for too long.
    if (
        lock_ids
        and not entry_ids
        and _has_duration_signal(text)
        and _contains_any(text, ("lock", "unlocked"))
    ):
        lock_id = lock_ids[0]
        threshold_hours = _extract_threshold_hours(text)
        default_rule_id = f"lock_unlocked_duration_{lock_id.replace('.', '_')}"
        return NormalizationResult(
            normalized=NormalizedRule(
                rule_id=_candidate_rule_id(candidate, default=default_rule_id),
                template_id="entity_state_duration",
                params={
                    "entity_id": lock_id,
                    "target_state": "unlocked",
                    "threshold_hours": threshold_hours,
                },
                severity="medium",
                confidence=float(candidate.get("confidence_hint", 0.75)),
                is_sensitive=True,
                suggested_actions=["lock.lock", "lock_entity"],
            )
        )

    # entity_state_duration: entry sensor open for too long.
    if (
        entry_ids
        and _has_duration_signal(text)
        and _contains_any(text, ("open", "window", "door", "entry"))
    ):
        entry_id = entry_ids[0]
        threshold_hours = _extract_threshold_hours(text)
        default_rule_id = f"entry_open_duration_{entry_id.replace('.', '_')}"
        return NormalizationResult(
            normalized=NormalizedRule(
                rule_id=_candidate_rule_id(candidate, default=default_rule_id),
                template_id="entity_state_duration",
                params={
                    "entity_id": entry_id,
                    "target_state": "on",
                    "threshold_hours": threshold_hours,
                },
                severity="medium",
                confidence=float(candidate.get("confidence_hint", 0.7)),
                is_sensitive=False,
                suggested_actions=["close_entry"],
            )
        )

    # unlocked_lock_while_away: lock unlocked when nobody is home.
    if lock_ids and not entry_ids and presence == "away" and _contains_any(text, ("lock", "unlocked")):
        lock_id = lock_ids[0]
        return NormalizationResult(
            normalized=NormalizedRule(
                rule_id=f"unlocked_lock_while_away_{lock_id.replace('.', '_')}",
                template_id="unlocked_lock_while_away",
                params={"lock_entity_id": lock_id},
                severity="high",
                confidence=float(candidate.get("confidence_hint", 0.85)),
                is_sensitive=True,
                suggested_actions=["lock.lock", "lock_entity"],
            )
        )

    if lock_ids and not entry_ids and _contains_any(text, ("lock", "unlocked")):
        lock_id = lock_ids[0]
        return NormalizationResult(
            normalized=NormalizedRule(
                rule_id=f"unlocked_lock_when_home_{lock_id.replace('.', '_')}",
                template_id="unlocked_lock_when_home",
                params={"lock_entity_id": lock_id},
                severity="medium",
                confidence=float(candidate.get("confidence_hint", 0.5)),
                is_sensitive=True,
                suggested_actions=["lock.lock", "lock_entity"],
            )
        )

    # multiple_entries_open_count: several entries open simultaneously.
    # Must precede the per-entry open branches below.
    if (
        len(entry_ids) >= 2
        and _has_multiple_signal(text)
        and _contains_any(text, ("open", "window", "door", "entry"))
    ):
        require_home = presence == "home"
        require_away = presence == "away"
        default_rule_id = "multiple_entries_open_count"
        return NormalizationResult(
            normalized=NormalizedRule(
                rule_id=_candidate_rule_id(candidate, default=default_rule_id),
                template_id="multiple_entries_open_count",
                params={
                    "entry_entity_ids": entry_ids,
                    "min_count": 2,
                    "require_home": require_home,
                    "require_away": require_away,
                },
                severity="high",
                confidence=float(candidate.get("confidence_hint", 0.75)),
                is_sensitive=True,
                suggested_actions=_entry_suggested_actions(entry_ids),
            )
        )

    if entry_ids and _contains_any(text, ("open", "window", "door", "entry")):
        if has_night and presence == "away":
            return NormalizationResult(
                normalized=_build_entry_rule(
                    candidate,
                    "open_entry_at_night_while_away",
                    f"open_entry_at_night_while_away_{entry_kind}",
                    entry_ids,
                )
            )
        if has_night and presence == "home":
            return NormalizationResult(
                normalized=_build_entry_rule(
                    candidate,
                    "open_entry_at_night_when_home",
                    f"open_entry_at_night_when_home_{entry_kind}",
                    entry_ids,
                )
            )
        if presence == "away":
            return NormalizationResult(
                normalized=_build_entry_rule(
                    candidate,
                    "open_entry_while_away",
                    f"open_entry_while_away_{entry_kind}",
                    entry_ids,
                )
            )
        if presence == "home":
            return NormalizationResult(
                normalized=_build_entry_rule(
                    candidate,
                    "open_entry_when_home",
                    f"open_entry_when_home_{entry_kind}",
                    entry_ids,
                )
            )
    if (
        not entry_ids
        and has_night
        and presence == "away"
        and _contains_any(text, ("open", "window"))
    ):
        return NormalizationResult(
            normalized=NormalizedRule(
                rule_id="open_any_window_at_night_while_away",
                template_id="open_any_window_at_night_while_away",
                params={"entry_selector": "window"},
                severity="high",
                confidence=float(candidate.get("confidence_hint", 0.6)),
                is_sensitive=True,
                suggested_actions=["close_entry"],
            )
        )

    if has_camera_signal and presence == "away" and has_unknown_person_signal:
        if camera_id:
            rule_id = f"unknown_person_camera_no_home_{camera_id.replace('.', '_')}"
            params = {"camera_entity_id": camera_id}
        else:
            rule_id = "unknown_person_camera_no_home_any_camera"
            params = {"camera_selector": "any"}
        return NormalizationResult(
            normalized=NormalizedRule(
                rule_id=rule_id,
                template_id="unknown_person_camera_no_home",
                params=params,
                severity="low",
                confidence=float(candidate.get("confidence_hint", 0.85)),
                is_sensitive=True,
                suggested_actions=["close_entry"],
            )
        )
    if has_camera_signal and presence == "home" and has_unknown_person_signal:
        if camera_id:
            default_rule_id = (
                f"unknown_person_camera_when_home_{camera_id.replace('.', '_')}"
            )
            params = {"camera_entity_id": camera_id}
        else:
            default_rule_id = "unknown_person_camera_when_home_any_camera"
            params = {"camera_selector": "any"}
        return NormalizationResult(
            normalized=NormalizedRule(
                rule_id=default_rule_id,
                template_id="unknown_person_camera_when_home",
                params=params,
                severity="low",
                confidence=float(candidate.get("confidence_hint", 0.7)),
                is_sensitive=False,
                suggested_actions=["close_entry"],
            )
        )

    if (
        motion_ids
        and camera_id
        and _contains_any(text, ("motion", "camera", "activity"))
    ):
        return NormalizationResult(
            normalized=NormalizedRule(
                rule_id=f"motion_without_camera_{camera_id.replace('.', '_')}",
                template_id="motion_without_camera_activity",
                params={
                    "motion_entity_ids": motion_ids,
                    "camera_entity_id": camera_id,
                },
                severity="low",
                confidence=float(candidate.get("confidence_hint", 0.5)),
                is_sensitive=False,
                suggested_actions=["check_camera"],
            )
        )

    if battery_sensor_ids and _contains_any(text, ("battery", "low", "below")):
        return NormalizationResult(
            normalized=NormalizedRule(
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
        )

    # sensor_threshold_condition: numeric sensor exceeds a threshold, with optional
    # night/away/home condition. Excludes battery sensors (handled above).
    non_battery_sensor_ids = [s for s in sensor_ids if s not in battery_sensor_ids]
    if (
        non_battery_sensor_ids
        and _has_power_energy_signal(text)
    ):
        threshold = _extract_threshold_numeric(text)
        if threshold is not None:
            sensor_id = non_battery_sensor_ids[0]
            default_rule_id = f"sensor_threshold_{sensor_id.replace('.', '_')}"
            return NormalizationResult(
                normalized=NormalizedRule(
                    rule_id=_candidate_rule_id(candidate, default=default_rule_id),
                    template_id="sensor_threshold_condition",
                    params={
                        "sensor_entity_id": sensor_id,
                        "threshold": threshold,
                        "require_night": has_night,
                        "require_away": presence == "away",
                        "require_home": presence == "home",
                    },
                    severity="low",
                    confidence=float(candidate.get("confidence_hint", 0.7)),
                    is_sensitive=False,
                    suggested_actions=["check_appliance"],
                )
            )

    if sensor_ids and _contains_any(text, ("unavailable", "offline", "unreachable")):
        if presence == "home":
            return NormalizationResult(
                normalized=NormalizedRule(
                    rule_id="unavailable_sensors_while_home",
                    template_id="unavailable_sensors_while_home",
                    params={"sensor_entity_ids": sensor_ids},
                    severity="low",
                    confidence=float(candidate.get("confidence_hint", 0.8)),
                    is_sensitive=False,
                    suggested_actions=["check_sensor"],
                )
            )
        return NormalizationResult(
            normalized=NormalizedRule(
                rule_id=_candidate_rule_id(candidate, default="unavailable_sensors"),
                template_id="unavailable_sensors",
                params={"sensor_entity_ids": sensor_ids},
                severity="low",
                confidence=float(candidate.get("confidence_hint", 0.6)),
                is_sensitive=False,
                suggested_actions=["check_sensor"],
            )
        )

    # entity_staleness: entity last_changed has not advanced past a threshold.
    # Matches person tracking (person_ids) or explicit sensor staleness signals.
    if (person_ids or sensor_ids) and _has_staleness_signal(text):
        entity_id = (person_ids or sensor_ids)[0]
        max_stale_hours = _extract_threshold_hours(text, default=_DEFAULT_STALE_HOURS)
        default_rule_id = f"entity_staleness_{entity_id.replace('.', '_')}"
        return NormalizationResult(
            normalized=NormalizedRule(
                rule_id=_candidate_rule_id(candidate, default=default_rule_id),
                template_id="entity_staleness",
                params={
                    "entity_id": entity_id,
                    "max_stale_hours": max_stale_hours,
                },
                severity="low",
                confidence=float(candidate.get("confidence_hint", 0.7)),
                is_sensitive=False,
                suggested_actions=["check_sensor"],
            )
        )

    return _normalization_failure(
        text=text,
        summary=summary,
        candidate=candidate,
        alarm_id=alarm_id,
        lock_ids=lock_ids,
        entry_ids=entry_ids,
        motion_ids=motion_ids,
        person_ids=person_ids,
        sensor_ids=sensor_ids,
        battery_sensor_ids=battery_sensor_ids,
        camera_id=camera_id,
        presence=presence,
        has_night=has_night,
    )


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
        suggested_actions=_entry_suggested_actions(entry_ids),
    )


def _entry_suggested_actions(entry_ids: list[str]) -> list[str]:
    if entry_ids and all(entity_id.startswith("cover.") for entity_id in entry_ids):
        return ["cover.close_cover", "close_entry"]
    return ["close_entry"]


def _normalization_failure(  # noqa: PLR0911, PLR0913
    *,
    text: str,
    summary: dict[str, Any],
    candidate: dict[str, Any],
    alarm_id: str | None,
    lock_ids: list[str],
    entry_ids: list[str],
    motion_ids: list[str],
    person_ids: list[str],
    sensor_ids: list[str],
    battery_sensor_ids: list[str],
    camera_id: str | None,
    presence: str,
    has_night: bool,
) -> NormalizationResult:
    if _contains_any(text, ("alarm", "disarmed", "armed")) and not alarm_id:
        return NormalizationResult(
            normalized=None,
            reason_code="missing_required_entities",
            details={"required": ["alarm_control_panel"], **summary},
        )
    if _contains_any(text, ("lock", "unlocked")) and not lock_ids:
        return NormalizationResult(
            normalized=None,
            reason_code="missing_required_entities",
            details={"required": ["lock"], **summary},
        )
    if _contains_any(text, ("open", "window", "door", "entry")) and not entry_ids:
        return NormalizationResult(
            normalized=None,
            reason_code="missing_required_entities",
            details={"required": ["entry"], **summary},
        )
    if _contains_any(text, ("motion", "vmd")) and not motion_ids:
        return NormalizationResult(
            normalized=None,
            reason_code="missing_required_entities",
            details={"required": ["motion"], **summary},
        )
    if _contains_any(text, ("battery", "low", "below")) and not battery_sensor_ids:
        return NormalizationResult(
            normalized=None,
            reason_code="missing_required_entities",
            details={"required": ["battery_sensor"], **summary},
        )
    if (
        _contains_any(text, ("unavailable", "offline", "unreachable"))
        and not sensor_ids
    ):
        return NormalizationResult(
            normalized=None,
            reason_code="missing_required_entities",
            details={"required": ["sensor"], **summary},
        )
    if (
        _contains_any(text, _CAMERA_TERMS)
        and camera_id is None
        and "camera" in text
        and _contains_any(text, _UNKNOWN_TERMS)
        and _contains_any(text, _PERSON_TERMS)
    ):
        return NormalizationResult(
            normalized=None,
            reason_code="missing_required_entities",
            details={"required": ["camera"], **summary},
        )
    if (
        any(
            (
                alarm_id,
                lock_ids,
                entry_ids,
                motion_ids,
                person_ids,
                sensor_ids,
                battery_sensor_ids,
                camera_id,
            )
        )
        or has_night
        or presence != "any"
    ):
        return NormalizationResult(
            normalized=None,
            reason_code="unsupported_pattern",
            details={"candidate_id": candidate.get("candidate_id"), **summary},
        )
    return NormalizationResult(
        normalized=None,
        reason_code="no_matching_entity_types",
        details={"candidate_id": candidate.get("candidate_id"), **summary},
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
            # Object ID without domain — check for entry keywords and accept
            # as-is (same tolerant pattern used by _find_sensor_entity_ids).
            if any(key in entity_id for key in ("door", "window", "entry")):
                ids.append(entity_id)
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
    if _contains_any(text, _AWAY_TERMS):
        return "away"
    if _contains_any(text, _HOME_TERMS):
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


def _find_camera_id(  # noqa: PLR0911
    evidence_paths: list[str], candidate: dict[str, Any]
) -> str | None:
    for path in evidence_paths:
        if path.startswith("camera_activity[entity_id="):
            return path.split("camera_activity[entity_id=", 1)[1].split("]", 1)[0]
        if path.startswith("camera_activity[camera_entity_id="):
            return path.split("camera_activity[camera_entity_id=", 1)[1].split("]", 1)[
                0
            ]
        if path.startswith("entities[entity_id=camera."):
            return path.split("entities[entity_id=", 1)[1].split("]", 1)[0]
    candidate_id = candidate.get("candidate_id")
    if not isinstance(candidate_id, str):
        return None
    normalized = re.sub(r"[^a-z0-9_]+", "_", candidate_id.lower())
    tokens = [token for token in normalized.split("_") if token]
    try:
        camera_idx = tokens.index("camera")
    except ValueError:
        return None
    suffix = tokens[camera_idx + 1 :]
    stopwords = {
        "home",
        "away",
        "while",
        "when",
        "day",
        "night",
        "during",
        "outside",
        "inside",
        "unknown",
        "person",
        "people",
        "and",
        "motion",
    }
    object_candidates = [
        token
        for token in suffix
        if token and token not in stopwords and len(token) >= _MIN_CAMERA_TOKEN_LEN
    ]
    if not object_candidates:
        return None
    return f"camera.{object_candidates[-1]}"
    return None


def _extract_threshold_hours(text: str, *, default: float = _DEFAULT_DURATION_HOURS) -> float:
    match = _HOURS_THRESHOLD_PATTERN.search(text)
    if match:
        try:
            return float(match.group(1))
        except ValueError:
            pass
    return default


def _extract_threshold_numeric(text: str) -> float | None:
    match = _NUMERIC_THRESHOLD_PATTERN.search(text)
    if not match:
        return None
    try:
        return float(match.group(1))
    except ValueError:
        return None


def _extract_alarm_state(text: str) -> str | None:
    for state in _ALARM_STATES:
        if state in text:
            return state
    return None


def _has_duration_signal(text: str) -> bool:
    return _contains_any(text, _DURATION_TERMS)


def _has_staleness_signal(text: str) -> bool:
    return _contains_any(text, _STALENESS_TERMS)


def _has_multiple_signal(text: str) -> bool:
    return _contains_any(text, _MULTIPLE_TERMS)


def _has_power_energy_signal(text: str) -> bool:
    return _contains_any(text, _POWER_ENERGY_TERMS)


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

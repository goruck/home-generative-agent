"""Deterministic evaluation for generated rules."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, cast

from .models import AnomalyFinding, Severity, build_anomaly_id
from .proposal_templates import SUPPORTED_TEMPLATES

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping

    from custom_components.home_generative_agent.snapshot.schema import (
        CameraActivity,
        FullStateSnapshot,
        SnapshotEntity,
    )

LOGGER = logging.getLogger(__name__)

_SEVERITIES = {"low", "medium", "high"}


def evaluate_dynamic_rules(
    snapshot: FullStateSnapshot, rules: Iterable[dict[str, Any]]
) -> list[AnomalyFinding]:
    """Evaluate generated rules deterministically."""
    entity_map = {entity["entity_id"]: entity for entity in snapshot["entities"]}
    camera_map = {
        activity["camera_entity_id"]: activity
        for activity in snapshot["camera_activity"]
    }

    findings: list[AnomalyFinding] = []
    for rule in rules:
        template_id = str(rule.get("template_id", ""))
        if template_id not in SUPPORTED_TEMPLATES:
            LOGGER.debug("Skipping unsupported dynamic template: %s", template_id)
            continue
        if template_id == "alarm_disarmed_open_entry":
            findings.extend(_eval_alarm_disarmed_open_entry(snapshot, rule, entity_map))
        elif template_id == "open_any_window_at_night_while_away":
            findings.extend(
                _eval_open_any_window_at_night_while_away(snapshot, rule, entity_map)
            )
        elif template_id == "unlocked_lock_when_home":
            findings.extend(_eval_unlocked_lock_when_home(snapshot, rule, entity_map))
        elif template_id == "open_entry_when_home":
            findings.extend(
                _eval_open_entry_with_context(
                    snapshot,
                    rule,
                    entity_map,
                    require_home=True,
                    require_away=False,
                    require_night=False,
                )
            )
        elif template_id == "open_entry_while_away":
            findings.extend(
                _eval_open_entry_with_context(
                    snapshot,
                    rule,
                    entity_map,
                    require_home=False,
                    require_away=True,
                    require_night=False,
                )
            )
        elif template_id == "open_entry_at_night_when_home":
            findings.extend(
                _eval_open_entry_with_context(
                    snapshot,
                    rule,
                    entity_map,
                    require_home=True,
                    require_away=False,
                    require_night=True,
                )
            )
        elif template_id == "open_entry_at_night_while_away":
            findings.extend(
                _eval_open_entry_with_context(
                    snapshot,
                    rule,
                    entity_map,
                    require_home=False,
                    require_away=True,
                    require_night=True,
                )
            )
        elif template_id == "motion_without_camera_activity":
            findings.extend(
                _eval_motion_without_camera_activity(
                    snapshot, rule, entity_map, camera_map
                )
            )
        elif template_id == "unavailable_sensors_while_home":
            findings.extend(
                _eval_unavailable_sensors_while_home(snapshot, rule, entity_map)
            )
    return findings


def _eval_alarm_disarmed_open_entry(
    _snapshot: FullStateSnapshot,
    rule: dict[str, Any],
    entity_map: Mapping[str, SnapshotEntity],
) -> list[AnomalyFinding]:
    params = _rule_params(rule)
    alarm_id = params.get("alarm_entity_id")
    entry_ids = params.get("entry_entity_ids")
    if not alarm_id or not isinstance(entry_ids, list):
        return []
    alarm = entity_map.get(alarm_id)
    if not alarm or alarm.get("state") != "disarmed":
        return []

    findings: list[AnomalyFinding] = []
    for entry_id in entry_ids:
        entry = entity_map.get(entry_id)
        if not entry or entry.get("state") != "on":
            continue
        evidence = {
            "rule_id": rule.get("rule_id"),
            "template_id": rule.get("template_id"),
            "alarm_entity_id": alarm_id,
            "entry_entity_id": entry_id,
            "entry_state": entry.get("state"),
            "alarm_state": alarm.get("state"),
            "last_changed": entry.get("last_changed"),
        }
        findings.append(_build_finding(rule, [alarm_id, entry_id], evidence))
    return findings


def _eval_unlocked_lock_when_home(
    snapshot: FullStateSnapshot,
    rule: dict[str, Any],
    entity_map: Mapping[str, SnapshotEntity],
) -> list[AnomalyFinding]:
    if not snapshot["derived"]["anyone_home"]:
        return []
    params = _rule_params(rule)
    lock_id = params.get("lock_entity_id")
    if not lock_id:
        return []
    lock = entity_map.get(lock_id)
    if not lock or lock.get("state") != "unlocked":
        return []
    evidence = {
        "rule_id": rule.get("rule_id"),
        "template_id": rule.get("template_id"),
        "lock_entity_id": lock_id,
        "lock_state": lock.get("state"),
        "anyone_home": snapshot["derived"]["anyone_home"],
        "last_changed": lock.get("last_changed"),
    }
    return [_build_finding(rule, [lock_id], evidence)]


def _eval_motion_without_camera_activity(
    _snapshot: FullStateSnapshot,
    rule: dict[str, Any],
    entity_map: Mapping[str, SnapshotEntity],
    camera_map: Mapping[str, CameraActivity],
) -> list[AnomalyFinding]:
    params = _rule_params(rule)
    motion_ids = params.get("motion_entity_ids")
    camera_id = params.get("camera_entity_id")
    if not camera_id or not isinstance(motion_ids, list):
        return []
    camera = camera_map.get(camera_id)
    if not camera or camera.get("last_activity"):
        return []

    findings: list[AnomalyFinding] = []
    for motion_id in motion_ids:
        motion = entity_map.get(motion_id)
        if not motion or motion.get("state") != "on":
            continue
        evidence = {
            "rule_id": rule.get("rule_id"),
            "template_id": rule.get("template_id"),
            "motion_entity_id": motion_id,
            "motion_state": motion.get("state"),
            "camera_entity_id": camera_id,
            "camera_last_activity": camera.get("last_activity"),
            "last_changed": motion.get("last_changed"),
        }
        findings.append(_build_finding(rule, [motion_id, camera_id], evidence))
    return findings


def _eval_unavailable_sensors_while_home(
    snapshot: FullStateSnapshot,
    rule: dict[str, Any],
    entity_map: Mapping[str, SnapshotEntity],
) -> list[AnomalyFinding]:
    if not snapshot["derived"]["anyone_home"]:
        return []
    params = _rule_params(rule)
    sensor_ids = params.get("sensor_entity_ids")
    if not isinstance(sensor_ids, list):
        return []

    required_entities: list[SnapshotEntity] = []
    for sensor_id in sensor_ids:
        entity = entity_map.get(sensor_id)
        if entity is None:
            return []
        required_entities.append(entity)

    findings: list[AnomalyFinding] = []
    for sensor_id, sensor in zip(sensor_ids, required_entities, strict=False):
        if sensor.get("state") != "unavailable":
            continue
        evidence = {
            "rule_id": rule.get("rule_id"),
            "template_id": rule.get("template_id"),
            "sensor_entity_id": sensor_id,
            "sensor_state": sensor.get("state"),
            "anyone_home": snapshot["derived"]["anyone_home"],
            "last_changed": sensor.get("last_changed"),
        }
        findings.append(_build_finding(rule, [sensor_id], evidence))
    return findings


def _eval_open_entry_with_context(  # noqa: PLR0913
    snapshot: FullStateSnapshot,
    rule: dict[str, Any],
    entity_map: Mapping[str, SnapshotEntity],
    *,
    require_home: bool,
    require_away: bool,
    require_night: bool,
) -> list[AnomalyFinding]:
    anyone_home = bool(snapshot["derived"]["anyone_home"])
    is_night = bool(snapshot["derived"]["is_night"])
    if require_home and not anyone_home:
        return []
    if require_away and anyone_home:
        return []
    if require_night and not is_night:
        return []

    params = _rule_params(rule)
    entry_ids = params.get("entry_entity_ids")
    if not isinstance(entry_ids, list):
        return []

    findings: list[AnomalyFinding] = []
    for entry_id in entry_ids:
        entry = entity_map.get(entry_id)
        if not entry or entry.get("state") != "on":
            continue
        evidence = {
            "rule_id": rule.get("rule_id"),
            "template_id": rule.get("template_id"),
            "entry_entity_id": entry_id,
            "entry_state": entry.get("state"),
            "anyone_home": anyone_home,
            "is_night": is_night,
            "last_changed": entry.get("last_changed"),
        }
        findings.append(_build_finding(rule, [entry_id], evidence))
    return findings


def _eval_open_any_window_at_night_while_away(
    snapshot: FullStateSnapshot,
    rule: dict[str, Any],
    entity_map: Mapping[str, SnapshotEntity],
) -> list[AnomalyFinding]:
    if snapshot["derived"]["anyone_home"]:
        return []
    if not snapshot["derived"]["is_night"]:
        return []
    window_ids = _find_open_window_entity_ids(entity_map)
    findings: list[AnomalyFinding] = []
    for window_id in window_ids:
        window = entity_map.get(window_id)
        if window is None:
            continue
        evidence = {
            "rule_id": rule.get("rule_id"),
            "template_id": rule.get("template_id"),
            "entry_selector": "window",
            "entry_entity_id": window_id,
            "entry_state": window.get("state"),
            "anyone_home": snapshot["derived"]["anyone_home"],
            "is_night": snapshot["derived"]["is_night"],
            "last_changed": window.get("last_changed"),
        }
        findings.append(_build_finding(rule, [window_id], evidence))
    return findings


def _find_open_window_entity_ids(
    entity_map: Mapping[str, SnapshotEntity],
) -> list[str]:
    window_ids: list[str] = []
    for entity_id, entity in entity_map.items():
        if entity.get("state") != "on":
            continue
        domain = entity.get("domain")
        if domain not in {"binary_sensor", "sensor"}:
            continue
        device_class = str(entity.get("attributes", {}).get("device_class", ""))
        if device_class == "window" or "window" in entity_id:
            window_ids.append(entity_id)
    return sorted(set(window_ids))


def _build_finding(
    rule: dict[str, Any],
    triggering_entities: list[str],
    evidence: dict[str, Any],
) -> AnomalyFinding:
    rule_id = str(rule.get("rule_id") or "dynamic_rule")
    severity_value = str(rule.get("severity") or "low")
    severity: Severity
    if severity_value in _SEVERITIES:
        severity = cast("Severity", severity_value)
    else:
        severity = "low"
    confidence = _coerce_float(rule.get("confidence"), default=0.5)
    suggested_actions = rule.get("suggested_actions") or []
    if not isinstance(suggested_actions, list):
        suggested_actions = []
    is_sensitive = bool(rule.get("is_sensitive", False))
    anomaly_id = build_anomaly_id(rule_id, triggering_entities, evidence)
    return AnomalyFinding(
        anomaly_id=anomaly_id,
        type=rule_id,
        severity=severity,
        confidence=confidence,
        triggering_entities=list(triggering_entities),
        evidence=evidence,
        suggested_actions=list(suggested_actions),
        is_sensitive=is_sensitive,
    )


def _coerce_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _rule_params(rule: dict[str, Any]) -> dict[str, Any]:
    """Return normalized params from a dynamic rule payload."""
    params = rule.get("params")
    if isinstance(params, dict):
        return params
    return {}

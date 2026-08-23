"""Sentinel models for anomaly findings."""

from __future__ import annotations

import hashlib
import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING, Any, Literal

from homeassistant.util import dt as dt_util

from custom_components.home_generative_agent.const import (
    RESERVED_IDENTITY_LABELS,
    SENTINEL_CAMERA_ACTIVITY_STALENESS_MINUTES,
    UNKNOWN_PERSON_LABEL,
)

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping

LOGGER = logging.getLogger(__name__)

Severity = Literal["low", "medium", "high"]

_UNKNOWN_PERSON_NORMALIZED = UNKNOWN_PERSON_LABEL.lower()

# A sighting timestamp this far in the future is a clock-skew or spoofed
# value, not freshness evidence: without this bound a camera publishing a
# far-future timestamp would keep its persisted sighting "fresh" until wall
# time caught up (adversarial review). Small skew is tolerated.
_FUTURE_SKEW_TOLERANCE_MINUTES = 2.0


def has_unknown_person(names: Iterable[str]) -> bool:
    """
    Return True when face recognition saw a person it could not identify.

    The snapshot's recognized_people list mixes enrolled names with reserved
    pipeline labels; "Unknown Person" is the positive stranger signal.
    Matching is normalized (strip + lowercase) because legacy gallery rows
    may carry variants like "unknown person".
    """
    return any(str(n).strip().lower() == _UNKNOWN_PERSON_NORMALIZED for n in names)


def enrolled_people(names: Iterable[str]) -> list[str]:
    """
    Return only the names that denote enrolled identities.

    Filters the reserved pipeline labels ("Unknown Person", "Indeterminate",
    legacy "None", empty) that the recognition pipeline emits alongside real
    enrolled names. Rules that mean "a known person was recognized" must use
    this instead of truthiness of recognized_people — the raw list is
    effectively never empty on a face-recognition install.
    """
    return [
        str(n) for n in names if str(n).strip().lower() not in RESERVED_IDENTITY_LABELS
    ]


def minutes_between(earlier_iso: str | None, later_iso: str | None) -> float | None:
    """
    Return elapsed minutes from *earlier_iso* to *later_iso*, or None.

    Both values are normalized to aware UTC before subtraction — a tz-naive
    string from a third-party camera attribute is interpreted as local time
    (HA convention) instead of raising TypeError, which on the dynamic-rule
    path would escape every exception boundary and kill the Sentinel run
    loop. A timestamp more than a small skew tolerance in the FUTURE returns
    None (freshness cannot be proven from a clock ahead of the snapshot);
    small negative deltas clamp to 0.
    """
    if not earlier_iso or not later_iso:
        return None
    t_earlier = dt_util.parse_datetime(earlier_iso)
    t_later = dt_util.parse_datetime(later_iso)
    if t_earlier is None or t_later is None:
        return None
    delta_minutes = (
        dt_util.as_utc(t_later) - dt_util.as_utc(t_earlier)
    ).total_seconds() / 60.0
    if delta_minutes < -_FUTURE_SKEW_TOLERANCE_MINUTES:
        return None
    return max(0.0, delta_minutes)


def sighting_timestamp(activity: Mapping[str, Any]) -> str | None:
    """
    Return the timestamp of the camera's face-recognition sighting.

    recognition_last_event is stamped by this integration's image entity from
    the same signal that carries recognized_people, so it dates the sighting
    itself. last_activity is only a fallback: on integrations that expose
    motion attributes it dates the latest MOTION, which pets or wind can
    refresh long after the recognized_people labels went stale.
    """
    return activity.get("recognition_last_event") or activity.get("last_activity")


def unknown_person_sighting_is_actionable(
    activity: Mapping[str, Any], generated_at: str
) -> bool:
    """
    Return True for a fresh, unaccompanied stranger sighting.

    Fire only on a positive "Unknown Person" label: the raw recognized_people
    list mixes in reserved labels ("Indeterminate" on every analyzed event),
    so truthiness would suppress unknown-person rules forever on
    face-recognition installs — and a real stranger writes "Unknown Person",
    which must fire the rule, not veto it.

    A stranger alongside an enrolled name is treated as the genuine-companion
    signal (a resident with a guest) and suppressed. Note the list is a
    batch-level union, not per-frame co-occurrence, so an identity-merge
    refusal (the same face flapping between a known name and "Unknown
    Person", issue #543) also lands here — suppressing it is deliberate,
    because firing on refused merges would re-create the phantom-stranger
    alerts that #543 eliminated.

    The staleness gate exists because recognized_people persists on the image
    entity until the next analyzed event; without it one sighting would
    re-fire for hours. A missing or unparseable timestamp cannot prove
    freshness and is skipped (logged at debug so the drop is observable).
    """
    people = activity.get("recognized_people") or []
    if not has_unknown_person(people):
        return False
    if enrolled_people(people):
        return False
    age_minutes = minutes_between(sighting_timestamp(activity), generated_at)
    if age_minutes is None:
        LOGGER.debug(
            "Unknown-person sighting on %s dropped: no provable freshness "
            "(sighting timestamp missing, unparseable, or in the future).",
            activity.get("camera_entity_id"),
        )
        return False
    return age_minutes <= SENTINEL_CAMERA_ACTIVITY_STALENESS_MINUTES


def _as_iso(value: datetime) -> str:
    return dt_util.as_utc(value).isoformat()


def _jsonify(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, datetime):
        return _as_iso(value)
    if isinstance(value, dict):
        return {str(k): _jsonify(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_jsonify(v) for v in value]
    return str(value)


def build_anomaly_id(
    anomaly_type: str, triggering_entities: list[str], evidence: dict[str, Any]
) -> str:
    """Create a stable hash for a finding."""
    payload = {
        "type": anomaly_type,
        "entities": sorted(triggering_entities),
        "evidence": _jsonify(evidence),
    }
    return hashlib.sha256(repr(payload).encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class AnomalyFinding:
    """Structured anomaly finding."""

    anomaly_id: str
    type: str
    severity: Severity
    confidence: float
    triggering_entities: list[str]
    evidence: dict[str, Any]
    suggested_actions: list[str]
    is_sensitive: bool
    detected_at: datetime = field(default_factory=dt_util.utcnow)

    def as_dict(self) -> dict[str, Any]:
        """Serialize the finding for storage/notifications."""
        return {
            "anomaly_id": self.anomaly_id,
            "type": self.type,
            "severity": self.severity,
            "confidence": self.confidence,
            "triggering_entities": list(self.triggering_entities),
            "evidence": _jsonify(self.evidence),
            "suggested_actions": list(self.suggested_actions),
            "is_sensitive": self.is_sensitive,
            "detected_at": _as_iso(self.detected_at),
        }


def _max_severity(findings: list[AnomalyFinding]) -> Severity:
    """Return the highest severity across a list of findings."""
    order: dict[Severity, int] = {"low": 0, "medium": 1, "high": 2}
    return max((f.severity for f in findings), key=lambda s: order[s])


def _merge_evidence(findings: list[AnomalyFinding]) -> dict[str, Any]:
    """Merge evidence dicts from constituent findings into a single dict."""
    merged: dict[str, Any] = {}
    for i, finding in enumerate(findings):
        for k, v in finding.evidence.items():
            key = f"{finding.type}.{k}" if k in merged else k
            merged[key] = v
        # Always store per-finding evidence under a namespaced key
        merged[f"constituent_{i}"] = _jsonify(finding.evidence)
    return merged


@dataclass(frozen=True)
class CompoundFinding:
    """
    A correlated group of related AnomalyFinding objects.

    Produced by SentinelCorrelator for findings detected in the same
    ``_run_once()`` call.  The object is frozen (immutable) once created.
    """

    compound_id: str
    constituent_findings: tuple[AnomalyFinding, ...]
    merged_evidence: dict[str, Any]
    severity: Severity
    confidence: float
    triggering_entities: tuple[str, ...]
    is_sensitive: bool

    @classmethod
    def from_findings(cls, findings: list[AnomalyFinding]) -> CompoundFinding:
        """Build a :class:`CompoundFinding` from a non-empty list of findings."""
        if not findings:
            msg = "CompoundFinding requires at least one constituent finding."
            raise ValueError(msg)
        all_entities: list[str] = []
        for f in findings:
            all_entities.extend(f.triggering_entities)
        return cls(
            compound_id=str(uuid.uuid4()),
            constituent_findings=tuple(findings),
            merged_evidence=_merge_evidence(findings),
            severity=_max_severity(findings),
            confidence=sum(f.confidence for f in findings) / len(findings),
            triggering_entities=tuple(dict.fromkeys(all_entities)),
            is_sensitive=any(f.is_sensitive for f in findings),
        )

    def as_dict(self) -> dict[str, Any]:
        """Serialize the compound finding for storage/notifications."""
        return {
            "compound_id": self.compound_id,
            "constituent_findings": [f.as_dict() for f in self.constituent_findings],
            "merged_evidence": _jsonify(self.merged_evidence),
            "severity": self.severity,
            "confidence": self.confidence,
            "triggering_entities": list(self.triggering_entities),
            "is_sensitive": self.is_sensitive,
        }


# Convenience union type used in the engine pipeline.
type Finding = AnomalyFinding | CompoundFinding

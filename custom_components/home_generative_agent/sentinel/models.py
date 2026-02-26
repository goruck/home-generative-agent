"""Sentinel models for anomaly findings."""

from __future__ import annotations

import hashlib
import uuid
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Literal

from homeassistant.util import dt as dt_util

Severity = Literal["low", "medium", "high"]


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

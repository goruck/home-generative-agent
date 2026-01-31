"""Sentinel models for anomaly findings."""

from __future__ import annotations

import hashlib
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
    digest = hashlib.sha256(repr(payload).encode("utf-8")).hexdigest()
    return digest


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

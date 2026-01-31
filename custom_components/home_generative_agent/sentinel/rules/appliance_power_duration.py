"""Rule: appliance power usage exceeds a duration threshold."""

from __future__ import annotations

from datetime import datetime, timedelta

from homeassistant.util import dt as dt_util

from ...snapshot.schema import FullStateSnapshot
from ..models import AnomalyFinding, build_anomaly_id

DEFAULT_POWER_THRESHOLD_W = 100.0
DEFAULT_DURATION_MIN = 60


class AppliancePowerDurationRule:
    """Detect appliances drawing power for too long."""

    rule_id = "appliance_power_duration"

    def __init__(
        self,
        power_threshold_w: float = DEFAULT_POWER_THRESHOLD_W,
        duration_min: int = DEFAULT_DURATION_MIN,
    ) -> None:
        self._power_threshold_w = power_threshold_w
        self._duration_min = duration_min

    def evaluate(self, snapshot: FullStateSnapshot) -> list[AnomalyFinding]:
        findings: list[AnomalyFinding] = []
        now = dt_util.parse_datetime(snapshot["derived"]["now"]) or dt_util.utcnow()
        threshold = timedelta(minutes=self._duration_min)

        for entity in snapshot["entities"]:
            if entity["domain"] != "sensor":
                continue
            attrs = entity["attributes"]
            device_class = attrs.get("device_class")
            unit = attrs.get("unit_of_measurement")
            if device_class != "power" and unit not in {"W", "kW"}:
                continue
            power_val = _coerce_float(entity["state"])
            if power_val is None:
                continue
            if unit == "kW":
                power_val *= 1000.0
            if power_val < self._power_threshold_w:
                continue
            last_changed = dt_util.parse_datetime(entity["last_changed"])
            if last_changed is None:
                continue
            if now - last_changed < threshold:
                continue

            evidence = {
                "entity_id": entity["entity_id"],
                "area": entity["area"],
                "power_w": power_val,
                "duration_min": int((now - last_changed).total_seconds() / 60),
                "threshold_min": self._duration_min,
            }
            anomaly_id = build_anomaly_id(self.rule_id, [entity["entity_id"]], evidence)
            findings.append(
                AnomalyFinding(
                    anomaly_id=anomaly_id,
                    type=self.rule_id,
                    severity="medium",
                    confidence=0.6,
                    triggering_entities=[entity["entity_id"]],
                    evidence=evidence,
                    suggested_actions=["check_appliance"],
                    is_sensitive=False,
                )
            )
        return findings


def _coerce_float(value: str) -> float | None:
    try:
        return float(value)
    except ValueError:
        return None

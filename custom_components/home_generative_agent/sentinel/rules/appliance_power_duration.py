"""Rule: appliance power usage exceeds a duration threshold."""

from __future__ import annotations

import math
from datetime import timedelta
from typing import TYPE_CHECKING

from homeassistant.util import dt as dt_util

from custom_components.home_generative_agent.sentinel.models import (
    AnomalyFinding,
    build_anomaly_id,
)

if TYPE_CHECKING:
    from datetime import datetime

    from custom_components.home_generative_agent.snapshot.schema import (
        FullStateSnapshot,
    )

DEFAULT_POWER_THRESHOLD_W = 100.0
DEFAULT_DURATION_MIN = 60

# Evidence keys excluded from the anomaly-id hash: friendly_name is cosmetic,
# and duration_min/power_w change on every evaluation cycle of an ongoing
# episode — hashing them would mint a new anomaly_id per cycle, defeating
# pending-prompt suppression and fragmenting the audit trail.
_UNSTABLE_EVIDENCE_KEYS = frozenset({"friendly_name", "duration_min", "power_w"})


class AppliancePowerDurationRule:
    """
    Detect appliances drawing power above a threshold for too long.

    Duration is measured by observation: the rule remembers when each sensor
    was first seen at or above ``power_threshold_w`` (rule instances persist
    across evaluation cycles) and reports the elapsed time since that rising
    edge.  HA's ``last_changed`` is deliberately not used — raw, it only
    advances when the *value* changes, so it measures how long a reading has
    been numerically static (issue #461); enriched by power_enrichment.py, it
    marks the recorder-derived crossing of the 10 W "off" level, which for an
    appliance idling between 10 W and the rule threshold could predate the
    actual threshold crossing by weeks.  Dropping below the threshold,
    becoming non-numeric (e.g. "unavailable"), or leaving the snapshot ends
    the episode.  Engine restart clears the tracker, so an already-running
    appliance is re-detected after a full ``duration_min`` of observation
    (same accepted tradeoff as the engine's cyclical sustained-deviation
    gate).
    """

    rule_id = "appliance_power_duration"

    def __init__(
        self,
        power_threshold_w: float = DEFAULT_POWER_THRESHOLD_W,
        duration_min: int = DEFAULT_DURATION_MIN,
    ) -> None:
        """Initialize thresholds for sustained appliance power usage."""
        self._power_threshold_w = power_threshold_w
        self._duration_min = duration_min
        # entity_id -> when the reading was first observed at or above the
        # power threshold in the current episode.
        self._above_since: dict[str, datetime] = {}

    def evaluate(self, snapshot: FullStateSnapshot) -> list[AnomalyFinding]:
        """Return findings for sensors above threshold for a long duration."""
        findings: list[AnomalyFinding] = []
        now = dt_util.as_utc(
            dt_util.parse_datetime(snapshot["derived"]["now"]) or dt_util.utcnow()
        )
        threshold = timedelta(minutes=self._duration_min)
        still_above: set[str] = set()

        for entity in snapshot["entities"]:
            if entity["domain"] != "sensor":
                continue
            attrs = entity["attributes"]
            device_class = attrs.get("device_class")
            unit = attrs.get("unit_of_measurement")
            if device_class != "power" and unit not in {"W", "kW"}:
                continue
            entity_id = entity["entity_id"]
            power_val = _coerce_float(entity["state"])
            if power_val is None:
                # Falling edge is applied eagerly (not only in the post-loop
                # sweep) so an exception later in the loop — swallowed by the
                # engine — cannot preserve a stale episode start.
                self._above_since.pop(entity_id, None)
                continue
            if unit == "kW":
                power_val *= 1000.0
            if power_val < self._power_threshold_w:
                self._above_since.pop(entity_id, None)
                continue

            still_above.add(entity_id)
            above_since = self._above_since.setdefault(entity_id, now)
            elapsed = now - above_since
            if elapsed < threshold:
                continue

            evidence = {
                "entity_id": entity_id,
                "area": entity["area"],
                "power_w": power_val,
                "duration_min": int(elapsed.total_seconds() / 60),
                "threshold_min": self._duration_min,
                "since": above_since.isoformat(),
                "friendly_name": entity["friendly_name"] or None,
            }
            hash_evidence = {
                k: v for k, v in evidence.items() if k not in _UNSTABLE_EVIDENCE_KEYS
            }
            anomaly_id = build_anomaly_id(self.rule_id, [entity_id], hash_evidence)
            findings.append(
                AnomalyFinding(
                    anomaly_id=anomaly_id,
                    type=self.rule_id,
                    severity="medium",
                    confidence=0.6,
                    triggering_entities=[entity_id],
                    evidence=evidence,
                    suggested_actions=["check_appliance"],
                    is_sensitive=False,
                )
            )

        # Entities that vanished from the snapshot (removed or renamed) also
        # end their episode.
        for entity_id in list(self._above_since):
            if entity_id not in still_above:
                del self._above_since[entity_id]
        return findings


def _coerce_float(value: str) -> float | None:
    try:
        parsed = float(value)
    except (ValueError, TypeError):
        return None
    # nan compares False against the threshold and would masquerade as an
    # above-threshold reading; inf would produce nonsense evidence.
    return parsed if math.isfinite(parsed) else None

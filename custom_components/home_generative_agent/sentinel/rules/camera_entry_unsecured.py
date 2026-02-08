"""Rule: camera activity near unsecured entry."""

from __future__ import annotations

from datetime import timedelta
from typing import TYPE_CHECKING

from homeassistant.util import dt as dt_util

from custom_components.home_generative_agent.sentinel.models import (
    AnomalyFinding,
    build_anomaly_id,
)

if TYPE_CHECKING:
    from custom_components.home_generative_agent.snapshot.schema import (
        FullStateSnapshot,
    )

ENTRY_CLASSES = {"door", "window", "opening"}
ACTIVITY_WINDOW_MIN = 10


class CameraEntryUnsecuredRule:
    """Detect camera activity while nearby entries are unsecured."""

    rule_id = "camera_entry_unsecured"

    def evaluate(self, snapshot: FullStateSnapshot) -> list[AnomalyFinding]:
        """Return findings for recent camera activity near unsecured entries."""
        findings: list[AnomalyFinding] = []
        now = dt_util.parse_datetime(snapshot["derived"]["now"]) or dt_util.utcnow()
        window = timedelta(minutes=ACTIVITY_WINDOW_MIN)

        unsecured_by_area: dict[str, list[str]] = {}
        for entity in snapshot["entities"]:
            area = entity.get("area")
            if not area:
                continue
            if entity["domain"] == "lock" and entity["state"] == "unlocked":
                unsecured_by_area.setdefault(area, []).append(entity["entity_id"])
                continue
            if entity["domain"] != "binary_sensor":
                continue
            if entity["attributes"].get("device_class") not in ENTRY_CLASSES:
                continue
            if entity["state"] != "on":
                continue
            unsecured_by_area.setdefault(area, []).append(entity["entity_id"])

        for activity in snapshot["camera_activity"]:
            area = activity.get("area")
            if not area or area not in unsecured_by_area:
                continue
            last_activity = activity.get("last_activity")
            if not last_activity:
                continue
            last_dt = dt_util.parse_datetime(last_activity)
            if last_dt is None:
                continue
            if now - last_dt > window:
                continue
            evidence = {
                "camera_entity_id": activity["camera_entity_id"],
                "area": area,
                "last_activity": last_activity,
                "unsecured_entities": sorted(unsecured_by_area[area]),
            }
            anomaly_id = build_anomaly_id(
                self.rule_id, [activity["camera_entity_id"]], evidence
            )
            findings.append(
                AnomalyFinding(
                    anomaly_id=anomaly_id,
                    type=self.rule_id,
                    severity="high",
                    confidence=0.6,
                    triggering_entities=[activity["camera_entity_id"]],
                    evidence=evidence,
                    suggested_actions=["check_entry"],
                    is_sensitive=True,
                )
            )
        return findings

"""Rule: vehicle parked near front gate while residents are home."""

from __future__ import annotations

from typing import TYPE_CHECKING

from custom_components.home_generative_agent.sentinel.models import (
    AnomalyFinding,
    build_anomaly_id,
)

if TYPE_CHECKING:
    from custom_components.home_generative_agent.snapshot.schema import (
        FullStateSnapshot,
    )

_FRONTGATE_CAM = "camera.frontgate"

_VEHICLE_KEYWORDS = frozenset(
    {
        "vehicle",
        "car",
        "suv",
        "truck",
        "van",
        "automobile",
        "sedan",
        "pickup",
        "minivan",
    }
)


def _snapshot_mentions_vehicle(summary: str) -> bool:
    """Return True if the snapshot summary contains a vehicle keyword."""
    lower = summary.lower()
    return any(kw in lower for kw in _VEHICLE_KEYWORDS)


class VehicleParkedNearFrontGateRule:
    """Detect a vehicle parked near the front gate while residents are home."""

    rule_id = "vehicle_parked_near_frontgate_home"

    def evaluate(self, snapshot: FullStateSnapshot) -> list[AnomalyFinding]:
        """Return a finding when a vehicle is observed near the front gate."""
        if not snapshot["derived"]["anyone_home"]:
            return []

        for activity in snapshot["camera_activity"]:
            if activity["camera_entity_id"] != _FRONTGATE_CAM:
                continue

            snapshot_summary = activity.get("snapshot_summary")
            if not snapshot_summary:
                return []

            if not _snapshot_mentions_vehicle(snapshot_summary):
                return []

            evidence = {
                "camera_entity_id": _FRONTGATE_CAM,
                "area": activity.get("area"),
                "snapshot_summary": snapshot_summary,
                "recognized_people": activity.get("recognized_people", []),
                "last_activity": activity.get("last_activity"),
                "is_night": snapshot["derived"]["is_night"],
                "anyone_home": snapshot["derived"]["anyone_home"],
                "people_home": snapshot["derived"]["people_home"],
            }
            anomaly_id = build_anomaly_id(self.rule_id, [_FRONTGATE_CAM], evidence)
            return [
                AnomalyFinding(
                    anomaly_id=anomaly_id,
                    type=self.rule_id,
                    severity="low",
                    confidence=0.6,
                    triggering_entities=[_FRONTGATE_CAM],
                    evidence=evidence,
                    suggested_actions=["close_entry"],
                    is_sensitive=False,
                )
            ]

        return []

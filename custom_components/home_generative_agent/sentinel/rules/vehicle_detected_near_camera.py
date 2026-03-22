"""Rule: vehicle detected near any monitored camera while residents are home."""

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


class VehicleDetectedNearCameraRule:
    """Detect a vehicle on any monitored camera while residents are home."""

    rule_id = "vehicle_detected_near_camera_home"

    def evaluate(self, snapshot: FullStateSnapshot) -> list[AnomalyFinding]:
        """Return findings when a vehicle is observed on any monitored camera."""
        if not snapshot["derived"]["anyone_home"]:
            return []

        findings: list[AnomalyFinding] = []
        for activity in snapshot["camera_activity"]:
            summary = activity.get("snapshot_summary")
            if not summary:
                continue
            # Require motion context — scopes to actively monitored/outdoor cameras.
            if (
                not activity.get("motion_entities")
                and not activity.get("vmd_entities")
                and not activity.get("last_activity")
            ):
                continue
            if not _snapshot_mentions_vehicle(summary):
                continue

            evidence = {
                "camera_entity_id": activity["camera_entity_id"],
                "area": activity.get("area"),
                "snapshot_summary": summary,
                "recognized_people": activity.get("recognized_people", []),
                "last_activity": activity.get("last_activity"),
                "is_night": snapshot["derived"]["is_night"],
                "anyone_home": snapshot["derived"]["anyone_home"],
                "people_home": snapshot["derived"]["people_home"],
            }
            anomaly_id = build_anomaly_id(
                self.rule_id, [activity["camera_entity_id"]], evidence
            )
            findings.append(
                AnomalyFinding(
                    anomaly_id=anomaly_id,
                    type=self.rule_id,
                    severity="low",
                    confidence=0.6,
                    triggering_entities=[activity["camera_entity_id"]],
                    evidence=evidence,
                    suggested_actions=["close_entry"],
                    is_sensitive=False,
                )
            )
        return findings

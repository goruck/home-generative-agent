"""Rule: camera missing snapshot summary at night while home is occupied."""

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

_BACKGARAGE_CAM = "camera.backgarage"


class CameraBackgarageMissingSnapshotRule:
    """Detect when the backgarage camera has no snapshot summary at night while home."""

    rule_id = "camera_backgarage_missing_snapshot_night_home"

    def evaluate(self, snapshot: FullStateSnapshot) -> list[AnomalyFinding]:
        """Return a finding when backgarage camera is missing a snapshot at night."""
        if not snapshot["derived"]["is_night"]:
            return []

        if not snapshot["derived"]["anyone_home"]:
            return []

        for activity in snapshot["camera_activity"]:
            if activity["camera_entity_id"] != _BACKGARAGE_CAM:
                continue

            snapshot_summary = activity.get("snapshot_summary")
            if snapshot_summary:
                # Camera is capturing images — no anomaly.
                return []

            evidence = {
                "camera_entity_id": _BACKGARAGE_CAM,
                "area": activity.get("area"),
                "snapshot_summary": None,
                "is_night": snapshot["derived"]["is_night"],
                "anyone_home": snapshot["derived"]["anyone_home"],
            }
            anomaly_id = build_anomaly_id(self.rule_id, [_BACKGARAGE_CAM], evidence)
            return [
                AnomalyFinding(
                    anomaly_id=anomaly_id,
                    type=self.rule_id,
                    severity="low",
                    confidence=0.6,
                    triggering_entities=[_BACKGARAGE_CAM],
                    evidence=evidence,
                    suggested_actions=["check_camera"],
                    is_sensitive=False,
                )
            ]

        # Camera entity not present in snapshot at all — also anomalous.
        evidence = {
            "camera_entity_id": _BACKGARAGE_CAM,
            "area": None,
            "snapshot_summary": None,
            "is_night": snapshot["derived"]["is_night"],
            "anyone_home": snapshot["derived"]["anyone_home"],
        }
        anomaly_id = build_anomaly_id(self.rule_id, [_BACKGARAGE_CAM], evidence)
        return [
            AnomalyFinding(
                anomaly_id=anomaly_id,
                type=self.rule_id,
                severity="low",
                confidence=0.6,
                triggering_entities=[_BACKGARAGE_CAM],
                evidence=evidence,
                suggested_actions=["check_camera"],
                is_sensitive=False,
            )
        ]

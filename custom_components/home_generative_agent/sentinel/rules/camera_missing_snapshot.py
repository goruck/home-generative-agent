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


class CameraMissingSnapshotRule:
    """Detect any monitored camera without a snapshot summary at night while home."""

    rule_id = "camera_missing_snapshot_night_home"

    def evaluate(self, snapshot: FullStateSnapshot) -> list[AnomalyFinding]:
        """Return findings for monitored cameras missing a snapshot at night."""
        if not snapshot["derived"]["is_night"]:
            return []

        if not snapshot["derived"]["anyone_home"]:
            return []

        findings: list[AnomalyFinding] = []
        for activity in snapshot["camera_activity"]:
            # Only fire for cameras with active motion sensors — best proxy for
            # "expected to capture" without hardcoding entity IDs.
            if not activity.get("motion_entities"):
                continue
            # Camera is capturing images — no anomaly.
            if activity.get("snapshot_summary"):
                continue

            evidence = {
                "camera_entity_id": activity["camera_entity_id"],
                "area": activity.get("area"),
                "snapshot_summary": None,
                "is_night": snapshot["derived"]["is_night"],
                "anyone_home": snapshot["derived"]["anyone_home"],
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
                    suggested_actions=["check_camera"],
                    is_sensitive=False,
                )
            )
        return findings

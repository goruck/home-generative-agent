"""Rule: unknown person on camera at night while someone is home."""

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


class UnknownPersonAtNightWhileHomeRule:
    """Detect an unrecognized person on camera at night while the home is occupied."""

    rule_id = "unknown_person_camera_night_home"

    def evaluate(self, snapshot: FullStateSnapshot) -> list[AnomalyFinding]:
        """Return findings when an unknown person is seen on camera at night."""
        if not snapshot["derived"]["is_night"]:
            return []

        if not snapshot["derived"]["anyone_home"]:
            return []

        findings: list[AnomalyFinding] = []
        for activity in snapshot["camera_activity"]:
            # Require a snapshot summary as evidence the camera captured something.
            snapshot_summary = activity.get("snapshot_summary")
            if not snapshot_summary:
                continue
            # Trigger only for unknown (unrecognized) persons.
            if activity.get("recognized_people"):
                continue
            # Require some indication of activity.
            if (
                not activity.get("last_activity")
                and not activity.get("motion_entities")
                and not activity.get("vmd_entities")
            ):
                continue

            evidence = {
                "camera_entity_id": activity["camera_entity_id"],
                "area": activity.get("area"),
                "snapshot_summary": snapshot_summary,
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
                    confidence=0.7,
                    triggering_entities=[activity["camera_entity_id"]],
                    evidence=evidence,
                    suggested_actions=["close_entry"],
                    is_sensitive=False,
                )
            )
        return findings

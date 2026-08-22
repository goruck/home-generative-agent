"""Rule: unknown person detected by camera while no one is home."""

from __future__ import annotations

from typing import TYPE_CHECKING

from custom_components.home_generative_agent.sentinel.models import (
    AnomalyFinding,
    build_anomaly_id,
    unknown_person_sighting_is_actionable,
)

if TYPE_CHECKING:
    from custom_components.home_generative_agent.snapshot.schema import (
        FullStateSnapshot,
    )


class UnknownPersonCameraNoHomeRule:
    """Detect an unrecognized person on camera while the home is unoccupied."""

    rule_id = "unknown_person_camera_no_home"

    def evaluate(self, snapshot: FullStateSnapshot) -> list[AnomalyFinding]:
        """Return findings when an unknown person is seen and no one is home."""
        if snapshot["derived"]["anyone_home"]:
            return []

        generated_at = snapshot["generated_at"]
        findings: list[AnomalyFinding] = []
        for activity in snapshot["camera_activity"]:
            # Fresh, unaccompanied stranger sighting — see the helper for the
            # full predicate rationale (reserved labels, companion
            # suppression, staleness).
            if not unknown_person_sighting_is_actionable(activity, generated_at):
                continue

            evidence = {
                "camera_entity_id": activity["camera_entity_id"],
                "area": activity.get("area"),
                "last_activity": activity.get("last_activity"),
                "recognition_last_event": activity.get("recognition_last_event"),
                "recognized_people": list(activity.get("recognized_people") or []),
                "motion_entities": activity.get("motion_entities", []),
                "vmd_entities": activity.get("vmd_entities", []),
                "anyone_home": snapshot["derived"]["anyone_home"],
                "is_night": snapshot["derived"]["is_night"],
            }
            anomaly_id = build_anomaly_id(
                self.rule_id, [activity["camera_entity_id"]], evidence
            )
            findings.append(
                AnomalyFinding(
                    anomaly_id=anomaly_id,
                    type=self.rule_id,
                    severity="low",
                    confidence=0.85,
                    triggering_entities=[activity["camera_entity_id"]],
                    evidence=evidence,
                    suggested_actions=["close_entry"],
                    is_sensitive=True,
                )
            )
        return findings

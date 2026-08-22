"""Rule: unknown person on camera at night while someone is home."""

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


class UnknownPersonAtNightWhileHomeRule:
    """Detect an unrecognized person on camera at night while the home is occupied."""

    rule_id = "unknown_person_camera_night_home"

    def evaluate(self, snapshot: FullStateSnapshot) -> list[AnomalyFinding]:
        """Return findings when an unknown person is seen on camera at night."""
        if not snapshot["derived"]["is_night"]:
            return []

        if not snapshot["derived"]["anyone_home"]:
            return []

        generated_at = snapshot["generated_at"]
        findings: list[AnomalyFinding] = []
        for activity in snapshot["camera_activity"]:
            # Require a snapshot summary as evidence the camera captured something.
            snapshot_summary = activity.get("snapshot_summary")
            if not snapshot_summary:
                continue
            # Fresh, unaccompanied stranger sighting — see the helper for the
            # full predicate rationale. The staleness gate also stops an
            # afternoon sighting from re-firing as a night finding hours
            # later.
            if not unknown_person_sighting_is_actionable(activity, generated_at):
                continue

            evidence = {
                "camera_entity_id": activity["camera_entity_id"],
                "area": activity.get("area"),
                "snapshot_summary": snapshot_summary,
                "recognized_people": list(activity.get("recognized_people") or []),
                "last_activity": activity.get("last_activity"),
                "recognition_last_event": activity.get("recognition_last_event"),
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

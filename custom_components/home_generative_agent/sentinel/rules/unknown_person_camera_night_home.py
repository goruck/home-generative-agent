"""Rule: unknown person on camera at night while someone is home."""

from __future__ import annotations

from typing import TYPE_CHECKING

from custom_components.home_generative_agent.const import (
    SENTINEL_CAMERA_ACTIVITY_STALENESS_MINUTES,
)
from custom_components.home_generative_agent.sentinel.models import (
    AnomalyFinding,
    build_anomaly_id,
    enrolled_people,
    has_unknown_person,
    minutes_between,
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
            people = activity.get("recognized_people") or []
            # Fire only on a positive stranger sighting. The raw list mixes in
            # reserved labels ("Indeterminate" on every analyzed event), so
            # truthiness would suppress the rule forever on face-recognition
            # installs — and a real stranger writes "Unknown Person", which
            # must fire the rule, not veto it.
            if not has_unknown_person(people):
                continue
            # A stranger alongside an enrolled person is the genuine-companion
            # signal (a resident with a guest), not an intrusion.
            if enrolled_people(people):
                continue
            # Staleness gate: recognized_people persists on the image entity
            # until the next analyzed event, so require the sighting itself to
            # be recent — a stranger seen in the afternoon must not re-fire as
            # a night finding hours later. A missing/unparseable timestamp
            # cannot prove freshness and is skipped.
            age_minutes = minutes_between(activity.get("last_activity"), generated_at)
            if (
                age_minutes is None
                or age_minutes > SENTINEL_CAMERA_ACTIVITY_STALENESS_MINUTES
            ):
                continue

            evidence = {
                "camera_entity_id": activity["camera_entity_id"],
                "area": activity.get("area"),
                "snapshot_summary": snapshot_summary,
                "recognized_people": list(people),
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

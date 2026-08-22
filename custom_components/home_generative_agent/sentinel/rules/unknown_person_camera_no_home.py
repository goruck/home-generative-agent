"""Rule: unknown person detected by camera while no one is home."""

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
            # be recent. A missing/unparseable timestamp cannot prove
            # freshness and is skipped.
            age_minutes = minutes_between(activity.get("last_activity"), generated_at)
            if (
                age_minutes is None
                or age_minutes > SENTINEL_CAMERA_ACTIVITY_STALENESS_MINUTES
            ):
                continue

            evidence = {
                "camera_entity_id": activity["camera_entity_id"],
                "area": activity.get("area"),
                "last_activity": activity["last_activity"],
                "recognized_people": list(people),
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

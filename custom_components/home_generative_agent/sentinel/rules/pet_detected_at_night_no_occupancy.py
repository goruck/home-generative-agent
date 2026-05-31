"""Rule: pet detected by camera at night while no one is home."""

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

_PET_KEYWORDS = frozenset(
    {
        "cat",
        "kitten",
        "dog",
        "puppy",
        "rabbit",
        "hamster",
        "bird",
        "parrot",
        "pet",
    }
)


def _snapshot_mentions_pet(summary: str) -> bool:
    """Return True if the snapshot summary contains a pet keyword."""
    lower = summary.lower()
    return any(kw in lower for kw in _PET_KEYWORDS)


class PetDetectedAtNightNoOccupancyRule:
    """Detect a pet on camera at night while no residents are home."""

    rule_id = "pet_detected_at_night_no_occupancy"

    def evaluate(self, snapshot: FullStateSnapshot) -> list[AnomalyFinding]:
        """Return findings when a pet is seen at night with no one home."""
        if not snapshot["derived"]["is_night"]:
            return []
        if snapshot["derived"]["anyone_home"]:
            return []

        findings: list[AnomalyFinding] = []
        for activity in snapshot["camera_activity"]:
            summary = activity.get("snapshot_summary")
            if not summary:
                continue
            if (
                not activity.get("motion_entities")
                and not activity.get("vmd_entities")
                and not activity.get("last_activity")
            ):
                continue
            if not _snapshot_mentions_pet(summary):
                continue

            evidence = {
                "camera_entity_id": activity["camera_entity_id"],
                "area": activity.get("area"),
                "snapshot_summary": summary,
                "last_activity": activity.get("last_activity"),
                "is_night": snapshot["derived"]["is_night"],
                "anyone_home": snapshot["derived"]["anyone_home"],
                "people_away": snapshot["derived"]["people_away"],
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
                    suggested_actions=[],
                    is_sensitive=False,
                )
            )
        return findings

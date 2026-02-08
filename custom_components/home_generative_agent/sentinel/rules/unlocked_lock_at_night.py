"""Rule: unlocked exterior lock at night."""

from __future__ import annotations

from typing import TYPE_CHECKING

from custom_components.home_generative_agent.sentinel.models import (
    AnomalyFinding,
    build_anomaly_id,
)

if TYPE_CHECKING:
    from custom_components.home_generative_agent.snapshot.schema import (
        FullStateSnapshot,
        SnapshotEntity,
    )

EXTERIOR_HINTS = (
    "front",
    "back",
    "exterior",
    "outside",
    "entry",
    "door",
    "garage",
    "gate",
    "patio",
)


class UnlockedLockAtNightRule:
    """Detect unlocked exterior locks at night."""

    rule_id = "unlocked_lock_at_night"

    def evaluate(self, snapshot: FullStateSnapshot) -> list[AnomalyFinding]:
        """Return findings for exterior locks left unlocked at night."""
        if not snapshot["derived"]["is_night"]:
            return []

        findings: list[AnomalyFinding] = []
        for entity in snapshot["entities"]:
            if entity["domain"] != "lock":
                continue
            if entity["state"] != "unlocked":
                continue
            if not _is_exterior_hint(entity):
                continue

            evidence = {
                "entity_id": entity["entity_id"],
                "area": entity["area"],
                "friendly_name": entity["friendly_name"],
                "state": entity["state"],
                "last_changed": entity["last_changed"],
                "is_night": snapshot["derived"]["is_night"],
            }
            anomaly_id = build_anomaly_id(self.rule_id, [entity["entity_id"]], evidence)
            findings.append(
                AnomalyFinding(
                    anomaly_id=anomaly_id,
                    type=self.rule_id,
                    severity="high",
                    confidence=0.7,
                    triggering_entities=[entity["entity_id"]],
                    evidence=evidence,
                    suggested_actions=["lock_entity"],
                    is_sensitive=True,
                )
            )
        return findings


def _is_exterior_hint(entity: SnapshotEntity) -> bool:
    area = (entity["area"] or "").lower()
    name = (entity["friendly_name"] or "").lower()
    return any(hint in area or hint in name for hint in EXTERIOR_HINTS)

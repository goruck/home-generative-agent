"""Rule: unlocked exterior lock at night."""

from __future__ import annotations

from typing import TYPE_CHECKING

from homeassistant.util import dt as dt_util

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
    "main",
    "side",
    "yard",
    "alley",
    "carport",
    "driveway",
    "outdoor",
)


class UnlockedLockAtNightRule:
    """Detect unlocked exterior locks at night."""

    rule_id = "unlocked_lock_at_night"
    requires: frozenset[str] = frozenset()
    cooldown_minutes = 0

    def evaluate(self, snapshot: FullStateSnapshot) -> list[AnomalyFinding]:
        """Return findings for exterior locks left unlocked at night."""
        if not snapshot["derived"]["is_night"]:
            return []

        anyone_home: bool = bool(snapshot["derived"].get("anyone_home", False))

        findings: list[AnomalyFinding] = []
        for entity in snapshot["entities"]:
            if entity["domain"] != "lock":
                continue
            if entity["state"] != "unlocked":
                continue
            if not _is_exterior_hint(entity):
                continue
            if _on_device_away_from_home(entity):
                continue

            evidence = {
                "entity_id": entity["entity_id"],
                "area": entity["area"],
                "friendly_name": entity["friendly_name"],
                "state": entity["state"],
                "last_changed": entity["last_changed"],
                "is_night": snapshot["derived"]["is_night"],
                "anyone_home": anyone_home,
            }
            anomaly_id = build_anomaly_id(self.rule_id, [entity["entity_id"]], evidence)
            findings.append(
                AnomalyFinding(
                    anomaly_id=anomaly_id,
                    type=self.rule_id,
                    severity="low" if anyone_home else "high",
                    confidence=0.7,
                    triggering_entities=[entity["entity_id"]],
                    evidence=evidence,
                    suggested_actions=["lock_entity"],
                    is_sensitive=True,  # lock location is always sensitive
                )
            )
        return findings


def _is_exterior_hint(entity: SnapshotEntity) -> bool:
    area = (entity["area"] or "").lower()
    name = (entity["friendly_name"] or "").lower()
    return any(hint in area or hint in name for hint in EXTERIOR_HINTS)


def _on_device_away_from_home(entity: SnapshotEntity) -> bool:
    """
    Return True for a lock on a device its GPS tracker places away from home.

    A car's door lock while the car is away is not a way into the house
    (#1088). The skip holds only while the tracker's report is at least as new
    as the lock's last change: a car whose cloud location still reads away
    after it came home, and was unlocked there since, alerts. A lock with no
    usable change time alerts too.
    """
    away_since = dt_util.parse_datetime(entity.get("device_away_since") or "")
    if away_since is None:
        return False
    changed = dt_util.parse_datetime(entity.get("last_changed") or "")
    if changed is None:
        return False
    return away_since >= changed

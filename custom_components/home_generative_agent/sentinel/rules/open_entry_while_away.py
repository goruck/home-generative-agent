"""Rule: open door/window while everyone is away."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from custom_components.home_generative_agent.sentinel.models import (
    AnomalyFinding,
    build_anomaly_id,
)

LOGGER = logging.getLogger(__name__)

if TYPE_CHECKING:
    from custom_components.home_generative_agent.snapshot.schema import (
        FullStateSnapshot,
    )

_ENTRY_CLASSES = {"door", "window", "opening"}


class OpenEntryWhileAwayRule:
    """Detect open entry sensors while the home is away."""

    rule_id = "open_entry_while_away"
    requires: frozenset[str] = frozenset()
    cooldown_minutes = 0

    def evaluate(self, snapshot: FullStateSnapshot) -> list[AnomalyFinding]:
        """Return findings for open entries while nobody is home."""
        if snapshot["derived"]["anyone_home"]:
            return []

        findings: list[AnomalyFinding] = []
        for entity in snapshot["entities"]:
            if entity["domain"] != "binary_sensor":
                continue
            device_class = entity["attributes"].get("device_class")
            if device_class not in _ENTRY_CLASSES:
                continue
            if entity["state"] != "on":
                continue
            # A car's door, window or lid while the car is away is not an
            # entry of the house (#1089). Parked at home, it still alerts.
            if entity.get("device_away_from_home"):
                LOGGER.debug(
                    "Skipping %s: its device's GPS tracker is away from home.",
                    entity["entity_id"],
                )
                continue

            evidence = {
                "entity_id": entity["entity_id"],
                "area": entity["area"],
                "device_class": device_class,
                "state": entity["state"],
                "last_changed": entity["last_changed"],
                "anyone_home": snapshot["derived"]["anyone_home"],
            }
            anomaly_id = build_anomaly_id(self.rule_id, [entity["entity_id"]], evidence)
            findings.append(
                AnomalyFinding(
                    anomaly_id=anomaly_id,
                    type=self.rule_id,
                    severity="high",
                    confidence=0.65,
                    triggering_entities=[entity["entity_id"]],
                    evidence=evidence,
                    suggested_actions=["close_entry"],
                    is_sensitive=True,
                )
            )
        return findings

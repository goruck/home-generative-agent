"""Rule: open door/window while everyone is away."""

from __future__ import annotations

from ...snapshot.schema import FullStateSnapshot
from ..models import AnomalyFinding, build_anomaly_id

_ENTRY_CLASSES = {"door", "window", "opening"}


class OpenEntryWhileAwayRule:
    """Detect open entry sensors while the home is away."""

    rule_id = "open_entry_while_away"

    def evaluate(self, snapshot: FullStateSnapshot) -> list[AnomalyFinding]:
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

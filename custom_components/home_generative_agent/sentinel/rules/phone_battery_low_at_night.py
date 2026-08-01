"""Rule: phone battery low at night while home is occupied."""

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

_PHONE_KEYWORDS: frozenset[str] = frozenset(
    {
        "phone",
        "iphone",
        "android",
        "pixel",
        "galaxy",
        "mobile",
        "smartphone",
        "handset",
    }
)

_LOW_BATTERY_THRESHOLD = 20


class PhoneBatteryLowAtNightRule:
    """Detect phone batteries that are low at night while the home is occupied."""

    rule_id = "phone_battery_low_at_night_home"

    def evaluate(self, snapshot: FullStateSnapshot) -> list[AnomalyFinding]:
        """Return findings for phone batteries below threshold at night while home."""
        if not snapshot["derived"]["is_night"]:
            return []
        if not snapshot["derived"]["anyone_home"]:
            return []

        findings: list[AnomalyFinding] = []
        for entity in snapshot["entities"]:
            if not _is_phone_battery_sensor(entity):
                continue
            try:
                level = float(entity["state"])
            except (ValueError, TypeError):
                continue
            if level >= _LOW_BATTERY_THRESHOLD:
                continue

            evidence = {
                "entity_id": entity["entity_id"],
                "battery_level": level,
                "friendly_name": entity.get("friendly_name"),
                "area": entity.get("area"),
                "is_night": snapshot["derived"]["is_night"],
                "anyone_home": snapshot["derived"]["anyone_home"],
            }
            anomaly_id = build_anomaly_id(self.rule_id, [entity["entity_id"]], evidence)
            findings.append(
                AnomalyFinding(
                    anomaly_id=anomaly_id,
                    type=self.rule_id,
                    severity="low",
                    confidence=0.7,
                    triggering_entities=[entity["entity_id"]],
                    evidence=evidence,
                    suggested_actions=["charge_device"],
                    is_sensitive=False,
                )
            )
        return findings


def _is_phone_battery_sensor(entity: SnapshotEntity) -> bool:
    """Return True if entity is a phone battery sensor."""
    if entity["domain"] != "sensor":
        return False
    if entity.get("attributes", {}).get("device_class") != "battery":
        return False
    searchable = " ".join(
        filter(
            None,
            [
                entity["entity_id"].lower(),
                (entity.get("friendly_name") or "").lower(),
            ],
        )
    )
    return any(kw in searchable for kw in _PHONE_KEYWORDS)

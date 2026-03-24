"""Rule: alarm disarmed while unknown person detected on outdoor camera."""

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

_DISARMED_STATE = "disarmed"


class AlarmDisarmedDuringExternalThreatRule:
    """Detect when alarm is disarmed while an unknown person is on outdoor cameras."""

    rule_id = "alarm_disarmed_during_external_threat"

    def evaluate(self, snapshot: FullStateSnapshot) -> list[AnomalyFinding]:
        """Return a finding when alarm is disarmed and unknown person is on camera."""
        # Collect all alarm_control_panel entities that are disarmed.
        # Scans generically — no hardcoded entity_id required.
        disarmed_panels = [
            e
            for e in snapshot["entities"]
            if e["domain"] == "alarm_control_panel" and e["state"] == _DISARMED_STATE
        ]
        if not disarmed_panels:
            return []

        disarmed_panel_ids = [e["entity_id"] for e in disarmed_panels]
        primary_alarm_id = disarmed_panel_ids[0]

        # Look for outdoor cameras with activity and no recognized people.
        findings: list[AnomalyFinding] = []
        for activity in snapshot["camera_activity"]:
            # Skip cameras with no recent activity.
            if (
                not activity.get("last_activity")
                and not activity.get("motion_entities")
                and not activity.get("vmd_entities")
                and not activity.get("snapshot_summary")
            ):
                continue
            # Skip cameras where the person is recognized (not a threat).
            if activity.get("recognized_people"):
                continue

            cam = activity["camera_entity_id"]
            evidence = {
                "camera_entity_id": cam,
                "area": activity.get("area"),
                "alarm_entity_id": primary_alarm_id,
                "alarm_entity_ids": disarmed_panel_ids,
                "alarm_state": _DISARMED_STATE,
                "recognized_people": activity.get("recognized_people", []),
                "last_activity": activity.get("last_activity"),
                "snapshot_summary": activity.get("snapshot_summary"),
            }
            anomaly_id = build_anomaly_id(
                self.rule_id, [primary_alarm_id, cam], evidence
            )
            findings.append(
                AnomalyFinding(
                    anomaly_id=anomaly_id,
                    type=self.rule_id,
                    severity="low",
                    confidence=0.9,
                    triggering_entities=[primary_alarm_id, cam],
                    evidence=evidence,
                    suggested_actions=["arm_alarm"],
                    is_sensitive=False,
                )
            )
        return findings

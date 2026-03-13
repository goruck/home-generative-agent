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

_ALARM_ENTITY_ID = "alarm_control_panel.home_alarm"
_DISARMED_STATE = "disarmed"


class AlarmDisarmedDuringExternalThreatRule:
    """Detect when alarm is disarmed while an unknown person is on outdoor cameras."""

    rule_id = "alarm_disarmed_during_external_threat"

    def evaluate(self, snapshot: FullStateSnapshot) -> list[AnomalyFinding]:
        """Return a finding when alarm is disarmed and unknown person is on camera."""
        # Check alarm state.
        alarm_state: str | None = None
        for entity in snapshot["entities"]:
            if entity["entity_id"] == _ALARM_ENTITY_ID:
                alarm_state = entity["state"]
                break

        if alarm_state is None or alarm_state != _DISARMED_STATE:
            return []

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
                "alarm_entity_id": _ALARM_ENTITY_ID,
                "alarm_state": alarm_state,
                "recognized_people": activity.get("recognized_people", []),
                "last_activity": activity.get("last_activity"),
                "snapshot_summary": activity.get("snapshot_summary"),
            }
            anomaly_id = build_anomaly_id(
                self.rule_id, [_ALARM_ENTITY_ID, cam], evidence
            )
            findings.append(
                AnomalyFinding(
                    anomaly_id=anomaly_id,
                    type=self.rule_id,
                    severity="low",
                    confidence=0.9,
                    triggering_entities=[_ALARM_ENTITY_ID, cam],
                    evidence=evidence,
                    suggested_actions=["close_entry"],
                    is_sensitive=False,
                )
            )
        return findings

"""Rule: alarm disarmed while unknown person detected on outdoor camera."""

from __future__ import annotations

from typing import TYPE_CHECKING

from homeassistant.util import dt as dt_util

from custom_components.home_generative_agent.const import (
    SENTINEL_CAMERA_ACTIVITY_STALENESS_MINUTES,
)
from custom_components.home_generative_agent.sentinel.models import (
    AnomalyFinding,
    build_anomaly_id,
)

if TYPE_CHECKING:
    from custom_components.home_generative_agent.snapshot.schema import (
        FullStateSnapshot,
    )

_DISARMED_STATE = "disarmed"


def _minutes_between(earlier_iso: str | None, later_iso: str | None) -> float | None:
    """Return elapsed minutes from *earlier_iso* to *later_iso*, or None."""
    if not earlier_iso or not later_iso:
        return None
    t_earlier = dt_util.parse_datetime(earlier_iso)
    t_later = dt_util.parse_datetime(later_iso)
    if t_earlier is None or t_later is None:
        return None
    return max(0.0, (t_later - t_earlier).total_seconds() / 60.0)


class AlarmDisarmedDuringExternalThreatRule:
    """Detect alarm disarmed while unrecognized outdoor camera activity is present."""

    rule_id = "alarm_disarmed_during_external_threat"

    def evaluate(self, snapshot: FullStateSnapshot) -> list[AnomalyFinding]:
        """Return a finding for disarmed alarm with unrecognized camera activity."""
        disarmed_panels = [
            e
            for e in snapshot["entities"]
            if e["domain"] == "alarm_control_panel" and e["state"] == _DISARMED_STATE
        ]
        if not disarmed_panels:
            return []

        disarmed_panel_ids = [e["entity_id"] for e in disarmed_panels]
        primary_alarm = disarmed_panels[0]
        primary_alarm_id = primary_alarm["entity_id"]
        generated_at = snapshot["generated_at"]

        entity_map = {e["entity_id"]: e for e in snapshot["entities"]}

        findings: list[AnomalyFinding] = []
        for activity in snapshot["camera_activity"]:
            has_other_evidence = bool(
                activity.get("motion_entities")
                or activity.get("vmd_entities")
                or activity.get("snapshot_summary")
            )
            if activity.get("recognized_people"):
                continue

            # Compute activity age. An unparseable timestamp is treated the same as
            # absent — the gate falls through to has_other_evidence.
            camera_activity_age_minutes: float | None = None
            if activity.get("last_activity"):
                camera_activity_age_minutes = _minutes_between(
                    activity["last_activity"], generated_at
                )

            last_activity_reliable = camera_activity_age_minutes is not None
            if not last_activity_reliable and not has_other_evidence:
                continue

            # Staleness gate: skip when reliable age exceeds the threshold.
            threshold = SENTINEL_CAMERA_ACTIVITY_STALENESS_MINUTES
            if (
                last_activity_reliable
                and camera_activity_age_minutes is not None
                and camera_activity_age_minutes > threshold
            ):
                continue

            cam = activity["camera_entity_id"]

            cam_entity = entity_map.get(cam)
            camera_friendly_name: str | None = (
                cam_entity.get("friendly_name") if cam_entity else None
            )
            alarm_friendly_name: str | None = primary_alarm.get("friendly_name")

            disarm_duration_minutes = _minutes_between(
                primary_alarm["last_changed"], generated_at
            )

            # Stable identity fields — used for the anomaly ID and cooldown key.
            # Must not include volatile display fields like age-in-minutes.
            id_evidence = {
                "camera_entity_id": cam,
                "alarm_entity_id": primary_alarm_id,
                "alarm_state": _DISARMED_STATE,
                "last_activity": activity.get("last_activity"),
                "alarm_last_changed": primary_alarm["last_changed"],
            }
            anomaly_id = build_anomaly_id(
                self.rule_id, [primary_alarm_id, cam], id_evidence
            )

            # Full evidence for notification, audit, and explain rendering.
            evidence = {
                **id_evidence,
                "area": activity.get("area"),
                "alarm_entity_ids": disarmed_panel_ids,
                "camera_friendly_name": camera_friendly_name,
                "alarm_friendly_name": alarm_friendly_name,
                "recognized_people": activity.get("recognized_people", []),
                "snapshot_summary": activity.get("snapshot_summary"),
                "camera_activity_age_minutes": camera_activity_age_minutes,
                "disarm_duration_minutes": disarm_duration_minutes,
                # Explicitly null — only true indoor motion/occupancy sensors qualify.
                # Do not derive from anyone_home, people_home, or person trackers.
                "indoor_occupancy_signal": None,
            }
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

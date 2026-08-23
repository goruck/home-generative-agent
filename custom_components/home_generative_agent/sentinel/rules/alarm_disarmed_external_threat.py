"""Rule: alarm disarmed while unknown person detected on outdoor camera."""

from __future__ import annotations

from typing import TYPE_CHECKING

from custom_components.home_generative_agent.sentinel.models import (
    AnomalyFinding,
    build_anomaly_id,
    minutes_between,
    sighting_timestamp,
    unknown_person_sighting_is_actionable,
)

if TYPE_CHECKING:
    from custom_components.home_generative_agent.snapshot.schema import (
        FullStateSnapshot,
    )

_DISARMED_STATE = "disarmed"


class AlarmDisarmedDuringExternalThreatRule:
    """Detect alarm disarmed while an unknown person is seen on a camera."""

    rule_id = "alarm_disarmed_during_external_threat"

    def evaluate(self, snapshot: FullStateSnapshot) -> list[AnomalyFinding]:
        """Return a finding for a disarmed alarm with a fresh stranger sighting."""
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
            # Fresh, unaccompanied stranger sighting — see the helper for the
            # full predicate rationale (reserved labels, companion
            # suppression, staleness).
            if not unknown_person_sighting_is_actionable(activity, generated_at):
                continue
            # Always a float here: the actionability gate above required a
            # fresh, parseable sighting timestamp.
            camera_activity_age_minutes = minutes_between(
                sighting_timestamp(activity), generated_at
            )

            cam = activity["camera_entity_id"]

            cam_entity = entity_map.get(cam)
            camera_friendly_name: str | None = (
                cam_entity.get("friendly_name") if cam_entity else None
            )
            alarm_friendly_name: str | None = primary_alarm.get("friendly_name")

            # A far-future alarm last_changed (panel clock skew) yields None
            # here rather than the old clamp-to-0 — consumers render the
            # phrase without a duration in that case.
            disarm_duration_minutes = minutes_between(
                primary_alarm["last_changed"], generated_at
            )

            # Stable identity fields — used for the anomaly ID and cooldown key.
            # Must not include volatile display fields like age-in-minutes.
            # Anchored on the sighting timestamp the rule fires on, so an
            # unrelated motion refresh of last_activity cannot mint a new
            # anomaly id for the same sighting.
            id_evidence = {
                "camera_entity_id": cam,
                "alarm_entity_id": primary_alarm_id,
                "alarm_state": _DISARMED_STATE,
                "sighting_last_event": sighting_timestamp(activity),
                "alarm_last_changed": primary_alarm["last_changed"] or None,
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
                "recognized_people": list(activity.get("recognized_people") or []),
                "recognition_last_event": activity.get("recognition_last_event"),
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

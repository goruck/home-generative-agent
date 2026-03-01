"""Rule: camera activity near unsecured entry."""

from __future__ import annotations

from datetime import timedelta
from typing import TYPE_CHECKING

from homeassistant.util import dt as dt_util

from custom_components.home_generative_agent.sentinel.models import (
    AnomalyFinding,
    build_anomaly_id,
)

if TYPE_CHECKING:
    from custom_components.home_generative_agent.snapshot.schema import (
        FullStateSnapshot,
    )

ENTRY_CLASSES = {"door", "window", "opening"}
ACTIVITY_WINDOW_MIN = 10


class CameraEntryUnsecuredRule:
    """Detect camera activity while nearby entries are unsecured."""

    rule_id = "camera_entry_unsecured"

    def evaluate(self, snapshot: FullStateSnapshot) -> list[AnomalyFinding]:  # noqa: PLR0912
        """Return findings for recent camera activity near unsecured entries."""
        findings: list[AnomalyFinding] = []
        now = dt_util.parse_datetime(snapshot["derived"]["now"]) or dt_util.utcnow()
        window = timedelta(minutes=ACTIVITY_WINDOW_MIN)

        unsecured_by_area: dict[str, list[str]] = {}
        for entity in snapshot["entities"]:
            area = entity.get("area")
            if not area:
                continue
            if entity["domain"] == "lock" and entity["state"] == "unlocked":
                unsecured_by_area.setdefault(area, []).append(entity["entity_id"])
                continue
            if entity["domain"] != "binary_sensor":
                continue
            if entity["attributes"].get("device_class") not in ENTRY_CLASSES:
                continue
            if entity["state"] != "on":
                continue
            unsecured_by_area.setdefault(area, []).append(entity["entity_id"])

        # All unsecured entities across the home, used as a fallback for
        # cameras whose area (e.g. "Outside") contains no entry-point sensors.
        all_unsecured: list[str] = sorted(
            {e for entities in unsecured_by_area.values() for e in entities}
        )

        # Index entity last_changed by entity_id for VMD/motion fallback lookup.
        last_changed_by_id: dict[str, str] = {
            e["entity_id"]: e["last_changed"] for e in snapshot["entities"]
        }

        for activity in snapshot["camera_activity"]:
            area = activity.get("area")
            if not area:
                continue
            last_activity = activity.get("last_activity")
            if not last_activity:
                # Camera has no activity timestamp attribute; use the most
                # recent last_changed of its associated VMD/motion sensors.
                sensor_ids = activity.get("vmd_entities", []) + activity.get(
                    "motion_entities", []
                )
                candidates = [
                    last_changed_by_id[sid]
                    for sid in sensor_ids
                    if sid in last_changed_by_id
                ]
                if not candidates:
                    # No linked sensors in camera_activity (camera doesn't
                    # advertise vmd_entity_id etc.); scan all binary sensors
                    # in the same area as a last resort.  Device-class is not
                    # checked because VMD sensors (e.g. Hikvision) often have
                    # no device_class; the area constraint is sufficient.
                    candidates = [
                        e["last_changed"]
                        for e in snapshot["entities"]
                        if e.get("area") == area
                        and e["domain"] == "binary_sensor"
                    ]
                last_activity = max(candidates) if candidates else None
            if not last_activity:
                continue
            last_dt = dt_util.parse_datetime(last_activity)
            if last_dt is None:
                continue
            if now - last_dt > window:
                continue
            # Prefer same-area unsecured entities; fall back to home-wide list
            # for exterior cameras whose area has no entry-point sensors.
            unsecured = unsecured_by_area.get(area) or all_unsecured
            if not unsecured:
                continue
            evidence = {
                "camera_entity_id": activity["camera_entity_id"],
                "area": area,
                "last_activity": last_activity,
                "unsecured_entities": sorted(unsecured),
            }
            anomaly_id = build_anomaly_id(
                self.rule_id, [activity["camera_entity_id"]], evidence
            )
            findings.append(
                AnomalyFinding(
                    anomaly_id=anomaly_id,
                    type=self.rule_id,
                    severity="high",
                    confidence=0.6,
                    triggering_entities=[activity["camera_entity_id"]],
                    evidence=evidence,
                    suggested_actions=["check_entry"],
                    is_sensitive=True,
                ),
            )
        return findings

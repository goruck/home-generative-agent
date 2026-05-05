"""Rule: camera activity near unsecured entry."""

from __future__ import annotations

import logging
from datetime import timedelta
from typing import TYPE_CHECKING

from homeassistant.util import dt as dt_util

from custom_components.home_generative_agent.sentinel.models import (
    AnomalyFinding,
    build_anomaly_id,
)

LOGGER = logging.getLogger(__name__)

if TYPE_CHECKING:
    from custom_components.home_generative_agent.snapshot.schema import (
        FullStateSnapshot,
        SnapshotEntity,
    )

ENTRY_CLASSES = {"door", "window", "opening"}
ACTIVITY_WINDOW_MIN = 10


class CameraEntryUnsecuredRule:
    """Detect camera activity while nearby entries are unsecured."""

    rule_id = "camera_entry_unsecured"

    def __init__(
        self,
        camera_entry_links: dict[str, list[str]] | None = None,
    ) -> None:
        """Initialise rule with optional cross-area camera→entry links."""
        self._camera_entry_links: dict[str, list[str]] = camera_entry_links or {}

    def evaluate(self, snapshot: FullStateSnapshot) -> list[AnomalyFinding]:  # noqa: PLR0912, PLR0915
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

        # Reverse map: entity_id → area, used to populate unsecured_entity_areas
        # in evidence so the LLM and correlator know where each entity lives.
        # Built from ALL snapshot entities (not just unsecured ones) so that
        # cross-area linked entities have their area available.
        all_entity_area_map: dict[str, str] = {
            e["entity_id"]: e.get("area") or "unknown" for e in snapshot["entities"]
        }

        # Index entities by id for linked-entity unsecured lookup.
        entity_by_id: dict[str, SnapshotEntity] = {
            e["entity_id"]: e for e in snapshot["entities"]
        }

        # Index entity last_changed by entity_id for VMD/motion fallback lookup.
        last_changed_by_id: dict[str, str] = {
            e["entity_id"]: e["last_changed"] for e in snapshot["entities"]
        }

        skip_counts = {
            "no_area": 0,
            "no_activity": 0,
            "parse_error": 0,
            "outside_window": 0,
            "missing_linked_entry": 0,
        }
        fallback_sensor_candidates = 0
        for activity in snapshot["camera_activity"]:
            cam = activity["camera_entity_id"]
            area = activity.get("area")
            if not area:
                skip_counts["no_area"] += 1
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
                    # checked because VMD sensors vary by manufacturer and
                    # often have no device_class; the area constraint is
                    # sufficient.
                    candidates = [
                        e["last_changed"]
                        for e in snapshot["entities"]
                        if e.get("area") == area and e["domain"] == "binary_sensor"
                    ]
                    fallback_sensor_candidates += len(candidates)
                last_activity = max(candidates) if candidates else None
            if not last_activity:
                skip_counts["no_activity"] += 1
                continue
            last_dt = dt_util.parse_datetime(last_activity)
            if last_dt is None:
                skip_counts["parse_error"] += 1
                continue
            if now - last_dt > window:
                skip_counts["outside_window"] += 1
                continue
            # Same-area unsecured entries (primary spatial relationship).
            unsecured_same_area: list[str] = list(unsecured_by_area.get(area) or [])

            # Cross-area linked entries: fire when camera.entity_id has an explicit
            # entry link configured via sentinel_camera_entry_links.  This covers
            # cameras that physically overlook an entry in a different HA area
            # (e.g. driveway camera → front door in "Front" area).
            unsecured_linked: list[str] = []
            for linked_eid in self._camera_entry_links.get(cam, []):
                entity = entity_by_id.get(linked_eid)
                if entity is None:
                    skip_counts["missing_linked_entry"] += 1
                    continue
                domain = entity["domain"]
                state = entity["state"]
                if (domain == "lock" and state == "unlocked") or (
                    domain == "binary_sensor"
                    and entity["attributes"].get("device_class") in ENTRY_CLASSES
                    and state == "on"
                ):
                    unsecured_linked.append(linked_eid)

            # Merge: same-area first, then linked (preserving order, deduplicating).
            same_area_set = set(unsecured_same_area)
            unsecured_all: list[str] = unsecured_same_area + [
                eid for eid in unsecured_linked if eid not in same_area_set
            ]

            if not unsecured_all:
                continue

            evidence = {
                "camera_entity_id": activity["camera_entity_id"],
                "area": area,  # kept for correlator Rule 1 compatibility
                "camera_area": area,  # explicit field for LLM spatial grounding
                "last_activity": last_activity,
                "unsecured_entities": sorted(unsecured_all),
                # Iterate sorted order so unsecured_entity_areas key order matches
                # unsecured_entities list order — consistent view for the LLM.
                "unsecured_entity_areas": {
                    eid: all_entity_area_map.get(eid, "unknown")
                    for eid in sorted(unsecured_all)
                },
            }
            # Hash only same-area unsecured entities so that changing the link
            # config does not invalidate suppression state for existing findings.
            anomaly_id = build_anomaly_id(
                self.rule_id,
                [activity["camera_entity_id"]],
                {
                    "camera_entity_id": activity["camera_entity_id"],
                    "area": area,
                    "last_activity": last_activity,
                    "unsecured_entities": sorted(unsecured_same_area),
                },
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
        if LOGGER.isEnabledFor(logging.DEBUG) and (
            findings or any(skip_counts.values())
        ):
            LOGGER.debug(
                "Camera-entry rule evaluated %d camera(s): findings=%d "
                "skips=%s fallback_sensor_candidates=%d.",
                len(snapshot["camera_activity"]),
                len(findings),
                skip_counts,
                fallback_sensor_candidates,
            )
        return findings

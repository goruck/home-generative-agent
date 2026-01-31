"""Snapshot schema for authoritative home state."""

from __future__ import annotations

from typing import Any, TypedDict, cast

import voluptuous as vol

SNAPSHOT_SCHEMA_VERSION = 1


class SnapshotEntity(TypedDict):
    """Serialized Home Assistant entity state."""

    entity_id: str
    domain: str
    state: str
    friendly_name: str | None
    area: str | None
    attributes: dict[str, Any]
    last_changed: str
    last_updated: str


class CameraActivity(TypedDict):
    """Serialized camera activity metadata."""

    camera_entity_id: str
    area: str | None
    last_activity: str | None
    motion_entities: list[str]
    vmd_entities: list[str]
    snapshot_summary: str | None
    recognized_people: list[str]
    latest_path: str | None


class DerivedContext(TypedDict):
    """Derived context for snapshot consumers."""

    now: str
    timezone: str
    is_night: bool
    anyone_home: bool
    last_motion_by_area: dict[str, str]


class FullStateSnapshot(TypedDict):
    """Full structured snapshot of home state."""

    schema_version: int
    generated_at: str
    entities: list[SnapshotEntity]
    camera_activity: list[CameraActivity]
    derived: DerivedContext


SNAPSHOT_SCHEMA = vol.Schema(
    {
        vol.Required("schema_version"): int,
        vol.Required("generated_at"): str,
        vol.Required("entities"): [
            {
                vol.Required("entity_id"): str,
                vol.Required("domain"): str,
                vol.Required("state"): str,
                vol.Required("friendly_name"): vol.Any(str, None),
                vol.Required("area"): vol.Any(str, None),
                vol.Required("attributes"): dict,
                vol.Required("last_changed"): str,
                vol.Required("last_updated"): str,
            }
        ],
        vol.Required("camera_activity"): [
            {
                vol.Required("camera_entity_id"): str,
                vol.Required("area"): vol.Any(str, None),
                vol.Required("last_activity"): vol.Any(str, None),
                vol.Required("motion_entities"): [str],
                vol.Required("vmd_entities"): [str],
                vol.Required("snapshot_summary"): vol.Any(str, None),
                vol.Required("recognized_people"): [str],
                vol.Required("latest_path"): vol.Any(str, None),
            }
        ],
        vol.Required("derived"): {
            vol.Required("now"): str,
            vol.Required("timezone"): str,
            vol.Required("is_night"): bool,
            vol.Required("anyone_home"): bool,
            vol.Required("last_motion_by_area"): dict,
        },
    }
)


def validate_snapshot(snapshot: dict[str, Any]) -> FullStateSnapshot:
    """Validate and return a snapshot using the canonical schema."""
    validated = SNAPSHOT_SCHEMA(snapshot)
    return cast(FullStateSnapshot, validated)

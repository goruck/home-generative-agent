"""Full state snapshot builder."""

from __future__ import annotations

from collections.abc import Iterable
from datetime import datetime
from typing import TYPE_CHECKING, Any

from homeassistant.helpers import area_registry as ar
from homeassistant.helpers import entity_registry as er
from homeassistant.util import dt as dt_util

from .camera_activity import extract_camera_activity
from .derived import derive_context
from .schema import (
    SNAPSHOT_SCHEMA_VERSION,
    FullStateSnapshot,
    SnapshotEntity,
    validate_snapshot,
)

if TYPE_CHECKING:
    from homeassistant.core import HomeAssistant, State


def _as_iso(value: datetime) -> str:
    return dt_util.as_utc(value).isoformat()


def _jsonify(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, datetime):
        return _as_iso(value)
    if isinstance(value, dict):
        return {str(k): _jsonify(v) for k, v in value.items()}
    if isinstance(value, Iterable) and not isinstance(value, (bytes, bytearray)):
        return [_jsonify(v) for v in value]
    return str(value)


def _build_entity_snapshot(state: State, area_name: str | None) -> SnapshotEntity:
    return {
        "entity_id": state.entity_id,
        "domain": state.domain,
        "state": state.state,
        "friendly_name": state.attributes.get("friendly_name"),
        "area": area_name,
        "attributes": _jsonify(state.attributes),
        "last_changed": _as_iso(state.last_changed),
        "last_updated": _as_iso(state.last_updated),
    }


def _build_area_lookup(hass: HomeAssistant) -> dict[str, str | None]:
    entity_registry = er.async_get(hass)
    area_registry = ar.async_get(hass)
    area_names = {area_id: area.name for area_id, area in area_registry.areas.items()}

    lookup: dict[str, str | None] = {}
    for entity_id, entry in entity_registry.entities.items():
        area_id = entry.area_id
        lookup[entity_id] = area_names.get(area_id) if area_id is not None else None
    return lookup


async def async_build_full_state_snapshot(hass: HomeAssistant) -> FullStateSnapshot:
    """Build a deterministic full state snapshot."""
    now = dt_util.now()
    timezone = hass.config.time_zone or str(dt_util.DEFAULT_TIME_ZONE)
    states = hass.states.async_all()
    area_lookup = _build_area_lookup(hass)
    image_states = hass.states.async_all("image")
    image_by_camera_id: dict[str, State] = {}
    for image_state in image_states:
        camera_id = image_state.attributes.get("camera_id")
        if isinstance(camera_id, str):
            image_by_camera_id[camera_id] = image_state

    entities = [
        _build_entity_snapshot(state, area_lookup.get(state.entity_id))
        for state in states
    ]
    entities.sort(key=lambda item: item["entity_id"])

    camera_activity = []
    for state in states:
        if state.domain != "camera":
            continue
        camera_activity.append(
            extract_camera_activity(
                state,
                area_lookup.get(state.entity_id),
                image_by_camera_id.get(state.entity_id),
            )
        )
    camera_activity.sort(key=lambda item: item["camera_entity_id"])

    derived = derive_context(
        now=now,
        timezone=timezone,
        sun_state=hass.states.get("sun.sun"),
        all_states=states,
        area_lookup=area_lookup,
    )

    snapshot: dict[str, Any] = {
        "schema_version": SNAPSHOT_SCHEMA_VERSION,
        "generated_at": _as_iso(now),
        "entities": entities,
        "camera_activity": camera_activity,
        "derived": derived,
    }

    return validate_snapshot(snapshot)

"""Full state snapshot builder."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING, Any

from homeassistant.helpers import area_registry as ar
from homeassistant.helpers import device_registry as dr
from homeassistant.helpers import entity_registry as er
from homeassistant.util import dt as dt_util

from .camera_activity import extract_camera_activity
from .derived import derive_context
from .network import NetworkBuildContext, async_build_network_snapshot
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


def _build_entity_snapshot(
    state: State, area_name: str | None, platform: str | None
) -> SnapshotEntity:
    return {
        "entity_id": state.entity_id,
        "domain": state.domain,
        "state": state.state,
        "friendly_name": state.attributes.get("friendly_name"),
        "area": area_name,
        "attributes": _jsonify(state.attributes),
        "last_changed": _as_iso(state.last_changed),
        "last_updated": _as_iso(state.last_updated),
        "platform": platform,
    }


@dataclass
class RegistryLookups:
    """Per-entity registry facts resolved once per snapshot build."""

    area: dict[str, str | None] = field(default_factory=dict)
    platform: dict[str, str | None] = field(default_factory=dict)
    entity_device: dict[str, str] = field(default_factory=dict)
    # device_id -> set of entity domains the device owns (for tagging a
    # device as a security device when one of its entities is a lock,
    # alarm panel, or camera).
    device_domains: dict[str, set[str]] = field(default_factory=dict)


def _build_registry_lookups(hass: HomeAssistant) -> RegistryLookups:
    entity_registry = er.async_get(hass)
    device_registry = dr.async_get(hass)
    area_registry = ar.async_get(hass)
    area_names = {area_id: area.name for area_id, area in area_registry.areas.items()}

    lookups = RegistryLookups()
    for entity_id, entry in entity_registry.entities.items():
        # Entity-level area takes precedence; fall back to the parent device's
        # area, which is how areas are most commonly assigned in HA.
        area_id = entry.area_id
        if area_id is None and entry.device_id is not None:
            # async_get rather than devices.get: HA 2026.9 turned ``devices`` into
            # a deprecated view that logs on every access and types as Collection.
            device = device_registry.async_get(entry.device_id)
            if device is not None:
                area_id = device.area_id
        lookups.area[entity_id] = (
            area_names.get(area_id) if area_id is not None else None
        )
        lookups.platform[entity_id] = entry.platform
        if entry.device_id is not None:
            lookups.entity_device[entity_id] = entry.device_id
            lookups.device_domains.setdefault(entry.device_id, set()).add(entry.domain)
    return lookups


async def async_build_full_state_snapshot(
    hass: HomeAssistant, *, network: NetworkBuildContext | None = None
) -> FullStateSnapshot:
    """
    Build a deterministic full state snapshot.

    ``network`` carries what the network section needs beyond ``hass`` (the
    effective options and the auth inventory); the Sentinel engine passes it,
    other callers get a section built with defaults.
    """
    now = dt_util.now()
    timezone = hass.config.time_zone or str(dt_util.DEFAULT_TIME_ZONE)
    states = hass.states.async_all()
    lookups = _build_registry_lookups(hass)
    area_lookup = lookups.area
    image_states = hass.states.async_all("image")
    image_by_camera_id: dict[str, State] = {}
    for image_state in image_states:
        camera_id = image_state.attributes.get("camera_id")
        if isinstance(camera_id, str):
            image_by_camera_id[camera_id] = image_state

    entities = [
        _build_entity_snapshot(
            state,
            area_lookup.get(state.entity_id),
            lookups.platform.get(state.entity_id),
        )
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

    network_section = await async_build_network_snapshot(
        hass,
        entities,
        network,
        now=now,
        entity_device=lookups.entity_device,
        device_domains=lookups.device_domains,
    )

    snapshot: dict[str, Any] = {
        "schema_version": SNAPSHOT_SCHEMA_VERSION,
        "generated_at": _as_iso(now),
        "entities": entities,
        "camera_activity": camera_activity,
        "derived": derived,
        "network": network_section,
    }

    return validate_snapshot(snapshot)

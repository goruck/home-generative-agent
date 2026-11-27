"""Full state snapshot builder."""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING, Any

from homeassistant.const import (
    ATTR_GPS_ACCURACY,
    ATTR_LATITUDE,
    ATTR_LONGITUDE,
    STATE_HOME,
    STATE_NOT_HOME,
    STATE_UNAVAILABLE,
    STATE_UNKNOWN,
)
from homeassistant.helpers import area_registry as ar
from homeassistant.helpers import device_registry as dr
from homeassistant.helpers import entity_registry as er
from homeassistant.util import dt as dt_util
from homeassistant.util.location import distance

from .camera_activity import extract_camera_activity
from .derived import derive_context
from .network import NetworkBuildContext, async_build_network_snapshot
from .router import TRACKER_DOMAIN
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


_SOURCE_TYPE_GPS = "gps"
_TRACKER_NO_READING = frozenset({STATE_UNKNOWN, STATE_UNAVAILABLE, ""})
_ZONE_DOMAIN = "zone"
_HOME_ZONE = "zone.home"
_ATTR_IN_ZONES = "in_zones"
_ATTR_RADIUS = "radius"
_ATTR_SOURCE_TYPE = "source_type"
_MAX_LATITUDE = 90.0
_MAX_LONGITUDE = 180.0


def _number(value: Any) -> float | None:
    """Return a finite real-number attribute as a float, else None."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    try:
        number = float(value)
    except OverflowError:
        return None
    return number if math.isfinite(number) else None


def _point(attributes: Mapping[str, Any]) -> tuple[float, float] | None:
    """Return a valid (latitude, longitude) from state attributes, else None."""
    lat = _number(attributes.get(ATTR_LATITUDE))
    lon = _number(attributes.get(ATTR_LONGITUDE))
    if lat is None or lon is None:
        return None
    if abs(lat) > _MAX_LATITUDE or abs(lon) > _MAX_LONGITUDE:
        return None
    return lat, lon


def _inside(
    point: tuple[float, float], zone: State | None, accuracy: float = 0.0
) -> bool | None:
    """
    Return whether ``point`` lies in ``zone``, or None when that is unknown.

    The test Home Assistant's zone helper uses: the distance to the zone's
    centre, less the GPS accuracy, is inside the radius. An unavailable zone
    or one without a usable centre and radius settles nothing.
    """
    if zone is None or zone.state == STATE_UNAVAILABLE:
        return None
    centre = _point(zone.attributes)
    radius = _number(zone.attributes.get(_ATTR_RADIUS))
    if centre is None or radius is None:
        return None
    gap = distance(point[0], point[1], centre[0], centre[1])
    if gap is None or not math.isfinite(gap):
        return None
    return gap - accuracy < radius


def _listed_zones_at_home(
    in_zones: list[Any], zones: Mapping[str, State]
) -> bool | None:
    """
    Read a tracker's ``in_zones`` list: home, away, or None when unsure.

    Home Assistant adds the enclosing zones only when it places a fix itself;
    a list the integration supplies may name a "Driveway" zone and not the home
    zone around it, so a listed zone whose centre is inside home counts as home.
    """
    home_zone = zones.get(_HOME_ZONE)
    unsure = False
    for zone_id in in_zones:
        if zone_id == _HOME_ZONE:
            return True
        zone = zones.get(zone_id) if isinstance(zone_id, str) else None
        centre = _point(zone.attributes) if zone is not None else None
        inside = _inside(centre, home_zone) if centre is not None else None
        if inside:
            return True
        if inside is None:
            unsure = True
    return None if unsure else False


def _tracker_at_home(state: State, zones: Mapping[str, State]) -> bool | None:
    """
    Return True (home), False (away) or None (no usable reading) for a tracker.

    Evidence in the order it can be trusted: Home Assistant's own ``home``;
    the fix's coordinates against the home zone; the zones the tracker lists
    (a tracker's state names only the closest zone, so a car parked in a
    "Driveway" zone inside the home radius reads "Driveway"); last, a location
    name the integration reports. An empty ``in_zones`` is what Home Assistant
    writes when it placed nothing, so it settles nothing. A reading nothing
    settles is no reading, and the lock keeps alerting (#1088).
    """
    reading = state.state.strip()
    if reading in _TRACKER_NO_READING:
        return None
    if reading == STATE_HOME:
        return True
    point = _point(state.attributes)
    if point is not None:
        accuracy = max(_number(state.attributes.get(ATTR_GPS_ACCURACY)) or 0.0, 0.0)
        inside = _inside(point, zones.get(_HOME_ZONE), accuracy)
        if inside is not None:
            return inside
    in_zones = state.attributes.get(_ATTR_IN_ZONES)
    if isinstance(in_zones, list) and in_zones:
        listed = _listed_zones_at_home(in_zones, zones)
        if listed is not None:
            return listed
    return _named_location_at_home(reading)


def _named_location_at_home(reading: str) -> bool | None:
    """Read a location name an integration reports: home, not_home, else None."""
    folded = reading.casefold()
    if folded == STATE_HOME:
        return True
    if folded == STATE_NOT_HOME:
        return False
    return None


def _devices_away_from_home(
    states: Iterable[State], entity_device: dict[str, str]
) -> set[str]:
    """
    Return the ids of devices a GPS device_tracker places away from home.

    A device counts as away when one of its GPS trackers places it away and
    none places it at home. A tracker with no usable reading says nothing, so
    a device whose only trackers have none is not away: its lock still alerts.
    The location is taken as the integration reports it; Home Assistant keeps
    no time of the last real fix, so a stale cloud location is not detected.
    """
    states = list(states)
    zones = {s.entity_id: s for s in states if s.domain == _ZONE_DOMAIN}
    away: set[str] = set()
    home: set[str] = set()
    for state in states:
        if state.domain != TRACKER_DOMAIN:
            continue
        if state.attributes.get(_ATTR_SOURCE_TYPE) != _SOURCE_TYPE_GPS:
            continue
        device_id = entity_device.get(state.entity_id)
        if device_id is None:
            continue
        at_home = _tracker_at_home(state, zones)
        if at_home is True:
            home.add(device_id)
        elif at_home is False:
            away.add(device_id)
    return away - home


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
    away_devices = _devices_away_from_home(states, lookups.entity_device)
    for entity in entities:
        if lookups.entity_device.get(entity["entity_id"]) in away_devices:
            entity["device_away_from_home"] = True

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

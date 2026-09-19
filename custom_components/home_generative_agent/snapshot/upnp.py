"""
UPnP/IGD adapter for the network snapshot (docs/network-security-plan.md, step 7).

The core ``upnp`` integration talks to the router's Internet Gateway Device
(IGD) service, which a router offers only when UPnP is on. That gives three
facts without any router-specific integration:

* **UPnP is on.** Evidence, strongest first: the SSDP discovery cache holds an
  IGD advertisement (``ssdp`` listens for these on every install); a loaded
  ``upnp`` config entry whose entities are live; or, only when the cache could
  not be read, an in-progress ``upnp`` discovery flow (Home Assistant aborts
  such a flow on a goodbye advertisement but not when the device merely
  expires, so a flow alone is not current evidence). The absence of all three
  is *not* evidence that UPnP is off (Home Assistant may sit on another
  network segment), so the capability is then reported missing with a note
  instead of asserted false.
* **The public IP address**, from the integration's external-IP sensor,
  pseudonymized before it enters the snapshot and compared with the previous
  Sentinel run to detect a change.
* **How many port mappings are open**, from the integration's port-mapping
  count sensor (disabled by default), compared with the previous run so a
  device punching a new hole through the firewall is reported.

Entity tier: the adapter reads the entity registry, entity state, config
entries and flows, and the SSDP cache through its public helper. It never
imports the ``upnp`` integration. The previous run's values come from the
engine's posture memory (persisted in the device inventory store), keyed to
the entity they were read from, so the change checks are listed as not run
on the very first run and after a source change rather than guessed.
"""

from __future__ import annotations

import asyncio
import ipaddress
import logging
import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from homeassistant.config_entries import ConfigEntryState
from homeassistant.helpers import entity_registry as er

from .network import AdapterResult, log_input_failure, sanitize_label

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from homeassistant.core import HomeAssistant

    from .network import NetworkBuildContext
    from .schema import SnapshotEntity

LOGGER = logging.getLogger(__name__)

UPNP_DOMAIN = "upnp"
# SSDP search targets of an Internet Gateway Device (UPnP Device Architecture
# 1.0 / 2.0). Kept local: importing the upnp integration's constants would
# import the integration.
IGD_SEARCH_TARGETS: tuple[str, ...] = (
    "urn:schemas-upnp-org:device:InternetGatewayDevice:1",
    "urn:schemas-upnp-org:device:InternetGatewayDevice:2",
)
# The SSDP helper is not a pure cache read: a location whose description was
# never fetched is fetched on demand, and the pinned description cache leaves
# its waiters hanging when that description fails to convert. Sentinel holds
# its run lock meanwhile, so the read is bounded.
SSDP_READ_TIMEOUT_S = 10

# Sensor keys of the upnp integration (its entity unique id is
# ``<udn>_<key>``, all in the ``sensor`` domain).
SENSOR_KEY_EXTERNAL_IP = "ip"
SENSOR_KEY_PORT_MAPPINGS = "port_mapping_number_of_entries"
SENSOR_KEYS: frozenset[str] = frozenset(
    {SENSOR_KEY_EXTERNAL_IP, SENSOR_KEY_PORT_MAPPINGS}
)
_UNKNOWN_STATES: frozenset[str] = frozenset({"", "unknown", "unavailable"})
# A UPnP port-mapping table larger than this is not a count, it is garbage.
MAX_PORT_MAPPING_COUNT = 65535

# Posture keys the engine remembers between runs for the change checks. The
# entity ids travel with the values so a value read from one gateway is never
# compared with another's.
POSTURE_MEMORY_KEYS: tuple[str, ...] = (
    "public_ip_key",
    "public_ip_entity_id",
    "upnp_port_mapping_count",
    "upnp_port_mapping_entity_id",
)
# Posture keys that describe a fact for display and never gate a rule; the
# merge step does not publish them as capabilities.
POSTURE_DISPLAY_KEYS: frozenset[str] = frozenset(
    {
        "upnp_evidence",
        "upnp_gateway_names",
        "public_ip_previous_key",
        "upnp_port_mapping_previous_count",
    }
)


@dataclass
class UpnpInputs:
    """
    What the adapter reads, collected by the async layer.

    ``igd_advertised`` is None when the SSDP cache could not be read (the
    ``ssdp`` component is not loaded, the read timed out, or the helper
    moved); False means the cache was read and holds no gateway.
    """

    igd_advertised: bool | None = None
    igd_names: list[str] = field(default_factory=list)
    # Config entries with source ``ignore`` (a dismissed discovery card) are
    # not counted: nothing is set up.
    entry_present: bool = False
    entry_loaded: bool = False
    flow_in_progress: bool = False
    # sensor key -> entity ids enabled in the registry, registry order.
    sensors: dict[str, list[str]] = field(default_factory=dict)
    # Same, for entities disabled in the registry (their state never exists).
    disabled_sensors: dict[str, list[str]] = field(default_factory=dict)
    # Every enabled upnp entity id, to tell a live entry from a dead gateway.
    entity_ids: list[str] = field(default_factory=list)


async def _collect_ssdp(inputs: UpnpInputs, hass: HomeAssistant) -> None:
    try:
        from homeassistant.components.ssdp import (  # noqa: PLC0415
            async_get_discovery_info_by_st,
        )
        from homeassistant.components.ssdp.const import (  # noqa: PLC0415
            DOMAIN as SSDP_DOMAIN,
        )
        from homeassistant.helpers.service_info.ssdp import (  # noqa: PLC0415
            ATTR_UPNP_FRIENDLY_NAME,
        )

        if SSDP_DOMAIN not in hass.config.components:
            # Discovery is off on this install (no default_config): unknown.
            return
        names: list[str] = []
        seen_usns: set[str] = set()
        async with asyncio.timeout(SSDP_READ_TIMEOUT_S):
            for target in IGD_SEARCH_TARGETS:
                for info in await async_get_discovery_info_by_st(hass, target):
                    usn = str(getattr(info, "ssdp_usn", "") or "")
                    if usn in seen_usns:
                        continue
                    seen_usns.add(usn)
                    upnp: Mapping[str, Any] = getattr(info, "upnp", None) or {}
                    name = sanitize_label(upnp.get(ATTR_UPNP_FRIENDLY_NAME))
                    names.append(name or "unnamed gateway")
        inputs.igd_advertised = bool(seen_usns)
        inputs.igd_names = sorted(set(names))
    except Exception as err:  # noqa: BLE001 - component API across HA versions
        # A loaded ssdp component whose scanner is not where the helper looks
        # (a renamed hass.data key) lands here too, so drift is logged once
        # instead of silently dropping the strongest evidence source.
        log_input_failure("SSDP discovery cache", err)


def _collect_entries(inputs: UpnpInputs, hass: HomeAssistant) -> None:
    try:
        entries = hass.config_entries.async_entries(
            UPNP_DOMAIN, include_ignore=False, include_disabled=False
        )
        inputs.entry_present = bool(entries)
        inputs.entry_loaded = any(
            entry.state is ConfigEntryState.LOADED for entry in entries
        )
        inputs.flow_in_progress = any(
            flow.get("handler") == UPNP_DOMAIN
            for flow in hass.config_entries.flow.async_progress()
        )
        registry = er.async_get(hass)
        for entry in entries:
            for reg_entry in er.async_entries_for_config_entry(
                registry, entry.entry_id
            ):
                _record_entity(inputs, reg_entry)
        inputs.entity_ids.sort()
    except Exception as err:  # noqa: BLE001
        log_input_failure("UPnP/IGD config entries", err)


def _sensor_key(entry: er.RegistryEntry) -> str | None:
    """Return the upnp sensor key for a registry entry, matched by unique id."""
    if entry.domain != "sensor":
        return None
    unique_id = str(entry.unique_id or "")
    for key in SENSOR_KEYS:
        if unique_id.endswith(f"_{key}"):
            return key
    return None


def _record_entity(inputs: UpnpInputs, entry: er.RegistryEntry) -> None:
    if entry.platform != UPNP_DOMAIN:
        return
    disabled = entry.disabled_by is not None
    if not disabled:
        inputs.entity_ids.append(entry.entity_id)
    key = _sensor_key(entry)
    if key is None:
        return
    target = inputs.disabled_sensors if disabled else inputs.sensors
    target.setdefault(key, []).append(entry.entity_id)


async def async_collect_upnp_inputs(hass: HomeAssistant) -> UpnpInputs:
    """Read every UPnP/IGD input, each guarded independently."""
    inputs = UpnpInputs()
    await _collect_ssdp(inputs, hass)
    _collect_entries(inputs, hass)
    return inputs


# ---------------------------------------------------------------------------
# Pure adapter
# ---------------------------------------------------------------------------


def _known_state(by_id: Mapping[str, SnapshotEntity], entity_id: str) -> str | None:
    entity = by_id.get(entity_id)
    if entity is None:
        return None
    state = entity["state"]
    return None if state in _UNKNOWN_STATES else state


def _live_sensor(
    inputs: UpnpInputs, by_id: Mapping[str, SnapshotEntity], key: str
) -> tuple[str, str] | None:
    """Return (entity id, state) of the first sensor for *key* with a known state."""
    for entity_id in inputs.sensors.get(key, []):
        state = _known_state(by_id, entity_id)
        if state is not None:
            return entity_id, state
    return None


def public_ip_value(state: str) -> str | None:
    """
    Return *state* when it is a routable-looking address, else None.

    Gateways answer ``0.0.0.0`` (or ``::``) while the WAN link is down, so
    that is "unknown", not an address the home briefly had.
    """
    try:
        address = ipaddress.ip_address(state.strip())
    except ValueError:
        return None
    if address.is_unspecified:
        return None
    return str(address)


def port_mapping_count(state: str) -> int | None:
    """Return *state* as a mapping count, or None when it is not one."""
    try:
        value = float(state)
    except ValueError:
        return None
    if not math.isfinite(value) or value < 0 or value > MAX_PORT_MAPPING_COUNT:
        return None
    return int(value)


def _upnp_evidence(
    inputs: UpnpInputs, by_id: Mapping[str, SnapshotEntity]
) -> tuple[str | None, str | None]:
    """
    Return (evidence source, note) for the UPnP-on fact.

    The source is None when nothing observed proves UPnP is on; the note then
    says why the check is not running on this home.
    """
    if inputs.igd_advertised:
        return "ssdp", None
    live = any(_known_state(by_id, e) is not None for e in inputs.entity_ids)
    if inputs.entry_loaded and live:
        return "integration", None
    if inputs.flow_in_progress and inputs.igd_advertised is None:
        return "discovery_flow", None
    if inputs.entry_present:
        return None, (
            "The UPnP/IGD integration is set up but its gateway is not answering "
            "and no gateway is announcing itself; UPnP is not audited until it "
            "does."
        )
    if inputs.igd_advertised is False:
        return None, (
            "No UPnP gateway has announced itself to Home Assistant, which usually "
            "means UPnP is off on the router; it can also mean Home Assistant is "
            "on a different network segment. UPnP is not audited."
        )
    return None, (
        "SSDP discovery is not available and the UPnP/IGD integration is not set "
        "up; UPnP is not audited."
    )


def _previous_value(
    previous: Mapping[str, Any], value_key: str, entity_key: str, entity_id: str
) -> Any:
    """Return the remembered value only when it was read from *entity_id*."""
    if previous.get(entity_key) != entity_id:
        return None
    return previous.get(value_key)


def public_ip_posture(
    context: NetworkBuildContext, entity_id: str, ip: str
) -> dict[str, Any]:
    """
    Return the public-IP posture keys for *ip* read from *entity_id*.

    Shared by every adapter with a public-IP sensor (UPnP/IGD, eero): the
    pseudonymized key, the entity it came from, and, when the engine's memory
    holds a value read from the same entity, whether it changed.
    """
    if context.pseudonymizer is None:
        return {}
    previous: Mapping[str, Any] = context.previous_posture or {}
    key = context.pseudonymizer.ip_key(ip)
    posture: dict[str, Any] = {
        "public_ip_key": key,
        "public_ip_entity_id": entity_id,
    }
    previous_key = _previous_value(
        previous, "public_ip_key", "public_ip_entity_id", entity_id
    )
    if isinstance(previous_key, str) and previous_key:
        posture["public_ip_changed"] = key != previous_key
        posture["public_ip_previous_key"] = previous_key
    return posture


def upnp_igd_adapter(
    inputs: UpnpInputs,
    entities: Sequence[SnapshotEntity],
    context: NetworkBuildContext,
) -> AdapterResult:
    """Pure adapter: UPnP posture from the SSDP cache and the upnp integration."""
    result = AdapterResult(name="upnp_igd")
    posture = result.posture
    previous: Mapping[str, Any] = context.previous_posture or {}
    by_id = {e["entity_id"]: e for e in entities}

    source, note = _upnp_evidence(inputs, by_id)
    if source is not None:
        posture["upnp_enabled"] = True
        posture["upnp_evidence"] = source
        posture["upnp_gateway_names"] = list(inputs.igd_names)
    elif note is not None:
        result.notes.append(note)

    ip_sensor = _live_sensor(inputs, by_id, SENSOR_KEY_EXTERNAL_IP)
    ip = public_ip_value(ip_sensor[1]) if ip_sensor else None
    if ip_sensor is not None and ip is not None and context.pseudonymizer is not None:
        posture.update(public_ip_posture(context, ip_sensor[0], ip))

    count_sensor = _live_sensor(inputs, by_id, SENSOR_KEY_PORT_MAPPINGS)
    count = port_mapping_count(count_sensor[1]) if count_sensor else None
    if count_sensor is not None and count is not None:
        entity_id = count_sensor[0]
        posture["upnp_port_mapping_count"] = count
        posture["upnp_port_mapping_entity_id"] = entity_id
        previous_count = _previous_value(
            previous,
            "upnp_port_mapping_count",
            "upnp_port_mapping_entity_id",
            entity_id,
        )
        if isinstance(previous_count, int) and not isinstance(previous_count, bool):
            posture["upnp_port_mappings_added"] = max(0, count - previous_count)
            posture["upnp_port_mapping_previous_count"] = previous_count
    elif inputs.disabled_sensors.get(SENSOR_KEY_PORT_MAPPINGS):
        result.notes.append(
            "Enable the UPnP/IGD port-mapping count sensor "
            f"({inputs.disabled_sensors[SENSOR_KEY_PORT_MAPPINGS][0]}) to be told "
            "when a device opens a new port through UPnP."
        )
    return result

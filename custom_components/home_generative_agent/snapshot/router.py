"""
Router adapters: clients from ``device_tracker`` entities, and eero posture.

Two entity-tier adapters (docs/network-security-plan.md, *Adapters*):

- ``generic_router_tracker`` turns every ``device_tracker`` entity whose
  ``source_type`` is ``router`` into a :class:`NetworkClient`. Home Assistant's
  scanner entities all expose the same attributes (``ip``, ``mac``,
  ``host_name``, ``source_type``), so FRITZ!Box, UniFi, eero, ASUSWRT, Nmap,
  and the rest are one adapter; the extras some of them add (``manufacturer``,
  ``connection_type``, ``ssid`` / ``essid`` / ``network_name``, ``vlan``,
  ``is_guest``) are read when present. The raw MAC is used for the
  per-install pseudonymized ``key`` and the locally-administered bit, and is
  then dropped: it never enters the snapshot.
- ``eero`` reads the community eero integration's network-level switches and
  sensors (``schmittx/home-assistant-eero``; entity unique id
  ``<network id>-<key>``) into router posture. Its UPnP switch outranks the
  UPnP/IGD inference and its public-IP sensor feeds the same change check.
  eero Plus features are simply absent on other accounts and are reported as
  missing capabilities, not as off.

Clients join the device inventory as the ``router`` source: the adapter asks
the inventory which keys are new (``new_clients``), attaches each recorded
client's ``first_seen`` so the grace period can be applied, and marks a client
``auto_trust`` when its MAC joins a registry device that a non-router
integration set up (the plan's narrow auto-trust rule). The engine commits
after dispatch.

Every read of Home Assistant state happens in the collector; the adapters
are pure over the snapshot entities and the collected inputs.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from homeassistant.helpers import device_registry as dr
from homeassistant.helpers import entity_registry as er

from custom_components.home_generative_agent.sentinel.network_inventory import (
    ROUTER_SOURCE,
    client_observation,
)

from .network import (
    DISCOVERY_SOURCES,
    ROUTER_PLATFORMS,
    AdapterResult,
    log_input_failure,
    sanitize_label,
)
from .upnp import public_ip_posture, public_ip_value

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from homeassistant.core import HomeAssistant

    from .network import NetworkBuildContext
    from .schema import NetworkClient, SnapshotEntity

EERO_DOMAIN = "eero"
TRACKER_DOMAIN = "device_tracker"
SOURCE_TYPE_ROUTER = "router"

# eero network-level switch key -> posture key. Keys are the integration's
# entity description keys (its unique id is ``<network id>-<key>``).
EERO_SWITCHES: dict[str, str] = {
    "upnp": "upnp_enabled",
    "wpa3": "wpa3_enabled",
    "guest_network_enabled": "guest_network_enabled",
    "ipv6_upstream": "ipv6_enabled",
    "ddns_enabled": "ddns_enabled",
    "block_malware": "malware_blocking_enabled",
    "ad_block": "ad_blocking_enabled",
}
EERO_SENSOR_PUBLIC_IP = "public_ip"
EERO_SENSOR_GUEST_CLIENTS = "connected_guest_clients_count"
EERO_SENSOR_THREATS_DAY = "blocked_day"
EERO_SENSORS: frozenset[str] = frozenset(
    {EERO_SENSOR_PUBLIC_IP, EERO_SENSOR_GUEST_CLIENTS, EERO_SENSOR_THREATS_DAY}
)
_UNKNOWN_STATES: frozenset[str] = frozenset({"", "unknown", "unavailable"})
# Attributes that name the network a client is on, most specific first.
_NETWORK_NAME_ATTRS: tuple[str, ...] = ("ssid", "essid", "network_name")
# Config-entry domains that prove nothing about a client: the router and
# tracker integrations create a registry device for every client they see.
_NON_QUALIFYING_DOMAINS: frozenset[str] = frozenset(
    {*ROUTER_PLATFORMS, *DISCOVERY_SOURCES, TRACKER_DOMAIN, "nmap_tracker"}
)
_MAC_OCTETS = 6
_OCTET_HEX_CHARS = 2
_BARE_MAC_CHARS = 12
COUNTER_CLIENT_COUNT = "network.client_count"
COUNTER_THREATS_DAY = "network.threats_day"


@dataclass(frozen=True)
class MacIndexEntry:
    """A registry device reachable by MAC, and whether it vouches for its client."""

    device_id: str
    integration: str | None
    # At least one config entry from a non-router integration owns a
    # non-tracker entity on this device (a Shelly plug, a printer).
    qualifying: bool


@dataclass
class RouterInputs:
    """What the adapters read from the registries, each field guarded."""

    # Tracker entity id -> registry platform (only trackers in the registry).
    tracker_platforms: dict[str, str] = field(default_factory=dict)
    # Tracker entity id -> registry device id.
    tracker_devices: dict[str, str] = field(default_factory=dict)
    # Normalized MAC -> registry device, for the auto-trust rule.
    mac_index: dict[str, MacIndexEntry] = field(default_factory=dict)
    eero_present: bool = False
    # eero network-level entities: key -> entity ids (registry order).
    eero_switches: dict[str, list[str]] = field(default_factory=dict)
    eero_sensors: dict[str, list[str]] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Collector (reads the registries)
# ---------------------------------------------------------------------------


def normalize_mac(value: Any) -> str | None:
    """Return *value* as a lower-case colon-separated MAC, or None."""
    if not isinstance(value, str):
        return None
    text = value.strip().lower().replace("-", ":").replace(".", "")
    if ":" not in text and len(text) == _BARE_MAC_CHARS:
        text = ":".join(text[i : i + 2] for i in range(0, _BARE_MAC_CHARS, 2))
    parts = text.split(":")
    if len(parts) != _MAC_OCTETS or any(
        len(p) != _OCTET_HEX_CHARS or any(c not in "0123456789abcdef" for c in p)
        for p in parts
    ):
        return None
    return text


def mac_is_randomized(mac: str) -> bool:
    """Return whether the locally-administered bit of a normalized MAC is set."""
    return bool(int(mac[:2], 16) & 0x02)


def _eero_network_key(entry: er.RegistryEntry) -> str | None:
    """Return the eero key of a network-level entity (unique id ``<net>-<key>``)."""
    parts = str(entry.unique_id or "").split("-")
    if len(parts) != 2:  # noqa: PLR2004 - resource entities have three parts
        return None
    return parts[1]


def _record_eero_entity(inputs: RouterInputs, entry: er.RegistryEntry) -> None:
    inputs.eero_present = True
    key = _eero_network_key(entry)
    if key is None:
        return
    if entry.domain == "switch" and key in EERO_SWITCHES:
        inputs.eero_switches.setdefault(key, []).append(entry.entity_id)
    elif entry.domain == "sensor" and key in EERO_SENSORS:
        inputs.eero_sensors.setdefault(key, []).append(entry.entity_id)


def _collect_entities(inputs: RouterInputs, hass: HomeAssistant) -> None:
    registry = er.async_get(hass)
    # device id -> {config entry id -> entity domains it owns on the device}
    owned: dict[str, dict[str, set[str]]] = {}
    for entry in registry.entities.values():
        if entry.disabled_by is not None:
            continue
        if entry.domain == TRACKER_DOMAIN:
            inputs.tracker_platforms[entry.entity_id] = entry.platform
            if entry.device_id:
                inputs.tracker_devices[entry.entity_id] = entry.device_id
        if entry.platform == EERO_DOMAIN:
            _record_eero_entity(inputs, entry)
        if entry.device_id and entry.config_entry_id:
            owned.setdefault(entry.device_id, {}).setdefault(
                entry.config_entry_id, set()
            ).add(entry.domain)
    _index_macs(inputs, hass, owned)


def _index_macs(
    inputs: RouterInputs,
    hass: HomeAssistant,
    owned: Mapping[str, Mapping[str, set[str]]],
) -> None:
    device_registry = dr.async_get(hass)
    for device_id, entries in owned.items():
        device = device_registry.async_get(device_id)
        if device is None:
            continue
        # ``async_get`` may return a child device entry, which has no
        # connections of its own.
        connections = getattr(device, "connections", None) or ()
        macs = [
            mac
            for kind, value in connections
            if kind == dr.CONNECTION_NETWORK_MAC
            and (mac := normalize_mac(value)) is not None
        ]
        if not macs:
            continue
        integration: str | None = None
        qualifying = False
        for entry_id, domains in entries.items():
            config_entry = hass.config_entries.async_get_entry(entry_id)
            domain = config_entry.domain if config_entry is not None else None
            if domain is None:
                continue
            integration = integration or domain
            if domain not in _NON_QUALIFYING_DOMAINS and domains - {TRACKER_DOMAIN}:
                integration = domain
                qualifying = True
        index = MacIndexEntry(
            device_id=device_id, integration=integration, qualifying=qualifying
        )
        for mac in macs:
            inputs.mac_index.setdefault(mac, index)


def async_collect_router_inputs(hass: HomeAssistant) -> RouterInputs:
    """Read the registries once; a failure leaves the inputs empty."""
    inputs = RouterInputs()
    try:
        _collect_entities(inputs, hass)
    except Exception as err:  # noqa: BLE001
        log_input_failure("router entity registry", err)
        return RouterInputs()
    return inputs


# ---------------------------------------------------------------------------
# The generic tracker adapter
# ---------------------------------------------------------------------------


def _label(value: Any) -> str | None:
    text = sanitize_label(value)
    return text or None


def _ip_value(value: Any) -> str | None:
    """Return *value* as a normalized address, or None for anything else."""
    return public_ip_value(value) if isinstance(value, str) else None


def _client_from_tracker(
    entity: SnapshotEntity,
    inputs: RouterInputs,
    context: NetworkBuildContext,
) -> NetworkClient | None:
    attrs: Mapping[str, Any] = entity.get("attributes") or {}
    mac = normalize_mac(attrs.get("mac"))
    if mac is None or context.pseudonymizer is None:
        return None
    index = inputs.mac_index.get(mac)
    ip = attrs.get("ip")
    client: NetworkClient = {
        "key": context.pseudonymizer.mac_key(mac),
        "connected": entity["state"] == "home",
        "name": _label(entity.get("friendly_name")),
        "ip": _ip_value(ip),
        "hostname": _label(attrs.get("host_name") or attrs.get("hostname")),
        "manufacturer": _label(attrs.get("manufacturer") or attrs.get("oui")),
        "connection_type": _connection_type(attrs),
        "network_name": next(
            (_label(attrs[a]) for a in _NETWORK_NAME_ATTRS if attrs.get(a)), None
        ),
        "last_seen": entity.get("last_changed"),
        "ha_device_id": (
            index.device_id
            if index
            else inputs.tracker_devices.get(entity["entity_id"])
        ),
        "ha_integration": (
            index.integration
            if index and index.integration
            else inputs.tracker_platforms.get(entity["entity_id"])
            or entity.get("platform")
        ),
        "tracker_entity_id": entity["entity_id"],
        "mac_randomized": mac_is_randomized(mac),
        "auto_trust": bool(index and index.qualifying),
    }
    if isinstance(attrs.get("is_guest"), bool):
        client["is_guest"] = attrs["is_guest"]
    if isinstance(attrs.get("vlan"), int) and not isinstance(attrs.get("vlan"), bool):
        client["vlan"] = attrs["vlan"]
    return client


def _connection_type(attrs: Mapping[str, Any]) -> str | None:
    value = attrs.get("connection_type")
    if isinstance(value, str) and value:
        text = value.strip().lower()
        if "wire" in text and "less" not in text:
            return "wired"
        return "wireless" if "wireless" in text or "wifi" in text else text
    wired = attrs.get("is_wired")
    if isinstance(wired, bool):
        return "wired" if wired else "wireless"
    return None


def generic_router_tracker_adapter(
    inputs: RouterInputs,
    entities: Sequence[SnapshotEntity],
    context: NetworkBuildContext,
) -> AdapterResult:
    """Pure adapter: every router-sourced ``device_tracker`` as a client."""
    result = AdapterResult(name="generic_router_tracker")
    trackers = [
        e
        for e in entities
        if e["domain"] == TRACKER_DOMAIN
        and (e.get("attributes") or {}).get("source_type") == SOURCE_TYPE_ROUTER
    ]
    if not trackers:
        return result
    if context.pseudonymizer is None:
        result.notes.append(
            "Router clients are not audited: the pseudonymization salt is not "
            "available, so client addresses cannot be tokenized."
        )
        return result
    clients: dict[str, NetworkClient] = {}
    for entity in sorted(trackers, key=lambda e: e["entity_id"]):
        client = _client_from_tracker(entity, inputs, context)
        if client is None:
            continue
        # Two trackers for one MAC (two integrations, or eero's per-connection
        # entities): the connected one wins, else the first by entity id.
        existing = clients.get(client["key"])
        if existing is None or (client["connected"] and not existing["connected"]):
            clients[client["key"]] = client
    result.clients = list(clients.values())
    result.counters[COUNTER_CLIENT_COUNT] = float(
        sum(1 for c in result.clients if c["connected"])
    )
    inventory = context.network_inventory
    if inventory is None:
        return result
    trusted_names = inventory.trusted_names(ROUTER_SOURCE)
    observations = []
    for client in result.clients:
        row = inventory.row(client_observation_key(client["key"]))
        if row is not None and isinstance(row.get("first_seen"), str):
            client["first_seen"] = row["first_seen"]
        if client.get("mac_randomized"):
            names = {
                (client.get("name") or "").lower(),
                (client.get("hostname") or "").lower(),
            } - {""}
            client["hostname_trusted"] = bool(names & trusted_names)
        observations.append(client_observation(client))
    delta = inventory.diff(observations, [ROUTER_SOURCE])
    # The inventory keys are source-qualified; the section carries client keys.
    prefix = f"{ROUTER_SOURCE}:"
    result.new_clients = [
        k.removeprefix(prefix) for k in delta.new_device_keys if k.startswith(prefix)
    ]
    return result


def client_observation_key(key: str) -> str:
    """Return the inventory key of a router client."""
    return f"{ROUTER_SOURCE}:{key}"


# ---------------------------------------------------------------------------
# The eero adapter
# ---------------------------------------------------------------------------


def _known_state(by_id: Mapping[str, SnapshotEntity], entity_id: str) -> str | None:
    entity = by_id.get(entity_id)
    if entity is None:
        return None
    state = entity["state"]
    return None if state in _UNKNOWN_STATES else state


def _live(
    ids: Mapping[str, list[str]], by_id: Mapping[str, SnapshotEntity], key: str
) -> tuple[str, str] | None:
    for entity_id in ids.get(key, []):
        state = _known_state(by_id, entity_id)
        if state is not None:
            return entity_id, state
    return None


def _count(state: str) -> int | None:
    try:
        value = float(state)
    except ValueError:
        return None
    if value != value or value < 0 or value > 1e9:  # noqa: PLR0124,PLR2004 - NaN/garbage
        return None
    return int(value)


def eero_adapter(
    inputs: RouterInputs,
    entities: Sequence[SnapshotEntity],
    context: NetworkBuildContext,
) -> AdapterResult:
    """Pure adapter: eero network posture from its switches and sensors."""
    result = AdapterResult(name="eero")
    if not inputs.eero_present:
        return result
    by_id = {e["entity_id"]: e for e in entities}
    posture = result.posture
    for key, field_name in EERO_SWITCHES.items():
        live = _live(inputs.eero_switches, by_id, key)
        if live is None:
            continue
        posture[field_name] = live[1] == "on"
        posture[f"{field_name}_entity_id"] = live[0]
    if "upnp_enabled" in posture:
        posture["upnp_evidence"] = "eero"

    guests = _live(inputs.eero_sensors, by_id, EERO_SENSOR_GUEST_CLIENTS)
    count = _count(guests[1]) if guests else None
    if guests is not None and count is not None:
        posture["guest_client_count"] = count
        posture["guest_client_count_entity_id"] = guests[0]

    threats = _live(inputs.eero_sensors, by_id, EERO_SENSOR_THREATS_DAY)
    count = _count(threats[1]) if threats else None
    if threats is not None and count is not None:
        result.counters[COUNTER_THREATS_DAY] = float(count)

    ip_sensor = _live(inputs.eero_sensors, by_id, EERO_SENSOR_PUBLIC_IP)
    ip = public_ip_value(ip_sensor[1]) if ip_sensor else None
    if ip_sensor is not None and ip is not None and context.pseudonymizer is not None:
        # Same posture keys and memory as the UPnP/IGD adapter: the change
        # check compares only values read from the same entity.
        posture.update(public_ip_posture(context, ip_sensor[0], ip))
    return result


def router_adapters(
    inputs: RouterInputs,
    entities: Sequence[SnapshotEntity],
    context: NetworkBuildContext,
) -> list[AdapterResult]:
    """Return the router adapters in merge order (generic first)."""
    return [
        generic_router_tracker_adapter(inputs, entities, context),
        eero_adapter(inputs, entities, context),
    ]

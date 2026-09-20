"""
eero runtime adapter: clients and settings read from the integration's own data.

The community eero integration (``schmittx/home-assistant-eero``, read at
version 1.8.1) registers a client's ``device_tracker`` only when it is set
up or reloaded, so a device that joins later has no entity and the tracker
adapter in ``snapshot/router.py`` cannot see it. The integration does hold
every client eero knows, refreshed on its poll interval, in the coordinator
it keeps at ``hass.data["eero"][<entry id>]["coordinator"]``. This runtime
adapter reads that object, following the plan's rules for the tier
(docs/network-security-plan.md, *Adapters*):

- nothing from the integration is imported; the objects are reached through
  ``hass.data`` and read attribute by attribute, each read guarded, so a
  renamed property degrades to a missing field and a broken coordinator to a
  missing capability with one rate-limited log line;
- only an explicit allowlist of attributes is read. The network object also
  carries the Wi-Fi and guest passwords and the Thread master key; those
  names are listed here as forbidden and a test proves they are never
  touched;
- the docstring names what is read: on ``coordinator.data`` (``EeroAccount``)
  the ``networks`` list; on each ``EeroNetwork`` the ``id``, ``name``,
  ``upnp``, ``wpa3``, ``guest_network_enabled``, ``ipv6_upstream``,
  ``ddns_enabled``, ``block_malware``, ``ad_block``, ``premium_enabled``,
  ``connected_guest_clients_count`` properties and the ``clients`` list; on
  each ``EeroClient`` ``mac``, ``ip``, ``hostname``, ``name``,
  ``manufacturer``, ``connection_type``, ``wireless``, ``connected``,
  ``last_active``, ``is_guest``, and ``device_type``.

The result runs last in the merge, so its client list and settings win over
the tracker adapter and the entity-tier eero adapter; the public-IP change
check and the threat counter stay with the entity adapter, whose sensors
exist whenever eero is loaded and keep the change memory on a stable
entity id. Clients from here carry no tracker entity id; the merge keeps
the one the tracker adapter found for the same key.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING, Any, Final

from homeassistant.config_entries import ConfigEntryState

from .network import AdapterResult, log_input_failure
from .router import (
    EERO_DOMAIN,
    ClientRead,
    RouterInputs,
    build_client,
    finalize_clients,
)

if TYPE_CHECKING:
    from homeassistant.core import HomeAssistant

    from .network import NetworkBuildContext
    from .schema import NetworkClient

EERO_INTEGRATION_VERSION: Final = "1.8.1"
DATA_COORDINATOR: Final = "coordinator"

# Every attribute this module reads, and nothing else.
NETWORK_ATTRS: Final[tuple[str, ...]] = (
    "id",
    "name",
    "upnp",
    "wpa3",
    "guest_network_enabled",
    "ipv6_upstream",
    "ddns_enabled",
    "block_malware",
    "ad_block",
    "premium_enabled",
    "connected_guest_clients_count",
)
CLIENT_ATTRS: Final[tuple[str, ...]] = (
    "mac",
    "ip",
    "hostname",
    "name",
    "manufacturer",
    "connection_type",
    "wireless",
    "connected",
    "last_active",
    "is_guest",
    "device_type",
)
# Attributes of the eero objects that must never be read: credentials and
# keys the integration exposes for its own entities. Kept next to the
# allowlist so a review sees both, and asserted disjoint by a test.
FORBIDDEN_ATTRS: Final[frozenset[str]] = frozenset(
    {
        "password",
        "guest_network_password",
        "thread_master_key",
        "thread_commissioning_credential",
        "thread_active_operational_dataset",
        "qr_code",
        "guest_network_qr_code",
    }
)

# Network settings that weaken the home when ON count if any network has
# them on; protections count only when every network has them on.
_ANY_ON: Final[frozenset[str]] = frozenset(
    {"upnp_enabled", "guest_network_enabled", "ipv6_enabled", "ddns_enabled"}
)
_SWITCH_TO_POSTURE: Final[dict[str, str]] = {
    "upnp": "upnp_enabled",
    "wpa3": "wpa3_enabled",
    "guest_network_enabled": "guest_network_enabled",
    "ipv6_upstream": "ipv6_enabled",
    "ddns_enabled": "ddns_enabled",
    "block_malware": "malware_blocking_enabled",
    "ad_block": "ad_blocking_enabled",
}

RUNTIME_FAILED_NOTE: Final = (
    "The eero integration's client list could not be read from the "
    "integration, so this run used its device trackers, which list a device "
    "only after the integration is reloaded."
)


@dataclass
class EeroNetworkRead:
    """One network as read from the coordinator (allowlisted fields only)."""

    id: str
    name: str | None = None
    settings: dict[str, bool] = field(default_factory=dict)
    premium_enabled: bool | None = None
    guest_client_count: int | None = None
    clients: list[ClientRead] = field(default_factory=list)


@dataclass
class EeroRuntimeInputs:
    """What the adapter reads, collected from ``hass.data`` with every read guarded."""

    present: bool = False
    failed: bool = False
    networks: list[EeroNetworkRead] = field(default_factory=list)


def _read(obj: Any, attr: str) -> Any:
    """Return ``obj.<attr>``, or None when the attribute is missing or raises."""
    if attr in FORBIDDEN_ATTRS:  # pragma: no cover - guarded by the allowlist test
        return None
    try:
        return getattr(obj, attr)
    except Exception:  # noqa: BLE001 - runtime objects across versions
        return None


def _read_client(client: Any, network_name: str | None) -> ClientRead | None:
    values = {attr: _read(client, attr) for attr in CLIENT_ATTRS}
    if not values["mac"]:
        return None
    wireless = values["wireless"]
    connection = values["connection_type"]
    if isinstance(connection, str) and connection:
        connection = connection.strip().lower()
    elif isinstance(wireless, bool):
        connection = "wireless" if wireless else "wired"
    else:
        connection = None
    last_active = values["last_active"]
    return ClientRead(
        mac=values["mac"],
        connected=bool(values["connected"]),
        platform=EERO_DOMAIN,
        name=values["name"],
        ip=values["ip"],
        hostname=values["hostname"],
        manufacturer=values["manufacturer"],
        connection_type=connection,
        network_name=network_name,
        last_seen=(
            last_active.isoformat() if isinstance(last_active, datetime) else None
        ),
        is_guest=values["is_guest"] if isinstance(values["is_guest"], bool) else None,
    )


def _read_network(network: Any) -> EeroNetworkRead | None:
    values = {attr: _read(network, attr) for attr in NETWORK_ATTRS}
    network_id = values["id"]
    if not network_id:
        return None
    name = values["name"]
    read = EeroNetworkRead(
        id=str(network_id),
        name=str(name) if name else None,
        premium_enabled=(
            values["premium_enabled"]
            if isinstance(values["premium_enabled"], bool)
            else None
        ),
    )
    for attr, posture_key in _SWITCH_TO_POSTURE.items():
        if isinstance(values[attr], bool):
            read.settings[posture_key] = values[attr]
    guests = values["connected_guest_clients_count"]
    if isinstance(guests, int) and not isinstance(guests, bool) and guests >= 0:
        read.guest_client_count = guests
    clients = _read(network, "clients") or []
    for client in clients:
        client_read = _read_client(client, read.name)
        if client_read is not None:
            read.clients.append(client_read)
    return read


def collect_eero_runtime_inputs(hass: HomeAssistant) -> EeroRuntimeInputs:
    """Read every loaded eero entry's coordinator; a failure leaves the inputs empty."""
    inputs = EeroRuntimeInputs()
    try:
        entries: Any = hass.data.get(EERO_DOMAIN) or {}
        for entry_id, data in entries.items():
            entry = hass.config_entries.async_get_entry(str(entry_id))
            if entry is None or entry.state is not ConfigEntryState.LOADED:
                continue
            coordinator = data.get(DATA_COORDINATOR) if isinstance(data, dict) else None
            account = getattr(coordinator, "data", None)
            if account is None:
                continue
            inputs.present = True
            for network in _read(account, "networks") or []:
                read = _read_network(network)
                if read is not None:
                    inputs.networks.append(read)
    except Exception as err:  # noqa: BLE001 - hass.data layout across versions
        log_input_failure("eero coordinator", err)
        return EeroRuntimeInputs(present=inputs.present, failed=True)
    return inputs


def eero_runtime_adapter(
    inputs: EeroRuntimeInputs,
    router_inputs: RouterInputs,
    context: NetworkBuildContext,
) -> AdapterResult:
    """Pure adapter: clients and settings from the eero coordinator read."""
    result = AdapterResult(name="eero_runtime")
    if not inputs.present:
        return result
    if inputs.failed:
        result.notes.append(RUNTIME_FAILED_NOTE)
        return result
    if context.pseudonymizer is None:
        return result

    clients: dict[str, NetworkClient] = {}
    for network in inputs.networks:
        for read in network.clients:
            client = build_client(read, router_inputs, context)
            if client is None:
                continue
            existing = clients.get(client["key"])
            if existing is None or (client["connected"] and not existing["connected"]):
                clients[client["key"]] = client
    if clients:
        finalize_clients(result, list(clients.values()), context)
    _aggregate_posture(inputs, result)
    return result


def _aggregate_posture(inputs: EeroRuntimeInputs, result: AdapterResult) -> None:
    """Judge the settings across every network (any-on weakens, all-on protects)."""
    posture = result.posture
    for key in _SWITCH_TO_POSTURE.values():
        states = [n.settings[key] for n in inputs.networks if key in n.settings]
        if not states:
            continue
        posture[key] = any(states) if key in _ANY_ON else all(states)
    if "upnp_enabled" in posture:
        posture["upnp_evidence"] = "eero"
    guests = [
        n.guest_client_count
        for n in inputs.networks
        if n.guest_client_count is not None
    ]
    if guests:
        posture["guest_client_count"] = sum(guests)
    if inputs.networks and not any(n.premium_enabled for n in inputs.networks):
        result.notes.append(
            "eero Plus is not active on this account, so the dynamic DNS, "
            "advanced security, and ad-blocking settings are not audited."
        )

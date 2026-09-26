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
- the docstring names what is read: on the coordinator ``data`` (an
  ``EeroAccount``) and ``last_update_success``; on ``data`` the ``networks``
  list; on each ``EeroNetwork`` the ``id``, ``name``, ``upnp``, ``wpa3``,
  ``guest_network_enabled``, ``ipv6_upstream``, ``ddns_enabled``,
  ``block_malware``, ``ad_block``, ``premium_enabled``,
  ``connected_guest_clients_count`` properties and the ``clients`` list; on
  each ``EeroClient`` ``mac``, ``ip``, ``hostname``, ``name``,
  ``manufacturer``, ``connection_type``, ``wireless``, ``connected``,
  ``last_active``, ``is_guest``, ``device_type``, and the figures ``usage_up``,
  ``usage_down`` (one poll's Mbps), ``signal`` (dBm), ``data_usage_day``
  (today's download and upload bytes; only present when the integration's
  Activity option requests it for clients), and ``blocked_day``. Only the networks the
  entry is configured for (the ``networks`` list beside the coordinator)
  are read, as the integration's own platforms do.

What counts as a usable read is strict, because the result replaces the
tracker adapter's client list in the merge and the inventory reconciles
the router source against it: the coordinator's last poll must have
succeeded, and every configured network's client list must have been read
without a fault. A partial read withholds the client list (the tracker
adapter and its reload caveat stand in) and says so in a note. A complete
read with no clients publishes an empty list, so the router source
bootstraps on a home whose eero knows no clients yet instead of trusting
the first one silently.

Settings are judged across the configured networks only when every one of
them reports the setting: a weakening setting is on if any network has it
on, a protection only if every network has it. The public-IP change check
and the threat counter stay with the entity adapter, whose sensors keep the
change memory on a stable entity id. Clients from here carry no tracker
entity id; the merge fills it from the tracker adapter's row for the same
key.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING, Any, Final

from homeassistant.config_entries import ConfigEntryState

from .network import AdapterResult, client_counter_id, log_input_failure
from .router import (
    EERO_ANY_ON,
    EERO_DOMAIN,
    EERO_SWITCHES,
    NO_SALT_NOTE,
    REGISTRY_FAILED_NOTE,
    ClientRead,
    RouterInputs,
    build_client,
    connection_type_from,
)

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping

    from homeassistant.core import HomeAssistant

    from .network import NetworkBuildContext
    from .schema import NetworkClient

LOGGER = logging.getLogger(__name__)

EERO_INTEGRATION_VERSION: Final = "1.8.1"
DATA_COORDINATOR: Final = "coordinator"
DATA_NETWORKS: Final = "networks"

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
    # Figures for the baseline step: instantaneous rates and the link signal
    # come with every poll; today's traffic and blocked count only when the
    # integration's Activity option requests them for clients.
    "usage_up",
    "usage_down",
    "signal",
    "data_usage_day",
    "blocked_day",
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

# Settings that exist only with an eero Plus subscription.
_PLUS_ONLY: Final[frozenset[str]] = frozenset(
    {"ddns_enabled", "malware_blocking_enabled", "ad_blocking_enabled"}
)

RUNTIME_FAILED_NOTE: Final = (
    "The eero integration's client list could not be read completely from the "
    "integration, so this run used its device trackers, which list a device "
    "only after the integration is reloaded."
)
STALE_POLL_NOTE: Final = (
    "The eero integration's last poll failed, so its client list is stale; "
    "this run used its device trackers."
)
PLUS_UNKNOWN_NOTE: Final = (
    "Whether eero Plus is active could not be read, so the dynamic DNS, "
    "advanced security, and ad-blocking settings are audited only where eero's "
    "own switches report them."
)
DATA_USAGE_NOTE: Final = (
    "Per-device data usage is not read: in the eero integration's options, "
    "under Activity, select Data Usage (Day) for clients so each device's "
    "daily traffic can be baselined."
)
NO_PLUS_NOTE: Final = (
    "eero Plus is not active on this account, so the dynamic DNS, advanced "
    "security, and ad-blocking settings are not audited."
)


class _Missing:
    """Sentinel for an attribute that could not be read (vs. a legitimate None)."""

    def __repr__(self) -> str:
        return "MISSING"


MISSING: Final = _Missing()


@dataclass
class EeroNetworkRead:
    """One network as read from the coordinator (allowlisted fields only)."""

    id: str
    name: str | None = None
    settings: dict[str, bool] = field(default_factory=dict)
    premium_enabled: bool | None = None
    guest_client_count: int | None = None
    clients: list[ClientRead] = field(default_factory=list)
    # The clients list itself could not be read; the network's clients are
    # unknown, not absent.
    clients_unreadable: bool = False


@dataclass
class EeroRuntimeInputs:
    """What the adapter reads, collected from ``hass.data`` with every read guarded."""

    present: bool = False
    # The read raised somewhere structural (the hass.data layout, the
    # networks list): nothing below can be trusted.
    failed: bool = False
    # The coordinator's last poll failed: its data is the previous poll's.
    stale: bool = False
    networks: list[EeroNetworkRead] = field(default_factory=list)

    @property
    def clients_complete(self) -> bool:
        """Return whether every configured network's client list was read."""
        return (
            self.present
            and not self.failed
            and not self.stale
            and all(not n.clients_unreadable for n in self.networks)
        )


def _read(obj: Any, attr: str) -> Any:
    """Return ``obj.<attr>``, or :data:`MISSING` when it cannot be read."""
    if attr in FORBIDDEN_ATTRS:  # pragma: no cover - guarded by the allowlist test
        return MISSING
    try:
        return getattr(obj, attr)
    except Exception:  # noqa: BLE001 - runtime objects across versions
        return MISSING


def _value(obj: Any, attr: str) -> Any:
    """Return ``obj.<attr>``, with an unreadable attribute read as None."""
    value = _read(obj, attr)
    return None if value is MISSING else value


def _read_client(client: Any, network_name: str | None) -> ClientRead | None:
    values = {attr: _value(client, attr) for attr in CLIENT_ATTRS}
    connected = values["connected"]
    # A client whose MAC or connection state cannot be read is not a
    # client the audit can place; the tracker adapter's row, if any, stands.
    if not values["mac"] or not isinstance(connected, bool):
        return None
    wireless = values["wireless"]
    connection = connection_type_from(
        {
            "connection_type": values["connection_type"],
            "is_wired": (not wireless) if isinstance(wireless, bool) else None,
        }
    )
    last_active = values["last_active"]
    return ClientRead(
        mac=values["mac"],
        connected=connected,
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
        **_client_figures(values),
    )


def _finite(value: Any) -> float | None:
    """Return *value* as a finite non-negative float, else None."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    return number if math.isfinite(number) and number >= 0 else None


def _count(value: Any) -> int | None:
    """Return *value* as a non-negative int, else None."""
    number = _finite(value)
    return int(number) if number is not None and float(number).is_integer() else None


def _client_figures(values: Mapping[str, Any]) -> dict[str, Any]:
    """
    Return the numeric figures of a client read, each None when unreadable.

    ``data_usage_day`` is eero's ``(download, upload)`` byte tuple and
    ``signal`` its ``(level, unit)`` tuple; a shape that differs from the
    pinned integration reads as absent rather than as a wrong number.
    """
    usage = values["data_usage_day"]
    down = up = None
    if isinstance(usage, tuple) and len(usage) == 2:  # noqa: PLR2004
        down, up = _count(usage[0]), _count(usage[1])
    signal = values["signal"]
    level = None
    if isinstance(signal, tuple) and len(signal) == 2:  # noqa: PLR2004
        raw, unit = signal
        if isinstance(raw, int) and not isinstance(raw, bool) and unit == "dBm":
            level = raw
    return {
        "usage_up_mbps": _finite(values["usage_up"]),
        "usage_down_mbps": _finite(values["usage_down"]),
        "data_up_day_bytes": up,
        "data_down_day_bytes": down,
        "blocked_day": _count(values["blocked_day"]),
        "signal_dbm": level,
    }


def _read_network(network: Any) -> EeroNetworkRead | None:
    values = {attr: _value(network, attr) for attr in NETWORK_ATTRS}
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
    for attr, posture_key in EERO_SWITCHES.items():
        if isinstance(values[attr], bool):
            read.settings[posture_key] = values[attr]
    guests = values["connected_guest_clients_count"]
    if isinstance(guests, int) and not isinstance(guests, bool) and guests >= 0:
        read.guest_client_count = guests
    clients = _read(network, "clients")
    if clients is MISSING or clients is None:
        read.clients_unreadable = True
        return read
    try:
        for client in clients:
            client_read = _read_client(client, read.name)
            if client_read is not None:
                read.clients.append(client_read)
    except Exception:  # noqa: BLE001 - the list itself is the integration's object
        read.clients_unreadable = True
    return read


def _configured_networks(data: Any) -> set[str] | None:
    """Return the network ids the entry is configured for, or None for all."""
    configured = data.get(DATA_NETWORKS) if isinstance(data, dict) else None
    if isinstance(configured, list | tuple | set) and configured:
        return {str(n) for n in configured}
    return None


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
            if coordinator is None or account is None:
                continue
            inputs.present = True
            if getattr(coordinator, "last_update_success", True) is False:
                inputs.stale = True
            configured = _configured_networks(data)
            networks = _read(account, "networks")
            if networks is MISSING or networks is None:
                inputs.failed = True
                continue
            for network in networks:
                read = _read_network(network)
                if read is None:
                    continue
                if configured is not None and read.id not in configured:
                    continue
                inputs.networks.append(read)
    except Exception as err:  # noqa: BLE001 - hass.data layout across versions
        log_input_failure("eero coordinator", err)
        return EeroRuntimeInputs(present=inputs.present, failed=True)
    if inputs.failed:
        log_input_failure("eero networks", RuntimeError("networks list unreadable"))
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
    if not inputs.stale:
        # A failed poll leaves the previous poll's values in the coordinator;
        # a cached guest count would keep resetting (or advancing) the guest
        # idle clock through an outage, so stale settings are not published.
        _aggregate_posture(inputs, router_inputs, result)
    blocker = _clients_blocker(inputs, router_inputs, context)
    LOGGER.debug(
        "eero runtime read: %d network(s), %d client(s), stale=%s, complete=%s%s",
        len(inputs.networks),
        sum(len(n.clients) for n in inputs.networks),
        inputs.stale,
        inputs.clients_complete,
        f"; clients withheld: {blocker}" if blocker else "",
    )
    if blocker is not None:
        result.notes.append(blocker)
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
    # A complete read with nothing in it is still a read: publish the empty
    # list so the router source bootstraps rather than trusting the first
    # client that appears later.
    result.clients = list(clients.values())
    _publish_client_counters(result, clients.values())
    return result


def _publish_client_counters(
    result: AdapterResult, clients: Iterable[NetworkClient]
) -> None:
    """
    Publish each connected client's traffic as ``network.client.<key>.*``.

    These feed the baseline step (the engine hands the section's counters to
    the baseline updater), keyed by the pseudonymized client key so the
    stored rows carry no address. Today's traffic needs the integration's
    Activity option; when no connected client carries it the note says so,
    since the usage check would otherwise be listed as not run with no hint.
    """
    connected = [c for c in clients if c.get("connected")]
    with_usage = 0
    for client in connected:
        for figure in _COUNTER_FIGURES:
            value = client.get(figure)
            if value is not None:
                result.counters[client_counter_id(client["key"], figure)] = float(value)
        if client.get("data_up_day_bytes") is not None:
            with_usage += 1
    if connected and not with_usage:
        result.notes.append(DATA_USAGE_NOTE)


# Only what a rule reads is stored as a baseline: the two traffic figures.
# The rates (one poll's Mbps) and the blocked count stay on the client for
# display and the audit tool.
_COUNTER_FIGURES: Final[tuple[str, ...]] = ("data_up_day_bytes", "data_down_day_bytes")


def _clients_blocker(
    inputs: EeroRuntimeInputs, router_inputs: RouterInputs, context: NetworkBuildContext
) -> str | None:
    """Return the note explaining why no client list is published, or None."""
    if inputs.stale:
        return STALE_POLL_NOTE
    if not inputs.clients_complete:
        return RUNTIME_FAILED_NOTE
    if context.pseudonymizer is None:
        return NO_SALT_NOTE
    if router_inputs.registry_failed:
        # Same rule as the tracker adapter: without the registries a client
        # cannot be joined to its device, and committing rows without their
        # device ids and auto-trust verdicts would make the inventory worse.
        return REGISTRY_FAILED_NOTE
    return None


def _deciding_network(
    states: list[tuple[str, bool]], *, any_on: bool
) -> tuple[bool, str]:
    """Return (value, id of the network that decided it) for one setting."""
    value = any(on for _n, on in states) if any_on else all(on for _n, on in states)
    decider = next((n for n, on in states if on is (value if any_on else False)), None)
    return value, decider or states[0][0]


def _aggregate_posture(
    inputs: EeroRuntimeInputs, router_inputs: RouterInputs, result: AdapterResult
) -> None:
    """
    Judge the settings across every configured network.

    A setting is published only when every network it applies to reports it:
    judging the networks that happen to be readable could turn a known "UPnP
    on" into "off" and silence a standing finding. A weakening setting is on
    if any network has it on; a protection only if every network has it on.

    The eero Plus settings apply only to networks where Plus is active. When
    Plus is known to be off everywhere they are withdrawn from the merge too,
    so a switch left over from a lapsed subscription cannot tell someone to
    turn on a feature they no longer have; when Plus cannot be read they are
    left to the entity tier and the gap is noted.

    The entity twin names the switch of the network that decided the value
    (from the registry read in ``snapshot/router.py``), so per-rule entity
    exclusions and the finding's identity are the same on both tiers.
    """
    posture = result.posture
    if not inputs.networks:
        return
    switch_key = {posture_key: key for key, posture_key in EERO_SWITCHES.items()}
    plus_networks = [n for n in inputs.networks if n.premium_enabled is True]
    for key in EERO_SWITCHES.values():
        networks = plus_networks if key in _PLUS_ONLY else inputs.networks
        if not networks or any(key not in n.settings for n in networks):
            continue
        value, decider = _deciding_network(
            [(n.id, n.settings[key]) for n in networks], any_on=key in EERO_ANY_ON
        )
        posture[key] = value
        switches = router_inputs.eero_switches.get(switch_key[key], [])
        entity_id = next((e for net, e in switches if net == decider), None)
        if entity_id:
            posture[f"{key}_entity_id"] = entity_id
    if "upnp_enabled" in posture:
        posture["upnp_evidence"] = "eero"
    guests = [n.guest_client_count for n in inputs.networks]
    if all(g is not None for g in guests):
        posture["guest_client_count"] = sum(g for g in guests if g is not None)
    plus = [n.premium_enabled for n in inputs.networks]
    if all(p is False for p in plus):
        result.posture_withdrawn |= set(_PLUS_ONLY)
        result.notes.append(NO_PLUS_NOTE)
    elif not plus_networks:
        result.notes.append(PLUS_UNKNOWN_NOTE)

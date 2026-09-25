"""
Network section of the home-state snapshot (docs/network-security-plan.md).

The section is assembled from *adapters*. An adapter is a pure function over
inputs that were read from Home Assistant, returning the normalized fields it
can assert; the merge step records which adapter provided each capability so
rules can declare what they need and be skipped, visibly, when a home cannot
provide it. Nothing here scans the network: every fact is derived from what
Home Assistant already ingests.

Phase 1 (HA-only MVP) ships the ``ha_native`` adapter, which audits Home
Assistant's own attack surface: users and tokens, exposed sensitive entities,
cloud remote access, pending updates, HTTP and auth-provider settings,
Supervisor add-ons, public webhook automations, unavailable security devices,
and discovered-but-unconfigured devices. The radio adapters (``radio.py``) and
the UPnP/IGD adapter (``upnp.py``) plug into the same merge; the remaining
router and DNS adapters land in later steps.

Runtime reads (auth store, Supervisor add-on info, HTTP server settings, the
automation entity component) are wrapped individually: a read that fails on
some Home Assistant version degrades to a missing capability with one log
line, never to a failed snapshot build.
"""

from __future__ import annotations

import logging
import re
import unicodedata
from dataclasses import dataclass, field
from datetime import timedelta
from typing import TYPE_CHECKING, Any, cast

from homeassistant.util import dt as dt_util

from custom_components.home_generative_agent.agent.automation_pin import (
    find_critical_automation_calls,
)
from custom_components.home_generative_agent.agent.helpers import (
    resolve_critical_action_policy,
)
from custom_components.home_generative_agent.const import DOMAIN
from custom_components.home_generative_agent.sentinel.auth_inventory import (
    TOKEN_TYPE_LONG_LIVED,
    ObservedToken,
    ObservedUser,
    token_labels,
)

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping, Sequence
    from datetime import datetime

    from homeassistant.core import HomeAssistant

    from custom_components.home_generative_agent.sentinel.auth_inventory import (
        AuthInventory,
    )
    from custom_components.home_generative_agent.sentinel.network_inventory import (
        NetworkInventory,
    )
    from custom_components.home_generative_agent.sentinel.pseudonymizer import (
        Pseudonymizer,
    )

    from .schema import NetworkClient, NetworkSnapshot, SnapshotEntity

LOGGER = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Capability paths
# ---------------------------------------------------------------------------

CAP_CLIENTS = "network.clients"
# Present once the device inventory has compared this run's clients.
CAP_NEW_CLIENTS = "network.new_clients"
# Some source says which clients are on the guest network. Published by the
# merge (``_publish_guest_flag``), which also names the connected clients no
# such source covers, so they are a stated gap rather than a silent pass.
CAP_GUEST_CLIENTS = "network.clients.is_guest"
# Connected clients carry today's traffic (eero's Activity option for
# clients); published by the merge, like the guest flag.
CAP_CLIENT_DATA_DAY = "network.clients.data_usage_day"
# The engine injected baseline statistics for the section's counters
# (``counter_baselines``); a rule that compares against them requires it.
CAP_COUNTER_BASELINES = "network.counter_baselines"
# Per-client counters: ``network.client.<key>.<figure>``.
CLIENT_COUNTER_PREFIX = "network.client."


def client_counter_id(key: str, figure: str) -> str:
    """Return the counter id for one figure of the client with *key*."""
    return f"{CLIENT_COUNTER_PREFIX}{key}.{figure}"


def client_counter_key(counter_id: str) -> tuple[str, str] | None:
    """Return ``(client key, figure)`` for a per-client counter id, else None."""
    if not counter_id.startswith(CLIENT_COUNTER_PREFIX):
        return None
    rest = counter_id[len(CLIENT_COUNTER_PREFIX) :]
    key, sep, figure = rest.partition(".")
    return (key, figure) if sep and key and figure else None


_CAP_RADIO_PREFIX = "network.radio."
_CAP_POSTURE_PREFIX = "network.posture."
_CAP_HA_PREFIX = "network.ha_security."


def posture_cap(key: str) -> str:
    """Return the capability path for a ``posture`` field."""
    return f"{_CAP_POSTURE_PREFIX}{key}"


def ha_cap(key: str) -> str:
    """Return the capability path for an ``ha_security`` field."""
    return f"{_CAP_HA_PREFIX}{key}"


def radio_cap(key: str) -> str:
    """Return the capability path for a radio field (``posture.<key>`` too)."""
    return f"{_CAP_RADIO_PREFIX}{key}"


# Integrations whose ``update.*`` entities describe router / gateway firmware.
# Matching on the entity-registry platform is a fact; matching on names is a
# guess, which is why SnapshotEntity carries ``platform``.
ROUTER_PLATFORMS: frozenset[str] = frozenset(
    {
        "fritz",
        "unifi",
        "eero",
        "asuswrt",
        "tplink_omada",
        "keenetic_ndms2",
        "mikrotik",
        "netgear",
        "freebox",
        "upnp",
    }
)

# Entity domains whose devices count as security devices.
SECURITY_DOMAINS: frozenset[str] = frozenset({"lock", "alarm_control_panel", "camera"})

# Cover device classes / name hints that make a cover an entry point. The
# hints are word-bounded so "indoor" / "outdoor" shades do not match.
_ENTRY_COVER_CLASSES: frozenset[str] = frozenset({"door", "garage", "gate"})
_ENTRY_COVER_HINT_RE = re.compile(r"\b(door|garage|gate)\b")

# Config-flow sources that mean "Home Assistant found this on the LAN".
DISCOVERY_SOURCES: frozenset[str] = frozenset({"ssdp", "zeroconf", "dhcp", "homekit"})

# Persistent-notification id Home Assistant's http component uses for failed
# logins (homeassistant.components.http.ban.NOTIFICATION_ID_LOGIN).
LOGIN_NOTIFICATION_ID = "http-login"

# Longest label copied from an untrusted source (mDNS name, add-on title,
# token client name) into evidence and notification text.
MAX_LABEL_CHARS = 64

# Once-per-process log guard for runtime reads that fail: the snapshot builds
# every few minutes and a version drift must not fill the log. Keyed by input
# and exception class so a new failure mode still logs once.
_LOGGED_INPUT_FAILURES: set[str] = set()


def sanitize_label(value: Any, *, limit: int = MAX_LABEL_CHARS) -> str:
    """
    Return *value* as a short, printable label safe for notification text.

    Names that reach this module come from the LAN (mDNS/SSDP advertisements,
    DHCP hostnames), from third-party add-on repositories, and from whoever
    minted a token. Control and format characters (bidi overrides, zero-width
    joiners) are dropped, whitespace is collapsed, and the length is capped so
    a hostile name cannot restyle a push notification or pad an LLM prompt.
    """
    text = "".join(
        ch
        for ch in str(value or "")
        if ch.isspace() or unicodedata.category(ch)[0] != "C"
    )
    text = " ".join(text.split())
    if len(text) > limit:
        text = text[: limit - 1].rstrip() + "\u2026"
    return text


# ---------------------------------------------------------------------------
# Build context and adapter result
# ---------------------------------------------------------------------------


@dataclass
class NetworkBuildContext:
    """
    What the snapshot builder needs beyond ``hass`` to build the section.

    ``options`` are the integration's effective options (for the critical
    action PIN policy); ``None`` means unknown, and the PIN capability is then
    reported as missing rather than guessed. ``auth_observation`` lets the
    Sentinel engine collect the auth state once, feed it into the build, and
    commit the same observation to the inventory afterwards.
    ``previous_posture`` is the engine's memory of the last run's posture
    values, for the checks that report a change.
    """

    enabled: bool = True
    options: Mapping[str, Any] | None = None
    pseudonymizer: Pseudonymizer | None = None
    auth_inventory: AuthInventory | None = None
    auth_observation: list[ObservedUser] | None = None
    network_inventory: NetworkInventory | None = None
    # Posture values the engine kept from its previous run (see
    # ``upnp.POSTURE_MEMORY_KEYS``): the change checks compare against these
    # and stay inactive when there is nothing to compare with.
    previous_posture: Mapping[str, Any] | None = None


@dataclass
class AdapterResult:
    """Partial network section produced by one adapter."""

    name: str
    posture: dict[str, Any] = field(default_factory=dict)
    ha_security: dict[str, Any] = field(default_factory=dict)
    # None means the adapter cannot see clients at all (capability absent);
    # an empty list means it can and there are none.
    clients: list[NetworkClient] | None = None
    # Posture keys an earlier source may have published that this source
    # knows must not stand (a feature the account does not have): the merge
    # removes them, their entity twins, and their capabilities.
    posture_withdrawn: set[str] = field(default_factory=set)
    # Client keys the device inventory did not know before this run; None
    # when no inventory compared them.
    new_clients: list[str] | None = None
    counters: dict[str, float] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)
    # Radio section fields (``devices``, ``new_devices``, ``present_sources``)
    # and radio posture, merged into ``network.radio``.
    radio: dict[str, Any] = field(default_factory=dict)
    radio_posture: dict[str, Any] = field(default_factory=dict)
    # Capability paths an adapter asserts that are not a field of its own
    # (Z-Wave security classes live on the registry adapter's devices).
    extra_capabilities: list[str] = field(default_factory=list)


# Radio fields that are bookkeeping for the engine, not rule inputs.
_RADIO_NON_CAPABILITY_KEYS: frozenset[str] = frozenset({"present_sources"})
# Posture fields that only describe a fact for display (evidence source,
# gateway names, the previous run's values) and never gate a rule. A key's
# ``<key>_entity_id`` twin (the entity a posture fact was read from) is
# display-only as well.
_POSTURE_NON_CAPABILITY_KEYS: frozenset[str] = frozenset(
    {
        "upnp_evidence",
        "upnp_gateway_names",
        "public_ip_previous_key",
        "upnp_port_mapping_previous_count",
        "guest_network_last_active",
        "guest_network_last_observed",
    }
)
# The one definition; snapshot/upnp.py re-exports it for its callers.
POSTURE_DISPLAY_KEYS = _POSTURE_NON_CAPABILITY_KEYS
_ENTITY_TWIN_SUFFIX = "_entity_id"
# Posture keys that only mean something together. When a later adapter
# provides the group's lead key, the earlier adapter's other members are
# dropped first, so a change flag computed against one gateway's sensor never
# survives next to another gateway's current value.
_POSTURE_GROUPS: dict[str, tuple[str, ...]] = {
    "public_ip_key": (
        "public_ip_key",
        "public_ip_entity_id",
        "public_ip_changed",
        "public_ip_previous_key",
    ),
}


def posture_is_capability(key: str) -> bool:
    """Return whether a posture key gates rules (vs. describing a fact)."""
    return key not in _POSTURE_NON_CAPABILITY_KEYS and not key.endswith(
        _ENTITY_TWIN_SUFFIX
    )


# Client fields a later source may leave unset while an earlier one knew them.
_CLIENT_FILL_FIELDS: tuple[str, ...] = (
    "name",
    "ip",
    "hostname",
    "manufacturer",
    "connection_type",
    "network_name",
    "last_seen",
    "ha_device_id",
    "ha_integration",
    "tracker_entity_id",
    "vlan",
)
COUNTER_CLIENT_COUNT = "network.client_count"


def _merge_client(earlier: NetworkClient, incoming: NetworkClient) -> NetworkClient:
    """
    Return *incoming* over *earlier* for one client key.

    The later source is fresher, so its ``connected`` and every value it
    supplies win; a field it does not know (a runtime source has no tracker
    entity id, a tracker has no eero nickname) keeps the earlier source's
    value rather than going blank.
    """
    merged: dict[str, Any] = {**earlier, **incoming}
    for field_name in _CLIENT_FILL_FIELDS:
        if incoming.get(field_name) is None and earlier.get(field_name) is not None:
            merged[field_name] = earlier[field_name]
    # Which network a client is on belongs to the observation that says it is
    # connected: an older source's guest flag (a tracker attribute from the
    # last reload) must not describe the fresher source's connection.
    if not isinstance(incoming.get("is_guest"), bool):
        merged.pop("is_guest", None)
    return cast("NetworkClient", merged)


def _merge_posture(
    result: AdapterResult, posture: dict[str, Any], sources: dict[str, str]
) -> None:
    """Fold one adapter's posture into the merged posture and its sources."""
    for lead, members in _POSTURE_GROUPS.items():
        if lead in result.posture:
            for member in members:
                posture.pop(member, None)
                sources.pop(posture_cap(member), None)
    for key in result.posture_withdrawn:
        posture.pop(key, None)
        posture.pop(f"{key}{_ENTITY_TWIN_SUFFIX}", None)
        sources.pop(posture_cap(key), None)
    for key, value in result.posture.items():
        posture[key] = value
        if posture_is_capability(key):
            sources[posture_cap(key)] = result.name
            twin = f"{key}{_ENTITY_TWIN_SUFFIX}"
            if twin not in result.posture:
                # The earlier source's entity no longer describes this
                # value; a stale twin would name the wrong evidence.
                posture.pop(twin, None)


def merge_adapter_results(results: Iterable[AdapterResult]) -> NetworkSnapshot:  # noqa: PLR0912 - one branch per section, kept flat on purpose
    """
    Merge adapter outputs into one section and derive the capability list.

    Later adapters override earlier ones for the same key, so callers order
    generic adapters before router-specific ones. Clients are merged by
    ``key``: a router-specific adapter that sees a client the generic tracker
    adapter also saw replaces that client rather than duplicating it.
    """
    posture: dict[str, Any] = {}
    ha_security: dict[str, Any] = {}
    clients: dict[str, NetworkClient] | None = None
    new_clients: list[str] | None = None
    counters: dict[str, float] = {}
    sources: dict[str, str] = {}
    notes: list[str] = []
    radio: dict[str, Any] = {}
    radio_posture: dict[str, Any] = {}
    radio_caps: set[str] = set()
    for result in results:
        _merge_posture(result, posture, sources)
        for key, value in result.ha_security.items():
            ha_security[key] = value
            sources[ha_cap(key)] = result.name
        if result.clients is not None:
            clients = clients or {}
            for incoming in result.clients:
                earlier = clients.get(incoming["key"])
                clients[incoming["key"]] = (
                    _merge_client(earlier, incoming) if earlier else incoming
                )
            sources[CAP_CLIENTS] = result.name
        if result.new_clients is not None:
            new_clients = sorted({*(new_clients or []), *result.new_clients})
            sources[CAP_NEW_CLIENTS] = result.name
        for key, value in result.radio.items():
            radio[key] = value
            if key not in _RADIO_NON_CAPABILITY_KEYS:
                sources[f"{_CAP_RADIO_PREFIX}{key}"] = result.name
                radio_caps.add(f"{_CAP_RADIO_PREFIX}{key}")
        for key, value in result.radio_posture.items():
            radio_posture[key] = value
            path = f"{_CAP_RADIO_PREFIX}posture.{key}"
            sources[path] = result.name
            radio_caps.add(path)
        for path in result.extra_capabilities:
            sources[path] = result.name
            if path.startswith(_CAP_RADIO_PREFIX):
                radio_caps.add(path)
        counters.update(result.counters)
        notes.extend(result.notes)
    if clients is not None:
        _publish_guest_flag(clients.values(), sources, notes)
        if any(
            c.get("connected") and c.get("data_up_day_bytes") is not None
            for c in clients.values()
        ):
            sources[CAP_CLIENT_DATA_DAY] = sources[CAP_CLIENTS]
        counters[COUNTER_CLIENT_COUNT] = float(
            sum(1 for c in clients.values() if c.get("connected"))
        )
    section: dict[str, Any] = {
        "capabilities": sorted(sources),
        "sources": sources,
        "clients": list(clients.values()) if clients else [],
        "posture": posture,  # type: ignore[typeddict-item]
        "ha_security": ha_security,  # type: ignore[typeddict-item]
        "counters": counters,
        "notes": notes,
    }
    if new_clients is not None:
        section["new_clients"] = new_clients
    if radio or radio_posture:
        section["radio"] = {
            "capabilities": sorted(radio_caps),
            "devices": radio.get("devices", []),
            "posture": radio_posture,
            **{k: v for k, v in radio.items() if k != "devices"},
        }
    return section  # type: ignore[return-value]


GUEST_FLAG_PARTIAL_NOTE = (
    "Guest Wi-Fi: {count} connected client(s) come from a router integration "
    "that does not say which network they are on, so they are not checked."
)


def _publish_guest_flag(
    clients: Iterable[Mapping[str, Any]], sources: dict[str, str], notes: list[str]
) -> None:
    """
    Publish the guest-flag capability for the merged client list.

    Decided after the merge because the list mixes sources. A client flagged
    as a guest is one whatever the others say, so one source reporting the
    flag is enough for the rule to run; connected clients without the flag
    (a second router integration's trackers) are counted in a note, since
    for them "not reported" does not mean "not a guest".
    """
    merged = list(clients)
    if not any(isinstance(c.get("is_guest"), bool) for c in merged):
        return
    sources[CAP_GUEST_CLIENTS] = sources[CAP_CLIENTS]
    uncovered = sum(
        1
        for c in merged
        if c.get("connected") and not isinstance(c.get("is_guest"), bool)
    )
    if uncovered:
        notes.append(GUEST_FLAG_PARTIAL_NOTE.format(count=uncovered))


# Names a router gives a client it cannot resolve; matching one of these
# against a trusted row would call a stranger "a known device".
_PLACEHOLDER_NAME_WORDS: tuple[str, ...] = ("unknown", "unnamed", "device", "[")
_MIN_MATCHABLE_NAME_CHARS = 4


def _matchable_names(client: Mapping[str, Any]) -> set[str]:
    """Return the client's names that are specific enough to identify it."""
    names = set()
    for value in (client.get("hostname"), client.get("name")):
        text = str(value or "").strip().lower()
        if len(text) < _MIN_MATCHABLE_NAME_CHARS or any(
            word in text for word in _PLACEHOLDER_NAME_WORDS
        ):
            continue
        names.add(text)
    return names


def attach_inventory(section: NetworkSnapshot, context: NetworkBuildContext) -> None:
    """
    Compare the merged client list with the device inventory, once per run.

    Runs after the merge so every source's clients are diffed together:
    attaches each recorded client's ``first_seen`` (the rule applies the
    grace period from it) and ``trusted`` verdict, marks a rotated random
    address whose name a trusted row carries, and publishes ``new_clients``
    with its capability.
    A section without the clients capability is left alone.
    """
    from custom_components.home_generative_agent.sentinel.network_inventory import (  # noqa: PLC0415
        ROUTER_SOURCE,
        client_key,
        client_observation,
    )

    inventory = context.network_inventory
    if inventory is None or CAP_CLIENTS not in section["capabilities"]:
        return
    trusted_names = inventory.trusted_names(ROUTER_SOURCE)
    observations = []
    for client in section["clients"]:
        row = inventory.row(client_key(client["key"]))
        if row is not None and isinstance(row.get("first_seen"), str):
            client["first_seen"] = row["first_seen"]
        if row is not None:
            client["trusted"] = bool(row.get("trusted"))
        if client.get("mac_randomized"):
            client["hostname_trusted"] = bool(_matchable_names(client) & trusted_names)
        observations.append(client_observation(client))
    delta = inventory.diff(observations, [ROUTER_SOURCE])
    prefix = f"{ROUTER_SOURCE}:"
    section["new_clients"] = [
        k.removeprefix(prefix) for k in delta.new_device_keys if k.startswith(prefix)
    ]
    _publish_capability(section, CAP_NEW_CLIENTS, "inventory")
    LOGGER.debug(
        "Network clients: %d from %s, %d connected, %d new to the inventory.",
        len(section["clients"]),
        section["sources"].get(CAP_CLIENTS),
        sum(1 for c in section["clients"] if c.get("connected")),
        len(section["new_clients"]),
    )


# The idle clock trusts its memory only across gaps this short: a longer
# stretch with no observation (the router integration removed, Home Assistant
# down) is time nobody watched, not time the guest network sat idle.
GUEST_OBSERVATION_GAP = timedelta(days=2)


def _day(moment: datetime) -> datetime:
    """Return *moment* in UTC truncated to its day."""
    return dt_util.as_utc(moment).replace(hour=0, minute=0, second=0, microsecond=0)


def _remembered_day(
    previous: Mapping[str, Any], key: str, today: datetime
) -> datetime | None:
    """Return a remembered day, or None when absent, garbled, or in the future."""
    parsed = dt_util.parse_datetime(str(previous.get(key) or ""))
    if parsed is None:
        return None
    day = _day(parsed)
    # A clock that once ran ahead must not pin the idle time at zero forever.
    return day if day <= today + timedelta(days=1) else None


def derive_guest_idle(
    section: NetworkSnapshot, context: NetworkBuildContext, now: datetime
) -> None:
    """
    Work out how long the guest Wi-Fi has sat unused, once per run.

    Runs after the merge. An *observation* needs the guest network's state
    and its connected-guest count from the same source (two sources can cover
    different networks, and mixing them would fabricate an idle network); a
    run without one leaves the engine's posture memory untouched, so a read
    gap neither restarts the clock nor counts as idle time.

    The memory holds two days, truncated to the day so it changes (and is
    written) at most daily: ``guest_network_last_active``, when a guest was
    last connected or the clock was restarted, and
    ``guest_network_last_observed``, when the network was last observed at
    all. A guest connected now, a network that is off, no usable memory, or
    more than :data:`GUEST_OBSERVATION_GAP` since the last observation all
    restart the clock; otherwise the remembered day stands and
    ``guest_network_idle_days`` is published. The first observation only
    starts the clock, so the rule is listed as not run rather than guessing.
    """
    posture: dict[str, Any] = section["posture"]  # type: ignore[assignment]
    enabled = posture.get("guest_network_enabled")
    count = posture.get("guest_client_count")
    sources = section["sources"]
    if (
        not isinstance(enabled, bool)
        or not isinstance(count, int)
        or isinstance(count, bool)
        or sources.get(posture_cap("guest_network_enabled"))
        != sources.get(posture_cap("guest_client_count"))
    ):
        return
    today = _day(now)
    previous = context.previous_posture or {}
    last_active = _remembered_day(previous, "guest_network_last_active", today)
    last_observed = _remembered_day(previous, "guest_network_last_observed", today)
    watched = (
        last_active is not None
        and last_observed is not None
        and today - last_observed <= GUEST_OBSERVATION_GAP
    )
    posture["guest_network_last_observed"] = today.isoformat()
    if not enabled or count > 0 or not watched or last_active is None:
        posture["guest_network_last_active"] = today.isoformat()
        if enabled and count > 0:
            posture["guest_network_idle_days"] = 0
            _publish_capability(
                section, posture_cap("guest_network_idle_days"), "posture_memory"
            )
        return
    posture["guest_network_last_active"] = last_active.isoformat()
    posture["guest_network_idle_days"] = max(0, (today - last_active).days)
    _publish_capability(
        section, posture_cap("guest_network_idle_days"), "posture_memory"
    )


def _publish_capability(section: NetworkSnapshot, path: str, source: str) -> None:
    """Record a capability a post-merge step established, and who supplied it."""
    section["sources"][path] = source
    section["capabilities"] = sorted(section["sources"])


def empty_network_snapshot(note: str | None = None) -> NetworkSnapshot:
    """Return a section with no capabilities (audit disabled or not built)."""
    return {
        "capabilities": [],
        "sources": {},
        "clients": [],
        "posture": {},
        "ha_security": {},
        "counters": {},
        "notes": [note] if note else [],
    }


# ---------------------------------------------------------------------------
# ha_native inputs (collected with I/O) and adapter (pure)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AddonInput:
    """Supervisor add-on facts relevant to exposure."""

    slug: str
    name: str
    host_ports: tuple[int, ...]
    protected: bool
    running: bool
    # False when the Supervisor has not reported this add-on's details yet;
    # ``host_ports`` and ``protected`` are then placeholders, not facts.
    info_known: bool = True


@dataclass(frozen=True)
class AutomationInput:
    """
    An automation entity and its configuration.

    ``config`` carries ``triggers`` and ``actions`` in Home Assistant's
    validated form (blueprints substituted, ``service:`` normalized to
    ``action:``) when the entity exposes them, falling back to the raw
    configuration otherwise.
    """

    entity_id: str
    config: Mapping[str, Any]


@dataclass(frozen=True)
class HttpInput:
    """HTTP server settings that matter for exposure."""

    trusted_proxies_configured: bool
    ip_ban_enabled: bool
    login_attempts_threshold: int


@dataclass
class HaNativeInputs:
    """
    Everything ``ha_native_adapter`` reads, collected by the async layer.

    ``None`` for a field means the read failed or the component is absent on
    this install; the adapter then leaves the matching capabilities out.
    """

    now: datetime
    auth: list[ObservedUser] | None = None
    failed_login_notification_present: bool | None = None
    exposed: dict[str, list[str]] | None = None  # assistant -> exposed ids
    cloud_remote_ui_enabled: bool | None = None
    http: HttpInput | None = None
    trusted_networks_bypass_login: bool | None = None
    addons: list[AddonInput] | None = None
    automations: list[AutomationInput] | None = None
    # Conversation engines used by Assist pipelines that this integration's
    # Critical Action PIN does not cover (built-in agent, other LLM agents).
    assist_agents_outside_pin: list[str] | None = None
    discovered_unconfigured: list[dict[str, str]] | None = None
    discovered_ignored: list[dict[str, str]] | None = None
    # Device-registry facts for tagging updates: entity -> device, device ->
    # set of entity domains it owns. Shared with the builder, never copied.
    entity_device: Mapping[str, str] = field(default_factory=dict)
    device_domains: Mapping[str, set[str]] = field(default_factory=dict)


def log_input_failure(name: str, err: Exception) -> None:
    """Log a failed runtime read once per input and exception class."""
    key = f"{name}:{type(err).__name__}"
    if key in _LOGGED_INPUT_FAILURES:
        return
    _LOGGED_INPUT_FAILURES.add(key)
    LOGGER.warning(
        "Network audit: could not read %s (%s: %s); the related checks are "
        "reported as missing capabilities.",
        name,
        type(err).__name__,
        err,
    )


_log_input_failure = log_input_failure


def _iso(value: Any) -> str | None:
    if value is None:
        return None
    if hasattr(value, "isoformat"):
        return dt_util.as_utc(value).isoformat()
    return str(value)


async def async_collect_auth_observation(
    hass: HomeAssistant, pseudonymizer: Pseudonymizer | None
) -> list[ObservedUser] | None:
    """
    Read users and refresh tokens from ``hass.auth`` without secrets.

    Token values, JWT keys, and credentials are never touched; the last-used
    address is pseudonymized before it leaves this function (or dropped when
    no pseudonymizer is available).
    """
    try:
        users = await hass.auth.async_get_users()
    except Exception as err:  # noqa: BLE001 - runtime read across HA versions
        _log_input_failure("auth users", err)
        return None
    observation: list[ObservedUser] = []
    try:
        for user in users:
            tokens: list[ObservedToken] = []
            for token in user.refresh_tokens.values():
                ip = getattr(token, "last_used_ip", None)
                tokens.append(
                    ObservedToken(
                        token_id=str(token.id),
                        user_id=str(user.id),
                        token_type=str(token.token_type),
                        client_name=(
                            sanitize_label(getattr(token, "client_name", None)) or None
                        ),
                        created_at=_iso(getattr(token, "created_at", None)),
                        last_used_at=_iso(getattr(token, "last_used_at", None)),
                        last_used_ip_key=(
                            pseudonymizer.ip_key(str(ip))
                            if ip and pseudonymizer is not None
                            else None
                        ),
                    )
                )
            observation.append(
                ObservedUser(
                    user_id=str(user.id),
                    name=sanitize_label(user.name) or None,
                    is_admin=bool(user.is_admin),
                    is_active=bool(user.is_active),
                    system_generated=bool(user.system_generated),
                    tokens=tuple(tokens),
                )
            )
    except Exception as err:  # noqa: BLE001
        _log_input_failure("auth token attributes", err)
        return None
    return observation


def _collect_failed_login(hass: HomeAssistant) -> bool | None:
    try:
        notifications = hass.data.get("persistent_notification")
    except Exception as err:  # noqa: BLE001
        _log_input_failure("persistent notifications", err)
        return None
    if notifications is None:
        return False
    return LOGIN_NOTIFICATION_ID in notifications


def _collect_exposed(
    hass: HomeAssistant, entity_ids: Sequence[str]
) -> dict[str, list[str]] | None:
    """
    Return ``assistant -> exposed entity ids`` for the given entities.

    Uses Home Assistant's own decision helper so entities that were never
    explicitly configured but fall under an assistant's default exposure
    (covers are a default-exposed domain) count as exposed, exactly as the
    assistant itself would treat them. Only the handful of sensitive entities
    is asked about, not the whole registry.
    """
    try:
        from homeassistant.components.homeassistant.const import (  # noqa: PLC0415
            DATA_EXPOSED_ENTITIES,
        )
        from homeassistant.components.homeassistant.exposed_entities import (  # noqa: PLC0415
            KNOWN_ASSISTANTS,
            async_should_expose,
        )

        if DATA_EXPOSED_ENTITIES not in hass.data:
            # The exposed-entities registry is created by the homeassistant
            # component at startup; absent means we cannot audit exposure.
            return None
        return {
            assistant: sorted(
                entity_id
                for entity_id in entity_ids
                if async_should_expose(hass, assistant, entity_id)
            )
            for assistant in KNOWN_ASSISTANTS
        }
    except Exception as err:  # noqa: BLE001
        _log_input_failure("exposed entities", err)
        return None


def _collect_assist_agents(hass: HomeAssistant) -> list[str] | None:
    """
    Return the Assist pipeline agents the Critical Action PIN does not cover.

    The exposure registry is shared by every conversation agent, but the PIN
    guards only this integration's own agent. A pipeline on Home Assistant's
    built-in agent (whose turn-off intent unlocks locks) or on another LLM
    integration can unlock an exposed lock with no PIN at all, so the
    exposure rule must know such pipelines exist. An empty list means every
    pipeline uses this integration; None means the read failed.
    """
    try:
        from homeassistant.components.assist_pipeline import (  # noqa: PLC0415
            async_get_pipelines,
        )
        from homeassistant.helpers import entity_registry as er  # noqa: PLC0415

        own = {
            entry.entity_id
            for entry in er.async_get(hass).entities.values()
            if entry.domain == "conversation" and entry.platform == DOMAIN
        }
        try:
            pipelines = async_get_pipelines(hass)
        except KeyError:
            # assist_pipeline not loaded: no voice pipeline exists at all.
            return []
        return sorted(
            {
                str(pipeline.conversation_engine)
                for pipeline in pipelines
                if str(pipeline.conversation_engine) not in own
            }
        )
    except Exception as err:  # noqa: BLE001
        _log_input_failure("Assist pipelines", err)
        return None


def _collect_cloud(hass: HomeAssistant) -> bool | None:
    if "cloud" not in hass.config.components:
        return None
    try:
        cloud = hass.data.get("cloud")
        if cloud is None:
            return None
        remote_enabled = bool(cloud.client.prefs.remote_enabled)
        logged_in = bool(getattr(cloud, "is_logged_in", False))
        return remote_enabled and logged_in  # noqa: TRY300
    except Exception as err:  # noqa: BLE001
        _log_input_failure("cloud remote UI preference", err)
        return None


def _collect_http(hass: HomeAssistant) -> HttpInput | None:
    server = getattr(hass, "http", None)
    if server is None:
        return None
    try:
        from homeassistant.components.http.ban import (  # noqa: PLC0415
            KEY_BAN_MANAGER,
            KEY_LOGIN_THRESHOLD,
        )

        app = server.app
        threshold = int(app.get(KEY_LOGIN_THRESHOLD, -1))
        # ip_ban_enabled reflects the http option (a ban manager exists);
        # whether a ban can ever trigger is the separate threshold fact: Home
        # Assistant ships ip_ban_enabled=true with no threshold (-1).
        return HttpInput(
            trusted_proxies_configured=bool(getattr(server, "trusted_proxies", None)),
            ip_ban_enabled=KEY_BAN_MANAGER in app,
            login_attempts_threshold=threshold,
        )
    except Exception as err:  # noqa: BLE001
        _log_input_failure("HTTP server settings", err)
        return None


def _collect_auth_providers(hass: HomeAssistant) -> bool | None:
    try:
        for provider in hass.auth.auth_providers:
            if getattr(provider, "type", None) != "trusted_networks":
                continue
            if bool(provider.config.get("allow_bypass_login", False)):
                return True
        return False  # noqa: TRY300
    except Exception as err:  # noqa: BLE001
        _log_input_failure("auth providers", err)
        return None


def _collect_addons(hass: HomeAssistant) -> list[AddonInput] | None:
    """
    Read installed add-ons from the Supervisor coordinator's caches.

    The add-on *list* (name, state) is refreshed on every coordinator update;
    the per-add-on *info* (host ports, protection mode) is fetched for every
    add-on at the first update and afterwards only for add-ons with
    subscribed entities, so ports and protection mode can lag until the next
    Home Assistant restart. Running state therefore comes from the list.
    """
    try:
        from homeassistant.helpers.hassio import is_hassio  # noqa: PLC0415

        if not is_hassio(hass):
            return None
        from homeassistant.components.hassio import (  # noqa: PLC0415
            HassioNotReadyError,
            get_addons_info,
            get_addons_list,
        )
    except Exception as err:  # noqa: BLE001
        _log_input_failure("Supervisor add-on info", err)
        return None
    try:
        listed = get_addons_list(hass)
        info = get_addons_info(hass)
    except HassioNotReadyError:
        # Normal during the first cycles after boot: the Supervisor
        # coordinator has not fetched add-on info yet. Not worth a warning.
        return None
    except Exception as err:  # noqa: BLE001
        _log_input_failure("Supervisor add-on info", err)
        return None
    addons: list[AddonInput] = []
    for entry in listed or []:
        if not isinstance(entry, dict) or not entry.get("slug"):
            continue
        slug = str(entry["slug"])
        raw = info.get(slug) if isinstance(info, dict) else None
        info_known = isinstance(raw, dict)
        data = raw if isinstance(raw, dict) else {}
        ports: list[int] = []
        network = data.get("network")
        if isinstance(network, dict):
            ports = sorted(
                int(host_port)
                for host_port in network.values()
                if isinstance(host_port, int) and not isinstance(host_port, bool)
            )
        addons.append(
            AddonInput(
                slug=slug,
                name=sanitize_label(entry.get("name") or data.get("name") or slug),
                host_ports=tuple(ports),
                protected=bool(data.get("protected", True)),
                running=str(entry.get("state") or data.get("state") or "") == "started",
                info_known=info_known,
            )
        )
    return addons


def _automation_config(entity: Any) -> Mapping[str, Any] | None:
    """
    Return an automation's validated ``triggers`` / ``actions``.

    ``AutomationEntity`` keeps the validated trigger list in
    ``_trigger_config`` and the validated action sequence on
    ``action_script.sequence`` (both post blueprint substitution). A blueprint
    automation's ``raw_config`` holds only ``use_blueprint``, so the raw form
    is the fallback, not the source.
    """
    triggers = getattr(entity, "_trigger_config", None)
    script = getattr(entity, "action_script", None)
    actions = getattr(script, "sequence", None)
    if isinstance(triggers, list) and isinstance(actions, list):
        return {"triggers": triggers, "actions": actions}
    raw = getattr(entity, "raw_config", None)
    return raw if isinstance(raw, dict) else None


def _collect_automations(hass: HomeAssistant) -> list[AutomationInput] | None:
    if "automation" not in hass.config.components:
        return None
    try:
        component = hass.data.get("automation")
        if component is None:
            return None
        automations: list[AutomationInput] = []
        for entity in component.entities:
            config = _automation_config(entity)
            if config is not None:
                automations.append(AutomationInput(entity.entity_id, config))
        return automations  # noqa: TRY300
    except Exception as err:  # noqa: BLE001
        _log_input_failure("automation configurations", err)
        return None


def _collect_discovery(
    hass: HomeAssistant,
) -> tuple[list[dict[str, str]] | None, list[dict[str, str]] | None]:
    unconfigured: list[dict[str, str]] | None
    ignored: list[dict[str, str]] | None
    try:
        unconfigured = []
        for flow in hass.config_entries.flow.async_progress():
            context = flow.get("context") or {}
            source = str(context.get("source") or "")
            if source not in DISCOVERY_SOURCES:
                continue
            placeholders = context.get("title_placeholders") or {}
            unconfigured.append(
                {
                    "handler": str(flow.get("handler") or ""),
                    "source": source,
                    "title": sanitize_label(placeholders.get("name")),
                }
            )
        unconfigured.sort(key=lambda d: (d["handler"], d["title"], d["source"]))
    except Exception as err:  # noqa: BLE001
        _log_input_failure("discovery flows", err)
        unconfigured = None
    try:
        ignored = []
        for entry in hass.config_entries.async_entries(include_ignore=True):
            if entry.source != "ignore":
                continue
            ignored.append(
                {
                    "handler": entry.domain,
                    "source": "ignore",
                    "title": sanitize_label(entry.title),
                }
            )
        ignored.sort(key=lambda d: (d["handler"], d["title"]))
    except Exception as err:  # noqa: BLE001
        _log_input_failure("ignored config entries", err)
        ignored = None
    return unconfigured, ignored


async def async_collect_ha_native_inputs(  # noqa: PLR0913
    hass: HomeAssistant,
    context: NetworkBuildContext,
    entities: Sequence[SnapshotEntity],
    *,
    now: datetime,
    entity_device: Mapping[str, str],
    device_domains: Mapping[str, set[str]],
) -> HaNativeInputs:
    """Read every HA-native input, each guarded independently."""
    auth = context.auth_observation
    if auth is None:
        auth = await async_collect_auth_observation(hass, context.pseudonymizer)
    unconfigured, ignored = _collect_discovery(hass)
    sensitive_ids = [e["entity_id"] for e in entities if is_sensitive_entity(e)]
    return HaNativeInputs(
        now=now,
        auth=auth,
        failed_login_notification_present=_collect_failed_login(hass),
        exposed=_collect_exposed(hass, sensitive_ids),
        cloud_remote_ui_enabled=_collect_cloud(hass),
        http=_collect_http(hass),
        trusted_networks_bypass_login=_collect_auth_providers(hass),
        addons=_collect_addons(hass),
        automations=_collect_automations(hass),
        assist_agents_outside_pin=_collect_assist_agents(hass),
        discovered_unconfigured=unconfigured,
        discovered_ignored=ignored,
        entity_device=entity_device,
        device_domains=device_domains,
    )


# --- pure helpers ----------------------------------------------------------


def _days_between(earlier: str | None, now: datetime) -> int | None:
    if not earlier:
        return None
    parsed = dt_util.parse_datetime(earlier)
    if parsed is None:
        return None
    return max(0, (dt_util.as_utc(now) - dt_util.as_utc(parsed)).days)


def _minutes_between(earlier: str, now: datetime) -> int | None:
    parsed = dt_util.parse_datetime(earlier)
    if parsed is None:
        return None
    seconds = (dt_util.as_utc(now) - dt_util.as_utc(parsed)).total_seconds()
    return max(0, int(seconds // 60))


def is_sensitive_entity(entity: SnapshotEntity) -> bool:
    """
    Return True for an entity whose voice exposure is a security concern.

    Locks and alarm panels always; covers only when they are a door, gate,
    or garage (by device class or name), mirroring the critical-action
    matcher so "sensitive" means the same thing in both places.
    """
    domain = entity["domain"]
    if domain in {"lock", "alarm_control_panel"}:
        return True
    if domain != "cover":
        return False
    device_class = str(entity["attributes"].get("device_class") or "").lower()
    if device_class in _ENTRY_COVER_CLASSES:
        return True
    haystack = f"{entity['entity_id']} {entity['friendly_name'] or ''}".lower()
    return _ENTRY_COVER_HINT_RE.search(haystack.replace("_", " ")) is not None


def automation_calls_critical_action(
    config: Mapping[str, Any], critical_actions: Sequence[Mapping[str, str]]
) -> bool:
    """
    Return True when any action in *config* can perform a critical action.

    Delegates to the screener the add-automation PIN gate uses: an allowlist
    over Home Assistant's script-action taxonomy that also recognizes device
    actions, ``homeassistant.turn_on`` aimed at a guarded domain, scene state
    reproduction, and script/scene indirection, and fails closed on steps
    whose real effect cannot be determined.
    """
    return bool(find_critical_automation_calls(config, critical_actions))


def automation_has_public_webhook(config: Mapping[str, Any]) -> bool:
    """
    Return True when a webhook trigger is reachable beyond the local network.

    Home Assistant defaults ``local_only`` to True, so only an explicit False
    counts.
    """
    triggers = config.get("triggers", config.get("trigger"))
    if isinstance(triggers, dict):
        triggers = [triggers]
    if not isinstance(triggers, list):
        return False
    for trigger in triggers:
        if not isinstance(trigger, dict):
            continue
        kind = trigger.get("trigger") or trigger.get("platform")
        if kind != "webhook":
            continue
        if trigger.get("local_only", True) is False:
            return True
    return False


def _auth_fields(
    inputs: HaNativeInputs, context: NetworkBuildContext
) -> dict[str, Any]:
    auth = inputs.auth
    auth_inventory = context.auth_inventory
    if auth is None:
        return {}
    labels = token_labels(auth)
    admins = 0
    long_lived = 0
    age_days: dict[str, int] = {}
    for user in auth:
        if user.is_admin and user.is_active and not user.system_generated:
            admins += 1
        for token in user.tokens:
            if token.token_type != TOKEN_TYPE_LONG_LIVED:
                continue
            long_lived += 1
            # Home Assistant logs refresh-token usage only on the access-token
            # exchange, which a long-lived token performs once at creation;
            # age is the only observable fact about it.
            days = _days_between(token.created_at, inputs.now)
            if days is not None:
                age_days[labels[token.token_id]] = days
    fields: dict[str, Any] = {
        "admin_user_count": admins,
        "long_lived_token_count": long_lived,
        "long_lived_token_age_days": age_days,
    }
    if auth_inventory is not None:
        fingerprint = (
            context.pseudonymizer.fingerprint
            if context.pseudonymizer is not None
            else None
        )
        delta = auth_inventory.diff(auth, salt_fingerprint=fingerprint)
        if not delta.bootstrap:
            fields["new_admin_users"] = list(delta.new_admin_users)
            fields["new_long_lived_tokens"] = list(delta.new_long_lived_tokens)
    return fields


def _update_fields(
    inputs: HaNativeInputs, entities: Sequence[SnapshotEntity]
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Return (ha_security fields, posture fields) derived from update entities."""
    pending: list[str] = []
    security_pending: list[str] = []
    router_pending: list[str] = []
    router_observed = False
    for entity in entities:
        if entity["domain"] != "update":
            continue
        platform = entity.get("platform") or ""
        is_router = platform in ROUTER_PLATFORMS
        router_observed = router_observed or is_router
        if entity["state"] != "on":
            continue
        entity_id = entity["entity_id"]
        pending.append(entity_id)
        device_id = inputs.entity_device.get(entity_id)
        domains = inputs.device_domains.get(device_id, set()) if device_id else set()
        if is_router or domains & SECURITY_DOMAINS:
            security_pending.append(entity_id)
        if is_router:
            router_pending.append(entity_id)
    ha_fields = {
        "pending_updates": sorted(pending),
        "pending_security_updates": sorted(security_pending),
    }
    posture: dict[str, Any] = {}
    if router_observed:
        posture["router_update_pending"] = bool(router_pending)
        posture["router_update_entities"] = sorted(router_pending)
    return ha_fields, posture


def _addon_fields(inputs: HaNativeInputs, result: AdapterResult) -> None:
    """Fill the add-on exposure fields; unknown add-ons are named, not assumed safe."""
    ha = result.ha_security
    if inputs.addons is not None:
        # An add-on whose details the Supervisor has not fetched yet carries
        # placeholder ports and protection; it is skipped and named, never
        # asserted safe.
        ha["addons_with_host_ports"] = {
            addon.slug: list(addon.host_ports)
            for addon in inputs.addons
            if addon.running and addon.info_known and addon.host_ports
        }
        ha["addons_unprotected"] = sorted(
            addon.slug
            for addon in inputs.addons
            if addon.running and addon.info_known and not addon.protected
        )
        ha["addon_names"] = {addon.slug: addon.name for addon in inputs.addons}
        pending = sorted(
            addon.name
            for addon in inputs.addons
            if addon.running and not addon.info_known
        )
        if pending:
            result.notes.append(
                "Supervisor has not reported details for "
                f"{len(pending)} running add-on(s) ({', '.join(pending)}); "
                "their host ports and protection mode are not audited yet."
            )
    else:
        result.notes.append(
            "Supervisor add-on data is unavailable on this install type; "
            "add-on exposure is not audited."
        )


def ha_native_adapter(  # noqa: PLR0912 - one branch per input, kept flat on purpose
    inputs: HaNativeInputs,
    entities: Sequence[SnapshotEntity],
    context: NetworkBuildContext,
) -> AdapterResult:
    """Pure adapter over HA-native inputs and the entity snapshot."""
    result = AdapterResult(name="ha_native")
    ha = result.ha_security

    ha.update(_auth_fields(inputs, context))

    if inputs.failed_login_notification_present is not None:
        ha["failed_login_notification_present"] = (
            inputs.failed_login_notification_present
        )

    if inputs.exposed is not None:
        by_id = {e["entity_id"]: e for e in entities}
        ha["exposed_sensitive_entities"] = {
            assistant: sorted(
                entity_id
                for entity_id in ids
                if entity_id in by_id and is_sensitive_entity(by_id[entity_id])
            )
            for assistant, ids in inputs.exposed.items()
        }

    if context.options is not None:
        ha["critical_action_pin_enabled"] = resolve_critical_action_policy(
            context.options
        ).enforceable
    if inputs.assist_agents_outside_pin is not None:
        ha["assist_agents_outside_pin"] = list(inputs.assist_agents_outside_pin)

    if inputs.cloud_remote_ui_enabled is not None:
        ha["cloud_remote_ui_enabled"] = inputs.cloud_remote_ui_enabled
    else:
        result.notes.append(
            "Home Assistant Cloud is not loaded; remote UI not audited."
        )

    update_fields, posture_fields = _update_fields(inputs, entities)
    ha.update(update_fields)
    result.posture.update(posture_fields)

    if inputs.http is not None:
        # Only the proxy list is observable on the server object; the
        # use_x_forwarded_for flag is consumed at setup and never stored, so
        # the snapshot does not pretend to know it.
        ha["http_trusted_proxies_configured"] = inputs.http.trusted_proxies_configured
        ha["http_ip_ban_enabled"] = inputs.http.ip_ban_enabled
        ha["http_login_attempts_threshold"] = inputs.http.login_attempts_threshold

    if inputs.trusted_networks_bypass_login is not None:
        ha["trusted_networks_bypass_login"] = inputs.trusted_networks_bypass_login

    _addon_fields(inputs, result)

    if inputs.automations is not None:
        critical_actions = (
            resolve_critical_action_policy(context.options).critical_actions
            if context.options is not None
            else resolve_critical_action_policy({}).critical_actions
        )
        public = sorted(
            a.entity_id
            for a in inputs.automations
            if automation_has_public_webhook(a.config)
        )
        ha["webhook_automations_public"] = public
        public_set = set(public)
        ha["webhook_automations_critical"] = sorted(
            a.entity_id
            for a in inputs.automations
            if a.entity_id in public_set
            and automation_calls_critical_action(a.config, critical_actions)
        )

    unavailable: dict[str, int] = {}
    for entity in entities:
        if entity["domain"] not in SECURITY_DOMAINS or entity["state"] != "unavailable":
            continue
        minutes = _minutes_between(entity["last_changed"], inputs.now)
        if minutes is not None:
            unavailable[entity["entity_id"]] = minutes
    ha["unavailable_security_devices"] = unavailable

    if inputs.discovered_unconfigured is not None:
        ha["discovered_unconfigured"] = list(inputs.discovered_unconfigured)
    if inputs.discovered_ignored is not None:
        ha["discovered_ignored"] = list(inputs.discovered_ignored)

    return result


# ---------------------------------------------------------------------------
# Entry point used by the snapshot builder
# ---------------------------------------------------------------------------


async def async_build_network_snapshot(  # noqa: PLR0913
    hass: HomeAssistant,
    entities: Sequence[SnapshotEntity],
    context: NetworkBuildContext | None,
    *,
    now: datetime,
    entity_device: Mapping[str, str],
    device_domains: Mapping[str, set[str]],
) -> NetworkSnapshot:
    """
    Build the ``network`` section; never raises for a failed input read.

    Only the Sentinel engine passes a context. Every other snapshot consumer
    (baseline, discovery, the preview service) gets an empty section, so the
    auth store and Supervisor are read on the detection cadence alone and the
    user's master switch is honored everywhere.
    """
    if context is None:
        return empty_network_snapshot("Network section not requested by caller.")
    if not context.enabled:
        return empty_network_snapshot("Network audit is disabled in Sentinel options.")
    # Imported here: the radio, UPnP, and router adapters build on this
    # module's helpers.
    from .eero import collect_eero_runtime_inputs, eero_runtime_adapter  # noqa: PLC0415
    from .radio import collect_radio_inputs, radio_adapters  # noqa: PLC0415
    from .router import async_collect_router_inputs, router_adapters  # noqa: PLC0415
    from .upnp import async_collect_upnp_inputs, upnp_igd_adapter  # noqa: PLC0415

    inputs = await async_collect_ha_native_inputs(
        hass,
        context,
        entities,
        now=now,
        entity_device=entity_device,
        device_domains=device_domains,
    )
    radio_inputs = collect_radio_inputs(
        hass, entity_device=entity_device, device_domains=device_domains
    )
    upnp_inputs = await async_collect_upnp_inputs(hass)
    router_inputs = async_collect_router_inputs(hass)
    eero_inputs = collect_eero_runtime_inputs(hass)
    # Router adapters come last: a router's own UPnP switch outranks the
    # IGD inference, and its public-IP sensor the gateway's.
    section = merge_adapter_results(
        [
            ha_native_adapter(inputs, entities, context),
            *radio_adapters(radio_inputs, entities, context.network_inventory),
            upnp_igd_adapter(upnp_inputs, entities, context),
            *router_adapters(
                router_inputs,
                entities,
                context,
                eero_runtime=eero_runtime_adapter(eero_inputs, router_inputs, context),
            ),
        ]
    )
    attach_inventory(section, context)
    derive_guest_idle(section, context, now)
    return section

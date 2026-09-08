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
and discovered-but-unconfigured devices. Router, DNS, and radio adapters land
in later phases and plug into the same merge.

Runtime reads (auth store, Supervisor add-on info, HTTP server settings, the
automation entity component) are wrapped individually: a read that fails on
some Home Assistant version degrades to a missing capability with one log
line, never to a failed snapshot build.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from homeassistant.util import dt as dt_util

from custom_components.home_generative_agent.agent.helpers import (
    matches_critical_rule,
    resolve_critical_action_policy,
)
from custom_components.home_generative_agent.sentinel.auth_inventory import (
    TOKEN_TYPE_LONG_LIVED,
    TOKEN_TYPE_SYSTEM,
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
    from custom_components.home_generative_agent.sentinel.pseudonymizer import (
        Pseudonymizer,
    )

    from .schema import NetworkClient, NetworkSnapshot, SnapshotEntity

LOGGER = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Capability paths
# ---------------------------------------------------------------------------

CAP_CLIENTS = "network.clients"
_CAP_POSTURE_PREFIX = "network.posture."
_CAP_HA_PREFIX = "network.ha_security."


def posture_cap(key: str) -> str:
    """Return the capability path for a ``posture`` field."""
    return f"{_CAP_POSTURE_PREFIX}{key}"


def ha_cap(key: str) -> str:
    """Return the capability path for an ``ha_security`` field."""
    return f"{_CAP_HA_PREFIX}{key}"


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

# Cover device classes / name hints that make a cover an entry point.
_ENTRY_COVER_CLASSES: frozenset[str] = frozenset({"door", "garage", "gate"})
_ENTRY_COVER_HINTS: tuple[str, ...] = ("door", "garage", "gate")

# Config-flow sources that mean "Home Assistant found this on the LAN".
DISCOVERY_SOURCES: frozenset[str] = frozenset({"ssdp", "zeroconf", "dhcp", "homekit"})

# Assistants whose exposure lists are audited (mirrors the core constant).
KNOWN_ASSISTANTS: tuple[str, ...] = (
    "cloud.alexa",
    "cloud.google_assistant",
    "conversation",
)

_LOGIN_NOTIFICATION_ID = "http-login"

# Once-per-process log guard for runtime reads that fail: the snapshot builds
# every few minutes and a version drift must not fill the log.
_LOGGED_INPUT_FAILURES: set[str] = set()


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
    """

    enabled: bool = True
    options: Mapping[str, Any] | None = None
    pseudonymizer: Pseudonymizer | None = None
    auth_inventory: AuthInventory | None = None
    auth_observation: list[ObservedUser] | None = None


@dataclass
class AdapterResult:
    """Partial network section produced by one adapter."""

    name: str
    posture: dict[str, Any] = field(default_factory=dict)
    ha_security: dict[str, Any] = field(default_factory=dict)
    # None means the adapter cannot see clients at all (capability absent);
    # an empty list means it can and there are none.
    clients: list[NetworkClient] | None = None
    counters: dict[str, float] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)


def merge_adapter_results(results: Iterable[AdapterResult]) -> NetworkSnapshot:
    """
    Merge adapter outputs into one section and derive the capability list.

    Later adapters override earlier ones for the same key, so callers order
    generic adapters before router-specific ones.
    """
    posture: dict[str, Any] = {}
    ha_security: dict[str, Any] = {}
    clients: list[NetworkClient] | None = None
    counters: dict[str, float] = {}
    sources: dict[str, str] = {}
    notes: list[str] = []
    for result in results:
        for key, value in result.posture.items():
            posture[key] = value
            sources[posture_cap(key)] = result.name
        for key, value in result.ha_security.items():
            ha_security[key] = value
            sources[ha_cap(key)] = result.name
        if result.clients is not None:
            clients = [*(clients or []), *result.clients]
            sources[CAP_CLIENTS] = result.name
        counters.update(result.counters)
        notes.extend(result.notes)
    return {
        "capabilities": sorted(sources),
        "sources": sources,
        "clients": clients or [],
        "posture": posture,  # type: ignore[typeddict-item]
        "ha_security": ha_security,  # type: ignore[typeddict-item]
        "counters": counters,
        "notes": notes,
    }


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


@dataclass(frozen=True)
class AutomationInput:
    """An automation entity and its raw configuration."""

    entity_id: str
    raw_config: Mapping[str, Any]


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
    discovered_unconfigured: list[dict[str, str]] | None = None
    discovered_ignored: list[dict[str, str]] | None = None
    # Device-registry facts for tagging updates: entity -> device, device ->
    # set of entity domains it owns.
    entity_device: dict[str, str] = field(default_factory=dict)
    device_domains: dict[str, set[str]] = field(default_factory=dict)


def _log_input_failure(name: str, err: Exception) -> None:
    if name in _LOGGED_INPUT_FAILURES:
        return
    _LOGGED_INPUT_FAILURES.add(name)
    LOGGER.warning(
        "Network audit: could not read %s (%s: %s); the related checks are "
        "reported as missing capabilities.",
        name,
        type(err).__name__,
        err,
    )


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
                        client_name=getattr(token, "client_name", None),
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
                    name=user.name,
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
    return _LOGIN_NOTIFICATION_ID in notifications


def _collect_exposed(hass: HomeAssistant) -> dict[str, list[str]] | None:
    try:
        from homeassistant.components.homeassistant.const import (  # noqa: PLC0415
            DATA_EXPOSED_ENTITIES,
        )
        from homeassistant.components.homeassistant.exposed_entities import (  # noqa: PLC0415
            async_get_assistant_settings,
        )

        if DATA_EXPOSED_ENTITIES not in hass.data:
            # The exposed-entities registry is created by the homeassistant
            # component at startup; absent means we cannot audit exposure.
            return None
        exposed: dict[str, list[str]] = {}
        for assistant in KNOWN_ASSISTANTS:
            settings = async_get_assistant_settings(hass, assistant)
            exposed[assistant] = sorted(
                entity_id
                for entity_id, cfg in settings.items()
                if bool(cfg.get("should_expose"))
            )
        return exposed  # noqa: TRY300
    except Exception as err:  # noqa: BLE001
        _log_input_failure("exposed entities", err)
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
        return HttpInput(
            trusted_proxies_configured=bool(getattr(server, "trusted_proxies", None)),
            ip_ban_enabled=KEY_BAN_MANAGER in app and threshold >= 1,
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
    try:
        from homeassistant.helpers.hassio import is_hassio  # noqa: PLC0415

        if not is_hassio(hass):
            return None
        from homeassistant.components.hassio import (  # noqa: PLC0415
            HassioNotReadyError,
            get_addons_info,
        )
    except Exception as err:  # noqa: BLE001
        _log_input_failure("Supervisor add-on info", err)
        return None
    try:
        info = get_addons_info(hass)
    except HassioNotReadyError:
        # Normal during the first cycles after boot: the Supervisor
        # coordinator has not fetched add-on info yet. Not worth a warning.
        return None
    except Exception as err:  # noqa: BLE001
        _log_input_failure("Supervisor add-on info", err)
        return None
    addons: list[AddonInput] = []
    for slug, data in (info or {}).items():
        if not isinstance(data, dict):
            continue
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
                slug=str(slug),
                name=str(data.get("name") or slug),
                host_ports=tuple(ports),
                protected=bool(data.get("protected", True)),
                running=str(data.get("state") or "") == "started",
            )
        )
    return addons


def _collect_automations(hass: HomeAssistant) -> list[AutomationInput] | None:
    if "automation" not in hass.config.components:
        return None
    try:
        component = hass.data.get("automation")
        if component is None:
            return None
        automations: list[AutomationInput] = []
        for entity in component.entities:
            raw = getattr(entity, "raw_config", None)
            if isinstance(raw, dict):
                automations.append(AutomationInput(entity.entity_id, raw))
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
                    "title": str(placeholders.get("name") or ""),
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
                {"handler": entry.domain, "source": "ignore", "title": entry.title}
            )
        ignored.sort(key=lambda d: (d["handler"], d["title"]))
    except Exception as err:  # noqa: BLE001
        _log_input_failure("ignored config entries", err)
        ignored = None
    return unconfigured, ignored


async def async_collect_ha_native_inputs(
    hass: HomeAssistant,
    context: NetworkBuildContext,
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
    return HaNativeInputs(
        now=now,
        auth=auth,
        failed_login_notification_present=_collect_failed_login(hass),
        exposed=_collect_exposed(hass),
        cloud_remote_ui_enabled=_collect_cloud(hass),
        http=_collect_http(hass),
        trusted_networks_bypass_login=_collect_auth_providers(hass),
        addons=_collect_addons(hass),
        automations=_collect_automations(hass),
        discovered_unconfigured=unconfigured,
        discovered_ignored=ignored,
        entity_device=dict(entity_device),
        device_domains={k: set(v) for k, v in device_domains.items()},
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
    return any(hint in haystack for hint in _ENTRY_COVER_HINTS)


def _walk_actions(node: Any) -> Iterable[dict[str, Any]]:
    """Yield every mapping in an automation's action tree (any nesting)."""
    if isinstance(node, dict):
        yield node
        for value in node.values():
            yield from _walk_actions(value)
    elif isinstance(node, list):
        for item in node:
            yield from _walk_actions(item)


def _action_entity_ids(call: Mapping[str, Any]) -> tuple[list[str], bool]:
    """Return (entity ids, unresolved-target) for one service-call mapping."""
    ids: list[str] = []
    unresolved = False
    for container in (call.get("target"), call.get("data"), call):
        if not isinstance(container, dict):
            continue
        raw = container.get("entity_id")
        if isinstance(raw, str):
            ids.append(raw)
        elif isinstance(raw, list):
            ids.extend(str(v) for v in raw)
        if any(
            k in container for k in ("area_id", "device_id", "label_id", "floor_id")
        ):
            unresolved = True
    if not ids:
        unresolved = True
    return ids, unresolved


def automation_calls_critical_action(
    raw_config: Mapping[str, Any], critical_actions: Sequence[Mapping[str, str]]
) -> bool:
    """Return True when any action in *raw_config* matches a critical rule."""
    actions = raw_config.get("actions", raw_config.get("action"))
    for call in _walk_actions(actions):
        service = call.get("action") or call.get("service")
        if not isinstance(service, str) or "." not in service:
            continue
        domain, _, name = service.partition(".")
        entity_ids, unresolved = _action_entity_ids(call)
        if matches_critical_rule(
            domain=domain,
            service=name,
            entity_ids=entity_ids,
            critical_actions=critical_actions,
            unresolved_target=unresolved,
        ):
            return True
    return False


def automation_has_public_webhook(raw_config: Mapping[str, Any]) -> bool:
    """
    Return True when a webhook trigger is reachable beyond the local network.

    Home Assistant defaults ``local_only`` to True, so only an explicit False
    counts.
    """
    triggers = raw_config.get("triggers", raw_config.get("trigger"))
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
    inputs: HaNativeInputs, auth_inventory: AuthInventory | None
) -> dict[str, Any]:
    auth = inputs.auth
    if auth is None:
        return {}
    labels = token_labels(auth)
    admins = 0
    long_lived = 0
    unused_days: dict[str, int] = {}
    for user in auth:
        if user.is_admin and user.is_active and not user.system_generated:
            admins += 1
        for token in user.tokens:
            if token.token_type == TOKEN_TYPE_SYSTEM:
                continue
            if token.token_type != TOKEN_TYPE_LONG_LIVED:
                continue
            long_lived += 1
            days = _days_between(token.last_used_at or token.created_at, inputs.now)
            if days is not None:
                unused_days[labels[token.token_id]] = days
    fields: dict[str, Any] = {
        "admin_user_count": admins,
        "long_lived_token_count": long_lived,
        "long_lived_tokens_unused_days": unused_days,
    }
    if auth_inventory is not None:
        delta = auth_inventory.diff(auth)
        if not delta.bootstrap:
            fields["new_admin_users"] = list(delta.new_admin_users)
            fields["new_long_lived_tokens"] = list(delta.new_long_lived_tokens)
            fields["refresh_tokens_from_new_ip"] = list(delta.tokens_from_new_ip)
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


def ha_native_adapter(  # noqa: PLR0912 - one branch per input, kept flat on purpose
    inputs: HaNativeInputs,
    entities: Sequence[SnapshotEntity],
    context: NetworkBuildContext,
) -> AdapterResult:
    """Pure adapter over HA-native inputs and the entity snapshot."""
    result = AdapterResult(name="ha_native")
    ha = result.ha_security

    ha.update(_auth_fields(inputs, context.auth_inventory))

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
        # Home Assistant validates use_x_forwarded_for and trusted_proxies as
        # an inclusive pair, so one implies the other; the server object only
        # exposes the proxy list.
        ha["http_use_x_forwarded_for"] = inputs.http.trusted_proxies_configured
        ha["http_trusted_proxies_configured"] = inputs.http.trusted_proxies_configured
        ha["http_ip_ban_enabled"] = inputs.http.ip_ban_enabled
        ha["http_login_attempts_threshold"] = inputs.http.login_attempts_threshold

    if inputs.trusted_networks_bypass_login is not None:
        ha["trusted_networks_bypass_login"] = inputs.trusted_networks_bypass_login

    if inputs.addons is not None:
        ha["addons_with_host_ports"] = {
            addon.slug: list(addon.host_ports)
            for addon in inputs.addons
            if addon.running and addon.host_ports
        }
        ha["addons_unprotected"] = sorted(
            addon.slug
            for addon in inputs.addons
            if addon.running and not addon.protected
        )
        ha["addon_names"] = {addon.slug: addon.name for addon in inputs.addons}
    else:
        result.notes.append(
            "Supervisor add-on data is unavailable on this install type; "
            "add-on exposure is not audited."
        )

    if inputs.automations is not None:
        critical_actions = (
            resolve_critical_action_policy(context.options).critical_actions
            if context.options is not None
            else resolve_critical_action_policy({}).critical_actions
        )
        public = sorted(
            a.entity_id
            for a in inputs.automations
            if automation_has_public_webhook(a.raw_config)
        )
        ha["webhook_automations_public"] = public
        public_set = set(public)
        ha["webhook_automations_critical"] = sorted(
            a.entity_id
            for a in inputs.automations
            if a.entity_id in public_set
            and automation_calls_critical_action(a.raw_config, critical_actions)
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
    """Build the ``network`` section; never raises for a failed input read."""
    ctx = context or NetworkBuildContext()
    if not ctx.enabled:
        return empty_network_snapshot("Network audit is disabled in Sentinel options.")
    inputs = await async_collect_ha_native_inputs(
        hass,
        ctx,
        now=now,
        entity_device=entity_device,
        device_domains=device_domains,
    )
    return merge_adapter_results([ha_native_adapter(inputs, entities, ctx)])

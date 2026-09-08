"""Snapshot schema for authoritative home state."""

from __future__ import annotations

from typing import Any, NotRequired, TypedDict, cast

import voluptuous as vol

# Version 2 adds the optional per-entity ``platform`` field and the optional
# top-level ``network`` section. Both are NotRequired so that snapshots
# persisted under version 1 (audit records) and older fixtures stay valid.
SNAPSHOT_SCHEMA_VERSION = 2


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
    # Entity-registry platform (integration domain) that owns the entity, e.g.
    # "fritz" or "eero". Network rules match on this fact instead of name
    # heuristics. None for entities with no registry entry.
    platform: NotRequired[str | None]


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
    # Timestamp of the face-recognition sighting itself (the image entity's
    # last_event, stamped by the same signal that carries recognized_people).
    # Distinct from last_activity, which camera attributes can refresh on any
    # motion long after the recognition labels went stale.
    recognition_last_event: NotRequired[str | None]


class DerivedContext(TypedDict):
    """Derived context for snapshot consumers."""

    now: str
    timezone: str
    is_night: bool
    anyone_home: bool
    people_home: list[str]
    people_away: list[str]
    last_motion_by_area: dict[str, str]
    baseline_ready_entities: NotRequired[list[str]]


# ---------------------------------------------------------------------------
# Network section (docs/network-security-plan.md)
# ---------------------------------------------------------------------------
#
# Every field below is optional at the data level: an adapter asserts only what
# the installed integrations can actually observe, and the ``capabilities``
# list names the dotted paths that are present. Rules declare what they need
# and are skipped, visibly, when a capability is missing.


class NetworkClient(TypedDict):
    """A client seen on the IP network by a router-class integration."""

    key: str  # stable pseudonymized id (HMAC of the MAC), see snapshot/network.py
    connected: bool
    mac: NotRequired[str | None]  # raw; stripped before any LLM call
    ip: NotRequired[str | None]
    hostname: NotRequired[str | None]
    manufacturer: NotRequired[str | None]
    connection_type: NotRequired[str | None]  # "wired" | "wireless" | None
    network_name: NotRequired[str | None]
    last_seen: NotRequired[str | None]
    ha_device_id: NotRequired[str | None]
    ha_integration: NotRequired[str | None]
    tracker_entity_id: NotRequired[str | None]
    signal: NotRequired[float | None]
    usage_day_bytes: NotRequired[int | None]
    blocked_day: NotRequired[int | None]
    is_guest: NotRequired[bool | None]
    vlan: NotRequired[int | None]


class NetworkPosture(TypedDict, total=False):
    """Router / DNS posture facts. Every key carries a ``<key>_entity_id`` twin."""

    upnp_enabled: bool
    wpa3_enabled: bool
    guest_network_enabled: bool
    guest_client_count: int
    ipv6_enabled: bool
    ddns_enabled: bool
    malware_blocking_enabled: bool
    ad_blocking_enabled: bool
    remote_management_enabled: bool
    router_update_pending: bool
    router_update_entities: list[str]
    public_ip_changed: bool
    port_forwards: list[dict[str, Any]]
    wlan_enabled: dict[str, bool]


class HaSecurityPosture(TypedDict, total=False):
    """Home Assistant's own attack surface, observed from HA-native sources."""

    admin_user_count: int
    long_lived_token_count: int
    long_lived_token_age_days: dict[str, int]  # token label -> days since created
    new_admin_users: list[str]  # names, vs. the persistent auth inventory
    new_long_lived_tokens: list[str]  # token labels, vs. the auth inventory
    failed_login_notification_present: bool
    exposed_sensitive_entities: dict[str, list[str]]  # assistant -> entity_ids
    critical_action_pin_enabled: bool
    cloud_remote_ui_enabled: bool
    pending_updates: list[str]  # update.* entity_ids in state "on"
    pending_security_updates: list[str]  # subset whose device is security-class
    http_use_x_forwarded_for: bool
    http_trusted_proxies_configured: bool
    http_ip_ban_enabled: bool
    http_login_attempts_threshold: int
    trusted_networks_bypass_login: bool
    addons_with_host_ports: dict[str, list[int]]  # slug -> host ports
    addons_unprotected: list[str]  # slugs with protection mode off
    addon_names: dict[str, str]  # slug -> display name
    webhook_automations_public: list[str]  # automation entity_ids
    webhook_automations_critical: list[str]  # subset that call critical actions
    unavailable_security_devices: dict[str, int]  # entity_id -> minutes
    discovered_unconfigured: list[dict[str, str]]  # {handler, source, title}
    discovered_ignored: list[dict[str, str]]


class RadioDevice(TypedDict):
    """A device on a non-IP radio (Zigbee, Z-Wave, Bluetooth, Matter, Thread)."""

    device_id: str
    protocol: str
    platform: str
    name: str | None
    is_security_device: bool
    manufacturer: NotRequired[str | None]
    model: NotRequired[str | None]
    first_seen: NotRequired[str | None]
    security_class: NotRequired[str | None]
    fabrics: NotRequired[list[dict[str, Any]]]


class RadioPosture(TypedDict, total=False):
    """Radio-protocol configuration facts."""

    zigbee_permit_join: bool
    zigbee_permit_join_entity_id: str
    zwave_inclusion_active: bool
    coordinator_update_pending: list[str]
    bluetooth_unknown_trackers: list[dict[str, Any]]
    thread_border_router_count: int


class RadioSnapshot(TypedDict):
    """Radio section of the network snapshot."""

    capabilities: list[str]
    devices: list[RadioDevice]
    posture: RadioPosture


class NetworkSnapshot(TypedDict):
    """Normalized network section built by the adapters in snapshot/network.py."""

    capabilities: list[str]  # dotted paths present, e.g. "network.posture.upnp_enabled"
    sources: dict[str, str]  # capability -> adapter that provided it
    clients: list[NetworkClient]
    posture: NetworkPosture
    ha_security: HaSecurityPosture
    counters: dict[str, float]
    radio: NotRequired[RadioSnapshot]
    # Human-readable provenance / privacy statements ("network audit disabled",
    # "Supervisor add-on data unavailable on this install type").
    notes: NotRequired[list[str]]


class FullStateSnapshot(TypedDict):
    """Full structured snapshot of home state."""

    schema_version: int
    generated_at: str
    entities: list[SnapshotEntity]
    camera_activity: list[CameraActivity]
    derived: DerivedContext
    # Always populated by the builder; NotRequired only protects snapshots
    # persisted before schema version 2.
    network: NotRequired[NetworkSnapshot]


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
                vol.Optional("platform"): vol.Any(str, None),
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
                vol.Optional("recognition_last_event"): vol.Any(str, None),
            }
        ],
        vol.Required("derived"): {
            vol.Required("now"): str,
            vol.Required("timezone"): str,
            vol.Required("is_night"): bool,
            vol.Required("anyone_home"): bool,
            vol.Required("people_home"): [str],
            vol.Required("people_away"): [str],
            vol.Required("last_motion_by_area"): dict,
        },
        vol.Optional("network"): {
            vol.Required("capabilities"): [str],
            vol.Required("sources"): dict,
            vol.Required("clients"): list,
            vol.Required("posture"): dict,
            vol.Required("ha_security"): dict,
            vol.Required("counters"): dict,
            vol.Optional("radio"): dict,
            vol.Optional("notes"): [str],
        },
    }
)


def validate_snapshot(snapshot: dict[str, Any]) -> FullStateSnapshot:
    """Validate and return a snapshot using the canonical schema."""
    validated = SNAPSHOT_SCHEMA(snapshot)
    return cast("FullStateSnapshot", validated)

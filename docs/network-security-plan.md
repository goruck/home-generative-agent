# Network Security & Privacy Audit Plan

Status: phase 1 in progress. Implementation-order steps 1–5 (the HA-only MVP: `platform` field, `network` snapshot section with the `ha_native` adapter, auth inventory, engine capability gating with health-sensor reporting, and the HA-native rules) are implemented; steps 6–11 (radio and router adapters, redaction, the `network_audit` feature type, the `audit_home_security` tool, baseline counters, discovery templates) are pending field validation of the MVP on a plain install. Two facts learned while implementing narrow the plan: Home Assistant records refresh-token use only on the access-token exchange, which a long-lived token performs once at creation, so `long_lived_tokens_unused_days` became `long_lived_token_age_days` and `refresh_tokens_from_new_ip` is collected into the inventory but not alerted on; and the per-type cooldown means every rule in this family emits one aggregated finding per cycle rather than one per item.

## Problem Statement

Home Assistant knows a great deal about a home's network, but that knowledge is
scattered across the device registry, the auth system, the exposed-entities
registry, `update` entities, and whichever router integration the user happens
to run. Add the Zigbee, Z-Wave, Bluetooth, and Matter integrations, which know the
security configuration of most locks and sensors. Nothing in HA or HGA reads
across those sources to answer "is my home network configured safely, and has
anything changed that I should know about?"

HGA is well placed to do this. It already builds an authoritative snapshot of
home state, evaluates deterministic Sentinel rules over it, learns per-home
baselines, and lets an LLM explain findings in plain language. This plan adds a
network dimension to each of those layers plus an on-demand agent tool, so the
agent can spot network security and privacy issues automatically or when asked,
and in a later phase fix them under the existing PIN gate.

The central design constraint is heterogeneity: every user has a different
network, a different router, and a different set of integrations. The feature
must never assume a topology. It models *capabilities* and degrades to whatever
the installed integrations can see.

## Goals

- Detect network security posture problems (UPnP on, WPA3 off, guest network
  left on, missing router firmware) deterministically from HA state.
- Detect network *changes* worth telling the user about (an unknown device
  joined, a security device went offline, a usage or threat spike).
- Audit HA's own attack surface (admin users, long-lived tokens, locks and
  alarms exposed to voice assistants, cloud remote access, pending updates).
- Audit radio-protocol configuration HA can observe: Z-Wave security
  classes, Zigbee permit-join, new radio devices, Bluetooth trackers,
  Matter fabrics.
- Explain findings and per-device privacy exposure in plain language via the
  LLM, using pseudonymized identifiers and an optional local-model override.
- Work on every install: a home with no router integration still gets a
  complete HA hardening audit (see HA-Only MVP); a home with Fritz!Box,
  UniFi, eero, or another supported router gets progressively more.
- Reuse existing Sentinel machinery (rules, baseline, discovery, notifier,
  audit trail, triage) rather than build a parallel pipeline.

## Non-Goals

- **No active scanning.** No nmap, ARP sweeps, or port probes from the HA host.
  Everything is derived from what HA already ingests. This is a hard scope
  decision, not a phasing one. A user who already runs the `nmap_tracker`
  integration is a data source like any other; HGA never initiates scans.
- **No fixes in phase 1.** The first implementation is read-only: it audits,
  notifies, and recommends. PIN-gated remediation is phase 2.
- No deep packet inspection or flow analysis; HGA never sees traffic.
- No detection of active radio-layer attacks: Zigbee key sniffing during
  join, Z-Wave S0 downgrade, Bluetooth pairing exploits, jamming, or
  replay. Those need RF monitoring HA does not do. Radio *configuration*
  weaknesses that HA can observe are in scope (see Radio Protocols).
- No new router integrations. HGA consumes entities from integrations the user
  already installed; it does not talk to router APIs itself.
- No change to the Sentinel safety invariant: the LLM never gates detection,
  never mutates findings, and never executes actions.

## Design Principles

1. **Model capabilities, not networks.** A normalized `network` snapshot
   section with every field optional, populated by small per-integration
   adapters, and a `capabilities` list saying what is present.
2. **Rules declare what they need.** A rule that needs `network.posture.upnp`
   is skipped, visibly, on a home that cannot provide it. No false positives
   from missing data.
3. **Learn the home rather than assume it.** Known-device inventory, baseline
   counters, and discovery proposals personalize the feature the same way
   they already personalize power and camera rules.
4. **Deterministic at runtime.** Adapters are pure functions over HA state.
   The LLM reasons over normalized, pseudonymized data and writes narrative;
   it never parses raw router entities in the detection path.
5. **Privacy of the audit itself.** MACs, IPs, and hostnames are tokenized
   before any model sees them, and the audit can be pinned to a local model.

## Architecture

```
HA state / registries / auth
        │
        ▼
snapshot/builder.py ──► snapshot/network.py (adapters) ──► snapshot["network"]
        │                                                        │
        │              ┌─────────────────────────────────────────┤
        ▼              ▼                     ▼                   ▼
 sentinel/rules/   sentinel/baseline.py   sentinel/discovery   agent/tools.py
 network_*.py      (network counters)     (network templates)  audit_home_security
        │              │                     │                   │
        └──────────────┴─────────┬───────────┘                   │
                                 ▼                               ▼
                     sentinel/engine.py ──► notifier ──► audit store   LLM narrative
                                 (capability gating)                (pseudonymized)
```

The only new runtime concept is the adapter layer in the snapshot. Everything
downstream consumes the normalized section through the same interfaces the
existing rules, baseline, discovery, and tools already use.

## Snapshot Changes

### 1. Add `platform` to `SnapshotEntity`

Rules today match entities by domain and name heuristics (see
`EXTERIOR_HINTS` in `sentinel/rules/unlocked_lock_at_night.py`). Network
rules must not do that: whether an entity is the eero UPnP switch is a fact
the entity registry already knows. Add:

```python
class SnapshotEntity(TypedDict):
    ...
    platform: NotRequired[str | None]  # entity-registry platform, e.g. "eero"
```

`snapshot/builder.py` fills it from `entity_registry.async_get(entity_id)`.
It is `NotRequired` so stored snapshots and existing tests remain valid. The
`SNAPSHOT_SCHEMA` gains `vol.Optional("platform")`. Bump
`SNAPSHOT_SCHEMA_VERSION` to 2; consumers that read persisted snapshots
(audit records) must tolerate the missing key.

### 2. New `network` section

```python
class NetworkClient(TypedDict):
    key: str  # stable id: pseudonymized MAC (see Privacy)
    mac: NotRequired[str | None]  # raw, stripped before any LLM call
    ip: NotRequired[str | None]
    hostname: NotRequired[str | None]
    manufacturer: NotRequired[str | None]
    connection_type: NotRequired[str | None]  # "wired" | "wireless" | None
    network_name: NotRequired[str | None]  # SSID / VLAN / "guest" when known
    connected: bool
    last_seen: NotRequired[str | None]
    ha_device_id: NotRequired[str | None]  # device-registry join, if any
    ha_integration: NotRequired[str | None]  # e.g. "shelly" when joined
    tracker_entity_id: NotRequired[str | None]
    signal: NotRequired[float | None]
    usage_day_bytes: NotRequired[int | None]
    blocked_day: NotRequired[int | None]
    is_guest: NotRequired[bool | None]  # UniFi, Fritz guest SSID
    vlan: NotRequired[int | None]  # UniFi


class NetworkPosture(TypedDict, total=False):
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
    port_forwards: list[
        dict[str, Any]
    ]  # {entity_id, external_port, protocol, description}
    wlan_enabled: dict[str, bool]  # WLAN name -> enabled
    # every key carries a companion "<key>_entity_id" for evidence/fixes


class HaSecurityPosture(TypedDict, total=False):
    admin_user_count: int
    long_lived_token_count: int
    long_lived_tokens_unused_days: dict[str, int]  # client_name -> days
    refresh_tokens_from_new_ip: list[str]  # client names
    failed_login_notification_present: bool
    exposed_sensitive_entities: dict[str, list[str]]  # assistant -> entity_ids
    critical_action_pin_enabled: bool
    cloud_remote_ui_enabled: bool
    pending_updates: list[str]  # update.* entity_ids
    http_use_x_forwarded_for: bool
    http_trusted_proxies_configured: bool
    http_ip_ban_enabled: bool
    trusted_networks_bypass_login: bool
    addons_with_host_ports: dict[str, list[int]]  # slug -> host ports
    addons_unprotected: list[str]  # slugs with protection mode off
    webhook_automations_public: list[str]  # automation entity_ids
    unavailable_security_devices: dict[str, int]  # entity_id -> minutes
    discovered_unconfigured: list[dict[str, str]]  # {handler, source, title}
    discovered_ignored: list[dict[str, str]]


class RadioDevice(TypedDict):
    device_id: str  # device-registry id
    protocol: str  # "zigbee" | "zwave" | "bluetooth" | "matter" | "thread"
    platform: str  # "zha" | "mqtt" (Zigbee2MQTT) | "zwave_js" | "bluetooth" | "matter"
    name: str | None
    manufacturer: NotRequired[str | None]
    model: NotRequired[str | None]
    is_security_device: bool  # lock, alarm, garage, camera
    first_seen: NotRequired[str | None]
    security_class: NotRequired[
        str | None
    ]  # zwave: "none" | "s0" | "s2_unauth" | "s2_auth" | "s2_access"
    fabrics: NotRequired[list[dict[str, Any]]]  # matter: {vendor_id, fabric_id, label}


class RadioPosture(TypedDict, total=False):
    zigbee_permit_join: bool
    zigbee_permit_join_entity_id: str
    zwave_inclusion_active: bool
    coordinator_update_pending: list[str]  # update.* for coordinators and proxies
    bluetooth_unknown_trackers: list[
        dict[str, Any]
    ]  # {key, first_seen, days_present, kind}
    thread_border_router_count: int


class RadioSnapshot(TypedDict):
    capabilities: list[str]
    devices: list[RadioDevice]
    posture: RadioPosture


class NetworkSnapshot(TypedDict):
    capabilities: list[str]  # dotted paths present, e.g. "network.posture.upnp_enabled"
    sources: dict[str, str]  # capability -> platform that provided it
    clients: list[NetworkClient]
    posture: NetworkPosture
    ha_security: HaSecurityPosture
    radio: NotRequired[RadioSnapshot]
    counters: dict[str, float]  # baseline inputs, see Baseline section
```

`FullStateSnapshot` gains `network: NotRequired[NetworkSnapshot]`. The
builder always populates it; `NotRequired` only protects older persisted
snapshots.

### 3. Adapters (`snapshot/network.py`)

An adapter is a pure function `(hass, entities, registries) -> partial
NetworkSnapshot`. The builder runs every adapter, merges results, and records
which adapter provided each capability. Order matters only for conflicts, where
a router-specific adapter wins over the generic one.

#### Adapter priority

Priority follows Home Assistant's public analytics, which are opt-in and
reported by roughly two-thirds of the installed base. Router-class
integrations, most popular first:

| Integration | Share of reporting installs | Notes |
|---|---|---|
| UPnP/IGD (`upnp`) | 32.8% | auto-discovered; its presence proves UPnP is on |
| FRITZ!Box Tools (`fritz`) | 8.8% | local API; richest consumer-router integration |
| UniFi Network (`unifi`) | 6.8% | local API; richest prosumer integration |
| AdGuard Home (`adguard`) | 4.3% | DNS protection switches and counters |
| Nmap Tracker (`nmap_tracker`) | 1.5% | client presence only, user-configured scanning |
| Pi-hole (`pi_hole`) | 1.3% | DNS counters, protection switch |
| ASUSWRT (`asuswrt`) | ~5,000 installs | clients, bandwidth |
| TP-Link Omada (`tplink_omada`) | ~3,300 installs | clients, PoE, WLAN |
| Keenetic NDMS2 (`keenetic_ndms2`) | ~3,200 installs | clients |
| MikroTik (`mikrotik`) | ~2,900 installs | clients |
| NETGEAR (`netgear`) | ~2,900 installs | clients, allow/block switches |
| eero (HACS, `schmittx/home-assistant-eero`) | not in core analytics | cloud-only, unofficial API |

Sources: the "used by N active installations" line on each integration's
page at home-assistant.io (September 2026). eero is a community integration
and does not appear in the core integration analytics; it is well below the
top two.

Phase 1 therefore ships `ha_native`, `generic_router_tracker`, `upnp_igd`,
`fritz`, `unifi`, and `adguard`. `eero` also ships in phase 1, not on
popularity but because the maintainer runs it and it is the only adapter that
can be validated on real hardware during development; it doubles as the
reference for cloud-only mesh integrations. The remaining routers are phase 3.

| Adapter | Selected when | Provides | Phase |
|---|---|---|---|
| `ha_native` | always | `ha_security.*`, `posture.pending_updates`, device-registry MAC index | 1 |
| `generic_router_tracker` | any `device_tracker` with `source_type: router` | `clients[]` with mac, ip, hostname when the tracker exposes them | 1 |
| `upnp_igd` | a loaded `upnp` config entry, or an in-progress `upnp` discovery flow | `posture.upnp_enabled = True`, `posture.public_ip_changed` from the external-IP sensor, WAN status | 1 |
| `fritz` | `platform == "fritz"` | clients, guest Wi-Fi, port forwards to the HA host, per-device internet access, firmware update | 1 |
| `unifi` | `platform == "unifi"` | clients with VLAN, guest flag, wired/wireless; block, port-forward, WLAN, firewall, traffic-rule switches; device updates | 1 |
| `adguard` | `platform == "adguard"` | protection, filtering, safe-browsing, parental switches; blocked-query counters | 1 |
| `eero` | `platform == "eero"` | everything in the eero table below | 1 |
| `zwave` | `platform == "zwave_js"` | per-node security class, inclusion state, controller update | 1 |
| `zigbee` | `platform == "zha"` or Zigbee2MQTT bridge entities via `mqtt` | permit-join state, coordinator update, device list | 1 |
| `radio_registry` | always | new radio devices from the device registry, per protocol | 1 |
| `bluetooth` | the `bluetooth` integration is loaded | persistent unknown tracker-class advertisements | 2 |
| `matter` | `platform == "matter"` | per-node fabric list | 3 |
| `pi_hole` | `platform == "pi_hole"` | protection switch, blocked-query counters | 3 |
| `asuswrt` / `tplink_omada` / `keenetic_ndms2` / `mikrotik` / `netgear` | by platform | clients; per-integration posture where exposed | 3 |

Adapters come in two tiers, and the tier is declared on the adapter:

- **Entity adapters** read entity state, attributes, and the entity and
  device registries only. They never import integration code, so they work
  whether the integration is core or HACS and are tested with plain
  `hass.states` fixtures. Every router and DNS adapter above is an entity
  adapter.
- **Runtime adapters** read an integration's runtime objects through that
  integration's own helper API, because the data is not on any entity.
  Z-Wave `highest_security_class` and controller inclusion state live on
  Z-Wave JS node and controller objects; ZHA permit-join state lives on the
  ZHA gateway. `zwave`, `zigbee` (ZHA half), `bluetooth`, and `matter` are
  runtime adapters. They are permitted under four rules: the import happens
  lazily inside the adapter function, never at module load; every read is
  wrapped so `ImportError`, `AttributeError`, `KeyError`, and `TypeError`
  degrade to a missing capability with one rate-limited log line, never an
  engine failure; each runtime adapter has a test that exercises the real
  integration objects from the pinned `homeassistant` version in
  `requirements/test.txt`, so an HA upgrade that moves the attribute fails a
  test instead of silently dropping the capability; and the adapter's
  docstring names the HA version and the exact objects it reads. A runtime
  adapter whose data cannot be obtained on the pinned version defers its
  rules rather than approximating from entity state.

#### FRITZ!Box adapter mapping

From the core `fritz` integration source. Trackers carry `mac_address`,
`last_time_reachable`, `connected_to`, `connection_type`, and `ssid`.

| Normalized field | fritz entity | Notes |
|---|---|---|
| `clients[].mac/hostname` | `device_tracker.*` | hostname is the entity name |
| `clients[].connection_type` | tracker attr `connection_type` | |
| `clients[].network_name` | tracker attr `ssid` | guest SSID identifies guest clients |
| `clients[].last_seen` | tracker attr `last_time_reachable` | |
| `posture.guest_network_enabled` | Wi-Fi network switch for the guest network | |
| `posture.port_forwards` | port-forward switches | only forwards whose target is the HA host are exposed |
| `posture.router_update_pending` | `update.*` with `platform == "fritz"` | |
| `posture.public_ip_changed` | external IP / IPv6 sensors | |
| fix: per-device internet access | `switch.*_internet_access` | phase 2 |

#### UniFi adapter mapping

From the core `unifi` integration source. Connected client trackers carry
`ip`, `mac`, `name`, `oui`, `essid`, `vlan`, `is_guest`, `is_wired`,
`authorized`, `ap_mac`, `radio`, and `radio_proto`.

| Normalized field | unifi entity | Notes |
|---|---|---|
| `clients[].mac/ip/hostname` | tracker attrs `mac`, `ip`, `name` | |
| `clients[].manufacturer` | tracker attr `oui` | |
| `clients[].connection_type` | tracker attr `is_wired` | |
| `clients[].network_name` | tracker attrs `essid`, `vlan` | VLAN is the strongest IoT-segmentation signal available |
| `clients[].is_guest` | tracker attr `is_guest` | new optional field |
| `posture.port_forwards` | port-forward switches | |
| `posture.wlan_enabled[]` | WLAN switches | |
| `posture.firewall_policies` | firewall-policy and traffic-rule switches | informational in phase 1 |
| `posture.router_update_pending` | `update.*` with `platform == "unifi"` | |
| `counters.client.<key>.usage` | per-client bandwidth sensors | baseline input |
| fix: block client | block-client switch | phase 2 |
| fix: PoE off | PoE port switch | phase 2, cameras and APs |

#### UPnP/IGD and AdGuard adapters

- `upnp_igd`: a loaded `upnp` config entry means the router answered an
  Internet Gateway Device discovery, which by definition means UPnP is on.
  This gives `posture.upnp_enabled` on roughly a third of installs with no
  router adapter at all. The integration's external-IP and WAN-status sensors
  feed `public_ip_changed` and a `wan_down` counter. An in-progress
  (discovered but not configured) `upnp` flow is treated the same way, since
  the discovery itself is the evidence.
- `adguard`: `switch.adguard_protection`, `switch.adguard_filtering`,
  `switch.adguard_safe_browsing`, and `switch.adguard_parental_control` map to
  `posture.malware_blocking_enabled` and `posture.ad_blocking_enabled`; the
  blocked-query and DNS-queries sensors feed `counters.network.threats_day`.

#### eero adapter mapping

Based on the entity keys in the community integration
(`schmittx/home-assistant-eero`). The integration is cloud-only through an
unofficial API and polls on an interval, so freshness is minutes, not seconds.

| Normalized field | eero entity | Notes |
|---|---|---|
| `clients[].mac/ip/hostname` | `device_tracker.*` attrs `mac`, `ip`, `host_name` | tracker `source_type` is `router` |
| `clients[].manufacturer` | tracker attr `manufacturer` | |
| `clients[].connection_type` | tracker attr `connection_type` | |
| `clients[].network_name` | tracker attr `network_name` | guest vs main only when the integration names them distinctly |
| `clients[].signal` | client `sensor.*_signal` | wireless only |
| `clients[].usage_day_bytes` | client `sensor.*_data_usage_day` | |
| `clients[].blocked_day` | client `sensor.*_blocked_day` | eero Secure |
| `posture.upnp_enabled` | `switch.*_upnp` | |
| `posture.wpa3_enabled` | `switch.*_wpa3` | |
| `posture.guest_network_enabled` | `switch.*_guest_network` | |
| `posture.guest_client_count` | `sensor.*_connected_guest_clients` | |
| `posture.ipv6_enabled` | `switch.*_ipv6` | |
| `posture.ddns_enabled` | `switch.*_dynamic_dns` | eero Plus |
| `posture.malware_blocking_enabled` | `switch.*_advanced_security` | eero Plus |
| `posture.ad_blocking_enabled` | `switch.*_ad_blocking` | eero Plus |
| `posture.router_update_pending` | `update.*` with `platform == "eero"` | |
| `posture.public_ip_changed` | `sensor.*_public_ip` vs previous run | |
| `counters.network.threats_day` | network `sensor.*_blocked_day` | baseline input |
| `counters.network.client_count` | count of connected trackers | baseline input |

Not available from eero and therefore never asserted by the adapter: per-client
guest flag, per-client blocked flag, port forwards, DHCP reservations, band or
SSID, device type. The entity is matched by `platform` plus the integration's
translation key or unique-id suffix, never by friendly name.

#### Auth inventory store

A new `sentinel/auth_inventory.py` store (`Store` under key
`home_generative_agent_sentinel_auth_inventory`) gives the auth change rules
something to compare against, because the existing audit store keeps only a
snapshot hash and HA's auth objects have no history.

- **Contents, all non-secret:** per user `user_id`, `is_admin`, `is_active`,
  `first_seen`; per refresh token `token_id` (HA's own identifier, not the
  token), `user_id`, `token_type`, `client_name`, `created_at`, `first_seen`,
  and `seen_ips`, a bounded list of HMAC-pseudonymized IPs with first and
  last seen timestamps. The raw IP is never stored; the rule only needs to
  know whether the current one was seen before.
- **Bootstrap:** the first run records every existing user and token as
  known without alerting, with one "auth inventory established"
  notification listing admin count and long-lived token count.
- **Retention and deletion:** a token row is deleted when its `token_id`
  no longer exists in HA; `seen_ips` keeps the most recent 20 entries and
  drops entries not seen for `sentinel_auth_ip_retention_days` (default 90).
  Deleting the Sentinel subentry deletes the store. A
  `sentinel_reset_auth_inventory` service clears it on demand.
- **Restart safety:** because the store is persistent, a restart does not
  re-bootstrap and does not re-alert on tokens seen before.

#### `ha_native` adapter details

- **Auth.** `hass.auth.async_get_users()` gives `is_admin`, `is_active`, and
  each user's refresh tokens with `id`, `token_type`, `client_name`,
  `created_at`, `last_used_at`, and `last_used_ip`. These objects describe
  the present only, so the change rules (`ha_new_admin_or_token`,
  `refresh_tokens_from_new_ip`) compare against a persistent auth inventory
  described below. Stale-token detection is computed from `last_used_at`
  directly. No token values, JWT keys, or secrets are ever read into the
  snapshot or the inventory.
- **Failed logins.** Presence of the `http-login` persistent notification.
- **Exposed entities.** Via the `homeassistant.exposed_entities` helpers, for
  each assistant list exposed entities whose domain is `lock`,
  `alarm_control_panel`, or `cover` with a door/gate/garage hint, reusing the
  critical-action matcher in `agent/helpers.py`.
- **Critical-action PIN.** From the config entry options.
- **Cloud remote UI.** From the cloud integration's state when loaded.
- **Pending updates.** Every `update.*` entity in state `on`, tagged as a
  security device when its device is a lock, alarm, camera, or router.
- **Device-registry MAC index.** `connections` entries of type `mac`, used to
  join router clients to HA devices and their integration domain.
- **Security device availability.** Every `lock`, `alarm_control_panel`, and
  `camera` entity whose state is `unavailable`, with minutes since
  `last_changed`. Needs no router; HA already knows when it lost a device.
- **Discovered but unconfigured devices.** In-progress config flows whose
  source is `ssdp`, `zeroconf`, `dhcp`, or `homekit`, plus config entries
  with source `ignore`. This is a partial LAN inventory HA collects on
  every install without any router integration.
- **Supervisor add-ons** (HA OS and Supervised only). Via the `hassio`
  component's add-on info: add-ons with container ports mapped to the host,
  and add-ons running with protection mode off. Absent on Container and
  Core installs, reported as a missing capability.
- **Webhook automations.** Automations whose raw config has a webhook
  trigger with `local_only: false`, read from the automation entities'
  stored config. These are reachable from the internet whenever remote
  access is on.
- **Auth providers.** A `trusted_networks` provider with
  `allow_bypass_login` enabled.
- **HTTP settings.** `use_x_forwarded_for` without trusted proxies, and IP
  banning disabled, from the `http` component's runtime configuration.

## Known-Device Inventory

A new `sentinel/network_inventory.py` store, shaped like `RuleRegistry`
(`Store` under key `home_generative_agent_sentinel_network_inventory`).

- Keyed by a source-qualified id: the pseudonymized MAC for router clients
  and the device-registry id for radio devices. Each row holds `source`
  (`router`, `zigbee`, `zwave`, `bluetooth`, `matter`), `first_seen`,
  `last_seen`, `trusted` (bool), the last observed name and manufacturer,
  and the HA device id when known. Router MACs are one source among
  several, not the inventory's identity.
- **Bootstrap:** on the first run after enablement every currently connected
  client is recorded with `trusted = True` and a single "inventory
  established" notification asks the user to review it. This mirrors the
  baseline establishment notification and avoids a flood of alerts on day one.
- **Auto-trust is narrow.** A device-registry match alone proves nothing:
  the FRITZ!Box and UniFi integrations create a registry device with a MAC
  connection for every client they track, so a registry hit would trust
  every new client and silence `network_unknown_device_joined`. A client is
  auto-trusted only when the joined registry device has at least one config
  entry whose domain is not a router, tracker, or discovery platform (the
  adapter platforms, `device_tracker`-only integrations, `dhcp`, `ssdp`,
  `zeroconf`) and that entry owns at least one non-`device_tracker` entity.
  A Shelly plug qualifies; a Fritz-created client device does not.
  Everything else stays untrusted until the user approves it.
- **Services:** `sentinel_get_network_inventory`,
  `sentinel_trust_network_device`, `sentinel_untrust_network_device`,
  `sentinel_reset_network_inventory`. The notifier gains a "Trust device"
  action button for unknown-device findings. Trusting a device is an
  inventory write, not network actuation, so it is allowed in phase 1.

## Sentinel Rules

All rules live in `sentinel/rules/` and follow the existing
`rule_id` + `evaluate(snapshot) -> list[AnomalyFinding]` shape. Each gains a
`requires: frozenset[str]` class attribute of capability paths.

### Capability gating in the engine

`SentinelEngine` checks `rule.requires <= set(snapshot["network"]["capabilities"])`
before calling `evaluate`. Static rules without `requires` are unchanged.
Skipped rules are counted in `run_stats["inactive_rules"]` as
`{rule_id: [missing capabilities]}` and surfaced on the `sentinel_health`
sensor so the user can see why a check is not running and which integration
would enable it.

### Rule catalogue

| rule_id | requires | severity | Phase |
|---|---|---|---|
| `network_unknown_device_joined` | `network.clients` | high while away or at night, medium otherwise | 1 |
| `network_upnp_enabled` | `network.posture.upnp_enabled` | medium | 1 |
| `security_device_unavailable` | none (HA-native); `network.clients` corroborates when present | high for lock/alarm/camera unavailable > N min | 1 |
| `network_router_update_pending` | `network.posture.router_update_pending` | medium | 1 |
| `ha_sensitive_entity_exposed_without_pin` | `network.ha_security.exposed_sensitive_entities` | high | 1 |
| `ha_long_lived_token_stale` | `network.ha_security.long_lived_tokens_unused_days` | low | 1 |
| `ha_new_admin_or_token` | `network.ha_security.*` | high | 1 |
| `ha_failed_logins` | `network.ha_security.failed_login_notification_present` | medium | 1 |
| `ha_cloud_remote_ui_enabled` | `network.ha_security.cloud_remote_ui_enabled` | low, informational | 1 |
| `ha_addon_exposed_port` | `network.ha_security.addons_with_host_ports` | medium; high for SSH, Samba, or a database | 1 |
| `ha_addon_unprotected` | `network.ha_security.addons_unprotected` | high | 1 |
| `ha_webhook_automation_public` | `network.ha_security.webhook_automations_public` | medium; high when the automation targets a critical action | 1 |
| `ha_trusted_networks_bypass_login` | `network.ha_security.trusted_networks_bypass_login` | medium | 1 |
| `ha_http_proxy_misconfigured` | `network.ha_security.http_*` | medium | 1 |
| `network_unconfigured_discovered_device` | `network.ha_security.discovered_unconfigured` | low; medium while away for a camera or NVR handler | 1 |
| `network_guest_network_idle` | `network.posture.guest_network_enabled`, `guest_client_count` | low, after N days idle | 1 |
| `network_wpa3_disabled` | `network.posture.wpa3_enabled` | low, advisory only | 1 |
| `network_protection_disabled` | `malware_blocking_enabled` or `ad_blocking_enabled` | medium | 1 |
| `network_ddns_enabled` | `network.posture.ddns_enabled` | low | 1 |
| `network_port_forward_active` | `network.posture.port_forwards` | medium; high when the target is a camera or NVR | 1 |
| `network_iot_device_on_main_vlan` | `network.clients[].vlan` | low, advisory | 1 |
| `network_guest_client_present` | `network.clients[].is_guest` | medium while away or at night | 1 |
| `zwave_insecure_security_class` | `network.radio.devices[].security_class` | high for locks and garage doors on S0 or none; low otherwise | 1 |
| `zigbee_permit_join_open` | `network.radio.posture.zigbee_permit_join` | medium; high while away | 1 |
| `radio_new_device_joined` | `network.radio.devices` | medium while away or at night, low otherwise | 1 |
| `radio_coordinator_update_pending` | `network.radio.posture.coordinator_update_pending` | medium | 1 |
| `bluetooth_unknown_tracker_present` | `network.radio.posture.bluetooth_unknown_trackers` | medium after N days | 2 |
| `matter_unexpected_fabric` | `network.radio.devices[].fabrics` | low | 3 |
| `network_client_identity_drift` | `network.clients` | low | 2 |
| `network_public_ip_changed` | `network.posture.public_ip_changed` | info | 2 |

Rule conventions:

- A rule may also declare `corroborates: frozenset[str]` for capabilities
  that improve its evidence but are not required. `security_device_unavailable`
  runs everywhere on entity availability and, when `network.clients` is
  present, adds whether the router still sees the device, which separates
  an integration outage from a device that left the network.

- `triggering_entities` holds the tracker, switch, or update entity id so the
  existing per-type entity exclusions and snooze machinery work unchanged.
- `evidence` carries the pseudonymized client key, never a raw MAC, plus the
  display fields. Display-only fields go through
  `DISPLAY_ONLY_EVIDENCE_KEYS` so anomaly ids stay stable.
- `is_sensitive = True` for every network and HA-security finding. The
  notifier already treats sensitive findings conservatively.
- `suggested_actions` are strings in phase 1 ("Disable UPnP on your router",
  "Trust this device if you recognize it"). Phase 2 turns them into
  structured fix descriptors (see Remediation).
- Unknown-device suppression: a client whose MAC has the locally
  administered bit set (randomized MAC) and whose hostname matches a trusted
  client is reported once at low severity, not as a new device every
  rotation. Phones do this constantly.

## HA-Only MVP

A home with no router or DNS integration still gets a complete, useful audit,
because the highest-value target on the network is Home Assistant itself and
HA already knows its own attack surface. This is the minimum every install
receives, and it is the first thing to build and validate.

| Rule | Runs with no third-party integration | Source |
|---|---|---|
| `ha_sensitive_entity_exposed_without_pin` | yes | exposed-entities registry, HGA options |
| `ha_new_admin_or_token` | yes | auth registry |
| `ha_long_lived_token_stale` | yes | auth registry |
| `ha_failed_logins` | yes | `http-login` persistent notification |
| `ha_cloud_remote_ui_enabled` | yes, when the cloud integration is loaded | cloud preferences |
| `ha_addon_exposed_port` / `ha_addon_unprotected` | yes on HA OS and Supervised | Supervisor add-on info |
| `ha_webhook_automation_public` | yes | automation raw config |
| `ha_trusted_networks_bypass_login` | yes | auth providers |
| `ha_http_proxy_misconfigured` | yes | `http` runtime config |
| `security_device_unavailable` | yes | entity availability |
| `network_unconfigured_discovered_device` | yes | in-progress and ignored discovery flows |
| `radio_new_device_joined` | yes, for any home with ZHA, Zigbee2MQTT, Z-Wave JS, Bluetooth, or Matter | device registry |
| `zwave_insecure_security_class`, `zigbee_permit_join_open` | yes, when the protocol integration is loaded | Z-Wave JS and Zigbee integrations (core, not third-party) |
| `network_router_update_pending` | partly | any `update.*` entity |
| `network_upnp_enabled`, `network_public_ip_changed` | on about a third of installs | the core, auto-discovered `upnp` integration |
| `network_unknown_device_joined`, guest, WPA3, port-forward, VLAN rules | no | need a router adapter |

The `audit_home_security` tool reports the last row as missing capabilities
and names the integrations that would unlock them, so a user without a router
integration is told exactly what they are not seeing rather than shown a
clean bill of health.

Validation order: build `ha_native` and the rules above first, run them on a
plain HA install with no router integration, and only then add the router
adapters. If the HA-only audit is not useful on its own, the router adapters
will not rescue it.

## Radio Protocols

The IP network is not the only network in the home. Zigbee, Z-Wave,
Bluetooth, Thread, and Matter devices include most locks and sensors, and HA
has direct visibility into their configuration through the integrations that
drive them. The same adapter and capability pattern applies; the data lands
in `snapshot["network"]["radio"]`.

What HA can observe, ranked by value:

| Check | Source | Phase |
|---|---|---|
| Z-Wave node security class | Z-Wave JS reports each node's highest security class. A lock or garage opener on S0 or with no security is a concrete, well-documented weakness. | 1 |
| Zigbee permit-join left open | Zigbee2MQTT exposes a bridge permit-join switch through MQTT discovery. ZHA exposes the coordinator's permit state through its gateway; confirm the access path against the pinned HA version at implementation time, and report the capability as missing if it is not readable. | 1 |
| New radio device joined | Device-registry delta filtered by integration domain (`zha`, `mqtt` with a Zigbee2MQTT identifier, `zwave_js`, `bluetooth`, `matter`). Feeds the general inventory. | 1 |
| Coordinator and proxy firmware | `update.*` entities whose device is a Zigbee or Z-Wave coordinator or an ESPHome Bluetooth proxy, weighted as security devices. | 1 |
| Z-Wave inclusion left active | Z-Wave JS controller inclusion state. | 1 |
| Unknown Bluetooth tracker | The Bluetooth integration already receives every BLE advertisement in range. A tracker-class advertisement (Apple Find My, Samsung SmartTag, Tile) that is not a configured device and stays present for days is a privacy finding. Passive: HA is already listening, so this stays inside the no-scanning rule. | 2 |
| Matter fabrics | Matter node diagnostics list the fabrics a device is commissioned to. A device still attached to a vendor fabric the user does not recognize is a privacy exposure. | 3 |
| Thread border routers | Count and dataset presence, informational. | 3 |

Adapter notes:

- `zwave`: reads node security class and controller state from the Z-Wave JS
  integration's node objects, never from the driver directly. Security class
  is normalized to `none`, `s0`, `s2_unauth`, `s2_auth`, `s2_access`.
- `zigbee`: two sources behind one adapter. Zigbee2MQTT bridge entities
  arrive through `mqtt` and are recognized by the bridge device identifier,
  not by friendly name. ZHA state comes from the ZHA gateway.
- `radio_registry`: pure device-registry read, always available, so
  `radio_new_device_joined` is part of the HA-only MVP for any home with a
  radio integration.
- `bluetooth`: advertisement addresses are pseudonymized like MACs. Tracker
  classification uses manufacturer-data prefixes and is conservative:
  unknown means "not a configured device and not an HA-known manufacturer",
  and the rule requires persistence across days before it fires. A resident's
  own tag registered through the Private BLE Device integration is a
  configured device and never flagged.
- `matter`: fabric vendor ids are matched against the device's own
  manufacturer and the user's configured ecosystems; anything else is
  reported, not judged.

Explicitly not covered: active protocol attacks such as Zigbee key sniffing
during join, Z-Wave S0 downgrade, Bluetooth pairing exploits, jamming, or
replay. These need RF monitoring that HA does not perform, and the audit says
so in its `privacy_notes` rather than implying radio coverage it lacks.

Remediation (phase 2): turn off permit join or inclusion via the exposing
switch, both PIN-gated. An insecure Z-Wave security class has no automated
fix; the finding tells the user to re-include the device with S2.

## Baseline Extension

`SentinelBaselineUpdater` currently evaluates power-class entities. Add a
generic numeric-counter path fed from `snapshot["network"]["counters"]`:

- `network.client_count` per hour-of-day and day-of-week.
- `network.threats_day` and `network.adblock_day` deltas per cycle.
- `network.client.<key>.usage_day_bytes` delta per cycle, per client.

Findings use the existing `baseline_deviation` type with a `subject` evidence
field so the notifier can say "your living room TV uploaded 4x its usual
amount today". The Welford and day-of-week code is reused as is; only the
extraction of the numeric value and the unit class differ.

## Discovery Extension

`sentinel/dynamic_rules.py` gains network templates so discovery can propose
per-home rules from the normalized section: `network_client_absent_when`
(a device that should be present at a given time is not), `network_client_present_when`
(a device that should not be present is), and `network_posture_equals`
(alert when a posture switch flips). The proposal, preview, approve, and reject
services and the proposals card work unchanged.

## Agent Tool: `audit_home_security`

Registered in `conversation.py` alongside the other LangChain tools and indexed
for RAG tool retrieval like the rest.

```python
@tool(parse_docstring=True)
async def audit_home_security(
    scope: Literal["all", "network", "home_assistant", "privacy"] = "all",
    include_devices: bool = False,
) -> dict[str, Any]:
    """Audit the home network and Home Assistant security posture.

    Reads the latest snapshot's network section, runs every network rule
    whose requirements are met, and returns structured findings plus a
    capability report. Never scans the network or changes anything.
    """
```

Returns:

```json
{
  "generated_at": "...",
  "capabilities": ["network.clients", "network.posture.upnp_enabled", ...],
  "missing_capabilities": {"network.posture.wpa3_enabled": "no router integration exposes this"},
  "findings": [ {AnomalyFinding.as_dict()} ],
  "inventory": {"trusted": 23, "untrusted": 1, "offline_security_devices": []},
  "devices": [ ... only when include_devices, pseudonymized ... ],
  "privacy_notes": ["Your router integration is cloud-only; client data transits the vendor's servers."]
}
```

The payload above is what the tool assembles internally and what the
`network_audit` model sees. What is returned to the conversation model is
the digest described under Local-model override; `devices` and the
detailed evidence stay inside the tool unless the conversation provider is
the audit provider. The tool reads the snapshot; it never polls
integrations, so it is safe to call repeatedly and respects the router
integration's own rate limits. The
system prompt gains a short instruction: use this tool when the user asks
about network security, privacy, unknown devices, or whether the home is
"safe"; report findings by severity; state which checks could not run and
why; never claim a check passed when its capability is missing.

Scheduled use is an ordinary HA automation calling the conversation agent,
or, more simply, the Sentinel rules above, which already run on the detection
interval. A dedicated `run_network_audit` service returns the same payload
for dashboards.

## Privacy of the Audit

### Pseudonymization

`snapshot/network.py` derives each client key as
`hmac_sha256(install_salt, mac)[:8]` with a per-install salt stored in the
config entry, reusing the salt pattern from the critical-action PIN. Before
any LLM call (explain, triage, discovery, or the tool result rendered for the
agent) a `redact_network_identifiers()` step, next to `_redact_person_names()`
in `explain/llm_explain.py`, strips `mac` and `ip`, replaces hostnames with
`manufacturer + key`, and keeps only the subnet class of IPs ("lan",
"public"). The notifier re-hydrates the display name from the inventory when
rendering the final notification, after the model has produced its text.

### Local-model override

Add `network_audit` to `FEATURE_DEFS` (`{"name": "Network Audit", "required": False}`)
mapped to category `chat` in `FEATURE_CATEGORY_MAP`. The existing feature
subentry flow then lets the user pin network-rule explanations, network
triage, network discovery, and the audit tool's own narrative step to a
specific provider, typically a local Ollama model. When the feature subentry
is absent, resolution falls back to the conversation provider exactly as
`conversation_summary` does today.

**What the override does not do.** The agent tool returns a `ToolMessage`
that the graph feeds back to the conversation model. A feature mapping
cannot change that: whatever the tool returns reaches the conversation
provider. So the override is designed around that fact rather than around
the assumption it can be avoided.

- The tool receives the `network_audit` model through the graph's
  `configurable` mapping, the same way `get_and_analyze_camera_image`
  receives `vlm_model`. It runs the narrative step on that model, inside the
  tool, over the full pseudonymized payload.
- The tool's return value to the conversation model is a **digest**, not the
  payload: severity counts, the narrative produced by the audit model, the
  capability and missing-capability lists, and per-finding `type`,
  `severity`, and pseudonymized `key`. No manufacturer strings, hostnames,
  usage figures, token client names, add-on slugs, or automation ids leave
  the tool. `include_devices` is honored only when the conversation
  provider is the same provider as `network_audit`; otherwise the tool
  returns the digest and says so in `privacy_notes`.
- When no `network_audit` feature is configured, the conversation provider
  is the audit model and receives the full pseudonymized payload. The
  README states this plainly: pinning `network_audit` to a local provider
  keeps audit detail local; without it, the conversation provider sees the
  pseudonymized audit.
- Sentinel-side calls (explain, triage, discovery) never touch the
  conversation graph, so for them the override is complete.

### Disclosure

The audit's `privacy_notes` always state where network data comes from. For
eero this means telling the user that the integration is cloud-only and that
eero Secure's inspected-content counters imply vendor-side inspection.

## Configuration

Extend the existing Sentinel subentry (Advanced setup) rather than add a new
subentry type; the rules run inside the Sentinel engine and share its
interval, quiet hours, cooldowns, and exclusions. New keys in `const.py`:

| Key | Default | Purpose |
|---|---|---|
| `sentinel_network_enabled` | `True` | master switch for network adapters and rules |
| `sentinel_network_router_platform` | `auto` | force a specific adapter when auto-detect picks wrong |
| `sentinel_network_unknown_device_grace_min` | `5` | avoid alerting on a device that appears for one poll |
| `sentinel_network_offline_device_min` | `30` | threshold for security-device-offline |
| `sentinel_network_guest_idle_days` | `7` | threshold for guest-network-idle |
| `sentinel_ha_token_stale_days` | `90` | threshold for stale long-lived tokens |
| `sentinel_auth_ip_retention_days` | `90` | how long a pseudonymized token IP stays in the auth inventory |
| `sentinel_network_fixes_enabled` | `False` | phase 2 only |

Basic setup leaves all of these at defaults. `docs/constants.md` and
`docs/configuration.md` are updated alongside.

## Remediation (Phase 2)

Every fix is an existing HA entity call, so no new actuation code is needed:

| Finding | Fix | Entity |
|---|---|---|
| unknown device | block or pause client | UniFi block-client switch; Fritz `switch.*_internet_access`; eero `switch.*_paused` |
| port forward active | turn off | Fritz and UniFi port-forward switches |
| Zigbee permit join open | turn off | Zigbee2MQTT bridge switch; ZHA permit service with duration 0 |
| Z-Wave inclusion active | stop inclusion | Z-Wave JS controller |
| UPnP enabled | turn off | `switch.*_upnp` |
| guest network idle | turn off | `switch.*_guest_network` |
| WPA3 disabled | turn on | `switch.*_wpa3` (advisory, may drop old devices) |
| protection disabled | turn on | `switch.*_advanced_security`, `switch.*_ad_blocking` |
| router update pending | install | `update.install` |
| stale long-lived token | revoke | HA auth API, via a new HGA service |
| exposed sensitive entity | unexpose | exposed-entities helper, via a new HGA service |

Mechanics:

- `AnomalyFinding.suggested_actions` stays `list[str]`. It is joined into
  notification text in `notify/actions.py`, split per entry by
  `sentinel/execution.py`, filtered for dotted service names in
  `sentinel/engine.py`, counted by triage, and exposed as a string list in
  event payloads and the proposals UI. Changing its type would break every
  one of those. Network fixes go in a new, additive field
  `remediations: list[Remediation]` with `field(default_factory=list)`,
  where `Remediation` is a frozen dataclass `{label, domain, service,
  entity_id, reversible: bool}`. `as_dict()` serializes it under
  `remediations`; audit records written before the field exists load with
  an empty list via `_migrate_record`; `build_anomaly_id()` excludes it
  like the display-only keys so ids stay stable. The notifier renders
  `remediations` as action buttons and continues to use `suggested_actions`
  for text. Existing rules are untouched.
- For autonomous execution, each `Remediation` is also mirrored into
  `suggested_actions` as the existing `domain.service` string form, so the
  `allowed_services` allowlist and the execution gate work without change.
- Every network fix is added to `RECOMMENDED_CRITICAL_ACTIONS` with
  `entity_match` patterns (`upnp`, `guest_network`, `paused`, `wpa3`), so both
  a direct agent call and an LLM-authored automation hit the PIN gate in
  `matches_critical_rule`.
- Autonomous execution at Sentinel level 2/3 uses the existing
  `allowed_services` allowlist, `max_actions_per_hour`, and canary mode; the
  defaults do not include any network service, so nothing runs on its own
  unless the user opts in.
- Fixes that are capability-gated and unavailable on a given home degrade to
  the phase 1 text recommendation.

## Phase 3: More Adapters and LLM-Assisted Mapping

- Pi-hole, ASUSWRT, TP-Link Omada, Keenetic, MikroTik, and NETGEAR adapters,
  each a fixture-tested mapping table like the phase 1 ones.
- For a router platform without an adapter, an advisory flow asks the
  configured model once to propose a mapping from that platform's entities to
  the normalized fields. The proposal is shown to the user, approved or
  rejected, and stored in a `network_adapter_registry` shaped like the rule
  registry. Runtime remains deterministic because the stored mapping is a
  plain lookup table; the model is only involved at proposal time. This is
  the same advisory pattern discovery uses for rules.

## Testing

- `tests/.../test_snapshot_network.py`: adapter unit tests with `hass.states`
  fixtures per platform (fritz and unifi fixtures built from the core
  integration sources, eero from the community integration's entity keys),
  merge and precedence, capability list correctness, MAC join to the device
  registry, pseudonymization stability across runs and instability across
  installs.
- `tests/.../test_snapshot_ha_native.py`: auth, exposed entities, add-on
  info, webhook automations, auth providers, discovery flows, and
  availability, each with and without the underlying component loaded, so
  a Container install reports add-on capabilities as missing rather than
  failing.
- `tests/.../test_snapshot_radio.py`: Z-Wave security-class normalization,
  Zigbee2MQTT bridge recognition by identifier, ZHA gateway state, device
  registry deltas per protocol, Bluetooth tracker classification with
  known-tag and Private BLE Device exclusions.
- `tests/.../test_rules_network.py`: each rule against synthetic snapshots,
  including the randomized-MAC suppression and grace windows.
- `tests/.../test_engine_capability_gating.py`: rules skipped when
  capabilities are missing and reported on the health sensor.
- `tests/.../test_network_inventory.py`: bootstrap, services, and
  auto-trust negative cases: a Fritz- or UniFi-created client device must
  not be auto-trusted; a Shelly device must.
- `tests/.../test_auth_inventory.py`: bootstrap without alerts, new admin
  and new token detection across a simulated restart, token-row deletion
  when HA drops the token, `seen_ips` bound and retention, no raw IP or
  token value anywhere in the persisted JSON.
- `tests/.../test_snapshot_radio.py` runs against the real Z-Wave JS and
  ZHA objects from the pinned `homeassistant` version, not mocks of them,
  so the runtime-adapter boundary is exercised by CI.
- `tests/.../test_audit_home_security_tool.py` asserts the digest contract:
  the returned payload contains no key outside the allowed set when the
  conversation provider differs from the audit provider.
- `tests/.../test_audit_home_security_tool.py`: tool output schema, no raw
  identifiers in the payload, missing-capability reporting.
- `tests/.../test_llm_explain.py`: extend for `redact_network_identifiers`.
- `tests/.../test_options_schema.py`: new Sentinel keys.
- README dashboard recipe test if a network card is added.

## Documentation

- README: new feature bullet and a "Network security audit" example.
- `docs/sentinel.md`: rules table rows, capability gating, inventory services.
- `docs/architecture.md`: snapshot `network` section and adapter diagram.
- `docs/configuration.md`, `docs/constants.md`: new keys.
- CHANGELOG entry per phase.

## Risks and Mitigations

- **Alert fatigue from unknown devices.** Mitigated by bootstrap trust,
  auto-trust via device-registry join, randomized-MAC suppression, the grace
  window, and the existing cooldown and snooze machinery.
- **Router integration churn.** Community integrations rename entity keys.
  Adapters match on `platform` plus translation key or unique-id suffix and
  are fixture-tested, so a rename fails a test rather than silently dropping
  a capability. Missing capabilities are reported, never assumed.
- **False sense of security.** The tool and notifications always list what
  could not be checked. The health sensor exposes inactive rules.
- **Privacy of the audit data.** Pseudonymization is unconditional; the local
  model override is optional on top.
- **PIN-gated fixes that break connectivity.** WPA3 and IPv6 changes are
  advisory only and never auto-executed; pause-client is reversible and its
  notification carries an "Unpause" action.

## Decisions

Resolved with the maintainer during design review.

1. **HA-security checks are on by default.** The token, admin-user,
   failed-login, and exposed-entity rules run at the severities in the
   catalogue. Users behind a VPN who consider them noise exclude them by type
   key like any other rule.
2. **Adapter priority follows install share, not the maintainer's hardware.**
   Fritz!Box and UniFi are the most-used router integrations by a wide margin
   and ship in phase 1 alongside the UPnP/IGD signal and AdGuard. eero ships
   in phase 1 as the dogfood and cloud-mesh reference adapter. eero Plus
   dependence needs no configuration: Plus-only capabilities are simply
   reported as missing on non-Plus accounts, and the docs say so.
3. **`network_client_identity_drift` is deferred** to phase 2 and ships only
   if phase 1 inventory data shows it would be quiet enough on real homes.
4. **Both pseudonymization and the local-model override are phase 1.**
   Pseudonymization is unconditional; the `network_audit` feature type lets
   the audit be pinned to a local provider.
5. **Radio protocols are in scope for configuration weaknesses HA can
   observe.** Z-Wave security class, Zigbee permit-join, new radio devices,
   and coordinator firmware are phase 1. Bluetooth tracker detection is
   phase 2, Matter fabrics phase 3. Active radio attacks are a stated
   non-goal.
6. **Review findings on PR #611, all accepted.** The tool returns a digest
   and runs its narrative on the `network_audit` model in-tool, because the
   conversation provider always receives the `ToolMessage`; auto-trust
   requires a non-router integration behind the registry device; auth
   change rules compare against a persistent, non-secret auth inventory;
   adapters are split into entity and runtime tiers with version-pinned
   tests for the latter; `suggested_actions` keeps its string contract and
   fixes live in an additive `remediations` field.

## Implementation Order (Phase 1)

1. `platform` field on `SnapshotEntity`, builder change, schema bump.
2. `snapshot/network.py` with the `ha_native` adapter only; pseudonymization
   helper; capability list.
3. `sentinel/auth_inventory.py` store and reset service.
4. Engine capability gating and health-sensor reporting.
5. The HA-only MVP rules, validated on an install with no router
   integration.
6. `radio_registry`, `zwave`, and `zigbee` adapters with their phase 1
   rules; the general device inventory.
7. `generic_router_tracker`, `upnp_igd`, `fritz`, `unifi`, `adguard`, and
   `eero` adapters, then the remaining phase 1 rules; router clients join
   the inventory as a second source.
8. `redact_network_identifiers` in the explain path; `network_audit` feature
   type and resolver plumbing.
9. `audit_home_security` tool, system prompt addition, `run_network_audit`
   service.
10. Baseline counters and discovery templates.
11. Docs and CHANGELOG.

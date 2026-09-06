# Network Security & Privacy Audit Plan

Status: design proposal (no code yet)

## Problem Statement

Home Assistant knows a great deal about a home's network, but that knowledge is
scattered across the device registry, the auth system, the exposed-entities
registry, `update` entities, and whichever router integration the user happens
to run. Nothing in HA or HGA reads across those sources to answer "is my home
network configured safely, and has anything changed that I should know about?"

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
- Explain findings and per-device privacy exposure in plain language via the
  LLM, using pseudonymized identifiers and an optional local-model override.
- Work on every install: a home with no router integration still gets the
  HA-native audit; a home with eero, UniFi, or another supported router gets
  progressively more.
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


class NetworkSnapshot(TypedDict):
    capabilities: list[str]  # dotted paths present, e.g. "network.posture.upnp_enabled"
    sources: dict[str, str]  # capability -> platform that provided it
    clients: list[NetworkClient]
    posture: NetworkPosture
    ha_security: HaSecurityPosture
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
| `pi_hole` | `platform == "pi_hole"` | protection switch, blocked-query counters | 3 |
| `asuswrt` / `tplink_omada` / `keenetic_ndms2` / `mikrotik` / `netgear` | by platform | clients; per-integration posture where exposed | 3 |

Adapters must not import integration code. They read entity state and
attributes only, so they work whether the integration is core or HACS and
they are testable with plain `hass.states` fixtures.

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

#### `ha_native` adapter details

- **Auth.** `hass.auth.async_get_users()` gives `is_admin`, `is_active`, and
  each user's refresh tokens with `token_type`, `client_name`, `created_at`,
  `last_used_at`, and `last_used_ip`. Long-lived tokens unused for more than a
  configurable number of days, and a refresh token whose `last_used_ip` was
  not seen in the previous run, become posture fields. No token values are
  ever read into the snapshot.
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

## Known-Device Inventory

A new `sentinel/network_inventory.py` store, shaped like `RuleRegistry`
(`Store` under key `home_generative_agent_sentinel_network_inventory`).

- Keyed by the pseudonymized client key, holding `first_seen`, `last_seen`,
  `trusted` (bool), the last observed hostname and manufacturer, and the HA
  device id when joined.
- **Bootstrap:** on the first run after enablement every currently connected
  client is recorded with `trusted = True` and a single "inventory
  established" notification asks the user to review it. This mirrors the
  baseline establishment notification and avoids a flood of alerts on day one.
- **Auto-trust:** a client that joins to an HA device in the registry is
  trusted automatically; HA already knows what it is.
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
| `network_security_device_offline` | `network.clients` | high for lock/alarm/camera devices offline > N min | 1 |
| `network_router_update_pending` | `network.posture.router_update_pending` | medium | 1 |
| `ha_sensitive_entity_exposed_without_pin` | `network.ha_security.exposed_sensitive_entities` | high | 1 |
| `ha_long_lived_token_stale` | `network.ha_security.long_lived_tokens_unused_days` | low | 1 |
| `ha_new_admin_or_token` | `network.ha_security.*` | high | 1 |
| `ha_failed_logins` | `network.ha_security.failed_login_notification_present` | medium | 1 |
| `network_guest_network_idle` | `network.posture.guest_network_enabled`, `guest_client_count` | low, after N days idle | 1 |
| `network_wpa3_disabled` | `network.posture.wpa3_enabled` | low, advisory only | 1 |
| `network_protection_disabled` | `malware_blocking_enabled` or `ad_blocking_enabled` | medium | 1 |
| `network_ddns_enabled` | `network.posture.ddns_enabled` | low | 1 |
| `network_port_forward_active` | `network.posture.port_forwards` | medium; high when the target is a camera or NVR | 1 |
| `network_iot_device_on_main_vlan` | `network.clients[].vlan` | low, advisory | 1 |
| `network_guest_client_present` | `network.clients[].is_guest` | medium while away or at night | 1 |
| `network_client_identity_drift` | `network.clients` | low | 2 |
| `network_public_ip_changed` | `network.posture.public_ip_changed` | info | 2 |

Rule conventions:

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

The tool reads the snapshot; it never polls integrations, so it is safe to
call repeatedly and respects the router integration's own rate limits. The
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
subentry flow then lets the user pin the audit tool, network-rule
explanations, and network discovery to a specific provider, typically a local
Ollama model, while conversation stays on a cloud provider. When the feature
subentry is absent, resolution falls back to the conversation provider exactly
as `conversation_summary` does today.

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
| `sentinel_network_fixes_enabled` | `False` | phase 2 only |

Basic setup leaves all of these at defaults. `docs/constants.md` and
`docs/configuration.md` are updated alongside.

## Remediation (Phase 2)

Every fix is an existing HA entity call, so no new actuation code is needed:

| Finding | Fix | Entity |
|---|---|---|
| unknown device | block or pause client | UniFi block-client switch; Fritz `switch.*_internet_access`; eero `switch.*_paused` |
| port forward active | turn off | Fritz and UniFi port-forward switches |
| UPnP enabled | turn off | `switch.*_upnp` |
| guest network idle | turn off | `switch.*_guest_network` |
| WPA3 disabled | turn on | `switch.*_wpa3` (advisory, may drop old devices) |
| protection disabled | turn on | `switch.*_advanced_security`, `switch.*_ad_blocking` |
| router update pending | install | `update.install` |
| stale long-lived token | revoke | HA auth API, via a new HGA service |
| exposed sensitive entity | unexpose | exposed-entities helper, via a new HGA service |

Mechanics:

- `suggested_actions` becomes a list of `{label, domain, service, entity_id}`
  descriptors. The notifier renders them as action buttons.
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
- `tests/.../test_rules_network.py`: each rule against synthetic snapshots,
  including the randomized-MAC suppression and grace windows.
- `tests/.../test_engine_capability_gating.py`: rules skipped when
  capabilities are missing and reported on the health sensor.
- `tests/.../test_network_inventory.py`: bootstrap, auto-trust, services.
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

## Implementation Order (Phase 1)

1. `platform` field on `SnapshotEntity`, builder change, schema bump.
2. `snapshot/network.py` with `ha_native`, `generic_router_tracker`,
   `upnp_igd`, `fritz`, `unifi`, `adguard`, and `eero` adapters;
   pseudonymization helper; capability list.
3. `sentinel/network_inventory.py` store and services.
4. Engine capability gating and health-sensor reporting.
5. Phase 1 rules from the catalogue, in the order listed.
6. `redact_network_identifiers` in the explain path; `network_audit` feature
   type and resolver plumbing.
7. `audit_home_security` tool, system prompt addition, `run_network_audit`
   service.
8. Baseline counters and discovery templates.
9. Docs and CHANGELOG.

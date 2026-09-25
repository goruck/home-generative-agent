# Network Security Plan, Step 6: Radio Adapters and Device Inventory

Status: implemented on `feat/network-radio-adapters`, with the three maintainer decisions below accepted as proposed; field validation on the maintainer's box pending. This implements implementation-order step 6 of
`docs/network-security-plan.md`: the `radio_registry`, `zwave`, and `zigbee`
adapters, their phase 1 rules, and the general device inventory. The parent
plan still governs; this document records what the pinned Home Assistant
version actually allows and the decisions that follow from it.

## What the pinned versions allow

Home Assistant 2026.9.1 pins `zha==2.2.1` (with `zigpy==2.1.0`) and
`zwave-js-server-python==0.73.1`.

| Fact the plan wants | Readable? | Source |
|---|---|---|
| Zigbee permit-join, ZHA | **No.** `ControllerApplication.permit()` broadcasts `Mgmt_Permit_Joining_req` and keeps no record of it. The frontend's "Add device" uses the `zha/devices/permit` websocket command, so no service-call event fires either. Neighbor-table `permit_joining` is refreshed only by topology scans, hours apart. | zigpy 2.1.0 `application.py`, HA `zha/websocket_api.py` |
| Zigbee permit-join, Zigbee2MQTT | Yes. The bridge publishes a `switch` with unique id `bridge_<coordinator ieee>_permit_join_<base topic>` on a device whose MQTT identifier is `zigbee2mqtt_bridge_<ieee>`. Z2M 2.x permits for at most 254 s per request. | Zigbee2MQTT `lib/extension/homeassistant.ts` |
| Z-Wave node security class | Yes. `Node.highest_security_class` gives `SecurityClass` (`NONE=-1`, `S2_UNAUTHENTICATED=0`, `S2_AUTHENTICATED=1`, `S2_ACCESS_CONTROL=2`, `S0_LEGACY=7`), or `None` before the interview finishes. | zwave-js-server-python 0.73.1 |
| Z-Wave inclusion active | Yes. `Controller.inclusion_state` (`IDLE`, `INCLUDING`, `EXCLUDING`, `BUSY`, `SMART_START`). | same |
| Radio devices joined | Yes. Device registry, config-entry domain plus identifiers. | HA device registry |
| Coordinator firmware | Partly. The ZHA coordinator has no `update` entity. Connect ZBT-1/ZBT-2 and Yellow radio firmware do (`homeassistant_sky_connect`, `homeassistant_connect_zbt2`, `homeassistant_yellow`), as does the Z2M bridge when it exposes one. ESPHome Bluetooth proxies are identified by joining `bluetooth.async_current_scanners()` sources to ESPHome devices' MAC connections. | HA components |

ZHA permit-join is therefore reported as a missing capability, with that
reason, rather than approximated. This is the parent plan's rule for runtime
adapters whose data the pinned version does not expose.

## Scope

### 1. Device inventory: `sentinel/network_inventory.py`

A `Store` under `home_generative_agent_sentinel_network_inventory`, shaped
like `AuthInventory`.

- **Rows** are keyed `"<source>:<device_id>"`, where source is one of
  `zigbee`, `zwave`, `bluetooth`, `matter` (router sources join in step 7).
  Each row holds `source`, `platform`, `ha_device_id`, `name`,
  `manufacturer`, `model`, `first_seen`, `last_seen`, and `trusted`. No
  IEEE addresses, node ids, or MACs are stored.
- **Bootstrap is per source.** The first time a source is observed, every
  device from it is recorded as trusted without alerting. One persistent
  notification per bootstrap names the counts. The parent plan bootstraps
  the whole inventory once. Per source is quieter and still correct: adding
  a Z-Wave stick next month records the controller, and each device included
  after that alerts.
- **Diff, then commit after dispatch, with hold-back.** This is the auth
  inventory's pattern. A device whose finding was not delivered stays out of
  the commit and is reported again on the next run.
- **Retention.** A row is deleted when its device leaves the device registry.
  Home Assistant restores a removed device's registry id when the same
  hardware is added back, so a device removed and re-paired alerts again only
  if a Sentinel run saw it gone in between; re-pairing a device that was never
  removed changes nothing, since the registry entry is the same.
- **Services:**
  - `sentinel_get_network_inventory` (read, response only).
  - `sentinel_trust_network_device` and `sentinel_untrust_network_device`
    (admin only).
  - `sentinel_reset_network_inventory` (admin only, logged with the caller,
    like the auth reset).
- **Trust button.** `radio_new_device_joined` pushes carry a "Trust device"
  action when the finding names one device; a finding naming several has
  no primary button and its body points at the trust service (one tap must
  never trust a device the 220-character push did not show). The handler
  trusts the one device named in the finding, refuses a tap naming several,
  and ignores the tap when the mobile-app event's user is not an admin.
- Deleting the Sentinel subentry deletes the store, next to the auth
  inventory and salt.

### 2. Adapters (in `snapshot/network.py`, or a new `snapshot/radio.py` if it gets long)

All output lands in `network.radio` (`RadioSnapshot`, already in the
schema). Capability paths are `network.radio.<field>`.

- **`radio_registry`** (entity tier, always runs). It reads the device
  registry and classifies each device:
  - `zha` entry → `zigbee`.
  - `mqtt` device with identifier matching `^zigbee2mqtt_(bridge_)?0x[0-9a-f]{16}$` → `zigbee`. Z2M group identifiers carry the base topic and do not match.
  - `zwave_js` entry, skipping `provision_` identifiers → `zwave`.
  - `matter` entry → `matter`.
  - A device with a `bluetooth` connection whose own entry is not the `bluetooth` adapter entry → `bluetooth`.

  `is_security_device` is true when the device owns a lock, alarm panel,
  camera, or entry cover, reusing `is_sensitive_entity`. `first_seen` comes
  from the inventory. The adapter provides `network.radio.devices` whenever
  any radio integration is loaded.
- **`zwave`** (runtime tier). For each loaded `zwave_js` entry it reads
  `entry.runtime_data.client.driver.controller`. It does not import
  `homeassistant.components.zwave_js`, so no new `after_dependencies` entry
  is needed. Each node is joined to its registry device by the
  `<home_id>-<node_id>` identifier. `int(highest_security_class)` is
  normalized to `none | s0 | s2_unauth | s2_auth | s2_access`; `None`
  (interview pending) is left unasserted. The controller node is skipped.
  `zwave_inclusion_active` is `inclusion_state == INCLUDING`. `SMART_START`
  is not counted, because only provisioned DSKs can join in that mode.
  Every read is wrapped, so a failure degrades to a missing capability with
  one log line.
- **`zigbee`** (entity tier). The Zigbee2MQTT half matches the bridge's
  permit-join switch by `platform == "mqtt"` and the unique-id pattern, never
  by name. It provides `zigbee_permit_join`,
  `zigbee_permit_join_entity_ids` (the switches that are on), and
  `zigbee_permit_join_switches` (every bridge switch, for the engine's
  wake-up). The ZHA half provides nothing (see above).
- **Coordinator updates** (in `radio_registry`) produce
  `coordinator_update_pending`: `update.*` entities in state `on` whose
  device is one of:
  - the root device (`via_device_id is None`) of a `zha` or `zwave_js` entry;
  - the Z2M bridge device;
  - a device from a hardware-firmware platform;
  - an ESPHome Bluetooth proxy.

  Proxies are read from the Bluetooth integration's per-scanner config
  entries (see Review round changes).

### 3. Rules (in `sentinel/rules/`, all added to `NETWORK_RULE_TYPES`)

Every rule aggregates to one finding per cycle, carries an English `summary`,
and gets `en`/`cs` labels. This matches the existing family.

| rule_id | requires | severity | Notes |
|---|---|---|---|
| `radio_new_device_joined` | `network.radio.devices`, `network.radio.new_devices` | medium while away or at night, low otherwise; high if a new device is a security device while away | Alerts once per device, then held back until delivered. Evidence carries device ids, never addresses. |
| `zwave_insecure_security_class` | `network.radio.devices.security_class` | high when a lock, alarm, or entry cover is on `s0` or `none`; low for any other device on `s0`. Other devices on `none` are not reported, because most sensors legitimately run without security and a daily low alert for each would nag | Standing condition with the 24 h posture floor. Suggested action: re-include with S2. |
| `zigbee_permit_join_open` | `network.radio.posture.zigbee_permit_join` | medium, high while away | Event-triggered (below). |
| `zwave_inclusion_active` | `network.radio.posture.zwave_inclusion_active` | medium, high while away | New rule id; the parent plan lists the check but no rule. Poll-only, see limitations. |
| `radio_coordinator_update_pending` | `network.radio.posture.coordinator_update_pending` | medium | Per-entity exclusions honored. |

**Event trigger.** Z2M permit-join lasts at most 254 s, which a 5-minute
poll would rarely see. The engine remembers the last snapshot's
`zigbee_permit_join_entity_id` and treats a change of that entity to `on` as
a trigger. It matches on the id taken from the registry, not a name pattern.

### 4. Audit report and health sensor

- `_CAPABILITY_REASONS` gains radio entries. ZHA's reason names the version
  gap; Z-Wave's names the missing integration.
- The `PRIVACY_NOTES` sentence saying radio checks are not part of this
  version is replaced by the parent plan's statement that active radio
  attacks (key sniffing, S0 downgrade, jamming) are not detected.
- The report gains `inventory: {trusted, untrusted, by_source}`. Counts
  only, so the tool's YAML grows by one line and no labels.

### 5. Tests

- `test_snapshot_radio.py`:
  - registry classification per protocol, including the Z2M group
    exclusion, the Bluetooth adapter exclusion, and the `provision_` skip;
  - security-class normalization against **real
    `zwave_js_server.model.controller.Controller` / `Node` objects** built
    from minimal state;
  - inclusion state;
  - Z2M bridge switch recognized by unique id and a lookalike name ignored;
  - coordinator-update tagging;
  - every runtime-read failure degrading to a missing capability.
- `requirements/test.txt` adds `zwave-js-server-python==0.73.1`.
- `test_network_inventory.py`:
  - per-source bootstrap without alerts;
  - new device on an already-bootstrapped source;
  - hold-back;
  - row deletion;
  - restart persistence;
  - trust, untrust, and reset;
  - no identifiers other than registry ids in the persisted JSON.
- `test_rules_network.py`: each new rule, including severity branches and
  exclusions.
- The engine tests cover the permit-join trigger, commit ordering, and
  inventory commit failure not ending the loop.
- The notifier tests cover the Trust button, handler admin gating, and
  en/cs label parity.

### 6. Docs

README and `docs/sentinel.md` (rules table, inventory services, the ZHA gap),
the `services.yaml` entries, CHANGELOG, and the parent plan's status line.

## Validation

On the maintainer's box, which runs ZHA with 10 devices, a Bluetooth adapter,
no Z-Wave, and no Z2M:

1. Bootstrap: one notification with a Zigbee count of 11 (including the
   coordinator) and no findings.
2. `run_network_audit`: `radio_new_device_joined` ran; the Z-Wave rules and
   `zigbee_permit_join_open` are listed as not run, with reasons.
3. Pair a Zigbee device the home has never had, or remove one from ZHA, wait
   for a Sentinel run to drop its inventory row, and pair it again; then confirm a single
   `radio_new_device_joined` push with a working Trust button, and no repeat
   on the next cycle.

The Z-Wave and Z2M paths are fixture-tested only. The PR says so.

## Review round changes

A Claude `/code-review high` pass (5 findings) and a Codex adversarial pass
(10 findings) ran on the first implementation. 13 were confirmed and fixed:

- **No commit without the device list.** Z-Wave or Zigbee2MQTT posture alone
  still built a `radio` section with `devices: []`, so a failed registry read
  would have deleted every row and re-alerted every device. The engine now
  commits only when `network.radio.devices` is a capability.
- **Pending alerts instead of hold-back.** Holding a device out of the
  inventory until its alert was delivered meant a "Snooze Always" or a triage
  suppression kept it out forever: never trustable, never counted, a
  suppressed audit row and a triage call every poll. A new device is now
  recorded at once as untrusted with `alerted: false`; delivery, a snooze,
  triage, a policy block, or trusting the device settles the alert, and a
  cooldown, quiet hours, or presence grace only postpone it.
- **Bluetooth proxies from the scanner's config entry.** A remote scanner
  registers under its Bluetooth MAC, which is not the ESP32's network MAC, so
  the MAC join missed ESPHome proxies. Home Assistant creates a `bluetooth`
  config entry per remote scanner with `source_device_id`; that is read
  instead, and the `bluetooth` `after_dependencies` entry is no longer needed.
- **Registry API tolerance.** The collector accepts both the 2026.9 registry
  (iterable `devices`, `config_entry_id`, `async_get_device_by_identifier`)
  and the earlier mapping API with multi-entry `config_entries`, since the
  HACS minimum is 2025.5.0; a device is classified by any of its entries.
- **Present sources are loaded integrations**, and the Bluetooth adapter's
  entry makes Bluetooth present, so a first ZHA setup does not report the
  devices it creates as new and the first paired BLE device alerts.
- **Z-Wave is all-or-nothing** across controllers; one unreadable stick
  reports the checks as not run.
- **Security devices from the entity registry** (disabled locks count).
- **Permit-join wake-up checks the registry on the event**, so a bridge added
  since the last snapshot is covered.
- **Unavailable update entities are not observations.**
- **Trust and reset surface storage failures** instead of reporting success.

Not changed: an unbounded device count through a compromised MQTT broker (a
broker that can forge devices can already command real ones; summaries are
capped), and the read-only inventory service being open to non-admin users
(consistent with `get_audit_records` and `run_network_audit`; device names
are visible to every Home Assistant user). A new-device alert that lands in
the notifier's burst batch counts as delivered; the batch digest carries no
Trust button and is lost if Home Assistant stops within its 30-second flush
window, which the existing *burst-batch digest* TODO covers. The device stays
recorded and trustable through the services either way.

## Known limitations (stated in the docs, not hidden)

- ZHA permit-join is not observable on this Home Assistant version.
- `zwave_inclusion_active` is poll-only. The Z-Wave JS integration exposes no
  entity for inclusion state, and subscribing to driver events would couple
  HGA to the client's event API. Inclusion windows shorter than the
  detection interval will usually be missed.
- `radio_new_device_joined` sees devices once Home Assistant registers them.
  A device that joins the Zigbee mesh but fails the interview never reaches
  the registry.

## Decisions for the maintainer

1. **Alert once per new device vs. a standing "untrusted devices" finding.**
   Proposed: alert once, like `ha_new_admin_or_token`. Pairing is almost
   always the owner, and a daily reminder until Trust is tapped would nag.
   The trust flag then feeds the audit report's counts and step 7's
   auto-trust.
2. **Per-source bootstrap** instead of the parent plan's single bootstrap.
3. **`zwave_inclusion_active` ships** despite being poll-only. The
   alternative is dropping it until an event-driven read exists.

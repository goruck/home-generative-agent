"""
Radio section of the network snapshot (docs/network-security-radio-plan.md).

Three adapters over one set of inputs read from Home Assistant:

* ``radio_registry`` (entity tier, always runs): the Zigbee, Z-Wave,
  Bluetooth, and Matter devices in the device registry, whether each is a
  security device, the inventory delta (which devices are new), and pending
  firmware updates for coordinators, radio sticks, and Bluetooth proxies.
* ``zwave`` (runtime tier): each node's highest security class and whether
  the controller is including. Reads ``zwave_js`` config entries'
  ``runtime_data.client.driver.controller`` (Home Assistant 2026.9.1,
  zwave-js-server-python 0.73.1: ``Controller.home_id``,
  ``Controller.inclusion_state``, ``Controller.nodes``,
  ``Node.node_id``, ``Node.is_controller_node``,
  ``Node.highest_security_class``) without importing the integration.
* ``zigbee`` (entity tier): the Zigbee2MQTT bridge's permit-join switch,
  matched by entity-registry unique id. ZHA's permit-join state is not
  observable on Home Assistant 2026.9.1: zigpy 2.1.0's
  ``ControllerApplication.permit()`` broadcasts the request and keeps no
  record of it, so the capability is reported missing instead of guessed.

No IEEE addresses, node ids, or MAC addresses leave this module; devices are
identified by their device-registry id and labelled with sanitized names.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from homeassistant.config_entries import ConfigEntryState
from homeassistant.helpers import device_registry as dr
from homeassistant.helpers import entity_registry as er

from .network import (
    SECURITY_DOMAINS,
    AdapterResult,
    is_sensitive_entity,
    log_input_failure,
    radio_cap,
    sanitize_label,
)

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from homeassistant.core import HomeAssistant

    from custom_components.home_generative_agent.sentinel.network_inventory import (
        NetworkInventory,
    )

    from .schema import RadioDevice, SnapshotEntity

LOGGER = logging.getLogger(__name__)

# Config-entry domains whose devices are radio devices, by protocol.
_DOMAIN_PROTOCOL: dict[str, str] = {
    "zha": "zigbee",
    "zwave_js": "zwave",
    "matter": "matter",
}
# Integrations whose config entry means the protocol is present on this home
# even before it has any device (so a new stick bootstraps on its own).
PRESENT_SOURCE_DOMAINS: dict[str, str] = {
    "zha": "zigbee",
    "zwave_js": "zwave",
    "matter": "matter",
}
# Zigbee2MQTT publishes its devices through MQTT discovery with the
# identifier ``zigbee2mqtt_<ieee>`` and its bridge as
# ``zigbee2mqtt_bridge_<ieee>``; group identifiers carry the base topic and
# never match.
_Z2M_DEVICE_RE = re.compile(r"^zigbee2mqtt_(bridge_)?0x[0-9a-f]{16}$")
# The bridge's permit-join switch unique id:
# ``bridge_<coordinator ieee>_permit_join_<base topic>``.
_Z2M_PERMIT_JOIN_RE = re.compile(r"^bridge_0x[0-9a-f]{16}_permit_join_.+$")
# Integrations whose own ``update`` entities are radio-stick firmware.
RADIO_FIRMWARE_PLATFORMS: frozenset[str] = frozenset(
    {
        "homeassistant_sky_connect",
        "homeassistant_connect_zbt2",
        "homeassistant_yellow",
    }
)
# zwave_js_server.const.SecurityClass values -> normalized names.
_ZWAVE_SECURITY_CLASS: dict[int, str] = {
    -1: "none",
    0: "s2_unauth",
    1: "s2_auth",
    2: "s2_access",
    7: "s0",
}
# zwave_js_server.const.InclusionState.INCLUDING. SMART_START (4) is not
# counted: only devices whose DSK was provisioned can join in that mode.
_ZWAVE_INCLUDING = 1

# ---------------------------------------------------------------------------
# Inputs (read from Home Assistant, no I/O)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RadioDeviceInput:
    """One radio device from the device registry."""

    device_id: str
    protocol: str
    platform: str
    name: str | None
    manufacturer: str | None
    model: str | None


@dataclass(frozen=True)
class ZwaveInput:
    """What the Z-Wave JS controllers report."""

    # device registry id -> raw SecurityClass value (None: not yet known)
    security_classes: Mapping[str, int | None]
    inclusion_active: bool


@dataclass
class RadioInputs:
    """
    Everything the radio adapters read, collected once per snapshot.

    ``None`` means the read failed or the integration is absent; the matching
    capabilities are then left out.
    """

    devices: list[RadioDeviceInput] | None = None
    present_sources: frozenset[str] = frozenset()
    zwave: ZwaveInput | None = None
    z2m_permit_join_switches: list[str] | None = None
    zha_present: bool = False
    # Devices whose firmware is radio infrastructure: ZHA / Z-Wave JS
    # coordinators, Zigbee2MQTT bridges, and Bluetooth proxies.
    coordinator_device_ids: frozenset[str] = frozenset()
    entity_device: Mapping[str, str] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)


def _label(value: Any) -> str | None:
    return sanitize_label(value) or None


def _z2m_identifier(device: dr.DeviceEntry) -> tuple[bool, bool]:
    """Return (is a Zigbee2MQTT device, is the Zigbee2MQTT bridge)."""
    for domain, identifier in device.identifiers:
        if domain != "mqtt":
            continue
        match = _Z2M_DEVICE_RE.match(str(identifier))
        if match:
            return True, match.group(1) is not None
    return False, False


def _classify(
    device: dr.DeviceEntry, entry_domains: Mapping[str, str]
) -> tuple[str, str, bool] | None:
    """Return (protocol, platform, is coordinator-class) for a radio device."""
    domain = entry_domains.get(device.config_entry_id or "", "")
    if domain in _DOMAIN_PROTOCOL:
        if domain == "zwave_js" and any(
            d == "zwave_js" and str(i).startswith("provision_")
            for d, i in device.identifiers
        ):
            return None
        root = device.via_device_id is None and domain in {"zha", "zwave_js"}
        return _DOMAIN_PROTOCOL[domain], domain, root
    if domain == "mqtt":
        is_z2m, is_bridge = _z2m_identifier(device)
        if is_z2m:
            return "zigbee", "mqtt", is_bridge
        return None
    if domain != "bluetooth" and any(
        kind == dr.CONNECTION_BLUETOOTH for kind, _ in device.connections
    ):
        return "bluetooth", domain, False
    return None


def _collect_devices(
    hass: HomeAssistant, inputs: RadioInputs
) -> tuple[list[RadioDeviceInput], set[str]] | None:
    try:
        entry_domains = {
            entry.entry_id: entry.domain
            for entry in hass.config_entries.async_entries(include_ignore=False)
        }
        present = {
            PRESENT_SOURCE_DOMAINS[domain]
            for domain in entry_domains.values()
            if domain in PRESENT_SOURCE_DOMAINS
        }
        inputs.zha_present = "zha" in entry_domains.values()
        coordinators: set[str] = set()
        devices: list[RadioDeviceInput] = []
        for device in dr.async_get(hass).devices:
            classified = _classify(device, entry_domains)
            if classified is None:
                continue
            protocol, platform, coordinator = classified
            if coordinator:
                coordinators.add(device.id)
            devices.append(
                RadioDeviceInput(
                    device_id=device.id,
                    protocol=protocol,
                    platform=platform,
                    name=_label(device.name_by_user or device.name),
                    manufacturer=_label(device.manufacturer),
                    model=_label(device.model),
                )
            )
        devices.sort(key=lambda d: (d.protocol, d.name or "", d.device_id))
        inputs.present_sources = frozenset(present)
    except Exception as err:  # noqa: BLE001 - registry API moved in 2026.9
        log_input_failure("radio devices from the device registry", err)
        return None
    return devices, coordinators


def _collect_bluetooth_proxies(hass: HomeAssistant) -> set[str]:
    """Return device ids of remote Bluetooth scanners (ESPHome, Shelly proxies)."""
    if "bluetooth" not in hass.config.components:
        return set()
    try:
        from homeassistant.components.bluetooth import (  # noqa: PLC0415
            async_current_scanners,
        )

        sources = {
            str(scanner.source).lower() for scanner in async_current_scanners(hass)
        }
        registry = dr.async_get(hass)
        entry_domains = {
            entry.entry_id: entry.domain
            for entry in hass.config_entries.async_entries(include_ignore=False)
        }
        return {
            device.id
            for device in registry.devices
            if entry_domains.get(device.config_entry_id or "") != "bluetooth"
            and any(
                kind == dr.CONNECTION_NETWORK_MAC and value.lower() in sources
                for kind, value in device.connections
            )
        }
    except Exception as err:  # noqa: BLE001
        log_input_failure("Bluetooth scanners", err)
        return set()


def _collect_zwave(hass: HomeAssistant) -> ZwaveInput | None:
    """Read security classes and inclusion state from loaded Z-Wave JS entries."""
    try:
        entries = [
            entry
            for entry in hass.config_entries.async_entries("zwave_js")
            if entry.state is ConfigEntryState.LOADED
        ]
    except Exception as err:  # noqa: BLE001
        log_input_failure("Z-Wave JS config entries", err)
        return None
    if not entries:
        return None
    registry = dr.async_get(hass)
    classes: dict[str, int | None] = {}
    including = False
    readable = False
    for entry in entries:
        try:
            driver = entry.runtime_data.client.driver
            if driver is None:
                # Client not connected to the Z-Wave JS server yet.
                continue
            controller = driver.controller
            home_id = controller.home_id
            including = including or int(controller.inclusion_state) == _ZWAVE_INCLUDING
            for node in controller.nodes.values():
                if node.is_controller_node:
                    continue
                device = registry.async_get_device_by_identifier(
                    ("zwave_js", f"{home_id}-{node.node_id}"), entry.entry_id
                )
                if device is None:
                    continue
                security_class = node.highest_security_class
                classes[device.id] = (
                    None if security_class is None else int(security_class)
                )
            readable = True
        except Exception as err:  # noqa: BLE001 - runtime objects across versions
            log_input_failure("Z-Wave JS controller state", err)
    if not readable:
        return None
    return ZwaveInput(security_classes=classes, inclusion_active=including)


def _collect_z2m_permit_join(hass: HomeAssistant) -> list[str] | None:
    try:
        return sorted(
            entry.entity_id
            for entry in er.async_get(hass).entities.values()
            if entry.platform == "mqtt"
            and entry.domain == "switch"
            and _Z2M_PERMIT_JOIN_RE.match(str(entry.unique_id))
        )
    except Exception as err:  # noqa: BLE001
        log_input_failure("Zigbee2MQTT permit-join switches", err)
        return None


def collect_radio_inputs(
    hass: HomeAssistant, *, entity_device: Mapping[str, str]
) -> RadioInputs:
    """Read every radio input, each guarded independently."""
    inputs = RadioInputs(entity_device=entity_device)
    collected = _collect_devices(hass, inputs)
    if collected is not None:
        devices, coordinators = collected
        inputs.devices = devices
        inputs.coordinator_device_ids = frozenset(
            coordinators | _collect_bluetooth_proxies(hass)
        )
    inputs.zwave = _collect_zwave(hass)
    inputs.z2m_permit_join_switches = _collect_z2m_permit_join(hass)
    return inputs


# ---------------------------------------------------------------------------
# Pure adapters over the collected inputs
# ---------------------------------------------------------------------------


def _security_device_ids(
    entities: Sequence[SnapshotEntity], entity_device: Mapping[str, str]
) -> set[str]:
    ids: set[str] = set()
    for entity in entities:
        device_id = entity_device.get(entity["entity_id"])
        if device_id is None:
            continue
        if entity["domain"] in SECURITY_DOMAINS or is_sensitive_entity(entity):
            ids.add(device_id)
    return ids


def radio_registry_adapter(
    inputs: RadioInputs,
    entities: Sequence[SnapshotEntity],
    inventory: NetworkInventory | None,
) -> AdapterResult:
    """Radio devices, the inventory delta, and coordinator firmware updates."""
    result = AdapterResult(name="radio_registry")
    if inputs.devices is None:
        return result
    security_ids = _security_device_ids(entities, inputs.entity_device)
    classes = inputs.zwave.security_classes if inputs.zwave is not None else {}
    devices: list[RadioDevice] = []
    for device in inputs.devices:
        record: RadioDevice = {
            "device_id": device.device_id,
            "protocol": device.protocol,
            "platform": device.platform,
            "name": device.name,
            "is_security_device": device.device_id in security_ids,
            "manufacturer": device.manufacturer,
            "model": device.model,
        }
        raw_class = classes.get(device.device_id)
        if device.protocol == "zwave" and raw_class is not None:
            record["security_class"] = _ZWAVE_SECURITY_CLASS.get(raw_class, "unknown")
        devices.append(record)
    result.radio["devices"] = devices
    result.radio["present_sources"] = sorted(
        inputs.present_sources | {d["protocol"] for d in devices}
    )
    if inventory is not None:
        delta = inventory.diff(devices, inputs.present_sources)
        result.radio["new_devices"] = list(delta.new_device_keys)

    pending: list[str] = []
    observed = False
    for entity in entities:
        if entity["domain"] != "update":
            continue
        device_id = inputs.entity_device.get(entity["entity_id"])
        if not (
            (entity.get("platform") or "") in RADIO_FIRMWARE_PLATFORMS
            or (device_id is not None and device_id in inputs.coordinator_device_ids)
        ):
            continue
        observed = True
        if entity["state"] == "on":
            pending.append(entity["entity_id"])
    if observed:
        result.radio_posture["coordinator_update_pending"] = sorted(pending)
    return result


def zwave_adapter(inputs: RadioInputs) -> AdapterResult:
    """Z-Wave security classes (attached by the registry adapter) and inclusion."""
    result = AdapterResult(name="zwave")
    if inputs.zwave is None:
        return result
    result.extra_capabilities.append(radio_cap("devices.security_class"))
    result.radio_posture["zwave_inclusion_active"] = inputs.zwave.inclusion_active
    return result


def zigbee_adapter(
    inputs: RadioInputs, entities: Sequence[SnapshotEntity]
) -> AdapterResult:
    """Zigbee2MQTT permit-join; ZHA's is not observable and is named as such."""
    result = AdapterResult(name="zigbee")
    switches = inputs.z2m_permit_join_switches
    states = {e["entity_id"]: e["state"] for e in entities}
    known = [s for s in switches or [] if states.get(s) in {"on", "off"}]
    if known:
        result.radio_posture["zigbee_permit_join"] = any(
            states[s] == "on" for s in known
        )
        result.radio_posture["zigbee_permit_join_entity_ids"] = sorted(
            s for s in known if states[s] == "on"
        )
    if switches:
        # Every bridge switch, available or not, so the engine can wake on
        # the moment one turns on (a join window lasts at most 254 s).
        result.radio_posture["zigbee_permit_join_switches"] = list(switches)
    elif inputs.zha_present:
        result.notes.append(
            "ZHA does not expose whether it is accepting new devices on this "
            "Home Assistant version; Zigbee permit-join is not audited."
        )
    return result


def radio_adapters(
    inputs: RadioInputs,
    entities: Sequence[SnapshotEntity],
    inventory: NetworkInventory | None,
) -> list[AdapterResult]:
    """Run the three radio adapters in merge order."""
    registry = radio_registry_adapter(inputs, entities, inventory)
    registry.notes.extend(inputs.notes)
    return [registry, zwave_adapter(inputs), zigbee_adapter(inputs, entities)]

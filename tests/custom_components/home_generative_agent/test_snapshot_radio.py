# ruff: noqa: S101
"""Tests for the radio adapters (docs/network-security-radio-plan.md)."""

from __future__ import annotations

from datetime import UTC, datetime
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any
from unittest.mock import MagicMock

import pytest
from homeassistant.config_entries import ConfigEntryState
from homeassistant.helpers import device_registry as dr
from homeassistant.helpers import entity_registry as er
from pytest_homeassistant_custom_component.common import MockConfigEntry
from zwave_js_server.model.controller import Controller

from custom_components.home_generative_agent.sentinel.network_inventory import (
    NetworkInventory,
)
from custom_components.home_generative_agent.snapshot import network as network_mod
from custom_components.home_generative_agent.snapshot.network import (
    NetworkBuildContext,
    async_build_network_snapshot,
    merge_adapter_results,
    radio_cap,
)
from custom_components.home_generative_agent.snapshot.radio import (
    RadioDeviceInput,
    RadioInputs,
    ZwaveInput,
    _device_entry_ids,
    _registry_devices,
    collect_radio_inputs,
    is_z2m_permit_join_entry,
    radio_adapters,
    radio_registry_adapter,
    zigbee_adapter,
    zwave_adapter,
)

if TYPE_CHECKING:
    from homeassistant.core import HomeAssistant

    from custom_components.home_generative_agent.snapshot.schema import (
        SnapshotEntity,
    )

NOW = datetime(2026, 9, 13, 12, 0, tzinfo=UTC)
HOME_ID = 3245146787


@pytest.fixture(autouse=True)
def _reset_logged_failures() -> None:
    network_mod._LOGGED_INPUT_FAILURES.clear()


def _entity(
    entity_id: str,
    state: str = "on",
    *,
    platform: str | None = None,
    attributes: dict[str, Any] | None = None,
) -> SnapshotEntity:
    return {
        "entity_id": entity_id,
        "domain": entity_id.partition(".")[0],
        "state": state,
        "friendly_name": None,
        "area": None,
        "attributes": attributes or {},
        "last_changed": NOW.isoformat(),
        "last_updated": NOW.isoformat(),
        "platform": platform,
    }


def _entry(hass: HomeAssistant, domain: str, **kwargs: Any) -> MockConfigEntry:
    entry = MockConfigEntry(domain=domain, **kwargs)
    entry.add_to_hass(hass)
    return entry


def _controller(inclusion_state: int = 0) -> Controller:
    """Build a real zwave-js-server-python Controller from minimal state."""
    return Controller(
        MagicMock(),
        {
            "controller": {
                "homeId": HOME_ID,
                "ownNodeId": 1,
                "inclusionState": inclusion_state,
            },
            "nodes": [
                {
                    "nodeId": 1,
                    "status": 4,
                    "isControllerNode": True,
                    "values": [],
                    "endpoints": [],
                },
                {
                    "nodeId": 5,
                    "status": 4,
                    "isControllerNode": False,
                    "highestSecurityClass": 7,
                    "values": [],
                    "endpoints": [],
                },
                {
                    "nodeId": 6,
                    "status": 4,
                    "isControllerNode": False,
                    "highestSecurityClass": -1,
                    "values": [],
                    "endpoints": [],
                },
                {
                    "nodeId": 7,
                    "status": 4,
                    "isControllerNode": False,
                    "highestSecurityClass": 2,
                    "values": [],
                    "endpoints": [],
                },
                # Interview not finished: security class unknown.
                {
                    "nodeId": 8,
                    "status": 4,
                    "isControllerNode": False,
                    "values": [],
                    "endpoints": [],
                },
            ],
        },
    )


# ---------------------------------------------------------------------------
# Device registry classification (real registry)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_registry_classifies_each_protocol(hass: HomeAssistant) -> None:
    """ZHA, Z2M (not groups), Z-Wave (not provisioning), BLE, Matter; others out."""
    registry = dr.async_get(hass)
    zha = _entry(hass, "zha")
    mqtt = _entry(hass, "mqtt")
    zwave = _entry(hass, "zwave_js")
    matter = _entry(hass, "matter")
    switchbot = _entry(hass, "switchbot")
    bluetooth = _entry(hass, "bluetooth")
    shelly = _entry(hass, "shelly")
    for loaded in (zha, zwave, bluetooth):
        loaded.mock_state(hass, ConfigEntryState.LOADED)

    coordinator = registry.async_get_or_create(
        config_entry_id=zha.entry_id,
        identifiers={("zha", "00:12:4b:00:1c:a1:b8:01")},
        manufacturer="Texas Instruments",
        model="CC2652",
        name="Coordinator",
    )
    lock = registry.async_get_or_create(
        config_entry_id=zha.entry_id,
        identifiers={("zha", "00:0d:6f:00:0a:90:69:e7")},
        manufacturer="Schlage",
        model="BE468",
        name="Front Door Lock",
        via_device_id=coordinator.id,
    )
    z2m_bridge = registry.async_get_or_create(
        config_entry_id=mqtt.entry_id,
        identifiers={("mqtt", "zigbee2mqtt_bridge_0x00124b001ca1b801")},
        name="Zigbee2MQTT Bridge",
    )
    z2m_device = registry.async_get_or_create(
        config_entry_id=mqtt.entry_id,
        identifiers={("mqtt", "zigbee2mqtt_0x00158d0001a2b3c4")},
        name="Hallway Sensor",
    )
    registry.async_get_or_create(
        config_entry_id=mqtt.entry_id,
        identifiers={("mqtt", "zigbee2mqtt_zigbee2mqtt_5")},
        name="Z2M group",
    )
    registry.async_get_or_create(
        config_entry_id=mqtt.entry_id,
        identifiers={("mqtt", "tasmota_plug")},
        name="Tasmota plug",
    )
    zwave_node = registry.async_get_or_create(
        config_entry_id=zwave.entry_id,
        identifiers={("zwave_js", f"{HOME_ID}-5")},
        name="Garage Opener",
    )
    registry.async_get_or_create(
        config_entry_id=zwave.entry_id,
        identifiers={("zwave_js", "provision_abcdef")},
        name="Provisioned only",
    )
    matter_device = registry.async_get_or_create(
        config_entry_id=matter.entry_id,
        identifiers={("matter", "deviceid_1")},
        name="Matter Plug",
    )
    ble = registry.async_get_or_create(
        config_entry_id=switchbot.entry_id,
        connections={(dr.CONNECTION_BLUETOOTH, "AA:BB:CC:DD:EE:01")},
        name="Curtain",
    )
    registry.async_get_or_create(
        config_entry_id=bluetooth.entry_id,
        connections={(dr.CONNECTION_BLUETOOTH, "AA:BB:CC:DD:EE:99")},
        name="hci0",
    )
    registry.async_get_or_create(
        config_entry_id=shelly.entry_id,
        connections={(dr.CONNECTION_NETWORK_MAC, "aa:bb:cc:dd:ee:02")},
        name="Shelly Plug",
    )

    inputs = collect_radio_inputs(hass, entity_device={})
    assert inputs.devices is not None
    by_id = {d.device_id: d for d in inputs.devices}
    assert set(by_id) == {
        coordinator.id,
        lock.id,
        z2m_bridge.id,
        z2m_device.id,
        zwave_node.id,
        matter_device.id,
        ble.id,
    }
    assert by_id[lock.id].protocol == "zigbee"
    assert by_id[z2m_device.id].protocol == "zigbee"
    assert by_id[z2m_device.id].platform == "mqtt"
    assert by_id[zwave_node.id].protocol == "zwave"
    assert by_id[matter_device.id].protocol == "matter"
    assert by_id[ble.id].protocol == "bluetooth"
    assert by_id[ble.id].platform == "switchbot"
    # Coordinator-class: the ZHA root device and the Z2M bridge, not children.
    assert coordinator.id in inputs.coordinator_device_ids
    assert z2m_bridge.id in inputs.coordinator_device_ids
    assert lock.id not in inputs.coordinator_device_ids
    # Only integrations that finished setting up count (matter is not loaded).
    assert inputs.present_sources == {"zigbee", "zwave", "bluetooth"}
    assert inputs.zha_present is True
    # Unload without importing the real integrations at teardown.
    for loaded in (zha, zwave, bluetooth):
        loaded.mock_state(hass, ConfigEntryState.NOT_LOADED)


@pytest.mark.asyncio
async def test_ignored_config_entry_does_not_mark_a_source_present(
    hass: HomeAssistant,
) -> None:
    """An ignored ZHA discovery is not a Zigbee network."""
    _entry(hass, "zha", source="ignore")
    inputs = collect_radio_inputs(hass, entity_device={})
    assert inputs.present_sources == frozenset()
    assert inputs.zha_present is False


@pytest.mark.asyncio
async def test_registry_read_failure_is_a_missing_capability(
    hass: HomeAssistant, monkeypatch: pytest.MonkeyPatch
) -> None:
    def _boom(_hass: Any) -> Any:
        msg = "moved"
        raise AttributeError(msg)

    monkeypatch.setattr(
        "custom_components.home_generative_agent.snapshot.radio.dr.async_get", _boom
    )
    inputs = collect_radio_inputs(hass, entity_device={})
    assert inputs.devices is None
    section = merge_adapter_results(radio_adapters(inputs, [], None))
    assert radio_cap("devices") not in section["capabilities"]


# ---------------------------------------------------------------------------
# Z-Wave runtime adapter (real zwave-js-server-python objects)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_zwave_reads_security_classes_from_real_controller(
    hass: HomeAssistant,
) -> None:
    registry = dr.async_get(hass)
    entry = _entry(hass, "zwave_js")
    entry.mock_state(hass, ConfigEntryState.LOADED)
    entry.runtime_data = SimpleNamespace(
        client=SimpleNamespace(driver=SimpleNamespace(controller=_controller()))
    )
    devices = {
        node: registry.async_get_or_create(
            config_entry_id=entry.entry_id,
            identifiers={("zwave_js", f"{HOME_ID}-{node}")},
            name=f"Node {node}",
        )
        for node in (1, 5, 6, 7, 8)
    }
    lock_entity = _entity("lock.node_5")
    inputs = collect_radio_inputs(hass, entity_device={"lock.node_5": devices[5].id})
    assert inputs.zwave is not None
    assert inputs.zwave.inclusion_active is False
    classes = inputs.zwave.security_classes
    assert devices[1].id not in classes  # controller node skipped
    assert classes[devices[5].id] == 7
    assert classes[devices[8].id] is None

    section = merge_adapter_results(radio_adapters(inputs, [lock_entity], None))
    radio = section.get("radio")
    assert radio is not None
    by_id = {d["device_id"]: d for d in radio["devices"]}
    assert by_id[devices[5].id].get("security_class") == "s0"
    assert by_id[devices[5].id]["is_security_device"] is True
    assert by_id[devices[6].id].get("security_class") == "none"
    assert by_id[devices[7].id].get("security_class") == "s2_access"
    assert "security_class" not in by_id[devices[8].id]
    assert radio_cap("devices.security_class") in section["capabilities"]
    assert radio["posture"].get("zwave_inclusion_active") is False
    assert section["sources"][radio_cap("devices.security_class")] == "zwave"


@pytest.mark.asyncio
async def test_zwave_inclusion_state_including_only(hass: HomeAssistant) -> None:
    """INCLUDING counts; SmartStart listening does not."""
    entry = _entry(hass, "zwave_js")
    entry.mock_state(hass, ConfigEntryState.LOADED)
    for state, expected in ((1, True), (4, False), (2, False)):
        entry.runtime_data = SimpleNamespace(
            client=SimpleNamespace(
                driver=SimpleNamespace(controller=_controller(inclusion_state=state))
            )
        )
        inputs = collect_radio_inputs(hass, entity_device={})
        assert inputs.zwave is not None
        assert inputs.zwave.inclusion_active is expected, state


@pytest.mark.asyncio
async def test_zwave_not_loaded_or_disconnected_is_missing(
    hass: HomeAssistant, caplog: pytest.LogCaptureFixture
) -> None:
    entry = _entry(hass, "zwave_js")
    # Not loaded: nothing read.
    assert collect_radio_inputs(hass, entity_device={}).zwave is None
    entry.mock_state(hass, ConfigEntryState.LOADED)
    entry.runtime_data = SimpleNamespace(client=SimpleNamespace(driver=None))
    assert collect_radio_inputs(hass, entity_device={}).zwave is None
    # A runtime object without the attribute degrades with one log line.
    entry.runtime_data = SimpleNamespace(client=SimpleNamespace())
    assert collect_radio_inputs(hass, entity_device={}).zwave is None
    assert "Z-Wave JS controller state" in caplog.text
    section = merge_adapter_results(
        radio_adapters(collect_radio_inputs(hass, entity_device={}), [], None)
    )
    assert radio_cap("devices.security_class") not in section["capabilities"]
    assert radio_cap("posture.zwave_inclusion_active") not in section["capabilities"]


# ---------------------------------------------------------------------------
# Zigbee adapter
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_z2m_permit_join_matched_by_unique_id(hass: HomeAssistant) -> None:
    entities = er.async_get(hass)
    mqtt = _entry(hass, "mqtt")
    bridge = entities.async_get_or_create(
        "switch",
        "mqtt",
        "bridge_0x00124b001ca1b801_permit_join_zigbee2mqtt",
        config_entry=mqtt,
        suggested_object_id="zigbee2mqtt_bridge_permit_join",
    )
    # A lookalike by name only is ignored.
    entities.async_get_or_create(
        "switch",
        "mqtt",
        "my_custom_permit_join",
        config_entry=mqtt,
        suggested_object_id="permit_join",
    )
    inputs = collect_radio_inputs(hass, entity_device={})
    assert inputs.z2m_permit_join_switches == [bridge.entity_id]

    closed = zigbee_adapter(inputs, [_entity(bridge.entity_id, "off")])
    assert closed.radio_posture["zigbee_permit_join"] is False
    assert is_z2m_permit_join_entry(entities.async_get(bridge.entity_id))
    assert not is_z2m_permit_join_entry(entities.async_get("switch.permit_join"))
    assert not is_z2m_permit_join_entry(None)
    opened = zigbee_adapter(inputs, [_entity(bridge.entity_id, "on")])
    assert opened.radio_posture["zigbee_permit_join"] is True
    assert opened.radio_posture["zigbee_permit_join_entity_ids"] == [bridge.entity_id]
    # Bridge offline: the fact is unknown, not "closed".
    offline = zigbee_adapter(inputs, [_entity(bridge.entity_id, "unavailable")])
    assert "zigbee_permit_join" not in offline.radio_posture


def test_zha_permit_join_is_named_not_guessed() -> None:
    inputs = RadioInputs(devices=[], z2m_permit_join_switches=[], zha_present=True)
    result = zigbee_adapter(inputs, [])
    assert result.radio_posture == {}
    assert any("ZHA" in note for note in result.notes)
    no_zigbee = zigbee_adapter(RadioInputs(z2m_permit_join_switches=[]), [])
    assert no_zigbee.notes == []


# ---------------------------------------------------------------------------
# Registry adapter: security devices, coordinator updates, inventory delta
# ---------------------------------------------------------------------------


def _inputs(**kwargs: Any) -> RadioInputs:
    devices = kwargs.pop(
        "devices",
        [
            RadioDeviceInput("coord", "zigbee", "zha", "Coordinator", None, None),
            RadioDeviceInput(
                "lock1", "zigbee", "zha", "Front Lock", "Schlage", "BE468"
            ),
            RadioDeviceInput("bulb1", "zigbee", "zha", "Bulb", None, None),
        ],
    )
    return RadioInputs(devices=devices, **kwargs)


def test_security_devices_from_owned_entities() -> None:
    inputs = _inputs(
        entity_device={
            "lock.front": "lock1",
            "light.bulb": "bulb1",
        }
    )
    result = radio_registry_adapter(
        inputs, [_entity("lock.front", "locked"), _entity("light.bulb")], None
    )
    flags = {d["device_id"]: d["is_security_device"] for d in result.radio["devices"]}
    assert flags == {"coord": False, "lock1": True, "bulb1": False}
    # No inventory: no delta, so the new-device capability is absent.
    assert "new_devices" not in result.radio


def test_coordinator_updates_by_device_and_firmware_platform() -> None:
    inputs = _inputs(
        coordinator_device_ids=frozenset({"coord", "proxy"}),
        entity_device={
            "update.coordinator": "coord",
            "update.bt_proxy": "proxy",
            "update.lock_firmware": "lock1",
        },
    )
    entities = [
        _entity("update.coordinator", "off"),
        _entity("update.bt_proxy", "on"),
        _entity("update.lock_firmware", "on"),
        _entity("update.zbt_1_firmware", "on", platform="homeassistant_sky_connect"),
    ]
    result = radio_registry_adapter(inputs, entities, None)
    assert result.radio_posture["coordinator_update_pending"] == [
        "update.bt_proxy",
        "update.zbt_1_firmware",
    ]
    # No coordinator-class update entity at all: capability absent.
    bare = radio_registry_adapter(
        _inputs(entity_device={"update.lock_firmware": "lock1"}),
        [_entity("update.lock_firmware", "on")],
        None,
    )
    assert "coordinator_update_pending" not in bare.radio_posture


@pytest.mark.asyncio
async def test_inventory_delta_lands_in_section(hass: HomeAssistant) -> None:
    inventory = NetworkInventory(hass)
    inputs = _inputs(present_sources=frozenset({"zigbee"}))
    first = merge_adapter_results(radio_adapters(inputs, [], inventory))
    radio = first.get("radio")
    assert radio is not None
    # Unbootstrapped source: nothing is new yet.
    assert radio.get("new_devices") == []
    assert radio_cap("new_devices") in first["capabilities"]
    assert radio_cap("present_sources") not in first["capabilities"]
    await inventory.async_commit(
        radio["devices"], NOW, present_sources=radio.get("present_sources", [])
    )
    grown = _inputs(present_sources=frozenset({"zigbee"}))
    assert grown.devices is not None
    grown.devices.append(RadioDeviceInput("new1", "zigbee", "zha", "New", None, None))
    second = merge_adapter_results(radio_adapters(grown, [], inventory))
    assert second["radio"]["new_devices"] == ["zigbee:new1"]  # type: ignore[typeddict-item]


@pytest.mark.asyncio
async def test_build_network_snapshot_runs_radio_adapters(hass: HomeAssistant) -> None:
    """The full build validates with a radio section on a plain test instance."""
    from custom_components.home_generative_agent.snapshot.schema import (  # noqa: PLC0415
        validate_snapshot,
    )

    zha = _entry(hass, "zha")
    dr.async_get(hass).async_get_or_create(
        config_entry_id=zha.entry_id,
        identifiers={("zha", "00:12:4b:00:1c:a1:b8:01")},
        name="Coordinator",
    )
    section = await async_build_network_snapshot(
        hass,
        [],
        NetworkBuildContext(options={}, network_inventory=NetworkInventory(hass)),
        now=NOW,
        entity_device={},
        device_domains={},
    )
    assert radio_cap("devices") in section["capabilities"]
    assert radio_cap("posture.zigbee_permit_join") not in section["capabilities"]
    assert any("ZHA" in note for note in section.get("notes", []))
    validate_snapshot(
        {
            "schema_version": 2,
            "generated_at": NOW.isoformat(),
            "entities": [],
            "camera_activity": [],
            "derived": {
                "now": NOW.isoformat(),
                "timezone": "UTC",
                "is_night": False,
                "anyone_home": True,
                "people_home": [],
                "people_away": [],
                "last_motion_by_area": {},
            },
            "network": section,
        }
    )


def test_zwave_adapter_without_input_asserts_nothing() -> None:
    assert zwave_adapter(RadioInputs()).radio_posture == {}
    result = zwave_adapter(
        RadioInputs(zwave=ZwaveInput(security_classes={}, inclusion_active=True))
    )
    assert result.radio_posture == {"zwave_inclusion_active": True}


@pytest.mark.asyncio
async def test_bluetooth_proxies_come_from_remote_scanner_entries(
    hass: HomeAssistant,
) -> None:
    """A remote scanner's bluetooth entry names its host device (ESPHome, Shelly)."""
    registry = dr.async_get(hass)
    esphome = _entry(hass, "esphome")
    proxy = registry.async_get_or_create(
        config_entry_id=esphome.entry_id,
        connections={(dr.CONNECTION_NETWORK_MAC, "aa:bb:cc:00:11:22")},
        name="Living Room Proxy",
    )
    plain = registry.async_get_or_create(
        config_entry_id=esphome.entry_id,
        connections={(dr.CONNECTION_NETWORK_MAC, "aa:bb:cc:00:11:33")},
        name="Garage Sensor",
    )
    # The scanner address is the Bluetooth MAC, not the network MAC above.
    _entry(
        hass,
        "bluetooth",
        data={"source": "AA:BB:CC:00:11:24", "source_device_id": proxy.id},
    )
    _entry(hass, "bluetooth", data={})  # the local adapter
    inputs = collect_radio_inputs(hass, entity_device={})
    assert proxy.id in inputs.coordinator_device_ids
    assert plain.id not in inputs.coordinator_device_ids


def test_registry_helpers_accept_the_pre_2026_9_api() -> None:
    """A mapping of devices and a config_entries set both work."""
    old_device = SimpleNamespace(id="d1", config_entries={"e1", "e2"})
    old_registry = SimpleNamespace(devices={"d1": old_device})
    assert _registry_devices(old_registry) == [old_device]  # type: ignore[arg-type]
    assert _device_entry_ids(old_device) == {"e1", "e2"}  # type: ignore[arg-type]
    new_device = SimpleNamespace(id="d2", config_entry_id="e3")
    assert _device_entry_ids(new_device) == {"e3"}  # type: ignore[arg-type]


@pytest.mark.asyncio
async def test_zwave_one_unreadable_controller_asserts_nothing(
    hass: HomeAssistant,
) -> None:
    """Two sticks, one disconnected: the checks are not run, not half-passed."""
    good = _entry(hass, "zwave_js")
    good.mock_state(hass, ConfigEntryState.LOADED)
    good.runtime_data = SimpleNamespace(
        client=SimpleNamespace(driver=SimpleNamespace(controller=_controller()))
    )
    assert collect_radio_inputs(hass, entity_device={}).zwave is not None
    bad = _entry(hass, "zwave_js")
    bad.mock_state(hass, ConfigEntryState.LOADED)
    bad.runtime_data = SimpleNamespace(client=SimpleNamespace(driver=None))
    assert collect_radio_inputs(hass, entity_device={}).zwave is None


def test_security_devices_include_registry_domains_without_state() -> None:
    """A lock whose entity is disabled (no state) still makes a security device."""
    inputs = _inputs(device_domains={"lock1": {"lock", "sensor"}, "bulb1": {"light"}})
    result = radio_registry_adapter(inputs, [], None)
    flags = {d["device_id"]: d["is_security_device"] for d in result.radio["devices"]}
    assert flags == {"coord": False, "lock1": True, "bulb1": False}


def test_unavailable_coordinator_update_is_not_an_observation() -> None:
    inputs = _inputs(
        coordinator_device_ids=frozenset({"coord"}),
        entity_device={"update.coordinator": "coord"},
    )
    result = radio_registry_adapter(
        inputs, [_entity("update.coordinator", "unavailable")], None
    )
    assert "coordinator_update_pending" not in result.radio_posture

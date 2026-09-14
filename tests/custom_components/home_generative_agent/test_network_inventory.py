# ruff: noqa: S101
"""Tests for the radio device inventory store."""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, Any
from unittest.mock import AsyncMock, patch

import pytest
from homeassistant.exceptions import HomeAssistantError

from custom_components.home_generative_agent.sentinel.network_inventory import (
    STORE_KEY,
    BootstrapSummary,
    NetworkInventory,
    device_key,
)

if TYPE_CHECKING:
    from homeassistant.core import HomeAssistant

    from custom_components.home_generative_agent.snapshot.schema import RadioDevice

NOW = datetime(2026, 9, 13, 12, 0, tzinfo=UTC)


def _device(
    device_id: str, protocol: str = "zigbee", name: str = "Device"
) -> RadioDevice:
    return {
        "device_id": device_id,
        "protocol": protocol,
        "platform": "zha" if protocol == "zigbee" else "zwave_js",
        "name": name,
        "is_security_device": False,
        "manufacturer": "Acme",
        "model": "X1",
    }


@pytest.mark.asyncio
async def test_first_commit_bootstraps_each_source_without_new_devices(
    hass: HomeAssistant,
) -> None:
    inventory = NetworkInventory(hass)
    devices = [_device("a"), _device("b"), _device("z1", "zwave")]
    delta = inventory.diff(devices, [])
    assert delta.new_device_keys == []
    assert delta.bootstrap_sources == ["zigbee", "zwave"]
    announced = await inventory.async_commit(devices, NOW)
    assert announced == [
        BootstrapSummary(source="zigbee", device_count=2),
        BootstrapSummary(source="zwave", device_count=1),
    ]
    summary = inventory.summary()
    assert summary["trusted"] == 3
    assert summary["untrusted"] == 0
    assert inventory.diff(devices, []).new_device_keys == []


@pytest.mark.asyncio
async def test_new_device_on_bootstrapped_source_is_new_and_untrusted(
    hass: HomeAssistant,
) -> None:
    inventory = NetworkInventory(hass)
    await inventory.async_commit([_device("a")], NOW)
    devices = [_device("a"), _device("b")]
    assert inventory.diff(devices, []).new_device_keys == ["zigbee:b"]
    assert await inventory.async_commit(devices, NOW) == []
    rows = {r["key"]: r for r in inventory.list_devices()}
    assert rows["zigbee:b"]["trusted"] is False
    assert rows["zigbee:a"]["trusted"] is True
    # Recorded, but its alert is still owed until the engine settles it.
    assert inventory.diff(devices, []).new_device_keys == ["zigbee:b"]


@pytest.mark.asyncio
async def test_new_source_bootstraps_silently_later(hass: HomeAssistant) -> None:
    """A Z-Wave stick added after Zigbee was recorded does not alert."""
    inventory = NetworkInventory(hass)
    await inventory.async_commit([_device("a")], NOW)
    # The integration exists with no devices yet: present, bootstrap it.
    assert inventory.diff([_device("a")], ["zwave"]).bootstrap_sources == ["zwave"]
    announced = await inventory.async_commit(
        [_device("a")], NOW, present_sources=["zwave"]
    )
    assert announced == [BootstrapSummary(source="zwave", device_count=0)]
    later = [_device("a"), _device("z1", "zwave")]
    assert inventory.diff(later, ["zwave"]).new_device_keys == ["zwave:z1"]


@pytest.mark.asyncio
async def test_unsettled_alert_stays_new_but_device_is_recorded(
    hass: HomeAssistant,
) -> None:
    """A postponed alert is owed again; the device is trustable meanwhile."""
    inventory = NetworkInventory(hass)
    await inventory.async_commit([_device("a")], NOW)
    devices = [_device("a"), _device("b")]
    await inventory.async_commit(devices, NOW)
    assert inventory.diff(devices, []).new_device_keys == ["zigbee:b"]
    assert inventory.summary()["untrusted"] == 1
    await inventory.async_commit(devices, NOW, alerted=["zigbee:b"])
    assert inventory.diff(devices, []).new_device_keys == []


@pytest.mark.asyncio
async def test_trusting_settles_a_pending_alert(hass: HomeAssistant) -> None:
    inventory = NetworkInventory(hass)
    await inventory.async_commit([_device("a")], NOW)
    devices = [_device("a"), _device("b")]
    await inventory.async_commit(devices, NOW)
    assert await inventory.async_set_trusted(["b"], trusted=True) == ["zigbee:b"]
    assert inventory.diff(devices, []).new_device_keys == []


@pytest.mark.asyncio
async def test_trust_save_failure_raises_and_reset_reports_removal_failure(
    hass: HomeAssistant,
) -> None:
    inventory = NetworkInventory(hass)
    await inventory.async_commit([_device("a")], NOW)
    with (
        patch.object(inventory, "async_save", AsyncMock(return_value=False)),
        pytest.raises(HomeAssistantError),
    ):
        await inventory.async_set_trusted(["a"], trusted=False)
    with patch.object(inventory._store, "async_remove", AsyncMock(side_effect=OSError)):
        assert await inventory.async_reset() is False
    assert await inventory.async_reset() is True


@pytest.mark.asyncio
async def test_row_deleted_when_device_leaves_registry(hass: HomeAssistant) -> None:
    inventory = NetworkInventory(hass)
    await inventory.async_commit([_device("a"), _device("b")], NOW)
    await inventory.async_commit([_device("a")], NOW)
    assert [r["key"] for r in inventory.list_devices()] == ["zigbee:a"]
    # A device whose row was dropped (a run saw it gone) is new when it returns.
    assert inventory.diff([_device("a"), _device("b")], []).new_device_keys == [
        "zigbee:b"
    ]


@pytest.mark.asyncio
async def test_persists_across_restart_and_writes_only_on_change(
    hass: HomeAssistant,
) -> None:
    inventory = NetworkInventory(hass)
    await inventory.async_commit([_device("a")], NOW)
    reloaded = NetworkInventory(hass)
    await reloaded.async_load()
    assert reloaded.is_source_bootstrapped("zigbee")
    assert reloaded.diff([_device("a"), _device("b")], []).new_device_keys == [
        "zigbee:b"
    ]
    with patch.object(reloaded, "async_save", AsyncMock(return_value=True)) as save:
        await reloaded.async_commit([_device("a")], NOW + timedelta(hours=1))
        save.assert_not_called()
        await reloaded.async_commit([_device("a")], NOW + timedelta(days=2))
        save.assert_called_once()
        await reloaded.async_commit(
            [_device("a", name="Renamed")], NOW + timedelta(days=2)
        )
        assert save.call_count == 2


@pytest.mark.asyncio
async def test_failed_save_retries_and_defers_announcement(
    hass: HomeAssistant,
) -> None:
    inventory = NetworkInventory(hass)
    with patch.object(inventory, "async_save", AsyncMock(return_value=False)):
        assert await inventory.async_commit([_device("a")], NOW) == []
    with patch.object(inventory, "async_save", AsyncMock(return_value=True)) as save:
        announced = await inventory.async_commit([_device("a")], NOW)
        save.assert_called_once()
    assert announced == [BootstrapSummary(source="zigbee", device_count=1)]


@pytest.mark.asyncio
async def test_trust_untrust_by_device_id_or_key(hass: HomeAssistant) -> None:
    inventory = NetworkInventory(hass)
    await inventory.async_commit([_device("a")], NOW)
    await inventory.async_commit([_device("a"), _device("b"), _device("c")], NOW)
    assert await inventory.async_set_trusted(["b", "zigbee:c"], trusted=True) == [
        "zigbee:b",
        "zigbee:c",
    ]
    assert await inventory.async_set_trusted(["b"], trusted=True) == []
    assert await inventory.async_set_trusted(["a"], trusted=False) == ["zigbee:a"]
    assert await inventory.async_set_trusted(["unknown"], trusted=True) == []
    summary = inventory.summary()
    assert summary["by_source"] == {"zigbee": {"trusted": 2, "untrusted": 1}}


@pytest.mark.asyncio
async def test_reset_forgets_everything(hass: HomeAssistant) -> None:
    inventory = NetworkInventory(hass)
    await inventory.async_commit([_device("a")], NOW)
    await inventory.async_reset()
    assert not inventory.is_source_bootstrapped("zigbee")
    reloaded = NetworkInventory(hass)
    await reloaded.async_load()
    assert reloaded.summary()["device_count"] == 0


@pytest.mark.asyncio
async def test_persisted_json_holds_no_radio_addresses(
    hass: HomeAssistant, hass_storage: dict[str, Any]
) -> None:
    inventory = NetworkInventory(hass)
    await inventory.async_commit([_device("dev123")], NOW)
    data = hass_storage[STORE_KEY]["data"]
    assert set(data) == {"sources", "devices"}
    (row,) = data["devices"].values()
    # Exactly the documented, address-free fields.
    assert set(row) == {
        "alerted",
        "source",
        "platform",
        "ha_device_id",
        "name",
        "manufacturer",
        "model",
        "first_seen",
        "last_seen",
        "trusted",
    }
    assert "ieee" not in json.dumps(data).lower()


@pytest.mark.asyncio
async def test_load_tolerates_corrupt_rows(
    hass: HomeAssistant, hass_storage: dict[str, Any]
) -> None:
    hass_storage[STORE_KEY] = {
        "version": 1,
        "key": STORE_KEY,
        "data": {
            "sources": {"zigbee": NOW.isoformat(), "bad": 5},
            "devices": {"zigbee:a": {"source": "zigbee"}, "broken": "x", "n": {}},
        },
    }
    inventory = NetworkInventory(hass)
    await inventory.async_load()
    assert inventory.is_source_bootstrapped("zigbee")
    assert not inventory.is_source_bootstrapped("bad")
    assert [r["key"] for r in inventory.list_devices()] == ["zigbee:a"]


@pytest.mark.asyncio
async def test_unreadable_store_starts_empty(hass: HomeAssistant) -> None:
    inventory = NetworkInventory(hass)
    with patch.object(
        inventory._store, "async_load", AsyncMock(side_effect=HomeAssistantError)
    ):
        await inventory.async_load()
    assert inventory.summary()["device_count"] == 0


def test_device_key_shape() -> None:
    assert device_key(_device("abc", "zwave")) == "zwave:abc"

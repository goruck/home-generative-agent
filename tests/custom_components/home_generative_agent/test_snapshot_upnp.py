# ruff: noqa: S101
"""Tests for the UPnP/IGD adapter (docs/network-security-plan.md, step 7)."""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

import pytest
from homeassistant.components.ssdp.const import DOMAIN as SSDP_DOMAIN
from homeassistant.components.ssdp.const import SSDP_SCANNER
from homeassistant.config_entries import SOURCE_IGNORE, ConfigEntryState
from homeassistant.helpers import entity_registry as er
from homeassistant.helpers.service_info.ssdp import SsdpServiceInfo
from pytest_homeassistant_custom_component.common import MockConfigEntry

from custom_components.home_generative_agent.sentinel.pseudonymizer import (
    Pseudonymizer,
)
from custom_components.home_generative_agent.snapshot import network as network_mod
from custom_components.home_generative_agent.snapshot.network import (
    POSTURE_DISPLAY_KEYS,
    NetworkBuildContext,
    async_build_network_snapshot,
    merge_adapter_results,
    posture_cap,
)
from custom_components.home_generative_agent.snapshot.upnp import (
    IGD_SEARCH_TARGETS,
    POSTURE_MEMORY_KEYS,
    UpnpInputs,
    async_collect_upnp_inputs,
    port_mapping_count,
    public_ip_value,
    upnp_igd_adapter,
)

if TYPE_CHECKING:
    from homeassistant.core import HomeAssistant

    from custom_components.home_generative_agent.snapshot.schema import (
        SnapshotEntity,
    )

NOW = datetime(2026, 9, 16, 12, 0, tzinfo=UTC)
PSEUDONYMIZER = Pseudonymizer("test-salt")


@pytest.fixture(autouse=True)
def _reset_logged_failures() -> None:
    network_mod._LOGGED_INPUT_FAILURES.clear()


def _entity(
    entity_id: str, state: str, platform: str | None = "upnp"
) -> SnapshotEntity:
    return {
        "entity_id": entity_id,
        "domain": entity_id.partition(".")[0],
        "state": state,
        "friendly_name": None,
        "area": None,
        "attributes": {},
        "last_changed": NOW.isoformat(),
        "last_updated": NOW.isoformat(),
        "platform": platform,
    }


def _context(
    previous: dict[str, Any] | None = None, *, pseudonymizer: Any = PSEUDONYMIZER
) -> NetworkBuildContext:
    return NetworkBuildContext(
        enabled=True, pseudonymizer=pseudonymizer, previous_posture=previous
    )


def _igd(usn: str, st: str, name: str | None = "eero") -> SsdpServiceInfo:
    return SsdpServiceInfo(
        ssdp_usn=usn,
        ssdp_st=st,
        upnp={"friendlyName": name} if name is not None else {},
        ssdp_udn=usn.partition("::")[0],
    )


class _Scanner:
    """Stands in for the ssdp component's Scanner behind its public helper."""

    def __init__(self, by_st: dict[str, list[SsdpServiceInfo]]) -> None:
        self.by_st = by_st

    async def async_get_discovery_info_by_st(self, st: str) -> list[SsdpServiceInfo]:
        return list(self.by_st.get(st, []))


class _BrokenScanner:
    async def async_get_discovery_info_by_st(self, st: str) -> list[SsdpServiceInfo]:
        msg = "scanner moved"
        raise AttributeError(msg)


class _HangingScanner:
    async def async_get_discovery_info_by_st(self, st: str) -> list[SsdpServiceInfo]:
        await asyncio.Event().wait()
        return []


# ---------------------------------------------------------------------------
# Collectors
# ---------------------------------------------------------------------------


def _ssdp_loaded(hass: HomeAssistant, scanner: Any) -> None:
    hass.config.components.add(SSDP_DOMAIN)
    hass.data[SSDP_DOMAIN] = {SSDP_SCANNER: scanner}


@pytest.mark.asyncio
async def test_collect_ssdp_reads_the_igd_cache_through_the_public_helper(
    hass: HomeAssistant,
) -> None:
    """Both IGD search targets are read; one gateway on both counts once."""
    v1, v2 = IGD_SEARCH_TARGETS
    _ssdp_loaded(
        hass,
        _Scanner(
            {
                v1: [_igd("uuid:aaa::" + v1, v1, "eero\u202e evil")],
                v2: [
                    _igd("uuid:aaa::" + v1, v2, "eero"),
                    _igd("uuid:bbb::" + v2, v2, None),
                ],
            }
        ),
    )
    inputs = await async_collect_upnp_inputs(hass)
    assert inputs.igd_advertised is True
    # Bidi override dropped; the nameless gateway gets a placeholder.
    assert inputs.igd_names == ["eero evil", "unnamed gateway"]


@pytest.mark.asyncio
async def test_collect_ssdp_absent_failing_or_hanging_is_not_evidence(
    hass: HomeAssistant,
    caplog: pytest.LogCaptureFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Not loaded: unknown silently. Loaded but unreadable or hanging: logged once."""
    if SSDP_DOMAIN in hass.config.components:
        hass.config.components.remove(SSDP_DOMAIN)
    hass.data.pop(SSDP_DOMAIN, None)
    inputs = await async_collect_upnp_inputs(hass)
    assert inputs.igd_advertised is None
    _ssdp_loaded(hass, _Scanner({}))
    inputs = await async_collect_upnp_inputs(hass)
    assert inputs.igd_advertised is False
    # Loaded, but the scanner is not where the helper looks (key drift).
    hass.data.pop(SSDP_DOMAIN)
    with caplog.at_level("WARNING"):
        inputs = await async_collect_upnp_inputs(hass)
    assert inputs.igd_advertised is None
    _ssdp_loaded(hass, _BrokenScanner())
    with caplog.at_level("WARNING"):
        inputs = await async_collect_upnp_inputs(hass)
        await async_collect_upnp_inputs(hass)
    assert inputs.igd_advertised is None
    warnings = [
        r.getMessage()
        for r in caplog.records
        if "SSDP discovery cache" in r.getMessage()
    ]
    assert len(warnings) == 2  # KeyError once, AttributeError once
    # A description fetch that never completes is bounded, never a stuck run.
    monkeypatch.setattr(
        "custom_components.home_generative_agent.snapshot.upnp.SSDP_READ_TIMEOUT_S",
        0.01,
    )
    _ssdp_loaded(hass, _HangingScanner())
    with caplog.at_level("WARNING"):
        inputs = await async_collect_upnp_inputs(hass)
    assert inputs.igd_advertised is None
    assert any("TimeoutError" in r.getMessage() for r in caplog.records)


@pytest.mark.asyncio
async def test_collect_entries_flows_and_entities(
    hass: HomeAssistant, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Entry state, a pending discovery flow, and the entry's entities by unique id."""
    ignored = MockConfigEntry(domain="upnp", source=SOURCE_IGNORE, data={})
    ignored.add_to_hass(hass)
    inputs = await async_collect_upnp_inputs(hass)
    assert inputs.entry_present is False  # an ignored discovery is not set up
    entry = MockConfigEntry(domain="upnp", data={"udn": "uuid:aaa"})
    entry.add_to_hass(hass)
    entry.mock_state(hass, ConfigEntryState.SETUP_RETRY)
    monkeypatch.setattr(
        hass.config_entries.flow,
        "async_progress",
        lambda: [
            {"handler": "upnp", "context": {"source": "ssdp"}},
            {"handler": "hue", "context": {"source": "ssdp"}},
        ],
    )
    registry = er.async_get(hass)
    registry.async_get_or_create("sensor", "upnp", "uuid:aaa_ip", config_entry=entry)
    registry.async_get_or_create(
        "sensor",
        "upnp",
        "uuid:aaa_port_mapping_number_of_entries",
        config_entry=entry,
        disabled_by=er.RegistryEntryDisabler.INTEGRATION,
    )
    registry.async_get_or_create(
        "binary_sensor", "upnp", "uuid:aaa_wan_status", config_entry=entry
    )
    # Another integration's entity, and a upnp entity of no entry: ignored.
    registry.async_get_or_create("sensor", "fritz", "fritz_ip")
    registry.async_get_or_create("sensor", "upnp", "uuid:zzz_ip")
    inputs = await async_collect_upnp_inputs(hass)
    assert inputs.entry_present is True
    assert inputs.entry_loaded is False
    assert inputs.flow_in_progress is True
    assert inputs.sensors == {"ip": ["sensor.upnp_uuid_aaa_ip"]}
    assert inputs.disabled_sensors == {
        "port_mapping_number_of_entries": [
            "sensor.upnp_uuid_aaa_port_mapping_number_of_entries"
        ]
    }
    assert inputs.entity_ids == [
        "binary_sensor.upnp_uuid_aaa_wan_status",
        "sensor.upnp_uuid_aaa_ip",
    ]
    entry.mock_state(hass, ConfigEntryState.LOADED)
    assert (await async_collect_upnp_inputs(hass)).entry_loaded is True


# ---------------------------------------------------------------------------
# Adapter: UPnP-on evidence
# ---------------------------------------------------------------------------


def test_ssdp_advertisement_is_the_strongest_evidence() -> None:
    inputs = UpnpInputs(igd_advertised=True, igd_names=["eero"])
    result = upnp_igd_adapter(inputs, [], _context())
    assert result.posture["upnp_enabled"] is True
    assert result.posture["upnp_evidence"] == "ssdp"
    assert result.posture["upnp_gateway_names"] == ["eero"]
    assert result.notes == []
    section = merge_adapter_results([result])
    assert posture_cap("upnp_enabled") in section["capabilities"]
    assert section["sources"][posture_cap("upnp_enabled")] == "upnp_igd"
    # Display facts ride along in the posture but are not capabilities.
    for key in POSTURE_DISPLAY_KEYS & set(result.posture):
        assert posture_cap(key) not in section["capabilities"]


def test_loaded_entry_counts_only_while_its_entities_are_live() -> None:
    inputs = UpnpInputs(
        igd_advertised=False,
        entry_present=True,
        entry_loaded=True,
        entity_ids=["sensor.gw_ip"],
    )
    live = upnp_igd_adapter(
        inputs, [_entity("sensor.gw_ip", "203.0.113.5")], _context()
    )
    assert live.posture["upnp_evidence"] == "integration"
    dead = upnp_igd_adapter(
        inputs, [_entity("sensor.gw_ip", "unavailable")], _context()
    )
    assert "upnp_enabled" not in dead.posture
    assert any("not answering" in note for note in dead.notes)


def test_discovery_flow_is_evidence_only_when_the_cache_is_unreadable() -> None:
    flow = upnp_igd_adapter(
        UpnpInputs(igd_advertised=None, flow_in_progress=True), [], _context()
    )
    assert flow.posture["upnp_evidence"] == "discovery_flow"
    # The cache was read and is empty: the gateway expired, the flow lingers.
    stale = upnp_igd_adapter(
        UpnpInputs(igd_advertised=False, flow_in_progress=True), [], _context()
    )
    assert "upnp_enabled" not in stale.posture
    assert any("usually means UPnP is off" in note for note in stale.notes)
    unknown = upnp_igd_adapter(UpnpInputs(igd_advertised=None), [], _context())
    assert "upnp_enabled" not in unknown.posture
    assert any("SSDP discovery is not available" in note for note in unknown.notes)


# ---------------------------------------------------------------------------
# Adapter: public IP and port mappings against the previous run
# ---------------------------------------------------------------------------


def test_public_ip_is_pseudonymized_and_compared_with_the_previous_run() -> None:
    inputs = UpnpInputs(sensors={"ip": ["sensor.gw_ip"]})
    entities = [_entity("sensor.gw_ip", "203.0.113.5")]
    first = upnp_igd_adapter(inputs, entities, _context())
    key = PSEUDONYMIZER.ip_key("203.0.113.5")
    assert first.posture["public_ip_key"] == key
    assert first.posture["public_ip_entity_id"] == "sensor.gw_ip"
    assert "public_ip_changed" not in first.posture
    assert "203.0.113.5" not in repr(first)
    remembered = {"public_ip_key": key, "public_ip_entity_id": "sensor.gw_ip"}
    same = upnp_igd_adapter(inputs, entities, _context(remembered))
    assert same.posture["public_ip_changed"] is False
    changed = upnp_igd_adapter(
        inputs, entities, _context({**remembered, "public_ip_key": "old1"})
    )
    assert changed.posture["public_ip_changed"] is True
    assert changed.posture["public_ip_previous_key"] == "old1"
    section = merge_adapter_results([changed])
    assert posture_cap("public_ip_changed") in section["capabilities"]
    # A value remembered from another gateway's sensor is never compared.
    drifted = upnp_igd_adapter(
        inputs,
        entities,
        _context({"public_ip_key": "old1", "public_ip_entity_id": "sensor.other"}),
    )
    assert "public_ip_changed" not in drifted.posture
    unknown = upnp_igd_adapter(inputs, [_entity("sensor.gw_ip", "unknown")], _context())
    assert "public_ip_key" not in unknown.posture
    no_pseudonymizer = upnp_igd_adapter(inputs, entities, _context(pseudonymizer=None))
    assert "public_ip_key" not in no_pseudonymizer.posture


def test_unspecified_or_malformed_public_ip_is_unknown() -> None:
    assert public_ip_value("203.0.113.5") == "203.0.113.5"
    assert public_ip_value(" 2001:db8::1 ") == "2001:db8::1"
    assert public_ip_value("0.0.0.0") is None  # noqa: S104 - WAN down
    assert public_ip_value("::") is None
    assert public_ip_value("not an ip") is None
    inputs = UpnpInputs(sensors={"ip": ["sensor.gw_ip"]})
    down = upnp_igd_adapter(
        inputs,
        [_entity("sensor.gw_ip", "0.0.0.0")],  # noqa: S104
        _context({"public_ip_key": "aaaa", "public_ip_entity_id": "sensor.gw_ip"}),
    )
    assert "public_ip_key" not in down.posture
    assert "public_ip_changed" not in down.posture


def test_port_mapping_count_delta_and_disabled_sensor_note() -> None:
    sensor = "sensor.gw_port_mapping_number_of_entries"
    inputs = UpnpInputs(sensors={"port_mapping_number_of_entries": [sensor]})
    first = upnp_igd_adapter(inputs, [_entity(sensor, "3")], _context())
    assert first.posture["upnp_port_mapping_count"] == 3
    assert first.posture["upnp_port_mapping_entity_id"] == sensor
    assert "upnp_port_mappings_added" not in first.posture
    remembered = {"upnp_port_mapping_count": 3, "upnp_port_mapping_entity_id": sensor}
    more = upnp_igd_adapter(inputs, [_entity(sensor, "5.0")], _context(remembered))
    assert more.posture["upnp_port_mappings_added"] == 2
    assert more.posture["upnp_port_mapping_previous_count"] == 3
    fewer = upnp_igd_adapter(inputs, [_entity(sensor, "1")], _context(remembered))
    assert fewer.posture["upnp_port_mappings_added"] == 0
    drifted = upnp_igd_adapter(
        inputs,
        [_entity(sensor, "9")],
        _context({**remembered, "upnp_port_mapping_entity_id": "sensor.other"}),
    )
    assert "upnp_port_mappings_added" not in drifted.posture
    disabled = upnp_igd_adapter(
        UpnpInputs(disabled_sensors={"port_mapping_number_of_entries": [sensor]}),
        [],
        _context(),
    )
    assert "upnp_port_mapping_count" not in disabled.posture
    assert any(sensor in note and "Enable" in note for note in disabled.notes)
    assert set(POSTURE_MEMORY_KEYS) == {
        "public_ip_key",
        "public_ip_entity_id",
        "upnp_port_mapping_count",
        "upnp_port_mapping_entity_id",
        "guest_network_last_active",
        "guest_network_last_observed",
    }


@pytest.mark.parametrize(
    ("state", "expected"),
    [
        ("3", 3),
        ("3.0", 3),
        ("0", 0),
        ("inf", None),  # int(float("inf")) would raise OverflowError
        ("1e999", None),
        ("nan", None),
        ("-1", None),
        ("70000", None),
        ("lots", None),
    ],
)
def test_port_mapping_count_rejects_what_is_not_a_count(
    state: str, expected: int | None
) -> None:
    assert port_mapping_count(state) == expected
    sensor = "sensor.gw_port_mapping_number_of_entries"
    inputs = UpnpInputs(sensors={"port_mapping_number_of_entries": [sensor]})
    result = upnp_igd_adapter(inputs, [_entity(sensor, state)], _context())
    assert result.posture.get("upnp_port_mapping_count") == expected


def test_first_live_sensor_wins_with_two_gateways() -> None:
    """An unavailable first gateway does not hide the second one's live sensor."""
    inputs = UpnpInputs(sensors={"ip": ["sensor.old_ip", "sensor.new_ip"]})
    result = upnp_igd_adapter(
        inputs,
        [
            _entity("sensor.old_ip", "unavailable"),
            _entity("sensor.new_ip", "203.0.113.9"),
        ],
        _context(),
    )
    assert result.posture["public_ip_entity_id"] == "sensor.new_ip"


@pytest.mark.asyncio
async def test_build_network_snapshot_merges_the_upnp_adapter(
    hass: HomeAssistant,
) -> None:
    """End to end through the builder entry point with a stubbed SSDP cache."""
    v1, _ = IGD_SEARCH_TARGETS
    _ssdp_loaded(hass, _Scanner({v1: [_igd("uuid:aaa::" + v1, v1, "eero")]}))
    section = await async_build_network_snapshot(
        hass, [], _context(), now=NOW, entity_device={}, device_domains={}
    )
    assert section["posture"].get("upnp_enabled") is True
    assert section["sources"][posture_cap("upnp_enabled")] == "upnp_igd"
    assert section["posture"].get("upnp_gateway_names") == ["eero"]

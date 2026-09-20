# ruff: noqa: S101
"""Router adapters: generic device_tracker clients and eero posture."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any, cast
from unittest.mock import MagicMock

import pytest
from homeassistant.helpers import device_registry as dr
from homeassistant.helpers import entity_registry as er
from pytest_homeassistant_custom_component.common import MockConfigEntry

from custom_components.home_generative_agent.sentinel.network_inventory import (
    NetworkInventory,
)
from custom_components.home_generative_agent.sentinel.pseudonymizer import (
    Pseudonymizer,
)
from custom_components.home_generative_agent.snapshot import network as network_mod
from custom_components.home_generative_agent.snapshot.network import (
    CAP_CLIENTS,
    CAP_NEW_CLIENTS,
    COUNTER_CLIENT_COUNT,
    NetworkBuildContext,
    async_build_network_snapshot,
    attach_inventory,
    merge_adapter_results,
    posture_cap,
)
from custom_components.home_generative_agent.snapshot.router import (
    COUNTER_THREATS_DAY,
    EERO_RELOAD_NOTE,
    MacIndexEntry,
    RouterInputs,
    async_collect_router_inputs,
    eero_adapter,
    generic_router_tracker_adapter,
    mac_is_randomized,
    normalize_mac,
)
from custom_components.home_generative_agent.snapshot.schema import validate_snapshot

if TYPE_CHECKING:
    from homeassistant.core import HomeAssistant

    from custom_components.home_generative_agent.snapshot.schema import (
        SnapshotEntity,
    )

NOW = datetime(2026, 9, 18, 12, 0, tzinfo=UTC)
PSEUDONYMIZER = Pseudonymizer("test-salt")
MAC_A = "a8:bb:cc:dd:ee:01"
MAC_B = "a8:bb:cc:dd:ee:02"
RANDOM_MAC = "d2:11:22:33:44:55"
KEY_A = PSEUDONYMIZER.mac_key(MAC_A)
KEY_B = PSEUDONYMIZER.mac_key(MAC_B)


@pytest.fixture(autouse=True)
def _reset_logged_failures() -> None:
    network_mod._LOGGED_INPUT_FAILURES.clear()


def _tracker(
    entity_id: str,
    state: str = "home",
    *,
    mac: str | None = MAC_A,
    platform: str | None = "eero",
    name: str | None = "Lindo's iPhone",
    **attrs: Any,
) -> SnapshotEntity:
    attributes: dict[str, Any] = {"source_type": "router", **attrs}
    if mac is not None:
        attributes["mac"] = mac
    return {
        "entity_id": entity_id,
        "domain": entity_id.partition(".")[0],
        "state": state,
        "friendly_name": name,
        "area": None,
        "attributes": attributes,
        "last_changed": NOW.isoformat(),
        "last_updated": NOW.isoformat(),
        "platform": platform,
    }


def _entity(entity_id: str, state: str, platform: str = "eero") -> SnapshotEntity:
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
    *,
    inventory: NetworkInventory | None = None,
    previous: dict[str, Any] | None = None,
    pseudonymizer: Any = PSEUDONYMIZER,
) -> NetworkBuildContext:
    return NetworkBuildContext(
        enabled=True,
        pseudonymizer=pseudonymizer,
        network_inventory=inventory,
        previous_posture=previous,
    )


def _plain(clients: list[Any] | None) -> list[dict[str, Any]]:
    """Widen NetworkClient rows to plain dicts for item access in assertions."""
    assert clients is not None
    return [cast("dict[str, Any]", c) for c in clients]


def _memory_inventory() -> NetworkInventory:
    inventory = NetworkInventory(MagicMock())
    store = MagicMock()

    async def _save(_data: dict[str, Any]) -> None:
        return None

    store.async_save = _save
    inventory._store = store  # type: ignore[assignment]
    return inventory


# ---------------------------------------------------------------------------
# MAC helpers
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("A8:BB:CC:DD:EE:01", MAC_A),
        ("a8-bb-cc-dd-ee-01", MAC_A),
        ("a8bb.ccdd.ee01", MAC_A),
        ("a8bbccddee01", MAC_A),
        (" a8:bb:cc:dd:ee:01 ", MAC_A),
        ("a8:bb:cc:dd:ee", None),
        ("zz:bb:cc:dd:ee:01", None),
        ("", None),
        (None, None),
        (42, None),
    ],
)
def test_normalize_mac(raw: Any, expected: str | None) -> None:
    assert normalize_mac(raw) == expected


def test_locally_administered_bit_marks_a_random_address() -> None:
    assert mac_is_randomized(RANDOM_MAC)
    assert mac_is_randomized("06:00:00:00:00:00")
    assert mac_is_randomized("aa:bb:cc:dd:ee:ff")
    assert not mac_is_randomized(MAC_A)
    assert not mac_is_randomized("00:11:22:33:44:55")


# ---------------------------------------------------------------------------
# generic_router_tracker
# ---------------------------------------------------------------------------


def test_router_trackers_become_clients_without_their_mac() -> None:
    entities = [
        _tracker(
            "device_tracker.iphone",
            ip="192.168.1.23",
            host_name="Lindos-iPhone",
            manufacturer="Apple",
            connection_type="wireless",
            network_name="Home",
        ),
        _tracker(
            "device_tracker.nas",
            "not_home",
            mac=MAC_B,
            platform="unifi",
            name="NAS",
            ip="192.168.1.5",
            oui="Synology",
            is_wired=True,
            essid="Home",
            vlan=20,
            is_guest=False,
        ),
        # A GPS tracker and a tracker without a MAC are not clients.
        _tracker("device_tracker.phone_gps", mac=None, platform="mobile_app"),
        {
            **_tracker("device_tracker.gps", platform="mobile_app"),
            "attributes": {"source_type": "gps", "mac": MAC_B},
        },
    ]
    result = generic_router_tracker_adapter(RouterInputs(), entities, _context())
    clients = _plain(result.clients)
    assert [c["key"] for c in clients] == [KEY_A, KEY_B]
    iphone, nas = clients
    assert "mac" not in iphone
    assert iphone["connected"] is True
    assert iphone["name"] == "Lindo's iPhone"
    assert iphone["ip"] == "192.168.1.23"
    assert iphone["hostname"] == "Lindos-iPhone"
    assert iphone["manufacturer"] == "Apple"
    assert iphone["connection_type"] == "wireless"
    assert iphone["network_name"] == "Home"
    assert iphone["tracker_entity_id"] == "device_tracker.iphone"
    assert iphone["ha_integration"] == "eero"
    assert iphone["mac_randomized"] is False
    assert iphone["auto_trust"] is False
    assert nas["connected"] is False
    assert nas["manufacturer"] == "Synology"
    assert nas["connection_type"] == "wired"
    assert nas["network_name"] == "Home"
    assert nas["vlan"] == 20
    assert nas["is_guest"] is False
    # The counter and the inventory diff happen once, after the merge.
    assert result.counters == {}
    assert merge_adapter_results([result])["counters"][COUNTER_CLIENT_COUNT] == 1.0
    assert result.new_clients is None
    assert result.clients is not None and len(result.clients) == 2  # noqa: PT018


def test_two_trackers_for_one_mac_keep_the_connected_one() -> None:
    entities = [
        _tracker("device_tracker.iphone_wired", "not_home", connection_type="wired"),
        _tracker("device_tracker.iphone_wireless", "home", connection_type="wireless"),
    ]
    result = generic_router_tracker_adapter(RouterInputs(), entities, _context())
    clients = _plain(result.clients)
    assert len(clients) == 1
    assert clients[0]["tracker_entity_id"] == "device_tracker.iphone_wireless"


def test_no_router_tracker_means_no_client_capability() -> None:
    result = generic_router_tracker_adapter(
        RouterInputs(), [_entity("switch.other", "on")], _context()
    )
    assert result.clients is None
    assert result.notes == []


def test_missing_pseudonymizer_withholds_clients_with_a_note() -> None:
    result = generic_router_tracker_adapter(
        RouterInputs(),
        [_tracker("device_tracker.iphone")],
        _context(pseudonymizer=None),
    )
    assert result.clients is None
    assert any("salt" in note for note in result.notes)


def test_registry_join_sets_device_integration_and_auto_trust() -> None:
    inputs = RouterInputs(
        mac_index={MAC_A: MacIndexEntry("dev-1", "shelly", qualifying=True)},
        tracker_platforms={"device_tracker.plug": "fritz"},
    )
    result = generic_router_tracker_adapter(
        inputs, [_tracker("device_tracker.plug", platform="fritz")], _context()
    )
    client = _plain(result.clients)[0]
    assert client["ha_device_id"] == "dev-1"
    assert client["ha_integration"] == "shelly"
    assert client["auto_trust"] is True


@pytest.mark.asyncio
async def test_attach_inventory_marks_new_clients_first_seen_and_known_names() -> None:
    inventory = _memory_inventory()
    # Bootstrap the router source with one trusted client named "Media NAS".
    await inventory.async_commit(
        [
            {
                "protocol": "router",
                "device_id": KEY_B,
                "platform": "eero",
                "name": "Media NAS",
                "manufacturer": None,
                "model": None,
            }
        ],
        NOW,
        present_sources=["router"],
    )
    random_key = PSEUDONYMIZER.mac_key(RANDOM_MAC)
    entities = [
        _tracker("device_tracker.iphone"),  # new
        _tracker("device_tracker.nas", mac=MAC_B, name="Media NAS"),  # known
        _tracker(
            "device_tracker.nas_random",
            mac=RANDOM_MAC,
            name="Unknown",
            host_name="media nas",
        ),  # rotated: the placeholder name is ignored, the hostname matches
    ]
    context = _context(inventory=inventory)
    section = merge_adapter_results(
        [generic_router_tracker_adapter(RouterInputs(), entities, context)]
    )
    attach_inventory(section, context)
    by_key = {c["key"]: c for c in _plain(list(section["clients"]))}
    assert section.get("new_clients") == sorted([KEY_A, random_key])
    assert CAP_NEW_CLIENTS in section["capabilities"]
    assert by_key[KEY_B]["first_seen"] == NOW.isoformat()
    assert "first_seen" not in by_key[KEY_A]
    # The row's verdict rides along; a client without a row carries none.
    assert by_key[KEY_B]["trusted"] is True
    assert "trusted" not in by_key[KEY_A]
    assert by_key[random_key]["mac_randomized"] is True
    assert by_key[random_key]["hostname_trusted"] is True
    assert "hostname_trusted" not in by_key[KEY_A]


# ---------------------------------------------------------------------------
# Collector (real registries)
# ---------------------------------------------------------------------------


def _entry(hass: HomeAssistant, domain: str) -> MockConfigEntry:
    entry = MockConfigEntry(domain=domain)
    entry.add_to_hass(hass)
    return entry


@pytest.mark.asyncio
async def test_collector_indexes_macs_and_eero_entities(hass: HomeAssistant) -> None:
    registry = er.async_get(hass)
    devices = dr.async_get(hass)
    fritz = _entry(hass, "fritz")
    shelly = _entry(hass, "shelly")
    eero = _entry(hass, "eero")
    # A client device the router created: tracker only, does not qualify.
    tracked = devices.async_get_or_create(
        config_entry_id=fritz.entry_id,
        connections={(dr.CONNECTION_NETWORK_MAC, "A8:BB:CC:DD:EE:01")},
        name="Guest phone",
    )
    registry.async_get_or_create(
        "device_tracker", "fritz", "t1", config_entry=fritz, device_id=tracked.id
    )
    # A Shelly plug: the shelly entry owns a switch on a device with a MAC.
    plug = devices.async_get_or_create(
        config_entry_id=shelly.entry_id,
        connections={(dr.CONNECTION_NETWORK_MAC, MAC_B)},
    )
    registry.async_get_or_create(
        "switch", "shelly", "s1", config_entry=shelly, device_id=plug.id
    )
    # The router also tracks the plug (same registry device via MAC merge).
    registry.async_get_or_create(
        "device_tracker", "fritz", "t2", config_entry=fritz, device_id=plug.id
    )
    # eero network-level and client-level entities.
    registry.async_get_or_create("switch", "eero", "net1-upnp", config_entry=eero)
    registry.async_get_or_create("sensor", "eero", "net1-public_ip", config_entry=eero)
    registry.async_get_or_create(
        "sensor", "eero", "net1-client1-blocked_day", config_entry=eero
    )
    disabled = registry.async_get_or_create(
        "switch", "eero", "net1-wpa3", config_entry=eero
    )
    registry.async_update_entity(
        disabled.entity_id, disabled_by=er.RegistryEntryDisabler.USER
    )

    inputs = async_collect_router_inputs(hass)

    assert inputs.mac_index[MAC_A].qualifying is False
    assert inputs.mac_index[MAC_A].integration == "fritz"
    assert inputs.mac_index[MAC_B].qualifying is True
    assert inputs.mac_index[MAC_B].integration == "shelly"
    assert inputs.mac_index[MAC_B].device_id == plug.id
    assert inputs.eero_present is True
    assert inputs.eero_switches == {"upnp": [("net1", "switch.eero_net1_upnp")]}
    assert inputs.eero_sensors == {
        "public_ip": [("net1", "sensor.eero_net1_public_ip")]
    }
    assert set(inputs.tracker_platforms.values()) == {"fritz"}
    assert "Guest phone" in inputs.tracker_device_names.values()


@pytest.mark.asyncio
async def test_collector_failure_leaves_inputs_empty(
    hass: HomeAssistant, monkeypatch: pytest.MonkeyPatch, caplog: Any
) -> None:
    def _boom(_hass: Any) -> Any:
        msg = "registry moved"
        raise AttributeError(msg)

    monkeypatch.setattr(
        "custom_components.home_generative_agent.snapshot.router.er.async_get", _boom
    )
    inputs = async_collect_router_inputs(hass)
    assert inputs == RouterInputs(registry_failed=True)
    assert "router entity registry" in caplog.text
    # The tracker adapter then withholds clients rather than commit
    # rows joined to nothing.
    result = generic_router_tracker_adapter(
        inputs, [_tracker("device_tracker.iphone")], _context()
    )
    assert result.clients is None
    assert any("registry" in note for note in result.notes)


# ---------------------------------------------------------------------------
# eero
# ---------------------------------------------------------------------------


def _eero_inputs() -> RouterInputs:
    return RouterInputs(
        eero_present=True,
        eero_switches={
            "upnp": [("n1", "switch.eero_upnp")],
            "wpa3": [("n1", "switch.eero_wpa3")],
            "guest_network_enabled": [("n1", "switch.eero_guest")],
            "ddns_enabled": [("n1", "switch.eero_ddns")],
        },
        eero_sensors={
            "public_ip": [("n1", "sensor.eero_public_ip")],
            "connected_guest_clients_count": [("n1", "sensor.eero_guests")],
            "blocked_day": [("n1", "sensor.eero_blocked")],
        },
    )


def test_eero_posture_is_judged_across_networks() -> None:
    """Several eero networks: a weakening setting counts if any has it on."""
    inputs = RouterInputs(
        eero_present=True,
        eero_switches={
            "upnp": [("n1", "switch.n1_upnp"), ("n2", "switch.n2_upnp")],
            "wpa3": [("n1", "switch.n1_wpa3"), ("n2", "switch.n2_wpa3")],
        },
        eero_sensors={
            "connected_guest_clients_count": [
                ("n1", "sensor.n1_guests"),
                ("n2", "sensor.n2_guests"),
            ],
        },
    )
    entities = [
        _entity("switch.n1_upnp", "off"),
        _entity("switch.n2_upnp", "on"),
        _entity("switch.n1_wpa3", "on"),
        _entity("switch.n2_wpa3", "off"),
        _entity("sensor.n1_guests", "1"),
        _entity("sensor.n2_guests", "2"),
    ]
    posture = eero_adapter(inputs, entities, _context()).posture
    assert posture["upnp_enabled"] is True
    assert posture["upnp_enabled_entity_id"] == "switch.n2_upnp"
    assert posture["wpa3_enabled"] is False
    assert posture["wpa3_enabled_entity_id"] == "switch.n2_wpa3"
    assert posture["guest_client_count"] == 3


def test_labels_that_are_addresses_are_dropped() -> None:
    """A router that names an unresolved client by its address leaks nothing."""
    entities = [
        _tracker(
            "device_tracker.a",
            name="a8:bb:cc:dd:ee:01 (Wireless)",
            host_name="192.168.1.9",
            manufacturer=None,
        ),
    ]
    client = _plain(
        generic_router_tracker_adapter(RouterInputs(), entities, _context()).clients
    )[0]
    # Not "[mac]": no name at all, so the client is shown by its key.
    assert client["name"] is None
    assert client["hostname"] is None


def test_registry_device_name_beats_the_composed_friendly_name() -> None:
    """Eero's friendly name repeats the device name; the registry name is clean."""
    inputs = RouterInputs(
        tracker_device_names={"device_tracker.laptop": "Nico's laptop (Wireless)"}
    )
    # Through the MAC index the same name reaches every source for this MAC.
    indexed = RouterInputs(
        mac_index={
            MAC_A: MacIndexEntry(
                "dev", "eero", qualifying=False, device_name="Nico's laptop (Wireless)"
            )
        }
    )
    entities = [
        _tracker(
            "device_tracker.laptop",
            name="Nico's laptop (Wireless) kro Nico's laptop (Wireless)",
        )
    ]
    client = _plain(
        generic_router_tracker_adapter(inputs, entities, _context()).clients
    )[0]
    assert client["name"] == "Nico's laptop (Wireless)"
    client = _plain(
        generic_router_tracker_adapter(indexed, entities, _context()).clients
    )[0]
    assert client["name"] == "Nico's laptop (Wireless)"


@pytest.mark.asyncio
async def test_unlisted_router_integration_cannot_vouch(hass: HomeAssistant) -> None:
    """A domain that provides trackers is a router, listed or not."""
    registry = er.async_get(hass)
    devices = dr.async_get(hass)
    router = _entry(hass, "some_new_router")
    client = devices.async_get_or_create(
        config_entry_id=router.entry_id,
        connections={(dr.CONNECTION_NETWORK_MAC, MAC_A)},
    )
    registry.async_get_or_create(
        "device_tracker",
        "some_new_router",
        "t1",
        config_entry=router,
        device_id=client.id,
    )
    registry.async_get_or_create(
        "sensor", "some_new_router", "s1", config_entry=router, device_id=client.id
    )
    inputs = async_collect_router_inputs(hass)
    assert inputs.mac_index[MAC_A].qualifying is False


def test_eero_posture_from_switches_and_sensors() -> None:
    entities = [
        _entity("switch.eero_upnp", "off"),
        _entity("switch.eero_wpa3", "on"),
        _entity("switch.eero_guest", "unavailable"),
        _entity("sensor.eero_public_ip", "203.0.113.9"),
        _entity("sensor.eero_guests", "2"),
        _entity("sensor.eero_blocked", "17"),
    ]
    result = eero_adapter(_eero_inputs(), entities, _context())
    posture = result.posture
    assert posture["upnp_enabled"] is False
    assert posture["upnp_enabled_entity_id"] == "switch.eero_upnp"
    assert posture["upnp_evidence"] == "eero"
    assert posture["wpa3_enabled"] is True
    assert "guest_network_enabled" not in posture  # unavailable is not off
    assert "ddns_enabled" not in posture  # no entity: eero Plus only
    assert posture["guest_client_count"] == 2
    assert posture["public_ip_key"] == PSEUDONYMIZER.ip_key("203.0.113.9")
    assert posture["public_ip_entity_id"] == "sensor.eero_public_ip"
    assert "public_ip_changed" not in posture  # nothing to compare with
    assert result.counters[COUNTER_THREATS_DAY] == 17.0
    # The audit tells the user what eero cannot show it.
    assert EERO_RELOAD_NOTE in result.notes


def test_eero_public_ip_change_uses_the_shared_memory() -> None:
    entities = [_entity("sensor.eero_public_ip", "203.0.113.10")]
    previous = {
        "public_ip_key": PSEUDONYMIZER.ip_key("203.0.113.9"),
        "public_ip_entity_id": "sensor.eero_public_ip",
    }
    posture = eero_adapter(
        _eero_inputs(), entities, _context(previous=previous)
    ).posture
    assert posture["public_ip_changed"] is True
    assert posture["public_ip_previous_key"] == previous["public_ip_key"]
    # A value remembered from another sensor is not comparable.
    other = {**previous, "public_ip_entity_id": "sensor.gateway_ip"}
    posture = eero_adapter(_eero_inputs(), entities, _context(previous=other)).posture
    assert "public_ip_changed" not in posture


def test_eero_absent_asserts_nothing() -> None:
    result = eero_adapter(
        RouterInputs(), [_entity("switch.eero_upnp", "on")], _context()
    )
    assert result.posture == {}
    assert result.counters == {}
    assert result.notes == []


# ---------------------------------------------------------------------------
# Merge and the full build
# ---------------------------------------------------------------------------


def test_merge_publishes_client_capabilities_and_hides_entity_twins() -> None:
    from custom_components.home_generative_agent.snapshot.network import (  # noqa: PLC0415
        AdapterResult,
    )

    generic = AdapterResult(
        name="generic_router_tracker",
        clients=[{"key": "k1", "connected": True, "name": "old"}],
        new_clients=["k1"],
    )
    specific = AdapterResult(
        name="eero",
        clients=[{"key": "k1", "connected": True, "name": "new"}],
        posture={"upnp_enabled": False, "upnp_enabled_entity_id": "switch.eero_upnp"},
    )
    section = merge_adapter_results([generic, specific])
    assert section["clients"] == [{"key": "k1", "connected": True, "name": "new"}]
    assert section.get("new_clients") == ["k1"]
    assert CAP_CLIENTS in section["capabilities"]
    assert CAP_NEW_CLIENTS in section["capabilities"]
    assert section["sources"][CAP_CLIENTS] == "eero"
    assert posture_cap("upnp_enabled") in section["capabilities"]
    assert posture_cap("upnp_enabled_entity_id") not in section["capabilities"]


def test_merge_publishes_the_guest_flag_and_counts_the_clients_it_cannot_cover() -> (
    None
):
    from custom_components.home_generative_agent.snapshot.network import (  # noqa: PLC0415
        CAP_GUEST_CLIENTS,
        GUEST_FLAG_PARTIAL_NOTE,
        AdapterResult,
    )

    def merged(*clients: dict[str, Any]) -> Any:
        return merge_adapter_results(
            [AdapterResult(name="eero_runtime", clients=list(clients))]  # type: ignore[arg-type]
        )

    # No client says anything about the guest network: no source reports it.
    section = merged({"key": "a", "connected": True})
    assert CAP_GUEST_CLIENTS not in section["capabilities"]
    assert section["notes"] == []
    # Every connected client carries the flag; an offline one without it
    # (a stale tracker the router forgot) does not matter.
    section = merged(
        {"key": "a", "connected": True, "is_guest": False},
        {"key": "b", "connected": True, "is_guest": True},
        {"key": "c", "connected": False},
    )
    assert CAP_GUEST_CLIENTS in section["capabilities"]
    assert section["sources"][CAP_GUEST_CLIENTS] == "eero_runtime"
    assert section["notes"] == []
    # A flagged guest is a guest whatever the others say, so the check runs;
    # the connected clients no source covers are a stated gap.
    section = merged(
        {"key": "a", "connected": True, "is_guest": True},
        {"key": "b", "connected": True},
        {"key": "c", "connected": True},
    )
    assert CAP_GUEST_CLIENTS in section["capabilities"]
    assert section["notes"] == [GUEST_FLAG_PARTIAL_NOTE.format(count=2)]


def test_merge_never_carries_an_older_sources_guest_flag_forward() -> None:
    from custom_components.home_generative_agent.snapshot.network import (  # noqa: PLC0415
        CAP_GUEST_CLIENTS,
        AdapterResult,
    )

    # The tracker attribute dates from the last reload, when the device was
    # on the guest SSID; the fresher source sees it connected and says
    # nothing about the network. The old flag must not describe the new
    # connection.
    stale = AdapterResult(
        name="generic_router_tracker",
        clients=[{"key": "k1", "connected": False, "is_guest": True, "vlan": 20}],
    )
    fresh = AdapterResult(
        name="eero_runtime", clients=[{"key": "k1", "connected": True}]
    )
    section = merge_adapter_results([stale, fresh])
    client = dict(section["clients"][0])
    assert "is_guest" not in client
    assert client["vlan"] == 20  # other unknown fields are still filled
    assert CAP_GUEST_CLIENTS not in section["capabilities"]
    # A fresher source that does know wins, either way.
    fresh.clients = [{"key": "k1", "connected": True, "is_guest": False}]
    remerged = merge_adapter_results([stale, fresh])["clients"][0]
    assert remerged.get("is_guest") is False


@pytest.mark.asyncio
async def test_build_network_snapshot_runs_router_adapters(hass: HomeAssistant) -> None:
    eero = _entry(hass, "eero")
    registry = er.async_get(hass)
    registry.async_get_or_create("switch", "eero", "net1-upnp", config_entry=eero)
    registry.async_get_or_create(
        "device_tracker", "eero", "net1-c1-device_tracker", config_entry=eero
    )
    entities = [
        _entity("switch.eero_net1_upnp", "on"),
        _tracker("device_tracker.eero_net1_c1_device_tracker", ip="192.168.1.23"),
    ]
    inventory = _memory_inventory()
    section = await async_build_network_snapshot(
        hass,
        entities,
        NetworkBuildContext(
            options={}, pseudonymizer=PSEUDONYMIZER, network_inventory=inventory
        ),
        now=NOW,
        entity_device={},
        device_domains={},
    )
    assert CAP_CLIENTS in section["capabilities"]
    assert CAP_NEW_CLIENTS in section["capabilities"]
    client = cast("dict[str, Any]", section["clients"][0])
    assert client["key"] == KEY_A
    assert "mac" not in client
    # The router source is not bootstrapped yet, so nothing is new.
    assert section.get("new_clients") == []
    posture = cast("dict[str, Any]", section["posture"])
    assert posture["upnp_enabled"] is True
    assert posture["upnp_evidence"] == "eero"
    assert section["sources"][posture_cap("upnp_enabled")] == "eero"
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

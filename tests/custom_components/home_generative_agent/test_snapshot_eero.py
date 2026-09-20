# ruff: noqa: S101
"""The eero runtime adapter: clients and settings read from the coordinator."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any, cast
from unittest.mock import MagicMock

import pytest
from homeassistant.config_entries import ConfigEntryState
from pytest_homeassistant_custom_component.common import MockConfigEntry

from custom_components.home_generative_agent.sentinel.network_inventory import (
    NetworkInventory,
)
from custom_components.home_generative_agent.sentinel.pseudonymizer import (
    Pseudonymizer,
)
from custom_components.home_generative_agent.snapshot import network as network_mod
from custom_components.home_generative_agent.snapshot.eero import (
    CLIENT_ATTRS,
    FORBIDDEN_ATTRS,
    NETWORK_ATTRS,
    RUNTIME_FAILED_NOTE,
    EeroRuntimeInputs,
    collect_eero_runtime_inputs,
    eero_runtime_adapter,
)
from custom_components.home_generative_agent.snapshot.network import (
    CAP_CLIENTS,
    CAP_NEW_CLIENTS,
    NetworkBuildContext,
    async_build_network_snapshot,
    merge_adapter_results,
    posture_cap,
)
from custom_components.home_generative_agent.snapshot.router import (
    EERO_RELOAD_NOTE,
    MacIndexEntry,
    RouterInputs,
    generic_router_tracker_adapter,
)

if TYPE_CHECKING:
    from homeassistant.core import HomeAssistant

NOW = datetime(2026, 9, 20, 2, 0, tzinfo=UTC)
PSEUDONYMIZER = Pseudonymizer("test-salt")
MAC_A = "a8:bb:cc:dd:ee:01"
MAC_B = "a8:bb:cc:dd:ee:02"
KEY_A = PSEUDONYMIZER.mac_key(MAC_A)
KEY_B = PSEUDONYMIZER.mac_key(MAC_B)


@pytest.fixture(autouse=True)
def _reset_logged_failures() -> None:
    network_mod._LOGGED_INPUT_FAILURES.clear()


class _Secret:
    """A property that fails the test if anything reads it."""

    def __get__(self, obj: Any, owner: Any = None) -> Any:
        msg = "a secret attribute was read"
        raise AssertionError(msg)


class FakeClient:
    """Duck-typed EeroClient (eero 1.8.1 property names)."""

    def __init__(self, **kw: Any) -> None:
        self.mac = kw.get("mac")
        self.ip = kw.get("ip")
        self.hostname = kw.get("hostname")
        self.name = kw.get("name")
        self.manufacturer = kw.get("manufacturer")
        self.connection_type = kw.get("connection_type")
        self.wireless = kw.get("wireless")
        self.connected = kw.get("connected", False)
        self.last_active = kw.get("last_active")
        self.is_guest = kw.get("is_guest")
        self.device_type = kw.get("device_type")


class FakeNetwork:
    """Duck-typed EeroNetwork with the secrets armed."""

    password = _Secret()
    guest_network_password = _Secret()
    thread_master_key = _Secret()
    thread_commissioning_credential = _Secret()
    thread_active_operational_dataset = _Secret()
    qr_code = _Secret()
    guest_network_qr_code = _Secret()

    def __init__(self, network_id: str = "558654", **kw: Any) -> None:
        self.id = network_id
        self.name = kw.get("name", "Kro")
        self.upnp = kw.get("upnp")
        self.wpa3 = kw.get("wpa3")
        self.guest_network_enabled = kw.get("guest_network_enabled")
        self.ipv6_upstream = kw.get("ipv6_upstream")
        self.ddns_enabled = kw.get("ddns_enabled")
        self.block_malware = kw.get("block_malware")
        self.ad_block = kw.get("ad_block")
        self.premium_enabled = kw.get("premium_enabled", False)
        self.connected_guest_clients_count = kw.get("connected_guest_clients_count", 0)
        self.clients = kw.get("clients", [])


def _hass_with(
    networks: list[Any], *, state: ConfigEntryState = ConfigEntryState.LOADED
) -> Any:
    hass = MagicMock()
    entry = MagicMock()
    entry.state = state
    hass.config_entries.async_get_entry.return_value = entry
    coordinator = MagicMock()
    coordinator.data = type("Account", (), {"networks": networks})()
    hass.data = {"eero": {"entry-1": {"coordinator": coordinator}}}
    return hass


def _context(inventory: NetworkInventory | None = None) -> NetworkBuildContext:
    return NetworkBuildContext(
        enabled=True, pseudonymizer=PSEUDONYMIZER, network_inventory=inventory
    )


def _plain(clients: list[Any] | None) -> list[dict[str, Any]]:
    assert clients is not None
    return [cast("dict[str, Any]", c) for c in clients]


# ---------------------------------------------------------------------------
# Collector
# ---------------------------------------------------------------------------


def test_allowlist_and_forbidden_attributes_are_disjoint() -> None:
    assert not (set(NETWORK_ATTRS) | set(CLIENT_ATTRS)) & FORBIDDEN_ATTRS


def test_collector_reads_allowlisted_fields_and_never_a_secret() -> None:
    laptop = FakeClient(
        mac=MAC_A,
        ip="192.168.1.23",
        hostname="Nicos-MacBook",
        name="Nico's laptop",
        manufacturer="Apple",
        wireless=True,
        connected=True,
        last_active=datetime(2026, 9, 20, 1, 50, tzinfo=UTC),
        is_guest=False,
    )
    guest = FakeClient(
        mac=MAC_B, connection_type="Wireless", connected=True, is_guest=True
    )
    no_mac = FakeClient(mac=None, name="ghost", connected=True)
    network = FakeNetwork(
        upnp=True,
        wpa3=False,
        guest_network_enabled=True,
        ddns_enabled=None,  # eero Plus only: absent
        premium_enabled=False,
        connected_guest_clients_count=1,
        clients=[laptop, guest, no_mac],
    )
    inputs = collect_eero_runtime_inputs(_hass_with([network]))
    assert inputs.present and not inputs.failed  # noqa: PT018
    assert len(inputs.networks) == 1
    read = inputs.networks[0]
    assert read.id == "558654"
    assert read.name == "Kro"
    assert read.settings == {
        "upnp_enabled": True,
        "wpa3_enabled": False,
        "guest_network_enabled": True,
    }
    assert read.guest_client_count == 1
    assert read.premium_enabled is False
    assert [c.mac for c in read.clients] == [MAC_A, MAC_B]
    assert read.clients[0].name == "Nico's laptop"
    assert read.clients[0].hostname == "Nicos-MacBook"
    assert read.clients[0].connection_type == "wireless"
    assert read.clients[0].last_seen == "2026-09-20T01:50:00+00:00"
    assert read.clients[0].network_name == "Kro"
    assert read.clients[1].connection_type == "wireless"
    assert read.clients[1].is_guest is True


def test_collector_skips_entries_that_are_not_loaded() -> None:
    inputs = collect_eero_runtime_inputs(
        _hass_with([FakeNetwork()], state=ConfigEntryState.SETUP_RETRY)
    )
    assert inputs == EeroRuntimeInputs()


def test_collector_without_eero_is_absent() -> None:
    hass = MagicMock()
    hass.data = {}
    assert collect_eero_runtime_inputs(hass) == EeroRuntimeInputs()


def test_collector_tolerates_a_renamed_property() -> None:
    """A property that raises or is missing degrades to a missing field."""

    class Drifted(FakeNetwork):
        @property
        def upnp(self) -> bool:  # type: ignore[override]
            msg = "moved"
            raise AttributeError(msg)

    network = FakeNetwork(wpa3=True, clients=[FakeClient(mac=MAC_A, connected=True)])
    network.__class__ = Drifted  # the class property now shadows the instance value
    del network.ad_block
    inputs = collect_eero_runtime_inputs(_hass_with([network]))
    assert inputs.networks[0].settings == {"wpa3_enabled": True}
    assert len(inputs.networks[0].clients) == 1


def test_collector_failure_is_logged_once_and_flagged(caplog: Any) -> None:
    hass = MagicMock()
    hass.data = {"eero": {"entry-1": {"coordinator": MagicMock()}}}
    hass.config_entries.async_get_entry.side_effect = RuntimeError("no registry")
    inputs = collect_eero_runtime_inputs(hass)
    assert inputs.failed is True
    assert "eero coordinator" in caplog.text


# ---------------------------------------------------------------------------
# Adapter
# ---------------------------------------------------------------------------


def _inputs(*networks: Any) -> EeroRuntimeInputs:
    return collect_eero_runtime_inputs(_hass_with(list(networks)))


def test_adapter_publishes_clients_and_posture() -> None:
    inventory = NetworkInventory(MagicMock())
    network = FakeNetwork(
        upnp=True,
        wpa3=True,
        guest_network_enabled=False,
        ipv6_upstream=True,
        block_malware=True,
        ad_block=False,
        premium_enabled=True,
        clients=[
            FakeClient(
                mac=MAC_A,
                name="Nico's laptop",
                manufacturer="Apple",
                wireless=True,
                connected=True,
            ),
            FakeClient(mac=MAC_B, name="Printer", wireless=False, connected=False),
        ],
    )
    router_inputs = RouterInputs(
        mac_index={MAC_B: MacIndexEntry("dev-p", "ipp", qualifying=True)}
    )
    result = eero_runtime_adapter(_inputs(network), router_inputs, _context(inventory))
    assert result.name == "eero_runtime"
    clients = _plain(result.clients)
    assert [c["key"] for c in clients] == [KEY_A, KEY_B]
    assert "mac" not in clients[0]
    assert clients[0]["name"] == "Nico's laptop"
    assert clients[0]["connection_type"] == "wireless"
    assert clients[0]["network_name"] == "Kro"
    assert clients[0]["ha_integration"] == "eero"
    assert clients[0]["tracker_entity_id"] is None
    assert clients[1]["connected"] is False
    assert clients[1]["connection_type"] == "wired"
    assert clients[1]["auto_trust"] is True
    assert clients[1]["ha_device_id"] == "dev-p"
    assert result.new_clients == []  # router source not bootstrapped yet
    assert result.posture == {
        "upnp_enabled": True,
        "upnp_evidence": "eero",
        "wpa3_enabled": True,
        "guest_network_enabled": False,
        "ipv6_enabled": True,
        "malware_blocking_enabled": True,
        "ad_blocking_enabled": False,
        "guest_client_count": 0,
    }
    assert result.notes == []


def test_adapter_judges_several_networks_together() -> None:
    main = FakeNetwork("1", upnp=False, wpa3=True, connected_guest_clients_count=2)
    cabin = FakeNetwork("2", upnp=True, wpa3=False, connected_guest_clients_count=1)
    posture = eero_runtime_adapter(
        _inputs(main, cabin), RouterInputs(), _context()
    ).posture
    assert posture["upnp_enabled"] is True
    assert posture["wpa3_enabled"] is False
    assert posture["guest_client_count"] == 3


def test_adapter_notes_missing_plus_and_failed_reads() -> None:
    result = eero_runtime_adapter(
        _inputs(FakeNetwork(premium_enabled=False)), RouterInputs(), _context()
    )
    assert any("eero Plus" in n for n in result.notes)
    failed = eero_runtime_adapter(
        EeroRuntimeInputs(present=True, failed=True), RouterInputs(), _context()
    )
    assert failed.clients is None
    assert failed.notes == [RUNTIME_FAILED_NOTE]
    absent = eero_runtime_adapter(EeroRuntimeInputs(), RouterInputs(), _context())
    assert absent.clients is None
    assert absent.posture == {}
    assert absent.notes == []


def test_runtime_clients_win_the_merge_but_keep_the_tracker_link() -> None:
    entities: list[Any] = [
        {
            "entity_id": "device_tracker.laptop",
            "domain": "device_tracker",
            "state": "not_home",  # eero's tracker is stale
            "friendly_name": "Nico's laptop (Wireless) kro Nico's laptop (Wireless)",
            "area": None,
            "attributes": {"source_type": "router", "mac": MAC_A},
            "last_changed": NOW.isoformat(),
            "last_updated": NOW.isoformat(),
            "platform": "eero",
        }
    ]
    trackers = generic_router_tracker_adapter(RouterInputs(), entities, _context())
    runtime = eero_runtime_adapter(
        _inputs(
            FakeNetwork(
                clients=[FakeClient(mac=MAC_A, name="Nico's laptop", connected=True)]
            )
        ),
        RouterInputs(),
        _context(),
    )
    section = merge_adapter_results([trackers, runtime])
    client = cast("dict[str, Any]", section["clients"][0])
    assert client["connected"] is True  # the runtime read is fresher
    assert client["name"] == "Nico's laptop"
    assert client["tracker_entity_id"] == "device_tracker.laptop"
    assert section["sources"][CAP_CLIENTS] == "eero_runtime"


@pytest.mark.asyncio
async def test_build_network_snapshot_sees_a_client_with_no_tracker(
    hass: HomeAssistant,
) -> None:
    """The case from the field: a device joined after setup, eero made no tracker."""
    entry = MockConfigEntry(domain="eero")
    entry.add_to_hass(hass)
    entry.mock_state(hass, ConfigEntryState.LOADED)
    coordinator = MagicMock()
    coordinator.data = type(
        "Account",
        (),
        {
            "networks": [
                FakeNetwork(
                    upnp=True,
                    clients=[
                        FakeClient(
                            mac=MAC_A,
                            name="Guest phone",
                            manufacturer="Apple",
                            connected=True,
                        )
                    ],
                )
            ]
        },
    )()
    hass.data["eero"] = {entry.entry_id: {"coordinator": coordinator}}
    inventory = NetworkInventory(MagicMock())
    store = MagicMock()

    async def _save(_data: dict[str, Any]) -> None:
        return None

    store.async_save = _save
    inventory._store = store  # type: ignore[assignment]
    await inventory.async_commit(
        [], NOW, present_sources=["router"]
    )  # bootstrapped, empty

    section = await async_build_network_snapshot(
        hass,
        [],  # no entities at all: no tracker exists for the phone
        NetworkBuildContext(
            options={}, pseudonymizer=PSEUDONYMIZER, network_inventory=inventory
        ),
        now=NOW,
        entity_device={},
        device_domains={},
    )
    assert CAP_CLIENTS in section["capabilities"]
    assert CAP_NEW_CLIENTS in section["capabilities"]
    assert section["sources"][CAP_CLIENTS] == "eero_runtime"
    assert cast("dict[str, Any]", section["clients"][0])["name"] == "Guest phone"
    assert section.get("new_clients") == [KEY_A]
    assert cast("dict[str, Any]", section["posture"])["upnp_enabled"] is True
    assert section["sources"][posture_cap("upnp_enabled")] == "eero_runtime"
    # The reload caveat is for the tracker path; the runtime path replaces it.
    assert EERO_RELOAD_NOTE not in section.get("notes", [])

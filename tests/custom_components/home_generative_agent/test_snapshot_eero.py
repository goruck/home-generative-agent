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
    NO_PLUS_NOTE,
    PLUS_UNKNOWN_NOTE,
    RUNTIME_FAILED_NOTE,
    STALE_POLL_NOTE,
    EeroRuntimeInputs,
    collect_eero_runtime_inputs,
    eero_runtime_adapter,
)
from custom_components.home_generative_agent.snapshot.network import (
    CAP_CLIENTS,
    CAP_NEW_CLIENTS,
    COUNTER_CLIENT_COUNT,
    AdapterResult,
    NetworkBuildContext,
    async_build_network_snapshot,
    merge_adapter_results,
    posture_cap,
)
from custom_components.home_generative_agent.snapshot.router import (
    EERO_RELOAD_NOTE,
    NO_SALT_NOTE,
    REGISTRY_FAILED_NOTE,
    MacIndexEntry,
    RouterInputs,
    generic_router_tracker_adapter,
    router_adapters,
)

if TYPE_CHECKING:
    from homeassistant.core import HomeAssistant

NOW = datetime(2026, 9, 20, 2, 0, tzinfo=UTC)
PSEUDONYMIZER = Pseudonymizer("test-salt")
MAC_A = "a8:bb:cc:dd:ee:01"
MAC_B = "a8:bb:cc:dd:ee:02"
KEY_A = PSEUDONYMIZER.mac_key(MAC_A)
KEY_B = PSEUDONYMIZER.mac_key(MAC_B)

# Every secret attribute access on a fake object lands here.
SECRET_ACCESSES: list[str] = []


@pytest.fixture(autouse=True)
def _reset() -> None:
    network_mod._LOGGED_INPUT_FAILURES.clear()
    SECRET_ACCESSES.clear()


class _Secret:
    """A property that records every read; the adapter must never trigger it."""

    def __set_name__(self, owner: Any, name: str) -> None:
        self._name = name

    def __get__(self, obj: Any, owner: Any = None) -> Any:
        SECRET_ACCESSES.append(self._name)
        return "s3cret"


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
    networks: list[Any],
    *,
    state: ConfigEntryState = ConfigEntryState.LOADED,
    configured: list[str] | None = None,
    last_update_success: bool = True,
) -> Any:
    hass = MagicMock()
    entry = MagicMock()
    entry.state = state
    hass.config_entries.async_get_entry.return_value = entry
    coordinator = MagicMock()
    coordinator.data = type("Account", (), {"networks": networks})()
    coordinator.last_update_success = last_update_success
    data: dict[str, Any] = {"coordinator": coordinator}
    if configured is not None:
        data["networks"] = configured
    hass.data = {"eero": {"entry-1": data}}
    return hass


def _context(
    inventory: NetworkInventory | None = None, *, salt: bool = True
) -> NetworkBuildContext:
    return NetworkBuildContext(
        enabled=True,
        pseudonymizer=PSEUDONYMIZER if salt else None,
        network_inventory=inventory,
    )


def _plain(clients: list[Any] | None) -> list[dict[str, Any]]:
    assert clients is not None
    return [cast("dict[str, Any]", c) for c in clients]


def _inputs(*networks: Any, **kw: Any) -> EeroRuntimeInputs:
    return collect_eero_runtime_inputs(_hass_with(list(networks), **kw))


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
    unknown_state = FakeClient(mac="a8:bb:cc:dd:ee:03", connected="yes")
    network = FakeNetwork(
        upnp=True,
        wpa3=False,
        guest_network_enabled=True,
        ddns_enabled=None,  # eero Plus only: absent
        premium_enabled=False,
        connected_guest_clients_count=1,
        clients=[laptop, guest, no_mac, unknown_state],
    )
    inputs = collect_eero_runtime_inputs(_hass_with([network]))
    assert inputs.present and not inputs.failed and not inputs.stale  # noqa: PT018
    assert inputs.clients_complete
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
    # A client whose MAC or connection state cannot be read is skipped.
    assert [c.mac for c in read.clients] == [MAC_A, MAC_B]
    assert read.clients[0].name == "Nico's laptop"
    assert read.clients[0].hostname == "Nicos-MacBook"
    assert read.clients[0].connection_type == "wireless"
    assert read.clients[0].last_seen == "2026-09-20T01:50:00+00:00"
    assert read.clients[0].network_name == "Kro"
    assert read.clients[1].connection_type == "wireless"
    assert read.clients[1].is_guest is True
    assert SECRET_ACCESSES == []


def test_collector_reads_only_the_configured_networks() -> None:
    home = FakeNetwork("1", upnp=False, clients=[FakeClient(mac=MAC_A, connected=True)])
    cabin = FakeNetwork("2", upnp=True, clients=[FakeClient(mac=MAC_B, connected=True)])
    inputs = _inputs(home, cabin, configured=["1"])
    assert [n.id for n in inputs.networks] == ["1"]
    # No selection recorded means every network on the account.
    assert [n.id for n in _inputs(home, cabin).networks] == ["1", "2"]


def test_collector_flags_a_failed_poll_and_skips_unloaded_entries() -> None:
    assert _inputs(FakeNetwork(), last_update_success=False).stale is True
    assert (
        _inputs(FakeNetwork(), state=ConfigEntryState.SETUP_RETRY)
        == EeroRuntimeInputs()
    )
    hass = MagicMock()
    hass.data = {}
    assert collect_eero_runtime_inputs(hass) == EeroRuntimeInputs()


def test_collector_tolerates_a_renamed_property_but_not_an_unreadable_client_list() -> (
    None
):
    class Drifted(FakeNetwork):
        @property
        def upnp(self) -> bool:  # type: ignore[override]
            msg = "moved"
            raise AttributeError(msg)

    network = FakeNetwork(wpa3=True, clients=[FakeClient(mac=MAC_A, connected=True)])
    network.__class__ = Drifted  # the class property now shadows the instance value
    del network.ad_block
    inputs = _inputs(network)
    assert inputs.networks[0].settings == {"wpa3_enabled": True}
    assert len(inputs.networks[0].clients) == 1
    assert inputs.clients_complete

    class NoClients(FakeNetwork):
        @property
        def clients(self) -> list[Any]:  # type: ignore[override]
            msg = "gone"
            raise RuntimeError(msg)

    broken = FakeNetwork()
    broken.__class__ = NoClients
    inputs = _inputs(FakeNetwork("1"), broken)
    assert inputs.networks[1].clients_unreadable is True
    assert not inputs.clients_complete


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


def test_adapter_publishes_clients_and_posture() -> None:
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
        mac_index={
            MAC_B: MacIndexEntry(
                "dev-p", "ipp", qualifying=True, device_name="Office printer"
            )
        }
    )
    result = eero_runtime_adapter(_inputs(network), router_inputs, _context())
    assert result.name == "eero_runtime"
    clients = _plain(result.clients)
    assert [c["key"] for c in clients] == [KEY_A, KEY_B]
    assert "mac" not in clients[0]
    assert clients[0]["name"] == "Nico's laptop"
    assert clients[0]["connection_type"] == "wireless"
    assert clients[0]["network_name"] == "Kro"
    assert clients[0]["ha_integration"] == "eero"
    assert clients[0]["tracker_entity_id"] is None
    # The registry device's name wins over eero's for a client joined to one.
    assert clients[1]["name"] == "Office printer"
    assert clients[1]["connected"] is False
    assert clients[1]["connection_type"] == "wired"
    assert clients[1]["auto_trust"] is True
    assert clients[1]["ha_device_id"] == "dev-p"
    assert result.new_clients is None  # the inventory diff runs once, after the merge
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


def test_adapter_judges_several_networks_together_only_when_all_report() -> None:
    main = FakeNetwork("1", upnp=False, wpa3=True, connected_guest_clients_count=2)
    cabin = FakeNetwork("2", upnp=True, wpa3=False, connected_guest_clients_count=1)
    posture = eero_runtime_adapter(
        _inputs(main, cabin), RouterInputs(), _context()
    ).posture
    assert posture["upnp_enabled"] is True
    assert posture["wpa3_enabled"] is False
    assert posture["guest_client_count"] == 3
    # A network that does not report a setting keeps the runtime silent on
    # it, so the entity tier's value (and a standing finding) survives.
    silent = FakeNetwork("2", upnp=None, wpa3=None)
    posture = eero_runtime_adapter(
        _inputs(main, silent), RouterInputs(), _context()
    ).posture
    assert "upnp_enabled" not in posture
    assert "wpa3_enabled" not in posture


PLUS_KEYS = {"ddns_enabled", "malware_blocking_enabled", "ad_blocking_enabled"}
PLUS_SETTINGS: dict[str, Any] = {
    "upnp": True,
    "ddns_enabled": True,
    "block_malware": False,
    "ad_block": False,
}


def test_plus_only_settings_follow_the_networks_that_have_plus() -> None:
    free = eero_runtime_adapter(
        _inputs(FakeNetwork(premium_enabled=False, **PLUS_SETTINGS)),
        RouterInputs(),
        _context(),
    )
    assert free.posture["upnp_enabled"] is True
    assert not PLUS_KEYS & set(free.posture)
    # Known off everywhere: withdrawn from the merge too, so a switch left
    # over from a lapsed subscription cannot raise the finding.
    assert free.posture_withdrawn == PLUS_KEYS
    assert NO_PLUS_NOTE in free.notes
    entity_tier = AdapterResult(
        name="eero",
        posture={
            "malware_blocking_enabled": False,
            "malware_blocking_enabled_entity_id": "switch.x",
        },
    )
    merged = cast(
        "dict[str, Any]", merge_adapter_results([entity_tier, free])["posture"]
    )
    assert "malware_blocking_enabled" not in merged
    assert "malware_blocking_enabled_entity_id" not in merged

    plus = eero_runtime_adapter(
        _inputs(FakeNetwork(premium_enabled=True, **PLUS_SETTINGS)),
        RouterInputs(),
        _context(),
    )
    assert plus.posture["malware_blocking_enabled"] is False
    assert plus.posture["ddns_enabled"] is True
    assert plus.posture_withdrawn == set()

    # One network with Plus, one without: the Plus network is still judged.
    mixed = eero_runtime_adapter(
        _inputs(
            FakeNetwork("1", premium_enabled=True, **PLUS_SETTINGS),
            FakeNetwork("2", premium_enabled=False, upnp=False),
        ),
        RouterInputs(),
        _context(),
    )
    assert mixed.posture["ddns_enabled"] is True
    assert mixed.posture_withdrawn == set()

    # Unknown Plus state: left to the entity tier, and the gap is noted.
    unknown = eero_runtime_adapter(
        _inputs(FakeNetwork(premium_enabled=None, **PLUS_SETTINGS)),
        RouterInputs(),
        _context(),
    )
    assert not PLUS_KEYS & set(unknown.posture)
    assert unknown.posture_withdrawn == set()
    assert PLUS_UNKNOWN_NOTE in unknown.notes


def test_the_deciding_networks_switch_is_the_entity_twin() -> None:
    """Same triggering entity on both tiers, so exclusions and identity agree."""
    router_inputs = RouterInputs(
        eero_switches={
            "upnp": [("1", "switch.home_upnp"), ("2", "switch.cabin_upnp")],
            "wpa3": [("1", "switch.home_wpa3")],
        }
    )
    posture = eero_runtime_adapter(
        _inputs(
            FakeNetwork("1", upnp=False, wpa3=False),
            FakeNetwork("2", upnp=True, wpa3=True),
        ),
        router_inputs,
        _context(),
    ).posture
    assert posture["upnp_enabled"] is True
    assert (
        posture["upnp_enabled_entity_id"] == "switch.cabin_upnp"
    )  # the one that is on
    assert posture["wpa3_enabled"] is False
    assert (
        posture["wpa3_enabled_entity_id"] == "switch.home_wpa3"
    )  # the one that is off


def test_a_stale_poll_publishes_no_settings() -> None:
    """Cached values would keep resetting, or advancing, the guest idle clock."""
    stale = eero_runtime_adapter(
        _inputs(
            FakeNetwork(upnp=True, connected_guest_clients_count=1),
            last_update_success=False,
        ),
        RouterInputs(),
        _context(),
    )
    assert stale.posture == {}
    assert STALE_POLL_NOTE in stale.notes


def test_adapter_withholds_clients_and_says_why() -> None:
    plain = FakeNetwork(
        premium_enabled=False, clients=[FakeClient(mac=MAC_A, connected=True)]
    )
    # eero Plus absent (known false) is noted; unknown is not.
    result = eero_runtime_adapter(_inputs(plain), RouterInputs(), _context())
    assert NO_PLUS_NOTE in result.notes
    unknown = FakeNetwork(premium_enabled=None)
    assert (
        NO_PLUS_NOTE
        not in eero_runtime_adapter(_inputs(unknown), RouterInputs(), _context()).notes
    )
    # Blockers, in order: structural failure, stale poll, partial read, no
    # salt, registry failure. Posture is still published where it can be.
    failed = eero_runtime_adapter(
        EeroRuntimeInputs(present=True, failed=True), RouterInputs(), _context()
    )
    assert failed.clients is None and failed.notes == [RUNTIME_FAILED_NOTE]  # noqa: PT018
    stale = eero_runtime_adapter(
        _inputs(plain, last_update_success=False), RouterInputs(), _context()
    )
    assert stale.clients is None and STALE_POLL_NOTE in stale.notes  # noqa: PT018
    assert stale.posture == {}  # a failed poll's cached settings are not published

    class NoClients(FakeNetwork):
        @property
        def clients(self) -> list[Any]:  # type: ignore[override]
            msg = "gone"
            raise RuntimeError(msg)

    broken = FakeNetwork()
    broken.__class__ = NoClients
    partial = eero_runtime_adapter(_inputs(plain, broken), RouterInputs(), _context())
    assert partial.clients is None and RUNTIME_FAILED_NOTE in partial.notes  # noqa: PT018
    no_salt = eero_runtime_adapter(_inputs(plain), RouterInputs(), _context(salt=False))
    assert no_salt.clients is None and NO_SALT_NOTE in no_salt.notes  # noqa: PT018
    no_registry = eero_runtime_adapter(
        _inputs(plain), RouterInputs(registry_failed=True), _context()
    )
    assert no_registry.clients is None and REGISTRY_FAILED_NOTE in no_registry.notes  # noqa: PT018
    absent = eero_runtime_adapter(EeroRuntimeInputs(), RouterInputs(), _context())
    assert absent.clients is None and absent.posture == {} and absent.notes == []  # noqa: PT018


def test_a_complete_empty_read_publishes_an_empty_client_list() -> None:
    """So the router source bootstraps instead of trusting the first client silently."""
    result = eero_runtime_adapter(
        _inputs(FakeNetwork(clients=[])), RouterInputs(), _context()
    )
    assert result.clients == []
    section = merge_adapter_results([result])
    assert CAP_CLIENTS in section["capabilities"]
    assert section["clients"] == []
    assert section["counters"][COUNTER_CLIENT_COUNT] == 0.0


def _tracker(state: str = "not_home", **attrs: Any) -> dict[str, Any]:
    return {
        "entity_id": "device_tracker.laptop",
        "domain": "device_tracker",
        "state": state,
        "friendly_name": "Nico's laptop (Wireless) kro Nico's laptop (Wireless)",
        "area": None,
        "attributes": {"source_type": "router", "mac": MAC_A, **attrs},
        "last_changed": NOW.isoformat(),
        "last_updated": NOW.isoformat(),
        "platform": "eero",
    }


def test_runtime_clients_win_the_merge_but_keep_what_the_tracker_knew() -> None:
    trackers = generic_router_tracker_adapter(
        RouterInputs(tracker_devices={"device_tracker.laptop": "dev-t"}),
        [cast("Any", _tracker(vlan=20, ip="192.168.1.23"))],
        _context(),
    )
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
    assert client["ha_device_id"] == "dev-t"  # the tracker's device link survives
    assert client["vlan"] == 20
    assert (
        client["ip"] == "192.168.1.23"
    )  # the runtime knew no IP; the tracker's stands
    assert section["sources"][CAP_CLIENTS] == "eero_runtime"
    assert section["counters"][COUNTER_CLIENT_COUNT] == 1.0


def test_runtime_posture_clears_the_entity_tiers_stale_twin() -> None:
    entity_tier = AdapterResult(
        name="eero",
        posture={"upnp_enabled": True, "upnp_enabled_entity_id": "switch.kro_upnp"},
    )
    runtime = AdapterResult(name="eero_runtime", posture={"upnp_enabled": False})
    posture = cast(
        "dict[str, Any]", merge_adapter_results([entity_tier, runtime])["posture"]
    )
    assert posture["upnp_enabled"] is False
    assert "upnp_enabled_entity_id" not in posture


def test_reload_note_only_when_the_runtime_supplied_no_clients() -> None:
    inputs = RouterInputs(eero_present=True)
    with_clients = AdapterResult(name="eero_runtime", clients=[])
    without = AdapterResult(name="eero_runtime", notes=[STALE_POLL_NOTE])
    notes = [
        n
        for r in router_adapters(inputs, [], _context(), eero_runtime=with_clients)
        for n in r.notes
    ]
    assert EERO_RELOAD_NOTE not in notes
    notes = [
        n
        for r in router_adapters(inputs, [], _context(), eero_runtime=without)
        for n in r.notes
    ]
    assert EERO_RELOAD_NOTE in notes
    assert STALE_POLL_NOTE in notes


@pytest.mark.asyncio
async def test_build_network_snapshot_sees_a_client_with_no_tracker(
    hass: HomeAssistant,
) -> None:
    """The case from the field: a device joined after setup, eero made no tracker."""
    entry = MockConfigEntry(domain="eero")
    entry.add_to_hass(hass)
    entry.mock_state(hass, ConfigEntryState.LOADED)
    coordinator = MagicMock()
    coordinator.last_update_success = True
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
    hass.data["eero"] = {
        entry.entry_id: {"coordinator": coordinator, "networks": ["558654"]}
    }
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
    assert SECRET_ACCESSES == []

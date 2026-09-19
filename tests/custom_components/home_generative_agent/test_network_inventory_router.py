# ruff: noqa: S101
"""The device inventory's router source, and the engine's client commit."""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, Any, cast
from unittest.mock import MagicMock

import pytest
import voluptuous as vol

import custom_components.home_generative_agent as _hga
from custom_components.home_generative_agent.const import (
    CONF_SENTINEL_NETWORK_UNKNOWN_DEVICE_GRACE_MIN,
)
from custom_components.home_generative_agent.sentinel.engine import SentinelEngine
from custom_components.home_generative_agent.sentinel.network_inventory import (
    ROUTER_SOURCE,
    NetworkInventory,
    client_observation,
    device_key,
)
from custom_components.home_generative_agent.sentinel.notifier import (
    _TRUST_DEVICE_TYPES,
)
from custom_components.home_generative_agent.sentinel.suppression import (
    SuppressionManager,
    SuppressionState,
)
from custom_components.home_generative_agent.snapshot.network import (
    CAP_CLIENTS,
    CAP_NEW_CLIENTS,
    radio_cap,
)
from custom_components.home_generative_agent.snapshot.schema import validate_snapshot

if TYPE_CHECKING:
    from homeassistant.core import HomeAssistant

    from custom_components.home_generative_agent.audit.store import AuditStore
    from custom_components.home_generative_agent.sentinel.notifier import (
        SentinelNotifier,
    )
    from custom_components.home_generative_agent.snapshot.network import (
        NetworkBuildContext,
    )
    from custom_components.home_generative_agent.snapshot.schema import (
        FullStateSnapshot,
    )

NOW = datetime(2026, 9, 18, 12, 0, tzinfo=UTC)
_SCHEMA = cast("Any", _hga).SENTINEL_NETWORK_DEVICE_SCHEMA


def _radio(device_id: str) -> dict[str, Any]:
    return {
        "device_id": device_id,
        "protocol": "zigbee",
        "platform": "zha",
        "name": "Bulb",
        "is_security_device": False,
        "manufacturer": None,
        "model": None,
    }


def _client(key: str, **extra: Any) -> dict[str, Any]:
    return {
        "key": key,
        "connected": True,
        "name": f"Device {key}",
        "hostname": f"host-{key}",
        "ip": "192.168.1.23",
        "manufacturer": "Apple",
        "ha_integration": "eero",
        "tracker_entity_id": f"device_tracker.{key}",
        **extra,
    }


def _memory_inventory() -> tuple[NetworkInventory, list[dict[str, Any]]]:
    inventory = NetworkInventory(MagicMock())
    store = MagicMock()
    saved: list[dict[str, Any]] = []

    async def _save(data: dict[str, Any]) -> None:
        saved.append(json.loads(json.dumps(data)))

    store.async_save = _save
    inventory._store = store  # type: ignore[assignment]
    return inventory, saved


# ---------------------------------------------------------------------------
# Inventory
# ---------------------------------------------------------------------------


def test_client_observation_carries_no_address() -> None:
    obs = client_observation(_client("k1", ha_device_id="dev-1", auto_trust=True))
    assert obs == {
        "protocol": ROUTER_SOURCE,
        "device_id": "k1",
        "platform": "eero",
        "ha_device_id": "dev-1",
        "name": "Device k1",
        "manufacturer": "Apple",
        "model": None,
        "auto_trust": True,
    }
    assert device_key(obs) == "router:k1"
    assert client_observation({"key": "k2", "hostname": "nas"})["name"] == "nas"


@pytest.mark.asyncio
async def test_commit_reconciles_only_the_sources_it_observed() -> None:
    inventory, _ = _memory_inventory()
    await inventory.async_commit(
        [_radio("z1"), client_observation(_client("k1"))],
        NOW,
        present_sources=["zigbee", ROUTER_SOURCE],
    )
    assert set(inventory.list_devices()[0].keys()) >= {"key", "source", "name"}
    assert {r["key"] for r in inventory.list_devices()} == {"zigbee:z1", "router:k1"}
    # A radio-only commit (router capability absent this run) keeps the client.
    await inventory.async_commit([_radio("z1")], NOW, present_sources=["zigbee"])
    assert {r["key"] for r in inventory.list_devices()} == {"zigbee:z1", "router:k1"}
    # A router-only commit keeps the radio device and, for 30 days, the client
    # it did not see (an integration reload must not turn it into a new one).
    await inventory.async_commit(
        [client_observation(_client("k2"))], NOW, present_sources=[ROUTER_SOURCE]
    )
    assert {r["key"] for r in inventory.list_devices()} == {
        "zigbee:z1",
        "router:k1",
        "router:k2",
    }
    await inventory.async_commit(
        [client_observation(_client("k2"))],
        NOW + timedelta(days=31),
        present_sources=[ROUTER_SOURCE],
    )
    assert {r["key"] for r in inventory.list_devices()} == {"zigbee:z1", "router:k2"}


@pytest.mark.asyncio
async def test_late_auto_trust_withdraws_a_pending_alert() -> None:
    inventory, _ = _memory_inventory()
    await inventory.async_commit(
        [client_observation(_client("k1"))], NOW, present_sources=[ROUTER_SOURCE]
    )
    pending = [client_observation(_client("k1")), client_observation(_client("k2"))]
    await inventory.async_commit(pending, NOW, present_sources=[ROUTER_SOURCE])
    assert inventory.diff(pending, [ROUTER_SOURCE]).new_device_keys == ["router:k2"]
    # The Shelly entry finished setting up: its device now vouches for k2.
    vouched = [
        client_observation(_client("k1")),
        client_observation(_client("k2", auto_trust=True)),
    ]
    assert inventory.diff(vouched, [ROUTER_SOURCE]).new_device_keys == []
    await inventory.async_commit(vouched, NOW, present_sources=[ROUTER_SOURCE])
    row = inventory.row("router:k2")
    assert row is not None
    assert row["trusted"] is True
    assert row["alerted"] is True


@pytest.mark.asyncio
async def test_router_bootstrap_then_new_and_auto_trusted_clients() -> None:
    inventory, saved = _memory_inventory()
    announced = await inventory.async_commit(
        [client_observation(_client("k1"))], NOW, present_sources=[ROUTER_SOURCE]
    )
    assert [(a.source, a.device_count) for a in announced] == [(ROUTER_SOURCE, 1)]
    row = inventory.row("router:k1")
    assert row is not None
    assert row["trusted"] is True
    assert row["first_seen"] == NOW.isoformat()
    later = NOW + timedelta(minutes=5)
    observations = [
        client_observation(_client("k1")),
        client_observation(_client("k2")),
        client_observation(_client("k3", auto_trust=True)),
    ]
    delta = inventory.diff(observations, [ROUTER_SOURCE])
    # The vouched-for client is never new; the other is.
    assert delta.new_device_keys == ["router:k2"]
    await inventory.async_commit(observations, later, present_sources=[ROUTER_SOURCE])
    k2 = inventory.row("router:k2")
    k3 = inventory.row("router:k3")
    assert k2 is not None and k2["trusted"] is False and k2["alerted"] is False  # noqa: PT018
    assert k3 is not None and k3["trusted"] is True and k3["alerted"] is True  # noqa: PT018
    # Still owed until settled; then trusted by key through the service path.
    assert inventory.diff(observations, [ROUTER_SOURCE]).new_device_keys == [
        "router:k2"
    ]
    assert await inventory.async_set_trusted(["router:k2"], trusted=True) == [
        "router:k2"
    ]
    assert inventory.diff(observations, [ROUTER_SOURCE]).new_device_keys == []
    assert inventory.trusted_names(ROUTER_SOURCE) == {
        "device k1",
        "device k2",
        "device k3",
    }
    assert inventory.summary()["by_source"][ROUTER_SOURCE] == {
        "trusted": 3,
        "untrusted": 0,
    }
    # Nothing persisted holds an address.
    blob = json.dumps(saved[-1])
    assert "192.168.1.23" not in blob
    assert "aa:bb" not in blob
    assert "host-k1" not in blob  # the tracker's name is stored, not the hostname


def test_trust_service_schema_accepts_keys_or_devices() -> None:
    assert _SCHEMA({"device_key": "router:k1"}) == {"device_key": ["router:k1"]}
    assert _SCHEMA({"device_id": ["d1"], "device_key": ["router:k1"]})
    with pytest.raises(vol.Invalid):
        _SCHEMA({})


def test_unknown_device_findings_get_the_trust_button() -> None:
    assert "network_unknown_device_joined" in _TRUST_DEVICE_TYPES


# ---------------------------------------------------------------------------
# Engine commit path
# ---------------------------------------------------------------------------


class _Suppression(SuppressionManager):
    def __init__(self) -> None:  # type: ignore[override]
        self._state = SuppressionState()

    @property
    def state(self) -> SuppressionState:  # type: ignore[override]
        return self._state

    @property
    def is_read_only(self) -> bool:  # type: ignore[override]
        return False

    async def async_save(self) -> None:  # type: ignore[override]
        return None


class _Notifier:
    def __init__(self) -> None:
        self.calls: list[Any] = []

    async def async_notify(self, finding, snapshot, explanation) -> None:  # type: ignore[no-untyped-def]
        self.calls.append(finding)


class _Audit:
    async def async_append_finding(self, *_a: Any, **_kw: Any) -> None:
        return None


def _snapshot(
    clients: list[dict[str, Any]] | None,
    new_clients: list[str] | None = None,
    *,
    radio: dict[str, Any] | None = None,
) -> FullStateSnapshot:
    caps: list[str] = []
    network: dict[str, Any] = {
        "capabilities": caps,
        "sources": {},
        "clients": clients or [],
        "posture": {},
        "ha_security": {},
        "counters": {},
    }
    if clients is not None:
        caps.append(CAP_CLIENTS)
        if new_clients is not None:
            caps.append(CAP_NEW_CLIENTS)
            network["new_clients"] = new_clients
    if radio is not None:
        caps.append(radio_cap("devices"))
        network["radio"] = {"capabilities": [], "posture": {}, **radio}
    return validate_snapshot(
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
            "network": network,
        }
    )


def _engine(
    monkeypatch: pytest.MonkeyPatch,
    snapshot: FullStateSnapshot,
    inventory: NetworkInventory,
    *,
    options: dict[str, Any] | None = None,
) -> tuple[SentinelEngine, _Notifier, list[dict[str, Any]]]:
    calls: list[dict[str, Any]] = []
    hass = MagicMock()

    async def _call(
        _domain: str, _service: str, data: dict[str, Any], **_kw: Any
    ) -> None:
        calls.append(data)

    hass.services.async_call = _call

    async def _fake_build(
        _hass: HomeAssistant, *, network: NetworkBuildContext | None = None
    ) -> FullStateSnapshot:
        del network
        return snapshot

    engine_mod = "custom_components.home_generative_agent.sentinel.engine"
    monkeypatch.setattr(f"{engine_mod}.async_build_full_state_snapshot", _fake_build)
    monkeypatch.setattr(f"{engine_mod}.async_dispatcher_send", lambda *_a, **_k: None)
    notifier = _Notifier()
    engine = SentinelEngine(
        hass=hass,
        options={
            "sentinel_cooldown_minutes": 0,
            "sentinel_entity_cooldown_minutes": 0,
            "sentinel_interval_seconds": 60,
            "explain_enabled": False,
            CONF_SENTINEL_NETWORK_UNKNOWN_DEVICE_GRACE_MIN: 0,
            **(options or {}),
        },
        suppression=_Suppression(),
        notifier=cast("SentinelNotifier", notifier),
        audit_store=cast("AuditStore", _Audit()),
        network_inventory=inventory,
    )
    return engine, notifier, calls


@pytest.mark.asyncio
async def test_engine_bootstraps_the_router_source_with_its_own_copy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    inventory, _ = _memory_inventory()
    engine, notifier, calls = _engine(
        monkeypatch, _snapshot([_client("k1"), _client("k2")], []), inventory
    )
    await engine._timed_run()
    assert inventory.is_source_bootstrapped(ROUTER_SOURCE)
    assert notifier.calls == []
    assert len(calls) == 1
    assert "2 network devices" in calls[0]["message"]
    assert "router" in calls[0]["message"]
    assert "never MAC or IP addresses" in calls[0]["message"]
    assert "radio" not in calls[0]["message"].split("recorded", 1)[1][:60]
    await engine._timed_run()
    assert len(calls) == 1


@pytest.mark.asyncio
async def test_engine_delivers_and_settles_a_new_client(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    inventory, _ = _memory_inventory()
    await inventory.async_commit(
        [client_observation(_client("k1"))], NOW, present_sources=[ROUTER_SOURCE]
    )
    engine, notifier, _ = _engine(
        monkeypatch, _snapshot([_client("k1"), _client("k2")], ["k2"]), inventory
    )
    await engine._timed_run()
    assert [f.type for f in notifier.calls] == ["network_unknown_device_joined"]
    assert notifier.calls[0].evidence["device_ids"] == ["router:k2"]
    row = inventory.row("router:k2")
    assert row is not None
    assert row["alerted"] is True
    assert row["trusted"] is False


@pytest.mark.asyncio
async def test_engine_settles_only_the_clients_the_finding_named(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A client still inside its grace period stays owed after another is reported."""
    inventory, _ = _memory_inventory()
    await inventory.async_commit(
        [client_observation(_client("k1"))], NOW, present_sources=[ROUTER_SOURCE]
    )
    old = _client("k2", first_seen=(NOW - timedelta(minutes=10)).isoformat())
    fresh = _client("k3")  # no first_seen: not recorded yet
    engine, notifier, _ = _engine(
        monkeypatch,
        _snapshot([_client("k1"), old, fresh], ["k2", "k3"]),
        inventory,
        options={CONF_SENTINEL_NETWORK_UNKNOWN_DEVICE_GRACE_MIN: 5},
    )
    await engine._timed_run()
    assert [f.evidence["client_keys"] for f in notifier.calls] == [["k2"]]
    k2 = inventory.row("router:k2")
    k3 = inventory.row("router:k3")
    assert k2 is not None and k2["alerted"] is True  # noqa: PT018
    assert k3 is not None and k3["alerted"] is False  # noqa: PT018


@pytest.mark.asyncio
async def test_engine_keeps_a_cooled_down_client_alert_owed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from custom_components.home_generative_agent.sentinel import (  # noqa: PLC0415
        engine as engine_mod,
    )
    from custom_components.home_generative_agent.sentinel.suppression import (  # noqa: PLC0415
        SuppressionDecision,
    )

    inventory, _ = _memory_inventory()
    await inventory.async_commit(
        [client_observation(_client("k1"))], NOW, present_sources=[ROUTER_SOURCE]
    )
    engine, notifier, _ = _engine(
        monkeypatch, _snapshot([_client("k1"), _client("k2")], ["k2"]), inventory
    )
    monkeypatch.setattr(
        engine_mod,
        "should_suppress",
        lambda *_a, **_k: SuppressionDecision(True, "type_cooldown"),  # noqa: FBT003
    )
    await engine._timed_run()
    assert notifier.calls == []
    row = inventory.row("router:k2")
    assert row is not None
    assert row["alerted"] is False  # recorded, alert still owed


@pytest.mark.asyncio
async def test_engine_leaves_router_rows_alone_when_clients_were_not_read(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    inventory, _ = _memory_inventory()
    await inventory.async_commit(
        [client_observation(_client("k1")), _radio("z1")],
        NOW,
        present_sources=[ROUTER_SOURCE, "zigbee"],
    )
    snapshot = _snapshot(
        None,
        radio={
            "devices": [_radio("z1")],
            "new_devices": [],
            "present_sources": ["zigbee"],
        },
    )
    engine, _, _ = _engine(monkeypatch, snapshot, inventory)
    await engine._timed_run()
    assert {r["key"] for r in inventory.list_devices()} == {"router:k1", "zigbee:z1"}

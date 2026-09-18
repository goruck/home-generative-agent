# ruff: noqa: S101
"""Engine capability gating, health-sensor reporting, and the auth-inventory commit."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, Any, cast
from unittest.mock import MagicMock

import pytest

from custom_components.home_generative_agent.const import (
    CONF_SENTINEL_NETWORK_ENABLED,
    SENTINEL_POSTURE_RULE_COOLDOWN_MINUTES,
)
from custom_components.home_generative_agent.core.sentinel_health_sensor import (
    SentinelHealthSensor,
)
from custom_components.home_generative_agent.sentinel.auth_inventory import (
    TOKEN_TYPE_LONG_LIVED,
    AuthInventory,
    ObservedToken,
    ObservedUser,
)
from custom_components.home_generative_agent.sentinel.engine import SentinelEngine
from custom_components.home_generative_agent.sentinel.models import AnomalyFinding
from custom_components.home_generative_agent.sentinel.network_inventory import (
    NetworkInventory,
)
from custom_components.home_generative_agent.sentinel.rules.network_common import (
    NETWORK_RULE_TYPES,
)
from custom_components.home_generative_agent.sentinel.suppression import (
    SuppressionManager,
    SuppressionState,
)
from custom_components.home_generative_agent.snapshot.network import (
    NetworkBuildContext,
    ha_cap,
)
from custom_components.home_generative_agent.snapshot.schema import validate_snapshot

if TYPE_CHECKING:
    from homeassistant.core import HomeAssistant

    from custom_components.home_generative_agent.audit.store import AuditStore
    from custom_components.home_generative_agent.sentinel.notifier import (
        SentinelNotifier,
    )
    from custom_components.home_generative_agent.snapshot.schema import (
        FullStateSnapshot,
        RadioDevice,
    )

NOW = datetime(2026, 9, 7, 12, 0, tzinfo=UTC)


class DummySuppression(SuppressionManager):
    """Suppression manager stub."""

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


class DummyNotifier:
    """Notifier stub recording findings."""

    def __init__(self) -> None:
        self.calls: list[Any] = []

    async def async_notify(self, finding, snapshot, explanation) -> None:  # type: ignore[no-untyped-def]
        self.calls.append(finding)


class DummyAudit:
    """Audit store stub."""

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    async def async_append_finding(  # type: ignore[no-untyped-def]
        self, snapshot, finding, explanation, **kwargs: Any
    ) -> None:
        self.calls.append({"finding": finding, "explanation": explanation, **kwargs})


def _snapshot(
    ha_security: dict[str, Any] | None = None,
    *,
    with_network: bool = True,
    radio: dict[str, Any] | None = None,
) -> FullStateSnapshot:
    ha = ha_security or {}
    caps = sorted(ha_cap(k) for k in ha)
    if radio is not None:
        caps = sorted(
            [*caps, *(f"network.radio.{k}" for k in radio if k != "posture")]
            + [f"network.radio.posture.{k}" for k in radio.get("posture", {})]
        )
    payload: dict[str, Any] = {
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
    }
    if with_network:
        payload["network"] = {
            "capabilities": caps,
            "sources": dict.fromkeys(caps, "ha_native"),
            "clients": [],
            "posture": {},
            "ha_security": ha,
            "counters": {},
        }
        if radio is not None:
            payload["network"]["radio"] = {
                "capabilities": [],
                "devices": [],
                "posture": {},
                **radio,
            }
    return validate_snapshot(payload)


def _engine(  # noqa: PLR0913
    monkeypatch: pytest.MonkeyPatch,
    snapshot: FullStateSnapshot,
    *,
    options: dict[str, Any] | None = None,
    auth_inventory: AuthInventory | None = None,
    hass: Any = None,
    explainer: Any = None,
    network_inventory: Any = None,
) -> tuple[SentinelEngine, list[NetworkBuildContext]]:
    contexts: list[NetworkBuildContext] = []

    async def _fake_build(
        _hass: HomeAssistant, *, network: NetworkBuildContext | None = None
    ) -> FullStateSnapshot:
        assert network is not None
        contexts.append(network)
        return snapshot

    monkeypatch.setattr(
        "custom_components.home_generative_agent.sentinel.engine."
        "async_build_full_state_snapshot",
        _fake_build,
    )
    # _timed_run fires the run-complete dispatcher signal, which needs a real
    # hass; the stub engine has none.
    monkeypatch.setattr(
        "custom_components.home_generative_agent.sentinel.engine.async_dispatcher_send",
        lambda *_args, **_kwargs: None,
    )
    engine = SentinelEngine(
        hass=hass if hass is not None else cast("HomeAssistant", object()),
        options={
            "sentinel_cooldown_minutes": 0,
            "sentinel_entity_cooldown_minutes": 0,
            "sentinel_interval_seconds": 60,
            "explain_enabled": False,
            **(options or {}),
        },
        suppression=DummySuppression(),
        notifier=cast("SentinelNotifier", DummyNotifier()),
        audit_store=cast("AuditStore", DummyAudit()),
        explainer=explainer,
        auth_inventory=auth_inventory,
        network_inventory=network_inventory,
    )
    return engine, contexts


@pytest.mark.asyncio
async def test_rules_skipped_and_reported_when_capabilities_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """With no capabilities every network rule is inactive and listed with its needs."""
    engine, _ = _engine(monkeypatch, _snapshot())
    await engine._timed_run()
    inactive = engine.run_stats["inactive_rules"]
    assert set(inactive) == NETWORK_RULE_TYPES
    assert inactive["ha_failed_logins"] == [ha_cap("failed_login_notification_present")]
    assert engine.run_stats["network_capabilities"] == []
    # Static rules that declare nothing are unaffected by gating.
    assert "unlocked_lock_at_night" not in inactive
    total_static = len(cast("Any", engine)._rules)
    assert engine.run_stats["active_rule_count"] == total_static - len(inactive)


@pytest.mark.asyncio
async def test_rule_runs_when_its_capabilities_are_present(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A rule whose capabilities exist evaluates and dispatches a finding."""
    engine, contexts = _engine(
        monkeypatch, _snapshot({"failed_login_notification_present": True})
    )
    await engine._timed_run()
    inactive = engine.run_stats["inactive_rules"]
    assert "ha_failed_logins" not in inactive
    assert len(inactive) == len(NETWORK_RULE_TYPES) - 1
    notifier = cast("DummyNotifier", cast("Any", engine)._notifier)
    assert [f.type for f in notifier.calls] == ["ha_failed_logins"]
    # The engine passed a live context (network enabled, options attached).
    assert contexts
    assert contexts[0].enabled is True
    assert contexts[0].options is not None


@pytest.mark.asyncio
async def test_pre_v2_snapshot_gates_everything(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A snapshot without a network section is treated as zero capabilities."""
    engine, _ = _engine(monkeypatch, _snapshot(with_network=False))
    await engine._timed_run()
    assert set(engine.run_stats["inactive_rules"]) == NETWORK_RULE_TYPES


@pytest.mark.asyncio
async def test_network_disabled_passes_disabled_context(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The master switch turns the section off at the builder."""
    engine, contexts = _engine(
        monkeypatch, _snapshot(), options={CONF_SENTINEL_NETWORK_ENABLED: False}
    )
    await engine._timed_run()
    assert contexts[0].enabled is False


class _RecordingExplainer:
    def __init__(self) -> None:
        self.calls: list[str] = []

    async def async_explain(self, finding: Any) -> str:
        self.calls.append(finding.type)
        return "model prose"


@pytest.mark.asyncio
async def test_explainer_skipped_for_security_copy_findings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Network findings never reach the explainer even with explanations on."""
    explainer = _RecordingExplainer()
    engine, _ = _engine(
        monkeypatch,
        _snapshot({"failed_login_notification_present": True}),
        options={"explain_enabled": True},
        explainer=explainer,
    )
    await engine._timed_run()
    notifier = cast("DummyNotifier", cast("Any", engine)._notifier)
    assert [f.type for f in notifier.calls] == ["ha_failed_logins"]
    assert explainer.calls == []
    # The audit row carries no explanation either.
    audit = cast("DummyAudit", cast("Any", engine)._audit_store)
    assert audit.calls[-1].get("explanation") is None
    # An ordinary finding still gets one.
    ordinary = AnomalyFinding(
        anomaly_id="plain",
        type="unlocked_lock_at_night",
        severity="low",
        confidence=0.9,
        triggering_entities=["lock.front"],
        evidence={},
        suggested_actions=[],
        is_sensitive=False,
    )
    assert (
        cast("Any", engine)._explainer_for(explain_enabled=True, finding=ordinary)
        is explainer
    )
    assert (
        cast("Any", engine)._explainer_for(explain_enabled=False, finding=ordinary)
        is None
    )


@pytest.mark.asyncio
async def test_posture_rule_cooldown_floor_applies(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A posture finding is not re-notified within its one-day floor."""
    engine, _ = _engine(
        monkeypatch, _snapshot({"failed_login_notification_present": True})
    )
    floor = cast("Any", engine)._rule_cooldown_floor("ha_failed_logins")
    assert floor == timedelta(minutes=SENTINEL_POSTURE_RULE_COOLDOWN_MINUTES)
    assert cast("Any", engine)._rule_cooldown_floor(
        "unlocked_lock_at_night"
    ) == timedelta(0)
    await engine._timed_run()
    # The first dispatch registers a pending prompt (the finding carries a
    # suggested action); clear it so the second run reaches the cooldown
    # check, where the configured 0-minute cooldown loses to the floor.
    cast("Any", engine)._suppression.state.pending_prompts.clear()
    await engine._timed_run()
    notifier = cast("DummyNotifier", cast("Any", engine)._notifier)
    assert len(notifier.calls) == 1
    audit = cast("DummyAudit", cast("Any", engine)._audit_store)
    assert audit.calls[-1]["suppression_reason_code"] == "type_cooldown"


@pytest.mark.asyncio
async def test_auth_inventory_bootstrap_commits_and_notifies_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """First run: observation collected, committed as bootstrap, one notification."""
    inventory = AuthInventory(MagicMock())
    store = MagicMock()
    saved: list[dict[str, Any]] = []

    async def _save(data: dict[str, Any]) -> None:
        saved.append(data)

    store.async_save = _save
    inventory._store = store  # type: ignore[assignment]
    observation = [
        ObservedUser(
            user_id="u1",
            name="Lindo",
            is_admin=True,
            is_active=True,
            system_generated=False,
            tokens=(
                ObservedToken(
                    "t1", "u1", TOKEN_TYPE_LONG_LIVED, "api", None, None, None
                ),
            ),
        )
    ]

    async def _collect(_hass: Any, _pseudonymizer: Any) -> list[ObservedUser]:
        return observation

    monkeypatch.setattr(
        "custom_components.home_generative_agent.sentinel.engine."
        "async_collect_auth_observation",
        _collect,
    )
    hass = MagicMock()
    service_calls: list[tuple[str, str, dict[str, Any]]] = []

    async def _call(
        domain: str, service: str, data: dict[str, Any], **_kw: Any
    ) -> None:
        service_calls.append((domain, service, data))

    hass.services.async_call = _call
    engine, contexts = _engine(
        monkeypatch, _snapshot(), auth_inventory=inventory, hass=hass
    )

    await engine._timed_run()
    assert contexts[0].auth_observation == observation
    assert inventory.is_bootstrapped
    assert len(saved) == 1
    assert len(service_calls) == 1
    assert "1 administrator" in service_calls[0][2]["message"]

    await engine._timed_run()
    # Nothing changed, so the inventory is not rewritten; no second
    # bootstrap notification either.
    assert len(saved) == 1
    assert len(service_calls) == 1


@pytest.mark.asyncio
async def test_health_sensor_exposes_inactive_rules() -> None:
    """The health sensor mirrors inactive_rules and the capability list."""
    hass = MagicMock()
    hass.async_create_task = MagicMock()
    engine = MagicMock()
    engine.run_stats = {
        "inactive_rules": {"ha_failed_logins": ["network.ha_security.x"]},
        "network_capabilities": ["network.ha_security.y"],
        "learned_suppressions_count": 0,
    }
    engine.learned_suppressions_count = 0
    engine.cyclical_entities_gated_count = 0
    sensor = SentinelHealthSensor(
        hass=hass,
        options={"sentinel_enabled": True},
        audit_store=None,
        sentinel=engine,
        entry_id="e1",
    )
    sensor.hass = hass
    sensor.async_write_ha_state = MagicMock()  # type: ignore[method-assign]
    await sensor._async_refresh()
    attrs = sensor.extra_state_attributes or {}
    assert attrs["inactive_rules"] == {"ha_failed_logins": ["network.ha_security.x"]}
    assert attrs["inactive_rule_count"] == 1
    assert attrs["network_capabilities"] == ["network.ha_security.y"]


@pytest.mark.asyncio
async def test_auth_inventory_commit_failure_does_not_end_the_run(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A raising commit is contained; the run still writes its stats."""
    inventory = MagicMock()
    inventory.is_bootstrapped = True
    calls = {"n": 0}

    async def _commit(*_a: Any, **_k: Any) -> bool:
        calls["n"] += 1
        if calls["n"] == 1:
            msg = "corrupt row"
            raise AttributeError(msg)
        return False

    inventory.async_commit = _commit
    inventory.diff = MagicMock(return_value=MagicMock(bootstrap=False))

    async def _collect(_hass: Any, _p: Any) -> list[ObservedUser]:
        return []

    monkeypatch.setattr(
        "custom_components.home_generative_agent.sentinel.engine."
        "async_collect_auth_observation",
        _collect,
    )
    engine, _ = _engine(
        monkeypatch, _snapshot(), auth_inventory=cast("AuthInventory", inventory)
    )
    await engine._timed_run()
    assert engine.run_stats["last_run_end"]
    await engine._timed_run()
    assert calls["n"] == 2


@pytest.mark.asyncio
async def test_suppressed_auth_change_is_held_back_and_refires(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A snoozed new-token alert is not baselined; it fires once the snooze ends."""
    from datetime import timedelta as _td  # noqa: PLC0415

    from custom_components.home_generative_agent.sentinel.auth_inventory import (  # noqa: PLC0415
        AuthDelta,
    )

    inventory = MagicMock()
    inventory.is_bootstrapped = True
    delta = AuthDelta(new_long_lived_tokens=["script"], new_token_ids=["t2"])
    inventory.diff = MagicMock(return_value=delta)
    commits: list[dict[str, Any]] = []

    async def _commit(*_a: Any, **kwargs: Any) -> bool:
        commits.append(kwargs)
        return False

    inventory.async_commit = _commit
    observation = [
        ObservedUser("u1", "Lindo", True, True, False, ())  # noqa: FBT003
    ]

    async def _collect(_hass: Any, _p: Any) -> list[ObservedUser]:
        return observation

    monkeypatch.setattr(
        "custom_components.home_generative_agent.sentinel.engine."
        "async_collect_auth_observation",
        _collect,
    )
    snapshot = _snapshot({"new_admin_users": [], "new_long_lived_tokens": ["script"]})
    engine, _ = _engine(monkeypatch, snapshot, auth_inventory=inventory)
    state = cast("Any", engine)._suppression.state
    # Snooze the type: the finding is produced but never delivered.
    state.snoozed_until["ha_new_admin_or_token"] = {
        "until": (NOW + _td(days=365)).isoformat()
    }
    await engine._timed_run()
    notifier = cast("DummyNotifier", cast("Any", engine)._notifier)
    assert notifier.calls == []
    assert commits[-1]["hold_back"] is delta

    state.snoozed_until.clear()
    await engine._timed_run()
    assert [f.type for f in notifier.calls] == ["ha_new_admin_or_token"]
    assert commits[-1]["hold_back"] is None


@pytest.mark.asyncio
async def test_posture_memory_feeds_the_next_run_and_survives_a_missing_value(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The engine remembers the UPnP posture keys between runs, in process only."""
    snapshot = _snapshot()
    posture = cast("dict[str, Any]", snapshot.get("network", {}).get("posture"))
    posture.update({"public_ip_key": "aaaa0000", "upnp_port_mapping_count": 2})
    engine, contexts = _engine(monkeypatch, snapshot)
    await engine._timed_run()
    assert contexts[0].previous_posture is None
    await engine._timed_run()
    assert contexts[1].previous_posture == {
        "public_ip_key": "aaaa0000",
        "upnp_port_mapping_count": 2,
    }
    # A value the snapshot no longer carries (sensor unavailable) is kept.
    posture.pop("upnp_port_mapping_count")
    posture["public_ip_key"] = "bbbb0000"
    await engine._timed_run()
    await engine._timed_run()
    assert contexts[3].previous_posture == {
        "public_ip_key": "bbbb0000",
        "upnp_port_mapping_count": 2,
    }


@pytest.mark.asyncio
async def test_undelivered_public_ip_change_keeps_the_previous_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A change alert stopped by a cooldown is compared against again next run."""
    from datetime import timedelta as _td  # noqa: PLC0415

    from custom_components.home_generative_agent.snapshot.network import (  # noqa: PLC0415
        posture_cap,
    )

    snapshot = _snapshot()
    network = cast("dict[str, Any]", snapshot.get("network"))
    network["capabilities"] = [posture_cap("public_ip_changed")]
    posture = cast("dict[str, Any]", network["posture"])
    posture.update(
        {
            "public_ip_changed": True,
            "public_ip_key": "bbbb0000",
            "public_ip_previous_key": "aaaa0000",
            "public_ip_entity_id": "sensor.gw_ip",
        }
    )
    engine, contexts = _engine(
        monkeypatch, snapshot, options={"sentinel_cooldown_minutes": 60}
    )
    notifier = cast("DummyNotifier", cast("Any", engine)._notifier)
    await engine._timed_run()
    assert [f.type for f in notifier.calls] == ["network_public_ip_changed"]
    # Delivered: the new key is remembered.
    await engine._timed_run()
    remembered = {"public_ip_key": "bbbb0000", "public_ip_entity_id": "sensor.gw_ip"}
    assert contexts[1].previous_posture == remembered
    # Second change inside the type cooldown: produced, not delivered, so the
    # remembered key stays and the change is re-detected on a later run.
    posture["public_ip_key"] = "cccc0000"
    await engine._timed_run()
    assert len(notifier.calls) == 1
    await engine._timed_run()
    assert contexts[3].previous_posture == remembered
    # A snooze settles it: the user chose not to hear it, so it is baselined.
    state = cast("Any", engine)._suppression.state
    state.snoozed_until["network_public_ip_changed"] = {
        "until": (NOW + _td(days=365)).isoformat()
    }
    await engine._timed_run()
    await engine._timed_run()
    assert contexts[5].previous_posture == {**remembered, "public_ip_key": "cccc0000"}


@pytest.mark.asyncio
async def test_posture_memory_survives_a_restart_through_the_inventory(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A new engine over the same inventory store starts with the last baseline."""
    inventory = _memory_inventory()
    snapshot = _snapshot()
    posture = cast("dict[str, Any]", snapshot.get("network", {}).get("posture"))
    posture.update({"public_ip_key": "aaaa0000", "public_ip_entity_id": "sensor.gw_ip"})
    engine, _ = _engine(monkeypatch, snapshot, network_inventory=inventory)
    await engine._timed_run()
    assert inventory.posture_memory == {
        "public_ip_key": "aaaa0000",
        "public_ip_entity_id": "sensor.gw_ip",
    }
    restarted, contexts = _engine(monkeypatch, snapshot, network_inventory=inventory)
    await restarted._timed_run()
    assert contexts[0].previous_posture == inventory.posture_memory


@pytest.mark.asyncio
async def test_undelivered_port_mapping_increase_keeps_count_and_entity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The count and the entity it came from are held back together."""
    from custom_components.home_generative_agent.snapshot.network import (  # noqa: PLC0415
        posture_cap,
    )

    snapshot = _snapshot()
    network = cast("dict[str, Any]", snapshot.get("network"))
    network["capabilities"] = [posture_cap("upnp_port_mappings_added")]
    posture = cast("dict[str, Any]", network["posture"])
    posture.update(
        {
            "upnp_port_mappings_added": 1,
            "upnp_port_mapping_count": 4,
            "upnp_port_mapping_entity_id": "sensor.gw_pm",
        }
    )
    engine, contexts = _engine(
        monkeypatch, snapshot, options={"sentinel_cooldown_minutes": 60}
    )
    notifier = cast("DummyNotifier", cast("Any", engine)._notifier)
    await engine._timed_run()
    assert [f.type for f in notifier.calls] == ["network_upnp_port_mapping_added"]
    posture["upnp_port_mapping_count"] = 5
    posture["upnp_port_mapping_entity_id"] = "sensor.gw2_pm"
    await engine._timed_run()  # inside the cooldown: held
    assert len(notifier.calls) == 1
    await engine._timed_run()
    assert contexts[2].previous_posture == {
        "upnp_port_mapping_count": 4,
        "upnp_port_mapping_entity_id": "sensor.gw_pm",
    }


def test_correlator_keeps_posture_findings_out_of_compounds() -> None:
    """A posture finding on lock.front never swallows a live event on lock.front."""
    from custom_components.home_generative_agent.sentinel.correlator import (  # noqa: PLC0415
        SentinelCorrelator,
    )
    from custom_components.home_generative_agent.sentinel.models import (  # noqa: PLC0415
        AnomalyFinding,
        CompoundFinding,
    )
    from custom_components.home_generative_agent.sentinel.rules.network_common import (  # noqa: PLC0415
        make_finding,
    )

    posture = make_finding(
        "ha_sensitive_entity_exposed_without_pin",
        severity="high",
        evidence={"exposures": {"conversation": ["lock.front"]}},
        summary="exposed",
        suggested_actions=["Unexpose it"],
        triggering_entities=["lock.front"],
    )
    live = AnomalyFinding(
        anomaly_id="live1",
        type="unlocked_lock_at_night",
        severity="high",
        confidence=0.7,
        triggering_entities=["lock.front"],
        evidence={"area": "front"},
        suggested_actions=["lock_entity"],
        is_sensitive=True,
    )
    out = SentinelCorrelator().correlate([posture, live])
    assert not any(isinstance(item, CompoundFinding) for item in out)
    assert {f.type for f in out if isinstance(f, AnomalyFinding)} == {
        "ha_sensitive_entity_exposed_without_pin",
        "unlocked_lock_at_night",
    }


# ---------------------------------------------------------------------------
# Radio device inventory and the permit-join wake-up
# ---------------------------------------------------------------------------


def _radio_device(device_id: str, name: str = "Sensor") -> RadioDevice:
    return {
        "device_id": device_id,
        "protocol": "zigbee",
        "platform": "zha",
        "name": name,
        "is_security_device": False,
        "manufacturer": None,
        "model": None,
    }


def _service_recorder() -> tuple[MagicMock, list[tuple[str, str, dict[str, Any]]]]:
    hass = MagicMock()
    calls: list[tuple[str, str, dict[str, Any]]] = []

    async def _call(
        domain: str, service: str, data: dict[str, Any], **_kw: Any
    ) -> None:
        calls.append((domain, service, data))

    hass.services.async_call = _call
    return hass, calls


def _memory_inventory() -> NetworkInventory:
    inventory = NetworkInventory(MagicMock())
    store = MagicMock()

    async def _save(_data: dict[str, Any]) -> None:
        return None

    store.async_save = _save
    inventory._store = store  # type: ignore[assignment]
    return inventory


@pytest.mark.asyncio
async def test_network_inventory_bootstrap_announces_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    inventory = _memory_inventory()
    hass, calls = _service_recorder()
    devices = [_radio_device("a"), _radio_device("b")]
    snapshot = _snapshot(
        radio={"devices": devices, "new_devices": [], "present_sources": ["zigbee"]}
    )
    engine, contexts = _engine(
        monkeypatch, snapshot, hass=hass, network_inventory=inventory
    )
    await engine._timed_run()
    assert contexts[0].network_inventory is inventory
    assert inventory.is_source_bootstrapped("zigbee")
    assert len(calls) == 1
    assert "2 Zigbee devices" in calls[0][2]["message"]
    await engine._timed_run()
    assert len(calls) == 1


@pytest.mark.asyncio
async def test_new_device_alert_postponed_by_cooldown_settled_by_snooze(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A cooldown keeps the alert owed; a snooze settles it; both record the device."""
    from datetime import timedelta as _td  # noqa: PLC0415

    from custom_components.home_generative_agent.sentinel import (  # noqa: PLC0415
        engine as engine_mod,
    )
    from custom_components.home_generative_agent.sentinel.suppression import (  # noqa: PLC0415
        SuppressionDecision,
    )

    inventory = _memory_inventory()
    await inventory.async_commit([_radio_device("a")], NOW)
    devices = [_radio_device("a"), _radio_device("b", "New Plug")]
    snapshot = _snapshot(
        radio={
            "devices": devices,
            "new_devices": ["zigbee:b"],
            "present_sources": ["zigbee"],
        }
    )
    hass, _calls = _service_recorder()
    engine, _ = _engine(monkeypatch, snapshot, hass=hass, network_inventory=inventory)
    notifier = cast("DummyNotifier", cast("Any", engine)._notifier)
    real_should_suppress = engine_mod.should_suppress
    monkeypatch.setattr(
        engine_mod,
        "should_suppress",
        lambda *_a, **_k: SuppressionDecision(True, "type_cooldown"),  # noqa: FBT003
    )
    await engine._timed_run()
    assert notifier.calls == []
    # Recorded (trustable, counted) but the alert is still owed.
    assert inventory.summary()["untrusted"] == 1
    assert inventory.diff(devices, []).new_device_keys == ["zigbee:b"]

    monkeypatch.setattr(engine_mod, "should_suppress", real_should_suppress)
    state = cast("Any", engine)._suppression.state
    state.snoozed_until["radio_new_device_joined"] = {
        "until": (NOW + _td(days=365)).isoformat()
    }
    await engine._timed_run()
    assert notifier.calls == []
    # The user asked not to hear about new devices: settled, not re-owed.
    assert inventory.diff(devices, []).new_device_keys == []


@pytest.mark.asyncio
async def test_delivered_new_device_alert_is_settled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    inventory = _memory_inventory()
    await inventory.async_commit([_radio_device("a")], NOW)
    devices = [_radio_device("a"), _radio_device("b", "New Plug")]
    snapshot = _snapshot(
        radio={"devices": devices, "new_devices": ["zigbee:b"]},
    )
    hass, _calls = _service_recorder()
    engine, _ = _engine(monkeypatch, snapshot, hass=hass, network_inventory=inventory)
    await engine._timed_run()
    notifier = cast("DummyNotifier", cast("Any", engine)._notifier)
    assert [f.type for f in notifier.calls] == ["radio_new_device_joined"]
    assert inventory.diff(devices, []).new_device_keys == []


@pytest.mark.asyncio
async def test_failed_registry_read_never_wipes_the_inventory(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Radio posture without the device capability commits nothing."""
    inventory = _memory_inventory()
    await inventory.async_commit([_radio_device("a"), _radio_device("b")], NOW)
    snapshot = _snapshot(radio={"posture": {"zwave_inclusion_active": False}})
    network = cast("Any", snapshot)["network"]
    assert "network.radio.devices" not in network["capabilities"]
    assert network["radio"]["devices"] == []
    engine, _ = _engine(monkeypatch, snapshot, network_inventory=inventory)
    await engine._timed_run()
    assert inventory.summary()["device_count"] == 2


@pytest.mark.asyncio
async def test_network_inventory_commit_failure_does_not_end_the_run(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    inventory = MagicMock()

    async def _commit(*_a: Any, **_k: Any) -> list[Any]:
        msg = "corrupt"
        raise KeyError(msg)

    inventory.async_commit = _commit
    snapshot = _snapshot(radio={"devices": [_radio_device("a")], "new_devices": []})
    engine, _ = _engine(monkeypatch, snapshot, network_inventory=inventory)
    await engine._timed_run()
    assert engine.run_stats["last_run_end"]


@pytest.mark.asyncio
async def test_permit_join_switch_turning_on_wakes_the_engine(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Matched on the event by registry unique id, with no snapshot needed first."""
    from types import SimpleNamespace  # noqa: PLC0415

    from homeassistant.core import State  # noqa: PLC0415

    switch = "switch.zigbee2mqtt_bridge_permit_join"
    entries = {
        switch: SimpleNamespace(
            platform="mqtt",
            domain="switch",
            unique_id="bridge_0x00124b001ca1b801_permit_join_zigbee2mqtt",
        ),
        "switch.kitchen": SimpleNamespace(
            platform="mqtt", domain="switch", unique_id="kitchen_permit_join"
        ),
    }
    monkeypatch.setattr(
        "custom_components.home_generative_agent.sentinel.engine.er.async_get",
        lambda _hass: SimpleNamespace(async_get=entries.get),
    )
    engine, _ = _engine(monkeypatch, _snapshot())
    enqueued: list[str] = []
    monkeypatch.setattr(
        cast("Any", engine)._trigger_scheduler,
        "enqueue",
        lambda record: enqueued.append(record.anomaly_type),
    )

    def _event(entity_id: str, state: str) -> Any:
        return MagicMock(
            data={"entity_id": entity_id, "new_state": State(entity_id, state)}
        )

    engine._on_state_changed(_event(switch, "on"))
    engine._on_state_changed(_event(switch, "off"))
    engine._on_state_changed(_event("switch.kitchen", "on"))
    engine._on_state_changed(_event("switch.unregistered", "on"))
    assert enqueued == ["zigbee_permit_join_open"]


@pytest.mark.asyncio
async def test_on_demand_audit_reports_inventory_counts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    inventory = _memory_inventory()
    await inventory.async_commit([_radio_device("a")], NOW)
    await inventory.async_commit([_radio_device("a"), _radio_device("b")], NOW)
    engine, _ = _engine(
        monkeypatch,
        _snapshot(radio={"devices": [], "new_devices": []}),
        network_inventory=inventory,
    )
    report = await engine.async_audit_network()
    assert report["inventory"] == {
        "trusted": 1,
        "untrusted": 1,
        "by_source": {"zigbee": {"trusted": 1, "untrusted": 1}},
    }
    # The audit never commits the inventory.
    assert inventory.diff([_radio_device("c")], []).new_device_keys == ["zigbee:c"]

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

    def set_silence_check(self, check: object) -> None:
        """Accept the engine's level-0 check; unused by this stub."""
        del check

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
    # Delivered: the new key is remembered, so the snapshot stops reporting
    # the change (the builder compares against that memory).
    posture["public_ip_changed"] = False
    await engine._timed_run()
    remembered = {"public_ip_key": "bbbb0000", "public_ip_entity_id": "sensor.gw_ip"}
    assert contexts[1].previous_posture == remembered
    # Second change inside the type cooldown: produced, not delivered, so the
    # remembered key stays and the change is re-detected on a later run.
    posture.update({"public_ip_changed": True, "public_ip_key": "cccc0000"})
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


@pytest.mark.asyncio
async def test_a_new_device_seen_at_level_zero_alerts_when_the_level_rises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    Level 0 holds a change alert; it is not lost (#783).

    The device is owed until an alert reaches the user, as a new admin or
    token already is, so it alerts once the level is raised.
    """
    inventory = _memory_inventory()
    await inventory.async_commit([_radio_device("a")], NOW)
    devices = [_radio_device("a"), _radio_device("b", "New Plug")]
    snapshot = _snapshot(radio={"devices": devices, "new_devices": ["zigbee:b"]})
    hass, _calls = _service_recorder()
    engine, _ = _engine(
        monkeypatch,
        snapshot,
        hass=hass,
        network_inventory=inventory,
        options={"sentinel_autonomy_level": 0},
    )
    engine._entry_id = "passive-entry"
    await engine._timed_run()
    notifier = cast("DummyNotifier", cast("Any", engine)._notifier)
    assert notifier.calls == []
    assert inventory.diff(devices, []).new_device_keys == ["zigbee:b"]

    engine._options["sentinel_autonomy_level"] = 1
    await engine._timed_run()
    assert [f.type for f in notifier.calls] == ["radio_new_device_joined"]
    assert inventory.diff(devices, []).new_device_keys == []


@pytest.mark.asyncio
async def test_a_held_change_is_recorded_once_not_every_cycle(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """While it is owed at level 0, a change is audited once per cooldown."""
    inventory = _memory_inventory()
    await inventory.async_commit([_radio_device("a")], NOW)
    devices = [_radio_device("a"), _radio_device("b", "New Plug")]
    snapshot = _snapshot(radio={"devices": devices, "new_devices": ["zigbee:b"]})
    hass, _calls = _service_recorder()
    engine, _ = _engine(
        monkeypatch,
        snapshot,
        hass=hass,
        network_inventory=inventory,
        options={"sentinel_autonomy_level": 0, "sentinel_cooldown_minutes": 30},
    )
    engine._entry_id = "passive-entry"
    audit = cast("DummyAudit", cast("Any", engine)._audit_store)
    await engine._timed_run()
    await engine._timed_run()
    held = [
        c
        for c in audit.calls
        if c.get("suppression_reason_code") == "autonomy_passive"
        and c["finding"].type == "radio_new_device_joined"
    ]
    assert len(held) == 1


def _public_ip_snapshot() -> tuple[FullStateSnapshot, dict[str, Any]]:
    """Return a snapshot reporting a public IP change, and its posture dict."""
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
    return snapshot, posture


def _upnp_snapshot() -> tuple[FullStateSnapshot, dict[str, Any]]:
    """Return a snapshot reporting one new UPnP port mapping, and its posture."""
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
            "upnp_port_mapping_count": 3,
            "upnp_port_mapping_previous_count": 2,
            "upnp_port_mapping_entity_id": "sensor.gw_upnp",
        }
    )
    return snapshot, posture


def _owed(engine: SentinelEngine) -> dict[str, list[dict[str, Any]]]:
    return cast("Any", engine)._suppression.state.owed_change_alerts


def _silenced(
    monkeypatch: pytest.MonkeyPatch, snapshot: FullStateSnapshot, **kwargs: Any
) -> tuple[SentinelEngine, DummyNotifier]:
    engine, _ = _engine(
        monkeypatch, snapshot, options={"sentinel_autonomy_level": 0}, **kwargs
    )
    engine._entry_id = "passive-entry"
    return engine, cast("DummyNotifier", cast("Any", engine)._notifier)


def _radio_snapshot() -> tuple[FullStateSnapshot, dict[str, Any]]:
    radio: dict[str, Any] = {
        "devices": [_radio_device("a"), _radio_device("b", "New Plug")],
        "new_devices": ["zigbee:b"],
    }
    snapshot = _snapshot(radio=radio)
    return snapshot, cast("dict[str, Any]", cast("Any", snapshot)["network"]["radio"])


async def _radio_engine(
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[SentinelEngine, DummyNotifier, dict[str, Any]]:
    inventory = _memory_inventory()
    await inventory.async_commit([_radio_device("a")], NOW)
    snapshot, section = _radio_snapshot()
    hass, _calls = _service_recorder()
    engine, notifier = _silenced(
        monkeypatch, snapshot, hass=hass, network_inventory=inventory
    )
    return engine, notifier, section


@pytest.mark.asyncio
async def test_a_change_undone_at_level_zero_still_alerts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A public IP that changes and changes back while silenced is told (#799)."""
    snapshot, posture = _public_ip_snapshot()
    engine, notifier = _silenced(monkeypatch, snapshot)
    await engine._timed_run()
    assert notifier.calls == []
    assert list(_owed(engine)) == ["network_public_ip_changed"]

    # Changed back before the level rises: the rule no longer fires.
    posture.update({"public_ip_changed": False, "public_ip_key": "aaaa0000"})
    await engine._timed_run()
    assert notifier.calls == []

    engine._options["sentinel_autonomy_level"] = 1
    await engine._timed_run()
    assert [f.type for f in notifier.calls] == ["network_public_ip_changed"]
    summary = notifier.calls[0].evidence["summary"]
    # The note leads: a push keeps only the start of a long summary.
    assert summary.startswith("Seen while Sentinel was at autonomy level 0 (")
    assert "it may have been undone since. Your home's public IP" in summary
    assert notifier.calls[0].evidence["seen_while_silenced"] is True
    assert _owed(engine) == {}

    # Settled: it is not sent again.
    await engine._timed_run()
    assert len(notifier.calls) == 1


@pytest.mark.asyncio
async def test_a_port_mapping_opened_and_closed_at_level_zero_alerts_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Several UPnP changes while silenced are told in one alert (#799)."""
    snapshot, posture = _upnp_snapshot()
    engine, notifier = _silenced(monkeypatch, snapshot)
    await engine._timed_run()
    posture.update({"upnp_port_mappings_added": 2, "upnp_port_mapping_count": 4})
    await engine._timed_run()
    # More mappings from the same baseline include the fewer: one is kept.
    assert len(_owed(engine)["network_upnp_port_mapping_added"]) == 1
    posture.update({"upnp_port_mappings_added": 0, "upnp_port_mapping_count": 2})
    await engine._timed_run()

    engine._options["sentinel_autonomy_level"] = 1
    await engine._timed_run()
    await engine._timed_run()
    assert [f.type for f in notifier.calls] == ["network_upnp_port_mapping_added"]
    summary = notifier.calls[0].evidence["summary"]
    assert "2 new port mappings were opened" in summary
    assert "(4 mappings open in total)" in summary
    assert _owed(engine) == {}


@pytest.mark.asyncio
async def test_a_change_still_there_when_the_level_rises_alerts_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The live finding carries an owed change that is still current; no repeat."""
    snapshot, _posture = _public_ip_snapshot()
    engine, notifier = _silenced(monkeypatch, snapshot)
    await engine._timed_run()
    engine._options["sentinel_autonomy_level"] = 1
    await engine._timed_run()
    assert [f.type for f in notifier.calls] == ["network_public_ip_changed"]
    assert "seen_while_silenced" not in notifier.calls[0].evidence
    assert _owed(engine) == {}


@pytest.mark.asyncio
async def test_an_owed_change_stays_while_its_live_alert_waits(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A live alert held by a cooldown settles nothing; the owed one waits too."""
    snapshot, _posture = _public_ip_snapshot()
    engine, notifier = _silenced(monkeypatch, snapshot)
    await engine._timed_run()
    state = cast("Any", engine)._suppression.state
    # The engine runs on the real clock.
    state.last_by_type["network_public_ip_changed"] = datetime.now(UTC).isoformat()
    engine._options["sentinel_autonomy_level"] = 1
    engine._options["sentinel_cooldown_minutes"] = 60
    await engine._timed_run()
    assert notifier.calls == []
    assert list(_owed(engine)) == ["network_public_ip_changed"]

    state.last_by_type.clear()
    await engine._timed_run()
    assert [f.type for f in notifier.calls] == ["network_public_ip_changed"]
    assert "seen_while_silenced" not in notifier.calls[0].evidence
    assert _owed(engine) == {}


@pytest.mark.asyncio
async def test_an_owed_change_stays_owed_until_it_can_be_sent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Quiet hours delay an owed replay without auditing every wait."""
    from custom_components.home_generative_agent.sentinel import (  # noqa: PLC0415
        engine as engine_mod,
    )
    from custom_components.home_generative_agent.sentinel.suppression import (  # noqa: PLC0415
        SuppressionDecision,
    )

    snapshot, posture = _public_ip_snapshot()
    engine, notifier = _silenced(monkeypatch, snapshot)
    await engine._timed_run()
    posture.update({"public_ip_changed": False, "public_ip_key": "aaaa0000"})
    real = engine_mod.should_suppress
    monkeypatch.setattr(
        engine_mod,
        "should_suppress",
        lambda *_a, **_k: SuppressionDecision(True, "quiet_hours"),  # noqa: FBT003
    )
    engine._options["sentinel_autonomy_level"] = 1
    audit = cast("DummyAudit", cast("Any", engine)._audit_store)
    before = len(audit.calls)
    await engine._timed_run()
    await engine._timed_run()
    assert notifier.calls == []
    assert len(audit.calls) == before
    assert list(_owed(engine)) == ["network_public_ip_changed"]

    monkeypatch.setattr(engine_mod, "should_suppress", real)
    await engine._timed_run()
    assert [f.type for f in notifier.calls] == ["network_public_ip_changed"]
    assert _owed(engine) == {}


@pytest.mark.asyncio
async def test_a_replay_does_not_wait_out_a_cooldown(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A live alert's cooldown would otherwise hold a replay back for good."""
    snapshot, posture = _public_ip_snapshot()
    engine, notifier = _silenced(monkeypatch, snapshot)
    await engine._timed_run()
    posture.update({"public_ip_changed": False, "public_ip_key": "aaaa0000"})
    _cooling(engine, "network_public_ip_changed")
    engine._options["sentinel_autonomy_level"] = 1
    await engine._timed_run()
    assert [f.type for f in notifier.calls] == ["network_public_ip_changed"]
    assert _owed(engine) == {}


@pytest.mark.asyncio
async def test_a_change_a_cooldown_stopped_at_level_zero_is_still_owed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A cooldown left from level 1 does not keep a level-0 change from being owed."""
    snapshot, posture = _public_ip_snapshot()
    engine, notifier = _silenced(monkeypatch, snapshot)
    state = cast("Any", engine)._suppression.state
    state.last_by_type["network_public_ip_changed"] = datetime.now(UTC).isoformat()
    engine._options["sentinel_cooldown_minutes"] = 60
    await engine._timed_run()
    assert list(_owed(engine)) == ["network_public_ip_changed"]

    posture.update({"public_ip_changed": False, "public_ip_key": "aaaa0000"})
    state.last_by_type.clear()
    engine._options["sentinel_autonomy_level"] = 1
    await engine._timed_run()
    assert [f.type for f in notifier.calls] == ["network_public_ip_changed"]
    assert notifier.calls[0].evidence["seen_while_silenced"] is True


@pytest.mark.asyncio
async def test_a_snoozed_change_at_level_zero_is_not_owed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    snapshot, _posture = _public_ip_snapshot()
    engine, _notifier = _silenced(monkeypatch, snapshot)
    state = cast("Any", engine)._suppression.state
    state.snoozed_until["network_public_ip_changed"] = {"until": "permanent"}
    await engine._timed_run()
    assert _owed(engine) == {}


@pytest.mark.asyncio
async def test_an_owed_change_the_user_snoozed_is_settled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A snooze is the user's answer: the owed change is not kept."""
    snapshot, posture = _public_ip_snapshot()
    engine, notifier = _silenced(monkeypatch, snapshot)
    await engine._timed_run()
    posture.update({"public_ip_changed": False, "public_ip_key": "aaaa0000"})
    state = cast("Any", engine)._suppression.state
    state.snoozed_until["network_public_ip_changed"] = {"until": "permanent"}
    engine._options["sentinel_autonomy_level"] = 1
    await engine._timed_run()
    assert notifier.calls == []
    assert _owed(engine) == {}


@pytest.mark.asyncio
async def test_a_replay_skips_triage(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """What was owed from level 0 is not the model's to drop."""
    snapshot, posture = _public_ip_snapshot()
    engine, notifier = _silenced(monkeypatch, snapshot)
    await engine._timed_run()
    posture.update({"public_ip_changed": False, "public_ip_key": "aaaa0000"})
    triage = MagicMock()

    async def _suppress(*_a: Any, **_k: Any) -> Any:
        return MagicMock(
            decision="suppress", reason_code="llm_suppress", triage_confidence=0.9
        )

    triage.triage = _suppress
    cast("Any", engine)._triage_service = triage
    engine._options["sentinel_autonomy_level"] = 1
    await engine._timed_run()
    assert [f.type for f in notifier.calls] == ["network_public_ip_changed"]
    assert _owed(engine) == {}


@pytest.mark.asyncio
async def test_an_owed_change_waits_while_its_rule_cannot_run(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A rule that did not run may still be reporting the change: no replay yet."""
    snapshot, posture = _public_ip_snapshot()
    engine, notifier = _silenced(monkeypatch, snapshot)
    await engine._timed_run()
    network = cast("dict[str, Any]", snapshot.get("network"))
    caps = list(network["capabilities"])
    network["capabilities"] = []
    engine._options["sentinel_autonomy_level"] = 1
    await engine._timed_run()
    assert notifier.calls == []
    assert list(_owed(engine)) == ["network_public_ip_changed"]

    # Readable again and still changed: the live alert tells it, once.
    network["capabilities"] = caps
    await engine._timed_run()
    assert [f.type for f in notifier.calls] == ["network_public_ip_changed"]
    assert "seen_while_silenced" not in notifier.calls[0].evidence
    del posture


@pytest.mark.asyncio
async def test_no_replay_while_the_suppression_state_is_read_only(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Nothing could be marked sent, so every restart would repeat the replay."""
    snapshot, posture = _public_ip_snapshot()
    engine, notifier = _silenced(monkeypatch, snapshot)
    await engine._timed_run()
    posture.update({"public_ip_changed": False, "public_ip_key": "aaaa0000"})
    monkeypatch.setattr(DummySuppression, "is_read_only", property(lambda _s: True))
    engine._options["sentinel_autonomy_level"] = 1
    await engine._timed_run()
    assert notifier.calls == []
    assert list(_owed(engine)) == ["network_public_ip_changed"]


@pytest.mark.asyncio
async def test_a_device_that_joins_and_leaves_at_level_zero_still_alerts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A radio device paired and removed while silenced is told (#799)."""
    engine, notifier, section = await _radio_engine(monkeypatch)
    await engine._timed_run()
    section["devices"] = [_radio_device("a")]
    section["new_devices"] = []
    await engine._timed_run()
    assert notifier.calls == []

    engine._options["sentinel_autonomy_level"] = 1
    await engine._timed_run()
    assert [f.type for f in notifier.calls] == ["radio_new_device_joined"]
    sent = notifier.calls[0]
    assert "New Plug" in sent.evidence["summary"]
    # A replay describes the past: no Trust device button.
    assert "device_ids" not in sent.evidence
    assert _owed(engine) == {}


@pytest.mark.asyncio
async def test_devices_seen_at_level_zero_are_told_in_one_alert(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Two devices paired and removed while silenced: one alert naming both."""
    engine, notifier, section = await _radio_engine(monkeypatch)
    await engine._timed_run()
    section["devices"] = [_radio_device("a"), _radio_device("c", "Other Plug")]
    section["new_devices"] = ["zigbee:c"]
    await engine._timed_run()
    assert len(_owed(engine)["radio_new_device_joined"]) == 2
    section["devices"] = [_radio_device("a")]
    section["new_devices"] = []

    engine._options["sentinel_autonomy_level"] = 1
    await engine._timed_run()
    assert len(notifier.calls) == 1
    sent = notifier.calls[0]
    assert sent.evidence["device_keys"] == ["zigbee:b", "zigbee:c"]
    assert "New Plug" in sent.evidence["summary"]
    assert "Other Plug" in sent.evidence["summary"]


@pytest.mark.asyncio
async def test_an_owed_device_named_again_by_a_live_alert_is_not_repeated(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A live alert naming the owed device and another settles the owed one."""
    engine, notifier, section = await _radio_engine(monkeypatch)
    await engine._timed_run()
    section["devices"] = [*section["devices"], _radio_device("c", "Other Plug")]
    section["new_devices"] = ["zigbee:b", "zigbee:c"]
    engine._options["sentinel_autonomy_level"] = 1
    await engine._timed_run()
    assert len(notifier.calls) == 1
    assert "seen_while_silenced" not in notifier.calls[0].evidence
    assert _owed(engine) == {}


@pytest.mark.asyncio
async def test_an_owed_device_the_live_alert_does_not_name_still_alerts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A live alert for another device does not settle the owed one."""
    engine, notifier, section = await _radio_engine(monkeypatch)
    await engine._timed_run()
    section["devices"] = [_radio_device("a"), _radio_device("c", "Other Plug")]
    section["new_devices"] = ["zigbee:c"]
    engine._options["sentinel_autonomy_level"] = 1
    await engine._timed_run()
    assert len(notifier.calls) == 2
    replays = [f for f in notifier.calls if f.evidence.get("seen_while_silenced")]
    assert len(replays) == 1
    assert "New Plug" in replays[0].evidence["summary"]
    assert "Other Plug" not in replays[0].evidence["summary"]
    assert _owed(engine) == {}


@pytest.mark.asyncio
async def test_an_owed_change_now_excluded_is_dropped(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An exclusion added while silenced applies to the owed change too."""
    snapshot, posture = _public_ip_snapshot()
    engine, notifier = _silenced(monkeypatch, snapshot)
    await engine._timed_run()
    posture.update({"public_ip_changed": False, "public_ip_key": "aaaa0000"})
    monkeypatch.setattr(
        engine,
        "_entity_excluded_for_type",
        lambda entity_id, _anomaly_type: entity_id == "sensor.gw_ip",
    )
    cast("Any", engine)._rule_entity_exclusions = {"*": object()}
    engine._options["sentinel_autonomy_level"] = 1
    await engine._timed_run()
    assert notifier.calls == []
    assert _owed(engine) == {}


@pytest.mark.asyncio
async def test_a_replay_overtaken_by_level_zero_stays_owed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Level 0 set while a replay is prepared: not sent, and owed as before."""
    snapshot, posture = _public_ip_snapshot()
    engine, notifier = _silenced(monkeypatch, snapshot)
    await engine._timed_run()
    stored = [dict(f) for f in _owed(engine)["network_public_ip_changed"]]
    posture.update({"public_ip_changed": False, "public_ip_key": "aaaa0000"})
    engine._options["sentinel_autonomy_level"] = 1
    real = engine._is_passive
    checks = iter([False, False, True])
    monkeypatch.setattr(engine, "_is_passive", lambda: next(checks, real()))
    monkeypatch.setattr(engine, "_explainer_for", lambda *_a, **_k: None, raising=False)
    abandoned: list[str] = []
    abandon = engine._abandon_for_silence

    async def _spy(held: Any, *args: Any) -> bool:
        abandoned.extend(f.type for f, _stamps in held)
        return await abandon(held, *args)

    monkeypatch.setattr(engine, "_abandon_for_silence", _spy)
    await engine._timed_run()
    assert abandoned == ["network_public_ip_changed"]
    assert notifier.calls == []
    assert _owed(engine)["network_public_ip_changed"] == stored


@pytest.mark.asyncio
async def test_the_owed_list_is_saved_only_when_it_changes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    snapshot, _posture = _public_ip_snapshot()
    engine, _notifier = _silenced(monkeypatch, snapshot)
    saves: list[int] = []
    suppression = cast("Any", engine)._suppression

    async def _save() -> None:
        saves.append(1)

    suppression.async_save = _save
    await engine._timed_run()
    first = len(saves)
    assert first >= 1
    await engine._timed_run()
    await engine._timed_run()
    assert len(saves) == first


def test_owed_change_alerts_survive_a_save_and_load() -> None:
    """The owed list persists, and older saved state without it loads empty."""
    from custom_components.home_generative_agent.sentinel.models import (  # noqa: PLC0415
        anomaly_finding_from_dict,
    )

    finding = AnomalyFinding(
        anomaly_id="id1",
        type="network_public_ip_changed",
        severity="low",
        confidence=0.9,
        triggering_entities=["sensor.gw_ip"],
        evidence={"public_ip_key": "bbbb0000", "summary": "s"},
        suggested_actions=["Nothing to do"],
        is_sensitive=True,
        detected_at=NOW,
    )
    state = SuppressionState(
        owed_change_alerts={"network_public_ip_changed": [finding.as_dict()]}
    )
    loaded = SuppressionState.from_dict(state.as_dict())
    stored = loaded.owed_change_alerts["network_public_ip_changed"]
    assert anomaly_finding_from_dict(stored[0]) == finding
    legacy = state.as_dict()
    del legacy["owed_change_alerts"]
    assert SuppressionState.from_dict(legacy).owed_change_alerts == {}


@pytest.mark.asyncio
async def test_an_admin_added_and_removed_at_level_zero_still_alerts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A new admin removed before the level rises is told, naming the account."""
    ha = {"new_admin_users": ["eve"], "new_long_lived_tokens": []}
    snapshot = _snapshot(ha)
    engine, notifier = _silenced(monkeypatch, snapshot)
    await engine._timed_run()
    ha_section = cast("dict[str, Any]", cast("Any", snapshot)["network"]["ha_security"])
    ha_section["new_admin_users"] = []
    engine._options["sentinel_autonomy_level"] = 1
    await engine._timed_run()
    assert [f.type for f in notifier.calls] == ["ha_new_admin_or_token"]
    sent = notifier.calls[0]
    assert sent.evidence["new_admin_users"] == ["eve"]
    assert ": 1 account or token;" in sent.evidence["summary"]
    assert _owed(engine) == {}


@pytest.mark.asyncio
async def test_an_admin_still_there_when_the_level_rises_alerts_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    snapshot = _snapshot({"new_admin_users": ["eve"], "new_long_lived_tokens": []})
    engine, notifier = _silenced(monkeypatch, snapshot)
    await engine._timed_run()
    engine._options["sentinel_autonomy_level"] = 1
    await engine._timed_run()
    assert len(notifier.calls) == 1
    assert "seen_while_silenced" not in notifier.calls[0].evidence
    assert _owed(engine) == {}


@pytest.mark.asyncio
async def test_a_port_mapping_still_open_when_the_level_rises_alerts_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    snapshot, _posture = _upnp_snapshot()
    engine, notifier = _silenced(monkeypatch, snapshot)
    await engine._timed_run()
    engine._options["sentinel_autonomy_level"] = 1
    await engine._timed_run()
    assert [f.type for f in notifier.calls] == ["network_upnp_port_mapping_added"]
    assert "seen_while_silenced" not in notifier.calls[0].evidence
    assert _owed(engine) == {}


@pytest.mark.asyncio
async def test_triage_dropping_the_live_alert_sends_the_owed_one(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Triage may drop a live alert, not what was owed from level 0."""
    snapshot, _posture = _public_ip_snapshot()
    engine, notifier = _silenced(monkeypatch, snapshot)
    await engine._timed_run()
    triage = MagicMock()

    async def _suppress(*_a: Any, **_k: Any) -> Any:
        return MagicMock(
            decision="suppress", reason_code="llm_suppress", triage_confidence=0.9
        )

    triage.triage = _suppress
    cast("Any", engine)._triage_service = triage
    engine._options["sentinel_autonomy_level"] = 1
    await engine._timed_run()
    assert [f.evidence.get("seen_while_silenced") for f in notifier.calls] == [True]
    assert _owed(engine) == {}


@pytest.mark.asyncio
async def test_a_device_still_there_is_not_named_again_by_the_replay(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Owed b and c paired together; c removed: b's live alert, c's replay."""
    engine, notifier, section = await _radio_engine(monkeypatch)
    section["devices"] = [*section["devices"], _radio_device("c", "Other Plug")]
    section["new_devices"] = ["zigbee:b", "zigbee:c"]
    await engine._timed_run()
    assert len(_owed(engine)["radio_new_device_joined"]) == 1
    section["devices"] = [_radio_device("a"), _radio_device("b", "New Plug")]
    section["new_devices"] = ["zigbee:b"]
    engine._options["sentinel_autonomy_level"] = 1
    await engine._timed_run()
    keys = [f.evidence["device_keys"] for f in notifier.calls]
    assert keys == [["zigbee:b"], ["zigbee:c"]]
    assert notifier.calls[1].evidence["seen_while_silenced"] is True
    assert _owed(engine) == {}


@pytest.mark.asyncio
async def test_a_partly_told_device_alert_keeps_what_its_live_alert_waits_on(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Owed b and c apart; b's live alert waits on a cooldown, c is replayed."""
    engine, notifier, section = await _radio_engine(monkeypatch)
    await engine._timed_run()
    section["devices"] = [_radio_device("a"), _radio_device("c", "Other Plug")]
    section["new_devices"] = ["zigbee:c"]
    await engine._timed_run()
    section["devices"] = [_radio_device("a"), _radio_device("b", "New Plug")]
    section["new_devices"] = ["zigbee:b"]
    state = cast("Any", engine)._suppression.state
    engine._options["sentinel_autonomy_level"] = 1
    engine._options["sentinel_cooldown_minutes"] = 60
    state.last_by_type["radio_new_device_joined"] = datetime.now(UTC).isoformat()
    await engine._timed_run()
    # c's replay goes out at once; b's live alert waits out the cooldown.
    assert [f.evidence["device_keys"] for f in notifier.calls] == [["zigbee:c"]]
    owed_now = _owed(engine)["radio_new_device_joined"]
    assert [m["evidence"]["device_keys"] for m in owed_now] == [["zigbee:b"]]

    state.last_by_type.clear()
    await engine._timed_run()
    assert [f.evidence["device_keys"] for f in notifier.calls][-1] == ["zigbee:b"]
    assert _owed(engine) == {}


@pytest.mark.asyncio
async def test_a_device_trusted_at_level_zero_is_not_replayed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    engine, notifier, section = await _radio_engine(monkeypatch)
    await engine._timed_run()
    inventory = cast("Any", engine)._network_inventory
    await inventory.async_set_trusted(["zigbee:b"], trusted=True)
    section["devices"] = [_radio_device("a"), _radio_device("b", "New Plug")]
    section["new_devices"] = []
    engine._options["sentinel_autonomy_level"] = 1
    await engine._timed_run()
    assert notifier.calls == []
    assert _owed(engine) == {}


@pytest.mark.asyncio
async def test_two_ip_changes_undone_at_level_zero_are_both_owed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A to B and back, then to C and back: one alert telling both."""
    snapshot, posture = _public_ip_snapshot()
    engine, notifier = _silenced(monkeypatch, snapshot)
    await engine._timed_run()
    posture.update({"public_ip_changed": False, "public_ip_key": "aaaa0000"})
    await engine._timed_run()
    posture.update({"public_ip_changed": True, "public_ip_key": "cccc0000"})
    await engine._timed_run()
    posture.update({"public_ip_changed": False, "public_ip_key": "aaaa0000"})
    await engine._timed_run()
    assert len(_owed(engine)["network_public_ip_changed"]) == 2
    engine._options["sentinel_autonomy_level"] = 1
    await engine._timed_run()
    assert [f.type for f in notifier.calls] == ["network_public_ip_changed"]
    assert _owed(engine) == {}


@pytest.mark.asyncio
async def test_only_kinds_whose_rule_ran_are_replayed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An IP change and a device owed; IP data unreadable: only the device."""
    from custom_components.home_generative_agent.snapshot.network import (  # noqa: PLC0415
        posture_cap,
    )

    inventory = _memory_inventory()
    await inventory.async_commit([_radio_device("a")], NOW)
    snapshot, section = _radio_snapshot()
    network = cast("dict[str, Any]", cast("Any", snapshot)["network"])
    ip_cap = posture_cap("public_ip_changed")
    network["capabilities"] = [*network["capabilities"], ip_cap]
    network["posture"].update(
        {
            "public_ip_changed": True,
            "public_ip_key": "bbbb0000",
            "public_ip_entity_id": "sensor.gw_ip",
        }
    )
    hass, _calls = _service_recorder()
    engine, notifier = _silenced(
        monkeypatch, snapshot, hass=hass, network_inventory=inventory
    )
    await engine._timed_run()
    assert set(_owed(engine)) == {
        "radio_new_device_joined",
        "network_public_ip_changed",
    }
    section["devices"] = [_radio_device("a")]
    section["new_devices"] = []
    network["capabilities"].remove(ip_cap)
    engine._options["sentinel_autonomy_level"] = 1
    await engine._timed_run()
    assert [f.type for f in notifier.calls] == ["radio_new_device_joined"]
    assert list(_owed(engine)) == ["network_public_ip_changed"]


@pytest.mark.asyncio
async def test_a_change_quiet_hours_stopped_at_level_zero_is_still_owed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from custom_components.home_generative_agent.sentinel import (  # noqa: PLC0415
        engine as engine_mod,
    )
    from custom_components.home_generative_agent.sentinel.suppression import (  # noqa: PLC0415
        SuppressionDecision,
    )

    snapshot, posture = _public_ip_snapshot()
    engine, notifier = _silenced(monkeypatch, snapshot)
    real = engine_mod.should_suppress
    monkeypatch.setattr(
        engine_mod,
        "should_suppress",
        lambda *_a, **_k: SuppressionDecision(True, "quiet_hours"),  # noqa: FBT003
    )
    await engine._timed_run()
    assert list(_owed(engine)) == ["network_public_ip_changed"]
    monkeypatch.setattr(engine_mod, "should_suppress", real)
    posture.update({"public_ip_changed": False, "public_ip_key": "aaaa0000"})
    engine._options["sentinel_autonomy_level"] = 1
    await engine._timed_run()
    assert [f.evidence.get("seen_while_silenced") for f in notifier.calls] == [True]


@pytest.mark.asyncio
async def test_triage_dropping_a_live_device_alert_sends_the_owed_one(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    engine, notifier, _section = await _radio_engine(monkeypatch)
    await engine._timed_run()
    triage = MagicMock()

    async def _suppress(*_a: Any, **_k: Any) -> Any:
        return MagicMock(
            decision="suppress", reason_code="llm_suppress", triage_confidence=0.9
        )

    triage.triage = _suppress
    cast("Any", engine)._triage_service = triage
    engine._options["sentinel_autonomy_level"] = 1
    await engine._timed_run()
    assert len(notifier.calls) == 1
    assert notifier.calls[0].evidence["device_keys"] == ["zigbee:b"]
    assert notifier.calls[0].evidence["seen_while_silenced"] is True
    assert _owed(engine) == {}


# --- #801: a change held back above level 0 is owed too -------------------


def _cooling(engine: SentinelEngine, kind: str) -> Any:
    """Start the type cooldown for ``kind`` on the real clock (60 minutes)."""
    state = cast("Any", engine)._suppression.state
    state.last_by_type[kind] = datetime.now(UTC).isoformat()
    engine._options["sentinel_cooldown_minutes"] = 60
    return state


@pytest.mark.asyncio
async def test_a_change_undone_during_a_cooldown_still_alerts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Level 1: a second IP change inside the cooldown, changed back, is told."""
    snapshot, posture = _public_ip_snapshot()
    engine, _ = _engine(monkeypatch, snapshot)
    notifier = cast("DummyNotifier", cast("Any", engine)._notifier)
    state = _cooling(engine, "network_public_ip_changed")
    await engine._timed_run()
    assert notifier.calls == []
    assert list(_owed(engine)) == ["network_public_ip_changed"]

    # Undone before the cooldown ends: the rule no longer fires, and the
    # replay does not wait the cooldown out.
    posture.update({"public_ip_changed": False, "public_ip_key": "aaaa0000"})
    await engine._timed_run()
    assert [f.type for f in notifier.calls] == ["network_public_ip_changed"]
    del state
    summary = notifier.calls[0].evidence["summary"]
    assert summary.startswith("Seen while its alert was held back (")
    assert "autonomy level 0" not in summary
    assert _owed(engine) == {}


@pytest.mark.asyncio
async def test_a_change_still_there_after_its_cooldown_alerts_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    snapshot, posture = _public_ip_snapshot()
    engine, _ = _engine(monkeypatch, snapshot)
    notifier = cast("DummyNotifier", cast("Any", engine)._notifier)
    state = _cooling(engine, "network_public_ip_changed")
    await engine._timed_run()
    await engine._timed_run()
    assert notifier.calls == []
    state.last_by_type.clear()
    await engine._timed_run()
    # Delivered: the remembered address moves on, so the rule goes quiet.
    posture["public_ip_changed"] = False
    await engine._timed_run()
    assert len(notifier.calls) == 1
    assert "seen_while_silenced" not in notifier.calls[0].evidence
    assert _owed(engine) == {}


@pytest.mark.asyncio
async def test_a_change_quiet_hours_held_back_is_owed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from custom_components.home_generative_agent.sentinel import (  # noqa: PLC0415
        engine as engine_mod,
    )
    from custom_components.home_generative_agent.sentinel.suppression import (  # noqa: PLC0415
        SuppressionDecision,
    )

    snapshot, posture = _public_ip_snapshot()
    engine, _ = _engine(monkeypatch, snapshot)
    notifier = cast("DummyNotifier", cast("Any", engine)._notifier)
    real = engine_mod.should_suppress
    monkeypatch.setattr(
        engine_mod,
        "should_suppress",
        lambda *_a, **_k: SuppressionDecision(True, "quiet_hours"),  # noqa: FBT003
    )
    await engine._timed_run()
    assert list(_owed(engine)) == ["network_public_ip_changed"]
    monkeypatch.setattr(engine_mod, "should_suppress", real)
    posture.update({"public_ip_changed": False, "public_ip_key": "aaaa0000"})
    await engine._timed_run()
    assert [f.evidence.get("seen_while_silenced") for f in notifier.calls] == [True]


@pytest.mark.asyncio
async def test_a_change_stopped_by_a_pending_prompt_is_owed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    A prompt is keyed by the alert's id, which a change made again can repeat.

    A UPnP mapping opened, closed and opened again inside the prompt's life
    has the first alert's id: owing it only waits the prompt out.
    """
    from custom_components.home_generative_agent.sentinel import (  # noqa: PLC0415
        engine as engine_mod,
    )
    from custom_components.home_generative_agent.sentinel.suppression import (  # noqa: PLC0415
        SuppressionDecision,
    )

    snapshot, _posture = _public_ip_snapshot()
    engine, _ = _engine(monkeypatch, snapshot)
    monkeypatch.setattr(
        engine_mod,
        "should_suppress",
        lambda *_a, **_k: SuppressionDecision(True, "pending_prompt"),  # noqa: FBT003
    )
    await engine._timed_run()
    assert list(_owed(engine)) == ["network_public_ip_changed"]


@pytest.mark.asyncio
async def test_a_snoozed_change_at_level_one_is_not_owed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    snapshot, _posture = _public_ip_snapshot()
    engine, _ = _engine(monkeypatch, snapshot)
    state = cast("Any", engine)._suppression.state
    state.snoozed_until["network_public_ip_changed"] = {"until": "permanent"}
    await engine._timed_run()
    assert _owed(engine) == {}


@pytest.mark.asyncio
async def test_a_device_paired_and_removed_during_a_cooldown_still_alerts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    inventory = _memory_inventory()
    await inventory.async_commit([_radio_device("a")], NOW)
    snapshot, section = _radio_snapshot()
    hass, _calls = _service_recorder()
    engine, _ = _engine(monkeypatch, snapshot, hass=hass, network_inventory=inventory)
    notifier = cast("DummyNotifier", cast("Any", engine)._notifier)
    state = _cooling(engine, "radio_new_device_joined")
    await engine._timed_run()
    assert notifier.calls == []
    section["devices"] = [_radio_device("a")]
    section["new_devices"] = []
    await engine._timed_run()
    state.last_by_type.clear()
    await engine._timed_run()
    assert [f.type for f in notifier.calls] == ["radio_new_device_joined"]
    assert "New Plug" in notifier.calls[0].evidence["summary"]
    assert _owed(engine) == {}


@pytest.mark.asyncio
async def test_an_admin_added_and_removed_during_a_cooldown_still_alerts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The case #801 leads with: a second admin inside the first one's cooldown."""
    ha = {"new_admin_users": ["eve"], "new_long_lived_tokens": []}
    snapshot = _snapshot(ha)
    engine, _ = _engine(monkeypatch, snapshot)
    notifier = cast("DummyNotifier", cast("Any", engine)._notifier)
    _cooling(engine, "ha_new_admin_or_token")
    await engine._timed_run()
    assert notifier.calls == []
    assert list(_owed(engine)) == ["ha_new_admin_or_token"]
    ha_section = cast("dict[str, Any]", cast("Any", snapshot)["network"]["ha_security"])
    ha_section["new_admin_users"] = []
    await engine._timed_run()
    assert [f.type for f in notifier.calls] == ["ha_new_admin_or_token"]
    sent = notifier.calls[0]
    assert sent.evidence["new_admin_users"] == ["eve"]
    assert sent.evidence["summary"].startswith("Seen while its alert was held back (")
    assert _owed(engine) == {}


@pytest.mark.asyncio
async def test_a_change_an_entity_cooldown_held_back_is_owed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from custom_components.home_generative_agent.sentinel import (  # noqa: PLC0415
        engine as engine_mod,
    )
    from custom_components.home_generative_agent.sentinel.suppression import (  # noqa: PLC0415
        SuppressionDecision,
    )

    snapshot, _posture = _public_ip_snapshot()
    engine, _ = _engine(monkeypatch, snapshot)
    monkeypatch.setattr(
        engine_mod,
        "should_suppress",
        lambda *_a, **_k: SuppressionDecision(True, "entity_cooldown"),  # noqa: FBT003
    )
    await engine._timed_run()
    assert list(_owed(engine)) == ["network_public_ip_changed"]


@pytest.mark.asyncio
async def test_more_port_mappings_from_the_same_baseline_settle_fewer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Mappings 4 to 5, then 4 to 6 in a cooldown: one alert, nothing 'undone'."""
    snapshot, posture = _upnp_snapshot()
    engine, _ = _engine(monkeypatch, snapshot)
    notifier = cast("DummyNotifier", cast("Any", engine)._notifier)
    state = _cooling(engine, "network_upnp_port_mapping_added")
    await engine._timed_run()
    posture.update({"upnp_port_mappings_added": 2, "upnp_port_mapping_count": 4})
    await engine._timed_run()
    state.last_by_type.clear()
    await engine._timed_run()
    posture["upnp_port_mappings_added"] = 0
    await engine._timed_run()
    assert len(notifier.calls) == 1
    assert "seen_while_silenced" not in notifier.calls[0].evidence
    assert _owed(engine) == {}


def test_a_replay_push_offers_no_snooze() -> None:
    """A snooze mutes the whole kind; a replay notice must not invite it."""
    from custom_components.home_generative_agent.sentinel import (  # noqa: PLC0415
        notifier as notifier_mod,
    )

    finding = AnomalyFinding(
        anomaly_id="a",
        type="network_public_ip_changed",
        severity="low",
        confidence=0.9,
        triggering_entities=[],
        evidence={"summary": "s", "seen_while_silenced": True},
        suggested_actions=[],
        is_sensitive=True,
    )
    titles = [a["title"] for a in notifier_mod._build_actions(finding)]
    assert "Snooze 24 h" not in titles
    assert "Snooze Always" not in titles
    assert "False Alarm" in titles
    live = notifier_mod._build_actions(
        AnomalyFinding(
            anomaly_id="b",
            type="network_public_ip_changed",
            severity="low",
            confidence=0.9,
            triggering_entities=[],
            evidence={"summary": "s"},
            suggested_actions=[],
            is_sensitive=True,
        )
    )
    assert "Snooze Always" in [a["title"] for a in live]


@pytest.mark.asyncio
async def test_a_kind_skips_its_cooldown_for_one_replay_per_window(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A churning kind cannot push a replay every run past its cooldown."""
    snapshot, posture = _public_ip_snapshot()
    engine, notifier = _silenced(monkeypatch, snapshot)
    await engine._timed_run()
    posture.update({"public_ip_changed": False, "public_ip_key": "aaaa0000"})
    _cooling(engine, "network_public_ip_changed")
    engine._options["sentinel_autonomy_level"] = 1
    await engine._timed_run()
    assert len(notifier.calls) == 1
    audit = cast("DummyAudit", cast("Any", engine)._audit_store)
    sent = [
        c
        for c in audit.calls
        if c["finding"].anomaly_id == notifier.calls[0].anomaly_id
    ]
    assert sent[-1]["suppression_reason_code"] == "not_suppressed"

    # Another change, held by the cooldown and undone: owed, but the kind
    # already skipped its cooldown this window, so it waits.
    posture.update({"public_ip_changed": True, "public_ip_key": "dddd0000"})
    await engine._timed_run()
    posture.update({"public_ip_changed": False, "public_ip_key": "aaaa0000"})
    await engine._timed_run()
    assert len(notifier.calls) == 1
    assert list(_owed(engine)) == ["network_public_ip_changed"]


@pytest.mark.asyncio
async def test_a_replay_waits_out_an_entity_cooldown(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """False Alarm lengthens the entity cooldown; a replay honours it."""
    from custom_components.home_generative_agent.sentinel import (  # noqa: PLC0415
        engine as engine_mod,
    )
    from custom_components.home_generative_agent.sentinel.suppression import (  # noqa: PLC0415
        SuppressionDecision,
    )

    snapshot, posture = _public_ip_snapshot()
    engine, notifier = _silenced(monkeypatch, snapshot)
    await engine._timed_run()
    posture.update({"public_ip_changed": False, "public_ip_key": "aaaa0000"})
    monkeypatch.setattr(
        engine_mod,
        "should_suppress",
        lambda *_a, **_k: SuppressionDecision(True, "entity_cooldown"),  # noqa: FBT003
    )
    engine._options["sentinel_autonomy_level"] = 1
    await engine._timed_run()
    assert notifier.calls == []
    assert list(_owed(engine)) == ["network_public_ip_changed"]


def _triage_suppresses(engine: SentinelEngine) -> None:
    triage = MagicMock()

    async def _suppress(*_a: Any, **_k: Any) -> Any:
        return MagicMock(
            decision="suppress", reason_code="llm_suppress", triage_confidence=0.9
        )

    triage.triage = _suppress
    cast("Any", engine)._triage_service = triage


async def _owed_ip_change_undone(
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[SentinelEngine, DummyNotifier, Any]:
    """Owe an IP change at level 0, undo it, start the cooldown, go to level 1."""
    snapshot, posture = _public_ip_snapshot()
    engine, notifier = _silenced(monkeypatch, snapshot)
    await engine._timed_run()
    posture.update({"public_ip_changed": False, "public_ip_key": "aaaa0000"})
    state = _cooling(engine, "network_public_ip_changed")
    engine._options["sentinel_autonomy_level"] = 1
    return engine, notifier, state


@pytest.mark.asyncio
async def test_a_replay_honours_an_entity_cooldown_false_alarm_lengthened(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Skipping the type cooldown does not skip what False Alarm asked for."""
    kind = "network_public_ip_changed"
    engine, notifier, state = await _owed_ip_change_undone(monkeypatch)
    engine._options["sentinel_entity_cooldown_minutes"] = 15
    state.last_by_entity["sensor.gw_ip"] = {kind: datetime.now(UTC).isoformat()}
    state.learned_cooldown_multipliers[f"{kind}:sensor.gw_ip"] = 2
    await engine._timed_run()
    assert notifier.calls == []
    assert list(_owed(engine)) == [kind]

    # The plain entity cooldown is renewed by every alert of the kind, as the
    # type one is: it does not hold the replay.
    state.learned_cooldown_multipliers.clear()
    await engine._timed_run()
    assert [f.evidence.get("seen_while_silenced") for f in notifier.calls] == [True]


@pytest.mark.asyncio
async def test_a_replay_past_the_cooldown_does_not_renew_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    kind = "network_public_ip_changed"
    engine, notifier, state = await _owed_ip_change_undone(monkeypatch)
    before = state.last_by_type[kind]
    await engine._timed_run()
    assert len(notifier.calls) == 1
    assert state.last_by_type[kind] == before
    assert kind not in state.last_by_entity.get("sensor.gw_ip", {})


@pytest.mark.asyncio
async def test_a_replay_the_notifier_drops_as_a_repeat_stays_owed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    kind = "network_public_ip_changed"
    engine, notifier, state = await _owed_ip_change_undone(monkeypatch)
    state.last_by_type.clear()
    sent: list[bool] = [False]

    async def _notify(finding: Any, _snapshot: Any, _explanation: Any) -> bool:
        notifier.calls.append(finding)
        return sent[0]

    cast("Any", notifier).async_notify = _notify
    await engine._timed_run()
    assert list(_owed(engine)) == [kind]
    # Retried once its prompt and cooldown are over.
    sent[0] = True
    state.pending_prompts.clear()
    state.last_by_type.clear()
    await engine._timed_run()
    assert _owed(engine) == {}


@pytest.mark.asyncio
async def test_a_held_back_change_triage_stopped_is_settled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Above level 0, triage stopping the covering live alert is its call."""
    snapshot, _posture = _public_ip_snapshot()
    engine, _ = _engine(monkeypatch, snapshot)
    notifier = cast("DummyNotifier", cast("Any", engine)._notifier)
    state = _cooling(engine, "network_public_ip_changed")
    await engine._timed_run()
    assert list(_owed(engine)) == ["network_public_ip_changed"]
    state.last_by_type.clear()
    _triage_suppresses(engine)
    await engine._timed_run()
    assert notifier.calls == []
    assert _owed(engine) == {}


@pytest.mark.asyncio
async def test_a_replay_that_told_a_new_token_commits_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Triage stops the live alert, the replay tells it: not alerted again."""
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

    async def _collect(_hass: Any, _p: Any) -> list[ObservedUser]:
        return [ObservedUser("u1", "Lindo", True, True, False, ())]  # noqa: FBT003

    monkeypatch.setattr(
        "custom_components.home_generative_agent.sentinel.engine."
        "async_collect_auth_observation",
        _collect,
    )
    snapshot = _snapshot({"new_admin_users": [], "new_long_lived_tokens": ["script"]})
    engine, notifier = _silenced(monkeypatch, snapshot, auth_inventory=inventory)
    await engine._timed_run()
    assert commits[-1]["hold_back"] is delta
    _triage_suppresses(engine)
    engine._options["sentinel_autonomy_level"] = 1
    await engine._timed_run()
    assert [f.evidence.get("seen_while_silenced") for f in notifier.calls] == [True]
    assert commits[-1]["hold_back"] is None


@pytest.mark.asyncio
async def test_a_waiting_live_alert_keeps_its_devices_out_of_the_replay(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Owed b and c together; b's live alert waits: c now, b with its alert."""
    engine, notifier, section = await _radio_engine(monkeypatch)
    section["devices"] = [*section["devices"], _radio_device("c", "Other Plug")]
    section["new_devices"] = ["zigbee:b", "zigbee:c"]
    await engine._timed_run()
    section["devices"] = [_radio_device("a"), _radio_device("b", "New Plug")]
    section["new_devices"] = ["zigbee:b"]
    state = _cooling(engine, "radio_new_device_joined")
    engine._options["sentinel_autonomy_level"] = 1
    await engine._timed_run()
    assert [f.evidence["device_keys"] for f in notifier.calls] == [["zigbee:c"]]
    owed_now = _owed(engine)["radio_new_device_joined"]
    assert [m["evidence"]["device_keys"] for m in owed_now] == [["zigbee:b"]]

    state.last_by_type.clear()
    await engine._timed_run()
    assert [f.evidence["device_keys"] for f in notifier.calls][-1] == ["zigbee:b"]
    assert "seen_while_silenced" not in notifier.calls[-1].evidence
    assert _owed(engine) == {}


def _owed_clients(*keys: str) -> AnomalyFinding:
    """Return an owed unknown-client alert as the rule reports it since #802."""
    return AnomalyFinding(
        anomaly_id="owed-clients-" + "-".join(keys),
        type="network_unknown_device_joined",
        severity="medium",
        confidence=0.9,
        triggering_entities=[f"device_tracker.{k}" for k in keys],
        evidence={
            "client_keys": list(keys),
            "summary": "New devices: " + ", ".join(keys) + ".",
            "item_labels": {k: f"Phone {k}" for k in keys},
            "item_entities": {k: f"device_tracker.{k}" for k in keys},
        },
        suggested_actions=["Check it"],
        is_sensitive=True,
        detected_at=NOW,
    )


@pytest.mark.asyncio
async def test_an_owed_kind_whose_rule_stops_running_expires(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """Warned after 7 idle days Sentinel ran, dropped and audited after 30 (#802)."""
    engine, _ = _engine(monkeypatch, _snapshot())
    state = cast("Any", engine)._suppression.state
    kind = "network_unknown_device_joined"
    state.owed_change_alerts[kind] = [_owed_clients("k1").as_dict()]
    expire = cast("Any", engine)._expire_idle_owed

    for day in range(29):
        # Two runs a day count once.
        await expire(set(), _snapshot(), NOW + timedelta(days=day))
        await expire(set(), _snapshot(), NOW + timedelta(days=day, hours=1))
    warned = [r for r in caplog.records if "they are dropped after 30" in r.message]
    assert len(warned) == 1
    assert state.owed_rule_idle[kind]["days"] == 29
    assert kind in state.owed_change_alerts

    await expire(set(), _snapshot(), NOW + timedelta(days=29))
    assert state.owed_change_alerts == {}
    assert state.owed_rule_idle == {}
    audit = cast("DummyAudit", cast("Any", engine)._audit_store)
    assert audit.calls[-1]["suppression_reason_code"] == "owed_expired"
    assert audit.calls[-1]["finding"].evidence["client_keys"] == ["k1"]


@pytest.mark.asyncio
async def test_only_days_sentinel_ran_count_as_idle(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A shutdown, clock jump or legacy alert cannot expire anything at once."""
    engine, _ = _engine(monkeypatch, _snapshot())
    state = cast("Any", engine)._suppression.state
    kind = "network_unknown_device_joined"
    # Owed by a build before #802: no idle record, first seen long ago.
    state.owed_change_alerts[kind] = [_owed_clients("k1").as_dict()]

    await cast("Any", engine)._expire_idle_owed(
        set(), _snapshot(), NOW + timedelta(days=90)
    )

    assert kind in state.owed_change_alerts
    assert state.owed_rule_idle[kind]["days"] == 1


@pytest.mark.asyncio
async def test_a_rule_that_runs_is_not_idle(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """A run resets the count; idle again later is counted and warned afresh."""
    engine, _ = _engine(monkeypatch, _snapshot())
    state = cast("Any", engine)._suppression.state
    kind = "network_unknown_device_joined"
    state.owed_change_alerts[kind] = [_owed_clients("k1").as_dict()]
    state.owed_rule_idle[kind] = {"days": 8, "last": "2026-09-06"}
    expire = cast("Any", engine)._expire_idle_owed
    await expire(set(), _snapshot(), NOW)
    assert any("they are dropped after 30" in r.message for r in caplog.records)
    caplog.clear()

    await expire({kind}, _snapshot(), NOW + timedelta(days=1))
    assert state.owed_rule_idle == {}
    assert kind in state.owed_change_alerts

    for day in range(2, 9):
        await expire(set(), _snapshot(), NOW + timedelta(days=day))
    assert state.owed_rule_idle[kind]["days"] == 7
    assert any("they are dropped after 30" in r.message for r in caplog.records)


@pytest.mark.asyncio
async def test_a_rule_that_raised_is_not_idle(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A broken rule is not a removed one: its owed alerts do not age out."""
    engine, _ = _engine(monkeypatch, _snapshot())
    state = cast("Any", engine)._suppression.state
    kind = "network_unknown_device_joined"
    state.owed_change_alerts[kind] = [_owed_clients("k1").as_dict()]
    state.owed_rule_idle[kind] = {"days": 29, "last": "2026-09-06"}

    await cast("Any", engine)._send_owed_changes(
        [],
        {},
        set(),
        _snapshot(),
        NOW,
        timedelta(0),
        timedelta(0),
        explain_enabled=False,
        trigger_source="poll",
        failed=frozenset({kind}),
    )

    assert state.owed_rule_idle == {}
    assert kind in state.owed_change_alerts


@pytest.mark.asyncio
async def test_unreadable_idle_state_drops_the_kind_not_sentinel(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """Saved state that cannot be read drops its kind instead of the run loop."""
    engine, _ = _engine(monkeypatch, _snapshot())
    state = cast("Any", engine)._suppression.state
    kind = "network_unknown_device_joined"
    state.owed_change_alerts[kind] = [_owed_clients("k1").as_dict()]
    state.owed_rule_idle[kind] = ["not", "a", "dict"]

    await cast("Any", engine)._expire_idle_owed(set(), _snapshot(), NOW)

    assert state.owed_change_alerts == {}
    assert state.owed_rule_idle == {}
    assert any("could not be read" in r.message for r in caplog.records)


@pytest.mark.asyncio
async def test_a_rule_running_at_level_zero_is_not_idle(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """At level 0 the rule still counts as running, so nothing ages out."""
    engine, _notifier, _section = await _radio_engine(monkeypatch)
    state = cast("Any", engine)._suppression.state
    state.owed_rule_idle["radio_new_device_joined"] = {"days": 3, "last": "2026-09-01"}

    await engine._timed_run()

    assert list(_owed(engine)) == ["radio_new_device_joined"]
    assert state.owed_rule_idle == {}
    assert engine.owed_status() == {
        "owed_change_alerts": {"radio_new_device_joined": 1}
    }


@pytest.mark.asyncio
async def test_an_exclusion_added_since_drops_only_that_device(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Excluding one owed client's tracker still tells the others (#802)."""
    engine, _ = _engine(
        monkeypatch,
        _snapshot(),
        options={
            "sentinel_rule_entity_exclusions": {
                "network_unknown_device_joined": ["device_tracker.k1"]
            }
        },
    )
    state = cast("Any", engine)._suppression.state
    state.owed_change_alerts["network_unknown_device_joined"] = [
        _owed_clients("k1", "k2").as_dict()
    ]
    notifier = cast("DummyNotifier", cast("Any", engine)._notifier)

    await cast("Any", engine)._send_owed_kind(
        "network_unknown_device_joined",
        [],
        {},
        _snapshot(),
        NOW,
        timedelta(0),
        timedelta(0),
        explain_enabled=False,
        trigger_source="poll",
        inventory_keys=[],
    )

    assert len(notifier.calls) == 1
    sent = notifier.calls[0]
    assert sent.evidence["client_keys"] == ["k2"]
    assert sent.triggering_entities == ["device_tracker.k2"]
    assert "Phone k1" not in sent.evidence["summary"]
    assert "New device on the network: Phone k2." in sent.evidence["summary"]
    assert state.owed_change_alerts == {}


@pytest.mark.asyncio
async def test_an_owed_alert_with_every_device_excluded_is_dropped(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    engine, _ = _engine(
        monkeypatch,
        _snapshot(),
        options={
            "sentinel_rule_entity_exclusions": {
                "network_unknown_device_joined": ["device_tracker.*"]
            }
        },
    )
    state = cast("Any", engine)._suppression.state
    state.owed_change_alerts["network_unknown_device_joined"] = [
        _owed_clients("k1", "k2").as_dict()
    ]
    notifier = cast("DummyNotifier", cast("Any", engine)._notifier)

    await cast("Any", engine)._send_owed_kind(
        "network_unknown_device_joined",
        [],
        {},
        _snapshot(),
        NOW,
        timedelta(0),
        timedelta(0),
        explain_enabled=False,
        trigger_source="poll",
        inventory_keys=[],
    )

    assert notifier.calls == []
    assert state.owed_change_alerts == {}

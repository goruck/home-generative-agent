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
        self.calls.append({"finding": finding, **kwargs})


def _snapshot(
    ha_security: dict[str, Any] | None = None, *, with_network: bool = True
) -> FullStateSnapshot:
    ha = ha_security or {}
    caps = sorted(ha_cap(k) for k in ha)
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
    return validate_snapshot(payload)


def _engine(
    monkeypatch: pytest.MonkeyPatch,
    snapshot: FullStateSnapshot,
    *,
    options: dict[str, Any] | None = None,
    auth_inventory: AuthInventory | None = None,
    hass: Any = None,
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
        explainer=None,
        auth_inventory=auth_inventory,
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

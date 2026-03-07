# ruff: noqa: S101
"""End-to-end sentinel test with synthetic snapshot."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any, cast
from unittest.mock import AsyncMock, MagicMock

import pytest

from custom_components.home_generative_agent.const import (
    CONF_SENTINEL_AUTO_EXEC_CANARY_MODE,
    CONF_SENTINEL_AUTO_EXECUTE_ALLOWED_SERVICES,
    CONF_SENTINEL_AUTO_EXECUTE_DEFAULT_MIN_CONFIDENCE,
    CONF_SENTINEL_AUTO_EXECUTE_MAX_ACTIONS_PER_HOUR,
    CONF_SENTINEL_AUTO_EXECUTION_ENABLED,
    CONF_SENTINEL_PENDING_PROMPT_TTL_MINUTES,
    CONF_SENTINEL_STALENESS_THRESHOLD_SECONDS,
)
from custom_components.home_generative_agent.sentinel.engine import SentinelEngine
from custom_components.home_generative_agent.sentinel.models import AnomalyFinding
from custom_components.home_generative_agent.sentinel.suppression import (
    SuppressionManager,
    SuppressionState,
)
from custom_components.home_generative_agent.snapshot.schema import (
    FullStateSnapshot,
    validate_snapshot,
)

if TYPE_CHECKING:
    from homeassistant.core import HomeAssistant

    from custom_components.home_generative_agent.audit.store import AuditStore
    from custom_components.home_generative_agent.sentinel.notifier import (
        SentinelNotifier,
    )


class DummySuppression(SuppressionManager):
    """Suppression manager stub."""

    def __init__(self) -> None:  # type: ignore[override]
        self._state = SuppressionState()
        self._read_only = False

    @property
    def state(self) -> SuppressionState:  # type: ignore[override]
        return self._state

    @property
    def is_read_only(self) -> bool:  # type: ignore[override]
        return self._read_only

    async def async_save(self) -> None:  # type: ignore[override]
        return None


class DummyNotifier:
    """Notification dispatcher stub."""

    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    async def async_notify(self, finding, snapshot, explanation) -> None:  # type: ignore[no-untyped-def]
        self.calls.append({"finding": finding, "snapshot": snapshot})


class DummyAudit:
    """Audit store stub."""

    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    async def async_append_finding(  # type: ignore[no-untyped-def]
        self, snapshot, finding, explanation, **kwargs: Any
    ) -> None:
        self.calls.append(
            {
                "finding": finding,
                "snapshot": snapshot,
                "suppression_reason_code": kwargs.get("suppression_reason_code"),
                "canary_would_execute": kwargs.get("canary_would_execute"),
                "action_policy_path": kwargs.get("action_policy_path"),
                "action_outcome": kwargs.get("action_outcome"),
            }
        )


@pytest.mark.asyncio
async def test_sentinel_end_to_end(monkeypatch: pytest.MonkeyPatch) -> None:
    """Sentinel processes a snapshot and emits a finding."""
    snapshot: FullStateSnapshot = validate_snapshot(
        {
            "schema_version": 1,
            "generated_at": "2025-01-01T00:00:00+00:00",
            "entities": [
                {
                    "entity_id": "binary_sensor.front_door",
                    "domain": "binary_sensor",
                    "state": "on",
                    "friendly_name": "Front Door",
                    "area": "Front",
                    "attributes": {"device_class": "door"},
                    "last_changed": "2025-01-01T00:00:00+00:00",
                    "last_updated": "2025-01-01T00:00:00+00:00",
                }
            ],
            "camera_activity": [],
            "derived": {
                "now": "2025-01-01T00:00:00+00:00",
                "timezone": "UTC",
                "is_night": False,
                "anyone_home": False,
                "people_home": [],
                "people_away": [],
                "last_motion_by_area": {},
            },
        }
    )

    async def _fake_build(_hass: HomeAssistant) -> FullStateSnapshot:
        return snapshot

    monkeypatch.setattr(
        "custom_components.home_generative_agent.sentinel.engine.async_build_full_state_snapshot",
        _fake_build,
    )

    engine = SentinelEngine(
        hass=cast("HomeAssistant", object()),
        options={
            "sentinel_cooldown_minutes": 0,
            "sentinel_entity_cooldown_minutes": 0,
            "sentinel_interval_seconds": 60,
            "explain_enabled": False,
        },
        suppression=DummySuppression(),
        notifier=cast("SentinelNotifier", DummyNotifier()),
        audit_store=cast("AuditStore", DummyAudit()),
        explainer=None,
    )

    await engine._run_once()

    notifier = cast("DummyNotifier", cast("Any", engine)._notifier)
    audit_store = cast("DummyAudit", cast("Any", engine)._audit_store)
    assert notifier.calls
    assert audit_store.calls
    assert audit_store.calls[0]["suppression_reason_code"] == "not_suppressed"


@pytest.mark.asyncio
async def test_sentinel_canary_mode_records_would_execute(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Canary mode records canary_would_execute=True without executing services."""
    snapshot: FullStateSnapshot = validate_snapshot(
        {
            "schema_version": 1,
            "generated_at": "2025-01-01T01:00:00+00:00",
            "entities": [
                {
                    "entity_id": "binary_sensor.front_door",
                    "domain": "binary_sensor",
                    "state": "on",
                    "friendly_name": "Front Door",
                    "area": "Front",
                    "attributes": {"device_class": "door"},
                    "last_changed": "2025-01-01T00:59:50+00:00",
                    "last_updated": "2025-01-01T00:59:50+00:00",
                }
            ],
            "camera_activity": [],
            "derived": {
                "now": "2025-01-01T01:00:00+00:00",
                "timezone": "UTC",
                "is_night": False,
                "anyone_home": False,
                "people_home": [],
                "people_away": [],
                "last_motion_by_area": {},
            },
        }
    )

    async def _fake_build(_hass: HomeAssistant) -> FullStateSnapshot:
        return snapshot

    monkeypatch.setattr(
        "custom_components.home_generative_agent.sentinel.engine.async_build_full_state_snapshot",
        _fake_build,
    )

    engine = SentinelEngine(
        hass=cast("HomeAssistant", object()),
        options={
            "sentinel_cooldown_minutes": 0,
            "sentinel_entity_cooldown_minutes": 0,
            "sentinel_interval_seconds": 60,
            "explain_enabled": False,
            CONF_SENTINEL_AUTO_EXEC_CANARY_MODE: True,
            CONF_SENTINEL_AUTO_EXECUTION_ENABLED: True,
            CONF_SENTINEL_AUTO_EXECUTE_DEFAULT_MIN_CONFIDENCE: 0.0,
            CONF_SENTINEL_AUTO_EXECUTE_MAX_ACTIONS_PER_HOUR: 10,
            CONF_SENTINEL_AUTO_EXECUTE_ALLOWED_SERVICES: [],
            CONF_SENTINEL_STALENESS_THRESHOLD_SECONDS: 3600,
            "sentinel_autonomy_level": 2,
        },
        suppression=DummySuppression(),
        notifier=cast("SentinelNotifier", DummyNotifier()),
        audit_store=cast("AuditStore", DummyAudit()),
        explainer=None,
    )

    # Patch get_autonomy_level so the engine uses level 2.
    monkeypatch.setattr(engine, "get_autonomy_level", lambda _entry_id: 2)
    engine._entry_id = "test_entry"

    await engine._run_once()

    audit_store = cast("DummyAudit", cast("Any", engine)._audit_store)
    assert audit_store.calls
    # At least one finding should have canary_would_execute recorded (True or False).
    canary_values = [c["canary_would_execute"] for c in audit_store.calls]
    assert any(v is not None for v in canary_values)


@pytest.mark.asyncio
async def test_sentinel_canary_mode_does_not_consume_live_auto_execute(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A canary-only pass must not burn the later live auto-execute slot."""
    engine, hass = _make_ae_engine(monkeypatch)
    engine._options[CONF_SENTINEL_AUTO_EXEC_CANARY_MODE] = True

    await engine._run_once()

    hass.services.async_call.assert_not_called()
    audit = cast("DummyAudit", cast("Any", engine)._audit_store)
    assert audit.calls
    assert audit.calls[0]["canary_would_execute"] is True
    assert audit.calls[0]["action_policy_path"] == "auto_execute"

    engine._options[CONF_SENTINEL_AUTO_EXEC_CANARY_MODE] = False
    await engine._run_once()

    hass.services.async_call.assert_called_once_with(
        "lock",
        "lock",
        {"entity_id": "lock.front_door"},
        blocking=True,
    )
    assert len(audit.calls) >= 2
    assert audit.calls[-1]["action_policy_path"] == "auto_execute"


# ---------------------------------------------------------------------------
# Issue #264 — Level 2 live auto-execute integration tests
# ---------------------------------------------------------------------------

_AE_SNAPSHOT: FullStateSnapshot = validate_snapshot(
    {
        "schema_version": 1,
        "generated_at": "2025-01-01T01:00:00+00:00",
        "entities": [
            {
                "entity_id": "lock.front_door",
                "domain": "lock",
                "state": "unlocked",
                "friendly_name": "Front Door Lock",
                "area": "Front",
                "attributes": {},
                "last_changed": "2025-01-01T00:59:50+00:00",
                "last_updated": "2025-01-01T00:59:50+00:00",
            }
        ],
        "camera_activity": [],
        "derived": {
            "now": "2025-01-01T01:00:00+00:00",
            "timezone": "UTC",
            "is_night": False,
            "anyone_home": False,
            "people_home": [],
            "people_away": [],
            "last_motion_by_area": {},
        },
    }
)


class StubAutoExecRule:
    """Rule that always emits a finding with a lock.lock service action."""

    rule_id = "stub_auto_exec"

    def evaluate(self, snapshot: FullStateSnapshot) -> list[AnomalyFinding]:  # type: ignore[override]
        """Return a fixed finding with a service-type suggested action."""
        return [
            AnomalyFinding(
                anomaly_id="stub-ae-finding",
                type="stub_auto_exec",
                severity="medium",
                confidence=0.9,
                triggering_entities=["lock.front_door"],
                evidence={},
                suggested_actions=["lock.lock"],
                is_sensitive=False,
            )
        ]


_AE_FROZEN_NOW = datetime(2025, 1, 1, 1, 0, 0, tzinfo=UTC)


def _make_ae_engine(
    monkeypatch: pytest.MonkeyPatch,
    *,
    autonomy_level: int = 2,
    max_actions_per_hour: int = 5,
    service_side_effect: Any = None,
) -> tuple[SentinelEngine, MagicMock]:
    """
    Build a live-auto-execute engine backed by a MagicMock hass.

    ``dt_util.utcnow`` is frozen to ``_AE_FROZEN_NOW`` so that the snapshot
    entity (last_changed 10 s before now) is always considered fresh, and
    time-sensitive guardrails (rate limit, idempotency) behave deterministically.
    ``CONF_SENTINEL_PENDING_PROMPT_TTL_MINUTES`` is set to 0 so that pending
    prompts expire immediately — allowing consecutive ``_run_once()`` calls to
    reach the execution-policy layer without being short-circuited by suppression.
    """
    hass = MagicMock()
    if service_side_effect is None:
        hass.services.async_call = AsyncMock(return_value=None)
    else:
        hass.services.async_call = AsyncMock(side_effect=service_side_effect)

    # Freeze time so snapshot entities are always fresh and guardrails are deterministic.
    monkeypatch.setattr(
        "homeassistant.util.dt.utcnow",
        lambda: _AE_FROZEN_NOW,
    )

    async def _fake_build(_hass: Any) -> FullStateSnapshot:
        return _AE_SNAPSHOT

    monkeypatch.setattr(
        "custom_components.home_generative_agent.sentinel.engine.async_build_full_state_snapshot",
        _fake_build,
    )

    engine = SentinelEngine(
        hass=hass,
        options={
            "sentinel_cooldown_minutes": 0,
            "sentinel_entity_cooldown_minutes": 0,
            "sentinel_interval_seconds": 60,
            "explain_enabled": False,
            CONF_SENTINEL_AUTO_EXEC_CANARY_MODE: False,
            CONF_SENTINEL_AUTO_EXECUTION_ENABLED: True,
            CONF_SENTINEL_AUTO_EXECUTE_DEFAULT_MIN_CONFIDENCE: 0.0,
            CONF_SENTINEL_AUTO_EXECUTE_MAX_ACTIONS_PER_HOUR: max_actions_per_hour,
            CONF_SENTINEL_AUTO_EXECUTE_ALLOWED_SERVICES: ["lock.lock"],
            CONF_SENTINEL_STALENESS_THRESHOLD_SECONDS: 3600,
            CONF_SENTINEL_PENDING_PROMPT_TTL_MINUTES: 0,
            "sentinel_autonomy_level": autonomy_level,
        },
        suppression=DummySuppression(),
        notifier=cast("SentinelNotifier", DummyNotifier()),
        audit_store=cast("AuditStore", DummyAudit()),
        explainer=None,
    )
    engine._rules = [StubAutoExecRule()]  # type: ignore[assignment]
    monkeypatch.setattr(engine, "get_autonomy_level", lambda _entry_id: autonomy_level)
    engine._entry_id = "test_entry"
    return engine, hass


@pytest.mark.asyncio
async def test_live_auto_execute_calls_ha_service(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Engine dispatches auto-execute finding and calls hass.services.async_call."""
    engine, hass = _make_ae_engine(monkeypatch)

    await engine._run_once()

    hass.services.async_call.assert_called_once_with(
        "lock",
        "lock",
        {"entity_id": "lock.front_door"},
        blocking=True,
    )


@pytest.mark.asyncio
async def test_live_auto_execute_audit_outcome_populated(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Audit record includes action_policy_path=auto_execute and action_outcome."""
    engine, _hass = _make_ae_engine(monkeypatch)

    await engine._run_once()

    audit = cast("DummyAudit", cast("Any", engine)._audit_store)
    ae_calls = [c for c in audit.calls if c["action_policy_path"] == "auto_execute"]
    assert ae_calls, "Expected at least one auto_execute audit record"
    assert cast("dict[str, str]", ae_calls[0]["action_outcome"])["status"] == "success"


@pytest.mark.asyncio
async def test_live_auto_execute_blocked_below_level_2(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Autonomy level < 2 prevents hass.services.async_call from being invoked."""
    engine, hass = _make_ae_engine(monkeypatch, autonomy_level=1)

    await engine._run_once()

    hass.services.async_call.assert_not_called()
    audit = cast("DummyAudit", cast("Any", engine)._audit_store)
    assert audit.calls, "Expected audit records"
    assert all(c["action_policy_path"] != "auto_execute" for c in audit.calls)


@pytest.mark.asyncio
async def test_live_auto_execute_idempotency_prevents_double_fire(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Same finding in two consecutive runs triggers hass.services.async_call only once."""
    # Use a high rate limit so only idempotency can block the second run.
    engine, hass = _make_ae_engine(monkeypatch, max_actions_per_hour=100)

    await engine._run_once()
    await engine._run_once()

    assert hass.services.async_call.call_count == 1
    audit = cast("DummyAudit", cast("Any", engine)._audit_store)
    paths = [c["action_policy_path"] for c in audit.calls]
    assert "auto_execute" in paths
    assert "prompt_user" in paths


@pytest.mark.asyncio
async def test_live_auto_execute_rate_limit_blocks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Rate limiter blocks auto-execution once max_actions_per_hour is exhausted."""
    engine, hass = _make_ae_engine(monkeypatch, max_actions_per_hour=1)

    await engine._run_once()
    await engine._run_once()

    assert hass.services.async_call.call_count == 1
    audit = cast("DummyAudit", cast("Any", engine)._audit_store)
    paths = [c["action_policy_path"] for c in audit.calls]
    assert "auto_execute" in paths
    assert "prompt_user" in paths


@pytest.mark.asyncio
async def test_live_auto_execute_failure_does_not_consume_retry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A failed service call must not burn idempotency or the rate-limit slot."""
    first_call = True

    async def _service_side_effect(*_args: Any, **_kwargs: Any) -> None:
        nonlocal first_call
        if first_call:
            first_call = False
            msg = "temporary HA failure"
            raise RuntimeError(msg)

    engine, hass = _make_ae_engine(
        monkeypatch,
        service_side_effect=_service_side_effect,
    )

    await engine._run_once()
    await engine._run_once()

    assert hass.services.async_call.call_count == 2
    audit = cast("DummyAudit", cast("Any", engine)._audit_store)
    first_outcome = cast("dict[str, Any]", audit.calls[0]["action_outcome"])
    second_outcome = cast("dict[str, Any]", audit.calls[1]["action_outcome"])
    assert first_outcome == {
        "status": "error",
        "actions": [
            {
                "service": "lock.lock",
                "status": "error",
                "error": "temporary HA failure",
            }
        ],
        "execution_id": first_outcome["execution_id"],
    }
    assert second_outcome == {
        "status": "success",
        "actions": [{"service": "lock.lock", "status": "ok", "error": None}],
        "execution_id": second_outcome["execution_id"],
    }

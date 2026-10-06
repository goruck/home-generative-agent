# ruff: noqa: S101
"""End-to-end sentinel test with synthetic snapshot."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, Any, Literal, cast
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from custom_components.home_generative_agent.const import (
    ACTION_POLICY_BLOCKED,
    CONF_SENTINEL_AUTO_EXEC_CANARY_MODE,
    CONF_SENTINEL_AUTO_EXECUTE_ALLOWED_SERVICES,
    CONF_SENTINEL_AUTO_EXECUTE_DEFAULT_MIN_CONFIDENCE,
    CONF_SENTINEL_AUTO_EXECUTE_MAX_ACTIONS_PER_HOUR,
    CONF_SENTINEL_AUTO_EXECUTION_ENABLED,
    CONF_SENTINEL_PENDING_PROMPT_TTL_MINUTES,
    CONF_SENTINEL_STALENESS_THRESHOLD_SECONDS,
)
from custom_components.home_generative_agent.sentinel.engine import (
    SentinelEngine,
    _delivered_reason,
)
from custom_components.home_generative_agent.sentinel.execution import (
    ActionPolicyResult,
)
from custom_components.home_generative_agent.sentinel.models import (
    AnomalyFinding,
    CompoundFinding,
)
from custom_components.home_generative_agent.sentinel.notifier import (
    MAX_ALSO_LINE_CHARS,
)
from custom_components.home_generative_agent.sentinel.suppression import (
    SUPPRESSION_REASON_POLICY_BLOCKED,
    SUPPRESSION_REASON_TRIAGE_SUPPRESSED,
    SuppressionManager,
    SuppressionState,
)
from custom_components.home_generative_agent.sentinel.triage import (
    TRIAGE_SUPPRESS,
    TriageDecision,
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

    async def async_notify(  # type: ignore[no-untyped-def]
        self, finding, snapshot, explanation, also_line=None
    ) -> bool | None:
        self.calls.append(
            {"finding": finding, "snapshot": snapshot, "also_line": also_line}
        )
        return None


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
                "shown_anomaly_id": kwargs.get("shown_anomaly_id"),
                "named_anomaly_ids": kwargs.get("named_anomaly_ids"),
                "triage_decision": kwargs.get("triage_decision"),
                "triage_reason_code": kwargs.get("triage_reason_code"),
                "canary_would_execute": kwargs.get("canary_would_execute"),
                "action_policy_path": kwargs.get("action_policy_path"),
                "action_outcome": kwargs.get("action_outcome"),
                "trigger_source": kwargs.get("trigger_source"),
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

    async def _fake_build(_hass: HomeAssistant, **_kwargs: Any) -> FullStateSnapshot:
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


def test_delivered_reason_keeps_the_code_unless_the_push_was_dropped() -> None:
    assert _delivered_reason("not_suppressed", sent=True) == "not_suppressed"
    assert _delivered_reason("not_suppressed", sent=None) == "not_suppressed"
    assert _delivered_reason("not_suppressed", sent=False) == "notifier_duplicate"
    assert _delivered_reason("type_cooldown", sent=False) == "notifier_duplicate"


class _DroppingNotifier(DummyNotifier):
    """A notifier that drops every push as a repeat within its cooldown."""

    async def async_notify(  # type: ignore[no-untyped-def]
        self, finding, snapshot, explanation, also_line=None
    ) -> bool:
        await super().async_notify(finding, snapshot, explanation, also_line)
        return False


@pytest.mark.asyncio
async def test_sentinel_audits_a_notifier_dropped_repeat_as_not_delivered(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A push the notifier drops as a repeat is not audited as reaching the user."""
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

    async def _fake_build(_hass: HomeAssistant, **_kwargs: Any) -> FullStateSnapshot:
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
        notifier=cast("SentinelNotifier", _DroppingNotifier()),
        audit_store=cast("AuditStore", DummyAudit()),
        explainer=None,
    )

    await engine._run_once()

    notifier = cast("_DroppingNotifier", cast("Any", engine)._notifier)
    audit_store = cast("DummyAudit", cast("Any", engine)._audit_store)
    assert notifier.calls
    assert audit_store.calls
    assert audit_store.calls[0]["suppression_reason_code"] == "notifier_duplicate"


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

    async def _fake_build(_hass: HomeAssistant, **_kwargs: Any) -> FullStateSnapshot:
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

    requires: frozenset[str] = frozenset()

    cooldown_minutes = 0

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

    async def _fake_build(_hass: Any, **_kwargs: Any) -> FullStateSnapshot:
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


# ---------------------------------------------------------------------------
# Helpers shared by Gap 1/2/3 tests
# ---------------------------------------------------------------------------


def _make_snapshot() -> FullStateSnapshot:
    return validate_snapshot(
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


def _make_engine(
    monkeypatch: pytest.MonkeyPatch,
    snapshot: FullStateSnapshot,
    *,
    cooldown_minutes: int = 30,
) -> tuple[SentinelEngine, DummyNotifier, DummyAudit]:
    """Return a wired SentinelEngine with DummyNotifier and DummyAudit."""

    async def _fake_build(_hass: HomeAssistant, **_kwargs: Any) -> FullStateSnapshot:
        return snapshot

    monkeypatch.setattr(
        "custom_components.home_generative_agent.sentinel.engine.async_build_full_state_snapshot",
        _fake_build,
    )
    notifier = DummyNotifier()
    audit = DummyAudit()
    engine = SentinelEngine(
        hass=cast("HomeAssistant", object()),
        options={
            "sentinel_cooldown_minutes": cooldown_minutes,
            "sentinel_entity_cooldown_minutes": cooldown_minutes,
            "sentinel_interval_seconds": 60,
            "explain_enabled": False,
        },
        suppression=DummySuppression(),
        notifier=cast("SentinelNotifier", notifier),
        audit_store=cast("AuditStore", audit),
        explainer=None,
    )
    return engine, notifier, audit


# ---------------------------------------------------------------------------
# Gap 1: suppressed findings produce an audit record
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_suppressed_finding_creates_audit_record(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A finding suppressed by cooldown must still produce an audit record."""
    snapshot = _make_snapshot()
    engine, notifier, audit = _make_engine(monkeypatch, snapshot, cooldown_minutes=30)

    # First run: finding fires and registers a cooldown.
    await engine._run_once("poll")
    assert len(notifier.calls) == 1

    # Second run: same finding is in cooldown — suppressed.
    await engine._run_once("poll")

    # Notifier must not have fired a second time.
    assert len(notifier.calls) == 1

    # Audit must have a second record for the suppressed finding.
    assert len(audit.calls) == 2
    suppressed_record = audit.calls[1]
    assert suppressed_record["suppression_reason_code"] is not None
    assert suppressed_record["suppression_reason_code"] != "not_suppressed"
    assert suppressed_record["action_policy_path"] is None


@pytest.mark.asyncio
async def test_suppression_gate_fires_before_triage(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A suppressed finding must produce exactly one audit record (no triage call)."""
    snapshot = _make_snapshot()
    engine, _notifier, audit = _make_engine(monkeypatch, snapshot, cooldown_minutes=30)

    # Prime the cooldown with first run.
    await engine._run_once("poll")
    first_count = len(audit.calls)

    # Second run: suppressed — must produce exactly one more audit record.
    await engine._run_once("poll")
    assert len(audit.calls) == first_count + 1

    # The suppressed record must have no triage decision (triage never ran).
    suppressed = audit.calls[-1]
    assert suppressed.get("suppression_reason_code") not in (None, "not_suppressed")


# ---------------------------------------------------------------------------
# Gap 2: BLOCKED findings do not notify but do produce an audit record
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_blocked_finding_no_notification(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A BLOCKED finding must not trigger async_notify."""
    snapshot = _make_snapshot()
    engine, notifier, _audit = _make_engine(monkeypatch, snapshot, cooldown_minutes=0)

    blocked_result = ActionPolicyResult(
        action_policy_path=ACTION_POLICY_BLOCKED,
        data_quality="unavailable",
        data_quality_details={},
        execution_id=None,
        block_reason="data_quality_unavailable",
    )

    with patch.object(
        cast("Any", engine)._execution_service,
        "evaluate_canary",
        return_value=blocked_result,
    ):
        await engine._run_once("poll")

    assert len(notifier.calls) == 0


@pytest.mark.asyncio
async def test_blocked_finding_creates_audit_record(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A BLOCKED finding must produce an audit record with action_policy_path='blocked'."""
    snapshot = _make_snapshot()
    engine, _notifier, audit = _make_engine(monkeypatch, snapshot, cooldown_minutes=0)

    blocked_result = ActionPolicyResult(
        action_policy_path=ACTION_POLICY_BLOCKED,
        data_quality="unavailable",
        data_quality_details={},
        execution_id=None,
        block_reason="data_quality_unavailable",
    )

    with patch.object(
        cast("Any", engine)._execution_service,
        "evaluate_canary",
        return_value=blocked_result,
    ):
        await engine._run_once("poll")

    assert len(audit.calls) == 1
    assert audit.calls[0]["action_policy_path"] == ACTION_POLICY_BLOCKED
    assert (
        audit.calls[0]["suppression_reason_code"] == SUPPRESSION_REASON_POLICY_BLOCKED
    )


@pytest.mark.asyncio
async def test_triage_suppressed_finding_audit_reason(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A triage-suppressed finding must audit as triage_suppressed, not notify."""
    snapshot = _make_snapshot()
    engine, notifier, audit = _make_engine(monkeypatch, snapshot, cooldown_minutes=0)

    triage_service = MagicMock()
    triage_service.triage = AsyncMock(
        return_value=TriageDecision(
            decision=TRIAGE_SUPPRESS,
            reason_code="routine_state",
            triage_confidence=0.9,
            summary="routine",
        )
    )
    cast("Any", engine)._triage_service = triage_service

    await engine._run_once("poll")

    assert len(notifier.calls) == 0
    assert len(audit.calls) == 1
    assert (
        audit.calls[0]["suppression_reason_code"]
        == SUPPRESSION_REASON_TRIAGE_SUPPRESSED
    )
    assert audit.calls[0]["triage_decision"] == TRIAGE_SUPPRESS
    assert audit.calls[0]["triage_reason_code"] == "routine_state"


@pytest.mark.asyncio
async def test_compound_blocked_finding_audit_reason(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A BLOCKED compound finding must audit as policy_blocked, not notify."""
    snapshot = _make_snapshot()
    engine, notifier, audit = _make_engine(monkeypatch, snapshot, cooldown_minutes=0)

    finding = AnomalyFinding(
        anomaly_id="test-compound-1",
        type="unlocked_lock_at_night",
        severity="high",
        confidence=0.9,
        triggering_entities=["lock.front_door"],
        evidence={"state": "unlocked"},
        suggested_actions=["lock.lock"],
        is_sensitive=False,
    )
    compound = CompoundFinding.from_findings([finding])

    blocked_result = ActionPolicyResult(
        action_policy_path=ACTION_POLICY_BLOCKED,
        data_quality="unavailable",
        data_quality_details={},
        execution_id=None,
        block_reason="data_quality_unavailable",
    )

    with patch.object(
        cast("Any", engine)._execution_service,
        "evaluate_canary",
        return_value=blocked_result,
    ):
        await engine._dispatch_compound(
            compound,
            snapshot,
            datetime.now(UTC),
            timedelta(minutes=0),
            timedelta(minutes=0),
            False,  # noqa: FBT003
        )

    assert len(notifier.calls) == 0
    assert len(audit.calls) == 1
    assert (
        audit.calls[0]["suppression_reason_code"] == SUPPRESSION_REASON_POLICY_BLOCKED
    )
    assert audit.calls[0]["action_policy_path"] == ACTION_POLICY_BLOCKED


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("notifier_factory", "expected"),
    [
        (_DroppingNotifier, "notifier_duplicate"),
        (DummyNotifier, "not_suppressed"),
    ],
)
async def test_compound_audit_reason_follows_the_notifier(
    monkeypatch: pytest.MonkeyPatch, notifier_factory: Any, expected: str
) -> None:
    """The compound path labels a dropped repeat the same way as a plain one."""
    snapshot = _make_snapshot()
    engine, _notifier, audit = _make_engine(monkeypatch, snapshot, cooldown_minutes=0)
    notifier = notifier_factory()
    cast("Any", engine)._notifier = notifier

    finding = AnomalyFinding(
        anomaly_id="test-compound-dup",
        type="open_entry_when_home_window",
        severity="medium",
        confidence=0.9,
        triggering_entities=["binary_sensor.kitchen_window"],
        evidence={"state": "on"},
        suggested_actions=[],
        is_sensitive=False,
    )
    compound = CompoundFinding.from_findings([finding])

    await engine._dispatch_compound(
        compound,
        snapshot,
        datetime.now(UTC),
        timedelta(minutes=0),
        timedelta(minutes=0),
        False,  # noqa: FBT003
    )

    assert len(notifier.calls) == 1
    assert len(audit.calls) == 1
    assert audit.calls[0]["suppression_reason_code"] == expected


def _standing_finding(
    anomaly_id: str,
    finding_type: str,
    *,
    severity: Literal["low", "medium", "high"] = "medium",
    confidence: float = 0.9,
    entities: list[str] | None = None,
) -> AnomalyFinding:
    return AnomalyFinding(
        anomaly_id=anomaly_id,
        type=finding_type,
        severity=severity,
        confidence=confidence,
        triggering_entities=(
            ["lock.garage_door_lock"] if entities is None else entities
        ),
        evidence={"state": "unlocked"},
        suggested_actions=[],
        is_sensitive=False,
    )


async def _dispatch_group(
    engine: SentinelEngine, *findings: AnomalyFinding, at: datetime
) -> bool:
    return await engine._dispatch_compound(
        CompoundFinding.from_findings(list(findings)),
        _make_snapshot(),
        at,
        timedelta(minutes=30),
        timedelta(minutes=15),
        False,  # noqa: FBT003
    )


@pytest.mark.asyncio
async def test_compound_shows_the_constituent_that_came_due(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    A compound push shows a constituent that passed, never a pending partner.

    Issue #723: the confident lock finding was still pending when the duration
    finding joined it, yet every push showed the lock again.  The pending
    lock's own prompt clock is left alone: it reminds when it runs out.
    """
    engine, notifier, _audit = _make_engine(monkeypatch, _make_snapshot())
    suppression = cast("DummySuppression", cast("Any", engine)._suppression)
    t0 = datetime(2025, 1, 1, 12, 0, tzinfo=UTC)
    lock = _standing_finding("lock-id", "unlocked_lock_when_home", confidence=0.9)
    duration = _standing_finding(
        "duration-id", "garage_door_unlocked_duration", confidence=0.5
    )

    assert await _dispatch_group(engine, lock, at=t0)
    assert await _dispatch_group(engine, lock, duration, at=t0 + timedelta(hours=2))

    assert notifier.calls[1]["finding"] is duration
    assert suppression.state.pending_prompts["lock-id"] == t0.isoformat()


@pytest.mark.asyncio
async def test_compound_does_not_mark_a_held_safety_finding_as_prompted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    A constituent held back this cycle is not registered by its partner's push.

    The open door is inside presence grace when the lock alerts; the push
    shows the lock, so the door must still alert once grace ends.
    """
    engine, notifier, _audit = _make_engine(monkeypatch, _make_snapshot())
    suppression = cast("DummySuppression", cast("Any", engine)._suppression)
    t0 = datetime(2025, 1, 1, 22, 0, tzinfo=UTC)
    suppression.state.presence_grace_until["person.someone"] = (
        t0 + timedelta(minutes=10)
    ).isoformat()
    lock = _standing_finding("lock-id", "unlocked_lock_at_night", confidence=0.9)
    door = _standing_finding("door-id", "open_entry_while_away", confidence=0.6)

    assert await _dispatch_group(engine, lock, door, at=t0)
    assert notifier.calls[0]["finding"] is lock
    assert "door-id" not in suppression.state.pending_prompts
    assert "open_entry_while_away" not in suppression.state.last_by_type

    assert await _dispatch_group(engine, lock, door, at=t0 + timedelta(minutes=15))
    assert notifier.calls[1]["finding"] is door


@pytest.mark.asyncio
async def test_compound_prompts_only_the_constituent_it_showed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    A passing constituent the push neither showed nor named is not prompted.

    With stable ids, a held loser matched its pending prompt and lost the
    pick again whenever both prompts expired, so it was never pushed while its
    partner stood.  It now only cools down and comes due on its own.  A
    finding with no devices cannot go on the also line (issue #727).
    """
    engine, notifier, _audit = _make_engine(monkeypatch, _make_snapshot())
    suppression = cast("DummySuppression", cast("Any", engine)._suppression)
    t0 = datetime(2025, 1, 1, 12, 0, tzinfo=UTC)
    door = _standing_finding("door-id", "open_entry_while_away", confidence=0.9)
    lock = _standing_finding(
        "lock-id", "garage_door_unlocked_duration", confidence=0.5, entities=[]
    )

    assert await _dispatch_group(engine, door, lock, at=t0)
    assert notifier.calls[0]["finding"] is door
    assert "lock-id" not in suppression.state.pending_prompts
    assert suppression.state.last_by_type["garage_door_unlocked_duration"] == (
        t0.isoformat()
    )

    # Inside the cooldown nothing is due; after it the lock is pushed itself.
    assert not await _dispatch_group(engine, door, lock, at=t0 + timedelta(minutes=20))
    assert await _dispatch_group(engine, door, lock, at=t0 + timedelta(minutes=31))
    assert notifier.calls[1]["finding"] is lock


@pytest.mark.asyncio
async def test_compound_shows_the_most_severe_due_constituent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A confident low-severity finding does not hide a high-severity one."""
    engine, notifier, _audit = _make_engine(monkeypatch, _make_snapshot())
    appliance = _standing_finding(
        "appliance-id", "appliance_power_duration", severity="low", confidence=0.95
    )
    lock = _standing_finding(
        "lock-id", "unlocked_lock_at_night", severity="high", confidence=0.4
    )

    assert await _dispatch_group(
        engine, appliance, lock, at=datetime(2025, 1, 1, tzinfo=UTC)
    )

    assert notifier.calls[0]["finding"] is lock


def _window(
    anomaly_id: str, entity_id: str, *, confidence: float = 0.6
) -> AnomalyFinding:
    return _standing_finding(
        anomaly_id,
        "open_entry_while_away_window",
        severity="medium",
        confidence=confidence,
        entities=[entity_id],
    )


@pytest.mark.asyncio
async def test_grouped_push_names_its_due_partners_and_prompts_them(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    One push covers the group: due partners are named and wait out the TTL.

    Issue #727: a 15-finding away group sent a push every 10-30 minutes, one
    per finding.  The push now names the others by device, and each named one
    gets the pending-prompt hold its own push would have given it.
    """
    engine, notifier, audit = _make_engine(monkeypatch, _make_snapshot())
    suppression = cast("DummySuppression", cast("Any", engine)._suppression)
    t0 = datetime(2025, 1, 1, 12, 0, tzinfo=UTC)
    door = _standing_finding(
        "door-id",
        "open_entry_while_away",
        severity="high",
        entities=["binary_sensor.front_door"],
    )
    kitchen = _window("kitchen-id", "binary_sensor.kitchen_window")
    landing = _window("landing-id", "binary_sensor.landing_window", confidence=0.7)
    # Same device as the shown finding: named too, never held silently.
    disarmed = _standing_finding(
        "disarmed-id",
        "alarm_disarmed_open_entry",
        severity="low",
        entities=["binary_sensor.front_door"],
    )

    assert await _dispatch_group(engine, door, kitchen, landing, disarmed, at=t0)

    assert notifier.calls[0]["finding"] is door
    # Front Door has a snapshot friendly name; the others fall back to the id.
    # Most severe, then most confident, first.
    assert notifier.calls[0]["also_line"] == (
        "Also: Landing Window, Kitchen Window, Front Door"
    )
    assert set(suppression.state.pending_prompts) == {
        "door-id",
        "kitchen-id",
        "landing-id",
        "disarmed-id",
    }
    assert audit.calls[-1]["shown_anomaly_id"] == "door-id"
    assert audit.calls[-1]["named_anomaly_ids"] == [
        "landing-id",
        "kitchen-id",
        "disarmed-id",
    ]
    # Every named finding is held, so the group stays quiet after the cooldown.
    assert not await _dispatch_group(
        engine, door, kitchen, landing, disarmed, at=t0 + timedelta(minutes=31)
    )
    assert len(notifier.calls) == 1


@pytest.mark.asyncio
async def test_grouped_push_dropped_as_repeat_names_nothing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A push the notifier dropped never reached the user: partners stay due."""
    engine, _notifier, audit = _make_engine(monkeypatch, _make_snapshot())
    notifier = _DroppingNotifier()
    cast("Any", engine)._notifier = notifier
    suppression = cast("DummySuppression", cast("Any", engine)._suppression)
    door = _standing_finding(
        "door-id", "open_entry_while_away", entities=["binary_sensor.front_door"]
    )
    kitchen = _window("kitchen-id", "binary_sensor.kitchen_window")

    await _dispatch_group(engine, door, kitchen, at=datetime(2025, 1, 1, tzinfo=UTC))

    assert notifier.calls[0]["also_line"] == "Also: Kitchen Window"
    assert "kitchen-id" not in suppression.state.pending_prompts
    assert audit.calls[-1]["named_anomaly_ids"] == []


@pytest.mark.asyncio
async def test_grouped_push_leaves_out_what_does_not_fit_and_keeps_it_due(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    The also line is capped; a finding it could not name gets its own push.

    Counting a finding as alerted when the push never named it would hide it
    for the prompt TTL behind a partner (the #723 review).
    """
    engine, notifier, _audit = _make_engine(monkeypatch, _make_snapshot())
    suppression = cast("DummySuppression", cast("Any", engine)._suppression)
    t0 = datetime(2025, 1, 1, 12, 0, tzinfo=UTC)
    door = _standing_finding(
        "door-id",
        "open_entry_while_away",
        severity="high",
        entities=["binary_sensor.front_door"],
    )
    windows = [
        _window(
            f"w{i}-id",
            f"binary_sensor.upstairs_guest_bedroom_window_{i}",
            confidence=0.9 - i / 100,
        )
        for i in range(8)
    ]

    assert await _dispatch_group(engine, door, *windows, at=t0)

    also_line = cast("str", notifier.calls[0]["also_line"])
    assert len(also_line) <= MAX_ALSO_LINE_CHARS
    named = [w for w in windows if w.anomaly_id in suppression.state.pending_prompts]
    left_out = [w for w in windows if w not in named]
    assert named
    assert left_out
    assert also_line.endswith(f"+{len(left_out)} more")
    for window in named:
        assert (
            window.triggering_entities[0].split(".")[1].replace("_", " ").title()
            in also_line
        )

    # After its cooldown the first one left out is shown itself.
    assert await _dispatch_group(engine, door, *windows, at=t0 + timedelta(minutes=31))
    assert notifier.calls[1]["finding"] is left_out[0]


@pytest.mark.asyncio
async def test_blocked_finding_registers_cooldown_not_prompt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A BLOCKED finding must register a type cooldown but not a pending prompt."""
    snapshot = _make_snapshot()
    engine, _notifier, _audit = _make_engine(monkeypatch, snapshot, cooldown_minutes=0)

    suppression = cast("DummySuppression", cast("Any", engine)._suppression)

    blocked_result = ActionPolicyResult(
        action_policy_path=ACTION_POLICY_BLOCKED,
        data_quality="unavailable",
        data_quality_details={},
        execution_id=None,
        block_reason="data_quality_unavailable",
    )

    with patch.object(
        cast("Any", engine)._execution_service,
        "evaluate_canary",
        return_value=blocked_result,
    ):
        await engine._run_once("poll")

    # Cooldown (last_by_type) must be registered.
    assert suppression.state.last_by_type  # non-empty

    # No pending prompt must be registered (BLOCKED never sends user a prompt).
    assert not suppression.state.pending_prompts


# ---------------------------------------------------------------------------
# Gap 3: trigger_source is populated in audit records
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_trigger_source_poll_in_audit(monkeypatch: pytest.MonkeyPatch) -> None:
    """Audit records from the polling path must have trigger_source='poll'."""
    snapshot = _make_snapshot()
    engine, _notifier, audit = _make_engine(monkeypatch, snapshot, cooldown_minutes=0)

    await engine._run_once("poll")

    assert audit.calls
    assert audit.calls[0]["trigger_source"] == "poll"


@pytest.mark.asyncio
async def test_trigger_source_event_in_audit(monkeypatch: pytest.MonkeyPatch) -> None:
    """Audit records from the event-driven path must have trigger_source='event'."""
    snapshot = _make_snapshot()
    engine, _notifier, audit = _make_engine(monkeypatch, snapshot, cooldown_minutes=0)

    await engine._run_once("event")

    assert audit.calls
    assert audit.calls[0]["trigger_source"] == "event"


@pytest.mark.asyncio
async def test_trigger_source_on_demand_in_audit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Audit records from the on-demand path must have trigger_source='on_demand'."""
    snapshot = _make_snapshot()
    engine, _notifier, audit = _make_engine(monkeypatch, snapshot, cooldown_minutes=0)

    await engine._run_once("on_demand")

    assert audit.calls
    assert audit.calls[0]["trigger_source"] == "on_demand"

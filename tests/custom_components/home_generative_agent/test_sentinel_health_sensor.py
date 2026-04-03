"""Tests for the Sentinel operational health sensor."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any, cast
from unittest.mock import AsyncMock, MagicMock

import pytest

from custom_components.home_generative_agent.core.sentinel_health_sensor import (
    SentinelHealthSensor,
    _compute_kpis,
    _compute_trigger_source_breakdown,
)

# ---------------------------------------------------------------------------
# _compute_kpis unit tests
# ---------------------------------------------------------------------------


def test_compute_kpis_empty_records() -> None:
    """Empty records should return zero-value KPIs with no division errors."""
    kpis = _compute_kpis([])

    assert kpis["findings_count_by_severity"] == {"low": 0, "medium": 0, "high": 0}
    assert kpis["trigger_source_stats"] == {}
    assert kpis["triage_suppress_rate"] is None
    assert kpis["auto_exec_count"] == 0
    assert kpis["auto_exec_failures"] == 0
    assert kpis["action_success_rate"] is None
    assert kpis["user_override_rate"] is None
    assert kpis["false_positive_rate_14d"] is None


def test_compute_kpis_severity_counts() -> None:
    """Severity counts should bucket findings by the finding.severity field."""
    records = [
        {"finding": {"severity": "low"}},
        {"finding": {"severity": "high"}},
        {"finding": {"severity": "medium"}},
        {"finding": {"severity": "high"}},
        {"finding": {}},  # missing severity — should not crash
    ]
    kpis = _compute_kpis(records)

    assert kpis["findings_count_by_severity"] == {"low": 1, "medium": 1, "high": 2}


def test_compute_kpis_trigger_source_stats() -> None:
    """trigger_source_stats should aggregate counts by trigger source."""
    records = [
        {"finding": {}, "trigger_source": "poll"},
        {"finding": {}, "trigger_source": "event"},
        {"finding": {}, "trigger_source": "poll"},
        {"finding": {}},  # no trigger_source
    ]
    kpis = _compute_kpis(records)

    assert kpis["trigger_source_stats"] == {"poll": 2, "event": 1}


def test_compute_kpis_triage_suppress_rate_50pct() -> None:
    """triage_suppress_rate should be 50% when half of triaged records are suppressed."""
    records = [
        {"finding": {}, "triage_decision": "suppress"},
        {"finding": {}, "triage_decision": "notify"},
        {"finding": {}},  # no triage data — not counted in denominator
    ]
    kpis = _compute_kpis(records)

    assert kpis["triage_suppress_rate"] == 50.0


def test_compute_kpis_triage_suppress_rate_none_when_no_triage() -> None:
    """triage_suppress_rate is None when no records have triage data."""
    records = [{"finding": {}}, {"finding": {}}]
    kpis = _compute_kpis(records)

    assert kpis["triage_suppress_rate"] is None


def test_compute_kpis_auto_exec_count_and_failures() -> None:
    """auto_exec_count and auto_exec_failures are counted from action_outcome."""
    records = [
        {"finding": {}, "action_outcome": {"status": "success"}},
        {"finding": {}, "action_outcome": {"status": "partial"}},
        {"finding": {}, "action_outcome": {"status": "error"}},
        {"finding": {}, "action_outcome": {"status": "no_actions"}},
        {
            "finding": {},
            "action_outcome": {"status": "agent_called"},
        },  # user-triggered, not auto-exec
        {"finding": {}, "action_outcome": {"status": "dismissed"}},  # not counted
        {"finding": {}},  # no action_outcome
    ]
    kpis = _compute_kpis(records)

    assert kpis["auto_exec_count"] == 4
    assert kpis["auto_exec_failures"] == 1


def test_compute_kpis_action_success_rate_includes_user_triggered() -> None:
    """action_success_rate counts both auto-exec and user-triggered outcomes."""
    records = [
        {"finding": {}, "action_outcome": {"status": "success"}},  # auto, success
        {"finding": {}, "action_outcome": {"status": "agent_called"}},  # user, success
        {"finding": {}, "action_outcome": {"status": "event_fired"}},  # user, success
        {"finding": {}, "action_outcome": {"status": "error"}},  # auto, failure
        {"finding": {}, "action_outcome": {"status": "blocked"}},  # user, failure
    ]
    kpis = _compute_kpis(records)

    # 3 successes (success, agent_called, event_fired), 5 total → 60%
    assert kpis["action_success_rate"] == 60.0


def test_compute_kpis_action_success_rate_auto_exec_only() -> None:
    """action_success_rate handles pure auto-exec mix."""
    records = [
        {"finding": {}, "action_outcome": {"status": "success"}},
        {"finding": {}, "action_outcome": {"status": "partial"}},
        {"finding": {}, "action_outcome": {"status": "error"}},
        {"finding": {}, "action_outcome": {"status": "error"}},
    ]
    kpis = _compute_kpis(records)

    # 2 success/partial, 4 total → 50%
    assert kpis["action_success_rate"] == 50.0


def test_compute_kpis_user_override_rate() -> None:
    """user_override_rate counts only not_suppressed records with user_response."""
    records = [
        {
            "finding": {},
            "suppression_reason_code": "not_suppressed",
            "user_response": {"action": "dismiss"},
        },
        {
            "finding": {},
            "suppression_reason_code": "not_suppressed",
            "user_response": {"action": "execute"},
        },
        # suppressed — user never saw it, should not count toward total
        {"finding": {}, "suppression_reason_code": "suppressed", "user_response": None},
        # not_suppressed but no response
        {"finding": {}, "suppression_reason_code": "not_suppressed"},
    ]
    kpis = _compute_kpis(records)

    # 2 responses out of 3 not_suppressed → 66.7%
    assert kpis["user_override_rate"] == 66.7


def test_compute_kpis_false_positive_rate_14d() -> None:
    """false_positive_rate_14d: % of last-14d not_suppressed records with false_positive."""
    now = datetime.now(UTC)
    recent_str = (now - timedelta(days=1)).isoformat()
    old_str = (now - timedelta(days=20)).isoformat()

    records = [
        # Recent, not_suppressed, false positive
        {
            "finding": {},
            "suppression_reason_code": "not_suppressed",
            "notification": {"notified_at": recent_str},
            "user_response": {"false_positive": True},
        },
        # Recent, not_suppressed, not false positive
        {
            "finding": {},
            "suppression_reason_code": "not_suppressed",
            "notification": {"notified_at": recent_str},
            "user_response": None,
        },
        # Recent but suppressed — must not count even if it had false_positive
        {
            "finding": {},
            "suppression_reason_code": "suppressed",
            "notification": {"notified_at": recent_str},
            "user_response": {"false_positive": True},
        },
        # Old not_suppressed — outside 14d window
        {
            "finding": {},
            "suppression_reason_code": "not_suppressed",
            "notification": {"notified_at": old_str},
            "user_response": {"false_positive": True},
        },
    ]
    kpis = _compute_kpis(records)

    # 1 FP out of 2 not_suppressed within-window records → 50%
    assert kpis["false_positive_rate_14d"] == 50.0


def test_compute_kpis_false_positive_rate_14d_none_when_no_recent() -> None:
    """false_positive_rate_14d is None when no not_suppressed records are in the 14d window."""
    old_str = (datetime.now(UTC) - timedelta(days=30)).isoformat()
    records = [
        {
            "finding": {},
            "suppression_reason_code": "not_suppressed",
            "notification": {"notified_at": old_str},
            "user_response": {"false_positive": True},
        },
    ]
    kpis = _compute_kpis(records)

    assert kpis["false_positive_rate_14d"] is None


def test_compute_kpis_bad_notified_at_does_not_crash() -> None:
    """Malformed notified_at strings should be silently skipped."""
    records = [
        {
            "finding": {},
            "suppression_reason_code": "not_suppressed",
            "notification": {"notified_at": "not-a-date"},
        },
        {
            "finding": {},
            "suppression_reason_code": "not_suppressed",
            "notification": {"notified_at": None},
        },
        {
            "finding": {},
            "suppression_reason_code": "not_suppressed",
            "notification": {},
        },
        {"finding": {}, "suppression_reason_code": "not_suppressed"},
    ]
    kpis = _compute_kpis(records)

    assert kpis["false_positive_rate_14d"] is None


# ---------------------------------------------------------------------------
# SentinelHealthSensor integration-style tests
# ---------------------------------------------------------------------------


def _make_sensor(
    *,
    sentinel_enabled: bool = True,
    records: list[dict[str, Any]] | None = None,
    run_stats: dict[str, Any] | None = None,
) -> SentinelHealthSensor:
    """Build a SentinelHealthSensor with lightweight fakes."""
    hass = MagicMock()
    hass.async_create_task = MagicMock()

    options: dict[str, Any] = {"sentinel_enabled": sentinel_enabled}

    audit_store = MagicMock()
    audit_store.async_get_latest = AsyncMock(return_value=records or [])

    sentinel_engine = MagicMock()
    sentinel_engine.run_stats = run_stats or {}

    sensor = SentinelHealthSensor(
        hass=hass,
        options=options,
        audit_store=audit_store,
        sentinel=sentinel_engine,
        entry_id="test_entry",
    )
    sensor.hass = hass
    return sensor


@pytest.mark.asyncio
async def test_refresh_disabled_sentinel() -> None:
    """When sentinel is disabled the state should be 'disabled' and attrs empty."""
    sensor = _make_sensor(sentinel_enabled=False)
    sensor.async_write_ha_state = MagicMock()  # suppress HA machinery

    await sensor._async_refresh()

    assert sensor._attr_native_value == "disabled"
    assert sensor._attrs == {}
    sensor.async_write_ha_state.assert_called_once()


@pytest.mark.asyncio
async def test_refresh_enabled_empty_records() -> None:
    """With no audit records the sensor state should be 'ok' with zero KPIs."""
    sensor = _make_sensor(sentinel_enabled=True, records=[])
    sensor.async_write_ha_state = MagicMock()

    await sensor._async_refresh()

    assert sensor._attr_native_value == "ok"
    assert sensor._attrs["auto_exec_count"] == 0
    assert sensor._attrs["findings_count_by_severity"] == {
        "low": 0,
        "medium": 0,
        "high": 0,
    }
    sensor.async_write_ha_state.assert_called_once()


@pytest.mark.asyncio
async def test_refresh_merges_run_stats() -> None:
    """run_stats from the engine should appear as sensor attributes."""
    run_stats = {
        "last_run_start": "2025-01-01T12:00:00+00:00",
        "last_run_end": "2025-01-01T12:00:01+00:00",
        "run_duration_ms": 1000,
        "active_rule_count": 12,
    }
    sensor = _make_sensor(run_stats=run_stats)
    sensor.async_write_ha_state = MagicMock()

    await sensor._async_refresh()

    assert sensor._attrs["last_run_start"] == "2025-01-01T12:00:00+00:00"
    assert sensor._attrs["run_duration_ms"] == 1000
    assert sensor._attrs["active_rule_count"] == 12


@pytest.mark.asyncio
async def test_refresh_run_stats_keys_present_when_no_engine() -> None:
    """All run_stats keys should be present (as None) when sentinel is None."""
    hass = MagicMock()
    sensor = SentinelHealthSensor(
        hass=hass,
        options={"sentinel_enabled": True},
        audit_store=None,
        sentinel=None,
        entry_id="test_entry",
    )
    sensor.async_write_ha_state = MagicMock()

    await sensor._async_refresh()

    for key in (
        "last_run_start",
        "last_run_end",
        "run_duration_ms",
        "active_rule_count",
    ):
        assert key in sensor._attrs
        assert sensor._attrs[key] is None


def test_unique_id_uses_entry_id() -> None:
    """unique_id should incorporate the entry_id for per-entry uniqueness."""
    sensor = _make_sensor()
    assert sensor._attr_unique_id == "sentinel_health::test_entry"


@pytest.mark.asyncio
async def test_refresh_includes_scheduler_stats_from_run_stats() -> None:
    """Scheduler stats nested in run_stats['scheduler'] appear as sensor attributes."""
    run_stats = {
        "last_run_start": "2025-01-01T12:00:00+00:00",
        "scheduler": {
            "triggers_coalesced": 5,
            "triggers_dropped_incoming": 1,
            "triggers_dropped_queued": 2,
            "triggers_ttl_expired": 0,
        },
    }
    sensor = _make_sensor(run_stats=run_stats)
    sensor.async_write_ha_state = MagicMock()

    await sensor._async_refresh()

    assert sensor._attrs["triggers_coalesced"] == 5
    assert sensor._attrs["triggers_dropped_incoming"] == 1
    assert sensor._attrs["triggers_dropped_queued"] == 2
    assert sensor._attrs["triggers_ttl_expired"] == 0


@pytest.mark.asyncio
async def test_refresh_scheduler_stats_absent_when_no_engine() -> None:
    """No scheduler stats in attrs when sentinel engine is None (no scheduler key)."""
    hass = MagicMock()
    sensor = SentinelHealthSensor(
        hass=hass,
        options={"sentinel_enabled": True},
        audit_store=None,
        sentinel=None,
        entry_id="test_entry",
    )
    sensor.async_write_ha_state = MagicMock()

    await sensor._async_refresh()

    # Scheduler stat keys should not appear when there is no engine.
    for key in ("triggers_coalesced", "triggers_dropped_incoming"):
        assert key not in sensor._attrs


@pytest.mark.asyncio
async def test_refresh_requests_all_available_records() -> None:
    """_async_refresh requests 1000 records so all available data contributes to KPIs."""
    sensor = _make_sensor(records=[])
    sensor.async_write_ha_state = MagicMock()

    await sensor._async_refresh()

    cast("MagicMock", sensor._audit_store).async_get_latest.assert_called_once_with(
        1000
    )


# ---------------------------------------------------------------------------
# _compute_trigger_source_breakdown unit tests
# ---------------------------------------------------------------------------


def test_trigger_source_breakdown_empty_records() -> None:
    """Empty records should return all-zero breakdown with the three standard keys."""
    result = _compute_trigger_source_breakdown([])
    assert result == {"poll": 0, "event": 0, "on_demand": 0}


def test_trigger_source_breakdown_counts_within_24h() -> None:
    """Only records notified within the last 24 hours are counted."""
    now = datetime.now(UTC)
    recent_str = (now - timedelta(hours=1)).isoformat()
    old_str = (now - timedelta(hours=25)).isoformat()
    records = [
        {"trigger_source": "poll", "notification": {"notified_at": recent_str}},
        {"trigger_source": "event", "notification": {"notified_at": recent_str}},
        {"trigger_source": "poll", "notification": {"notified_at": recent_str}},
        # older than 24h — must be excluded
        {"trigger_source": "on_demand", "notification": {"notified_at": old_str}},
    ]
    result = _compute_trigger_source_breakdown(records)
    assert result["poll"] == 2
    assert result["event"] == 1
    assert result["on_demand"] == 0


def test_trigger_source_breakdown_unknown_key_included() -> None:
    """A trigger_source value not in the known keys is still counted (open-ended)."""
    now = datetime.now(UTC)
    recent_str = (now - timedelta(minutes=5)).isoformat()
    records = [
        {"trigger_source": "poll", "notification": {"notified_at": recent_str}},
        {
            "trigger_source": "unknown_future_type",
            "notification": {"notified_at": recent_str},
        },
    ]
    result = _compute_trigger_source_breakdown(records)
    assert result["poll"] == 1
    assert result["unknown_future_type"] == 1


def test_trigger_source_breakdown_missing_notification_skipped() -> None:
    """Records with no notification.notified_at are silently skipped."""
    now = datetime.now(UTC)
    recent_str = (now - timedelta(minutes=5)).isoformat()
    records = [
        {"trigger_source": "poll", "notification": {"notified_at": recent_str}},
        {"trigger_source": "event"},  # no notification key
        {"trigger_source": "event", "notification": {}},  # notified_at absent
        {
            "trigger_source": "event",
            "notification": {"notified_at": None},
        },  # None value
        {
            "trigger_source": "event",
            "notification": {"notified_at": "not-a-date"},
        },  # malformed
    ]
    result = _compute_trigger_source_breakdown(records)
    assert result["poll"] == 1
    assert result["event"] == 0


def test_trigger_source_breakdown_no_trigger_source_skipped() -> None:
    """Records within 24h but with no trigger_source do not affect counts."""
    now = datetime.now(UTC)
    recent_str = (now - timedelta(minutes=5)).isoformat()
    records = [
        {"notification": {"notified_at": recent_str}},  # trigger_source absent
    ]
    result = _compute_trigger_source_breakdown(records)
    assert result == {"poll": 0, "event": 0, "on_demand": 0}


# ---------------------------------------------------------------------------
# Sensor integration: trigger_source_breakdown attribute
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_refresh_exposes_trigger_source_breakdown_with_audit_store() -> None:
    """trigger_source_breakdown attribute is populated from audit records when store present."""
    now = datetime.now(UTC)
    recent_str = (now - timedelta(hours=2)).isoformat()
    records = [
        {"trigger_source": "poll", "notification": {"notified_at": recent_str}},
        {"trigger_source": "event", "notification": {"notified_at": recent_str}},
        {"trigger_source": "poll", "notification": {"notified_at": recent_str}},
    ]
    sensor = _make_sensor(records=records)
    sensor.async_write_ha_state = MagicMock()

    await sensor._async_refresh()

    breakdown = sensor._attrs["trigger_source_breakdown"]
    assert isinstance(breakdown, dict)
    assert breakdown["poll"] == 2
    assert breakdown["event"] == 1
    assert breakdown["on_demand"] == 0


@pytest.mark.asyncio
async def test_refresh_trigger_source_breakdown_none_when_no_audit_store() -> None:
    """trigger_source_breakdown is None when audit_store is absent."""
    hass = MagicMock()
    sensor = SentinelHealthSensor(
        hass=hass,
        options={"sentinel_enabled": True},
        audit_store=None,
        sentinel=None,
        entry_id="test_entry",
    )
    sensor.async_write_ha_state = MagicMock()

    await sensor._async_refresh()

    assert sensor._attrs["trigger_source_breakdown"] is None

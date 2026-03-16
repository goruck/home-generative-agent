# ruff: noqa: S101
"""Tests for the Sentinel operational health sensor."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from custom_components.home_generative_agent.core.sentinel_health_sensor import (
    SentinelHealthSensor,
    _compute_kpis,
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
        {"finding": {}, "action_outcome": {"status": "dismissed"}},  # not auto-exec
        {"finding": {}},  # no action_outcome
    ]
    kpis = _compute_kpis(records)

    assert kpis["auto_exec_count"] == 4
    assert kpis["auto_exec_failures"] == 1


def test_compute_kpis_action_success_rate_75pct() -> None:
    """action_success_rate is (success+partial) / auto_exec_total * 100."""
    records = [
        {"finding": {}, "action_outcome": {"status": "success"}},
        {"finding": {}, "action_outcome": {"status": "partial"}},
        {"finding": {}, "action_outcome": {"status": "error"}},
        {"finding": {}, "action_outcome": {"status": "error"}},
    ]
    kpis = _compute_kpis(records)

    # 2 success/partial, 4 total auto-exec → 50%
    assert kpis["action_success_rate"] == 50.0


def test_compute_kpis_user_override_rate() -> None:
    """user_override_rate is (records with user_response / total) * 100."""
    records = [
        {"finding": {}, "user_response": {"action": "dismiss"}},
        {"finding": {}, "user_response": {"action": "execute"}},
        {"finding": {}},
        {"finding": {}},
    ]
    kpis = _compute_kpis(records)

    assert kpis["user_override_rate"] == 50.0


def test_compute_kpis_false_positive_rate_14d() -> None:
    """false_positive_rate_14d: % of last-14d notified records with false_positive=True."""
    now = datetime.now(UTC)
    recent_str = (now - timedelta(days=1)).isoformat()
    old_str = (now - timedelta(days=20)).isoformat()

    records = [
        # Recent, false positive
        {
            "finding": {},
            "notification": {"notified_at": recent_str},
            "user_response": {"false_positive": True},
        },
        # Recent, not false positive
        {
            "finding": {},
            "notification": {"notified_at": recent_str},
            "user_response": None,
        },
        # Old — outside 14d window, should not count
        {
            "finding": {},
            "notification": {"notified_at": old_str},
            "user_response": {"false_positive": True},
        },
    ]
    kpis = _compute_kpis(records)

    # 1 FP out of 2 within-window records → 50%
    assert kpis["false_positive_rate_14d"] == 50.0


def test_compute_kpis_false_positive_rate_14d_none_when_no_recent() -> None:
    """false_positive_rate_14d is None when no records fall within the 14d window."""
    old_str = (datetime.now(UTC) - timedelta(days=30)).isoformat()
    records = [
        {
            "finding": {},
            "notification": {"notified_at": old_str},
            "user_response": {"false_positive": True},
        },
    ]
    kpis = _compute_kpis(records)

    assert kpis["false_positive_rate_14d"] is None


def test_compute_kpis_bad_notified_at_does_not_crash() -> None:
    """Malformed notified_at strings should be silently skipped."""
    records = [
        {"finding": {}, "notification": {"notified_at": "not-a-date"}},
        {"finding": {}, "notification": {"notified_at": None}},
        {"finding": {}, "notification": {}},
        {"finding": {}},
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

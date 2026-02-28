# ruff: noqa: S101
"""Tests for SentinelBaselineUpdater — sentinel/baseline.py (Issue #265)."""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, Any, cast
from unittest.mock import AsyncMock, MagicMock

import pytest

from custom_components.home_generative_agent.sentinel.baseline import (
    BASELINE_FRESH,
    BASELINE_STALE,
    BASELINE_UNAVAILABLE,
    METRIC_HOURLY_PREFIX,
    METRIC_ROLLING_AVG,
    SentinelBaselineUpdater,
    evaluate_baseline_deviation,
    evaluate_time_of_day_anomaly,
)
from custom_components.home_generative_agent.sentinel.models import AnomalyFinding

if TYPE_CHECKING:
    from custom_components.home_generative_agent.snapshot.schema import (
        FullStateSnapshot,
    )

# ---------------------------------------------------------------------------
# Snapshot helpers
# ---------------------------------------------------------------------------


def _entity(
    entity_id: str,
    state: str,
    domain: str | None = None,
    area: str = "Living Room",
) -> dict[str, Any]:
    d = domain or entity_id.split(".", maxsplit=1)[0]
    return {
        "entity_id": entity_id,
        "state": state,
        "domain": d,
        "area": area,
        "attributes": {},
        "last_changed": "2025-01-01T00:00:00+00:00",
        "last_updated": "2025-01-01T00:00:00+00:00",
    }


def _snapshot(
    entities: list[dict[str, Any]] | None = None,
    is_night: bool = False,  # noqa: FBT001, FBT002
    anyone_home: bool = True,  # noqa: FBT001, FBT002
    now: str = "2025-01-01T10:00:00+00:00",
) -> FullStateSnapshot:
    return cast(
        "FullStateSnapshot",
        {
            "schema_version": 1,
            "generated_at": "2025-01-01T00:00:00+00:00",
            "entities": entities or [],
            "camera_activity": [],
            "derived": {
                "now": now,
                "timezone": "UTC",
                "is_night": is_night,
                "anyone_home": anyone_home,
                "people_home": [],
                "people_away": [],
                "last_motion_by_area": {},
            },
        },
    )


def _rule(  # noqa: PLR0913
    rule_id: str = "baseline_deviation",
    template_id: str = "baseline_deviation",
    entity_id: str = "sensor.temperature",
    threshold_pct: float = 50.0,
    metric: str = METRIC_ROLLING_AVG,
    severity: str = "low",
    confidence: float = 0.7,
) -> dict[str, Any]:
    return {
        "rule_id": rule_id,
        "template_id": template_id,
        "params": {
            "entity_id": entity_id,
            "threshold_pct": threshold_pct,
            "metric": metric,
        },
        "severity": severity,
        "confidence": confidence,
        "is_sensitive": False,
        "suggested_actions": [],
    }


# ---------------------------------------------------------------------------
# 1. evaluate_baseline_deviation — basic logic
# ---------------------------------------------------------------------------


def test_baseline_deviation_fires_when_exceeds_threshold() -> None:
    """A finding is returned when deviation exceeds threshold_pct."""
    snapshot = _snapshot([_entity("sensor.temperature", "30")])
    baselines = {"sensor.temperature": {METRIC_ROLLING_AVG: 20.0}}
    rule = _rule(threshold_pct=40.0)  # 50% deviation > 40% threshold

    findings = evaluate_baseline_deviation(snapshot, rule, baselines)

    assert len(findings) == 1
    assert isinstance(findings[0], AnomalyFinding)
    assert findings[0].type == "baseline_deviation"
    assert "deviation_pct" in findings[0].evidence
    assert findings[0].evidence["deviation_pct"] == pytest.approx(50.0)


def test_baseline_deviation_no_finding_below_threshold() -> None:
    """No finding is returned when deviation is within threshold."""
    snapshot = _snapshot([_entity("sensor.temperature", "22")])
    baselines = {"sensor.temperature": {METRIC_ROLLING_AVG: 20.0}}
    rule = _rule(threshold_pct=50.0)  # 10% deviation < 50% threshold

    findings = evaluate_baseline_deviation(snapshot, rule, baselines)

    assert findings == []


def test_baseline_deviation_no_finding_when_entity_missing_from_snapshot() -> None:
    """No finding when the entity_id is absent from the snapshot."""
    snapshot = _snapshot([])  # no entities
    baselines = {"sensor.temperature": {METRIC_ROLLING_AVG: 20.0}}
    rule = _rule()

    findings = evaluate_baseline_deviation(snapshot, rule, baselines)

    assert findings == []


def test_baseline_deviation_no_finding_when_baseline_unavailable() -> None:
    """No finding when no baseline exists for the entity."""
    snapshot = _snapshot([_entity("sensor.temperature", "30")])
    baselines: dict[str, dict[str, float]] = {}
    rule = _rule()

    findings = evaluate_baseline_deviation(snapshot, rule, baselines)

    assert findings == []


def test_baseline_deviation_no_finding_when_state_non_numeric() -> None:
    """No finding when entity state is non-numeric."""
    snapshot = _snapshot([_entity("binary_sensor.door", "on")])
    baselines = {"binary_sensor.door": {METRIC_ROLLING_AVG: 0.0}}
    rule = _rule(entity_id="binary_sensor.door")

    findings = evaluate_baseline_deviation(snapshot, rule, baselines)

    assert findings == []


def test_baseline_deviation_zero_baseline_non_zero_current_fires() -> None:
    """When baseline is 0.0 and current is non-zero, deviation is treated as 100%."""
    snapshot = _snapshot([_entity("sensor.power", "5")])
    baselines = {"sensor.power": {METRIC_ROLLING_AVG: 0.0}}
    rule = _rule(entity_id="sensor.power", threshold_pct=50.0)

    findings = evaluate_baseline_deviation(snapshot, rule, baselines)

    assert len(findings) == 1
    assert findings[0].evidence["deviation_pct"] == pytest.approx(100.0)


def test_baseline_deviation_zero_baseline_zero_current_no_finding() -> None:
    """When both baseline and current are 0.0, no deviation fires."""
    snapshot = _snapshot([_entity("sensor.power", "0")])
    baselines = {"sensor.power": {METRIC_ROLLING_AVG: 0.0}}
    rule = _rule(entity_id="sensor.power", threshold_pct=50.0)

    findings = evaluate_baseline_deviation(snapshot, rule, baselines)

    assert findings == []


def test_baseline_deviation_missing_entity_id_in_params() -> None:
    """No finding when entity_id is absent from rule params."""
    snapshot = _snapshot([_entity("sensor.temperature", "30")])
    baselines = {"sensor.temperature": {METRIC_ROLLING_AVG: 20.0}}
    rule = {"rule_id": "r1", "params": {}, "severity": "low", "confidence": 0.7}

    findings = evaluate_baseline_deviation(snapshot, rule, baselines)

    assert findings == []


def test_baseline_deviation_finding_has_correct_evidence_fields() -> None:
    """AnomalyFinding evidence includes all required baseline fields."""
    snapshot = _snapshot([_entity("sensor.temperature", "40")])
    baselines = {"sensor.temperature": {METRIC_ROLLING_AVG: 20.0}}
    rule = _rule(threshold_pct=50.0)

    findings = evaluate_baseline_deviation(snapshot, rule, baselines)

    assert len(findings) == 1
    ev = findings[0].evidence
    assert ev["entity_id"] == "sensor.temperature"
    assert ev["current_value"] == pytest.approx(40.0)
    assert ev["baseline_value"] == pytest.approx(20.0)
    assert ev["metric"] == METRIC_ROLLING_AVG
    assert ev["threshold_pct"] == pytest.approx(50.0)
    assert "deviation_pct" in ev


# ---------------------------------------------------------------------------
# 2. evaluate_time_of_day_anomaly
# ---------------------------------------------------------------------------


def test_time_of_day_anomaly_uses_hourly_metric() -> None:
    """evaluate_time_of_day_anomaly delegates to evaluate_baseline_deviation with hourly metric."""
    # "now" is 10:xx — expect hourly_avg_10 metric to be used
    snapshot = _snapshot(
        [_entity("sensor.power", "50")],
        now="2025-01-01T10:00:00+00:00",
    )
    baselines = {
        "sensor.power": {
            f"{METRIC_HOURLY_PREFIX}10": 20.0,  # 150% deviation
        }
    }
    rule = _rule(entity_id="sensor.power", threshold_pct=100.0)

    findings = evaluate_time_of_day_anomaly(snapshot, rule, baselines)

    assert len(findings) == 1
    assert findings[0].evidence["metric"] == f"{METRIC_HOURLY_PREFIX}10"


def test_time_of_day_anomaly_no_finding_below_threshold() -> None:
    """No finding when hourly deviation is within threshold."""
    snapshot = _snapshot(
        [_entity("sensor.power", "21")],
        now="2025-01-01T14:00:00+00:00",
    )
    baselines = {"sensor.power": {f"{METRIC_HOURLY_PREFIX}14": 20.0}}
    rule = _rule(entity_id="sensor.power", threshold_pct=50.0)

    findings = evaluate_time_of_day_anomaly(snapshot, rule, baselines)

    assert findings == []


def test_time_of_day_anomaly_no_finding_when_hourly_baseline_missing() -> None:
    """No finding when no hourly baseline exists for the current hour."""
    snapshot = _snapshot(
        [_entity("sensor.power", "50")],
        now="2025-01-01T10:00:00+00:00",
    )
    baselines = {"sensor.power": {METRIC_ROLLING_AVG: 20.0}}  # rolling only, no hourly
    rule = _rule(entity_id="sensor.power", threshold_pct=50.0)

    findings = evaluate_time_of_day_anomaly(snapshot, rule, baselines)

    assert findings == []


def test_time_of_day_anomaly_invalid_now_returns_empty() -> None:
    """Invalid 'now' string in derived context returns empty list."""
    snapshot = _snapshot([_entity("sensor.power", "50")], now="not-a-date")
    baselines = {"sensor.power": {f"{METRIC_HOURLY_PREFIX}10": 20.0}}
    rule = _rule(entity_id="sensor.power")

    findings = evaluate_time_of_day_anomaly(snapshot, rule, baselines)

    assert findings == []


# ---------------------------------------------------------------------------
# 3. SentinelBaselineUpdater.check_freshness
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_check_freshness_returns_unavailable_when_no_row() -> None:
    """check_freshness returns UNAVAILABLE when no baseline row exists."""
    mock_cursor = MagicMock()
    mock_cursor.execute = AsyncMock()
    mock_cursor.fetchone = AsyncMock(return_value=None)
    mock_cursor.__aenter__ = AsyncMock(return_value=mock_cursor)
    mock_cursor.__aexit__ = AsyncMock(return_value=False)

    mock_conn = MagicMock()
    mock_conn.cursor = MagicMock(return_value=mock_cursor)
    mock_conn.__aenter__ = AsyncMock(return_value=mock_conn)
    mock_conn.__aexit__ = AsyncMock(return_value=False)

    mock_pool = MagicMock()
    mock_pool.connection = MagicMock(return_value=mock_conn)

    mock_hass = MagicMock()
    updater = SentinelBaselineUpdater(mock_hass, mock_pool, {})

    result = await updater.check_freshness("sensor.temp", METRIC_ROLLING_AVG, "rolling")

    assert result == BASELINE_UNAVAILABLE


@pytest.mark.asyncio
async def test_check_freshness_returns_fresh_for_recent_row() -> None:
    """check_freshness returns FRESH when updated_at is within the threshold."""
    recent_time = datetime.now(tz=UTC) - timedelta(seconds=60)
    row = {"updated_at": recent_time}

    mock_cursor = MagicMock()
    mock_cursor.execute = AsyncMock()
    mock_cursor.fetchone = AsyncMock(return_value=row)
    mock_cursor.__aenter__ = AsyncMock(return_value=mock_cursor)
    mock_cursor.__aexit__ = AsyncMock(return_value=False)

    mock_conn = MagicMock()
    mock_conn.cursor = MagicMock(return_value=mock_cursor)
    mock_conn.__aenter__ = AsyncMock(return_value=mock_conn)
    mock_conn.__aexit__ = AsyncMock(return_value=False)

    mock_pool = MagicMock()
    mock_pool.connection = MagicMock(return_value=mock_conn)

    mock_hass = MagicMock()
    # freshness threshold default is 3600 seconds; 60 seconds ago is fresh
    updater = SentinelBaselineUpdater(mock_hass, mock_pool, {})

    result = await updater.check_freshness("sensor.temp", METRIC_ROLLING_AVG, "rolling")

    assert result == BASELINE_FRESH


@pytest.mark.asyncio
async def test_check_freshness_returns_stale_for_old_row() -> None:
    """check_freshness returns STALE when updated_at is older than the threshold."""
    old_time = datetime.now(tz=UTC) - timedelta(hours=2)
    row = {"updated_at": old_time}

    mock_cursor = MagicMock()
    mock_cursor.execute = AsyncMock()
    mock_cursor.fetchone = AsyncMock(return_value=row)
    mock_cursor.__aenter__ = AsyncMock(return_value=mock_cursor)
    mock_cursor.__aexit__ = AsyncMock(return_value=False)

    mock_conn = MagicMock()
    mock_conn.cursor = MagicMock(return_value=mock_cursor)  # cursor, not conn
    mock_conn.__aenter__ = AsyncMock(return_value=mock_conn)
    mock_conn.__aexit__ = AsyncMock(return_value=False)

    mock_pool = MagicMock()
    mock_pool.connection = MagicMock(return_value=mock_conn)

    mock_hass = MagicMock()
    # freshness threshold default 3600 s; 7200 s is stale
    updater = SentinelBaselineUpdater(mock_hass, mock_pool, {})

    result = await updater.check_freshness("sensor.temp", METRIC_ROLLING_AVG, "rolling")

    assert result == BASELINE_STALE


@pytest.mark.asyncio
async def test_check_freshness_returns_unavailable_on_db_error() -> None:
    """check_freshness returns UNAVAILABLE when the DB query raises an exception."""
    mock_pool = MagicMock()
    mock_pool.connection = MagicMock(side_effect=OSError("DB down"))

    mock_hass = MagicMock()
    updater = SentinelBaselineUpdater(mock_hass, mock_pool, {})

    result = await updater.check_freshness("sensor.temp", METRIC_ROLLING_AVG, "rolling")

    assert result == BASELINE_UNAVAILABLE


# ---------------------------------------------------------------------------
# 4. SentinelBaselineUpdater lifecycle: start / stop idempotency
# ---------------------------------------------------------------------------


def test_start_creates_background_task_and_is_idempotent() -> None:
    """start() creates a background task; calling it twice is idempotent."""
    task_calls: list[Any] = []

    mock_hass = MagicMock()

    def _fake_create_task(coro: Any) -> Any:
        task_calls.append(coro)
        return MagicMock()

    mock_hass.async_create_task = _fake_create_task

    mock_pool = MagicMock()
    updater = SentinelBaselineUpdater(mock_hass, mock_pool, {})

    updater.start()
    updater.start()  # idempotent

    assert len(task_calls) == 1

    # Close the unawaited coroutine to prevent RuntimeWarning leaking into later tests.
    for coro in task_calls:
        coro.close()


@pytest.mark.asyncio
async def test_stop_cancels_task_and_sets_none() -> None:
    """stop() cancels the running task and sets _task to None."""
    mock_hass = MagicMock()
    mock_pool = MagicMock()
    updater = SentinelBaselineUpdater(mock_hass, mock_pool, {})

    # Use a real Task so `await self._task` works correctly in stop()
    async def _never_ending() -> None:
        await asyncio.sleep(9999)

    task = asyncio.get_event_loop().create_task(_never_ending())
    updater._task = task  # type: ignore[assignment]

    await updater.stop()

    assert task.cancelled()
    assert updater._task is None


@pytest.mark.asyncio
async def test_stop_when_not_started_is_noop() -> None:
    """stop() when no task is running must not raise."""
    mock_hass = MagicMock()
    mock_pool = MagicMock()
    updater = SentinelBaselineUpdater(mock_hass, mock_pool, {})

    await updater.stop()  # must not raise

    assert updater._task is None


# ---------------------------------------------------------------------------
# 5. _update_baselines writes correct rows for numeric entities
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_update_baselines_upserts_two_rows_per_numeric_entity() -> None:
    """_update_baselines must upsert rolling_avg AND hourly_avg_<H> for each numeric entity."""
    executed_params: list[tuple[Any, ...]] = []

    mock_cursor = MagicMock()

    async def _fake_execute(_sql: str, params: tuple[Any, ...]) -> None:
        executed_params.append(params)

    mock_cursor.execute = _fake_execute
    mock_cursor.__aenter__ = AsyncMock(return_value=mock_cursor)
    mock_cursor.__aexit__ = AsyncMock(return_value=False)

    mock_conn = MagicMock()
    mock_conn.cursor = MagicMock(return_value=mock_cursor)
    mock_conn.commit = AsyncMock()
    mock_conn.__aenter__ = AsyncMock(return_value=mock_conn)
    mock_conn.__aexit__ = AsyncMock(return_value=False)

    mock_pool = MagicMock()
    mock_pool.connection = MagicMock(return_value=mock_conn)

    mock_hass = MagicMock()
    updater = SentinelBaselineUpdater(mock_hass, mock_pool, {})

    snapshot = _snapshot(
        [
            _entity("sensor.temperature", "22.5"),
            _entity("binary_sensor.door", "on"),  # non-numeric, must be skipped
        ]
    )

    await updater._update_baselines(snapshot)

    # Only sensor.temperature is numeric; 2 rows (rolling + hourly) must be written
    assert len(executed_params) == 2
    metrics = {p[1] for p in executed_params}
    assert METRIC_ROLLING_AVG in metrics
    assert any(m.startswith(METRIC_HOURLY_PREFIX) for m in metrics)

    # entity_id must be correct in both rows
    for params in executed_params:
        assert params[0] == "sensor.temperature"
        assert params[3] == pytest.approx(22.5)


@pytest.mark.asyncio
async def test_update_baselines_skips_non_numeric_entities() -> None:
    """_update_baselines must not write rows for non-numeric entity states."""
    executed_params: list[tuple[Any, ...]] = []

    mock_cursor = MagicMock()

    async def _fake_execute(_sql: str, params: tuple[Any, ...]) -> None:
        executed_params.append(params)

    mock_cursor.execute = _fake_execute
    mock_cursor.__aenter__ = AsyncMock(return_value=mock_cursor)
    mock_cursor.__aexit__ = AsyncMock(return_value=False)

    mock_conn = MagicMock()
    mock_conn.cursor = MagicMock(return_value=mock_cursor)
    mock_conn.commit = AsyncMock()
    mock_conn.__aenter__ = AsyncMock(return_value=mock_conn)
    mock_conn.__aexit__ = AsyncMock(return_value=False)

    mock_pool = MagicMock()
    mock_pool.connection = MagicMock(return_value=mock_conn)

    mock_hass = MagicMock()
    updater = SentinelBaselineUpdater(mock_hass, mock_pool, {})

    snapshot = _snapshot(
        [
            _entity("binary_sensor.door", "on"),
            _entity("binary_sensor.motion", "off"),
            _entity("select.mode", "auto"),
        ]
    )

    await updater._update_baselines(snapshot)

    assert executed_params == []


@pytest.mark.asyncio
async def test_update_baselines_empty_snapshot_is_noop() -> None:
    """_update_baselines with an empty snapshot must not touch the DB."""
    mock_pool = MagicMock()
    mock_pool.connection = MagicMock(side_effect=AssertionError("should not be called"))

    mock_hass = MagicMock()
    updater = SentinelBaselineUpdater(mock_hass, mock_pool, {})

    await updater._update_baselines(_snapshot([]))
    # No AssertionError raised means the pool was not accessed.

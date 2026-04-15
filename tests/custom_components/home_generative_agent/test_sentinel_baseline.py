# ruff: noqa: S101
"""Tests for SentinelBaselineUpdater — sentinel/baseline.py (Issue #265)."""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, Any, cast
from unittest.mock import AsyncMock, MagicMock

import pytest

from custom_components.home_generative_agent.const import (
    CONF_SENTINEL_BASELINE_DRIFT_THRESHOLD_PCT,
    CONF_SENTINEL_BASELINE_MAX_SAMPLES,
    CONF_SENTINEL_BASELINE_MIN_SAMPLES,
    CONF_SENTINEL_BASELINE_WEEKLY_PATTERNS,
    RECOMMENDED_SENTINEL_BASELINE_MIN_SAMPLES,
)
from custom_components.home_generative_agent.sentinel.baseline import (
    _DOW_METRIC_PARTS,
    BASELINE_FRESH,
    BASELINE_STALE,
    BASELINE_UNAVAILABLE,
    METRIC_DOW_AVG_PREFIX,
    METRIC_DOW_STD_PREFIX,
    METRIC_HOURLY_PREFIX,
    METRIC_ROLLING_AVG,
    SentinelBaselineUpdater,
    evaluate_baseline_deviation,
    evaluate_time_of_day_anomaly,
)
from custom_components.home_generative_agent.sentinel.engine import SentinelEngine
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
    assert findings[0].evidence["deviation_direction"] == "above"


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
    assert ev["deviation_direction"] == "above"


def test_baseline_deviation_direction_below() -> None:
    """deviation_direction is 'below' when current is less than baseline."""
    snapshot = _snapshot([_entity("sensor.power", "0.5")])
    baselines = {"sensor.power": {METRIC_ROLLING_AVG: 40.7}}
    rule = _rule(entity_id="sensor.power", threshold_pct=50.0)

    findings = evaluate_baseline_deviation(snapshot, rule, baselines)

    assert len(findings) == 1
    assert findings[0].evidence["deviation_direction"] == "below"


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
    upsert_params: list[tuple[Any, ...]] = []

    mock_cursor = MagicMock()

    async def _fake_execute(sql: str, params: tuple[Any, ...]) -> None:
        if "INSERT INTO" in sql:
            upsert_params.append(params)

    mock_cursor.execute = _fake_execute
    mock_cursor.fetchall = AsyncMock(return_value=[])  # _FETCH_REFS_SQL returns no refs
    mock_cursor.fetchone = AsyncMock(return_value=None)  # sample_count check
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

    # Only sensor.temperature is numeric; 2 INSERT rows (rolling + hourly) must be written
    assert len(upsert_params) == 2
    metrics = {p[1] for p in upsert_params}
    assert METRIC_ROLLING_AVG in metrics
    assert any(m.startswith(METRIC_HOURLY_PREFIX) for m in metrics)

    # entity_id must be correct in both rows
    for params in upsert_params:
        assert params[0] == "sensor.temperature"
        assert params[3] == pytest.approx(22.5)


@pytest.mark.asyncio
async def test_update_baselines_skips_non_numeric_entities() -> None:
    """_update_baselines must not write rows for non-numeric entity states."""
    mock_pool = MagicMock()
    mock_pool.connection = MagicMock(side_effect=AssertionError("should not be called"))

    mock_hass = MagicMock()
    updater = SentinelBaselineUpdater(mock_hass, mock_pool, {})

    snapshot = _snapshot(
        [
            _entity("binary_sensor.door", "on"),
            _entity("binary_sensor.motion", "off"),
            _entity("select.mode", "auto"),
        ]
    )

    # entity_values will be empty → returns early before touching the DB
    await updater._update_baselines(snapshot)


@pytest.mark.asyncio
async def test_update_baselines_empty_snapshot_is_noop() -> None:
    """_update_baselines with an empty snapshot must not touch the DB."""
    mock_pool = MagicMock()
    mock_pool.connection = MagicMock(side_effect=AssertionError("should not be called"))

    mock_hass = MagicMock()
    updater = SentinelBaselineUpdater(mock_hass, mock_pool, {})

    await updater._update_baselines(_snapshot([]))
    # No AssertionError raised means the pool was not accessed.


# ---------------------------------------------------------------------------
# 6. async_fetch_baselines
# ---------------------------------------------------------------------------


def _make_pool_with_rows(rows: list[Any]) -> MagicMock:
    """Return a mock pool whose fetchall() returns *rows*."""
    mock_cursor = MagicMock()
    mock_cursor.execute = AsyncMock()
    mock_cursor.fetchall = AsyncMock(return_value=rows)
    mock_cursor.__aenter__ = AsyncMock(return_value=mock_cursor)
    mock_cursor.__aexit__ = AsyncMock(return_value=False)

    mock_conn = MagicMock()
    mock_conn.cursor = MagicMock(return_value=mock_cursor)
    mock_conn.__aenter__ = AsyncMock(return_value=mock_conn)
    mock_conn.__aexit__ = AsyncMock(return_value=False)

    mock_pool = MagicMock()
    mock_pool.connection = MagicMock(return_value=mock_conn)
    return mock_pool


@pytest.mark.asyncio
async def test_fetch_baselines_returns_nested_dict_from_dict_rows() -> None:
    """async_fetch_baselines builds {entity_id: {metric: value}} from dict rows."""
    rows = [
        {
            "entity_id": "sensor.temperature",
            "metric": METRIC_ROLLING_AVG,
            "value": 21.5,
        },
        {
            "entity_id": "sensor.temperature",
            "metric": f"{METRIC_HOURLY_PREFIX}10",
            "value": 20.0,
        },
        {"entity_id": "sensor.power", "metric": METRIC_ROLLING_AVG, "value": 150.0},
    ]
    updater = SentinelBaselineUpdater(MagicMock(), _make_pool_with_rows(rows), {})

    result = await updater.async_fetch_baselines()

    assert result["sensor.temperature"][METRIC_ROLLING_AVG] == pytest.approx(21.5)
    assert result["sensor.temperature"][f"{METRIC_HOURLY_PREFIX}10"] == pytest.approx(
        20.0
    )
    assert result["sensor.power"][METRIC_ROLLING_AVG] == pytest.approx(150.0)


@pytest.mark.asyncio
async def test_fetch_baselines_returns_nested_dict_from_tuple_rows() -> None:
    """async_fetch_baselines handles tuple rows (non-dict cursor)."""
    rows = [
        ("sensor.temperature", METRIC_ROLLING_AVG, 21.5),
        ("sensor.power", METRIC_ROLLING_AVG, 150.0),
    ]
    updater = SentinelBaselineUpdater(MagicMock(), _make_pool_with_rows(rows), {})

    result = await updater.async_fetch_baselines()

    assert result["sensor.temperature"][METRIC_ROLLING_AVG] == pytest.approx(21.5)
    assert result["sensor.power"][METRIC_ROLLING_AVG] == pytest.approx(150.0)


@pytest.mark.asyncio
async def test_fetch_baselines_returns_empty_dict_on_db_error() -> None:
    """async_fetch_baselines returns {} when the DB query raises."""
    mock_pool = MagicMock()
    mock_pool.connection = MagicMock(side_effect=OSError("DB down"))

    updater = SentinelBaselineUpdater(MagicMock(), mock_pool, {})
    result = await updater.async_fetch_baselines()

    assert result == {}


@pytest.mark.asyncio
async def test_fetch_baselines_returns_empty_dict_when_no_rows() -> None:
    """async_fetch_baselines returns {} when the table is empty."""
    updater = SentinelBaselineUpdater(MagicMock(), _make_pool_with_rows([]), {})
    result = await updater.async_fetch_baselines()

    assert result == {}


# ---------------------------------------------------------------------------
# 7. async_fetch_baselines — min_samples filter
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_fetch_baselines_passes_min_samples_as_sql_param() -> None:
    """async_fetch_baselines passes min_samples to the SQL query."""
    received_params: list[Any] = []

    mock_cursor = MagicMock()

    async def _capture_execute(_sql: str, params: Any) -> None:
        received_params.append(params)

    mock_cursor.execute = _capture_execute
    mock_cursor.fetchall = AsyncMock(return_value=[])
    mock_cursor.__aenter__ = AsyncMock(return_value=mock_cursor)
    mock_cursor.__aexit__ = AsyncMock(return_value=False)

    mock_conn = MagicMock()
    mock_conn.cursor = MagicMock(return_value=mock_cursor)
    mock_conn.__aenter__ = AsyncMock(return_value=mock_conn)
    mock_conn.__aexit__ = AsyncMock(return_value=False)

    mock_pool = MagicMock()
    mock_pool.connection = MagicMock(return_value=mock_conn)

    updater = SentinelBaselineUpdater(MagicMock(), mock_pool, {})
    await updater.async_fetch_baselines(min_samples=10)

    assert len(received_params) == 1
    # min_samples must be the first (and only) param in the WHERE clause
    assert received_params[0] == (10,)


@pytest.mark.asyncio
async def test_fetch_baselines_uses_recommended_default_when_min_samples_none() -> None:
    """async_fetch_baselines uses RECOMMENDED_SENTINEL_BASELINE_MIN_SAMPLES when not given."""
    received_params: list[Any] = []

    mock_cursor = MagicMock()

    async def _capture_execute(_sql: str, params: Any) -> None:
        received_params.append(params)

    mock_cursor.execute = _capture_execute
    mock_cursor.fetchall = AsyncMock(return_value=[])
    mock_cursor.__aenter__ = AsyncMock(return_value=mock_cursor)
    mock_cursor.__aexit__ = AsyncMock(return_value=False)

    mock_conn = MagicMock()
    mock_conn.cursor = MagicMock(return_value=mock_cursor)
    mock_conn.__aenter__ = AsyncMock(return_value=mock_conn)
    mock_conn.__aexit__ = AsyncMock(return_value=False)

    mock_pool = MagicMock()
    mock_pool.connection = MagicMock(return_value=mock_conn)

    updater = SentinelBaselineUpdater(MagicMock(), mock_pool, {})
    await updater.async_fetch_baselines()  # min_samples=None → uses default

    assert received_params[0] == (RECOMMENDED_SENTINEL_BASELINE_MIN_SAMPLES,)


# ---------------------------------------------------------------------------
# 8. async_fetch_full_baselines
# ---------------------------------------------------------------------------


def _make_pool_with_full_rows(rows: list[Any]) -> MagicMock:
    """Return a mock pool returning *rows* from fetchall (no filter param)."""
    mock_cursor = MagicMock()
    mock_cursor.execute = AsyncMock()
    mock_cursor.fetchall = AsyncMock(return_value=rows)
    mock_cursor.__aenter__ = AsyncMock(return_value=mock_cursor)
    mock_cursor.__aexit__ = AsyncMock(return_value=False)

    mock_conn = MagicMock()
    mock_conn.cursor = MagicMock(return_value=mock_cursor)
    mock_conn.__aenter__ = AsyncMock(return_value=mock_conn)
    mock_conn.__aexit__ = AsyncMock(return_value=False)

    mock_pool = MagicMock()
    mock_pool.connection = MagicMock(return_value=mock_conn)
    return mock_pool


@pytest.mark.asyncio
async def test_fetch_full_baselines_returns_rich_dict() -> None:
    """async_fetch_full_baselines returns {entity_id: {metric: {value, sample_count, updated_at, freshness}}}."""
    recent = datetime.now(UTC)
    rows = [
        {
            "entity_id": "sensor.power",
            "metric": METRIC_ROLLING_AVG,
            "value": 100.0,
            "sample_count": 25,
            "updated_at": recent,
        }
    ]
    updater = SentinelBaselineUpdater(MagicMock(), _make_pool_with_full_rows(rows), {})
    result = await updater.async_fetch_full_baselines()

    assert result is not None
    assert "sensor.power" in result
    entry = result["sensor.power"][METRIC_ROLLING_AVG]
    assert entry["value"] == pytest.approx(100.0)
    assert entry["sample_count"] == 25
    assert entry["freshness"] == BASELINE_FRESH


@pytest.mark.asyncio
async def test_fetch_full_baselines_marks_stale_rows() -> None:
    """async_fetch_full_baselines marks freshness=stale for old rows."""
    old = datetime.now(UTC) - timedelta(hours=5)
    rows = [
        {
            "entity_id": "sensor.temp",
            "metric": METRIC_ROLLING_AVG,
            "value": 22.0,
            "sample_count": 30,
            "updated_at": old,
        }
    ]
    updater = SentinelBaselineUpdater(MagicMock(), _make_pool_with_full_rows(rows), {})
    result = await updater.async_fetch_full_baselines()

    assert result is not None
    assert result["sensor.temp"][METRIC_ROLLING_AVG]["freshness"] == BASELINE_STALE


@pytest.mark.asyncio
async def test_fetch_full_baselines_returns_none_on_db_error() -> None:
    """async_fetch_full_baselines returns None when the DB raises."""
    mock_pool = MagicMock()
    mock_pool.connection = MagicMock(side_effect=OSError("DB down"))
    updater = SentinelBaselineUpdater(MagicMock(), mock_pool, {})
    result = await updater.async_fetch_full_baselines()
    assert result is None


@pytest.mark.asyncio
async def test_fetch_full_baselines_handles_none_updated_at() -> None:
    """async_fetch_full_baselines sets freshness=unavailable when updated_at is None."""
    rows = [
        {
            "entity_id": "sensor.x",
            "metric": METRIC_ROLLING_AVG,
            "value": 1.0,
            "sample_count": 5,
            "updated_at": None,
        }
    ]
    updater = SentinelBaselineUpdater(MagicMock(), _make_pool_with_full_rows(rows), {})
    result = await updater.async_fetch_full_baselines()
    assert result is not None
    assert result["sensor.x"][METRIC_ROLLING_AVG]["freshness"] == BASELINE_UNAVAILABLE


# ---------------------------------------------------------------------------
# 9. async_reset_baseline
# ---------------------------------------------------------------------------


def _make_pool_with_commit_and_rowcount(rowcount: int) -> MagicMock:
    """Return a mock pool whose cursor has a given rowcount after DELETE."""
    mock_cursor = MagicMock()
    mock_cursor.execute = AsyncMock()
    mock_cursor.rowcount = rowcount
    mock_cursor.__aenter__ = AsyncMock(return_value=mock_cursor)
    mock_cursor.__aexit__ = AsyncMock(return_value=False)

    mock_conn = MagicMock()
    mock_conn.cursor = MagicMock(return_value=mock_cursor)
    mock_conn.commit = AsyncMock()
    mock_conn.__aenter__ = AsyncMock(return_value=mock_conn)
    mock_conn.__aexit__ = AsyncMock(return_value=False)

    mock_pool = MagicMock()
    mock_pool.connection = MagicMock(return_value=mock_conn)
    return mock_pool


@pytest.mark.asyncio
async def test_reset_baseline_specific_entity_removes_from_established() -> None:
    """async_reset_baseline removes entity from _established set."""
    updater = SentinelBaselineUpdater(
        MagicMock(), _make_pool_with_commit_and_rowcount(2), {}
    )
    updater._established = {"sensor.power", "sensor.temp"}

    await updater.async_reset_baseline("sensor.power")

    assert "sensor.power" not in updater._established
    assert "sensor.temp" in updater._established  # other entity untouched


@pytest.mark.asyncio
async def test_reset_baseline_all_entities_clears_established() -> None:
    """async_reset_baseline(None) clears the entire _established set."""
    updater = SentinelBaselineUpdater(
        MagicMock(), _make_pool_with_commit_and_rowcount(10), {}
    )
    updater._established = {"sensor.a", "sensor.b", "sensor.c"}

    await updater.async_reset_baseline(None)

    assert updater._established == set()


@pytest.mark.asyncio
async def test_reset_baseline_returns_deleted_count() -> None:
    """async_reset_baseline returns the rowcount from the DELETE statement."""
    updater = SentinelBaselineUpdater(
        MagicMock(), _make_pool_with_commit_and_rowcount(5), {}
    )
    deleted = await updater.async_reset_baseline("sensor.power")
    assert deleted == 5


@pytest.mark.asyncio
async def test_reset_baseline_returns_negative_on_db_error() -> None:
    """async_reset_baseline returns -1 when the DB raises."""
    mock_pool = MagicMock()
    mock_pool.connection = MagicMock(side_effect=OSError("DB down"))
    updater = SentinelBaselineUpdater(MagicMock(), mock_pool, {})
    deleted = await updater.async_reset_baseline("sensor.power")
    assert deleted == -1


# ---------------------------------------------------------------------------
# 10. async_fetch_ready_entity_ids
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_fetch_ready_entity_ids_returns_list_from_dict_rows() -> None:
    """async_fetch_ready_entity_ids returns entity_ids from dict rows."""
    rows = [{"entity_id": "sensor.a"}, {"entity_id": "sensor.b"}]
    updater = SentinelBaselineUpdater(MagicMock(), _make_pool_with_rows(rows), {})
    result = await updater.async_fetch_ready_entity_ids()
    assert set(result) == {"sensor.a", "sensor.b"}


@pytest.mark.asyncio
async def test_fetch_ready_entity_ids_returns_list_from_tuple_rows() -> None:
    """async_fetch_ready_entity_ids handles tuple rows."""
    rows = [("sensor.x",), ("sensor.y",)]
    updater = SentinelBaselineUpdater(MagicMock(), _make_pool_with_rows(rows), {})
    result = await updater.async_fetch_ready_entity_ids()
    assert set(result) == {"sensor.x", "sensor.y"}


@pytest.mark.asyncio
async def test_fetch_ready_entity_ids_returns_empty_on_db_error() -> None:
    """async_fetch_ready_entity_ids returns [] when the DB raises."""
    mock_pool = MagicMock()
    mock_pool.connection = MagicMock(side_effect=OSError("DB down"))
    updater = SentinelBaselineUpdater(MagicMock(), mock_pool, {})
    result = await updater.async_fetch_ready_entity_ids()
    assert result == []


# ---------------------------------------------------------------------------
# 11. EMA aging/decay — max_samples cap
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_update_baselines_passes_max_samples_to_upsert() -> None:
    """_update_baselines includes max_samples in all three LEAST() positions in upsert SQL."""
    captured_sqls: list[tuple[str, Any]] = []

    mock_cursor = MagicMock()

    async def _capture(sql: str, params: Any) -> None:
        captured_sqls.append((sql, params))

    mock_cursor.execute = _capture
    mock_cursor.fetchall = AsyncMock(return_value=[])
    mock_cursor.fetchone = AsyncMock(return_value=None)
    mock_cursor.__aenter__ = AsyncMock(return_value=mock_cursor)
    mock_cursor.__aexit__ = AsyncMock(return_value=False)

    mock_conn = MagicMock()
    mock_conn.cursor = MagicMock(return_value=mock_cursor)
    mock_conn.commit = AsyncMock()
    mock_conn.__aenter__ = AsyncMock(return_value=mock_conn)
    mock_conn.__aexit__ = AsyncMock(return_value=False)

    mock_pool = MagicMock()
    mock_pool.connection = MagicMock(return_value=mock_conn)

    options = {CONF_SENTINEL_BASELINE_MAX_SAMPLES: 100}
    updater = SentinelBaselineUpdater(MagicMock(), mock_pool, options)

    snapshot = _snapshot([_entity("sensor.temperature", "22.0")])
    await updater._update_baselines(snapshot)

    # Find the rolling_avg upsert (first upsert call after the FETCH_REFS SELECT).
    upsert_calls = [(sql, p) for sql, p in captured_sqls if "INSERT INTO" in sql]
    assert len(upsert_calls) >= 1
    # max_samples=100 must appear three times in the rolling_avg upsert params.
    rolling_upsert = upsert_calls[0]
    params = rolling_upsert[1]
    assert params.count(100) >= 3


# ---------------------------------------------------------------------------
# 12. Establishment notification tracking
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_update_baselines_fires_establishment_notification_on_crossing() -> None:
    """_update_baselines fires an establishment notification when sample_count crosses min_samples."""
    # sample_count returned after upsert is exactly min_samples (20).
    min_samples = 20
    count_row = {"sample_count": min_samples}

    call_log: list[tuple[str, str, dict[str, Any]]] = []

    mock_cursor = MagicMock()

    async def _execute(sql: str, params: Any) -> None:
        pass

    async def _fetchall() -> list[Any]:
        return []  # no existing refs

    async def _fetchone() -> dict[str, Any] | None:
        return count_row

    mock_cursor.execute = _execute
    mock_cursor.fetchall = _fetchall
    mock_cursor.fetchone = _fetchone
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

    async def _fake_service_call(
        domain: str, service: str, data: dict[str, Any], **_kw: Any
    ) -> None:
        call_log.append((domain, service, data))

    mock_hass.services = MagicMock()
    mock_hass.services.async_call = _fake_service_call

    options = {CONF_SENTINEL_BASELINE_MIN_SAMPLES: min_samples}
    updater = SentinelBaselineUpdater(mock_hass, mock_pool, options)
    # entity NOT in _established → will trigger notification
    updater._established = set()

    snapshot = _snapshot([_entity("sensor.temperature", "22.0")])
    await updater._update_baselines(snapshot)

    # Must be in _established after update
    assert "sensor.temperature" in updater._established
    # Notification must have been fired
    assert any(svc == "create" for _, svc, _ in call_log)


@pytest.mark.asyncio
async def test_update_baselines_skips_notification_for_already_established() -> None:
    """_update_baselines does NOT re-fire when entity is already in _established."""
    call_log: list[Any] = []

    mock_cursor = MagicMock()
    mock_cursor.execute = AsyncMock()
    mock_cursor.fetchall = AsyncMock(return_value=[])
    mock_cursor.fetchone = AsyncMock(return_value=None)
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

    async def _fake_service_call(*args: Any, **_kw: Any) -> None:
        call_log.append(args)

    mock_hass.services = MagicMock()
    mock_hass.services.async_call = _fake_service_call

    updater = SentinelBaselineUpdater(mock_hass, mock_pool, {})
    # Entity already established — should not fire notification or query sample_count
    updater._established = {"sensor.temperature"}

    snapshot = _snapshot([_entity("sensor.temperature", "22.0")])
    await updater._update_baselines(snapshot)

    # _FETCH_SAMPLE_COUNT_SQL must NOT have been called (entity skipped)
    assert call_log == []


# ---------------------------------------------------------------------------
# 13. Drift notification
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_update_baselines_fires_drift_notification_when_threshold_exceeded() -> (
    None
):
    """_update_baselines fires a drift notification when drift_pct >= threshold."""
    drift_notifications: list[dict[str, Any]] = []

    # reference_value in DB is 20.0; new value is 30.0 → 50% drift
    ref_rows = [
        {"entity_id": "sensor.power", "reference_value": 20.0, "reference_at": None}
    ]
    count_row = {"sample_count": 1}

    mock_cursor = MagicMock()
    call_count = 0

    async def _execute(sql: str, params: Any) -> None:
        pass

    async def _fetchall() -> list[Any]:
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return ref_rows
        return []

    async def _fetchone() -> dict[str, Any] | None:
        return count_row

    mock_cursor.execute = _execute
    mock_cursor.fetchall = _fetchall
    mock_cursor.fetchone = _fetchone
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

    async def _fake_service_call(
        _domain: str, _service: str, data: dict[str, Any], **_kw: Any
    ) -> None:
        if "Drift" in data.get("title", ""):
            drift_notifications.append(data)

    mock_hass.services = MagicMock()
    mock_hass.services.async_call = _fake_service_call

    options = {CONF_SENTINEL_BASELINE_DRIFT_THRESHOLD_PCT: 30.0}
    updater = SentinelBaselineUpdater(mock_hass, mock_pool, options)
    updater._established = {"sensor.power"}  # already established

    snapshot = _snapshot([_entity("sensor.power", "30.0")])
    await updater._update_baselines(snapshot)

    assert len(drift_notifications) == 1
    assert "sensor.power" in drift_notifications[0]["message"]


# ---------------------------------------------------------------------------
# 14. Appliance completion detection
# ---------------------------------------------------------------------------


def _power_entity(
    entity_id: str,
    state: str,
    device_class: str = "power",
    unit: str = "W",
    last_changed: str | None = None,
) -> dict[str, Any]:
    """Return a snapshot entity with power-related attributes."""
    ts = last_changed or datetime.now(UTC).isoformat()
    return {
        "entity_id": entity_id,
        "state": state,
        "domain": "sensor",
        "area": "Laundry",
        "attributes": {
            "device_class": device_class,
            "unit_of_measurement": unit,
        },
        "last_changed": ts,
        "last_updated": ts,
    }


def test_baseline_deviation_near_zero_power_is_completion() -> None:
    """Power entity dropping to <10% of active baseline → is_completion=True, severity='low'."""
    snapshot = _snapshot(
        [_power_entity("sensor.washer_power", "0.5", device_class="power", unit="W")]
    )
    # 800 W is a realistic active-cycle baseline (>= COMPLETION_MIN_ACTIVE_WATTS).
    baselines = {"sensor.washer_power": {METRIC_ROLLING_AVG: 800.0}}
    rule = _rule(entity_id="sensor.washer_power", threshold_pct=50.0, severity="medium")

    findings = evaluate_baseline_deviation(snapshot, rule, baselines)

    assert len(findings) == 1
    f = findings[0]
    assert f.evidence.get("is_completion") is True
    assert f.severity == "low"
    assert f.suggested_actions == []  # no action needed for a completed cycle


def test_baseline_deviation_standby_baseline_not_completion() -> None:
    """
    Appliance with low standby baseline dropping to 0 W → finding suppressed entirely.

    Reproduces the dishwasher/washer false-positive where the rolling average
    reflects standby power (~21 W) rather than an active cycle (~1200 W).
    A 21 W → 0 W swing is below MINIMUM_POWER_DEVIATION_WATTS and is pure
    standby noise — no finding should be emitted at all.
    """
    snapshot = _snapshot(
        [
            _power_entity(
                "sensor.dishwasher_power", "0.0", device_class="power", unit="W"
            )
        ]
    )
    # 21 W is a standby/idle baseline — absolute swing of 21 W is below the
    # MINIMUM_POWER_DEVIATION_WATTS threshold, so no finding should fire.
    baselines = {"sensor.dishwasher_power": {METRIC_ROLLING_AVG: 21.0}}
    rule = _rule(
        entity_id="sensor.dishwasher_power", threshold_pct=50.0, severity="medium"
    )

    findings = evaluate_baseline_deviation(snapshot, rule, baselines)

    # Standby-level noise is suppressed — no actionable finding.
    assert len(findings) == 0


def test_baseline_deviation_stale_idle_appliance_not_completion() -> None:
    """
    Appliance idle for longer than COMPLETION_RECENCY_SECS → no is_completion.

    Reproduces the dishwasher false-positive: the rolling average still reflects
    historical active power but the appliance has been idle for hours.  Without
    the recency guard this fires every evaluation cycle.
    """
    snapshot = _snapshot(
        [
            _power_entity(
                "sensor.dishwasher_power",
                "0.5",
                device_class="power",
                unit="W",
                last_changed="2025-01-01T00:00:00+00:00",  # hours ago
            )
        ]
    )
    baselines = {"sensor.dishwasher_power": {METRIC_ROLLING_AVG: 800.0}}
    rule = _rule(
        entity_id="sensor.dishwasher_power", threshold_pct=50.0, severity="medium"
    )

    findings = evaluate_baseline_deviation(snapshot, rule, baselines)

    assert len(findings) == 1
    f = findings[0]
    assert "is_completion" not in f.evidence
    assert f.severity == "medium"


def test_baseline_deviation_mid_cycle_power_drop_not_completion() -> None:
    """
    Power entity at 50% of baseline — above completion threshold, above min-watt guard.

    400 W is 50% of 800 W baseline (above COMPLETION_THRESHOLD_PCT=10%) and the
    absolute swing of 400 W is above MINIMUM_POWER_DEVIATION_WATTS=50 W.  The
    rule should emit a finding with no is_completion flag.
    """
    snapshot = _snapshot(
        [_power_entity("sensor.washer_power", "400.0", device_class="power", unit="W")]
    )
    baselines = {"sensor.washer_power": {METRIC_ROLLING_AVG: 800.0}}
    rule = _rule(entity_id="sensor.washer_power", threshold_pct=40.0, severity="medium")

    findings = evaluate_baseline_deviation(snapshot, rule, baselines)

    assert len(findings) == 1
    f = findings[0]
    assert "is_completion" not in f.evidence
    assert f.severity == "medium"


def test_baseline_deviation_generic_power_sensor_no_completion() -> None:
    """
    Generic power sensor (no appliance keyword) near zero → no is_completion.

    UPS, whole-home power, server rack, fridge, etc. must not be silently
    downgraded to low severity by the completion heuristic.  The standby guard
    is scoped to appliance circuits only, so a 40 W UPS dropping to 5 W fires
    normally even though the absolute swing (35 W) is below MINIMUM_POWER_DEVIATION_WATTS.
    """
    snapshot = _snapshot(
        [_power_entity("sensor.ups_power", "5.0", device_class="power", unit="W")]
    )
    # 40 W is a realistic UPS idle baseline.  Absolute swing ~35 W — this would
    # have been suppressed by the old broad guard; with the appliance-scoped guard
    # it fires correctly because "ups" is not in _APPLIANCE_HINTS.
    baselines = {"sensor.ups_power": {METRIC_ROLLING_AVG: 40.0}}
    rule = _rule(entity_id="sensor.ups_power", threshold_pct=50.0, severity="medium")

    findings = evaluate_baseline_deviation(snapshot, rule, baselines)

    assert len(findings) == 1
    f = findings[0]
    assert "is_completion" not in f.evidence
    assert f.severity == "medium"


def test_baseline_deviation_kw_standby_suppressed() -> None:
    """
    Power entity reporting in kW with small standby-level deviation is suppressed.

    0.03 kW → 0 kW is 30 W in absolute terms — below MINIMUM_POWER_DEVIATION_WATTS.
    Without kW-normalisation the guard would compare 0.03 < 50 (true) and suppress,
    but with the correct normalisation 0.03 kW * 1000 = 30 W < 50 W → still suppressed.
    Verifies the normalisation path for the "correctly suppressed" case.
    """
    snapshot = _snapshot(
        [_power_entity("sensor.dryer_power_kw", "0.0", device_class="power", unit="kW")]
    )
    baselines = {"sensor.dryer_power_kw": {METRIC_ROLLING_AVG: 0.03}}
    rule = _rule(
        entity_id="sensor.dryer_power_kw", threshold_pct=50.0, severity="medium"
    )

    findings = evaluate_baseline_deviation(snapshot, rule, baselines)

    # 30 W equivalent — standby noise, no finding.
    assert len(findings) == 0


def test_baseline_deviation_kw_large_drop_fires() -> None:
    """
    Power entity reporting in kW with a large absolute deviation is NOT suppressed.

    30 kW → 0 kW is 30,000 W equivalent, well above MINIMUM_POWER_DEVIATION_WATTS=50.
    Without kW-normalisation abs(30.0 - 0.0) = 30 < 50 → wrongly suppressed.
    With normalisation 30 * 1000 = 30,000 W > 50 W → finding fires correctly.
    """
    snapshot = _snapshot(
        [
            _power_entity(
                "sensor.washer_power_kw", "0.0", device_class="power", unit="kW"
            )
        ]
    )
    baselines = {"sensor.washer_power_kw": {METRIC_ROLLING_AVG: 30.0}}  # 30 kW
    rule = _rule(
        entity_id="sensor.washer_power_kw", threshold_pct=50.0, severity="medium"
    )

    findings = evaluate_baseline_deviation(snapshot, rule, baselines)

    # 30 kW load going dark — real event, not noise.
    assert len(findings) == 1


def test_baseline_deviation_anomaly_id_stable_without_friendly_name() -> None:
    """
    anomaly_id is identical regardless of whether friendly_name is present.

    friendly_name is excluded from the hash to prevent duplicate audit records
    when the entity is briefly unavailable and friendly_name="" differs from
    the normal-case "Dishwasher Power".
    """
    pinned_ts = "2025-01-15T10:00:00+00:00"

    entity_with_name = _power_entity(
        "sensor.dishwasher_power",
        "0.5",
        device_class="power",
        unit="W",
        last_changed=pinned_ts,
    )
    entity_with_name["friendly_name"] = "Dishwasher Power"

    entity_no_name = _power_entity(
        "sensor.dishwasher_power",
        "0.5",
        device_class="power",
        unit="W",
        last_changed=pinned_ts,
    )
    entity_no_name["friendly_name"] = ""

    baselines = {"sensor.dishwasher_power": {METRIC_ROLLING_AVG: 800.0}}
    rule = _rule(
        entity_id="sensor.dishwasher_power", threshold_pct=50.0, severity="medium"
    )

    findings_with = evaluate_baseline_deviation(
        _snapshot([entity_with_name]), rule, baselines
    )
    findings_without = evaluate_baseline_deviation(
        _snapshot([entity_no_name]), rule, baselines
    )

    assert len(findings_with) == 1
    assert len(findings_without) == 1
    assert findings_with[0].anomaly_id == findings_without[0].anomaly_id


def test_baseline_deviation_non_power_entity_no_completion() -> None:
    """Non-power entity (device_class=temperature) near zero → no is_completion key."""
    snapshot = _snapshot(
        [
            {
                "entity_id": "sensor.temp",
                "state": "0.5",
                "domain": "sensor",
                "area": "Living Room",
                "attributes": {
                    "device_class": "temperature",
                    "unit_of_measurement": "°C",
                },
                "last_changed": "2025-01-01T00:00:00+00:00",
                "last_updated": "2025-01-01T00:00:00+00:00",
            }
        ]
    )
    baselines = {"sensor.temp": {METRIC_ROLLING_AVG: 20.0}}
    rule = _rule(entity_id="sensor.temp", threshold_pct=50.0, severity="medium")

    findings = evaluate_baseline_deviation(snapshot, rule, baselines)

    assert len(findings) == 1
    assert "is_completion" not in findings[0].evidence


# ---------------------------------------------------------------------------
# DOW (day-of-week) baseline tests — Sprint 3 PR2
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Welford's algorithm — _update_dow_welford
# ---------------------------------------------------------------------------


def test_dow_welford_single_observation() -> None:
    """After one observation, mean equals the value and stddev is 0."""
    mock_hass = MagicMock()
    mock_pool = MagicMock()
    updater = SentinelBaselineUpdater(mock_hass, mock_pool, {})

    mean, stddev = updater._update_dow_welford(
        "sensor.temp", dow=0, hour=14, new_value=20.0
    )

    assert mean == pytest.approx(20.0)
    assert stddev == pytest.approx(0.0)
    assert updater._dow_state["sensor.temp:0:14"] == pytest.approx((20.0, 0.0, 1))


def test_dow_welford_multiple_observations_converge() -> None:
    """After N observations, running mean and stddev converge correctly."""
    mock_hass = MagicMock()
    mock_pool = MagicMock()
    updater = SentinelBaselineUpdater(mock_hass, mock_pool, {})

    values = [10.0, 20.0, 30.0, 40.0]
    final_mean = 0.0
    final_stddev = 0.0
    for v in values:
        final_mean, final_stddev = updater._update_dow_welford(
            "sensor.temp", dow=5, hour=8, new_value=v
        )

    # Population mean = 25; sample stddev = sqrt(((10-25)^2+(20-25)^2+(30-25)^2+(40-25)^2)/3)
    # = sqrt((225+25+25+225)/3) = sqrt(500/3) ≈ 12.91
    assert final_mean == pytest.approx(25.0)
    assert final_stddev == pytest.approx((500.0 / 3) ** 0.5, rel=1e-4)
    _, _, count = updater._dow_state["sensor.temp:5:8"]
    assert count == 4


def test_dow_welford_separate_slots_are_independent() -> None:
    """DOW slots for different entities/hours don't interfere."""
    mock_hass = MagicMock()
    mock_pool = MagicMock()
    updater = SentinelBaselineUpdater(mock_hass, mock_pool, {})

    updater._update_dow_welford("sensor.a", dow=0, hour=0, new_value=100.0)
    updater._update_dow_welford("sensor.b", dow=0, hour=0, new_value=5.0)

    mean_a, _, count_a = updater._dow_state["sensor.a:0:0"]
    mean_b, _, count_b = updater._dow_state["sensor.b:0:0"]

    assert mean_a == pytest.approx(100.0)
    assert mean_b == pytest.approx(5.0)
    assert count_a == 1
    assert count_b == 1


# ---------------------------------------------------------------------------
# evaluate_time_of_day_anomaly — DOW blending
# ---------------------------------------------------------------------------


def _dow_rule(entity_id: str = "sensor.temp") -> dict[str, Any]:
    return {
        "rule_id": "time_of_day_anomaly",
        "template_id": "time_of_day_anomaly",
        "params": {"entity_id": entity_id},
        "severity": "low",
        "confidence": 0.7,
        "is_sensitive": False,
        "suggested_actions": [],
    }


def _dow_snapshot(entity_id: str, state: str, hour: int, dow: int) -> Any:
    """Snapshot with a single entity and a specific local datetime."""
    # Create an ISO string matching the given DOW and hour.
    # We use a fixed base date (2026-01-05 = Monday) and offset by dow days.
    base_monday = datetime(2026, 1, 5, 0, 0, 0, tzinfo=UTC)
    dt = base_monday.replace(hour=hour) + timedelta(days=dow)
    now_str = dt.isoformat()
    return cast(
        "FullStateSnapshot",
        {
            "schema_version": 1,
            "generated_at": now_str,
            "entities": [
                {
                    "entity_id": entity_id,
                    "state": state,
                    "domain": entity_id.split(".", maxsplit=1)[0],
                    "area": "Living Room",
                    "attributes": {},
                    "last_changed": now_str,
                    "last_updated": now_str,
                }
            ],
            "camera_activity": [],
            "derived": {
                "now": now_str,
                "timezone": "UTC",
                "is_night": False,
                "anyone_home": True,
                "people_home": [],
                "people_away": [],
                "last_motion_by_area": {},
            },
        },
    )


def test_dow_blend_at_zero_samples_uses_global_mean() -> None:
    """With count=0 (no DOW data), falls back fully to global hourly mean."""
    # No DOW data — dow_data is empty — should fall back to global hourly detection.
    snapshot = _dow_snapshot("sensor.temp", "200", hour=14, dow=0)
    # Global hourly mean of 20 at hour 14; current=200 → 900% deviation
    baselines = {"sensor.temp": {f"{METRIC_HOURLY_PREFIX}14": 20.0}}
    rule = _dow_rule()

    # With no DOW data at all, evaluates as pure global
    findings = evaluate_time_of_day_anomaly(
        snapshot, rule, baselines, dow_data={}, dow_min_samples=4
    )

    assert len(findings) == 1


def test_dow_blend_at_full_samples_uses_dow_mean() -> None:
    """With count >= dow_min_samples, expected is entirely the DOW mean."""
    dow_min_samples = 4
    entity_id = "sensor.temp"
    dow, hour = 0, 14  # Monday 14:00

    snapshot = _dow_snapshot(entity_id, "25", hour=hour, dow=dow)
    # DOW mean=20, global mean=50 (very different to confirm DOW is used)
    dow_mean_key = f"{METRIC_DOW_AVG_PREFIX}{dow}_{hour}"
    dow_std_key = f"{METRIC_DOW_STD_PREFIX}{dow}_{hour}"
    global_key = f"{METRIC_HOURLY_PREFIX}{hour}"
    baselines = {entity_id: {global_key: 50.0}}
    dow_data = {
        entity_id: {
            dow_mean_key: (20.0, dow_min_samples),  # (value, count) — fully warm
            dow_std_key: (1.0, dow_min_samples),  # tight stddev
        }
    }

    findings = evaluate_time_of_day_anomaly(
        snapshot,
        rule=_dow_rule(entity_id),
        baselines=baselines,
        dow_data=dow_data,
        dow_min_samples=dow_min_samples,
    )

    # expected = 20.0 (DOW), deviation = |25-20| = 5, threshold = max(1*2, 20*0.3)=6 → no fire
    assert findings == []


def test_dow_blend_fires_for_genuine_outlier_vs_dow_mean() -> None:
    """A genuine outlier relative to the DOW mean fires even if near the global mean."""
    dow_min_samples = 4
    entity_id = "sensor.temp"
    dow, hour = 5, 20  # Saturday 20:00

    snapshot = _dow_snapshot(entity_id, "100", hour=hour, dow=dow)
    dow_mean_key = f"{METRIC_DOW_AVG_PREFIX}{dow}_{hour}"
    dow_std_key = f"{METRIC_DOW_STD_PREFIX}{dow}_{hour}"
    global_key = f"{METRIC_HOURLY_PREFIX}{hour}"
    # DOW mean=20, stddev=2; global=90 (close to current); current=100
    baselines = {entity_id: {global_key: 90.0}}
    dow_data = {
        entity_id: {
            dow_mean_key: (20.0, dow_min_samples),
            dow_std_key: (2.0, dow_min_samples),
        }
    }

    findings = evaluate_time_of_day_anomaly(
        snapshot,
        rule=_dow_rule(entity_id),
        baselines=baselines,
        dow_data=dow_data,
        dow_min_samples=dow_min_samples,
    )

    # expected=20 (fully DOW), threshold=max(2*2,20*0.3)=6, deviation=|100-20|=80 → fires
    assert len(findings) == 1
    ev = findings[0].evidence
    assert ev["dow_mean"] == pytest.approx(20.0)
    assert ev["blend_weight"] == pytest.approx(1.0)
    assert ev["deviation_direction"] == "above"


def test_dow_blend_no_fire_for_in_range_value() -> None:
    """A value within DOW mean ± 2*stddev does NOT fire."""
    dow_min_samples = 4
    entity_id = "sensor.temp"
    dow, hour = 6, 9  # Sunday 09:00

    snapshot = _dow_snapshot(entity_id, "22", hour=hour, dow=dow)
    dow_mean_key = f"{METRIC_DOW_AVG_PREFIX}{dow}_{hour}"
    dow_std_key = f"{METRIC_DOW_STD_PREFIX}{dow}_{hour}"
    global_key = f"{METRIC_HOURLY_PREFIX}{hour}"
    baselines = {entity_id: {global_key: 20.0}}
    dow_data = {
        entity_id: {
            dow_mean_key: (20.0, dow_min_samples),
            dow_std_key: (3.0, dow_min_samples),  # threshold = max(6, 20*0.3) = 6
        }
    }

    findings = evaluate_time_of_day_anomaly(
        snapshot,
        rule=_dow_rule(entity_id),
        baselines=baselines,
        dow_data=dow_data,
        dow_min_samples=dow_min_samples,
    )

    # deviation=2, threshold=6 → no fire
    assert findings == []


def test_dow_partial_blend_midpoint() -> None:
    """At count = dow_min_samples/2, blend weight is 0.5."""
    dow_min_samples = 4
    entity_id = "sensor.temp"
    dow, hour = 1, 7  # Tuesday 07:00
    half_count = dow_min_samples // 2  # 2

    snapshot = _dow_snapshot(entity_id, "200", hour=hour, dow=dow)
    dow_mean_key = f"{METRIC_DOW_AVG_PREFIX}{dow}_{hour}"
    global_key = f"{METRIC_HOURLY_PREFIX}{hour}"
    # DOW mean=10, global mean=10; both at 10; current=200 → fires regardless of blend
    baselines = {entity_id: {global_key: 10.0}}
    dow_data = {
        entity_id: {
            dow_mean_key: (10.0, half_count),  # w=0.5
        }
    }

    findings = evaluate_time_of_day_anomaly(
        snapshot,
        rule=_dow_rule(entity_id),
        baselines=baselines,
        dow_data=dow_data,
        dow_min_samples=dow_min_samples,
    )

    # expected=0.5*10 + 0.5*10 = 10; deviation=190 → fires
    assert len(findings) == 1
    assert findings[0].evidence["blend_weight"] == pytest.approx(0.5)


def test_dow_fallback_to_global_when_no_dow_data() -> None:
    """When dow_data=None, falls back to global hourly comparison."""
    entity_id = "sensor.temp"
    dow, hour = 0, 10  # Monday 10:00

    snapshot = _dow_snapshot(entity_id, "100", hour=hour, dow=dow)
    global_key = f"{METRIC_HOURLY_PREFIX}{hour}"
    baselines = {entity_id: {global_key: 20.0}}

    findings = evaluate_time_of_day_anomaly(
        snapshot,
        rule=_dow_rule(entity_id),
        baselines=baselines,
        dow_data=None,
    )

    # Global: deviation = 80/20 = 400% > 30% → fires
    assert len(findings) == 1


# ---------------------------------------------------------------------------
# DOW migration index — idempotency via CREATE INDEX IF NOT EXISTS
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_async_initialize_dow_index_idempotent() -> None:
    """async_initialize() must succeed when called twice (index DDL is IF NOT EXISTS)."""
    executed: list[str] = []
    mock_cursor = MagicMock()

    async def _exec(sql: str, *_args: Any) -> None:
        executed.append(sql.strip().split("\n")[0])

    mock_cursor.execute = _exec
    mock_cursor.fetchall = AsyncMock(return_value=[])
    mock_cursor.fetchone = AsyncMock(return_value=None)

    mock_conn = MagicMock()
    mock_conn.commit = AsyncMock()
    mock_conn.cursor = MagicMock(
        return_value=MagicMock(
            __aenter__=AsyncMock(return_value=mock_cursor),
            __aexit__=AsyncMock(return_value=False),
        )
    )
    mock_pool = MagicMock()
    mock_pool.connection = MagicMock(
        return_value=MagicMock(
            __aenter__=AsyncMock(return_value=mock_conn),
            __aexit__=AsyncMock(return_value=False),
        )
    )

    mock_hass = MagicMock()
    updater = SentinelBaselineUpdater(mock_hass, mock_pool, {})

    # First init
    await updater.async_initialize()
    # Second init — must not raise (idempotent DDL)
    await updater.async_initialize()

    # The index DDL should have been executed twice (once per init call).
    index_calls = [s for s in executed if "idx_sentinel_baselines" in s]
    assert len(index_calls) == 2


# ---------------------------------------------------------------------------
# DOW upserts written when weekly_patterns=True
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_update_baselines_writes_dow_rows_when_enabled() -> None:
    """When weekly_patterns=True, _update_baselines writes DOW mean+stddev rows."""
    written_metrics: list[str] = []

    mock_cursor = MagicMock()

    async def _exec(_sql: str, params: Any = None) -> None:
        if params and len(params) >= 2:
            written_metrics.append(str(params[1]))

    mock_cursor.execute = _exec
    mock_cursor.fetchone = AsyncMock(return_value={"sample_count": 5})
    mock_cursor.fetchall = AsyncMock(return_value=[])

    mock_conn = MagicMock()
    mock_conn.commit = AsyncMock()
    mock_conn.cursor = MagicMock(
        return_value=MagicMock(
            __aenter__=AsyncMock(return_value=mock_cursor),
            __aexit__=AsyncMock(return_value=False),
        )
    )
    mock_pool = MagicMock()
    mock_pool.connection = MagicMock(
        return_value=MagicMock(
            __aenter__=AsyncMock(return_value=mock_conn),
            __aexit__=AsyncMock(return_value=False),
        )
    )
    mock_hass = MagicMock()
    updater = SentinelBaselineUpdater(
        mock_hass,
        mock_pool,
        {CONF_SENTINEL_BASELINE_WEEKLY_PATTERNS: True},
    )

    snapshot = cast(
        "FullStateSnapshot",
        {
            "schema_version": 1,
            "generated_at": "2026-01-05T14:00:00+00:00",
            "entities": [
                {
                    "entity_id": "sensor.temp",
                    "state": "22.0",
                    "domain": "sensor",
                    "area": "Living Room",
                    "attributes": {},
                    "last_changed": "2026-01-05T14:00:00+00:00",
                    "last_updated": "2026-01-05T14:00:00+00:00",
                }
            ],
            "camera_activity": [],
            "derived": {
                "now": "2026-01-05T14:00:00+00:00",
                "timezone": "UTC",
                "is_night": False,
                "anyone_home": True,
                "people_home": [],
                "people_away": [],
                "last_motion_by_area": {},
            },
        },
    )

    await updater._update_baselines(snapshot)

    # Should write rolling_avg, hourly_avg_<H>, hourly_avg_<DOW>_<H>, hourly_stddev_<DOW>_<H>
    dow_mean_written = any(
        METRIC_DOW_AVG_PREFIX in m and m.count("_") >= _DOW_METRIC_PARTS + 1
        for m in written_metrics
    )
    dow_std_written = any(METRIC_DOW_STD_PREFIX in m for m in written_metrics)

    assert dow_mean_written, f"No DOW mean metric written; got: {written_metrics}"
    assert dow_std_written, f"No DOW stddev metric written; got: {written_metrics}"


@pytest.mark.asyncio
async def test_update_baselines_skips_dow_rows_when_disabled() -> None:
    """When weekly_patterns=False (default), no DOW rows are written."""
    written_metrics: list[str] = []

    mock_cursor = MagicMock()

    async def _exec(_sql: str, params: Any = None) -> None:
        if params and len(params) >= 2:
            written_metrics.append(str(params[1]))

    mock_cursor.execute = _exec
    mock_cursor.fetchone = AsyncMock(return_value={"sample_count": 5})
    mock_cursor.fetchall = AsyncMock(return_value=[])

    mock_conn = MagicMock()
    mock_conn.commit = AsyncMock()
    mock_conn.cursor = MagicMock(
        return_value=MagicMock(
            __aenter__=AsyncMock(return_value=mock_cursor),
            __aexit__=AsyncMock(return_value=False),
        )
    )
    mock_pool = MagicMock()
    mock_pool.connection = MagicMock(
        return_value=MagicMock(
            __aenter__=AsyncMock(return_value=mock_conn),
            __aexit__=AsyncMock(return_value=False),
        )
    )
    mock_hass = MagicMock()
    # weekly_patterns NOT set (defaults to False)
    updater = SentinelBaselineUpdater(mock_hass, mock_pool, {})

    snapshot = cast(
        "FullStateSnapshot",
        {
            "schema_version": 1,
            "generated_at": "2026-01-05T14:00:00+00:00",
            "entities": [
                {
                    "entity_id": "sensor.temp",
                    "state": "22.0",
                    "domain": "sensor",
                    "area": "Living Room",
                    "attributes": {},
                    "last_changed": "2026-01-05T14:00:00+00:00",
                    "last_updated": "2026-01-05T14:00:00+00:00",
                }
            ],
            "camera_activity": [],
            "derived": {
                "now": "2026-01-05T14:00:00+00:00",
                "timezone": "UTC",
                "is_night": False,
                "anyone_home": True,
                "people_home": [],
                "people_away": [],
                "last_motion_by_area": {},
            },
        },
    )

    await updater._update_baselines(snapshot)

    dow_std_written = any(METRIC_DOW_STD_PREFIX in m for m in written_metrics)
    assert not dow_std_written, (
        f"DOW stddev should not be written; got: {written_metrics}"
    )


# ---------------------------------------------------------------------------
# DOW warm-restart — _dow_state reconstructed from DB rows in async_initialize
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_async_initialize_reconstructs_dow_state_from_db() -> None:
    """
    async_initialize() must reconstruct _dow_state from existing DB rows.

    Simulates a restart: DB contains a mean row and a stddev row for entity
    sensor.power at DOW=0 (Monday), hour=9. Verifies that _dow_state is
    populated with the correct (mean, m2, count) tuple where
    m2 = stddev**2 * (count - 1).
    """
    # Pre-existing DB rows: mean=100.0 (count=5) and stddev=10.0 for Mon 09:00.
    dow, hour = 0, 9
    mean_metric = f"{METRIC_DOW_AVG_PREFIX}{dow}_{hour}"
    std_metric = f"{METRIC_DOW_STD_PREFIX}{dow}_{hour}"
    db_rows = [
        {
            "entity_id": "sensor.power",
            "metric": mean_metric,
            "value": 100.0,
            "sample_count": 5,
        },
        {
            "entity_id": "sensor.power",
            "metric": std_metric,
            "value": 10.0,
            "sample_count": 5,
        },
    ]

    call_count = 0

    mock_cursor = MagicMock()

    async def _exec(_sql: str, *_args: Any) -> None:
        pass

    async def _fetchall() -> list[Any]:
        nonlocal call_count
        call_count += 1
        # First fetchall: established-set query (returns empty).
        # Second fetchall: DOW warm-restart query (returns pre-populated rows).
        if call_count == 1:
            return []
        return db_rows

    mock_cursor.execute = _exec
    mock_cursor.fetchall = _fetchall
    mock_cursor.fetchone = AsyncMock(return_value=None)

    mock_conn = MagicMock()
    mock_conn.commit = AsyncMock()
    mock_conn.cursor = MagicMock(
        return_value=MagicMock(
            __aenter__=AsyncMock(return_value=mock_cursor),
            __aexit__=AsyncMock(return_value=False),
        )
    )
    mock_pool = MagicMock()
    mock_pool.connection = MagicMock(
        return_value=MagicMock(
            __aenter__=AsyncMock(return_value=mock_conn),
            __aexit__=AsyncMock(return_value=False),
        )
    )

    mock_hass = MagicMock()
    updater = SentinelBaselineUpdater(mock_hass, mock_pool, {})

    assert updater._dow_state == {}, "State must be empty before init"

    await updater.async_initialize()

    key = f"sensor.power:{dow}:{hour}"
    assert key in updater._dow_state, f"Expected key {key!r} in _dow_state"

    reconstructed_mean, reconstructed_m2, reconstructed_count = updater._dow_state[key]

    # mean should equal the DB value exactly.
    assert reconstructed_mean == pytest.approx(100.0)
    # count should equal the DB sample_count.
    assert reconstructed_count == 5
    # m2 = stddev**2 * (count - 1) = 10**2 * 4 = 400.
    expected_m2 = 10.0**2 * (5 - 1)
    assert reconstructed_m2 == pytest.approx(expected_m2)


# ---------------------------------------------------------------------------
# Cyclical load sustained deviation gate tests
# ---------------------------------------------------------------------------
# These tests exercise SentinelEngine._apply_sustained_gate() directly.
# To avoid instantiating the full engine (which requires hass, suppression,
# notifier, etc.) we create a minimal uninitialized instance and seed only
# the attributes the method accesses.


def _make_gate_engine() -> SentinelEngine:
    """Return a minimal SentinelEngine with only gate-relevant state populated."""
    engine = object.__new__(SentinelEngine)
    engine._cyclical_deviation_above_since = {}
    return engine


def _make_deviation_finding(
    entity_id: str,
    friendly_name: str = "",
    template_id: str = "baseline_deviation",
) -> AnomalyFinding:
    """Construct a minimal AnomalyFinding for gate testing."""
    return AnomalyFinding(
        anomaly_id=f"test_{entity_id}",
        type="high_energy_consumption",
        severity="medium",
        confidence=0.9,
        triggering_entities=[entity_id],
        evidence={
            "template_id": template_id,
            "friendly_name": friendly_name,
        },
        suggested_actions=[],
        is_sensitive=False,
    )


def _gate_now() -> datetime:
    return datetime(2026, 4, 14, 12, 0, 0, tzinfo=UTC)


def test_gate_suppresses_below_duration() -> None:
    """Entity above threshold for < sustained_minutes → finding suppressed."""
    engine = _make_gate_engine()
    finding = _make_deviation_finding("sensor.fridge_power", "Fridge")
    now = _gate_now()

    # First call: clock starts, finding suppressed.
    result = SentinelEngine._apply_sustained_gate(engine, [finding], now, 20)
    assert result == [], "First appearance should be suppressed (clock started)"

    # Advance 10 minutes — still below 20-minute gate.
    later = now + timedelta(minutes=10)
    result = SentinelEngine._apply_sustained_gate(engine, [finding], later, 20)
    assert result == [], "10 min elapsed < 20 min gate — still suppressed"


def test_gate_fires_after_duration() -> None:
    """Entity above threshold for >= sustained_minutes → finding fires."""
    engine = _make_gate_engine()
    finding = _make_deviation_finding("sensor.refrigerator_power", "Refrigerator")
    now = _gate_now()

    # Seed the clock as if it started 21 minutes ago.
    entity_id = finding.triggering_entities[0]
    engine._cyclical_deviation_above_since[entity_id] = now - timedelta(minutes=21)

    result = SentinelEngine._apply_sustained_gate(engine, [finding], now, 20)
    assert len(result) == 1, "Gate exceeded — finding should fire"
    assert result[0] is finding

    # Clock should have been reset to now.
    assert engine._cyclical_deviation_above_since[entity_id] == now


def test_gate_clears_when_entity_absent() -> None:
    """Entity drops below threshold → clock cleared; next cycle waits full duration."""
    engine = _make_gate_engine()
    finding = _make_deviation_finding("sensor.freezer_power", "Freezer")
    now = _gate_now()

    # Seed the clock.
    entity_id = finding.triggering_entities[0]
    engine._cyclical_deviation_above_since[entity_id] = now - timedelta(minutes=5)

    # Run with an empty findings list — entity has dropped below threshold.
    result = SentinelEngine._apply_sustained_gate(engine, [], now, 20)
    assert result == []
    assert entity_id not in engine._cyclical_deviation_above_since

    # Next appearance restarts clock — should be suppressed again.
    result = SentinelEngine._apply_sustained_gate(engine, [finding], now, 20)
    assert result == [], "Clock reset — should be suppressed from scratch"


def test_gate_applies_to_time_of_day_anomaly() -> None:
    """time_of_day_anomaly findings for cyclical entities are gated."""
    engine = _make_gate_engine()
    finding = _make_deviation_finding(
        "sensor.fridge_energy",
        "Fridge",
        template_id="time_of_day_anomaly",
    )
    now = _gate_now()

    result = SentinelEngine._apply_sustained_gate(engine, [finding], now, 20)
    assert result == [], "time_of_day_anomaly for cyclical entity should be gated"
    assert "sensor.fridge_energy" in engine._cyclical_deviation_above_since


def test_gate_passes_non_cyclical_entity() -> None:
    """Non-cyclical entity findings pass through the gate unchanged."""
    engine = _make_gate_engine()
    finding = _make_deviation_finding("sensor.living_room_tv_power", "TV")
    now = _gate_now()

    result = SentinelEngine._apply_sustained_gate(engine, [finding], now, 20)
    assert len(result) == 1, "Non-cyclical entity should pass through unchanged"
    assert result[0] is finding
    assert engine._cyclical_deviation_above_since == {}


def test_gate_disabled_when_zero() -> None:
    """sustained_minutes = 0 → gate disabled; all findings pass through unchanged."""
    engine = _make_gate_engine()
    fridge_finding = _make_deviation_finding("sensor.fridge_power", "Fridge")
    tv_finding = _make_deviation_finding("sensor.tv_power", "TV")
    now = _gate_now()

    result = SentinelEngine._apply_sustained_gate(
        engine, [fridge_finding, tv_finding], now, 0
    )
    assert len(result) == 2, "sustained_minutes=0 — all findings should pass through"
    assert engine._cyclical_deviation_above_since == {}


def test_gate_dot_split_entity_id() -> None:
    """Entity IDs with dots (sensor.refrigerator_power) match CYCLICAL_LOAD_HINTS."""
    engine = _make_gate_engine()
    # 'refrigerator' should match after splitting on the dot.
    finding = _make_deviation_finding("sensor.refrigerator_power", "")
    now = _gate_now()

    result = SentinelEngine._apply_sustained_gate(engine, [finding], now, 20)
    assert result == [], "sensor.refrigerator_power should be gated (dot-split match)"

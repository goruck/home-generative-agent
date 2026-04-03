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
    RECOMMENDED_SENTINEL_BASELINE_MIN_SAMPLES,
)
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
    is_night: bool = False,
    anyone_home: bool = True,
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


def _rule(
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

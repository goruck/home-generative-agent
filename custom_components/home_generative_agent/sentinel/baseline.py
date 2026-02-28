"""
Sentinel baseline storage and temporal detectors — Issue #265.

Provides ``SentinelBaselineUpdater``, an independent background task that
writes rolling statistical summaries (per entity, per metric) to the
``sentinel_baselines`` PostgreSQL table on a configurable cadence
(default 15 min).

Temporal detection helpers
--------------------------
``evaluate_baseline_deviation`` — fires when a numeric entity state
    deviates from its rolling average by more than ``threshold_pct`` percent.
``evaluate_time_of_day_anomaly`` — fires when a numeric entity state differs
    from the expected hour-of-day rolling average by more than
    ``threshold_pct`` percent.

Both helpers return lists of ``AnomalyFinding`` and can be registered as
dynamic-rule evaluators inside ``sentinel/dynamic_rules.py``.

Baseline freshness
------------------
``check_baseline_freshness`` returns ``"fresh"``, ``"stale"``, or
``"unavailable"`` for a given entity_id/metric, based on the age of the
most-recent update row in PostgreSQL.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
from datetime import UTC
from typing import TYPE_CHECKING, Any

from homeassistant.util import dt as dt_util

from custom_components.home_generative_agent.const import (
    CONF_SENTINEL_BASELINE_FRESHNESS_THRESHOLD_SECONDS,
    CONF_SENTINEL_BASELINE_UPDATE_INTERVAL_MINUTES,
    RECOMMENDED_SENTINEL_BASELINE_FRESHNESS_THRESHOLD_SECONDS,
    RECOMMENDED_SENTINEL_BASELINE_UPDATE_INTERVAL_MINUTES,
)

from .models import AnomalyFinding, Severity, build_anomaly_id

if TYPE_CHECKING:
    from homeassistant.core import HomeAssistant
    from psycopg import AsyncConnection
    from psycopg.rows import DictRow
    from psycopg_pool import AsyncConnectionPool

    from custom_components.home_generative_agent.snapshot.schema import (
        FullStateSnapshot,
    )

LOGGER = logging.getLogger(__name__)

# Baseline freshness states.
BASELINE_FRESH = "fresh"
BASELINE_STALE = "stale"
BASELINE_UNAVAILABLE = "unavailable"

# Rolling metric name written for the global time-window average.
METRIC_ROLLING_AVG = "rolling_avg"
# Rolling metric name keyed by hour-of-day (e.g. "hourly_avg_14").
METRIC_HOURLY_PREFIX = "hourly_avg_"

# DDL for the baseline table.
_CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS sentinel_baselines (
    entity_id   TEXT NOT NULL,
    metric      TEXT NOT NULL,
    period      TEXT NOT NULL,
    value       DOUBLE PRECISION NOT NULL,
    sample_count INTEGER NOT NULL DEFAULT 0,
    updated_at  TIMESTAMPTZ NOT NULL,
    PRIMARY KEY (entity_id, metric, period)
)
"""

# Upsert that maintains an exponential moving average with exact sample count.
_UPSERT_SQL = """
INSERT INTO sentinel_baselines
    (entity_id, metric, period, value, sample_count, updated_at)
VALUES
    (%s, %s, %s, %s, 1, %s)
ON CONFLICT (entity_id, metric, period) DO UPDATE SET
    value = (
        sentinel_baselines.value
        * sentinel_baselines.sample_count
        + EXCLUDED.value
    ) / (sentinel_baselines.sample_count + 1),
    sample_count = sentinel_baselines.sample_count + 1,
    updated_at   = EXCLUDED.updated_at
"""

_FRESHNESS_SQL = """
SELECT updated_at
FROM sentinel_baselines
WHERE entity_id = %s AND metric = %s AND period = %s
LIMIT 1
"""


class SentinelBaselineUpdater:
    """
    Independent background task that maintains rolling baselines in PostgreSQL.

    Lifecycle: call ``start()`` after the pool is ready; call ``stop()`` on
    integration unload.  The updater runs independently of the main sentinel
    detection loop so that slow DB writes never delay rule evaluation.
    """

    def __init__(
        self,
        hass: HomeAssistant,
        pool: AsyncConnectionPool[AsyncConnection[DictRow]],
        options: dict[str, Any],
    ) -> None:
        """Initialise with an open psycopg connection pool."""
        self._hass = hass
        self._pool = pool
        self._options = options
        self._task: asyncio.Task[None] | None = None
        self._stop_event = asyncio.Event()

    # ---------------------------------------------------------------------- #
    # Lifecycle
    # ---------------------------------------------------------------------- #

    async def async_initialize(self) -> None:
        """Create the ``sentinel_baselines`` table if it does not exist."""
        try:
            async with self._pool.connection() as conn, conn.cursor() as cur:
                await cur.execute(_CREATE_TABLE_SQL)
                await conn.commit()
            LOGGER.debug("sentinel_baselines table ready.")
        except Exception:
            LOGGER.exception(
                "Failed to initialise sentinel_baselines table; "
                "baseline updates will be skipped."
            )

    def start(self) -> None:
        """Start the background update loop."""
        if self._task is not None:
            return
        self._stop_event.clear()
        self._task = self._hass.async_create_task(self._run_loop())
        LOGGER.debug("SentinelBaselineUpdater started.")

    async def stop(self) -> None:
        """Stop the background update loop."""
        if self._task is None:
            return
        self._stop_event.set()
        self._task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await self._task
        self._task = None
        LOGGER.debug("SentinelBaselineUpdater stopped.")

    # ---------------------------------------------------------------------- #
    # Run loop
    # ---------------------------------------------------------------------- #

    async def _run_loop(self) -> None:
        interval_minutes = _coerce_int(
            self._options.get(CONF_SENTINEL_BASELINE_UPDATE_INTERVAL_MINUTES),
            default=RECOMMENDED_SENTINEL_BASELINE_UPDATE_INTERVAL_MINUTES,
        )
        interval_seconds = interval_minutes * 60
        LOGGER.info("Baseline update loop started (interval=%d min).", interval_minutes)
        while not self._stop_event.is_set():
            try:
                from custom_components.home_generative_agent.snapshot.builder import (  # noqa: PLC0415
                    async_build_full_state_snapshot,
                )

                snapshot = await async_build_full_state_snapshot(self._hass)
                await self._update_baselines(snapshot)
            except (ValueError, TypeError, KeyError, asyncio.CancelledError):
                raise
            except Exception:
                LOGGER.exception("Baseline update failed; will retry next cycle.")
            try:
                await asyncio.wait_for(
                    asyncio.shield(self._stop_event.wait()),
                    timeout=interval_seconds,
                )
                break  # stop_event was set
            except TimeoutError:
                continue

    # ---------------------------------------------------------------------- #
    # Baseline writes
    # ---------------------------------------------------------------------- #

    async def _update_baselines(self, snapshot: FullStateSnapshot) -> None:
        """Upsert rolling stats for all numeric entities in *snapshot*."""
        now = dt_util.utcnow()
        hour_str = str(now.hour)

        rows: list[tuple[str, str, str, float, Any]] = []
        for entity in snapshot.get("entities", []):
            entity_id = entity.get("entity_id", "")
            if not entity_id:
                continue
            try:
                value = float(str(entity.get("state", "")))
            except (TypeError, ValueError):
                continue
            # Global rolling average.
            rows.append((entity_id, METRIC_ROLLING_AVG, "rolling", value, now))
            # Hour-of-day rolling average.
            rows.append(
                (
                    entity_id,
                    f"{METRIC_HOURLY_PREFIX}{hour_str}",
                    "hourly",
                    value,
                    now,
                )
            )

        if not rows:
            return

        try:
            async with self._pool.connection() as conn, conn.cursor() as cur:
                for entity_id, metric, period, value, updated_at in rows:
                    await cur.execute(
                        _UPSERT_SQL,
                        (entity_id, metric, period, value, updated_at),
                    )
                await conn.commit()
            LOGGER.debug(
                "Baseline upsert completed for %d entity-metric pairs.", len(rows)
            )
        except Exception:
            LOGGER.exception("Baseline DB write failed.")

    # ---------------------------------------------------------------------- #
    # Freshness query
    # ---------------------------------------------------------------------- #

    async def check_freshness(
        self,
        entity_id: str,
        metric: str = METRIC_ROLLING_AVG,
        period: str = "rolling",
    ) -> str:
        """
        Return ``"fresh"``, ``"stale"``, or ``"unavailable"``.

        ``"unavailable"`` means no baseline row exists yet.
        ``"stale"`` means the most-recent update is older than the configured
        freshness threshold.
        ``"fresh"`` means the baseline was updated within the threshold.
        """
        freshness_threshold_seconds = _coerce_int(
            self._options.get(CONF_SENTINEL_BASELINE_FRESHNESS_THRESHOLD_SECONDS),
            default=RECOMMENDED_SENTINEL_BASELINE_FRESHNESS_THRESHOLD_SECONDS,
        )

        try:
            async with self._pool.connection() as conn, conn.cursor() as cur:
                await cur.execute(_FRESHNESS_SQL, (entity_id, metric, period))
                row = await cur.fetchone()
        except Exception:  # noqa: BLE001
            LOGGER.debug(
                "Baseline freshness query failed for %s/%s/%s.",
                entity_id,
                metric,
                period,
            )
            return BASELINE_UNAVAILABLE

        if row is None:
            return BASELINE_UNAVAILABLE

        updated_at = row.get("updated_at") if isinstance(row, dict) else row[0]
        if updated_at is None:
            return BASELINE_UNAVAILABLE

        now = dt_util.utcnow()
        try:
            age_seconds = (now - updated_at).total_seconds()
        except (TypeError, AttributeError):
            return BASELINE_STALE

        if age_seconds <= freshness_threshold_seconds:
            return BASELINE_FRESH
        return BASELINE_STALE


# ---------------------------------------------------------------------------
# Temporal deviation detector
# ---------------------------------------------------------------------------


def evaluate_baseline_deviation(  # noqa: PLR0911
    snapshot: FullStateSnapshot,
    rule: dict[str, Any],
    baselines: dict[str, dict[str, float]],
) -> list[AnomalyFinding]:
    """
    Detect when an entity's state deviates from its rolling average.

    *baselines* is a nested dict: ``{entity_id: {metric: value}}``.
    Rule params:
      ``entity_id`` — required.
      ``threshold_pct`` — percentage deviation that triggers a finding
          (default 50.0).
      ``metric`` — which baseline metric to compare against
          (default ``"rolling_avg"``).

    Returns an empty list when the baseline is unavailable or no deviation.
    """
    params: dict[str, Any] = rule.get("params") or {}
    entity_id = params.get("entity_id")
    if not entity_id:
        return []

    threshold_pct = _coerce_float(params.get("threshold_pct"), default=50.0)
    metric = str(params.get("metric") or METRIC_ROLLING_AVG)

    entity_map = {e["entity_id"]: e for e in snapshot.get("entities", [])}
    entity = entity_map.get(entity_id)
    if entity is None:
        return []

    try:
        current_value = float(str(entity.get("state", "")))
    except (TypeError, ValueError):
        return []

    baseline_value = (baselines.get(entity_id) or {}).get(metric)
    if baseline_value is None:
        return []

    if baseline_value == 0.0:
        # Avoid division by zero; treat any non-zero current as deviation.
        if current_value == 0.0:
            return []
        deviation_pct = 100.0
    else:
        diff = abs(current_value - baseline_value)
        deviation_pct = diff / abs(baseline_value) * 100.0

    if deviation_pct < threshold_pct:
        return []

    rule_id = str(rule.get("rule_id") or "baseline_deviation")
    evidence = {
        "rule_id": rule_id,
        "template_id": rule.get("template_id", "baseline_deviation"),
        "entity_id": entity_id,
        "current_value": current_value,
        "baseline_value": baseline_value,
        "deviation_pct": round(deviation_pct, 2),
        "threshold_pct": threshold_pct,
        "metric": metric,
        "last_changed": entity.get("last_changed"),
    }
    severity_raw = str(rule.get("severity") or "low")
    severity: Severity = (  # type: ignore[assignment]
        severity_raw if severity_raw in {"low", "medium", "high"} else "low"
    )
    confidence = _coerce_float(rule.get("confidence"), default=0.7)
    suggested_actions = rule.get("suggested_actions") or []
    anomaly_id = build_anomaly_id(rule_id, [entity_id], evidence)

    return [
        AnomalyFinding(
            anomaly_id=anomaly_id,
            type=rule_id,
            severity=severity,
            confidence=confidence,
            triggering_entities=[entity_id],
            evidence=evidence,
            suggested_actions=(
                list(suggested_actions) if isinstance(suggested_actions, list) else []
            ),
            is_sensitive=bool(rule.get("is_sensitive", False)),
        )
    ]


def evaluate_time_of_day_anomaly(
    snapshot: FullStateSnapshot,
    rule: dict[str, Any],
    baselines: dict[str, dict[str, float]],
) -> list[AnomalyFinding]:
    """
    Detect when an entity's state differs from the expected hour-of-day average.

    Rule params are identical to ``evaluate_baseline_deviation`` but the
    metric compared is the hour-of-day rolling average for the current hour.
    """
    params: dict[str, Any] = rule.get("params") or {}
    entity_id = params.get("entity_id")
    if not entity_id:
        return []

    derived = snapshot.get("derived", {})
    now_str = derived.get("now", "")
    try:
        from datetime import datetime  # noqa: PLC0415

        now_dt = datetime.fromisoformat(now_str)
        if now_dt.tzinfo is None:
            now_dt = now_dt.replace(tzinfo=UTC)
        hour_str = str(now_dt.hour)
    except (TypeError, ValueError):
        return []

    hourly_rule = dict(rule)
    hourly_params = dict(params)
    hourly_params["metric"] = f"{METRIC_HOURLY_PREFIX}{hour_str}"
    hourly_rule["params"] = hourly_params
    if not hourly_rule.get("rule_id"):
        hourly_rule["rule_id"] = "time_of_day_anomaly"
    if not hourly_rule.get("template_id"):
        hourly_rule["template_id"] = "time_of_day_anomaly"

    return evaluate_baseline_deviation(snapshot, hourly_rule, baselines)


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _coerce_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _coerce_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default

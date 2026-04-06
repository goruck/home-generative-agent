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
from datetime import UTC, timedelta
from typing import TYPE_CHECKING, Any

from homeassistant.util import dt as dt_util

from custom_components.home_generative_agent.const import (
    CONF_SENTINEL_BASELINE_DRIFT_THRESHOLD_PCT,
    CONF_SENTINEL_BASELINE_FRESHNESS_THRESHOLD_SECONDS,
    CONF_SENTINEL_BASELINE_MAX_SAMPLES,
    CONF_SENTINEL_BASELINE_MIN_SAMPLES,
    CONF_SENTINEL_BASELINE_UPDATE_INTERVAL_MINUTES,
    CONF_SENTINEL_BASELINE_WEEKLY_PATTERNS,
    RECOMMENDED_SENTINEL_BASELINE_DOW_MIN_SAMPLES,
    RECOMMENDED_SENTINEL_BASELINE_DRIFT_THRESHOLD_PCT,
    RECOMMENDED_SENTINEL_BASELINE_FRESHNESS_THRESHOLD_SECONDS,
    RECOMMENDED_SENTINEL_BASELINE_MAX_SAMPLES,
    RECOMMENDED_SENTINEL_BASELINE_MIN_SAMPLES,
    RECOMMENDED_SENTINEL_BASELINE_UPDATE_INTERVAL_MINUTES,
    RECOMMENDED_SENTINEL_BASELINE_WEEKLY_PATTERNS,
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

# Appliance cycle-completion threshold: <10% of baseline → treat as cycle end.
COMPLETION_THRESHOLD_PCT = 0.10  # <10% of baseline → treat as cycle completion
# Minimum baseline power to consider a completion event meaningful.  A baseline
# below this value means the appliance was in standby (not an active cycle), so
# a drop to near-zero is just "turned off", not "cycle finished".
# Typical active draw: dishwasher ~1200 W, washer ~500 W, dryer ~3000 W.
# 100 W is safely above any standby level while below any active cycle.
COMPLETION_MIN_ACTIVE_WATTS = 100.0
# Recency window: only flag completion if last_changed is within this window.
# Prevents stale-baseline false positives when an appliance has been idle for
# hours but the rolling average hasn't converged to the new resting state yet.
# Set to 3x the default evaluation interval (3 x 300 s = 900 s / 15 min).
COMPLETION_RECENCY_SECS = 900
# Minimum absolute power deviation (watts) required to fire a baseline anomaly
# on a power-class entity.  A 2.9 W → 0 W swing is 100% relative deviation but
# is purely standby-level noise.  Any swing below this threshold is suppressed
# regardless of the configured threshold_pct.
MINIMUM_POWER_DEVIATION_WATTS = 50.0

# Entity-name keywords that identify dedicated appliance circuits.  Only these
# entities qualify for completion detection; generic power sensors (UPS, fridge,
# whole-home, server rack) are intentionally excluded.
_APPLIANCE_HINTS: frozenset[str] = frozenset(
    {
        "washer",
        "dryer",
        "dishwasher",
        "washing",
        "laundry",
        "oven",
        "microwave",
        "coffee",
        "kettle",
        "toaster",
        "iron",
        "vacuum",
        "robot",
    }
)

# Baseline freshness states.
BASELINE_FRESH = "fresh"
BASELINE_STALE = "stale"
BASELINE_UNAVAILABLE = "unavailable"

# Rolling metric name written for the global time-window average.
METRIC_ROLLING_AVG = "rolling_avg"
# Global hourly metric keyed by hour-of-day (e.g. "hourly_avg_14").
METRIC_HOURLY_PREFIX = "hourly_avg_"
# DOW (day-of-week) metric prefixes — per (DOW, hour) slot.
# Examples: "hourly_avg_5_14" = Friday 14:00 mean, "hourly_stddev_5_14" = stddev.
# DOW follows Python weekday(): 0=Monday, 6=Sunday.
# DST note: buckets use local time via dt_util.now(). The ambiguous fall-back
# hour during DST transitions double-counts one slot once per year — acceptable;
# do not "fix" by converting to UTC (would break weekly pattern separation).
METRIC_DOW_AVG_PREFIX = "hourly_avg_"  # same prefix, but with two numeric suffixes
METRIC_DOW_STD_PREFIX = "hourly_stddev_"
# Number of numeric parts in a DOW metric name (e.g. "5_14" = DOW 5, hour 14).
# Global hourly metrics have one part ("14"); DOW metrics have two ("5_14").
_DOW_METRIC_PARTS = 2

# DDL for the baseline table.
_CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS sentinel_baselines (
    entity_id       TEXT NOT NULL,
    metric          TEXT NOT NULL,
    period          TEXT NOT NULL,
    value           DOUBLE PRECISION NOT NULL,
    sample_count    INTEGER NOT NULL DEFAULT 0,
    updated_at      TIMESTAMPTZ NOT NULL,
    reference_value DOUBLE PRECISION,
    reference_at    TIMESTAMPTZ,
    PRIMARY KEY (entity_id, metric, period)
)
"""

# ALTER TABLE migrations for columns added after initial release.
_MIGRATE_REFERENCE_VALUE_SQL = """
ALTER TABLE sentinel_baselines
ADD COLUMN IF NOT EXISTS reference_value DOUBLE PRECISION
"""

_MIGRATE_REFERENCE_AT_SQL = """
ALTER TABLE sentinel_baselines
ADD COLUMN IF NOT EXISTS reference_at TIMESTAMPTZ
"""

# Upsert that maintains a capped EMA.
# MAX_SAMPLES (%s) caps old-count so new-sample weight stays >= 1/MAX_SAMPLES.
# Params: (entity_id, metric, period, value, updated_at, max_samples x3).
# The last three params are all MAX_SAMPLES (same value, used in LEAST expressions).
# RETURNING sample_count eliminates a post-commit SELECT per entity.
_UPSERT_SQL = """
INSERT INTO sentinel_baselines
    (entity_id, metric, period, value, sample_count, updated_at)
VALUES
    (%s, %s, %s, %s, 1, %s)
ON CONFLICT (entity_id, metric, period) DO UPDATE SET
    value = (
        sentinel_baselines.value
        * LEAST(sentinel_baselines.sample_count, %s - 1)
        + EXCLUDED.value
    ) / (LEAST(sentinel_baselines.sample_count, %s - 1) + 1),
    sample_count = LEAST(sentinel_baselines.sample_count + 1, %s),
    updated_at   = EXCLUDED.updated_at
RETURNING sample_count
"""

# Upsert that also updates the drift reference columns.
# Params: (entity_id, metric, period, value, updated_at, max_samples x3, ref_value, ref_at)  # noqa: E501
# RETURNING sample_count eliminates a post-commit SELECT per entity.
_UPSERT_WITH_REF_SQL = """
INSERT INTO sentinel_baselines
    (entity_id, metric, period, value, sample_count,
     updated_at, reference_value, reference_at)
VALUES
    (%s, %s, %s, %s, 1, %s, %s, %s)
ON CONFLICT (entity_id, metric, period) DO UPDATE SET
    value = (
        sentinel_baselines.value
        * LEAST(sentinel_baselines.sample_count, %s - 1)
        + EXCLUDED.value
    ) / (LEAST(sentinel_baselines.sample_count, %s - 1) + 1),
    sample_count    = LEAST(sentinel_baselines.sample_count + 1, %s),
    updated_at      = EXCLUDED.updated_at,
    reference_value = EXCLUDED.reference_value,
    reference_at    = EXCLUDED.reference_at
RETURNING sample_count
"""

_FRESHNESS_SQL = """
SELECT updated_at
FROM sentinel_baselines
WHERE entity_id = %s AND metric = %s AND period = %s
LIMIT 1
"""

# Fast fetch for rule evaluation — only returns entities above min_samples threshold.
_FETCH_ALL_SQL = """
SELECT entity_id, metric, value
FROM sentinel_baselines
WHERE sample_count >= %s
"""

# Rich fetch for the sentinel_get_baselines service — returns all columns.
_FETCH_FULL_SQL = """
SELECT entity_id, metric, value, sample_count, updated_at
FROM sentinel_baselines
ORDER BY entity_id, metric
"""

# Fetch reference values for drift detection (rolling_avg only).
_FETCH_REFS_SQL = """
SELECT entity_id, reference_value, reference_at
FROM sentinel_baselines
WHERE metric = %s AND period = 'rolling' AND entity_id = ANY(%s)
"""

# Seed the established set at startup: entities already above min_samples.
_FETCH_ESTABLISHED_SQL = """
SELECT DISTINCT entity_id
FROM sentinel_baselines
WHERE metric = %s AND sample_count >= %s
"""

# Delete all baselines for a specific entity (or all entities if no filter).
_DELETE_ENTITY_SQL = """
DELETE FROM sentinel_baselines WHERE entity_id = %s
"""

_DELETE_ALL_SQL = """
DELETE FROM sentinel_baselines
"""

# DOW index for fast per-(entity, metric) lookups added in async_initialize().
_CREATE_DOW_INDEX_SQL = """
CREATE INDEX IF NOT EXISTS idx_sentinel_baselines_entity_metric
ON sentinel_baselines (entity_id, metric)
"""

# Direct upsert for DOW slots — bypasses EMA because DOW slots are sparse
# (each updated at most once per week per entity).  EMA on sparse slots
# produces interpolation artefacts; Welford's algorithm gives exact running
# mean + variance without requiring continuous data.
_DOW_UPSERT_SQL = """
INSERT INTO sentinel_baselines
    (entity_id, metric, period, value, sample_count, updated_at)
VALUES
    (%s, %s, 'dow', %s, %s, %s)
ON CONFLICT (entity_id, metric, period) DO UPDATE SET
    value        = EXCLUDED.value,
    sample_count = EXCLUDED.sample_count,
    updated_at   = EXCLUDED.updated_at
"""

# Fetch all DOW rows (both mean and stddev) — no min-samples filter because
# the blend weight formula handles sparse slots gracefully (w → 0 as count → 0).
_FETCH_DOW_SQL = """
SELECT entity_id, metric, value, sample_count
FROM sentinel_baselines
WHERE period = 'dow'
"""

# Warm-restart query: fetch existing DOW mean + stddev rows on startup so
# Welford's _dow_state (mean, m2, count) can be reconstructed without full
# retraining.
_FETCH_DOW_WARMUP_SQL = """
SELECT entity_id, metric, value, sample_count
FROM sentinel_baselines
WHERE period = 'dow'
"""

# Aggregate stats query for health sensor — avoids full table scan.
# Params: (metric, freshness_threshold_seconds, min_samples).
_FETCH_STATS_SQL = """
SELECT
    COUNT(DISTINCT entity_id)                                           AS entity_count,
    COUNT(*) FILTER (WHERE metric = %s
        AND EXTRACT(EPOCH FROM (NOW() - updated_at)) <= %s)             AS fresh_count,
    COUNT(*) FILTER (WHERE metric = %s
        AND EXTRACT(EPOCH FROM (NOW() - updated_at)) > %s)              AS stale_count,
    COUNT(*) FILTER (WHERE metric = %s AND sample_count < %s)  AS rules_waiting,
    MAX(updated_at)                                                     AS latest_update
FROM sentinel_baselines
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
        # entity_ids that crossed MIN_SAMPLES (no re-notification on restart).
        self._established: set[str] = set()
        # In-memory Welford state for DOW slots: key = "entity_id:dow:hour",
        # value = (mean, m2, count).  m2 is the sum of squared deviations
        # (Welford's accumulator for variance).  Populated on async_initialize()
        # from DB values and updated on every _update_baselines() call.
        self._dow_state: dict[str, tuple[float, float, int]] = {}

    # ---------------------------------------------------------------------- #
    # Lifecycle
    # ---------------------------------------------------------------------- #

    async def async_initialize(self) -> None:  # noqa: PLR0912
        """Create/migrate the ``sentinel_baselines`` table and seed state."""
        try:
            async with self._pool.connection() as conn, conn.cursor() as cur:
                await cur.execute(_CREATE_TABLE_SQL)
                # Idempotent migrations for columns added after initial release.
                await cur.execute(_MIGRATE_REFERENCE_VALUE_SQL)
                await cur.execute(_MIGRATE_REFERENCE_AT_SQL)
                # DOW index — fast per-(entity, metric) lookups.
                await cur.execute(_CREATE_DOW_INDEX_SQL)
                await conn.commit()
                # Seed the established set so restarts don't re-fire notifications.
                min_samples = _coerce_int(
                    self._options.get(CONF_SENTINEL_BASELINE_MIN_SAMPLES),
                    default=RECOMMENDED_SENTINEL_BASELINE_MIN_SAMPLES,
                )
                await cur.execute(
                    _FETCH_ESTABLISHED_SQL, (METRIC_ROLLING_AVG, min_samples)
                )
                rows = await cur.fetchall()
                for row in rows:
                    eid = row.get("entity_id") if isinstance(row, dict) else row[0]
                    if eid:
                        self._established.add(str(eid))
                # Warm-restart Welford _dow_state from existing DB rows so that
                # historical variance is preserved across HA restarts.
                await cur.execute(_FETCH_DOW_WARMUP_SQL)
                dow_rows = await cur.fetchall()
                mean_rows: dict[str, dict[str, tuple[float, int]]] = {}
                std_rows: dict[str, dict[str, float]] = {}
                for row in dow_rows:
                    if isinstance(row, dict):
                        eid = str(row.get("entity_id", "") or "")
                        metric = str(row.get("metric", "") or "")
                        val = row.get("value")
                        cnt = row.get("sample_count", 0)
                    else:
                        eid, metric, val, cnt = (
                            str(row[0] or ""),
                            str(row[1] or ""),
                            row[2],
                            row[3],
                        )
                    if not eid or not metric or val is None:
                        continue
                    if metric.startswith(
                        METRIC_DOW_STD_PREFIX
                    ) and not metric.startswith(METRIC_DOW_AVG_PREFIX):
                        std_rows.setdefault(eid, {})[metric] = float(val)
                    elif metric.startswith(METRIC_DOW_AVG_PREFIX):
                        # hourly_avg_{dow}_{hour} rows have two numeric suffixes
                        parts = metric[len(METRIC_DOW_AVG_PREFIX) :].split("_")
                        if (
                            len(parts) == _DOW_METRIC_PARTS
                        ):  # DOW (not global hourly_avg_{H})
                            mean_rows.setdefault(eid, {})[metric] = (
                                float(val),
                                int(cnt or 0),
                            )
                # Reconstruct _dow_state: key = "entity_id:dow:hour"
                for eid, metrics in mean_rows.items():
                    for metric, (mean_val, count) in metrics.items():
                        parts = metric[len(METRIC_DOW_AVG_PREFIX) :].split("_")
                        if len(parts) != _DOW_METRIC_PARTS:
                            continue
                        dow_str, hour_str = parts
                        std_metric = f"{METRIC_DOW_STD_PREFIX}{dow_str}_{hour_str}"
                        stddev = (std_rows.get(eid) or {}).get(std_metric, 0.0)
                        m2 = stddev**2 * max(count - 1, 0)
                        key = f"{eid}:{dow_str}:{hour_str}"
                        self._dow_state[key] = (mean_val, m2, count)
            LOGGER.debug(
                "sentinel_baselines table ready; %d baselines established; "
                "%d DOW slots warmed.",
                len(self._established),
                len(self._dow_state),
            )
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

    async def _update_baselines(self, snapshot: FullStateSnapshot) -> None:  # noqa: PLR0912, PLR0915
        """Upsert rolling stats for all numeric entities in *snapshot*."""
        now = dt_util.utcnow()
        # Use local time for DOW bucket assignment so that weekday patterns align with
        # the user's actual schedule.  UTC-based bucketing would produce cross-timezone
        # artefacts (e.g. Saturday night classified as Sunday morning).
        local_now = dt_util.now()
        hour_str = str(now.hour)

        min_samples = _coerce_int(
            self._options.get(CONF_SENTINEL_BASELINE_MIN_SAMPLES),
            default=RECOMMENDED_SENTINEL_BASELINE_MIN_SAMPLES,
        )
        max_samples = max(
            1,
            _coerce_int(
                self._options.get(CONF_SENTINEL_BASELINE_MAX_SAMPLES),
                default=RECOMMENDED_SENTINEL_BASELINE_MAX_SAMPLES,
            ),
        )
        drift_threshold_pct = _coerce_float(
            self._options.get(CONF_SENTINEL_BASELINE_DRIFT_THRESHOLD_PCT),
            default=RECOMMENDED_SENTINEL_BASELINE_DRIFT_THRESHOLD_PCT,
        )
        weekly_patterns = bool(
            self._options.get(
                CONF_SENTINEL_BASELINE_WEEKLY_PATTERNS,
                RECOMMENDED_SENTINEL_BASELINE_WEEKLY_PATTERNS,
            )
        )

        # Collect numeric entities and their current values.
        entity_values: dict[str, float] = {}
        for entity in snapshot.get("entities", []):
            entity_id = entity.get("entity_id", "")
            if not entity_id:
                continue
            try:
                value = float(str(entity.get("state", "")))
            except (TypeError, ValueError):
                continue
            entity_values[entity_id] = value

        if not entity_values:
            return

        try:
            async with self._pool.connection() as conn, conn.cursor() as cur:
                # Batch-fetch reference values for drift detection (rolling_avg only).
                entity_ids_list = list(entity_values.keys())
                await cur.execute(
                    _FETCH_REFS_SQL, (METRIC_ROLLING_AVG, entity_ids_list)
                )
                ref_rows = await cur.fetchall()
                refs: dict[str, float | None] = {}
                for row in ref_rows:
                    if isinstance(row, dict):
                        eid = row.get("entity_id", "")
                        rv = row.get("reference_value")
                    else:
                        eid, rv = row[0], row[1]
                    if eid:
                        refs[str(eid)] = rv

                # Collect sample_count from RETURNING clause — eliminates N+1 SELECTs.
                rolling_sample_counts: dict[str, int] = {}

                for entity_id, value in entity_values.items():
                    # --- rolling_avg upsert ---
                    ref_val = refs.get(entity_id)
                    drift_detected = False
                    if ref_val is not None:
                        if ref_val == 0.0:
                            drift_pct = 100.0 if value != 0.0 else 0.0
                        else:
                            drift_pct = abs(value - ref_val) / abs(ref_val) * 100.0
                        drift_detected = drift_pct >= drift_threshold_pct

                    if drift_detected:
                        await cur.execute(
                            _UPSERT_WITH_REF_SQL,
                            (
                                entity_id,
                                METRIC_ROLLING_AVG,
                                "rolling",
                                value,
                                now,
                                value,  # new reference_value
                                now,  # new reference_at
                                max_samples,
                                max_samples,
                                max_samples,
                            ),
                        )
                    else:
                        await cur.execute(
                            _UPSERT_SQL,
                            (
                                entity_id,
                                METRIC_ROLLING_AVG,
                                "rolling",
                                value,
                                now,
                                max_samples,
                                max_samples,
                                max_samples,
                            ),
                        )
                    row = await cur.fetchone()
                    if row is not None:
                        sc = (
                            row.get("sample_count") if isinstance(row, dict) else row[0]
                        )
                        if sc is not None:
                            rolling_sample_counts[entity_id] = int(sc)

                    # --- hourly_avg upsert ---
                    hourly_metric = f"{METRIC_HOURLY_PREFIX}{hour_str}"
                    await cur.execute(
                        _UPSERT_SQL,
                        (
                            entity_id,
                            hourly_metric,
                            "hourly",
                            value,
                            now,
                            max_samples,
                            max_samples,
                            max_samples,
                        ),
                    )
                    # Discard hourly RETURNING row (not used for establishment).
                    await cur.fetchone()

                    # --- DOW (day-of-week) Welford upsert (when enabled) ---
                    if weekly_patterns:
                        dow = local_now.weekday()  # 0=Monday, 6=Sunday
                        local_hour = local_now.hour
                        dow_mean, dow_stddev = self._update_dow_welford(
                            entity_id, dow, local_hour, value
                        )
                        dow_mean_metric = f"{METRIC_DOW_AVG_PREFIX}{dow}_{local_hour}"
                        dow_std_metric = f"{METRIC_DOW_STD_PREFIX}{dow}_{local_hour}"
                        _, _, dow_count = self._dow_state[
                            f"{entity_id}:{dow}:{local_hour}"
                        ]
                        await cur.execute(
                            _DOW_UPSERT_SQL,
                            (entity_id, dow_mean_metric, dow_mean, dow_count, now),
                        )
                        await cur.execute(
                            _DOW_UPSERT_SQL,
                            (entity_id, dow_std_metric, dow_stddev, dow_count, now),
                        )

                await conn.commit()

                # Check establishment crossings using RETURNING sample_count values.
                for entity_id, count in rolling_sample_counts.items():
                    if entity_id in self._established:
                        continue
                    if count >= min_samples:
                        self._established.add(entity_id)
                        await self._fire_establishment_notification(
                            entity_id, entity_values[entity_id], snapshot
                        )

            LOGGER.debug(
                "Baseline upsert completed for %d entities.", len(entity_values)
            )

            # Fire drift notifications after commit (outside the conn context).
            # Only fire for established entities — skip entities still in build-up.
            for entity_id, value in entity_values.items():
                if entity_id not in self._established:
                    continue
                ref_val = refs.get(entity_id)
                if ref_val is None:
                    continue
                if ref_val == 0.0:
                    drift_pct = 100.0 if value != 0.0 else 0.0
                else:
                    drift_pct = abs(value - ref_val) / abs(ref_val) * 100.0
                if drift_pct >= drift_threshold_pct:
                    await self._fire_drift_notification(
                        entity_id, value, ref_val, drift_pct, snapshot
                    )

        except Exception:
            LOGGER.exception("Baseline DB write failed.")

    async def _fire_establishment_notification(
        self,
        entity_id: str,
        value: float,
        snapshot: FullStateSnapshot,
    ) -> None:
        """Fire a persistent HA notification when a baseline is established."""
        # Attempt to find a friendly name from the snapshot entities.
        friendly_name = entity_id
        for entity in snapshot.get("entities", []):
            if entity.get("entity_id") == entity_id:
                friendly_name = entity.get("friendly_name") or entity_id
                break
        message = (
            f"Baseline established for {friendly_name}: normal \u2248 {value:.2f}."
        )
        LOGGER.info("Baseline established for %s (value=%.2f).", entity_id, value)
        try:
            await self._hass.services.async_call(
                "persistent_notification",
                "create",
                {
                    "title": "Sentinel Baseline Established",
                    "message": message,
                    "notification_id": f"hga_baseline_established_{entity_id}",
                },
                blocking=False,
            )
        except Exception:  # noqa: BLE001
            LOGGER.debug("Could not fire establishment notification for %s.", entity_id)

    async def _fire_drift_notification(
        self,
        entity_id: str,
        current_value: float,
        reference_value: float,
        drift_pct: float,
        snapshot: FullStateSnapshot,
    ) -> None:
        """Fire a persistent HA notification when an entity's baseline drifts."""
        friendly_name = entity_id
        for entity in snapshot.get("entities", []):
            if entity.get("entity_id") == entity_id:
                friendly_name = entity.get("friendly_name") or entity_id
                break
        message = (
            f"Baseline drift detected for {friendly_name}: "
            f"was {reference_value:.2f}, now {current_value:.2f} "
            f"({drift_pct:.1f}% change). Reference updated."
        )
        LOGGER.info(
            "Baseline drift for %s: %.2f -> %.2f (%.1f%%).",
            entity_id,
            reference_value,
            current_value,
            drift_pct,
        )
        try:
            await self._hass.services.async_call(
                "persistent_notification",
                "create",
                {
                    "title": "Sentinel Baseline Drift Detected",
                    "message": message,
                    "notification_id": f"hga_baseline_drift_{entity_id}",
                },
                blocking=False,
            )
        except Exception:  # noqa: BLE001
            LOGGER.debug("Could not fire drift notification for %s.", entity_id)

    # ---------------------------------------------------------------------- #
    # DOW Welford helpers
    # ---------------------------------------------------------------------- #

    def _update_dow_welford(
        self,
        entity_id: str,
        dow: int,
        hour: int,
        new_value: float,
    ) -> tuple[float, float]:
        """
        Update Welford's online mean/variance for the given (entity, DOW, hour) slot.

        Returns ``(mean, stddev)`` after incorporating *new_value*.

        Welford's algorithm accumulates mean and M2 (sum of squared deviations)
        without storing all past values.  This is correct for sparse slots that
        update at most once per week; EMA would produce interpolation artefacts.

        The in-memory state ``_dow_state`` is the only source of M2; DB stores
        only mean, stddev, and count.  On restart, M2 is reconstructed from
        ``stddev**2 * (count - 1)`` in ``async_initialize()``.
        """
        key = f"{entity_id}:{dow}:{hour}"
        mean, m2, count = self._dow_state.get(key, (0.0, 0.0, 0))
        count += 1
        delta = new_value - mean
        mean = mean + delta / count
        delta2 = new_value - mean
        m2 = m2 + delta * delta2
        stddev = (m2 / (count - 1)) ** 0.5 if count > 1 else 0.0
        self._dow_state[key] = (mean, m2, count)
        return mean, stddev

    async def async_fetch_dow_data(
        self,
    ) -> dict[str, dict[str, tuple[float, int]]]:
        """
        Return DOW mean/stddev values and sample counts for all entities.

        Returns ``{entity_id: {metric_key: (value, count)}}``.
        No min-samples filter — the blend-weight formula handles sparse slots
        (``w = min(count / dow_min_samples, 1.0)``; w=0 ⟹ pure global mean).
        Returns an empty dict on any DB error.
        """
        try:
            async with self._pool.connection() as conn, conn.cursor() as cur:
                await cur.execute(_FETCH_DOW_SQL)
                rows = await cur.fetchall()
        except Exception:  # noqa: BLE001
            LOGGER.debug("DOW baseline fetch failed; returning empty dict.")
            return {}

        result: dict[str, dict[str, tuple[float, int]]] = {}
        for row in rows:
            if isinstance(row, dict):
                entity_id = str(row.get("entity_id", "") or "")
                metric = str(row.get("metric", "") or "")
                value = row.get("value")
                count = row.get("sample_count", 0)
            else:
                entity_id, metric, value, count = (
                    str(row[0] or ""),
                    str(row[1] or ""),
                    row[2],
                    row[3],
                )
            if not entity_id or not metric or value is None:
                continue
            result.setdefault(entity_id, {})[metric] = (float(value), int(count or 0))

        LOGGER.debug("Fetched DOW baseline data for %d entities.", len(result))
        return result

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

    async def async_fetch_baselines(
        self, min_samples: int | None = None
    ) -> dict[str, dict[str, float]]:
        """
        Return baseline values for entities above the min-sample threshold.

        ``{entity_id: {metric: value}}`` — used by the sentinel engine every
        detection cycle.  Entities with fewer than *min_samples* rows are
        excluded so rules don't fire on statistically insufficient data.
        Returns an empty dict on any DB error.
        """
        if min_samples is None:
            min_samples = _coerce_int(
                self._options.get(CONF_SENTINEL_BASELINE_MIN_SAMPLES),
                default=RECOMMENDED_SENTINEL_BASELINE_MIN_SAMPLES,
            )
        try:
            async with self._pool.connection() as conn, conn.cursor() as cur:
                await cur.execute(_FETCH_ALL_SQL, (min_samples,))
                rows = await cur.fetchall()
        except Exception:  # noqa: BLE001
            LOGGER.debug("Baseline fetch failed; returning empty baselines.")
            return {}

        result: dict[str, dict[str, float]] = {}
        for row in rows:
            if isinstance(row, dict):
                entity_id = row.get("entity_id", "")
                metric = row.get("metric", "")
                value = row.get("value")
            else:
                entity_id, metric, value = row[0], row[1], row[2]
            if not entity_id or not metric or value is None:
                continue
            result.setdefault(entity_id, {})[metric] = float(value)

        LOGGER.debug(
            "Fetched baselines for %d entities (min_samples=%d).",
            len(result),
            min_samples,
        )
        return result

    async def async_fetch_full_baselines(self) -> dict[str, dict[str, Any]] | None:
        """
        Return rich baseline data for all entities.

        ``{entity_id: {metric: {value, sample_count, updated_at, freshness}}}``
        Used by the ``sentinel_get_baselines`` service.
        Returns ``None`` on any DB error so callers can distinguish "no baselines
        yet" from "backend broken".
        """
        freshness_threshold = _coerce_int(
            self._options.get(CONF_SENTINEL_BASELINE_FRESHNESS_THRESHOLD_SECONDS),
            default=RECOMMENDED_SENTINEL_BASELINE_FRESHNESS_THRESHOLD_SECONDS,
        )
        now = dt_util.utcnow()
        try:
            async with self._pool.connection() as conn, conn.cursor() as cur:
                await cur.execute(_FETCH_FULL_SQL)
                rows = await cur.fetchall()
        except Exception:  # noqa: BLE001
            LOGGER.warning("Full baseline fetch failed.", exc_info=True)
            return None

        result: dict[str, dict[str, Any]] = {}
        for row in rows:
            if isinstance(row, dict):
                entity_id = row.get("entity_id", "")
                metric = row.get("metric", "")
                value = row.get("value")
                sample_count = row.get("sample_count", 0)
                updated_at = row.get("updated_at")
            else:
                entity_id, metric, value, sample_count, updated_at = (
                    row[0],
                    row[1],
                    row[2],
                    row[3],
                    row[4],
                )
            if not entity_id or not metric or value is None:
                continue
            freshness = BASELINE_UNAVAILABLE
            if updated_at is not None:
                try:
                    age = (now - updated_at).total_seconds()
                    freshness = (
                        BASELINE_FRESH if age <= freshness_threshold else BASELINE_STALE
                    )
                except (TypeError, AttributeError):
                    freshness = BASELINE_STALE
            result.setdefault(entity_id, {})[metric] = {
                "value": float(value),
                "sample_count": int(sample_count or 0),
                "updated_at": updated_at.isoformat()
                if updated_at is not None
                else None,
                "freshness": freshness,
            }

        return result

    async def async_fetch_baseline_stats(self) -> dict[str, Any] | None:
        """
        Return aggregate baseline statistics for health monitoring.

        Runs a single SQL aggregate query instead of a full table scan.
        Returns a dict with keys: entity_count, fresh_count, stale_count,
        rules_waiting, latest_update.  Returns ``None`` on any DB error so
        callers can avoid caching zeros that misrepresent the system state.
        """
        freshness_threshold = _coerce_int(
            self._options.get(CONF_SENTINEL_BASELINE_FRESHNESS_THRESHOLD_SECONDS),
            default=RECOMMENDED_SENTINEL_BASELINE_FRESHNESS_THRESHOLD_SECONDS,
        )
        min_samples = _coerce_int(
            self._options.get(CONF_SENTINEL_BASELINE_MIN_SAMPLES),
            default=RECOMMENDED_SENTINEL_BASELINE_MIN_SAMPLES,
        )
        try:
            async with self._pool.connection() as conn, conn.cursor() as cur:
                await cur.execute(
                    _FETCH_STATS_SQL,
                    (
                        METRIC_ROLLING_AVG,
                        freshness_threshold,
                        METRIC_ROLLING_AVG,
                        freshness_threshold,
                        METRIC_ROLLING_AVG,
                        min_samples,
                    ),
                )
                row = await cur.fetchone()
        except Exception:  # noqa: BLE001
            LOGGER.warning("Baseline stats fetch failed.", exc_info=True)
            return None
        if row is None:
            return {}
        if isinstance(row, dict):
            return {
                "entity_count": int(row.get("entity_count") or 0),
                "fresh_count": int(row.get("fresh_count") or 0),
                "stale_count": int(row.get("stale_count") or 0),
                "rules_waiting": int(row.get("rules_waiting") or 0),
                "latest_update": row.get("latest_update"),
            }
        return {
            "entity_count": int(row[0] or 0),
            "fresh_count": int(row[1] or 0),
            "stale_count": int(row[2] or 0),
            "rules_waiting": int(row[3] or 0),
            "latest_update": row[4],
        }

    async def async_reset_baseline(self, entity_id: str | None) -> int:
        """
        Delete baseline rows for *entity_id* (or all rows if ``None``).

        Returns the number of rows deleted.  The entity_id must already be
        validated by the caller (``domain.object_id`` format or ``None``).
        """
        try:
            async with self._pool.connection() as conn, conn.cursor() as cur:
                if entity_id is not None:
                    await cur.execute(_DELETE_ENTITY_SQL, (entity_id,))
                else:
                    await cur.execute(_DELETE_ALL_SQL)
                deleted = cur.rowcount if cur.rowcount is not None else 0
                await conn.commit()
                # Update in-memory state only after the DB commit succeeds.
                if entity_id is not None:
                    self._established.discard(entity_id)
                else:
                    self._established.clear()
        except Exception:
            LOGGER.exception("Baseline reset failed.")
            return -1
        LOGGER.info(
            "Baseline reset: deleted %d rows (entity_id=%s).", deleted, entity_id
        )
        return int(deleted)

    async def async_fetch_ready_entity_ids(self) -> list[str]:
        """
        Return entity_ids whose rolling_avg has crossed min_samples.

        Used by the sentinel engine to inject ``baseline_ready_entities`` into
        the discovery snapshot so the LLM can propose baseline rules.
        """
        min_samples = _coerce_int(
            self._options.get(CONF_SENTINEL_BASELINE_MIN_SAMPLES),
            default=RECOMMENDED_SENTINEL_BASELINE_MIN_SAMPLES,
        )
        try:
            async with self._pool.connection() as conn, conn.cursor() as cur:
                await cur.execute(
                    _FETCH_ESTABLISHED_SQL, (METRIC_ROLLING_AVG, min_samples)
                )
                rows = await cur.fetchall()
        except Exception:  # noqa: BLE001
            LOGGER.debug("Could not fetch baseline-ready entity ids.")
            return []
        result = []
        for row in rows:
            eid = row.get("entity_id") if isinstance(row, dict) else row[0]
            if eid:
                result.append(str(eid))
        return result


# ---------------------------------------------------------------------------
# Temporal deviation detector
# ---------------------------------------------------------------------------


def evaluate_baseline_deviation(  # noqa: PLR0911, PLR0912, PLR0915
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

    # Determine power-class and appliance-class early; both are used for the
    # standby guard and for completion detection below.
    _attrs: dict[str, Any] = (entity.get("attributes") or {}) if entity else {}
    _unit = str(_attrs.get("unit_of_measurement") or "")
    is_power_entity = str(_attrs.get("device_class") or "") == "power" or _unit in {
        "W",
        "kW",
    }
    friendly_name = str(entity.get("friendly_name") or "").lower()
    is_appliance = is_power_entity and any(
        hint in entity_id.lower() or hint in friendly_name for hint in _APPLIANCE_HINTS
    )

    # Suppress standby-noise false positives for *appliance* circuits only.
    # A 2.9 W → 0 W swing on a washer sensor is pure noise; a 40 W UPS or
    # 30 W switch going dark is a real event worth notifying on.
    # Normalize to watts first: kW entities report values like 30.0 (kW).
    _deviation_w = abs(current_value - baseline_value) * (
        1000.0 if _unit == "kW" else 1.0
    )
    if is_appliance and _deviation_w < MINIMUM_POWER_DEVIATION_WATTS:
        return []

    rule_id = str(rule.get("rule_id") or "baseline_deviation")
    evidence = {
        "rule_id": rule_id,
        "template_id": rule.get("template_id", "baseline_deviation"),
        "entity_id": entity_id,
        "friendly_name": entity.get("friendly_name") or "",
        "current_value": current_value,
        "baseline_value": baseline_value,
        "deviation_pct": round(deviation_pct, 2),
        "deviation_direction": "above" if current_value > baseline_value else "below",
        "threshold_pct": threshold_pct,
        "metric": metric,
        "last_changed": entity.get("last_changed"),
    }

    # Appliance completion detection: power-class entity whose name identifies
    # a dedicated appliance circuit (washer, dryer, dishwasher, etc.).
    # Generic power sensors (UPS, fridge, whole-home, server rack) are
    # intentionally excluded to avoid silencing real faults.
    # (is_appliance already computed above for the standby guard)
    if (
        is_appliance
        and baseline_value >= COMPLETION_MIN_ACTIVE_WATTS
        and current_value < COMPLETION_THRESHOLD_PCT * baseline_value
    ):
        # Only treat as completion if the state changed recently.  If the
        # appliance has been idle for longer than COMPLETION_RECENCY_SECS the
        # low reading is just its resting state — not a fresh cycle end — and
        # the rolling average baseline hasn't converged yet.
        last_changed_raw = entity.get("last_changed")
        recently_changed = False
        if last_changed_raw:
            try:
                last_changed_dt = dt_util.parse_datetime(str(last_changed_raw))
                if last_changed_dt is not None:
                    now_utc = dt_util.utcnow()
                    recently_changed = (now_utc - last_changed_dt) <= timedelta(
                        seconds=COMPLETION_RECENCY_SECS
                    )
            except (ValueError, TypeError):
                pass
        if recently_changed:
            evidence["is_completion"] = True

    severity_raw = str(rule.get("severity") or "low")
    severity: Severity = (  # type: ignore[assignment]
        severity_raw if severity_raw in {"low", "medium", "high"} else "low"
    )
    # Override severity to low for appliance cycle completions.
    if evidence.get("is_completion"):
        severity = "low"
    confidence = _coerce_float(rule.get("confidence"), default=0.7)
    suggested_actions = rule.get("suggested_actions") or []
    # Appliance finished its cycle normally — no user action required.
    if evidence.get("is_completion"):
        suggested_actions = []
    # Exclude display-only fields from the hash so anomaly_id is stable even
    # when friendly_name is temporarily unavailable (returns "").
    _hash_evidence = {k: v for k, v in evidence.items() if k != "friendly_name"}
    anomaly_id = build_anomaly_id(rule_id, [entity_id], _hash_evidence)

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


def evaluate_time_of_day_anomaly(  # noqa: PLR0913
    snapshot: FullStateSnapshot,
    rule: dict[str, Any],
    baselines: dict[str, dict[str, float]],
    dow_data: dict[str, dict[str, tuple[float, int]]] | None = None,
    dow_min_samples: int = RECOMMENDED_SENTINEL_BASELINE_DOW_MIN_SAMPLES,
    global_drift_pct: float = RECOMMENDED_SENTINEL_BASELINE_DRIFT_THRESHOLD_PCT,
) -> list[AnomalyFinding]:
    """
    Detect when an entity's state differs from the expected hour-of-day average.

    When *dow_data* is provided and the entity has DOW baseline data, the
    expected value is a weighted blend of the DOW-hour mean and the global
    hourly mean:

        w = min(count / dow_min_samples, 1.0)
        expected = w * dow_mean + (1 - w) * global_mean

    At ``count = 0``: pure global.  At ``count >= dow_min_samples``: pure DOW.

    The anomaly threshold is entity-specific when a DOW stddev is available:

        threshold = max(dow_std * 2.0, abs(expected) * drift_pct / 100)

    Without DOW data the function falls back to the existing global-hourly
    behaviour (delegates to ``evaluate_baseline_deviation``).
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
        dow = now_dt.weekday()  # 0=Monday, 6=Sunday
    except (TypeError, ValueError):
        return []

    # --- DOW-blended path ---
    if dow_data is not None:
        entity_dow = dow_data.get(entity_id, {})
        dow_mean_key = f"{METRIC_DOW_AVG_PREFIX}{dow}_{now_dt.hour}"
        dow_std_key = f"{METRIC_DOW_STD_PREFIX}{dow}_{now_dt.hour}"

        dow_mean_entry = entity_dow.get(dow_mean_key)
        dow_std_entry = entity_dow.get(dow_std_key)

        if dow_mean_entry is not None:
            return _evaluate_dow_anomaly(
                snapshot=snapshot,
                rule=rule,
                baselines=baselines,
                entity_id=entity_id,
                hour_str=hour_str,
                dow=dow,
                local_hour=now_dt.hour,
                dow_mean=dow_mean_entry[0],
                dow_count=dow_mean_entry[1],
                dow_std=dow_std_entry[0] if dow_std_entry is not None else None,
                dow_min_samples=dow_min_samples,
                global_drift_pct=global_drift_pct,
            )

    # --- Global-hourly fallback (no DOW data) ---
    hourly_rule = dict(rule)
    hourly_params = dict(params)
    hourly_params["metric"] = f"{METRIC_HOURLY_PREFIX}{hour_str}"
    hourly_rule["params"] = hourly_params
    if not hourly_rule.get("rule_id"):
        hourly_rule["rule_id"] = "time_of_day_anomaly"
    if not hourly_rule.get("template_id"):
        hourly_rule["template_id"] = "time_of_day_anomaly"

    return evaluate_baseline_deviation(snapshot, hourly_rule, baselines)


def _evaluate_dow_anomaly(  # noqa: PLR0913
    snapshot: FullStateSnapshot,
    rule: dict[str, Any],
    baselines: dict[str, dict[str, float]],
    entity_id: str,
    hour_str: str,
    dow: int,
    local_hour: int,
    dow_mean: float,
    dow_count: int,
    dow_std: float | None,
    dow_min_samples: int,
    global_drift_pct: float,
) -> list[AnomalyFinding]:
    """
    Compute a DOW-blended time-of-day anomaly finding.

    Called only when a DOW mean row exists for the given entity/DOW/hour.
    The blend weight is proportional to DOW observation count, so the model
    transitions smoothly from global-hourly to DOW-specific baselines as data
    accumulates.  No hard cliff at dow_min_samples.
    """
    entity_map = {e["entity_id"]: e for e in snapshot.get("entities", [])}
    entity = entity_map.get(entity_id)
    if entity is None:
        return []

    try:
        current_value = float(str(entity.get("state", "")))
    except (TypeError, ValueError):
        return []

    # Global hourly mean for this hour (the fallback anchor).
    global_mean = (baselines.get(entity_id) or {}).get(
        f"{METRIC_HOURLY_PREFIX}{hour_str}"
    )
    if global_mean is None:
        global_mean = dow_mean  # no global baseline yet — use DOW mean as anchor

    # Weighted blend: w=0 ⟹ pure global; w=1 ⟹ pure DOW.
    w = min(dow_count / dow_min_samples, 1.0) if dow_min_samples > 0 else 1.0
    expected = w * dow_mean + (1.0 - w) * global_mean

    # Entity-specific threshold: use DOW stddev when the slot is warm enough.
    if dow_std is not None and dow_count >= dow_min_samples and dow_std > 0.0:
        threshold = max(dow_std * 2.0, abs(expected) * (global_drift_pct / 100.0))
    else:
        threshold = abs(expected) * (global_drift_pct / 100.0)

    deviation = abs(current_value - expected)
    if deviation <= threshold:
        return []

    rule_id = str(rule.get("rule_id") or "time_of_day_anomaly")
    deviation_pct = deviation / abs(expected) * 100.0 if expected != 0.0 else 100.0
    evidence = {
        "rule_id": rule_id,
        "template_id": rule.get("template_id", "time_of_day_anomaly"),
        "entity_id": entity_id,
        "friendly_name": entity.get("friendly_name") or "",
        "current_value": current_value,
        "expected_value": round(expected, 4),
        "dow_mean": dow_mean,
        "global_mean": global_mean,
        "blend_weight": round(w, 4),
        "dow_count": dow_count,
        "dow_stddev": round(dow_std, 4) if dow_std is not None else None,
        "deviation": round(deviation, 4),
        "deviation_pct": round(deviation_pct, 2),
        "deviation_direction": "above" if current_value > expected else "below",
        "threshold": round(threshold, 4),
        "dow": dow,
        "hour": local_hour,
        "last_changed": entity.get("last_changed"),
    }

    severity_raw = str(rule.get("severity") or "low")
    severity: Severity = (  # type: ignore[assignment]
        severity_raw if severity_raw in {"low", "medium", "high"} else "low"
    )
    confidence = _coerce_float(rule.get("confidence"), default=0.7)
    _hash_evidence = {k: v for k, v in evidence.items() if k != "friendly_name"}
    anomaly_id = build_anomaly_id(rule_id, [entity_id], _hash_evidence)
    suggested_actions = rule.get("suggested_actions") or []

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

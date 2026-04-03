"""Sentinel operational health sensor."""

from __future__ import annotations

import logging
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, Any

from homeassistant.components.sensor import SensorEntity, SensorEntityDescription
from homeassistant.core import HomeAssistant, callback
from homeassistant.helpers.dispatcher import async_dispatcher_connect
from homeassistant.util import dt as dt_util

from ..const import (  # noqa: TID252
    CONF_SENTINEL_ENABLED,
    RECOMMENDED_SENTINEL_ENABLED,
    SIGNAL_SENTINEL_RUN_COMPLETE,
)

if TYPE_CHECKING:
    from custom_components.home_generative_agent.audit.store import AuditStore
    from custom_components.home_generative_agent.sentinel.baseline import (
        SentinelBaselineUpdater,
    )
    from custom_components.home_generative_agent.sentinel.discovery_engine import (
        SentinelDiscoveryEngine,
    )
    from custom_components.home_generative_agent.sentinel.engine import SentinelEngine

LOGGER = logging.getLogger(__name__)

_DESC = SensorEntityDescription(
    key="sentinel_health",
    icon="mdi:shield-check",
)

# Maximum audit records to fetch per health-sensor refresh.  Using a value well
# above any realistic hot-store size ensures all records contribute to KPI accuracy.
_MAX_AUDIT_RECORDS = 1000

# action_outcome statuses that represent a completed autonomous-execute attempt.
_AUTO_EXEC_TERMINAL: frozenset[str] = frozenset(
    {"success", "partial", "error", "no_actions"}
)

# action_outcome statuses written by ActionHandler for user-triggered executions.
# "agent_called" = conversation agent invoked; "event_fired" = blueprint hook fired.
# Both are treated as successful for action_success_rate purposes.
_USER_EXEC_SUCCESS: frozenset[str] = frozenset({"agent_called", "event_fired"})
_USER_EXEC_FAILURE: frozenset[str] = frozenset({"blocked", "missing_finding"})

# Known trigger source keys exposed in the 24-hour breakdown attribute.
_TRIGGER_SOURCE_KEYS: tuple[str, ...] = ("poll", "event", "on_demand")


def _compute_kpis(records: list[dict[str, Any]]) -> dict[str, Any]:  # noqa: PLR0912, PLR0915
    """Compute Sentinel KPIs from a list of raw audit record dicts."""
    cutoff_14d = datetime.now(UTC) - timedelta(days=14)

    severity_counts: dict[str, int] = {"low": 0, "medium": 0, "high": 0}
    trigger_source_counts: dict[str, int] = {}
    triage_total = 0
    triage_suppress = 0
    auto_exec_count = 0
    auto_exec_failure_count = 0
    action_success_count = 0
    action_total = 0
    user_response_count = 0
    fp_14d_total = 0
    fp_14d_count = 0

    for r in records:
        finding = r.get("finding", {})
        sev = finding.get("severity")
        if sev in severity_counts:
            severity_counts[sev] += 1

        trigger = r.get("trigger_source")
        if trigger:
            trigger_source_counts[trigger] = trigger_source_counts.get(trigger, 0) + 1

        triage_decision = r.get("triage_decision")
        if triage_decision is not None:
            triage_total += 1
            if triage_decision == "suppress":
                triage_suppress += 1

        action_outcome = r.get("action_outcome")
        if action_outcome is not None:
            status = action_outcome.get("status", "")
            if status in _AUTO_EXEC_TERMINAL:
                auto_exec_count += 1
                action_total += 1
                if status in {"success", "partial"}:
                    action_success_count += 1
                elif status == "error":
                    auto_exec_failure_count += 1
            elif status in _USER_EXEC_SUCCESS:
                action_total += 1
                action_success_count += 1
            elif status in _USER_EXEC_FAILURE:
                action_total += 1

        # Only count user-facing metrics for findings that were actually notified.
        # Suppressed findings never reach the user and cannot receive responses.
        notified = r.get("suppression_reason_code") == "not_suppressed"

        user_response = r.get("user_response")
        if notified and user_response is not None:
            user_response_count += 1

        if notified:
            notified_at_str = r.get("notification", {}).get("notified_at")
            if notified_at_str:
                notified_dt = dt_util.parse_datetime(str(notified_at_str))
                if notified_dt is not None and notified_dt >= cutoff_14d:
                    fp_14d_total += 1
                    if user_response and user_response.get("false_positive"):
                        fp_14d_count += 1

    notified_total = sum(
        1 for r in records if r.get("suppression_reason_code") == "not_suppressed"
    )
    return {
        "findings_count_by_severity": severity_counts,
        "trigger_source_stats": trigger_source_counts,
        "triage_suppress_rate": (
            round(triage_suppress / triage_total * 100, 1) if triage_total > 0 else None
        ),
        "auto_exec_count": auto_exec_count,
        "auto_exec_failures": auto_exec_failure_count,
        "action_success_rate": (
            round(action_success_count / action_total * 100, 1)
            if action_total > 0
            else None
        ),
        "user_override_rate": (
            round(user_response_count / notified_total * 100, 1)
            if notified_total > 0
            else None
        ),
        "false_positive_rate_14d": (
            round(fp_14d_count / fp_14d_total * 100, 1) if fp_14d_total > 0 else None
        ),
    }


def _compute_trigger_source_breakdown(
    records: list[dict[str, Any]],
) -> dict[str, int]:
    """
    Return trigger-source counts for findings notified in the last 24 hours.

    Only records whose ``notification.notified_at`` timestamp falls within the
    rolling 24-hour window are included.  Known trigger sources
    (``_TRIGGER_SOURCE_KEYS``) are always present (value = 0 when no matching
    records exist).  Unknown trigger sources are added dynamically, so the
    returned dict may contain extra keys.
    """
    cutoff_24h = datetime.now(UTC) - timedelta(hours=24)
    counts: dict[str, int] = dict.fromkeys(_TRIGGER_SOURCE_KEYS, 0)
    for r in records:
        notified_at_str = r.get("notification", {}).get("notified_at")
        if not notified_at_str:
            continue
        notified_dt = dt_util.parse_datetime(str(notified_at_str))
        if notified_dt is None:
            continue
        if notified_dt < cutoff_24h:
            continue
        trigger = r.get("trigger_source")
        if trigger:
            counts[trigger] = counts.get(trigger, 0) + 1
    return counts


class SentinelHealthSensor(SensorEntity):
    """Sentinel operational health sensor exposing KPIs as attributes."""

    entity_description: SensorEntityDescription = _DESC
    _attr_has_entity_name = True
    _attr_icon = "mdi:shield-check"

    def __init__(  # noqa: PLR0913
        self,
        hass: HomeAssistant,
        options: dict[str, Any],
        audit_store: AuditStore | None,
        sentinel: SentinelEngine | None,
        entry_id: str,
        baseline_updater: SentinelBaselineUpdater | None = None,
        discovery_engine: SentinelDiscoveryEngine | None = None,
    ) -> None:
        """Initialize the sentinel health sensor."""
        self.hass = hass
        self._options = options
        self._audit_store = audit_store
        self._sentinel = sentinel
        self._baseline_updater = baseline_updater
        self._discovery_engine = discovery_engine
        self._attr_name = "Sentinel Health"
        self._attr_unique_id = f"sentinel_health::{entry_id}"
        self._attr_native_value = "ok"
        self._attrs: dict[str, Any] = {}
        self._attr_extra_state_attributes = self._attrs
        self._baseline_stats_cache: dict[str, Any] = {}
        self._baseline_stats_cache_ts: datetime | None = None
        self._baseline_stats_ttl = timedelta(seconds=60)

    async def async_added_to_hass(self) -> None:
        """Subscribe to run-complete signal and perform an initial refresh."""

        @callback
        def _on_run_complete() -> None:
            self.hass.async_create_task(self._async_refresh())

        remove = async_dispatcher_connect(
            self.hass, SIGNAL_SENTINEL_RUN_COMPLETE, _on_run_complete
        )
        self.async_on_remove(remove)
        await self._async_refresh()

    async def _async_refresh(self) -> None:
        """Recompute KPIs from the audit store and update HA state."""
        enabled = bool(
            self._options.get(CONF_SENTINEL_ENABLED, RECOMMENDED_SENTINEL_ENABLED)
        )
        if not enabled:
            self._attr_native_value = "disabled"
            self._attrs.clear()
            self.async_write_ha_state()
            return

        records: list[dict[str, Any]] = []
        if self._audit_store is not None:
            records = await self._audit_store.async_get_latest(_MAX_AUDIT_RECORDS)

        kpis = _compute_kpis(records)
        self._attrs.update(kpis)

        # 24-hour trigger-source breakdown (separate from the all-time stats above).
        self._attrs["trigger_source_breakdown"] = (
            _compute_trigger_source_breakdown(records)
            if self._audit_store is not None
            else None
        )

        # Merge run-timing telemetry from the engine.
        run_stats: dict[str, Any] = (
            self._sentinel.run_stats if self._sentinel is not None else {}
        )
        for key in (
            "last_run_start",
            "last_run_end",
            "run_duration_ms",
            "active_rule_count",
        ):
            self._attrs[key] = run_stats.get(key)

        # Merge trigger scheduler statistics.
        scheduler_stats: dict[str, Any] = run_stats.get("scheduler", {})
        for key, val in scheduler_stats.items():
            self._attrs[key] = val

        # Discovery cycle stats from the most recent discovery run.
        if self._discovery_engine is not None:
            disc_stats = self._discovery_engine.discovery_cycle_stats
            for key, val in disc_stats.items():
                self._attrs[f"discovery_{key}"] = val
        else:
            for key in (
                "candidates_generated",
                "candidates_novel",
                "candidates_deduplicated",
                "proposals_promoted",
                "unsupported_ttl_expired",
            ):
                self._attrs[f"discovery_{key}"] = None

        # Baseline health statistics.
        await self._refresh_baseline_attrs()

        self._attr_native_value = "ok"
        self.async_write_ha_state()

    async def _refresh_baseline_attrs(self) -> None:
        """Populate baseline_* attributes from the baseline updater."""
        if self._baseline_updater is None:
            for key in (
                "baseline_entity_count",
                "baseline_fresh_count",
                "baseline_stale_count",
                "baseline_rules_waiting",
                "baseline_last_update",
            ):
                self._attrs[key] = None
            return

        # Use a 60-second TTL cache to avoid a DB round-trip on every HA state refresh.
        now = datetime.now(tz=UTC)
        if (
            self._baseline_stats_cache_ts is None
            or (now - self._baseline_stats_cache_ts) >= self._baseline_stats_ttl
        ):
            fetched = await self._baseline_updater.async_fetch_baseline_stats()
            # Only update the cache on success; a None return means DB error —
            # keep the previous cached values so the sensor doesn't report "0
            # baselines" when the backend is temporarily unavailable.
            if fetched is not None:
                self._baseline_stats_cache = fetched
            else:
                LOGGER.warning("Baseline stats unavailable; using cached values.")
            self._baseline_stats_cache_ts = now

        stats = self._baseline_stats_cache
        updated_at = stats.get("latest_update")
        self._attrs["baseline_entity_count"] = stats.get("entity_count", 0)
        self._attrs["baseline_fresh_count"] = stats.get("fresh_count", 0)
        self._attrs["baseline_stale_count"] = stats.get("stale_count", 0)
        self._attrs["baseline_rules_waiting"] = stats.get("rules_waiting", 0)
        self._attrs["baseline_last_update"] = (
            updated_at.isoformat()
            if updated_at is not None and hasattr(updated_at, "isoformat")
            else updated_at
        )

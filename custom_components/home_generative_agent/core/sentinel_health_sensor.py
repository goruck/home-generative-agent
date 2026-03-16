"""Sentinel operational health sensor."""

from __future__ import annotations

import logging
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, Any

from homeassistant.components.sensor import SensorEntity, SensorEntityDescription
from homeassistant.core import HomeAssistant, callback
from homeassistant.helpers.dispatcher import async_dispatcher_connect

from ..const import (  # noqa: TID252
    CONF_SENTINEL_ENABLED,
    RECOMMENDED_SENTINEL_ENABLED,
    SIGNAL_SENTINEL_RUN_COMPLETE,
)

if TYPE_CHECKING:
    from custom_components.home_generative_agent.audit.store import AuditStore
    from custom_components.home_generative_agent.sentinel.engine import SentinelEngine

LOGGER = logging.getLogger(__name__)

_DESC = SensorEntityDescription(
    key="sentinel_health",
    icon="mdi:shield-check",
)

# action_outcome statuses that represent a completed auto-execute attempt.
_AUTO_EXEC_TERMINAL: frozenset[str] = frozenset(
    {"success", "partial", "error", "no_actions"}
)


def _compute_kpis(records: list[dict[str, Any]]) -> dict[str, Any]:  # noqa: PLR0912
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

        user_response = r.get("user_response")
        if user_response is not None:
            user_response_count += 1

        notified_at_str = r.get("notification", {}).get("notified_at")
        if notified_at_str:
            try:
                notified_dt = datetime.fromisoformat(notified_at_str)
                if notified_dt >= cutoff_14d:
                    fp_14d_total += 1
                    if user_response and user_response.get("false_positive"):
                        fp_14d_count += 1
            except (ValueError, TypeError):
                pass

    total = len(records)
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
            round(user_response_count / total * 100, 1) if total > 0 else None
        ),
        "false_positive_rate_14d": (
            round(fp_14d_count / fp_14d_total * 100, 1) if fp_14d_total > 0 else None
        ),
    }


class SentinelHealthSensor(SensorEntity):
    """Sentinel operational health sensor exposing KPIs as attributes."""

    entity_description: SensorEntityDescription = _DESC
    _attr_has_entity_name = True
    _attr_icon = "mdi:shield-check"

    def __init__(
        self,
        hass: HomeAssistant,
        options: dict[str, Any],
        audit_store: AuditStore | None,
        sentinel: SentinelEngine | None,
        entry_id: str,
    ) -> None:
        """Initialize the sentinel health sensor."""
        self.hass = hass
        self._options = options
        self._audit_store = audit_store
        self._sentinel = sentinel
        self._attr_name = "Sentinel Health"
        self._attr_unique_id = f"sentinel_health::{entry_id}"
        self._attr_native_value = "ok"
        self._attrs: dict[str, Any] = {}
        self._attr_extra_state_attributes = self._attrs

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
            records = await self._audit_store.async_get_latest(200)

        kpis = _compute_kpis(records)
        self._attrs.update(kpis)

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

        self._attr_native_value = "ok"
        self.async_write_ha_state()

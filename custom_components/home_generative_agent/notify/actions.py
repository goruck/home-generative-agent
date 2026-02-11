"""Notification action handling for sentinel findings."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from homeassistant.util import dt as dt_util

if TYPE_CHECKING:
    from homeassistant.core import HomeAssistant

    from custom_components.home_generative_agent.audit.store import AuditStore
    from custom_components.home_generative_agent.sentinel.models import AnomalyFinding
from custom_components.home_generative_agent.sentinel.suppression import (
    SuppressionManager,
    resolve_prompt,
)

LOGGER = logging.getLogger(__name__)

ACTION_PREFIX = "hga_sentinel_"
ACTION_ID_PARTS = 2
EVENT_SENTINEL_EXECUTE_REQUESTED = "hga_sentinel_execute_requested"


class ActionHandler:
    """Handle notification actions for sentinel findings."""

    def __init__(
        self,
        hass: HomeAssistant,
        suppression: SuppressionManager,
        audit_store: AuditStore,
    ) -> None:
        """Initialize action handling dependencies."""
        self._hass = hass
        self._suppression = suppression
        self._audit_store = audit_store
        self._pending_findings: dict[str, AnomalyFinding] = {}

    def register_finding(self, finding: AnomalyFinding) -> None:
        """Register a finding for action callbacks."""
        self._pending_findings[finding.anomaly_id] = finding

    async def handle_action(self, action_id: str, payload: dict[str, Any]) -> None:
        """Handle a mobile app action."""
        if not action_id.startswith(ACTION_PREFIX):
            return
        parts = action_id.removeprefix(ACTION_PREFIX).split("_", 1)
        if len(parts) != ACTION_ID_PARTS:
            return
        action, anomaly_id = parts
        LOGGER.info("Handling sentinel action %s for %s.", action, anomaly_id)
        finding = self._pending_findings.get(anomaly_id)
        resolve_prompt(self._suppression.state, anomaly_id)
        await self._suppression.async_save()

        response = {
            "action": action,
            "payload": payload,
        }
        outcome: dict[str, Any] | None = None
        if action == "execute":
            if finding is None:
                outcome = {"status": "missing_finding"}
            elif finding.is_sensitive:
                outcome = {
                    "status": "blocked",
                    "reason": "Sensitive action requires explicit confirmation.",
                }
            else:
                event_data = _build_execute_event_data(finding, payload)
                self._hass.bus.async_fire(
                    EVENT_SENTINEL_EXECUTE_REQUESTED,
                    event_data,
                )
                outcome = {
                    "status": "event_fired",
                    "event_type": EVENT_SENTINEL_EXECUTE_REQUESTED,
                }

        if anomaly_id:
            self._pending_findings.pop(anomaly_id, None)
            await self._audit_store.async_update_response(
                anomaly_id=anomaly_id,
                response=response,
                outcome=outcome,
            )


def _build_execute_event_data(
    finding: AnomalyFinding, payload: dict[str, Any]
) -> dict[str, Any]:
    """Build deterministic payload for automation execution hooks."""
    return {
        "requested_at": dt_util.as_utc(dt_util.utcnow()).isoformat(),
        "anomaly_id": finding.anomaly_id,
        "type": finding.type,
        "severity": finding.severity,
        "confidence": finding.confidence,
        "triggering_entities": list(finding.triggering_entities),
        "suggested_actions": list(finding.suggested_actions),
        "is_sensitive": finding.is_sensitive,
        "evidence": finding.evidence,
        "mobile_action_payload": dict(payload),
    }

"""Notification action handling for sentinel findings."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from homeassistant.core import HomeAssistant

LOGGER = logging.getLogger(__name__)
from ..audit.store import AuditStore
from ..sentinel.models import AnomalyFinding
from ..sentinel.suppression import SuppressionManager, resolve_prompt

ACTION_PREFIX = "hga_sentinel_"


class ActionHandler:
    """Handle notification actions for sentinel findings."""

    def __init__(
        self,
        _hass: HomeAssistant,
        suppression: SuppressionManager,
        audit_store: AuditStore,
    ) -> None:
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
        if len(parts) != 2:
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
                outcome = {
                    "status": "skipped",
                    "reason": "No deterministic executor configured.",
                }

        if anomaly_id:
            self._pending_findings.pop(anomaly_id, None)
            await self._audit_store.async_update_response(
                anomaly_id=anomaly_id,
                response=response,
                outcome=outcome,
            )

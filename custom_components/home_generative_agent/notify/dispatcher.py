"""Notification dispatcher for sentinel findings."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from homeassistant.core import callback

if TYPE_CHECKING:
    from collections.abc import Callable

    from homeassistant.core import Event, HomeAssistant

    from custom_components.home_generative_agent.sentinel.models import AnomalyFinding
    from custom_components.home_generative_agent.snapshot.schema import (
        FullStateSnapshot,
    )

from custom_components.home_generative_agent.const import CONF_NOTIFY_SERVICE

from .actions import ACTION_PREFIX, ActionHandler

LOGGER = logging.getLogger(__name__)


class NotificationDispatcher:
    """Dispatch notifications and handle mobile app actions."""

    def __init__(
        self,
        hass: HomeAssistant,
        options: dict[str, Any],
        action_handler: ActionHandler,
    ) -> None:
        """Initialize notification dispatcher state."""
        self._hass = hass
        self._options = options
        self._action_handler = action_handler
        self._unsub: Callable[[], None] | None = None

    def start(self) -> None:
        """Start listening for action events."""
        if self._unsub is not None:
            return
        self._unsub = self._hass.bus.async_listen(
            "mobile_app_notification_action", self._handle_action_event
        )

    def stop(self) -> None:
        """Stop listening for action events."""
        if self._unsub is not None:
            self._unsub()
            self._unsub = None

    async def async_notify(
        self,
        finding: AnomalyFinding,
        _snapshot: FullStateSnapshot,
        explanation: str | None,
    ) -> None:
        """Send a proactive notification for a finding."""
        self._action_handler.register_finding(finding)

        title = "Home Generative Agent alert"
        message = explanation or _fallback_message(finding)
        actions = _build_actions(finding)

        notify_service = self._options.get(CONF_NOTIFY_SERVICE)
        if notify_service and isinstance(notify_service, str):
            LOGGER.info("Sending sentinel notification via %s.", notify_service)
            domain, _, service = notify_service.partition(".")
            if not service:
                service = notify_service
                domain = "notify"
            tag = f"hga_sentinel_{finding.anomaly_id[:32]}"
            data = {
                "title": title,
                "message": message,
                "data": {
                    "actions": actions,
                    "tag": tag,
                },
            }
            await self._hass.services.async_call(domain, service, data, blocking=False)
            return

        LOGGER.info("Sending sentinel notification via persistent_notification.")
        await self._hass.services.async_call(
            "persistent_notification",
            "create",
            {
                "title": title,
                "message": message,
                "notification_id": f"hga_sentinel_{finding.anomaly_id}",
            },
            blocking=False,
        )

    @callback
    def _handle_action_event(self, event: Event) -> None:
        action = event.data.get("action")
        if not isinstance(action, str):
            return
        if not action.startswith(ACTION_PREFIX):
            return
        self._hass.async_create_task(
            self._action_handler.handle_action(action, dict(event.data))
        )


def _build_actions(finding: AnomalyFinding) -> list[dict[str, Any]]:
    base = [
        {"action": f"{ACTION_PREFIX}ack_{finding.anomaly_id}", "title": "Acknowledge"},
        {"action": f"{ACTION_PREFIX}ignore_{finding.anomaly_id}", "title": "Ignore"},
        {"action": f"{ACTION_PREFIX}later_{finding.anomaly_id}", "title": "Ask Later"},
    ]
    if not finding.is_sensitive and finding.suggested_actions:
        base.append(
            {
                "action": f"{ACTION_PREFIX}execute_{finding.anomaly_id}",
                "title": "Execute",
            }
        )
    return base


def _fallback_message(finding: AnomalyFinding) -> str:
    entities = ", ".join(finding.triggering_entities) or "unknown"
    return (
        f"Sentinel detected {finding.type} "
        f"(severity {finding.severity}) for {entities}."
    )

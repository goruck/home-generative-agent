"""Notification dispatcher for sentinel findings."""

from __future__ import annotations

import logging
import re
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
MAX_MOBILE_MESSAGE_CHARS = 220


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
        mobile_message = _mobile_message(explanation, finding)
        persistent_message = _persistent_message(explanation, finding)
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
                "message": mobile_message,
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
                "message": persistent_message,
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
        {"action": f"{ACTION_PREFIX}later_{finding.anomaly_id}", "title": "Later"},
    ]
    if not finding.is_sensitive and finding.suggested_actions:
        base.append(
            {
                "action": f"{ACTION_PREFIX}execute_{finding.anomaly_id}",
                "title": "Execute",
            }
        )
    return base


def _normalize_text(text: str) -> str:
    normalized = text.replace("**", "").replace("`", "")
    normalized = normalized.replace("\r", " ").replace("\n", " ")
    return re.sub(r"\s+", " ", normalized).strip()


def _friendly_type(anomaly_type: str) -> str:
    known = {
        "open_entry_while_away": "Open entry while away",
        "open_entry_at_night_when_home": "Open entry at night",
        "open_entry_at_night_when_home_window": "Open entry at night",
        "open_entry_at_night_while_away": "Open entry at night",
        "open_any_window_at_night_while_away": "Window open at night",
        "unlocked_lock_at_night": "Door lock left unlocked",
        "camera_entry_unsecured": "Activity near unsecured entry",
    }
    if anomaly_type in known:
        return known[anomaly_type]
    return anomaly_type.replace("_", " ").strip().capitalize()


def _friendly_entity(entity_id: str) -> str:
    if "." in entity_id:
        _, _, name = entity_id.partition(".")
    else:
        name = entity_id
    return name.replace("_", " ").strip().title()


def _fallback_message(finding: AnomalyFinding) -> str:
    summary = _friendly_type(finding.type)
    entity = (
        _friendly_entity(finding.triggering_entities[0])
        if finding.triggering_entities
        else "Unknown entity"
    )
    return f"{summary}: {entity}. {_severity_action_hint(finding.severity)}"


def _mobile_message(explanation: str | None, finding: AnomalyFinding) -> str:
    if explanation:
        text = _normalize_text(explanation)
        if text and len(text) <= MAX_MOBILE_MESSAGE_CHARS:
            return text
    return _fallback_message(finding)[:MAX_MOBILE_MESSAGE_CHARS].rstrip()


def _persistent_message(explanation: str | None, finding: AnomalyFinding) -> str:
    if explanation:
        text = _normalize_text(explanation)
        if text:
            return text

    entities = ", ".join(
        _friendly_entity(entity) for entity in finding.triggering_entities
    )
    entities = entities or "Unknown entity"
    return (
        f"{_friendly_type(finding.type)} "
        f"(severity {finding.severity}) for {entities}. "
        + _severity_action_hint(finding.severity)
    )


def _severity_action_hint(severity: str) -> str:
    if severity == "high":
        return "Urgent: check and secure it now."
    if severity == "medium":
        return "Check soon and secure it if unexpected."
    return "Review when convenient."

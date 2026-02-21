"""Notification action handling for sentinel findings."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from homeassistant.helpers import entity_registry as er
from homeassistant.util import dt as dt_util

from custom_components.home_generative_agent.const import DOMAIN
from custom_components.home_generative_agent.sentinel.suppression import (
    SuppressionManager,
    resolve_prompt,
)

if TYPE_CHECKING:
    from homeassistant.core import HomeAssistant

    from custom_components.home_generative_agent.audit.store import AuditStore
    from custom_components.home_generative_agent.sentinel.models import AnomalyFinding

LOGGER = logging.getLogger(__name__)

ACTION_PREFIX = "hga_sentinel_"
ACTION_ID_PARTS = 2
EVENT_SENTINEL_EXECUTE_REQUESTED = "hga_sentinel_execute_requested"
EVENT_SENTINEL_ASK_REQUESTED = "hga_sentinel_ask_requested"


class ActionHandler:
    """Handle notification actions for sentinel findings."""

    def __init__(
        self,
        hass: HomeAssistant,
        suppression: SuppressionManager,
        audit_store: AuditStore,
        *,
        entry_id: str = "",
        notify_service: str | None = None,
    ) -> None:
        """Initialize action handling dependencies."""
        self._hass = hass
        self._suppression = suppression
        self._audit_store = audit_store
        self._entry_id = entry_id
        self._notify_service = notify_service
        self._pending_findings: dict[str, AnomalyFinding] = {}
        # Allows tests to inject a known entity_id without a real entity registry.
        self._conversation_entity_id_override: str | None = None

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
            outcome = await self._outcome_for_execute(finding, payload)
        elif action == "handoff":
            # Delegate to the conversation agent for PIN-gated or sensitive actions.
            outcome = await self._outcome_for_handoff(finding, payload)

        if anomaly_id:
            self._pending_findings.pop(anomaly_id, None)
            await self._audit_store.async_update_response(
                anomaly_id=anomaly_id,
                response=response,
                outcome=outcome,
            )

    def _resolve_agent_entity_id(self) -> str | None:
        """Resolve the HGA conversation entity_id from the entity registry."""
        if self._conversation_entity_id_override is not None:
            return self._conversation_entity_id_override
        if not self._entry_id:
            return None
        registry = er.async_get(self._hass)
        return registry.async_get_entity_id("conversation", DOMAIN, self._entry_id)

    async def _outcome_for_execute(
        self, finding: AnomalyFinding | None, payload: dict[str, Any]
    ) -> dict[str, Any]:
        """Call the conversation agent to execute; fall back to event for blueprints."""
        if finding is None:
            LOGGER.warning("Execute: finding not found (cleaned up or expired).")
            return {"status": "missing_finding"}
        if finding.is_sensitive:
            return {
                "status": "blocked",
                "reason": "Sensitive action requires agent confirmation.",
            }
        entity_id = self._resolve_agent_entity_id()
        if not entity_id:
            # No entity found — fire the event so a blueprint can handle it.
            LOGGER.info("No conversation entity; firing execute_requested event.")
            event_data = _build_execute_event_data(finding, payload)
            self._hass.bus.async_fire(EVENT_SENTINEL_EXECUTE_REQUESTED, event_data)
            return {
                "status": "event_fired",
                "event_type": EVENT_SENTINEL_EXECUTE_REQUESTED,
            }
        LOGGER.debug("Calling conversation agent %s for execute.", entity_id)
        prompt = _build_execute_prompt(finding)
        result = await self._hass.services.async_call(
            "conversation",
            "process",
            {"agent_id": entity_id, "text": prompt},
            blocking=True,
            return_response=True,
        )
        reply = _extract_speech(result)
        reply_sent = False
        if reply and self._notify_service:
            await self._async_send_agent_reply(reply)
            reply_sent = True
            LOGGER.debug("Agent reply sent via %s.", self._notify_service)
        elif not reply:
            LOGGER.warning("Agent returned no speech reply.")
        elif not self._notify_service:
            LOGGER.debug("Agent called successfully but no notify_service configured.")
        return {
            "status": "agent_called",
            "entity_id": entity_id,
            "reply_sent": reply_sent,
        }

    async def _outcome_for_handoff(
        self, finding: AnomalyFinding | None, payload: dict[str, Any]
    ) -> dict[str, Any]:
        """Call the conversation agent directly; fall back to event for blueprints."""
        if finding is None:
            LOGGER.warning("Handoff: finding not found (cleaned up or expired).")
            return {"status": "missing_finding"}
        entity_id = self._resolve_agent_entity_id()
        if not entity_id:
            # No entity found — fire the event so a blueprint can handle it.
            LOGGER.info("No conversation entity; firing ask_requested event.")
            event_data = _build_ask_event_data(finding, payload)
            self._hass.bus.async_fire(EVENT_SENTINEL_ASK_REQUESTED, event_data)
            return {"status": "event_fired", "event_type": EVENT_SENTINEL_ASK_REQUESTED}
        LOGGER.debug("Calling conversation agent %s for handoff.", entity_id)
        prompt = _build_ask_prompt(finding)
        result = await self._hass.services.async_call(
            "conversation",
            "process",
            {"agent_id": entity_id, "text": prompt},
            blocking=True,
            return_response=True,
        )
        reply = _extract_speech(result)
        reply_sent = False
        if reply and self._notify_service:
            await self._async_send_agent_reply(reply)
            reply_sent = True
            LOGGER.debug("Agent reply sent via %s.", self._notify_service)
        elif not reply:
            LOGGER.warning("Agent returned no speech reply.")
        elif not self._notify_service:
            LOGGER.debug("Agent called successfully but no notify_service configured.")
        return {
            "status": "agent_called",
            "entity_id": entity_id,
            "reply_sent": reply_sent,
        }

    async def _async_send_agent_reply(self, reply_text: str) -> None:
        """Send the agent reply as a push notification."""
        if not self._notify_service:
            return
        domain, _, service = self._notify_service.partition(".")
        if not service:
            service = self._notify_service
            domain = "notify"
        # Add notification data to make it actionable when tapped.
        notification_data = {
            "title": "Home Generative Agent",
            "message": reply_text,
            "data": {
                # Tag to group with other HGA notifications
                "tag": "hga_sentinel_reply",
                # For iOS: tapping could open HA to a view (no conversation URL)
                # "url": "/config/integrations"
            },
        }
        await self._hass.services.async_call(
            domain,
            service,
            notification_data,
            blocking=False,
        )


def _extract_speech(response: dict[str, Any] | None) -> str | None:
    """Extract the plain-speech text from a conversation.process response."""
    try:
        return response["response"]["speech"]["plain"]["speech"]  # type: ignore[index]
    except (KeyError, TypeError):
        return None


def _build_execute_prompt(finding: AnomalyFinding) -> str:
    """Build a natural-language prompt for non-sensitive execute actions."""
    entity_id = finding.evidence.get("entity_id") or (
        finding.triggering_entities[0] if finding.triggering_entities else None
    )
    friendly = finding.evidence.get("friendly_name") or (
        entity_id.split(".")[-1].replace("_", " ").title() if entity_id else "an entity"
    )
    suggested = (
        ", ".join(finding.suggested_actions)
        if finding.suggested_actions
        else "take appropriate action"
    )
    entity_clause = f" (entity_id: {entity_id})" if entity_id else ""
    extra = [e for e in finding.triggering_entities if e != entity_id]
    extra_clause = f" Also involved: {', '.join(extra)}." if extra else ""
    return (
        f"Sentinel alert — {finding.severity} severity, "
        f"{finding.confidence:.0%} confidence. "
        f"Primary device: {friendly}{entity_clause}.{extra_clause} "
        f"Suggested action: {suggested}. "
        f"Use GetLiveContext to check current state, then take appropriate action. "
        f"Report what you did or if you need additional information."
    )


def _build_ask_prompt(finding: AnomalyFinding) -> str:
    """Build a natural-language prompt from a finding for agent handoff."""
    entity_id = finding.evidence.get("entity_id") or (
        finding.triggering_entities[0] if finding.triggering_entities else None
    )
    friendly = finding.evidence.get("friendly_name") or (
        entity_id.split(".")[-1].replace("_", " ").title() if entity_id else "an entity"
    )
    suggested = (
        ", ".join(finding.suggested_actions)
        if finding.suggested_actions
        else "take appropriate action"
    )
    entity_clause = f" (entity_id: {entity_id})" if entity_id else ""
    extra = [e for e in finding.triggering_entities if e != entity_id]
    extra_clause = f" Also involved: {', '.join(extra)}." if extra else ""
    return (
        f"Sentinel security alert — {finding.severity} severity, "
        f"{finding.confidence:.0%} confidence. "
        f"Primary device: {friendly}{entity_clause}.{extra_clause} "
        f"Suggested remediation: {suggested}. "
        f"Use GetLiveContext to check current state if needed, then act. "
        f"Do not ask clarifying questions — proceed autonomously based on available "
        f"context. If the action requires a PIN or alarm code that you cannot obtain, "
        f"report that you cannot complete the action in this automated alert context "
        f"and instruct the user to handle it manually via the Home Assistant app."
    )


def _build_ask_event_data(
    finding: AnomalyFinding, payload: dict[str, Any]
) -> dict[str, Any]:
    """Build payload for the hga_sentinel_ask_requested event."""
    return {
        **_build_execute_event_data(finding, payload),
        "suggested_prompt": _build_ask_prompt(finding),
    }


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

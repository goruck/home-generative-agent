# ruff: noqa: S101
"""Tests for Sentinel mobile action handling."""

from __future__ import annotations

from typing import Any

import pytest

from custom_components.home_generative_agent.notify.actions import (
    ACTION_PREFIX,
    EVENT_SENTINEL_ASK_REQUESTED,
    EVENT_SENTINEL_EXECUTE_REQUESTED,
    ActionHandler,
)
from custom_components.home_generative_agent.sentinel.models import AnomalyFinding
from custom_components.home_generative_agent.sentinel.suppression import (
    SuppressionState,
)


class DummyBus:
    """Minimal bus recorder."""

    def __init__(self) -> None:
        self.events: list[dict[str, Any]] = []

    def async_fire(
        self,
        event_type: str,
        event_data: dict[str, Any] | None = None,
        origin: Any | None = None,
        context: Any | None = None,
        time_fired: Any | None = None,
    ) -> None:
        self.events.append(
            {
                "event_type": event_type,
                "event_data": event_data or {},
            }
        )


class DummyServices:
    """Service call recorder; returns a canned conversation response."""

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    async def async_call(
        self,
        domain: str,
        service: str,
        data: dict[str, Any],
        *,
        blocking: bool = False,
        return_response: bool = False,
    ) -> dict[str, Any] | None:
        self.calls.append(
            {
                "domain": domain,
                "service": service,
                "data": data,
                "blocking": blocking,
                "return_response": return_response,
            }
        )
        if domain == "conversation" and service == "process" and return_response:
            return {
                "response": {
                    "speech": {
                        "plain": {
                            "speech": "I have locked the front door.",
                        }
                    }
                }
            }
        return None


class DummyHass:
    """Minimal hass stub for action tests."""

    def __init__(self) -> None:
        self.bus = DummyBus()
        self.services = DummyServices()


class DummySuppressionManager:
    """Suppression state stub."""

    def __init__(self) -> None:
        self.state = SuppressionState()
        self.save_calls = 0

    async def async_save(self) -> None:
        self.save_calls += 1


class DummyAuditStore:
    """Audit update recorder."""

    def __init__(self) -> None:
        self.updates: list[dict[str, Any]] = []

    async def async_update_response(
        self,
        anomaly_id: str,
        response: dict[str, Any],
        outcome: dict[str, Any] | None,
    ) -> None:
        self.updates.append(
            {
                "anomaly_id": anomaly_id,
                "response": response,
                "outcome": outcome,
            }
        )


def _finding(*, anomaly_id: str, is_sensitive: bool) -> AnomalyFinding:
    return AnomalyFinding(
        anomaly_id=anomaly_id,
        type="appliance_power_duration",
        severity="medium",
        confidence=0.6,
        triggering_entities=["sensor.fridge_power"],
        evidence={"entity_id": "sensor.fridge_power", "power_w": 170.0},
        suggested_actions=["check_appliance"],
        is_sensitive=is_sensitive,
    )


def _sensitive_finding(anomaly_id: str = "sens-1") -> AnomalyFinding:
    return AnomalyFinding(
        anomaly_id=anomaly_id,
        type="unlocked_lock_at_night",
        severity="high",
        confidence=0.7,
        triggering_entities=["lock.front_door"],
        evidence={"entity_id": "lock.front_door", "friendly_name": "Front Door Lock"},
        suggested_actions=["lock_entity"],
        is_sensitive=True,
    )


@pytest.mark.asyncio
async def test_execute_fires_event_when_no_conversation_entity() -> None:
    """When entry_id is not set, the execute event fires as a blueprint fallback."""
    hass = DummyHass()
    suppression = DummySuppressionManager()
    suppression.state.pending_prompts["finding-1"] = "2026-01-01T00:00:00+00:00"
    audit = DummyAuditStore()
    handler = ActionHandler(
        hass=hass,  # type: ignore[arg-type]
        suppression=suppression,  # type: ignore[arg-type]
        audit_store=audit,  # type: ignore[arg-type]
    )
    finding = _finding(anomaly_id="finding-1", is_sensitive=False)
    handler.register_finding(finding)

    await handler.handle_action(
        f"{ACTION_PREFIX}execute_{finding.anomaly_id}",
        {"action_data": "from_test"},
    )

    assert len(hass.bus.events) == 1
    event = hass.bus.events[0]
    assert event["event_type"] == EVENT_SENTINEL_EXECUTE_REQUESTED
    event_data = event["event_data"]
    assert event_data["anomaly_id"] == finding.anomaly_id
    assert event_data["type"] == finding.type
    assert event_data["suggested_actions"] == ["check_appliance"]
    assert event_data["mobile_action_payload"] == {"action_data": "from_test"}
    assert "requested_at" in event_data

    assert suppression.save_calls == 1
    assert "finding-1" not in suppression.state.pending_prompts

    assert len(audit.updates) == 1
    assert audit.updates[0]["anomaly_id"] == "finding-1"
    assert audit.updates[0]["outcome"] == {
        "status": "event_fired",
        "event_type": EVENT_SENTINEL_EXECUTE_REQUESTED,
    }


@pytest.mark.asyncio
async def test_execute_calls_agent_and_sends_reply() -> None:
    """When a conversation entity is found, the agent is called directly."""
    hass = DummyHass()
    suppression = DummySuppressionManager()
    suppression.state.pending_prompts["finding-1"] = "2026-01-01T00:00:00+00:00"
    audit = DummyAuditStore()
    handler = ActionHandler(
        hass=hass,  # type: ignore[arg-type]
        suppression=suppression,  # type: ignore[arg-type]
        audit_store=audit,  # type: ignore[arg-type]
        entry_id="test-entry",
        notify_service="notify.mobile_app_phone",
    )
    handler._conversation_entity_id_override = "conversation.home_generative_agent"
    finding = _finding(anomaly_id="finding-1", is_sensitive=False)
    handler.register_finding(finding)

    await handler.handle_action(
        f"{ACTION_PREFIX}execute_{finding.anomaly_id}",
        {"action_data": "from_test"},
    )

    # No event fired — agent called directly.
    assert hass.bus.events == []

    # conversation.process was called with the right agent and prompt.
    conv_calls = [c for c in hass.services.calls if c["domain"] == "conversation"]
    assert len(conv_calls) == 1
    assert conv_calls[0]["data"]["agent_id"] == "conversation.home_generative_agent"
    assert "check_appliance" in conv_calls[0]["data"]["text"]
    assert conv_calls[0]["return_response"] is True

    # Reply notification sent via the configured notify service.
    notify_calls = [c for c in hass.services.calls if c["domain"] == "notify"]
    assert len(notify_calls) == 1
    assert notify_calls[0]["service"] == "mobile_app_phone"
    assert notify_calls[0]["data"]["message"] == "I have locked the front door."

    assert "finding-1" not in suppression.state.pending_prompts
    assert suppression.save_calls == 1
    assert finding.anomaly_id not in handler._pending_findings

    assert audit.updates[0]["outcome"] == {
        "status": "agent_called",
        "entity_id": "conversation.home_generative_agent",
        "reply_sent": True,
    }


@pytest.mark.asyncio
async def test_execute_blocked_when_sensitive() -> None:
    hass = DummyHass()
    suppression = DummySuppressionManager()
    audit = DummyAuditStore()
    handler = ActionHandler(
        hass=hass,  # type: ignore[arg-type]
        suppression=suppression,  # type: ignore[arg-type]
        audit_store=audit,  # type: ignore[arg-type]
    )
    finding = _finding(anomaly_id="finding-2", is_sensitive=True)
    handler.register_finding(finding)

    await handler.handle_action(f"{ACTION_PREFIX}execute_{finding.anomaly_id}", {})

    assert hass.bus.events == []
    assert len(audit.updates) == 1
    assert audit.updates[0]["outcome"] == {
        "status": "blocked",
        "reason": "Sensitive action requires agent confirmation.",
    }


@pytest.mark.asyncio
async def test_execute_missing_finding_records_outcome() -> None:
    hass = DummyHass()
    suppression = DummySuppressionManager()
    audit = DummyAuditStore()
    handler = ActionHandler(
        hass=hass,  # type: ignore[arg-type]
        suppression=suppression,  # type: ignore[arg-type]
        audit_store=audit,  # type: ignore[arg-type]
    )

    await handler.handle_action(f"{ACTION_PREFIX}execute_missing-id", {"source": "app"})

    assert hass.bus.events == []
    assert len(audit.updates) == 1
    assert audit.updates[0]["anomaly_id"] == "missing-id"
    assert audit.updates[0]["outcome"] == {"status": "missing_finding"}


# ---------------------------------------------------------------------------
# Agent handoff flow tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_handoff_fires_ask_event_when_no_conversation_entity() -> None:
    """When entry_id is not set, the ask event fires as a blueprint fallback."""
    hass = DummyHass()
    suppression = DummySuppressionManager()
    suppression.state.pending_prompts["sens-1"] = "2026-01-01T00:00:00+00:00"
    audit = DummyAuditStore()
    handler = ActionHandler(
        hass=hass,  # type: ignore[arg-type]
        suppression=suppression,  # type: ignore[arg-type]
        audit_store=audit,  # type: ignore[arg-type]
    )
    finding = _sensitive_finding()
    handler.register_finding(finding)

    await handler.handle_action(f"{ACTION_PREFIX}handoff_{finding.anomaly_id}", {})

    assert len(hass.bus.events) == 1
    event = hass.bus.events[0]
    assert event["event_type"] == EVENT_SENTINEL_ASK_REQUESTED
    event_data = event["event_data"]
    assert event_data["anomaly_id"] == finding.anomaly_id
    assert event_data["type"] == finding.type
    assert event_data["is_sensitive"] is True
    assert "suggested_prompt" in event_data
    assert "Front Door Lock" in event_data["suggested_prompt"]

    # Suppression prompt resolved — user responded.
    assert "sens-1" not in suppression.state.pending_prompts
    assert suppression.save_calls == 1

    # Finding cleaned up.
    assert finding.anomaly_id not in handler._pending_findings

    assert audit.updates[0]["outcome"] == {
        "status": "event_fired",
        "event_type": EVENT_SENTINEL_ASK_REQUESTED,
    }


@pytest.mark.asyncio
async def test_handoff_missing_finding_records_outcome() -> None:
    hass = DummyHass()
    suppression = DummySuppressionManager()
    audit = DummyAuditStore()
    handler = ActionHandler(
        hass=hass,  # type: ignore[arg-type]
        suppression=suppression,  # type: ignore[arg-type]
        audit_store=audit,  # type: ignore[arg-type]
    )

    await handler.handle_action(f"{ACTION_PREFIX}handoff_missing-id", {})

    assert hass.bus.events == []
    assert audit.updates[0]["outcome"] == {"status": "missing_finding"}


@pytest.mark.asyncio
async def test_handoff_calls_agent_and_sends_reply() -> None:
    """When a conversation entity is found, the agent is called directly."""
    hass = DummyHass()
    suppression = DummySuppressionManager()
    suppression.state.pending_prompts["sens-1"] = "2026-01-01T00:00:00+00:00"
    audit = DummyAuditStore()
    handler = ActionHandler(
        hass=hass,  # type: ignore[arg-type]
        suppression=suppression,  # type: ignore[arg-type]
        audit_store=audit,  # type: ignore[arg-type]
        entry_id="test-entry",
        notify_service="notify.mobile_app_phone",
    )
    handler._conversation_entity_id_override = "conversation.home_generative_agent"
    finding = _sensitive_finding()
    handler.register_finding(finding)

    await handler.handle_action(f"{ACTION_PREFIX}handoff_{finding.anomaly_id}", {})

    # No event fired — agent called directly.
    assert hass.bus.events == []

    # conversation.process was called with the right agent and prompt.
    conv_calls = [c for c in hass.services.calls if c["domain"] == "conversation"]
    assert len(conv_calls) == 1
    assert conv_calls[0]["data"]["agent_id"] == "conversation.home_generative_agent"
    assert "Front Door Lock" in conv_calls[0]["data"]["text"]
    assert conv_calls[0]["return_response"] is True

    # Reply notification sent via the configured notify service.
    notify_calls = [c for c in hass.services.calls if c["domain"] == "notify"]
    assert len(notify_calls) == 1
    assert notify_calls[0]["service"] == "mobile_app_phone"
    assert notify_calls[0]["data"]["message"] == "I have locked the front door."
    assert notify_calls[0]["data"]["title"] == "Home Generative Agent"

    assert "sens-1" not in suppression.state.pending_prompts
    assert suppression.save_calls == 1
    assert finding.anomaly_id not in handler._pending_findings

    assert audit.updates[0]["outcome"] == {
        "status": "agent_called",
        "entity_id": "conversation.home_generative_agent",
        "reply_sent": True,
    }


@pytest.mark.asyncio
async def test_handoff_calls_agent_without_reply_when_no_notify_service() -> None:
    """Agent is called but no reply is sent when notify_service is not configured."""
    hass = DummyHass()
    suppression = DummySuppressionManager()
    audit = DummyAuditStore()
    handler = ActionHandler(
        hass=hass,  # type: ignore[arg-type]
        suppression=suppression,  # type: ignore[arg-type]
        audit_store=audit,  # type: ignore[arg-type]
        entry_id="test-entry",
    )
    handler._conversation_entity_id_override = "conversation.home_generative_agent"
    finding = _sensitive_finding(anomaly_id="sens-2")
    handler.register_finding(finding)

    await handler.handle_action(f"{ACTION_PREFIX}handoff_{finding.anomaly_id}", {})

    assert hass.bus.events == []
    conv_calls = [c for c in hass.services.calls if c["domain"] == "conversation"]
    assert len(conv_calls) == 1
    notify_calls = [c for c in hass.services.calls if c["domain"] == "notify"]
    assert notify_calls == []

    assert audit.updates[0]["outcome"] == {
        "status": "agent_called",
        "entity_id": "conversation.home_generative_agent",
        "reply_sent": False,
    }

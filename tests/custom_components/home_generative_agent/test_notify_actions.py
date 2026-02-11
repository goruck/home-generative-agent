# ruff: noqa: S101
"""Tests for Sentinel mobile action handling."""

from __future__ import annotations

from typing import Any

import pytest

from custom_components.home_generative_agent.notify.actions import (
    ACTION_PREFIX,
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


class DummyHass:
    """Minimal hass stub for action tests."""

    def __init__(self) -> None:
        self.bus = DummyBus()


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


@pytest.mark.asyncio
async def test_execute_fires_event_and_updates_audit() -> None:
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
        "reason": "Sensitive action requires explicit confirmation.",
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

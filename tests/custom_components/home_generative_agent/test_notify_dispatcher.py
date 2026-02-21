# ruff: noqa: S101
"""Tests for Sentinel notification copy and actions."""

from __future__ import annotations

from typing import Any

import pytest

from custom_components.home_generative_agent.notify.actions import ACTION_PREFIX
from custom_components.home_generative_agent.notify.dispatcher import (
    NotificationDispatcher,
    _build_actions,
)
from custom_components.home_generative_agent.sentinel.models import AnomalyFinding


class DummyServices:
    """Service call recorder."""

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    async def async_call(
        self,
        domain: str,
        service: str,
        data: dict[str, Any],
        *,
        blocking: bool = False,
    ) -> None:
        self.calls.append(
            {
                "domain": domain,
                "service": service,
                "data": data,
                "blocking": blocking,
            }
        )


class DummyHass:
    """Minimal hass stub for notification tests."""

    def __init__(self) -> None:
        self.services = DummyServices()


class DummyActionHandler:
    """Action handler stub."""

    def __init__(self) -> None:
        self.findings: list[AnomalyFinding] = []

    def register_finding(self, finding: AnomalyFinding) -> None:
        self.findings.append(finding)


def _finding() -> AnomalyFinding:
    return AnomalyFinding(
        anomaly_id="abc123",
        type="open_entry_at_night_when_home_window",
        severity="high",
        confidence=0.62,
        triggering_entities=["binary_sensor.garage_and_play_room_doors"],
        evidence={"entity_id": "binary_sensor.garage_and_play_room_doors"},
        suggested_actions=["close_entry"],
        is_sensitive=True,
    )


def _low_finding() -> AnomalyFinding:
    return AnomalyFinding(
        anomaly_id="abc124",
        type="camera_entry_unsecured",
        severity="low",
        confidence=0.31,
        triggering_entities=["camera.driveway"],
        evidence={"camera_entity_id": "camera.driveway"},
        suggested_actions=["check_entry"],
        is_sensitive=True,
    )


@pytest.mark.asyncio
async def test_mobile_notify_uses_compact_message_and_short_labels() -> None:
    hass = DummyHass()
    action_handler = DummyActionHandler()
    dispatcher = NotificationDispatcher(
        hass=hass,  # type: ignore[arg-type]
        options={"notify_service": "notify.mobile_app_phone"},
        action_handler=action_handler,  # type: ignore[arg-type]
    )

    await dispatcher.async_notify(
        _finding(),
        {},  # type: ignore[arg-type]
        "**rule** `open_entry_at_night_when_home_window` " + ("x " * 180),
    )

    assert len(hass.services.calls) == 1
    call = hass.services.calls[0]
    assert call["domain"] == "notify"
    assert call["service"] == "mobile_app_phone"
    message = str(call["data"]["message"])
    assert len(message) <= 220
    assert "**" not in message
    assert "`" not in message
    assert "open_entry_at_night_when_home_window" not in message
    actions = call["data"]["data"]["actions"]
    assert [a["title"] for a in actions] == [
        "Acknowledge",
        "Ignore",
        "Later",
        "Ask Agent",
    ]


@pytest.mark.asyncio
async def test_persistent_notify_uses_detailed_fallback() -> None:
    hass = DummyHass()
    action_handler = DummyActionHandler()
    dispatcher = NotificationDispatcher(
        hass=hass,  # type: ignore[arg-type]
        options={},
        action_handler=action_handler,  # type: ignore[arg-type]
    )

    await dispatcher.async_notify(_finding(), {}, None)  # type: ignore[arg-type]

    assert len(hass.services.calls) == 1
    call = hass.services.calls[0]
    assert call["domain"] == "persistent_notification"
    assert call["service"] == "create"
    message = str(call["data"]["message"])
    assert "severity high" in message
    assert "Open entry at night" in message
    assert "Urgent: check and secure it now." in message


@pytest.mark.asyncio
async def test_persistent_notify_low_severity_uses_relaxed_hint() -> None:
    hass = DummyHass()
    action_handler = DummyActionHandler()
    dispatcher = NotificationDispatcher(
        hass=hass,  # type: ignore[arg-type]
        options={},
        action_handler=action_handler,  # type: ignore[arg-type]
    )

    await dispatcher.async_notify(_low_finding(), {}, None)  # type: ignore[arg-type]

    assert len(hass.services.calls) == 1
    call = hass.services.calls[0]
    message = str(call["data"]["message"])
    assert "severity low" in message
    assert "Review when convenient." in message


# ---------------------------------------------------------------------------
# _build_actions button logic tests
# ---------------------------------------------------------------------------


def _non_sensitive_finding() -> AnomalyFinding:
    return AnomalyFinding(
        anomaly_id="ns1",
        type="appliance_power_duration",
        severity="medium",
        confidence=0.6,
        triggering_entities=["sensor.fridge_power"],
        evidence={},
        suggested_actions=["check_appliance"],
        is_sensitive=False,
    )


def _sensitive_finding_no_actions() -> AnomalyFinding:
    return AnomalyFinding(
        anomaly_id="s2",
        type="unlocked_lock_at_night",
        severity="high",
        confidence=0.7,
        triggering_entities=["lock.front_door"],
        evidence={},
        suggested_actions=[],
        is_sensitive=True,
    )


def test_sensitive_finding_shows_ask_agent_button() -> None:
    finding = _finding()  # is_sensitive=True, has suggested_actions
    actions = _build_actions(finding)
    titles = [a["title"] for a in actions]
    ids = [a["action"] for a in actions]
    assert "Ask Agent" in titles
    assert "Execute" not in titles
    assert "Confirm to Execute" not in titles
    assert any(aid.startswith(f"{ACTION_PREFIX}handoff_") for aid in ids)


def test_non_sensitive_finding_shows_execute_button() -> None:
    finding = _non_sensitive_finding()
    actions = _build_actions(finding)
    titles = [a["title"] for a in actions]
    assert "Execute" in titles
    assert "Confirm to Execute" not in titles


def test_sensitive_finding_without_suggested_actions_has_no_extra_button() -> None:
    finding = _sensitive_finding_no_actions()
    actions = _build_actions(finding)
    titles = [a["title"] for a in actions]
    assert titles == ["Acknowledge", "Ignore", "Later"]

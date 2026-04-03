"""Tests for SentinelNotifier — sentinel/notifier.py (Issue #261)."""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

import custom_components.home_generative_agent.sentinel.notifier as _notifier_mod
from custom_components.home_generative_agent.const import (
    ACT_SNOOZE_24H,
    ACT_SNOOZE_ALWAYS,
    ACT_SNOOZE_CANCEL,
    ACT_SNOOZE_CONFIRM,
    ACTION_PREFIX,
    CONF_NOTIFY_SERVICE,
    CONF_SENTINEL_AREA_NOTIFY_MAP,
    SNOOZE_PERMANENT,
)
from custom_components.home_generative_agent.sentinel.models import AnomalyFinding
from custom_components.home_generative_agent.sentinel.notifier import (
    SentinelNotifier,
    _friendly_type,
    _redact_if_sensitive,
)
from custom_components.home_generative_agent.sentinel.suppression import (
    SuppressionState,
)

# ---------------------------------------------------------------------------
# Minimal stubs
# ---------------------------------------------------------------------------


class DummyServices:
    """Records async_call() invocations."""

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    async def async_call(
        self,
        domain: str,
        service: str,
        data: dict[str, Any] | None = None,
        *,
        blocking: bool = False,
        return_response: bool = False,
    ) -> None:
        self.calls.append({"domain": domain, "service": service, "data": data or {}})


class DummyBus:
    """Records async_listen() subscriptions and returns a no-op unsub."""

    def async_listen(self, event_type: str, callback: Any) -> Any:
        return lambda: None


class DummyHass:
    """Minimal HomeAssistant stub with task draining support."""

    def __init__(self) -> None:
        self.services = DummyServices()
        self.bus = DummyBus()
        self._pending_tasks: list[asyncio.Task[Any]] = []

    def async_create_task(self, coro: Any) -> asyncio.Task[Any]:
        loop = asyncio.get_event_loop()
        task = loop.create_task(coro)
        self._pending_tasks.append(task)
        return task

    async def drain_tasks(self) -> None:
        while self._pending_tasks:
            task = self._pending_tasks.pop(0)
            await task


class DummySuppressionManager:
    """SuppressionManager stub."""

    def __init__(self) -> None:
        self.state = SuppressionState()
        self.is_read_only = False
        self.save_called = False
        self.save_count = 0

    async def async_save(self) -> None:
        self.save_called = True
        self.save_count += 1


class DummyActionHandler:
    """ActionHandler stub recording register_finding and handle_action calls."""

    def __init__(self) -> None:
        self._pending_findings: dict[str, AnomalyFinding] = {}
        self.register_calls: list[AnomalyFinding] = []
        self.handle_calls: list[tuple[str, dict[str, Any]]] = []

    def register_finding(self, finding: AnomalyFinding) -> None:
        self._pending_findings[finding.anomaly_id] = finding
        self.register_calls.append(finding)

    async def handle_action(self, action_id: str, payload: dict[str, Any]) -> None:
        self.handle_calls.append((action_id, payload))


class DummyEvent:
    """Minimal HA Event stub."""

    def __init__(self, data: dict[str, Any]) -> None:
        self._data = data

    @property
    def data(self) -> dict[str, Any]:
        return self._data


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _finding(
    anomaly_id: str = "abc123",
    ftype: str = "open_entry_while_away",
    is_sensitive: bool = False,
    recognized_people: list[str] | None = None,
) -> AnomalyFinding:
    evidence: dict[str, Any] = {}
    if recognized_people is not None:
        evidence["recognized_people"] = recognized_people
    return AnomalyFinding(
        anomaly_id=anomaly_id,
        type=ftype,
        severity="medium",
        confidence=0.75,
        triggering_entities=["binary_sensor.front_door"],
        evidence=evidence,
        suggested_actions=["close_entry"],
        is_sensitive=is_sensitive,
    )


def _make_notifier(
    options: dict[str, Any] | None = None,
    hass: DummyHass | None = None,
    suppression: DummySuppressionManager | None = None,
    action_handler: DummyActionHandler | None = None,
) -> tuple[SentinelNotifier, DummyHass, DummySuppressionManager, DummyActionHandler]:
    h = hass or DummyHass()
    s = suppression or DummySuppressionManager()
    a = action_handler or DummyActionHandler()
    opts = options if options is not None else {}
    notifier = SentinelNotifier(
        hass=h,  # type: ignore[arg-type]
        options=opts,
        suppression=s,  # type: ignore[arg-type]
        action_handler=a,  # type: ignore[arg-type]
    )
    return notifier, h, s, a


def _minimal_snapshot(area: str = "Living Room") -> dict[str, Any]:
    return {
        "schema_version": 1,
        "generated_at": "2025-01-01T00:00:00+00:00",
        "entities": [
            {
                "entity_id": "binary_sensor.front_door",
                "state": "on",
                "domain": "binary_sensor",
                "area": area,
                "attributes": {},
                "last_changed": "2025-01-01T00:00:00+00:00",
                "last_updated": "2025-01-01T00:00:00+00:00",
            }
        ],
        "camera_activity": [],
        "derived": {
            "now": "2025-01-01T10:00:00+00:00",
            "timezone": "UTC",
            "is_night": False,
            "anyone_home": False,
            "people_home": [],
            "people_away": [],
            "last_motion_by_area": {},
        },
    }


# ---------------------------------------------------------------------------
# 1. ``always`` confirmation guard
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_snooze_always_sends_confirmation_not_snooze() -> None:
    """
    Tapping 'Snooze Always' sends a confirmation notification.

    register_snooze must NOT be called at this point.
    """
    options = {CONF_NOTIFY_SERVICE: "notify.mobile_app_phone"}
    notifier, hass, suppression, action_handler = _make_notifier(options)
    finding = _finding(anomaly_id="abc123")
    action_handler.register_finding(finding)

    await notifier._handle_snooze(ACT_SNOOZE_ALWAYS, "abc123")

    # A confirmation notification must have been sent.
    assert len(hass.services.calls) == 1
    call = hass.services.calls[0]
    assert call["domain"] == "notify"
    assert call["service"] == "mobile_app_phone"
    assert "Confirm permanent snooze" in call["data"]["title"]

    # Snooze must NOT have been written to suppression state yet.
    assert finding.type not in suppression.state.snoozed_until
    assert suppression.save_called is False

    # The pending intent must be recorded.
    assert "abc123" in notifier._pending_always_snooze
    assert notifier._pending_always_snooze["abc123"] == finding.type


@pytest.mark.asyncio
async def test_snooze_confirm_writes_permanent_after_always() -> None:
    """After 'Snooze Always' → 'Confirm', SNOOZE_PERMANENT is written."""
    options = {CONF_NOTIFY_SERVICE: "notify.mobile_app_phone"}
    notifier, _hass, suppression, action_handler = _make_notifier(options)
    finding = _finding(anomaly_id="abc123")
    action_handler.register_finding(finding)

    await notifier._handle_snooze(ACT_SNOOZE_ALWAYS, "abc123")
    assert finding.type not in suppression.state.snoozed_until

    await notifier._handle_snooze(ACT_SNOOZE_CONFIRM, "abc123")

    assert finding.type in suppression.state.snoozed_until
    entry = suppression.state.snoozed_until[finding.type]
    assert entry["until"] == SNOOZE_PERMANENT
    assert suppression.save_called is True
    assert "abc123" not in notifier._pending_always_snooze


@pytest.mark.asyncio
async def test_snooze_confirm_without_prior_always_is_noop() -> None:
    """A stray 'Confirm' with no prior 'Snooze Always' must be a no-op."""
    notifier, hass, suppression, action_handler = _make_notifier()
    finding = _finding(anomaly_id="abc123")
    action_handler.register_finding(finding)

    await notifier._handle_snooze(ACT_SNOOZE_CONFIRM, "abc123")

    assert finding.type not in suppression.state.snoozed_until
    assert suppression.save_called is False
    assert hass.services.calls == []


# ---------------------------------------------------------------------------
# 2. Snooze 24 h
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_snooze_24h_writes_to_suppression() -> None:
    """snooze24h registers a 24-hour snooze and calls async_save()."""
    notifier, _hass, suppression, action_handler = _make_notifier()
    finding = _finding(anomaly_id="abc123", ftype="open_entry_while_away")
    action_handler.register_finding(finding)

    await notifier._handle_snooze(ACT_SNOOZE_24H, "abc123")

    assert finding.type in suppression.state.snoozed_until
    entry = suppression.state.snoozed_until[finding.type]
    assert entry["until"] != SNOOZE_PERMANENT
    assert suppression.save_called is True
    assert suppression.save_count == 1


@pytest.mark.asyncio
async def test_snooze_24h_unknown_finding_is_noop() -> None:
    """snooze24h for an unknown anomaly_id must not crash or write state."""
    notifier, _hass, suppression, _action_handler = _make_notifier()

    await notifier._handle_snooze(ACT_SNOOZE_24H, "nonexistent")

    assert suppression.state.snoozed_until == {}
    assert suppression.save_called is False


# ---------------------------------------------------------------------------
# 3. Sensitive-flag redacts person names
# ---------------------------------------------------------------------------


def test_redact_if_sensitive_replaces_names() -> None:
    """_redact_if_sensitive replaces known names with 'a recognised person'."""
    finding = _finding(is_sensitive=True, recognized_people=["John Doe"])
    explanation = "John Doe was seen near the front door."

    result = _redact_if_sensitive(explanation, finding)

    assert result is not None
    assert "John Doe" not in result
    assert "a recognised person" in result


def test_redact_if_sensitive_multiple_names() -> None:
    """All names in recognized_people are redacted."""
    finding = _finding(
        is_sensitive=True, recognized_people=["Alice Smith", "Bob Jones"]
    )
    explanation = "Alice Smith and Bob Jones were detected."

    result = _redact_if_sensitive(explanation, finding)

    assert result is not None
    assert "Alice Smith" not in result
    assert "Bob Jones" not in result
    assert result.count("a recognised person") == 2


def test_redact_if_sensitive_case_insensitive() -> None:
    """Redaction is case-insensitive."""
    finding = _finding(is_sensitive=True, recognized_people=["John Doe"])
    explanation = "JOHN DOE was detected."

    result = _redact_if_sensitive(explanation, finding)

    assert result is not None
    assert "JOHN DOE" not in result
    assert "a recognised person" in result


def test_no_redaction_when_not_sensitive() -> None:
    """Names are NOT redacted when is_sensitive=False."""
    finding = _finding(is_sensitive=False, recognized_people=["John Doe"])
    explanation = "John Doe was seen near the front door."

    result = _redact_if_sensitive(explanation, finding)

    assert result == explanation


def test_no_redaction_when_no_recognized_people() -> None:
    """Explanation is returned unchanged when recognized_people is empty."""
    finding = _finding(is_sensitive=True, recognized_people=None)
    explanation = "Motion detected near the front door."

    result = _redact_if_sensitive(explanation, finding)

    assert result == explanation


@pytest.mark.asyncio
async def test_async_notify_redacts_sensitive_message() -> None:
    """async_notify sends a redacted message when finding.is_sensitive=True."""
    options = {CONF_NOTIFY_SERVICE: "notify.mobile_app_phone"}
    notifier, hass, _suppression, _action_handler = _make_notifier(options)
    snapshot = _minimal_snapshot()

    sensitive_finding = _finding(
        anomaly_id="sens1",
        is_sensitive=True,
        recognized_people=["John Doe"],
    )
    explanation = "John Doe was seen near the front door at 10 PM."

    await notifier.async_notify(sensitive_finding, snapshot, explanation)  # type: ignore[arg-type]

    assert len(hass.services.calls) == 1
    call = hass.services.calls[0]
    message = call["data"]["message"]
    assert "John Doe" not in message
    assert "a recognised person" in message


# ---------------------------------------------------------------------------
# 3b. _normalize_text strips think blocks
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_async_notify_strips_think_blocks_from_explanation() -> None:
    """<think> blocks in explanation text must be stripped before the notification is sent."""
    options = {CONF_NOTIFY_SERVICE: "notify.mobile_app_phone"}
    notifier, hass, _suppression, _action_handler = _make_notifier(options)
    snapshot = _minimal_snapshot()
    finding = _finding(anomaly_id="think1")

    explanation = (
        "<think>internal reasoning</think>Front door open recently. Close it now."
    )
    await notifier.async_notify(finding, snapshot, explanation)  # type: ignore[arg-type]

    assert len(hass.services.calls) == 1
    message = hass.services.calls[0]["data"]["message"]
    assert "<think>" not in message
    assert "internal reasoning" not in message
    assert "Front door open recently." in message


# ---------------------------------------------------------------------------
# 4. Per-area routing
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_area_routing_uses_mapped_service() -> None:
    """When entity area matches CONF_SENTINEL_AREA_NOTIFY_MAP, use mapped service."""
    options = {
        CONF_NOTIFY_SERVICE: "notify.mobile_app_global",
        CONF_SENTINEL_AREA_NOTIFY_MAP: {"Living Room": "notify.mobile_app_alice"},
    }
    notifier, hass, _suppression, _action_handler = _make_notifier(options)
    snapshot = _minimal_snapshot(area="Living Room")
    finding = _finding(anomaly_id="route1")

    await notifier.async_notify(finding, snapshot, "Door is open.")  # type: ignore[arg-type]

    assert len(hass.services.calls) == 1
    call = hass.services.calls[0]
    assert call["domain"] == "notify"
    assert call["service"] == "mobile_app_alice"


@pytest.mark.asyncio
async def test_area_routing_falls_back_to_global_service() -> None:
    """When no area matches the map, the global notify service is used."""
    options = {
        CONF_NOTIFY_SERVICE: "notify.mobile_app_global",
        CONF_SENTINEL_AREA_NOTIFY_MAP: {"Kitchen": "notify.mobile_app_bob"},
    }
    notifier, hass, _suppression, _action_handler = _make_notifier(options)
    snapshot = _minimal_snapshot(area="Living Room")
    finding = _finding(anomaly_id="route2")

    await notifier.async_notify(finding, snapshot, "Door is open.")  # type: ignore[arg-type]

    assert len(hass.services.calls) == 1
    call = hass.services.calls[0]
    assert call["domain"] == "notify"
    assert call["service"] == "mobile_app_global"


@pytest.mark.asyncio
async def test_no_notify_service_uses_persistent_notification() -> None:
    """When no notify service is configured, a persistent notification is sent."""
    notifier, hass, _suppression, _action_handler = _make_notifier(options={})
    snapshot = _minimal_snapshot()
    finding = _finding(anomaly_id="persist1")

    await notifier.async_notify(finding, snapshot, "Door is open.")  # type: ignore[arg-type]

    assert len(hass.services.calls) == 1
    call = hass.services.calls[0]
    assert call["domain"] == "persistent_notification"
    assert call["service"] == "create"


@pytest.mark.asyncio
async def test_area_map_only_without_global_service_routes_correctly() -> None:
    """Area map works even when no global CONF_NOTIFY_SERVICE is set."""
    options = {
        CONF_SENTINEL_AREA_NOTIFY_MAP: {"Living Room": "notify.mobile_app_alice"},
    }
    notifier, hass, _suppression, _action_handler = _make_notifier(options)
    snapshot = _minimal_snapshot(area="Living Room")
    finding = _finding(anomaly_id="route3")

    await notifier.async_notify(finding, snapshot, "Door is open.")  # type: ignore[arg-type]

    assert len(hass.services.calls) == 1
    call = hass.services.calls[0]
    assert call["service"] == "mobile_app_alice"


# ---------------------------------------------------------------------------
# 5. Non-snooze actions delegated to ActionHandler
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_execute_action_delegated_to_action_handler() -> None:
    """execute_<id> mobile actions are delegated to ActionHandler.handle_action()."""
    notifier, hass, _suppression, action_handler = _make_notifier()
    finding = _finding(anomaly_id="abc123")
    action_handler.register_finding(finding)
    notifier.start()

    action_str = f"{ACTION_PREFIX}execute_abc123"
    event = DummyEvent({"action": action_str, "extra": "payload"})
    notifier._handle_action_event(event)  # type: ignore[arg-type]

    await hass.drain_tasks()

    assert len(action_handler.handle_calls) == 1
    called_action_id, called_payload = action_handler.handle_calls[0]
    assert called_action_id == action_str
    assert called_payload["action"] == action_str


@pytest.mark.asyncio
async def test_non_prefixed_action_not_delegated() -> None:
    """Actions not starting with ACTION_PREFIX are silently ignored."""
    notifier, hass, _suppression, action_handler = _make_notifier()
    notifier.start()

    event = DummyEvent({"action": "some_other_app_action"})
    notifier._handle_action_event(event)  # type: ignore[arg-type]

    await hass.drain_tasks()

    assert action_handler.handle_calls == []
    assert hass.services.calls == []


@pytest.mark.asyncio
async def test_handoff_action_delegated_to_action_handler() -> None:
    """handoff_<id> mobile actions are delegated to ActionHandler."""
    notifier, hass, _suppression, action_handler = _make_notifier()
    finding = _finding(anomaly_id="abc123")
    action_handler.register_finding(finding)
    notifier.start()

    action_str = f"{ACTION_PREFIX}handoff_abc123"
    event = DummyEvent({"action": action_str})
    notifier._handle_action_event(event)  # type: ignore[arg-type]

    await hass.drain_tasks()

    assert len(action_handler.handle_calls) == 1
    assert action_handler.handle_calls[0][0] == action_str


# ---------------------------------------------------------------------------
# 6. Snooze cancel clears pending state
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_snooze_cancel_clears_pending_always() -> None:
    """'Snooze Cancel' after 'Snooze Always' discards the pending intent."""
    options = {CONF_NOTIFY_SERVICE: "notify.mobile_app_phone"}
    notifier, _hass, suppression, action_handler = _make_notifier(options)
    finding = _finding(anomaly_id="abc123")
    action_handler.register_finding(finding)

    await notifier._handle_snooze(ACT_SNOOZE_ALWAYS, "abc123")
    assert "abc123" in notifier._pending_always_snooze

    await notifier._handle_snooze(ACT_SNOOZE_CANCEL, "abc123")

    assert "abc123" not in notifier._pending_always_snooze
    assert finding.type not in suppression.state.snoozed_until
    assert suppression.save_called is False


@pytest.mark.asyncio
async def test_snooze_cancel_without_prior_always_is_noop() -> None:
    """A stray 'Cancel' with no prior 'Snooze Always' must be a silent no-op."""
    notifier, _hass, suppression, action_handler = _make_notifier()
    finding = _finding(anomaly_id="abc123")
    action_handler.register_finding(finding)

    await notifier._handle_snooze(ACT_SNOOZE_CANCEL, "abc123")

    assert "abc123" not in notifier._pending_always_snooze
    assert suppression.state.snoozed_until == {}
    assert suppression.save_called is False


@pytest.mark.asyncio
async def test_confirm_after_cancel_is_noop() -> None:
    """Confirm after Cancel finds no pending intent and must be a no-op."""
    options = {CONF_NOTIFY_SERVICE: "notify.mobile_app_phone"}
    notifier, hass, suppression, action_handler = _make_notifier(options)
    finding = _finding(anomaly_id="abc123")
    action_handler.register_finding(finding)

    await notifier._handle_snooze(ACT_SNOOZE_ALWAYS, "abc123")
    await notifier._handle_snooze(ACT_SNOOZE_CANCEL, "abc123")

    hass.services.calls.clear()
    suppression.save_called = False

    await notifier._handle_snooze(ACT_SNOOZE_CONFIRM, "abc123")

    assert finding.type not in suppression.state.snoozed_until
    assert suppression.save_called is False
    assert hass.services.calls == []


# ---------------------------------------------------------------------------
# 7. End-to-end event-driven paths
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_event_driven_snooze_24h_via_handle_action_event() -> None:
    """End-to-end: a mobile_app_notification_action event for snooze24h writes to suppression."""
    notifier, hass, suppression, action_handler = _make_notifier()
    finding = _finding(anomaly_id="abc123")
    action_handler.register_finding(finding)
    notifier.start()

    action_str = f"{ACTION_PREFIX}{ACT_SNOOZE_24H}_abc123"
    event = DummyEvent({"action": action_str})
    notifier._handle_action_event(event)  # type: ignore[arg-type]

    await hass.drain_tasks()

    assert finding.type in suppression.state.snoozed_until
    assert suppression.save_called is True


@pytest.mark.asyncio
async def test_event_driven_snooze_always_via_handle_action_event() -> None:
    """End-to-end: snoozealways mobile event stores pending intent."""
    options = {CONF_NOTIFY_SERVICE: "notify.mobile_app_phone"}
    notifier, hass, suppression, action_handler = _make_notifier(options)
    finding = _finding(anomaly_id="abc123")
    action_handler.register_finding(finding)
    notifier.start()

    action_str = f"{ACTION_PREFIX}{ACT_SNOOZE_ALWAYS}_abc123"
    event = DummyEvent({"action": action_str})
    notifier._handle_action_event(event)  # type: ignore[arg-type]

    await hass.drain_tasks()

    assert len(hass.services.calls) == 1
    assert "abc123" in notifier._pending_always_snooze
    assert finding.type not in suppression.state.snoozed_until


# ---------------------------------------------------------------------------
# 8. Lifecycle: start / stop
# ---------------------------------------------------------------------------


def test_start_subscribes_to_event_bus() -> None:
    """start() registers an event listener; calling it twice is idempotent."""
    subscribe_calls: list[str] = []

    class TrackingBus:
        def async_listen(self, event_type: str, callback: Any) -> Any:
            subscribe_calls.append(event_type)
            return lambda: None

    hass = DummyHass()
    hass.bus = TrackingBus()  # type: ignore[assignment]
    notifier, *_ = _make_notifier(hass=hass)

    notifier.start()
    notifier.start()  # idempotent

    assert len(subscribe_calls) == 1
    assert subscribe_calls[0] == "mobile_app_notification_action"


def test_stop_unsubscribes_and_is_idempotent() -> None:
    """stop() calls the unsub callback and is safe to call multiple times."""
    unsub_calls: list[int] = []

    class TrackingBus:
        def async_listen(self, event_type: str, callback: Any) -> Any:
            def _unsub() -> None:
                unsub_calls.append(1)

            return _unsub

    hass = DummyHass()
    hass.bus = TrackingBus()  # type: ignore[assignment]
    notifier, *_ = _make_notifier(hass=hass)

    notifier.start()
    notifier.stop()
    notifier.stop()  # idempotent

    assert len(unsub_calls) == 1


# ---------------------------------------------------------------------------
# 9. iOS notification priority tiers
# ---------------------------------------------------------------------------


def _finding_with_severity(
    severity: str,
    anomaly_id: str = "sev1",
    ftype: str = "open_entry_while_away",
) -> AnomalyFinding:
    return AnomalyFinding(
        anomaly_id=anomaly_id,
        type=ftype,
        severity=severity,  # type: ignore[arg-type]
        confidence=0.75,
        triggering_entities=["binary_sensor.front_door"],
        evidence={},
        suggested_actions=["close_entry"],
        is_sensitive=False,
    )


@pytest.mark.asyncio
async def test_async_notify_high_severity_uses_time_sensitive_interruption() -> None:
    """severity=high → push interruption-level == 'time-sensitive', title == 'Security Alert'."""
    options = {CONF_NOTIFY_SERVICE: "notify.mobile_app_phone"}
    notifier, hass, _suppression, _action_handler = _make_notifier(options)
    snapshot = _minimal_snapshot()
    finding = _finding_with_severity("high", anomaly_id="high1")

    await notifier.async_notify(finding, snapshot, "Front door unlocked.")  # type: ignore[arg-type]

    assert len(hass.services.calls) == 1
    call = hass.services.calls[0]
    assert call["data"]["title"] == "Security Alert"
    assert call["data"]["data"]["push"]["interruption-level"] == "time-sensitive"


@pytest.mark.asyncio
async def test_async_notify_low_severity_uses_passive_interruption() -> None:
    """severity=low → push interruption-level == 'passive', title == 'Home Update'."""
    options = {CONF_NOTIFY_SERVICE: "notify.mobile_app_phone"}
    notifier, hass, _suppression, _action_handler = _make_notifier(options)
    snapshot = _minimal_snapshot()
    finding = _finding_with_severity("low", anomaly_id="low1")

    await notifier.async_notify(finding, snapshot, "Appliance finished.")  # type: ignore[arg-type]

    assert len(hass.services.calls) == 1
    call = hass.services.calls[0]
    assert call["data"]["title"] == "Home Update"
    assert call["data"]["data"]["push"]["interruption-level"] == "passive"


@pytest.mark.asyncio
async def test_async_notify_subtitle_is_friendly_type() -> None:
    """data['data']['subtitle'] is a non-empty string derived from finding type."""
    options = {CONF_NOTIFY_SERVICE: "notify.mobile_app_phone"}
    notifier, hass, _suppression, _action_handler = _make_notifier(options)
    snapshot = _minimal_snapshot()
    finding = _finding_with_severity(
        "medium", anomaly_id="sub1", ftype="unlocked_lock_at_night"
    )

    await notifier.async_notify(finding, snapshot, "Lock left unlocked.")  # type: ignore[arg-type]

    assert len(hass.services.calls) == 1
    subtitle = hass.services.calls[0]["data"]["data"]["subtitle"]
    assert isinstance(subtitle, str)
    assert len(subtitle) > 0


# ---------------------------------------------------------------------------
# 10. Notification batching
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_async_notify_batching_holds_after_rate_limit() -> None:
    """Send 4 low-severity notifications; first 3 dispatched; 4th held in _held_batch."""
    options = {CONF_NOTIFY_SERVICE: "notify.mobile_app_phone"}

    cancel_calls: list[int] = []

    def _fake_async_call_later(_hass: Any, _delay: float, _cb: Any) -> Any:
        def _cancel() -> None:
            cancel_calls.append(1)

        return _cancel

    notifier, hass, _suppression, _action_handler = _make_notifier(options)
    snapshot = _minimal_snapshot()

    # Patch async_call_later via the top-level module reference.
    original = _notifier_mod.async_call_later
    _notifier_mod.async_call_later = _fake_async_call_later  # type: ignore[assignment]
    try:
        for i in range(4):
            f = _finding_with_severity("low", anomaly_id=f"batch{i}")
            await notifier.async_notify(f, snapshot, f"msg {i}")  # type: ignore[arg-type]
    finally:
        _notifier_mod.async_call_later = original  # type: ignore[assignment]

    # First 3 dispatched, 4th held.
    assert len(hass.services.calls) == 3
    assert len(notifier._held_batch) == 1


@pytest.mark.asyncio
async def test_async_flush_batch_sends_summary_no_actions() -> None:
    """After _async_flush_batch(), dispatched message has no 'actions' key in data['data']."""
    options = {CONF_NOTIFY_SERVICE: "notify.mobile_app_phone"}
    notifier, hass, _suppression, _action_handler = _make_notifier(options)

    # Pre-load the held batch.
    finding = _finding_with_severity("low", anomaly_id="flush1")
    notifier._held_batch.append((finding, "Some message", "notify.mobile_app_phone"))

    notifier._async_flush_batch()
    await hass.drain_tasks()

    assert len(hass.services.calls) == 1
    call = hass.services.calls[0]
    assert "actions" not in call["data"]["data"]
    assert call["data"]["title"] == "Home Update"


@pytest.mark.asyncio
async def test_high_severity_bypasses_batch() -> None:
    """More than 3 high-severity notifications all dispatched immediately (no batching)."""
    options = {CONF_NOTIFY_SERVICE: "notify.mobile_app_phone"}
    notifier, hass, _suppression, _action_handler = _make_notifier(options)
    snapshot = _minimal_snapshot()

    for i in range(5):
        f = _finding_with_severity("high", anomaly_id=f"high{i}")
        await notifier.async_notify(f, snapshot, f"urgent {i}")  # type: ignore[arg-type]

    # All 5 dispatched immediately; nothing held.
    assert len(hass.services.calls) == 5
    assert len(notifier._held_batch) == 0


# ---------------------------------------------------------------------------
# 13. Per-finding cooldown
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_async_notify_cooldown_suppresses_duplicate() -> None:
    """Second notification for the same anomaly_id within cooldown is suppressed."""
    options = {CONF_NOTIFY_SERVICE: "notify.mobile_app_phone"}
    notifier, hass, _suppression, _action_handler = _make_notifier(options)
    snapshot = _minimal_snapshot()

    f = _finding_with_severity("medium", anomaly_id="fridge_abc")
    await notifier.async_notify(f, snapshot, "Fridge running high.")  # type: ignore[arg-type]
    # Second call with the same anomaly_id should be silently dropped.
    await notifier.async_notify(f, snapshot, "Fridge still running high.")  # type: ignore[arg-type]

    assert len(hass.services.calls) == 1


@pytest.mark.asyncio
async def test_async_notify_cooldown_allows_after_expiry() -> None:
    """Notification fires again once the cooldown entry expires."""
    options = {CONF_NOTIFY_SERVICE: "notify.mobile_app_phone"}
    notifier_obj, hass, _suppression, _action_handler = _make_notifier(options)
    snapshot = _minimal_snapshot()

    f = _finding_with_severity("medium", anomaly_id="fridge_exp")
    await notifier_obj.async_notify(f, snapshot, "First.")  # type: ignore[arg-type]
    assert len(hass.services.calls) == 1

    # Simulate expiry: temporarily zero the cooldown window so the next call passes.
    old_cd = _notifier_mod._FINDING_COOLDOWN_SECS
    _notifier_mod._FINDING_COOLDOWN_SECS = 0  # type: ignore[attr-defined]
    try:
        await notifier_obj.async_notify(f, snapshot, "After expiry.")  # type: ignore[arg-type]
    finally:
        _notifier_mod._FINDING_COOLDOWN_SECS = old_cd  # type: ignore[attr-defined]

    assert len(hass.services.calls) == 2


@pytest.mark.asyncio
async def test_async_notify_cooldown_bypassed_for_high_severity() -> None:
    """High-severity findings always fire even if the same anomaly_id was seen recently."""
    options = {CONF_NOTIFY_SERVICE: "notify.mobile_app_phone"}
    notifier_obj, hass, _suppression, _action_handler = _make_notifier(options)
    snapshot = _minimal_snapshot()

    f = _finding_with_severity("high", anomaly_id="camera_xyz")
    await notifier_obj.async_notify(f, snapshot, "Alert 1.")  # type: ignore[arg-type]
    await notifier_obj.async_notify(f, snapshot, "Alert 2.")  # type: ignore[arg-type]
    await notifier_obj.async_notify(f, snapshot, "Alert 3.")  # type: ignore[arg-type]

    assert len(hass.services.calls) == 3


# ---------------------------------------------------------------------------
# 14. _friendly_type — prefix stripping
# ---------------------------------------------------------------------------


def test_friendly_type_strips_candidate_prefix() -> None:
    """'candidate_foo_bar' should display as 'Foo bar', not 'Candidate foo bar'."""
    assert (
        _friendly_type("candidate_appliance_power_spike_away")
        == "Appliance power spike away"
    )
    assert _friendly_type("candidate_washing_machine_away") == "Washing machine away"
    # Known types are unaffected.
    assert _friendly_type("unlocked_lock_at_night") == "Door lock left unlocked"
    # Non-candidate unknown types still work.
    assert _friendly_type("time_of_day_anomaly") == "Time of day anomaly"


def test_friendly_type_strips_rule_number_prefix() -> None:
    """'rule_NN_...' IDs strip the internal numbering prefix."""
    # The fridge notification bug: "rule_02_high_energy_consumption_away"
    # was showing as "Rule 02 high energy consumption away".
    assert (
        _friendly_type("rule_02_high_energy_consumption_away")
        == "High energy consumption away"
    )
    assert _friendly_type("rule_01_door_open_at_night") == "Door open at night"
    # Multi-digit numbers.
    assert _friendly_type("rule_12_motion_while_away") == "Motion while away"
    # candidate_ + rule_NN_ combined (LLM-proposed numbered rule).
    assert (
        _friendly_type("candidate_rule_03_fridge_power_spike") == "Fridge power spike"
    )
    # No rule_NN prefix — unchanged stripping behaviour.
    assert _friendly_type("rule_custom_check") == "Rule custom check"


# ---------------------------------------------------------------------------
# 14. Appliance completion subtitle
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_async_notify_completion_subtitle_uses_appliance_name() -> None:
    """is_completion=True findings use '[Appliance name] finished' as subtitle."""
    options = {CONF_NOTIFY_SERVICE: "notify.mobile_app_phone"}
    notifier, hass, _suppression, _action_handler = _make_notifier(options)
    snapshot = _minimal_snapshot()
    finding = AnomalyFinding(
        anomaly_id="comp1",
        type="candidate_appliance_power_spike_away",
        severity="low",
        confidence=0.8,
        triggering_entities=["sensor.dishwasher_power"],
        evidence={"is_completion": True, "friendly_name": "Dishwasher Power"},
        suggested_actions=[],
        is_sensitive=False,
    )

    await notifier.async_notify(finding, snapshot, "Dishwasher finished its cycle.")  # type: ignore[arg-type]

    assert len(hass.services.calls) == 1
    subtitle = hass.services.calls[0]["data"]["data"]["subtitle"]
    assert subtitle == "Dishwasher finished"


@pytest.mark.asyncio
async def test_async_notify_non_completion_subtitle_uses_friendly_type() -> None:
    """Without is_completion, subtitle falls back to _friendly_type(finding.type)."""
    options = {CONF_NOTIFY_SERVICE: "notify.mobile_app_phone"}
    notifier, hass, _suppression, _action_handler = _make_notifier(options)
    snapshot = _minimal_snapshot()
    finding = AnomalyFinding(
        anomaly_id="noncomp1",
        type="candidate_appliance_power_spike_away",
        severity="medium",
        confidence=0.8,
        triggering_entities=["sensor.dishwasher_power"],
        evidence={},
        suggested_actions=["check_appliance"],
        is_sensitive=False,
    )

    await notifier.async_notify(finding, snapshot, "Dishwasher may have stopped.")  # type: ignore[arg-type]

    assert len(hass.services.calls) == 1
    subtitle = hass.services.calls[0]["data"]["data"]["subtitle"]
    # "candidate_" stripped → "Appliance power spike away"
    assert subtitle == "Appliance power spike away"


# ---------------------------------------------------------------------------
# Daily digest notification (Item 4)
# ---------------------------------------------------------------------------


class _DummyAuditStore:
    """Minimal AuditStore stub for digest tests."""

    def __init__(self, records: list[dict[str, Any]] | None = None) -> None:
        self._records = records or []

    async def async_get_latest(self, _limit: int) -> list[dict[str, Any]]:
        return list(self._records)


def _notified_record(notified_at: str, severity: str = "medium") -> dict[str, Any]:
    """Build a minimal audit record that counts as a notified finding."""
    return {
        "suppression_reason_code": "not_suppressed",
        "finding": {"severity": severity},
        "notification": {"notified_at": notified_at},
    }


def _suppressed_record(notified_at: str) -> dict[str, Any]:
    """Build a minimal audit record that is suppressed (should not count)."""
    return {
        "suppression_reason_code": "suppressed",
        "finding": {"severity": "low"},
        "notification": {"notified_at": notified_at},
    }


def _make_digest_notifier(
    options: dict[str, Any] | None = None,
    records: list[dict[str, Any]] | None = None,
) -> tuple[SentinelNotifier, DummyHass, _DummyAuditStore]:
    """Create a SentinelNotifier wired with an audit store for digest tests."""
    from custom_components.home_generative_agent.const import (
        CONF_SENTINEL_DAILY_DIGEST_ENABLED,
        CONF_SENTINEL_DAILY_DIGEST_TIME,
    )

    h = DummyHass()
    s = DummySuppressionManager()
    a = DummyActionHandler()
    store = _DummyAuditStore(records)
    opts: dict[str, Any] = {
        CONF_NOTIFY_SERVICE: "notify.mobile_app_phone",
        CONF_SENTINEL_DAILY_DIGEST_ENABLED: True,
        CONF_SENTINEL_DAILY_DIGEST_TIME: "08:00",
    }
    if options:
        opts.update(options)
    notifier = SentinelNotifier(
        hass=h,  # type: ignore[arg-type]
        options=opts,
        suppression=s,  # type: ignore[arg-type]
        action_handler=a,  # type: ignore[arg-type]
        audit_store=store,  # type: ignore[arg-type]
    )
    return notifier, h, store


@pytest.mark.asyncio
async def test_daily_digest_sends_summary_for_notified_findings() -> None:
    """_async_run_daily_digest fires a mobile push with correct count and severity."""
    from homeassistant.util import dt as dt_util

    now_iso = dt_util.utcnow().isoformat()
    records = [
        _notified_record(now_iso, severity="high"),
        _notified_record(now_iso, severity="medium"),
        _notified_record(now_iso, severity="low"),
    ]
    notifier, hass, _store = _make_digest_notifier(records=records)

    await notifier._async_run_daily_digest()

    assert len(hass.services.calls) == 1
    call = hass.services.calls[0]
    assert call["domain"] == "notify"
    assert call["service"] == "mobile_app_phone"
    msg: str = call["data"]["message"]
    assert "3 alerts" in msg
    # All three severities must appear in the summary.
    assert "high" in msg
    assert "medium" in msg
    assert "low" in msg
    assert call["data"]["title"] == "Sentinel Daily Digest"
    assert call["data"]["data"]["tag"] == "hga_sentinel_daily_digest"
    assert call["data"]["data"]["push"]["interruption-level"] == "passive"


@pytest.mark.asyncio
async def test_daily_digest_skips_suppressed_records() -> None:
    """Suppressed findings must not be included in the digest count."""
    from homeassistant.util import dt as dt_util

    now_iso = dt_util.utcnow().isoformat()
    records = [
        _notified_record(now_iso, severity="high"),
        _suppressed_record(now_iso),  # must not count
    ]
    notifier, hass, _store = _make_digest_notifier(records=records)

    await notifier._async_run_daily_digest()

    assert len(hass.services.calls) == 1
    msg: str = hass.services.calls[0]["data"]["message"]
    assert "1 alert" in msg
    assert "2" not in msg.split()[1]  # count word is "1"


@pytest.mark.asyncio
async def test_daily_digest_skips_old_records() -> None:
    """Records with notified_at older than 24 h must be excluded."""
    from datetime import timedelta

    from homeassistant.util import dt as dt_util

    old_iso = (dt_util.utcnow() - timedelta(hours=25)).isoformat()
    recent_iso = dt_util.utcnow().isoformat()
    records = [
        _notified_record(old_iso, severity="high"),  # too old
        _notified_record(recent_iso, severity="medium"),
    ]
    notifier, hass, _store = _make_digest_notifier(records=records)

    await notifier._async_run_daily_digest()

    assert len(hass.services.calls) == 1
    msg: str = hass.services.calls[0]["data"]["message"]
    assert "1 alert" in msg
    assert "high" not in msg


@pytest.mark.asyncio
async def test_daily_digest_no_findings_sends_nothing() -> None:
    """When there are no notified findings in 24 h, no notification is sent."""
    notifier, hass, _store = _make_digest_notifier(records=[])

    await notifier._async_run_daily_digest()

    assert len(hass.services.calls) == 0


@pytest.mark.asyncio
async def test_daily_digest_falls_back_to_persistent_notification() -> None:
    """When CONF_NOTIFY_SERVICE is absent, a persistent_notification is created."""
    from homeassistant.util import dt as dt_util

    from custom_components.home_generative_agent.const import (
        CONF_SENTINEL_DAILY_DIGEST_ENABLED,
        CONF_SENTINEL_DAILY_DIGEST_TIME,
    )

    now_iso = dt_util.utcnow().isoformat()
    records = [_notified_record(now_iso)]
    h = DummyHass()
    s = DummySuppressionManager()
    a = DummyActionHandler()
    store = _DummyAuditStore(records)
    notifier = SentinelNotifier(
        hass=h,  # type: ignore[arg-type]
        options={
            CONF_SENTINEL_DAILY_DIGEST_ENABLED: True,
            CONF_SENTINEL_DAILY_DIGEST_TIME: "08:00",
            # No CONF_NOTIFY_SERVICE
        },
        suppression=s,  # type: ignore[arg-type]
        action_handler=a,  # type: ignore[arg-type]
        audit_store=store,  # type: ignore[arg-type]
    )

    await notifier._async_run_daily_digest()

    assert len(h.services.calls) == 1
    call = h.services.calls[0]
    assert call["domain"] == "persistent_notification"
    assert call["service"] == "create"
    assert call["data"]["notification_id"] == "hga_sentinel_daily_digest"


@pytest.mark.asyncio
async def test_daily_digest_audit_store_none_sends_nothing() -> None:
    """If audit_store is None, _async_run_daily_digest must exit early."""
    notifier, hass, _s, _a = _make_notifier(
        options={CONF_NOTIFY_SERVICE: "notify.mobile_app_phone"}
    )
    # audit_store defaults to None when not passed to _make_notifier.
    await notifier._async_run_daily_digest()

    assert len(hass.services.calls) == 0


def test_daily_digest_start_registers_time_change_when_enabled() -> None:
    """start() must register an async_track_time_change listener when digest is enabled."""
    from unittest.mock import MagicMock, patch

    from custom_components.home_generative_agent.const import (
        CONF_SENTINEL_DAILY_DIGEST_ENABLED,
        CONF_SENTINEL_DAILY_DIGEST_TIME,
    )

    notifier, _hass, _store = _make_digest_notifier(
        options={
            CONF_SENTINEL_DAILY_DIGEST_ENABLED: True,
            CONF_SENTINEL_DAILY_DIGEST_TIME: "07:30",
        },
    )
    unsub_mock = MagicMock()
    with (
        patch(
            "custom_components.home_generative_agent.sentinel.notifier.async_track_time_change",
            return_value=unsub_mock,
        ) as track_mock,
        patch(
            "custom_components.home_generative_agent.sentinel.notifier.async_call_later",
            return_value=MagicMock(),
        ),
    ):
        notifier.start()

    track_mock.assert_called_once()
    _kw = track_mock.call_args.kwargs
    assert _kw["hour"] == 7
    assert _kw["minute"] == 30
    assert _kw["second"] == 0
    assert notifier._digest_unsub is unsub_mock


def test_daily_digest_start_skips_registration_when_disabled() -> None:
    """start() must NOT register async_track_time_change when digest is disabled."""
    from unittest.mock import MagicMock, patch

    from custom_components.home_generative_agent.const import (
        CONF_SENTINEL_DAILY_DIGEST_ENABLED,
    )

    notifier, _hass, _store = _make_digest_notifier(
        options={CONF_SENTINEL_DAILY_DIGEST_ENABLED: False},
    )
    with (
        patch(
            "custom_components.home_generative_agent.sentinel.notifier.async_track_time_change",
        ) as track_mock,
        patch(
            "custom_components.home_generative_agent.sentinel.notifier.async_call_later",
            return_value=MagicMock(),
        ),
    ):
        notifier.start()

    track_mock.assert_not_called()
    assert notifier._digest_unsub is None


def test_daily_digest_stop_cancels_unsub() -> None:
    """stop() must call the unsub callable and clear _digest_unsub."""
    from unittest.mock import MagicMock, patch

    from custom_components.home_generative_agent.const import (
        CONF_SENTINEL_DAILY_DIGEST_ENABLED,
        CONF_SENTINEL_DAILY_DIGEST_TIME,
    )

    notifier, _hass, _store = _make_digest_notifier(
        options={
            CONF_SENTINEL_DAILY_DIGEST_ENABLED: True,
            CONF_SENTINEL_DAILY_DIGEST_TIME: "08:00",
        },
    )
    unsub_mock = MagicMock()
    with (
        patch(
            "custom_components.home_generative_agent.sentinel.notifier.async_track_time_change",
            return_value=unsub_mock,
        ),
        patch(
            "custom_components.home_generative_agent.sentinel.notifier.async_call_later",
            return_value=MagicMock(),
        ),
    ):
        notifier.start()

    notifier.stop()

    unsub_mock.assert_called_once()
    assert notifier._digest_unsub is None


def test_daily_digest_stop_cancels_task() -> None:
    """stop() must cancel a pending digest task and clear _digest_task."""
    from unittest.mock import MagicMock

    notifier, _hass, _store = _make_digest_notifier()
    task_mock = MagicMock()
    notifier._digest_task = task_mock  # inject a fake in-flight task

    notifier.stop()

    task_mock.cancel.assert_called_once()
    assert notifier._digest_task is None


def test_daily_digest_invalid_time_falls_back_to_0800() -> None:
    """A malformed CONF_SENTINEL_DAILY_DIGEST_TIME must fall back to 08:00."""
    from unittest.mock import MagicMock, patch

    from custom_components.home_generative_agent.const import (
        CONF_SENTINEL_DAILY_DIGEST_ENABLED,
        CONF_SENTINEL_DAILY_DIGEST_TIME,
    )

    notifier, _hass, _store = _make_digest_notifier(
        options={
            CONF_SENTINEL_DAILY_DIGEST_ENABLED: True,
            CONF_SENTINEL_DAILY_DIGEST_TIME: "NOT_A_TIME",
        },
    )
    with (
        patch(
            "custom_components.home_generative_agent.sentinel.notifier.async_track_time_change",
            return_value=MagicMock(),
        ) as track_mock,
        patch(
            "custom_components.home_generative_agent.sentinel.notifier.async_call_later",
            return_value=MagicMock(),
        ),
    ):
        notifier.start()

    _kw = track_mock.call_args.kwargs
    assert _kw["hour"] == 8
    assert _kw["minute"] == 0


@pytest.mark.asyncio
async def test_daily_digest_callback_dispatches_coroutine() -> None:
    """_async_send_daily_digest (sync callback) must schedule _async_run_daily_digest."""
    from unittest.mock import AsyncMock, patch

    notifier, hass, _store = _make_digest_notifier(records=[])

    with patch.object(
        notifier,
        "_async_run_daily_digest",
        new_callable=AsyncMock,
    ) as run_mock:
        notifier._async_send_daily_digest()
        await hass.drain_tasks()

    run_mock.assert_awaited_once()

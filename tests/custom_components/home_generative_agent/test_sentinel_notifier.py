# ruff: noqa: S101, PLC0415
"""Tests for SentinelNotifier — sentinel/notifier.py (Issue #261)."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
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
    CONF_SENTINEL_RESPONSE_LANGUAGE,
    SNOOZE_PERMANENT,
)
from custom_components.home_generative_agent.sentinel.models import AnomalyFinding
from custom_components.home_generative_agent.sentinel.notifier import (
    MAX_MOBILE_MESSAGE_CHARS,
    SentinelNotifier,
    _alarm_disarmed_mobile_message,
    _alarm_disarmed_open_entry_mobile_message,
    _appliance_power_duration_mobile_message,
    _baseline_deviation_mobile_message,
    _build_actions,
    _build_subtitle,
    _display_type,
    _entity_staleness_mobile_message,
    _friendly_type,
    _is_power_class_evidence,
    _mobile_message,
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
        # Real HomeAssistant objects always carry a config with a language;
        # model that here so notifier tests exercise the normal language
        # resolution path (cs tests override .language after construction).
        self.config = SimpleNamespace(language="en")
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
    is_sensitive: bool = False,  # noqa: FBT001, FBT002
    recognized_people: list[str] | None = None,
    triggering_entities: list[str] | None = None,
) -> AnomalyFinding:
    evidence: dict[str, Any] = {}
    if recognized_people is not None:
        evidence["recognized_people"] = recognized_people
    return AnomalyFinding(
        anomaly_id=anomaly_id,
        type=ftype,
        severity="medium",
        confidence=0.75,
        triggering_entities=(
            triggering_entities
            if triggering_entities is not None
            else ["binary_sensor.front_door"]
        ),
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


@pytest.mark.asyncio
async def test_snooze_24h_records_feedback_for_each_triggering_entity() -> None:
    """snooze24h increments cooldown multiplier for every triggering entity."""
    notifier, _hass, suppression, action_handler = _make_notifier()
    finding = _finding(
        anomaly_id="abc123",
        ftype="unlocked_lock_at_night",
        triggering_entities=["lock.front", "lock.back"],
    )
    action_handler.register_finding(finding)

    await notifier._handle_snooze(ACT_SNOOZE_24H, "abc123")

    multipliers = suppression.state.learned_cooldown_multipliers
    assert multipliers.get("unlocked_lock_at_night:lock.front", 1) == 2
    assert multipliers.get("unlocked_lock_at_night:lock.back", 1) == 2
    assert suppression.save_called is True


@pytest.mark.asyncio
async def test_snooze_24h_finding_none_no_feedback() -> None:
    """snooze24h with an expired finding must not call record_cooldown_feedback."""
    notifier, _hass, suppression, _action_handler = _make_notifier()
    # No finding registered — simulates an expired/cleaned-up finding.
    await notifier._handle_snooze(ACT_SNOOZE_24H, "gone123")

    assert suppression.state.learned_cooldown_multipliers == {}
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


def test_redact_if_sensitive_ignores_unknown_person_label() -> None:
    """
    The reserved 'Unknown Person' label must never be redacted.

    Unknown-person findings now carry ['Unknown Person'] in evidence;
    substituting it would rewrite 'an unknown person was seen' into
    'a recognised person was seen', inverting the security meaning.
    """
    finding = _finding(is_sensitive=True, recognized_people=["Unknown Person"])
    explanation = "An unknown person was seen in the backyard."

    result = _redact_if_sensitive(explanation, finding)

    assert result == explanation


def test_redact_if_sensitive_ignores_reserved_labels_but_redacts_names() -> None:
    """Reserved labels pass through while enrolled names are still redacted."""
    finding = _finding(
        is_sensitive=True,
        recognized_people=["Indeterminate", "Unknown Person", "John Doe"],
    )
    explanation = "John Doe stood near an unknown person."

    result = _redact_if_sensitive(explanation, finding)

    assert result is not None
    assert "John Doe" not in result
    assert "unknown person" in result
    assert "a recognised person" in result


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


def test_friendly_type_motion_at_night_while_away() -> None:
    """Issue #516: the motion-while-away rule ID gets a clean label."""
    assert (
        _friendly_type("motion_detected_at_night_while_away")
        == "Motion at night while away"
    )


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


def test_friendly_type_alarm_disarmed_external_threat() -> None:
    """alarm_disarmed_during_external_threat gets a clean user-facing label."""
    assert (
        _friendly_type("alarm_disarmed_during_external_threat")
        == "Outdoor activity while alarm disarmed"
    )


def test_friendly_type_open_entry_at_night_variants() -> None:
    """Issue #504: presence-agnostic night rule IDs get the clean entry label."""
    for anomaly_type in (
        "open_entry_at_night",
        "open_entry_at_night_window",
        "open_entry_at_night_door",
        "open_entry_at_night_entry",
    ):
        assert _friendly_type(anomaly_type) == "Open entry at night"


# ---------------------------------------------------------------------------
# 15. alarm_disarmed_during_external_threat — deterministic mobile copy
# ---------------------------------------------------------------------------


def _alarm_finding(
    anomaly_id: str = "adt1",
    evidence: dict[str, Any] | None = None,
) -> AnomalyFinding:
    return AnomalyFinding(
        anomaly_id=anomaly_id,
        type="alarm_disarmed_during_external_threat",
        severity="low",
        confidence=0.9,
        triggering_entities=["alarm_control_panel.home_alarm", "camera.front_door"],
        evidence=evidence or {},
        suggested_actions=["arm_alarm"],
        is_sensitive=False,
    )


def test_build_actions_arm_alarm_label() -> None:
    """suggested_actions containing arm_alarm should yield 'Arm Alarm', not 'Execute'."""
    finding = _alarm_finding()
    actions = _build_actions(finding)
    primary = actions[0]
    assert primary["title"] == "Arm Alarm"
    assert "execute_adt1" in primary["action"]


def test_build_actions_execute_label_for_other_actions() -> None:
    """Findings with other suggested_actions keep the generic 'Execute' label."""
    finding = _finding(ftype="camera_entry_unsecured", anomaly_id="xyz")
    # _finding sets suggested_actions=["close_entry"]
    actions = _build_actions(finding)
    assert actions[0]["title"] == "Execute"


def test_alarm_disarmed_mobile_message_full_evidence() -> None:
    """Full evidence produces a camera-name + age + alarm-time message."""
    ev: dict[str, Any] = {
        "camera_friendly_name": "Front Door",
        "camera_entity_id": "camera.front_door",
        "camera_activity_age_minutes": 3.0,
        "alarm_last_changed": "2025-01-01T18:14:00+00:00",
    }
    msg = _alarm_disarmed_mobile_message(_alarm_finding(evidence=ev))
    assert "Front Door" in msg
    assert "3 min ago" in msg
    assert "disarmed since" in msg
    assert len(msg) <= MAX_MOBILE_MESSAGE_CHARS


def test_alarm_disarmed_mobile_message_missing_timestamps() -> None:
    """
    Missing timestamps drop the age phrase and say 'currently disarmed'.

    Newly generated findings always carry a numeric age (the rule's freshness
    gate guarantees it), but findings persisted before that gate existed can
    still re-render through this branch.
    """
    ev: dict[str, Any] = {
        "camera_friendly_name": "Front Door",
        "camera_entity_id": "camera.front_door",
        "camera_activity_age_minutes": None,
        "alarm_last_changed": None,
    }
    msg = _alarm_disarmed_mobile_message(_alarm_finding(evidence=ev))
    assert "Front Door" in msg
    assert "unrecognized person" in msg
    assert "min ago" not in msg
    assert "currently disarmed" in msg
    assert len(msg) <= MAX_MOBILE_MESSAGE_CHARS


def test_alarm_disarmed_mobile_message_no_camera_name() -> None:
    """Missing camera name falls back to 'A camera'."""
    ev: dict[str, Any] = {
        "camera_friendly_name": None,
        "camera_entity_id": None,
        "camera_activity_age_minutes": 7.0,
        "alarm_last_changed": None,
    }
    msg = _alarm_disarmed_mobile_message(_alarm_finding(evidence=ev))
    assert msg.startswith("A camera")
    assert "7 min ago" in msg
    assert len(msg) <= MAX_MOBILE_MESSAGE_CHARS


def test_alarm_disarmed_mobile_message_no_llm_fallback() -> None:
    """Deterministic builder is used even when an LLM explanation is available."""
    finding = _alarm_finding(
        evidence={
            "camera_friendly_name": "Backyard",
            "camera_activity_age_minutes": 2.0,
            "alarm_last_changed": None,
        }
    )
    llm_explanation = (
        "The home alarm was recently disarmed while someone is still inside."
    )
    msg = _mobile_message(llm_explanation, finding)
    assert "someone is still inside" not in msg
    assert "Backyard" in msg


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
# 15. Appliance power duration deterministic message
# ---------------------------------------------------------------------------


def _appliance_finding(
    anomaly_id: str = "apd1",
    evidence: dict[str, Any] | None = None,
) -> AnomalyFinding:
    return AnomalyFinding(
        anomaly_id=anomaly_id,
        type="appliance_power_duration",
        severity="medium",
        confidence=0.6,
        triggering_entities=["sensor.washer_power"],
        evidence=evidence
        or {
            "entity_id": "sensor.washer_power",
            "area": "Laundry",
            "power_w": 296.0,
            "duration_min": 633,
            "threshold_min": 60,
            "friendly_name": "Washer Power",
        },
        suggested_actions=["check_appliance"],
        is_sensitive=False,
    )


def test_appliance_power_duration_strips_power_suffix() -> None:
    """'Washer Power' friendly_name should render as 'Washer' in the message."""
    msg = _appliance_power_duration_mobile_message(_appliance_finding())
    assert msg.startswith("Washer ")
    assert "Power" not in msg.split()[0]
    assert "296" in msg
    assert "633 min" in msg
    assert "60 min" in msg
    assert len(msg) <= MAX_MOBILE_MESSAGE_CHARS


def test_appliance_power_duration_preserves_casing() -> None:
    """User-authored casing (e.g. 'EV Charger') must not be title-cased away."""
    ev: dict[str, Any] = {
        "entity_id": "sensor.ev_charger_power",
        "area": "Garage",
        "power_w": 7200.0,
        "duration_min": 120,
        "threshold_min": 60,
        "friendly_name": "EV Charger Power",
    }
    msg = _appliance_power_duration_mobile_message(_appliance_finding(evidence=ev))
    assert msg.startswith("EV Charger ")


def test_appliance_power_duration_lowercase_friendly_name() -> None:
    """Lowercase friendly_name suffix should still be stripped correctly."""
    ev: dict[str, Any] = {
        "entity_id": "sensor.washer_power",
        "area": "Laundry",
        "power_w": 250.0,
        "duration_min": 90,
        "threshold_min": 60,
        "friendly_name": "washer power",
    }
    msg = _appliance_power_duration_mobile_message(_appliance_finding(evidence=ev))
    assert msg.startswith("washer ")
    assert "power" not in msg.split()[0].lower() or msg.startswith("washer ")


def test_appliance_power_duration_fallback_to_entity_id() -> None:
    """None and empty-string friendly_name both fall back to the entity ID display name."""
    for name_val in (None, ""):
        ev: dict[str, Any] = {
            "entity_id": "sensor.washer_power",
            "area": "Laundry",
            "power_w": 250.0,
            "duration_min": 90,
            "threshold_min": 60,
            "friendly_name": name_val,
        }
        msg = _appliance_power_duration_mobile_message(_appliance_finding(evidence=ev))
        assert msg.startswith("Washer drew ")
        assert len(msg) <= MAX_MOBILE_MESSAGE_CHARS


def test_appliance_power_duration_mobile_message_wins_over_llm() -> None:
    """Deterministic copy is used even when the LLM explanation mentions 'An appliance'."""
    finding = _appliance_finding()
    llm_explanation = (
        "An appliance in Laundry recently drew about 296 W for 633 minutes,"
        " exceeding the 60-minute threshold. Check the appliance."
    )
    msg = _mobile_message(llm_explanation, finding)
    assert msg.startswith("Washer ")
    assert "An appliance" not in msg


# ---------------------------------------------------------------------------
# entity_staleness mobile message
# ---------------------------------------------------------------------------


def _staleness_finding(
    entity_id: str = "person.lindo_st_angel",
    friendly_name: str | None = "Lindo St Angel",
    age_hours: float = 42.0,
) -> AnomalyFinding:
    return AnomalyFinding(
        anomaly_id="stale-1",
        type="person_tracking_staleness",
        severity="low",
        confidence=0.9,
        triggering_entities=[entity_id],
        evidence={
            "template_id": "entity_staleness",
            "entity_id": entity_id,
            "friendly_name": friendly_name,
            "state": "not_home",
            "max_stale_hours": 24.0,
            "age_hours": age_hours,
        },
        suggested_actions=["check_sensor"],
        is_sensitive=False,
    )


def test_entity_staleness_mobile_message_person_name() -> None:
    """Mobile message includes the person's friendly name."""
    msg = _entity_staleness_mobile_message(_staleness_finding())
    assert "Lindo St Angel" in msg
    assert len(msg) <= MAX_MOBILE_MESSAGE_CHARS


def test_entity_staleness_mobile_message_age_about_1_day() -> None:
    """42-hour staleness rounds to 'about 1 day'."""
    msg = _entity_staleness_mobile_message(_staleness_finding(age_hours=42.0))
    assert "about 1 day" in msg


def test_entity_staleness_mobile_message_age_over_2_days() -> None:
    """50-hour staleness rounds to 'about 2 days'."""
    msg = _entity_staleness_mobile_message(_staleness_finding(age_hours=50.0))
    assert "2 days" in msg


def test_entity_staleness_mobile_message_person_fallback_name() -> None:
    """Falls back to entity_id-derived name when friendly_name is absent."""
    finding = _staleness_finding(friendly_name=None)
    msg = _entity_staleness_mobile_message(finding)
    assert "Lindo St Angel" in msg


def test_entity_staleness_mobile_message_non_person() -> None:
    """Non-person entity uses 'data has been outdated' phrasing."""
    finding = _staleness_finding(
        entity_id="sensor.front_door_battery",
        friendly_name="Front Door Battery",
        age_hours=30.0,
    )
    msg = _entity_staleness_mobile_message(finding)
    assert "Front Door Battery" in msg
    assert "data has been outdated" in msg


def test_entity_staleness_mobile_message_wins_over_llm() -> None:
    """Deterministic copy wins even when an LLM explanation is available."""
    finding = _staleness_finding()
    llm_msg = "The person tracking data has been outdated for about 42 hours."
    msg = _mobile_message(llm_msg, finding)
    assert "Lindo St Angel" in msg
    assert "person tracking data" not in msg


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


def test_daily_digest_start_hhmmss_format_parsed_correctly() -> None:
    """start() must parse 'HH:MM:SS' (3-part) digest time without ValueError."""
    from unittest.mock import MagicMock, patch

    from custom_components.home_generative_agent.const import (
        CONF_SENTINEL_DAILY_DIGEST_ENABLED,
        CONF_SENTINEL_DAILY_DIGEST_TIME,
    )

    notifier, _hass, _store = _make_digest_notifier(
        options={
            CONF_SENTINEL_DAILY_DIGEST_ENABLED: True,
            CONF_SENTINEL_DAILY_DIGEST_TIME: "08:00:00",
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


# ---------------------------------------------------------------------------
# alarm_disarmed_open_entry — mobile message and subtitle
# ---------------------------------------------------------------------------


def _open_entry_finding(
    entry_id: str = "binary_sensor.family_room_right_window",
    alarm_last_changed: str | None = "2025-01-01T22:15:00+00:00",
    entry_last_changed: str | None = "2025-01-01T21:00:00+00:00",
) -> AnomalyFinding:
    return AnomalyFinding(
        anomaly_id="oe1",
        type="alarm_disarmed_open_entry_alarm_control_panel_home_alarm",
        severity="high",
        confidence=0.6,
        triggering_entities=["alarm_control_panel.home_alarm", entry_id],
        evidence={
            "template_id": "alarm_disarmed_open_entry",
            "alarm_entity_id": "alarm_control_panel.home_alarm",
            "entry_entity_id": entry_id,
            "entry_state": "on",
            "alarm_state": "disarmed",
            "entry_last_changed": entry_last_changed,
            "alarm_last_changed": alarm_last_changed,
        },
        suggested_actions=["close_entry"],
        is_sensitive=True,
    )


def test_alarm_disarmed_open_entry_mobile_message_with_alarm_time() -> None:
    """Mobile copy shows entry name and the alarm disarm time, not the entry timestamp."""
    finding = _open_entry_finding(alarm_last_changed="2025-01-01T22:15:00+00:00")
    msg = _alarm_disarmed_open_entry_mobile_message(finding)
    assert "Family Room Right Window" in msg
    assert "disarmed since" in msg
    assert "close" in msg.lower() or "snooze" in msg.lower()
    assert len(msg) <= MAX_MOBILE_MESSAGE_CHARS


def test_alarm_disarmed_open_entry_mobile_message_no_alarm_time() -> None:
    """Falls back gracefully when alarm_last_changed is absent."""
    finding = _open_entry_finding(alarm_last_changed=None)
    msg = _alarm_disarmed_open_entry_mobile_message(finding)
    assert "Family Room Right Window" in msg
    assert "disarmed" in msg
    assert len(msg) <= MAX_MOBILE_MESSAGE_CHARS


def test_alarm_disarmed_open_entry_mobile_message_no_llm_fallback() -> None:
    """Template-id dispatch routes to deterministic builder, not the LLM explanation."""
    finding = _open_entry_finding()
    llm_explanation = "The alarm is disarmed and the window is open."
    msg = _mobile_message(llm_explanation, finding)
    # Deterministic copy must be used — LLM text should NOT appear verbatim.
    assert "window is open" not in msg
    assert "Family Room Right Window" in msg


def test_alarm_disarmed_open_entry_subtitle() -> None:
    """Subtitle shows entry name and rule context, not raw rule id."""
    finding = _open_entry_finding()
    subtitle = _build_subtitle(finding)
    assert "Family Room Right Window" in subtitle
    assert "open" in subtitle.lower()
    assert "alarm disarmed" in subtitle.lower()


def test_display_type_prefers_template_label_for_slug_rule_ids() -> None:
    """Slugified candidate rule IDs display the curated template label."""
    finding = AnomalyFinding(
        anomaly_id="slug-1",
        type="v1_subject_motion_sensor_candidate_slug",
        severity="medium",
        confidence=0.8,
        triggering_entities=["binary_sensor.hall_motion"],
        evidence={"template_id": "motion_detected_at_night_while_away"},
        suggested_actions=["check_camera"],
        is_sensitive=False,
    )
    assert _display_type(finding) == "Motion at night while away"
    # Unknown template IDs keep the prefix-stripped fallback.
    assert _display_type(_finding(ftype="candidate_washing_machine_away")) == (
        "Washing machine away"
    )


def test_friendly_type_motion_while_away() -> None:
    """Issue #518: the day-agnostic away-motion template gets a clean label."""
    assert _friendly_type("motion_detected_while_away") == "Motion while away"
    finding = AnomalyFinding(
        anomaly_id="slug-518",
        type="motion_kitchen_while_away",
        severity="low",
        confidence=0.6,
        triggering_entities=["binary_sensor.xiao_esp32_c5_espectre_motion"],
        evidence={"template_id": "motion_detected_while_away"},
        suggested_actions=["check_camera"],
        is_sensitive=False,
    )
    assert _display_type(finding) == "Motion while away"


# ---------------------------------------------------------------------------
# response_language vs deterministic mobile formatters (PR #523 field report)
# ---------------------------------------------------------------------------


def test_deterministic_mobile_message_wins_without_response_language() -> None:
    """Default behaviour is unchanged: deterministic English copy still wins."""
    finding = _open_entry_finding()
    msg = _mobile_message("The alarm is disarmed and the window is open.", finding)
    assert "window is open" not in msg
    assert "Family Room Right Window" in msg


def test_response_language_lets_translated_explanation_win_template_id() -> None:
    """A configured response language routes template_id findings to the explanation."""
    finding = _staleness_finding()
    explanation = "Sledování polohy je zastaralé už 42 hodin."
    msg = _mobile_message(explanation, finding, "Czech")
    assert msg == explanation


def test_security_copy_never_defers_to_translation_finding_type() -> None:
    """Security copy keeps camera, disarm time, and CTA even under a language."""
    finding = _alarm_finding(
        evidence={
            "camera_friendly_name": "Backyard",
            "camera_activity_age_minutes": 2.0,
            "alarm_last_changed": None,
        }
    )
    explanation = "Alarm byl vypnut, zatímco je někdo stále uvnitř."
    msg = _mobile_message(explanation, finding, "Czech")
    assert msg != explanation
    assert "Backyard" in msg
    assert "stále uvnitř" not in msg


def test_security_copy_never_defers_to_translation_template_id() -> None:
    """The alarm_disarmed_open_entry template is security copy too."""
    finding = _open_entry_finding()
    explanation = "Alarm je vypnutý a okno v obývacím pokoji je otevřené."
    msg = _mobile_message(explanation, finding, "Czech")
    assert msg != explanation
    assert "Family Room Right Window" in msg


def test_response_language_lets_translated_explanation_win_finding_type() -> None:
    """Type-dispatched formatters (no template_id) honour the language too."""
    finding = _appliance_finding()
    explanation = "Pračka v prádelně běží už 633 minut. Zkontrolujte spotřebič."
    msg = _mobile_message(explanation, finding, "Czech")
    assert msg == explanation
    assert not msg.startswith("Washer ")


def test_response_language_falls_back_to_deterministic_when_no_explanation() -> None:
    """No translation available → accurate English, not the generic fallback."""
    finding = _staleness_finding()
    msg = _mobile_message(None, finding, "Czech")
    assert "Lindo St Angel" in msg
    assert "data has been outdated" in msg or "location tracking" in msg


def test_response_language_falls_back_to_deterministic_when_explanation_too_long() -> (
    None
):
    """
    Over-cap text falls back to the deterministic string, not _fallback_message.

    In production LLMExplainer caps output at the same 220 characters and
    returns None rather than an English compact fallback when a language is
    set, so this is the notifier's own belt-and-braces guard.
    """
    finding = _staleness_finding()
    long_explanation = "Č" * (MAX_MOBILE_MESSAGE_CHARS + 1)
    msg = _mobile_message(long_explanation, finding, "Czech")
    assert long_explanation not in msg
    assert "Lindo St Angel" in msg
    assert len(msg) <= MAX_MOBILE_MESSAGE_CHARS


def test_untemplated_finding_still_falls_back_to_generic_message() -> None:
    """Findings with no deterministic formatter keep the generic English fallback."""
    finding = _finding(anomaly_id="nofmt")
    msg = _mobile_message(None, finding, "Czech")
    assert (
        msg
        == _notifier_mod._fallback_message(finding)[:MAX_MOBILE_MESSAGE_CHARS].rstrip()
    )


@pytest.mark.asyncio
async def test_async_notify_mobile_and_persistent_agree_under_response_language() -> (
    None
):
    """The mobile push must not be English while the persistent text is translated."""
    options = {
        CONF_NOTIFY_SERVICE: "notify.mobile_app_phone",
        CONF_SENTINEL_RESPONSE_LANGUAGE: "Czech",
    }
    notifier, hass, _suppression, _action_handler = _make_notifier(options)
    explanation = "Pračka v prádelně běží už 633 minut. Zkontrolujte spotřebič."

    await notifier.async_notify(_appliance_finding(), _minimal_snapshot(), explanation)  # type: ignore[arg-type]

    assert len(hass.services.calls) == 1
    assert hass.services.calls[0]["data"]["message"] == explanation


@pytest.mark.asyncio
async def test_async_notify_keeps_deterministic_copy_without_response_language() -> (
    None
):
    """Unset language keeps the exact-figure English copy on the mobile push."""
    options = {CONF_NOTIFY_SERVICE: "notify.mobile_app_phone"}
    notifier, hass, _suppression, _action_handler = _make_notifier(options)
    explanation = "An appliance in Laundry recently drew about 296 W."

    await notifier.async_notify(_appliance_finding(), _minimal_snapshot(), explanation)  # type: ignore[arg-type]

    message = hass.services.calls[0]["data"]["message"]
    assert message.startswith("Washer ")
    assert "An appliance" not in message


@pytest.mark.asyncio
async def test_held_batch_stores_redacted_explanation() -> None:
    """Buffered findings must hold the redacted text, never the raw explanation."""
    options = {CONF_NOTIFY_SERVICE: "notify.mobile_app_phone"}
    notifier, _hass, _suppression, _action_handler = _make_notifier(options)
    snapshot = _minimal_snapshot()

    def _fake_async_call_later(_hass: Any, _delay: float, _cb: Any) -> Any:
        return lambda: None

    original = _notifier_mod.async_call_later
    _notifier_mod.async_call_later = _fake_async_call_later  # type: ignore[assignment]
    try:
        # Fill the rate-limit window so the next finding is buffered, not sent.
        for i in range(3):
            await notifier.async_notify(_finding(anomaly_id=f"warm{i}"), snapshot, None)  # type: ignore[arg-type]

        sensitive = _finding(
            anomaly_id="held-sens", is_sensitive=True, recognized_people=["John Doe"]
        )
        explanation = "John Doe was seen near the front door."
        await notifier.async_notify(sensitive, snapshot, explanation)  # type: ignore[arg-type]
    finally:
        _notifier_mod.async_call_later = original  # type: ignore[assignment]

    assert len(notifier._held_batch) == 1
    held_explanation = notifier._held_batch[0][1]
    assert held_explanation is not None
    assert "John Doe" not in held_explanation
    assert "a recognised person" in held_explanation


# ---------------------------------------------------------------------------
# 22. Localized notification chrome (PR #565)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_async_notify_czech_title_and_subtitle() -> None:
    """
    A cs-configured hass produces Czech title and type-label subtitle.

    This is the only test shape that fails if the hass threading is dropped
    at a call site (every localized helper defaults hass=None → English).
    """
    options = {CONF_NOTIFY_SERVICE: "notify.mobile_app_phone"}
    notifier, hass, _suppression, _action_handler = _make_notifier(options)
    hass.config.language = "cs"
    snapshot = _minimal_snapshot()
    finding = _finding_with_severity(
        "high", anomaly_id="cs1", ftype="unlocked_lock_at_night"
    )

    await notifier.async_notify(finding, snapshot, None)  # type: ignore[arg-type]

    assert len(hass.services.calls) == 1
    call = hass.services.calls[0]
    assert call["data"]["title"] == "Bezpečnostní výstraha"
    assert call["data"]["data"]["subtitle"] == "Zámek ponechán odemčený"


@pytest.mark.asyncio
async def test_security_body_stays_english_under_czech_hass() -> None:
    """
    Deterministic security copy is never localized (PR #531 invariant).

    Even against the new vector — hass itself Czech-configured — the body
    stays English, while the chrome (title) around it IS Czech in the same
    dispatch.
    """
    options = {
        CONF_NOTIFY_SERVICE: "notify.mobile_app_phone",
        CONF_SENTINEL_RESPONSE_LANGUAGE: "Czech",
    }
    notifier, hass, _suppression, _action_handler = _make_notifier(options)
    hass.config.language = "cs"
    snapshot = _minimal_snapshot()
    finding = _alarm_finding(
        anomaly_id="cssec1",
        evidence={
            "camera_friendly_name": "Backyard",
            "camera_activity_age_minutes": 2.0,
            "alarm_last_changed": None,
        },
    )
    explanation = "Alarm byl vypnut, zatímco je někdo stále uvnitř."

    await notifier.async_notify(finding, snapshot, explanation)  # type: ignore[arg-type]

    assert len(hass.services.calls) == 1
    call = hass.services.calls[0]
    # Chrome is Czech (alarm finding is low severity → low title).
    assert call["data"]["title"] == "Novinka z domova"
    # Body keeps the exact English deterministic security copy.
    body: str = call["data"]["message"]
    assert body != explanation
    assert "Backyard" in body
    assert "stále uvnitř" not in body


@pytest.mark.asyncio
async def test_flush_batch_czech_title_and_type_label() -> None:
    """Batch summary title and type labels localize under a cs hass."""
    options = {CONF_NOTIFY_SERVICE: "notify.mobile_app_phone"}
    notifier, hass, _suppression, _action_handler = _make_notifier(options)
    hass.config.language = "cs"
    finding = _finding_with_severity(
        "low", anomaly_id="csflush1", ftype="motion_detected_while_away"
    )
    notifier._held_batch.append((finding, "Some message", "notify.mobile_app_phone"))

    notifier._async_flush_batch()
    await hass.drain_tasks()

    assert len(hass.services.calls) == 1
    call = hass.services.calls[0]
    assert call["data"]["title"] == "Novinka z domova"
    assert "Pohyb v nepřítomnosti" in call["data"]["message"]


@pytest.mark.asyncio
async def test_daily_digest_czech_title_and_severity_words() -> None:
    """Digest title and severity words localize under a cs hass."""
    from homeassistant.util import dt as dt_util

    now_iso = dt_util.utcnow().isoformat()
    records = [
        _notified_record(now_iso, severity="high"),
        _notified_record(now_iso, severity="low"),
    ]
    notifier, hass, _store = _make_digest_notifier(records=records)
    hass.config.language = "cs"

    await notifier._async_run_daily_digest()

    assert len(hass.services.calls) == 1
    call = hass.services.calls[0]
    assert call["data"]["title"] == "Denní přehled Sentinelu"
    msg: str = call["data"]["message"]
    assert "vysoká" in msg
    assert "nízká" in msg
    assert "za posledních 24 h" in msg


@pytest.mark.asyncio
async def test_snooze_confirmation_czech() -> None:
    """The permanent-snooze confirmation localizes title, message, and label."""
    options = {CONF_NOTIFY_SERVICE: "notify.mobile_app_phone"}
    notifier, hass, _suppression, action_handler = _make_notifier(options)
    hass.config.language = "cs"
    finding = _finding(anomaly_id="cssnooze1")
    action_handler.register_finding(finding)

    await notifier._handle_snooze(ACT_SNOOZE_ALWAYS, "cssnooze1")

    assert len(hass.services.calls) == 1
    call = hass.services.calls[0]
    assert call["data"]["title"] == "Potvrdit trvalé ztlumení"
    # The friendly type label inside the message is Czech too.
    assert "Otevřený vstup v nepřítomnosti" in call["data"]["message"]


@pytest.mark.asyncio
async def test_unknown_severity_falls_back_to_medium_title() -> None:
    """An unrecognized severity maps to the medium title in both languages."""
    options = {CONF_NOTIFY_SERVICE: "notify.mobile_app_phone"}
    for language, expected in (("en", "Home Alert"), ("cs", "Upozornění z domova")):
        notifier, hass, _suppression, _action_handler = _make_notifier(options)
        hass.config.language = language
        snapshot = _minimal_snapshot()
        finding = _finding_with_severity("critical", anomaly_id=f"sevx-{language}")

        await notifier.async_notify(finding, snapshot, "msg")  # type: ignore[arg-type]

        assert hass.services.calls[0]["data"]["title"] == expected


@pytest.mark.asyncio
async def test_daily_digest_survives_malformed_records() -> None:
    """A null/non-dict finding or None severity must not kill the digest."""
    from homeassistant.util import dt as dt_util

    now_iso = dt_util.utcnow().isoformat()
    records = [
        _notified_record(now_iso, severity="high"),
        {
            "suppression_reason_code": "not_suppressed",
            "finding": None,
            "notification": {"notified_at": now_iso},
        },
        {
            "suppression_reason_code": "not_suppressed",
            "finding": "corrupt",
            "notification": {"notified_at": now_iso},
        },
        {
            "suppression_reason_code": "not_suppressed",
            "finding": {"severity": None},
            "notification": {"notified_at": now_iso},
        },
    ]
    notifier, hass, _store = _make_digest_notifier(records=records)

    await notifier._async_run_daily_digest()

    assert len(hass.services.calls) == 1
    msg: str = hass.services.calls[0]["data"]["message"]
    assert "4 alerts" in msg
    assert "3 unknown" in msg


@pytest.mark.asyncio
async def test_daily_digest_never_treats_stored_severity_as_message_key() -> None:
    """
    A corrupted severity colliding with a message id must render verbatim.

    Routing stored data through notif_msg as a key would surface unrelated
    copy (e.g. the action-hint sentence) inside the digest summary.
    """
    from homeassistant.util import dt as dt_util

    now_iso = dt_util.utcnow().isoformat()
    records = [_notified_record(now_iso, severity="action_hint_high")]
    notifier, hass, _store = _make_digest_notifier(records=records)

    await notifier._async_run_daily_digest()

    msg: str = hass.services.calls[0]["data"]["message"]
    assert "1 action_hint_high" in msg
    assert "Urgent" not in msg


def test_persistent_fallback_localizes_severity_word() -> None:
    """The persistent fallback renders the localized severity word, not raw."""
    finding = _finding(anomaly_id="persloc1")  # medium severity
    cs_hass = SimpleNamespace(config=SimpleNamespace(language="cs"))
    msg = _notifier_mod._persistent_message(None, finding, cs_hass)  # type: ignore[arg-type]
    assert "závažnost střední" in msg
    en_msg = _notifier_mod._persistent_message(None, finding, None)
    assert "(severity medium)" in en_msg


# ---------------------------------------------------------------------------
# baseline_deviation / time_of_day_anomaly — metric-aware copy
# ---------------------------------------------------------------------------


def _baseline_finding(
    evidence: dict[str, Any],
    anomaly_type: str = "baseline_deviation",
) -> AnomalyFinding:
    return AnomalyFinding(
        anomaly_id="bl1",
        type=anomaly_type,
        severity="low",
        confidence=0.7,
        triggering_entities=[str(evidence.get("entity_id") or "sensor.x")],
        evidence=evidence,
        suggested_actions=[],
        is_sensitive=False,
    )


def _humidity_evidence(**overrides: Any) -> dict[str, Any]:
    ev: dict[str, Any] = {
        "template_id": "baseline_deviation",
        "entity_id": "sensor.playroom_attic_humidity",
        "friendly_name": "Playroom Attic Humidity",
        "unit_of_measurement": "%",
        "device_class": "humidity",
        "current_value": 65.0,
        "baseline_value": 49.2,
        "deviation_pct": 32.0,
        "deviation_direction": "above",
    }
    ev.update(overrides)
    return ev


def _power_evidence(**overrides: Any) -> dict[str, Any]:
    ev: dict[str, Any] = {
        "template_id": "baseline_deviation",
        "entity_id": "sensor.dishwasher_power",
        "friendly_name": "Dishwasher Power",
        "unit_of_measurement": "W",
        "device_class": "power",
        "current_value": 1200.0,
        "baseline_value": 400.0,
        "deviation_pct": 200.0,
        "deviation_direction": "above",
    }
    ev.update(overrides)
    return ev


def test_baseline_mobile_message_humidity_never_says_power() -> None:
    """A humidity finding renders its unit and neutral copy, not power wording."""
    finding = _baseline_finding(_humidity_evidence())
    msg = _baseline_deviation_mobile_message(finding)
    assert "power" not in msg.lower()
    assert "Check appliance" not in msg
    assert "65.0% vs usual 49.2%" in msg
    assert "(32% above normal)" in msg
    assert len(msg) <= MAX_MOBILE_MESSAGE_CHARS


def test_baseline_mobile_message_power_copy_unchanged() -> None:
    """Power findings keep the appliance wording and W unit."""
    finding = _baseline_finding(_power_evidence())
    msg = _baseline_deviation_mobile_message(finding)
    assert "1200.0W vs usual 400.0W" in msg
    assert "Check appliance." in msg


def test_baseline_mobile_message_dow_expected_value_renders_values() -> None:
    """
    DOW time_of_day_anomaly findings carry expected_value, not baseline_value.

    Regression: these previously fell through to the value-less
    "power N% above normal" fallback for every sensor class.
    """
    ev = _humidity_evidence(template_id="time_of_day_anomaly")
    del ev["baseline_value"]
    ev["expected_value"] = 49.2
    finding = _baseline_finding(ev, anomaly_type="time_of_day_anomaly")
    msg = _baseline_deviation_mobile_message(finding)
    assert "65.0% vs usual 49.2%" in msg
    assert "power" not in msg.lower()


def test_baseline_mobile_message_valueless_fallback_is_metric_aware() -> None:
    """Without any comparison value, non-power copy says reading, not power."""
    ev = _humidity_evidence()
    del ev["baseline_value"]
    del ev["current_value"]
    finding = _baseline_finding(ev)
    msg = _baseline_deviation_mobile_message(finding)
    assert msg == "Playroom Attic Humidity reading 32% above normal. Worth checking."

    ev_power = _power_evidence()
    del ev_power["baseline_value"]
    del ev_power["current_value"]
    msg_power = _baseline_deviation_mobile_message(_baseline_finding(ev_power))
    assert "power 200% above normal. Check appliance." in msg_power


def test_baseline_mobile_message_legacy_evidence_keeps_power_copy() -> None:
    """
    Findings persisted before unit/device_class were captured still render.

    The entity_id substring heuristic preserves the historical power wording
    and inferred W unit for power entities.
    """
    ev = _power_evidence()
    del ev["unit_of_measurement"]
    del ev["device_class"]
    finding = _baseline_finding(ev)
    msg = _baseline_deviation_mobile_message(finding)
    assert "1200.0W vs usual 400.0W" in msg
    assert "Check appliance." in msg


def test_baseline_mobile_message_legacy_non_power_evidence_is_neutral() -> None:
    """Legacy findings for non-power entities render unitless neutral copy."""
    ev = _humidity_evidence()
    del ev["unit_of_measurement"]
    del ev["device_class"]
    finding = _baseline_finding(ev)
    msg = _baseline_deviation_mobile_message(finding)
    assert "power" not in msg.lower()
    assert "65.0 vs usual 49.2" in msg


def test_baseline_subtitle_humidity_says_reading() -> None:
    """Non-power baseline findings get the reading-deviation subtitle."""
    finding = _baseline_finding(_humidity_evidence())
    subtitle = _build_subtitle(finding)
    assert subtitle == "Playroom Attic Humidity: reading higher than expected"


def test_baseline_subtitle_power_says_power() -> None:
    """Power baseline findings keep the power-deviation subtitle."""
    finding = _baseline_finding(_power_evidence())
    subtitle = _build_subtitle(finding)
    assert subtitle == "Dishwasher: power higher than expected"


def test_baseline_subtitle_czech_reading_deviation() -> None:
    """A cs-configured hass localizes the reading-deviation subtitle."""
    finding = _baseline_finding(_humidity_evidence())
    cs_hass = SimpleNamespace(config=SimpleNamespace(language="cs"))
    subtitle = _build_subtitle(finding, cs_hass)  # type: ignore[arg-type]
    assert subtitle == "Playroom Attic Humidity: hodnota vyšší, než se čekalo"


def test_is_power_class_evidence_classification() -> None:
    """device_class wins; unit decides next; entity_id is the legacy fallback."""
    assert _is_power_class_evidence({"device_class": "power"})
    assert _is_power_class_evidence({"device_class": "energy"})
    assert not _is_power_class_evidence(
        {"device_class": "humidity", "entity_id": "sensor.solar_power_humidity"}
    )
    assert _is_power_class_evidence({"unit_of_measurement": "W"})
    assert _is_power_class_evidence({"unit_of_measurement": "kWh"})
    assert not _is_power_class_evidence({"unit_of_measurement": "%"})
    assert _is_power_class_evidence({"entity_id": "sensor.dishwasher_power"})
    assert _is_power_class_evidence({"entity_id": "sensor.home_energy"})
    assert not _is_power_class_evidence({"entity_id": "sensor.attic_humidity"})


def test_is_power_class_evidence_unit_gates_entity_id_fallback() -> None:
    """A non-power unit decides False; the entity_id substring never runs."""
    assert not _is_power_class_evidence(
        {"unit_of_measurement": "%", "entity_id": "sensor.dishwasher_power"}
    )


def test_baseline_mobile_message_legacy_energy_evidence_infers_kwh() -> None:
    """Legacy energy findings (no unit/device_class) infer kWh from entity_id."""
    ev = _power_evidence(
        entity_id="sensor.home_energy",
        friendly_name="Home Energy",
        current_value=12.5,
        baseline_value=5.0,
        deviation_pct=150.0,
    )
    del ev["unit_of_measurement"]
    del ev["device_class"]
    finding = _baseline_finding(ev)
    msg = _baseline_deviation_mobile_message(finding)
    assert "12.5kWh vs usual 5.0kWh" in msg
    assert "Check appliance." in msg


def test_baseline_mobile_message_non_finite_evidence_never_raises() -> None:
    """
    NaN/inf values in persisted evidence must degrade, not crash dispatch.

    Regression: round(float('nan')) raised ValueError inside the notifier,
    which killed the Sentinel run loop until reload (red-team finding).
    """
    ev = _humidity_evidence(
        current_value=float("nan"),
        baseline_value=float("inf"),
        deviation_pct=float("nan"),
    )
    msg = _baseline_deviation_mobile_message(_baseline_finding(ev))
    assert msg == "Playroom Attic Humidity reading above normal. Worth checking."

    ev_str = _humidity_evidence(deviation_pct="not-a-number")
    msg_str = _baseline_deviation_mobile_message(_baseline_finding(ev_str))
    assert "65.0% vs usual 49.2%" in msg_str
    assert "not-a-number" not in msg_str


def test_baseline_mobile_message_new_unitless_sensor_no_fabricated_unit() -> None:
    """
    A new finding with a present-but-empty unit gets no fabricated W/kWh.

    Key ABSENCE marks legacy evidence; sensor.energy_score with captured
    empty metadata is a unitless score, not an energy circuit.
    """
    ev = _humidity_evidence(
        entity_id="sensor.energy_score",
        friendly_name="Energy Score",
        unit_of_measurement="",
        device_class="",
    )
    msg = _baseline_deviation_mobile_message(_baseline_finding(ev))
    assert "kWh" not in msg
    assert "65.0 vs usual 49.2" in msg
    assert "power" not in msg.lower()
    assert "Worth checking." in msg


def test_baseline_mobile_message_unitless_power_sensor_keeps_power_copy() -> None:
    """device_class=power with an empty captured unit: power copy, no unit."""
    ev = _power_evidence(unit_of_measurement="")
    msg = _baseline_deviation_mobile_message(_baseline_finding(ev))
    assert "1200.0 vs usual 400.0" in msg
    assert "Check appliance." in msg


def test_is_power_class_evidence_unit_wins_over_exotic_device_class() -> None:
    """A power-dimension unit classifies even under a non-power device_class."""
    assert _is_power_class_evidence(
        {"device_class": "energy_storage", "unit_of_measurement": "kWh"}
    )
    assert _is_power_class_evidence(
        {"device_class": "Power", "unit_of_measurement": "W"}
    )
    assert not _is_power_class_evidence(
        {
            "device_class": "",
            "unit_of_measurement": "",
            "entity_id": "sensor.energy_score",
        }
    )


def test_baseline_mobile_message_unit_control_chars_stripped() -> None:
    """Bidi/format control characters in the unit never reach the push text."""
    rlo = "\u202e"  # right-to-left override (bidi spoofing)
    zwsp = "\u200b"  # zero-width space
    ev = _power_evidence(unit_of_measurement=f"{rlo}W evil{zwsp}")
    msg = _baseline_deviation_mobile_message(_baseline_finding(ev))
    assert rlo not in msg
    assert zwsp not in msg
    assert "W evil" in msg

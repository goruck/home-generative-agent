# ruff: noqa: S101
"""Tests for SentinelExecutionService guardrails — Issue #259."""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any, cast

import pytest
from homeassistant.util import dt as dt_util

from custom_components.home_generative_agent.const import (
    ACTION_POLICY_AUTO_EXECUTE,
    ACTION_POLICY_PROMPT_USER,
    CONF_SENTINEL_AUTO_EXECUTE_ALLOWED_SERVICES,
    CONF_SENTINEL_AUTO_EXECUTE_MAX_ACTIONS_PER_HOUR,
    CONF_SENTINEL_AUTO_EXECUTION_ENABLED,
    CONF_SENTINEL_EXECUTION_IDEMPOTENCY_WINDOW_MINUTES,
    CONF_SENTINEL_STALENESS_THRESHOLD_SECONDS,
    DATA_QUALITY_FRESH,
    DATA_QUALITY_STALE,
    DATA_QUALITY_UNAVAILABLE,
)
from custom_components.home_generative_agent.sentinel.execution import (
    SentinelExecutionService,
)
from custom_components.home_generative_agent.sentinel.models import (
    AnomalyFinding,
    Severity,
)
from custom_components.home_generative_agent.snapshot.schema import (
    FullStateSnapshot,
    validate_snapshot,
)

from .sentinel_actions import service_actions

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_finding(  # noqa: PLR0913
    *,
    anomaly_id: str = "a1",
    severity: Severity = "medium",
    confidence: float = 0.9,
    triggering_entities: list[str] | None = None,
    is_sensitive: bool = False,
    suggested_actions: list[str] | None = None,
) -> AnomalyFinding:
    return AnomalyFinding(
        anomaly_id=anomaly_id,
        type="open_entry_while_away",
        severity=severity,
        confidence=confidence,
        triggering_entities=triggering_entities or ["binary_sensor.front_door"],
        evidence={},
        suggested_actions=suggested_actions or [],
        is_sensitive=is_sensitive,
        actions=service_actions(
            suggested_actions, triggering_entities or ["binary_sensor.front_door"]
        ),
    )


def _make_snapshot(
    *,
    entity_id: str = "binary_sensor.front_door",
    last_changed: str = "2025-01-01T00:00:00+00:00",
) -> FullStateSnapshot:
    return validate_snapshot(
        {
            "schema_version": 1,
            "generated_at": "2025-01-01T00:00:00+00:00",
            "entities": [
                {
                    "entity_id": entity_id,
                    "domain": "binary_sensor",
                    "state": "on",
                    "friendly_name": "Front Door",
                    "area": None,
                    "attributes": {"device_class": "door"},
                    "last_changed": last_changed,
                    "last_updated": last_changed,
                }
            ],
            "camera_activity": [],
            "derived": {
                "now": "2025-01-01T00:00:00+00:00",
                "timezone": "UTC",
                "is_night": False,
                "anyone_home": False,
                "people_home": [],
                "people_away": [],
                "last_motion_by_area": {},
            },
        }
    )


_NOW: datetime = cast("datetime", dt_util.parse_datetime("2025-01-01T01:00:00+00:00"))

_FULL_OPTIONS: dict[str, Any] = {
    CONF_SENTINEL_AUTO_EXECUTION_ENABLED: True,
    CONF_SENTINEL_AUTO_EXECUTE_MAX_ACTIONS_PER_HOUR: 5,
    CONF_SENTINEL_AUTO_EXECUTE_ALLOWED_SERVICES: ["light.turn_off"],
    CONF_SENTINEL_STALENESS_THRESHOLD_SECONDS: 3600,
}


# ---------------------------------------------------------------------------
# 1. Autonomy level gate
# ---------------------------------------------------------------------------


def test_autonomy_level_below_2_returns_prompt_user() -> None:
    svc = SentinelExecutionService({})
    finding = _make_finding()
    snapshot = _make_snapshot()

    for level in (0, 1):
        result = svc.evaluate_canary(finding, snapshot, level, _NOW)
        assert result.action_policy_path == ACTION_POLICY_PROMPT_USER
        assert result.block_reason == "autonomy_level_below_2"


# ---------------------------------------------------------------------------
# 2. Data quality gate — stale
# ---------------------------------------------------------------------------


def test_stale_entity_returns_prompt_user_with_stale_quality() -> None:
    """Entity last changed >threshold seconds ago → stale → prompt_user."""
    # Entity changed at 00:00, now is 01:00, threshold 300 s → stale
    opts = {
        **_FULL_OPTIONS,
        CONF_SENTINEL_STALENESS_THRESHOLD_SECONDS: 300,
    }
    svc = SentinelExecutionService(opts)
    # Snapshot entity last_changed is 00:00:00, _NOW is 01:00:00 → 3600 s old
    snapshot = _make_snapshot(last_changed="2025-01-01T00:00:00+00:00")
    finding = _make_finding(suggested_actions=["light.turn_off"])

    result = svc.evaluate_canary(finding, snapshot, 2, _NOW)

    assert result.data_quality == DATA_QUALITY_STALE
    assert result.action_policy_path == ACTION_POLICY_PROMPT_USER
    assert result.block_reason == "data_quality_stale"


# ---------------------------------------------------------------------------
# 2b. Data quality gate — unavailable
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("severity", ["low", "medium", "high"])
def test_a_missing_sensor_stops_the_action_and_never_the_alert(
    severity: Severity,
) -> None:
    """Entity not in the snapshot: the action does not run, the user is told."""
    svc = SentinelExecutionService(_FULL_OPTIONS)
    snapshot = _make_snapshot()  # has binary_sensor.front_door
    finding = _make_finding(
        triggering_entities=["lock.missing_lock"],  # not in snapshot
        severity=severity,
        suggested_actions=["light.turn_off"],
    )

    result = svc.evaluate_canary(finding, snapshot, 2, _NOW)

    assert result.data_quality == DATA_QUALITY_UNAVAILABLE
    assert result.action_policy_path == ACTION_POLICY_PROMPT_USER
    assert result.block_reason == "data_quality_unavailable"


# ---------------------------------------------------------------------------
# 3. What no longer gates an action (#789)
# ---------------------------------------------------------------------------


def test_a_sensitive_finding_with_an_action_still_runs() -> None:
    """`is_sensitive` governs what text is shared, not whether a rule may act."""
    snapshot = _make_snapshot(last_changed="2025-01-01T00:59:50+00:00")
    svc = SentinelExecutionService(_FULL_OPTIONS)
    finding = _make_finding(is_sensitive=True, suggested_actions=["light.turn_off"])

    result = svc.evaluate_canary(finding, snapshot, 2, _NOW)

    assert result.action_policy_path == ACTION_POLICY_AUTO_EXECUTE


def test_confidence_does_not_gate_an_action() -> None:
    """A model picks a discovered rule's confidence; it decides nothing here."""
    snapshot = _make_snapshot(last_changed="2025-01-01T00:59:50+00:00")
    svc = SentinelExecutionService(_FULL_OPTIONS)
    finding = _make_finding(confidence=0.0, suggested_actions=["light.turn_off"])

    result = svc.evaluate_canary(finding, snapshot, 2, _NOW)

    assert result.action_policy_path == ACTION_POLICY_AUTO_EXECUTE


def test_a_finding_that_cannot_act_is_never_dropped_for_its_data() -> None:
    """An alert about a sensor that is gone must still reach the user."""
    snapshot = _make_snapshot(entity_id="binary_sensor.other")
    svc = SentinelExecutionService(_FULL_OPTIONS)
    finding = _make_finding(severity="low")

    result = svc.evaluate_canary(finding, snapshot, 2, _NOW)

    assert result.action_policy_path == ACTION_POLICY_PROMPT_USER
    assert result.block_reason == "no_executable_action"


# ---------------------------------------------------------------------------
# 4. Auto-execution disabled gate
# ---------------------------------------------------------------------------


def test_auto_execution_disabled_returns_prompt_user() -> None:
    opts = {**_FULL_OPTIONS, CONF_SENTINEL_AUTO_EXECUTION_ENABLED: False}
    svc = SentinelExecutionService(opts)
    recent = "2025-01-01T00:59:50+00:00"
    snapshot = _make_snapshot(last_changed=recent)
    finding = _make_finding()

    result = svc.evaluate_canary(finding, snapshot, 2, _NOW)

    assert result.action_policy_path == ACTION_POLICY_PROMPT_USER
    assert result.block_reason == "auto_execution_disabled"


# ---------------------------------------------------------------------------
# 6. Allowlist gate
# ---------------------------------------------------------------------------


def test_service_not_allowlisted_returns_prompt_user() -> None:
    opts = {
        **_FULL_OPTIONS,
        CONF_SENTINEL_AUTO_EXECUTE_ALLOWED_SERVICES: ["light.turn_off"],
    }
    svc = SentinelExecutionService(opts)
    recent = "2025-01-01T00:59:50+00:00"
    snapshot = _make_snapshot(last_changed=recent)
    finding = _make_finding(
        suggested_actions=["lock.lock"]  # not allowlisted
    )

    result = svc.evaluate_canary(finding, snapshot, 2, _NOW)

    assert result.action_policy_path == ACTION_POLICY_PROMPT_USER
    assert result.block_reason == "service_not_allowlisted"


def test_empty_allowlist_blocks_when_actions_present() -> None:
    opts = {
        **_FULL_OPTIONS,
        CONF_SENTINEL_AUTO_EXECUTE_ALLOWED_SERVICES: [],
    }
    svc = SentinelExecutionService(opts)
    recent = "2025-01-01T00:59:50+00:00"
    snapshot = _make_snapshot(last_changed=recent)
    finding = _make_finding(suggested_actions=["light.turn_off"])

    result = svc.evaluate_canary(finding, snapshot, 2, _NOW)

    assert result.action_policy_path == ACTION_POLICY_PROMPT_USER
    assert result.block_reason == "service_not_allowlisted"


def test_a_finding_with_nothing_to_run_is_left_to_the_user() -> None:
    """No suggested `domain.service` action: nothing to run, the user decides."""
    opts = {
        **_FULL_OPTIONS,
        CONF_SENTINEL_AUTO_EXECUTE_ALLOWED_SERVICES: [],
    }
    svc = SentinelExecutionService(opts)
    recent = "2025-01-01T00:59:50+00:00"
    snapshot = _make_snapshot(last_changed=recent)
    finding = _make_finding(suggested_actions=[])

    result = svc.evaluate_canary(finding, snapshot, 2, _NOW)
    assert result.action_policy_path == ACTION_POLICY_PROMPT_USER
    assert result.block_reason == "no_executable_action"


# ---------------------------------------------------------------------------
# 7. Rate limit gate
# ---------------------------------------------------------------------------


def test_rate_limit_blocks_after_max_actions() -> None:
    opts = {
        **_FULL_OPTIONS,
        CONF_SENTINEL_AUTO_EXECUTE_MAX_ACTIONS_PER_HOUR: 2,
        CONF_SENTINEL_AUTO_EXECUTE_ALLOWED_SERVICES: ["light.turn_off"],
    }
    svc = SentinelExecutionService(opts)
    recent = "2025-01-01T00:59:50+00:00"
    snapshot = _make_snapshot(last_changed=recent)

    # Pre-fill the rate-limit counter with two recent actions.
    svc._recent_action_times = [
        _NOW - timedelta(minutes=10),
        _NOW - timedelta(minutes=5),
    ]

    finding = _make_finding(anomaly_id="fresh_a1", suggested_actions=["light.turn_off"])
    result = svc.evaluate_canary(finding, snapshot, 2, _NOW)

    assert result.action_policy_path == ACTION_POLICY_PROMPT_USER
    assert result.block_reason == "rate_limit_exceeded"


def test_actions_older_than_an_hour_do_not_count() -> None:
    """The hourly limit reads the last hour, and a commit drops the rest."""
    opts = {**_FULL_OPTIONS, CONF_SENTINEL_AUTO_EXECUTE_MAX_ACTIONS_PER_HOUR: 2}
    svc = SentinelExecutionService(opts)
    snapshot = _make_snapshot(last_changed="2025-01-01T00:59:50+00:00")
    svc._recent_action_times = [
        _NOW - timedelta(hours=2),
        _NOW - timedelta(hours=1, minutes=1),
    ]
    finding = _make_finding(
        anomaly_id="after_prune", suggested_actions=["light.turn_off"]
    )

    result = svc.evaluate_canary(finding, snapshot, 2, _NOW)

    assert result.action_policy_path == ACTION_POLICY_AUTO_EXECUTE
    svc.commit_auto_execute(cast("str", result.execution_id), _NOW)
    assert svc._recent_action_times == [_NOW]


# ---------------------------------------------------------------------------
# 8. Idempotency gate
# ---------------------------------------------------------------------------


def test_a_committed_action_is_a_duplicate_until_its_window_ends() -> None:
    """The same decision is refused inside the window and allowed after it."""
    svc = SentinelExecutionService(_FULL_OPTIONS)
    snapshot = _make_snapshot(last_changed="2025-01-01T00:59:50+00:00")
    finding = _make_finding(
        anomaly_id="idem_test", suggested_actions=["light.turn_off"]
    )

    first = svc.evaluate_canary(finding, snapshot, 2, _NOW)
    assert first.action_policy_path == ACTION_POLICY_AUTO_EXECUTE
    svc.commit_auto_execute(cast("str", first.execution_id), _NOW)

    second = svc.evaluate_canary(finding, snapshot, 2, _NOW)
    assert second.action_policy_path == ACTION_POLICY_PROMPT_USER
    assert second.block_reason == "idempotency_duplicate"
    assert second.execution_id == first.execution_id

    svc._idempotency_window_reset_at -= 16 * 60  # the default window is 15 min
    third = svc.evaluate_canary(finding, snapshot, 2, _NOW)
    assert third.action_policy_path == ACTION_POLICY_AUTO_EXECUTE
    # The next commit starts a new window holding only its own id.
    svc.commit_auto_execute("next", _NOW)
    assert svc._idempotency_seen == {"next"}


@pytest.mark.parametrize("window", [0, -5, None, "soon"])
def test_an_unusable_window_setting_does_not_break_the_gate(window: object) -> None:
    """A hand-edited window of 0 divided by zero; the finding was never sent."""
    opts = {
        **_FULL_OPTIONS,
        CONF_SENTINEL_EXECUTION_IDEMPOTENCY_WINDOW_MINUTES: window,
        CONF_SENTINEL_AUTO_EXECUTE_MAX_ACTIONS_PER_HOUR: None,
    }
    svc = SentinelExecutionService(opts)
    snapshot = _make_snapshot(last_changed="2025-01-01T00:59:50+00:00")
    finding = _make_finding(suggested_actions=["light.turn_off"])

    result = svc.evaluate_canary(finding, snapshot, 2, _NOW)

    assert result.action_policy_path == ACTION_POLICY_AUTO_EXECUTE
    if not isinstance(window, str) and window is not None:
        assert svc._idempotency_window_seconds() == 60


@pytest.mark.parametrize(
    ("limit", "reason"),
    [
        (-1, "rate_limit_exceeded"),
        (0, "rate_limit_exceeded"),
        ("x", None),
    ],
)
def test_an_hourly_limit_is_read_as_a_whole_number_or_the_default(
    limit: object, reason: str | None
) -> None:
    """Below zero is zero (nothing runs); not a number is the default."""
    opts = {**_FULL_OPTIONS, CONF_SENTINEL_AUTO_EXECUTE_MAX_ACTIONS_PER_HOUR: limit}
    svc = SentinelExecutionService(opts)
    snapshot = _make_snapshot(last_changed="2025-01-01T00:59:50+00:00")
    finding = _make_finding(suggested_actions=["light.turn_off"])

    assert svc.evaluate_canary(finding, snapshot, 2, _NOW).block_reason == reason


@pytest.mark.parametrize(
    ("stored", "read"),
    [("3.0", 3), ("1e3", 60), (2.9, 2), (-4, 0), ("x", 5), (None, 5), (10**9, 60)],
)
def test_a_numeric_setting_is_read_within_its_bounds(stored: object, read: int) -> None:
    """Text that is a number is that number; anything else is the default."""
    svc = SentinelExecutionService({"some_setting": stored})
    assert svc._int_option("some_setting", 5, minimum=0, maximum=60) == read


@pytest.mark.parametrize("allowed", [5, True, None, "light.turn_off", ["light", 1]])
def test_an_allowlist_that_is_not_a_list_of_text_does_not_break_the_gate(
    allowed: object,
) -> None:
    """A hand-edited allowlist raised, and the run's later findings went unsent."""
    opts = {**_FULL_OPTIONS, CONF_SENTINEL_AUTO_EXECUTE_ALLOWED_SERVICES: allowed}
    svc = SentinelExecutionService(opts)
    snapshot = _make_snapshot(last_changed="2025-01-01T00:59:50+00:00")
    finding = _make_finding(suggested_actions=["light.turn_off"])

    result = svc.evaluate_canary(finding, snapshot, 2, _NOW)

    # Only the list holding the text "light" allows the action.
    expected = None if isinstance(allowed, list) else "service_not_allowlisted"
    assert result.block_reason == expected


@pytest.mark.parametrize("threshold", ["soon", None, float("inf"), 10**18])
def test_an_unusable_staleness_setting_does_not_break_the_gate(
    threshold: object,
) -> None:
    """It raised out of the gate, before the finding was sent."""
    opts = {**_FULL_OPTIONS, CONF_SENTINEL_STALENESS_THRESHOLD_SECONDS: threshold}
    svc = SentinelExecutionService(opts)
    snapshot = _make_snapshot(last_changed="2025-01-01T00:59:50+00:00")
    finding = _make_finding(suggested_actions=["light.turn_off"])

    result = svc.evaluate_canary(finding, snapshot, 2, _NOW)

    assert result.action_policy_path == ACTION_POLICY_AUTO_EXECUTE


# ---------------------------------------------------------------------------
# 9. Happy-path: all guardrails pass
# ---------------------------------------------------------------------------


def test_all_guardrails_pass_returns_auto_execute() -> None:
    recent = "2025-01-01T00:59:50+00:00"
    snapshot = _make_snapshot(last_changed=recent)
    svc = SentinelExecutionService(
        {
            **_FULL_OPTIONS,
            CONF_SENTINEL_AUTO_EXECUTE_ALLOWED_SERVICES: ["light.turn_off"],
        }
    )
    finding = _make_finding(
        anomaly_id="happy_path",
        confidence=0.95,
        is_sensitive=False,
        suggested_actions=["light.turn_off"],
    )

    result = svc.evaluate_canary(finding, snapshot, 2, _NOW)

    assert result.action_policy_path == ACTION_POLICY_AUTO_EXECUTE
    assert result.data_quality == DATA_QUALITY_FRESH
    assert result.execution_id is not None
    assert result.block_reason is None


# ---------------------------------------------------------------------------
# Data quality helper — fresh entity
# ---------------------------------------------------------------------------


def test_fresh_entity_yields_fresh_quality() -> None:
    """Entity changed <threshold seconds ago → fresh quality."""
    opts = {CONF_SENTINEL_STALENESS_THRESHOLD_SECONDS: 3600}
    svc = SentinelExecutionService(opts)
    # _NOW is 01:00:00, entity last changed 00:59:50 → 70 s old < 3600 s
    recent = "2025-01-01T00:59:50+00:00"
    snapshot = _make_snapshot(last_changed=recent)
    finding = _make_finding()

    dq, details = svc._compute_data_quality(finding, snapshot, _NOW)

    assert dq == DATA_QUALITY_FRESH
    assert details["fresh_entities"] == ["binary_sensor.front_door"]
    assert details["stale_entities"] == []


@pytest.mark.parametrize(
    ("last_reported", "fresh"),
    [("2025-01-01T00:59:59+00:00", True), ("2024-12-31T01:00:00+00:00", False)],
)
def test_data_quality_reads_when_a_power_sensor_last_reported(
    last_reported: str,
    fresh: bool,  # noqa: FBT001
) -> None:
    """
    A power sensor is fresh while it keeps reporting, stale once it stops (#982).

    Its last_changed is the start of its current run (a day back here), set by
    power enrichment, which also stores last_reported.
    """
    svc = SentinelExecutionService(_FULL_OPTIONS)
    snapshot = _make_snapshot(last_changed="2024-12-31T01:00:00+00:00")
    cast("Any", snapshot["entities"][0])["last_reported"] = last_reported

    dq, details = svc._compute_data_quality(_make_finding(), snapshot, _NOW)

    if fresh:
        assert dq == DATA_QUALITY_FRESH
        assert details["stale_entities"] == []
    else:
        assert details["stale_entities"] == ["binary_sensor.front_door"]


# ---------------------------------------------------------------------------
# Policy version — deterministic
# ---------------------------------------------------------------------------


def test_policy_version_is_deterministic() -> None:
    svc = SentinelExecutionService(_FULL_OPTIONS)
    v1 = svc._compute_policy_version(2)
    v2 = svc._compute_policy_version(2)
    assert v1 == v2


def test_policy_version_changes_on_config_change() -> None:
    svc = SentinelExecutionService(_FULL_OPTIONS)
    v1 = svc._compute_policy_version(2)
    v2 = svc._compute_policy_version(3)  # autonomy level changed
    assert v1 != v2


# ---------------------------------------------------------------------------
# 10. Canary mode — evaluate_canary() has no side effects
# ---------------------------------------------------------------------------


def test_canary_no_side_effects_when_would_auto_execute() -> None:
    """evaluate_canary() returns auto_execute but does not mutate rate/idempotency state."""
    recent = "2025-01-01T00:59:50+00:00"
    snapshot = _make_snapshot(last_changed=recent)
    svc = SentinelExecutionService(
        {
            **_FULL_OPTIONS,
            CONF_SENTINEL_AUTO_EXECUTE_ALLOWED_SERVICES: ["light.turn_off"],
        }
    )
    finding = _make_finding(
        anomaly_id="canary_happy",
        confidence=0.95,
        suggested_actions=["light.turn_off"],
    )

    # Capture state before canary call.
    times_before = list(svc._recent_action_times)
    seen_before = set(svc._idempotency_seen)

    result = svc.evaluate_canary(finding, snapshot, 2, _NOW)

    assert result.action_policy_path == ACTION_POLICY_AUTO_EXECUTE
    assert result.execution_id is not None
    # State must be unchanged.
    assert svc._recent_action_times == times_before
    assert svc._idempotency_seen == seen_before


def test_the_hourly_limit_counts_committed_actions_only() -> None:
    """A decision takes no slot; an action that ran does."""
    snapshot = _make_snapshot(last_changed="2025-01-01T00:59:50+00:00")
    max_actions = 2
    svc = SentinelExecutionService(
        {**_FULL_OPTIONS, CONF_SENTINEL_AUTO_EXECUTE_MAX_ACTIONS_PER_HOUR: max_actions}
    )
    for i in range(max_actions):
        f = _make_finding(anomaly_id=f"fill_{i}", suggested_actions=["light.turn_off"])
        approved = svc.evaluate_canary(f, snapshot, 2, _NOW)
        assert approved.action_policy_path == ACTION_POLICY_AUTO_EXECUTE
        svc.commit_auto_execute(cast("str", approved.execution_id), _NOW)
    one_more = _make_finding(anomaly_id="over", suggested_actions=["light.turn_off"])
    result = svc.evaluate_canary(one_more, snapshot, 2, _NOW)
    assert result.action_policy_path == ACTION_POLICY_PROMPT_USER
    assert result.block_reason == "rate_limit_exceeded"
    assert len(svc._recent_action_times) == max_actions


def test_commit_auto_execute_records_state_after_pure_evaluation() -> None:
    """Pure evaluation stays read-only until commit_auto_execute() is called."""
    recent = "2025-01-01T00:59:50+00:00"
    snapshot = _make_snapshot(last_changed=recent)
    svc = SentinelExecutionService(
        {
            **_FULL_OPTIONS,
            CONF_SENTINEL_AUTO_EXECUTE_ALLOWED_SERVICES: ["light.turn_off"],
        }
    )
    finding = _make_finding(
        anomaly_id="commit_me",
        confidence=0.95,
        suggested_actions=["light.turn_off"],
    )

    result = svc.evaluate_canary(finding, snapshot, 2, _NOW)

    assert result.action_policy_path == ACTION_POLICY_AUTO_EXECUTE
    assert result.execution_id is not None
    assert not svc._recent_action_times
    assert not svc._idempotency_seen

    svc.commit_auto_execute(cast("str", result.execution_id), _NOW)

    assert len(svc._recent_action_times) == 1
    assert result.execution_id in svc._idempotency_seen


def test_commit_auto_execute_enables_future_duplicate_block() -> None:
    """Once committed, the same decision is blocked by idempotency."""
    recent = "2025-01-01T00:59:50+00:00"
    snapshot = _make_snapshot(last_changed=recent)
    svc = SentinelExecutionService(
        {
            **_FULL_OPTIONS,
            CONF_SENTINEL_AUTO_EXECUTE_ALLOWED_SERVICES: ["light.turn_off"],
        }
    )
    finding = _make_finding(
        anomaly_id="dup_after_commit",
        confidence=0.95,
        suggested_actions=["light.turn_off"],
    )

    first = svc.evaluate_canary(finding, snapshot, 2, _NOW)
    svc.commit_auto_execute(cast("str", first.execution_id), _NOW)
    second = svc.evaluate_canary(finding, snapshot, 2, _NOW)

    assert first.action_policy_path == ACTION_POLICY_AUTO_EXECUTE
    assert second.action_policy_path == ACTION_POLICY_PROMPT_USER
    assert second.block_reason == "idempotency_duplicate"

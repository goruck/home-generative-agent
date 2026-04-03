"""Tests for SentinelExecutionService guardrails — Issue #259."""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any, cast

from homeassistant.util import dt as dt_util

from custom_components.home_generative_agent.const import (
    ACTION_POLICY_AUTO_EXECUTE,
    ACTION_POLICY_BLOCKED,
    ACTION_POLICY_HANDOFF,
    ACTION_POLICY_PROMPT_USER,
    CONF_SENTINEL_AUTO_EXECUTE_ALLOWED_SERVICES,
    CONF_SENTINEL_AUTO_EXECUTE_DEFAULT_MIN_CONFIDENCE,
    CONF_SENTINEL_AUTO_EXECUTE_MAX_ACTIONS_PER_HOUR,
    CONF_SENTINEL_AUTO_EXECUTION_ENABLED,
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

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_finding(
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
    CONF_SENTINEL_AUTO_EXECUTE_DEFAULT_MIN_CONFIDENCE: 0.70,
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
        result = svc.evaluate(finding, snapshot, level, _NOW)
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
    finding = _make_finding()

    result = svc.evaluate(finding, snapshot, 2, _NOW)

    assert result.data_quality == DATA_QUALITY_STALE
    assert result.action_policy_path == ACTION_POLICY_PROMPT_USER
    assert result.block_reason == "data_quality_stale"


# ---------------------------------------------------------------------------
# 2b. Data quality gate — unavailable
# ---------------------------------------------------------------------------


def test_unavailable_entity_low_severity_returns_blocked() -> None:
    """Entity not in snapshot → unavailable; low/medium severity → blocked."""
    svc = SentinelExecutionService(_FULL_OPTIONS)
    snapshot = _make_snapshot()  # has binary_sensor.front_door
    finding = _make_finding(
        triggering_entities=["lock.missing_lock"],  # not in snapshot
        severity="medium",
    )

    result = svc.evaluate(finding, snapshot, 2, _NOW)

    assert result.data_quality == DATA_QUALITY_UNAVAILABLE
    assert result.action_policy_path == ACTION_POLICY_BLOCKED


def test_unavailable_entity_high_severity_returns_prompt_user() -> None:
    """High-severity unavailable findings route to prompt_user (not blocked)."""
    svc = SentinelExecutionService(_FULL_OPTIONS)
    snapshot = _make_snapshot()
    finding = _make_finding(
        triggering_entities=["lock.missing_lock"],
        severity="high",
    )

    result = svc.evaluate(finding, snapshot, 2, _NOW)

    assert result.data_quality == DATA_QUALITY_UNAVAILABLE
    assert result.action_policy_path == ACTION_POLICY_PROMPT_USER


# ---------------------------------------------------------------------------
# 3. Sensitivity gate
# ---------------------------------------------------------------------------


def test_sensitive_finding_returns_handoff() -> None:
    """Sensitive findings are handed off to the conversation agent."""
    # Fresh entity — entity last_changed at _NOW - 10s, threshold 3600
    recent = "2025-01-01T00:59:50+00:00"
    snapshot = _make_snapshot(last_changed=recent)
    svc = SentinelExecutionService(_FULL_OPTIONS)
    finding = _make_finding(is_sensitive=True)

    result = svc.evaluate(finding, snapshot, 2, _NOW)

    assert result.action_policy_path == ACTION_POLICY_HANDOFF
    assert result.block_reason == "finding_sensitive"


# ---------------------------------------------------------------------------
# 4. Auto-execution disabled gate
# ---------------------------------------------------------------------------


def test_auto_execution_disabled_returns_prompt_user() -> None:
    opts = {**_FULL_OPTIONS, CONF_SENTINEL_AUTO_EXECUTION_ENABLED: False}
    svc = SentinelExecutionService(opts)
    recent = "2025-01-01T00:59:50+00:00"
    snapshot = _make_snapshot(last_changed=recent)
    finding = _make_finding()

    result = svc.evaluate(finding, snapshot, 2, _NOW)

    assert result.action_policy_path == ACTION_POLICY_PROMPT_USER
    assert result.block_reason == "auto_execution_disabled"


# ---------------------------------------------------------------------------
# 5. Confidence gate
# ---------------------------------------------------------------------------


def test_confidence_below_threshold_returns_prompt_user() -> None:
    opts = {**_FULL_OPTIONS, CONF_SENTINEL_AUTO_EXECUTE_DEFAULT_MIN_CONFIDENCE: 0.80}
    svc = SentinelExecutionService(opts)
    recent = "2025-01-01T00:59:50+00:00"
    snapshot = _make_snapshot(last_changed=recent)
    finding = _make_finding(confidence=0.75)  # below 0.80

    result = svc.evaluate(finding, snapshot, 2, _NOW)

    assert result.action_policy_path == ACTION_POLICY_PROMPT_USER
    assert result.block_reason == "confidence_below_threshold"


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

    result = svc.evaluate(finding, snapshot, 2, _NOW)

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

    result = svc.evaluate(finding, snapshot, 2, _NOW)

    assert result.action_policy_path == ACTION_POLICY_PROMPT_USER
    assert result.block_reason == "service_not_allowlisted"


def test_empty_suggested_actions_passes_allowlist() -> None:
    """Finding with no suggested_actions passes any allowlist."""
    opts = {
        **_FULL_OPTIONS,
        CONF_SENTINEL_AUTO_EXECUTE_ALLOWED_SERVICES: [],
    }
    svc = SentinelExecutionService(opts)
    recent = "2025-01-01T00:59:50+00:00"
    snapshot = _make_snapshot(last_changed=recent)
    finding = _make_finding(suggested_actions=[])

    # Will be blocked by rate or idempotency, not allowlist.
    result = svc.evaluate(finding, snapshot, 2, _NOW)
    assert result.block_reason != "service_not_allowlisted"


# ---------------------------------------------------------------------------
# 7. Rate limit gate
# ---------------------------------------------------------------------------


def test_rate_limit_blocks_after_max_actions() -> None:
    opts = {
        **_FULL_OPTIONS,
        CONF_SENTINEL_AUTO_EXECUTE_MAX_ACTIONS_PER_HOUR: 2,
        CONF_SENTINEL_AUTO_EXECUTE_ALLOWED_SERVICES: [],  # empty = allow no-action findings
    }
    svc = SentinelExecutionService(opts)
    recent = "2025-01-01T00:59:50+00:00"
    snapshot = _make_snapshot(last_changed=recent)

    # Pre-fill the rate-limit counter with two recent actions.
    svc._recent_action_times = [
        _NOW - timedelta(minutes=10),
        _NOW - timedelta(minutes=5),
    ]

    finding = _make_finding(anomaly_id="fresh_a1", suggested_actions=[])
    result = svc.evaluate(finding, snapshot, 2, _NOW)

    assert result.action_policy_path == ACTION_POLICY_PROMPT_USER
    assert result.block_reason == "rate_limit_exceeded"


def test_expired_rate_limit_entries_are_pruned() -> None:
    """Actions older than 1 hour are pruned before the rate check."""
    opts = {
        **_FULL_OPTIONS,
        CONF_SENTINEL_AUTO_EXECUTE_MAX_ACTIONS_PER_HOUR: 2,
        CONF_SENTINEL_AUTO_EXECUTE_ALLOWED_SERVICES: [],
    }
    svc = SentinelExecutionService(opts)
    recent = "2025-01-01T00:59:50+00:00"
    snapshot = _make_snapshot(last_changed=recent)

    # Both old actions are >1 hour ago → pruned → rate not exceeded.
    svc._recent_action_times = [
        _NOW - timedelta(hours=2),
        _NOW - timedelta(hours=1, minutes=1),
    ]

    finding = _make_finding(anomaly_id="after_prune", suggested_actions=[])
    result = svc.evaluate(finding, snapshot, 2, _NOW)

    assert result.block_reason != "rate_limit_exceeded"


# ---------------------------------------------------------------------------
# 8. Idempotency gate
# ---------------------------------------------------------------------------


def test_idempotency_deduplicates_within_window() -> None:
    """Second evaluate() call with same finding returns idempotency block."""
    opts = {
        **_FULL_OPTIONS,
        CONF_SENTINEL_AUTO_EXECUTE_ALLOWED_SERVICES: [],
    }
    svc = SentinelExecutionService(opts)
    recent = "2025-01-01T00:59:50+00:00"
    snapshot = _make_snapshot(last_changed=recent)
    finding = _make_finding(anomaly_id="idem_test", suggested_actions=[])

    first = svc.evaluate(finding, snapshot, 2, _NOW)
    assert first.action_policy_path == ACTION_POLICY_AUTO_EXECUTE

    second = svc.evaluate(finding, snapshot, 2, _NOW)
    assert second.action_policy_path == ACTION_POLICY_PROMPT_USER
    assert second.block_reason == "idempotency_duplicate"
    assert second.execution_id == first.execution_id


# ---------------------------------------------------------------------------
# 9. Happy-path: all guardrails pass
# ---------------------------------------------------------------------------


def test_all_guardrails_pass_returns_auto_execute() -> None:
    recent = "2025-01-01T00:59:50+00:00"
    snapshot = _make_snapshot(last_changed=recent)
    svc = SentinelExecutionService(
        {
            **_FULL_OPTIONS,
            CONF_SENTINEL_AUTO_EXECUTE_ALLOWED_SERVICES: [],  # empty allowed services, empty actions → pass
        }
    )
    finding = _make_finding(
        anomaly_id="happy_path",
        confidence=0.95,
        is_sensitive=False,
        suggested_actions=[],
    )

    result = svc.evaluate(finding, snapshot, 2, _NOW)

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


# ---------------------------------------------------------------------------
# Policy version — deterministic
# ---------------------------------------------------------------------------


def test_policy_version_is_deterministic() -> None:
    svc = SentinelExecutionService(_FULL_OPTIONS)
    v1 = svc._compute_policy_version(2, 0.70)
    v2 = svc._compute_policy_version(2, 0.70)
    assert v1 == v2


def test_policy_version_changes_on_config_change() -> None:
    svc = SentinelExecutionService(_FULL_OPTIONS)
    v1 = svc._compute_policy_version(2, 0.70)
    v2 = svc._compute_policy_version(3, 0.70)  # autonomy level changed
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
            CONF_SENTINEL_AUTO_EXECUTE_ALLOWED_SERVICES: [],
        }
    )
    finding = _make_finding(
        anomaly_id="canary_happy",
        confidence=0.95,
        suggested_actions=[],
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


def test_canary_distinguishable_from_live() -> None:
    """Live evaluate() mutates state; evaluate_canary() does not."""
    recent = "2025-01-01T00:59:50+00:00"
    snapshot = _make_snapshot(last_changed=recent)
    svc = SentinelExecutionService(
        {
            **_FULL_OPTIONS,
            CONF_SENTINEL_AUTO_EXECUTE_ALLOWED_SERVICES: [],
        }
    )
    finding = _make_finding(
        anomaly_id="live_vs_canary",
        confidence=0.95,
        suggested_actions=[],
    )

    # Canary first — no state change.
    canary = svc.evaluate_canary(finding, snapshot, 2, _NOW)
    assert canary.action_policy_path == ACTION_POLICY_AUTO_EXECUTE
    assert not svc._recent_action_times
    assert not svc._idempotency_seen

    # Live evaluate — mutates state.
    live = svc.evaluate(finding, snapshot, 2, _NOW)
    assert live.action_policy_path == ACTION_POLICY_AUTO_EXECUTE
    assert len(svc._recent_action_times) == 1
    assert live.execution_id in svc._idempotency_seen


def test_canary_rate_limit_is_read_only() -> None:
    """Canary respects rate limit without consuming the slot."""
    recent = "2025-01-01T00:59:50+00:00"
    snapshot = _make_snapshot(last_changed=recent)
    max_actions = 2
    svc = SentinelExecutionService(
        {
            **_FULL_OPTIONS,
            CONF_SENTINEL_AUTO_EXECUTE_ALLOWED_SERVICES: [],
            CONF_SENTINEL_AUTO_EXECUTE_MAX_ACTIONS_PER_HOUR: max_actions,
        }
    )
    # Fill rate-limit bucket with live calls.
    for i in range(max_actions):
        f = _make_finding(anomaly_id=f"fill_{i}", confidence=0.95, suggested_actions=[])
        svc.evaluate(f, snapshot, 2, _NOW)
    assert len(svc._recent_action_times) == max_actions

    # Canary on a new finding should report rate_limit_exceeded without mutating.
    new_finding = _make_finding(
        anomaly_id="new_canary", confidence=0.95, suggested_actions=[]
    )
    result = svc.evaluate_canary(new_finding, snapshot, 2, _NOW)
    assert result.action_policy_path == ACTION_POLICY_PROMPT_USER
    assert result.block_reason == "rate_limit_exceeded"
    # Still the same count — canary didn't add another slot.
    assert len(svc._recent_action_times) == max_actions


def test_commit_auto_execute_records_state_after_pure_evaluation() -> None:
    """Pure evaluation stays read-only until commit_auto_execute() is called."""
    recent = "2025-01-01T00:59:50+00:00"
    snapshot = _make_snapshot(last_changed=recent)
    svc = SentinelExecutionService(
        {
            **_FULL_OPTIONS,
            CONF_SENTINEL_AUTO_EXECUTE_ALLOWED_SERVICES: [],
        }
    )
    finding = _make_finding(
        anomaly_id="commit_me",
        confidence=0.95,
        suggested_actions=[],
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
            CONF_SENTINEL_AUTO_EXECUTE_ALLOWED_SERVICES: [],
        }
    )
    finding = _make_finding(
        anomaly_id="dup_after_commit",
        confidence=0.95,
        suggested_actions=[],
    )

    first = svc.evaluate_canary(finding, snapshot, 2, _NOW)
    svc.commit_auto_execute(cast("str", first.execution_id), _NOW)
    second = svc.evaluate_canary(finding, snapshot, 2, _NOW)

    assert first.action_policy_path == ACTION_POLICY_AUTO_EXECUTE
    assert second.action_policy_path == ACTION_POLICY_PROMPT_USER
    assert second.block_reason == "idempotency_duplicate"

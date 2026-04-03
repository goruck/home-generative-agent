"""
Tests for suppression upgrades — Issue #260.

Covers: snooze routing, per-person presence grace, quiet hours, reason codes
for all suppression paths, v1→v2 migration, and read-only compatibility mode.
"""

from __future__ import annotations

from datetime import timedelta

from homeassistant.util import dt as dt_util

from custom_components.home_generative_agent.const import (
    SNOOZE_7D,
    SNOOZE_24H,
    SNOOZE_PERMANENT,
)
from custom_components.home_generative_agent.sentinel.models import (
    AnomalyFinding,
    Severity,
)
from custom_components.home_generative_agent.sentinel.suppression import (
    SUPPRESSION_REASON_NOT_SUPPRESSED,
    SUPPRESSION_REASON_PENDING_PROMPT,
    SUPPRESSION_REASON_PRESENCE_GRACE,
    SUPPRESSION_REASON_QUIET_HOURS,
    SUPPRESSION_REASON_TYPE_COOLDOWN,
    SUPPRESSION_REASON_USER_SNOOZE_7D,
    SUPPRESSION_REASON_USER_SNOOZE_24H,
    SUPPRESSION_REASON_USER_SNOOZE_PERMANENT,
    SUPPRESSION_STATE_VERSION,
    SuppressionState,
    _migrate_suppression_state,
    register_finding,
    register_presence_grace,
    register_prompt,
    register_snooze,
    should_suppress,
)

# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


_ZERO_COOLDOWN = timedelta(minutes=0)
_COOLDOWN_10M = timedelta(minutes=10)


def _finding(
    *,
    anomaly_id: str = "a1",
    finding_type: str = "open_entry_while_away",
    severity: Severity = "low",
) -> AnomalyFinding:
    return AnomalyFinding(
        anomaly_id=anomaly_id,
        type=finding_type,
        severity=severity,
        confidence=0.8,
        triggering_entities=["binary_sensor.front_door"],
        evidence={},
        suggested_actions=[],
        is_sensitive=False,
    )


# ---------------------------------------------------------------------------
# Snooze routing
# ---------------------------------------------------------------------------


def test_snooze_24h_suppresses() -> None:
    """A 24-hour snooze suppresses the finding during the window."""
    state = SuppressionState()
    now = dt_util.utcnow()
    finding = _finding()

    register_snooze(state, finding.type, SNOOZE_24H, now)

    decision = should_suppress(
        state, finding, now + timedelta(hours=1), _ZERO_COOLDOWN, _ZERO_COOLDOWN
    )
    assert decision.suppress is True
    assert decision.reason_code == SUPPRESSION_REASON_USER_SNOOZE_24H


def test_snooze_7d_suppresses() -> None:
    """A 7-day snooze suppresses the finding for 7 days."""
    state = SuppressionState()
    now = dt_util.utcnow()
    finding = _finding()

    register_snooze(state, finding.type, SNOOZE_7D, now)

    decision = should_suppress(
        state, finding, now + timedelta(days=3), _ZERO_COOLDOWN, _ZERO_COOLDOWN
    )
    assert decision.suppress is True
    assert decision.reason_code == SUPPRESSION_REASON_USER_SNOOZE_7D


def test_snooze_permanent_suppresses_indefinitely() -> None:
    """A permanent snooze suppresses the finding indefinitely."""
    state = SuppressionState()
    now = dt_util.utcnow()
    finding = _finding()

    register_snooze(state, finding.type, SNOOZE_PERMANENT, now)

    decision = should_suppress(
        state, finding, now + timedelta(days=365), _ZERO_COOLDOWN, _ZERO_COOLDOWN
    )
    assert decision.suppress is True
    assert decision.reason_code == SUPPRESSION_REASON_USER_SNOOZE_PERMANENT


def test_expired_snooze_does_not_suppress() -> None:
    """An expired snooze entry is cleaned up and the finding passes through."""
    state = SuppressionState()
    now = dt_util.utcnow()
    finding = _finding()

    # Register a 24h snooze but check 25h later
    register_snooze(state, finding.type, SNOOZE_24H, now)

    future = now + timedelta(hours=25)
    decision = should_suppress(state, finding, future, _ZERO_COOLDOWN, _ZERO_COOLDOWN)
    assert decision.suppress is False
    # Expired entry cleaned up
    assert finding.type not in state.snoozed_until


def test_snooze_different_type_does_not_suppress_other_type() -> None:
    """Snooze is scoped to the finding type."""
    state = SuppressionState()
    now = dt_util.utcnow()
    register_snooze(state, "open_entry_while_away", SNOOZE_PERMANENT, now)

    other_finding = _finding(finding_type="unlocked_lock_at_night")
    decision = should_suppress(
        state, other_finding, now, _ZERO_COOLDOWN, _ZERO_COOLDOWN
    )
    assert decision.suppress is False


# ---------------------------------------------------------------------------
# Presence grace window
# ---------------------------------------------------------------------------


def test_presence_grace_suppresses_presence_sensitive_type() -> None:
    """An active grace window suppresses presence-sensitive findings."""
    state = SuppressionState()
    now = dt_util.utcnow()

    register_presence_grace(state, "person.alice", now, grace_minutes=10)

    finding = _finding(finding_type="open_entry_while_away")
    decision = should_suppress(
        state, finding, now + timedelta(minutes=5), _ZERO_COOLDOWN, _ZERO_COOLDOWN
    )
    assert decision.suppress is True
    assert decision.reason_code == SUPPRESSION_REASON_PRESENCE_GRACE


def test_presence_grace_does_not_suppress_non_sensitive_type() -> None:
    """Grace window only applies to presence-sensitive finding types."""
    state = SuppressionState()
    now = dt_util.utcnow()

    register_presence_grace(state, "person.alice", now, grace_minutes=10)

    # appliance_power_duration is not presence-sensitive
    finding = _finding(finding_type="appliance_power_duration")
    decision = should_suppress(
        state, finding, now + timedelta(minutes=5), _ZERO_COOLDOWN, _ZERO_COOLDOWN
    )
    assert decision.suppress is False


def test_expired_presence_grace_does_not_suppress() -> None:
    """An expired grace window is cleaned up and does not suppress."""
    state = SuppressionState()
    now = dt_util.utcnow()

    register_presence_grace(state, "person.alice", now, grace_minutes=10)

    future = now + timedelta(minutes=15)
    finding = _finding(finding_type="open_entry_while_away")
    decision = should_suppress(state, finding, future, _ZERO_COOLDOWN, _ZERO_COOLDOWN)
    assert decision.suppress is False
    # Expired entry cleaned up
    assert "person.alice" not in state.presence_grace_until


def test_multiple_persons_grace_any_active_suppresses() -> None:
    """If at least one person is in grace, the finding is suppressed."""
    state = SuppressionState()
    now = dt_util.utcnow()

    register_presence_grace(state, "person.alice", now, grace_minutes=10)
    register_presence_grace(
        state, "person.bob", now - timedelta(minutes=15), grace_minutes=10
    )

    finding = _finding(finding_type="open_entry_while_away")
    # alice's window is active, bob's is expired
    decision = should_suppress(
        state, finding, now + timedelta(minutes=5), _ZERO_COOLDOWN, _ZERO_COOLDOWN
    )
    assert decision.suppress is True
    assert decision.reason_code == SUPPRESSION_REASON_PRESENCE_GRACE
    # Bob's expired entry cleaned up
    assert "person.bob" not in state.presence_grace_until


# ---------------------------------------------------------------------------
# Quiet hours
# ---------------------------------------------------------------------------


def _utc_dt(date_str: str) -> object:
    return dt_util.parse_datetime(date_str)


def test_quiet_hours_suppresses_matching_severity() -> None:
    """Quiet hours suppress low-severity findings during the configured window."""
    state = SuppressionState()
    # 23:00 UTC — within quiet hours 22-06
    now = dt_util.parse_datetime("2025-01-01T23:00:00+00:00")
    assert now is not None

    finding = _finding(severity="low")
    decision = should_suppress(
        state,
        finding,
        now,
        _ZERO_COOLDOWN,
        _ZERO_COOLDOWN,
        snapshot_timezone="UTC",
        quiet_hours_start=22,
        quiet_hours_end=6,
        quiet_hours_severities=["low"],
    )
    assert decision.suppress is True
    assert decision.reason_code == SUPPRESSION_REASON_QUIET_HOURS


def test_quiet_hours_does_not_suppress_high_severity() -> None:
    """High-severity findings pass through quiet hours."""
    state = SuppressionState()
    now = dt_util.parse_datetime("2025-01-01T23:00:00+00:00")
    assert now is not None

    finding = _finding(severity="high")
    decision = should_suppress(
        state,
        finding,
        now,
        _ZERO_COOLDOWN,
        _ZERO_COOLDOWN,
        snapshot_timezone="UTC",
        quiet_hours_start=22,
        quiet_hours_end=6,
        quiet_hours_severities=["low"],
    )
    assert decision.suppress is False


def test_outside_quiet_hours_does_not_suppress() -> None:
    """Findings outside quiet hours pass through."""
    state = SuppressionState()
    # 14:00 UTC — outside 22-06
    now = dt_util.parse_datetime("2025-01-01T14:00:00+00:00")
    assert now is not None

    finding = _finding(severity="low")
    decision = should_suppress(
        state,
        finding,
        now,
        _ZERO_COOLDOWN,
        _ZERO_COOLDOWN,
        snapshot_timezone="UTC",
        quiet_hours_start=22,
        quiet_hours_end=6,
        quiet_hours_severities=["low"],
    )
    assert decision.suppress is False


def test_quiet_hours_disabled_when_start_is_none() -> None:
    """Quiet hours are disabled when start_hour is None."""
    state = SuppressionState()
    now = dt_util.parse_datetime("2025-01-01T23:00:00+00:00")
    assert now is not None

    finding = _finding(severity="low")
    decision = should_suppress(
        state,
        finding,
        now,
        _ZERO_COOLDOWN,
        _ZERO_COOLDOWN,
        quiet_hours_start=None,
        quiet_hours_end=6,
        quiet_hours_severities=["low"],
    )
    assert decision.suppress is False


# ---------------------------------------------------------------------------
# Existing suppression paths still emit correct reason codes
# ---------------------------------------------------------------------------


def test_pending_prompt_reason_code_unchanged() -> None:
    """Pending prompt suppression still emits SUPPRESSION_REASON_PENDING_PROMPT."""
    state = SuppressionState()
    now = dt_util.utcnow()
    finding = _finding()
    register_prompt(state, finding, now)

    decision = should_suppress(state, finding, now, _ZERO_COOLDOWN, _ZERO_COOLDOWN)
    assert decision.suppress is True
    assert decision.reason_code == SUPPRESSION_REASON_PENDING_PROMPT


def test_type_cooldown_reason_code_unchanged() -> None:
    """Type cooldown suppression still emits SUPPRESSION_REASON_TYPE_COOLDOWN."""
    state = SuppressionState()
    now = dt_util.utcnow()
    finding = _finding()
    register_finding(state, finding, now)

    decision = should_suppress(
        state, finding, now + timedelta(minutes=5), _COOLDOWN_10M, _ZERO_COOLDOWN
    )
    assert decision.suppress is True
    assert decision.reason_code == SUPPRESSION_REASON_TYPE_COOLDOWN


def test_not_suppressed_reason_code_unchanged() -> None:
    state = SuppressionState()
    now = dt_util.utcnow()
    finding = _finding()

    decision = should_suppress(state, finding, now, _ZERO_COOLDOWN, _ZERO_COOLDOWN)
    assert decision.suppress is False
    assert decision.reason_code == SUPPRESSION_REASON_NOT_SUPPRESSED


# ---------------------------------------------------------------------------
# SuppressionState versioning
# ---------------------------------------------------------------------------


def test_suppression_state_default_version() -> None:
    state = SuppressionState()
    assert state.version == SUPPRESSION_STATE_VERSION


def test_suppression_state_serialization_includes_version() -> None:
    state = SuppressionState()
    data = state.as_dict()
    assert "version" in data
    assert data["version"] == SUPPRESSION_STATE_VERSION


def test_suppression_state_serialization_includes_new_fields() -> None:
    state = SuppressionState()
    data = state.as_dict()
    assert "snoozed_until" in data
    assert "presence_grace_until" in data


# ---------------------------------------------------------------------------
# v1 → v2 migration
# ---------------------------------------------------------------------------


def _v1_data() -> dict:
    return {
        "last_by_type": {"open_entry_while_away": "2025-01-01T00:00:00+00:00"},
        "last_by_entity": {},
        "pending_prompts": {},
        # No version key — classic v1 layout
    }


def test_migrate_v1_adds_version_and_new_fields() -> None:
    data = _v1_data()
    result = _migrate_suppression_state(data)

    assert result is data  # in-place
    assert result["version"] == 4
    assert "snoozed_until" in result
    assert "presence_grace_until" in result
    assert result["learned_cooldown_multipliers"] == {}


def test_migrate_v1_preserves_existing_data() -> None:
    data = _v1_data()
    _migrate_suppression_state(data)
    assert data["last_by_type"] == {
        "open_entry_while_away": "2025-01-01T00:00:00+00:00"
    }


def test_migrate_is_idempotent() -> None:
    data = _v1_data()
    _migrate_suppression_state(data)
    snapshot = dict(data)
    _migrate_suppression_state(data)
    assert data == snapshot


def test_migrate_v3_upgrades_to_v4() -> None:
    """v3 state is upgraded to v4 — learned_cooldown_multipliers added."""
    data: dict = {
        "last_by_type": {},
        "last_by_entity": {},
        "pending_prompts": {},
        "version": 3,
        "snoozed_until": {},
        "presence_grace_until": {},
    }
    _migrate_suppression_state(data)
    assert data["version"] == 4
    assert data["learned_cooldown_multipliers"] == {}


def test_migrate_skips_already_v4() -> None:
    """v4 state is not mutated by migration."""
    data: dict = {
        "last_by_type": {},
        "last_by_entity": {},
        "pending_prompts": {},
        "version": 4,
        "snoozed_until": {},
        "presence_grace_until": {},
        "learned_cooldown_multipliers": {"lock.front": 2},
    }
    original = dict(data)
    _migrate_suppression_state(data)
    assert data == original


def test_from_dict_handles_v1_data() -> None:
    """SuppressionState.from_dict can load v1 data (missing new fields)."""
    data = _v1_data()
    # Simulate migration having happened
    _migrate_suppression_state(data)
    state = SuppressionState.from_dict(data)

    assert state.version == 4
    assert state.snoozed_until == {}
    assert state.presence_grace_until == {}
    assert state.learned_cooldown_multipliers == {}
    assert "open_entry_while_away" in state.last_by_type


# ---------------------------------------------------------------------------
# Suppression priority ordering
# ---------------------------------------------------------------------------


def test_pending_prompt_takes_priority_over_snooze() -> None:
    """Pending prompt is checked before snooze."""
    state = SuppressionState()
    now = dt_util.utcnow()
    finding = _finding()

    register_prompt(state, finding, now)
    register_snooze(state, finding.type, SNOOZE_PERMANENT, now)

    decision = should_suppress(state, finding, now, _ZERO_COOLDOWN, _ZERO_COOLDOWN)
    assert decision.reason_code == SUPPRESSION_REASON_PENDING_PROMPT


def test_snooze_takes_priority_over_presence_grace() -> None:
    """Snooze is checked before presence grace."""
    state = SuppressionState()
    now = dt_util.utcnow()
    finding = _finding(finding_type="open_entry_while_away")

    register_snooze(state, finding.type, SNOOZE_PERMANENT, now)
    register_presence_grace(state, "person.alice", now, grace_minutes=10)

    decision = should_suppress(state, finding, now, _ZERO_COOLDOWN, _ZERO_COOLDOWN)
    assert decision.reason_code == SUPPRESSION_REASON_USER_SNOOZE_PERMANENT


# ---------------------------------------------------------------------------
# v2 → v3 migration: presence_grace_until key cleanup
# ---------------------------------------------------------------------------


def _v2_data_with_friendly_name_keys() -> dict:
    """Return a v2 suppression record with a mix of entity-ID and friendly-name keys."""
    return {
        "last_by_type": {},
        "last_by_entity": {},
        "pending_prompts": {},
        "version": 2,
        "snoozed_until": {},
        "presence_grace_until": {
            "Alice": "2099-01-01T00:00:00+00:00",  # stale friendly-name key
            "person.bob": "2099-01-01T00:00:00+00:00",  # valid entity-ID key
        },
    }


def test_migrate_v2_drops_friendly_name_presence_grace_keys() -> None:
    """v2→v3 migration must drop presence_grace_until keys that aren't person entity IDs."""
    data = _v2_data_with_friendly_name_keys()
    _migrate_suppression_state(data)

    assert "Alice" not in data["presence_grace_until"]
    assert "person.bob" in data["presence_grace_until"]
    assert data["version"] == 4


def test_migrate_v2_preserves_valid_entity_id_keys() -> None:
    """v2→v3 migration must not drop keys that already start with 'person.'."""
    data: dict = {
        "last_by_type": {},
        "last_by_entity": {},
        "pending_prompts": {},
        "version": 2,
        "snoozed_until": {},
        "presence_grace_until": {"person.alice": "2099-01-01T00:00:00+00:00"},
    }
    _migrate_suppression_state(data)
    assert data["presence_grace_until"] == {"person.alice": "2099-01-01T00:00:00+00:00"}


def test_migrate_v2_to_v3_is_idempotent() -> None:
    """Running migration twice on the same v2 record must be a no-op on the second pass."""
    data = _v2_data_with_friendly_name_keys()
    _migrate_suppression_state(data)
    snapshot = dict(data)
    snapshot["presence_grace_until"] = dict(data["presence_grace_until"])
    _migrate_suppression_state(data)
    assert data == snapshot


def test_migrate_v2_logs_warning_for_dropped_keys() -> None:
    """v2→v3 migration must emit a warning for each dropped friendly-name key."""
    from unittest.mock import patch

    import custom_components.home_generative_agent.sentinel.suppression as _mod

    data = _v2_data_with_friendly_name_keys()
    with patch.object(_mod.LOGGER, "warning") as mock_warn:
        _migrate_suppression_state(data)

    mock_warn.assert_called_once()
    assert "Alice" in mock_warn.call_args[0][1]

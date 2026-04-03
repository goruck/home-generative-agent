"""Tests for suppression logic."""

from __future__ import annotations

from datetime import timedelta

from homeassistant.util import dt as dt_util

from custom_components.home_generative_agent.sentinel.models import AnomalyFinding
from custom_components.home_generative_agent.sentinel.suppression import (
    MAX_COOLDOWN_MULTIPLIER,
    PENDING_PROMPT_DEFAULT_TTL,
    SUPPRESSION_REASON_ENTITY_COOLDOWN,
    SUPPRESSION_REASON_NOT_SUPPRESSED,
    SUPPRESSION_REASON_PENDING_PROMPT,
    SUPPRESSION_REASON_TYPE_COOLDOWN,
    SuppressionState,
    _migrate_suppression_state,
    purge_expired_prompts,
    record_cooldown_feedback,
    register_finding,
    register_prompt,
    resolve_prompt,
    should_suppress,
)


def _finding() -> AnomalyFinding:
    return AnomalyFinding(
        anomaly_id="a1",
        type="rule",
        severity="low",
        confidence=0.5,
        triggering_entities=["lock.front"],
        evidence={"entity_id": "lock.front"},
        suggested_actions=[],
        is_sensitive=True,
    )


def test_cooldown_suppresses() -> None:
    state = SuppressionState()
    finding = _finding()
    now = dt_util.utcnow()
    register_finding(state, finding, now)

    decision = should_suppress(
        state,
        finding,
        now + timedelta(minutes=1),
        cooldown_type=timedelta(minutes=10),
        cooldown_entity=timedelta(minutes=5),
    )
    assert decision.suppress
    assert decision.reason_code == SUPPRESSION_REASON_TYPE_COOLDOWN
    assert decision.context["type"] == finding.type


def test_prompt_suppresses_until_resolved() -> None:
    state = SuppressionState()
    finding = _finding()
    now = dt_util.utcnow()
    register_prompt(state, finding, now)

    decision = should_suppress(
        state,
        finding,
        now,
        cooldown_type=timedelta(minutes=0),
        cooldown_entity=timedelta(minutes=0),
    )
    assert decision.suppress
    assert decision.reason_code == SUPPRESSION_REASON_PENDING_PROMPT

    resolve_prompt(state, finding.anomaly_id)
    decision = should_suppress(
        state,
        finding,
        now,
        cooldown_type=timedelta(minutes=0),
        cooldown_entity=timedelta(minutes=0),
    )
    assert not decision.suppress
    assert decision.reason_code == SUPPRESSION_REASON_NOT_SUPPRESSED


def test_prompt_suppresses_within_ttl() -> None:
    state = SuppressionState()
    finding = _finding()
    now = dt_util.utcnow()
    register_prompt(state, finding, now)

    decision = should_suppress(
        state,
        finding,
        now + timedelta(hours=1),
        cooldown_type=timedelta(minutes=0),
        cooldown_entity=timedelta(minutes=0),
    )
    assert decision.suppress
    assert decision.reason_code == SUPPRESSION_REASON_PENDING_PROMPT


def test_prompt_expires_at_ttl_boundary() -> None:
    state = SuppressionState()
    finding = _finding()
    now = dt_util.utcnow()
    register_prompt(state, finding, now)

    decision = should_suppress(
        state,
        finding,
        now + PENDING_PROMPT_DEFAULT_TTL,
        cooldown_type=timedelta(minutes=0),
        cooldown_entity=timedelta(minutes=0),
    )
    assert not decision.suppress
    assert decision.reason_code == SUPPRESSION_REASON_NOT_SUPPRESSED
    assert finding.anomaly_id not in state.pending_prompts


def test_prompt_with_invalid_timestamp_treated_as_expired() -> None:
    state = SuppressionState(pending_prompts={"a1": "not-a-datetime"})
    finding = _finding()
    now = dt_util.utcnow()

    decision = should_suppress(
        state,
        finding,
        now,
        cooldown_type=timedelta(minutes=0),
        cooldown_entity=timedelta(minutes=0),
    )
    assert not decision.suppress
    assert decision.reason_code == SUPPRESSION_REASON_NOT_SUPPRESSED
    assert finding.anomaly_id not in state.pending_prompts


def test_purge_expired_prompts_removes_only_expired() -> None:
    now = dt_util.utcnow()
    state = SuppressionState(
        pending_prompts={
            "expired": dt_util.as_utc(now - timedelta(hours=6)).isoformat(),
            "fresh": dt_util.as_utc(now - timedelta(minutes=30)).isoformat(),
            "invalid": "bad-timestamp",
        }
    )

    changed = purge_expired_prompts(
        state,
        now,
        pending_prompt_ttl=PENDING_PROMPT_DEFAULT_TTL,
    )

    assert changed == 2
    assert "expired" not in state.pending_prompts
    assert "invalid" not in state.pending_prompts
    assert "fresh" in state.pending_prompts


def test_entity_cooldown_suppresses() -> None:
    now = dt_util.utcnow()
    state = SuppressionState(
        last_by_entity={"lock.front": {"rule": dt_util.as_utc(now).isoformat()}}
    )
    finding = _finding()

    decision = should_suppress(
        state,
        finding,
        now + timedelta(minutes=1),
        cooldown_type=timedelta(minutes=0),
        cooldown_entity=timedelta(minutes=5),
    )
    assert decision.suppress
    assert decision.reason_code == SUPPRESSION_REASON_ENTITY_COOLDOWN
    assert decision.context["entity_id"] == "lock.front"


# ---------------------------------------------------------------------------
# Learned cooldown multipliers (v4)
# ---------------------------------------------------------------------------


def test_record_cooldown_feedback_increments_multiplier() -> None:
    state = SuppressionState()
    new_val = record_cooldown_feedback(state, "lock.front")
    assert new_val == 2
    assert state.learned_cooldown_multipliers["lock.front"] == 2


def test_record_cooldown_feedback_caps_at_max() -> None:
    state = SuppressionState(
        learned_cooldown_multipliers={"lock.front": MAX_COOLDOWN_MULTIPLIER}
    )
    new_val = record_cooldown_feedback(state, "lock.front")
    assert new_val == MAX_COOLDOWN_MULTIPLIER
    assert state.learned_cooldown_multipliers["lock.front"] == MAX_COOLDOWN_MULTIPLIER


def test_entity_cooldown_respects_learned_multiplier() -> None:
    """Multiplier=2 doubles the effective cooldown window."""
    now = dt_util.utcnow()
    state = SuppressionState(
        last_by_entity={"lock.front": {"rule": dt_util.as_utc(now).isoformat()}},
        learned_cooldown_multipliers={"lock.front": 2},
    )
    finding = _finding()
    base_cooldown = timedelta(minutes=5)

    # At 1x + 1 min (6 min after last) -> still within 2x window (10 min) -> suppressed.
    decision = should_suppress(
        state,
        finding,
        now + timedelta(minutes=6),
        cooldown_type=timedelta(minutes=0),
        cooldown_entity=base_cooldown,
    )
    assert decision.suppress
    assert decision.reason_code == SUPPRESSION_REASON_ENTITY_COOLDOWN
    assert decision.context["multiplier"] == 2

    # At 2x + 1 min (11 min after last) -> outside 2x window -> not suppressed.
    decision2 = should_suppress(
        state,
        finding,
        now + timedelta(minutes=11),
        cooldown_type=timedelta(minutes=0),
        cooldown_entity=base_cooldown,
    )
    assert not decision2.suppress


def test_entity_cooldown_multiplier_context_field() -> None:
    """SuppressionDecision.context always carries the 'multiplier' key."""
    now = dt_util.utcnow()
    state = SuppressionState(
        last_by_entity={"lock.front": {"rule": dt_util.as_utc(now).isoformat()}}
    )
    finding = _finding()

    decision = should_suppress(
        state,
        finding,
        now + timedelta(minutes=1),
        cooldown_type=timedelta(minutes=0),
        cooldown_entity=timedelta(minutes=5),
    )
    assert decision.suppress
    assert "multiplier" in decision.context
    assert decision.context["multiplier"] == 1  # no feedback -> default 1x


def test_migration_v3_to_v4_adds_empty_multipliers() -> None:
    """v3 state gains an empty learned_cooldown_multipliers dict on migration."""
    v3_data: dict = {
        "version": 3,
        "last_by_type": {},
        "last_by_entity": {},
        "pending_prompts": {},
        "snoozed_until": {},
        "presence_grace": {},
        "quiet_hours": None,
    }
    migrated = _migrate_suppression_state(v3_data)
    assert migrated["version"] == 4
    assert migrated["learned_cooldown_multipliers"] == {}


def test_from_dict_round_trip_preserves_multipliers() -> None:
    """as_dict / from_dict round-trip keeps learned_cooldown_multipliers intact."""
    state = SuppressionState(
        learned_cooldown_multipliers={"lock.front": 3, "sensor.motion": 1}
    )
    restored = SuppressionState.from_dict(state.as_dict())
    assert restored.learned_cooldown_multipliers == {
        "lock.front": 3,
        "sensor.motion": 1,
    }

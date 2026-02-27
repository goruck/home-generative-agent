# ruff: noqa: S101
"""Tests for suppression logic."""

from __future__ import annotations

from datetime import timedelta

from homeassistant.util import dt as dt_util

from custom_components.home_generative_agent.sentinel.models import AnomalyFinding
from custom_components.home_generative_agent.sentinel.suppression import (
    PENDING_PROMPT_DEFAULT_TTL,
    SUPPRESSION_REASON_ENTITY_COOLDOWN,
    SUPPRESSION_REASON_NOT_SUPPRESSED,
    SUPPRESSION_REASON_PENDING_PROMPT,
    SUPPRESSION_REASON_TYPE_COOLDOWN,
    SuppressionState,
    purge_expired_prompts,
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

    assert changed is True
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

# ruff: noqa: S101
"""Tests for suppression logic."""

from __future__ import annotations

from datetime import timedelta

from homeassistant.util import dt as dt_util

from custom_components.home_generative_agent.sentinel.models import AnomalyFinding
from custom_components.home_generative_agent.sentinel.suppression import (
    SuppressionState,
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

    assert should_suppress(
        state,
        finding,
        now + timedelta(minutes=1),
        cooldown_type=timedelta(minutes=10),
        cooldown_entity=timedelta(minutes=5),
    )


def test_prompt_suppresses_until_resolved() -> None:
    state = SuppressionState()
    finding = _finding()
    now = dt_util.utcnow()
    register_prompt(state, finding, now)

    assert should_suppress(
        state,
        finding,
        now,
        cooldown_type=timedelta(minutes=0),
        cooldown_entity=timedelta(minutes=0),
    )

    resolve_prompt(state, finding.anomaly_id)
    assert not should_suppress(
        state,
        finding,
        now,
        cooldown_type=timedelta(minutes=0),
        cooldown_entity=timedelta(minutes=0),
    )

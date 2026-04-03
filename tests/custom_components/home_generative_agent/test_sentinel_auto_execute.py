"""Tests for Level 2 live auto-execute — Issue #264."""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from custom_components.home_generative_agent.sentinel.engine import (
    _auto_execute_finding,
)
from custom_components.home_generative_agent.sentinel.models import AnomalyFinding

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_finding(
    *,
    anomaly_id: str = "f1",
    suggested_actions: list[str] | None = None,
    triggering_entities: list[str] | None = None,
) -> AnomalyFinding:
    return AnomalyFinding(
        anomaly_id=anomaly_id,
        type="test_rule",
        severity="medium",
        confidence=0.9,
        triggering_entities=(
            ["lock.front_door"] if triggering_entities is None else triggering_entities
        ),
        evidence={},
        suggested_actions=suggested_actions if suggested_actions is not None else [],
        is_sensitive=False,
    )


def _make_hass(*, service_raises: Exception | None = None) -> MagicMock:
    hass = MagicMock()
    if service_raises is not None:
        hass.services.async_call = AsyncMock(side_effect=service_raises)
    else:
        hass.services.async_call = AsyncMock(return_value=None)
    return hass


# ---------------------------------------------------------------------------
# _auto_execute_finding helper
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_auto_execute_no_actions_returns_no_actions() -> None:
    """Finding with no suggested_actions yields status='no_actions'."""
    hass = _make_hass()
    finding = _make_finding(suggested_actions=[])

    result = await _auto_execute_finding(hass, finding, "exec_001")

    assert result["status"] == "no_actions"
    assert result["actions"] == []
    assert result["execution_id"] == "exec_001"
    hass.services.async_call.assert_not_called()


@pytest.mark.asyncio
async def test_auto_execute_advisory_only_skips_all() -> None:
    """Actions without '.' are advisory text and should not trigger HA service calls."""
    hass = _make_hass()
    finding = _make_finding(suggested_actions=["lock_entity", "close_entry"])

    result = await _auto_execute_finding(hass, finding, "exec_002")

    assert result["status"] == "no_actions"
    hass.services.async_call.assert_not_called()


@pytest.mark.asyncio
async def test_auto_execute_domain_service_calls_ha() -> None:
    """'domain.service' actions call hass.services.async_call."""
    hass = _make_hass()
    finding = _make_finding(
        suggested_actions=["lock.lock"],
        triggering_entities=["lock.front_door"],
    )

    result = await _auto_execute_finding(hass, finding, "exec_003")

    assert result["status"] == "success"
    assert len(result["actions"]) == 1
    assert result["actions"][0]["service"] == "lock.lock"
    assert result["actions"][0]["status"] == "ok"
    hass.services.async_call.assert_called_once_with(
        "lock",
        "lock",
        {"entity_id": "lock.front_door"},
        blocking=True,
    )


@pytest.mark.asyncio
async def test_auto_execute_multiple_triggering_entities_passed_as_list() -> None:
    """Multiple triggering entities are passed as a list to the service call."""
    hass = _make_hass()
    finding = _make_finding(
        suggested_actions=["lock.lock"],
        triggering_entities=["lock.front_door", "lock.back_door"],
    )

    result = await _auto_execute_finding(hass, finding, "exec_004")

    assert result["status"] == "success"
    hass.services.async_call.assert_called_once_with(
        "lock",
        "lock",
        {"entity_id": ["lock.front_door", "lock.back_door"]},
        blocking=True,
    )


@pytest.mark.asyncio
async def test_auto_execute_service_failure_returns_error() -> None:
    """Service call failure → status='error'."""
    hass = _make_hass(service_raises=RuntimeError("HA unavailable"))
    finding = _make_finding(suggested_actions=["lock.lock"])

    result = await _auto_execute_finding(hass, finding, "exec_005")

    assert result["status"] == "error"
    assert result["actions"][0]["status"] == "error"
    assert "HA unavailable" in result["actions"][0]["error"]


@pytest.mark.asyncio
async def test_auto_execute_partial_failure() -> None:
    """One ok, one error → status='partial'."""
    call_count = 0

    async def _side_effect(*_args: Any, **_kwargs: Any) -> None:
        nonlocal call_count
        call_count += 1
        if call_count == 2:
            msg = "second call fails"
            raise RuntimeError(msg)

    hass = MagicMock()
    hass.services.async_call = AsyncMock(side_effect=_side_effect)
    finding = _make_finding(
        suggested_actions=["lock.lock", "notify.notify"],
        triggering_entities=["lock.front_door"],
    )

    result = await _auto_execute_finding(hass, finding, "exec_006")

    assert result["status"] == "partial"
    assert result["actions"][0]["status"] == "ok"
    assert result["actions"][1]["status"] == "error"


@pytest.mark.asyncio
async def test_auto_execute_mixed_advisory_and_service() -> None:
    """Mix of advisory and domain.service actions: only service actions are dispatched."""
    hass = _make_hass()
    finding = _make_finding(
        suggested_actions=["check_entry", "lock.lock", "alert_user"],
        triggering_entities=["lock.front_door"],
    )

    result = await _auto_execute_finding(hass, finding, "exec_007")

    assert result["status"] == "success"
    assert len(result["actions"]) == 1
    assert result["actions"][0]["service"] == "lock.lock"
    hass.services.async_call.assert_called_once()


@pytest.mark.asyncio
async def test_auto_execute_no_triggering_entities_omits_entity_id() -> None:
    """Finding with empty triggering_entities calls service without entity_id."""
    hass = _make_hass()
    finding = _make_finding(
        suggested_actions=["lock.lock"],
        triggering_entities=[],
    )

    result = await _auto_execute_finding(hass, finding, "exec_008")

    assert result["status"] == "success"
    hass.services.async_call.assert_called_once_with(
        "lock",
        "lock",
        {},
        blocking=True,
    )

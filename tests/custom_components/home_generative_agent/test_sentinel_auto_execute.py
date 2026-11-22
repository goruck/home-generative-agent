# ruff: noqa: S101
"""Tests for Level 2 live auto-execute — Issue #264."""

from __future__ import annotations

import asyncio
from dataclasses import replace
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from custom_components.home_generative_agent.sentinel import engine as engine_module
from custom_components.home_generative_agent.sentinel.engine import (
    _auto_execute_finding,
    outcome_may_have_acted,
)
from custom_components.home_generative_agent.sentinel.models import (
    AnomalyFinding,
    FindingAction,
)

from .sentinel_actions import service_actions

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
        actions=service_actions(
            suggested_actions,
            ["lock.front_door"] if triggering_entities is None else triggering_entities,
        ),
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
        {"entity_id": ["lock.front_door"]},
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
        suggested_actions=["lock.lock", "switch.turn_off"],
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
async def test_an_action_without_a_target_is_never_called() -> None:
    """Home Assistant would run an untargeted call on the whole domain."""
    hass = _make_hass()
    finding = _make_finding(suggested_actions=["lock.lock"], triggering_entities=[])

    result = await _auto_execute_finding(hass, finding, "exec_008")

    assert result["status"] == "error"
    assert result["actions"][0]["error"].startswith("not a targeted service call")
    hass.services.async_call.assert_not_called()


@pytest.mark.asyncio
async def test_a_service_named_in_the_advice_is_never_called() -> None:
    """Only a finding's own actions run; its suggested actions are advice."""
    hass = _make_hass()
    finding = replace(_make_finding(suggested_actions=["lock.lock"]), actions=())

    result = await _auto_execute_finding(hass, finding, "exec_009")

    assert result["status"] == "no_actions"
    hass.services.async_call.assert_not_called()


@pytest.mark.asyncio
async def test_an_action_goes_to_its_own_targets_not_the_trigger() -> None:
    """A power sensor triggers; the plug behind it is what gets switched off."""
    hass = _make_hass()
    finding = replace(
        _make_finding(triggering_entities=["sensor.drier_power", "person.andy"]),
        actions=(FindingAction("switch.turn_off", ("switch.drier",)),),
    )

    result = await _auto_execute_finding(hass, finding, "exec_010")

    assert result["status"] == "success"
    assert result["actions"][0]["entity_ids"] == ["switch.drier"]
    hass.services.async_call.assert_called_once_with(
        "switch", "turn_off", {"entity_id": ["switch.drier"]}, blocking=True
    )


@pytest.mark.asyncio
async def test_a_call_that_never_answers_is_given_up_on(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A hung device costs one timeout, and the next action still runs."""
    monkeypatch.setattr(engine_module, "_ACTION_CALL_TIMEOUT_S", 0.01)
    calls: list[str] = []

    async def _call(domain: str, service: str, *_a: Any, **_k: Any) -> None:
        calls.append(f"{domain}.{service}")
        if domain == "switch":
            await asyncio.Event().wait()

    hass = MagicMock()
    hass.services.async_call = _call
    finding = replace(
        _make_finding(),
        actions=(
            FindingAction("switch.turn_off", ("switch.drier",)),
            FindingAction("lock.lock", ("lock.front_door",)),
        ),
    )

    result = await _auto_execute_finding(hass, finding, "exec_011")

    assert calls == ["switch.turn_off", "lock.lock"]
    assert result["status"] == "partial"
    hung, done = result["actions"]
    assert hung["status"] == "timeout"
    assert "the device may still have acted" in hung["error"]
    assert done["status"] == "ok"
    # Not cancelled at the limit: a slow device is left to finish.
    (late,) = engine_module._GIVEN_UP
    assert not late.done()
    assert outcome_may_have_acted(result)

    # While it runs, the device is not called again.
    again = await _auto_execute_finding(
        hass,
        replace(
            _make_finding(),
            actions=(FindingAction("switch.turn_on", ("switch.drier",)),),
        ),
        "exec_011b",
    )
    assert calls == ["switch.turn_off", "lock.lock"]
    assert again["actions"][0]["error"] == (
        "not called: an earlier call on switch.drier has not ended"
    )
    assert not outcome_may_have_acted(again)

    # A stopped engine ends it, and its result is collected.
    engine_module.cancel_outstanding_calls()
    await asyncio.gather(late, return_exceptions=True)
    await asyncio.sleep(0)
    assert not engine_module._GIVEN_UP
    assert not engine_module._CALLS


@pytest.mark.asyncio
async def test_a_call_that_only_timed_out_still_counts_as_attempted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """It may have acted, so it uses up the hourly limit and duplicate check."""
    monkeypatch.setattr(engine_module, "_ACTION_CALL_TIMEOUT_S", 0.01)

    async def _call(*_a: Any, **_k: Any) -> None:
        await asyncio.Event().wait()

    hass = MagicMock()
    hass.services.async_call = _call
    finding = replace(
        _make_finding(), actions=(FindingAction("switch.turn_off", ("switch.drier",)),)
    )

    result = await _auto_execute_finding(hass, finding, "exec_012")

    assert result["status"] == "error"
    assert outcome_may_have_acted(result)
    late = list(engine_module._GIVEN_UP)
    engine_module.cancel_outstanding_calls()
    await asyncio.gather(*late, return_exceptions=True)


@pytest.mark.asyncio
async def test_a_failed_call_does_not_count_as_attempted() -> None:
    """A call Home Assistant refused did nothing: it may be tried again."""
    hass = _make_hass(service_raises=RuntimeError("no such service"))
    finding = replace(
        _make_finding(), actions=(FindingAction("switch.turn_off", ("switch.drier",)),)
    )

    result = await _auto_execute_finding(hass, finding, "exec_013")

    assert result["status"] == "error"
    assert not outcome_may_have_acted(result)


@pytest.mark.parametrize(
    "bad",
    [
        FindingAction("switch", ("switch.drier",)),
        FindingAction("switch.turn_off", ("Switch.Drier",)),
        FindingAction("switch.turn_off", ("switch.bad id",)),
        FindingAction("switch.turn_off", ("light.hall",)),
    ],
)
@pytest.mark.asyncio
async def test_a_malformed_action_stops_the_valid_one_beside_it(
    bad: FindingAction,
) -> None:
    """As at the gate: part of a finding's actions never runs without the rest."""
    hass = _make_hass()
    finding = replace(
        _make_finding(),
        actions=(FindingAction("lock.lock", ("lock.front_door",)), bad),
    )

    result = await _auto_execute_finding(hass, finding, "exec_014")

    assert result["status"] == "error"
    assert [a["status"] for a in result["actions"]] == ["error", "error"]
    assert not outcome_may_have_acted(result)
    hass.services.async_call.assert_not_called()


@pytest.mark.asyncio
async def test_a_target_gone_since_the_gate_is_not_called() -> None:
    """Home Assistant would skip it and return: recorded as not called."""
    hass = _make_hass()
    hass.states.get = lambda entity_id: (
        None if entity_id == "switch.drier" else MagicMock(state="on")
    )
    finding = replace(
        _make_finding(),
        actions=(
            FindingAction("switch.turn_off", ("switch.drier",)),
            FindingAction("lock.lock", ["lock.front_door"]),  # type: ignore[arg-type]
        ),
    )

    result = await _auto_execute_finding(hass, finding, "exec_015")

    assert [a["status"] for a in result["actions"]] == ["error", "ok"]
    assert result["actions"][0]["error"] == (
        "not called: switch.drier is missing or unavailable"
    )
    hass.services.async_call.assert_called_once_with(
        "lock", "lock", {"entity_id": ["lock.front_door"]}, blocking=True
    )


@pytest.mark.asyncio
async def test_a_failure_with_no_message_still_says_what_failed() -> None:
    """An exception with no text is named by its type."""
    hass = _make_hass(service_raises=KeyError())
    finding = replace(
        _make_finding(), actions=(FindingAction("switch.turn_off", ("switch.drier",)),)
    )

    result = await _auto_execute_finding(hass, finding, "exec_016")

    assert result["actions"][0]["error"] == "KeyError"

# ruff: noqa: S101
"""Tests for sentinel alarm history enrichment."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
from homeassistant.helpers.recorder import DATA_INSTANCE

from custom_components.home_generative_agent.sentinel.alarm_enrichment import (
    async_enrich_alarm_last_changed,
)
from custom_components.home_generative_agent.snapshot.schema import validate_snapshot


def _base_snapshot(**kwargs: Any) -> Any:
    base: dict[str, Any] = {
        "schema_version": 1,
        "generated_at": "2026-06-22T05:16:00+00:00",
        "entities": [],
        "camera_activity": [],
        "derived": {
            "now": "2026-06-22T05:16:00+00:00",
            "timezone": "UTC",
            "is_night": False,
            "anyone_home": True,
            "people_home": [],
            "people_away": [],
            "last_motion_by_area": {},
        },
    }
    base.update(kwargs)
    return validate_snapshot(base)


def _alarm_entity(
    state: str = "disarmed", last_changed: str = "2026-06-22T05:16:00+00:00"
) -> dict[str, Any]:
    return {
        "entity_id": "alarm_control_panel.home",
        "domain": "alarm_control_panel",
        "state": state,
        "friendly_name": "Home Alarm",
        "area": None,
        "attributes": {},
        "last_changed": last_changed,
        "last_updated": last_changed,
    }


def _mock_state(state: str, last_changed: datetime) -> MagicMock:
    s = MagicMock()
    s.state = state
    s.last_changed = last_changed
    return s


def _make_hass(recorder_states: list[MagicMock] | None = None) -> MagicMock:
    """Return a mock hass with recorder available."""
    hass = MagicMock()
    instance = MagicMock()
    if recorder_states is not None:
        instance.async_add_executor_job = AsyncMock(
            return_value={"alarm_control_panel.home": recorder_states}
        )
    else:
        instance.async_add_executor_job = AsyncMock(return_value={})
    hass.data = {DATA_INSTANCE: instance}
    return hass


@pytest.mark.asyncio
async def test_no_recorder_skips_enrichment() -> None:
    """If recorder isn't loaded, enrichment is a no-op."""
    hass = MagicMock()
    hass.data = {}  # no DATA_INSTANCE
    snapshot = _base_snapshot()
    snapshot["entities"] = [_alarm_entity(last_changed="2026-06-22T05:16:00+00:00")]

    await async_enrich_alarm_last_changed(hass, snapshot)

    assert snapshot["entities"][0]["last_changed"] == "2026-06-22T05:16:00+00:00"


@pytest.mark.asyncio
async def test_no_disarmed_alarm_skips_enrichment() -> None:
    """If no alarm is disarmed, no recorder query is made."""
    hass = MagicMock()
    instance = MagicMock()
    instance.async_add_executor_job = AsyncMock()
    hass.data = {DATA_INSTANCE: instance}

    snapshot = _base_snapshot()
    snapshot["entities"] = [_alarm_entity(state="armed_away")]

    await async_enrich_alarm_last_changed(hass, snapshot)

    instance.async_add_executor_job.assert_not_called()


@pytest.mark.asyncio
async def test_corrects_startup_reset_last_changed() -> None:
    """Alarm disarmed before HA startup (no unavailable) gets last_changed corrected."""
    actual_disarm = datetime(2026, 6, 20, 10, 0, 0, tzinfo=UTC)
    ha_startup = datetime(2026, 6, 22, 5, 16, 0, tzinfo=UTC)

    recorder_states = [
        _mock_state("armed_away", datetime(2026, 6, 20, 9, 0, 0, tzinfo=UTC)),
        _mock_state("disarmed", actual_disarm),  # actual disarm
        _mock_state("disarmed", ha_startup),  # HA restart reset (no unavailable)
    ]
    hass = _make_hass(recorder_states)

    snapshot = _base_snapshot()
    snapshot["entities"] = [_alarm_entity(last_changed="2026-06-22T05:16:00+00:00")]

    await async_enrich_alarm_last_changed(hass, snapshot)

    assert snapshot["entities"][0]["last_changed"] == actual_disarm.isoformat()


@pytest.mark.asyncio
async def test_corrects_startup_reset_via_unavailable() -> None:
    """Real-world case: HA restart sets unavailable then disarmed; finds true disarm."""
    actual_disarm = datetime(2026, 6, 20, 10, 0, 0, tzinfo=UTC)
    ha_startup = datetime(2026, 6, 22, 5, 16, 0, tzinfo=UTC)

    recorder_states = [
        _mock_state("armed_away", datetime(2026, 6, 20, 9, 0, 0, tzinfo=UTC)),
        _mock_state("disarmed", actual_disarm),  # actual disarm days ago
        _mock_state(
            "unavailable", ha_startup
        ),  # integration lost connection on restart
        _mock_state(
            "disarmed", ha_startup
        ),  # integration reconnected → re-reports state
    ]
    hass = _make_hass(recorder_states)

    snapshot = _base_snapshot()
    snapshot["entities"] = [_alarm_entity(last_changed="2026-06-22T05:16:00+00:00")]

    await async_enrich_alarm_last_changed(hass, snapshot)

    assert snapshot["entities"][0]["last_changed"] == actual_disarm.isoformat()


@pytest.mark.asyncio
async def test_multiple_restarts_with_unavailable() -> None:
    """Multiple HA restarts each producing unavailable→disarmed; finds original."""
    original_disarm = datetime(2026, 6, 18, 14, 0, 0, tzinfo=UTC)
    restart1 = datetime(2026, 6, 20, 8, 0, 0, tzinfo=UTC)
    restart2 = datetime(2026, 6, 22, 5, 16, 0, tzinfo=UTC)

    recorder_states = [
        _mock_state("armed_away", datetime(2026, 6, 18, 13, 0, 0, tzinfo=UTC)),
        _mock_state("disarmed", original_disarm),
        _mock_state("unavailable", restart1),
        _mock_state("disarmed", restart1),
        _mock_state("unavailable", restart2),
        _mock_state("disarmed", restart2),
    ]
    hass = _make_hass(recorder_states)

    snapshot = _base_snapshot()
    snapshot["entities"] = [_alarm_entity(last_changed="2026-06-22T05:16:00+00:00")]

    await async_enrich_alarm_last_changed(hass, snapshot)

    assert snapshot["entities"][0]["last_changed"] == original_disarm.isoformat()


@pytest.mark.asyncio
async def test_genuine_disarm_after_unavailable_not_changed() -> None:
    """Genuine armed→unavailable→disarmed sequence keeps the disarm time."""
    disarm_time = datetime(2026, 6, 22, 5, 16, 0, tzinfo=UTC)

    recorder_states = [
        _mock_state("armed_home", datetime(2026, 6, 21, 22, 0, 0, tzinfo=UTC)),
        _mock_state("unavailable", datetime(2026, 6, 22, 5, 15, 0, tzinfo=UTC)),
        _mock_state("disarmed", disarm_time),  # genuine disarm after brief outage
    ]
    hass = _make_hass(recorder_states)

    snapshot = _base_snapshot()
    snapshot["entities"] = [_alarm_entity(last_changed="2026-06-22T05:16:00+00:00")]

    await async_enrich_alarm_last_changed(hass, snapshot)

    assert snapshot["entities"][0]["last_changed"] == "2026-06-22T05:16:00+00:00"


@pytest.mark.asyncio
async def test_no_change_when_disarm_is_genuine() -> None:
    """If the alarm really was just disarmed at startup time, no change is made."""
    disarm_time = datetime(2026, 6, 22, 5, 16, 0, tzinfo=UTC)

    recorder_states = [
        _mock_state("armed_home", datetime(2026, 6, 21, 22, 0, 0, tzinfo=UTC)),
        _mock_state("disarmed", disarm_time),  # genuine disarm = startup time
    ]
    hass = _make_hass(recorder_states)

    snapshot = _base_snapshot()
    snapshot["entities"] = [_alarm_entity(last_changed="2026-06-22T05:16:00+00:00")]

    await async_enrich_alarm_last_changed(hass, snapshot)

    assert snapshot["entities"][0]["last_changed"] == "2026-06-22T05:16:00+00:00"


@pytest.mark.asyncio
async def test_multiple_ha_restarts_finds_original_disarm() -> None:
    """Multiple HA restarts produce multiple disarmed records; finds the first."""
    original_disarm = datetime(2026, 6, 18, 14, 0, 0, tzinfo=UTC)

    recorder_states = [
        _mock_state("armed_away", datetime(2026, 6, 18, 13, 0, 0, tzinfo=UTC)),
        _mock_state("disarmed", original_disarm),
        _mock_state(
            "disarmed", datetime(2026, 6, 20, 8, 0, 0, tzinfo=UTC)
        ),  # restart 1
        _mock_state(
            "disarmed", datetime(2026, 6, 22, 5, 16, 0, tzinfo=UTC)
        ),  # restart 2
    ]
    hass = _make_hass(recorder_states)

    snapshot = _base_snapshot()
    snapshot["entities"] = [_alarm_entity(last_changed="2026-06-22T05:16:00+00:00")]

    await async_enrich_alarm_last_changed(hass, snapshot)

    assert snapshot["entities"][0]["last_changed"] == original_disarm.isoformat()


@pytest.mark.asyncio
async def test_insufficient_history_leaves_unchanged() -> None:
    """Fewer than 2 history records → snapshot left unchanged."""
    hass = _make_hass(
        recorder_states=[
            _mock_state("disarmed", datetime(2026, 6, 22, 5, 16, 0, tzinfo=UTC)),
        ]
    )

    snapshot = _base_snapshot()
    snapshot["entities"] = [_alarm_entity(last_changed="2026-06-22T05:16:00+00:00")]

    await async_enrich_alarm_last_changed(hass, snapshot)

    assert snapshot["entities"][0]["last_changed"] == "2026-06-22T05:16:00+00:00"


@pytest.mark.asyncio
async def test_recorder_error_leaves_unchanged() -> None:
    """If the recorder query fails, the snapshot is left unchanged."""
    hass = MagicMock()
    instance = MagicMock()
    instance.async_add_executor_job = AsyncMock(side_effect=RuntimeError("db error"))
    hass.data = {DATA_INSTANCE: instance}

    snapshot = _base_snapshot()
    snapshot["entities"] = [_alarm_entity(last_changed="2026-06-22T05:16:00+00:00")]

    await async_enrich_alarm_last_changed(hass, snapshot)

    assert snapshot["entities"][0]["last_changed"] == "2026-06-22T05:16:00+00:00"


@pytest.mark.asyncio
async def test_all_disarmed_within_window_uses_oldest() -> None:
    """All records disarmed within window: armed record purged, oldest used as best estimate."""
    oldest_disarm = datetime(2026, 6, 18, 10, 0, 0, tzinfo=UTC)
    recorder_states = [
        _mock_state("disarmed", oldest_disarm),
        _mock_state("disarmed", datetime(2026, 6, 20, 8, 0, 0, tzinfo=UTC)),
        _mock_state("disarmed", datetime(2026, 6, 22, 5, 16, 0, tzinfo=UTC)),
    ]
    hass = _make_hass(recorder_states)

    snapshot = _base_snapshot()
    snapshot["entities"] = [_alarm_entity(last_changed="2026-06-22T05:16:00+00:00")]

    await async_enrich_alarm_last_changed(hass, snapshot)

    assert snapshot["entities"][0]["last_changed"] == oldest_disarm.isoformat()


@pytest.mark.asyncio
async def test_real_world_armed_record_purged() -> None:
    """
    Mirrors the actual debug log: armed record purged, all 12 records are disarmed.

    The alarm was disarmed 8 days ago; both the preceding "armed" record AND any
    include_start_time_state synthetic anchor were purged from the short-term DB.
    The oldest surviving record IS the actual disarm, and the fallback path uses it.
    """
    actual_disarm = datetime(2026, 6, 14, 20, 46, 53, tzinfo=UTC)
    recorder_states = [
        # Oldest surviving record — the actual disarm 8 days ago.
        _mock_state("disarmed", actual_disarm),
        # 11 subsequent HA-restart re-reports of "disarmed".
        _mock_state("disarmed", datetime(2026, 6, 15, 12, 55, 35, tzinfo=UTC)),
        _mock_state("disarmed", datetime(2026, 6, 15, 19, 28, 37, tzinfo=UTC)),
        _mock_state("disarmed", datetime(2026, 6, 17, 2, 7, 0, tzinfo=UTC)),
        _mock_state("disarmed", datetime(2026, 6, 18, 19, 28, 49, tzinfo=UTC)),
        _mock_state("disarmed", datetime(2026, 6, 19, 12, 33, 50, tzinfo=UTC)),
        _mock_state("disarmed", datetime(2026, 6, 19, 18, 25, 35, tzinfo=UTC)),
        _mock_state("disarmed", datetime(2026, 6, 20, 5, 14, 18, tzinfo=UTC)),
        _mock_state("disarmed", datetime(2026, 6, 20, 5, 16, 33, tzinfo=UTC)),
        _mock_state("disarmed", datetime(2026, 6, 22, 13, 13, 48, tzinfo=UTC)),
        _mock_state("disarmed", datetime(2026, 6, 22, 18, 41, 8, tzinfo=UTC)),
        _mock_state("disarmed", datetime(2026, 6, 22, 19, 8, 19, tzinfo=UTC)),
    ]
    hass = _make_hass(recorder_states)

    snapshot = _base_snapshot()
    snapshot["entities"] = [_alarm_entity(last_changed="2026-06-22T19:08:19+00:00")]

    await async_enrich_alarm_last_changed(hass, snapshot)

    assert snapshot["entities"][0]["last_changed"] == actual_disarm.isoformat()


@pytest.mark.asyncio
async def test_real_world_with_armed_anchor_in_window() -> None:
    """Armed record still within recorder window: transition detection finds it directly."""
    actual_disarm = datetime(2026, 6, 14, 20, 46, 53, tzinfo=UTC)
    recorder_states = [
        # Armed record still in the short-term DB (not yet purged).
        _mock_state("armed_away", datetime(2026, 5, 28, 18, 0, 0, tzinfo=UTC)),
        # Actual disarm 8 days ago — found via transition detection.
        _mock_state("disarmed", actual_disarm),
        # 10 subsequent HA-restart re-reports of "disarmed".
        _mock_state("disarmed", datetime(2026, 6, 15, 12, 55, 35, tzinfo=UTC)),
        _mock_state("disarmed", datetime(2026, 6, 15, 19, 28, 37, tzinfo=UTC)),
        _mock_state("disarmed", datetime(2026, 6, 17, 2, 7, 0, tzinfo=UTC)),
        _mock_state("disarmed", datetime(2026, 6, 18, 19, 28, 49, tzinfo=UTC)),
        _mock_state("disarmed", datetime(2026, 6, 19, 12, 33, 50, tzinfo=UTC)),
        _mock_state("disarmed", datetime(2026, 6, 20, 1, 25, 35, tzinfo=UTC)),
        _mock_state("disarmed", datetime(2026, 6, 20, 12, 14, 18, tzinfo=UTC)),
        _mock_state("disarmed", datetime(2026, 6, 20, 12, 16, 33, tzinfo=UTC)),
        _mock_state("disarmed", datetime(2026, 6, 22, 13, 13, 48, tzinfo=UTC)),
        _mock_state("disarmed", datetime(2026, 6, 23, 1, 41, 8, tzinfo=UTC)),
    ]
    hass = _make_hass(recorder_states)

    snapshot = _base_snapshot()
    snapshot["entities"] = [_alarm_entity(last_changed="2026-06-23T01:41:08+00:00")]

    await async_enrich_alarm_last_changed(hass, snapshot)

    assert snapshot["entities"][0]["last_changed"] == actual_disarm.isoformat()


@pytest.mark.asyncio
async def test_all_disarmed_beyond_lookback_window() -> None:
    """Alarm disarmed longer than _LOOKBACK_DAYS: synthetic anchor is also disarmed."""
    recorder_states = [
        # Synthetic anchor — alarm was already disarmed before the window started.
        _mock_state("disarmed", datetime(2026, 5, 1, 10, 0, 0, tzinfo=UTC)),
        _mock_state("disarmed", datetime(2026, 5, 20, 8, 0, 0, tzinfo=UTC)),
        _mock_state("disarmed", datetime(2026, 6, 22, 5, 16, 0, tzinfo=UTC)),
    ]
    hass = _make_hass(recorder_states)

    snapshot = _base_snapshot()
    snapshot["entities"] = [_alarm_entity(last_changed="2026-06-22T05:16:00+00:00")]

    await async_enrich_alarm_last_changed(hass, snapshot)

    assert snapshot["entities"][0]["last_changed"] == ""

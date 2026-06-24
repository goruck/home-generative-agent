# ruff: noqa: S101
"""Tests for sentinel lock history enrichment."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
from homeassistant.helpers.recorder import DATA_INSTANCE

from custom_components.home_generative_agent.sentinel.lock_enrichment import (
    async_enrich_lock_last_changed,
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


def _lock_entity(
    state: str = "unlocked", last_changed: str = "2026-06-22T05:16:00+00:00"
) -> dict[str, Any]:
    return {
        "entity_id": "lock.garage_door",
        "domain": "lock",
        "state": state,
        "friendly_name": "Garage Door Lock",
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
    hass = MagicMock()
    instance = MagicMock()
    if recorder_states is not None:
        instance.async_add_executor_job = AsyncMock(
            return_value={"lock.garage_door": recorder_states}
        )
    else:
        instance.async_add_executor_job = AsyncMock(return_value={})
    hass.data = {DATA_INSTANCE: instance}
    return hass


@pytest.mark.asyncio
async def test_no_recorder_skips_enrichment() -> None:
    """If recorder isn't loaded, enrichment is a no-op."""
    hass = MagicMock()
    hass.data = {}
    snapshot = _base_snapshot()
    snapshot["entities"] = [_lock_entity(last_changed="2026-06-22T05:16:00+00:00")]

    await async_enrich_lock_last_changed(hass, snapshot)

    assert snapshot["entities"][0]["last_changed"] == "2026-06-22T05:16:00+00:00"


@pytest.mark.asyncio
async def test_no_unlocked_lock_skips_enrichment() -> None:
    """If no lock is unlocked, no recorder query is made."""
    hass = MagicMock()
    instance = MagicMock()
    instance.async_add_executor_job = AsyncMock()
    hass.data = {DATA_INSTANCE: instance}

    snapshot = _base_snapshot()
    snapshot["entities"] = [_lock_entity(state="locked")]

    await async_enrich_lock_last_changed(hass, snapshot)

    instance.async_add_executor_job.assert_not_called()


@pytest.mark.asyncio
async def test_corrects_startup_reset_direct_unlock() -> None:
    """Lock unlocked before HA startup (no unavailable) gets last_changed corrected."""
    actual_unlock = datetime(2026, 6, 20, 10, 0, 0, tzinfo=UTC)
    ha_startup = datetime(2026, 6, 22, 5, 16, 0, tzinfo=UTC)

    recorder_states = [
        _mock_state("locked", datetime(2026, 6, 20, 9, 0, 0, tzinfo=UTC)),
        _mock_state("unlocked", actual_unlock),
        _mock_state("unlocked", ha_startup),  # HA restart re-report
    ]
    hass = _make_hass(recorder_states)

    snapshot = _base_snapshot()
    snapshot["entities"] = [_lock_entity(last_changed="2026-06-22T05:16:00+00:00")]

    await async_enrich_lock_last_changed(hass, snapshot)

    assert snapshot["entities"][0]["last_changed"] == actual_unlock.isoformat()


@pytest.mark.asyncio
async def test_corrects_startup_reset_via_unavailable() -> None:
    """HA restart: unavailable → unlocked re-report; finds true unlock."""
    actual_unlock = datetime(2026, 6, 20, 10, 0, 0, tzinfo=UTC)
    ha_startup = datetime(2026, 6, 22, 5, 16, 0, tzinfo=UTC)

    recorder_states = [
        _mock_state("locked", datetime(2026, 6, 20, 9, 0, 0, tzinfo=UTC)),
        _mock_state("unlocked", actual_unlock),
        _mock_state("unavailable", ha_startup),
        _mock_state("unlocked", ha_startup),
    ]
    hass = _make_hass(recorder_states)

    snapshot = _base_snapshot()
    snapshot["entities"] = [_lock_entity(last_changed="2026-06-22T05:16:00+00:00")]

    await async_enrich_lock_last_changed(hass, snapshot)

    assert snapshot["entities"][0]["last_changed"] == actual_unlock.isoformat()


@pytest.mark.asyncio
async def test_unlocking_intermediate_state_handled() -> None:
    """Locked → unlocking → unlocked sequence; true unlock identified correctly."""
    actual_unlock = datetime(2026, 6, 22, 5, 16, 0, tzinfo=UTC)

    recorder_states = [
        _mock_state("locked", datetime(2026, 6, 22, 5, 15, 0, tzinfo=UTC)),
        _mock_state("unlocking", datetime(2026, 6, 22, 5, 15, 59, tzinfo=UTC)),
        _mock_state("unlocked", actual_unlock),
    ]
    hass = _make_hass(recorder_states)

    snapshot = _base_snapshot()
    snapshot["entities"] = [_lock_entity(last_changed="2026-06-22T05:16:00+00:00")]

    await async_enrich_lock_last_changed(hass, snapshot)

    # unlocking acts as the non-unlocked anchor; unlocked at actual_unlock is correct
    assert snapshot["entities"][0]["last_changed"] == "2026-06-22T05:16:00+00:00"


@pytest.mark.asyncio
async def test_genuine_unlock_not_changed() -> None:
    """Lock truly unlocked at startup time; no correction needed."""
    unlock_time = datetime(2026, 6, 22, 5, 16, 0, tzinfo=UTC)

    recorder_states = [
        _mock_state("locked", datetime(2026, 6, 21, 22, 0, 0, tzinfo=UTC)),
        _mock_state("unlocked", unlock_time),
    ]
    hass = _make_hass(recorder_states)

    snapshot = _base_snapshot()
    snapshot["entities"] = [_lock_entity(last_changed="2026-06-22T05:16:00+00:00")]

    await async_enrich_lock_last_changed(hass, snapshot)

    assert snapshot["entities"][0]["last_changed"] == "2026-06-22T05:16:00+00:00"


@pytest.mark.asyncio
async def test_multiple_restarts_finds_original_unlock() -> None:
    """Multiple HA restarts all re-reporting unlocked; finds first unlock."""
    original_unlock = datetime(2026, 6, 18, 14, 0, 0, tzinfo=UTC)

    recorder_states = [
        _mock_state("locked", datetime(2026, 6, 18, 13, 0, 0, tzinfo=UTC)),
        _mock_state("unlocked", original_unlock),
        _mock_state("unavailable", datetime(2026, 6, 20, 8, 0, 0, tzinfo=UTC)),
        _mock_state("unlocked", datetime(2026, 6, 20, 8, 0, 0, tzinfo=UTC)),
        _mock_state("unavailable", datetime(2026, 6, 22, 5, 16, 0, tzinfo=UTC)),
        _mock_state("unlocked", datetime(2026, 6, 22, 5, 16, 0, tzinfo=UTC)),
    ]
    hass = _make_hass(recorder_states)

    snapshot = _base_snapshot()
    snapshot["entities"] = [_lock_entity(last_changed="2026-06-22T05:16:00+00:00")]

    await async_enrich_lock_last_changed(hass, snapshot)

    assert snapshot["entities"][0]["last_changed"] == original_unlock.isoformat()


@pytest.mark.asyncio
async def test_armed_record_purged_uses_oldest_within_window() -> None:
    """All records unlocked, locked record purged: uses oldest as best estimate."""
    oldest_unlock = datetime(2026, 6, 14, 20, 46, 53, tzinfo=UTC)

    recorder_states = [
        _mock_state("unlocked", oldest_unlock),
        _mock_state("unlocked", datetime(2026, 6, 15, 12, 0, 0, tzinfo=UTC)),
        _mock_state("unlocked", datetime(2026, 6, 22, 5, 16, 0, tzinfo=UTC)),
    ]
    hass = _make_hass(recorder_states)

    snapshot = _base_snapshot()
    snapshot["entities"] = [_lock_entity(last_changed="2026-06-22T05:16:00+00:00")]

    await async_enrich_lock_last_changed(hass, snapshot)

    assert snapshot["entities"][0]["last_changed"] == oldest_unlock.isoformat()


@pytest.mark.asyncio
async def test_all_unlocked_beyond_window_clears_last_changed() -> None:
    """Synthetic anchor also unlocked (lock open > 30 days): clears last_changed."""
    recorder_states = [
        _mock_state(
            "unlocked", datetime(2026, 5, 1, 10, 0, 0, tzinfo=UTC)
        ),  # before window
        _mock_state("unlocked", datetime(2026, 5, 20, 8, 0, 0, tzinfo=UTC)),
        _mock_state("unlocked", datetime(2026, 6, 22, 5, 16, 0, tzinfo=UTC)),
    ]
    hass = _make_hass(recorder_states)

    snapshot = _base_snapshot()
    snapshot["entities"] = [_lock_entity(last_changed="2026-06-22T05:16:00+00:00")]

    await async_enrich_lock_last_changed(hass, snapshot)

    assert snapshot["entities"][0]["last_changed"] == ""


@pytest.mark.asyncio
async def test_insufficient_history_leaves_unchanged() -> None:
    """Fewer than 2 history records → snapshot unchanged."""
    hass = _make_hass(
        recorder_states=[
            _mock_state("unlocked", datetime(2026, 6, 22, 5, 16, 0, tzinfo=UTC)),
        ]
    )

    snapshot = _base_snapshot()
    snapshot["entities"] = [_lock_entity(last_changed="2026-06-22T05:16:00+00:00")]

    await async_enrich_lock_last_changed(hass, snapshot)

    assert snapshot["entities"][0]["last_changed"] == "2026-06-22T05:16:00+00:00"


@pytest.mark.asyncio
async def test_recorder_error_leaves_unchanged() -> None:
    """Recorder query failure leaves snapshot unchanged."""
    hass = MagicMock()
    instance = MagicMock()
    instance.async_add_executor_job = AsyncMock(side_effect=RuntimeError("db error"))
    hass.data = {DATA_INSTANCE: instance}

    snapshot = _base_snapshot()
    snapshot["entities"] = [_lock_entity(last_changed="2026-06-22T05:16:00+00:00")]

    await async_enrich_lock_last_changed(hass, snapshot)

    assert snapshot["entities"][0]["last_changed"] == "2026-06-22T05:16:00+00:00"


@pytest.mark.asyncio
async def test_locked_entity_not_enriched() -> None:
    """Locked (not unlocked) entities are skipped entirely."""
    hass = MagicMock()
    instance = MagicMock()
    instance.async_add_executor_job = AsyncMock()
    hass.data = {DATA_INSTANCE: instance}

    snapshot = _base_snapshot()
    snapshot["entities"] = [_lock_entity(state="locked")]

    await async_enrich_lock_last_changed(hass, snapshot)

    instance.async_add_executor_job.assert_not_called()

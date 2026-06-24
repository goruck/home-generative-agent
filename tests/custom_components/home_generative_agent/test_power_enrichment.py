# ruff: noqa: S101
"""Tests for sentinel power sensor history enrichment."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
from homeassistant.helpers.recorder import DATA_INSTANCE

from custom_components.home_generative_agent.sentinel.power_enrichment import (
    async_enrich_power_last_changed,
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


def _power_entity(
    state: str = "1498.5",
    unit: str = "W",
    last_changed: str = "2026-06-22T05:16:00+00:00",
) -> dict[str, Any]:
    return {
        "entity_id": "sensor.dishwasher_power",
        "domain": "sensor",
        "state": state,
        "friendly_name": "Dishwasher Power",
        "area": None,
        "attributes": {"device_class": "power", "unit_of_measurement": unit},
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
            return_value={"sensor.dishwasher_power": recorder_states}
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
    snapshot["entities"] = [_power_entity(last_changed="2026-06-22T05:16:00+00:00")]

    await async_enrich_power_last_changed(hass, snapshot)

    assert snapshot["entities"][0]["last_changed"] == "2026-06-22T05:16:00+00:00"


@pytest.mark.asyncio
async def test_off_sensor_skips_enrichment() -> None:
    """Power sensor below off threshold is not queried."""
    hass = MagicMock()
    instance = MagicMock()
    instance.async_add_executor_job = AsyncMock()
    hass.data = {DATA_INSTANCE: instance}

    snapshot = _base_snapshot()
    snapshot["entities"] = [_power_entity(state="2.0")]  # effectively off

    await async_enrich_power_last_changed(hass, snapshot)

    instance.async_add_executor_job.assert_not_called()


@pytest.mark.asyncio
async def test_corrects_startup_reset() -> None:
    """Power sensor 'on' since before restart: corrects last_changed to true on-time."""
    true_on_time = datetime(2026, 6, 22, 3, 0, 0, tzinfo=UTC)
    ha_startup = datetime(2026, 6, 22, 5, 16, 0, tzinfo=UTC)

    recorder_states = [
        _mock_state("0.5", datetime(2026, 6, 22, 2, 59, 0, tzinfo=UTC)),  # off
        _mock_state("1498.5", true_on_time),  # appliance turned on
        _mock_state("1501.2", datetime(2026, 6, 22, 4, 0, 0, tzinfo=UTC)),
        _mock_state("unavailable", ha_startup),
        _mock_state("1500.1", ha_startup),  # restart re-report
    ]
    hass = _make_hass(recorder_states)

    snapshot = _base_snapshot()
    snapshot["entities"] = [_power_entity(last_changed="2026-06-22T05:16:00+00:00")]

    await async_enrich_power_last_changed(hass, snapshot)

    assert snapshot["entities"][0]["last_changed"] == true_on_time.isoformat()


@pytest.mark.asyncio
async def test_genuine_on_since_startup_not_changed() -> None:
    """Appliance genuinely started at HA startup: no correction needed."""
    startup_and_on = datetime(2026, 6, 22, 5, 16, 0, tzinfo=UTC)

    recorder_states = [
        _mock_state(
            "0.5", datetime(2026, 6, 22, 5, 15, 0, tzinfo=UTC)
        ),  # off before startup
        _mock_state("1498.5", startup_and_on),  # actually started at startup
    ]
    hass = _make_hass(recorder_states)

    snapshot = _base_snapshot()
    snapshot["entities"] = [_power_entity(last_changed="2026-06-22T05:16:00+00:00")]

    await async_enrich_power_last_changed(hass, snapshot)

    assert snapshot["entities"][0]["last_changed"] == "2026-06-22T05:16:00+00:00"


@pytest.mark.asyncio
async def test_multiple_restarts_finds_true_on_time() -> None:
    """Multiple HA restarts after appliance started: finds original on-time."""
    true_on_time = datetime(2026, 6, 22, 1, 0, 0, tzinfo=UTC)

    recorder_states = [
        _mock_state("0.0", datetime(2026, 6, 22, 0, 59, 0, tzinfo=UTC)),
        _mock_state("1498.5", true_on_time),
        _mock_state("unavailable", datetime(2026, 6, 22, 3, 0, 0, tzinfo=UTC)),
        _mock_state("1500.0", datetime(2026, 6, 22, 3, 0, 0, tzinfo=UTC)),
        _mock_state("unavailable", datetime(2026, 6, 22, 5, 16, 0, tzinfo=UTC)),
        _mock_state("1499.5", datetime(2026, 6, 22, 5, 16, 0, tzinfo=UTC)),
    ]
    hass = _make_hass(recorder_states)

    snapshot = _base_snapshot()
    snapshot["entities"] = [_power_entity(last_changed="2026-06-22T05:16:00+00:00")]

    await async_enrich_power_last_changed(hass, snapshot)

    assert snapshot["entities"][0]["last_changed"] == true_on_time.isoformat()


@pytest.mark.asyncio
async def test_kw_unit_sensor_corrected() -> None:
    """kW-unit sensor: off threshold applied in native kW units."""
    true_on_time = datetime(2026, 6, 22, 3, 0, 0, tzinfo=UTC)

    recorder_states = [
        _mock_state("0.001", datetime(2026, 6, 22, 2, 59, 0, tzinfo=UTC)),  # 1W — off
        _mock_state("1.498", true_on_time),  # 1498W — on
        _mock_state("unavailable", datetime(2026, 6, 22, 5, 16, 0, tzinfo=UTC)),
        _mock_state("1.500", datetime(2026, 6, 22, 5, 16, 0, tzinfo=UTC)),
    ]
    hass = _make_hass(recorder_states)

    snapshot = _base_snapshot()
    snapshot["entities"] = [
        _power_entity(
            state="1.500", unit="kW", last_changed="2026-06-22T05:16:00+00:00"
        )
    ]

    await async_enrich_power_last_changed(hass, snapshot)

    assert snapshot["entities"][0]["last_changed"] == true_on_time.isoformat()


@pytest.mark.asyncio
async def test_all_on_within_window_uses_oldest() -> None:
    """No off reading in window: uses oldest within-window record as best estimate."""
    oldest_on = datetime(2026, 6, 16, 10, 0, 0, tzinfo=UTC)

    recorder_states = [
        _mock_state("1498.5", oldest_on),
        _mock_state("1501.0", datetime(2026, 6, 18, 8, 0, 0, tzinfo=UTC)),
        _mock_state("1499.8", datetime(2026, 6, 22, 5, 16, 0, tzinfo=UTC)),
    ]
    hass = _make_hass(recorder_states)

    snapshot = _base_snapshot()
    snapshot["entities"] = [_power_entity(last_changed="2026-06-22T05:16:00+00:00")]

    await async_enrich_power_last_changed(hass, snapshot)

    assert snapshot["entities"][0]["last_changed"] == oldest_on.isoformat()


@pytest.mark.asyncio
async def test_all_on_beyond_window_leaves_unchanged() -> None:
    """Synthetic anchor also on (appliance on > lookback): last_changed left unchanged."""
    recorder_states = [
        _mock_state(
            "1498.5", datetime(2026, 5, 1, 0, 0, 0, tzinfo=UTC)
        ),  # well before 30-day window
        _mock_state("1500.0", datetime(2026, 6, 18, 8, 0, 0, tzinfo=UTC)),
        _mock_state("1499.8", datetime(2026, 6, 22, 5, 16, 0, tzinfo=UTC)),
    ]
    hass = _make_hass(recorder_states)

    snapshot = _base_snapshot()
    snapshot["entities"] = [_power_entity(last_changed="2026-06-22T05:16:00+00:00")]

    await async_enrich_power_last_changed(hass, snapshot)

    # Cannot determine true on-since beyond lookback window; leave unchanged
    assert snapshot["entities"][0]["last_changed"] == "2026-06-22T05:16:00+00:00"


@pytest.mark.asyncio
async def test_recorder_error_leaves_unchanged() -> None:
    """Recorder query failure leaves snapshot unchanged."""
    hass = MagicMock()
    instance = MagicMock()
    instance.async_add_executor_job = AsyncMock(side_effect=RuntimeError("db error"))
    hass.data = {DATA_INSTANCE: instance}

    snapshot = _base_snapshot()
    snapshot["entities"] = [_power_entity(last_changed="2026-06-22T05:16:00+00:00")]

    await async_enrich_power_last_changed(hass, snapshot)

    assert snapshot["entities"][0]["last_changed"] == "2026-06-22T05:16:00+00:00"


@pytest.mark.asyncio
async def test_insufficient_history_leaves_unchanged() -> None:
    """Fewer than 2 history records → snapshot unchanged."""
    hass = _make_hass(
        recorder_states=[
            _mock_state("1498.5", datetime(2026, 6, 22, 5, 16, 0, tzinfo=UTC)),
        ]
    )

    snapshot = _base_snapshot()
    snapshot["entities"] = [_power_entity(last_changed="2026-06-22T05:16:00+00:00")]

    await async_enrich_power_last_changed(hass, snapshot)

    assert snapshot["entities"][0]["last_changed"] == "2026-06-22T05:16:00+00:00"


@pytest.mark.asyncio
async def test_non_power_sensor_not_enriched() -> None:
    """Non-power sensor domain is not enriched."""
    hass = MagicMock()
    instance = MagicMock()
    instance.async_add_executor_job = AsyncMock()
    hass.data = {DATA_INSTANCE: instance}

    snapshot = _base_snapshot()
    snapshot["entities"] = [
        {
            "entity_id": "sensor.temperature",
            "domain": "sensor",
            "state": "21.5",
            "friendly_name": "Temperature",
            "area": None,
            "attributes": {"device_class": "temperature", "unit_of_measurement": "°C"},
            "last_changed": "2026-06-22T05:16:00+00:00",
            "last_updated": "2026-06-22T05:16:00+00:00",
        }
    ]

    await async_enrich_power_last_changed(hass, snapshot)

    instance.async_add_executor_job.assert_not_called()

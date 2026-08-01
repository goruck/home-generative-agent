# ruff: noqa: S101
"""Tests for sentinel power sensor history enrichment."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from homeassistant.helpers.recorder import DATA_INSTANCE

from custom_components.home_generative_agent.sentinel.power_enrichment import (
    async_enrich_power_last_changed,
)
from custom_components.home_generative_agent.snapshot.schema import validate_snapshot

_NOW = datetime(2026, 6, 22, 5, 16, 0, tzinfo=UTC)


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


def _mock_state(
    state: str, last_changed: datetime, unit: str | None = None
) -> MagicMock:
    s = MagicMock()
    s.state = state
    s.last_changed = last_changed
    # No unit → empty attributes dict: the walk falls back to the sensor's
    # current unit factor, matching rows recorded without attribute payloads.
    s.attributes = {"unit_of_measurement": unit} if unit is not None else {}
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


@pytest.fixture(autouse=True)
def freeze_utcnow():
    with patch("homeassistant.util.dt.utcnow", return_value=_NOW):
        yield


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
    true_on_time = _NOW - timedelta(hours=2, minutes=16)  # ~June 22 03:00

    recorder_states = [
        _mock_state("0.5", _NOW - timedelta(hours=2, minutes=17)),  # off
        _mock_state("1498.5", true_on_time),  # appliance turned on
        _mock_state("1501.2", _NOW - timedelta(hours=1, minutes=16)),
        _mock_state("unavailable", _NOW),
        _mock_state("1500.1", _NOW),  # restart re-report
    ]
    hass = _make_hass(recorder_states)

    snapshot = _base_snapshot()
    snapshot["entities"] = [_power_entity(last_changed="2026-06-22T05:16:00+00:00")]

    await async_enrich_power_last_changed(hass, snapshot)

    assert snapshot["entities"][0]["last_changed"] == true_on_time.isoformat()


@pytest.mark.asyncio
async def test_genuine_on_since_startup_not_changed() -> None:
    """Appliance genuinely started at HA startup: no correction needed."""
    startup_and_on = _NOW

    recorder_states = [
        _mock_state("0.5", _NOW - timedelta(minutes=1)),  # off before startup
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
    true_on_time = _NOW - timedelta(hours=4, minutes=16)  # ~June 22 01:00

    recorder_states = [
        _mock_state("0.0", _NOW - timedelta(hours=4, minutes=17)),
        _mock_state("1498.5", true_on_time),
        _mock_state("unavailable", _NOW - timedelta(hours=2, minutes=16)),
        _mock_state("1500.0", _NOW - timedelta(hours=2, minutes=16)),
        _mock_state("unavailable", _NOW),
        _mock_state("1499.5", _NOW),
    ]
    hass = _make_hass(recorder_states)

    snapshot = _base_snapshot()
    snapshot["entities"] = [_power_entity(last_changed="2026-06-22T05:16:00+00:00")]

    await async_enrich_power_last_changed(hass, snapshot)

    assert snapshot["entities"][0]["last_changed"] == true_on_time.isoformat()


@pytest.mark.asyncio
async def test_kw_unit_sensor_corrected() -> None:
    """kW-unit sensor: off threshold applied in native kW units."""
    true_on_time = _NOW - timedelta(hours=2, minutes=16)  # ~June 22 03:00

    recorder_states = [
        _mock_state("0.001", _NOW - timedelta(hours=2, minutes=17)),  # 1W — off
        _mock_state("1.498", true_on_time),  # 1498W — on
        _mock_state("unavailable", _NOW),
        _mock_state("1.500", _NOW),
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
async def test_mw_unit_sensor_corrected() -> None:
    """MW-unit sensor: off threshold applied in native MW units."""
    true_on_time = _NOW - timedelta(hours=2, minutes=16)

    recorder_states = [
        _mock_state("0.000001", _NOW - timedelta(hours=2, minutes=17)),  # 1W — off
        _mock_state("0.0015", true_on_time),  # 1500W — on
        _mock_state("unavailable", _NOW),
        _mock_state("0.0015", _NOW),
    ]
    hass = _make_hass(recorder_states)

    snapshot = _base_snapshot()
    snapshot["entities"] = [
        _power_entity(
            state="0.0015", unit="MW", last_changed="2026-06-22T05:16:00+00:00"
        )
    ]

    await async_enrich_power_last_changed(hass, snapshot)

    assert snapshot["entities"][0]["last_changed"] == true_on_time.isoformat()


@pytest.mark.asyncio
async def test_non_power_unit_sensor_skipped() -> None:
    """A device_class:power sensor in VA cannot be compared to the watts off level."""
    hass = MagicMock()
    instance = MagicMock()
    instance.async_add_executor_job = AsyncMock()
    hass.data = {DATA_INSTANCE: instance}

    snapshot = _base_snapshot()
    snapshot["entities"] = [
        _power_entity(state="1500", unit="VA", last_changed="2026-06-22T05:16:00+00:00")
    ]

    await async_enrich_power_last_changed(hass, snapshot)

    instance.async_add_executor_job.assert_not_called()
    assert snapshot["entities"][0]["last_changed"] == "2026-06-22T05:16:00+00:00"


@pytest.mark.asyncio
async def test_unit_only_sensor_enriched() -> None:
    """A sensor with a power unit but no device_class is still enriched."""
    true_on_time = _NOW - timedelta(hours=2, minutes=16)

    recorder_states = [
        _mock_state("2", _NOW - timedelta(hours=2, minutes=17)),  # off
        _mock_state("1500", true_on_time),  # on
        _mock_state("1500", _NOW),
    ]
    hass = _make_hass(recorder_states)

    snapshot = _base_snapshot()
    entity = _power_entity(state="1500", last_changed="2026-06-22T05:16:00+00:00")
    entity["attributes"] = {"unit_of_measurement": "W"}
    snapshot["entities"] = [entity]

    await async_enrich_power_last_changed(hass, snapshot)

    assert snapshot["entities"][0]["last_changed"] == true_on_time.isoformat()


@pytest.mark.asyncio
async def test_missing_unit_sensor_treated_as_watts() -> None:
    """A device_class:power sensor with no unit is compared in watts."""
    true_on_time = _NOW - timedelta(hours=2, minutes=16)

    recorder_states = [
        _mock_state("2", _NOW - timedelta(hours=2, minutes=17)),  # off
        _mock_state("1500", true_on_time),  # on
        _mock_state("1500", _NOW),
    ]
    hass = _make_hass(recorder_states)

    snapshot = _base_snapshot()
    entity = _power_entity(state="1500", last_changed="2026-06-22T05:16:00+00:00")
    entity["attributes"] = {"device_class": "power"}
    snapshot["entities"] = [entity]

    await async_enrich_power_last_changed(hass, snapshot)

    assert snapshot["entities"][0]["last_changed"] == true_on_time.isoformat()


@pytest.mark.asyncio
async def test_historical_rows_honor_their_own_unit() -> None:
    """
    A sensor reconfigured from W to kW mid-window walks history correctly.

    The historical 5 (W, off) row must read as 5 W, not 5 kW: interpreting old
    rows in the sensor's *current* unit would miss the off boundary and push
    the on-since arbitrarily far back.
    """
    true_on_time = _NOW - timedelta(hours=2, minutes=16)

    recorder_states = [
        _mock_state("5", _NOW - timedelta(hours=2, minutes=17), unit="W"),  # off
        _mock_state("1498", true_on_time, unit="W"),  # on, recorded as W
        _mock_state("1.500", _NOW, unit="kW"),  # after unit reconfiguration
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
async def test_non_finite_states_never_treated_as_on() -> None:
    """
    nan/inf states are rejected at both the current-state gate and the walk.

    nan compares False against the off threshold, so without the isfinite
    guard a nan reading would look 'on' and could be captured as the on-since
    boundary — rewriting last_changed from garbage.
    """
    # Current state nan: enrichment skips the sensor without querying.
    hass = MagicMock()
    instance = MagicMock()
    instance.async_add_executor_job = AsyncMock()
    hass.data = {DATA_INSTANCE: instance}
    snapshot = _base_snapshot()
    snapshot["entities"] = [
        _power_entity(state="nan", last_changed="2026-06-22T05:16:00+00:00")
    ]
    await async_enrich_power_last_changed(hass, snapshot)
    instance.async_add_executor_job.assert_not_called()

    # Historical nan/inf rows are skipped like transients, not read as 'on'.
    true_on_time = _NOW - timedelta(hours=2, minutes=16)
    recorder_states = [
        _mock_state("0.5", _NOW - timedelta(hours=2, minutes=17)),  # off
        _mock_state("nan", _NOW - timedelta(hours=2, minutes=17)),
        _mock_state("1498.5", true_on_time),  # true on-start
        _mock_state("1e309", _NOW - timedelta(hours=1)),  # inf
        _mock_state("1500.1", _NOW),
    ]
    hass = _make_hass(recorder_states)
    snapshot = _base_snapshot()
    snapshot["entities"] = [_power_entity(last_changed="2026-06-22T05:16:00+00:00")]
    await async_enrich_power_last_changed(hass, snapshot)
    assert snapshot["entities"][0]["last_changed"] == true_on_time.isoformat()


@pytest.mark.asyncio
async def test_unhashable_unit_attribute_does_not_crash() -> None:
    """
    A poisoned unit_of_measurement (list) must not raise from the filter.

    Raw frozenset membership hashes its operand; an unhashable unit on a
    sensor without device_class:power previously raised TypeError, which
    propagated out of the engine's unguarded enrichment call and killed the
    Sentinel run loop until reload.
    """
    hass = MagicMock()
    instance = MagicMock()
    instance.async_add_executor_job = AsyncMock()
    hass.data = {DATA_INSTANCE: instance}

    snapshot = _base_snapshot()
    entity = _power_entity(state="1500", last_changed="2026-06-22T05:16:00+00:00")
    entity["attributes"] = {"unit_of_measurement": ["W"]}  # no device_class
    snapshot["entities"] = [entity]

    await async_enrich_power_last_changed(hass, snapshot)

    instance.async_add_executor_job.assert_not_called()
    assert snapshot["entities"][0]["last_changed"] == "2026-06-22T05:16:00+00:00"


@pytest.mark.asyncio
async def test_all_on_within_window_uses_oldest() -> None:
    """No off reading in window: uses oldest within-window record as best estimate."""
    oldest_on = _NOW - timedelta(days=5, hours=19, minutes=16)  # ~June 16 10:00

    recorder_states = [
        _mock_state("1498.5", oldest_on),
        _mock_state("1501.0", _NOW - timedelta(days=3, hours=21, minutes=16)),
        _mock_state("1499.8", _NOW),
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
        _mock_state("1498.5", _NOW - timedelta(days=52)),  # before 30-day window
        _mock_state("1500.0", _NOW - timedelta(days=3, hours=21, minutes=16)),
        _mock_state("1499.8", _NOW),
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
            _mock_state("1498.5", _NOW),
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

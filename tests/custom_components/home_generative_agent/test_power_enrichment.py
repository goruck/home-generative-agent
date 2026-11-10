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
_PE = "custom_components.home_generative_agent.sentinel.power_enrichment"


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


class _FakeRecorder:
    """
    Serve one sensor's rows the way state_changes_during_period does.

    Rows are given oldest first.  A query returns rows after ``start`` and
    before ``end``; with ``descending`` and ``limit`` the newest ``limit``
    of them, newest first; and with ``include_start_time_state`` the newest
    row at or before ``start`` last.
    """

    def __init__(self, rows: list[MagicMock]) -> None:
        self.rows = rows
        self.queries: list[dict[str, Any]] = []

    def __call__(self, job: Any) -> dict[str, list[MagicMock]]:
        _hass, start, end, entity_id = job.args
        kw = job.keywords
        self.queries.append({"start": start, "end": end, **kw})
        assert kw["descending"] is True
        window = [
            r
            for r in self.rows
            if r.last_changed > start and (end is None or r.last_changed < end)
        ]
        page = list(reversed(window))[: kw["limit"]]
        if kw["include_start_time_state"]:
            before = [r for r in self.rows if r.last_changed <= start]
            if before:
                page.append(before[-1])
        return {entity_id: page} if page else {}


def _make_hass(recorder_states: list[MagicMock] | None = None) -> MagicMock:
    hass = MagicMock()
    instance = MagicMock()
    instance.async_add_executor_job = AsyncMock(
        side_effect=_FakeRecorder(recorder_states or [])
    )
    hass.data = {DATA_INSTANCE: instance}
    return hass


def _recorder(hass: MagicMock) -> _FakeRecorder:
    return hass.data[DATA_INSTANCE].async_add_executor_job.side_effect


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


def _on_rows(newest: datetime, count: int, step: timedelta) -> list[MagicMock]:
    """``count`` 'on' readings ending at ``newest``, oldest first."""
    return [_mock_state("1500", newest - step * i) for i in range(count - 1, -1, -1)]


async def _run(
    hass: MagicMock, cache: dict[str, Any], state: str = "1500", unit: str = "W"
) -> str:
    snapshot = _base_snapshot()
    snapshot["entities"] = [
        _power_entity(state=state, unit=unit, last_changed=_NOW.isoformat())
    ]
    await async_enrich_power_last_changed(hass, snapshot, cache)
    return snapshot["entities"][0]["last_changed"]


@pytest.mark.asyncio
async def test_walks_back_a_page_at_a_time() -> None:
    """The off reading several pages back is found, one page per query."""
    true_on = _NOW - timedelta(minutes=10)
    rows = [
        _mock_state("0.5", true_on - timedelta(minutes=1)),
        *_on_rows(_NOW, 11, timedelta(minutes=1)),
    ]
    hass = _make_hass(rows)

    with patch(f"{_PE}._PAGE_ROWS", 3):
        assert await _run(hass, {}) == true_on.isoformat()

    queries = _recorder(hass).queries
    assert [q["limit"] for q in queries] == [3, 3, 3, 3]
    assert queries[0]["end"] is None
    # Each page ends where the one before it stopped.
    assert queries[1]["end"] == _NOW - timedelta(minutes=2)


@pytest.mark.asyncio
async def test_stops_at_the_first_off_reading() -> None:
    """Older history past the last off reading is never read."""
    true_on = _NOW - timedelta(minutes=2)
    rows = [
        *_on_rows(_NOW - timedelta(hours=1), 50, timedelta(seconds=10)),
        _mock_state("0.5", _NOW - timedelta(minutes=3)),
        *_on_rows(_NOW, 3, timedelta(minutes=1)),
    ]
    hass = _make_hass(rows)

    with patch(f"{_PE}._PAGE_ROWS", 10):
        assert await _run(hass, {}) == true_on.isoformat()

    assert len(_recorder(hass).queries) == 1


@pytest.mark.asyncio
async def test_row_cap_leaves_the_start_unknown_and_is_remembered() -> None:
    """
    Past the cap the start is unknown, and later runs do not walk back again.

    A sensor that reports every second and has been on for hours would
    otherwise be read back to its last off reading every run.
    """
    rows = [
        _mock_state("0.5", _NOW - timedelta(hours=3)),
        *_on_rows(_NOW, 10, timedelta(minutes=1)),
    ]
    hass = _make_hass(rows)
    cache: dict[str, Any] = {}

    with patch(f"{_PE}._PAGE_ROWS", 2), patch(f"{_PE}._MAX_ROWS", 4):
        assert await _run(hass, cache) == _NOW.isoformat()
        assert sum(q["limit"] for q in _recorder(hass).queries) == 4
        assert cache["sensor.dishwasher_power"].unknown

        _recorder(hass).queries.clear()
        rows.append(_mock_state("1500", _NOW + timedelta(minutes=1)))
        assert await _run(hass, cache) == _NOW.isoformat()

    (query,) = _recorder(hass).queries
    assert query["start"] == _NOW
    assert query["include_start_time_state"] is False


@pytest.mark.asyncio
async def test_next_run_reads_only_newer_readings() -> None:
    """The episode start found once is reused; only new readings are read."""
    true_on = _NOW - timedelta(hours=2)
    rows = [
        _mock_state("0.5", true_on - timedelta(minutes=1)),
        _mock_state("1500", true_on),
        _mock_state("1500", _NOW),
    ]
    hass = _make_hass(rows)
    cache: dict[str, Any] = {}

    assert await _run(hass, cache) == true_on.isoformat()
    _recorder(hass).queries.clear()
    rows.append(_mock_state("1490", _NOW + timedelta(minutes=5)))

    assert await _run(hass, cache) == true_on.isoformat()
    (query,) = _recorder(hass).queries
    assert query["start"] == _NOW
    assert query["end"] is None


@pytest.mark.asyncio
async def test_off_and_on_again_between_runs_starts_a_new_episode() -> None:
    """An off reading among the new readings moves the start forward."""
    true_on = _NOW - timedelta(hours=2)
    rows = [
        _mock_state("0.5", true_on - timedelta(minutes=1)),
        _mock_state("1500", true_on),
        _mock_state("1500", _NOW),
    ]
    hass = _make_hass(rows)
    cache: dict[str, Any] = {}
    assert await _run(hass, cache) == true_on.isoformat()

    restarted = _NOW + timedelta(minutes=3)
    rows += [
        _mock_state("0.2", _NOW + timedelta(minutes=1)),
        _mock_state("1500", restarted),
        _mock_state("1500", _NOW + timedelta(minutes=4)),
    ]
    assert await _run(hass, cache) == restarted.isoformat()


@pytest.mark.asyncio
async def test_episode_beginning_after_an_off_run_is_found() -> None:
    """Seen off at the last run, then on: the start is the first new reading."""
    rows = [
        _mock_state("1500", _NOW - timedelta(hours=1)),
        _mock_state("0.5", _NOW),
    ]
    hass = _make_hass(rows)
    cache: dict[str, Any] = {}
    # The recorder lags the live reading: its newest row is still off.
    assert await _run(hass, cache) == _NOW.isoformat()
    assert cache["sensor.dishwasher_power"].since is None

    started = _NOW + timedelta(minutes=1)
    rows += [
        _mock_state("1500", started),
        _mock_state("1500", _NOW + timedelta(minutes=2)),
    ]
    assert await _run(hass, cache) == started.isoformat()


@pytest.mark.asyncio
async def test_off_or_missing_sensor_is_forgotten() -> None:
    """A sensor that reads off this run is dropped, so its next episode is read anew."""
    rows = [
        _mock_state("0.5", _NOW - timedelta(hours=1)),
        _mock_state("1500", _NOW - timedelta(minutes=30)),
        _mock_state("1500", _NOW),
    ]
    hass = _make_hass(rows)
    cache: dict[str, Any] = {}
    await _run(hass, cache)
    assert "sensor.dishwasher_power" in cache

    await _run(hass, cache, state="2.0")
    assert cache == {}

    await _run(hass, cache)
    snapshot = _base_snapshot()  # the sensor is gone
    await async_enrich_power_last_changed(hass, snapshot, cache)
    assert cache == {}


@pytest.mark.asyncio
async def test_unit_change_walks_back_again() -> None:
    """A remembered start read in one unit is not reused after a unit change."""
    true_on = _NOW - timedelta(hours=2)
    rows = [
        _mock_state("5", true_on - timedelta(minutes=1), unit="W"),
        _mock_state("1500", true_on, unit="W"),
        _mock_state("1500", _NOW, unit="W"),
    ]
    hass = _make_hass(rows)
    cache: dict[str, Any] = {}
    assert await _run(hass, cache) == true_on.isoformat()

    _recorder(hass).queries.clear()
    rows.append(_mock_state("1.5", _NOW + timedelta(minutes=1), unit="kW"))
    assert await _run(hass, cache, state="1.5", unit="kW") == true_on.isoformat()
    assert _recorder(hass).queries[0]["include_start_time_state"] is True

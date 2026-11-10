# ruff: noqa: S101
"""Tests for sentinel power sensor history enrichment."""

from __future__ import annotations

import asyncio
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
    Serve one sensor's rows the way the recorder history API does.

    Rows are given oldest first.  get_last_state_changes returns the newest
    ``number_of_states`` rows, oldest first.  state_changes_during_period
    returns the rows after ``start`` and before ``end``, oldest first, the
    oldest ``limit`` of them; descending queries are refused, since with a
    limit Home Assistant before 2026.11 returns the oldest rows for them.
    """

    def __init__(self, rows: list[MagicMock]) -> None:
        self.rows = rows
        self.queries: list[dict[str, Any]] = []
        # Enrichment swallows recorder errors, so a refused query must be
        # seen here: every test checks this stays empty.
        self.refused: list[str] = []

    def __call__(self, job: Any) -> dict[str, list[MagicMock]]:
        try:
            return self._serve(job)
        except AssertionError as err:
            self.refused.append(repr(err) or job.func.__name__)
            raise

    def _serve(self, job: Any) -> dict[str, list[MagicMock]]:
        name = job.func.__name__
        if name == "get_last_state_changes":
            _hass, count, entity_id = job.args
            ordered = sorted(self.rows, key=lambda r: r.last_changed)
            page = ordered[-count:]
            self.queries.append({"newest": count, "returned": len(page)})
            return {entity_id: page} if page else {}
        assert name == "state_changes_during_period"
        _hass, start, end, entity_id = job.args
        kw = job.keywords
        assert kw["descending"] is False
        assert kw["no_attributes"] is False  # each row's own unit is needed
        assert kw["limit"]
        assert kw["include_start_time_state"] is False
        page = [r for r in self.rows if start < r.last_changed < end][: kw["limit"]]
        self.queries.append({"start": start, "end": end, **kw, "returned": len(page)})
        return {entity_id: page} if page else {}


_RECORDERS: list[_FakeRecorder] = []


def _make_hass(recorder_states: list[MagicMock] | None = None) -> MagicMock:
    hass = MagicMock()
    instance = MagicMock()
    recorder = _FakeRecorder(recorder_states or [])
    _RECORDERS.append(recorder)
    instance.async_add_executor_job = AsyncMock(side_effect=recorder)
    hass.data = {DATA_INSTANCE: instance}
    return hass


def _recorder(hass: MagicMock) -> _FakeRecorder:
    return hass.data[DATA_INSTANCE].async_add_executor_job.side_effect


_CLOCK = [_NOW]


_MONO = [1000.0]


@pytest.fixture(autouse=True)
def freeze_utcnow():
    _CLOCK[0] = _NOW
    _MONO[0] = 1000.0
    _RECORDERS.clear()
    with (
        patch("homeassistant.util.dt.utcnow", side_effect=lambda: _CLOCK[0]),
        patch(f"{_PE}.time.monotonic", side_effect=lambda: _MONO[0]),
    ):
        yield
    assert [r.refused for r in _RECORDERS if r.refused] == []


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
async def test_on_since_before_the_window_is_on_at_least_since_its_start() -> None:
    """
    On at the window's start: last_changed becomes the window's start.

    The appliance has been on at least 30 days, so a long run never reads as
    one that just changed (the baseline cycle-completion check would read a
    fresh last_changed as a cycle that just ended).
    """
    recorder_states = [
        _mock_state("1498.5", _NOW - timedelta(days=52)),  # before 30-day window
        _mock_state("1500.0", _NOW - timedelta(days=3, hours=21, minutes=16)),
        _mock_state("1499.8", _NOW),
    ]
    hass = _make_hass(recorder_states)
    cache: dict[str, Any] = {}

    snapshot = _base_snapshot()
    snapshot["entities"] = [_power_entity(last_changed="2026-06-22T05:16:00+00:00")]

    await async_enrich_power_last_changed(hass, snapshot, cache)

    window_start = _NOW - timedelta(days=30)
    assert snapshot["entities"][0]["last_changed"] == window_start.isoformat()
    assert cache["sensor.dishwasher_power"].unknown


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
        _power_entity(state=state, unit=unit, last_changed=_CLOCK[0].isoformat())
    ]
    await async_enrich_power_last_changed(hass, snapshot, cache)
    return snapshot["entities"][0]["last_changed"]


def _advance(by: timedelta) -> datetime:
    """Move the wall clock and, when time really passes, the monotonic one."""
    _CLOCK[0] += by
    _MONO[0] += max(by.total_seconds(), 0.0)
    return _CLOCK[0]


@pytest.mark.asyncio
async def test_readings_after_the_run_began_are_not_read() -> None:
    """An off and on recorded while the walk runs belongs to the next run."""
    true_on = _NOW - timedelta(hours=2)
    rows = [
        _mock_state("0.5", true_on - timedelta(minutes=1)),
        _mock_state("1500", true_on),
        _mock_state("1500", _NOW),
        _mock_state("0.5", _NOW + timedelta(seconds=3)),
        _mock_state("1500", _NOW + timedelta(seconds=6)),
    ]
    hass = _make_hass(rows)

    assert await _run(hass, {}) == true_on.isoformat()


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
        _mock_state("1500", _advance(timedelta(minutes=4))),
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
        _mock_state("1500", _advance(timedelta(minutes=2))),
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


@pytest.mark.parametrize("state", ["unavailable", "unknown", "garbage"])
@pytest.mark.asyncio
async def test_a_brief_drop_out_keeps_the_remembered_start(state: str) -> None:
    """One unavailable reading must not cost the start and a walk from scratch."""
    true_on = _NOW - timedelta(hours=8)
    rows = [
        _mock_state("0.5", true_on - timedelta(minutes=1)),
        _mock_state("1500", true_on),
        _mock_state("1500", _NOW),
    ]
    hass = _make_hass(rows)
    cache: dict[str, Any] = {}
    assert await _run(hass, cache) == true_on.isoformat()

    _recorder(hass).queries.clear()
    _advance(timedelta(minutes=5))
    await _run(hass, cache, state=state)
    assert "sensor.dishwasher_power" in cache
    assert _recorder(hass).queries == []

    rows.append(_mock_state("1500", _advance(timedelta(minutes=5))))
    assert await _run(hass, cache) == true_on.isoformat()
    assert all(not q["include_start_time_state"] for q in _recorder(hass).queries)


@pytest.mark.asyncio
async def test_recorder_error_keeps_the_remembered_start() -> None:
    """A failed query leaves the snapshot and the remembered entry alone."""
    true_on = _NOW - timedelta(hours=2)
    rows = [
        _mock_state("0.5", true_on - timedelta(minutes=1)),
        _mock_state("1500", true_on),
        _mock_state("1500", _NOW),
    ]
    hass = _make_hass(rows)
    cache: dict[str, Any] = {}
    await _run(hass, cache)
    before = cache["sensor.dishwasher_power"]

    hass.data[DATA_INSTANCE].async_add_executor_job.side_effect = RuntimeError("db")
    _advance(timedelta(minutes=5))
    assert await _run(hass, cache) == _CLOCK[0].isoformat()
    assert cache["sensor.dishwasher_power"] is before


@pytest.mark.asyncio
async def test_clock_set_back_walks_back_again() -> None:
    """A remembered position later than now would hide every new reading."""
    true_on = _NOW - timedelta(hours=2)
    rows = [
        _mock_state("0.5", true_on - timedelta(minutes=1)),
        _mock_state("1500", true_on),
        _mock_state("1500", _NOW),
    ]
    hass = _make_hass(rows)
    cache: dict[str, Any] = {}
    assert await _run(hass, cache) == true_on.isoformat()

    restarted = _NOW - timedelta(minutes=30)
    rows[-1:] = [
        _mock_state("0.2", restarted - timedelta(minutes=1)),
        _mock_state("1500", restarted),
    ]
    _advance(timedelta(minutes=-20))  # NTP pulled the clock back
    assert await _run(hass, cache) == restarted.isoformat()


@pytest.mark.asyncio
async def test_a_found_start_is_not_marked_a_lower_bound() -> None:
    true_on = _NOW - timedelta(hours=2)
    rows = [
        _mock_state("0.5", true_on - timedelta(minutes=1)),
        _mock_state("1500", true_on),
        _mock_state("1500", _NOW),
    ]
    hass = _make_hass(rows)
    snapshot = _base_snapshot()
    snapshot["entities"] = [_power_entity(state="1500", last_changed=_NOW.isoformat())]
    await async_enrich_power_last_changed(hass, snapshot, {})
    assert "last_changed_is_lower_bound" not in snapshot["entities"][0]


def _newest_queries(hass: MagicMock) -> list[int]:
    return [q["newest"] for q in _recorder(hass).queries if "newest" in q]


@pytest.mark.asyncio
async def test_stops_at_the_newest_off_reading() -> None:
    """One read of the newest readings; older history is never read."""
    true_on = _NOW - timedelta(minutes=2)
    rows = [
        *_on_rows(_NOW - timedelta(hours=1), 50, timedelta(seconds=10)),
        _mock_state("0.5", _NOW - timedelta(minutes=3)),
        *_on_rows(_NOW, 3, timedelta(minutes=1)),
    ]
    hass = _make_hass(rows)

    assert await _run(hass, {}) == true_on.isoformat()
    assert _newest_queries(hass) == [1000]


@pytest.mark.asyncio
async def test_reads_further_back_when_the_newest_hold_no_off_reading() -> None:
    """No 'off' among the newest readings: read up to the cap and find it."""
    true_on = _NOW - timedelta(minutes=10)
    rows = [
        _mock_state("0.5", true_on - timedelta(minutes=1)),
        *_on_rows(_NOW, 11, timedelta(minutes=1)),
    ]
    hass = _make_hass(rows)

    with patch(f"{_PE}._RECENT_ROWS", 4), patch(f"{_PE}._MAX_ROWS", 50):
        assert await _run(hass, {}) == true_on.isoformat()
    assert _newest_queries(hass) == [4, 50]


@pytest.mark.asyncio
async def test_row_cap_gives_on_at_least_since_and_is_remembered() -> None:
    """
    Past the cap, last_changed is the oldest reading read, marked and kept.

    A sensor that reports every second and has been on for hours would
    otherwise have its raw last_changed (seconds old) read as a cycle that
    just ended.
    """
    rows = [
        _mock_state("0.5", _NOW - timedelta(hours=3)),
        *_on_rows(_NOW, 30, timedelta(seconds=20)),
    ]
    hass = _make_hass(rows)
    cache: dict[str, Any] = {}
    snapshot = _base_snapshot()
    snapshot["entities"] = [_power_entity(state="1500", last_changed=_NOW.isoformat())]

    with patch(f"{_PE}._RECENT_ROWS", 4), patch(f"{_PE}._MAX_ROWS", 10):
        await async_enrich_power_last_changed(hass, snapshot, cache)
        entity = snapshot["entities"][0]
        oldest_read = _NOW - timedelta(seconds=20 * 9)
        assert entity["last_changed"] == oldest_read.isoformat()
        assert entity.get("last_changed_is_lower_bound") is True
        assert cache["sensor.dishwasher_power"].unknown

        _recorder(hass).queries.clear()
        rows.append(_mock_state("1500", _advance(timedelta(minutes=1))))
        assert await _run(hass, cache) == oldest_read.isoformat()

    # Only readings since the newest one read last time.
    (query,) = _recorder(hass).queries
    assert query["start"] == _NOW - timedelta(microseconds=1)


@pytest.mark.asyncio
async def test_history_with_no_off_reading_is_a_lower_bound() -> None:
    """All the history there is shows the sensor on: its start was not seen."""
    rows = [
        _mock_state("20", _NOW - timedelta(minutes=5)),
        _mock_state("21", _NOW),
    ]
    hass = _make_hass(rows)
    snapshot = _base_snapshot()
    snapshot["entities"] = [_power_entity(state="21", last_changed=_NOW.isoformat())]

    await async_enrich_power_last_changed(hass, snapshot, {})

    entity = snapshot["entities"][0]
    assert entity["last_changed"] == (_NOW - timedelta(minutes=5)).isoformat()
    assert entity.get("last_changed_is_lower_bound") is True
    assert _newest_queries(hass) == [1000]  # all the history: no wider read


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
    rows.append(_mock_state("1490", _advance(timedelta(minutes=5))))

    assert await _run(hass, cache) == true_on.isoformat()
    (query,) = _recorder(hass).queries
    assert query["start"] == _NOW - timedelta(microseconds=1)
    assert query["descending"] is False


@pytest.mark.asyncio
async def test_a_reading_committed_late_is_read_on_the_next_run() -> None:
    """
    The next run reads from the newest reading read, not from when it ran.

    The recorder commits a sensor's readings in order, so a reading not yet
    committed at the last run is newer than every reading that run read,
    however late it was committed.
    """
    true_on = _NOW - timedelta(hours=2)
    rows = [
        _mock_state("0.5", true_on - timedelta(minutes=1)),
        _mock_state("1500", true_on),
        _mock_state("1500", _NOW - timedelta(minutes=20)),  # newest committed
    ]
    hass = _make_hass(rows)
    cache: dict[str, Any] = {}
    assert await _run(hass, cache) == true_on.isoformat()
    # A quiet run reads nothing new: the position stays at the newest reading.
    last_run = _advance(timedelta(minutes=5))
    assert await _run(hass, cache) == true_on.isoformat()

    # Stamped 10 minutes before that run, committed after it.
    restarted = last_run - timedelta(minutes=9)
    rows += [
        _mock_state("0.2", last_run - timedelta(minutes=10)),
        _mock_state("1500", restarted),
        _mock_state("1500", _advance(timedelta(minutes=5))),
    ]
    assert await _run(hass, cache) == restarted.isoformat()


@pytest.mark.asyncio
async def test_unknown_start_becomes_known_once_an_off_reading_is_read() -> None:
    """After the cap, the next off-then-on among new readings gives a real start."""
    rows = _on_rows(_NOW, 30, timedelta(seconds=20))
    hass = _make_hass(rows)
    cache: dict[str, Any] = {}
    with patch(f"{_PE}._RECENT_ROWS", 4), patch(f"{_PE}._MAX_ROWS", 10):
        await _run(hass, cache)
        assert cache["sensor.dishwasher_power"].unknown

        restarted = _NOW + timedelta(minutes=2)
        rows += [
            _mock_state("0.2", _NOW + timedelta(minutes=1)),
            _mock_state("1500", restarted),
            _mock_state("1500", _advance(timedelta(minutes=3))),
        ]
        assert await _run(hass, cache) == restarted.isoformat()
    assert not cache["sensor.dishwasher_power"].unknown


@pytest.mark.asyncio
async def test_too_many_new_readings_reads_the_newest_back_again() -> None:
    """More new readings than the cap: the remembered start may be stale."""
    true_on = _NOW - timedelta(hours=2)
    rows = [
        _mock_state("0.5", true_on - timedelta(minutes=1)),
        _mock_state("1500", true_on),
        _mock_state("1500", _NOW),
    ]
    hass = _make_hass(rows)
    cache: dict[str, Any] = {}
    with patch(f"{_PE}._RECENT_ROWS", 4), patch(f"{_PE}._MAX_ROWS", 10):
        assert await _run(hass, cache) == true_on.isoformat()
        _recorder(hass).queries.clear()
        rows += _on_rows(_advance(timedelta(minutes=10)), 30, timedelta(seconds=10))
        result = await _run(hass, cache)

    assert _newest_queries(hass) == [4, 10]
    entry = cache["sensor.dishwasher_power"]
    assert entry.unknown
    assert entry.since is not None
    assert entry.since > true_on
    assert result == entry.since.isoformat()


@pytest.mark.asyncio
async def test_unit_change_reads_the_newest_back_again() -> None:
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
    rows.append(_mock_state("1.5", _advance(timedelta(minutes=1)), unit="kW"))
    assert await _run(hass, cache, state="1.5", unit="kW") == true_on.isoformat()
    assert _newest_queries(hass) == [1000]


@pytest.mark.asyncio
async def test_memory_older_than_the_window_reads_the_newest_back_again() -> None:
    """After a long gap between runs, what was read has left the window."""
    true_on = _NOW - timedelta(hours=2)
    rows = [
        _mock_state("0.5", true_on - timedelta(minutes=1)),
        _mock_state("1500", true_on),
        _mock_state("1500", _NOW),
    ]
    hass = _make_hass(rows)
    cache: dict[str, Any] = {}
    assert await _run(hass, cache) == true_on.isoformat()

    restarted = _NOW + timedelta(days=2)
    rows += [
        _mock_state("0.2", restarted - timedelta(minutes=1)),
        _mock_state("1500", restarted),
        _mock_state("1500", _advance(timedelta(days=31))),
    ]
    _recorder(hass).queries.clear()
    assert await _run(hass, cache) == restarted.isoformat()
    assert _newest_queries(hass) == [1000]


@pytest.mark.asyncio
async def test_never_asks_the_recorder_for_newest_first_with_a_limit() -> None:
    """
    No query relies on descending order with a limit.

    Before Home Assistant 2026.11, state_changes_during_period applied the
    limit before the order, so it returned the oldest readings; the fake
    refuses such a query.
    """
    rows = [
        _mock_state("0.5", _NOW - timedelta(hours=3)),
        *_on_rows(_NOW, 30, timedelta(minutes=1)),
    ]
    hass = _make_hass(rows)
    cache: dict[str, Any] = {}
    await _run(hass, cache)
    rows.append(_mock_state("1500", _advance(timedelta(minutes=1))))
    await _run(hass, cache)
    assert _recorder(hass).refused == []
    assert any("start" in q for q in _recorder(hass).queries)


@pytest.mark.asyncio
async def test_a_reading_after_the_run_began_does_not_skip_the_wider_read() -> None:
    """
    Readings newer than the run's start are dropped after the count is taken.

    A full page that loses one of them is still a full page: the newest
    _MAX_ROWS are read, not the oldest of a short read taken as all history.
    """
    true_on = _NOW - timedelta(minutes=10)
    rows = [
        _mock_state("0.5", true_on - timedelta(minutes=1)),
        *_on_rows(_NOW, 11, timedelta(minutes=1)),
        _mock_state("1500", _NOW + timedelta(seconds=1)),
    ]
    hass = _make_hass(rows)
    with patch(f"{_PE}._RECENT_ROWS", 4), patch(f"{_PE}._MAX_ROWS", 50):
        assert await _run(hass, {}) == true_on.isoformat()
    assert _newest_queries(hass) == [4, 50]


@pytest.mark.asyncio
async def test_an_unusable_reading_before_the_window_proves_nothing() -> None:
    """Unavailable 40 days ago and on for 2 days is not on for 30 days."""
    two_days = _NOW - timedelta(days=2)
    rows = [
        _mock_state("unavailable", _NOW - timedelta(days=40)),
        _mock_state("1500", two_days),
        _mock_state("1500", _NOW),
    ]
    hass = _make_hass(rows)
    cache: dict[str, Any] = {}
    assert await _run(hass, cache) == two_days.isoformat()
    assert cache["sensor.dishwasher_power"].unknown
    assert _newest_queries(hass) == [1000]  # reached the window: no wider read


@pytest.mark.asyncio
async def test_an_off_reading_before_the_window_dates_the_start() -> None:
    """Off just before the window and on ever since: the start is found."""
    first_on = _NOW - timedelta(days=29)
    rows = [
        _mock_state("0.5", _NOW - timedelta(days=31)),
        _mock_state("1500", first_on),
        _mock_state("1500", _NOW),
    ]
    hass = _make_hass(rows)
    cache: dict[str, Any] = {}
    assert await _run(hass, cache) == first_on.isoformat()
    assert not cache["sensor.dishwasher_power"].unknown


@pytest.mark.asyncio
async def test_a_small_clock_step_back_reads_the_newest_back_again() -> None:
    """
    A clock set back, then caught up past the remembered position.

    Readings stamped in the repeated stretch sit below that position, so an
    off and on there would be missed; comparing with the monotonic clock
    shows the step and the remembered start is not used.
    """
    true_on = _NOW - timedelta(hours=2)
    rows = [
        _mock_state("0.5", true_on - timedelta(minutes=1)),
        _mock_state("1500", true_on),
        _mock_state("1500", _NOW),
    ]
    hass = _make_hass(rows)
    cache: dict[str, Any] = {}
    assert await _run(hass, cache) == true_on.isoformat()

    # 10 minutes pass, but the wall clock was stepped back 5 of them.
    _MONO[0] += 600
    _CLOCK[0] += timedelta(minutes=5)
    restarted = _NOW - timedelta(seconds=30)
    rows += [
        _mock_state("0.2", _NOW - timedelta(minutes=1)),
        _mock_state("1500", restarted),
        _mock_state("1500", _CLOCK[0]),
    ]
    _recorder(hass).queries.clear()
    assert await _run(hass, cache) == restarted.isoformat()
    assert _newest_queries(hass) == [1000]


@pytest.mark.asyncio
async def test_seen_off_then_off_and_on_again_is_dated_by_the_last_off() -> None:
    """No episode at the last run; new readings hold an off, then on."""
    rows = [
        _mock_state("1500", _NOW - timedelta(hours=1)),
        _mock_state("0.5", _NOW),
    ]
    hass = _make_hass(rows)
    cache: dict[str, Any] = {}
    await _run(hass, cache)
    assert cache["sensor.dishwasher_power"].since is None

    started = _NOW + timedelta(minutes=3)
    rows += [
        _mock_state("1500", _NOW + timedelta(minutes=1)),
        _mock_state("0.3", _NOW + timedelta(minutes=2)),
        _mock_state("1500", started),
        _mock_state("1500", _advance(timedelta(minutes=4))),
    ]
    assert await _run(hass, cache) == started.isoformat()


@pytest.mark.asyncio
async def test_an_unbounded_start_takes_the_first_reading_seen() -> None:
    """Unknown with nothing usable read, then on readings: they bound it."""
    rows = [
        _mock_state("unavailable", _NOW - timedelta(minutes=2)),
        _mock_state("unknown", _NOW),
    ]
    hass = _make_hass(rows)
    cache: dict[str, Any] = {}
    await _run(hass, cache)
    entry = cache["sensor.dishwasher_power"]
    assert entry.unknown
    assert entry.since is None

    first = _NOW + timedelta(minutes=1)
    rows += [
        _mock_state("1500", first),
        _mock_state("1500", _advance(timedelta(minutes=2))),
    ]
    assert await _run(hass, cache) == first.isoformat()
    assert cache["sensor.dishwasher_power"].unknown


@pytest.mark.asyncio
async def test_a_bounded_start_over_the_cap_reads_the_newest_back_again() -> None:
    """A lower bound, then more new readings than the cap: read the newest back."""
    rows = _on_rows(_NOW, 30, timedelta(seconds=20))
    hass = _make_hass(rows)
    cache: dict[str, Any] = {}
    with patch(f"{_PE}._RECENT_ROWS", 4), patch(f"{_PE}._MAX_ROWS", 10):
        await _run(hass, cache)
        assert cache["sensor.dishwasher_power"].unknown
        _recorder(hass).queries.clear()
        rows += _on_rows(_advance(timedelta(minutes=10)), 30, timedelta(seconds=10))
        await _run(hass, cache)
    assert _newest_queries(hass) == [4, 10]
    assert cache["sensor.dishwasher_power"].since > _NOW


@pytest.mark.asyncio
async def test_an_off_below_an_unusable_pre_window_reading_dates_the_start() -> None:
    """The newest usable reading before the window decides, not just the newest."""
    first_on = _NOW - timedelta(days=29)
    rows = [
        _mock_state("0.5", _NOW - timedelta(days=40)),
        _mock_state("unavailable", _NOW - timedelta(days=35)),
        _mock_state("1500", first_on),
        _mock_state("1500", _NOW),
    ]
    hass = _make_hass(rows)
    cache: dict[str, Any] = {}
    assert await _run(hass, cache) == first_on.isoformat()
    assert not cache["sensor.dishwasher_power"].unknown


@pytest.mark.asyncio
async def test_a_run_cut_short_keeps_what_it_did_not_reach() -> None:
    """A reload or shutdown mid-run must not forget the sensors after it."""
    true_on = _NOW - timedelta(hours=2)
    rows = [
        _mock_state("0.5", true_on - timedelta(minutes=1)),
        _mock_state("1500", true_on),
        _mock_state("1500", _NOW),
    ]
    hass = _make_hass(rows)
    cache: dict[str, Any] = {}
    snapshot = _base_snapshot()
    first = _power_entity(state="1500", last_changed=_NOW.isoformat())
    second = {**_power_entity(state="1500"), "entity_id": "sensor.dryer_power"}
    snapshot["entities"] = [first, second]
    await async_enrich_power_last_changed(hass, snapshot, cache)
    assert set(cache) == {"sensor.dishwasher_power", "sensor.dryer_power"}

    recorder = _recorder(hass)

    def _cut(job: Any) -> Any:
        if job.args[-1] == "sensor.dishwasher_power":
            raise asyncio.CancelledError
        return recorder(job)

    hass.data[DATA_INSTANCE].async_add_executor_job.side_effect = _cut
    _advance(timedelta(minutes=5))
    snapshot = _base_snapshot()
    snapshot["entities"] = [first, second]
    with pytest.raises(asyncio.CancelledError):
        await async_enrich_power_last_changed(hass, snapshot, cache)
    assert set(cache) == {"sensor.dishwasher_power", "sensor.dryer_power"}


@pytest.mark.asyncio
async def test_newest_readings_all_after_the_run_began_read_wider() -> None:
    """After a large clock step back the newest page can all be in the future."""
    true_on = _NOW - timedelta(hours=2)
    rows = [
        _mock_state("0.5", true_on - timedelta(minutes=1)),
        _mock_state("1500", true_on),
        _mock_state("1500", _NOW),
        *_on_rows(_NOW + timedelta(hours=1), 4, timedelta(minutes=1)),
    ]
    hass = _make_hass(rows)
    with patch(f"{_PE}._RECENT_ROWS", 4), patch(f"{_PE}._MAX_ROWS", 50):
        assert await _run(hass, {}) == true_on.isoformat()
    assert _newest_queries(hass) == [4, 50]


@pytest.mark.asyncio
async def test_after_a_clock_step_back_the_next_run_reads_from_before_it() -> None:
    """Readings stamped after the step may commit below the newest one read."""
    true_on = _NOW - timedelta(hours=2)
    rows = [
        _mock_state("0.5", true_on - timedelta(minutes=1)),
        _mock_state("1500", true_on),
        _mock_state("1500", _NOW),
    ]
    hass = _make_hass(rows)
    cache: dict[str, Any] = {}
    assert await _run(hass, cache) == true_on.isoformat()

    _MONO[0] += 600
    _CLOCK[0] += timedelta(minutes=5)  # 10 minutes passed, 5 of them repeated
    await _run(hass, cache)
    assert cache["sensor.dishwasher_power"].checked <= _NOW - timedelta(minutes=5)


@pytest.mark.asyncio
async def test_a_position_older_than_the_kept_history_reads_the_newest_back() -> None:
    """Unavailable for longer than the recorder keeps: an off may be purged."""
    true_on = _NOW - timedelta(hours=2)
    rows = [
        _mock_state("0.5", true_on - timedelta(minutes=1)),
        _mock_state("1500", true_on),
        _mock_state("1500", _NOW),
    ]
    hass = _make_hass(rows)
    cache: dict[str, Any] = {}
    assert await _run(hass, cache) == true_on.isoformat()

    later = _advance(timedelta(days=11))
    restarted = later - timedelta(days=1)
    rows[:] = [  # the recorder purged everything older than 10 days
        _mock_state("0.2", restarted - timedelta(minutes=1)),
        _mock_state("1500", restarted),
        _mock_state("1500", later),
    ]
    oldest = (later - timedelta(days=10)).timestamp()
    hass.data[DATA_INSTANCE].states_manager.oldest_ts = oldest
    _recorder(hass).queries.clear()
    assert await _run(hass, cache) == restarted.isoformat()
    assert _newest_queries(hass) == [1000]


@pytest.mark.asyncio
async def test_no_value_change_moves_the_position_up_to_the_slack() -> None:
    """
    A value that holds while attributes churn shows no new value changes.

    The position still moves up to _COMMIT_SLACK before now, so the next
    query does not step through every attribute-only update since then.
    """
    true_on = _NOW - timedelta(hours=2)
    rows = [
        _mock_state("0.5", true_on - timedelta(minutes=1)),
        _mock_state("1500", true_on),
        _mock_state("1500", _NOW - timedelta(hours=1)),
    ]
    hass = _make_hass(rows)
    cache: dict[str, Any] = {}
    assert await _run(hass, cache) == true_on.isoformat()

    later = _advance(timedelta(hours=3))
    assert await _run(hass, cache) == true_on.isoformat()
    assert cache["sensor.dishwasher_power"].checked == later - timedelta(minutes=10)


@pytest.mark.asyncio
async def test_enrichment_keeps_when_the_sensor_last_reported() -> None:
    """
    Moving last_changed back to the run start keeps when it last reported (#982).

    Staleness and data quality read last_reported, so an always-on sensor that
    reports every second does not read as a day old.
    """
    true_on = _NOW - timedelta(days=1)
    rows = [_mock_state("0.5", true_on - timedelta(minutes=1))]
    rows += _on_rows(_NOW, 2, timedelta(days=1))
    hass = _make_hass(rows)
    reported = MagicMock()
    reported.state = "1498.5"  # still the snapshot's reading
    reported.last_reported = _NOW - timedelta(seconds=1)
    hass.states.get.return_value = reported

    snapshot = _base_snapshot()
    snapshot["entities"] = [_power_entity(last_changed=_NOW.isoformat())]
    await async_enrich_power_last_changed(hass, snapshot, {})

    entity = snapshot["entities"][0]
    assert entity["last_changed"] == true_on.isoformat()
    assert entity["last_reported"] == (_NOW - timedelta(seconds=1)).isoformat()
    hass.states.get.assert_called_with("sensor.dishwasher_power")


@pytest.mark.asyncio
async def test_an_always_on_sensor_reporting_every_second_is_not_stale() -> None:
    """
    End to end (#982): enrichment output, validated, through a staleness rule.

    A UPS drawing power for two days and reporting every second: its
    last_changed moves back two days, and a 24-hour staleness rule must not
    fire while it keeps reporting.
    """
    from custom_components.home_generative_agent.sentinel.dynamic_rules import (  # noqa: PLC0415
        evaluate_dynamic_rule,
    )
    from custom_components.home_generative_agent.snapshot.schema import (  # noqa: PLC0415
        validate_snapshot,
    )

    rows = [_mock_state("0.5", _NOW - timedelta(days=2, minutes=1))]
    rows += _on_rows(_NOW, 3, timedelta(days=1))
    hass = _make_hass(rows)
    reported = MagicMock()
    reported.state = "1498.5"
    reported.last_reported = _NOW
    hass.states.get.return_value = reported

    snapshot = _base_snapshot()
    snapshot["entities"] = [_power_entity(last_changed=_NOW.isoformat())]
    await async_enrich_power_last_changed(hass, snapshot, {})
    snapshot = validate_snapshot(snapshot)
    entity_id = snapshot["entities"][0]["entity_id"]
    rule = {
        "rule_id": "ups_stale",
        "template_id": "entity_staleness",
        "params": {"entity_id": entity_id, "max_stale_hours": 24.0},
        "severity": "low",
        "confidence": 0.6,
        "is_sensitive": False,
        "suggested_actions": [],
    }

    assert (
        snapshot["entities"][0]["last_changed"]
        < (_NOW - timedelta(hours=24)).isoformat()
    )
    assert evaluate_dynamic_rule(snapshot, rule) == []


@pytest.mark.asyncio
async def test_a_sensor_that_changed_during_the_queries_keeps_the_snapshots_time() -> (
    None
):
    """
    A live state that no longer holds the snapshot's reading is not used (#982).

    A sensor that went unavailable while the history was read must not make
    the snapshot's old reading look fresh to auto-execute's data-quality gate.
    """
    true_on = _NOW - timedelta(days=1)
    rows = [_mock_state("0.5", true_on - timedelta(minutes=1))]
    rows += _on_rows(_NOW, 2, timedelta(days=1))
    hass = _make_hass(rows)
    live = MagicMock()
    live.state = "unavailable"
    live.last_reported = _NOW + timedelta(seconds=5)
    hass.states.get.return_value = live

    snapshot = _base_snapshot()
    entity = _power_entity(last_changed=_NOW.isoformat())
    entity["last_updated"] = (_NOW - timedelta(days=2)).isoformat()
    snapshot["entities"] = [entity]
    await async_enrich_power_last_changed(hass, snapshot, {})

    assert snapshot["entities"][0]["last_reported"] == entity["last_updated"]


@pytest.mark.asyncio
async def test_an_on_at_least_since_start_still_keeps_last_reported() -> None:
    """The lower-bound case (no off reading found) gets last_reported too."""
    hass = _make_hass(_on_rows(_NOW, 3, timedelta(days=1)))
    reported = MagicMock()
    reported.state = "1498.5"
    reported.last_reported = _NOW
    hass.states.get.return_value = reported

    snapshot = _base_snapshot()
    snapshot["entities"] = [_power_entity(last_changed=_NOW.isoformat())]
    await async_enrich_power_last_changed(hass, snapshot, {})

    entity = snapshot["entities"][0]
    assert entity.get("last_changed_is_lower_bound") is True
    assert entity["last_reported"] == _NOW.isoformat()

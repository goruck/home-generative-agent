"""Power sensor history enrichment for accurate appliance 'on since' timestamps."""

from __future__ import annotations

import functools
import logging
import math
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import TYPE_CHECKING, Any

from homeassistant.components.recorder import history as recorder_history
from homeassistant.helpers.recorder import DATA_INSTANCE
from homeassistant.helpers.recorder import get_instance as get_recorder_instance
from homeassistant.util import dt as dt_util

from .power_units import is_power_unit, watts_per_unit

if TYPE_CHECKING:
    from homeassistant.core import HomeAssistant

    from custom_components.home_generative_agent.snapshot.schema import (
        FullStateSnapshot,
    )

_LOGGER = logging.getLogger(__name__)

_LOOKBACK_DAYS = 30
# A sensor with nothing remembered is read from its newest readings back:
# first this many, then _MAX_ROWS if no 'off' reading is among them.
_RECENT_ROWS = 1000
# Most readings read for one sensor in one run: about 5.5 h of a sensor that
# reports every second.  Past it the start is not found, and the oldest
# reading read stands in as "on at least since".
_MAX_ROWS = 20_000
# Query bounds are exclusive; this makes a bound inclusive.
_TICK = timedelta(microseconds=1)
# A wall clock that moved this much less than the monotonic clock between two
# runs was set back: readings stamped in that stretch may sit below the
# remembered position, so it is not used.
_CLOCK_STEP = timedelta(minutes=1)
# With no value change since the remembered position, it moves up to this
# long before now: a value that has not changed cannot hide an off reading,
# and the next query would otherwise step through every attribute-only update
# since then.  A reading committed later than this is missed.
_COMMIT_SLACK = timedelta(minutes=10)
# Below this wattage the appliance is effectively off.  Used when walking
# recorder history to locate the last off→on transition.
_POWER_OFF_W = 10.0
# Connectivity-loss states — skip when searching for the power transition.
_TRANSIENT_STATES = frozenset({"unavailable", "unknown"})


def _parse_float(raw: str) -> float | None:
    try:
        parsed = float(raw)
    except (ValueError, TypeError):
        return None
    # nan compares False against the off threshold and would masquerade as an
    # 'on' reading (rewriting last_changed from garbage); inf likewise.
    return parsed if math.isfinite(parsed) else None


def _row_watts(state: Any, current_factor: float) -> float | None:
    """
    Return a recorder row's reading in watts, or None if unusable.

    The row's own unit_of_measurement is honored when present — a sensor whose
    unit was reconfigured mid-window (W → kW template edit) must not have its
    old readings interpreted in the new unit.  Rows without a unit attribute
    fall back to the sensor's current factor; rows in a non-power unit are
    skipped like transients.
    """
    val = _parse_float(state.state)
    if val is None:
        return None
    attrs = getattr(state, "attributes", None) or {}
    if "unit_of_measurement" in attrs:
        factor = watts_per_unit(attrs.get("unit_of_measurement"))
        if factor is None:
            return None
    else:
        factor = current_factor
    val_w = val * factor
    # Finite native value can overflow to inf in watts (1e300 TW); inf
    # compares False against the off level and would masquerade as 'on'.
    return val_w if math.isfinite(val_w) else None


def _at(state: Any) -> datetime:
    return dt_util.as_utc(state.last_changed)


@dataclass(slots=True, frozen=True)
class _Reading:
    """The parts of a recorder row the walk reads."""

    state: str
    last_changed: datetime
    attributes: dict[str, Any]

    @classmethod
    def of(cls, row: Any) -> _Reading:
        attrs = getattr(row, "attributes", None) or {}
        unit = (
            {"unit_of_measurement": attrs.get("unit_of_measurement")}
            if "unit_of_measurement" in attrs
            else {}
        )
        return cls(str(row.state), _at(row), unit)


@dataclass(slots=True)
class _Walk:
    """A newest-to-oldest walk through one sensor's readings."""

    # Oldest usable 'on' reading so far: the episode's start once an 'off'
    # reading is found below it.
    prev: _Reading | None = None

    def feed(self, state: _Reading, factor: float) -> bool:
        """Take the next older reading; True when it is an 'off' reading."""
        if state.state in _TRANSIENT_STATES:
            return False
        val_w = _row_watts(state, factor)
        if val_w is None:
            return False
        if val_w <= _POWER_OFF_W:
            return True
        self.prev = state
        return False

    def back(self, rows: list[_Reading], factor: float) -> bool:
        """Walk ``rows`` (oldest first) newest first; True at an 'off' reading."""
        return any(self.feed(state, factor) for state in reversed(rows))

    @property
    def since(self) -> datetime | None:
        return None if self.prev is None else self.prev.last_changed


@dataclass(slots=True)
class PowerOnSince:
    """
    What is known about one power sensor's current 'on' episode.

    ``since`` is when it started.  With ``unknown`` the start was not found
    and ``since`` is only "on at least since" (the oldest reading read), or
    None when nothing usable was read.  None without ``unknown`` means no
    episode was seen.  ``checked`` is the newest reading read: the recorder
    commits a sensor's readings in order, so any not yet committed are newer,
    and the next run reads from there.  Held by the engine between runs, not
    saved.
    """

    since: datetime | None
    unknown: bool
    checked: datetime
    factor: float
    # When the run that found this ran, by the wall and the monotonic clock.
    ran_at: datetime | None = None
    ran_mono: float = 0.0


async def _newest(
    hass: HomeAssistant, instance: Any, entity_id: str, count: int, now: datetime
) -> tuple[list[_Reading], bool]:
    """
    Return the sensor's newest readings up to ``now``, oldest first.

    Also says whether that is all the history there is: the recorder returned
    fewer than ``count``.  get_last_state_changes takes the newest ``count``
    inside the query on every supported Home Assistant version, with no time
    bound, so readings recorded after the run began are dropped here (left
    for the next run) after that count is taken.
    """
    states = await instance.async_add_executor_job(
        functools.partial(
            recorder_history.get_last_state_changes, hass, count, entity_id
        )
    )
    fetched = states.get(entity_id) or []
    rows = sorted(
        (r for r in map(_Reading.of, fetched) if r.last_changed <= now),
        key=lambda r: r.last_changed,
    )
    return rows, len(fetched) < count


async def _since_checked(
    hass: HomeAssistant, instance: Any, entity_id: str, checked: datetime, now: datetime
) -> list[_Reading] | None:
    """
    Return the readings after ``checked`` up to ``now``, oldest first.

    None when there are _MAX_ROWS or more: too many to tell what happened.
    Readings at ``checked`` itself are read again, which changes nothing.
    """
    states = await instance.async_add_executor_job(
        functools.partial(
            recorder_history.state_changes_during_period,
            hass,
            checked - _TICK,
            now + _TICK,
            entity_id,
            no_attributes=False,
            descending=False,
            limit=_MAX_ROWS,
            include_start_time_state=False,
        )
    )
    rows = [_Reading.of(row) for row in states.get(entity_id) or []]
    return None if len(rows) >= _MAX_ROWS else rows


async def _resolve_fresh(  # noqa: PLR0913
    hass: HomeAssistant,
    instance: Any,
    entity_id: str,
    factor: float,
    start_time: datetime,
    now: datetime,
) -> PowerOnSince | None:
    """
    Read the sensor's newest readings back to its last 'off' reading.

    The reading just newer than that 'off' is the start.  Without an 'off'
    reading in the window, the start is not found: the oldest reading read
    (or the window's start, when the sensor was already on then) is only
    "on at least since".
    """
    rows: list[_Reading] = []
    walk = _Walk()
    for count in (_RECENT_ROWS, _MAX_ROWS):
        rows, all_there_is = await _newest(hass, instance, entity_id, count, now)
        if len(rows) < 2:  # noqa: PLR2004
            if all_there_is:
                return None  # too little history to say anything
            continue  # the newest are after the run began (clock set back)
        walk = _Walk()
        in_window = [row for row in rows if row.last_changed > start_time]
        checked = rows[-1].last_changed
        if walk.back(in_window, factor):
            return PowerOnSince(
                walk.since, unknown=False, checked=checked, factor=factor
            )
        before = rows[: len(rows) - len(in_window)]
        for row in reversed(before):
            # The newest usable reading from before the window shows the
            # state at its start.
            if walk.feed(row, factor):
                return PowerOnSince(
                    walk.since, unknown=False, checked=checked, factor=factor
                )
            if walk.prev is row:
                return PowerOnSince(
                    start_time, unknown=True, checked=checked, factor=factor
                )
        if before or all_there_is:
            break  # no 'off' reading in all the history there is
    if len(rows) < 2:  # noqa: PLR2004
        return None
    return PowerOnSince(
        walk.since, unknown=True, checked=rows[-1].last_changed, factor=factor
    )


async def _resolve(  # noqa: PLR0913
    hass: HomeAssistant,
    instance: Any,
    entity_id: str,
    factor: float,
    start_time: datetime,
    now: datetime,
    cached: PowerOnSince | None,
    mono: float,
) -> PowerOnSince | None:
    """
    Return the sensor's current 'on' episode, or None to leave it alone.

    With a remembered episode, only the readings since it was last read are
    read.  Without one (a new sensor, a unit change, the clock set back, a
    remembered position older than the window, or too many new readings),
    the newest readings are read back.
    """
    stepped_back = cached is not None and _clock_step_back(cached, now, mono)
    if (
        cached is not None
        and cached.factor == factor
        and start_time <= cached.checked <= now
        and not stepped_back
        and _still_recorded(instance, cached.checked)
    ):
        rows = await _since_checked(hass, instance, entity_id, cached.checked, now)
        if rows is not None:
            checked = max(
                max((r.last_changed for r in rows), default=cached.checked),
                now - _COMMIT_SLACK,
            )
            walk = _Walk()
            if walk.back(rows, factor) or (
                # The episode began after the last run saw the sensor off.
                walk.prev is not None and not cached.unknown and cached.since is None
            ):
                return PowerOnSince(
                    walk.since, unknown=False, checked=checked, factor=factor
                )
            since = cached.since
            if since is None and cached.unknown:
                since = walk.since
            return PowerOnSince(
                since, unknown=cached.unknown, checked=checked, factor=factor
            )
    result = await _resolve_fresh(hass, instance, entity_id, factor, start_time, now)
    if result is not None and stepped_back and cached and cached.ran_at:
        # Readings stamped after the step may still be committed below the
        # newest one read: read from before the step again next run.
        result.checked = min(result.checked, cached.ran_at - stepped_back)
    return result


def _clock_step_back(cached: PowerOnSince, now: datetime, mono: float) -> timedelta:
    """How far the wall clock went back since that run; zero if not past _CLOCK_STEP."""
    if cached.ran_at is None:
        return timedelta(0)
    back = timedelta(seconds=mono - cached.ran_mono) - (now - cached.ran_at)
    return back if back > _CLOCK_STEP else timedelta(0)


def _still_recorded(instance: Any, checked: datetime) -> bool:
    """
    Whether the recorder still keeps readings from ``checked`` on.

    A sensor unavailable for longer than the recorder keeps history keeps its
    remembered position; an off reading since then may have been purged.
    """
    oldest = getattr(getattr(instance, "states_manager", None), "oldest_ts", None)
    if not isinstance(oldest, int | float):
        return True
    return checked.timestamp() >= oldest


async def async_enrich_power_last_changed(
    hass: HomeAssistant,
    snapshot: FullStateSnapshot,
    on_since: dict[str, PowerOnSince] | None = None,
) -> None:
    """
    Correct last_changed for power sensors reset by HA startup.

    When HA restarts, a power sensor re-reports its current wattage, creating a
    new last_changed at startup time.  This function queries the recorder to
    find when the sensor last crossed from off to on and corrects last_changed
    before rules evaluate.  The appliance duration rule does NOT consume this
    (it measures duration by direct observation); the enriched value informs
    advisory context only — dynamic rules, triage, and the baseline
    cycle-completion recency check.

    When the start cannot be found, last_changed becomes the oldest reading
    read (the appliance has been on at least that long) and the entity is
    marked ``last_changed_is_lower_bound``, so a long run never reads as one
    that just changed.

    ``on_since`` carries each sensor's result to the next run, so that run
    reads only newer readings.  A sensor that reads off or is gone is dropped
    from it; one that is briefly unavailable keeps its entry.  Without it
    every run reads from the newest reading back.
    """
    if on_since is None:
        on_since = {}
    if DATA_INSTANCE not in getattr(hass, "data", {}):
        return

    power_entities = [
        e
        for e in snapshot["entities"]
        if e["domain"] == "sensor"
        and (
            e["attributes"].get("device_class") == "power"
            or is_power_unit(e["attributes"].get("unit_of_measurement"))
        )
    ]
    keep: set[str] = set()
    if power_entities:
        await _enrich(hass, power_entities, on_since, keep)
    # Only after a whole run: one cut short keeps what it did not reach.
    for entity_id in [k for k in on_since if k not in keep]:
        del on_since[entity_id]


async def _enrich(
    hass: HomeAssistant,
    power_entities: list[Any],
    on_since: dict[str, PowerOnSince],
    keep: set[str],
) -> None:
    instance = get_recorder_instance(hass)
    now = dt_util.utcnow()  # read nothing recorded after this run began
    mono = time.monotonic()
    start_time = now - timedelta(days=_LOOKBACK_DAYS)

    for power_entity in power_entities:
        entity_id = power_entity["entity_id"]
        unit_factor = watts_per_unit(
            power_entity["attributes"].get("unit_of_measurement")
        )

        current_val = _parse_float(power_entity["state"])
        if current_val is None:
            # Unavailable, unknown or garbage for now: say nothing, but keep
            # what is remembered so a brief drop-out does not lose the start.
            keep.add(entity_id)
            continue
        if unit_factor is None:
            # A device_class:power sensor in a non-power unit (e.g. VA) whose
            # history cannot be compared against the watts off level.
            continue
        current_w = current_val * unit_factor
        if not math.isfinite(current_w) or current_w <= _POWER_OFF_W:
            continue  # off, or garbage that overflowed to inf in watts
        keep.add(entity_id)

        try:
            result = await _resolve(
                hass,
                instance,
                entity_id,
                unit_factor,
                start_time,
                now,
                on_since.get(entity_id),
                mono,
            )
        except Exception:  # noqa: BLE001
            _LOGGER.debug(
                "Could not query recorder history for power entity %s", entity_id
            )
            continue
        if result is None:
            on_since.pop(entity_id, None)
            continue
        result.ran_at, result.ran_mono = now, mono
        on_since[entity_id] = result
        if result.unknown:
            # Only "on at least since" (Home Assistant's own value is one too
            # when nothing was walked): the baseline completion check must not
            # read it as a recent change.
            power_entity["last_changed_is_lower_bound"] = True
        if result.since is None:
            continue

        true_last_changed = result.since.isoformat()
        if true_last_changed != power_entity["last_changed"]:
            _LOGGER.debug(
                "Power sensor %s: corrected last_changed %s → %s (%s; now %.1f W)",
                entity_id,
                power_entity["last_changed"],
                true_last_changed,
                "on at least since" if result.unknown else "true on-since",
                current_w,
            )
            power_entity["last_changed"] = true_last_changed

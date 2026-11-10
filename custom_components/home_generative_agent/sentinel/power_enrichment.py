"""Power sensor history enrichment for accurate appliance 'on since' timestamps."""

from __future__ import annotations

import functools
import logging
import math
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


async def _newest(
    hass: HomeAssistant, instance: Any, entity_id: str, count: int, now: datetime
) -> list[_Reading]:
    """
    Return the sensor's newest ``count`` readings up to ``now``, oldest first.

    get_last_state_changes orders by time, newest first, inside the query on
    every supported Home Assistant version.  Readings recorded after the run
    began are left for the next run.
    """
    states = await instance.async_add_executor_job(
        functools.partial(
            recorder_history.get_last_state_changes, hass, count, entity_id
        )
    )
    rows = [_Reading.of(row) for row in states.get(entity_id) or []]
    return [row for row in rows if row.last_changed <= now]


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
    for count in (_RECENT_ROWS, _MAX_ROWS):
        rows = await _newest(hass, instance, entity_id, count, now)
        if len(rows) < 2:  # noqa: PLR2004
            return None  # too little history to say anything
        walk = _Walk()
        in_window = [row for row in rows if row.last_changed > start_time]
        checked = rows[-1].last_changed
        if walk.back(in_window, factor):
            return PowerOnSince(
                walk.since, unknown=False, checked=checked, factor=factor
            )
        if len(in_window) < len(rows):
            # A reading from before the window shows the state at its start.
            if walk.feed(rows[len(rows) - len(in_window) - 1], factor):
                return PowerOnSince(
                    walk.since, unknown=False, checked=checked, factor=factor
                )
            return PowerOnSince(
                start_time, unknown=True, checked=checked, factor=factor
            )
        if len(rows) < count:
            break  # all the history there is, and no 'off' reading in it
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
) -> PowerOnSince | None:
    """
    Return the sensor's current 'on' episode, or None to leave it alone.

    With a remembered episode, only the readings since it was last read are
    read.  Without one (a new sensor, a unit change, the clock set back, a
    remembered position older than the window, or too many new readings),
    the newest readings are read back.
    """
    if (
        cached is not None
        and cached.factor == factor
        and start_time <= cached.checked <= now
    ):
        rows = await _since_checked(hass, instance, entity_id, cached.checked, now)
        if rows is not None:
            checked = rows[-1].last_changed if rows else cached.checked
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
    return await _resolve_fresh(hass, instance, entity_id, factor, start_time, now)


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
    try:
        if power_entities:
            await _enrich(hass, power_entities, on_since, keep)
    finally:
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
            )
        except Exception:  # noqa: BLE001
            _LOGGER.debug(
                "Could not query recorder history for power entity %s", entity_id
            )
            continue
        if result is None:
            on_since.pop(entity_id, None)
            continue
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

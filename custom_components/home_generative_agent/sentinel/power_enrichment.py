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
# Readings per recorder query, newest first.  The walk usually stops in the
# first page, at the reading where the appliance was last off.
_PAGE_ROWS = 1000
# Most readings read for one sensor in one run: about 5.5 h of a sensor that
# reports every second.  Past it the start is left unknown, as for a sensor on
# since before the window.
_MAX_ROWS = 20_000
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


@dataclass(slots=True)
class _Walk:
    """A newest-to-oldest walk through one sensor's readings."""

    # Oldest usable 'on' reading so far: the episode's start once an 'off'
    # reading is found below it.
    prev: Any | None = None
    rows: int = 0

    def feed(self, state: Any, factor: float) -> bool:
        """Take the next older reading; True when it is an 'off' reading."""
        self.rows += 1
        if state.state in _TRANSIENT_STATES:
            return False
        val_w = _row_watts(state, factor)
        if val_w is None:
            return False
        if val_w <= _POWER_OFF_W:
            return True
        self.prev = state
        return False


# How a walk ended: at an 'off' reading, at the start of what it was asked to
# read, or at _MAX_ROWS.
_OFF = "off"
_START = "start"
_CAP = "cap"


@dataclass(slots=True)
class PowerOnSince:
    """
    What is known about one power sensor's current 'on' episode.

    ``since`` is when it started; None with ``unknown`` means it started
    before anything that was read, and None without it means no episode was
    seen.  ``checked`` is the newest reading read, so the next run reads only
    newer ones.  Held by the engine between runs and not saved.
    """

    since: datetime | None
    unknown: bool
    checked: datetime | None
    factor: float


async def _walk_back(  # noqa: PLR0913
    hass: HomeAssistant,
    instance: Any,
    entity_id: str,
    factor: float,
    start: datetime,
    *,
    include_start: bool,
) -> tuple[_Walk, str, datetime | None]:
    """
    Walk readings newer than ``start``, newest first, a page at a time.

    Returns the walk, how it ended, and the newest reading's time.  Pages
    are cut by an exclusive end at the oldest reading so far; two readings of
    one sensor at the same instant across a page boundary would lose one.
    """
    walk = _Walk()
    newest: datetime | None = None
    end: datetime | None = None  # first page: no upper bound
    while walk.rows < _MAX_ROWS:
        limit = min(_PAGE_ROWS, _MAX_ROWS - walk.rows)
        states = await instance.async_add_executor_job(
            functools.partial(
                recorder_history.state_changes_during_period,
                hass,
                start,
                end,
                entity_id,
                no_attributes=False,
                descending=True,
                limit=limit,
                include_start_time_state=include_start,
            )
        )
        rows = states.get(entity_id) or []
        # Newest first; the state at ``start``, when asked for, comes last.
        page = [s for s in rows if dt_util.as_utc(s.last_changed) > start]
        if page and newest is None:
            newest = dt_util.as_utc(page[0].last_changed)
        for state in page:
            if walk.feed(state, factor):
                return walk, _OFF, newest
        if len(page) < limit:
            for state in rows[len(page) :]:
                if walk.feed(state, factor):
                    return walk, _OFF, newest
            return walk, _START, newest
        end = dt_util.as_utc(page[-1].last_changed)
    return walk, _CAP, newest


def _since(walk: _Walk) -> datetime | None:
    return None if walk.prev is None else dt_util.as_utc(walk.prev.last_changed)


async def _resolve_newer(
    hass: HomeAssistant,
    instance: Any,
    entity_id: str,
    start_time: datetime,
    cached: PowerOnSince,
) -> PowerOnSince:
    """Update a remembered episode from the readings newer than it."""
    factor = cached.factor
    walk, how, newest = await _walk_back(
        hass,
        instance,
        entity_id,
        factor,
        max(cached.checked or start_time, start_time),
        include_start=False,
    )
    checked = newest or cached.checked
    if how == _CAP:
        return PowerOnSince(None, unknown=True, checked=checked, factor=factor)
    if how == _OFF or (
        # The episode began after the last run saw the sensor off.
        walk.prev is not None and cached.since is None and not cached.unknown
    ):
        return PowerOnSince(_since(walk), unknown=False, checked=checked, factor=factor)
    return PowerOnSince(
        cached.since, unknown=cached.unknown, checked=checked, factor=factor
    )


async def _resolve(  # noqa: PLR0913
    hass: HomeAssistant,
    instance: Any,
    entity_id: str,
    factor: float,
    start_time: datetime,
    cached: PowerOnSince | None,
) -> PowerOnSince | None:
    """
    Return the sensor's current 'on' episode, or None to leave it alone.

    The first 'off' reading below the newest readings ends the walk; the
    reading just newer than it is the start.  With no 'off' reading in the
    window, the oldest reading in it is the best estimate, unless the sensor
    was already on at the window's start.
    """
    if cached is not None and cached.factor == factor:
        return await _resolve_newer(hass, instance, entity_id, start_time, cached)

    walk, how, newest = await _walk_back(
        hass, instance, entity_id, factor, start_time, include_start=True
    )
    if how == _START and walk.rows < 2:  # noqa: PLR2004
        return None  # too little history to say anything
    if how == _CAP:
        return PowerOnSince(None, unknown=True, checked=newest, factor=factor)
    since = _since(walk)
    if how == _START and since is not None and since <= start_time:
        # On at the window's start: started before anything here.
        return PowerOnSince(None, unknown=True, checked=newest, factor=factor)
    return PowerOnSince(since, unknown=False, checked=newest, factor=factor)


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

    ``on_since`` carries each sensor's result to the next run, so that run
    reads only newer readings.  A sensor that is off or gone is dropped from
    it.  Without it every run walks from the newest reading back.
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
    on_ids: set[str] = set()
    try:
        if power_entities:
            await _enrich(hass, power_entities, on_since, on_ids)
    finally:
        for entity_id in [k for k in on_since if k not in on_ids]:
            del on_since[entity_id]


async def _enrich(
    hass: HomeAssistant,
    power_entities: list[Any],
    on_since: dict[str, PowerOnSince],
    on_ids: set[str],
) -> None:
    instance = get_recorder_instance(hass)
    start_time = dt_util.utcnow() - timedelta(days=_LOOKBACK_DAYS)

    for power_entity in power_entities:
        entity_id = power_entity["entity_id"]
        unit_factor = watts_per_unit(
            power_entity["attributes"].get("unit_of_measurement")
        )

        current_val = _parse_float(power_entity["state"])
        if current_val is None or unit_factor is None:
            # Non-numeric reading, or a device_class:power sensor in a
            # non-power unit (e.g. VA) whose history cannot be compared
            # against the watts off level.
            continue
        current_w = current_val * unit_factor
        if not math.isfinite(current_w) or current_w <= _POWER_OFF_W:
            continue  # off, or garbage that overflowed to inf in watts
        on_ids.add(entity_id)

        try:
            result = await _resolve(
                hass,
                instance,
                entity_id,
                unit_factor,
                start_time,
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
        if result.since is None:
            continue

        true_last_changed = result.since.isoformat()
        if true_last_changed != power_entity["last_changed"]:
            _LOGGER.debug(
                "Power sensor %s: corrected last_changed %s → %s "
                "(true on-since; now %.1f W)",
                entity_id,
                power_entity["last_changed"],
                true_last_changed,
                current_w,
            )
            power_entity["last_changed"] = true_last_changed

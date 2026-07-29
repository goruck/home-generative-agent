"""Power sensor history enrichment for accurate appliance 'on since' timestamps."""

from __future__ import annotations

import functools
import logging
import math
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


def _find_true_on_since(
    state_list: list[Any],
    current_factor: float,
    start_time: datetime,
) -> Any | None:
    """
    Return the state when the sensor last transitioned from off to on.

    Walk newest-to-oldest, skip transients.  The first reading at or below
    the off level (compared in watts, honoring each row's own unit) marks the
    boundary of the current 'on' episode; the state immediately after it
    (newer) is the true start.

    If no 'off' reading exists in the window, the oldest within-window record
    is returned as the best available approximation (the sensor has been on
    since before the window, but it is still more accurate than a restart-reset
    last_changed).  Returns None when nothing useful can be determined.
    """
    prev = None
    for state in reversed(state_list):
        if state.state in _TRANSIENT_STATES:
            continue
        val_w = _row_watts(state, current_factor)
        if val_w is None:
            continue
        if val_w <= _POWER_OFF_W:
            # 'Off' reading found — prev is the true on-start (may be None if
            # the first non-transient record is already below threshold).
            return prev
        prev = state

    # No 'off' reading in window.  Use oldest within-window 'on' record.
    first = next(
        (
            s
            for s in state_list
            if s.state not in _TRANSIENT_STATES
            and _row_watts(s, current_factor) is not None
        ),
        None,
    )
    if first is not None and dt_util.as_utc(first.last_changed) > start_time:
        return first
    return None


async def async_enrich_power_last_changed(
    hass: HomeAssistant, snapshot: FullStateSnapshot
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
    """
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
    if not power_entities:
        return

    instance = get_recorder_instance(hass)
    now = dt_util.utcnow()
    start_time = now - timedelta(days=_LOOKBACK_DAYS)

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

        try:
            states = await instance.async_add_executor_job(
                functools.partial(
                    recorder_history.state_changes_during_period,
                    hass,
                    start_time,
                    now,
                    entity_id,
                    no_attributes=False,
                    descending=False,
                    limit=None,
                    include_start_time_state=True,
                )
            )
        except Exception:  # noqa: BLE001
            _LOGGER.debug(
                "Could not query recorder history for power entity %s", entity_id
            )
            continue

        state_list = states.get(entity_id) or []

        if len(state_list) < 2:  # noqa: PLR2004
            continue

        true_on_state = _find_true_on_since(state_list, unit_factor, start_time)
        if true_on_state is None:
            continue

        true_last_changed = dt_util.as_utc(true_on_state.last_changed).isoformat()
        if true_last_changed != power_entity["last_changed"]:
            _LOGGER.debug(
                "Power sensor %s: corrected last_changed %s → %s "
                "(true on-since; %s at %.1f %s)",
                entity_id,
                power_entity["last_changed"],
                true_last_changed,
                true_on_state.state,
                current_w,
                "W",
            )
            power_entity["last_changed"] = true_last_changed

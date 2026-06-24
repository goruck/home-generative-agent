"""Power sensor history enrichment for accurate appliance 'on since' timestamps."""

from __future__ import annotations

import functools
import logging
from datetime import datetime, timedelta
from typing import TYPE_CHECKING, Any

from homeassistant.components.recorder import history as recorder_history
from homeassistant.helpers.recorder import DATA_INSTANCE
from homeassistant.helpers.recorder import get_instance as get_recorder_instance
from homeassistant.util import dt as dt_util

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
        return float(raw)
    except (ValueError, TypeError):
        return None


def _find_true_on_since(
    state_list: list[Any],
    off_threshold_native: float,
    start_time: datetime,
) -> Any | None:
    """
    Return the state when the sensor last transitioned from off to on.

    Walk newest-to-oldest, skip transients.  The first reading at or below
    off_threshold_native marks the boundary of the current 'on' episode; the
    state immediately after it (newer) is the true start.

    If no 'off' reading exists in the window, the oldest within-window record
    is returned as the best available approximation (the sensor has been on
    since before the window, but it is still more accurate than a restart-reset
    last_changed).  Returns None when nothing useful can be determined.
    """
    prev = None
    for state in reversed(state_list):
        if state.state in _TRANSIENT_STATES:
            continue
        val = _parse_float(state.state)
        if val is None:
            continue
        if val <= off_threshold_native:
            # 'Off' reading found — prev is the true on-start (may be None if
            # the first non-transient record is already below threshold).
            return prev
        prev = state

    # No 'off' reading in window.  Use oldest within-window 'on' record.
    first = next(
        (
            s
            for s in state_list
            if s.state not in _TRANSIENT_STATES and _parse_float(s.state) is not None
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
    new last_changed at startup time.  The appliance duration rule then computes
    a falsely short duration and fires too late (or not at all).  This function
    queries the recorder to find when the sensor last crossed from off to on and
    corrects last_changed before rules evaluate.
    """
    if DATA_INSTANCE not in getattr(hass, "data", {}):
        return

    power_entities = [
        e
        for e in snapshot["entities"]
        if e["domain"] == "sensor"
        and (
            e["attributes"].get("device_class") == "power"
            or e["attributes"].get("unit_of_measurement") in {"W", "kW"}
        )
    ]
    if not power_entities:
        return

    instance = get_recorder_instance(hass)
    now = dt_util.utcnow()
    start_time = now - timedelta(days=_LOOKBACK_DAYS)

    for power_entity in power_entities:
        entity_id = power_entity["entity_id"]
        unit = str(power_entity["attributes"].get("unit_of_measurement") or "W")

        current_val = _parse_float(power_entity["state"])
        if current_val is None:
            continue
        current_w = current_val * 1000.0 if unit == "kW" else current_val
        if current_w <= _POWER_OFF_W:
            continue  # appliance is off — nothing to correct

        off_threshold_native = _POWER_OFF_W / (1000.0 if unit == "kW" else 1.0)

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

        true_on_state = _find_true_on_since(
            state_list, off_threshold_native, start_time
        )
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

"""Alarm entity history enrichment for accurate disarm timestamps."""

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

_DISARMED = "disarmed"
# How far back to search for an armed→disarmed transition.  Using a date-range
# query (not count-based) so HA includes a synthetic start-of-period state via
# include_start_time_state=True, which acts as an "armed" anchor even when the
# actual record has been purged from the short-term DB.
_LOOKBACK_DAYS = 30
# HA sets these transient states when an integration loses/lacks connectivity.
# They are not intentional alarm state changes and must be ignored when
# searching for the armed→disarmed transition.
_TRANSIENT_STATES = frozenset({"unavailable", "unknown"})


def _find_true_disarm(
    state_list: list[Any], start_time: datetime
) -> tuple[Any | None, bool]:
    """
    Return (true_disarm_state, should_clear) from ascending state history.

    Walk newest-to-oldest, skip transient connectivity states, look for the
    armed→disarmed transition.  If none is found, fall back to using the oldest
    surviving record when it is genuinely within the window (i.e. last_changed >
    start_time, so it is not a synthetic start-state injected by HA).  If even
    that fails, return (None, True) to signal that last_changed should be cleared.
    """
    prev = None
    for state in reversed(state_list):
        if state.state in _TRANSIENT_STATES:
            continue
        if prev is not None and state.state != _DISARMED and prev.state == _DISARMED:
            return prev, False
        prev = state

    # No transition found.
    # If the oldest non-transient record is within the window its "armed"
    # predecessor was purged from the short-term DB but the disarm itself
    # survived — use it as the best available approximation.
    # If it is at or before start_time, include_start_time_state injected it as
    # a synthetic anchor, meaning the alarm was disarmed before the window.
    first = next((s for s in state_list if s.state not in _TRANSIENT_STATES), None)
    if (
        first is not None
        and first.state == _DISARMED
        and dt_util.as_utc(first.last_changed) > start_time
    ):
        return first, False
    return None, True


async def async_enrich_alarm_last_changed(
    hass: HomeAssistant, snapshot: FullStateSnapshot
) -> None:
    """Correct last_changed for disarmed alarm entities reset by HA startup."""
    if DATA_INSTANCE not in getattr(hass, "data", {}):
        return

    disarmed_alarms = [
        e
        for e in snapshot["entities"]
        if e["domain"] == "alarm_control_panel" and e["state"] == _DISARMED
    ]
    if not disarmed_alarms:
        return

    instance = get_recorder_instance(hass)
    now = dt_util.utcnow()
    start_time = now - timedelta(days=_LOOKBACK_DAYS)

    for alarm_entity in disarmed_alarms:
        entity_id = alarm_entity["entity_id"]
        try:
            # Use a date-range query with include_start_time_state=True so that
            # HA inserts a synthetic record showing the state at the start of
            # the window.  This acts as an "armed" anchor even when the actual
            # transition record was purged by the recorder's short-term DB.
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
                "Could not query recorder history for alarm entity %s", entity_id
            )
            continue

        state_list = states.get(entity_id) or []

        _LOGGER.debug(
            "Alarm %s history (%d records, %d-day window): %s",
            entity_id,
            len(state_list),
            _LOOKBACK_DAYS,
            [
                (s.state, dt_util.as_local(s.last_changed).isoformat())
                for s in state_list
            ],
        )

        if len(state_list) < 2:  # noqa: PLR2004
            continue

        true_disarm_state, should_clear = _find_true_disarm(state_list, start_time)

        if should_clear:
            _LOGGER.debug(
                "Alarm %s: no armed→disarmed transition in %d-day window; "
                "clearing last_changed to suppress misleading HA restart time",
                entity_id,
                _LOOKBACK_DAYS,
            )
            alarm_entity["last_changed"] = ""
            continue

        if true_disarm_state is None:
            continue

        true_last_changed = dt_util.as_utc(true_disarm_state.last_changed).isoformat()
        if true_last_changed != alarm_entity["last_changed"]:
            _LOGGER.debug(
                "Alarm %s: corrected last_changed %s → %s",
                entity_id,
                alarm_entity["last_changed"],
                true_last_changed,
            )
            alarm_entity["last_changed"] = true_last_changed
        else:
            _LOGGER.debug(
                "Alarm %s: last_changed %s already correct, no update needed",
                entity_id,
                alarm_entity["last_changed"],
            )

"""Lock entity history enrichment for accurate unlock timestamps."""

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

_UNLOCKED = "unlocked"
_LOOKBACK_DAYS = 30
# Connectivity-loss states injected by HA during startup/reconnect — not
# intentional lock operations; skip when searching for the transition.
_TRANSIENT_STATES = frozenset({"unavailable", "unknown"})


def _find_true_unlock(
    state_list: list[Any], start_time: datetime
) -> tuple[Any | None, bool]:
    """
    Return (true_unlock_state, should_clear) from ascending state history.

    Walk newest-to-oldest, skip transient connectivity states, look for the
    locked→unlocked transition.  If none is found, fall back to using the oldest
    surviving record when it is genuinely within the window (last_changed >
    start_time), meaning the preceding locked record was purged from the DB.
    If even that fails, return (None, True) to clear last_changed.
    """
    prev = None
    for state in reversed(state_list):
        if state.state in _TRANSIENT_STATES:
            continue
        if prev is not None and state.state != _UNLOCKED and prev.state == _UNLOCKED:
            return prev, False
        prev = state

    # No transition found.
    first = next((s for s in state_list if s.state not in _TRANSIENT_STATES), None)
    if (
        first is not None
        and first.state == _UNLOCKED
        and dt_util.as_utc(first.last_changed) > start_time
    ):
        return first, False
    return None, True


async def async_enrich_lock_last_changed(
    hass: HomeAssistant, snapshot: FullStateSnapshot
) -> None:
    """Correct last_changed for unlocked lock entities reset by HA startup."""
    if DATA_INSTANCE not in getattr(hass, "data", {}):
        return

    unlocked_locks = [
        e
        for e in snapshot["entities"]
        if e["domain"] == "lock" and e["state"] == _UNLOCKED
    ]
    if not unlocked_locks:
        return

    instance = get_recorder_instance(hass)
    now = dt_util.utcnow()
    start_time = now - timedelta(days=_LOOKBACK_DAYS)

    for lock_entity in unlocked_locks:
        entity_id = lock_entity["entity_id"]
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
                "Could not query recorder history for lock entity %s", entity_id
            )
            continue

        state_list = states.get(entity_id) or []

        _LOGGER.debug(
            "Lock %s history (%d records, %d-day window): %s",
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

        true_unlock_state, should_clear = _find_true_unlock(state_list, start_time)

        if should_clear:
            _LOGGER.debug(
                "Lock %s: no locked→unlocked transition in %d-day window; "
                "clearing last_changed to suppress misleading HA restart time",
                entity_id,
                _LOOKBACK_DAYS,
            )
            lock_entity["last_changed"] = ""
            continue

        if true_unlock_state is None:
            continue

        true_last_changed = dt_util.as_utc(true_unlock_state.last_changed).isoformat()
        if true_last_changed != lock_entity["last_changed"]:
            _LOGGER.debug(
                "Lock %s: corrected last_changed %s → %s",
                entity_id,
                lock_entity["last_changed"],
                true_last_changed,
            )
            lock_entity["last_changed"] = true_last_changed
        else:
            _LOGGER.debug(
                "Lock %s: last_changed %s already correct, no update needed",
                entity_id,
                lock_entity["last_changed"],
            )

"""Derived snapshot context helpers."""

from __future__ import annotations

from typing import TYPE_CHECKING

from homeassistant.util import dt as dt_util

if TYPE_CHECKING:
    from collections.abc import Iterable
    from datetime import datetime

    from homeassistant.core import State

    from .schema import DerivedContext

_MOTION_DEVICE_CLASSES = {"motion", "occupancy"}


def _as_iso(value: datetime) -> str:
    return dt_util.as_utc(value).isoformat()


def _iter_motion_entities(states: Iterable[State]) -> Iterable[State]:
    for state in states:
        if state.domain != "binary_sensor":
            continue
        device_class = state.attributes.get("device_class")
        if device_class in _MOTION_DEVICE_CLASSES:
            yield state


def derive_context(
    now: datetime,
    timezone: str,
    sun_state: State | None,
    all_states: list[State],
    area_lookup: dict[str, str | None],
) -> DerivedContext:
    """Build derived context for the snapshot."""
    is_night = False
    if sun_state is not None:
        is_night = sun_state.state == "below_horizon"

    anyone_home = any(
        state.domain == "person" and state.state == "home" for state in all_states
    )

    last_motion_by_area: dict[str, str] = {}
    for state in _iter_motion_entities(all_states):
        area_name = area_lookup.get(state.entity_id)
        if not area_name:
            continue
        existing = last_motion_by_area.get(area_name)
        changed = _as_iso(state.last_changed)
        if existing is None or changed > existing:
            last_motion_by_area[area_name] = changed

    return {
        "now": _as_iso(now),
        "timezone": timezone,
        "is_night": is_night,
        "anyone_home": anyone_home,
        "last_motion_by_area": dict(sorted(last_motion_by_area.items())),
    }

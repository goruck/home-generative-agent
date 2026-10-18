"""
Read Sentinel's autonomy settings the same way everywhere (#783).

The options form, the Basic setup carry and the engine all read the autonomy
level and the override duration, from the Sentinel subentry or from the entry
options an older install kept them in. Read differently, an unchanged form
save could change a value the engine was running with (a legacy level ``0.0``
shown as 1), and a duration the form refuses could reach the engine (0, which
ends every override at once, or one too large for a ``timedelta``).
"""

from __future__ import annotations

from ..const import (  # noqa: TID252
    RECOMMENDED_SENTINEL_AUTONOMY_LEVEL,
    RECOMMENDED_SENTINEL_RUNTIME_OVERRIDE_TTL_MINUTES,
    SENTINEL_AUTONOMY_LEVELS,
)

# An override can last up to 30 days.
MAX_OVERRIDE_TTL_MINUTES = 30 * 24 * 60


def _whole_number(value: object) -> int | None:
    """Return a whole number given as an int, a float or text, else None."""
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value) if value.is_integer() else None
    if isinstance(value, str):
        try:
            number = float(value.strip())
        except ValueError:
            return None
        return int(number) if number.is_integer() else None
    return None


def autonomy_level(value: object) -> int:
    """Return a stored autonomy level, or the default for one unreadable."""
    level = _whole_number(value)
    return (
        level
        if level is not None and level in SENTINEL_AUTONOMY_LEVELS
        else RECOMMENDED_SENTINEL_AUTONOMY_LEVEL
    )


def override_minutes(value: object) -> int:
    """Return a stored override duration: 1 minute to 30 days, else the default."""
    minutes = _whole_number(value)
    if minutes is None or minutes < 1:
        return RECOMMENDED_SENTINEL_RUNTIME_OVERRIDE_TTL_MINUTES
    return min(minutes, MAX_OVERRIDE_TTL_MINUTES)

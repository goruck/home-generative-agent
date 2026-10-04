# ruff: noqa: S101
"""get_entity_history reads its times as the home's local wall clock."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING
from zoneinfo import ZoneInfo

import pytest
from homeassistant.util import dt as dt_util

from custom_components.home_generative_agent.agent.tools import _as_utc

if TYPE_CHECKING:
    from collections.abc import Iterator

_DEFAULT = datetime(2000, 1, 1, tzinfo=UTC)


@pytest.fixture(autouse=True)
def _los_angeles() -> Iterator[None]:
    previous = dt_util.get_default_time_zone()
    dt_util.set_default_time_zone(ZoneInfo("America/Los_Angeles"))
    yield
    dt_util.set_default_time_zone(previous)


@pytest.mark.parametrize(
    "value",
    [
        # The field case: local wall clock labelled as UTC.
        "2026-10-04T00:00:00+00:00",
        # The correct offset, and no offset at all.
        "2026-10-04T00:00:00-07:00",
        "2026-10-04T00:00:00",
        # A wrong offset in the strftime "%z" form.
        "2026-10-04T00:00:00+0530",
    ],
)
def test_local_midnight_is_read_as_local_whatever_the_offset(value: str) -> None:
    """Local midnight in October is 07:00 UTC, so the morning stays in range."""
    assert _as_utc(value, _DEFAULT, "bad") == datetime(2026, 10, 4, 7, tzinfo=UTC)


def test_winter_wall_clock_uses_standard_time() -> None:
    """A guessed summer offset must not shift a January window by an hour."""
    assert _as_utc("2026-01-15T00:00:00-07:00", _DEFAULT, "bad") == datetime(
        2026, 1, 15, 8, tzinfo=UTC
    )


def test_field_case_window_covers_the_morning_departure() -> None:
    """The live call's window (00:00 to 13:57) must include a 07:00 departure."""
    start = _as_utc("2026-10-04T00:00:00+00:00", _DEFAULT, "bad")
    end = _as_utc("2026-10-04T13:57:32+00:00", _DEFAULT, "bad")
    departure = datetime(2026, 10, 4, 7, 0, 21, tzinfo=ZoneInfo("America/Los_Angeles"))
    assert start <= departure <= end
    assert end - start == timedelta(hours=13, minutes=57, seconds=32)


def test_missing_value_uses_the_default() -> None:
    assert _as_utc(None, _DEFAULT, "bad") == _DEFAULT  # type: ignore[arg-type]

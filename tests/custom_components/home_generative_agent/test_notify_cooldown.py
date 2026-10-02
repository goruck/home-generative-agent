# ruff: noqa: S101
"""Tests for the pure notification cooldown policy (issue #672)."""

from __future__ import annotations

import pytest

from custom_components.home_generative_agent.core.notify_cooldown import (
    CooldownAction,
    CooldownWindow,
    HumanLevel,
    active_window,
    decide,
    human_level,
    notification_tag,
)

_NONE = HumanLevel.NONE
_KNOWN = HumanLevel.KNOWN
_UNIDENTIFIED = HumanLevel.UNIDENTIFIED


def _window(
    level: HumanLevel = _NONE, *, started: float = 100.0, last_capture: float = 1000.0
) -> CooldownWindow:
    return CooldownWindow(
        started=started, window_id=1_790_000_000, level=level, last_capture=last_capture
    )


@pytest.mark.parametrize(
    ("names", "caption_has_human", "window_level", "expected"),
    [
        # An explicit unknown face is unidentified, whatever else is present.
        (["Unknown Person"], False, None, _UNIDENTIFIED),
        (["Lindo", "Unknown Person"], True, _KNOWN, _UNIDENTIFIED),
        # Pre-guard gallery rows may carry label variants.
        ([" unknown person "], False, _KNOWN, _UNIDENTIFIED),
        # Enrolled names only are known, with or without a caption human.
        (["Lindo"], False, None, _KNOWN),
        (["Lindo", "Indeterminate"], True, None, _KNOWN),
        # A caption human with no face result is unidentified...
        ([], True, None, _UNIDENTIFIED),
        (["Indeterminate"], True, _NONE, _UNIDENTIFIED),
        ([], True, _UNIDENTIFIED, _UNIDENTIFIED),
        # ...except inside a KNOWN window: the resident turned away.
        ([], True, _KNOWN, _KNOWN),
        (["Indeterminate"], True, _KNOWN, _KNOWN),
        # No human evidence at all, including legacy placeholders.
        ([], False, None, _NONE),
        (["Indeterminate", "None", ""], False, _KNOWN, _NONE),
    ],
)
def test_human_level(
    names: list[str],
    caption_has_human: bool,  # noqa: FBT001
    window_level: HumanLevel | None,
    expected: HumanLevel,
) -> None:
    """Each row of the batch human-level table."""
    assert (
        human_level(
            names, caption_has_human=caption_has_human, window_level=window_level
        )
        is expected
    )


def test_active_window_expires_at_the_exact_boundary() -> None:
    """A window is open strictly inside the cooldown, closed at and after it."""
    window = _window(started=100.0)
    assert active_window(window, now=219.9, cooldown_s=120) is window
    assert active_window(window, now=220.0, cooldown_s=120) is None
    assert active_window(None, now=0.0, cooldown_s=120) is None


@pytest.mark.parametrize(
    ("window_level", "batch_level", "capture_ts", "expected"),
    [
        # Escalation sounds; anything else inside the window is quiet.
        (_NONE, _KNOWN, 1001.0, CooldownAction.BYPASS),
        (_NONE, _UNIDENTIFIED, 1001.0, CooldownAction.BYPASS),
        (_KNOWN, _UNIDENTIFIED, 1001.0, CooldownAction.BYPASS),
        (_KNOWN, _KNOWN, 1001.0, CooldownAction.QUIET),
        (_UNIDENTIFIED, _KNOWN, 1001.0, CooldownAction.QUIET),
        (_UNIDENTIFIED, _UNIDENTIFIED, 1001.0, CooldownAction.QUIET),
        (_KNOWN, _NONE, 1001.0, CooldownAction.QUIET),
        # The same capture second is not stale; an older batch is.
        (_KNOWN, _KNOWN, 1000.0, CooldownAction.QUIET),
        (_KNOWN, _KNOWN, 999.0, CooldownAction.SKIP_STALE),
        # A sounding push is never withheld, however old its batch.
        (_KNOWN, _UNIDENTIFIED, 999.0, CooldownAction.BYPASS),
    ],
)
def test_decide_inside_a_window(
    window_level: HumanLevel,
    batch_level: HumanLevel,
    capture_ts: float,
    expected: CooldownAction,
) -> None:
    """Each row of the decision table for an open window."""
    window = _window(window_level, last_capture=1000.0)
    assert decide(window, level=batch_level, capture_ts=capture_ts) is expected


def test_decide_without_a_window_opens_one() -> None:
    """No active window always opens, even for an old batch with no human."""
    assert decide(None, level=_NONE, capture_ts=0.0) is CooldownAction.OPEN


def test_alternating_subjects_cannot_flap() -> None:
    """A level never goes back down, so resident/stranger/resident sounds twice."""
    window = _window(_NONE)
    sounding = 0
    for batch_level in (_KNOWN, _UNIDENTIFIED, _KNOWN, _UNIDENTIFIED, _KNOWN):
        action = decide(window, level=batch_level, capture_ts=1001.0)
        if action is CooldownAction.BYPASS:
            sounding += 1
            window = _window(batch_level)
    assert sounding == 2
    assert window.level is _UNIDENTIFIED


def test_notification_tag_is_per_window() -> None:
    """A new window id gives a new card; the same window reuses its tag."""
    first = _window()
    second = CooldownWindow(
        started=first.started,
        window_id=first.window_id + 60,
        level=first.level,
        last_capture=first.last_capture,
    )
    assert notification_tag("front_door", first) == "hga_camera_front_door_1790000000"
    assert notification_tag("front_door", first) != notification_tag(
        "front_door", second
    )

# ruff: noqa: S101
"""Tests for the pure notification cooldown policy (issue #672)."""

from __future__ import annotations

import pytest

from custom_components.home_generative_agent.core.notify_cooldown import (
    BatchEvidence,
    CooldownAction,
    CooldownWindow,
    HumanLevel,
    active_window,
    batch_evidence,
    decide,
    next_window,
    notification_tag,
)

_NONE = HumanLevel.NONE
_KNOWN = HumanLevel.KNOWN
_UNIDENTIFIED = HumanLevel.UNIDENTIFIED


def _window(
    level: HumanLevel = _NONE,
    *,
    unknown_face: bool = False,
    started: float = 100.0,
    window_id: int = 1_790_000_000,
) -> CooldownWindow:
    return CooldownWindow(
        started=started, window_id=window_id, level=level, unknown_face=unknown_face
    )


def _ev(level: HumanLevel, *, unknown_face: bool = False) -> BatchEvidence:
    return BatchEvidence(level, unknown_face=unknown_face)


@pytest.mark.parametrize(
    ("names", "caption_has_human", "window_level", "expected"),
    [
        # An explicit unknown face is unidentified and face-confirmed.
        (["Unknown Person"], False, None, _ev(_UNIDENTIFIED, unknown_face=True)),
        (
            ["Lindo", "Unknown Person"],
            True,
            _KNOWN,
            _ev(_UNIDENTIFIED, unknown_face=True),
        ),
        # Pre-guard gallery rows may carry label variants.
        ([" unknown person "], False, _KNOWN, _ev(_UNIDENTIFIED, unknown_face=True)),
        # Enrolled names only are known, with or without a caption human.
        (["Lindo"], False, None, _ev(_KNOWN)),
        (["Lindo", "Indeterminate"], True, None, _ev(_KNOWN)),
        # A caption human with no face result is unidentified, never
        # face-confirmed...
        ([], True, None, _ev(_UNIDENTIFIED)),
        (["Indeterminate"], True, _NONE, _ev(_UNIDENTIFIED)),
        ([], True, _UNIDENTIFIED, _ev(_UNIDENTIFIED)),
        # ...except inside a KNOWN window: the resident turned away.
        ([], True, _KNOWN, _ev(_KNOWN)),
        (["Indeterminate"], True, _KNOWN, _ev(_KNOWN)),
        # No human evidence at all, including legacy placeholders.
        ([], False, None, _ev(_NONE)),
        (["Indeterminate", "None", ""], False, _KNOWN, _ev(_NONE)),
    ],
)
def test_batch_evidence(
    names: list[str],
    caption_has_human: bool,  # noqa: FBT001
    window_level: HumanLevel | None,
    expected: BatchEvidence,
) -> None:
    """Each row of the batch evidence table."""
    assert (
        batch_evidence(
            names, caption_has_human=caption_has_human, window_level=window_level
        )
        == expected
    )


def test_active_window_expires_at_the_exact_boundary() -> None:
    """A window is open strictly inside the cooldown, closed at and after it."""
    window = _window(started=100.0)
    assert active_window(window, now=219.9, cooldown_s=120) is window
    assert active_window(window, now=220.0, cooldown_s=120) is None
    assert active_window(None, now=0.0, cooldown_s=120) is None


@pytest.mark.parametrize(
    ("window", "evidence", "expected"),
    [
        # A higher human level sounds.
        (_window(_NONE), _ev(_KNOWN), CooldownAction.BYPASS),
        (_window(_NONE), _ev(_UNIDENTIFIED), CooldownAction.BYPASS),
        (_window(_KNOWN), _ev(_UNIDENTIFIED, unknown_face=True), CooldownAction.BYPASS),
        # The same level replaces the window's card quietly.
        (_window(_NONE), _ev(_NONE), CooldownAction.QUIET),
        (_window(_KNOWN), _ev(_KNOWN), CooldownAction.QUIET),
        (_window(_UNIDENTIFIED), _ev(_UNIDENTIFIED), CooldownAction.QUIET),
        # A calmer scene goes to the follow-up card, not the alert's card.
        (_window(_KNOWN), _ev(_NONE), CooldownAction.FOLLOW_UP),
        (_window(_UNIDENTIFIED), _ev(_KNOWN), CooldownAction.FOLLOW_UP),
        (_window(_UNIDENTIFIED), _ev(_NONE), CooldownAction.FOLLOW_UP),
        # The first face-confirmed unknown person sounds even when a caption
        # had already put the window at UNIDENTIFIED...
        (
            _window(_UNIDENTIFIED),
            _ev(_UNIDENTIFIED, unknown_face=True),
            CooldownAction.BYPASS,
        ),
        # ...and only once per window.
        (
            _window(_UNIDENTIFIED, unknown_face=True),
            _ev(_UNIDENTIFIED, unknown_face=True),
            CooldownAction.QUIET,
        ),
    ],
)
def test_decide_inside_a_window(
    window: CooldownWindow, evidence: BatchEvidence, expected: CooldownAction
) -> None:
    """Each row of the decision table for an open window."""
    assert decide(window, evidence) is expected


def test_decide_without_a_window_opens_one() -> None:
    """No active window always opens, even with no human in the batch."""
    assert decide(None, _ev(_NONE)) is CooldownAction.OPEN


def test_alternating_subjects_cannot_flap() -> None:
    """A level never goes back down, so resident/stranger/resident sounds twice."""
    window = _window(_NONE)
    sounding = 0
    for evidence in (
        _ev(_KNOWN),
        _ev(_UNIDENTIFIED, unknown_face=True),
        _ev(_KNOWN),
        _ev(_UNIDENTIFIED, unknown_face=True),
        _ev(_KNOWN),
    ):
        if decide(window, evidence) is CooldownAction.BYPASS:
            sounding += 1
            window = next_window(
                window, evidence, now=window.started + 1, wall_time=0.0, carry=window
            )
    assert sounding == 2
    assert window.level is _UNIDENTIFIED
    assert window.unknown_face


def test_next_window_from_a_bypass_keeps_what_the_window_showed() -> None:
    """A bypass restarts the clock but never lowers the level or the face flag."""
    carried = _window(_UNIDENTIFIED, unknown_face=True, started=100.0)

    restarted = next_window(
        carried, _ev(_KNOWN), now=150.0, wall_time=1_790_000_050.0, carry=carried
    )

    assert restarted.started == 150.0
    assert restarted.level is _UNIDENTIFIED
    assert restarted.unknown_face
    assert restarted.window_id == 1_790_000_050


def test_next_window_after_expiry_starts_from_the_batch_alone() -> None:
    """An expired window contributes nothing but its id."""
    expired = _window(_UNIDENTIFIED, unknown_face=True)

    opened = next_window(
        expired, _ev(_NONE), now=900.0, wall_time=1_790_000_800.0, carry=None
    )

    assert opened.level is _NONE
    assert not opened.unknown_face


@pytest.mark.parametrize(
    "wall_time", [1_790_000_000.0, 1_790_000_000.9, 1_789_999_000.0]
)
def test_window_ids_are_unique_within_a_second_and_across_clock_steps(
    wall_time: float,
) -> None:
    """Two windows in one second, or after a clock step back, get distinct ids."""
    previous = _window(window_id=1_790_000_000)

    following = next_window(
        previous, _ev(_KNOWN), now=101.0, wall_time=wall_time, carry=previous
    )

    assert following.window_id == previous.window_id + 1


def test_notification_tag_is_per_window_and_per_card() -> None:
    """A new window gives a new card; the follow-up card has its own tag."""
    first = _window()
    second = _window(window_id=first.window_id + 60)
    camera = "camera.front_door"

    assert notification_tag(camera, first) == notification_tag(camera, first)
    assert notification_tag(camera, first) != notification_tag(camera, second)
    assert notification_tag(camera, first) != notification_tag(
        camera, first, follow_up=True
    )
    assert notification_tag(camera, first) != notification_tag("camera.side", first)


def test_notification_tag_fits_the_apns_collapse_id_limit() -> None:
    """The tag stays under 64 bytes however long the entity id is."""
    camera = "camera." + "very_long_camera_object_id_" * 8

    tag = notification_tag(camera, _window(), follow_up=True)

    assert len(tag.encode()) <= 64

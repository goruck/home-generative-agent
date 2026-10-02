# ruff: noqa: S101
"""Tests for the pure notification cooldown policy (issue #672)."""

from __future__ import annotations

import pytest

from custom_components.home_generative_agent.core.notify_cooldown import (
    CooldownAction,
    CooldownWindow,
    active_window,
    card_id_after,
    decide,
    has_unknown_face,
    notification_tag,
    opened_window,
    window_after_unknown_face,
)

_TARGET = "notify.mobile_app_phone"


def _window(
    *,
    unknown_face: bool = False,
    started: float = 100.0,
    card_id: int = 1_790_000_000,
    target: str = _TARGET,
) -> CooldownWindow:
    return CooldownWindow(
        started=started, card_id=card_id, target=target, unknown_face=unknown_face
    )


@pytest.mark.parametrize(
    ("names", "expected"),
    [
        (["Unknown Person"], True),
        (["Lindo", "Unknown Person"], True),
        # Pre-guard gallery rows may carry label variants.
        ([" unknown person "], True),
        (["Lindo"], False),
        (["Indeterminate", "None", ""], False),
        ([], False),
    ],
)
def test_has_unknown_face(names: list[str], expected: bool) -> None:  # noqa: FBT001
    """Only the face-recognition label counts, never a name that resembles it."""
    assert has_unknown_face(names) is expected


def test_active_window_expires_at_the_exact_boundary() -> None:
    """A window is open strictly inside the cooldown, closed at and after it."""
    window = _window(started=100.0)
    assert active_window(window, now=219.9, cooldown_s=120, target=_TARGET) is window
    assert active_window(window, now=220.0, cooldown_s=120, target=_TARGET) is None
    assert active_window(None, now=0.0, cooldown_s=120, target=_TARGET) is None


def test_active_window_belongs_to_its_notify_target() -> None:
    """A target that did not get the sounding push has no open window."""
    window = _window(started=100.0)
    assert (
        active_window(
            window, now=110.0, cooldown_s=120, target="notify.mobile_app_other"
        )
        is None
    )


@pytest.mark.parametrize(
    ("window", "unknown_in_batch", "unknown_displayed", "expected"),
    [
        # No window: always open, whatever the batch holds.
        (None, False, False, CooldownAction.OPEN),
        (None, True, True, CooldownAction.OPEN),
        # The window's first unknown face sounds, shown or not.
        (_window(), True, True, CooldownAction.UNKNOWN_FACE),
        (_window(), True, False, CooldownAction.UNKNOWN_FACE),
        # Nothing else sounds.
        (_window(), False, False, CooldownAction.QUIET),
        # After the unknown-face alert: the same subject replaces its card...
        (_window(unknown_face=True), True, True, CooldownAction.QUIET),
        # ...and a push that does not show an unknown face leaves it alone.
        (_window(unknown_face=True), False, False, CooldownAction.FOLLOW_UP),
        (_window(unknown_face=True), True, False, CooldownAction.FOLLOW_UP),
    ],
)
def test_decide(
    window: CooldownWindow | None,
    unknown_in_batch: bool,  # noqa: FBT001
    unknown_displayed: bool,  # noqa: FBT001
    expected: CooldownAction,
) -> None:
    """Each row of the decision table."""
    assert (
        decide(
            window,
            unknown_in_batch=unknown_in_batch,
            unknown_displayed=unknown_displayed,
        )
        is expected
    )


def test_opened_window_starts_from_the_batch_alone() -> None:
    """An expired window contributes nothing but its card id."""
    expired = _window(unknown_face=True)

    opened = opened_window(
        expired,
        now=900.0,
        wall_time=1_790_000_800.0,
        target=_TARGET,
        unknown_face=False,
    )

    assert opened == CooldownWindow(
        started=900.0, card_id=1_790_000_800, target=_TARGET, unknown_face=False
    )


def test_unknown_face_gets_a_new_card_without_moving_the_window() -> None:
    """The window's fixed length is what bounds every quiet period."""
    window = _window(started=100.0)

    after = window_after_unknown_face(window, wall_time=1_790_000_050.0)

    assert after.started == window.started
    assert after.unknown_face
    assert after.card_id == 1_790_000_050
    assert after.target == window.target


@pytest.mark.parametrize(
    "wall_time", [1_790_000_000.0, 1_790_000_000.9, 1_789_999_000.0]
)
def test_card_ids_are_unique_within_a_second_and_across_clock_steps(
    wall_time: float,
) -> None:
    """Two cards in one second, or after a clock step back, get distinct ids."""
    previous = _window(card_id=1_790_000_000)

    assert card_id_after(previous, wall_time=wall_time) == previous.card_id + 1
    assert card_id_after(None, wall_time=wall_time) == int(wall_time)


def test_notification_tag_is_per_card() -> None:
    """A new card id gives a new card; the follow-up card has its own tag."""
    first = _window()
    second = _window(card_id=first.card_id + 60)
    camera = "camera.front_door"

    assert notification_tag(camera, first) == notification_tag(camera, first)
    assert notification_tag(camera, first) != notification_tag(camera, second)
    assert notification_tag(camera, first, follow_up=True) == (
        notification_tag(camera, first) + "_more"
    )
    assert notification_tag(camera, first) != notification_tag("camera.side", first)


def test_notification_tag_fits_the_apns_collapse_id_limit() -> None:
    """The tag stays under 64 bytes however long the entity id is."""
    camera = "camera." + "very_long_camera_object_id_" * 8

    tag = notification_tag(camera, _window(), follow_up=True)

    assert len(tag.encode()) <= 64

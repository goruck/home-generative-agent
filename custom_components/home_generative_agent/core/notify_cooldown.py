"""
Per-camera notification cooldown policy for the video analyzer (issue #672).

Pure decision logic: no Home Assistant objects, no I/O, no clock reads. The
analyzer owns the per-camera lock, the state dict and the dispatch; this module
only answers "what should this push do to the camera's notification cards".

A window opens with a sounding push and runs for a fixed time. Later pushes
inside it replace that card quietly. The one exception is face recognition
reporting an unknown person for the first time in the window: that sounds once.

Captions deliberately decide nothing here. Guessing "a new person appeared"
from free text was tried and is unsafe for a security alert: a model's wording
for a person is open-ended ("a figure", "someone"), and a wrong guess means a
silent notification. Face recognition gives a result, not a guess.
"""

from __future__ import annotations

import dataclasses
import hashlib
from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING

from ..const import UNKNOWN_PERSON_LABEL  # noqa: TID252

if TYPE_CHECKING:
    from collections.abc import Iterable

_UNKNOWN_PERSON: str = UNKNOWN_PERSON_LABEL.lower()
_CAMERA_HASH_CHARS = 12


class CooldownAction(StrEnum):
    """What one push does to the camera's notification cards."""

    # No open window (or it expired): sounding push, new card, new window.
    OPEN = "open"
    # Open window, first face-confirmed unknown person in it: sounding push on
    # a new card. The window keeps its start time.
    UNKNOWN_FACE = "unknown_face"
    # Open window, nothing that must sound: replace the current card quietly.
    QUIET = "quiet"
    # Open window whose card shows an unknown person, and this push does not:
    # update a second card quietly, so the alert's card keeps explaining the
    # sound.
    FOLLOW_UP = "follow_up"


@dataclass(frozen=True, slots=True)
class CooldownWindow:
    """One camera's open notification window."""

    # Monotonic seconds at the push that opened the window. Never moved: the
    # window has a fixed length, so nothing can stay quiet longer than that.
    started: float
    # Identifies the window's current alert card; see card_id_after.
    card_id: int
    # The notify service ("domain.service") the window's pushes went to.
    target: str
    # A face-confirmed unknown person has already sounded in this window.
    unknown_face: bool


def has_unknown_face(names: Iterable[str]) -> bool:
    """Return True when face recognition reported an unknown person."""
    return any(name.strip().lower() == _UNKNOWN_PERSON for name in names)


def active_window(
    window: CooldownWindow | None, *, now: float, cooldown_s: float, target: str
) -> CooldownWindow | None:
    """
    Return `window` while it is open at monotonic `now` for this `target`.

    A window belongs to the notify target that received its sounding push; a
    different target has not been alerted, so it starts its own window.
    """
    if window is None or window.target != target:
        return None
    if now - window.started >= cooldown_s:
        return None
    return window


def decide(
    window: CooldownWindow | None, *, unknown_in_batch: bool, unknown_displayed: bool
) -> CooldownAction:
    """
    Decide what a push that passed the novelty check does.

    `window` is the camera's ACTIVE window (see active_window), or None.
    `unknown_in_batch` is whether any frame of the batch carried an unknown
    face; `unknown_displayed` is whether the frames the notification's text and
    image were built from did. They differ when the unknown face sits in a
    frame the summary did not keep.
    """
    if window is None:
        return CooldownAction.OPEN
    if unknown_in_batch and not window.unknown_face:
        return CooldownAction.UNKNOWN_FACE
    if window.unknown_face and not unknown_displayed:
        return CooldownAction.FOLLOW_UP
    return CooldownAction.QUIET


def card_id_after(previous: CooldownWindow | None, *, wall_time: float) -> int:
    """
    Return an id for a new alert card, unique per camera.

    Wall-clock seconds keep ids distinct across restarts (state is in memory);
    stepping past the previous id keeps them distinct within one second and
    when the clock is set back.
    """
    card_id = int(wall_time)
    if previous is not None and card_id <= previous.card_id:
        card_id = previous.card_id + 1
    return card_id


def opened_window(
    previous: CooldownWindow | None,
    *,
    now: float,
    wall_time: float,
    target: str,
    unknown_face: bool,
) -> CooldownWindow:
    """Build the window an OPEN push starts."""
    return CooldownWindow(
        started=now,
        card_id=card_id_after(previous, wall_time=wall_time),
        target=target,
        unknown_face=unknown_face,
    )


def window_after_unknown_face(
    window: CooldownWindow, *, wall_time: float
) -> CooldownWindow:
    """Give an open window a new alert card for its first unknown face."""
    return dataclasses.replace(
        window,
        card_id=card_id_after(window, wall_time=wall_time),
        unknown_face=True,
    )


def notification_tag(
    camera_id: str, window: CooldownWindow, *, follow_up: bool = False
) -> str:
    """
    Return the tag that makes later pushes replace one card.

    The camera is hashed so the tag stays far below the 64 bytes APNs allows
    for a collapse id, whatever the entity id's length.
    """
    digest = hashlib.sha256(camera_id.encode()).hexdigest()[:_CAMERA_HASH_CHARS]
    suffix = "_more" if follow_up else ""
    return f"hga_cam_{digest}_{window.card_id}{suffix}"

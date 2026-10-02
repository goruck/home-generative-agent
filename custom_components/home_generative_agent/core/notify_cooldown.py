"""
Per-camera notification cooldown policy for the video analyzer (issue #672).

Pure decision logic: no Home Assistant objects, no I/O, no clock reads. The
analyzer owns the per-camera lock, the state dict and the dispatch; this module
only answers "what should this push do to the camera's notification card".

A window opens with a sounding push. Later pushes inside it replace the same
card quietly, unless the scene escalates: a human appears where there was none,
or an unidentified person appears where only enrolled people had been seen.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum, StrEnum
from typing import TYPE_CHECKING

from ..const import RESERVED_IDENTITY_LABELS, UNKNOWN_PERSON_LABEL  # noqa: TID252

if TYPE_CHECKING:
    from collections.abc import Iterable

_UNKNOWN_PERSON: str = UNKNOWN_PERSON_LABEL.lower()


class HumanLevel(IntEnum):
    """Who a window (or one batch) has shown. Ordered: a higher level escalates."""

    NONE = 0
    KNOWN = 1
    UNIDENTIFIED = 2


class CooldownAction(StrEnum):
    """What one push does to the camera's notification card."""

    # No open window (or it expired): sounding push, new card, new window.
    OPEN = "open"
    # Open window, the scene escalated: sounding push, new card, window restarts.
    BYPASS = "bypass"
    # Open window, nothing new: replace the window's card without sound.
    QUIET = "quiet"
    # Open window, but this batch is older than the one already on the card.
    SKIP_STALE = "skip_stale"


@dataclass(frozen=True, slots=True)
class CooldownWindow:
    """One camera's open notification window."""

    # Monotonic seconds at the window's last sounding push.
    started: float
    # Wall-clock epoch seconds at that push; makes the card's tag unique.
    window_id: int
    level: HumanLevel
    # Capture time (epoch seconds) of the batch the card currently shows.
    last_capture: float


def active_window(
    window: CooldownWindow | None, *, now: float, cooldown_s: float
) -> CooldownWindow | None:
    """Return `window` while it is still open at monotonic time `now`."""
    if window is None or now - window.started >= cooldown_s:
        return None
    return window


def human_level(
    names: Iterable[str],
    *,
    caption_has_human: bool,
    window_level: HumanLevel | None,
) -> HumanLevel:
    """
    Classify one batch from its face-recognition names and its caption.

    An explicit "Unknown Person" result is unidentified; enrolled names alone
    are known. A caption human with no face result is unidentified too, except
    inside a window already at KNOWN: there it is most likely the recognized
    resident turned away from the camera, so it must not escalate.
    """
    normalized = {name.strip().lower() for name in names}
    if _UNKNOWN_PERSON in normalized:
        return HumanLevel.UNIDENTIFIED
    if normalized - RESERVED_IDENTITY_LABELS:
        return HumanLevel.KNOWN
    if caption_has_human:
        if window_level is HumanLevel.KNOWN:
            return HumanLevel.KNOWN
        return HumanLevel.UNIDENTIFIED
    return HumanLevel.NONE


def decide(
    window: CooldownWindow | None, *, level: HumanLevel, capture_ts: float
) -> CooldownAction:
    """
    Decide what a push that passed the novelty check does.

    `window` is the camera's ACTIVE window (see active_window), or None.
    Sounding pushes are never withheld; only a quiet update can be stale.
    """
    if window is None:
        return CooldownAction.OPEN
    if level > window.level:
        return CooldownAction.BYPASS
    if capture_ts < window.last_capture:
        return CooldownAction.SKIP_STALE
    return CooldownAction.QUIET


def notification_tag(camera_name: str, window: CooldownWindow) -> str:
    """Return the per-window tag that makes later pushes replace one card."""
    return f"hga_camera_{camera_name}_{window.window_id}"

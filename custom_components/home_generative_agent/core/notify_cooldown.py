"""
Per-camera notification cooldown policy for the video analyzer (issue #672).

Pure decision logic: no Home Assistant objects, no I/O, no clock reads. The
analyzer owns the per-camera lock, the state dict and the dispatch; this module
only answers "what should this push do to the camera's notification cards".

A window opens with a sounding push. Later pushes inside it replace that card
quietly, unless the scene escalates: a human appears where there was none, an
unidentified person appears where only enrolled people had been seen, or face
recognition reports an unknown person for the first time in the window.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from enum import IntEnum, StrEnum
from typing import TYPE_CHECKING

from ..const import RESERVED_IDENTITY_LABELS, UNKNOWN_PERSON_LABEL  # noqa: TID252

if TYPE_CHECKING:
    from collections.abc import Iterable

_UNKNOWN_PERSON: str = UNKNOWN_PERSON_LABEL.lower()
_CAMERA_HASH_CHARS = 12


class HumanLevel(IntEnum):
    """Who a window (or one batch) has shown. Ordered: a higher level escalates."""

    NONE = 0
    KNOWN = 1
    UNIDENTIFIED = 2


class CooldownAction(StrEnum):
    """What one push does to the camera's notification cards."""

    # No open window (or it expired): sounding push, new card, new window.
    OPEN = "open"
    # Open window, the scene escalated: sounding push, new card, window restarts.
    BYPASS = "bypass"
    # Open window, same human level: replace the window's card without sound.
    QUIET = "quiet"
    # Open window, a calmer scene than the one that sounded: update a second
    # card without sound, so the alert's own card keeps explaining the sound.
    FOLLOW_UP = "follow_up"


@dataclass(frozen=True, slots=True)
class BatchEvidence:
    """What one analyzed batch shows about humans."""

    level: HumanLevel
    # Face recognition itself reported "Unknown Person" (not a caption guess).
    unknown_face: bool


@dataclass(frozen=True, slots=True)
class CooldownWindow:
    """One camera's open notification window."""

    # Monotonic seconds at the window's last sounding push.
    started: float
    # Makes the window's card tag unique; see next_window_id.
    window_id: int
    level: HumanLevel
    # A face-confirmed unknown person has already sounded in this window.
    unknown_face: bool


def active_window(
    window: CooldownWindow | None, *, now: float, cooldown_s: float
) -> CooldownWindow | None:
    """Return `window` while it is still open at monotonic time `now`."""
    if window is None or now - window.started >= cooldown_s:
        return None
    return window


def batch_evidence(
    names: Iterable[str],
    *,
    caption_has_human: bool,
    window_level: HumanLevel | None,
) -> BatchEvidence:
    """
    Classify one batch from its face-recognition names and its caption.

    An explicit "Unknown Person" result is unidentified; enrolled names alone
    are known. A caption human with no face result is unidentified too, except
    inside a window already at KNOWN: there it is most likely the recognized
    resident turned away from the camera, so it must not escalate.
    """
    normalized = {name.strip().lower() for name in names}
    if _UNKNOWN_PERSON in normalized:
        return BatchEvidence(HumanLevel.UNIDENTIFIED, unknown_face=True)
    if normalized - RESERVED_IDENTITY_LABELS:
        return BatchEvidence(HumanLevel.KNOWN, unknown_face=False)
    if caption_has_human:
        level = (
            HumanLevel.KNOWN
            if window_level is HumanLevel.KNOWN
            else HumanLevel.UNIDENTIFIED
        )
        return BatchEvidence(level, unknown_face=False)
    return BatchEvidence(HumanLevel.NONE, unknown_face=False)


def decide(window: CooldownWindow | None, evidence: BatchEvidence) -> CooldownAction:
    """
    Decide what a push that passed the novelty check does.

    `window` is the camera's ACTIVE window (see active_window), or None.
    """
    if window is None:
        return CooldownAction.OPEN
    if evidence.level > window.level:
        return CooldownAction.BYPASS
    # A caption is a guess; a face result is not. A caption that put the window
    # at UNIDENTIFIED (even a wrong one, "no visible people") must not keep the
    # first face-confirmed unknown person in the window from sounding.
    if evidence.unknown_face and not window.unknown_face:
        return CooldownAction.BYPASS
    if evidence.level < window.level:
        return CooldownAction.FOLLOW_UP
    return CooldownAction.QUIET


def next_window(
    previous: CooldownWindow | None,
    evidence: BatchEvidence,
    *,
    now: float,
    wall_time: float,
    carry: CooldownWindow | None,
) -> CooldownWindow:
    """
    Build the window a sounding push starts.

    `previous` is the camera's last window, open or expired; its id is only
    used to keep ids unique when two windows start in the same second.
    `carry` is the still-open window a BYPASS restarts (None for OPEN): what it
    already showed stays shown, so a level never goes back down inside it.
    """
    window_id = int(wall_time)
    if previous is not None and window_id <= previous.window_id:
        window_id = previous.window_id + 1
    level = evidence.level
    unknown_face = evidence.unknown_face
    if carry is not None:
        level = max(level, carry.level)
        unknown_face = unknown_face or carry.unknown_face
    return CooldownWindow(
        started=now, window_id=window_id, level=level, unknown_face=unknown_face
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
    return f"hga_cam_{digest}_{window.window_id}{suffix}"

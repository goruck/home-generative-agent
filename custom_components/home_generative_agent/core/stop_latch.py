"""A latch that refuses an object's side effects once its owner has stopped."""

from __future__ import annotations

import logging

LOGGER = logging.getLogger(__name__)


class StopLatch:
    """
    Refuse an object's side effects once its owner has stopped (#988).

    ``cancel_and_wait`` gives up on a run that keeps catching its cancel, so a
    stop can return while that run is still going. Closed once the stop has
    stopped waiting, a latch makes such a run's later writes, notifications
    and actions no-ops: every write is refused, so nothing it changed in
    memory reaches storage, and a reload's replacement starts from what was
    saved before the stop. A run that ends within the wait is not affected.
    """

    def __init__(self, what: str) -> None:
        """Name the owner for the log line; open until ``close``."""
        self._what = what
        self._closed = False
        self._logged = False

    @property
    def closed(self) -> bool:
        """Return True once ``close`` has been called."""
        return self._closed

    def close(self) -> None:
        """Refuse every later side effect. Cannot be reopened."""
        self._closed = True

    def refuses(self, effect: str) -> bool:
        """
        Return True, and log, when *effect* must not happen: the latch is closed.

        Logged as a warning once per latch, since it means a run outlived its
        stop, and at debug level after that.
        """
        if not self._closed:
            return False
        if self._logged:
            LOGGER.debug("%s has stopped; dropped a late %s.", self._what, effect)
        else:
            self._logged = True
            LOGGER.warning(
                "%s has stopped, but a run that outlived the stop tried to "
                "write; dropped its %s and any later change (issue #988).",
                self._what,
                effect,
            )
        return True

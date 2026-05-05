"""Small logging helpers for Sentinel runtime loops."""

from __future__ import annotations

import logging
from typing import Any


class RepeatingLogLimiter:
    """Limit repeated operational warnings while preserving recovery signal."""

    def __init__(
        self,
        logger: logging.Logger,
        *,
        every: int = 10,
        recovery_level: int = logging.DEBUG,
    ) -> None:
        """Initialize the limiter for a logger."""
        self._logger = logger
        self._every = every
        self._recovery_level = recovery_level
        self._counts: dict[str, int] = {}

    def warning(
        self,
        key: str,
        msg: str,
        *args: Any,
        exc_info: bool | BaseException | tuple[Any, Any, Any] | None = None,
    ) -> None:
        """Log the first warning, then every Nth repeat for the same key."""
        count = self._counts.get(key, 0) + 1
        self._counts[key] = count
        if count != 1 and count % self._every != 0:
            return
        if count > 1:
            msg = f"{msg.rstrip('.')} (occurrence={count})."
        self._logger.warning(msg, *args, exc_info=exc_info)

    def recovered(self, key: str, msg: str, *args: Any) -> None:
        """Log once when a previously repeating condition has recovered."""
        count = self._counts.pop(key, 0)
        if count:
            self._logger.log(self._recovery_level, msg, *args, count)

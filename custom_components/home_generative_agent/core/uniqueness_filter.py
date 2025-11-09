"""Perceptual hash-based frame deduplication."""

from __future__ import annotations

import logging
from collections import deque
from pathlib import Path
from time import monotonic
from typing import TYPE_CHECKING

import aiofiles
from PIL import Image

from .image_utils import ImageUtils

if TYPE_CHECKING:
    from homeassistant.core import HomeAssistant

LOGGER = logging.getLogger(__name__)


class UniquenessFilter:
    """Perceptual hash-based frame deduplication with heartbeat."""

    def __init__(
        self,
        hass: HomeAssistant,
        enabled: bool = False,
        hash_size: int = 8,
        hamming_threshold: int = 4,
        heartbeat_sec: int = 10,
        history_size: int = 2,
    ) -> None:
        """
        Initialize uniqueness filter.

        Args:
            hass: Home Assistant instance
            enabled: Whether filtering is enabled
            hash_size: dHash grid size (8 -> 64-bit hash)
            hamming_threshold: Max Hamming distance to consider duplicate
            heartbeat_sec: Always allow a frame at least this often
            history_size: Number of recent hashes to compare against

        """
        self.hass = hass
        self._enabled = enabled
        self._hash_size = hash_size
        self._hamming_max = hamming_threshold
        self._heartbeat_sec = heartbeat_sec
        self._history_size = history_size

        # Per-camera state
        self._last_hashes: dict[str, deque[int]] = {}
        self._last_unique_ts: dict[str, float] = {}

    async def should_process(self, camera_id: str, image_path: Path) -> bool:
        """
        Check if frame is unique enough to process.

        Args:
            camera_id: Camera identifier
            image_path: Path to image file

        Returns:
            True to process, False to skip as duplicate

        """
        if not self._enabled:
            return True

        now = monotonic()

        # Check heartbeat - always allow if it's been long enough
        last_ok = self._last_unique_ts.get(camera_id, 0.0)
        heartbeat_due = now - last_ok >= self._heartbeat_sec

        # Load and hash image
        try:
            async with aiofiles.open(image_path, "rb") as f:
                data = await f.read()
            image_hash = ImageUtils.compute_dhash(data, self._hash_size)
        except (FileNotFoundError, OSError, Image.UnidentifiedImageError, ValueError):
            # On error, be permissive - allow processing
            LOGGER.debug(
                "[%s] Could not hash %s, allowing processing", camera_id, image_path
            )
            return True

        # Get or initialize history for this camera
        hist = self._last_hashes.setdefault(camera_id, deque(maxlen=self._history_size))

        # If no history, seed it and allow
        if not hist:
            hist.append(image_hash)
            self._last_unique_ts[camera_id] = now
            return True

        # Compute minimum Hamming distance to recent frames
        min_hamming = min(
            ImageUtils.hamming_distance(
                image_hash, prev, max_bits=self._hash_size * self._hash_size
            )
            for prev in hist
        )

        # Too similar and no heartbeat due -> skip
        if min_hamming <= self._hamming_max and not heartbeat_due:
            LOGGER.debug(
                "[%s] Skipping near-duplicate (Hamming=%d)", camera_id, min_hamming
            )
            return False

        # Accept: update history and timestamp
        hist.append(image_hash)
        self._last_unique_ts[camera_id] = now
        return True

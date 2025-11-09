"""Snapshot capture and storage management."""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

from homeassistant.components.camera.const import DOMAIN as CAMERA_DOMAIN
from homeassistant.exceptions import HomeAssistantError

from .datetime_utils import DateTimeUtils
from .path_utils import PathUtils

if TYPE_CHECKING:
    from homeassistant.core import HomeAssistant

LOGGER = logging.getLogger(__name__)


class SnapshotManager:
    """Handles snapshot capture and storage."""

    def __init__(self, hass: HomeAssistant, snapshot_root: Path) -> None:
        """
        Initialize snapshot manager.

        Args:
            hass: Home Assistant instance
            snapshot_root: Root directory for snapshots

        """
        self.hass = hass
        self.snapshot_root = snapshot_root
        self._initialized_dirs: set[str] = set()

    async def get_snapshot_dir(self, camera_id: str) -> Path:
        """
        Get or create snapshot directory for camera.

        Args:
            camera_id: Camera entity ID

        Returns:
            Path to camera's snapshot directory

        """
        if camera_id not in self._initialized_dirs:
            cam_dir = PathUtils.camera_snapshot_dir(self.snapshot_root, camera_id)
            await self.hass.async_add_executor_job(PathUtils.ensure_dir, cam_dir)
            self._initialized_dirs.add(camera_id)

            # Check if directory was already populated
            dir_not_empty = await self.hass.async_add_executor_job(
                lambda: any(cam_dir.iterdir())
            )
            if dir_not_empty:
                LOGGER.info(
                    "[%s] Folder not empty. Existing snapshots will not be pruned.",
                    camera_id,
                )

        return PathUtils.camera_snapshot_dir(self.snapshot_root, camera_id)

    async def take_snapshot(self, camera_id: str, timestamp: datetime) -> Path | None:
        """
        Take and save a single snapshot.

        Args:
            camera_id: Camera entity ID
            timestamp: Timestamp for filename

        Returns:
            Path to saved snapshot, or None if failed

        """
        snapshot_dir = await self.get_snapshot_dir(camera_id)
        timestamp_str = DateTimeUtils.snapshot_timestamp(timestamp)
        path = snapshot_dir / PathUtils.snapshot_filename(timestamp_str)

        try:
            # Call camera snapshot service
            await self.hass.services.async_call(
                CAMERA_DOMAIN,
                "snapshot",
                {"entity_id": camera_id, "filename": str(path)},
                blocking=False,
            )

            # Wait for file to appear (up to 10 seconds)
            for i in range(50):
                exists = await self.hass.async_add_executor_job(path.exists)
                if exists:
                    LOGGER.debug("[%s] Snapshot captured: %s", camera_id, path.name)
                    return path
                await asyncio.sleep(0.2)

            LOGGER.warning(
                "[%s] Snapshot failed to appear on disk after waiting: %s",
                camera_id,
                path,
            )
            return None

        except HomeAssistantError:
            LOGGER.warning("Snapshot failed for %s", camera_id, exc_info=True)
            return None

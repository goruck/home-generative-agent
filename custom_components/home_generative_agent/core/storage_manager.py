"""Snapshot storage, retention, and vector database management."""

from __future__ import annotations

import logging
from collections import deque
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING

import async_timeout

from ..const import (  # noqa: TID252
    VIDEO_ANALYZER_LATEST_NAME,
    VIDEO_ANALYZER_LATEST_SUBFOLDER,
)

if TYPE_CHECKING:
    from homeassistant.core import HomeAssistant
    from langgraph.store.base import BaseStore

LOGGER = logging.getLogger(__name__)


class StorageManager:
    """Manages snapshot storage, retention, and vector DB."""

    def __init__(
        self,
        hass: HomeAssistant,
        store: BaseStore,
        snapshots_to_keep: int = 100,
    ) -> None:
        """
        Initialize storage manager.

        Args:
            hass: Home Assistant instance
            store: LangGraph store for vector database
            snapshots_to_keep: Number of snapshots to retain per camera

        """
        self.hass = hass
        self.store = store
        self.snapshots_to_keep = snapshots_to_keep
        self._retention_deques: dict[str, deque[Path]] = {}

    async def store_results(
        self, camera_id: str, batch: list[Path], summary: str
    ) -> None:
        """
        Store analysis results in vector database.

        Args:
            camera_id: Camera entity ID
            batch: List of snapshot paths
            summary: Analysis summary text

        """
        camera_name = camera_id.split(".")[-1]

        try:
            async with async_timeout.timeout(10):
                await self.store.aput(
                    namespace=("video_analysis", camera_name),
                    key=batch[0].name,  # Use first snapshot as key
                    value={"content": summary, "snapshots": [str(p) for p in batch]},
                )
            LOGGER.debug(
                "[%s] Stored analysis in vector DB: %s", camera_id, batch[0].name
            )
        except TimeoutError:
            LOGGER.warning(
                "[%s] Storing results timed out, skipping vector DB", camera_id
            )

    async def prune_old_snapshots(
        self,
        camera_id: str,
        batch: list[Path],
        is_protected: Callable[[Path], bool],
    ) -> None:
        """
        Delete old snapshots respecting retention policy.

        Args:
            camera_id: Camera entity ID
            batch: Newly processed snapshots to add to retention
            is_protected: Function to check if a path is protected from deletion

        """
        # Get or create retention deque for this camera
        retention = self._retention_deques.setdefault(camera_id, deque())

        # Add new snapshots to retention
        for path in batch:
            retention.append(path)

            # Prune oldest while over limit
            while len(retention) > self.snapshots_to_keep:
                old = retention.popleft()

                # Don't delete "latest" assets
                if (
                    old.name == VIDEO_ANALYZER_LATEST_NAME
                    or old.parent.name == VIDEO_ANALYZER_LATEST_SUBFOLDER
                ):
                    # Put it back at the end and stop pruning this round
                    retention.append(old)
                    break

                # Don't delete if protected (e.g., by recent notification)
                if is_protected(old):
                    # Put it back at the end and stop pruning this round
                    retention.append(old)
                    break

                # Delete the file
                try:
                    await self.hass.async_add_executor_job(old.unlink)
                    LOGGER.debug("[%s] Deleted old snapshot: %s", camera_id, old)
                except OSError as err:
                    LOGGER.warning("[%s] Failed to delete %s: %s", camera_id, old, err)

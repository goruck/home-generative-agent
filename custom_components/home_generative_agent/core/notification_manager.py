"""Notification and anomaly detection management."""

from __future__ import annotations

import asyncio
import logging
from datetime import timedelta
from pathlib import Path
from time import monotonic
from typing import TYPE_CHECKING

import async_timeout
import homeassistant.util.dt as dt_util

from ..const import (  # noqa: TID252
    SIGNAL_HGA_NEW_LATEST,
    SIGNAL_HGA_RECOGNIZED,
    VIDEO_ANALYZER_SIMILARITY_THRESHOLD,
    VIDEO_ANALYZER_TIME_OFFSET,
)
from .datetime_utils import DateTimeUtils
from .utils import discover_mobile_notify_service, dispatch_on_loop
from .video_helpers import latest_target, publish_latest_atomic

if TYPE_CHECKING:
    from homeassistant.core import HomeAssistant
    from langgraph.store.base import BaseStore

LOGGER = logging.getLogger(__name__)


class NotificationManager:
    """Handles user notifications and anomaly detection."""

    def __init__(
        self,
        hass: HomeAssistant,
        store: BaseStore,
        snapshot_root: Path,
        notify_service: str | None = None,
    ) -> None:
        """
        Initialize notification manager.

        Args:
            hass: Home Assistant instance
            store: LangGraph store for anomaly detection
            snapshot_root: Root directory for snapshots
            notify_service: Notification service name (optional)

        """
        self.hass = hass
        self.store = store
        self.snapshot_root = snapshot_root
        self.notify_service = notify_service
        self._protected_images: dict[Path, float] = {}  # path -> expiry_time

    def protect_image(self, path: Path, ttl_sec: int = 1800) -> None:
        """
        Mark image as protected from pruning.

        Args:
            path: Path to protect
            ttl_sec: Time to live in seconds (default 30 minutes)

        """
        self._protected_images[path] = monotonic() + ttl_sec

    def is_protected(self, path: Path) -> bool:
        """
        Check if image is still protected.

        Args:
            path: Path to check

        Returns:
            True if protected, False otherwise

        """
        now = monotonic()

        # Clean up expired entries
        expired = [k for k, t in self._protected_images.items() if t < now]
        for k in expired:
            self._protected_images.pop(k, None)

        return self._protected_images.get(path, 0) > now

    async def is_anomaly(
        self, camera_name: str, summary: str, first_snapshot_name: str
    ) -> bool:
        """
        Check if event is anomalous via semantic search.

        Args:
            camera_name: Camera name (without domain)
            summary: Event summary text
            first_snapshot_name: Filename of first snapshot in batch

        Returns:
            True if anomalous, False if normal

        """
        try:
            async with async_timeout.timeout(10):
                search_results = await self.store.asearch(
                    ("video_analysis", camera_name), query=summary, limit=10
                )
        except TimeoutError:
            LOGGER.warning(
                "[%s] Anomaly detection timed out, assuming not anomalous", camera_name
            )
            return False

        # Parse timestamp from first snapshot
        try:
            first_dt = DateTimeUtils.parse_snapshot_timestamp(first_snapshot_name)
        except ValueError:
            LOGGER.warning(
                "Could not parse timestamp from %s, assuming not anomalous",
                first_snapshot_name,
            )
            return False

        # Time threshold - snapshots older than offset are considered anomalous
        time_threshold = dt_util.now() - timedelta(minutes=VIDEO_ANALYZER_TIME_OFFSET)

        # Anomaly if:
        # 1. First snapshot is older than time threshold, OR
        # 2. Any search result has low similarity score
        if first_dt < time_threshold:
            return True

        return any(
            r.score < VIDEO_ANALYZER_SIMILARITY_THRESHOLD
            for r in search_results
            if r.score is not None
        )

    async def send_notification(
        self, message: str, camera_name: str, image_path: Path
    ) -> None:
        """
        Send push notification with image.

        Args:
            message: Notification message
            camera_name: Camera name for title
            image_path: Path to image attachment

        """
        # Determine service to use
        full_service = self.notify_service
        if full_service and full_service.startswith("notify."):
            domain, service = full_service.split(".", 1)
        else:
            service = discover_mobile_notify_service(self.hass)
            LOGGER.debug("Discovered notify service: %s", service)
            if not service:
                LOGGER.warning("No notify.mobile_app_* service found.")
                return
            domain = "notify"

        # Send notification
        await self.hass.services.async_call(
            domain,
            service,
            {
                "message": message,
                "title": f"Camera Alert from {camera_name}!",
                "data": {"image": str(image_path)},
            },
            blocking=False,
        )
        LOGGER.debug("[%s] Sent notification: %s", camera_name, message)

    async def handle_notification(
        self,
        camera_id: str,
        summary: str,
        batch: list[Path],
        mode: str,
        recognized_people: list[str],
        queue_size: int = 0,
    ) -> None:
        """
        Decide whether to notify and send notification.

        Args:
            camera_id: Camera entity ID
            summary: Event summary
            batch: List of snapshot paths in this event
            mode: Notification mode ("notify_on_all" or "notify_on_anomaly")
            recognized_people: List of recognized person names
            queue_size: Current queue backlog size

        """
        camera_name = camera_id.split(".")[-1]

        # Choose middle frame as representative
        chosen = batch[len(batch) // 2]

        # Publish as "latest" image
        latest_path = latest_target(self.snapshot_root, camera_id)
        await publish_latest_atomic(self.hass, chosen, latest_path)

        # Fire bus event
        self.hass.bus.async_fire(
            "hga_last_event_frame",
            {
                "camera_id": camera_id,
                "summary": summary,
                "path": str(chosen),
                "latest": str(latest_path),
            },
        )

        # Notify ImageEntity listeners
        dispatch_on_loop(
            self.hass,
            SIGNAL_HGA_NEW_LATEST,
            camera_id,
            str(latest_path),
            summary,
            list(recognized_people),
            dt_util.utcnow().isoformat(),
        )

        # Notify Sensor listeners
        dispatch_on_loop(
            self.hass,
            SIGNAL_HGA_RECOGNIZED,
            camera_id,
            list(recognized_people),
            summary,
            dt_util.utcnow().isoformat(),
            str(latest_path),
        )

        # Wait if backlog is high
        if queue_size > 10:
            await asyncio.sleep(0.5)

        # Prepare notification image path
        # Use /media/local prefix for Home Assistant media serving
        media_dir = "/media/local"
        notify_img = Path(media_dir) / Path(*chosen.parts[-3:])

        # Decide whether to send notification
        should_notify = False

        if mode == "notify_on_anomaly":
            first_snapshot = batch[0].parts[-1]
            is_anom = await self.is_anomaly(camera_name, summary, first_snapshot)
            if is_anom:
                LOGGER.debug("[%s] Video is an anomaly!", camera_id)
                should_notify = True
        else:
            # notify_on_all mode
            should_notify = True

        if should_notify:
            # Protect the image from pruning for 30 minutes
            self.protect_image(chosen, ttl_sec=1800)
            await self.send_notification(summary, camera_name, notify_img)

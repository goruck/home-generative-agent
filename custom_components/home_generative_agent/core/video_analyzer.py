"""Video analyzer for recording and motion-triggered cameras - REFACTORED."""

from __future__ import annotations

import asyncio
import logging
import re
from collections.abc import Callable
from datetime import datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Final

import homeassistant.util.dt as dt_util
from homeassistant.core import HomeAssistant, callback
from homeassistant.exceptions import HomeAssistantError
from homeassistant.helpers.event import async_track_time_interval

from ..const import (  # noqa: TID252
    CONF_NOTIFY_SERVICE,
    CONF_VIDEO_ANALYZER_MODE,
    VIDEO_ANALYZER_MOTION_CAMERA_MAP,
    VIDEO_ANALYZER_SCAN_INTERVAL,
    VIDEO_ANALYZER_SNAPSHOT_ROOT,
    VIDEO_ANALYZER_SNAPSHOTS_TO_KEEP,
    VIDEO_ANALYZER_TRIGGER_ON_MOTION,
)
from .face_recognition_service import FaceRecognitionService
from .frame_processor import FrameProcessor
from .notification_manager import NotificationManager
from .queue_manager import QueueManager
from .snapshot_manager import SnapshotManager
from .storage_manager import StorageManager
from .uniqueness_filter import UniquenessFilter
from .video_analyzer_metrics import VideoAnalyzerMetrics
from .video_summarizer import VideoSummarizer

if TYPE_CHECKING:
    from homeassistant.core import Event

    from .runtime import HGAConfigEntry

LOGGER = logging.getLogger(__name__)

# Tuning constants
_MAX_BATCH: Final[int] = 5
_QUEUE_MAXSIZE: Final[int] = 50
_FRAME_DEADLINE_SEC: Final[int] = 240
_SUMMARY_TIMEOUT_SEC: Final[int] = 60
_FACE_TIMEOUT_SEC: Final[int] = 10
_VISION_TIMEOUT_SEC: Final[int] = 60
_GLOBAL_VISION_CONCURRENCY: Final[int] = 3
_UNIQUENESS_ENABLED: Final[bool] = False
_METRICS_REPORT_INTERVAL_SEC: Final[int] = 3600

# Global vision concurrency semaphore
_global_vision_sem = asyncio.Semaphore(_GLOBAL_VISION_CONCURRENCY)


class VideoAnalyzer:
    """Analyze video from recording or motion-triggered cameras - Orchestrator."""

    def __init__(self, hass: HomeAssistant, entry: HGAConfigEntry) -> None:
        """Initialize the video analyzer with specialized components.

        Args:
            hass: Home Assistant instance
            entry: Config entry with runtime data
        """
        self.hass = hass
        self.entry = entry

        snapshot_root = Path(VIDEO_ANALYZER_SNAPSHOT_ROOT)

        # Initialize specialized services via dependency injection
        self.snapshot_mgr = SnapshotManager(hass, snapshot_root)

        self.uniqueness = UniquenessFilter(
            hass, enabled=_UNIQUENESS_ENABLED
        )

        self.face_service = FaceRecognitionService(
            hass,
            entry.runtime_data.face_api_url,
            entry.runtime_data.person_gallery,
            snapshot_root,
            timeout=_FACE_TIMEOUT_SEC,
            save_debug_crops=entry.runtime_data.face_mode == "debug",
        )

        self.frame_processor = FrameProcessor(
            hass,
            entry.runtime_data.vision_model,
            self.face_service,
            _global_vision_sem,
            face_timeout=_FACE_TIMEOUT_SEC,
            vision_timeout=_VISION_TIMEOUT_SEC,
            frame_deadline=_FRAME_DEADLINE_SEC,
        )

        self.summarizer = VideoSummarizer(
            entry.runtime_data.summarization_model, timeout=_SUMMARY_TIMEOUT_SEC
        )

        self.notifier = NotificationManager(
            hass,
            entry.runtime_data.store,
            snapshot_root,
            entry.options.get(CONF_NOTIFY_SERVICE),
        )

        self.storage = StorageManager(
            hass, entry.runtime_data.store, VIDEO_ANALYZER_SNAPSHOTS_TO_KEEP
        )

        self.metrics = VideoAnalyzerMetrics(
            report_interval_sec=_METRICS_REPORT_INTERVAL_SEC
        )

        # Queue coordination
        self.queue_mgr: QueueManager[Path] = QueueManager(
            hass, worker_factory=self._snapshot_worker, max_size=_QUEUE_MAXSIZE
        )

        # Motion-triggered camera tracking
        self._active_motion_cameras: dict[str, asyncio.Task] = {}
        self._last_recognized: dict[str, list[str]] = {}

        # Lifecycle management
        self._cancel_track: Callable[[], None] | None = None
        self._cancel_listen: Callable[[], None] | None = None
        self._cancel_motion_listen: Callable[[], None] | None = None
        self._metrics_job_cancel: Callable[[], None] | None = None

    async def _get_batch(
        self, queue: asyncio.Queue[Path], camera_id: str
    ) -> list[Path]:
        """Get a batch of snapshots from queue.

        Args:
            queue: Queue to pull from
            camera_id: Camera identifier for logging

        Returns:
            List of snapshot paths (up to _MAX_BATCH)
        """
        # Wait for first item
        first = await queue.get()
        batch = [first]

        # Try to get more (non-blocking)
        n = max(_MAX_BATCH - 1, 0)
        batch.extend(QueueManager.drain(queue)[:n])

        LOGGER.debug(
            "[%s] Start t=%s batch=%d qsize=%d",
            camera_id,
            dt_util.utcnow().isoformat(),
            len(batch),
            queue.qsize(),
        )
        return batch

    async def _snapshot_worker(self, camera_id: str) -> None:
        """Consume and process snapshots for one camera.

        Args:
            camera_id: Camera entity ID
        """
        queue = self.queue_mgr.get_queue(camera_id)
        if not queue:
            LOGGER.error("[%s] Worker started without queue!", camera_id)
            return

        LOGGER.debug("[%s] Worker started", camera_id)

        try:
            while True:
                # Get batch
                batch = await self._get_batch(queue, camera_id)

                # Order by timestamp
                ordered = self.frame_processor.order_by_timestamp(batch)
                if not ordered:
                    continue

                # Check age
                first_epoch = ordered[0][1]
                age = int(dt_util.utcnow().timestamp() - first_epoch)
                LOGGER.debug(
                    "[%s] Dequeue age=%ds qsize=%d", camera_id, age, queue.qsize()
                )

                # Process batch
                await self._analyze_and_finalize(camera_id, ordered)

        except asyncio.CancelledError:
            LOGGER.debug("[%s] Worker cancelled", camera_id)
            raise
        except Exception:
            LOGGER.exception("[%s] Worker crashed", camera_id)

    async def _analyze_and_finalize(
        self, camera_id: str, ordered: list[tuple[Path, int]]
    ) -> None:
        """Process batch: analyze frames, summarize, notify, store.

        Args:
            camera_id: Camera entity ID
            ordered: List of (path, epoch) tuples sorted by time
        """
        # 1. Process frames
        frame_descs, recognized = await self.frame_processor.process_batch(
            camera_id, ordered
        )
        if not frame_descs:
            return

        self._last_recognized[camera_id] = recognized

        # 2. Summarize
        summary = await self.summarizer.summarize_with_timeout(camera_id, frame_descs)
        if not summary:
            return

        # 3. Notify
        mode = self.entry.options.get(CONF_VIDEO_ANALYZER_MODE, "notify_on_all")
        queue = self.queue_mgr.get_queue(camera_id)
        queue_size = queue.qsize() if queue else 0

        await self.notifier.handle_notification(
            camera_id,
            summary,
            [p for p, _ in ordered],
            mode,
            recognized,
            queue_size,
        )

        # 4. Store
        await self.storage.store_results(camera_id, [p for p, _ in ordered], summary)

        # 5. Prune
        await self.storage.prune_old_snapshots(
            camera_id, [p for p, _ in ordered], self.notifier.is_protected
        )

    async def _process_snapshot_queue(self, camera_id: str) -> None:
        """Flush any queued snapshots for a camera as one ordered batch.

        Args:
            camera_id: Camera entity ID
        """
        queue = self.queue_mgr.get_queue(camera_id)
        if not queue:
            return

        batch = QueueManager.drain(queue)
        if not batch:
            return

        # Order and process
        ordered = self.frame_processor.order_by_timestamp(batch)
        if ordered:
            await self._analyze_and_finalize(camera_id, ordered)

    async def _take_single_snapshot(
        self, camera_id: str, now: datetime
    ) -> Path | None:
        """Take a snapshot and enqueue if unique enough.

        Args:
            camera_id: Camera entity ID
            now: Current timestamp

        Returns:
            Path to snapshot if successful, None otherwise
        """
        # 1. Take snapshot
        path = await self.snapshot_mgr.take_snapshot(camera_id, now)
        if not path:
            return None

        self.metrics.increment(camera_id, "captured")

        # 2. Check uniqueness
        should_process = await self.uniqueness.should_process(camera_id, path)
        if not should_process:
            self.metrics.increment(camera_id, "skipped_duplicate")
            LOGGER.debug("[%s] Skipping near-duplicate: %s", camera_id, path.name)
            return None

        # 3. Enqueue
        queue = self.queue_mgr.get_or_create(camera_id)
        await QueueManager.put_with_backpressure(queue, path)
        self.metrics.increment(camera_id, "enqueued")
        LOGGER.debug("[%s] Enqueued: %s", camera_id, path.name)

        return path

    async def _motion_snapshot_loop(self, camera_id: str) -> None:
        """Continuous snapshot loop while motion is detected.

        Args:
            camera_id: Camera entity ID
        """
        try:
            while True:
                now = dt_util.utcnow()
                await self._take_single_snapshot(camera_id, now)
                await asyncio.sleep(VIDEO_ANALYZER_SCAN_INTERVAL)
        except asyncio.CancelledError:
            LOGGER.debug("Snapshot loop cancelled for camera: %s", camera_id)

    def _resolve_camera_from_motion(self, motion_entity_id: str) -> str | None:
        """Resolve camera entity ID from motion sensor ID.

        Args:
            motion_entity_id: Motion sensor entity ID

        Returns:
            Camera entity ID, or None if not found
        """
        # Check overrides
        overrides: dict = VIDEO_ANALYZER_MOTION_CAMERA_MAP
        camera_id = overrides.get(motion_entity_id)
        if camera_id and self.hass.states.get(camera_id):
            return camera_id

        # Infer from motion sensor name
        base = motion_entity_id.replace("binary_sensor.", "")
        base = re.sub(r"_vmd\d+.*", "", base)
        inferred_camera_id = f"camera.{base}"

        if self.hass.states.get(inferred_camera_id):
            return inferred_camera_id

        return None

    @callback
    def _handle_motion_event(self, event: Event) -> None:
        """Handle motion sensor state change.

        Args:
            event: State change event
        """
        entity_id = event.data.get("entity_id")
        if not entity_id or not entity_id.startswith("binary_sensor."):
            return

        new_state = event.data.get("new_state")
        old_state = event.data.get("old_state")
        if new_state is None:
            return

        camera_id = self._resolve_camera_from_motion(entity_id)
        if not camera_id:
            return

        # Motion ON -> start snapshot loop
        if new_state.state == "on" and (old_state is None or old_state.state != "on"):
            if camera_id not in self._active_motion_cameras:
                LOGGER.debug("Motion ON: Starting snapshot loop for %s", camera_id)
                task = self.hass.async_create_task(
                    self._motion_snapshot_loop(camera_id)
                )
                self._active_motion_cameras[camera_id] = task

        # Motion OFF -> stop loop and process queue
        elif new_state.state == "off" and camera_id in self._active_motion_cameras:
            LOGGER.debug("Motion OFF: Stopping snapshot loop for %s", camera_id)
            task = self._active_motion_cameras.pop(camera_id, None)
            if task and not task.done():
                task.cancel()
                self.hass.async_create_task(self._process_snapshot_queue(camera_id))

    @callback
    def _get_recording_cameras(self) -> list[str]:
        """Get list of cameras currently recording.

        Returns:
            List of camera entity IDs
        """
        return [
            state.entity_id
            for state in self.hass.states.async_all("camera")
            if state.state == "recording"
        ]

    async def _take_snapshots_from_recording_cameras(self, now: datetime) -> None:
        """Take snapshots from all recording cameras.

        Args:
            now: Current timestamp
        """
        for camera_id in self._get_recording_cameras():
            try:
                path = await self._take_single_snapshot(camera_id, now)
                if path:
                    LOGGER.debug(
                        "[%s] Enqueued snapshot for processing: %s", camera_id, path
                    )
            except HomeAssistantError:
                LOGGER.exception("[%s] Failed to take/enqueue snapshot.", camera_id)

    @callback
    def _handle_camera_recording_state_change(self, event: Event) -> None:
        """Handle camera recording state change.

        Args:
            event: State change event
        """
        entity_id = event.data.get("entity_id")
        if not entity_id or not entity_id.startswith("camera."):
            return

        old_state = event.data.get("old_state")
        new_state = event.data.get("new_state")
        if old_state is None or new_state is None:
            return

        # Camera stopped recording -> process queue
        if old_state.state == "recording" and new_state.state != "recording":
            self.hass.async_create_task(self._process_snapshot_queue(entity_id))

    def start(self) -> None:
        """Start the video analyzer and all sub-services."""
        if self._cancel_track is not None:
            LOGGER.warning("VideoAnalyzer already started.")
            return

        # Start face recognition service
        self.hass.async_create_task(self.face_service.start())

        # Register time-based snapshot capture for recording cameras
        self._cancel_track = async_track_time_interval(
            self.hass,
            self._take_snapshots_from_recording_cameras,
            timedelta(seconds=VIDEO_ANALYZER_SCAN_INTERVAL),
        )

        # Listen for camera recording state changes
        self._cancel_listen = self.hass.bus.async_listen(
            "state_changed", self._handle_camera_recording_state_change
        )

        # Optionally listen for motion events
        if VIDEO_ANALYZER_TRIGGER_ON_MOTION:
            self._cancel_motion_listen = self.hass.bus.async_listen(
                "state_changed", self._handle_motion_event
            )

        # Start hourly metrics reporting
        self._metrics_job_cancel = async_track_time_interval(
            self.hass,
            self.metrics.flush_and_report,
            timedelta(seconds=_METRICS_REPORT_INTERVAL_SEC),
        )

        LOGGER.info("Video analyzer started.")

    async def stop(self) -> None:
        """Stop the video analyzer and all sub-services."""
        if self._cancel_track is None:
            LOGGER.warning("VideoAnalyzer not started.")
            return

        # Stop motion camera loops
        for task in self._active_motion_cameras.values():
            task.cancel()
        self._active_motion_cameras.clear()

        # Stop queue workers
        await self.queue_mgr.stop_all(timeout=5.0)

        # Unregister event listeners
        if self._cancel_track:
            self._cancel_track()
            self._cancel_track = None

        if self._cancel_listen:
            self._cancel_listen()
            self._cancel_listen = None

        if self._cancel_motion_listen:
            self._cancel_motion_listen()
            self._cancel_motion_listen = None

        if self._metrics_job_cancel:
            self._metrics_job_cancel()
            self._metrics_job_cancel = None

        # Stop face recognition service
        await self.face_service.stop()

        LOGGER.info("Video analyzer stopped.")

    def is_running(self) -> bool:
        """Check if the video analyzer is running.

        Returns:
            True if running, False otherwise
        """
        return self._cancel_track is not None and self._cancel_listen is not None

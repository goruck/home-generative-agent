"""Video analyzer for recording and motion-triggered cameras."""

from __future__ import annotations

import asyncio
import calendar
import io
import logging
import re
from collections import deque
from datetime import datetime, timedelta
from pathlib import Path
from time import monotonic
from typing import TYPE_CHECKING
from urllib.parse import urljoin

import aiofiles
import async_timeout
import homeassistant.util.dt as dt_util
import httpx
from homeassistant.components.camera.const import DOMAIN as CAMERA_DOMAIN
from homeassistant.core import HomeAssistant, callback
from homeassistant.exceptions import HomeAssistantError
from homeassistant.helpers.event import async_track_time_interval
from langchain.schema import HumanMessage, SystemMessage
from PIL import Image

from ..agent.tools import analyze_image  # noqa: TID252
from ..const import (  # noqa: TID252
    CONF_NOTIFY_SERVICE,
    CONF_VIDEO_ANALYZER_MODE,
    REASONING_DELIMITERS,
    VIDEO_ANALYZER_FACE_CROP,
    VIDEO_ANALYZER_MOTION_CAMERA_MAP,
    VIDEO_ANALYZER_PROMPT,
    VIDEO_ANALYZER_SCAN_INTERVAL,
    VIDEO_ANALYZER_SIMILARITY_THRESHOLD,
    VIDEO_ANALYZER_SNAPSHOT_ROOT,
    VIDEO_ANALYZER_SNAPSHOTS_TO_KEEP,
    VIDEO_ANALYZER_SYSTEM_MESSAGE,
    VIDEO_ANALYZER_TIME_OFFSET,
    VIDEO_ANALYZER_TRIGGER_ON_MOTION,
)
from .utils import (
    discover_mobile_notify_service,
)

if TYPE_CHECKING:
    from homeassistant.core import Event, HomeAssistant

    from .runtime import HGAConfigEntry


LOGGER = logging.getLogger(__name__)


class VideoAnalyzer:
    """Analyze video from recording or motion-triggered cameras."""

    def __init__(self, hass: HomeAssistant, entry: HGAConfigEntry) -> None:
        """Initialize the video analyzer."""
        self.hass = hass
        self.entry = entry
        self._snapshot_queues: dict[str, asyncio.Queue[Path]] = {}
        self._retention_deques: dict[str, deque[Path]] = {}
        self._active_queue_tasks: dict[str, asyncio.Task] = {}
        self._initialized_dirs: set[str] = set()
        self._active_motion_cameras: dict[str, asyncio.Task] = {}
        self._last_recognized: dict[str, list[str]] = {}
        # Protect images referenced in notifications from immediate pruning
        self._notify_protected: dict[Path, float] = {}  # path -> expiry time

    def _protect_notify_image(self, p: Path, ttl_sec: int = 1800) -> None:
        """Mark a snapshot as protected from pruning for ttl_sec seconds."""
        self._notify_protected[p] = monotonic() + ttl_sec

    def _is_protected(self, p: Path) -> bool:
        """Return True if snapshot is still within its protection TTL."""
        now = monotonic()
        # Drop expired entries
        expired = [k for k, t in self._notify_protected.items() if t < now]
        for k in expired:
            self._notify_protected.pop(k, None)
        return self._notify_protected.get(p, 0) > now

    def _get_snapshot_queue(self, camera_id: str) -> asyncio.Queue[Path]:
        if camera_id not in self._snapshot_queues:
            queue: asyncio.Queue[Path] = asyncio.Queue()
            self._snapshot_queues[camera_id] = queue
            task = self.hass.async_create_task(self._process_snapshot_queue(camera_id))
            self._active_queue_tasks[camera_id] = task
        return self._snapshot_queues[camera_id]

    async def _send_notification(
        self, msg: str, camera_name: str, notify_img_path: Path
    ) -> None:
        # Prefer configured option; fall back to discovery
        full_service = self.entry.options.get(CONF_NOTIFY_SERVICE)
        if full_service and full_service.startswith("notify."):
            domain, service = full_service.split(".", 1)
        else:
            service = discover_mobile_notify_service(self.hass)
            LOGGER.debug("Discovered notify service: %s", service)
            if not service:
                LOGGER.warning("No notify.mobile_app_* service found.")
                return
            domain = "notify"

        await self.hass.services.async_call(
            domain,
            service,
            {
                "message": msg,
                "title": f"Camera Alert from {camera_name}!",
                "data": {"image": str(notify_img_path)},
            },
            blocking=False,
        )

    async def _generate_summary(
        self, frame_descriptions: list[dict[str, list[str]]], cam_id: str
    ) -> str:
        await asyncio.sleep(0)  # yield control
        if not frame_descriptions:
            msg = "At least one frame description required."
            raise ValueError(msg)

        ftag = "\n<frame description>\n{}\n</frame description>"
        ptag = "\n<person identity>\n{}\n</person identity>"
        prompt = " ".join(
            [VIDEO_ANALYZER_PROMPT]
            + [
                ftag.format(frame) + "".join([ptag.format(p) for p in people])
                for entry in frame_descriptions
                for frame, people in entry.items()
            ]
        )

        LOGGER.debug("Prompt: %s", prompt)

        messages = [
            SystemMessage(content=VIDEO_ANALYZER_SYSTEM_MESSAGE),
            HumanMessage(content=prompt),
        ]
        model = self.entry.runtime_data.summarization_model
        resp = await model.ainvoke(messages)

        summary = resp.content
        LOGGER.debug("Summary for %s: %s", cam_id, summary)
        first, sep, last = summary.partition(REASONING_DELIMITERS.get("end", ""))
        return (last if sep else first).strip("\n")

    async def _is_anomaly(self, camera_name: str, msg: str, first_path: str) -> bool:
        async with async_timeout.timeout(10):
            search_results = await self.entry.runtime_data.store.asearch(
                ("video_analysis", camera_name), query=msg, limit=10
            )

        # Calculate a "no newer than" time threshold from first snapshot time
        # by delaying it by the time offset.
        # Snapshot names are in the form "snapshot_20250426_002804.jpg".
        first_str = first_path.replace("snapshot_", "").replace(".jpg", "")
        first_dt = dt_util.as_local(datetime.strptime(first_str, "%Y%m%d_%H%M%S"))  # noqa: DTZ007

        # Simple anomaly detection.
        # If the first snapshot is older then the time threshold or if any search
        # result has a lower score then the similarity threshold, declare the current
        # video analysis as an anomaly.
        return first_dt < dt_util.now() - timedelta(
            minutes=VIDEO_ANALYZER_TIME_OFFSET
        ) or any(
            r.score < VIDEO_ANALYZER_SIMILARITY_THRESHOLD
            for r in search_results
            if r.score is not None
        )

    async def _prune_old_snapshots(self, camera_id: str, batch: list[Path]) -> None:
        retention = self._retention_deques.setdefault(camera_id, deque())
        for path in batch:
            retention.append(path)
            while len(retention) > VIDEO_ANALYZER_SNAPSHOTS_TO_KEEP:
                old = retention.popleft()
                # Skip if protected by a recent notification
                if self._is_protected(old):
                    retention.append(
                        old
                    )  # push it to the end once; stop pruning this round
                    break
                try:
                    await self.hass.async_add_executor_job(old.unlink)
                    LOGGER.debug("[%s] Deleted old snapshot: %s", camera_id, old)
                except OSError as err:
                    LOGGER.warning("[%s] Failed to delete %s: %s", camera_id, old, err)

    async def _recognize_faces(self, data: bytes, camera_id: str) -> list[str]:  # noqa: PLR0912, PLR0915
        """Call face API to recognize faces in the snapshot image."""
        face_mode = self.entry.runtime_data.face_mode
        if not face_mode or face_mode == "disable":
            return []

        base_url = self.entry.runtime_data.face_api_url

        # --- call face API with timeout & specific exception handling ---
        try:
            async with httpx.AsyncClient(timeout=15.0) as client:
                resp = await client.post(
                    urljoin(base_url.rstrip("/") + "/", "analyze"),
                    files={"file": ("snapshot.jpg", data, "image/jpeg")},
                )
                resp.raise_for_status()
                face_res = resp.json()
        except asyncio.CancelledError:
            raise
        except httpx.HTTPStatusError as err:
            LOGGER.warning("Face API HTTP %s: %s", err.response.status_code, err)
            return []
        except httpx.RequestError as err:
            LOGGER.warning("Face API request error: %s", err)
            return []
        except ValueError as err:  # JSON parsing
            LOGGER.warning("Face API returned invalid JSON: %s", err)
            return []

        faces = face_res.get("faces", [])
        if not faces:
            return ["Indeterminate"]

        # --- helper: offload Pillow + filesystem sync work to executor ---
        def _load_img(buf: bytes) -> Image.Image:
            return Image.open(io.BytesIO(buf)).convert("RGB")

        def _ensure_dir(p: Path) -> None:
            p.mkdir(parents=True, exist_ok=True)

        def _crop_resize_encode(
            img: Image.Image, bbox: list[int], pad: float, min_px: int
        ) -> bytes | None:
            x1, y1, x2, y2 = map(int, bbox)
            w, h = x2 - x1, y2 - y1
            if w <= 0 or h <= 0:
                return None
            dx, dy = int(w * pad), int(h * pad)
            x1, y1 = max(0, x1 - dx), max(0, y1 - dy)
            x2, y2 = min(img.width, x2 + dx), min(img.height, y2 + dy)
            if x2 <= x1 or y2 <= y1:
                return None
            crop = img.crop((x1, y1, x2, y2))
            if crop.width < min_px or crop.height < min_px:
                crop = crop.resize((min_px, min_px), resample=Image.Resampling.LANCZOS)
            out = io.BytesIO()
            crop.save(out, format="JPEG", quality=95, subsampling=0)
            return out.getvalue()

        # --- decode snapshot off the loop (Pillow is sync) ---
        try:
            img = await self.hass.async_add_executor_job(_load_img, data)
        except asyncio.CancelledError:
            raise
        except (Image.UnidentifiedImageError, OSError) as err:
            LOGGER.warning("Failed to decode snapshot for crops: %s", err)
            img = None  # still return recognition results below

        dao = self.entry.runtime_data.person_gallery
        recognized: list[str] = []

        timestamp = dt_util.now().strftime("%Y%m%d_%H%M%S")
        face_debug_root = (
            Path(VIDEO_ANALYZER_SNAPSHOT_ROOT) / "faces" / camera_id.replace(".", "_")
        )

        # ensure directory off the loop
        try:
            await self.hass.async_add_executor_job(_ensure_dir, face_debug_root)
        except OSError as err:
            LOGGER.debug("Could not ensure face debug dir: %s", err)

        insightface_bbox_length = 4
        small_crop_threshold = 128  # pixels

        for idx, face in enumerate(faces):
            emb = face["embedding"]
            # Let unexpected DAO errors propagate; don't catch blindly
            name = await dao.recognize_person(emb)
            recognized.append(name)

            # optional debug crop
            if not VIDEO_ANALYZER_FACE_CROP or not img:
                continue

            bbox = face.get("bbox")
            if bbox and len(bbox) == insightface_bbox_length:
                try:
                    # crop/resize/encode off the loop
                    jpeg_bytes = await self.hass.async_add_executor_job(
                        _crop_resize_encode, img, bbox, 0.3, small_crop_threshold
                    )
                    if not jpeg_bytes:
                        continue
                    face_file = face_debug_root / f"face_{timestamp}_{idx}_{name}.jpg"
                    # async write
                    async with aiofiles.open(face_file, "wb") as f:
                        await f.write(jpeg_bytes)
                    LOGGER.debug("Saved face crop: %s", face_file)
                except asyncio.CancelledError:
                    raise
                except OSError as err:
                    LOGGER.warning("Failed to save face crop: %s", err)

        return recognized

    def _log_snapshot_error(self, camera_id: str, path: Path, exc: Exception) -> None:
        """Log errors from snapshot processing."""
        if isinstance(exc, FileNotFoundError):
            LOGGER.warning("[%s] Snapshot not found: %s", camera_id, path)
        elif isinstance(exc, TimeoutError):
            LOGGER.warning("[%s] Image analysis timed out for %s", camera_id, path)
        elif isinstance(exc, HomeAssistantError):
            LOGGER.exception("[%s] Error analyzing %s", camera_id, path)
        else:
            LOGGER.exception("[%s] Unexpected error analyzing %s.", camera_id, path)

    def _drain_queue(self, queue: asyncio.Queue[Path]) -> list[Path]:
        """Drain all items from the asyncio queue into a list."""
        batch: list[Path] = []
        try:
            while True:
                batch.append(queue.get_nowait())
        except asyncio.QueueEmpty:
            pass
        return batch

    async def _process_snapshot(
        self, path: Path, camera_id: str, prev_text: str | None = None
    ) -> dict[str, list[str]]:
        """Process a single snapshot: recognize faces and describe the frame."""
        try:
            async with aiofiles.open(path, "rb") as file:
                data = await file.read()
            async with async_timeout.timeout(30):
                faces_in_frame = await self._recognize_faces(data, camera_id)
                frame_description: str = await analyze_image(
                    self.entry.runtime_data.vision_model,
                    data,
                    None,  # detection_keywords
                    prev_text=prev_text,  # previous description (text only)
                )
        except (FileNotFoundError, TimeoutError, HomeAssistantError) as exc:
            self._log_snapshot_error(camera_id, path, exc)
            return {}
        else:
            return {frame_description: faces_in_frame}

    async def _handle_notification(
        self, camera_id: str, msg: str, batch: list[Path]
    ) -> None:
        """Decide whether to notify and send if needed."""
        camera_name = camera_id.split(".")[-1]
        chosen = batch[len(batch) // 2]

        # If backlog is hot, let filesystem settle before notifying
        queue = self._snapshot_queues.get(camera_id)
        backlog_limit = 10
        if queue and queue.qsize() > backlog_limit:
            await asyncio.sleep(0.5)

        # If the Home Assistant media_dirs option is set in configuration.yaml,
        # ensure that /media is added as an option for '/media/local' to work.
        # (If its not set then the default is /media.)
        # E.g:
        # homeassistant:
        #   media_dirs:
        #       local: /media # restore default 'local'
        #       foo: /media/snapshots/foo  # optional extra source
        media_dir = "/media/local"
        notify_img = Path(media_dir) / Path(*chosen.parts[-3:])

        mode = self.entry.options.get(CONF_VIDEO_ANALYZER_MODE)
        if mode == "notify_on_anomaly":
            first_snapshot = batch[0].parts[-1]
            LOGGER.debug("[%s] First snapshot: %s", camera_id, first_snapshot)
            if await self._is_anomaly(camera_name, msg, first_snapshot):
                LOGGER.debug("[%s] Video is an anomaly!", camera_id)
                # Protect the chosen file from pruning for 30 minutes
                self._protect_notify_image(chosen, ttl_sec=1800)
                await self._send_notification(msg, camera_name, notify_img)
        else:
            self._protect_notify_image(chosen, ttl_sec=1800)
            await self._send_notification(msg, camera_name, notify_img)

    async def _store_results(self, camera_id: str, batch: list[Path], msg: str) -> None:
        """Store the analysis results in the vector DB."""
        camera_name = camera_id.split(".")[-1]
        async with async_timeout.timeout(10):
            await self.entry.runtime_data.store.aput(
                namespace=("video_analysis", camera_name),
                key=batch[0].name,
                value={"content": msg, "snapshots": [str(p) for p in batch]},
            )

    async def _process_snapshot_queue(self, camera_id: str) -> None:
        """Process snapshots from the queue in batches."""
        queue: asyncio.Queue[Path] | None = self._snapshot_queues.get(camera_id)
        if not queue:
            return

        batch = self._drain_queue(queue)
        if not batch:
            return

        def _epoch_from_path(path: Path) -> int:
            s = path.stem.removeprefix("snapshot_")  # "YYYYMMDD_HHMMSS"
            y, mo, d = int(s[0:4]), int(s[4:6]), int(s[6:8])
            hh, mm, ss = int(s[9:11]), int(s[11:13]), int(s[13:15])
            # UTC epoch seconds without creating a datetime object
            return calendar.timegm((y, mo, d, hh, mm, ss, 0, 0, 0))

        # Sort by actual timestamp (handles out-of-order filenames)
        ordered = sorted(
            ((path, _epoch_from_path(path)) for path in batch), key=lambda x: x[1]
        )
        if not ordered:
            return
        t0 = ordered[0][1]

        # Collect results SEQUENTIALLY so we can thread prev_text
        frame_descriptions: list[dict[str, list[str]]] = []
        prev_text: str | None = None

        for path, ts in ordered:
            fd = await self._process_snapshot(path, camera_id, prev_text=prev_text)
            if not fd:
                continue

            # Extract description and faces
            frame_description, faces = next(iter(fd.items()))

            # Save time-prefixed description for downstream summarizer
            timed_desc = f"t+{ts - t0}s. {frame_description}"
            frame_descriptions.append({timed_desc: faces})

            # Update prev_text with the raw description (NO time prefix)
            prev_text = frame_description

        if not frame_descriptions:
            return

        # Deduplicate recognized people and cache last seen.
        recognized_people = [
            p for d in frame_descriptions for v in d.values() for p in v if p != "None"
        ]
        self._last_recognized[camera_id] = sorted(set(recognized_people))

        # Generate summary
        async with async_timeout.timeout(60):
            msg = await self._generate_summary(frame_descriptions, camera_id)

        LOGGER.info("[%s] Video analysis: %s", camera_id, msg)

        await self._handle_notification(camera_id, msg, batch)
        await self._store_results(camera_id, batch, msg)
        await self._prune_old_snapshots(camera_id, batch)

    def _resolve_camera_from_motion(self, motion_entity_id: str) -> str | None:
        """Resolve a camera entity ID from a motion sensor ID."""
        overrides: dict = VIDEO_ANALYZER_MOTION_CAMERA_MAP
        camera_id = overrides.get(motion_entity_id)
        if camera_id and self.hass.states.get(camera_id):
            return camera_id

        base = motion_entity_id.replace("binary_sensor.", "")
        base = re.sub(r"_vmd\d+.*", "", base)
        inferred_camera_id = f"camera.{base}"
        if self.hass.states.get(inferred_camera_id):
            return inferred_camera_id

        return None

    async def _get_snapshot_dir(self, camera_id: str) -> Path:
        if camera_id not in self._initialized_dirs:
            cam_dir = Path(VIDEO_ANALYZER_SNAPSHOT_ROOT) / camera_id.replace(".", "_")
            cam_dir.mkdir(parents=True, exist_ok=True)
            self._initialized_dirs.add(camera_id)
            dir_not_empty = await self.hass.async_add_executor_job(
                lambda: any(cam_dir.iterdir()),
            )
            if dir_not_empty:
                msg = "[{id}] Folder not empty. Existing snapshots will not be pruned."
                LOGGER.info(msg.format(id=camera_id))
        return Path(VIDEO_ANALYZER_SNAPSHOT_ROOT) / camera_id.replace(".", "_")

    async def _take_single_snapshot(self, camera_id: str, now: datetime) -> Path | None:
        snapshot_dir = await self._get_snapshot_dir(camera_id)
        timestamp = dt_util.as_local(now).strftime("%Y%m%d_%H%M%S")
        path = snapshot_dir / f"snapshot_{timestamp}.jpg"

        try:
            start_time = dt_util.utcnow()
            LOGGER.debug("[%s] Initiating snapshot at %s", camera_id, start_time)

            await self.hass.services.async_call(
                CAMERA_DOMAIN,
                "snapshot",
                {"entity_id": camera_id, "filename": str(path)},
                blocking=False,
            )
            LOGGER.debug("[%s] Snapshot service call completed.", camera_id)

            for i in range(50):
                exists = await self.hass.async_add_executor_job(path.exists)
                if exists:
                    LOGGER.debug(
                        "[%s] Snapshot appeared after %.2f s.",
                        camera_id,
                        (dt_util.utcnow() - start_time).total_seconds(),
                    )
                    break
                await asyncio.sleep(0.2)
                LOGGER.debug(
                    "[%s] Waiting for snapshot to appear... attempt %d",
                    camera_id,
                    i + 1,
                )
            else:
                LOGGER.warning(
                    "[%s] Snapshot failed to appear on disk after waiting: %s",
                    camera_id,
                    path,
                )
                return None

            queue = self._get_snapshot_queue(camera_id)
            await queue.put(path)
            LOGGER.debug("[%s] Enqueued snapshot %s", camera_id, path)
        except HomeAssistantError as err:
            LOGGER.warning("Snapshot failed for %s: %s", camera_id, err)
        else:
            return path
        return None

    async def _motion_snapshot_loop(self, camera_id: str) -> None:
        try:
            while True:
                now = dt_util.utcnow()
                await self._take_single_snapshot(camera_id, now)
                await asyncio.sleep(VIDEO_ANALYZER_SCAN_INTERVAL)
        except asyncio.CancelledError:
            LOGGER.debug("Snapshot loop cancelled for camera: %s", camera_id)

    @callback
    def _handle_motion_event(self, event: Event) -> None:
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

        if new_state.state == "on" and (old_state is None or old_state.state != "on"):
            if camera_id not in self._active_motion_cameras:
                LOGGER.debug("Motion ON: Starting snapshot loop for %s", camera_id)
                task = self.hass.async_create_task(
                    self._motion_snapshot_loop(camera_id),
                )
                self._active_motion_cameras[camera_id] = task

        elif new_state.state == "off" and camera_id in self._active_motion_cameras:
            LOGGER.debug("Motion OFF: Stopping snapshot loop for %s", camera_id)
            task = self._active_motion_cameras.pop(camera_id, None)
            if task and not task.done():
                task.cancel()
                self.hass.async_create_task(self._process_snapshot_queue(camera_id))

    @callback
    def _get_recording_cameras(self) -> list[str]:
        return [
            state.entity_id
            for state in self.hass.states.async_all("camera")
            if state.state == "recording"
        ]

    async def _take_snapshots_from_recording_cameras(self, now: datetime) -> None:
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
        entity_id = event.data.get("entity_id")
        if not entity_id or not entity_id.startswith("camera."):
            return

        old_state = event.data.get("old_state")
        new_state = event.data.get("new_state")
        if old_state is None or new_state is None:
            return

        if old_state.state == "recording" and new_state.state != "recording":
            self.hass.async_create_task(self._process_snapshot_queue(entity_id))

    def start(self) -> None:
        """Start the video analyzer."""
        if hasattr(self, "_cancel_track"):
            LOGGER.warning("VideoAnalyzer already started.")
            return

        self._cancel_track = async_track_time_interval(
            self.hass,
            self._take_snapshots_from_recording_cameras,
            timedelta(seconds=VIDEO_ANALYZER_SCAN_INTERVAL),
        )
        self._cancel_listen = self.hass.bus.async_listen(
            "state_changed", self._handle_camera_recording_state_change
        )
        if VIDEO_ANALYZER_TRIGGER_ON_MOTION:
            self._cancel_motion_listen = self.hass.bus.async_listen(
                "state_changed", self._handle_motion_event
            )
        LOGGER.info("Video analyzer started.")

    async def stop(self) -> None:
        """Stop the video analyzer."""
        if not hasattr(self, "_cancel_track"):
            LOGGER.warning("VideoAnalyzer not started.")
            return

        tasks_to_await: list[asyncio.Task] = []

        for task in self._active_motion_cameras.values():
            task.cancel()
            tasks_to_await.append(task)
        self._active_motion_cameras.clear()

        for task in self._active_queue_tasks.values():
            task.cancel()
            tasks_to_await.append(task)
        self._active_queue_tasks.clear()

        try:
            self._cancel_track()
        except HomeAssistantError:
            LOGGER.warning("Error unsubscribing time interval listener", exc_info=True)

        try:
            self._cancel_listen()
        except HomeAssistantError:
            LOGGER.warning(
                "Error unsubscribing recording state listener", exc_info=True
            )

        if hasattr(self, "_cancel_motion_listen"):
            try:
                self._cancel_motion_listen()
            except HomeAssistantError:
                LOGGER.warning(
                    "Error unsubscribing motion event listener", exc_info=True
                )

        if tasks_to_await:
            _, pending = await asyncio.wait(tasks_to_await, timeout=5)
            for task in pending:
                LOGGER.warning("Task did not cancel in time: %s", task)

        LOGGER.info("Video analyzer stopped.")

    def is_running(self) -> bool:
        """Check if the video analyzer is running."""
        return hasattr(self, "_cancel_track") and hasattr(self, "_cancel_listen")

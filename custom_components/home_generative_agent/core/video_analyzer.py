"""Video analyzer for recording and motion-triggered cameras."""

from __future__ import annotations

import asyncio
import contextlib
import logging
import re
import statistics
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from time import monotonic
from typing import TYPE_CHECKING, Final
from urllib.parse import urljoin

import aiofiles
import async_timeout
import homeassistant.util.dt as dt_util
import httpx
from homeassistant.components.camera.const import DOMAIN as CAMERA_DOMAIN
from homeassistant.core import HomeAssistant, callback
from homeassistant.exceptions import HomeAssistantError
from homeassistant.helpers.event import async_track_time_interval
from langchain_core.messages import HumanMessage, SystemMessage
from PIL import Image

from ..agent.tools import analyze_image  # noqa: TID252
from ..const import (  # noqa: TID252
    CONF_NOTIFY_SERVICE,
    CONF_VIDEO_ANALYZER_MODE,
    SIGNAL_HGA_NEW_LATEST,
    SIGNAL_HGA_RECOGNIZED,
    VIDEO_ANALYZER_FACE_CROP,
    VIDEO_ANALYZER_LATEST_NAME,
    VIDEO_ANALYZER_LATEST_SUBFOLDER,
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
    dispatch_on_loop,
    extract_final,
)
from .video_helpers import (
    crop_resize_encode_jpeg,
    dedupe_desc,
    dhash_bytes,
    ensure_dir,
    epoch_from_path,
    hamming64,
    latest_target,
    load_image_rgb,
    order_batch,
    publish_latest_atomic,
    put_with_backpressure,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable

    from homeassistant.core import Event, HomeAssistant

    from .runtime import HGAConfigEntry

LOGGER = logging.getLogger(__name__)

# --- Video analyzer tuning constants ---
_MAX_BATCH: Final[int] = 5  # frames per batch
_QUEUE_MAXSIZE: Final[int] = 50  # per-camera backlog cap
_FRAME_DEADLINE_SEC: Final[int] = 600  # skip frames older than this
_SUMMARY_TIMEOUT_SEC: Final[int] = 60  # was 35
_FACE_TIMEOUT_SEC: Final[int] = 10  # was 10 (keep)
_VISION_TIMEOUT_SEC: Final[int] = 90  # was 30
_GLOBAL_VISION_CONCURRENCY: Final[int] = 3  # tune per hardware

# --- Uniqueness gate tuning ---
_UNIQUENESS_ENABLED: Final[bool] = False
_UNIQUENESS_HAMMING_MAX: Final[int] = 4  # <= this => "too similar" (tune 4-10)
_UNIQUENESS_HISTORY: Final[int] = 2  # compare against last N accepted hashes
_UNIQUENESS_HEARTBEAT_SEC: Final[int] = 10  # always allow a frame at least this often

# --- Metrics reporting ---
_METRICS_REPORT_INTERVAL_SEC: Final[int] = 3600  # once per hour
_METRICS_LAT_HISTORY: Final[int] = 512  # keep up to 512 lat samples per camera

_global_vision_sem = asyncio.Semaphore(_GLOBAL_VISION_CONCURRENCY)


@dataclass
class _Metrics:
    captured: int = 0
    enqueued: int = 0
    skipped_duplicate: int = 0
    dropped_stale: int = 0
    analyzed: int = 0
    timeouts: int = 0
    # PEP 585: deque is subscriptable; keep in a field to avoid shared default
    lat_ms: deque[float] = field(
        default_factory=lambda: deque(maxlen=_METRICS_LAT_HISTORY)
    )


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
        self._httpx_client: httpx.AsyncClient | None = None
        self._last_hashes: dict[
            str, deque[int]
        ] = {}  # camera_id -> deque of recent dHashes
        self._last_unique_ts: dict[
            str, float
        ] = {}  # camera_id -> monotonic() of last accepted
        # Per-camera counters and latency samples
        self._metrics: dict[str, _Metrics] = defaultdict(_Metrics)
        self._metrics_job_cancel: Callable[[], None] | None = (
            None  # hourly report handle
        )

    def _pctl(self, values: Iterable[float], q: float) -> float:
        """Nearest-rank percentile for a finite iterable."""
        xs = list(values)
        if not xs:
            return 0.0
        xs.sort()
        if len(xs) == 1:
            return float(xs[0])
        # clamp rank to [0, len(xs)-1]
        k = max(0, min(len(xs) - 1, round((q / 100.0) * (len(xs) - 1))))
        return float(xs[k])

    def _m_inc(self, camera_id: str, key: str, n: int = 1) -> None:
        """Increment a metrics counter by key."""
        m = self._metrics[camera_id]
        if key == "captured":
            m.captured += n
        elif key == "enqueued":
            m.enqueued += n
        elif key == "skipped_duplicate":
            m.skipped_duplicate += n
        elif key == "dropped_stale":
            m.dropped_stale += n
        elif key == "analyzed":
            m.analyzed += n
        elif key == "timeouts":
            m.timeouts += n
        else:
            # ignore unknown keys silently to avoid noisy logs in prod
            return

    def _m_add_latency(self, camera_id: str, ms: float) -> None:
        """Record a single latency sample in milliseconds."""
        self._metrics[camera_id].lat_ms.append(float(ms))

    async def _metrics_flush_report(self, _now: datetime) -> None:
        """Aggregate and log per-camera metrics, then reset counters and samples."""
        for cam, m in self._metrics.items():
            lat_list = list(m.lat_ms)
            avg_ms = statistics.fmean(lat_list) if lat_list else 0.0
            p95_ms = self._pctl(lat_list, 95.0) if lat_list else 0.0

            msg = (
                "[%s] Metrics (last interval): "
                "captured=%d enqueued=%d skipped_duplicate=%d "
                "dropped_stale=%d analyzed=%d timeouts=%d "
                "avg_latency_ms=%.1f p95_latency_ms=%.1f"
            )
            LOGGER.info(
                msg,
                cam,
                m.captured,
                m.enqueued,
                m.skipped_duplicate,
                m.dropped_stale,
                m.analyzed,
                m.timeouts,
                avg_ms,
                p95_ms,
            )

            # reset counters and samples
            m.captured = 0
            m.enqueued = 0
            m.skipped_duplicate = 0
            m.dropped_stale = 0
            m.analyzed = 0
            m.timeouts = 0
            m.lat_ms.clear()

    def protect_notify_image(self, p: Path, ttl_sec: int = 1800) -> None:
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
            queue: asyncio.Queue[Path] = asyncio.Queue(maxsize=_QUEUE_MAXSIZE)
            self._snapshot_queues[camera_id] = queue
            task = self.hass.async_create_task(self._snapshot_worker(camera_id))
            self._active_queue_tasks[camera_id] = task
        return self._snapshot_queues[camera_id]

    async def _get_batch(
        self,
        queue: asyncio.Queue[Path],
        camera_id: str,
        *,
        max_batch: int = _MAX_BATCH,
    ) -> list[Path]:
        first: Path = await queue.get()
        batch: list[Path] = [first]

        n: int = max(max_batch - 1, 0)
        with contextlib.suppress(asyncio.QueueEmpty):
            batch.extend(queue.get_nowait() for _ in range(n))

        LOGGER.debug(
            "[%s] Start t=%s batch=%d qsize=%d",
            camera_id,
            dt_util.utcnow().isoformat(),
            len(batch),
            queue.qsize(),
        )
        return batch

    async def _process_batch(
        self,
        camera_id: str,
        ordered: list[tuple[Path, int]],
    ) -> tuple[list[dict[str, list[str]]], list[str]]:
        if not ordered:
            return [], []

        t0: int = ordered[0][1]
        frame_descriptions: list[dict[str, list[str]]] = []
        prev_text: str | None = None

        for path, ts in ordered:
            fd = await self._process_snapshot(path, camera_id, prev_text=prev_text)
            if not fd:
                continue
            frame_desc, faces = next(iter(fd.items()))
            frame_descriptions.append({f"t+{ts - t0}s. {frame_desc}": faces})
            prev_text = frame_desc

        # dedupe near-identical, then cap to last 8
        frame_descriptions = dedupe_desc(frame_descriptions)[-8:]

        recognized: list[str] = sorted(
            {
                p
                for d in frame_descriptions
                for v in d.values()
                for p in v
                if p != "None"
            }
        )
        return frame_descriptions, recognized

    async def _summarize(
        self, camera_id: str, frame_descriptions: list[dict[str, list[str]]]
    ) -> str | None:
        if not frame_descriptions:
            return None
        try:
            async with async_timeout.timeout(_SUMMARY_TIMEOUT_SEC):
                msg: str = await self._generate_summary(frame_descriptions)
        except TimeoutError as exc:
            LOGGER.warning("[%s] Summary timed out: %s", camera_id, exc)
            return None
        else:
            LOGGER.debug("[%s] Video analysis: %s", camera_id, msg)
            return msg

    async def _finalize(self, camera_id: str, batch: list[Path], msg: str) -> None:
        await self._handle_notification(camera_id, msg, batch)
        await self._store_results(camera_id, batch, msg)
        await self._prune_old_snapshots(camera_id, batch)

    async def _analyze_and_finalize(
        self, camera_id: str, ordered: list[tuple[Path, int]]
    ) -> None:
        frame_descs, recognized = await self._process_batch(camera_id, ordered)
        if not frame_descs:
            return
        self._last_recognized[camera_id] = recognized
        msg = await self._summarize(camera_id, frame_descs)
        if not msg:
            return
        await self._finalize(camera_id, [p for p, _ in ordered], msg)

    async def _snapshot_worker(self, camera_id: str) -> None:
        """Consume and process snapshots for one camera."""
        queue = self._snapshot_queues[camera_id]
        LOGGER.debug("[%s] Worker started", camera_id)
        try:
            while True:
                batch = await self._get_batch(queue, camera_id)
                ordered = order_batch(batch)

                if ordered:
                    first_epoch = ordered[0][1]
                    age = int(dt_util.utcnow().timestamp() - first_epoch)
                    LOGGER.debug(
                        "[%s] Dequeue age=%ds qsize=%d", camera_id, age, queue.qsize()
                    )

                await self._analyze_and_finalize(camera_id, ordered)
        except asyncio.CancelledError:
            LOGGER.debug("[%s] Worker cancelled", camera_id)
            raise
        except Exception:
            LOGGER.exception("[%s] Worker crashed", camera_id)

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
        self, frame_descriptions: list[dict[str, list[str]]]
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
        summary = await model.ainvoke(messages)
        LOGGER.debug("Raw video analyzer summary: %s", summary)

        text = extract_final(getattr(summary, "content", "") or "")
        if text:
            return text

        msg = "Empty model content after parsing."
        raise ValueError(msg)

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

                # Do not delete published "latest" assets
                if (
                    old.name == VIDEO_ANALYZER_LATEST_NAME
                    or old.parent.name == VIDEO_ANALYZER_LATEST_SUBFOLDER
                ):
                    retention.append(old)
                    break

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

    async def recognize_faces(self, data: bytes, camera_id: str) -> list[str]:  # noqa: PLR0912, PLR0915
        """Call face API to recognize faces in the snapshot image."""
        face_mode = self.entry.runtime_data.face_mode
        if not face_mode or face_mode == "disable":
            return []

        base_url = self.entry.runtime_data.face_api_url
        client = self._httpx_client
        if client is None:
            # fallback if start() wasn't called yet
            client = httpx.AsyncClient(timeout=_FACE_TIMEOUT_SEC)

        # Call face API with timeout & specific exception handling
        try:
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

        # --- decode snapshot off the loop (Pillow is sync) ---
        try:
            img = await self.hass.async_add_executor_job(load_image_rgb, data)
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
            await self.hass.async_add_executor_job(ensure_dir, face_debug_root)
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
                        crop_resize_encode_jpeg, img, bbox, 0.3, small_crop_threshold
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
        # freshness gate
        try:
            epoch = epoch_from_path(path)
        except (OSError, FileNotFoundError, ValueError):
            epoch = 0
        age = dt_util.utcnow().timestamp() - float(epoch)
        if age > _FRAME_DEADLINE_SEC:
            self._m_inc(camera_id, "dropped_stale")
            LOGGER.debug(
                "[%s] Skipping stale snapshot (%ds): %s", camera_id, int(age), path
            )
            return {}

        start_ns = monotonic()
        try:
            async with aiofiles.open(path, "rb") as file:
                data = await file.read()

            # Face recognition: short timeout
            try:
                async with async_timeout.timeout(_FACE_TIMEOUT_SEC):
                    faces_in_frame = await self.recognize_faces(data, camera_id)
            except TimeoutError:
                LOGGER.warning(
                    "[%s] Face recognition timed out for %s", camera_id, path
                )
                faces_in_frame = []

            # Vision: global concurrency + short timeout
            try:
                async with (
                    _global_vision_sem,
                    async_timeout.timeout(_VISION_TIMEOUT_SEC),
                ):
                    frame_description = await analyze_image(
                        self.entry.runtime_data.vision_model,
                        data,
                        None,
                        prev_text=prev_text,
                    )
            except TimeoutError as exc:
                self._m_inc(camera_id, "timeouts")
                self._log_snapshot_error(camera_id, path, exc)
                return {}
        except (FileNotFoundError, HomeAssistantError) as exc:
            self._log_snapshot_error(camera_id, path, exc)
            return {}
        else:
            dur_ms = (monotonic() - start_ns) * 1000.0
            self._m_add_latency(camera_id, dur_ms)
            self._m_inc(camera_id, "analyzed")
            return {frame_description: faces_in_frame}

    async def _handle_notification(
        self, camera_id: str, msg: str, batch: list[Path]
    ) -> None:
        """Decide whether to notify and send if needed."""
        camera_name = camera_id.split(".")[-1]
        chosen = batch[len(batch) // 2]

        dst = latest_target(Path(VIDEO_ANALYZER_SNAPSHOT_ROOT), camera_id)
        await publish_latest_atomic(self.hass, chosen, dst)

        # Fire bus event when a new latest is published
        self.hass.bus.async_fire(
            "hga_last_event_frame",
            {
                "camera_id": camera_id,
                "summary": msg,
                "path": str(chosen),
                "latest": str(dst),
            },
        )

        # Notify ImageEntity listeners (latest frame)
        dispatch_on_loop(
            self.hass,
            SIGNAL_HGA_NEW_LATEST,
            camera_id,
            str(dst),
            msg,
            list(self._last_recognized.get(camera_id, [])),
            dt_util.utcnow().isoformat(),
        )

        # Notify Sensor listeners (recognized names + summary + path)
        dispatch_on_loop(
            self.hass,
            SIGNAL_HGA_RECOGNIZED,
            camera_id,
            list(self._last_recognized.get(camera_id, [])),
            msg,
            dt_util.utcnow().isoformat(),
            str(dst),
        )

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
                self.protect_notify_image(chosen, ttl_sec=1800)
                await self._send_notification(msg, camera_name, notify_img)
        else:
            self.protect_notify_image(chosen, ttl_sec=1800)
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
        """Flush any queued snapshots for a camera as one ordered batch."""
        queue: asyncio.Queue[Path] | None = self._snapshot_queues.get(camera_id)
        if not queue:
            return

        batch = self._drain_queue(queue)
        if not batch:
            return

        # Sort by actual timestamp (handles out-of-order filenames)
        ordered: list[tuple[Path, int]] = sorted(
            ((path, epoch_from_path(path)) for path in batch),
            key=lambda x: x[1],
        )
        if not ordered:
            return

        # Reuse the same path as the live worker (process → summarize → finalize)
        await self._analyze_and_finalize(camera_id, ordered)

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

    async def _is_unique_enough(self, camera_id: str, path: Path) -> bool:
        """
        Gate snapshots by perceptual uniqueness + heartbeat.

        Returns True to enqueue, False to skip.
        """
        if not _UNIQUENESS_ENABLED:
            return True

        # Heartbeat: always allow at least one frame every N seconds
        now = monotonic()
        last_ok = self._last_unique_ts.get(camera_id, 0.0)
        heartbeat_due = now - last_ok >= _UNIQUENESS_HEARTBEAT_SEC

        # Load bytes async, compute hash off the loop if needed
        try:
            async with aiofiles.open(path, "rb") as f:
                data = await f.read()
        except (FileNotFoundError, OSError):
            return True  # if we can't read, don't block processing

        # Hash compute is fast; do it inline (PIL decode already occurs)
        try:
            dh = dhash_bytes(data)
        except (Image.UnidentifiedImageError, OSError, ValueError):
            return True  # be permissive on failures

        hist = self._last_hashes.setdefault(
            camera_id, deque(maxlen=_UNIQUENESS_HISTORY)
        )
        # If we have no history, accept and seed
        if not hist:
            hist.append(dh)
            self._last_unique_ts[camera_id] = now
            return True

        # Compute min Hamming distance to recent accepted frames
        min_h = min(hamming64(dh, prev) for prev in hist)

        if min_h <= _UNIQUENESS_HAMMING_MAX and not heartbeat_due:
            # Too similar and no heartbeat due → skip
            return False

        # Accept: update history and last-unique time
        hist.append(dh)
        self._last_unique_ts[camera_id] = now
        return True

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

            self._m_inc(camera_id, "captured")
            queue = self._get_snapshot_queue(camera_id)

            # --- Uniqueness gate: skip near-duplicate frames unless heartbeat due ---
            try:
                should_enqueue = await self._is_unique_enough(camera_id, path)
            except (OSError, FileNotFoundError, ValueError):
                should_enqueue = True  # fail-open

            if not should_enqueue:
                self._m_inc(camera_id, "skipped_duplicate")
                LOGGER.debug(
                    "[%s] Skipping enqueue (near-duplicate): %s", camera_id, path
                )
                return None

            await put_with_backpressure(queue, path)
            self._m_inc(camera_id, "enqueued")
            LOGGER.debug(
                "[%s] Enqueue t=%s %s", camera_id, dt_util.utcnow().isoformat(), path
            )
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

        # Create a reusable httpx client for face API
        if self._httpx_client is None:
            self._httpx_client = httpx.AsyncClient(timeout=_FACE_TIMEOUT_SEC)

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

        # hourly metrics reporting
        self._metrics_job_cancel = async_track_time_interval(
            self.hass,
            self._metrics_flush_report,
            timedelta(seconds=_METRICS_REPORT_INTERVAL_SEC),
        )

        LOGGER.info("Video analyzer started.")

    async def stop(self) -> None:  # noqa: PLR0912
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

        # close reusable httpx client
        if self._httpx_client is not None:
            try:
                await self._httpx_client.aclose()
            finally:
                self._httpx_client = None

        # cancel hourly metrics reporting
        if self._metrics_job_cancel is not None:
            try:
                self._metrics_job_cancel()
            except HomeAssistantError:
                LOGGER.warning("Error unsubscribing metrics reporter", exc_info=True)
            finally:
                self._metrics_job_cancel = None

        LOGGER.info("Video analyzer stopped.")

    def is_running(self) -> bool:
        """Check if the video analyzer is running."""
        return hasattr(self, "_cancel_track") and hasattr(self, "_cancel_listen")

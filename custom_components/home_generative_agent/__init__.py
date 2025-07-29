"""Home Generative Agent Initialization."""

from __future__ import annotations

import asyncio
import logging
import re
from asyncio import QueueEmpty
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING

import aiofiles
import async_timeout
import homeassistant.util.dt as dt_util
from homeassistant.components.camera.const import DOMAIN as CAMERA_DOMAIN
from homeassistant.const import (
    CONF_API_KEY,
    Platform,
)
from homeassistant.core import HomeAssistant, callback
from homeassistant.exceptions import HomeAssistantError
from homeassistant.helpers.event import async_track_time_interval
from homeassistant.helpers.httpx_client import get_async_client
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import ConfigurableField
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.store.postgres import AsyncPostgresStore
from psycopg.rows import DictRow, dict_row
from psycopg_pool import AsyncConnectionPool, PoolTimeout

from .const import (
    CONF_SUMMARIZATION_MODEL,
    CONF_SUMMARIZATION_MODEL_TEMPERATURE,
    CONF_SUMMARIZATION_MODEL_TOP_P,
    CONF_VIDEO_ANALYZER_MODE,
    DB_URI,
    EDGE_CHAT_MODEL_URL,
    EMBEDDING_MODEL_CTX,
    EMBEDDING_MODEL_DIMS,
    EMBEDDING_MODEL_URL,
    RECOMMENDED_EDGE_CHAT_MODEL,
    RECOMMENDED_EMBEDDING_MODEL,
    RECOMMENDED_SUMMARIZATION_MODEL,
    RECOMMENDED_SUMMARIZATION_MODEL_TEMPERATURE,
    RECOMMENDED_SUMMARIZATION_MODEL_TOP_P,
    RECOMMENDED_VLM,
    SUMMARIZATION_MODEL_CTX,
    SUMMARIZATION_MODEL_PREDICT,
    SUMMARIZATION_MODEL_REASONING_DELIMITER,
    SUMMARIZATION_MODEL_URL,
    VIDEO_ANALYZER_MOBILE_APP,
    VIDEO_ANALYZER_MOTION_CAMERA_MAP,
    VIDEO_ANALYZER_PROMPT,
    VIDEO_ANALYZER_SCAN_INTERVAL,
    VIDEO_ANALYZER_SIMILARITY_THRESHOLD,
    VIDEO_ANALYZER_SNAPSHOT_ROOT,
    VIDEO_ANALYZER_SNAPSHOTS_TO_KEEP,
    VIDEO_ANALYZER_SYSTEM_MESSAGE,
    VIDEO_ANALYZER_TIME_OFFSET,
    VIDEO_ANALYZER_TRIGGER_ON_MOTION,
    VLM_URL,
)
from .tools import analyze_image

if TYPE_CHECKING:
    from collections.abc import Sequence

    from homeassistant.config_entries import ConfigEntry
    from homeassistant.core import Event, HomeAssistant
    from langchain_core.language_models import BaseMessage, LanguageModelInput
    from langchain_core.runnables import RunnableConfig
    from langchain_core.runnables.base import RunnableSerializable
    from langgraph.store.postgres.base import PostgresIndexConfig
    from psycopg import AsyncConnection

LOGGER = logging.getLogger(__name__)

PLATFORMS = (Platform.CONVERSATION,)

type HGAConfigEntry = ConfigEntry[HGAData]

@dataclass
class HGAData:
    """Run-time data for Home Generative Agent."""

    chat_model: RunnableSerializable[LanguageModelInput, BaseMessage]
    edge_chat_model: RunnableSerializable[LanguageModelInput, BaseMessage]
    vision_model: RunnableSerializable[LanguageModelInput, BaseMessage]
    summarization_model: RunnableSerializable[LanguageModelInput, BaseMessage]
    pool: AsyncConnectionPool[AsyncConnection[DictRow]]
    store: AsyncPostgresStore
    video_analyzer: VideoAnalyzer
    checkpointer: AsyncPostgresSaver
    embedding_model: OllamaEmbeddings

class VideoAnalyzer:
    """Analyze video from recording or motion-triggered cameras."""

    def __init__(self, hass: HomeAssistant, entry: HGAConfigEntry) -> None:
        """Init analyzer."""
        self.hass = hass
        self.entry = entry
        self._snapshot_queues: dict[str, asyncio.Queue[Path]] = {}
        self._retention_deques: dict[str, deque[Path]] = {}
        self._active_queue_tasks: dict[str, asyncio.Task] = {}
        self._initialized_dirs: set[str] = set()
        self._active_motion_cameras: dict[str, asyncio.Task] = {}
        self._sum_model_cfg: RunnableConfig = {
            "configurable": {
                "model": self.entry.options.get(
                    CONF_SUMMARIZATION_MODEL, RECOMMENDED_SUMMARIZATION_MODEL
                ),
                "temperature": self.entry.options.get(
                    CONF_SUMMARIZATION_MODEL_TEMPERATURE,
                    RECOMMENDED_SUMMARIZATION_MODEL_TEMPERATURE,
                ),
                "top_p": self.entry.options.get(
                    CONF_SUMMARIZATION_MODEL_TOP_P,
                    RECOMMENDED_SUMMARIZATION_MODEL_TOP_P,
                ),
                "num_predict": SUMMARIZATION_MODEL_PREDICT,
                "num_ctx": SUMMARIZATION_MODEL_CTX,
            }
        }

    def _get_snapshot_queue(self, camera_id: str) -> asyncio.Queue[Path]:
        """Lazily create a Queue and start its processing task."""
        if camera_id not in self._snapshot_queues:
            queue: asyncio.Queue[Path] = asyncio.Queue()
            self._snapshot_queues[camera_id] = queue
            # Kick off the consumer.
            task = self.hass.async_create_task(
                self._process_snapshot_queue(camera_id)
            )
            self._active_queue_tasks[camera_id] = task
        return self._snapshot_queues[camera_id]

    async def _send_notification(
            self,
            msg: str,
            camera_name: str,
            notify_img_path: Path
        ) -> None:
        """Send notification to the mobile app."""
        await self.hass.services.async_call(
            "notify",
            VIDEO_ANALYZER_MOBILE_APP,
            {
                "message": msg,
                "title": f"Camera Alert from {camera_name}!",
                "data": {
                    "image": str(notify_img_path)
                }
            },
            blocking=False
        )

    async def _generate_summary(self, frames: list[str], cam_id: str) -> str:
        """Generate a summarized analysis from frame descriptions."""
        await asyncio.sleep(0) # avoid blocking the event loop

        if not frames:
            msg = "At least one frame description required."
            raise ValueError(msg)

        if len(frames) == 1:
            return frames[0]

        tag = "\n<frame description>\n{}\n</frame description>"
        prompt = " ".join([
            VIDEO_ANALYZER_PROMPT
        ] + [tag.format(i) for i in frames])

        LOGGER.debug("Prompt: %s", prompt)

        messages = [
            SystemMessage(content=VIDEO_ANALYZER_SYSTEM_MESSAGE),
            HumanMessage(content=prompt),
        ]
        model = self.entry.runtime_data.summarization_model.with_config(
            config=self._sum_model_cfg
        )
        resp = await model.ainvoke(messages)

        summary = resp.content
        LOGGER.debug("Summary for %s: %s", cam_id, summary)
        first, sep, last = summary.partition(
            SUMMARIZATION_MODEL_REASONING_DELIMITER.get("end", "")
        )
        return (last if sep else first).strip("\n")

    async def _is_anomaly(
            self,
            camera_name: str,
            msg: str,
            first_path: str
        ) -> bool:
        """Perform anomaly detection on video analysis."""
        # Sematic search of the store with the video analysis as query.
        async with async_timeout.timeout(10):
            search_results = await self.entry.runtime_data.store.asearch(
                ("video_analysis", camera_name),
                query=msg,
                limit=10
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
        return (
            first_dt < dt_util.now() - timedelta(minutes=VIDEO_ANALYZER_TIME_OFFSET) or
            any(
                r.score < VIDEO_ANALYZER_SIMILARITY_THRESHOLD
                for r in search_results if r.score is not None
            )
        )

    async def _prune_old_snapshots(self, camera_id: str, batch: list[Path]) -> None:
        """Retain and prune old snapshots."""
        retention = self._retention_deques.setdefault(camera_id, deque())
        for path in batch:
            retention.append(path)
            if len(retention) > VIDEO_ANALYZER_SNAPSHOTS_TO_KEEP:
                old = retention.popleft()
                try:
                    await self.hass.async_add_executor_job(old.unlink)
                    LOGGER.debug("[%s] Deleted old snapshot: %s", camera_id, old)
                except OSError as e:
                    LOGGER.warning("[%s] Failed to delete %s: %s", camera_id, old, e)

    async def _process_snapshot_queue(self, camera_id: str) -> None:
        """
        Drain snapshot queue and process all frames as one batch.

        Generate frame descriptions, summarize, notify, store, and prune.
        """
        queue = self._snapshot_queues.get(camera_id)
        if not queue:
            return

        batch: list[Path] = []
        # Drain queue.
        try:
            while True:
                batch.append(queue.get_nowait())
        except QueueEmpty:
            pass

        camera_name = camera_id.split(".")[-1]

        # Generate frame descriptions.
        frame_descriptions: list[str] = []
        for path in batch:
            try:
                async with aiofiles.open(path, "rb") as f:
                    data = await f.read()
                async with async_timeout.timeout(30):
                    desc = await analyze_image(
                        self.entry.runtime_data.vision_model,
                        self.entry.options,
                        data,
                        None
                    )
                frame_descriptions.append(desc)
            except FileNotFoundError:
                LOGGER.warning("[%s] Snapshot not found: %s", camera_id, path)
                continue
            except TimeoutError:
                LOGGER.warning("[%s] Image analysis timed out for %s", camera_id, path)
                continue
            except HomeAssistantError:
                LOGGER.exception("[%s] Error analyzing %s", camera_id, path)
                continue
            except Exception:
                LOGGER.exception("[%s] Unexpected error analyzing %s.", camera_id, path)
                continue

        if not frame_descriptions:
            return

        # Summarize all frames at once.
        async with async_timeout.timeout(60):
            msg = await self._generate_summary(frame_descriptions, camera_id)

        # Pick middle snapshot for notification image.
        mid = batch[len(batch) // 2]
        notify_img = Path("/media/local") / Path(*mid.parts[-3:])

        mode = self.entry.options.get(CONF_VIDEO_ANALYZER_MODE)
        if mode == "notify_on_anomaly":
            first_snapshot = batch[0].parts[-1]
            LOGGER.debug("[%s] First snapshot: %s", camera_id, first_snapshot)
            if await self._is_anomaly(camera_name, msg, first_snapshot):
                LOGGER.debug("[%s] Video is an anomaly!", camera_id)
                await self._send_notification(msg, camera_name, notify_img)
        else:
            await self._send_notification(msg, camera_name, notify_img)

        # Store the result.
        async with async_timeout.timeout(10):
            await self.entry.runtime_data.store.aput(
                namespace=("video_analysis", camera_name),
                key=batch[0].name,
                value={"content": msg, "snapshots": [str(p) for p in batch]},
            )

        # Retain and prune old snapshots.
        await self._prune_old_snapshots(camera_id, batch)

    def _resolve_camera_from_motion(self, motion_entity_id: str) -> str | None:
        """Resolve a camera entity ID from a motion sensor ID."""
        # Explicit override.
        overrides: dict = VIDEO_ANALYZER_MOTION_CAMERA_MAP
        camera_id = overrides.get(motion_entity_id)
        if camera_id and self.hass.states.get(camera_id):
            return camera_id

        # Attempt Axis-style VMD fallback.
        base = motion_entity_id.replace("binary_sensor.", "")
        base = re.sub(r"_vmd\d+.*", "", base)
        inferred_camera_id = f"camera.{base}"
        if self.hass.states.get(inferred_camera_id):
            return inferred_camera_id

        return None

    async def _get_snapshot_dir(self, camera_id: str) -> Path:
        """Create snapshot folder once per camera."""
        if camera_id not in self._initialized_dirs:
            cam_dir = Path(VIDEO_ANALYZER_SNAPSHOT_ROOT) / camera_id.replace(".", "_")
            cam_dir.mkdir(parents=True, exist_ok=True)
            self._initialized_dirs.add(camera_id)
            dir_not_empty = await self.hass.async_add_executor_job(
                lambda: any(cam_dir.iterdir())
            )
            if dir_not_empty:
                msg = "[{id}] Folder not empty. Existing snapshots will not be pruned."
                LOGGER.info(msg.format(id=camera_id))
        return Path(VIDEO_ANALYZER_SNAPSHOT_ROOT) / camera_id.replace(".", "_")

    async def _take_single_snapshot(self, camera_id: str, now: datetime) -> Path | None:
        """Take a snapshot and enqueue it for processing."""
        snapshot_dir = await self._get_snapshot_dir(camera_id)
        timestamp = dt_util.as_local(now).strftime("%Y%m%d_%H%M%S")
        path = snapshot_dir / f"snapshot_{timestamp}.jpg"

        try:
            start_time = dt_util.utcnow()
            LOGGER.debug("[%s] Initiating snapshot at %s", camera_id, start_time)

            await self.hass.services.async_call(
                CAMERA_DOMAIN,
                "snapshot",
                {
                    "entity_id": camera_id,
                    "filename": str(path),
                },
                blocking=False,
            )
            LOGGER.debug("[%s] Snapshot service call completed.", camera_id)

            # Wait loop: up to 10 seconds, polling every 0.2 seconds.
            for i in range(50):
                exists = await self.hass.async_add_executor_job(path.exists)
                if exists:
                    LOGGER.debug(
                        "[%s] Snapshot appeared on disk after %.2f seconds.",
                        camera_id,
                        (dt_util.utcnow() - start_time).total_seconds()
                    )
                    break
                await asyncio.sleep(0.2)
                LOGGER.debug("[%s] Waiting for snapshot to appear... attempt %d",
                            camera_id, i + 1)
            else:
                LOGGER.warning(
                    "[%s] Snapshot failed to appear on disk after waiting: %s",
                    camera_id, path
                )
                return None

            queue = self._get_snapshot_queue(camera_id)
            await queue.put(path)
            LOGGER.debug("[%s] Enqueued snapshot %s", camera_id, path)
        except HomeAssistantError as e:
            LOGGER.warning("Snapshot failed for %s: %s", camera_id, e)
        else:
            return path

    async def _motion_snapshot_loop(self, camera_id: str) -> None:
        """Take snapshots while motion is active."""
        try:
            while True:
                now = dt_util.utcnow()
                await self._take_single_snapshot(camera_id, now)
                # Take snapshots no faster than the scan interval.
                await asyncio.sleep(VIDEO_ANALYZER_SCAN_INTERVAL)
        except asyncio.CancelledError:
            LOGGER.debug("Snapshot loop cancelled for camera: %s", camera_id)

    @callback
    def _handle_motion_event(self, event: Event) -> None:
        """Respond to motion sensor changes by starting/stopping snapshot loop."""
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
                    self._motion_snapshot_loop(camera_id)
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
        """Return a list of cameras currently recording."""
        return [
            state.entity_id for state in self.hass.states.async_all("camera")
            if state.state == "recording"
        ]

    async def _take_snapshots_from_recording_cameras(self, now: datetime) -> None:
        """
        Take snapshots from all recording cameras and enqueue them for processing.

        Relies on _take_single_snapshot to handle file creation and queueing.
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
        """Handle camera recording state changes to trigger processing."""
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
            timedelta(seconds=VIDEO_ANALYZER_SCAN_INTERVAL)
        )
        self._cancel_listen = self.hass.bus.async_listen(
            "state_changed",
            self._handle_camera_recording_state_change
        )
        if VIDEO_ANALYZER_TRIGGER_ON_MOTION:
            self._cancel_motion_listen = self.hass.bus.async_listen(
                "state_changed",
                self._handle_motion_event
            )
        LOGGER.info("Video analyzer started.")

    async def stop(self) -> None:
        """Stop the video analyzer: cancel tasks and unsubscribe listeners."""
        if not hasattr(self, "_cancel_track"):
            LOGGER.warning("VideoAnalyzer not started.")
            return

        tasks_to_await: list[asyncio.Task] = []

        # Cancel active motion-triggered snapshot loops.
        for task in self._active_motion_cameras.values():
            task.cancel()
            tasks_to_await.append(task)
        self._active_motion_cameras.clear()

        # Cancel all snapshot queue consumer tasks.
        for task in self._active_queue_tasks.values():
            task.cancel()
            tasks_to_await.append(task)
        self._active_queue_tasks.clear()

        # Unsubscribe the interval update and state listeners.
        try:
            self._cancel_track()
        except HomeAssistantError:
            LOGGER.warning("Error unsubscribing time interval listener", exc_info=True)

        try:
            self._cancel_listen()
        except HomeAssistantError:
            LOGGER.warning(
                "Error unsubscribing recording state listener", exc_info=True)

        if hasattr(self, "_cancel_motion_listen"):
            try:
                self._cancel_motion_listen()
            except HomeAssistantError:
                LOGGER.warning(
                    "Error unsubscribing motion event listener", exc_info=True
                )

        # Await cancellation of all background tasks, with timeout.
        if tasks_to_await:
            done, pending = await asyncio.wait(tasks_to_await, timeout=5)
            for task in pending:
                LOGGER.warning("Task did not cancel in time: %s", task)

        LOGGER.info("Video analyzer stopped.")

    def is_running(self) -> bool:
        """Check if video analyzer is running."""
        return (hasattr(self, "_cancel_track") and hasattr(self, "_cancel_listen"))

async def async_setup_entry(hass: HomeAssistant, entry: HGAConfigEntry) -> bool:
    """Set up Home Generative Agent from a config entry."""
    # Initialize models and verify they were setup correctly.
    # TODO(goruck): fix blocking call.
    # https://github.com/goruck/home-generative-agent/issues/110
    chat_model = ChatOpenAI(
        api_key=entry.data.get(CONF_API_KEY),
        timeout=10,
        http_async_client=get_async_client(hass),
    ).configurable_fields(
        model_name=ConfigurableField(id="model_name"),
        temperature=ConfigurableField(id="temperature"),
        top_p=ConfigurableField(id="top_p"),
        max_tokens=ConfigurableField(id="max_tokens"),
    )
    try:
        await hass.async_add_executor_job(chat_model.get_name)
    except HomeAssistantError:
        LOGGER.exception("Error setting up chat model.")
        return False

    edge_chat_model = ChatOllama(
        model=RECOMMENDED_EDGE_CHAT_MODEL,
        base_url=EDGE_CHAT_MODEL_URL
    ).configurable_fields(
        model=ConfigurableField(id="model"),
        format=ConfigurableField(id="format"),
        temperature=ConfigurableField(id="temperature"),
        top_p=ConfigurableField(id="top_p"),
        num_predict=ConfigurableField(id="num_predict"),
        num_ctx=ConfigurableField(id="num_ctx"),
    )
    try:
        await hass.async_add_executor_job(edge_chat_model.get_name)
    except HomeAssistantError:
        LOGGER.exception("Error setting up edge chat model.")
        return False

    vision_model = ChatOllama(
        model=RECOMMENDED_VLM,
        base_url=VLM_URL,
    ).configurable_fields(
        model=ConfigurableField(id="model"),
        format=ConfigurableField(id="format"),
        temperature=ConfigurableField(id="temperature"),
        top_p=ConfigurableField(id="top_p"),
        num_predict=ConfigurableField(id="num_predict"),
        num_ctx=ConfigurableField(id="num_ctx"),
    )
    try:
        await hass.async_add_executor_job(vision_model.get_name)
    except HomeAssistantError:
        LOGGER.exception("Error setting up VLM.")
        return False

    summarization_model = ChatOllama(
        model=RECOMMENDED_SUMMARIZATION_MODEL,
        base_url=SUMMARIZATION_MODEL_URL
    ).configurable_fields(
        model=ConfigurableField(id="model"),
        format=ConfigurableField(id="format"),
        temperature=ConfigurableField(id="temperature"),
        top_p=ConfigurableField(id="top_p"),
        num_predict=ConfigurableField(id="num_predict"),
        num_ctx=ConfigurableField(id="num_ctx"),
    )
    try:
        await hass.async_add_executor_job(vision_model.get_name)
    except HomeAssistantError:
        LOGGER.exception("Error setting up summarization model.")
        return False

    embedding_model = OllamaEmbeddings(
        model=RECOMMENDED_EMBEDDING_MODEL,
        base_url=EMBEDDING_MODEL_URL,
        num_ctx=EMBEDDING_MODEL_CTX
    )

    # Open postgresql database for short-term and long-term memory.
    connection_kwargs = {
        "autocommit": True,
        "prepare_threshold": 0,
        "row_factory": dict_row
    }
    pool: AsyncConnectionPool[AsyncConnection[DictRow]] = AsyncConnectionPool(
        conninfo=DB_URI,
        min_size=5,
        max_size=20,
        kwargs=connection_kwargs,
        open=False
    )
    try:
        await pool.open()
    except PoolTimeout:
        LOGGER.exception("Error opening postgresql db.")
        return False

    # Initialize store for session-based (long-term) memory with semantic search.
    async def _generate_embeddings(texts: Sequence[str]) -> list[list[float]]:
        """Generate embeddings from a list of text."""
        return await embedding_model.aembed_documents(list(texts))
    index_config: PostgresIndexConfig= {
        "embed": _generate_embeddings,
        "dims": EMBEDDING_MODEL_DIMS,
        "fields": ["content"]
    }
    # NOTE: must call .setup() the first time store is used.
    store = AsyncPostgresStore(
        pool,
        index=index_config,
    )
    # NOTE: must call .setup() the first time store is used.
    #await store.setup()  # noqa: ERA001

    # Initialize database for thread-based (short-term) memory.
    checkpointer = AsyncPostgresSaver(pool)
    # NOTE: must call .setup() the first time checkpointer is used.
    #await checkpointer.setup()  # noqa: ERA001

    # Initialize video analyzer and start if option is set.
    video_analyzer = VideoAnalyzer(hass, entry)
    if entry.options.get(CONF_VIDEO_ANALYZER_MODE) != "disable":
        video_analyzer.start()

    entry.runtime_data = HGAData(
        chat_model=chat_model,
        edge_chat_model=edge_chat_model,
        vision_model=vision_model,
        summarization_model=summarization_model,
        pool=pool,
        store=store,
        video_analyzer=video_analyzer,
        checkpointer=checkpointer,
        embedding_model=embedding_model
    )

    # Setup conversation platform.
    await hass.config_entries.async_forward_entry_setups(entry, PLATFORMS)

    return True

async def async_unload_entry(hass: HomeAssistant, entry: HGAConfigEntry) -> bool:
    """Unload Home Generative Agent."""
    pool = entry.runtime_data.pool
    await pool.close()

    video_analyzer = entry.runtime_data.video_analyzer
    await video_analyzer.stop()

    await hass.config_entries.async_unload_platforms(entry, PLATFORMS)

    return True

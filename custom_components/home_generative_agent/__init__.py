"""Home Generative Agent Initialization."""

from __future__ import annotations

import asyncio
import logging
import re
from dataclasses import dataclass
from datetime import datetime, timedelta
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING

import aiofiles
import homeassistant.util.dt as dt_util
from homeassistant.components.camera import DOMAIN as CAMERA_DOMAIN
from homeassistant.const import (
    CONF_API_KEY,
    EVENT_STATE_CHANGED,
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
from langgraph.store.postgres import AsyncPostgresStore
from psycopg.rows import dict_row
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
    from homeassistant.config_entries import ConfigEntry
    from homeassistant.core import Event, HomeAssistant

LOGGER = logging.getLogger(__name__)

PLATFORMS: list[Platform] = (Platform.CONVERSATION,)

type HGAConfigEntry = ConfigEntry[HGAData]

@dataclass
class HGAData:
    """Data for Home Generative Assistant."""

    chat_model: ChatOpenAI
    edge_chat_model: ChatOllama
    vision_model: ChatOllama
    summarization_model: ChatOllama
    pool: AsyncConnectionPool
    store: AsyncPostgresStore
    video_analyzer: VideoAnalyzer

async def _generate_embeddings(
        texts: list[str],
        model: OllamaEmbeddings
    ) -> list[list[float]]:
    """Generate embeddings from a list of text."""
    return await model.aembed_documents(texts)

class VideoAnalyzer:
    """Analyze video from recording or motion-triggered cameras."""

    def __init__(self, hass: HomeAssistant, entry: HGAConfigEntry) -> None:  # noqa: D107
        self.hass = hass
        self.entry = entry
        self.camera_snapshots = {}
        self.camera_write_locks = {}
        self.active_motion_cameras = {}

    async def _send_notification(
            self,
            msg: str,
            camera_name: str,
            camera_id: str,
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
                    "entity_id:": camera_id,
                    "image": str(notify_img_path)
                }
            },
            blocking=True
        )

    async def _generate_summary(self, frames: list[str], cam_id: str) -> str:
        """Generate a summarized analysis from frame descriptions."""
        if not frames:
            return ValueError("At least one frame description required.")

        if len(frames) == 1:
            return frames[0]

        opts = self.entry.options
        tag = "\n<frame description>\n{}\n</frame description>"
        prompt = " ".join([
            VIDEO_ANALYZER_PROMPT
        ] + [tag.format(i) for i in frames])

        LOGGER.debug("Prompt: %s", prompt)

        cfg = {
            "model": opts.get(
                CONF_SUMMARIZATION_MODEL, RECOMMENDED_SUMMARIZATION_MODEL
            ),
            "temperature": opts.get(
                CONF_SUMMARIZATION_MODEL_TEMPERATURE,
                RECOMMENDED_SUMMARIZATION_MODEL_TEMPERATURE,
            ),
            "top_p": opts.get(
                CONF_SUMMARIZATION_MODEL_TOP_P,
                RECOMMENDED_SUMMARIZATION_MODEL_TOP_P,
            ),
            "num_predict": SUMMARIZATION_MODEL_PREDICT,
            "num_ctx": SUMMARIZATION_MODEL_CTX,
        }

        messages = [
            SystemMessage(content=VIDEO_ANALYZER_SYSTEM_MESSAGE),
            HumanMessage(content=prompt),
        ]
        model = self.entry.summarization_model.with_config(config=cfg)
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
            first_path_parts: tuple[str]
        ) -> bool:
        """Perform anomaly detection on video analysis."""
        # Sematic search of the store with the video analysis as query.
        search_results = await self.entry.store.asearch(
            ("video_analysis", camera_name),
            query=msg,
            limit=10
        )
        LOGGER.debug("Search results: %s", search_results)

        # Calculate a "no newer than" time threshold from first snapshot time
        # by delaying it by the time offset.
        # Snapshot names are in the form "snapshot_20250426_002804.jpg".
        first_str = first_path_parts[-1].replace("snapshot_", "").replace(".jpg", "")
        first_dt = dt_util.as_local(datetime.strptime(first_str, "%Y%m%d_%H%M%S"))  # noqa: DTZ007
        no_newer_dt = first_dt - timedelta(minutes=VIDEO_ANALYZER_TIME_OFFSET)

        # Simple anomaly detection.
        # If all search results are older then the time threshold or if any are
        # newer or equal to it, and have a lower score then the similarity
        # threshold, declare the current video analysis as an anomaly.
        return (
            all(r.updated_at < no_newer_dt for r in search_results) or
            any(r.updated_at >= no_newer_dt and
                r.score < VIDEO_ANALYZER_SIMILARITY_THRESHOLD
                for r in search_results)
        )

    async def _process_snapshots(self, camera_id: str) -> None:
        """Process snapshots after they are written."""
        lock = self.camera_write_locks.get(camera_id)
        if lock:
            LOGGER.debug("[%s] Waiting for snapshot writes to finish...", camera_id)
            async with lock:
                LOGGER.debug("[%s] Done waiting for writes.", camera_id)

        snapshots: list[Path] = self.camera_snapshots.get(camera_id, [])

        if not snapshots:
            return

        camera_name = camera_id.split(".")[-1]

        options = self.entry.options

        LOGGER.debug("[%s] Processing %s snapshots...", camera_id, len(snapshots))
        frame_descriptions: list[str] = []
        for path in snapshots:
            LOGGER.debug(" - %s", path)

            async with aiofiles.open(path, "rb") as file:
                image = await file.read()

                detection_keywords = None
                frame_description = await analyze_image(
                    self.entry.vision_model, options, image, detection_keywords
                )
                LOGGER.debug("Analysis for %s: %s", path, frame_description)
                frame_descriptions.append(frame_description)

        msg = await self._generate_summary(frame_descriptions, camera_id)

        first_path_parts = snapshots[0].parts

        # Use mid snapshot as a static image in the mobile app notification.
        mid_index = len(snapshots) // 2
        mid_path_parts = snapshots[mid_index].parts
        notify_img_path = Path("/media/local") / Path(*mid_path_parts[-3:])

        if (mode := options.get(CONF_VIDEO_ANALYZER_MODE)) == "notify_on_anomaly":
            is_anomaly = await self._is_anomaly(camera_name, msg, first_path_parts)
            LOGGER.debug("Is anomaly: %s", is_anomaly)

            if is_anomaly:
                await self._send_notification(
                    msg, camera_name, camera_id, notify_img_path
                )
        elif mode == "always_notify":
            await self._send_notification(msg, camera_name, camera_id, notify_img_path)

        # Store current msg and associated snapshots.
        await self.entry.store.aput(
            namespace=("video_analysis", camera_name),
            key=first_path_parts[-1], # key is date and time of first snapshot
            value={
                "content": msg, "snapshots": [str(s) for s in snapshots]
            },
        )

        # Get a list of snapshot paths sorted by modification time (newest first).
        folder = Path(VIDEO_ANALYZER_SNAPSHOT_ROOT) / camera_id.replace(".", "_")
        snapshots_in_folder = await self.hass.async_add_executor_job(
            lambda: sorted(
                [f for f in folder.iterdir() if f.is_file()],
                key=lambda x: x.stat().st_mtime,
                reverse=True
            )
        )

        # Delete snapshots beyond number to keep.
        for path in snapshots_in_folder[VIDEO_ANALYZER_SNAPSHOTS_TO_KEEP:]:
            try:
                await self.hass.async_add_executor_job(path.unlink)
                LOGGER.debug("Deleted snapshot: %s", path)
            except OSError:
                LOGGER.warning("Failed to delete snapshot: %s", path)

        self.camera_snapshots[camera_id] = []

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

        LOGGER.debug(
            "Failed to resolve camera for motion sensor %s (override: %s, fallback: %s)",
            motion_entity_id, camera_id, inferred_camera_id
        )
        return None

    async def _take_single_snapshot(self, camera_id: str, now: datetime) -> Path | None:
        """Take a snapshot from a specific camera."""
        snapshot_root_path = Path(VIDEO_ANALYZER_SNAPSHOT_ROOT)
        snapshot_root_path.mkdir(parents=True, exist_ok=True)

        timestamp = dt_util.as_local(now).strftime("%Y%m%d_%H%M%S")
        cam_dir = snapshot_root_path / camera_id.replace(".", "_")
        cam_dir.mkdir(parents=True, exist_ok=True)
        snapshot_path = cam_dir / f"snapshot_{timestamp}.jpg"

        lock = self.camera_write_locks.setdefault(camera_id, asyncio.Lock())

        try:
            async with lock:
                await self.hass.services.async_call(
                    CAMERA_DOMAIN,
                    "snapshot",
                    {
                        "entity_id": camera_id,
                        "filename": str(snapshot_path),
                    },
                    blocking=True,
                )
                self.camera_snapshots.setdefault(camera_id, []).append(snapshot_path)
                LOGGER.debug("[%s] Snapshot saved to %s", camera_id, snapshot_path)
                return snapshot_path
        except HomeAssistantError as e:
            LOGGER.warning("Snapshot failed for %s: %s", camera_id, e)
            return None

    async def _motion_snapshot_loop(self, camera_id: str) -> None:
        """Continuously take snapshots while motion is active."""
        try:
            while True:
                now = dt_util.utcnow()
                await self._take_single_snapshot(camera_id, now)
                await asyncio.sleep(1)
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
            LOGGER.debug("No camera resolved for motion sensor: %s", entity_id)
            return

        if new_state.state == "on" and (old_state is None or old_state.state != "on"):
            if camera_id not in self.active_motion_cameras:
                LOGGER.debug("Motion ON: Starting snapshot loop for %s", camera_id)
                task = self.hass.async_create_task(self._motion_snapshot_loop(camera_id))
                self.active_motion_cameras[camera_id] = task

        elif new_state.state == "off" and camera_id in self.active_motion_cameras:
            LOGGER.debug("Motion OFF: Stopping snapshot loop for %s", camera_id)
            task = self.active_motion_cameras.pop(camera_id, None)
            if task and not task.done():
                task.cancel()

            # Wait briefly to allow final snapshot to write
            async def _delayed_process() -> None:
                await asyncio.sleep(1)
                await self._process_snapshots(camera_id)

            self.hass.async_create_task(_delayed_process())

    @callback
    def _get_recording_cameras(self) -> list[str]:
        """Return a list of cameras currently recording."""
        return [
            state.entity_id for state in self.hass.states.async_all("camera")
            if state.state == "recording"
        ]

    async def _take_snapshot(self, now: datetime) -> None:
        """Take snapshots from all recording cameras."""
        snapshot_root_path = Path(VIDEO_ANALYZER_SNAPSHOT_ROOT)
        snapshot_root_path.mkdir(parents=True, exist_ok=True)

        for camera_id in self._get_recording_cameras():
            timestamp = dt_util.as_local(now).strftime("%Y%m%d_%H%M%S")
            cam_dir = snapshot_root_path / camera_id.replace(".", "_")
            cam_dir.mkdir(parents=True, exist_ok=True)
            snapshot_path = cam_dir / f"snapshot_{timestamp}.jpg"

            # Create a lock to track pending snapshot write.
            lock = self.camera_write_locks.setdefault(camera_id, asyncio.Lock())

            async with lock:
                await self.hass.services.async_call(
                    CAMERA_DOMAIN,
                    "snapshot",
                    {
                        "entity_id": camera_id,
                        "filename": str(snapshot_path),
                    },
                    blocking=True,
                )
                self.camera_snapshots.setdefault(camera_id, []).append(snapshot_path)
                LOGGER.debug("[%s] Snapshot saved to %s", camera_id, snapshot_path)

    @callback
    def _handle_camera_state_change(self, event: Event) -> None:
        """Handle camera recording state changes to trigger processing."""
        entity_id = event.data.get("entity_id")
        if not entity_id or not entity_id.startswith("camera."):
            return

        old_state = event.data.get("old_state")
        new_state = event.data.get("new_state")
        if old_state is None or new_state is None:
            return

        if old_state.state == "recording" and new_state.state != "recording":
            # Debounce: wait 1 second before processing.
            async def _delayed_process() -> None:
                await asyncio.sleep(1)
                await self._process_snapshots(entity_id)

            self.hass.async_create_task(_delayed_process())

    def start(self) -> None:
        """Start the video analyzer."""
        self.cancel_track = async_track_time_interval(
            self.hass,
            self._take_snapshot,
            timedelta(seconds=VIDEO_ANALYZER_SCAN_INTERVAL)
        )
        self.cancel_listen = self.hass.bus.async_listen(
            EVENT_STATE_CHANGED,
            self._handle_camera_state_change
        )
        if VIDEO_ANALYZER_TRIGGER_ON_MOTION:
            self.cancel_motion_listen = self.hass.bus.async_listen(
                EVENT_STATE_CHANGED,
                self._handle_motion_event
            )
        LOGGER.info("Video analyzer started.")

    def stop(self) -> None:
        """Stop the video analyzer."""
        for task in self.active_motion_cameras.values():
            task.cancel()
        self.active_motion_cameras.clear()
        if self.is_running():
            self.cancel_track()
            self.cancel_listen()
            if hasattr(self, "cancel_motion_listen"):
                self.cancel_motion_listen()
            LOGGER.info("Video analyzer stopped.")

    def is_running(self) -> bool:
        """Check if video analyzer is running."""
        return hasattr(self, "cancel_track") and hasattr(self, "cancel_listen")

async def async_setup_entry(hass: HomeAssistant, entry: HGAConfigEntry) -> bool:
    """Set up Home Generative Agent from a config entry."""
    # Initialize models and verify they were setup correctly.
    chat_model = ChatOpenAI( #TODO: fix blocking call
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
    except HomeAssistantError as err:
        LOGGER.error("Error setting up chat model: %s", err)
        return False
    entry.chat_model = chat_model

    edge_chat_model = ChatOllama(
        model=RECOMMENDED_EDGE_CHAT_MODEL,
        base_url=EDGE_CHAT_MODEL_URL,
        http_async_client=get_async_client(hass)
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
    except HomeAssistantError as err:
        LOGGER.error("Error setting up edge chat model: %s", err)
        return False
    entry.edge_chat_model = edge_chat_model

    vision_model = ChatOllama(
        model=RECOMMENDED_VLM,
        base_url=VLM_URL,
        http_async_client=get_async_client(hass)
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
    except HomeAssistantError as err:
        LOGGER.error("Error setting up VLM: %s", err)
        return False
    entry.vision_model = vision_model

    summarization_model = ChatOllama(
        model=RECOMMENDED_SUMMARIZATION_MODEL,
        base_url=SUMMARIZATION_MODEL_URL,
        http_async_client=get_async_client(hass)
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
    except HomeAssistantError as err:
        LOGGER.error("Error setting up summarization model: %s", err)
        return False
    entry.summarization_model = summarization_model

    embedding_model = OllamaEmbeddings(
        model=RECOMMENDED_EMBEDDING_MODEL,
        base_url=EMBEDDING_MODEL_URL,
        num_ctx=EMBEDDING_MODEL_CTX
    )
    # TODO: find a way to verify embedding model was setup correctly.
    #try:
        #await hass.async_add_executor_job(embedding_model.get_name)
    #except HomeAssistantError as err:
        #LOGGER.error("Error setting up embedding model: %s", err)
        #return False
    entry.embedding_model = embedding_model

    # Open postgresql database for short-term and long-term memory.
    connection_kwargs = {
        "autocommit": True,
        "prepare_threshold": 0,
        "row_factory": dict_row
    }
    pool = AsyncConnectionPool(
        conninfo=DB_URI,
        min_size=5,
        max_size=20,
        kwargs=connection_kwargs,
        open=False
    )
    try:
        await pool.open()
    except PoolTimeout as err:
        LOGGER.error("Error opening postgresql db: %s", err)
        return False
    entry.pool = pool

    # Initialize store for session-based (long-term) memory with semantic search.
    store = AsyncPostgresStore(
        pool,
        index={
            "embed": partial(
                _generate_embeddings,
                model=embedding_model
            ),
            "dims": EMBEDDING_MODEL_DIMS,
            "fields": ["content"]
        }
    )
    # NOTE: must call .setup() the first time store is used.
    #await store.setup()  # noqa: ERA001
    entry.store = store

    # Initialize video analyzer and start if option is set.
    video_analyzer = VideoAnalyzer(hass, entry)
    if entry.options.get(CONF_VIDEO_ANALYZER_MODE) != "disable":
        video_analyzer.start()
    entry.video_analyzer = video_analyzer

    # Setup conversation platform.
    await hass.config_entries.async_forward_entry_setups(entry, PLATFORMS)

    return True

async def async_unload_entry(hass: HomeAssistant, entry: HGAConfigEntry) -> bool:
    """Unload Home Generative Agent."""
    pool = entry.pool
    await pool.close()

    video_analyzer = entry.video_analyzer
    video_analyzer.stop()

    await hass.config_entries.async_unload_platforms(entry, PLATFORMS)

    return True

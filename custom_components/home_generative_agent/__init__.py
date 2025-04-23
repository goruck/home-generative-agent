"""Home Generative Agent Initialization."""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
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
from psycopg.rows import dict_row
from psycopg_pool import AsyncConnectionPool, PoolTimeout

from .const import (
    CONF_SUMMARIZATION_MODEL,
    CONF_SUMMARIZATION_MODEL_TEMPERATURE,
    CONF_SUMMARIZATION_MODEL_TOP_P,
    CONF_VIDEO_ANALYZER_ENABLE,
    DB_URI,
    EDGE_CHAT_MODEL_URL,
    EMBEDDING_MODEL_CTX,
    EMBEDDING_MODEL_URL,
    RECOMMENDED_EDGE_CHAT_MODEL,
    RECOMMENDED_EMBEDDING_MODEL,
    RECOMMENDED_SUMMARIZATION_MODEL,
    RECOMMENDED_SUMMARIZATION_MODEL_TEMPERATURE,
    RECOMMENDED_SUMMARIZATION_MODEL_TOP_P,
    RECOMMENDED_VIDEO_ANALYZER_ENABLE,
    RECOMMENDED_VLM,
    SUMMARIZATION_MODEL_CTX,
    SUMMARIZATION_MODEL_PREDICT,
    SUMMARIZATION_MODEL_URL,
    VIDEO_ANALYZER_MOBILE_APP,
    VIDEO_ANALYZER_PROMPT,
    VIDEO_ANALYZER_SCAN_INTERVAL,
    VIDEO_ANALYZER_SNAPSHOT_ROOT,
    VIDEO_ANALYZER_SYSTEM_MESSAGE,
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
    video_analyzer: VideoAnalyzer

class VideoAnalyzer:
    """Analyze video from recording cameras."""

    def __init__(self, hass: HomeAssistant, entry: HGAConfigEntry) -> None:
        """Init the video analyzer."""
        # Track snapshots and pending writes per camera.
        self.camera_snapshots = {}
        self.camera_write_locks = {}

        self.hass = hass
        self.entry = entry

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

    async def _process_snapshots(self, camera_id: str) -> None:
        """Process snapshots after a camera stops recording."""
        lock = self.camera_write_locks.get(camera_id)
        if lock:
            LOGGER.debug("[%s] Waiting for snapshot writes to finish...", camera_id)
            async with lock:
                LOGGER.debug("[%s] Done waiting for writes.", camera_id)

        snapshots = self.camera_snapshots.get(camera_id, [])
        if not snapshots:
            return

        camera_name = camera_id.split(".")[-1]

        options = self.entry.options

        LOGGER.debug("[%s] Processing %s snapshots...", camera_id, len(snapshots))
        frame_descriptions = []
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

        if len(frame_descriptions) > 1:
            prompt_start = VIDEO_ANALYZER_PROMPT
            tag_template = "\n<start frame description> {i} <end frame description>"
            prompt_parts = [tag_template.format(i=i) for i in frame_descriptions]
            prompt_parts.insert(0, prompt_start)
            prompt = " ".join(prompt_parts)
            LOGGER.debug("Prompt: %s", prompt)
            system_message = VIDEO_ANALYZER_SYSTEM_MESSAGE
            messages = [
                SystemMessage(content=system_message),
                HumanMessage(content=prompt)
            ]
            model = self.entry.summarization_model
            model_with_config = model.with_config(
                config={
                    "model": options.get(
                        CONF_SUMMARIZATION_MODEL,
                        RECOMMENDED_SUMMARIZATION_MODEL,
                    ),
                    "temperature": options.get(
                        CONF_SUMMARIZATION_MODEL_TEMPERATURE,
                        RECOMMENDED_SUMMARIZATION_MODEL_TEMPERATURE,
                    ),
                    "top_p": options.get(
                        CONF_SUMMARIZATION_MODEL_TOP_P,
                        RECOMMENDED_SUMMARIZATION_MODEL_TOP_P,
                    ),
                    "num_predict": SUMMARIZATION_MODEL_PREDICT,
                    "num_ctx": SUMMARIZATION_MODEL_CTX,
                }
            )
            summary = await model_with_config.ainvoke(messages)
            LOGGER.debug("Summary for %s: %s", camera_id, summary.content)

            notify_msg = summary.content
        else:
            notify_msg = frame_description

        # Grab first snapshot to display as a static image in the notification.
        img_path_parts = snapshots[0].parts
        notify_img_path = Path("/media/local") / Path(*img_path_parts[-3:])

        await self.hass.services.async_call(
            "notify",
            VIDEO_ANALYZER_MOBILE_APP,
            {
                "message": notify_msg,
                "title": f"Camera Alert from {camera_name}!",
                "data": {
                    "entity_id:": camera_id,
                    "image": str(notify_img_path)
                }
            },
            blocking=True
        )

        self.camera_snapshots[camera_id] = []

    @callback
    def _handle_camera_state_change(self, event: Event) -> None:
        """Handle camera state changes to trigger processing."""
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
        # Start video analyzer snapshot job.
        self.cancel_track = async_track_time_interval(
            self.hass,
            self._take_snapshot,
            timedelta(seconds=VIDEO_ANALYZER_SCAN_INTERVAL)
        )
        # Watch for recording cameras and analyze video.
        self.cancel_listen = self.hass.bus.async_listen(
            EVENT_STATE_CHANGED,
            self._handle_camera_state_change
        )
        LOGGER.info("Video analyzer started.")

    def stop(self) -> None:
        """Stop the video analyzer."""
        if self.is_running():
            self.cancel_track()
            self.cancel_listen()
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

    # Initialize video analyzer and start if option is set.
    video_analyzer = VideoAnalyzer(hass, entry)
    if entry.options.get(
        CONF_VIDEO_ANALYZER_ENABLE, RECOMMENDED_VIDEO_ANALYZER_ENABLE
    ):
        video_analyzer.start()
    entry.video_analyzer = video_analyzer

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

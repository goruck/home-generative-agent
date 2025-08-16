"""Home Generative Agent Initialization."""

from __future__ import annotations

import asyncio
import logging
import re
from asyncio import QueueEmpty
from collections import deque
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, Any

import aiofiles
import async_timeout
import homeassistant.util.dt as dt_util
from homeassistant.components.camera.const import DOMAIN as CAMERA_DOMAIN
from homeassistant.config_entries import ConfigEntry
from homeassistant.const import CONF_API_KEY, Platform
from homeassistant.core import HomeAssistant, callback
from homeassistant.exceptions import HomeAssistantError
from homeassistant.helpers.event import async_track_time_interval
from homeassistant.helpers.httpx_client import get_async_client
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import ConfigurableField
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.store.postgres import AsyncPostgresStore
from langgraph.store.postgres.base import PostgresIndexConfig
from psycopg.rows import DictRow, dict_row
from psycopg_pool import AsyncConnectionPool, PoolTimeout

from .const import (
    CHAT_MODEL_MAX_TOKENS,
    CHAT_MODEL_NUM_CTX,
    CHAT_MODEL_TOP_P,
    CONF_CHAT_MODEL_PROVIDER,
    CONF_CHAT_MODEL_TEMPERATURE,
    CONF_EMBEDDING_MODEL_PROVIDER,
    CONF_OLLAMA_CHAT_MODEL,
    CONF_OLLAMA_EMBEDDING_MODEL,
    CONF_OLLAMA_SUMMARIZATION_MODEL,
    CONF_OLLAMA_VLM,
    CONF_OPENAI_CHAT_MODEL,
    CONF_OPENAI_EMBEDDING_MODEL,
    CONF_OPENAI_SUMMARIZATION_MODEL,
    CONF_OPENAI_VLM,
    CONF_SUMMARIZATION_MODEL_PROVIDER,
    CONF_SUMMARIZATION_MODEL_TEMPERATURE,
    CONF_VIDEO_ANALYZER_MODE,
    CONF_VLM_PROVIDER,
    CONF_VLM_TEMPERATURE,
    DB_URI,
    DOMAIN,
    EMBEDDING_MODEL_CTX,
    EMBEDDING_MODEL_DIMS,
    HTTP_STATUS_BAD_REQUEST,
    OLLAMA_URL,
    REASONING_DELIMITERS,
    RECOMMENDED_CHAT_MODEL_PROVIDER,
    RECOMMENDED_CHAT_MODEL_TEMPERATURE,
    RECOMMENDED_EMBEDDING_MODEL_PROVIDER,
    RECOMMENDED_OLLAMA_CHAT_MODEL,
    RECOMMENDED_OLLAMA_EMBEDDING_MODEL,
    RECOMMENDED_OLLAMA_SUMMARIZATION_MODEL,
    RECOMMENDED_OLLAMA_VLM,
    RECOMMENDED_OPENAI_CHAT_MODEL,
    RECOMMENDED_OPENAI_EMBEDDING_MODEL,
    RECOMMENDED_OPENAI_SUMMARIZATION_MODEL,
    RECOMMENDED_OPENAI_VLM,
    RECOMMENDED_SUMMARIZATION_MODEL_PROVIDER,
    RECOMMENDED_SUMMARIZATION_MODEL_TEMPERATURE,
    RECOMMENDED_VLM_PROVIDER,
    RECOMMENDED_VLM_TEMPERATURE,
    SUMMARIZATION_MODEL_CTX,
    SUMMARIZATION_MODEL_PREDICT,
    SUMMARIZATION_MODEL_TOP_P,
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
    VLM_NUM_CTX,
    VLM_NUM_PREDICT,
    VLM_TOP_P,
)
from .tools import analyze_image

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator, Sequence

    from homeassistant.config_entries import ConfigEntry
    from homeassistant.core import Event, HomeAssistant
    from langchain_core.language_models import BaseMessage, LanguageModelInput
    from langchain_core.runnables.base import RunnableSerializable
    from psycopg import AsyncConnection

LOGGER = logging.getLogger(__name__)

PLATFORMS = (Platform.CONVERSATION,)


# ---------------- Utilities ----------------


def _ensure_http_url(url: str) -> str:
    """Ensure a URL has an explicit scheme."""
    if url.startswith(("http://", "https://")):
        return url
    return f"http://{url}"


async def _ollama_healthy(
    hass: HomeAssistant, base_url: str, timeout_s: float = 2.0
) -> bool:
    """Quick reachability check for Ollama."""
    url = _ensure_http_url(base_url).rstrip("/") + "/api/tags"
    client = get_async_client(hass)
    try:
        async with async_timeout.timeout(timeout_s):
            resp = await client.get(url)
        if resp.status_code < HTTP_STATUS_BAD_REQUEST:
            return True
        LOGGER.warning("Ollama health check HTTP %s: %s", resp.status_code, resp.text)
    except Exception as err:  # noqa: BLE001
        LOGGER.warning("Ollama health check failed at %s: %s", url, err)
    return False


async def _openai_healthy(
    hass: HomeAssistant, api_key: str | None, timeout_s: float = 2.0
) -> bool:
    """Quick reachability check for OpenAI."""
    if not api_key:
        LOGGER.warning("OpenAI health check skipped: missing API key.")
        return False
    client = get_async_client(hass)
    headers = {"Authorization": f"Bearer {api_key}"}
    try:
        async with async_timeout.timeout(timeout_s):
            resp = await client.get("https://api.openai.com/v1/models", headers=headers)
        if resp.status_code < HTTP_STATUS_BAD_REQUEST:
            return True
        LOGGER.warning("OpenAI health check HTTP %s: %s", resp.status_code, resp.text)
    except Exception as err:  # noqa: BLE001
        LOGGER.warning("OpenAI health check failed: %s", err)
    return False


async def _generate_embeddings(
    emb: OpenAIEmbeddings | OllamaEmbeddings, texts: Sequence[str]
) -> list[list[float]]:
    """Generate embeddings from a list of text."""
    return await emb.aembed_documents(list(texts))


class NullChat:
    """Non-throwing fallback implementing ainvoke/astream/with_config."""

    async def ainvoke(self, _input: Any, **_kw: Any) -> str:
        """Return a placeholder response."""
        return "LLM unavailable."

    async def astream(self, _input: Any, **_kw: Any) -> AsyncGenerator[str, Any]:
        """Return a placeholder response."""
        yield "LLM unavailable."

    def with_config(self, **_cfg: Any) -> NullChat:
        """Return self, as this is a no-op."""
        return self


# ---------------- Runtime data ----------------


@dataclass
class HGAData:
    """Run-time data for Home Generative Agent."""

    # Selected models for roles (never None; may be NullChat)
    chat_model: Any
    vision_model: Any
    summarization_model: Any

    # Storage
    pool: AsyncConnectionPool[AsyncConnection[DictRow]]
    store: AsyncPostgresStore
    checkpointer: AsyncPostgresSaver

    # Video analyzer
    video_analyzer: VideoAnalyzer


# After HGAData is defined, we can specialize the ConfigEntry type.
type HGAConfigEntry = ConfigEntry[HGAData]


# ---------------- Video Analyzer ----------------


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

        sum_provider = self.entry.options.get(
            CONF_SUMMARIZATION_MODEL_PROVIDER, RECOMMENDED_SUMMARIZATION_MODEL_PROVIDER
        )
        temp = self.entry.options.get(
            CONF_SUMMARIZATION_MODEL_TEMPERATURE,
            RECOMMENDED_SUMMARIZATION_MODEL_TEMPERATURE,
        )

        if sum_provider == "openai":
            self._sum_model_cfg = {
                "configurable": {
                    "model_name": self.entry.options.get(
                        CONF_OPENAI_SUMMARIZATION_MODEL,
                        RECOMMENDED_OPENAI_SUMMARIZATION_MODEL,
                    ),
                    "temperature": temp,
                    "top_p": SUMMARIZATION_MODEL_TOP_P,
                    "max_tokens": SUMMARIZATION_MODEL_PREDICT,
                }
            }
        else:  # ollama
            self._sum_model_cfg = {
                "configurable": {
                    "model": self.entry.options.get(
                        CONF_OLLAMA_SUMMARIZATION_MODEL,
                        RECOMMENDED_OLLAMA_SUMMARIZATION_MODEL,
                    ),
                    "temperature": temp,
                    "top_p": SUMMARIZATION_MODEL_TOP_P,
                    "num_predict": SUMMARIZATION_MODEL_PREDICT,
                    "num_ctx": SUMMARIZATION_MODEL_CTX,
                }
            }

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
        await self.hass.services.async_call(
            "notify",
            VIDEO_ANALYZER_MOBILE_APP,
            {
                "message": msg,
                "title": f"Camera Alert from {camera_name}!",
                "data": {"image": str(notify_img_path)},
            },
            blocking=False,
        )

    async def _generate_summary(self, frames: list[str], cam_id: str) -> str:
        await asyncio.sleep(0)
        if not frames:
            msg = "At least one frame description required."
            raise ValueError(msg)

        if len(frames) == 1:
            return frames[0]

        tag = "\n<frame description>\n{}\n</frame description>"
        prompt = " ".join([VIDEO_ANALYZER_PROMPT] + [tag.format(i) for i in frames])

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
            REASONING_DELIMITERS.get("end", "")
        )
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
            if len(retention) > VIDEO_ANALYZER_SNAPSHOTS_TO_KEEP:
                old = retention.popleft()
                try:
                    await self.hass.async_add_executor_job(old.unlink)
                    LOGGER.debug("[%s] Deleted old snapshot: %s", camera_id, old)
                except OSError as err:
                    LOGGER.warning("[%s] Failed to delete %s: %s", camera_id, old, err)

    async def _process_snapshot_queue(self, camera_id: str) -> None:
        queue = self._snapshot_queues.get(camera_id)
        if not queue:
            return

        batch: list[Path] = []
        try:
            while True:
                batch.append(queue.get_nowait())
        except QueueEmpty:
            pass

        camera_name = camera_id.split(".")[-1]

        frame_descriptions: list[str] = []
        for path in batch:
            try:
                async with aiofiles.open(path, "rb") as file:
                    data = await file.read()
                async with async_timeout.timeout(30):
                    desc = await analyze_image(
                        self.entry.runtime_data.vision_model,
                        data,
                        None,
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

        async with async_timeout.timeout(60):
            msg = await self._generate_summary(frame_descriptions, camera_id)

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

        async with async_timeout.timeout(10):
            await self.entry.runtime_data.store.aput(
                namespace=("video_analysis", camera_name),
                key=batch[0].name,
                value={"content": msg, "snapshots": [str(p) for p in batch]},
            )

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
            done, pending = await asyncio.wait(tasks_to_await, timeout=5)
            for task in pending:
                LOGGER.warning("Task did not cancel in time: %s", task)

        LOGGER.info("Video analyzer stopped.")

    def is_running(self) -> bool:
        """Check if the video analyzer is running."""
        return hasattr(self, "_cancel_track") and hasattr(self, "_cancel_listen")


# ---------------- Home Assistant entrypoints ----------------


async def async_setup_entry(hass: HomeAssistant, entry: HGAConfigEntry) -> bool:  # noqa: PLR0912, PLR0915
    """Set up Home Generative Agent from a config entry."""
    hass.data.setdefault(DOMAIN, {})

    api_key = entry.data.get(CONF_API_KEY)

    # Health checks (fast, non-fatal)
    health_timeout = 2.0
    ollama_ok, openai_ok = await asyncio.gather(
        _ollama_healthy(hass, OLLAMA_URL, timeout_s=health_timeout),
        _openai_healthy(hass, api_key, timeout_s=health_timeout),
    )

    http_client = get_async_client(hass)

    # Instantiate providers.
    openai_provider: RunnableSerializable[LanguageModelInput, BaseMessage] | None = None
    if openai_ok:
        try:
            openai_provider = ChatOpenAI(
                api_key=api_key,
                timeout=10,
                http_async_client=http_client,
            ).configurable_fields(
                model_name=ConfigurableField(id="model_name"),
                temperature=ConfigurableField(id="temperature"),
                top_p=ConfigurableField(id="top_p"),
                max_tokens=ConfigurableField(id="max_tokens"),
            )
        except Exception:
            LOGGER.exception("OpenAI provider init failed; continuing without it.")

    ollama_provider: RunnableSerializable[LanguageModelInput, BaseMessage] | None = None
    if ollama_ok:
        try:
            ollama_provider = ChatOllama(
                model=RECOMMENDED_OLLAMA_CHAT_MODEL,
                base_url=_ensure_http_url(OLLAMA_URL),
            ).configurable_fields(
                model=ConfigurableField(id="model"),
                format=ConfigurableField(id="format"),
                temperature=ConfigurableField(id="temperature"),
                top_p=ConfigurableField(id="top_p"),
                num_predict=ConfigurableField(id="num_predict"),
                num_ctx=ConfigurableField(id="num_ctx"),
            )
        except Exception:
            LOGGER.exception("Ollama provider init failed; continuing without it.")

    # Embeddings: instantiate both, then select based on provider
    openai_embeddings: OpenAIEmbeddings | None = None
    if openai_ok:
        try:
            openai_embeddings = OpenAIEmbeddings(
                api_key=api_key,
                model=entry.options.get(
                    CONF_OPENAI_EMBEDDING_MODEL, RECOMMENDED_OPENAI_EMBEDDING_MODEL
                ),
                dimensions=EMBEDDING_MODEL_DIMS,
            )
        except Exception:
            LOGGER.exception("OpenAI embeddings init failed; continuing without them.")

    ollama_embeddings: OllamaEmbeddings | None = None
    if ollama_ok:
        try:
            ollama_embeddings = OllamaEmbeddings(
                model=entry.options.get(
                    CONF_OLLAMA_EMBEDDING_MODEL, RECOMMENDED_OLLAMA_EMBEDDING_MODEL
                ),
                base_url=_ensure_http_url(OLLAMA_URL),
                num_ctx=EMBEDDING_MODEL_CTX,
            )
        except Exception:
            LOGGER.exception("Ollama embeddings init failed; continuing without them.")

    # Choose active embedding provider
    embedding_model: OpenAIEmbeddings | OllamaEmbeddings | None = None
    embedding_provider = entry.options.get(
        CONF_EMBEDDING_MODEL_PROVIDER, RECOMMENDED_EMBEDDING_MODEL_PROVIDER
    )
    index_config: PostgresIndexConfig | None = None
    if embedding_provider == "openai":
        embedding_model = openai_embeddings
    else:
        embedding_model = ollama_embeddings

    if embedding_model is None:
        LOGGER.warning(
            "No embeddings provider available; vector store will be limited.",
        )
    else:
        index_config = PostgresIndexConfig(
            embed=partial(_generate_embeddings, embedding_model),
            dims=EMBEDDING_MODEL_DIMS,
            fields=["content"],
        )

    # Open Postgres
    connection_kwargs = {
        "autocommit": True,
        "prepare_threshold": 0,
        "row_factory": dict_row,
    }
    pool: AsyncConnectionPool[AsyncConnection[DictRow]] = AsyncConnectionPool(
        conninfo=DB_URI, min_size=5, max_size=20, kwargs=connection_kwargs, open=False
    )
    try:
        await pool.open()
    except PoolTimeout:
        LOGGER.exception("Error opening postgresql db.")
        return False

    store = AsyncPostgresStore(
        pool,
        index=index_config if index_config else None,
    )
    # NOTE: must call .setup() the first time store is used.
    # await store.setup()  # noqa: ERA001

    # Initialize database for thread-based (short-term) memory.
    checkpointer = AsyncPostgresSaver(pool)
    # NOTE: must call .setup() the first time store is used.
    # await checkpointer.setup()  # noqa: ERA001

    # ----- Choose concrete models for roles from constants -----

    # CHAT
    chat_provider = entry.options.get(
        CONF_CHAT_MODEL_PROVIDER, RECOMMENDED_CHAT_MODEL_PROVIDER
    )
    chat_temp = entry.options.get(
        CONF_CHAT_MODEL_TEMPERATURE, RECOMMENDED_CHAT_MODEL_TEMPERATURE
    )
    if chat_provider == "openai":
        chat_model = (openai_provider or NullChat()).with_config(
            config={
                "configurable": {
                    "model_name": entry.options.get(
                        CONF_OPENAI_CHAT_MODEL, RECOMMENDED_OPENAI_CHAT_MODEL
                    ),
                    "temperature": chat_temp,
                    "top_p": CHAT_MODEL_TOP_P,
                    "max_tokens": CHAT_MODEL_MAX_TOKENS,
                }
            }
        )
    else:
        chat_model = (ollama_provider or NullChat()).with_config(
            config={
                "configurable": {
                    "model": entry.options.get(
                        CONF_OLLAMA_CHAT_MODEL, RECOMMENDED_OLLAMA_CHAT_MODEL
                    ),
                    "temperature": chat_temp,
                    "top_p": CHAT_MODEL_TOP_P,
                    "num_predict": CHAT_MODEL_MAX_TOKENS,
                    "num_ctx": CHAT_MODEL_NUM_CTX,
                }
            }
        )

    # VLM
    vlm_provider = entry.options.get(CONF_VLM_PROVIDER, RECOMMENDED_VLM_PROVIDER)
    vlm_temp = entry.options.get(CONF_VLM_TEMPERATURE, RECOMMENDED_VLM_TEMPERATURE)
    if vlm_provider == "openai":
        vision_model = (openai_provider or NullChat()).with_config(
            config={
                "configurable": {
                    "model_name": entry.options.get(
                        CONF_OPENAI_VLM, RECOMMENDED_OPENAI_VLM
                    ),
                    "temperature": vlm_temp,
                    "top_p": VLM_TOP_P,
                    "max_tokens": VLM_NUM_PREDICT,
                }
            }
        )
    else:
        vision_model = (ollama_provider or NullChat()).with_config(
            config={
                "configurable": {
                    "model": entry.options.get(CONF_OLLAMA_VLM, RECOMMENDED_OLLAMA_VLM),
                    "temperature": vlm_temp,
                    "top_p": VLM_TOP_P,
                    "num_predict": VLM_NUM_PREDICT,
                    "num_ctx": VLM_NUM_CTX,
                }
            }
        )

    # SUMMARIZATION
    sum_provider = entry.options.get(
        CONF_SUMMARIZATION_MODEL_PROVIDER, RECOMMENDED_SUMMARIZATION_MODEL_PROVIDER
    )
    sum_temp = entry.options.get(
        CONF_SUMMARIZATION_MODEL_TEMPERATURE,
        RECOMMENDED_SUMMARIZATION_MODEL_TEMPERATURE,
    )
    if sum_provider == "openai":
        summarization_model = (openai_provider or NullChat()).with_config(
            config={
                "configurable": {
                    "model_name": entry.options.get(
                        CONF_OPENAI_SUMMARIZATION_MODEL,
                        RECOMMENDED_OPENAI_SUMMARIZATION_MODEL,
                    ),
                    "temperature": sum_temp,
                    "top_p": SUMMARIZATION_MODEL_TOP_P,
                    "max_tokens": SUMMARIZATION_MODEL_PREDICT,
                }
            }
        )
    else:
        summarization_model = (ollama_provider or NullChat()).with_config(
            config={
                "configurable": {
                    "model": entry.options.get(
                        CONF_OLLAMA_SUMMARIZATION_MODEL,
                        RECOMMENDED_OLLAMA_SUMMARIZATION_MODEL,
                    ),
                    "temperature": sum_temp,
                    "top_p": SUMMARIZATION_MODEL_TOP_P,
                    "num_predict": SUMMARIZATION_MODEL_PREDICT,
                    "num_ctx": SUMMARIZATION_MODEL_CTX,
                }
            }
        )

    # Video analyzer
    video_analyzer = VideoAnalyzer(hass, entry)
    if entry.options.get(CONF_VIDEO_ANALYZER_MODE) != "disable":
        video_analyzer.start()

    # Save runtime data.
    entry.runtime_data = HGAData(
        chat_model=chat_model,
        vision_model=vision_model,
        summarization_model=summarization_model,
        store=store,
        video_analyzer=video_analyzer,
        checkpointer=checkpointer,
        pool=pool,
    )

    await hass.config_entries.async_forward_entry_setups(entry, PLATFORMS)

    msg = (
        "Home Generative Agent initialized with the following models: "
        "chat=%s, vlm=%s, summarization=%s. "
        "OpenAI ok=%s, Ollama ok=%s."
    )
    LOGGER.info(
        msg,
        chat_provider,
        vlm_provider,
        sum_provider,
        openai_ok,
        ollama_ok,
    )
    return True


async def async_unload_entry(hass: HomeAssistant, entry: HGAConfigEntry) -> bool:
    """Unload the config entry."""
    await entry.runtime_data.pool.close()
    await entry.runtime_data.video_analyzer.stop()
    await hass.config_entries.async_unload_platforms(entry, PLATFORMS)
    return True

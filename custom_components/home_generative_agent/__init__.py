"""Home Generative Agent Initialization."""

from __future__ import annotations

import asyncio
import calendar
import io
import logging
import re
from collections import deque
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, Any
from urllib.parse import urljoin

import aiofiles
import async_timeout
import homeassistant.util.dt as dt_util
import httpx
import numpy as np
import voluptuous as vol
from homeassistant.components.camera.const import DOMAIN as CAMERA_DOMAIN
from homeassistant.config_entries import ConfigEntry
from homeassistant.const import CONF_API_KEY, Platform
from homeassistant.core import HomeAssistant, ServiceCall, callback
from homeassistant.exceptions import HomeAssistantError
from homeassistant.helpers import config_validation as cv
from homeassistant.helpers.event import async_track_time_interval
from homeassistant.helpers.httpx_client import get_async_client
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import ConfigurableField
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.store.postgres import AsyncPostgresStore
from langgraph.store.postgres.base import PostgresIndexConfig
from PIL import Image
from psycopg.rows import DictRow, dict_row
from psycopg_pool import AsyncConnectionPool, PoolTimeout

from .const import (
    CHAT_MODEL_MAX_TOKENS,
    CHAT_MODEL_NUM_CTX,
    CHAT_MODEL_REPEAT_PENALTY,
    CHAT_MODEL_TOP_P,
    CONF_CHAT_MODEL_PROVIDER,
    CONF_CHAT_MODEL_TEMPERATURE,
    CONF_DB_BOOTSTRAPPED,
    CONF_DB_URI,
    CONF_EMBEDDING_MODEL_PROVIDER,
    CONF_FACE_API_URL,
    CONF_FACE_RECOGNITION_MODE,
    CONF_GEMINI_API_KEY,
    CONF_GEMINI_CHAT_MODEL,
    CONF_GEMINI_EMBEDDING_MODEL,
    CONF_GEMINI_SUMMARIZATION_MODEL,
    CONF_GEMINI_VLM,
    CONF_NOTIFY_SERVICE,
    CONF_OLLAMA_CHAT_MODEL,
    CONF_OLLAMA_EMBEDDING_MODEL,
    CONF_OLLAMA_SUMMARIZATION_MODEL,
    CONF_OLLAMA_URL,
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
    DOMAIN,
    EMBEDDING_MODEL_CTX,
    EMBEDDING_MODEL_DIMS,
    HTTP_STATUS_BAD_REQUEST,
    REASONING_DELIMITERS,
    RECOMMENDED_CHAT_MODEL_PROVIDER,
    RECOMMENDED_CHAT_MODEL_TEMPERATURE,
    RECOMMENDED_DB_URI,
    RECOMMENDED_EMBEDDING_MODEL_PROVIDER,
    RECOMMENDED_FACE_API_URL,
    RECOMMENDED_FACE_RECOGNITION_MODE,
    RECOMMENDED_GEMINI_CHAT_MODEL,
    RECOMMENDED_GEMINI_EMBEDDING_MODEL,
    RECOMMENDED_GEMINI_SUMMARIZATION_MODEL,
    RECOMMENDED_GEMINI_VLM,
    RECOMMENDED_OLLAMA_CHAT_MODEL,
    RECOMMENDED_OLLAMA_EMBEDDING_MODEL,
    RECOMMENDED_OLLAMA_SUMMARIZATION_MODEL,
    RECOMMENDED_OLLAMA_URL,
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
    SUMMARIZATION_MODEL_REPEAT_PENALTY,
    SUMMARIZATION_MODEL_TOP_P,
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
    VLM_NUM_CTX,
    VLM_NUM_PREDICT,
    VLM_REPEAT_PENALTY,
    VLM_TOP_P,
)
from .tools import analyze_image

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator, Callable

    from homeassistant.config_entries import ConfigEntry
    from homeassistant.core import Event, HomeAssistant
    from langchain_core.language_models import BaseMessage, LanguageModelInput
    from langchain_core.runnables.base import RunnableSerializable
    from psycopg import AsyncConnection, AsyncCursor

LOGGER = logging.getLogger(__name__)

PLATFORMS = (Platform.CONVERSATION,)

SERVICE_ENROLL_PERSON = "enroll_person"

ENROLL_SCHEMA = vol.Schema(
    {
        vol.Required("name"): cv.string,
        vol.Required("file_path"): cv.isfile,  # path to a local uploaded snapshot
    }
)


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


async def _gemini_healthy(
    hass: HomeAssistant, api_key: str | None, timeout_s: float = 2.0
) -> bool:
    """Quick reachability check for Gemini."""
    if not api_key:
        LOGGER.warning("Gemini health check skipped: missing API key.")
        return False
    client = get_async_client(hass)
    try:
        # Public models list endpoint requires key query param
        url = f"https://generativelanguage.googleapis.com/v1/models?key={api_key}"
        async with async_timeout.timeout(timeout_s):
            resp = await client.get(url)
        if resp.status_code < HTTP_STATUS_BAD_REQUEST:
            return True
        LOGGER.warning("Gemini health check HTTP %s: %s", resp.status_code, resp.text)
    except Exception as err:  # noqa: BLE001
        LOGGER.warning("Gemini health check failed: %s", err)
    return False


async def _generate_embeddings(
    emb: OpenAIEmbeddings | OllamaEmbeddings | GoogleGenerativeAIEmbeddings,
    texts: Sequence[str],
) -> list[list[float]]:
    """
    Generate embeddings from a list of text.

    Note: Gemini supports custom output dimensionality; force 1024 to match our index.
    """
    texts_list = list(texts)
    # If it's Gemini, ask for 1024-d explicitly
    if isinstance(emb, GoogleGenerativeAIEmbeddings):
        return await emb.aembed_documents(
            texts_list, output_dimensionality=EMBEDDING_MODEL_DIMS
        )
    # OpenAI / Ollama paths unchanged
    return await emb.aembed_documents(texts_list)


def _discover_mobile_notify_service(hass: HomeAssistant) -> str | None:
    # Returns just the service *name* (e.g., "mobile_app_lindos_iphone")
    services = hass.services.async_services().get("notify", {})
    # `services` is a dict mapping service_name -> Service object
    for svc_name in services:
        if svc_name.startswith("mobile_app_"):
            return svc_name
    return None


async def _bootstrap_db_once(
    hass: HomeAssistant,
    entry: ConfigEntry,
    store: AsyncPostgresStore,
    checkpointer: AsyncPostgresSaver,
) -> None:
    if entry.data.get(CONF_DB_BOOTSTRAPPED):
        return

    # First time only
    await store.setup()
    await checkpointer.setup()

    # Persist the flag so it survives restarts
    hass.config_entries.async_update_entry(
        entry, data={**entry.data, CONF_DB_BOOTSTRAPPED: True}
    )


async def migration_1(cur: AsyncCursor[DictRow]) -> None:
    """Migration 1: create person_gallery and bump schema version."""
    await cur.execute(
        """
        CREATE TABLE IF NOT EXISTS person_gallery (
            id SERIAL PRIMARY KEY,
            name TEXT NOT NULL,
            embedding VECTOR(512),
            added_at TIMESTAMP DEFAULT NOW()
        )
        """
    )
    await cur.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_person_gallery_embedding
        ON person_gallery USING ivfflat (embedding vector_l2_ops)
        WITH (lists = 100)
        """
    )

    # bump version to 1
    await cur.execute(
        "INSERT INTO hga_schema_version (id, version) VALUES (1, 1) "
        "ON CONFLICT(id) DO UPDATE SET version = 1"
    )


# Registry of migrations in order
MIGRATIONS: dict[int, Callable] = {
    1: migration_1,
    # 2: migration_2, 3: migration_3, ...
}


async def _migrate_person_gallery_db_schema(
    pool: AsyncConnectionPool[AsyncConnection[DictRow]],
) -> None:
    async with pool.connection() as conn, conn.cursor() as cur:
        # --- Schema version table ---
        await cur.execute(
            """
            CREATE TABLE IF NOT EXISTS hga_schema_version (
                id INTEGER PRIMARY KEY DEFAULT 1,
                version INTEGER NOT NULL
            )
            """
        )
        await cur.execute("SELECT version FROM hga_schema_version WHERE id = 1")

        row: dict[str, int] | None = await cur.fetchone()
        current_version: int = row["version"] if row else 0

        # --- Run pending migrations ---
        for version in sorted(MIGRATIONS.keys()):
            if current_version < version:
                await MIGRATIONS[version](cur)
                current_version = version


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

    # Face recognition
    face_api_url: str
    face_mode: str
    person_gallery: Any


# After HGAData is defined, we can specialize the ConfigEntry type.
type HGAConfigEntry = ConfigEntry[HGAData]

# ---------------- Face Recognition DAO ---------------

Embedding = Sequence[float]
FACE_EMBEDDING_DIMS = 512


class PersonGalleryDAO:
    """Access layer for person recognition using pgvector cosine distance."""

    def __init__(
        self, pool: AsyncConnectionPool[AsyncConnection[DictRow]]
    ) -> None:  # RuffANN001 fixed
        """Initialize with a psycopg_pool AsyncConnectionPool."""
        self.pool = pool

    def _normalize(self, embedding: Embedding) -> list[float]:
        """Return L2-normalized list of floats."""
        v = np.array(embedding, dtype=np.float32)
        norm = float(np.linalg.norm(v))
        if norm == 0.0:
            msg = "Zero vector cannot be normalized"
            raise ValueError(msg)
        return (v / norm).tolist()

    def _as_pgvector(self, embedding: Embedding) -> str:
        """Format floats as pgvector literal string with full precision."""
        if len(embedding) != FACE_EMBEDDING_DIMS:
            msg = f"Expected {FACE_EMBEDDING_DIMS} dims, got {len(embedding)}"
            raise ValueError(msg)
        return "[" + ",".join(format(float(x), ".17g") for x in embedding) + "]"

    async def enroll_from_image(
        self, face_api_url: str, name: str, image_bytes: bytes
    ) -> bool:
        """Detect face in image, extract embedding, and add to gallery."""
        async with httpx.AsyncClient() as client:
            resp = await client.post(
                urljoin(face_api_url.rstrip("/") + "/", "analyze"),
                files={"file": ("snapshot.jpg", image_bytes, "image/jpeg")},
            )
            resp.raise_for_status()
            data = resp.json()

        faces = data.get("faces", [])
        if not faces:
            LOGGER.warning("No face detected for enrollment of %s", name)
            return False

        emb: Embedding = faces[0]["embedding"]
        await self.add_person(name, emb)
        LOGGER.info("Enrolled new person '%s' with embedding.", name)
        return True

    async def add_person(self, name: str, embedding: Embedding) -> None:
        """Insert normalized embedding into gallery (cosine distance ready)."""
        normed = self._normalize(embedding)
        vec_str = self._as_pgvector(normed)

        sql = """
            INSERT INTO public.person_gallery (name, embedding)
            VALUES (%s, %s::vector(512))
        """
        async with (
            self.pool.connection() as conn,
            conn.cursor(row_factory=dict_row) as cur,
        ):
            await cur.execute(sql, (name, vec_str))
            LOGGER.debug("Inserted %s into person_gallery", name)

    async def recognize_person(
        self, embedding: Embedding, threshold: float = 0.7
    ) -> str:
        """Return best cosine match or 'Unknown Person'."""
        normed = self._normalize(embedding)
        vec_str = self._as_pgvector(normed)

        sql = """
            SELECT name, embedding <=> %s::vector(512) AS distance
            FROM public.person_gallery
            ORDER BY distance
            LIMIT 1
        """
        async with (
            self.pool.connection() as conn,
            conn.cursor(row_factory=dict_row) as cur,
        ):
            await cur.execute(sql, (vec_str,))
            row = await cur.fetchone()

            if not row:
                LOGGER.error("Recognition query returned no rows")
                return "Unknown Person"

            dist = float(row["distance"])
            LOGGER.debug("Closest match=%s cosine_distance=%.6f", row["name"], dist)
            return row["name"] if dist < threshold else "Unknown Person"

    async def list_people(self) -> list[str]:
        """Return list of distinct enrolled person names."""
        async with (
            self.pool.connection() as conn,
            conn.cursor(row_factory=dict_row) as cur,
        ):
            await cur.execute(
                "SELECT DISTINCT name FROM public.person_gallery ORDER BY name"
            )
            rows = await cur.fetchall()
            return [r["name"] for r in rows]


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
        self._last_recognized: dict[str, list[str]] = {}

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
                    "repeat_penalty": SUMMARIZATION_MODEL_REPEAT_PENALTY,
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
        # Prefer configured option; fall back to discovery
        full_service = self.entry.options.get(CONF_NOTIFY_SERVICE)
        if full_service and full_service.startswith("notify."):
            domain, service = full_service.split(".", 1)
        else:
            service = _discover_mobile_notify_service(self.hass)
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
        model = self.entry.runtime_data.summarization_model.with_config(
            config=self._sum_model_cfg
        )
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
            if len(retention) > VIDEO_ANALYZER_SNAPSHOTS_TO_KEEP:
                old = retention.popleft()
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
        notify_img = Path("/media/local") / Path(*batch[len(batch) // 2].parts[-3:])

        mode = self.entry.options.get(CONF_VIDEO_ANALYZER_MODE)
        if mode == "notify_on_anomaly":
            first_snapshot = batch[0].parts[-1]
            LOGGER.debug("[%s] First snapshot: %s", camera_id, first_snapshot)
            if await self._is_anomaly(camera_name, msg, first_snapshot):
                LOGGER.debug("[%s] Video is an anomaly!", camera_id)
                await self._send_notification(msg, camera_name, notify_img)
        else:
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


# ---------------- Home Assistant entrypoints ----------------


async def async_setup_entry(hass: HomeAssistant, entry: HGAConfigEntry) -> bool:  # noqa: C901, PLR0912, PLR0915
    """Set up Home Generative Agent from a config entry."""
    hass.data.setdefault(DOMAIN, {})

    # Options override data.
    # entry.options is where data lives if added or changed later via Options.
    # Merging options over data guarantees you see the most recent values.
    conf = {**entry.data, **entry.options}
    api_key = conf.get(CONF_API_KEY)
    gemini_key = conf.get(CONF_GEMINI_API_KEY)
    ollama_url = conf.get(CONF_OLLAMA_URL, RECOMMENDED_OLLAMA_URL)
    ollama_url = _ensure_http_url(ollama_url)
    face_api_url = conf.get(CONF_FACE_API_URL, RECOMMENDED_FACE_API_URL)
    face_api_url = _ensure_http_url(face_api_url)

    # Health checks (fast, non-fatal)
    health_timeout = 2.0
    ollama_ok, openai_ok, gemini_ok = await asyncio.gather(
        _ollama_healthy(hass, ollama_url, timeout_s=health_timeout),
        _openai_healthy(hass, api_key, timeout_s=health_timeout),
        _gemini_healthy(hass, gemini_key, timeout_s=health_timeout),
    )

    http_client = get_async_client(hass)

    # Instantiate providers.
    openai_provider: RunnableSerializable[LanguageModelInput, BaseMessage] | None = None
    if openai_ok:
        try:
            openai_provider = ChatOpenAI(
                api_key=api_key,
                timeout=120,
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
                base_url=ollama_url,
            ).configurable_fields(
                model=ConfigurableField(id="model"),
                format=ConfigurableField(id="format"),
                temperature=ConfigurableField(id="temperature"),
                top_p=ConfigurableField(id="top_p"),
                num_predict=ConfigurableField(id="num_predict"),
                num_ctx=ConfigurableField(id="num_ctx"),
                repeat_penalty=ConfigurableField(id="repeat_penalty"),
            )
        except Exception:
            LOGGER.exception("Ollama provider init failed; continuing without it.")

    gemini_provider: RunnableSerializable[LanguageModelInput, BaseMessage] | None = None
    if gemini_ok:
        try:
            gemini_provider = ChatGoogleGenerativeAI(
                api_key=gemini_key,
                model=RECOMMENDED_GEMINI_CHAT_MODEL,  # default, will get overridden
            ).configurable_fields(
                model=ConfigurableField(id="model"),
                temperature=ConfigurableField(id="temperature"),
                top_p=ConfigurableField(id="top_p"),
                max_output_tokens=ConfigurableField(id="max_tokens"),
            )
        except Exception:
            LOGGER.exception("Gemini provider init failed; continuing without it.")

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
                base_url=ollama_url,
                num_ctx=EMBEDDING_MODEL_CTX,
            )
        except Exception:
            LOGGER.exception("Ollama embeddings init failed; continuing without them.")

    gemini_embeddings: GoogleGenerativeAIEmbeddings | None = None
    if gemini_ok:
        try:
            gemini_embeddings = GoogleGenerativeAIEmbeddings(
                google_api_key=gemini_key,
                model=entry.options.get(
                    CONF_GEMINI_EMBEDDING_MODEL, RECOMMENDED_GEMINI_EMBEDDING_MODEL
                ),
            )
        except Exception:
            LOGGER.exception("Gemini embeddings init failed; continuing without them.")

    # Choose active embedding provider
    embedding_model: (
        OpenAIEmbeddings | OllamaEmbeddings | GoogleGenerativeAIEmbeddings | None
    ) = None
    embedding_provider = entry.options.get(
        CONF_EMBEDDING_MODEL_PROVIDER, RECOMMENDED_EMBEDDING_MODEL_PROVIDER
    )
    index_config: PostgresIndexConfig | None = None
    if embedding_provider == "openai":
        embedding_model = openai_embeddings
    elif embedding_provider == "gemini":
        embedding_model = gemini_embeddings
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
    db_uri = conf.get(CONF_DB_URI, RECOMMENDED_DB_URI)
    pool: AsyncConnectionPool[AsyncConnection[DictRow]] = AsyncConnectionPool(
        conninfo=db_uri, min_size=5, max_size=20, kwargs=connection_kwargs, open=False
    )
    try:
        await pool.open()
    except PoolTimeout:
        LOGGER.exception("Error opening postgresql db.")
        return False

    # Initialize database for long-term memory.
    store = AsyncPostgresStore(
        pool,
        index=index_config if index_config else None,
    )
    # Initialize database for thread-based (short-term) memory.
    checkpointer = AsyncPostgresSaver(pool)
    # First-time setup (if needed)
    await _bootstrap_db_once(hass, entry, store, checkpointer)

    # Migrate person gallery DB schema (if needed)
    try:
        await _migrate_person_gallery_db_schema(pool)
    except Exception:
        LOGGER.exception("Error migrating person_gallery database schema.")
        return False

    person_gallery = PersonGalleryDAO(pool)

    async with pool.connection() as conn, conn.cursor(row_factory=dict_row) as cur:
        await cur.execute("""
                SELECT current_database() AS db,
                    current_user     AS usr,
                    inet_server_addr()::text AS host,
                    inet_server_port()       AS port,
                    current_schemas(true)    AS schemas,
                    current_setting('search_path', true) AS search_path
            """)
        env = await cur.fetchone()
        if env:
            LOGGER.info(
                "DB env: db=%s user=%s host=%s port=%s schemas=%s search_path=%s",
                env["db"],
                env["usr"],
                env["host"],
                env["port"],
                env["schemas"],
                env["search_path"],
            )

        await cur.execute("SELECT COUNT(*) AS total FROM public.person_gallery")
        resp = await cur.fetchone()
        if resp:
            LOGGER.info("Gallery rows visible to this connection: %s", resp["total"])

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
    elif chat_provider == "gemini":
        chat_model = (gemini_provider or NullChat()).with_config(
            config={
                "configurable": {
                    "model": entry.options.get(
                        CONF_GEMINI_CHAT_MODEL, RECOMMENDED_GEMINI_CHAT_MODEL
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
                    "repeat_penalty": CHAT_MODEL_REPEAT_PENALTY,
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
    elif vlm_provider == "gemini":
        vision_model = (gemini_provider or NullChat()).with_config(
            config={
                "configurable": {
                    "model": entry.options.get(CONF_GEMINI_VLM, RECOMMENDED_GEMINI_VLM),
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
                    "repeat_penalty": VLM_REPEAT_PENALTY,
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
    elif sum_provider == "gemini":
        summarization_model = (gemini_provider or NullChat()).with_config(
            config={
                "configurable": {
                    "model": entry.options.get(
                        CONF_GEMINI_SUMMARIZATION_MODEL,
                        RECOMMENDED_GEMINI_SUMMARIZATION_MODEL,
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
                    "repeat_penalty": SUMMARIZATION_MODEL_REPEAT_PENALTY,
                }
            }
        )

    # Video analyzer
    video_analyzer = VideoAnalyzer(hass, entry)
    if entry.options.get(CONF_VIDEO_ANALYZER_MODE) != "disable":
        video_analyzer.start()

    # Face recognition
    face_mode = entry.options.get(
        CONF_FACE_RECOGNITION_MODE, RECOMMENDED_FACE_RECOGNITION_MODE
    )

    # Save runtime data.
    entry.runtime_data = HGAData(
        chat_model=chat_model,
        vision_model=vision_model,
        summarization_model=summarization_model,
        store=store,
        video_analyzer=video_analyzer,
        checkpointer=checkpointer,
        pool=pool,
        face_mode=face_mode,
        face_api_url=face_api_url,
        person_gallery=person_gallery,
    )

    await hass.config_entries.async_forward_entry_setups(entry, PLATFORMS)

    msg = (
        "Home Generative Agent initialized with the following models: "
        "chat=%s, vlm=%s, summarization=%s. embeddings=%s. "
        "OpenAI ok=%s, Ollama ok=%s, Gemini ok=%s."
    )
    LOGGER.info(
        msg,
        chat_provider,
        vlm_provider,
        sum_provider,
        embedding_provider,
        openai_ok,
        ollama_ok,
        gemini_ok,
    )

    async def _handle_enroll_person(call: ServiceCall) -> None:
        name: str = call.data["name"]
        file_path: str = call.data["file_path"]

        try:
            async with aiofiles.open(file_path, "rb") as f:
                img_bytes = await f.read()
        except OSError as err:
            msg = f"Could not read file: {err}"
            raise HomeAssistantError(msg) from err

        dao: PersonGalleryDAO = entry.runtime_data.person_gallery
        ok = await dao.enroll_from_image(
            entry.runtime_data.face_api_url, name, img_bytes
        )
        if not ok:
            msg = f"No face found in image for {name}"
            raise HomeAssistantError(msg)

    hass.services.async_register(
        DOMAIN,
        SERVICE_ENROLL_PERSON,
        _handle_enroll_person,
        schema=ENROLL_SCHEMA,
    )

    return True


async def async_unload_entry(hass: HomeAssistant, entry: HGAConfigEntry) -> bool:
    """Unload the config entry."""
    await entry.runtime_data.pool.close()
    await entry.runtime_data.video_analyzer.stop()
    await hass.config_entries.async_unload_platforms(entry, PLATFORMS)
    return True

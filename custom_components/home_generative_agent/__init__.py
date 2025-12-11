"""Home Generative Agent Initialization."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Sequence
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import aiofiles
import voluptuous as vol
from homeassistant.components.camera.const import DOMAIN as CAMERA_DOMAIN
from homeassistant.config_entries import ConfigEntry
from homeassistant.const import CONF_API_KEY, Platform
from homeassistant.core import HomeAssistant
from homeassistant.exceptions import HomeAssistantError
from homeassistant.helpers import config_validation as cv
from homeassistant.helpers.httpx_client import get_async_client
from homeassistant.helpers.target import (
    TargetSelectorData,
    async_extract_referenced_entity_ids,
)
from homeassistant.util import dt as dt_util
from langchain_core.runnables import ConfigurableField
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.store.postgres import AsyncPostgresStore
from langgraph.store.postgres.base import PostgresIndexConfig
from psycopg.rows import DictRow, dict_row
from psycopg_pool import AsyncConnectionPool, PoolTimeout

from .agent.tools import analyze_image
from .const import (
    CHAT_MODEL_MAX_TOKENS,
    CHAT_MODEL_REPEAT_PENALTY,
    CHAT_MODEL_TOP_P,
    CONF_CHAT_MODEL_PROVIDER,
    CONF_CHAT_MODEL_TEMPERATURE,
    CONF_DB_BOOTSTRAPPED,
    CONF_DB_URI,
    CONF_EMBEDDING_MODEL_PROVIDER,
    CONF_FACE_API_URL,
    CONF_FACE_RECOGNITION,
    CONF_GEMINI_API_KEY,
    CONF_GEMINI_CHAT_MODEL,
    CONF_GEMINI_EMBEDDING_MODEL,
    CONF_GEMINI_SUMMARIZATION_MODEL,
    CONF_GEMINI_VLM,
    CONF_OLLAMA_CHAT_CONTEXT_SIZE,
    CONF_OLLAMA_CHAT_KEEPALIVE,
    CONF_OLLAMA_CHAT_MODEL,
    CONF_OLLAMA_EMBEDDING_MODEL,
    CONF_OLLAMA_REASONING,
    CONF_OLLAMA_SUMMARIZATION_CONTEXT_SIZE,
    CONF_OLLAMA_SUMMARIZATION_KEEPALIVE,
    CONF_OLLAMA_SUMMARIZATION_MODEL,
    CONF_OLLAMA_URL,
    CONF_OLLAMA_VLM,
    CONF_OLLAMA_VLM_CONTEXT_SIZE,
    CONF_OLLAMA_VLM_KEEPALIVE,
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
    RECOMMENDED_CHAT_MODEL_PROVIDER,
    RECOMMENDED_CHAT_MODEL_TEMPERATURE,
    RECOMMENDED_DB_URI,
    RECOMMENDED_EMBEDDING_MODEL_PROVIDER,
    RECOMMENDED_FACE_API_URL,
    RECOMMENDED_FACE_RECOGNITION,
    RECOMMENDED_GEMINI_CHAT_MODEL,
    RECOMMENDED_GEMINI_EMBEDDING_MODEL,
    RECOMMENDED_GEMINI_SUMMARIZATION_MODEL,
    RECOMMENDED_GEMINI_VLM,
    RECOMMENDED_OLLAMA_CHAT_KEEPALIVE,
    RECOMMENDED_OLLAMA_CHAT_MODEL,
    RECOMMENDED_OLLAMA_CONTEXT_SIZE,
    RECOMMENDED_OLLAMA_EMBEDDING_MODEL,
    RECOMMENDED_OLLAMA_REASONING,
    RECOMMENDED_OLLAMA_SUMMARIZATION_KEEPALIVE,
    RECOMMENDED_OLLAMA_SUMMARIZATION_MODEL,
    RECOMMENDED_OLLAMA_URL,
    RECOMMENDED_OLLAMA_VLM,
    RECOMMENDED_OLLAMA_VLM_KEEPALIVE,
    RECOMMENDED_OPENAI_CHAT_MODEL,
    RECOMMENDED_OPENAI_EMBEDDING_MODEL,
    RECOMMENDED_OPENAI_SUMMARIZATION_MODEL,
    RECOMMENDED_OPENAI_VLM,
    RECOMMENDED_SUMMARIZATION_MODEL_PROVIDER,
    RECOMMENDED_SUMMARIZATION_MODEL_TEMPERATURE,
    RECOMMENDED_VLM_PROVIDER,
    RECOMMENDED_VLM_TEMPERATURE,
    SIGNAL_HGA_NEW_LATEST,
    SIGNAL_HGA_RECOGNIZED,
    SUMMARIZATION_MIRO_STAT,
    SUMMARIZATION_MODEL_PREDICT,
    SUMMARIZATION_MODEL_REPEAT_PENALTY,
    SUMMARIZATION_MODEL_TOP_P,
    VIDEO_ANALYZER_SNAPSHOT_ROOT,
    VLM_MIRO_STAT,
    VLM_NUM_PREDICT,
    VLM_REPEAT_PENALTY,
    VLM_TOP_P,
)
from .core.migrations import migrate_person_gallery
from .core.person_gallery import PersonGalleryDAO
from .core.runtime import HGAConfigEntry, HGAData
from .core.utils import (
    configured_ollama_urls,
    dispatch_on_loop,
    ensure_http_url,
    gemini_healthy,
    generate_embeddings,
    ollama_healthy,
    ollama_url_for_category,
    openai_healthy,
    reasoning_field,
)
from .core.video_analyzer import VideoAnalyzer
from .core.video_helpers import latest_target, publish_latest_atomic

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator

    from homeassistant.config_entries import ConfigEntry
    from homeassistant.core import HomeAssistant, ServiceCall
    from homeassistant.helpers.typing import ConfigType
    from langchain_core.language_models import BaseMessage, LanguageModelInput
    from langchain_core.runnables.base import RunnableSerializable
    from psycopg import AsyncConnection

LOGGER = logging.getLogger(__name__)

PLATFORMS = (Platform.CONVERSATION, "image", "sensor")

SERVICE_ENROLL_PERSON = "enroll_person"

ENROLL_SCHEMA = vol.Schema(
    {
        vol.Required("name"): cv.string,
        vol.Required("file_path"): cv.isfile,  # path to a local uploaded snapshot
    }
)

Embedding = Sequence[float]


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


def _register_services(hass: HomeAssistant, entry: HGAConfigEntry) -> None:
    """Register integration services."""

    async def _handle_save_and_analyze_snapshot(call: ServiceCall) -> None:
        """Capture a snapshot, analyze it, and publish 'latest' for targeted cameras."""
        # 1) Resolve Target selector (entity/device/area/label[/floor])
        # raw 'target:' from the service call (may be absent)
        raw_target: ConfigType = cast("ConfigType", call.data.get("target", {}))
        selector = TargetSelectorData(raw_target)
        refs = async_extract_referenced_entity_ids(hass, selector, expand_group=True)
        entity_ids = sorted(refs.referenced | refs.indirectly_referenced)

        # Back-compat: honor legacy data.entity_id if someone passed it
        legacy = call.data.get("entity_id")
        if isinstance(legacy, str):
            entity_ids.append(legacy)
        elif isinstance(legacy, list):
            entity_ids.extend(str(e) for e in legacy)

        # Keep only cameras
        entity_ids = [e for e in entity_ids if e.startswith("camera.")]
        if not entity_ids:
            msg = "Please target at least one camera entity."
            raise HomeAssistantError(msg)

        protect_minutes = int(call.data.get("protect_minutes", 30))

        # Access models from config entry runtime
        vision_model = entry.runtime_data.vision_model
        video_analyzer = entry.runtime_data.video_analyzer

        for camera_id in entity_ids:
            # 2) Compute destination and ensure directory exists
            dst = latest_target(Path(VIDEO_ANALYZER_SNAPSHOT_ROOT), camera_id)
            await hass.async_add_executor_job(
                partial(dst.parent.mkdir, parents=True, exist_ok=True)
            )

            # 3) Capture snapshot to a temp filename
            tmp_name = f"snapshot_{dt_util.utcnow().strftime('%Y%m%d_%H%M%S')}.jpg"
            tmp_path = dst.parent / tmp_name

            await hass.services.async_call(
                CAMERA_DOMAIN,
                "snapshot",
                {"entity_id": camera_id, "filename": str(tmp_path)},
                blocking=True,
            )

            # 4) Read bytes for analysis
            try:
                async with aiofiles.open(tmp_path, "rb") as f:
                    img_bytes = await f.read()
            except OSError as err:
                msg = f"Failed to read snapshot {tmp_path}: {err}"
                raise HomeAssistantError(msg) from err

            # 5) Run AI analysis (summary) and face recognition if available.
            summary: str | None
            summary = await analyze_image(vision_model, img_bytes, None, prev_text=None)

            people: list[str] = []
            people = await video_analyzer.recognize_faces(img_bytes, camera_id)

            # 6) Publish to _latest atomically
            await publish_latest_atomic(hass, tmp_path, dst)

            # 7) Notify listeners (image entity + sensor + event)
            iso = dt_util.utcnow().isoformat()

            # Image entity (path update)
            dispatch_on_loop(hass, SIGNAL_HGA_NEW_LATEST, camera_id, str(dst))

            # Sensor + metadata (recognized + summary + timestamp + latest path)
            dispatch_on_loop(
                hass,
                SIGNAL_HGA_RECOGNIZED,
                camera_id,
                people,
                summary,
                iso,
                str(dst),
            )

            # Bus event for anything else listening, and a safety net for image entities
            hass.bus.async_fire(
                "hga_last_event_frame",
                {
                    "camera_id": camera_id,
                    "summary": summary,
                    "path": str(tmp_path),
                    "latest": str(dst),
                },
            )

            video_analyzer.protect_notify_image(dst, ttl_sec=protect_minutes * 60)

    # Register the service
    hass.services.async_register(
        DOMAIN, "save_and_analyze_snapshot", _handle_save_and_analyze_snapshot
    )


async def async_setup_entry(hass: HomeAssistant, entry: HGAConfigEntry) -> bool:  # noqa: C901, PLR0912, PLR0915
    """Set up Home Generative Agent from a config entry."""
    hass.data.setdefault(DOMAIN, {})

    _register_services(hass, entry)

    # Options override data.
    # entry.options is where data lives if added or changed later via Options.
    # Merging options over data guarantees you see the most recent values.
    conf = {**entry.data, **entry.options}
    api_key = conf.get(CONF_API_KEY)
    gemini_key = conf.get(CONF_GEMINI_API_KEY)
    base_ollama_url = ensure_http_url(conf.get(CONF_OLLAMA_URL, RECOMMENDED_OLLAMA_URL))
    ollama_chat_url = (
        ollama_url_for_category(conf, "chat", fallback=base_ollama_url)
        or base_ollama_url
    )
    ollama_vlm_url = (
        ollama_url_for_category(conf, "vlm", fallback=base_ollama_url)
        or base_ollama_url
    )
    ollama_sum_url = (
        ollama_url_for_category(conf, "summarization", fallback=base_ollama_url)
        or base_ollama_url
    )
    face_api_url = conf.get(CONF_FACE_API_URL, RECOMMENDED_FACE_API_URL)
    face_api_url = ensure_http_url(face_api_url)

    # Health checks (fast, non-fatal)
    health_timeout = 2.0
    ollama_urls = configured_ollama_urls(conf, fallback=base_ollama_url)
    ollama_health: dict[str, bool] = {}
    if ollama_urls:
        ollama_results = await asyncio.gather(
            *(
                ollama_healthy(hass, url, timeout_s=health_timeout)
                for url in ollama_urls
            )
        )
        ollama_health = dict(zip(ollama_urls, ollama_results, strict=False))

    openai_ok, gemini_ok = await asyncio.gather(
        openai_healthy(hass, api_key, timeout_s=health_timeout),
        gemini_healthy(hass, gemini_key, timeout_s=health_timeout),
    )
    ollama_any_ok = any(ollama_health.values())

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

    def _build_ollama_provider(
        url: str,
    ) -> RunnableSerializable[LanguageModelInput, BaseMessage]:
        return ChatOllama(
            model=RECOMMENDED_OLLAMA_CHAT_MODEL,
            base_url=url,
        ).configurable_fields(
            model=ConfigurableField(id="model"),
            format=ConfigurableField(id="format"),
            temperature=ConfigurableField(id="temperature"),
            top_p=ConfigurableField(id="top_p"),
            num_predict=ConfigurableField(id="num_predict"),
            num_ctx=ConfigurableField(id="num_ctx"),
            repeat_penalty=ConfigurableField(id="repeat_penalty"),
            reasoning=ConfigurableField(id="reasoning"),
            mirostat=ConfigurableField(id="mirostat"),
            keep_alive=ConfigurableField(id="keep_alive"),
        )

    ollama_providers: dict[
        str, RunnableSerializable[LanguageModelInput, BaseMessage]
    ] = {}
    for url, healthy in ollama_health.items():
        if not healthy:
            continue
        try:
            ollama_providers[url] = _build_ollama_provider(url)
        except Exception:
            LOGGER.exception(
                "Ollama provider init failed for %s; continuing without it.", url
            )

    gemini_provider: RunnableSerializable[LanguageModelInput, BaseMessage] | None = None
    if gemini_ok:
        try:
            gemini_provider = ChatGoogleGenerativeAI(
                api_key=gemini_key,
                model=RECOMMENDED_GEMINI_CHAT_MODEL,
            ).configurable_fields(
                model=ConfigurableField(id="model"),
                temperature=ConfigurableField(id="temperature"),
                top_p=ConfigurableField(id="top_p"),
                max_output_tokens=ConfigurableField(id="max_output_tokens"),
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
    if ollama_health.get(base_ollama_url):
        try:
            ollama_embeddings = OllamaEmbeddings(
                model=entry.options.get(
                    CONF_OLLAMA_EMBEDDING_MODEL, RECOMMENDED_OLLAMA_EMBEDDING_MODEL
                ),
                base_url=base_ollama_url,
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
            embed=partial(generate_embeddings, embedding_model),
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
        await migrate_person_gallery(pool)
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

    ollama_reasoning: bool = entry.options.get(
        CONF_OLLAMA_REASONING, RECOMMENDED_OLLAMA_REASONING
    )
    chat_ollama_provider = ollama_providers.get(ollama_chat_url)
    vlm_ollama_provider = ollama_providers.get(ollama_vlm_url)
    summarization_ollama_provider = ollama_providers.get(ollama_sum_url)

    # CHAT
    chat_provider = entry.options.get(
        CONF_CHAT_MODEL_PROVIDER, RECOMMENDED_CHAT_MODEL_PROVIDER
    )
    chat_temp = entry.options.get(
        CONF_CHAT_MODEL_TEMPERATURE, RECOMMENDED_CHAT_MODEL_TEMPERATURE
    )
    ollama_chat_keep_alive = entry.options.get(
        CONF_OLLAMA_CHAT_KEEPALIVE, RECOMMENDED_OLLAMA_CHAT_KEEPALIVE
    )
    ollama_chat_context_size = entry.options.get(
        CONF_OLLAMA_CHAT_CONTEXT_SIZE, RECOMMENDED_OLLAMA_CONTEXT_SIZE
    )
    ollama_chat_model_options = {
        "temperature": chat_temp,
        "top_p": CHAT_MODEL_TOP_P,
        "num_predict": CHAT_MODEL_MAX_TOKENS,
        "num_ctx": ollama_chat_context_size,
        "repeat_penalty": CHAT_MODEL_REPEAT_PENALTY,
        "keep_alive": ollama_chat_keep_alive,
    }
    if chat_provider == "openai":
        chat_model = (openai_provider or NullChat()).with_config(
            config={
                "configurable": {
                    "model_name": entry.options.get(
                        CONF_OPENAI_CHAT_MODEL, RECOMMENDED_OPENAI_CHAT_MODEL
                    ),
                    "temperature": chat_temp,
                    "top_p": CHAT_MODEL_TOP_P,
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
                }
            }
        )
    else:
        ollama_chat_model = entry.options.get(
            CONF_OLLAMA_CHAT_MODEL, RECOMMENDED_OLLAMA_CHAT_MODEL
        )
        rf_chat = reasoning_field(model=ollama_chat_model, enabled=ollama_reasoning)
        ollama_chat_model_options = {**ollama_chat_model_options, **rf_chat}
        chat_model = (chat_ollama_provider or NullChat()).with_config(
            config={
                "configurable": {
                    "model": ollama_chat_model,
                    "temperature": chat_temp,
                    "top_p": CHAT_MODEL_TOP_P,
                    "num_predict": CHAT_MODEL_MAX_TOKENS,
                    "num_ctx": ollama_chat_context_size,
                    "repeat_penalty": CHAT_MODEL_REPEAT_PENALTY,
                    "keep_alive": ollama_chat_keep_alive,
                    **rf_chat,
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
                }
            }
        )
    else:
        ollama_vlm = entry.options.get(CONF_OLLAMA_VLM, RECOMMENDED_OLLAMA_VLM)
        rf_vlm = reasoning_field(model=ollama_vlm, enabled=ollama_reasoning)
        vision_model = (vlm_ollama_provider or NullChat()).with_config(
            config={
                "configurable": {
                    "model": ollama_vlm,
                    "temperature": vlm_temp,
                    "top_p": VLM_TOP_P,
                    "num_predict": VLM_NUM_PREDICT,
                    "num_ctx": entry.options.get(
                        CONF_OLLAMA_VLM_CONTEXT_SIZE, RECOMMENDED_OLLAMA_CONTEXT_SIZE
                    ),
                    "repeat_penalty": VLM_REPEAT_PENALTY,
                    "mirostat": VLM_MIRO_STAT,
                    "keep_alive": entry.options.get(
                        CONF_OLLAMA_VLM_KEEPALIVE, RECOMMENDED_OLLAMA_VLM_KEEPALIVE
                    ),
                    **rf_vlm,
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
                }
            }
        )
    else:
        ollama_summarization_model = entry.options.get(
            CONF_OLLAMA_SUMMARIZATION_MODEL,
            RECOMMENDED_OLLAMA_SUMMARIZATION_MODEL,
        )
        rf_summarization = reasoning_field(
            model=ollama_summarization_model, enabled=ollama_reasoning
        )
        summarization_model = (summarization_ollama_provider or NullChat()).with_config(
            config={
                "configurable": {
                    "model": ollama_summarization_model,
                    "temperature": sum_temp,
                    "top_p": SUMMARIZATION_MODEL_TOP_P,
                    "num_predict": SUMMARIZATION_MODEL_PREDICT,
                    "num_ctx": entry.options.get(
                        CONF_OLLAMA_SUMMARIZATION_CONTEXT_SIZE,
                        RECOMMENDED_OLLAMA_CONTEXT_SIZE,
                    ),
                    "repeat_penalty": SUMMARIZATION_MODEL_REPEAT_PENALTY,
                    "mirostat": SUMMARIZATION_MIRO_STAT,
                    "keep_alive": entry.options.get(
                        CONF_OLLAMA_SUMMARIZATION_KEEPALIVE,
                        RECOMMENDED_OLLAMA_SUMMARIZATION_KEEPALIVE,
                    ),
                    **rf_summarization,
                }
            }
        )

    video_analyzer = VideoAnalyzer(hass, entry)

    face_recognition = entry.options.get(
        CONF_FACE_RECOGNITION, RECOMMENDED_FACE_RECOGNITION
    )

    # Save runtime data.
    entry.runtime_data = HGAData(
        chat_model=chat_model,
        chat_model_options=ollama_chat_model_options,
        vision_model=vision_model,
        summarization_model=summarization_model,
        store=store,
        video_analyzer=video_analyzer,
        checkpointer=checkpointer,
        pool=pool,
        face_recognition=face_recognition,
        face_api_url=face_api_url,
        person_gallery=person_gallery,
        pending_actions={},
    )

    await hass.config_entries.async_forward_entry_setups(entry, PLATFORMS)

    if entry.options.get(CONF_VIDEO_ANALYZER_MODE) != "disable":
        video_analyzer.start()

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
        ollama_any_ok,
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

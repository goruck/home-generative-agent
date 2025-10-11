"""Home Generative Agent Initialization."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Sequence
from functools import partial
from typing import TYPE_CHECKING, Any

import aiofiles
import voluptuous as vol
from homeassistant.config_entries import ConfigEntry
from homeassistant.const import CONF_API_KEY, Platform
from homeassistant.core import HomeAssistant
from homeassistant.exceptions import HomeAssistantError
from homeassistant.helpers import config_validation as cv
from homeassistant.helpers.httpx_client import get_async_client
from langchain_core.runnables import ConfigurableField
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
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
    VLM_NUM_CTX,
    VLM_NUM_PREDICT,
    VLM_REPEAT_PENALTY,
    VLM_TOP_P,
)
from .core.migrations import migrate_person_gallery
from .core.person_gallery import PersonGalleryDAO
from .core.runtime import HGAConfigEntry, HGAData
from .core.utils import (
    ensure_http_url,
    gemini_healthy,
    generate_embeddings,
    ollama_healthy,
    openai_healthy,
)
from .core.video_analyzer import VideoAnalyzer

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator

    from homeassistant.config_entries import ConfigEntry
    from homeassistant.core import HomeAssistant, ServiceCall
    from langchain_core.language_models import BaseMessage, LanguageModelInput
    from langchain_core.runnables.base import RunnableSerializable
    from psycopg import AsyncConnection

LOGGER = logging.getLogger(__name__)

PLATFORMS = (Platform.CONVERSATION,)

SERVICE_ENROLL_PERSON = "enroll_person"

ENROLL_SCHEMA = vol.Schema(
    {
        vol.Required("name"): cv.string,
        vol.Required("file_path"): cv.isfile,  # path to a local uploaded snapshot
    }
)


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


Embedding = Sequence[float]


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
    ollama_url = ensure_http_url(ollama_url)
    face_api_url = conf.get(CONF_FACE_API_URL, RECOMMENDED_FACE_API_URL)
    face_api_url = ensure_http_url(face_api_url)

    # Health checks (fast, non-fatal)
    health_timeout = 2.0
    ollama_ok, openai_ok, gemini_ok = await asyncio.gather(
        ollama_healthy(hass, ollama_url, timeout_s=health_timeout),
        openai_healthy(hass, api_key, timeout_s=health_timeout),
        gemini_healthy(hass, gemini_key, timeout_s=health_timeout),
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

    video_analyzer = VideoAnalyzer(hass, entry)

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

"""Home Generative Agent Initialization."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Mapping, Sequence
from functools import partial
from pathlib import Path
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, cast
from urllib.parse import unquote

import aiofiles
import httpx
import voluptuous as vol
from homeassistant.components import media_source
from homeassistant.components.camera.const import DOMAIN as CAMERA_DOMAIN
from homeassistant.components.http import StaticPathConfig
from homeassistant.config_entries import ConfigEntry, ConfigSubentry
from homeassistant.const import (
    CONF_API_KEY,
    CONF_HOST,
    CONF_PASSWORD,
    CONF_PORT,
    CONF_USERNAME,
    EVENT_HOMEASSISTANT_STARTED,
    EVENT_HOMEASSISTANT_STOP,
    Platform,
)
from homeassistant.core import HomeAssistant, SupportsResponse
from homeassistant.exceptions import HomeAssistantError
from homeassistant.helpers import config_validation as cv
from homeassistant.helpers.httpx_client import get_async_client
from homeassistant.helpers.network import get_url
from homeassistant.helpers.target import (
    TargetSelectorData,
    async_extract_referenced_entity_ids,
)
from homeassistant.util import dt as dt_util
from langchain_core.runnables import ConfigurableField
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langgraph.checkpoint.memory import MemorySaver
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.store.postgres import AsyncPostgresStore
from langgraph.store.postgres.base import PostgresIndexConfig
from psycopg.rows import DictRow, dict_row
from psycopg_pool import AsyncConnectionPool, PoolTimeout
from pydantic import SecretStr

from .agent.tools import analyze_image
from .audit.store import AuditStore
from .const import (
    CHAT_MODEL_MAX_TOKENS,
    CHAT_MODEL_REPEAT_PENALTY,
    CHAT_MODEL_TOP_P,
    CONF_CHAT_MODEL_PROVIDER,
    CONF_CHAT_MODEL_TEMPERATURE,
    CONF_DB_BOOTSTRAPPED,
    CONF_DB_NAME,
    CONF_DB_PARAMS,
    CONF_DB_URI,
    CONF_EMBEDDING_MODEL_PROVIDER,
    CONF_EXPLAIN_ENABLED,
    CONF_FACE_API_URL,
    CONF_FACE_RECOGNITION,
    CONF_FEATURE_MODEL,
    CONF_FEATURE_MODEL_CONTEXT_SIZE,
    CONF_FEATURE_MODEL_KEEPALIVE,
    CONF_FEATURE_MODEL_NAME,
    CONF_FEATURE_MODEL_REASONING,
    CONF_FEATURE_MODEL_TEMPERATURE,
    CONF_GEMINI_API_KEY,
    CONF_GEMINI_CHAT_MODEL,
    CONF_GEMINI_EMBEDDING_MODEL,
    CONF_GEMINI_SUMMARIZATION_MODEL,
    CONF_GEMINI_VLM,
    CONF_NOTIFY_SERVICE,
    CONF_OLLAMA_CHAT_CONTEXT_SIZE,
    CONF_OLLAMA_CHAT_KEEPALIVE,
    CONF_OLLAMA_CHAT_MODEL,
    CONF_OLLAMA_CHAT_URL,
    CONF_OLLAMA_EMBEDDING_MODEL,
    CONF_OLLAMA_REASONING,
    CONF_OLLAMA_SUMMARIZATION_CONTEXT_SIZE,
    CONF_OLLAMA_SUMMARIZATION_KEEPALIVE,
    CONF_OLLAMA_SUMMARIZATION_MODEL,
    CONF_OLLAMA_SUMMARIZATION_URL,
    CONF_OLLAMA_URL,
    CONF_OLLAMA_VLM,
    CONF_OLLAMA_VLM_CONTEXT_SIZE,
    CONF_OLLAMA_VLM_KEEPALIVE,
    CONF_OLLAMA_VLM_URL,
    CONF_OPENAI_CHAT_MODEL,
    CONF_OPENAI_EMBEDDING_MODEL,
    CONF_OPENAI_SUMMARIZATION_MODEL,
    CONF_OPENAI_VLM,
    CONF_SENTINEL_COOLDOWN_MINUTES,
    CONF_SENTINEL_DISCOVERY_ENABLED,
    CONF_SENTINEL_DISCOVERY_INTERVAL_SECONDS,
    CONF_SENTINEL_DISCOVERY_MAX_RECORDS,
    CONF_SENTINEL_ENABLED,
    CONF_SENTINEL_ENTITY_COOLDOWN_MINUTES,
    CONF_SENTINEL_INTERVAL_SECONDS,
    CONF_SUMMARIZATION_MODEL_PROVIDER,
    CONF_SUMMARIZATION_MODEL_TEMPERATURE,
    CONF_VECTORS_BOOTSTRAPPED,
    CONF_VIDEO_ANALYZER_MODE,
    CONF_VLM_PROVIDER,
    CONF_VLM_TEMPERATURE,
    CONFIG_ENTRY_VERSION,
    DEFAULT_FEATURE_TYPES,
    DOMAIN,
    EMBEDDING_MODEL_CTX,
    EMBEDDING_MODEL_DIMS,
    FEATURE_CATEGORY_MAP,
    FEATURE_NAMES,
    HGA_CARD_STATIC_PATH,
    HGA_CARD_STATIC_PATH_LEGACY,
    MODEL_CATEGORY_SPECS,
    RECOMMENDED_CHAT_MODEL_PROVIDER,
    RECOMMENDED_CHAT_MODEL_TEMPERATURE,
    RECOMMENDED_DB_HOST,
    RECOMMENDED_DB_NAME,
    RECOMMENDED_DB_PARAMS,
    RECOMMENDED_DB_PASSWORD,
    RECOMMENDED_DB_PORT,
    RECOMMENDED_DB_USERNAME,
    RECOMMENDED_EMBEDDING_MODEL_PROVIDER,
    RECOMMENDED_EXPLAIN_ENABLED,
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
    RECOMMENDED_SENTINEL_COOLDOWN_MINUTES,
    RECOMMENDED_SENTINEL_DISCOVERY_ENABLED,
    RECOMMENDED_SENTINEL_DISCOVERY_INTERVAL_SECONDS,
    RECOMMENDED_SENTINEL_DISCOVERY_MAX_RECORDS,
    RECOMMENDED_SENTINEL_ENABLED,
    RECOMMENDED_SENTINEL_ENTITY_COOLDOWN_MINUTES,
    RECOMMENDED_SENTINEL_INTERVAL_SECONDS,
    RECOMMENDED_SUMMARIZATION_MODEL_PROVIDER,
    RECOMMENDED_SUMMARIZATION_MODEL_TEMPERATURE,
    RECOMMENDED_VLM_PROVIDER,
    RECOMMENDED_VLM_TEMPERATURE,
    SIGNAL_HGA_NEW_LATEST,
    SIGNAL_HGA_RECOGNIZED,
    SUBENTRY_TYPE_DATABASE,
    SUBENTRY_TYPE_FEATURE,
    SUBENTRY_TYPE_MODEL_PROVIDER,
    SUBENTRY_TYPE_SENTINEL,
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
from .core.db_utils import parse_postgres_uri
from .core.migrations import migrate_person_gallery
from .core.person_gallery import PersonGalleryDAO
from .core.runtime import HGAConfigEntry, HGAData
from .core.subentry_resolver import (
    build_database_uri_from_entry,
    legacy_feature_configs,
    legacy_model_provider_configs,
    resolve_model_provider_configs,
    resolve_runtime_options,
)
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
from .explain.llm_explain import LLMExplainer
from .http import EnrollPersonView
from .notify.actions import ActionHandler
from .notify.dispatcher import NotificationDispatcher
from .sentinel.discovery_engine import SentinelDiscoveryEngine
from .sentinel.discovery_semantic import candidate_semantic_key, rule_semantic_key
from .sentinel.discovery_store import DiscoveryStore
from .sentinel.engine import SentinelEngine
from .sentinel.proposal_store import ProposalStore
from .sentinel.proposal_templates import normalize_candidate
from .sentinel.rule_registry import RuleRegistry
from .sentinel.suppression import SuppressionManager

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator

    from homeassistant.config_entries import ConfigEntry
    from homeassistant.core import HomeAssistant, ServiceCall
    from homeassistant.helpers.typing import ConfigType
    from langchain_core.language_models import BaseMessage, LanguageModelInput
    from langchain_core.runnables.base import RunnableSerializable
    from psycopg import AsyncConnection

    from .core.subentry_types import ModelProviderConfig

LOGGER = logging.getLogger(__name__)

_SERVICE_RESPONSE_ONLY = SupportsResponse.ONLY

PLATFORMS = (Platform.CONVERSATION, Platform.STT, "image", "sensor")

SERVICE_ENROLL_PERSON = "enroll_person"
SERVICE_GET_AUDIT_RECORDS = "get_audit_records"
SERVICE_GET_DISCOVERY_RECORDS = "get_discovery_records"
SERVICE_PROMOTE_DISCOVERY_CANDIDATE = "promote_discovery_candidate"
SERVICE_GET_PROPOSAL_DRAFTS = "get_proposal_drafts"
SERVICE_APPROVE_RULE_PROPOSAL = "approve_rule_proposal"
SERVICE_REJECT_RULE_PROPOSAL = "reject_rule_proposal"

ENROLL_SCHEMA = vol.Schema(
    {
        vol.Required("name"): cv.string,
        vol.Required("file_path"): cv.string,
    }
)

GET_AUDIT_SCHEMA = vol.Schema(
    {
        vol.Optional("limit", default=20): vol.Coerce(int),
    }
)

GET_DISCOVERY_SCHEMA = vol.Schema(
    {
        vol.Optional("limit", default=20): vol.Coerce(int),
    }
)

PROMOTE_DISCOVERY_SCHEMA = vol.Schema(
    {
        vol.Required("candidate_id"): cv.string,
        vol.Optional("notes"): cv.string,
    }
)

GET_PROPOSAL_SCHEMA = vol.Schema(
    {
        vol.Optional("limit", default=50): vol.Coerce(int),
    }
)

REVIEW_PROPOSAL_SCHEMA = vol.Schema(
    {
        vol.Required("candidate_id"): cv.string,
        vol.Optional("notes"): cv.string,
    }
)

Embedding = Sequence[float]


def _default_feature_payload(feature_type: str) -> dict[str, Any]:
    return {
        "feature_type": feature_type,
        "name": FEATURE_NAMES.get(feature_type, feature_type),
        "model_provider_id": None,
        CONF_FEATURE_MODEL: {},
        "config": {},
    }


def _provider_api_key(
    providers: Mapping[str, ModelProviderConfig], provider_type: str
) -> str | None:
    for provider in providers.values():
        if provider.provider_type != provider_type:
            continue
        settings = provider.data.get("settings", {})
        api_key = settings.get("api_key")
        if api_key:
            return str(api_key)
    return None


def _resolve_www_dir() -> str | None:
    www_dir = Path(__file__).resolve().parent / "www"
    if not www_dir.is_dir():
        return None
    return str(www_dir)


async def _read_enroll_image_bytes(hass: HomeAssistant, file_ref: str) -> bytes:
    if media_source.is_media_source_id(file_ref):
        if file_ref.startswith("media-source://media_source/local/"):
            relative = unquote(file_ref.split("/local/", 1)[1])
            local_path = Path(hass.config.path("media", relative))
            async with aiofiles.open(local_path, "rb") as f:
                return await f.read()

        media = await media_source.async_resolve_media(hass, file_ref, None)
        url = media.url
        if url.startswith("/"):
            url = f"{get_url(hass)}{url}"
        client = get_async_client(hass)
        response = await client.get(url)
        response.raise_for_status()
        return await response.aread()

    async with aiofiles.open(file_ref, "rb") as f:
        return await f.read()


def _default_model_data(category: str, provider_type: str) -> dict[str, Any]:
    spec = MODEL_CATEGORY_SPECS.get(category, {})
    model_data: dict[str, Any] = {}
    model_name = spec.get("recommended_models", {}).get(provider_type)
    if model_name:
        model_data[CONF_FEATURE_MODEL_NAME] = model_name
    temp = spec.get("recommended_temperature")
    if temp is not None:
        model_data[CONF_FEATURE_MODEL_TEMPERATURE] = temp
    if provider_type == "ollama":
        keepalive_map = {
            "chat": RECOMMENDED_OLLAMA_CHAT_KEEPALIVE,
            "vlm": RECOMMENDED_OLLAMA_VLM_KEEPALIVE,
            "summarization": RECOMMENDED_OLLAMA_SUMMARIZATION_KEEPALIVE,
        }
        keepalive = keepalive_map.get(category)
        if keepalive is not None:
            model_data[CONF_FEATURE_MODEL_KEEPALIVE] = keepalive
        model_data[CONF_FEATURE_MODEL_CONTEXT_SIZE] = RECOMMENDED_OLLAMA_CONTEXT_SIZE
        if category == "chat":
            model_data[CONF_FEATURE_MODEL_REASONING] = RECOMMENDED_OLLAMA_REASONING
    return model_data


def _ensure_default_feature_subentries(hass: HomeAssistant, entry: ConfigEntry) -> None:
    for feature_type in DEFAULT_FEATURE_TYPES:
        exists = any(
            s.subentry_type == SUBENTRY_TYPE_FEATURE
            and s.data.get("feature_type") == feature_type
            for s in entry.subentries.values()
        )
        if exists:
            continue
        payload = _default_feature_payload(feature_type)
        subentry = ConfigSubentry(
            subentry_type=SUBENTRY_TYPE_FEATURE,
            title=FEATURE_NAMES.get(feature_type, feature_type),
            unique_id=f"{entry.entry_id}_{feature_type}",
            data=MappingProxyType(payload),
        )
        hass.config_entries.async_add_subentry(entry, subentry)


def _assign_first_provider_if_needed(hass: HomeAssistant, entry: ConfigEntry) -> None:
    providers = [
        s
        for s in entry.subentries.values()
        if s.subentry_type == SUBENTRY_TYPE_MODEL_PROVIDER
    ]
    if len(providers) != 1:
        return
    provider = providers[0]
    provider_type = provider.data.get("provider_type", "ollama")

    for subentry in entry.subentries.values():
        if subentry.subentry_type != SUBENTRY_TYPE_FEATURE:
            continue
        data = dict(subentry.data)
        if data.get("model_provider_id"):
            continue
        feature_type = data.get("feature_type")
        category = (
            FEATURE_CATEGORY_MAP.get(feature_type)
            if isinstance(feature_type, str)
            else None
        )
        model_data = dict(data.get(CONF_FEATURE_MODEL, {}))
        if not model_data and category:
            model_data = _default_model_data(category, provider_type)
        data["model_provider_id"] = provider.subentry_id
        data[CONF_FEATURE_MODEL] = model_data
        hass.config_entries.async_update_subentry(  # type: ignore[attr-defined]
            entry, subentry, data=MappingProxyType(data), title=subentry.title
        )


def _ensure_default_sentinel_subentry(hass: HomeAssistant, entry: ConfigEntry) -> None:
    """Ensure a singleton sentinel subentry exists for deterministic settings."""
    exists = any(
        s.subentry_type == SUBENTRY_TYPE_SENTINEL for s in entry.subentries.values()
    )
    if exists:
        return

    payload = {
        CONF_SENTINEL_ENABLED: RECOMMENDED_SENTINEL_ENABLED,
        CONF_SENTINEL_INTERVAL_SECONDS: RECOMMENDED_SENTINEL_INTERVAL_SECONDS,
        CONF_SENTINEL_COOLDOWN_MINUTES: RECOMMENDED_SENTINEL_COOLDOWN_MINUTES,
        CONF_SENTINEL_ENTITY_COOLDOWN_MINUTES: (
            RECOMMENDED_SENTINEL_ENTITY_COOLDOWN_MINUTES
        ),
        CONF_SENTINEL_DISCOVERY_ENABLED: RECOMMENDED_SENTINEL_DISCOVERY_ENABLED,
        CONF_SENTINEL_DISCOVERY_INTERVAL_SECONDS: (
            RECOMMENDED_SENTINEL_DISCOVERY_INTERVAL_SECONDS
        ),
        CONF_SENTINEL_DISCOVERY_MAX_RECORDS: RECOMMENDED_SENTINEL_DISCOVERY_MAX_RECORDS,
        CONF_EXPLAIN_ENABLED: RECOMMENDED_EXPLAIN_ENABLED,
    }
    subentry = ConfigSubentry(
        subentry_type=SUBENTRY_TYPE_SENTINEL,
        title="Sentinel",
        unique_id=f"{entry.entry_id}_sentinel",
        data=MappingProxyType(payload),
    )
    hass.config_entries.async_add_subentry(entry, subentry)


# Database and vector index bootstrapping.
# store.setup() only runs the vector migrations when store.index_config is set.
# If index_config is None (no embeddings configured yet), setup() runs only the
# base store migrations and skips VECTOR_MIGRATIONS, which is where store_vectors
# and its ANN index are created. Adding a model provider with embeddings later
# will trigger a separate setup() call that creates the vector index then if not
# already configured during initial bootstrap.


async def _bootstrap_db_once(
    hass: HomeAssistant,
    entry: ConfigEntry,
    store: AsyncPostgresStore,
    checkpointer: AsyncPostgresSaver,
) -> None:
    """Bootstrap database if needed."""
    if entry.data.get(CONF_DB_BOOTSTRAPPED):
        return

    await store.setup()
    await checkpointer.setup()

    hass.config_entries.async_update_entry(
        entry, data={**entry.data, CONF_DB_BOOTSTRAPPED: True}
    )


async def _bootstrap_vectors_once(
    hass: HomeAssistant,
    entry: ConfigEntry,
    store: AsyncPostgresStore,
) -> None:
    """Bootstrap vector index if needed."""
    if not store.index_config:
        return
    if entry.data.get(CONF_VECTORS_BOOTSTRAPPED):
        return

    await store.setup()

    hass.config_entries.async_update_entry(
        entry, data={**entry.data, CONF_VECTORS_BOOTSTRAPPED: True}
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


class NullStore:
    """Non-throwing fallback for memory store operations."""

    async def asearch(self, *_args: Any, **_kwargs: Any) -> list[Any]:
        """Return an empty result set."""
        return []

    async def aput(self, *_args: Any, **_kwargs: Any) -> None:
        """No-op write."""
        return


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
    _ensure_default_feature_subentries(hass, entry)
    _ensure_default_sentinel_subentry(hass, entry)
    _assign_first_provider_if_needed(hass, entry)

    # Resolve effective options (data + options + subentries).
    options = resolve_runtime_options(entry)
    conf = dict(options)
    providers = resolve_model_provider_configs(entry, options)
    api_key = conf.get(CONF_API_KEY) or _provider_api_key(providers, "openai")
    openai_secret = SecretStr(api_key) if api_key else None
    gemini_key = conf.get(CONF_GEMINI_API_KEY) or _provider_api_key(providers, "gemini")
    gemini_secret = SecretStr(gemini_key) if gemini_key else None
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
                api_key=openai_secret,
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
                api_key=gemini_secret,
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
                api_key=openai_secret,
                model=options.get(
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
                model=options.get(
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
                google_api_key=gemini_secret,
                model=options.get(
                    CONF_GEMINI_EMBEDDING_MODEL, RECOMMENDED_GEMINI_EMBEDDING_MODEL
                ),
            )
        except Exception:
            LOGGER.exception("Gemini embeddings init failed; continuing without them.")

    # Choose active embedding provider
    embedding_model: (
        OpenAIEmbeddings | OllamaEmbeddings | GoogleGenerativeAIEmbeddings | None
    ) = None
    embedding_provider = options.get(
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

    db_uri = build_database_uri_from_entry(entry)

    if db_uri is not None:
        pool: AsyncConnectionPool[AsyncConnection[DictRow]] | None = (
            AsyncConnectionPool(
                conninfo=db_uri,
                min_size=5,
                max_size=20,
                kwargs=connection_kwargs,
                open=False,
            )
        )
        try:
            await pool.open()
        except PoolTimeout:
            LOGGER.exception("Error opening postgresql db.")
            return False

        try:
            # Initialize database for long-term memory.
            store = AsyncPostgresStore(
                pool,
                index=index_config or None,
            )
            # Initialize database for thread-based (short-term) memory.
            checkpointer = AsyncPostgresSaver(pool)
            # First-time setup (if needed)
            await _bootstrap_db_once(hass, entry, store, checkpointer)
            await _bootstrap_vectors_once(hass, entry, store)

            # Migrate person gallery DB schema (if needed)
            try:
                await migrate_person_gallery(pool)
            except Exception:
                LOGGER.exception("Error migrating person_gallery database schema.")
                raise

            person_gallery = PersonGalleryDAO(pool, hass)

            async with (
                pool.connection() as conn,
                conn.cursor(row_factory=dict_row) as cur,
            ):
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
                        """
                        DB env: db=%s user=%s host=%s port=%s schemas=%s search_path=%s
                        """,
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
                    LOGGER.info(
                        "Gallery rows visible to this connection: %s", resp["total"]
                    )
        except Exception:
            LOGGER.exception("Postgresql setup failed; closing pool.")
            await pool.close()
            return False
    else:
        person_gallery = None
        checkpointer = MemorySaver()
        pool = None
        store = NullStore()

    # ----- Choose concrete models for roles from constants -----

    ollama_reasoning: bool = options.get(
        CONF_OLLAMA_REASONING, RECOMMENDED_OLLAMA_REASONING
    )
    chat_ollama_provider = ollama_providers.get(ollama_chat_url)
    vlm_ollama_provider = ollama_providers.get(ollama_vlm_url)
    summarization_ollama_provider = ollama_providers.get(ollama_sum_url)

    # CHAT
    chat_provider = options.get(
        CONF_CHAT_MODEL_PROVIDER, RECOMMENDED_CHAT_MODEL_PROVIDER
    )
    chat_temp = options.get(
        CONF_CHAT_MODEL_TEMPERATURE, RECOMMENDED_CHAT_MODEL_TEMPERATURE
    )
    ollama_chat_keep_alive = options.get(
        CONF_OLLAMA_CHAT_KEEPALIVE, RECOMMENDED_OLLAMA_CHAT_KEEPALIVE
    )
    ollama_chat_context_size = options.get(
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
                    "model_name": options.get(
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
                    "model": options.get(
                        CONF_GEMINI_CHAT_MODEL, RECOMMENDED_GEMINI_CHAT_MODEL
                    ),
                    "temperature": chat_temp,
                    "top_p": CHAT_MODEL_TOP_P,
                }
            }
        )
    else:
        ollama_chat_model = options.get(
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
    vlm_provider = options.get(CONF_VLM_PROVIDER, RECOMMENDED_VLM_PROVIDER)
    vlm_temp = options.get(CONF_VLM_TEMPERATURE, RECOMMENDED_VLM_TEMPERATURE)
    if vlm_provider == "openai":
        vision_model = (openai_provider or NullChat()).with_config(
            config={
                "configurable": {
                    "model_name": options.get(CONF_OPENAI_VLM, RECOMMENDED_OPENAI_VLM),
                    "temperature": vlm_temp,
                    "top_p": VLM_TOP_P,
                }
            }
        )
    elif vlm_provider == "gemini":
        vision_model = (gemini_provider or NullChat()).with_config(
            config={
                "configurable": {
                    "model": options.get(CONF_GEMINI_VLM, RECOMMENDED_GEMINI_VLM),
                    "temperature": vlm_temp,
                    "top_p": VLM_TOP_P,
                }
            }
        )
    else:
        ollama_vlm = options.get(CONF_OLLAMA_VLM, RECOMMENDED_OLLAMA_VLM)
        rf_vlm = reasoning_field(model=ollama_vlm, enabled=ollama_reasoning)
        vision_model = (vlm_ollama_provider or NullChat()).with_config(
            config={
                "configurable": {
                    "model": ollama_vlm,
                    "temperature": vlm_temp,
                    "top_p": VLM_TOP_P,
                    "num_predict": VLM_NUM_PREDICT,
                    "num_ctx": options.get(
                        CONF_OLLAMA_VLM_CONTEXT_SIZE, RECOMMENDED_OLLAMA_CONTEXT_SIZE
                    ),
                    "repeat_penalty": VLM_REPEAT_PENALTY,
                    "mirostat": VLM_MIRO_STAT,
                    "keep_alive": options.get(
                        CONF_OLLAMA_VLM_KEEPALIVE, RECOMMENDED_OLLAMA_VLM_KEEPALIVE
                    ),
                    **rf_vlm,
                }
            }
        )

    # SUMMARIZATION
    sum_provider = options.get(
        CONF_SUMMARIZATION_MODEL_PROVIDER, RECOMMENDED_SUMMARIZATION_MODEL_PROVIDER
    )
    sum_temp = options.get(
        CONF_SUMMARIZATION_MODEL_TEMPERATURE,
        RECOMMENDED_SUMMARIZATION_MODEL_TEMPERATURE,
    )
    if sum_provider == "openai":
        summarization_model = (openai_provider or NullChat()).with_config(
            config={
                "configurable": {
                    "model_name": options.get(
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
                    "model": options.get(
                        CONF_GEMINI_SUMMARIZATION_MODEL,
                        RECOMMENDED_GEMINI_SUMMARIZATION_MODEL,
                    ),
                    "temperature": sum_temp,
                    "top_p": SUMMARIZATION_MODEL_TOP_P,
                }
            }
        )
    else:
        ollama_summarization_model = options.get(
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
                    "num_ctx": options.get(
                        CONF_OLLAMA_SUMMARIZATION_CONTEXT_SIZE,
                        RECOMMENDED_OLLAMA_CONTEXT_SIZE,
                    ),
                    "repeat_penalty": SUMMARIZATION_MODEL_REPEAT_PENALTY,
                    "mirostat": SUMMARIZATION_MIRO_STAT,
                    "keep_alive": options.get(
                        CONF_OLLAMA_SUMMARIZATION_KEEPALIVE,
                        RECOMMENDED_OLLAMA_SUMMARIZATION_KEEPALIVE,
                    ),
                    **rf_summarization,
                }
            }
        )

    video_analyzer = VideoAnalyzer(hass, entry)
    suppression = SuppressionManager(hass)
    await suppression.async_load()
    audit_store = AuditStore(hass)
    await audit_store.async_load()
    discovery_max = int(
        options.get(
            CONF_SENTINEL_DISCOVERY_MAX_RECORDS,
            RECOMMENDED_SENTINEL_DISCOVERY_MAX_RECORDS,
        )
    )
    discovery_store = DiscoveryStore(hass, max_records=discovery_max)
    await discovery_store.async_load()
    proposal_store = ProposalStore(hass)
    await proposal_store.async_load()
    rule_registry = RuleRegistry(hass)
    await rule_registry.async_load()
    action_handler = ActionHandler(hass, suppression, audit_store)
    notifier = NotificationDispatcher(hass, options, action_handler)
    notifier.start()
    explainer = None
    if options.get(CONF_EXPLAIN_ENABLED, RECOMMENDED_EXPLAIN_ENABLED):
        explainer = LLMExplainer(chat_model)
    sentinel = SentinelEngine(
        hass,
        options,
        suppression,
        notifier,
        audit_store,
        explainer,
        rule_registry=rule_registry,
    )
    discovery_engine = SentinelDiscoveryEngine(
        hass=hass,
        options=options,
        model=chat_model,
        store=discovery_store,
        rule_registry=rule_registry,
        proposal_store=proposal_store,
    )

    face_recognition = options.get(CONF_FACE_RECOGNITION, RECOMMENDED_FACE_RECOGNITION)

    # Save runtime data.
    entry.runtime_data = HGAData(
        options=options,
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
        suppression=suppression,
        sentinel=sentinel,
        notifier=notifier,
        action_handler=action_handler,
        audit_store=audit_store,
        explainer=explainer,
        discovery_store=discovery_store,
        discovery_engine=discovery_engine,
        proposal_store=proposal_store,
        rule_registry=rule_registry,
    )

    if not hass.data[DOMAIN].get("http_registered"):
        hass.http.register_view(EnrollPersonView(hass, entry))
        www_dir = await hass.async_add_executor_job(_resolve_www_dir)
        if www_dir is not None:
            await hass.http.async_register_static_paths(
                [
                    # Canonical prefix for all HGA frontend card modules.
                    StaticPathConfig(
                        HGA_CARD_STATIC_PATH,
                        str(www_dir),
                        cache_headers=True,
                    ),
                    # Backward-compatible alias for existing enroll card resources.
                    StaticPathConfig(
                        HGA_CARD_STATIC_PATH_LEGACY,
                        str(www_dir),
                        cache_headers=True,
                    ),
                ]
            )
        hass.data[DOMAIN]["http_registered"] = True

    await hass.config_entries.async_forward_entry_setups(entry, PLATFORMS)

    if options.get(CONF_VIDEO_ANALYZER_MODE) != "disable":
        video_analyzer.start()
    if options.get(CONF_SENTINEL_ENABLED, RECOMMENDED_SENTINEL_ENABLED):
        if hass.is_running:
            sentinel.start()
        else:

            def _start_sentinel(_event: object) -> None:
                hass.loop.call_soon_threadsafe(sentinel.start)

            hass.bus.async_listen_once(EVENT_HOMEASSISTANT_STARTED, _start_sentinel)
    if options.get(
        CONF_SENTINEL_DISCOVERY_ENABLED,
        RECOMMENDED_SENTINEL_DISCOVERY_ENABLED,
    ):
        if hass.is_running:
            discovery_engine.start()
        else:

            def _start_discovery(_event: object) -> None:
                hass.loop.call_soon_threadsafe(discovery_engine.start)

            hass.bus.async_listen_once(EVENT_HOMEASSISTANT_STARTED, _start_discovery)

    async def _stop_background_tasks(_event: object) -> None:
        await sentinel.stop()
        await discovery_engine.stop()

    hass.bus.async_listen_once(EVENT_HOMEASSISTANT_STOP, _stop_background_tasks)

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
            img_bytes = await _read_enroll_image_bytes(hass, file_path)
        except (
            OSError,
            httpx.HTTPError,
            media_source.MediaSourceError,
            ValueError,
        ) as err:
            msg = f"Could not read media: {err}"
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

    async def _handle_get_audit(call: ServiceCall) -> dict[str, Any]:
        limit = int(call.data.get("limit", 20))
        limit = max(1, min(limit, 200))
        audit_store = entry.runtime_data.audit_store
        if audit_store is None:
            return {"records": []}
        records = await audit_store.async_get_latest(limit)
        return {"records": records}

    hass.services.async_register(
        DOMAIN,
        SERVICE_GET_AUDIT_RECORDS,
        _handle_get_audit,
        schema=GET_AUDIT_SCHEMA,
        supports_response=_SERVICE_RESPONSE_ONLY,
    )

    async def _handle_get_discovery(call: ServiceCall) -> dict[str, Any]:
        limit = int(call.data.get("limit", 20))
        limit = max(1, min(limit, 200))
        discovery_store = entry.runtime_data.discovery_store
        if discovery_store is None:
            return {"records": []}
        records = await discovery_store.async_get_latest(limit)
        return {"records": records}

    hass.services.async_register(
        DOMAIN,
        SERVICE_GET_DISCOVERY_RECORDS,
        _handle_get_discovery,
        schema=GET_DISCOVERY_SCHEMA,
        supports_response=_SERVICE_RESPONSE_ONLY,
    )

    def _covered_rule_id_for_candidate(candidate: dict[str, Any]) -> str | None:
        rule_registry = entry.runtime_data.rule_registry
        if rule_registry is None:
            return None
        candidate_key = candidate_semantic_key(candidate)
        if not candidate_key:
            return None
        for rule in rule_registry.list_rules():
            if rule_semantic_key(rule) != candidate_key:
                continue
            rule_id = str(rule.get("rule_id", ""))
            if rule_id:
                return rule_id
        return None

    async def _handle_promote_discovery(call: ServiceCall) -> dict[str, Any]:
        candidate_id = str(call.data.get("candidate_id"))
        notes = str(call.data.get("notes", "") or "")
        discovery_store = entry.runtime_data.discovery_store
        proposal_store = entry.runtime_data.proposal_store
        rule_registry = entry.runtime_data.rule_registry
        if discovery_store is None or proposal_store is None:
            return {"status": "unavailable"}

        candidate = discovery_store.find_candidate(candidate_id)
        if candidate is None:
            return {"status": "not_found"}

        covered_rule_id = _covered_rule_id_for_candidate(candidate)
        if covered_rule_id is not None:
            LOGGER.info(
                "Skipping promote for %s: candidate already covered by %s.",
                candidate_id,
                covered_rule_id,
            )
            return {
                "status": "already_active",
                "candidate_id": candidate_id,
                "rule_id": covered_rule_id,
            }

        normalized = normalize_candidate(candidate)
        if normalized is not None:
            existing_draft = proposal_store.find_by_rule_id(normalized.rule_id)
            if existing_draft is not None:
                LOGGER.info(
                    "Skipping promote for %s: draft already exists for rule %s.",
                    candidate_id,
                    normalized.rule_id,
                )
                return {
                    "status": "exists",
                    "candidate_id": candidate_id,
                    "rule_id": normalized.rule_id,
                }
            if rule_registry is not None and rule_registry.find_rule(
                normalized.rule_id
            ):
                LOGGER.info(
                    "Skipping promote for %s: rule %s already active.",
                    candidate_id,
                    normalized.rule_id,
                )
                return {
                    "status": "already_active",
                    "candidate_id": candidate_id,
                    "rule_id": normalized.rule_id,
                }

        draft = {
            "candidate_id": candidate_id,
            "candidate": candidate,
            "notes": notes,
            "status": "draft",
            "created_at": dt_util.utcnow().isoformat(),
        }
        if normalized is not None:
            draft["rule_id"] = normalized.rule_id
            draft["template_id"] = normalized.template_id
        await proposal_store.async_append(draft)
        notify_service = entry.runtime_data.options.get(CONF_NOTIFY_SERVICE)
        if notify_service and isinstance(notify_service, str):
            domain, _, service = notify_service.partition(".")
            if not service:
                service = notify_service
                domain = "notify"
            await hass.services.async_call(
                domain,
                service,
                {
                    "title": "HGA proposal draft",
                    "message": (
                        f"New proposal draft created for candidate {candidate_id}."
                    ),
                    "data": {"tag": f"hga_proposal_{candidate_id[:16]}"},
                },
                blocking=False,
            )
        return {"status": "ok", "candidate_id": candidate_id}

    hass.services.async_register(
        DOMAIN,
        SERVICE_PROMOTE_DISCOVERY_CANDIDATE,
        _handle_promote_discovery,
        schema=PROMOTE_DISCOVERY_SCHEMA,
        supports_response=_SERVICE_RESPONSE_ONLY,
    )

    async def _handle_get_proposals(call: ServiceCall) -> dict[str, Any]:
        limit = int(call.data.get("limit", 50))
        limit = max(1, min(limit, 200))
        proposal_store = entry.runtime_data.proposal_store
        if proposal_store is None:
            return {"records": []}
        records = await proposal_store.async_get_latest(limit)
        return {"records": records}

    hass.services.async_register(
        DOMAIN,
        SERVICE_GET_PROPOSAL_DRAFTS,
        _handle_get_proposals,
        schema=GET_PROPOSAL_SCHEMA,
        supports_response=_SERVICE_RESPONSE_ONLY,
    )

    async def _handle_approve_proposal(call: ServiceCall) -> dict[str, Any]:
        candidate_id = str(call.data.get("candidate_id"))
        notes = str(call.data.get("notes", "") or "")
        proposal_store = entry.runtime_data.proposal_store
        rule_registry = entry.runtime_data.rule_registry
        if proposal_store is None or rule_registry is None:
            LOGGER.info(
                "Proposal approval unavailable for candidate %s (stores not ready).",
                candidate_id,
            )
            return {"status": "unavailable"}
        record = proposal_store.find_by_candidate_id(candidate_id)
        candidate = record.get("candidate") if record else None
        if not candidate:
            LOGGER.info(
                "Proposal approval failed: candidate %s not found.",
                candidate_id,
            )
            return {"status": "not_found", "candidate_id": candidate_id}

        normalized = normalize_candidate(candidate)
        if normalized is None:
            covered_rule_id = _covered_rule_id_for_candidate(candidate)
            if covered_rule_id is not None:
                await proposal_store.async_update_status(
                    candidate_id,
                    "covered_by_existing_rule",
                    notes,
                    extra={"covered_rule_id": covered_rule_id},
                )
                LOGGER.info(
                    "Proposal %s marked covered_by_existing_rule (%s).",
                    candidate_id,
                    covered_rule_id,
                )
                return {
                    "status": "covered_by_existing_rule",
                    "candidate_id": candidate_id,
                    "rule_id": covered_rule_id,
                }
            await proposal_store.async_update_status(
                candidate_id,
                "unsupported",
                notes,
            )
            LOGGER.info(
                (
                    "Proposal %s marked unsupported "
                    "(candidate could not map to a template)."
                ),
                candidate_id,
            )
            return {"status": "unsupported", "candidate_id": candidate_id}

        rule_spec = normalized.as_dict()
        rule_spec["created_at"] = dt_util.utcnow().isoformat()
        rule_spec["source_candidate_id"] = candidate_id
        added = await rule_registry.async_add_rule(rule_spec)
        await proposal_store.async_update_status(
            candidate_id,
            "approved",
            notes,
            extra={"rule_id": rule_spec["rule_id"], "rule_spec": rule_spec},
        )
        if added:
            LOGGER.info(
                "Approved proposal %s and registered dynamic rule %s (%s).",
                candidate_id,
                rule_spec["rule_id"],
                rule_spec["template_id"],
            )
        else:
            await proposal_store.async_update_status(
                candidate_id,
                "covered_by_existing_rule",
                notes,
                extra={"covered_rule_id": rule_spec["rule_id"]},
            )
            LOGGER.info(
                "Approved proposal %s but dynamic rule %s already existed.",
                candidate_id,
                rule_spec["rule_id"],
            )
            return {
                "status": "covered_by_existing_rule",
                "candidate_id": candidate_id,
                "rule_id": rule_spec["rule_id"],
            }
        return {
            "status": "ok",
            "candidate_id": candidate_id,
            "rule_id": rule_spec["rule_id"],
        }

    hass.services.async_register(
        DOMAIN,
        SERVICE_APPROVE_RULE_PROPOSAL,
        _handle_approve_proposal,
        schema=REVIEW_PROPOSAL_SCHEMA,
        supports_response=_SERVICE_RESPONSE_ONLY,
    )

    async def _handle_reject_proposal(call: ServiceCall) -> dict[str, Any]:
        candidate_id = str(call.data.get("candidate_id"))
        notes = str(call.data.get("notes", "") or "")
        proposal_store = entry.runtime_data.proposal_store
        if proposal_store is None:
            return {"status": "unavailable"}
        ok = await proposal_store.async_update_status(candidate_id, "rejected", notes)
        return {"status": "ok" if ok else "not_found", "candidate_id": candidate_id}

    hass.services.async_register(
        DOMAIN,
        SERVICE_REJECT_RULE_PROPOSAL,
        _handle_reject_proposal,
        schema=REVIEW_PROPOSAL_SCHEMA,
        supports_response=_SERVICE_RESPONSE_ONLY,
    )

    return True


async def async_unload_entry(hass: HomeAssistant, entry: HGAConfigEntry) -> bool:
    """Unload the config entry."""
    if entry.runtime_data.pool is not None:
        await entry.runtime_data.pool.close()
    await entry.runtime_data.video_analyzer.stop()
    if entry.runtime_data.sentinel is not None:
        await entry.runtime_data.sentinel.stop()
    if entry.runtime_data.discovery_engine is not None:
        await entry.runtime_data.discovery_engine.stop()
    if entry.runtime_data.notifier is not None:
        entry.runtime_data.notifier.stop()
    await hass.config_entries.async_unload_platforms(entry, PLATFORMS)
    return True


async def async_migrate_entry(hass: HomeAssistant, config_entry: ConfigEntry) -> bool:  # noqa: C901, PLR0912, PLR0915
    """
    Migrate config entry to the latest version.

    - v1 -> v2: move CONF_DB_URI into a database subentry.
    - v2 -> v3: create model provider + feature subentries from legacy options.
    - v3 -> v4: move model settings into feature subentries and trim options.
    - v4 -> v5: move sentinel settings into a sentinel subentry.
    """
    current_version = config_entry.version or 1
    new_data = dict(config_entry.data)
    new_options = dict(config_entry.options)
    merged_options = {**new_data, **new_options}

    if current_version < 2:  # noqa: PLR2004
        LOGGER.info(
            "Migrating %s config entry %s -> v2",
            config_entry.domain,
            config_entry.entry_id,
        )

        db_uri = new_data.pop(CONF_DB_URI, None) or new_options.pop(CONF_DB_URI, None)

        if db_uri:
            parsed = parse_postgres_uri(db_uri)

            db_subentry_data = {
                CONF_USERNAME: parsed.get("username") or RECOMMENDED_DB_USERNAME,
                CONF_PASSWORD: parsed.get("password") or RECOMMENDED_DB_PASSWORD,
                CONF_HOST: parsed.get("host") or RECOMMENDED_DB_HOST,
                CONF_PORT: parsed.get("port") or RECOMMENDED_DB_PORT,
                CONF_DB_NAME: parsed.get("dbname") or RECOMMENDED_DB_NAME,
                CONF_DB_PARAMS: parsed.get("params") or RECOMMENDED_DB_PARAMS,
            }

            if any(
                s.subentry_type == SUBENTRY_TYPE_DATABASE
                for s in config_entry.subentries.values()
            ):
                LOGGER.debug(
                    "Database subentry already exists for entry %s, skipping creation",
                    config_entry.entry_id,
                )
            else:
                try:
                    subentry = ConfigSubentry(
                        subentry_type=SUBENTRY_TYPE_DATABASE,
                        title="Database",
                        unique_id=f"{config_entry.entry_id}_database",
                        data=MappingProxyType(db_subentry_data),
                    )
                    hass.config_entries.async_add_subentry(config_entry, subentry)
                    LOGGER.info(
                        "Created database subentry for entry %s",
                        config_entry.entry_id,
                    )
                except Exception:
                    LOGGER.exception(
                        "Migration failed for entry %s during database creation",
                        config_entry.entry_id,
                    )
                    return False
        current_version = 2
        merged_options = {**new_data, **new_options}

    if current_version < 3:  # noqa: PLR2004
        LOGGER.info(
            "Migrating %s config entry %s -> v%s",
            config_entry.domain,
            config_entry.entry_id,
            3,
        )

        existing_provider_subentries = [
            s
            for s in config_entry.subentries.values()
            if s.subentry_type == SUBENTRY_TYPE_MODEL_PROVIDER
        ]
        providers = legacy_model_provider_configs(config_entry, merged_options)
        if not existing_provider_subentries:
            for provider in providers.values():
                try:
                    hass.config_entries.async_add_subentry(
                        config_entry,
                        ConfigSubentry(
                            subentry_type=SUBENTRY_TYPE_MODEL_PROVIDER,
                            title=provider.name,
                            unique_id=provider.entry_id,
                            data=MappingProxyType(
                                {
                                    "provider_type": provider.provider_type,
                                    "capabilities": sorted(provider.capabilities),
                                    "settings": provider.data.get("settings", {}),
                                    "name": provider.name,
                                }
                            ),
                        ),
                    )
                except Exception:
                    LOGGER.exception(
                        "Failed to create model provider subentry for %s",
                        provider.name,
                    )

        provider_id_by_type: dict[str, str] = {}
        for subentry in config_entry.subentries.values():
            if subentry.subentry_type != SUBENTRY_TYPE_MODEL_PROVIDER:
                continue
            provider_type = subentry.data.get("provider_type")
            if provider_type and provider_type not in provider_id_by_type:
                provider_id_by_type[provider_type] = subentry.subentry_id

        existing_features = [
            s
            for s in config_entry.subentries.values()
            if s.subentry_type == SUBENTRY_TYPE_FEATURE
        ]
        if not existing_features and provider_id_by_type:
            features = legacy_feature_configs(config_entry, providers, merged_options)
            provider_lookup = {p.entry_id: p for p in providers.values()}
            for feature in features.values():
                provider = provider_lookup.get(feature.model_provider_id or "")
                provider_id = (
                    provider_id_by_type.get(provider.provider_type)
                    if provider
                    else None
                )
                if not provider_id:
                    continue
                try:
                    hass.config_entries.async_add_subentry(
                        config_entry,
                        ConfigSubentry(
                            subentry_type=SUBENTRY_TYPE_FEATURE,
                            title=feature.name,
                            unique_id=f"{config_entry.entry_id}_{feature.feature_type}",
                            data=MappingProxyType(
                                {
                                    "feature_type": feature.feature_type,
                                    "model_provider_id": provider_id,
                                    "name": feature.name,
                                    CONF_FEATURE_MODEL: feature.model,
                                    "config": feature.config,
                                }
                            ),
                        ),
                    )
                except Exception:
                    LOGGER.exception(
                        "Failed to create feature subentry %s", feature.feature_type
                    )
        current_version = 3

    if current_version < CONFIG_ENTRY_VERSION:
        LOGGER.info(
            "Migrating %s config entry %s -> v%s",
            config_entry.domain,
            config_entry.entry_id,
            CONFIG_ENTRY_VERSION,
        )

        provider_settings: dict[str, dict[str, Any]] = {}
        provider_types: dict[str, str] = {}
        for subentry in config_entry.subentries.values():
            if subentry.subentry_type != SUBENTRY_TYPE_MODEL_PROVIDER:
                continue
            settings = dict(subentry.data.get("settings", {}))
            provider_settings[subentry.subentry_id] = settings
            provider_types[subentry.subentry_id] = subentry.data.get(
                "provider_type", "ollama"
            )

        for subentry in list(config_entry.subentries.values()):
            if subentry.subentry_type != SUBENTRY_TYPE_FEATURE:
                continue
            feature_type = subentry.data.get("feature_type")
            category = (
                FEATURE_CATEGORY_MAP.get(feature_type)
                if isinstance(feature_type, str)
                else None
            )
            model_data = dict(subentry.data.get(CONF_FEATURE_MODEL, {}))
            if not model_data:
                provider_id = subentry.data.get("model_provider_id")
                provider_type = provider_types.get(provider_id or "", "ollama")
                settings = provider_settings.get(provider_id or "", {})
                spec = MODEL_CATEGORY_SPECS.get(category or "", {})
                model_name = spec.get("recommended_models", {}).get(provider_type)
                if category and provider_type in {"ollama", "openai", "gemini"}:
                    model_name = settings.get(f"{category}_model") or model_name

                if model_name:
                    model_data[CONF_FEATURE_MODEL_NAME] = model_name
                temp_key = spec.get("temperature_key")
                if temp_key and merged_options.get(temp_key) is not None:
                    model_data[CONF_FEATURE_MODEL_TEMPERATURE] = merged_options.get(
                        temp_key
                    )
                if provider_type == "ollama" and category:
                    keepalive_key = f"{category}_keepalive"
                    context_key = f"{category}_context"
                    if settings.get(keepalive_key) is not None:
                        model_data[CONF_FEATURE_MODEL_KEEPALIVE] = settings.get(
                            keepalive_key
                        )
                    if settings.get(context_key) is not None:
                        model_data[CONF_FEATURE_MODEL_CONTEXT_SIZE] = settings.get(
                            context_key
                        )
                    if category == "chat" and settings.get("reasoning") is not None:
                        model_data[CONF_FEATURE_MODEL_REASONING] = settings.get(
                            "reasoning"
                        )

            updated = dict(subentry.data)
            updated[CONF_FEATURE_MODEL] = model_data
            updated.setdefault("config", {})
            hass.config_entries.async_update_subentry(  # type: ignore[attr-defined]
                config_entry,
                subentry,
                data=MappingProxyType(updated),
                title=subentry.title,
            )

        for subentry in list(config_entry.subentries.values()):
            if subentry.subentry_type != SUBENTRY_TYPE_MODEL_PROVIDER:
                continue
            settings = dict(subentry.data.get("settings", {}))
            base_url = (
                settings.get("base_url")
                or settings.get("chat_url")
                or settings.get("vlm_url")
            )
            trimmed_settings = {}
            if base_url:
                trimmed_settings["base_url"] = base_url
            if api_key := settings.get("api_key"):
                trimmed_settings["api_key"] = api_key
            updated = dict(subentry.data)
            updated["settings"] = trimmed_settings
            hass.config_entries.async_update_subentry(  # type: ignore[attr-defined]
                config_entry,
                subentry,
                data=MappingProxyType(updated),
                title=subentry.title,
            )

        _ensure_default_feature_subentries(hass, config_entry)

        for key in (
            CONF_API_KEY,
            CONF_GEMINI_API_KEY,
            CONF_OLLAMA_URL,
            CONF_OLLAMA_CHAT_URL,
            CONF_OLLAMA_VLM_URL,
            CONF_OLLAMA_SUMMARIZATION_URL,
            CONF_CHAT_MODEL_PROVIDER,
            CONF_VLM_PROVIDER,
            CONF_SUMMARIZATION_MODEL_PROVIDER,
            CONF_EMBEDDING_MODEL_PROVIDER,
            CONF_CHAT_MODEL_TEMPERATURE,
            CONF_VLM_TEMPERATURE,
            CONF_SUMMARIZATION_MODEL_TEMPERATURE,
            CONF_OLLAMA_CHAT_MODEL,
            CONF_OPENAI_CHAT_MODEL,
            CONF_GEMINI_CHAT_MODEL,
            CONF_OLLAMA_VLM,
            CONF_OPENAI_VLM,
            CONF_GEMINI_VLM,
            CONF_OLLAMA_SUMMARIZATION_MODEL,
            CONF_OPENAI_SUMMARIZATION_MODEL,
            CONF_GEMINI_SUMMARIZATION_MODEL,
            CONF_OLLAMA_EMBEDDING_MODEL,
            CONF_OPENAI_EMBEDDING_MODEL,
            CONF_GEMINI_EMBEDDING_MODEL,
            CONF_OLLAMA_CHAT_KEEPALIVE,
            CONF_OLLAMA_VLM_KEEPALIVE,
            CONF_OLLAMA_SUMMARIZATION_KEEPALIVE,
            CONF_OLLAMA_CHAT_CONTEXT_SIZE,
            CONF_OLLAMA_VLM_CONTEXT_SIZE,
            CONF_OLLAMA_SUMMARIZATION_CONTEXT_SIZE,
            CONF_OLLAMA_REASONING,
        ):
            new_options.pop(key, None)

        sentinel_subentry_exists = any(
            s.subentry_type == SUBENTRY_TYPE_SENTINEL
            for s in config_entry.subentries.values()
        )
        if not sentinel_subentry_exists:
            sentinel_data = {
                CONF_SENTINEL_ENABLED: merged_options.get(
                    CONF_SENTINEL_ENABLED, RECOMMENDED_SENTINEL_ENABLED
                ),
                CONF_SENTINEL_INTERVAL_SECONDS: merged_options.get(
                    CONF_SENTINEL_INTERVAL_SECONDS,
                    RECOMMENDED_SENTINEL_INTERVAL_SECONDS,
                ),
                CONF_SENTINEL_COOLDOWN_MINUTES: merged_options.get(
                    CONF_SENTINEL_COOLDOWN_MINUTES,
                    RECOMMENDED_SENTINEL_COOLDOWN_MINUTES,
                ),
                CONF_SENTINEL_ENTITY_COOLDOWN_MINUTES: merged_options.get(
                    CONF_SENTINEL_ENTITY_COOLDOWN_MINUTES,
                    RECOMMENDED_SENTINEL_ENTITY_COOLDOWN_MINUTES,
                ),
                CONF_SENTINEL_DISCOVERY_ENABLED: merged_options.get(
                    CONF_SENTINEL_DISCOVERY_ENABLED,
                    RECOMMENDED_SENTINEL_DISCOVERY_ENABLED,
                ),
                CONF_SENTINEL_DISCOVERY_INTERVAL_SECONDS: merged_options.get(
                    CONF_SENTINEL_DISCOVERY_INTERVAL_SECONDS,
                    RECOMMENDED_SENTINEL_DISCOVERY_INTERVAL_SECONDS,
                ),
                CONF_SENTINEL_DISCOVERY_MAX_RECORDS: merged_options.get(
                    CONF_SENTINEL_DISCOVERY_MAX_RECORDS,
                    RECOMMENDED_SENTINEL_DISCOVERY_MAX_RECORDS,
                ),
                CONF_EXPLAIN_ENABLED: merged_options.get(
                    CONF_EXPLAIN_ENABLED, RECOMMENDED_EXPLAIN_ENABLED
                ),
            }
            hass.config_entries.async_add_subentry(
                config_entry,
                ConfigSubentry(
                    subentry_type=SUBENTRY_TYPE_SENTINEL,
                    title="Sentinel",
                    unique_id=f"{config_entry.entry_id}_sentinel",
                    data=MappingProxyType(sentinel_data),
                ),
            )

        for key in (
            CONF_SENTINEL_ENABLED,
            CONF_SENTINEL_INTERVAL_SECONDS,
            CONF_SENTINEL_COOLDOWN_MINUTES,
            CONF_SENTINEL_ENTITY_COOLDOWN_MINUTES,
            CONF_SENTINEL_DISCOVERY_ENABLED,
            CONF_SENTINEL_DISCOVERY_INTERVAL_SECONDS,
            CONF_SENTINEL_DISCOVERY_MAX_RECORDS,
            CONF_EXPLAIN_ENABLED,
        ):
            new_options.pop(key, None)

        try:
            hass.config_entries.async_update_entry(
                config_entry,
                data=new_data,
                options=new_options,
                version=CONFIG_ENTRY_VERSION,
            )
        except Exception:
            LOGGER.exception(
                "Failed to update config entry %s during migration",
                config_entry.entry_id,
            )
            return False

    return True

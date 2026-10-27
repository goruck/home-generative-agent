"""Home Generative Agent Initialization."""

from __future__ import annotations

import asyncio
import inspect
import logging
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import replace
from functools import partial
from pathlib import Path
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, cast
from urllib.parse import unquote

import aiofiles
import httpx
import psycopg
import voluptuous as vol
from homeassistant.components import media_source
from homeassistant.components.camera.const import DOMAIN as CAMERA_DOMAIN
from homeassistant.components.http.server import StaticPathConfig
from homeassistant.config_entries import (
    SIGNAL_CONFIG_ENTRY_CHANGED,
    ConfigEntry,
    ConfigEntryChange,
    ConfigSubentry,
)
from homeassistant.const import (
    CONF_API_KEY,
    CONF_HOST,
    CONF_LLM_HASS_API,
    CONF_PASSWORD,
    CONF_PORT,
    CONF_USERNAME,
    EVENT_HOMEASSISTANT_STOP,
    Platform,
)
from homeassistant.core import HomeAssistant, SupportsResponse, callback
from homeassistant.exceptions import ConfigEntryError, HomeAssistantError
from homeassistant.helpers import config_validation as cv
from homeassistant.helpers.dispatcher import (
    async_dispatcher_connect,
    async_dispatcher_send,
)
from homeassistant.helpers.httpx_client import get_async_client
from homeassistant.helpers.network import get_url
from homeassistant.helpers.target import (
    async_extract_referenced_entity_ids,
)
from homeassistant.util import dt as dt_util
from homeassistant.util.ssl import SSL_ALPN_HTTP11, client_context
from langchain_core.runnables import ConfigurableField
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langgraph.checkpoint.memory import MemorySaver
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.store.postgres import AsyncPostgresStore
from langgraph.store.postgres.base import PostgresIndexConfig
from ollama import ResponseError as OllamaResponseError
from psycopg.rows import DictRow, dict_row
from psycopg_pool import AsyncConnectionPool
from pydantic import SecretStr

from .agent.tools import VLMPromptOverrides, analyze_image
from .audit.store import AuditStore
from .const import (
    CHAT_MODEL_MAX_TOKENS,
    CHAT_MODEL_REPEAT_PENALTY,
    CHAT_MODEL_TOP_P,
    CONF_ANTHROPIC_API_KEY,
    CONF_ANTHROPIC_CHAT_MODEL,
    CONF_ANTHROPIC_SUMMARIZATION_MODEL,
    CONF_ANTHROPIC_VLM,
    CONF_AUDIT_HOT_MAX_RECORDS,
    CONF_CHAT_MODEL_PROVIDER,
    CONF_CHAT_MODEL_TEMPERATURE,
    CONF_CHAT_REASONING,
    CONF_CHAT_REASONING_BUDGET,
    CONF_CHAT_REASONING_BY_MODEL,
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
    CONF_MODEL_PROVIDER_UNCONTENDED,
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
    CONF_OPENAI_COMPATIBLE_API_KEY,
    CONF_OPENAI_COMPATIBLE_BASE_URL,
    CONF_OPENAI_COMPATIBLE_CHAT_MODEL,
    CONF_OPENAI_COMPATIBLE_EMBEDDING_API_KEY,
    CONF_OPENAI_COMPATIBLE_EMBEDDING_DIMS,
    CONF_OPENAI_COMPATIBLE_EMBEDDING_MODEL,
    CONF_OPENAI_COMPATIBLE_EMBEDDING_URL,
    CONF_OPENAI_COMPATIBLE_SUMMARIZATION_MODEL,
    CONF_OPENAI_COMPATIBLE_VLM,
    CONF_OPENAI_EMBEDDING_MODEL,
    CONF_OPENAI_SUMMARIZATION_MODEL,
    CONF_OPENAI_VLM,
    CONF_SENTINEL_BASELINE_ENABLED,
    CONF_SENTINEL_COOLDOWN_MINUTES,
    CONF_SENTINEL_DISCOVERY_ENABLED,
    CONF_SENTINEL_DISCOVERY_INTERVAL_SECONDS,
    CONF_SENTINEL_DISCOVERY_MAX_RECORDS,
    CONF_SENTINEL_ENABLED,
    CONF_SENTINEL_ENTITY_COOLDOWN_MINUTES,
    CONF_SENTINEL_INTERVAL_SECONDS,
    CONF_SENTINEL_PENDING_PROMPT_TTL_MINUTES,
    CONF_SENTINEL_RESPONSE_LANGUAGE,
    CONF_SENTINEL_TRIAGE_ENABLED,
    CONF_SENTINEL_TRIAGE_MAX_PER_RUN,
    CONF_SENTINEL_TRIAGE_TIMEOUT_SECONDS,
    CONF_SUMMARIZATION_MODEL_PROVIDER,
    CONF_SUMMARIZATION_MODEL_TEMPERATURE,
    CONF_VECTORS_BOOTSTRAPPED,
    CONF_VIDEO_ANALYZER_MODE,
    CONF_VLM_PROMPT_EXTRA,
    CONF_VLM_PROVIDER,
    CONF_VLM_RESPONSE_LANGUAGE,
    CONF_VLM_TEMPERATURE,
    CONFIG_ENTRY_VERSION,
    DEFAULT_FEATURE_TYPES,
    DOMAIN,
    EMBEDDING_MODEL_CTX,
    EMBEDDING_MODEL_DIMS,
    FALLBACK_CIRCUIT_BREAKER_COOLDOWN_SECONDS,
    FALLBACK_CIRCUIT_BREAKER_THRESHOLD,
    FALLBACK_CIRCUIT_BREAKER_WINDOW_SECONDS,
    FEATURE_CATEGORY_MAP,
    FEATURE_NAMES,
    HGA_CARD_STATIC_PATH,
    HGA_CARD_STATIC_PATH_LEGACY,
    MODEL_CATEGORY_SPECS,
    NO_DATABASE_ENROLL_MESSAGE,
    NO_DATABASE_REMEDIATION,
    OPENAI_CHAT_COMPLETIONS_PATH,
    RECOMMENDED_ANTHROPIC_CHAT_MODEL,
    RECOMMENDED_ANTHROPIC_SUMMARIZATION_MODEL,
    RECOMMENDED_ANTHROPIC_VLM,
    RECOMMENDED_AUDIT_HOT_MAX_RECORDS,
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
    RECOMMENDED_MODEL_PROVIDER_UNCONTENDED,
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
    RECOMMENDED_OPENAI_COMPATIBLE_CHAT_MODEL,
    RECOMMENDED_OPENAI_COMPATIBLE_EMBEDDING_DIMS,
    RECOMMENDED_OPENAI_COMPATIBLE_EMBEDDING_MODEL,
    RECOMMENDED_OPENAI_COMPATIBLE_SUMMARIZATION_MODEL,
    RECOMMENDED_OPENAI_COMPATIBLE_VLM,
    RECOMMENDED_OPENAI_EMBEDDING_MODEL,
    RECOMMENDED_OPENAI_SUMMARIZATION_MODEL,
    RECOMMENDED_OPENAI_VLM,
    RECOMMENDED_SENTINEL_BASELINE_ENABLED,
    RECOMMENDED_SENTINEL_COOLDOWN_MINUTES,
    RECOMMENDED_SENTINEL_DISCOVERY_ENABLED,
    RECOMMENDED_SENTINEL_DISCOVERY_INTERVAL_SECONDS,
    RECOMMENDED_SENTINEL_DISCOVERY_MAX_RECORDS,
    RECOMMENDED_SENTINEL_ENABLED,
    RECOMMENDED_SENTINEL_ENTITY_COOLDOWN_MINUTES,
    RECOMMENDED_SENTINEL_INTERVAL_SECONDS,
    RECOMMENDED_SENTINEL_PENDING_PROMPT_TTL_MINUTES,
    RECOMMENDED_SENTINEL_TRIAGE_ENABLED,
    RECOMMENDED_SENTINEL_TRIAGE_MAX_PER_RUN,
    RECOMMENDED_SENTINEL_TRIAGE_TIMEOUT_SECONDS,
    RECOMMENDED_SUMMARIZATION_MODEL_PROVIDER,
    RECOMMENDED_SUMMARIZATION_MODEL_TEMPERATURE,
    RECOMMENDED_VLM_PROMPT_EXTRA,
    RECOMMENDED_VLM_PROVIDER,
    RECOMMENDED_VLM_RESPONSE_LANGUAGE,
    RECOMMENDED_VLM_TEMPERATURE,
    SENTINEL_AUTONOMY_LEVELS,
    SIGNAL_HGA_NEW_LATEST,
    SIGNAL_HGA_RECOGNIZED,
    SIGNAL_SENTINEL_RUN_COMPLETE,
    SIGNAL_TOOL_INDEX_UPDATED,
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
from .core.anthropic_client import SharedClientChatAnthropic, async_prime_async_client
from .core.caption_store import async_backfill_caption_cameras
from .core.database_guard import (
    async_check_database_reachable,
    async_clear_database_issue,
    async_ensure_pgvector,
    async_sync_database_issue,
    database_unreachable,
    is_connection_failure,
)
from .core.db_utils import parse_postgres_uri
from .core.embedding_model_guard import (
    async_clear_ollama_embedding_model_issue,
    async_sync_ollama_embedding_model_issue,
)
from .core.embedding_record import (
    DIGEST_FETCH_TIMEOUT_S,
    EmbeddingFallbackGate,
    FallbackBuild,
    async_clear_embedding_model_changed_issue,
    async_clear_other_build_fallback_issue,
    async_recorded_ollama_digest,
    async_sync_embedding_record,
    async_sync_other_build_fallback_issue,
    async_verify_embedding_fallback,
    embedding_fingerprint,
)
from .core.embedding_store import (
    CHECKPOINT_MIGRATIONS,
    CHECKPOINT_TABLES,
    GEMINI,
    MIGRATION_RECORDS,
    OPENAI_COMPATIBLE,
    STORE_MIGRATIONS,
    STORE_TABLES,
    VECTOR_MIGRATIONS,
    VECTOR_TABLES,
    UnavailableEmbeddings,
    async_clear_embedding_size_issue,
    async_clear_other_model_fallback_issue,
    async_forget_migrations,
    async_store_tables,
    async_sync_embedding_size_issue,
    async_sync_other_model_fallback_issue,
    embedding_size_check_provider,
    missing_tables,
)
from .core.face_guard import (
    async_check_unknown_person_rules,
    async_clear_unknown_person_issue,
)
from .core.fallback import (
    CircuitBreaker,
    FallbackChatModel,
    FallbackEmbeddings,
    FallbackVLM,
    _safe_err_summary,
)
from .core.lifecycle import defer_start_until_hass_started
from .core.migrations import migrate_person_gallery
from .core.person_gallery import PersonGalleryDAO
from .core.pipeline_guard import async_clear_pin_pipeline_issue
from .core.prompt_cache import CACHE_CONTROL_EPHEMERAL
from .core.provider_recheck import (
    ProviderProbe,
    async_schedule_provider_recheck,
    endpoint_label,
)
from .core.provider_stall import ProviderStallTracker
from .core.runtime import HGAConfigEntry, HGAData
from .core.subentry_resolver import (
    build_database_uri_from_entry,
    build_model_deployments,
    chosen_fallback_ids,
    legacy_feature_configs,
    legacy_model_provider_configs,
    resolve_fallback_chains,
    resolve_model_provider_configs,
    resolve_runtime_options,
    split_embedding_chain,
)
from .core.utils import (
    anthropic_healthy,
    builds_differ,
    configured_ollama_urls,
    dispatch_on_loop,
    ensure_http_url,
    gemini_healthy,
    gemini_sampling_configurable,
    generate_embeddings,
    make_gated_probe_runner,
    normalize_openai_compatible_base_url,
    normalized_ollama_url,
    ollama_healthy,
    ollama_model_digests,
    ollama_provider_settings_url,
    ollama_url_for_category,
    openai_compatible_healthy,
    openai_healthy,
    reasoning_field,
    redact_url,
    set_model_provider_uncontended,
    thinking_configurable,
    validate_anthropic_key,
    validate_gemini_key,
    validate_ollama_url,
    validate_openai_compatible_url,
    validate_openai_key,
)
from .core.video_analyzer import VideoAnalyzer
from .core.video_helpers import latest_target, publish_latest_atomic
from .explain.llm_explain import LLMExplainer
from .http import EnrollPersonView
from .notify.actions import ActionHandler
from .sentinel.auth_inventory import AuthInventory
from .sentinel.baseline import SentinelBaselineUpdater
from .sentinel.discovery_engine import SentinelDiscoveryEngine
from .sentinel.discovery_semantic import (
    candidate_semantic_key,
    rule_key_covers_candidate_key,
    rule_semantic_key,
    sanitize_environmental_candidate,
)
from .sentinel.discovery_store import DiscoveryStore
from .sentinel.dynamic_rules import evaluate_dynamic_rule
from .sentinel.engine import SentinelEngine
from .sentinel.evidence_paths import (
    NETWORK_POSTURE_ALERT_VALUE,
    NETWORK_POSTURE_STATIC_RULES,
    network_posture_expected,
    network_posture_keys,
)
from .sentinel.network_audit import empty_report as empty_network_audit_report
from .sentinel.network_inventory import NetworkInventory
from .sentinel.notifier import SentinelNotifier
from .sentinel.proposal_store import ProposalStore
from .sentinel.proposal_templates import explain_normalize_candidate
from .sentinel.pseudonymizer import (
    async_load_pseudonymizer,
    async_remove_pseudonymizer_salt,
)
from .sentinel.rule_registry import RuleRegistry
from .sentinel.suppression import SuppressionManager
from .sentinel.triage import SentinelTriageService
from .snapshot.builder import async_build_full_state_snapshot

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator

    from homeassistant.auth.models import User
    from homeassistant.config_entries import ConfigEntry
    from homeassistant.core import HomeAssistant, ServiceCall
    from homeassistant.helpers.typing import ConfigType
    from langchain_core.language_models import BaseMessage, LanguageModelInput
    from langchain_core.runnables.base import RunnableSerializable
    from psycopg import AsyncConnection

    from .core.subentry_types import ModelProviderConfig

# Newer cores renamed TargetSelectorData to TargetSelection and deprecated the
# old name (removal in HA 2026.12.0). Keep importing the old name as a fallback
# so cores that predate the rename still load.
try:
    from homeassistant.helpers.target import TargetSelection
except ImportError:  # pragma: no cover - older cores only expose the old name.
    from homeassistant.helpers.target import (  # type: ignore[assignment]
        TargetSelectorData as TargetSelection,
    )

LOGGER = logging.getLogger(__name__)

_SERVICE_RESPONSE_ONLY = SupportsResponse.ONLY

PLATFORMS = (Platform.CONVERSATION, Platform.STT, Platform.TTS, "image", "sensor")

SERVICE_ENROLL_PERSON = "enroll_person"
SERVICE_GET_AUDIT_RECORDS = "get_audit_records"
SERVICE_GET_DISCOVERY_RECORDS = "get_discovery_records"
SERVICE_TRIGGER_SENTINEL_DISCOVERY = "trigger_sentinel_discovery"
SERVICE_PROMOTE_DISCOVERY_CANDIDATE = "promote_discovery_candidate"
SERVICE_GET_PROPOSAL_DRAFTS = "get_proposal_drafts"
SERVICE_PREVIEW_RULE_PROPOSAL = "preview_rule_proposal"
SERVICE_APPROVE_RULE_PROPOSAL = "approve_rule_proposal"
SERVICE_REJECT_RULE_PROPOSAL = "reject_rule_proposal"
SERVICE_GET_DYNAMIC_RULES = "get_dynamic_rules"
SERVICE_DEACTIVATE_DYNAMIC_RULE = "deactivate_dynamic_rule"
SERVICE_REACTIVATE_DYNAMIC_RULE = "reactivate_dynamic_rule"
SERVICE_PATCH_DYNAMIC_RULE = "patch_dynamic_rule"
SERVICE_SENTINEL_SET_AUTONOMY_LEVEL = "sentinel_set_autonomy_level"
SERVICE_SENTINEL_GET_BASELINES = "sentinel_get_baselines"
SERVICE_SENTINEL_RESET_BASELINE = "sentinel_reset_baseline"
SERVICE_SENTINEL_RESET_AUTH_INVENTORY = "sentinel_reset_auth_inventory"
SERVICE_RUN_NETWORK_AUDIT = "run_network_audit"
SERVICE_SENTINEL_GET_NETWORK_INVENTORY = "sentinel_get_network_inventory"
SERVICE_SENTINEL_TRUST_NETWORK_DEVICE = "sentinel_trust_network_device"
SERVICE_SENTINEL_UNTRUST_NETWORK_DEVICE = "sentinel_untrust_network_device"
SERVICE_SENTINEL_RESET_NETWORK_INVENTORY = "sentinel_reset_network_inventory"
SENTINEL_NETWORK_DEVICE_SCHEMA = vol.Schema(
    vol.All(
        vol.Schema(
            {
                # Registry devices (radio devices, and router clients joined
                # to one) or inventory keys (``router:<key>``) from
                # ``sentinel_get_network_inventory``; at least one of the two.
                vol.Optional("device_id"): vol.All(cv.ensure_list, [cv.string]),
                vol.Optional("device_key"): vol.All(cv.ensure_list, [cv.string]),
            }
        ),
        cv.has_at_least_one_key("device_id", "device_key"),
    )
)

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

TRIGGER_DISCOVERY_SCHEMA = vol.Schema({})

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

PREVIEW_PROPOSAL_SCHEMA = vol.Schema(
    {
        vol.Required("candidate_id"): cv.string,
    }
)

REVIEW_PROPOSAL_SCHEMA = vol.Schema(
    {
        vol.Required("candidate_id"): cv.string,
        vol.Optional("notes"): cv.string,
    }
)

GET_DYNAMIC_RULES_SCHEMA = vol.Schema(
    {
        vol.Optional("limit", default=200): vol.Coerce(int),
    }
)

TOGGLE_DYNAMIC_RULE_SCHEMA = vol.Schema(
    {
        vol.Required("rule_id"): cv.string,
    }
)

PATCH_DYNAMIC_RULE_SCHEMA = vol.Schema(
    {
        vol.Required("rule_id"): cv.string,
        vol.Required("params"): dict,
    }
)

SET_AUTONOMY_LEVEL_SCHEMA = vol.Schema(
    {
        vol.Required("level"): vol.All(
            vol.Coerce(int), vol.In(SENTINEL_AUTONOMY_LEVELS)
        ),
        vol.Optional("pin"): cv.string,
    }
)

SENTINEL_GET_BASELINES_SCHEMA = vol.Schema({})

# An entity id, or a network counter id (``network.client.<key>.<figure>``)
# from the baseline step, which the reset must reach as well.
_ENTITY_ID_RE = r"^(?:[a-z_]+\.[a-z0-9_-]+|network\.[a-z0-9_.-]+)$"

SENTINEL_RESET_BASELINE_SCHEMA = vol.Schema(
    {
        vol.Optional("entity_id"): vol.All(
            cv.string,
            vol.Match(
                _ENTITY_ID_RE,
                msg="entity_id must match domain.object_id or a network.* counter",
            ),
        ),
    }
)

Embedding = Sequence[float]


def _rule_entity_ids(params: dict[str, Any]) -> list[str]:
    entity_ids: list[str] = []
    for key, value in params.items():
        if key.endswith("_entity_id") and isinstance(value, str) and value:
            entity_ids.append(value)
        elif key.endswith("_entity_ids") and isinstance(value, list):
            entity_ids.extend(item for item in value if isinstance(item, str) and item)
    return sorted(set(entity_ids))


def _candidate_entity_ids(candidate: dict[str, Any]) -> list[str]:
    explain = explain_normalize_candidate(candidate)
    normalized = explain.normalized
    if normalized is None:
        return []
    return _rule_entity_ids(normalized.params)


def _covered_rule_for_candidate(
    entry: HGAConfigEntry,
    candidate: dict[str, Any],
) -> tuple[str, list[str]] | None:
    rule_registry = entry.runtime_data.rule_registry
    candidate_key = candidate_semantic_key(candidate)
    candidate_entities = _candidate_entity_ids(candidate)
    if rule_registry is not None and candidate_key:
        for rule in rule_registry.list_rules():
            rkey = rule_semantic_key(rule)
            if not rkey or not rule_key_covers_candidate_key(rkey, candidate_key):
                continue
            rule_id = str(rule.get("rule_id", ""))
            if rule_id:
                overlapping_entities = sorted(
                    set(candidate_entities).intersection(
                        _rule_entity_ids(rule.get("params") or {})
                    )
                )
                return rule_id, overlapping_entities
    return _covered_builtin_rule_for_candidate(candidate)


def _camera_entity_from_paths(paths: list[str]) -> str | None:
    """
    Return the first camera.entity_id found in evidence paths.

    Handles two formats:
      camera_activity[camera_entity_id=camera.front_porch].snapshot_summary
      camera_activity[camera_entity_id=front_porch].snapshot_summary  (no domain)
    In the second case the "camera." domain prefix is prepended automatically.
    """
    _key = "camera_entity_id="
    for path in paths:
        # Format 1: full entity_id with "camera." domain prefix anywhere in path.
        idx = path.find("camera.")
        if idx >= 0:
            end = path.find("]", idx)
            return path[idx:end] if end >= 0 else path[idx:]
        # Format 2: camera_entity_id=<object_id> without domain prefix.
        kidx = path.find(_key)
        if kidx >= 0:
            start = kidx + len(_key)
            end = path.find("]", start)
            object_id = path[start:end] if end >= 0 else path[start:]
            if object_id:
                return object_id if "." in object_id else f"camera.{object_id}"
    return None


def _covered_builtin_rule_for_candidate(
    candidate: dict[str, Any],
) -> tuple[str, list[str]] | None:
    text = " ".join(
        [
            str(candidate.get("title", "")),
            str(candidate.get("summary", "")),
            str(candidate.get("pattern", "")),
            str(candidate.get("suggested_type", "")),
        ]
    ).lower()
    evidence_paths = [
        item for item in candidate.get("evidence_paths", []) if isinstance(item, str)
    ]
    camera_entity = _camera_entity_from_paths(evidence_paths)
    for posture_key in network_posture_keys(evidence_paths):
        expected = network_posture_expected(posture_key, text)
        if expected is None:
            expected = NETWORK_POSTURE_ALERT_VALUE[posture_key]
        covered = NETWORK_POSTURE_STATIC_RULES.get((posture_key, expected))
        if covered is not None:
            return covered, []
    if (
        camera_entity is not None
        and "vehicle" in text
        and any(term in text for term in ("home", "resident", "occupant"))
    ):
        return "vehicle_detected_near_camera_home", [camera_entity]
    if (
        camera_entity is not None
        and all(term in text for term in ("snapshot", "night"))
        and any(term in text for term in ("missing", "no "))
        and any(term in text for term in ("home", "present", "occupant", "resident"))
    ):
        return "camera_missing_snapshot_night_home", [camera_entity]
    _phone_keywords = {
        "phone",
        "iphone",
        "android",
        "pixel",
        "galaxy",
        "mobile",
        "smartphone",
        "handset",
    }
    if (
        "battery" in text
        and "night" in text
        and any(kw in text for kw in _phone_keywords)
    ):
        entity_id = next(
            (
                path
                for path in evidence_paths
                if "battery" in path and any(kw in path for kw in _phone_keywords)
            ),
            None,
        )
        return "phone_battery_low_at_night_home", ([entity_id] if entity_id else [])
    return None


def _covered_specific_rule_for_any_camera_normalized(
    entry: HGAConfigEntry,
    template_id: str,
    params: dict[str, Any],
) -> tuple[str, list[str]] | None:
    rule_registry = entry.runtime_data.rule_registry
    if rule_registry is None:
        return None
    if params.get("camera_selector") != "any":
        return None
    if template_id not in {
        "unknown_person_camera_no_home",
        "unknown_person_camera_when_home",
    }:
        return None
    for rule in rule_registry.list_rules():
        if str(rule.get("template_id", "")) != template_id:
            continue
        rule_params = rule.get("params") or {}
        if not isinstance(rule_params, dict):
            continue
        camera_entity_id = str(rule_params.get("camera_entity_id", ""))
        if not camera_entity_id:
            continue
        rule_id = str(rule.get("rule_id", ""))
        if rule_id:
            return rule_id, [camera_entity_id]
    return None


async def _trigger_sentinel_discovery(entry: HGAConfigEntry) -> dict[str, Any]:
    discovery_engine = entry.runtime_data.discovery_engine
    if discovery_engine is None:
        return {"status": "unavailable"}
    started = await discovery_engine.async_run_now()
    return {"status": "ok" if started else "busy"}


async def _promote_discovery_candidate(  # noqa: PLR0911, PLR0912
    hass: HomeAssistant,
    entry: HGAConfigEntry,
    *,
    candidate_id: str,
    notes: str = "",
) -> dict[str, Any]:
    discovery_store = entry.runtime_data.discovery_store
    proposal_store = entry.runtime_data.proposal_store
    rule_registry = entry.runtime_data.rule_registry
    if discovery_store is None or proposal_store is None:
        return {"status": "unavailable"}

    candidate = discovery_store.find_candidate(candidate_id)
    if candidate is None:
        return {"status": "not_found"}

    # Environmental candidates stored before the ingestion sanitizer shipped
    # (or promoted straight from discovery history) still carry occupancy
    # decoration — sanitize at promote time too, so a pre-fix stored
    # candidate can never mint a new decorated draft (red-team review).
    sanitized = sanitize_environmental_candidate(candidate)
    if sanitized is not None:
        candidate = sanitized

    covered = _covered_rule_for_candidate(entry, candidate)
    if covered is not None:
        covered_rule_id, overlapping_entities = covered
        return {
            "status": "already_active",
            "candidate_id": candidate_id,
            "rule_id": covered_rule_id,
            "overlapping_entity_ids": overlapping_entities,
        }

    normalization = explain_normalize_candidate(candidate)
    normalized = normalization.normalized
    if normalized is not None:
        covered_specific = _covered_specific_rule_for_any_camera_normalized(
            entry,
            normalized.template_id,
            normalized.params,
        )
        if covered_specific is not None:
            covered_specific_rule_id, overlapping_entities = covered_specific
            return {
                "status": "already_active",
                "candidate_id": candidate_id,
                "rule_id": covered_specific_rule_id,
                "overlapping_entity_ids": overlapping_entities,
            }
        existing_draft = proposal_store.find_by_rule_id(normalized.rule_id)
        if existing_draft is not None:
            return {
                "status": "exists",
                "candidate_id": candidate_id,
                "rule_id": normalized.rule_id,
            }
        if rule_registry is not None and rule_registry.find_rule(normalized.rule_id):
            overlapping_entities = _rule_entity_ids(normalized.params)
            return {
                "status": "already_active",
                "candidate_id": candidate_id,
                "rule_id": normalized.rule_id,
                "overlapping_entity_ids": overlapping_entities,
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
        draft["severity"] = normalized.severity
        draft["confidence"] = normalized.confidence
    await proposal_store.async_append(draft)
    notify_service = entry.runtime_data.options.get(CONF_NOTIFY_SERVICE)
    if notify_service and isinstance(notify_service, str):
        domain, _, service = notify_service.partition(".")
        if not service:
            service = notify_service
            domain = "notify"
        if normalized is None:
            message = f"New proposal draft created for candidate {candidate_id}."
            data: dict[str, Any] = {"tag": f"hga_proposal_{candidate_id[:16]}"}
        else:
            confidence_pct = round(normalized.confidence * 100)
            message = (
                f"New {normalized.severity.upper()}-severity proposal: "
                f"{normalized.template_id} ({confidence_pct}% confident) - "
                "call approve_rule_proposal to activate."
            )
            data = {
                "tag": f"hga_proposal_{candidate_id[:16]}",
                "candidate_id": candidate_id,
                "template_id": normalized.template_id,
                "severity": normalized.severity,
                "confidence": normalized.confidence,
                "service_hint": "approve_rule_proposal",
            }
        await hass.services.async_call(
            domain,
            service,
            {
                "title": "HGA proposal draft",
                "message": message,
                "data": data,
            },
            blocking=False,
        )
    return {"status": "ok", "candidate_id": candidate_id}


def _is_battery_like_state(entity_id: str, state: Any) -> bool:
    """
    Return True when a live state plausibly is a battery-percent reading.

    The issue #522 locale fallback promotes sensor IDs its English-only
    kind-token filter cannot classify, so approval must gate on the live
    state (red-team review): a battery device_class or a percent unit
    (excluding percent sensors of another declared class, e.g. humidity),
    or a legacy English "battery" ID token. The state must also be numeric
    or transiently unavailable — a text state would permanently deadlock
    the all-of low_battery_sensors evaluator.
    """
    attributes = getattr(state, "attributes", {}) or {}
    device_class = attributes.get("device_class")
    unit = attributes.get("unit_of_measurement")
    if device_class == "battery":
        plausible = True
    elif device_class is None:
        # The battery ID token only grandfathers sensors whose metadata does
        # not contradict it: sensor.front_door_battery_voltage (unit V,
        # state 3.1) must not pass and then permanently satisfy the percent
        # evaluator (issue #522 Codex verification round).
        plausible = unit == "%" or (unit is None and "battery" in entity_id.lower())
    else:
        plausible = False
    if not plausible:
        return False
    value = getattr(state, "state", None)
    if value in ("unavailable", "unknown"):
        return True
    try:
        float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return False
    return True


async def _validate_low_battery_params(
    *,
    hass: HomeAssistant | None,
    proposal_store: Any,
    candidate_id: str,
    notes: str,
    normalized: Any,
) -> tuple[Any, dict[str, Any] | None]:
    """
    Live-validate low_battery_sensors params at approval (issue #522).

    Battery sensor IDs can come from the locale fallback, which promotes an
    unnamed sensor.* evidence ID on text signals alone — evidence paths are
    LLM output, so a hallucinated or paraphrased ID would register a
    low_battery_sensors rule whose all-of numeric evaluator silently never
    fires while its semantic key suppresses re-proposals (same
    permanent-monitoring-gap class as the #518 motion validation). Resolves
    against live states, drops unknown IDs, and returns an error response
    (second element) when no usable battery sensor remains.
    """
    battery_sensor_ids = (
        normalized.params.get("sensor_entity_ids")
        if normalized.template_id == "low_battery_sensors"
        else None
    )
    if (
        hass is None
        or not isinstance(battery_sensor_ids, list)
        or not battery_sensor_ids
    ):
        return normalized, None
    resolved_battery_ids: list[str] = []
    implausible_battery_ids: list[str] = []
    unresolved_battery_ids: list[str] = []
    for entity_id in battery_sensor_ids:
        state = hass.states.get(entity_id) if isinstance(entity_id, str) else None
        if state is None and isinstance(entity_id, str) and "." not in entity_id:
            # Legacy drafts store domainless object IDs; the evaluator's
            # _resolve_sensor_entity_id tries the sensor/binary_sensor
            # domains, so approval must use the same resolution or it
            # newly refuses previously-supported candidates (issue #522
            # Codex verification round).
            for qualified_id in (
                f"sensor.{entity_id}",
                f"binary_sensor.{entity_id}",
            ):
                state = hass.states.get(qualified_id)
                if state is not None:
                    break
        if state is None:
            unresolved_battery_ids.append(entity_id)
        elif _is_battery_like_state(entity_id, state):
            resolved_battery_ids.append(entity_id)
        else:
            # Exists but is not a battery-percent reading (issue #522
            # red-team review): the English-only kind-token filter in
            # the locale fallback cannot recognize e.g. a Czech
            # voltage/temperature sensor, and registering one would
            # false-fire the percent-threshold evaluator every cycle
            # (3.1 V <= 40) or deadlock it on a non-numeric state.
            # This live-state gate is the authoritative filter.
            implausible_battery_ids.append(entity_id)
    if not resolved_battery_ids:
        rejected_ids = unresolved_battery_ids + implausible_battery_ids
        reason_code = (
            "not_battery_sensor" if implausible_battery_ids else "entities_unresolved"
        )
        LOGGER.info(
            "Proposal approval rejected for candidate %s: no usable "
            "battery sensor entity (%s: %s)",
            candidate_id,
            reason_code,
            rejected_ids,
        )
        await proposal_store.async_update_status(
            candidate_id,
            "unsupported",
            notes,
            extra={
                "normalization_reason": reason_code,
                "unresolved_entity_ids": rejected_ids,
            },
        )
        return normalized, {
            "status": "unsupported",
            "candidate_id": candidate_id,
            "reason_code": reason_code,
            "details": {"unresolved_entity_ids": rejected_ids},
        }
    if unresolved_battery_ids or implausible_battery_ids:
        normalized = replace(
            normalized,
            params={
                **normalized.params,
                "sensor_entity_ids": resolved_battery_ids,
            },
        )
    return normalized, None


async def _approve_rule_proposal(  # noqa: PLR0911, PLR0912, PLR0915
    entry: HGAConfigEntry,
    *,
    hass: HomeAssistant | None = None,
    candidate_id: str,
    notes: str = "",
) -> dict[str, Any]:
    proposal_store = entry.runtime_data.proposal_store
    rule_registry = entry.runtime_data.rule_registry
    if proposal_store is None or rule_registry is None:
        return {"status": "unavailable"}
    record = proposal_store.find_by_candidate_id(candidate_id)
    candidate = record.get("candidate") if record else None
    if not candidate:
        return {"status": "not_found", "candidate_id": candidate_id}

    # Drafts persisted before the environmental sanitizer shipped still
    # carry occupancy decoration; sanitize at this boundary too so an old
    # card cannot activate (or preview) a rule whose ID and prose advertise
    # conditioning the rule never has (Codex structured review P1).
    sanitized_candidate = sanitize_environmental_candidate(candidate)
    if sanitized_candidate is not None:
        candidate = sanitized_candidate

    normalization = explain_normalize_candidate(candidate)
    normalized = normalization.normalized

    covered = _covered_rule_for_candidate(entry, candidate)
    if covered is not None:
        covered_rule_id, overlapping_entities = covered
        # Key-based DYNAMIC-rule coverage is coarser than template
        # semantics: an unknown-person camera candidate that also cites a
        # motion sensor must not be swallowed as "already active" by a
        # plain motion rule — the sensitive camera semantics would be
        # silently lost — and an unsupported candidate (e.g. a stale-sensor
        # request keying like an active motion rule) gets the honest
        # "unsupported" response instead of a false cover (issue #518
        # verification reviews). Built-in rule covers (not present in the
        # dynamic registry) are matched by their own targeted semantics in
        # _covered_builtin_rule_for_candidate and always stand.
        covered_rule = rule_registry.find_rule(covered_rule_id)
        covered_template = str((covered_rule or {}).get("template_id") or "")
        if covered_rule is None or (
            normalized is not None
            and covered_template
            and covered_template == normalized.template_id
        ):
            await proposal_store.async_update_status(
                candidate_id,
                "covered_by_existing_rule",
                notes,
                extra={
                    "covered_rule_id": covered_rule_id,
                    "overlapping_entity_ids": overlapping_entities,
                },
            )
            return {
                "status": "covered_by_existing_rule",
                "candidate_id": candidate_id,
                "rule_id": covered_rule_id,
                "overlapping_entity_ids": overlapping_entities,
            }

    if normalized is None:
        LOGGER.info(
            "Proposal approval unsupported for candidate %s: reason=%s details=%s",
            candidate_id,
            normalization.reason_code,
            normalization.details or {},
        )
        await proposal_store.async_update_status(
            candidate_id,
            "unsupported",
            notes,
            extra={
                "normalization_reason": normalization.reason_code,
                "normalization_details": normalization.details or {},
            },
        )
        return {
            "status": "unsupported",
            "candidate_id": candidate_id,
            "reason_code": normalization.reason_code,
            "details": normalization.details or {},
        }

    # Motion entity IDs can come from candidate prose (index-based evidence
    # paths, issue #518), and prose is LLM output: a hallucinated or
    # paraphrased sensor name would register a rule whose evaluator fails
    # closed forever while its semantic key suppresses re-proposals — a
    # silent, permanent monitoring gap the user believes is active
    # (issue #518 red-team + Codex reviews). Resolve against live states;
    # drop unknown IDs, and refuse the approval when none remain.
    motion_entity_ids = normalized.params.get("motion_entity_ids")
    if hass is not None and isinstance(motion_entity_ids, list) and motion_entity_ids:
        resolved_motion_ids = [
            entity_id
            for entity_id in motion_entity_ids
            if isinstance(entity_id, str) and hass.states.get(entity_id) is not None
        ]
        unresolved_motion_ids = [
            entity_id
            for entity_id in motion_entity_ids
            if entity_id not in resolved_motion_ids
        ]
        if not resolved_motion_ids:
            LOGGER.info(
                "Proposal approval rejected for candidate %s: no configured "
                "motion entity exists (%s)",
                candidate_id,
                unresolved_motion_ids,
            )
            await proposal_store.async_update_status(
                candidate_id,
                "unsupported",
                notes,
                extra={
                    "normalization_reason": "entities_unresolved",
                    "unresolved_entity_ids": unresolved_motion_ids,
                },
            )
            return {
                "status": "unsupported",
                "candidate_id": candidate_id,
                "reason_code": "entities_unresolved",
                "details": {"unresolved_entity_ids": unresolved_motion_ids},
            }
        if unresolved_motion_ids:
            normalized = replace(
                normalized,
                params={
                    **normalized.params,
                    "motion_entity_ids": resolved_motion_ids,
                },
            )
    # Battery sensor IDs can come from the locale fallback (issue #522) —
    # live-validate them (see _validate_low_battery_params) and refuse the
    # approval when no usable battery sensor remains.
    normalized, battery_error = await _validate_low_battery_params(
        hass=hass,
        proposal_store=proposal_store,
        candidate_id=candidate_id,
        notes=notes,
        normalized=normalized,
    )
    if battery_error is not None:
        return battery_error
    # Same-template any-of superset coverage: the key-based check above uses
    # exact entity-set equality, so an existing [kitchen, hall] rule does not
    # cover a kitchen-only candidate there — yet it already alerts on every
    # sensor this candidate would, and registering a second rule guarantees
    # duplicate findings on each event. Runs for fully resolved candidates
    # and for sets reduced by the hallucination filter alike (issue #518
    # verification rounds 3-4). Restricted to the two ANY-OF away-motion
    # templates: motion_while_alarm_disarmed_and_home_present is all-of (a
    # [kitchen, hall] rule fires only when BOTH are active, so it does NOT
    # cover a kitchen-only candidate), and alarm-motion templates carry
    # alarm params that must also match (verification round 5).
    final_motion_ids = normalized.params.get("motion_entity_ids")
    if (
        normalized.template_id
        in {"motion_detected_while_away", "motion_detected_at_night_while_away"}
        and isinstance(final_motion_ids, list)
        and final_motion_ids
    ):
        for existing_rule in rule_registry.list_rules():
            if str(existing_rule.get("template_id") or "") != normalized.template_id:
                continue
            existing_motion = {
                entity_id
                for entity_id in (existing_rule.get("params") or {}).get(
                    "motion_entity_ids", []
                )
                if isinstance(entity_id, str)
            }
            if not existing_motion or not existing_motion.issuperset(final_motion_ids):
                continue
            overlapping_entities = sorted(
                existing_motion.intersection(final_motion_ids)
            )
            existing_rule_id = str(existing_rule.get("rule_id") or "")
            await proposal_store.async_update_status(
                candidate_id,
                "covered_by_existing_rule",
                notes,
                extra={
                    "covered_rule_id": existing_rule_id,
                    "overlapping_entity_ids": overlapping_entities,
                },
            )
            return {
                "status": "covered_by_existing_rule",
                "candidate_id": candidate_id,
                "rule_id": existing_rule_id,
                "overlapping_entity_ids": overlapping_entities,
            }

    covered_specific = _covered_specific_rule_for_any_camera_normalized(
        entry,
        normalized.template_id,
        normalized.params,
    )
    if covered_specific is not None:
        covered_specific_rule_id, overlapping_entities = covered_specific
        await proposal_store.async_update_status(
            candidate_id,
            "covered_by_existing_rule",
            notes,
            extra={
                "covered_rule_id": covered_specific_rule_id,
                "overlapping_entity_ids": overlapping_entities,
            },
        )
        return {
            "status": "covered_by_existing_rule",
            "candidate_id": candidate_id,
            "rule_id": covered_specific_rule_id,
            "overlapping_entity_ids": overlapping_entities,
        }

    existing_rule = rule_registry.find_rule(normalized.rule_id)
    if existing_rule is not None:
        overlapping_entities = sorted(
            set(_rule_entity_ids(normalized.params)).intersection(
                _rule_entity_ids(existing_rule.get("params") or {})
            )
        )
        await proposal_store.async_update_status(
            candidate_id,
            "covered_by_existing_rule",
            notes,
            extra={
                "covered_rule_id": normalized.rule_id,
                "overlapping_entity_ids": overlapping_entities,
            },
        )
        return {
            "status": "covered_by_existing_rule",
            "candidate_id": candidate_id,
            "rule_id": normalized.rule_id,
            "overlapping_entity_ids": overlapping_entities,
        }

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
        sentinel = entry.runtime_data.sentinel
        if sentinel is not None:
            await sentinel.async_run_now()
        return {
            "status": "ok",
            "candidate_id": candidate_id,
            "rule_id": rule_spec["rule_id"],
        }

    await proposal_store.async_update_status(
        candidate_id,
        "covered_by_existing_rule",
        notes,
        extra={"covered_rule_id": rule_spec["rule_id"]},
    )
    return {
        "status": "covered_by_existing_rule",
        "candidate_id": candidate_id,
        "rule_id": rule_spec["rule_id"],
    }


async def _preview_rule_proposal(
    hass: HomeAssistant,
    entry: HGAConfigEntry,
    *,
    candidate_id: str,
) -> dict[str, Any]:
    proposal_store = entry.runtime_data.proposal_store
    if proposal_store is None:
        return {"status": "unavailable"}

    record = proposal_store.find_by_candidate_id(candidate_id)
    candidate = record.get("candidate") if record else None
    if not candidate:
        return {"status": "not_found", "candidate_id": candidate_id}

    # Drafts persisted before the environmental sanitizer shipped still
    # carry occupancy decoration; sanitize at this boundary too so an old
    # card cannot activate (or preview) a rule whose ID and prose advertise
    # conditioning the rule never has (Codex structured review P1).
    sanitized_candidate = sanitize_environmental_candidate(candidate)
    if sanitized_candidate is not None:
        candidate = sanitized_candidate

    normalization = explain_normalize_candidate(candidate)
    normalized = normalization.normalized
    if normalized is None:
        return {
            "status": "unsupported",
            "candidate_id": candidate_id,
            "reason_code": normalization.reason_code,
            "details": normalization.details or {},
        }

    snapshot = await async_build_full_state_snapshot(hass)
    findings = evaluate_dynamic_rule(snapshot, normalized.as_dict())
    matching_entity_ids = sorted(
        {entity_id for finding in findings for entity_id in finding.triggering_entities}
    )
    return {
        "status": "ok",
        "candidate_id": candidate_id,
        "rule_id": normalized.rule_id,
        "template_id": normalized.template_id,
        "would_trigger": bool(findings),
        "matching_entity_ids": matching_entity_ids,
        "findings": [finding.as_dict() for finding in findings],
    }


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


def _configured_embedding_dims(options: Mapping[str, Any]) -> int:
    """
    Return the vector size a new memory store is created with.

    The configured dimensions for an OpenAI-compatible primary, else the
    size the other providers are asked for. An existing store keeps the size
    it was created with, which the embedding-model check reads back.
    """
    if (
        options.get(CONF_EMBEDDING_MODEL_PROVIDER, RECOMMENDED_EMBEDDING_MODEL_PROVIDER)
        == "openai_compatible"
    ):
        return options.get(
            CONF_OPENAI_COMPATIBLE_EMBEDDING_DIMS,
            RECOMMENDED_OPENAI_COMPATIBLE_EMBEDDING_DIMS,
        )
    return EMBEDDING_MODEL_DIMS


def _configured_embedding_model(options: Mapping[str, Any], provider_type: str) -> str:
    """Return the embedding model a provider type is configured with."""
    keys = {
        "openai": (CONF_OPENAI_EMBEDDING_MODEL, RECOMMENDED_OPENAI_EMBEDDING_MODEL),
        GEMINI: (CONF_GEMINI_EMBEDDING_MODEL, RECOMMENDED_GEMINI_EMBEDDING_MODEL),
        OPENAI_COMPATIBLE: (
            CONF_OPENAI_COMPATIBLE_EMBEDDING_MODEL,
            RECOMMENDED_OPENAI_COMPATIBLE_EMBEDDING_MODEL,
        ),
    }
    key, default = keys.get(
        provider_type, (CONF_OLLAMA_EMBEDDING_MODEL, RECOMMENDED_OLLAMA_EMBEDDING_MODEL)
    )
    return str(options.get(key, default))


def _ollama_embedding_urls_to_check(
    *,
    primary_url: str,
    primary_selected: bool,
    fallback_urls: list[str],
    health: Mapping[str, bool],
    has_database: bool,
) -> list[str]:
    """
    Return the Ollama servers whose embedding model setup should verify.

    The primary only when Ollama is the selected embedding provider, plus each
    Ollama fallback in the embedding chain, which embeds with the same model
    (#749). Only servers that passed the health check: an unreachable one is
    already reported, and its embeddings client is never built. None without a
    database, where nothing uses embeddings.
    """
    if not has_database:
        return []
    candidates = [primary_url] if primary_selected else []
    candidates.extend(fallback_urls)
    urls: list[str] = []
    for url in candidates:
        if url and health.get(url) and url not in urls:
            urls.append(url)
    return urls


def _ollama_provider_url(provider: ModelProviderConfig) -> str:
    """Return the Ollama URL a provider's models use, as setup keys it."""
    return ollama_provider_settings_url(provider.data.get("settings", {}))


@callback
def _retry_failed_tool_index(
    hass: HomeAssistant, entry: ConfigEntry, provider_id: str
) -> None:
    """
    Let the tool index build again once a held embedding fallback is admitted.

    A held fallback refuses calls until its build is checked (#857); a tool
    index built while it was the only provider left that could answer failed,
    and a failed index is not built again. It holds the store's build, so an
    index that did build stays as it is. An index still being written is
    marked, so a failure it ends in does not latch either.
    """
    runtime_data = getattr(entry, "runtime_data", None)
    if runtime_data is None:
        return
    if runtime_data.tool_indexing_in_progress:
        runtime_data.tool_index_rearmed = True
        return
    if not runtime_data.tool_index_failed:
        return
    runtime_data.tool_index_failed = False
    LOGGER.info(
        "Embedding provider %s was admitted after its build check; the tool "
        "index that failed meanwhile is built again on the next indexing pass.",
        provider_id,
    )
    async_dispatcher_send(hass, SIGNAL_TOOL_INDEX_UPDATED, "stale", 0)


def _hold_unchecked_embedding_fallbacks(
    hass: HomeAssistant,
    entry: ConfigEntry,
    chain: FallbackEmbeddings,
    unchecked: Sequence[FallbackBuild],
    *,
    on_admitted: Callable[[str], None] | None = None,
) -> None:
    """
    Keep Ollama embedding fallbacks setup could not fully check from answering.

    One holding another build than the store's would write its vectors there
    with nothing to say so (#857): each answers only once its build is read
    and matches, checked when the chain first needs it or by the record check
    at the end of setup, whichever comes first. `on_admitted` runs too when
    one is first admitted.
    """
    if not unchecked:
        return

    @callback
    def _admitted(provider_id: str) -> None:
        _retry_failed_tool_index(hass, entry, provider_id)
        if on_admitted is not None:
            on_admitted(provider_id)

    chain.hold_until_verified(
        [fb.provider_id for fb in unchecked],
        partial(
            async_verify_embedding_fallback,
            hass,
            entry,
            {fb.provider_id: fb for fb in unchecked},
            chain=chain,
            warned=set(),
        ),
        admitted_callback=_admitted,
    )


def _provider_setting(
    providers: Mapping[str, ModelProviderConfig], provider_type: str, key: str
) -> str | None:
    """Return a named setting for the first matching provider type, or None."""
    for provider in providers.values():
        if provider.provider_type != provider_type:
            continue
        value = provider.data.get("settings", {}).get(key)
        if value:
            return str(value)
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
        if category in keepalive_map:
            model_data[CONF_FEATURE_MODEL_CONTEXT_SIZE] = (
                RECOMMENDED_OLLAMA_CONTEXT_SIZE
            )
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
        capabilities = {str(c) for c in provider.data.get("capabilities") or []}
        if category and capabilities and category not in capabilities:
            continue
        model_data = dict(data.get(CONF_FEATURE_MODEL, {}))
        if not model_data and category:
            model_data = _default_model_data(category, provider_type)
        data["model_provider_id"] = provider.subentry_id
        data[CONF_FEATURE_MODEL] = model_data
        hass.config_entries.async_update_subentry(  # type: ignore[attr-defined]
            entry, subentry, data=MappingProxyType(data), title=subentry.title
        )


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
    pool: AsyncConnectionPool[AsyncConnection[DictRow]],
) -> None:
    """
    Bootstrap database if needed.

    Again when its tables are gone after it was bootstrapped, such as a
    Database subentry pointed at a fresh database or a dropped table (#760).
    """
    if entry.data.get(CONF_DB_BOOTSTRAPPED):
        tables = await async_store_tables(pool)
        missing = missing_tables(tables, (*STORE_TABLES, *CHECKPOINT_TABLES))
        if tables is None or not missing:
            return
        LOGGER.warning(
            "Memory store tables %s are missing (a new or reset database?); "
            "creating them again.",
            ", ".join(missing),
        )
        await _async_rebuild_tables(
            pool,
            tables,
            (STORE_MIGRATIONS, CHECKPOINT_MIGRATIONS),
            (store.setup, checkpointer.setup),
        )
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
    pool: AsyncConnectionPool[AsyncConnection[DictRow]],
) -> None:
    """
    Bootstrap vector index if needed.

    Again when the vector table is gone after it was bootstrapped, such as
    one dropped to change its size (#760): it is created at the configured
    size. langgraph skips the vector migrations it has recorded, so they are
    forgotten first.
    """
    if not store.index_config:
        return
    if entry.data.get(CONF_VECTORS_BOOTSTRAPPED):
        tables = await async_store_tables(pool)
        if tables is None or not missing_tables(tables, VECTOR_TABLES):
            return
        LOGGER.warning(
            "The memory store's vector table is missing; creating it again for "
            "%s-dimension vectors. Memories saved before it was removed are "
            "kept but are not found by memory search until saved again.",
            store.index_config.get("dims"),
        )
        await _async_rebuild_tables(pool, tables, (VECTOR_MIGRATIONS,), (store.setup,))
        return

    await store.setup()

    hass.config_entries.async_update_entry(
        entry, data={**entry.data, CONF_VECTORS_BOOTSTRAPPED: True}
    )


async def _async_rebuild_tables(
    pool: AsyncConnectionPool[AsyncConnection[DictRow]],
    tables: frozenset[str],
    records: tuple[str, ...],
    setups: tuple[Callable[[], Awaitable[None]], ...],
) -> None:
    """
    Create missing memory store tables again (#760).

    A setup that fails is logged and the next one still runs, so a store that
    cannot be rebuilt (a vector size its index cannot hold) does not keep the
    checkpointer's tables from coming back, and the integration still loads:
    a missing table only ever failed the writes to it. A lost connection is
    raised, for setup's retry.
    """
    names = tuple(name for record in records for name in MIGRATION_RECORDS[record])
    steps: tuple[Callable[[], Awaitable[None]], ...] = (
        partial(async_forget_migrations, pool, tables, records),
        *setups,
    )
    for step in steps:
        try:
            await step()
        except psycopg.Error as err:
            if is_connection_failure(err):
                raise
            LOGGER.error(  # noqa: TRY400
                "Could not create memory store tables again, so writes to them "
                "fail: %s",
                err,
            )
            if step is steps[0]:
                return
    if missing := missing_tables(await async_store_tables(pool), names):
        LOGGER.error(
            "Memory store tables %s are still missing after setup, so writes "
            "to them fail.",
            ", ".join(missing),
        )


class NullChat:
    """Non-throwing fallback implementing common chat model methods."""

    async def ainvoke(self, _input: Any, _config: Any = None, **_kw: Any) -> str:
        """Return a placeholder response."""
        return "LLM unavailable."

    async def astream(
        self, _input: Any, _config: Any = None, **_kw: Any
    ) -> AsyncGenerator[str, Any]:
        """Return a placeholder response."""
        yield "LLM unavailable."

    def bind_tools(self, _tools: Any) -> NullChat:
        """Return self, as tool binding is a no-op for the fallback model."""
        return self

    def with_config(self, _config: Any = None, **_cfg: Any) -> NullChat:
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


def _ollama_httpx_client_kwargs() -> dict[str, Any]:
    """Return httpx kwargs that avoid creating SSL contexts in the event loop."""
    return {"verify": client_context(alpn_protocols=SSL_ALPN_HTTP11)}


async def _async_require_admin(
    hass: HomeAssistant, call: ServiceCall, service: str
) -> User:
    """
    Return the calling admin user, or raise for anonymous and non-admin callers.

    Shared by the services that can weaken Sentinel (raising the autonomy
    level, resetting the auth inventory): both must refuse a stolen
    low-privilege session or an automation running as a non-admin user.
    """
    if not call.context.user_id:
        msg = f"{service} requires an authenticated user."
        raise HomeAssistantError(msg)
    user = await hass.auth.async_get_user(call.context.user_id)
    if user is None or not user.is_admin:
        msg = f"{service} is restricted to admin users."
        raise HomeAssistantError(msg)
    return user


def _register_network_inventory_services(
    hass: HomeAssistant, entry: HGAConfigEntry
) -> None:
    """Register the device-inventory services (read, trust, untrust, reset)."""

    def _inventory() -> NetworkInventory | None:
        return entry.runtime_data.network_inventory

    async def _handle_get(call: ServiceCall) -> dict[str, Any]:
        """Return every known radio device with its trust flag."""
        _ = call
        inventory = _inventory()
        if inventory is None:
            return {"status": "unavailable", "summary": None, "devices": []}
        return {
            "status": "ok",
            "summary": inventory.summary(),
            "devices": inventory.list_devices(),
        }

    def _make_trust_handler(
        service: str, *, trusted: bool
    ) -> Callable[[ServiceCall], Any]:
        async def _handle(call: ServiceCall) -> dict[str, Any]:
            """
            Set the trust flag on known devices.

            Admin-only: the flag records what the owner recognizes, and the
            audit report counts untrusted devices.
            """
            user = await _async_require_admin(hass, call, service)
            inventory = _inventory()
            if inventory is None:
                return {"status": "unavailable", "changed": []}
            changed = await inventory.async_set_trusted(
                [*call.data.get("device_id", []), *call.data.get("device_key", [])],
                trusted=trusted,
            )
            LOGGER.info(
                "Sentinel network inventory: user %s set trusted=%s on %d device(s).",
                user.name,
                trusted,
                len(changed),
            )
            return {"status": "ok", "changed": changed}

        return _handle

    async def _handle_reset(call: ServiceCall) -> dict[str, Any]:
        """
        Clear the device inventory; the next run bootstraps every source.

        Admin-only and logged, like the auth inventory reset: a reset makes
        every paired device known without alerting.
        """
        user = await _async_require_admin(
            hass, call, SERVICE_SENTINEL_RESET_NETWORK_INVENTORY
        )
        inventory = _inventory()
        if inventory is None:
            return {"status": "unavailable"}
        before = inventory.summary()
        LOGGER.warning(
            "Sentinel network inventory reset by user %s (%s): %d device(s) "
            "forgotten; the next run records paired devices without alerting.",
            user.name,
            user.id,
            before["device_count"],
        )
        if not await inventory.async_reset():
            msg = (
                "The device inventory was cleared in memory but its file could "
                "not be removed; it will come back after a restart."
            )
            raise HomeAssistantError(msg)
        return {"status": "ok", "cleared": before}

    _register_entry_service(
        hass,
        entry,
        SERVICE_SENTINEL_GET_NETWORK_INVENTORY,
        _handle_get,
        schema=vol.Schema({}),
        supports_response=SupportsResponse.ONLY,
    )
    for service, trusted in (
        (SERVICE_SENTINEL_TRUST_NETWORK_DEVICE, True),
        (SERVICE_SENTINEL_UNTRUST_NETWORK_DEVICE, False),
    ):
        _register_entry_service(
            hass,
            entry,
            service,
            _make_trust_handler(service, trusted=trusted),
            schema=SENTINEL_NETWORK_DEVICE_SCHEMA,
            supports_response=SupportsResponse.OPTIONAL,
        )
    _register_entry_service(
        hass,
        entry,
        SERVICE_SENTINEL_RESET_NETWORK_INVENTORY,
        _handle_reset,
        schema=vol.Schema({}),
        supports_response=SupportsResponse.ONLY,
    )


def _register_entry_service(  # noqa: PLR0913
    hass: HomeAssistant,
    entry: HGAConfigEntry,
    service: str,
    handler: Callable[[ServiceCall], Any],
    *,
    schema: vol.Schema | None = None,
    supports_response: SupportsResponse = SupportsResponse.NONE,
) -> None:
    """
    Register a domain service and remove it when this entry unloads.

    Service handlers close over this entry's ``runtime_data``, so leaving them
    registered after unload keeps a live API surface (`hga.*` service calls)
    reaching closed pools and stopped engines. Home Assistant also runs
    on-unload callbacks after a FAILED setup, so registering the removal here
    — immediately after the registration itself — covers every abort path and
    guarantees the remove always has a matching register.
    """
    hass.services.async_register(
        DOMAIN,
        service,
        handler,
        schema=schema,
        supports_response=supports_response,
    )
    entry.async_on_unload(partial(hass.services.async_remove, DOMAIN, service))


def _register_services(hass: HomeAssistant, entry: HGAConfigEntry) -> None:
    """Register integration services."""

    async def _handle_save_and_analyze_snapshot(call: ServiceCall) -> None:
        """Capture a snapshot, analyze it, and publish 'latest' for targeted cameras."""
        # 1) Resolve Target selector (entity/device/area/label[/floor])
        # raw 'target:' from the service call (may be absent)
        raw_target: ConfigType = cast("ConfigType", call.data.get("target", {}))
        selector = TargetSelection(raw_target)
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
            try:
                vlm_overrides = VLMPromptOverrides(
                    response_language=entry.runtime_data.options.get(
                        CONF_VLM_RESPONSE_LANGUAGE, RECOMMENDED_VLM_RESPONSE_LANGUAGE
                    ),
                    prompt_extra=entry.runtime_data.options.get(
                        CONF_VLM_PROMPT_EXTRA, RECOMMENDED_VLM_PROMPT_EXTRA
                    ),
                )
                summary = await analyze_image(
                    vision_model,
                    img_bytes,
                    None,
                    prev_text=None,
                    overrides=vlm_overrides,
                )
            except OllamaResponseError as err:
                msg = f"VLM analysis failed for {camera_id}: {err}"
                raise HomeAssistantError(msg) from err

            face_hits = await video_analyzer.recognize_faces(img_bytes, camera_id)
            people: list[str] = [hit.name for hit in face_hits]

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
    _register_entry_service(
        hass, entry, "save_and_analyze_snapshot", _handle_save_and_analyze_snapshot
    )


async def _log_ollama_server_info(
    hass: HomeAssistant,
    urls_by_category: dict[str, str],
) -> None:
    """Query Ollama /api/ps at startup and log loaded models and swap-risk notes."""
    client = get_async_client(hass)

    # Log which categories share each URL so operators can spot contention.
    url_to_cats: dict[str, list[str]] = {}
    for cat, url in urls_by_category.items():
        url_to_cats.setdefault(url, []).append(cat)
    for url, cats in url_to_cats.items():
        LOGGER.info(
            "subsystem=model_server url=%s categories=%s",
            redact_url(url),
            ",".join(sorted(cats)),
        )

    # Query /api/ps on each unique URL and log loaded models.
    for url, cats in url_to_cats.items():
        try:
            async with asyncio.timeout(5.0):
                resp = await client.get(f"{url.rstrip('/')}/api/ps")
            resp.raise_for_status()
            data = resp.json()
        except Exception as err:  # noqa: BLE001
            reason = (
                f"HTTP {err.response.status_code}"
                if isinstance(err, httpx.HTTPStatusError)
                else type(err).__name__
            )
            LOGGER.warning(
                "subsystem=model_server url=%s probe_failed reason=%s",
                redact_url(url),
                reason,
            )
            continue

        models = data.get("models", [])
        if not models:
            LOGGER.info(
                "subsystem=model_server url=%s loaded_models=none", redact_url(url)
            )
            continue

        for m in models:
            name = m.get("name", "unknown")
            size_mb = int(m.get("size", 0)) // (1024 * 1024)
            vram_mb = int(m.get("size_vram", 0)) // (1024 * 1024)
            LOGGER.info(
                "subsystem=model_server url=%s model=%s size_mb=%d vram_mb=%d",
                redact_url(url),
                name,
                size_mb,
                vram_mb,
            )

        # A fallback only runs once its primary fails, so only primaries that
        # share a server swap models on it in normal use.
        primaries = [cat for cat in cats if "_fallback" not in cat]
        if len(primaries) > 1:
            LOGGER.info(
                "subsystem=model_server url=%s shared_by=%d categories=%s "
                "note=model_swapping_may_add_latency",
                redact_url(url),
                len(primaries),
                ",".join(sorted(primaries)),
            )


async def async_setup_entry(hass: HomeAssistant, entry: HGAConfigEntry) -> bool:  # noqa: C901, PLR0912, PLR0915
    """Set up Home Generative Agent from a config entry."""
    hass.data.setdefault(DOMAIN, {})

    # Reset admission-control globals in case a prior entry unload left them
    # dirty (e.g. mid-chat reload where the task was not cleanly cancelled).
    import custom_components.home_generative_agent.core.utils as _utils_mod  # noqa: PLC0415

    _utils_mod._chat_active_count = 0  # noqa: SLF001
    _utils_mod._chat_idle.set()  # noqa: SLF001
    _utils_mod._chat_last_end = 0.0  # noqa: SLF001
    _utils_mod._uncontended = False  # noqa: SLF001
    _utils_mod._gate_generation += 1  # noqa: SLF001
    _utils_mod._sentinel_last_call_end = 0.0  # noqa: SLF001
    _utils_mod._sentinel_last_defer = 0.0  # noqa: SLF001
    _utils_mod._video_active_count = 0  # noqa: SLF001
    _utils_mod._video_idle.set()  # noqa: SLF001
    _utils_mod._sentinel_last_run = 0.0  # noqa: SLF001
    _utils_mod._sentinel_first_defer = 0.0  # noqa: SLF001
    _utils_mod._sentinel_defer_count = 0  # noqa: SLF001
    for _t in _utils_mod._sentinel_llm_tasks:  # noqa: SLF001
        _t.cancel()
    _utils_mod._sentinel_llm_tasks = set()  # noqa: SLF001

    _register_services(hass, entry)
    _ensure_default_feature_subentries(hass, entry)
    _assign_first_provider_if_needed(hass, entry)

    db_uri = build_database_uri_from_entry(entry)

    # "Configured" is a property of the entry, not of the connection: clear
    # the not-configured issue before connecting, so a database that is
    # configured but unreachable (the retry raised below) never leaves a stale
    # issue claiming the subentry does not exist.
    if db_uri is not None:
        async_sync_database_issue(hass, entry.entry_id, database_missing=False)
        # An unreachable database retries setup with backoff instead of
        # failing it for good (#748); this connect also gives the card the
        # server's reason, which the pool's timeout would hide. It runs before
        # any provider probe or client build, so a retry while the database
        # is down costs one connect attempt and sends no fallback notices.
        await async_check_database_reachable(db_uri)

    # Resolve effective options (data + options + subentries).
    options = resolve_runtime_options(entry)
    conf = dict(options)
    providers = resolve_model_provider_configs(entry, options)
    model_deployments = build_model_deployments(entry, providers, options)
    fallback_chains = resolve_fallback_chains(entry, providers, options)
    # The store holds one model's vectors, so an embedding fallback is left
    # out of the chain unless it is another Ollama server behind an Ollama
    # primary, whose build is checked (#751, #853).
    embedding_chain, embedding_left_out = split_embedding_chain(
        fallback_chains.get("embedding", [])
    )
    if embedding_left_out:
        fallback_chains["embedding"] = embedding_chain
        chosen = chosen_fallback_ids(entry, providers, options, "embedding")
        embedding_left_out_chosen = [
            p.name for p in embedding_left_out if p.entry_id in chosen
        ]
        (LOGGER.warning if embedding_left_out_chosen else LOGGER.info)(
            "Embedding fallback provider(s) %s left out: only another Ollama "
            "server behind an Ollama primary can be an embedding fallback. Any "
            "other fallback of the primary provider %s adds no failover (a "
            "second OpenAI or Gemini provider shares its client) or could mix "
            "models in the memory store.",
            ", ".join(p.name for p in embedding_left_out),
            embedding_chain[0].name,
        )
    else:
        embedding_left_out_chosen = []
    notified_model_fallbacks: set[tuple[str, str]] = set()
    setup_model_fallbacks: list[tuple[str, str, str, str]] = []

    def _provider_name(provider_id: str | None) -> str:
        """Return a readable provider name for fallback notices."""
        if not provider_id:
            return "unknown provider"
        provider = providers.get(provider_id)
        name = provider.name if provider else provider_id
        for prefix in ("Primary ", "Fallback "):
            if name.startswith(prefix):
                return name.removeprefix(prefix)
        return name

    async def _async_notify_model_fallback(
        category: str,
        primary_provider_id: str | None,
        fallback_provider_id: str | None,
        fallback_deployment: str | None,
        *,
        setup_selected: bool = False,
    ) -> None:
        """Notify the user once when model fallback is active."""
        if not fallback_provider_id:
            return
        dedupe_key = (category, fallback_provider_id)
        if dedupe_key in notified_model_fallbacks:
            return
        notified_model_fallbacks.add(dedupe_key)

        category_label = category.replace("_", " ")
        primary_name = _provider_name(primary_provider_id)
        fallback_name = _provider_name(fallback_provider_id)
        cloud_note = (
            " Cloud model usage may incur provider costs."
            if fallback_deployment == "cloud"
            else ""
        )
        if setup_selected:
            message = (
                f"Using {fallback_name} for {category_label}; {primary_name} "
                f"was unavailable at startup.{cloud_note}"
            )
        else:
            message = (
                f"Using {fallback_name} for {category_label}; {primary_name} "
                f"failed.{cloud_note}"
            )

        notify_service = options.get(CONF_NOTIFY_SERVICE)
        tag = f"hga_model_fallback_{category}_{fallback_provider_id[:16]}"
        if notify_service and isinstance(notify_service, str):
            domain, _, service = notify_service.partition(".")
            if not service:
                service = notify_service
                domain = "notify"
            try:
                await hass.services.async_call(
                    domain,
                    service,
                    {
                        "title": "HGA model fallback active",
                        "message": message,
                        "data": {"tag": tag},
                    },
                    blocking=False,
                )
            except HomeAssistantError as err:
                LOGGER.warning(
                    "Could not send model fallback notification via %s: %s",
                    notify_service,
                    err,
                )
            return

        try:
            await hass.services.async_call(
                "persistent_notification",
                "create",
                {
                    "title": "HGA model fallback active",
                    "message": message,
                    "notification_id": tag,
                },
                blocking=False,
            )
        except HomeAssistantError as err:
            LOGGER.warning("Could not send model fallback notification: %s", err)

    def _schedule_model_fallback_notification(
        category: str,
        primary_provider_id: str,
        fallback_provider_id: str | None,
        fallback_deployment: str | None,
    ) -> None:
        """Schedule a runtime fallback notification without blocking model calls."""
        if not fallback_provider_id:
            return
        hass.async_create_task(
            _async_notify_model_fallback(
                category,
                primary_provider_id,
                fallback_provider_id,
                fallback_deployment,
            )
        )

    def _model_fallback_alert(category: str) -> Any:
        """Return a runtime fallback alert callback for model wrappers."""

        def _alert(
            provider_id: str,
            err: Exception,
            *,
            fallback_to: str | None = None,
            fallback_deployment: str | None = None,
        ) -> None:
            if fallback_to:
                LOGGER.warning(
                    "Fallback activated: provider %s failed (%s), switching to %s",
                    provider_id,
                    _safe_err_summary(err),
                    fallback_to,
                )
            else:
                LOGGER.error(
                    "All providers failed. Last provider %s error: %s",
                    provider_id,
                    _safe_err_summary(err),
                )
            _schedule_model_fallback_notification(
                category, provider_id, fallback_to, fallback_deployment
            )

        return _alert

    def _reset_breakers(provider_id: str) -> None:
        """Make a recovered provider usable at once, whatever its timeouts opened."""
        for breaker in (_chat_cb, _vlm_cb, _sum_cb, _emb_cb):
            breaker.record_success(provider_id)

    # A local server with capacity to spare bypasses every local gate: the
    # chat session, Sentinel admission and its grace, and the probe gate.
    set_model_provider_uncontended(
        bool(
            options.get(
                CONF_MODEL_PROVIDER_UNCONTENDED, RECOMMENDED_MODEL_PROVIDER_UNCONTENDED
            )
        )
    )

    def _provider_deployment(provider_id: str) -> str | None:
        provider = providers.get(provider_id)
        return provider.deployment if provider else None

    # One stall state per provider, shared by the chat, VLM and summarization
    # chains: a provider that stopped answering for one is skipped by all
    # (#828). Changes refresh sensor.sentinel_health. On an edge deployment
    # the probe is not sent while chat or video holds the GPU (#866).
    stall_tracker = ProviderStallTracker(
        hass,
        entry,
        provider_name=_provider_name,
        on_change=lambda: async_dispatcher_send(hass, SIGNAL_SENTINEL_RUN_COMPLETE),
        on_recovered=_reset_breakers,
        run_probe=make_gated_probe_runner(_provider_deployment),
    )

    _chat_cb = CircuitBreaker(
        threshold=FALLBACK_CIRCUIT_BREAKER_THRESHOLD,
        window_seconds=FALLBACK_CIRCUIT_BREAKER_WINDOW_SECONDS,
        cooldown_seconds=FALLBACK_CIRCUIT_BREAKER_COOLDOWN_SECONDS,
    )
    _vlm_cb = CircuitBreaker(
        threshold=FALLBACK_CIRCUIT_BREAKER_THRESHOLD,
        window_seconds=FALLBACK_CIRCUIT_BREAKER_WINDOW_SECONDS,
        cooldown_seconds=FALLBACK_CIRCUIT_BREAKER_COOLDOWN_SECONDS,
    )
    _sum_cb = CircuitBreaker(
        threshold=FALLBACK_CIRCUIT_BREAKER_THRESHOLD,
        window_seconds=FALLBACK_CIRCUIT_BREAKER_WINDOW_SECONDS,
        cooldown_seconds=FALLBACK_CIRCUIT_BREAKER_COOLDOWN_SECONDS,
    )
    _emb_cb = CircuitBreaker(
        threshold=FALLBACK_CIRCUIT_BREAKER_THRESHOLD,
        window_seconds=FALLBACK_CIRCUIT_BREAKER_WINDOW_SECONDS,
        cooldown_seconds=FALLBACK_CIRCUIT_BREAKER_COOLDOWN_SECONDS,
    )

    def _model_instance_for_provider(provider: Any) -> Any:
        """Map a ModelProviderConfig to the initialized LangChain instance."""
        pt = provider.provider_type
        if pt == "openai":
            return openai_provider
        if pt == "openai_compatible":
            return openai_compatible_provider
        if pt == "gemini":
            return gemini_provider
        if pt == "anthropic":
            return anthropic_provider
        if pt == "ollama":
            return ollama_providers.get(_ollama_provider_url(provider))
        return None

    api_key = conf.get(CONF_API_KEY) or _provider_api_key(providers, "openai")
    openai_secret = SecretStr(api_key) if api_key else None
    gemini_key = conf.get(CONF_GEMINI_API_KEY) or _provider_api_key(providers, "gemini")
    gemini_secret = SecretStr(gemini_key) if gemini_key else None
    anthropic_key = conf.get(CONF_ANTHROPIC_API_KEY) or _provider_api_key(
        providers, "anthropic"
    )
    anthropic_secret = SecretStr(anthropic_key) if anthropic_key else None
    openai_compatible_base_url = conf.get(
        CONF_OPENAI_COMPATIBLE_BASE_URL
    ) or _provider_setting(providers, "openai_compatible", "base_url")
    openai_compatible_api_key = (
        conf.get(CONF_OPENAI_COMPATIBLE_API_KEY)
        or _provider_setting(providers, "openai_compatible", "api_key")
        or "none"
    )
    # Embeddings may live on a separate OpenAI-compatible server from chat.
    openai_compatible_embedding_url = (
        conf.get(CONF_OPENAI_COMPATIBLE_EMBEDDING_URL) or openai_compatible_base_url
    )
    openai_compatible_embedding_api_key = (
        conf.get(CONF_OPENAI_COMPATIBLE_EMBEDDING_API_KEY) or openai_compatible_api_key
    )
    base_ollama_url = (
        normalized_ollama_url(conf.get(CONF_OLLAMA_URL)) or RECOMMENDED_OLLAMA_URL
    )
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
    ollama_embedding_url = (
        ollama_url_for_category(conf, "embedding", fallback=base_ollama_url)
        or base_ollama_url
    )
    face_api_url = conf.get(CONF_FACE_API_URL, RECOMMENDED_FACE_API_URL)
    face_api_url = ensure_http_url(face_api_url)

    # Health checks (fast, non-fatal)
    health_timeout = 2.0
    ollama_urls = configured_ollama_urls(conf, fallback=base_ollama_url)
    # Only each category's head provider reaches the options, so an Ollama
    # provider used only as a fallback would never be checked or built at its
    # own URL, leaving its chain empty (issue #749). A head runs at its
    # category's URL, which the options already cover.
    chained = [p for chain in fallback_chains.values() for p in chain]
    fallback_ollama_urls = {
        category: [
            _ollama_provider_url(p) for p in chain[1:] if p.provider_type == "ollama"
        ]
        for category, chain in fallback_chains.items()
    }
    for urls in fallback_ollama_urls.values():
        ollama_urls.extend(url for url in urls if url not in ollama_urls)
    ollama_health: dict[str, bool] = {}
    if ollama_urls:
        ollama_results = await asyncio.gather(
            *(
                ollama_healthy(hass, url, timeout_s=health_timeout)
                for url in ollama_urls
            )
        )
        ollama_health = dict(zip(ollama_urls, ollama_results, strict=False))

    openai_ok, gemini_ok, openai_compatible_ok, anthropic_ok = await asyncio.gather(
        openai_healthy(hass, api_key, timeout_s=health_timeout),
        gemini_healthy(hass, gemini_key, timeout_s=health_timeout),
        openai_compatible_healthy(
            hass,
            openai_compatible_base_url,
            openai_compatible_api_key,
            timeout_s=health_timeout,
        ),
        anthropic_healthy(hass, anthropic_key, timeout_s=health_timeout),
    )
    if (
        openai_compatible_embedding_url == openai_compatible_base_url
        and openai_compatible_embedding_api_key == openai_compatible_api_key
    ):
        openai_compatible_embedding_ok = openai_compatible_ok
    elif openai_compatible_embedding_url:
        openai_compatible_embedding_ok = await openai_compatible_healthy(
            hass,
            openai_compatible_embedding_url,
            openai_compatible_embedding_api_key,
            timeout_s=health_timeout,
        )
    else:
        openai_compatible_embedding_ok = False
    ollama_any_ok = any(ollama_health.values())

    # A failed check is not final: re-probe after HA has started and reload
    # when the provider answers (issue #711). Only providers some category's
    # chain uses count: a spare key, a stale legacy Ollama URL or the default
    # Ollama URL coming back would reload the entry for nothing.
    chained_types = {p.provider_type for p in chained}
    used_ollama_urls = {url for urls in fallback_ollama_urls.values() for url in urls}
    for category, url in (
        ("chat", ollama_chat_url),
        ("vlm", ollama_vlm_url),
        ("summarization", ollama_sum_url),
        ("embedding", ollama_embedding_url),
    ):
        # The head runs at the category's URL; fallbacks at their own, above.
        chain = fallback_chains.get(category)
        if chain and chain[0].provider_type == "ollama":
            used_ollama_urls.add(url)

    recheck_probes: list[ProviderProbe] = [
        ProviderProbe(
            endpoint_label("Ollama", url),
            partial(validate_ollama_url, hass, url, health_timeout),
        )
        for url, healthy in ollama_health.items()
        if not healthy and url in used_ollama_urls
    ]
    for provider_type, label, key, ok, validate in (
        ("openai", "OpenAI", api_key, openai_ok, validate_openai_key),
        ("gemini", "Gemini", gemini_key, gemini_ok, validate_gemini_key),
        ("anthropic", "Anthropic", anthropic_key, anthropic_ok, validate_anthropic_key),
    ):
        if key and not ok and provider_type in chained_types:
            recheck_probes.append(
                ProviderProbe(label, partial(validate, hass, key, health_timeout))
            )

    # Setup builds one OpenAI-compatible chat client from the base URL and one
    # embedding client from the embedding URL, so each URL counts only when a
    # chain of its own kind uses an OpenAI-compatible provider.
    def _chains_use_compatible(categories: tuple[str, ...]) -> bool:
        return any(
            p.provider_type == "openai_compatible"
            for category in categories
            for p in fallback_chains.get(category, [])
        )

    # Keyed by URL and key: setup shares one check only when both match.
    compatible_endpoints: dict[tuple[str, str], bool] = {}
    if openai_compatible_base_url and _chains_use_compatible(
        ("chat", "vlm", "summarization")
    ):
        compatible_endpoints[
            (openai_compatible_base_url, openai_compatible_api_key)
        ] = openai_compatible_ok
    if openai_compatible_embedding_url and _chains_use_compatible(("embedding",)):
        compatible_endpoints.setdefault(
            (openai_compatible_embedding_url, openai_compatible_embedding_api_key),
            openai_compatible_embedding_ok,
        )
    recheck_probes.extend(
        ProviderProbe(
            endpoint_label("OpenAI-compatible", url),
            partial(
                validate_openai_compatible_url,
                hass,
                url,
                key,
                health_timeout,
                capability_path=OPENAI_CHAT_COMPLETIONS_PATH,
            ),
        )
        for (url, key), ok in compatible_endpoints.items()
        if not ok
    )

    http_async_client = get_async_client(hass)
    openai_http_client = await hass.async_add_executor_job(
        partial(httpx.Client, timeout=120)
    )

    # Close registered here, at construction, rather than in
    # async_unload_entry. Home Assistant runs an entry's on-unload callbacks on
    # EVERY failed setup, not only when a loaded entry unloads —
    # `finally: if not result and domain_is_integration:
    # await self._async_process_on_unload(hass)` in ConfigEntry.async_setup —
    # so one registration here covers every abort between this line and the end
    # of setup: both database `return False`s, any uncaught raise out of model
    # selection or platform forwarding, and ConfigEntryNotReady. Patching exit
    # paths one at a time instead would miss the ten or so that are not
    # `return False`, and those are the ones users hit repeatedly (reload until
    # the database comes up), which is exactly the accumulation this is meant
    # to stop.
    #
    # It also fixes the ordering for free: on a successful unload these
    # callbacks run only after async_unload_entry returns, hence after
    # async_unload_platforms, so no live entity is left holding a closed
    # transport. Sentinel's LLM path is the sharp case — it prefers the SYNC
    # invoke on an executor thread that cancellation cannot interrupt.
    #
    # Sync client, so closing it on the loop would block.
    async def _close_openai_http_client() -> None:
        await hass.async_add_executor_job(openai_http_client.close)

    entry.async_on_unload(_close_openai_http_client)

    # Instantiate providers.
    openai_provider: RunnableSerializable[LanguageModelInput, BaseMessage] | None = None
    if openai_ok:
        try:
            openai_provider = ChatOpenAI(
                api_key=openai_secret,
                timeout=120,
                http_client=openai_http_client,
                http_async_client=http_async_client,
                streaming=True,
                stream_usage=True,
                # langchain-openai 1.6 routes some model names (gpt-6 with
                # tools, *codex*, *-pro) to the Responses API on its own; this
                # integration is built and tested on Chat Completions.
                use_responses_api=False,
            ).configurable_fields(
                model_name=ConfigurableField(id="model_name"),
                temperature=ConfigurableField(id="temperature"),
                top_p=ConfigurableField(id="top_p"),
                max_tokens=ConfigurableField(id="max_tokens"),
                reasoning_effort=ConfigurableField(id="reasoning_effort"),
            )
        except Exception:
            LOGGER.exception("OpenAI provider init failed; continuing without it.")

    def _build_ollama_provider(
        url: str,
    ) -> RunnableSerializable[LanguageModelInput, BaseMessage]:
        httpx_kwargs = _ollama_httpx_client_kwargs()
        return ChatOllama(
            model=RECOMMENDED_OLLAMA_CHAT_MODEL,
            base_url=url,
            sync_client_kwargs=httpx_kwargs,
            async_client_kwargs=httpx_kwargs,
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
        except Exception as err:  # noqa: BLE001
            # No traceback: the client's validation error quotes the raw URL.
            LOGGER.error(  # noqa: TRY400
                "Ollama provider init failed for %s (%s); continuing without it.",
                redact_url(url),
                type(err).__name__,
            )

    gemini_provider: RunnableSerializable[LanguageModelInput, BaseMessage] | None = None
    if gemini_ok:
        try:
            gemini_provider = ChatGoogleGenerativeAI(
                google_api_key=gemini_secret,
                model=RECOMMENDED_GEMINI_CHAT_MODEL,
            ).configurable_fields(
                model=ConfigurableField(id="model"),
                temperature=ConfigurableField(id="temperature"),
                top_p=ConfigurableField(id="top_p"),
                max_output_tokens=ConfigurableField(id="max_output_tokens"),
                # No thinking_level field: the pinned generativelanguage
                # protobuf cannot marshal it (see _thinking_gemini).
                thinking_budget=ConfigurableField(id="thinking_budget"),
            )
        except Exception:
            LOGGER.exception("Gemini provider init failed; continuing without it.")

    anthropic_provider: RunnableSerializable[LanguageModelInput, BaseMessage] | None = (
        None
    )
    if anthropic_ok:
        # Top-level cache_control = Anthropic's automatic breakpoint on the
        # LAST block of every request (history reuse inside a tool loop). The
        # stable system prefix gets its own explicit breakpoint per call in
        # core/prompt_cache.py (issue #617).
        try:
            anthropic_chat = SharedClientChatAnthropic(  # type: ignore[call-arg]
                anthropic_api_key=anthropic_secret,  # type: ignore[call-arg]
                model=RECOMMENDED_ANTHROPIC_CHAT_MODEL,  # type: ignore[call-arg]
                model_kwargs={"cache_control": dict(CACHE_CONTROL_EPHEMERAL)},
                streaming=True,
            )
            # Build the SDK client (and its SSL context) in a thread once,
            # BEFORE the provider is published: the per-request copies
            # configurable_fields makes inherit it instead of constructing
            # their own on the event loop (issues #587, #618). If the build
            # fails here it would fail the same way on the loop, so the
            # provider stays unset and the fallback chain takes over.
            await async_prime_async_client(hass, anthropic_chat)
            anthropic_provider = anthropic_chat.configurable_fields(
                model=ConfigurableField(id="model"),
                temperature=ConfigurableField(id="temperature"),
                thinking=ConfigurableField(id="thinking"),
                max_tokens=ConfigurableField(id="max_tokens"),
            )
        except Exception:
            LOGGER.exception("Anthropic provider init failed; continuing without it.")

    openai_compatible_provider: (
        RunnableSerializable[LanguageModelInput, BaseMessage] | None
    ) = None
    if openai_compatible_ok and openai_compatible_base_url:
        try:
            openai_compatible_provider = ChatOpenAI(
                api_key=SecretStr(openai_compatible_api_key),
                base_url=normalize_openai_compatible_base_url(
                    openai_compatible_base_url
                ),
                timeout=120,
                http_client=openai_http_client,
                http_async_client=http_async_client,
                # Most OpenAI-compatible servers serve Chat Completions only;
                # keep langchain-openai 1.6 from switching to /responses by
                # model name.
                use_responses_api=False,
            ).configurable_fields(
                model_name=ConfigurableField(id="model_name"),
                temperature=ConfigurableField(id="temperature"),
                top_p=ConfigurableField(id="top_p"),
                max_tokens=ConfigurableField(id="max_tokens"),
                reasoning_effort=ConfigurableField(id="reasoning_effort"),
                extra_body=ConfigurableField(id="extra_body"),
            )
        except Exception:
            LOGGER.exception(
                "OpenAI-compatible provider init failed; continuing without it."
            )

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
                http_client=openai_http_client,
                http_async_client=http_async_client,
            )
        except Exception:
            LOGGER.exception("OpenAI embeddings init failed; continuing without them.")

    # Every Ollama server embeds with the one configured model, a fallback
    # server included: provider settings hold only a URL, and the same model
    # keeps fallback vectors comparable with the ones already stored.
    ollama_embedding_model = options.get(
        CONF_OLLAMA_EMBEDDING_MODEL, RECOMMENDED_OLLAMA_EMBEDDING_MODEL
    )

    def _build_ollama_embeddings(url: str) -> OllamaEmbeddings | None:
        """Build Ollama embeddings at ``url``, or None if the client fails."""
        try:
            return OllamaEmbeddings(
                model=ollama_embedding_model,
                base_url=url,
                num_ctx=EMBEDDING_MODEL_CTX,
                sync_client_kwargs=_ollama_httpx_client_kwargs(),
                async_client_kwargs=_ollama_httpx_client_kwargs(),
            )
        except Exception as err:  # noqa: BLE001
            # No traceback: the client's validation error quotes the raw URL.
            LOGGER.error(  # noqa: TRY400
                "Ollama embeddings init failed for %s (%s); continuing without them.",
                redact_url(url),
                type(err).__name__,
            )
            return None

    ollama_embeddings = (
        _build_ollama_embeddings(ollama_embedding_url)
        if ollama_health.get(ollama_embedding_url)
        else None
    )

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

    openai_compatible_embeddings: OpenAIEmbeddings | None = None
    if openai_compatible_embedding_ok and openai_compatible_embedding_url:
        try:
            openai_compatible_embeddings = OpenAIEmbeddings(
                api_key=SecretStr(openai_compatible_embedding_api_key),
                base_url=normalize_openai_compatible_base_url(
                    openai_compatible_embedding_url
                ),
                model=options.get(
                    CONF_OPENAI_COMPATIBLE_EMBEDDING_MODEL,
                    RECOMMENDED_OPENAI_COMPATIBLE_EMBEDDING_MODEL,
                ),
                check_embedding_ctx_length=False,
                http_client=openai_http_client,
                http_async_client=http_async_client,
            )
        except Exception:
            LOGGER.exception(
                "OpenAI-compatible embeddings init failed; continuing without them."
            )

    fallback_ollama_embeddings: dict[str, OllamaEmbeddings | None] = {}

    def _ollama_embeddings_for_provider(provider: ModelProviderConfig) -> Any:
        """Return Ollama embeddings at the provider's own URL, if it answered."""
        url = _ollama_provider_url(provider)
        if url == ollama_embedding_url:
            return ollama_embeddings
        # A fallback-only Ollama provider on a server of its own (issue #749),
        # one client per server however many providers point at it.
        if not ollama_health.get(url):
            return None
        if url not in fallback_ollama_embeddings:
            fallback_ollama_embeddings[url] = _build_ollama_embeddings(url)
        return fallback_ollama_embeddings[url]

    def _embedding_instance_for_provider(provider: ModelProviderConfig) -> Any:
        """Map a ModelProviderConfig to the initialized embedding instance."""
        if provider.provider_type == "openai":
            return openai_embeddings
        if provider.provider_type == "openai_compatible":
            return openai_compatible_embeddings
        if provider.provider_type == "gemini":
            return gemini_embeddings
        if provider.provider_type == "ollama":
            return _ollama_embeddings_for_provider(provider)
        return None

    # Choose active embedding provider and build fallback chain
    embedding_model: Any = None
    embedding_provider = options.get(
        CONF_EMBEDDING_MODEL_PROVIDER, RECOMMENDED_EMBEDDING_MODEL_PROVIDER
    )
    index_config: PostgresIndexConfig | None = None
    # Each Ollama embedding server's build of the model: a fallback holding
    # another build embeds differently, and the primary's goes into the
    # store's model record (#855). Asked only of servers that just answered,
    # with a longer wait than the health check's: Home Assistant's busy
    # startup often ran past that one, and a build unread here leaves the
    # fallback check below blind (#860).
    ollama_embedding_digests: dict[str, str] = {}
    if embedding_provider == "ollama" and db_uri is not None:
        digest_urls = [
            url
            for url in (
                ollama_embedding_url,
                *fallback_ollama_urls.get("embedding", []),
            )
            if ollama_health.get(url)
        ]
        ollama_embedding_digests = await ollama_model_digests(
            hass,
            list(dict.fromkeys(digest_urls)),
            ollama_embedding_model,
            timeout_s=DIGEST_FETCH_TIMEOUT_S,
        )
    primary_embedding_digest = ollama_embedding_digests.get(ollama_embedding_url)
    # With the primary down there is nothing live to compare a fallback's build
    # with, and the fallback is what writes then: compare it with the build
    # the store's record names instead. A primary that answered but whose
    # build read timed out is compared with later, once the record check has
    # read it: the record may be stale after a pull on both servers (#860).
    reference_embedding_digest = primary_embedding_digest
    if (
        reference_embedding_digest is None
        and db_uri is not None
        and ollama_embeddings is None
        and any(url != ollama_embedding_url for url in ollama_embedding_digests)
    ):
        reference_embedding_digest = await async_recorded_ollama_digest(
            db_uri, ollama_embedding_model
        )
    embedding_other_build: list[str] = []
    # Kept Ollama fallbacks not compared with the primary's live build (theirs
    # unread, or the primary's): checked again after setup (#860).
    embedding_fallback_unchecked: list[FallbackBuild] = []
    if embedding_provider == "openai":
        embedding_model = openai_embeddings
    elif embedding_provider == "openai_compatible":
        embedding_model = openai_compatible_embeddings
    elif embedding_provider == "gemini":
        embedding_model = gemini_embeddings
    else:
        embedding_model = ollama_embeddings
    # The store's re-embed (#855) runs on the primary's own client: a fallback
    # in the chain may hold a build nobody read.
    primary_embeddings = embedding_model
    # Read by the embedding-model repair check at the end of setup.
    ollama_embeddings_selected = embedding_model is not None and (
        embedding_model is ollama_embeddings
    )

    # Setup notices of held fallbacks, sent when the chain admits them.
    held_fallback_notices: dict[str, str | None] = {}

    @callback
    def _announce_admitted_embedding_fallback(provider_id: str) -> None:
        if provider_id not in held_fallback_notices:
            return
        # Only now the deployment that embeds: held, it answered nothing.
        if (deployment := held_fallback_notices[provider_id]) is not None:
            model_deployments["embedding"] = deployment
        _schedule_model_fallback_notification(
            "embedding",
            _emb_fb[0].entry_id,
            provider_id,
            held_fallback_notices.pop(provider_id),
        )

    async def _async_announce_setup_embedding_fallback(
        active: tuple[Any, str, str],
    ) -> None:
        """
        Say that setup starts embedding on a fallback, the primary being down.

        A fallback held until its build is checked (#857) may answer nothing,
        so it is logged as held instead of announced as standing in.
        """
        _model, deployment, active_provider_id = active
        if active_provider_id == _emb_fb[0].entry_id:
            return
        provider = providers.get(active_provider_id)
        held = next(
            (
                fb
                for fb in embedding_fallback_unchecked
                if fb.provider_id == active_provider_id
            ),
            None,
        )
        if held is not None:
            # Announced once the chain admits it, if it ever does.
            held_fallback_notices[active_provider_id] = (
                provider.deployment if provider else deployment
            )
            LOGGER.warning(
                "The embedding primary %s is down; fallback provider %s stands in "
                "once its build of %s is read and matches the memory store's, "
                "checked at its first call. Until then saving memories and "
                "captions fails.",
                _emb_fb[0].name,
                held.name,
                ollama_embedding_model,
            )
            return
        model_deployments["embedding"] = provider.deployment if provider else deployment
        await _async_notify_model_fallback(
            "embedding",
            _emb_fb[0].entry_id,
            active_provider_id,
            provider.deployment if provider else deployment,
            setup_selected=True,
        )

    # Wrap embedding model with fallback if multiple providers available
    _emb_fb = fallback_chains.get("embedding", [])
    if _emb_fb:
        _emb_chain = []
        if embedding_model is not None:
            _emb_chain.append(
                (embedding_model, _emb_fb[0].deployment, _emb_fb[0].entry_id)
            )
        for p in _emb_fb[1:]:
            m = _embedding_instance_for_provider(p)
            # An Ollama fallback at the primary's URL is the primary's client:
            # the same client twice is no failover, only a false "fell back"
            # and a needless tool index rebuild (#751 review).
            if m is None or any(m is c for c, _d, _p in _emb_chain):
                continue
            # Another build of the model on a fallback server embeds
            # differently from the primary's; an unknown digest decides
            # nothing (#855). Every fallback here is an Ollama server: the
            # split above leaves out any other (#853).
            digest = ollama_embedding_digests.get(_ollama_provider_url(p))
            if builds_differ(digest, reference_embedding_digest):
                embedding_other_build.append(p.name)
                continue
            # Without a database no build was read and there is no store to
            # protect, so there is nothing to check later either.
            if db_uri is not None and (
                digest is None or primary_embedding_digest is None
            ):
                embedding_fallback_unchecked.append(
                    FallbackBuild(p.entry_id, p.name, _ollama_provider_url(p), digest)
                )
            _emb_chain.append((m, p.deployment, p.entry_id))
        if embedding_other_build:
            LOGGER.warning(
                "Embedding fallback provider(s) %s left out: their server holds "
                "another build of %s than the primary provider %s (or, while it "
                "is down, than the memory store's record), which would mix "
                "models in the memory store.",
                ", ".join(embedding_other_build),
                ollama_embedding_model,
                _emb_fb[0].name,
            )
        if len(_emb_chain) > 1:
            await _async_announce_setup_embedding_fallback(_emb_chain[0])

            def _mark_tool_index_stale(
                previous_provider_id: str, provider_id: str, *, notify: bool = True
            ) -> None:
                """
                Mark tool vectors stale after an embedding provider switch.

                Without ``notify``, for a fallback removed from the chain for
                its build (#860): nothing fell back, so no notice is sent.
                """
                runtime_data = getattr(entry, "runtime_data", None)
                if runtime_data is None:
                    return
                runtime_data.tool_content_hashes.clear()
                runtime_data.tool_index_ready = False
                runtime_data.tool_index_failed = False
                # A build in flight embedded with the previous provider: let
                # it finish and it would declare that index ready.
                task = runtime_data.tool_index_task
                if task is not None and not task.done():
                    task.cancel()
                LOGGER.warning(
                    "Tool index marked stale after embedding provider %s %s to "
                    "%s; it will be rebuilt on the next indexing pass.",
                    previous_provider_id,
                    "switched" if notify else "was removed, handing the chain",
                    provider_id,
                )
                async_dispatcher_send(hass, SIGNAL_TOOL_INDEX_UPDATED, "stale", 0)
                if not notify:
                    return
                provider = providers.get(provider_id)
                _schedule_model_fallback_notification(
                    "embedding",
                    previous_provider_id,
                    provider_id,
                    provider.deployment if provider else None,
                )

            embedding_model = FallbackEmbeddings(
                _emb_chain,
                circuit_breaker=_emb_cb,
                switch_callback=_mark_tool_index_stale,
                removed_callback=partial(_mark_tool_index_stale, notify=False),
            )
            _hold_unchecked_embedding_fallbacks(
                hass,
                entry,
                embedding_model,
                embedding_fallback_unchecked,
                on_admitted=_announce_admitted_embedding_fallback,
            )
        elif _emb_chain:
            await _async_announce_setup_embedding_fallback(_emb_chain[0])
            embedding_model = _emb_chain[0][0]
            if embedding_fallback_unchecked:
                # Only an unchecked fallback answered at setup: wrapped, so
                # it is held like one in a longer chain (#857).
                embedding_model = FallbackEmbeddings(_emb_chain, circuit_breaker=None)
                _hold_unchecked_embedding_fallbacks(
                    hass,
                    entry,
                    embedding_model,
                    embedding_fallback_unchecked,
                    on_admitted=_announce_admitted_embedding_fallback,
                )
        else:
            embedding_model = None

    # Read by the embedding-size repair check at the end of setup (#760).
    embedding_size_provider = embedding_size_check_provider(
        _emb_fb[0] if _emb_fb else None,
        primary_type=embedding_provider,
        primary_embeddings=primary_embeddings,
    )

    if embedding_model is None and embedding_chain:
        # Configured but nothing in the chain could be built (down at boot, or
        # a client that failed to build): keep the store's index with a client
        # that fails at once, so a memory or caption saved meanwhile fails
        # (langgraph embeds before it writes) instead of being saved without a
        # vector no search finds again, and memory search fails open without
        # waiting on a stalled server. The provider re-check reloads once the
        # primary answers (#751 review).
        embedding_model = UnavailableEmbeddings(embedding_chain[0].name)
        LOGGER.warning(
            "Embedding provider %s is not available; saving memories and camera "
            "captions fails until it answers and Home Generative Agent reloads.",
            embedding_chain[0].name,
        )

    embedding_dims = _configured_embedding_dims(options)
    # The model the store's vectors should come from, compared with the
    # store's record at the end of setup (#855).
    current_embedding_fingerprint = (
        embedding_fingerprint(
            embedding_provider,
            _configured_embedding_model(options, embedding_provider),
            dims=embedding_dims,
            digest=primary_embedding_digest if embedding_provider == "ollama" else None,
        )
        if embedding_model is not None
        else None
    )
    if embedding_model is None:
        LOGGER.warning(
            "No embeddings provider available; vector store will be limited.",
        )
    else:
        index_config = PostgresIndexConfig(
            embed=partial(generate_embeddings, embedding_model),
            dims=embedding_dims,
            fields=["content"],
        )

    # Open Postgres
    connection_kwargs = {
        "autocommit": True,
        "prepare_threshold": 0,
        "row_factory": dict_row,
    }

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
        # Does not wait for a connection (psycopg-pool's open defaults to
        # wait=False), so it cannot fail on an unreachable server; the first
        # use below does.
        await pool.open()

        try:
            # Initialize database for long-term memory.
            store = AsyncPostgresStore(
                pool,
                index=index_config or None,
            )
            # Initialize database for thread-based (short-term) memory.
            checkpointer = AsyncPostgresSaver(pool)
            # The schema needs pgvector in this database (#756).
            await async_ensure_pgvector(pool)
            # First-time setup (if needed)
            await _bootstrap_db_once(hass, entry, store, checkpointer, pool)
            await _bootstrap_vectors_once(hass, entry, store, pool)

            # Migrate person gallery DB schema (if needed)
            try:
                await migrate_person_gallery(pool)
            except Exception as err:
                # A lost connection is retried below; logging it here would
                # read as a migration bug.
                if not is_connection_failure(err):
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
        except Exception as err:
            await pool.close()
            # A ConfigEntryError already says what is wrong and how to fix
            # it; HA shows it on the integration card and logs it.
            if isinstance(err, ConfigEntryError):
                raise
            # The database went away after the preflight connect (a restart
            # mid-setup): retry, as for one that was down from the start.
            if is_connection_failure(err):
                raise database_unreachable(err) from err
            LOGGER.exception("Postgresql setup failed; pool closed.")
            return False
    else:
        # Log loudly: nothing downstream fails without a database (chat
        # included), so without this line and the repair issue raised at the
        # end of setup the gap is invisible. A fresh entry with no Model
        # Provider yet is mid-onboarding, not misconfigured: info only.
        LOGGER.log(
            logging.WARNING if providers else logging.INFO,
            "No database is configured for this entry; running with in-memory "
            "storage only. Conversation history is lost on every reload or "
            "restart, long-term memory (conversation summaries, camera "
            "activity recall, alarm-arming consent) is unavailable, the person "
            "gallery (face recognition) is unavailable, and the Sentinel "
            "baseline anomaly detector is off. %s",
            NO_DATABASE_REMEDIATION,
        )
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
    chat_reasoning = options.get(CONF_CHAT_REASONING)
    chat_reasoning_budget = options.get(CONF_CHAT_REASONING_BUDGET)
    chat_reasoning_by_model: dict[str, Any] = dict(
        options.get(CONF_CHAT_REASONING_BY_MODEL) or {}
    )

    def _chat_thinking_for_model(
        provider_type: str, model_name: Any, *, default_to_flat: bool = True
    ) -> dict[str, Any]:
        """
        Thinking configurable entries for a chat model, keyed by model name.

        The flat effective setting mirrors the primary chat model, so fallback
        models pass default_to_flat=False and only apply a remembered per-model
        entry.
        """
        remembered = chat_reasoning_by_model.get(str(model_name))
        if isinstance(remembered, Mapping):
            reasoning_val = remembered.get("reasoning")
            budget_val = remembered.get("budget")
            # A budget-only entry (older stored data) must not mask the flat
            # setting for the primary model.
            if reasoning_val is None and default_to_flat:
                reasoning_val = chat_reasoning
                budget_val = budget_val or chat_reasoning_budget
        elif default_to_flat:
            reasoning_val = chat_reasoning
            budget_val = chat_reasoning_budget
        else:
            return {}
        try:
            budget_int = int(budget_val) if budget_val else None
        except (TypeError, ValueError):
            budget_int = None
        return thinking_configurable(
            provider_type=provider_type,
            reasoning=reasoning_val,
            budget=budget_int,
            model=str(model_name),
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
        openai_chat_model = options.get(
            CONF_OPENAI_CHAT_MODEL, RECOMMENDED_OPENAI_CHAT_MODEL
        )
        chat_model = (openai_provider or NullChat()).with_config(
            config={
                "configurable": {
                    "model_name": openai_chat_model,
                    "temperature": chat_temp,
                    "top_p": CHAT_MODEL_TOP_P,
                    **_chat_thinking_for_model("openai", openai_chat_model),
                }
            }
        )
    elif chat_provider == "openai_compatible":
        openai_compatible_chat_model = options.get(
            CONF_OPENAI_COMPATIBLE_CHAT_MODEL,
            RECOMMENDED_OPENAI_COMPATIBLE_CHAT_MODEL,
        )
        chat_model = (openai_compatible_provider or NullChat()).with_config(
            config={
                "configurable": {
                    "model_name": openai_compatible_chat_model,
                    "temperature": chat_temp,
                    "top_p": CHAT_MODEL_TOP_P,
                    **_chat_thinking_for_model(
                        "openai_compatible", openai_compatible_chat_model
                    ),
                }
            }
        )
    elif chat_provider == "gemini":
        gemini_chat_model = options.get(
            CONF_GEMINI_CHAT_MODEL, RECOMMENDED_GEMINI_CHAT_MODEL
        )
        chat_model = (gemini_provider or NullChat()).with_config(
            config={
                "configurable": {
                    "model": gemini_chat_model,
                    **gemini_sampling_configurable(
                        model=gemini_chat_model,
                        temperature=chat_temp,
                        top_p=CHAT_MODEL_TOP_P,
                        recommended_temperature=RECOMMENDED_CHAT_MODEL_TEMPERATURE,
                    ),
                    **_chat_thinking_for_model("gemini", gemini_chat_model),
                }
            }
        )
    elif chat_provider == "anthropic":
        anthropic_chat_model = options.get(
            CONF_ANTHROPIC_CHAT_MODEL, RECOMMENDED_ANTHROPIC_CHAT_MODEL
        )
        chat_model = (anthropic_provider or NullChat()).with_config(
            config={
                "configurable": {
                    "model": anthropic_chat_model,
                    "temperature": chat_temp,
                    **_chat_thinking_for_model("anthropic", anthropic_chat_model),
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
    elif vlm_provider == "openai_compatible":
        vision_model = (openai_compatible_provider or NullChat()).with_config(
            config={
                "configurable": {
                    "model_name": options.get(
                        CONF_OPENAI_COMPATIBLE_VLM, RECOMMENDED_OPENAI_COMPATIBLE_VLM
                    ),
                    "temperature": vlm_temp,
                    "top_p": VLM_TOP_P,
                }
            }
        )
    elif vlm_provider == "gemini":
        gemini_vlm = options.get(CONF_GEMINI_VLM, RECOMMENDED_GEMINI_VLM)
        vision_model = (gemini_provider or NullChat()).with_config(
            config={
                "configurable": {
                    "model": gemini_vlm,
                    **gemini_sampling_configurable(
                        model=gemini_vlm,
                        temperature=vlm_temp,
                        top_p=VLM_TOP_P,
                        recommended_temperature=RECOMMENDED_VLM_TEMPERATURE,
                    ),
                }
            }
        )
    elif vlm_provider == "anthropic":
        vision_model = (anthropic_provider or NullChat()).with_config(
            config={
                "configurable": {
                    "model": options.get(CONF_ANTHROPIC_VLM, RECOMMENDED_ANTHROPIC_VLM),
                    "temperature": vlm_temp,
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
    elif sum_provider == "openai_compatible":
        summarization_model = (openai_compatible_provider or NullChat()).with_config(
            config={
                "configurable": {
                    "model_name": options.get(
                        CONF_OPENAI_COMPATIBLE_SUMMARIZATION_MODEL,
                        RECOMMENDED_OPENAI_COMPATIBLE_SUMMARIZATION_MODEL,
                    ),
                    "temperature": sum_temp,
                    "top_p": SUMMARIZATION_MODEL_TOP_P,
                }
            }
        )
    elif sum_provider == "gemini":
        gemini_sum_model = options.get(
            CONF_GEMINI_SUMMARIZATION_MODEL,
            RECOMMENDED_GEMINI_SUMMARIZATION_MODEL,
        )
        summarization_model = (gemini_provider or NullChat()).with_config(
            config={
                "configurable": {
                    "model": gemini_sum_model,
                    **gemini_sampling_configurable(
                        model=gemini_sum_model,
                        temperature=sum_temp,
                        top_p=SUMMARIZATION_MODEL_TOP_P,
                        recommended_temperature=(
                            RECOMMENDED_SUMMARIZATION_MODEL_TEMPERATURE
                        ),
                    ),
                }
            }
        )
    elif sum_provider == "anthropic":
        summarization_model = (anthropic_provider or NullChat()).with_config(
            config={
                "configurable": {
                    "model": options.get(
                        CONF_ANTHROPIC_SUMMARIZATION_MODEL,
                        RECOMMENDED_ANTHROPIC_SUMMARIZATION_MODEL,
                    ),
                    "temperature": sum_temp,
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

    def _fallback_model_name(provider: ModelProviderConfig, category: str) -> Any:
        """Return the model name to use for a fallback provider/category."""
        settings = provider.data.get("settings", {})
        spec = MODEL_CATEGORY_SPECS.get(category, {})
        return settings.get(f"{category}_model") or spec.get(
            "recommended_models", {}
        ).get(provider.provider_type)

    def _category_top_p(category: str) -> float | None:
        """Return the top-p value for chat-like categories."""
        if category == "chat":
            return CHAT_MODEL_TOP_P
        if category == "vlm":
            return VLM_TOP_P
        if category == "summarization":
            return SUMMARIZATION_MODEL_TOP_P
        return None

    def _configured_cloud_model(
        model: Any,
        provider: ModelProviderConfig,
        category: str,
        model_name: Any,
        temperature: Any,
    ) -> Any:
        """Configure a cloud fallback model for its provider/category."""
        model_key = (
            "model_name"
            if provider.provider_type in ("openai", "openai_compatible")
            else "model"
        )
        config = {model_key: model_name, "temperature": temperature}
        if (top_p := _category_top_p(category)) is not None:
            config["top_p"] = top_p
        if provider.provider_type == "gemini":
            spec = MODEL_CATEGORY_SPECS.get(category, {})
            config.update(
                gemini_sampling_configurable(
                    model=model_name,
                    temperature=temperature,
                    top_p=top_p,
                    recommended_temperature=spec.get("recommended_temperature"),
                )
            )
        if category == "chat":
            config.update(
                _chat_thinking_for_model(
                    provider.provider_type, model_name, default_to_flat=False
                )
            )
        return model.with_config(config={"configurable": config})

    def _configured_ollama_model(
        model: Any,
        provider: ModelProviderConfig,
        category: str,
        model_name: Any,
        temperature: Any,
    ) -> Any:
        """Configure an Ollama fallback model for its category."""
        settings = provider.data.get("settings", {})
        reasoning_enabled = settings.get("reasoning", ollama_reasoning)
        if category == "chat":
            remembered = chat_reasoning_by_model.get(str(model_name))
            if isinstance(remembered, Mapping) and "reasoning" in remembered:
                reasoning_enabled = remembered["reasoning"]
        rf_fallback = reasoning_field(model=str(model_name), enabled=reasoning_enabled)
        config = {
            "model": model_name,
            "temperature": temperature,
            **rf_fallback,
        }
        if category == "chat":
            config.update(
                {
                    "top_p": CHAT_MODEL_TOP_P,
                    "num_predict": CHAT_MODEL_MAX_TOKENS,
                    "num_ctx": settings.get(
                        "chat_context", RECOMMENDED_OLLAMA_CONTEXT_SIZE
                    ),
                    "repeat_penalty": CHAT_MODEL_REPEAT_PENALTY,
                    "keep_alive": settings.get(
                        "chat_keepalive", RECOMMENDED_OLLAMA_CHAT_KEEPALIVE
                    ),
                }
            )
        elif category == "vlm":
            config.update(
                {
                    "top_p": VLM_TOP_P,
                    "num_predict": VLM_NUM_PREDICT,
                    "num_ctx": settings.get(
                        "vlm_context", RECOMMENDED_OLLAMA_CONTEXT_SIZE
                    ),
                    "repeat_penalty": VLM_REPEAT_PENALTY,
                    "mirostat": VLM_MIRO_STAT,
                    "keep_alive": settings.get(
                        "vlm_keepalive", RECOMMENDED_OLLAMA_VLM_KEEPALIVE
                    ),
                }
            )
        elif category == "summarization":
            config.update(
                {
                    "top_p": SUMMARIZATION_MODEL_TOP_P,
                    "num_predict": SUMMARIZATION_MODEL_PREDICT,
                    "num_ctx": settings.get(
                        "summarization_context", RECOMMENDED_OLLAMA_CONTEXT_SIZE
                    ),
                    "repeat_penalty": SUMMARIZATION_MODEL_REPEAT_PENALTY,
                    "mirostat": SUMMARIZATION_MIRO_STAT,
                    "keep_alive": settings.get(
                        "summarization_keepalive",
                        RECOMMENDED_OLLAMA_SUMMARIZATION_KEEPALIVE,
                    ),
                }
            )
        return model.with_config(config={"configurable": config})

    def _configured_model_for_provider(
        provider: ModelProviderConfig, category: str
    ) -> Any:
        """Return a category-configured model instance for a fallback provider."""
        model = _model_instance_for_provider(provider)
        if model is None:
            return None

        model_name = _fallback_model_name(provider, category)
        spec = MODEL_CATEGORY_SPECS.get(category, {})
        temp_key = spec.get("temperature_key")
        temperature = options.get(temp_key) if temp_key else None
        if temperature is None:
            temperature = spec.get("recommended_temperature")

        if provider.provider_type in (
            "openai",
            "openai_compatible",
            "gemini",
        ):
            return _configured_cloud_model(
                model, provider, category, model_name, temperature
            )
        if provider.provider_type == "anthropic":
            anthropic_config = {"model": model_name, "temperature": temperature}
            if category == "chat":
                anthropic_config.update(
                    _chat_thinking_for_model(
                        "anthropic", model_name, default_to_flat=False
                    )
                )
            return model.with_config(config={"configurable": anthropic_config})
        if provider.provider_type == "ollama":
            return _configured_ollama_model(
                model, provider, category, model_name, temperature
            )
        return model

    def _is_unavailable_model(model: Any) -> bool:
        """Return True for placeholder models that should not block fallback."""
        return isinstance(model, NullChat)

    def _append_available_model(
        chain: list[tuple[Any, str, str]],
        model: Any,
        deployment: str,
        provider_id: str,
        category: str,
    ) -> None:
        """Append a model to a fallback chain if it is usable."""
        if _is_unavailable_model(model):
            LOGGER.debug(
                "Skipping unavailable %s model for fallback provider %s.",
                category,
                provider_id,
            )
            return
        chain.append((model, deployment, provider_id))

    def _breaker_for(
        chain: list[Any], breaker: CircuitBreaker
    ) -> CircuitBreaker | None:
        """
        Return the breaker for a chain, or None for a lone provider.

        A breaker only helps by skipping to another provider; on a lone one
        it would refuse calls for its whole cooldown after the server is back.
        """
        return breaker if len(chain) > 1 else None

    def _finalize_model_chain(
        category: str,
        chain: list[tuple[Any, str, str]],
        configured_providers: list[ModelProviderConfig],
        unavailable_model: Any,
        wrapper_factory: Any,
    ) -> Any:
        """Finalize fallback chain and update effective deployment metadata."""
        primary_provider = configured_providers[0]
        if not chain:
            configured = [
                f"{provider.entry_id}:{provider.deployment}"
                for provider in configured_providers
            ]
            if len(configured_providers) <= 1:
                LOGGER.debug(
                    "No configured fallback for %s; primary provider %s is "
                    "unavailable and will remain the placeholder model until "
                    "the integration is reloaded.",
                    category,
                    primary_provider.entry_id,
                )
            else:
                LOGGER.warning(
                    "No available %s model providers in resolved fallback chain %s; "
                    "keeping unavailable primary provider %s. Check fallback "
                    "provider health and %s capability.",
                    category,
                    configured,
                    primary_provider.entry_id,
                    category,
                )
            return unavailable_model

        _active_model, active_deployment, active_provider_id = chain[0]
        model_deployments[category] = active_deployment
        if active_provider_id != primary_provider.entry_id:
            LOGGER.warning(
                "Fallback selected at setup for %s: primary provider %s was "
                "unavailable; using provider %s (deployment=%s). The primary is "
                "re-checked in the background, and Home Generative Agent "
                "reloads to switch back once it answers.",
                category,
                primary_provider.entry_id,
                active_provider_id,
                active_deployment,
            )
            setup_model_fallbacks.append(
                (
                    category,
                    primary_provider.entry_id,
                    active_provider_id,
                    active_deployment,
                )
            )

        # A single provider is wrapped too, so a stall is detected, shown and
        # failed fast even with no fallback to go to (#828).
        return wrapper_factory(chain)

    # ----- Wrap chat / VLM / summarization with fallback chains -----
    # Primary model already carries per-feature config (model_name, temp, etc.).
    # Fallbacks are configured per provider/category before entering the chain.
    _chat_fb = fallback_chains.get("chat", [])
    if _chat_fb:
        _chat_chain = []
        _append_available_model(
            _chat_chain,
            chat_model,
            _chat_fb[0].deployment,
            _chat_fb[0].entry_id,
            "chat",
        )
        for p in _chat_fb[1:]:
            m = _configured_model_for_provider(p, "chat")
            if m is not None:
                _append_available_model(
                    _chat_chain, m, p.deployment, p.entry_id, "chat"
                )
        chat_model = _finalize_model_chain(
            "chat",
            _chat_chain,
            _chat_fb,
            chat_model,
            lambda chain: FallbackChatModel(
                chain,
                circuit_breaker=_breaker_for(chain, _chat_cb),
                alert_callback=_model_fallback_alert("chat"),
                stall_tracker=stall_tracker,
            ),
        )

    _vlm_fb = fallback_chains.get("vlm", [])
    if _vlm_fb:
        _vlm_chain = []
        _append_available_model(
            _vlm_chain,
            vision_model,
            _vlm_fb[0].deployment,
            _vlm_fb[0].entry_id,
            "vlm",
        )
        for p in _vlm_fb[1:]:
            m = _configured_model_for_provider(p, "vlm")
            if m is not None:
                _append_available_model(_vlm_chain, m, p.deployment, p.entry_id, "vlm")
        vision_model = _finalize_model_chain(
            "vlm",
            _vlm_chain,
            _vlm_fb,
            vision_model,
            lambda chain: FallbackVLM(
                chain,
                circuit_breaker=_breaker_for(chain, _vlm_cb),
                alert_callback=_model_fallback_alert("vlm"),
                stall_tracker=stall_tracker,
            ),
        )

    _sum_fb = fallback_chains.get("summarization", [])
    if _sum_fb:
        _sum_chain = []
        _append_available_model(
            _sum_chain,
            summarization_model,
            _sum_fb[0].deployment,
            _sum_fb[0].entry_id,
            "summarization",
        )
        for p in _sum_fb[1:]:
            m = _configured_model_for_provider(p, "summarization")
            if m is not None:
                _append_available_model(
                    _sum_chain, m, p.deployment, p.entry_id, "summarization"
                )
        summarization_model = _finalize_model_chain(
            "summarization",
            _sum_chain,
            _sum_fb,
            summarization_model,
            lambda chain: FallbackChatModel(
                chain,
                circuit_breaker=_breaker_for(chain, _sum_cb),
                alert_callback=_model_fallback_alert("summarization"),
                stall_tracker=stall_tracker,
            ),
        )

    for category, primary_id, fallback_id, fallback_deployment in setup_model_fallbacks:
        await _async_notify_model_fallback(
            category,
            primary_id,
            fallback_id,
            fallback_deployment,
            setup_selected=True,
        )

    video_analyzer = VideoAnalyzer(hass, entry)
    suppression = SuppressionManager(hass)
    await suppression.async_load()
    audit_max = int(
        options.get(CONF_AUDIT_HOT_MAX_RECORDS, RECOMMENDED_AUDIT_HOT_MAX_RECORDS)
    )
    audit_store = AuditStore(hass, max_records=audit_max)
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
    # Network / HA-security audit (docs/network-security-plan.md): the
    # per-install salt pseudonymizes addresses before anything persists them,
    # and the auth inventory is what the auth-change rules compare against.
    # Created only when Sentinel is enabled so that removing the Sentinel
    # subentry (which deletes both files) does not see them recreated on the
    # reload that follows.
    pseudonymizer = None
    auth_inventory = None
    network_inventory = None
    if options.get(CONF_SENTINEL_ENABLED, RECOMMENDED_SENTINEL_ENABLED):
        pseudonymizer = await async_load_pseudonymizer(hass)
        auth_inventory = AuthInventory(hass)
        await auth_inventory.async_load()
        network_inventory = NetworkInventory(hass)
        await network_inventory.async_load()
    action_handler = ActionHandler(
        hass,
        suppression,
        audit_store,
        entry_id=entry.entry_id,
        notify_service=options.get(CONF_NOTIFY_SERVICE),
    )
    action_handler.network_inventory = network_inventory
    notifier = SentinelNotifier(
        hass, options, suppression, action_handler, audit_store=audit_store
    )
    notifier.start()
    sentinel_run_stats: dict[str, Any] = {
        "sentinel_admission_degraded": False,
        "sentinel_admission_degraded_category": None,
        "sentinel_admission_consecutive_deferrals": 0,
        "sentinel_admission_starved_for_s": 0,
    }
    chat_deployment = model_deployments.get("chat", "edge")
    explainer = None
    if options.get(CONF_EXPLAIN_ENABLED, RECOMMENDED_EXPLAIN_ENABLED):
        explainer = LLMExplainer(
            chat_model,
            deployment=chat_deployment,
            health_stats=sentinel_run_stats,
            response_language=options.get(CONF_SENTINEL_RESPONSE_LANGUAGE, ""),
        )

    # Issue #262: optional LLM triage service.
    triage_service: SentinelTriageService | None = None
    if options.get(CONF_SENTINEL_TRIAGE_ENABLED, RECOMMENDED_SENTINEL_TRIAGE_ENABLED):
        triage_timeout = int(
            options.get(
                CONF_SENTINEL_TRIAGE_TIMEOUT_SECONDS,
                RECOMMENDED_SENTINEL_TRIAGE_TIMEOUT_SECONDS,
            )
        )
        triage_max_per_run = int(
            options.get(
                CONF_SENTINEL_TRIAGE_MAX_PER_RUN,
                RECOMMENDED_SENTINEL_TRIAGE_MAX_PER_RUN,
            )
        )
        triage_service = SentinelTriageService(
            chat_model,
            timeout_seconds=triage_timeout,
            deployment=chat_deployment,
            health_stats=sentinel_run_stats,
            max_per_run=triage_max_per_run,
        )
        LOGGER.info(
            "Sentinel LLM triage enabled (timeout=%ds, at most %d call(s) per run).",
            triage_timeout,
            triage_max_per_run,
        )
    # Issue #265: optional baseline updater (requires PostgreSQL pool).
    # Baseline collection is gated on sentinel_enabled acting as master switch.
    baseline_updater: SentinelBaselineUpdater | None = None
    if (
        pool is not None
        and options.get(CONF_SENTINEL_ENABLED, RECOMMENDED_SENTINEL_ENABLED)
        and options.get(
            CONF_SENTINEL_BASELINE_ENABLED, RECOMMENDED_SENTINEL_BASELINE_ENABLED
        )
    ):
        baseline_updater = SentinelBaselineUpdater(hass, pool, dict(options))
        await baseline_updater.async_initialize()
        LOGGER.info("Sentinel baseline updater enabled.")
    sentinel = SentinelEngine(
        hass,
        options,
        suppression,
        notifier,
        audit_store,
        explainer,
        rule_registry=rule_registry,
        entry_id=entry.entry_id,
        triage_service=triage_service,
        baseline_updater=baseline_updater,
        run_stats=sentinel_run_stats,
        auth_inventory=auth_inventory,
        pseudonymizer=pseudonymizer,
        network_inventory=network_inventory,
    )
    discovery_engine = SentinelDiscoveryEngine(
        hass=hass,
        options=options,
        model=chat_model,
        store=discovery_store,
        rule_registry=rule_registry,
        proposal_store=proposal_store,
        baseline_updater=baseline_updater,
        deployment=chat_deployment,
        health_stats=sentinel_run_stats,
    )

    face_recognition = options.get(CONF_FACE_RECOGNITION, RECOMMENDED_FACE_RECOGNITION)
    if face_recognition and person_gallery is None:
        # The root cause (no database) was already logged at WARNING above;
        # this is the consequence, not a second problem.
        LOGGER.info(
            "Face recognition is enabled but person gallery is unavailable; "
            "disabling face recognition for this entry."
        )
        face_recognition = False

    # Save runtime data.
    entry.runtime_data = HGAData(
        options=options,
        chat_model=chat_model,
        chat_model_options=ollama_chat_model_options,
        vision_model=vision_model,
        summarization_model=summarization_model,
        model_deployments=model_deployments,
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
        baseline_updater=baseline_updater,
        auth_inventory=auth_inventory,
        network_inventory=network_inventory,
        openai_http_client=openai_http_client,
        stall_tracker=stall_tracker,
        embeddings=embedding_model,
        primary_embeddings=primary_embeddings,
        embedding_fingerprint=current_embedding_fingerprint,
        embedding_digest_source=(
            (ollama_embedding_url, ollama_embedding_model)
            if embedding_provider == "ollama"
            else None
        ),
        embedding_fallback_gate=EmbeddingFallbackGate(
            primary=embedding_chain[0].name if embedding_chain else "",
            model=ollama_embedding_model,
            left_out=tuple(embedding_other_build),
            unchecked=tuple(embedding_fallback_unchecked),
        ),
    )

    # Registered once per Home Assistant run: aiohttp routes cannot be removed,
    # so the view resolves the currently loaded entry per request instead of
    # pinning the one that happened to register it (see EnrollPersonView).
    if not hass.data[DOMAIN].get("http_registered"):
        hass.http.register_view(EnrollPersonView(hass))
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

    # Describe only the Ollama servers a chain uses and that answered: with a
    # cloud primary and an Ollama fallback, the category URLs are an unused
    # localhost default.
    ollama_probe_urls: dict[str, str] = {}
    for category, head_url in (
        ("chat", ollama_chat_url),
        ("vlm", ollama_vlm_url),
        ("summarization", ollama_sum_url),
        ("embedding", ollama_embedding_url),
    ):
        chain = fallback_chains.get(category)
        if chain and chain[0].provider_type == "ollama" and ollama_health.get(head_url):
            ollama_probe_urls[category] = head_url
        for index, url in enumerate(fallback_ollama_urls.get(category, []), 1):
            if ollama_health.get(url):
                ollama_probe_urls[f"{category}_fallback{index}"] = url
    if ollama_probe_urls:
        hass.async_create_task(_log_ollama_server_info(hass, ollama_probe_urls))

    if options.get(CONF_VIDEO_ANALYZER_MODE) != "disable":
        if hass.is_running:
            video_analyzer.start()
        else:
            defer_start_until_hass_started(hass, entry, video_analyzer.start)
    if options.get(CONF_SENTINEL_ENABLED, RECOMMENDED_SENTINEL_ENABLED):
        if hass.is_running:
            sentinel.start()
        else:
            defer_start_until_hass_started(hass, entry, sentinel.start)
    if options.get(CONF_SENTINEL_ENABLED, RECOMMENDED_SENTINEL_ENABLED) and options.get(
        CONF_SENTINEL_DISCOVERY_ENABLED,
        RECOMMENDED_SENTINEL_DISCOVERY_ENABLED,
    ):
        if hass.is_running:
            discovery_engine.start()
        else:
            defer_start_until_hass_started(hass, entry, discovery_engine.start)

    if baseline_updater is not None:
        if hass.is_running:
            baseline_updater.start()
        else:
            defer_start_until_hass_started(hass, entry, baseline_updater.start)

    # Home Assistant does not unload config entries at shutdown — its
    # ConfigEntries._async_shutdown only calls entry.async_shutdown(), which
    # cancels retry setup — so this stays the only stop path for a still-loaded
    # entry. Note it is deliberately NARROWER than async_unload_entry: the
    # video analyzer, the notifier, and the DB pool are not torn down here,
    # because the process is exiting anyway.
    #
    # Wrapped in async_on_unload because it was previously registered
    # unconditionally on every setup and never removed: each reload added
    # another closure capturing that generation's engines and client, so
    # shutdown after N reloads fired N+1 of them — stopping long-dead objects
    # and keeping every stale generation's object graph reachable until the
    # process exited.
    stop_listener_fired = False

    async def _stop_background_tasks() -> None:
        await sentinel.stop()
        await discovery_engine.stop()
        if baseline_updater is not None:
            await baseline_updater.stop()
        await hass.async_add_executor_job(openai_http_client.close)

    # @callback, so the flag is set inline in the same loop tick that
    # _OneTimeListener unsubscribes. Setting it inside the coroutine instead
    # would be correct only by accident: an async listener makes HassJob infer
    # Coroutinefunction and dispatch through create_eager_task, which happens
    # to run to the first suspension synchronously. Any await added above the
    # assignment — or the loss of eager start — would reopen the window the
    # flag exists to close.
    @callback
    def _on_hass_stop(_event: object) -> None:
        nonlocal stop_listener_fired
        stop_listener_fired = True
        hass.async_create_task(_stop_background_tasks())

    remove_stop_listener = hass.bus.async_listen_once(
        EVENT_HOMEASSISTANT_STOP, _on_hass_stop
    )

    # The same fired-flag guard the deferred-start helper uses, and for the
    # same reason: async_listen_once hands back the very remover its
    # _OneTimeListener consumes when the event fires, so a second call reaches
    # EventBus._async_remove_listener and logs "Unable to remove unknown job
    # listener" with a traceback.
    #
    # This is NOT unreachable. entry.async_on_unload callbacks run only after
    # async_unload_entry RETURNS, so the listener stays armed for that whole
    # function — a shutdown landing during an in-flight reload runs both paths.
    # Stopping twice is harmless (the engines latch, httpx.Client.close is
    # state-guarded), but the second remove is not, and pytest teardown hits
    # this ordering routinely.
    @callback
    def _cancel_stop_listener() -> None:
        if not stop_listener_fired:
            remove_stop_listener()

    entry.async_on_unload(_cancel_stop_listener)

    async_schedule_provider_recheck(hass, entry, recheck_probes)

    msg = (
        "Home Generative Agent initialized with the following models: "
        "chat=%s, vlm=%s, summarization=%s. embeddings=%s. "
        "OpenAI ok=%s, Ollama ok=%s, Gemini ok=%s, Anthropic ok=%s."
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
        anthropic_ok,
    )

    async def _handle_enroll_person(call: ServiceCall) -> None:
        name: str = call.data["name"]
        file_path: str = call.data["file_path"]

        # Captured before the first await: a reload landing during the media
        # read deletes ``entry.runtime_data`` (the service is deregistered on
        # unload, but an in-flight call is not cancelled). And refused before
        # any I/O: an entry with no gallery must not read a caller-chosen
        # file or fetch a media source on behalf of a call that cannot succeed.
        runtime_data = entry.runtime_data
        dao: PersonGalleryDAO | None = runtime_data.person_gallery
        if dao is None:
            raise HomeAssistantError(NO_DATABASE_ENROLL_MESSAGE)

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

        ok = await dao.enroll_from_image(runtime_data.face_api_url, name, img_bytes)
        if not ok:
            msg = f"No face found in image for {name}"
            raise HomeAssistantError(msg)

    _register_entry_service(
        hass,
        entry,
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

    _register_entry_service(
        hass,
        entry,
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

    _register_entry_service(
        hass,
        entry,
        SERVICE_GET_DISCOVERY_RECORDS,
        _handle_get_discovery,
        schema=GET_DISCOVERY_SCHEMA,
        supports_response=_SERVICE_RESPONSE_ONLY,
    )

    async def _handle_trigger_discovery(_call: ServiceCall) -> dict[str, Any]:
        return await _trigger_sentinel_discovery(entry)

    _register_entry_service(
        hass,
        entry,
        SERVICE_TRIGGER_SENTINEL_DISCOVERY,
        _handle_trigger_discovery,
        schema=TRIGGER_DISCOVERY_SCHEMA,
        supports_response=_SERVICE_RESPONSE_ONLY,
    )

    def _rule_entity_ids(params: dict[str, Any]) -> list[str]:
        entity_ids: list[str] = []
        for key, value in params.items():
            if key.endswith("_entity_id") and isinstance(value, str) and value:
                entity_ids.append(value)
            elif key.endswith("_entity_ids") and isinstance(value, list):
                entity_ids.extend(
                    item for item in value if isinstance(item, str) and item
                )
        return sorted(set(entity_ids))

    def _candidate_entity_ids(candidate: dict[str, Any]) -> list[str]:
        explain = explain_normalize_candidate(candidate)
        normalized = explain.normalized
        if normalized is None:
            return []
        return _rule_entity_ids(normalized.params)

    def _covered_rule_for_candidate(
        candidate: dict[str, Any],
    ) -> tuple[str, list[str]] | None:
        rule_registry = entry.runtime_data.rule_registry
        if rule_registry is None:
            return None
        candidate_key = candidate_semantic_key(candidate)
        if not candidate_key:
            return None
        candidate_entities = _candidate_entity_ids(candidate)
        for rule in rule_registry.list_rules():
            rkey = rule_semantic_key(rule)
            if not rkey or not rule_key_covers_candidate_key(rkey, candidate_key):
                continue
            rule_id = str(rule.get("rule_id", ""))
            if rule_id:
                overlapping_entities = sorted(
                    set(candidate_entities).intersection(
                        _rule_entity_ids(rule.get("params") or {})
                    )
                )
                return rule_id, overlapping_entities
        return None

    def _covered_specific_rule_for_any_camera_normalized(
        template_id: str,
        params: dict[str, Any],
    ) -> tuple[str, list[str]] | None:
        rule_registry = entry.runtime_data.rule_registry
        if rule_registry is None:
            return None
        if params.get("camera_selector") != "any":
            return None
        if template_id not in {
            "unknown_person_camera_no_home",
            "unknown_person_camera_when_home",
        }:
            return None
        for rule in rule_registry.list_rules():
            if str(rule.get("template_id", "")) != template_id:
                continue
            rule_params = rule.get("params") or {}
            if not isinstance(rule_params, dict):
                continue
            camera_entity_id = str(rule_params.get("camera_entity_id", ""))
            if not camera_entity_id:
                continue
            rule_id = str(rule.get("rule_id", ""))
            if rule_id:
                return rule_id, [camera_entity_id]
        return None

    async def _handle_promote_discovery(
        call: ServiceCall,
    ) -> dict[str, Any]:
        candidate_id = str(call.data.get("candidate_id"))
        notes = str(call.data.get("notes", "") or "")
        return await _promote_discovery_candidate(
            hass,
            entry,
            candidate_id=candidate_id,
            notes=notes,
        )

    _register_entry_service(
        hass,
        entry,
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

    _register_entry_service(
        hass,
        entry,
        SERVICE_GET_PROPOSAL_DRAFTS,
        _handle_get_proposals,
        schema=GET_PROPOSAL_SCHEMA,
        supports_response=_SERVICE_RESPONSE_ONLY,
    )

    async def _handle_preview_proposal(
        call: ServiceCall,
    ) -> dict[str, Any]:
        candidate_id = str(call.data.get("candidate_id"))
        return await _preview_rule_proposal(
            hass,
            entry,
            candidate_id=candidate_id,
        )

    _register_entry_service(
        hass,
        entry,
        SERVICE_PREVIEW_RULE_PROPOSAL,
        _handle_preview_proposal,
        schema=PREVIEW_PROPOSAL_SCHEMA,
        supports_response=_SERVICE_RESPONSE_ONLY,
    )

    async def _handle_get_dynamic_rules(call: ServiceCall) -> dict[str, Any]:
        limit = int(call.data.get("limit", 200))
        limit = max(1, min(limit, 500))
        rule_registry = entry.runtime_data.rule_registry
        if rule_registry is None:
            return {"records": []}
        records = rule_registry.list_rules(include_disabled=True)
        return {"records": records[:limit]}

    _register_entry_service(
        hass,
        entry,
        SERVICE_GET_DYNAMIC_RULES,
        _handle_get_dynamic_rules,
        schema=GET_DYNAMIC_RULES_SCHEMA,
        supports_response=_SERVICE_RESPONSE_ONLY,
    )

    async def _handle_approve_proposal(
        call: ServiceCall,
    ) -> dict[str, Any]:
        candidate_id = str(call.data.get("candidate_id"))
        notes = str(call.data.get("notes", "") or "")
        return await _approve_rule_proposal(
            entry,
            hass=hass,
            candidate_id=candidate_id,
            notes=notes,
        )

    _register_entry_service(
        hass,
        entry,
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

    _register_entry_service(
        hass,
        entry,
        SERVICE_REJECT_RULE_PROPOSAL,
        _handle_reject_proposal,
        schema=REVIEW_PROPOSAL_SCHEMA,
        supports_response=_SERVICE_RESPONSE_ONLY,
    )

    async def _handle_deactivate_dynamic_rule(call: ServiceCall) -> dict[str, Any]:
        rule_id = str(call.data.get("rule_id"))
        rule_registry = entry.runtime_data.rule_registry
        if rule_registry is None:
            return {"status": "unavailable", "rule_id": rule_id}
        ok = await rule_registry.async_set_rule_enabled(rule_id, enabled=False)
        return {"status": "ok" if ok else "not_found", "rule_id": rule_id}

    _register_entry_service(
        hass,
        entry,
        SERVICE_DEACTIVATE_DYNAMIC_RULE,
        _handle_deactivate_dynamic_rule,
        schema=TOGGLE_DYNAMIC_RULE_SCHEMA,
        supports_response=_SERVICE_RESPONSE_ONLY,
    )

    async def _handle_reactivate_dynamic_rule(call: ServiceCall) -> dict[str, Any]:
        rule_id = str(call.data.get("rule_id"))
        rule_registry = entry.runtime_data.rule_registry
        if rule_registry is None:
            return {"status": "unavailable", "rule_id": rule_id}
        ok = await rule_registry.async_set_rule_enabled(rule_id, enabled=True)
        return {"status": "ok" if ok else "not_found", "rule_id": rule_id}

    _register_entry_service(
        hass,
        entry,
        SERVICE_REACTIVATE_DYNAMIC_RULE,
        _handle_reactivate_dynamic_rule,
        schema=TOGGLE_DYNAMIC_RULE_SCHEMA,
        supports_response=_SERVICE_RESPONSE_ONLY,
    )

    async def _handle_patch_dynamic_rule(call: ServiceCall) -> dict[str, Any]:
        rule_id = str(call.data.get("rule_id"))
        params_patch = dict(call.data.get("params", {}))
        rule_registry = entry.runtime_data.rule_registry
        if rule_registry is None:
            return {"status": "unavailable", "rule_id": rule_id}
        ok = await rule_registry.async_patch_rule_params(rule_id, params_patch)
        return {"status": "ok" if ok else "not_found", "rule_id": rule_id}

    _register_entry_service(
        hass,
        entry,
        SERVICE_PATCH_DYNAMIC_RULE,
        _handle_patch_dynamic_rule,
        schema=PATCH_DYNAMIC_RULE_SCHEMA,
        supports_response=_SERVICE_RESPONSE_ONLY,
    )

    async def _handle_sentinel_set_autonomy_level(call: ServiceCall) -> dict[str, Any]:
        """Set the runtime autonomy level (admin-only)."""
        level = int(call.data["level"])
        pin: str | None = call.data.get("pin")

        await _async_require_admin(hass, call, SERVICE_SENTINEL_SET_AUTONOMY_LEVEL)

        sentinel = entry.runtime_data.sentinel
        if sentinel is None:
            return {"status": "unavailable", "entry_id": entry.entry_id}

        sentinel.set_autonomy_level(entry.entry_id, level, pin=pin)
        # Show the new level on the health sensor now, not after the next run:
        # a level 0 that silences Sentinel must be visible at once (#783).
        async_dispatcher_send(hass, SIGNAL_SENTINEL_RUN_COMPLETE)
        return {"status": "ok", "entry_id": entry.entry_id, "level": level}

    _register_entry_service(
        hass,
        entry,
        SERVICE_SENTINEL_SET_AUTONOMY_LEVEL,
        _handle_sentinel_set_autonomy_level,
        schema=SET_AUTONOMY_LEVEL_SCHEMA,
        supports_response=_SERVICE_RESPONSE_ONLY,
    )

    async def _handle_sentinel_get_baselines(call: ServiceCall) -> dict[str, Any]:  # noqa: ARG001
        """Return full baseline statistics for entities above min-sample threshold."""
        baseline_updater = entry.runtime_data.baseline_updater
        if baseline_updater is None:
            return {"status": "unavailable", "baselines": {}}
        baselines = await baseline_updater.async_fetch_full_baselines()
        if baselines is None:
            return {
                "status": "error",
                "message": "DB query failed; see logs for details.",
            }
        return {"status": "ok", "baselines": baselines}

    _register_entry_service(
        hass,
        entry,
        SERVICE_SENTINEL_GET_BASELINES,
        _handle_sentinel_get_baselines,
        schema=SENTINEL_GET_BASELINES_SCHEMA,
        supports_response=_SERVICE_RESPONSE_ONLY,
    )

    async def _handle_sentinel_reset_baseline(call: ServiceCall) -> dict[str, Any]:
        """Delete baseline data for one entity or all entities."""
        entity_id: str | None = call.data.get("entity_id")
        baseline_updater = entry.runtime_data.baseline_updater
        if baseline_updater is None:
            return {"status": "unavailable"}
        deleted = await baseline_updater.async_reset_baseline(entity_id)
        if deleted < 0:
            return {
                "status": "error",
                "message": "DB reset failed; see logs for details.",
                "entity_id": entity_id,
            }
        scope = entity_id or "all entities"
        await hass.services.async_call(
            "persistent_notification",
            "create",
            {
                "title": "Sentinel Baseline Reset",
                "message": f"Baseline reset for {scope}. {deleted} record(s) removed.",
                "notification_id": f"sentinel_baseline_reset_{entry.entry_id}",
            },
            blocking=False,
        )
        return {"status": "ok", "entity_id": entity_id, "deleted": deleted}

    _register_entry_service(
        hass,
        entry,
        SERVICE_SENTINEL_RESET_BASELINE,
        _handle_sentinel_reset_baseline,
        schema=SENTINEL_RESET_BASELINE_SCHEMA,
        supports_response=_SERVICE_RESPONSE_ONLY,
    )

    async def _handle_sentinel_reset_auth_inventory(
        call: ServiceCall,
    ) -> dict[str, Any]:
        """
        Clear the auth inventory; the next Sentinel run re-bootstraps it.

        Admin-only, like ``sentinel_set_autonomy_level``: a reset makes every
        current user and token "known" without alerting, so a non-admin
        caller could blind the new-admin / new-token detector before creating
        exactly what it exists to catch. The reset is logged with the caller's
        user id so it leaves a trace in the Home Assistant log as well.
        """
        user = await _async_require_admin(
            hass, call, SERVICE_SENTINEL_RESET_AUTH_INVENTORY
        )
        inventory = entry.runtime_data.auth_inventory
        if inventory is None:
            return {"status": "unavailable"}
        before = inventory.summary()
        LOGGER.warning(
            "Sentinel auth inventory reset by user %s (%s): %d user(s), %d "
            "token(s) forgotten; the next run bootstraps without alerting.",
            user.name,
            user.id,
            before["user_count"],
            before["token_count"],
        )
        await inventory.async_reset()
        await hass.services.async_call(
            "persistent_notification",
            "create",
            {
                "title": "Sentinel auth inventory reset",
                "message": (
                    "The Home Assistant user and token inventory was cleared "
                    f"({before['user_count']} user(s), {before['token_count']} "
                    "token(s)). The next Sentinel run records the current state "
                    "as known without alerting."
                ),
                "notification_id": f"sentinel_auth_inventory_reset_{entry.entry_id}",
            },
            blocking=False,
        )
        return {"status": "ok", "cleared": before}

    _register_entry_service(
        hass,
        entry,
        SERVICE_SENTINEL_RESET_AUTH_INVENTORY,
        _handle_sentinel_reset_auth_inventory,
        schema=vol.Schema({}),
        supports_response=_SERVICE_RESPONSE_ONLY,
    )

    async def _handle_run_network_audit(call: ServiceCall) -> dict[str, Any]:
        """
        Run the network / HA-security rules now and return the report.

        Read-only and side-effect free: nothing is notified, stored, or
        cooled down, and the auth inventory is not committed. The same
        report backs the ``audit_home_security`` agent tool.
        """
        _ = call
        sentinel = entry.runtime_data.sentinel
        if sentinel is None:
            # Same shape as every other outcome so automations can rely on it.
            return dict(
                empty_network_audit_report(
                    "unavailable",
                    dt_util.utcnow(),
                    "Sentinel is not enabled for this entry.",
                )
            )
        return dict(await sentinel.async_audit_network())

    _register_entry_service(
        hass,
        entry,
        SERVICE_RUN_NETWORK_AUDIT,
        _handle_run_network_audit,
        schema=vol.Schema({}),
        supports_response=_SERVICE_RESPONSE_ONLY,
    )

    _register_network_inventory_services(hass, entry)

    # Reload whenever a Sentinel subentry is added, updated, or removed so that
    # tasks started during setup are stopped or started to match the current
    # subentry configuration.
    #
    # We use SIGNAL_CONFIG_ENTRY_CHANGED (dispatcher) rather than
    # entry.add_update_listener because OptionsFlowWithReload forbids update
    # listeners and raises ValueError when the options flow completes.
    #
    # SIGNAL_CONFIG_ENTRY_CHANGED fires for all entry mutations AND for every
    # entry state transition (setup-in-progress → loaded, etc.) via
    # ConfigEntry._async_set_state.  Reacting to state transitions would create
    # an infinite reload loop.  Guard by snapshotting {subentry_id: dict(data)}
    # at setup time; state transitions never alter subentry data.
    _sentinel_snapshot: dict[str, dict[str, object]] = {
        s.subentry_id: dict(s.data)
        for s in entry.subentries.values()
        if s.subentry_type == SUBENTRY_TYPE_SENTINEL
    }

    @callback
    def _on_entry_changed(
        change_type: ConfigEntryChange, changed_entry: HGAConfigEntry
    ) -> None:
        if changed_entry.entry_id != entry.entry_id:
            return
        if change_type != ConfigEntryChange.UPDATED:
            return
        current_snapshot = {
            s.subentry_id: dict(s.data)
            for s in changed_entry.subentries.values()
            if s.subentry_type == SUBENTRY_TYPE_SENTINEL
        }
        if current_snapshot == _sentinel_snapshot:
            return
        # Sentinel subentry was added, removed, or its data changed.
        # When removed, stop tasks immediately before the reload unloads the entry.
        if not current_snapshot:
            rd = changed_entry.runtime_data
            if rd.sentinel is not None:
                hass.async_create_task(rd.sentinel.stop())

            async def _forget_network_audit() -> None:
                # Deleting Sentinel deletes what the network audit persisted
                # (the auth inventory and the pseudonymization salt), but
                # only once the engine has stopped: a mid-cycle commit racing
                # the deletion would recreate the inventory file.
                if rd.sentinel is not None:
                    await rd.sentinel.stop()
                if rd.auth_inventory is not None:
                    await rd.auth_inventory.async_reset()
                if rd.network_inventory is not None:
                    await rd.network_inventory.async_reset()
                await async_remove_pseudonymizer_salt(hass)

            cleanup = hass.async_create_task(_forget_network_audit())
            if rd.baseline_updater is not None:
                hass.async_create_task(rd.baseline_updater.stop())
            if rd.discovery_engine is not None:
                hass.async_create_task(rd.discovery_engine.stop())
            if rd.notifier is not None:
                rd.notifier.stop()
            # Reload only after the inventory and salt are gone: a replacement
            # setup racing the cleanup could load them just before deletion.
            cleanup.add_done_callback(
                lambda _task: hass.config_entries.async_schedule_reload(entry.entry_id)
            )
            return
        hass.config_entries.async_schedule_reload(entry.entry_id)

    entry.async_on_unload(
        async_dispatcher_connect(hass, SIGNAL_CONFIG_ENTRY_CHANGED, _on_entry_changed)
    )

    # Raised last so it is only ever true of a *loaded* entry: every exit
    # above is a failed setup, which is its own visible state. Gated on a
    # Model Provider because a freshly added entry has neither a provider nor
    # a database and is mid-onboarding (README steps 5-6), not misconfigured;
    # the feature subentries cannot serve as that gate — setup creates them.
    async_sync_database_issue(
        hass,
        entry.entry_id,
        database_missing=db_uri is None and bool(providers),
    )

    # Same placement. Checked off the setup path: a model listing and a size
    # request per server that just answered its health check, plus one read of
    # the store's vector size, and it only ever informs.
    entry.async_create_background_task(
        hass,
        async_sync_ollama_embedding_model_issue(
            hass,
            entry.entry_id,
            urls=_ollama_embedding_urls_to_check(
                primary_url=ollama_embedding_url,
                primary_selected=ollama_embeddings_selected,
                fallback_urls=fallback_ollama_urls.get("embedding", []),
                health=ollama_health,
                has_database=db_uri is not None,
            ),
            # Set whenever Ollama is the configured primary, even one that is
            # down at boot: the check must then wait for it to be measured,
            # not judge the fallbacks as if no Ollama primary existed.
            primary_url=(
                ollama_embedding_url if embedding_provider == "ollama" else None
            ),
            model=ollama_embedding_model,
            configured_dims=embedding_dims,
            pool=pool,
        ),
        name="hga_ollama_embedding_model_check",
    )
    # Same placement, same reason, for the other embedding providers: one
    # embedding from an OpenAI-compatible server, the others' size is fixed
    # (#760).
    entry.async_create_background_task(
        hass,
        async_sync_embedding_size_issue(
            hass,
            entry.entry_id,
            provider=embedding_size_provider,
            pool=pool,
        ),
        name="hga_embedding_size_check",
    )
    async_sync_other_model_fallback_issue(
        hass,
        entry.entry_id,
        primary=embedding_chain[0].name if embedding_chain else "",
        left_out=embedding_left_out_chosen,
    )
    # With kept fallbacks still to be judged, the record check syncs this
    # issue itself once it knows (#860): syncing it here too would delete a
    # standing, perhaps dismissed, issue on no evidence, only for the check to
    # raise it again fresh, or, with the primary down and no check to come,
    # for good while the fallback it names is still held, unjudged (#857).
    if embedding_other_build or not embedding_fallback_unchecked:
        async_sync_other_build_fallback_issue(
            hass,
            entry.entry_id,
            primary=embedding_chain[0].name if embedding_chain else "",
            model=ollama_embedding_model,
            left_out=embedding_other_build,
        )
    # Same placement, same reason: one read of the store's model record, and
    # a write when there is none (#855). It also resumes a re-embed the
    # repair started.
    entry.async_create_background_task(
        hass,
        async_sync_embedding_record(hass, entry),
        name="hga_embedding_record_check",
    )
    # Same placement: captions stored before they named their camera get it,
    # in batches, so each camera's reads find them again (#706). Kept to be
    # cancelled at unload before the pool it writes on is closed.
    if pool is not None:
        entry.runtime_data.caption_backfill_task = entry.async_create_background_task(
            hass,
            async_backfill_caption_cameras(pool),
            name="hga_caption_camera_backfill",
        )

    # Same placement, same reason: four Sentinel rule ids (five evaluators) key
    # on a positive "Unknown Person" label that only the face pipeline writes,
    # so without it they are permanently inert with nothing logged. Raised here
    # rather than where `face_recognition` is computed so it can only ever be
    # true of a loaded entry. It receives the raw option, the gallery and the
    # analyzer mode separately so the notice can name the state the install is
    # actually in; the analyzer counts because `recognize_faces` only ever runs
    # from there.
    async_check_unknown_person_rules(
        hass,
        entry.entry_id,
        sentinel_enabled=bool(
            options.get(CONF_SENTINEL_ENABLED, RECOMMENDED_SENTINEL_ENABLED)
        ),
        # The three facts separately, not the collapsed `face_recognition`: the
        # notice has to tell a user who switched face recognition ON that the
        # database is missing, rather than tell them to switch it on.
        face_recognition_configured=bool(
            options.get(CONF_FACE_RECOGNITION, RECOMMENDED_FACE_RECOGNITION)
        ),
        person_gallery_available=person_gallery is not None,
        # Mirrors the gate that actually starts the analyzer above, literal
        # and defaultless included: the guard must agree with what runs, not
        # with RECOMMENDED_VIDEO_ANALYZER_MODE, or it would claim the rules
        # are inert on an install whose analyzer is running (unset reads as
        # enabled there).
        video_analysis_enabled=options.get(CONF_VIDEO_ANALYZER_MODE) != "disable",
    )

    return True


# How long one cancellation gets to land, and how many are sent, before the
# unload stops waiting on a task (see _cancel_and_wait).
_CANCEL_WAIT_S = 2.0
_CANCEL_ATTEMPTS = 5


async def _cancel_and_wait(task: object, what: str) -> None:
    """
    Cancel *task* and wait for it to finish; a no-op for None or a done task.

    *what* names the task in the give-up warning: a task we do not own (the
    store worker) carries only an auto-generated name.

    ``asyncio.wait`` rather than ``await task``: the latter would raise the
    task's CancelledError here, and suppressing it would also swallow a
    cancellation of the unload itself (Home Assistant shutdown, a reload
    cancelled), which must keep propagating.

    The wait is bounded and the cancel repeated, because one cancel is not
    guaranteed to end a task we do not own: langgraph's batch worker catches
    any ordinary exception out of the in-flight batch (handing it to callers
    still waiting, if any) and goes back to its queue, so a cancel that the
    database driver turns into such an exception leaves the worker alive.
    Waiting on it without a timeout held the entry in UNLOAD_IN_PROGRESS for
    good, taking every entity with it. A repeated cancel ends a worker that
    is back on its queue; one that lands in another batch can be absorbed
    again, hence several attempts. A task that outlives them all is logged
    and left behind rather than allowed to block the unload.
    """
    if not isinstance(task, asyncio.Task) or task.done():
        return
    for _ in range(_CANCEL_ATTEMPTS):
        task.cancel()
        done, _pending = await asyncio.wait([task], timeout=_CANCEL_WAIT_S)
        if done:
            return
    LOGGER.warning(
        "The %s (%s) did not stop after %d cancellations; continuing the "
        "unload without it. It may keep a database connection open.",
        what,
        task.get_name(),
        _CANCEL_ATTEMPTS,
    )


async def async_unload_entry(hass: HomeAssistant, entry: HGAConfigEntry) -> bool:
    """Unload the config entry."""
    # Platforms come down FIRST, and a refusal aborts before anything is torn
    # down. Every teardown below is irreversible — the engines set a one-way
    # _stopped latch, and the pool cannot be reopened — while a False return
    # puts the entry in FAILED_UNLOAD, which Home Assistant treats as
    # non-recoverable and which leaves runtime_data and any entity that refused
    # to unload in place. Doing the teardown first would therefore strand live
    # entities on dead resources with no way back short of a restart.
    #
    # Unloading first also keeps the conversation entity from outliving what it
    # borrows: runtime_data.chat_model, which may be built on the OpenAI client
    # closed by the on-unload hook, and the store/checkpointer backed by the
    # pool closed below. (The STT entity is not affected — stt.py uses Home
    # Assistant's shared async client, not this one.)
    if not await hass.config_entries.async_unload_platforms(entry, PLATFORMS):
        return False

    # Every teardown step below is contained: one raising step must not skip
    # the remaining stops, and it must not escape this function. Home Assistant
    # catches an exception out of async_unload_entry, sets FAILED_UNLOAD —
    # declared non-recoverable — and returns WITHOUT running the on-unload
    # callbacks, so an uncontained raise would leave every deferred-start
    # listener armed and every engine below the raising line un-stopped and
    # un-latched: exactly the orphan-start this teardown exists to prevent.
    async def _teardown(step: str, run: Callable[[], Awaitable[object] | None]) -> None:
        try:
            result = run()
            # isawaitable, not a None-check: a future sync step that returns
            # a value must not be awaited (TypeError misread as step failure).
            if inspect.isawaitable(result):
                await result
        except Exception:
            LOGGER.exception("Unload step %s failed; continuing teardown", step)

    rd = entry.runtime_data
    await _teardown("video_analyzer.stop", rd.video_analyzer.stop)
    if rd.sentinel is not None:
        await _teardown("sentinel.stop", rd.sentinel.stop)
    if rd.discovery_engine is not None:
        await _teardown("discovery_engine.stop", rd.discovery_engine.stop)
    if rd.baseline_updater is not None:
        await _teardown("baseline_updater.stop", rd.baseline_updater.stop)
    if rd.notifier is not None:
        await _teardown("notifier.stop", rd.notifier.stop)
    if (audit_store := rd.audit_store) is not None:
        # Saves are delayed; write now and close the store, so a reload's
        # fresh store loads the newest records and no late write from this
        # one can overwrite them. The lambda keeps the attribute lookup inside
        # the contained step.
        await _teardown(
            "audit_store.async_close",
            lambda: audit_store.async_close(),  # noqa: PLW0108
        )

    async def _cancel_tool_index() -> None:
        # The index write runs on the pool closed next; a reload that raced
        # it used to end in PoolClosed, logged as a failed index. Two tasks
        # are involved: the runner that awaits the store, and the store's
        # own batch worker (langgraph's AsyncBatchedBaseStore queues each
        # put and runs the embed + INSERT batch in a task of its own), which
        # cancelling the runner alone would leave writing.
        await _cancel_and_wait(rd.tool_index_task, "tool index write")
        rd.tool_index_task = None
        await _cancel_and_wait(getattr(rd.store, "_task", None), "store batch worker")

    await _teardown("tool_index.cancel", _cancel_tool_index)

    async def _cancel_reembed() -> None:
        # Writes on the pool closed next; it resumes at the next setup (#855).
        await _cancel_and_wait(rd.reembed_task, "memory store re-embed")
        rd.reembed_task = None
        # Same for the caption camera backfill (#706).
        await _cancel_and_wait(rd.caption_backfill_task, "caption camera backfill")
        rd.caption_backfill_task = None

    await _teardown("reembed.cancel", _cancel_reembed)
    if rd.pool is not None:
        await _teardown("pool.close", rd.pool.close)
    # The OpenAI http client is NOT closed here — it is closed by the on-unload
    # callback registered where it is built, which also covers failed setups.
    # Home Assistant runs those callbacks only when this returns True, so the
    # abort above correctly leaves the client open for the still-loaded entry.
    #
    # An unloaded entry is not "running without a database": drop the issue
    # (a disable would otherwise leave it standing); the next successful
    # setup re-raises it if the database is still missing.
    async_clear_database_issue(hass, entry.entry_id)
    async_clear_ollama_embedding_model_issue(hass, entry.entry_id)
    async_clear_embedding_size_issue(hass, entry.entry_id)
    # Same for the left-out embedding fallback: setup re-syncs it, so a reload
    # keeps the user's dismissal; a disable drops it.
    if entry.disabled_by is not None:
        async_clear_other_model_fallback_issue(hass, entry.entry_id)
        async_clear_other_build_fallback_issue(hass, entry.entry_id)
        async_clear_embedding_model_changed_issue(hass, entry.entry_id)
    # A disabled entry runs no rules at all, so singling four out is wrong the
    # same way. A reload (every Sentinel config save) keeps it: deleting it
    # would also delete the user's dismissal, so a dismissed notice came back
    # on every save (#810). Setup re-checks it, and clears it once it no
    # longer applies.
    if entry.disabled_by is not None:
        async_clear_unknown_person_issue(hass, entry.entry_id)
    return True


async def async_remove_entry(hass: HomeAssistant, entry: HGAConfigEntry) -> None:
    """
    Clear every per-entry repair issue when the entry is deleted for good.

    Every one is keyed by entry_id, so a remove-and-re-add (new entry_id)
    would otherwise leave the old ones standing until a restart.
    """
    async_clear_database_issue(hass, entry.entry_id)
    async_clear_ollama_embedding_model_issue(hass, entry.entry_id)
    async_clear_embedding_size_issue(hass, entry.entry_id)
    async_clear_other_model_fallback_issue(hass, entry.entry_id)
    async_clear_other_build_fallback_issue(hass, entry.entry_id)
    async_clear_embedding_model_changed_issue(hass, entry.entry_id)
    async_clear_pin_pipeline_issue(hass, entry.entry_id)
    async_clear_unknown_person_issue(hass, entry.entry_id)


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
                CONF_SENTINEL_PENDING_PROMPT_TTL_MINUTES: merged_options.get(
                    CONF_SENTINEL_PENDING_PROMPT_TTL_MINUTES,
                    RECOMMENDED_SENTINEL_PENDING_PROMPT_TTL_MINUTES,
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
            CONF_SENTINEL_PENDING_PROMPT_TTL_MINUTES,
            CONF_SENTINEL_DISCOVERY_ENABLED,
            CONF_SENTINEL_DISCOVERY_INTERVAL_SECONDS,
            CONF_SENTINEL_DISCOVERY_MAX_RECORDS,
            CONF_EXPLAIN_ENABLED,
        ):
            new_options.pop(key, None)

        current_version = 5

        if current_version < 6:  # noqa: PLR2004
            # v5 -> v6: normalize CONF_LLM_HASS_API from a string to a list.
            # "none" sentinel and absent key both become [] (no APIs).
            api_val = new_options.get(CONF_LLM_HASS_API)
            if isinstance(api_val, str):
                new_options[CONF_LLM_HASS_API] = [] if api_val == "none" else [api_val]
            elif api_val is None:
                new_options[CONF_LLM_HASS_API] = []
            current_version = 6

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

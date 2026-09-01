"""Constants for Home Generative Agent."""

from typing import Annotated, Any, Literal, get_args

from annotated_types import Ge, Le

DOMAIN = "home_generative_agent"
HGA_CARD_STATIC_PATH = "/hga-card"
HGA_CARD_STATIC_PATH_LEGACY = "/hga-enroll-card"

CONFIG_ENTRY_VERSION = 6

SUBENTRY_TYPE_DATABASE = "database"
SUBENTRY_TYPE_MODEL_PROVIDER = "model_provider"
SUBENTRY_TYPE_FEATURE = "feature"
SUBENTRY_TYPE_STT_PROVIDER = "stt_provider"
SUBENTRY_TYPE_SENTINEL = "sentinel"

HTTP_STATUS_UNAUTHORIZED = 401
HTTP_STATUS_FORBIDDEN = 403
HTTP_STATUS_BAD_REQUEST = 400
HTTP_STATUS_WEBPAGE_NOT_FOUND = 404
HTTP_STATUS_OK = 200
HTTP_STATUS_REQUEST_TOO_LARGE = 413
HTTP_STATUS_SERVICE_UNAVAILABLE = 503

# ---- Critical action guard ----
CONF_CRITICAL_ACTION_PIN_ENABLED = "critical_action_pin_enabled"
CONF_CRITICAL_ACTION_PIN = "critical_action_pin"
CONF_CRITICAL_ACTION_PIN_HASH = "critical_action_pin_hash"
CONF_CRITICAL_ACTION_PIN_SALT = "critical_action_pin_salt"
CONF_CRITICAL_ACTIONS = "critical_actions"
# Matched by `matches_critical_rule` (agent/helpers.py) for both direct tool
# calls and the contents of LLM-authored automations. An `entity_match` rule
# also fires when the call's real targets cannot be resolved at check time
# (an area, device, label, floor, group, registry ID, or template) — see
# `agent/automation_pin.py`.
RECOMMENDED_CRITICAL_ACTIONS: list[dict[str, str]] = [
    {"domain": "lock", "service": "unlock"},
    {"domain": "lock", "service": "open"},
    # `toggle` on a locked lock unlocks it.
    {"domain": "lock", "service": "toggle"},
    # Covers: guard only doors/gates/garages, not windows/shades.
    # `toggle` and `set_cover_position` open a closed door just as `open_cover`
    # does, so every opening service needs the same guard.
    {"domain": "cover", "service": "open_cover", "entity_match": "door"},
    {"domain": "cover", "service": "open_cover", "entity_match": "gate"},
    {"domain": "cover", "service": "open_cover", "entity_match": "garage"},
    {"domain": "cover", "service": "open", "entity_match": "door"},
    {"domain": "cover", "service": "open", "entity_match": "gate"},
    {"domain": "cover", "service": "open", "entity_match": "garage"},
    {"domain": "cover", "service": "toggle", "entity_match": "door"},
    {"domain": "cover", "service": "toggle", "entity_match": "gate"},
    {"domain": "cover", "service": "toggle", "entity_match": "garage"},
    {"domain": "cover", "service": "set_cover_position", "entity_match": "door"},
    {"domain": "cover", "service": "set_cover_position", "entity_match": "gate"},
    {"domain": "cover", "service": "set_cover_position", "entity_match": "garage"},
    {"domain": "garage_door", "service": "open"},
    {"domain": "garage_door", "service": "toggle"},
]
CRITICAL_PIN_MIN_LEN = 4
CRITICAL_PIN_MAX_LEN = 10

# ---- PostgreSQL (vector store + checkpointer) ----

# -- version 1
CONF_DB_URI = "db_uri"

# -- version 2
CONF_DB_NAME = "db_name"
CONF_DB_PARAMS = "db_params"
RECOMMENDED_DB_USERNAME = "ha_user"
RECOMMENDED_DB_PASSWORD = "ha_password"  # noqa: S105
RECOMMENDED_DB_HOST = "localhost"
RECOMMENDED_DB_PORT = 5432
RECOMMENDED_DB_NAME = "ha_db"
RECOMMENDED_DB_PARAMS = [{"key": "sslmode", "value": "disable"}]

CONF_DB_BOOTSTRAPPED = "db_bootstrapped"
CONF_VECTORS_BOOTSTRAPPED = "vectors_bootstrapped"

# ---- Notify service (for mobile push notifications) ----
CONF_NOTIFY_SERVICE = "notify_service"

# ---- LangChain logging ----
# See https://python.langchain.com/docs/how_to/debugging/
LANGCHAIN_LOGGING_LEVEL: Literal["disable", "verbose", "debug"] = "disable"


# ---- Global Ollama Options ----
RECOMMENDED_OLLAMA_CONTEXT_SIZE = 32000

# Ollama keepalive limits (seconds)
KEEPALIVE_MIN_SECONDS: int = 0  # 0 = unload immediately
KEEPALIVE_MAX_SECONDS: int = 15 * 60  # 900 = 15 minutes
KEEPALIVE_SENTINEL: int = -1  # never unload

KeepAliveSeconds = (
    Annotated[int, Ge(KEEPALIVE_MIN_SECONDS), Le(KEEPALIVE_MAX_SECONDS)] | Literal[-1]
)

CONF_OLLAMA_URL = "ollama_url"
RECOMMENDED_OLLAMA_URL = "http://localhost:11434"

CONF_OLLAMA_CHAT_URL = "ollama_chat_url"
RECOMMENDED_OLLAMA_CHAT_URL = RECOMMENDED_OLLAMA_URL
CONF_OLLAMA_VLM_URL = "ollama_vlm_url"
RECOMMENDED_OLLAMA_VLM_URL = RECOMMENDED_OLLAMA_URL
CONF_OLLAMA_SUMMARIZATION_URL = "ollama_summarization_url"
RECOMMENDED_OLLAMA_SUMMARIZATION_URL = RECOMMENDED_OLLAMA_URL
CONF_OLLAMA_EMBEDDING_URL = "ollama_embedding_url"
RECOMMENDED_OLLAMA_EMBEDDING_URL = RECOMMENDED_OLLAMA_URL
OLLAMA_CATEGORY_URL_KEYS = {
    "chat": CONF_OLLAMA_CHAT_URL,
    "vlm": CONF_OLLAMA_VLM_URL,
    "summarization": CONF_OLLAMA_SUMMARIZATION_URL,
    "embedding": CONF_OLLAMA_EMBEDDING_URL,
}

CONF_OLLAMA_REASONING = "ollama_reasoning"
RECOMMENDED_OLLAMA_REASONING: bool = False
OLLAMA_GPT_EFFORT = "low"
OLLAMA_OSS_TAG = "gpt-oss"
OLLAMA_BOOL_HINT_TAGS = {
    "deepseek-r1",
    "qwen3",
    "deepseek-v3.1",
    "magistral",
}

# ---- Global options ----
CONF_RECOMMENDED = "recommended"
CONF_PROMPT = "prompt"
CONF_VLM_RESPONSE_LANGUAGE = "vlm_response_language"
RECOMMENDED_VLM_RESPONSE_LANGUAGE = ""
CONF_VLM_PROMPT_EXTRA = "vlm_prompt_extra"
RECOMMENDED_VLM_PROMPT_EXTRA = ""
CONF_SCHEMA_FIRST_YAML = "schema_first_yaml"
CONF_DISABLED_FEATURES = "disabled_features"

# ---- STT hallucination filter (phantom transcriptions of silence/noise) ----
CONF_STT_HALLUCINATION_PATTERNS = "stt_hallucination_patterns"
CONF_STT_HALLUCINATION_EXACT_PATTERNS = "stt_hallucination_exact_patterns"
DEFAULT_STT_HALLUCINATION_PATTERNS: list[str] = []
DEFAULT_STT_HALLUCINATION_EXACT_PATTERNS: list[str] = []

# ---- Audit store ----
CONF_AUDIT_HOT_MAX_RECORDS = "audit_hot_max_records"
CONF_AUDIT_ARCHIVAL_BACKLOG_MAX = "audit_archival_backlog_max"
CONF_AUDIT_RETENTION_DAYS = "audit_retention_days"
CONF_AUDIT_HIGH_RETENTION_DAYS = "audit_high_retention_days"

# ---- Proactive sentinel ----
CONF_SENTINEL_ENABLED = "sentinel_enabled"
CONF_SENTINEL_INTERVAL_SECONDS = "sentinel_interval_seconds"
CONF_SENTINEL_COOLDOWN_MINUTES = "sentinel_cooldown_minutes"
CONF_SENTINEL_ENTITY_COOLDOWN_MINUTES = "sentinel_entity_cooldown_minutes"
CONF_SENTINEL_PENDING_PROMPT_TTL_MINUTES = "sentinel_pending_prompt_ttl_minutes"
CONF_EXPLAIN_ENABLED = "explain_enabled"
CONF_SENTINEL_DISCOVERY_ENABLED = "sentinel_discovery_enabled"
CONF_SENTINEL_DISCOVERY_INTERVAL_SECONDS = "sentinel_discovery_interval_seconds"
CONF_SENTINEL_DISCOVERY_MAX_RECORDS = "sentinel_discovery_max_records"
# Optional language override for the LLM-authored Sentinel finding
# explanation (explain/llm_explain.LLMExplainer -- the text shown in mobile
# and persistent notifications). Empty string means "let the model use its
# default (English)" -- mirrors CONF_VLM_RESPONSE_LANGUAGE's convention.
# Deliberately scoped to the explainer only: discovery candidate title/
# summary and Sentinel Triage decision/reason_code/summary are parsed by
# code downstream (normalization, dedup, evidence-mismatch matching) and
# must not be translated. See #523/#524 discussion.
CONF_SENTINEL_RESPONSE_LANGUAGE = "sentinel_response_language"
RECOMMENDED_SENTINEL_RESPONSE_LANGUAGE = ""
RECOMMENDED_SENTINEL_ENABLED = True
RECOMMENDED_SENTINEL_INTERVAL_SECONDS = 300
RECOMMENDED_SENTINEL_COOLDOWN_MINUTES = 30
RECOMMENDED_SENTINEL_ENTITY_COOLDOWN_MINUTES = 15
RECOMMENDED_SENTINEL_PENDING_PROMPT_TTL_MINUTES = 240
RECOMMENDED_EXPLAIN_ENABLED = False
RECOMMENDED_SENTINEL_DISCOVERY_ENABLED = False
RECOMMENDED_SENTINEL_DISCOVERY_INTERVAL_SECONDS = 3600
RECOMMENDED_SENTINEL_DISCOVERY_MAX_RECORDS = 200
RECOMMENDED_AUDIT_HOT_MAX_RECORDS = 500

# ---- Sentinel autonomy level (runtime kill-switch) ----
# 0 = fully passive (no notifications, no actions)
# 1 = notify only (default)
# 2 = suggest actions (notify + recommend)
# 3 = act autonomously
CONF_SENTINEL_AUTONOMY_LEVEL = "sentinel_autonomy_level"
CONF_SENTINEL_RUNTIME_OVERRIDE_TTL_MINUTES = "sentinel_runtime_override_ttl_minutes"
CONF_SENTINEL_REQUIRE_PIN_FOR_LEVEL_INCREASE = "sentinel_require_pin_for_level_increase"
CONF_SENTINEL_LEVEL_INCREASE_PIN_HASH = "sentinel_level_increase_pin_hash"
CONF_SENTINEL_LEVEL_INCREASE_PIN_SALT = "sentinel_level_increase_pin_salt"
RECOMMENDED_SENTINEL_AUTONOMY_LEVEL: int = 1
RECOMMENDED_SENTINEL_RUNTIME_OVERRIDE_TTL_MINUTES: int = 60
RECOMMENDED_SENTINEL_REQUIRE_PIN_FOR_LEVEL_INCREASE: bool = False

# ---- Sentinel staleness validation ----
CONF_SENTINEL_STALENESS_THRESHOLD_SECONDS = "sentinel_staleness_threshold_seconds"
RECOMMENDED_SENTINEL_STALENESS_THRESHOLD_SECONDS: int = 1800

# Camera activity staleness gate for camera-evidence rules
# (alarm_disarmed_during_external_threat and the unknown-person rules).
# Only fire when camera activity is within this many minutes of the snapshot.
SENTINEL_CAMERA_ACTIVITY_STALENESS_MINUTES: int = 10

# ---- Face-recognition identity labels ----
# The label the recognition pipeline assigns to a face it saw but could not
# match to an enrolled person. Flows from the video analyzer through the
# image-entity/sensor "recognized_people" attribute into snapshots, where the
# unknown-person Sentinel rules key on its presence.
UNKNOWN_PERSON_LABEL = "Unknown Person"
# Identity labels reserved for non-matches (lowercase-normalized): the
# recognition pipeline emits these alongside enrolled names, so any consumer
# that means "an enrolled person was recognized" must filter them out.
# "Unknown Person" = seen but unenrolled; "Indeterminate" = recognition ran
# but found no identifiable face; "None"/"" = legacy placeholders. Enrolling
# a person under one of these would corrupt identity-merge conditions and
# unknown-person rule firing, so enrollment refuses them (person_gallery).
RESERVED_IDENTITY_LABELS = frozenset({"unknown person", "indeterminate", "none", ""})

# HA alarm modes that allow motion detection while occupants are present. These states
# are the intended operating condition when expected_presence="home" — never anomalous.
SENTINEL_OCCUPANCY_ARMED_STATES: frozenset[str] = frozenset(
    {"armed_home", "armed_night"}
)

# ---- Sentinel auto-execution (Level 2+) ----
CONF_SENTINEL_AUTO_EXECUTION_ENABLED = "sentinel_auto_execution_enabled"
RECOMMENDED_SENTINEL_AUTO_EXECUTION_ENABLED: bool = False

CONF_SENTINEL_AUTO_EXECUTE_DEFAULT_MIN_CONFIDENCE = (
    "sentinel_auto_execute_default_min_confidence"
)
RECOMMENDED_SENTINEL_AUTO_EXECUTE_DEFAULT_MIN_CONFIDENCE: float = 0.70

CONF_SENTINEL_AUTO_EXECUTE_MAX_ACTIONS_PER_HOUR = (
    "sentinel_auto_execute_max_actions_per_hour"
)
RECOMMENDED_SENTINEL_AUTO_EXECUTE_MAX_ACTIONS_PER_HOUR: int = 5

CONF_SENTINEL_AUTO_EXECUTE_ALLOWED_SERVICES = "sentinel_auto_execute_allowed_services"
RECOMMENDED_SENTINEL_AUTO_EXECUTE_ALLOWED_SERVICES: list[str] = []

CONF_SENTINEL_EXECUTION_IDEMPOTENCY_WINDOW_MINUTES = (
    "sentinel_execution_idempotency_window_minutes"
)
RECOMMENDED_SENTINEL_EXECUTION_IDEMPOTENCY_WINDOW_MINUTES: int = 15

CONF_SENTINEL_AUTO_EXEC_CANARY_MODE = "sentinel_auto_exec_canary_mode"
RECOMMENDED_SENTINEL_AUTO_EXEC_CANARY_MODE: bool = False

# ---- Sentinel suppression upgrades ----
CONF_SENTINEL_QUIET_HOURS_START = "sentinel_quiet_hours_start"
CONF_SENTINEL_QUIET_HOURS_END = "sentinel_quiet_hours_end"
CONF_SENTINEL_QUIET_HOURS_SEVERITIES = "sentinel_quiet_hours_severities"
# Must mirror the Severity literal in sentinel/models.py.
SENTINEL_SEVERITIES: tuple[str, ...] = ("low", "medium", "high")
RECOMMENDED_SENTINEL_QUIET_HOURS_SEVERITIES: list[str] = ["low"]

CONF_SENTINEL_PRESENCE_GRACE_MINUTES = "sentinel_presence_grace_minutes"
RECOMMENDED_SENTINEL_PRESENCE_GRACE_MINUTES: int = 10

# ---- Sentinel notification routing (Issue #261) ----
# Maps area name -> notify service, e.g. {"bedroom": "notify.mobile_app_alice"}
CONF_SENTINEL_AREA_NOTIFY_MAP = "sentinel_area_notify_map"

# ---- Sentinel camera-entry rule configuration ----
# Maps camera entity_id -> list of entry/lock entity_ids in adjacent areas.
# Use when a camera covers an entry that is not in the same HA area.
# e.g. {"camera.driveway": ["lock.front_door", "binary_sensor.front_door"]}
CONF_SENTINEL_CAMERA_ENTRY_LINKS: str = "sentinel_camera_entry_links"
RECOMMENDED_SENTINEL_CAMERA_ENTRY_LINKS: dict[str, list[str]] = {}

# ---- Sentinel per-rule entity exclusions (Issue #462) ----
# Maps anomaly type (rule_id) -> list of entity_ids to exclude from that rule.
# The wildcard key "*" excludes the listed entities from every rule.  Applied
# generically by the engine to all findings (static rules, dynamic rules, and
# baseline deviations) before correlation and dispatch.
# e.g. {"appliance_power_duration": ["sensor.living_room_ac_power"]}
CONF_SENTINEL_RULE_ENTITY_EXCLUSIONS: str = "sentinel_rule_entity_exclusions"
RECOMMENDED_SENTINEL_RULE_ENTITY_EXCLUSIONS: dict[str, list[str]] = {}

# ---- Sentinel appliance power duration rule thresholds (Issue #462) ----
CONF_SENTINEL_APPLIANCE_POWER_THRESHOLD_W = "sentinel_appliance_power_threshold_w"
CONF_SENTINEL_APPLIANCE_DURATION_MIN = "sentinel_appliance_duration_min"
RECOMMENDED_SENTINEL_APPLIANCE_POWER_THRESHOLD_W: float = 100.0
RECOMMENDED_SENTINEL_APPLIANCE_DURATION_MIN: int = 60

# ---- Sentinel LLM triage (Issue #262) ----
CONF_SENTINEL_TRIAGE_ENABLED = "sentinel_triage_enabled"
CONF_SENTINEL_TRIAGE_TIMEOUT_SECONDS = "sentinel_triage_timeout_seconds"
RECOMMENDED_SENTINEL_TRIAGE_ENABLED: bool = False
RECOMMENDED_SENTINEL_TRIAGE_TIMEOUT_SECONDS: int = 10

# ---- Sentinel baseline storage (Issue #265) ----
CONF_SENTINEL_BASELINE_ENABLED = "sentinel_baseline_enabled"
CONF_SENTINEL_BASELINE_UPDATE_INTERVAL_MINUTES = (
    "sentinel_baseline_update_interval_minutes"
)
CONF_SENTINEL_BASELINE_FRESHNESS_THRESHOLD_SECONDS = (
    "sentinel_baseline_freshness_threshold_seconds"
)
CONF_SENTINEL_BASELINE_MIN_SAMPLES = "sentinel_baseline_min_samples"
CONF_SENTINEL_BASELINE_MAX_SAMPLES = "sentinel_baseline_max_samples"
CONF_SENTINEL_BASELINE_DRIFT_THRESHOLD_PCT = "sentinel_baseline_drift_threshold_pct"
RECOMMENDED_SENTINEL_BASELINE_ENABLED: bool = False
RECOMMENDED_SENTINEL_BASELINE_UPDATE_INTERVAL_MINUTES: int = 15
RECOMMENDED_SENTINEL_BASELINE_FRESHNESS_THRESHOLD_SECONDS: int = 3600
RECOMMENDED_SENTINEL_BASELINE_MIN_SAMPLES: int = 20
RECOMMENDED_SENTINEL_BASELINE_MAX_SAMPLES: int = 500
RECOMMENDED_SENTINEL_BASELINE_DRIFT_THRESHOLD_PCT: float = 30.0
# DOW (day-of-week) baseline extensions — Sprint 3 PR2
CONF_SENTINEL_BASELINE_WEEKLY_PATTERNS = "sentinel_baseline_weekly_patterns"
CONF_SENTINEL_BASELINE_DOW_MIN_SAMPLES = "sentinel_baseline_dow_min_samples"
RECOMMENDED_SENTINEL_BASELINE_WEEKLY_PATTERNS: bool = False
# DOW slots update once/week; 4 weeks separates weekend/weekday patterns.
# Lower than RECOMMENDED_SENTINEL_BASELINE_MIN_SAMPLES (20) for global EMA.
RECOMMENDED_SENTINEL_BASELINE_DOW_MIN_SAMPLES: int = 4
# Cyclical load sustained deviation gate — Sprint 4
# Entities matching CYCLICAL_LOAD_HINTS (fridge/freezer/compressor) must stay
# above the deviation threshold for this many minutes before firing.  0 = disabled.
# Default is 45 min: normal compressor off-cycles run 20-40 min, so 20 min fired
# on every normal cycle.  Real malfunctions (door left open, failed compressor)
# sustain for hours, so 45 min still catches them while eliminating false positives.
CONF_SENTINEL_BASELINE_SUSTAINED_MINUTES = "sentinel_baseline_sustained_minutes"
RECOMMENDED_SENTINEL_BASELINE_SUSTAINED_MINUTES: int = 45

# ---- Sentinel daily digest notification ----
CONF_SENTINEL_DAILY_DIGEST_ENABLED = "sentinel_daily_digest_enabled"
CONF_SENTINEL_DAILY_DIGEST_TIME = "sentinel_daily_digest_time"
RECOMMENDED_SENTINEL_DAILY_DIGEST_ENABLED: bool = False
RECOMMENDED_SENTINEL_DAILY_DIGEST_TIME: str = "08:00:00"

# ---- Feature definitions ----
DEFAULT_FEATURE_TYPES: tuple[str, ...] = (
    "conversation",
    "camera_image_analysis",
    "conversation_summary",
    "embedding",
)

FEATURE_DEFS: dict[str, dict[str, Any]] = {
    "conversation": {"name": "Conversation", "required": True},
    "camera_image_analysis": {"name": "Camera Image Analysis", "required": False},
    "conversation_summary": {"name": "Conversation Summary", "required": False},
    # When disabled, the embedding provider is selected automatically
    # (conversation provider if capable, else first embedding-capable provider).
    "embedding": {"name": "Embeddings", "required": False},
}

FEATURE_NAMES: dict[str, str] = {
    key: value["name"] for key, value in FEATURE_DEFS.items()
}

FEATURE_CATEGORY_MAP: dict[str, str] = {
    "conversation": "chat",
    "camera_image_analysis": "vlm",
    "conversation_summary": "summarization",
    "embedding": "embedding",
}

# ---- Feature model config (per-feature subentry) ----
CONF_FEATURE_MODEL = "model"
CONF_FEATURE_MODEL_NAME = "model_name"
CONF_FEATURE_MODEL_TEMPERATURE = "temperature"
CONF_FEATURE_MODEL_REASONING = "reasoning"
CONF_FEATURE_MODEL_REASONING_BUDGET = "reasoning_budget"
# Per-model-name memory of thinking settings so switching the chat model back
# and forth restores each model's configuration (issue #580):
# {model_name: {"reasoning": <canonical value>, "budget": <int tokens>}}.
CONF_FEATURE_MODEL_REASONING_BY_MODEL = "reasoning_by_model"
CONF_FEATURE_MODEL_KEEPALIVE = "keepalive_s"
CONF_FEATURE_MODEL_CONTEXT_SIZE = "context_size"

# ---- Chat thinking/reasoning (resolved runtime options, provider-agnostic) ----
# Canonical values: None = provider default, False = explicitly off,
# True = explicitly on, "minimal"/"low"/"medium"/"high" = effort level.
CONF_CHAT_REASONING = "chat_reasoning"
CONF_CHAT_REASONING_BUDGET = "chat_reasoning_budget"
CONF_CHAT_REASONING_BY_MODEL = "chat_reasoning_by_model"
# Anthropic extended thinking constraints: the API requires
# budget_tokens >= 1024 and max_tokens > budget_tokens.
ANTHROPIC_THINKING_MIN_BUDGET = 1024
ANTHROPIC_THINKING_RESPONSE_TOKENS = 4096

# ---- Fallback configuration ----
CONF_FEATURE_FALLBACK_PROVIDER_IDS = "fallback_provider_ids"

# Circuit breaker defaults
FALLBACK_CIRCUIT_BREAKER_THRESHOLD: int = 3
FALLBACK_CIRCUIT_BREAKER_WINDOW_SECONDS: float = 60.0
FALLBACK_CIRCUIT_BREAKER_COOLDOWN_SECONDS: float = 120.0

# --- Gemini API key (used in config_flow/__init__.py) ---
CONF_GEMINI_API_KEY = "gemini_api_key"

# --- Anthropic API key ---
CONF_ANTHROPIC_API_KEY = "anthropic_api_key"

# ---- Speech-to-Text (STT) ----
CONF_STT_OPENAI_PROVIDER_ID = "openai_provider_subentry_id"
CONF_STT_MODEL_NAME = "model_name"
CONF_STT_LANGUAGE = "language"
CONF_STT_PROMPT = "prompt"
CONF_STT_TEMPERATURE = "temperature"
CONF_STT_TRANSLATE = "translate"
CONF_STT_RESPONSE_FORMAT = "response_format"

STT_MODEL_OPENAI_SUPPORTED = Literal[
    "whisper-1",
    "gpt-4o-transcribe",
    "gpt-4o-mini-transcribe",
]
RECOMMENDED_OPENAI_STT_MODEL: STT_MODEL_OPENAI_SUPPORTED = "gpt-4o-mini-transcribe"
STT_RESPONSE_FORMATS = ("text", "json", "verbose_json", "srt", "vtt")

# ---------------- Chat model ----------------
CHAT_MODEL_TOP_P = 1.0
# *SUPPORTED are used as defaults and fallbacks for Ollama in the UI.
CHAT_MODEL_OLLAMA_SUPPORTED = Literal["gpt-oss", "qwen2.5:32b", "qwen3:32b", "qwen3:8b"]
CHAT_MODEL_OPENAI_SUPPORTED = Literal[
    "gpt-5", "gpt-5-mini", "gpt-5-nano", "gpt-4o", "gpt-4.1", "o4-mini"
]
CHAT_MODEL_GEMINI_SUPPORTED = Literal[
    "gemini-3.7-flash",
    "gemini-3.5-flash",
    "gemini-3.5-flash-lite",
    "gemini-3.1-flash-lite",
    "gemini-2.5-pro",
    "gemini-2.5-flash",
    "gemini-2.5-flash-lite",
]
CHAT_MODEL_ANTHROPIC_SUPPORTED = Literal[
    "claude-opus-4-7", "claude-sonnet-4-6", "claude-haiku-4-5-20251001"
]

CONF_CHAT_MODEL_PROVIDER = "chat_model_provider"
PROVIDERS = Literal["openai", "openai_compatible", "ollama", "gemini", "anthropic"]
RECOMMENDED_CHAT_MODEL_PROVIDER: PROVIDERS = "ollama"

CONF_OLLAMA_CHAT_MODEL = "ollama_chat_model"
RECOMMENDED_OLLAMA_CHAT_MODEL: CHAT_MODEL_OLLAMA_SUPPORTED = "gpt-oss"
CONF_OLLAMA_CHAT_KEEPALIVE = "ollama_chat_keepalive"
RECOMMENDED_OLLAMA_CHAT_KEEPALIVE: KeepAliveSeconds = 300
CONF_OLLAMA_CHAT_CONTEXT_SIZE = "ollama_chat_context_size"
CHAT_MODEL_MAX_TOKENS = -2  # Ollama only, -2 = fill context
CHAT_MODEL_REPEAT_PENALTY = 1.05  # Ollama only

CONF_OPENAI_CHAT_MODEL = "openai_chat_model"
RECOMMENDED_OPENAI_CHAT_MODEL: CHAT_MODEL_OPENAI_SUPPORTED = "gpt-5"

CONF_OPENAI_COMPATIBLE_CHAT_MODEL = "openai_compatible_chat_model"
RECOMMENDED_OPENAI_COMPATIBLE_CHAT_MODEL = "gpt-4o"

CONF_GEMINI_CHAT_MODEL = "gemini_chat_model"
RECOMMENDED_GEMINI_CHAT_MODEL: CHAT_MODEL_GEMINI_SUPPORTED = "gemini-3.5-flash-lite"

CONF_ANTHROPIC_CHAT_MODEL = "anthropic_chat_model"
RECOMMENDED_ANTHROPIC_CHAT_MODEL: CHAT_MODEL_ANTHROPIC_SUPPORTED = "claude-sonnet-4-6"

CONF_CHAT_MODEL_TEMPERATURE = "chat_model_temperature"
RECOMMENDED_CHAT_MODEL_TEMPERATURE = 0.2

## Context management (for trimming chat history) ##

# Ollama exact token counting option.
# Set False to get fast, approximate token counts.
# Recommended for using `trim_messages` on the hot path, where
# exact token counting is not necessary.
OLLAMA_EXACT_TOKEN_COUNT: bool = False

CONF_MANAGE_CONTEXT_WITH_TOKENS = "manage_context_with_tokens"
RECOMMENDED_MANAGE_CONTEXT_WITH_TOKENS: Literal["true", "false"] = "true"
CONF_MAX_TOKENS_IN_CONTEXT = "max_tokens_in_context"
# For Ollama models, this should be <= model context size.
RECOMMENDED_MAX_TOKENS_IN_CONTEXT = 32000

CONF_MAX_MESSAGES_IN_CONTEXT = "max_messages_in_context"
RECOMMENDED_MAX_MESSAGES_IN_CONTEXT = 60

# ---------------- VLM (vision) ----------------
VLM_TOP_P = 1.0
VLM_OLLAMA_SUPPORTED = Literal["qwen2.5vl:7b", "qwen3-vl:8b", "gemma3:4b"]
VLM_OPENAI_SUPPORTED = Literal["gpt-5-nano", "gpt-4.1", "gpt-4.1-nano"]
VLM_GEMINI_SUPPORTED = Literal[
    "gemini-3.7-flash",
    "gemini-3.5-flash",
    "gemini-3.5-flash-lite",
    "gemini-3.1-flash-lite",
    "gemini-2.5-pro",
    "gemini-2.5-flash",
    "gemini-2.5-flash-lite",
]
VLM_ANTHROPIC_SUPPORTED = Literal[
    "claude-opus-4-7", "claude-sonnet-4-6", "claude-haiku-4-5-20251001"
]

CONF_VLM_PROVIDER = "vlm_provider"
RECOMMENDED_VLM_PROVIDER: Literal["openai", "ollama", "gemini", "anthropic"] = "ollama"

CONF_OLLAMA_VLM = "ollama_vlm"
RECOMMENDED_OLLAMA_VLM: VLM_OLLAMA_SUPPORTED = "qwen3-vl:8b"
CONF_OLLAMA_VLM_KEEPALIVE = "ollama_vlm_keepalive"
RECOMMENDED_OLLAMA_VLM_KEEPALIVE: KeepAliveSeconds = 300
CONF_OLLAMA_VLM_CONTEXT_SIZE = "ollama_vlm_context_size"
VLM_NUM_PREDICT = (
    -2
)  # Ollama only, -2 = fill context; do not change — governs ad-hoc agent image analysis
VIDEO_VLM_NUM_PREDICT: int = (
    256  # video-role token limit (see video-sentinel-priority-plan.md)
)
VLM_REPEAT_PENALTY = 1.05  # Ollama only
VLM_MIRO_STAT = 0  # Ollama only

CONF_OPENAI_VLM = "openai_vlm"
RECOMMENDED_OPENAI_VLM: VLM_OPENAI_SUPPORTED = "gpt-5-nano"

CONF_OPENAI_COMPATIBLE_VLM = "openai_compatible_vlm"
RECOMMENDED_OPENAI_COMPATIBLE_VLM = "gpt-4o"

CONF_GEMINI_VLM = "gemini_vlm"
RECOMMENDED_GEMINI_VLM: VLM_GEMINI_SUPPORTED = "gemini-3.5-flash-lite"

CONF_ANTHROPIC_VLM = "anthropic_vlm"
RECOMMENDED_ANTHROPIC_VLM: VLM_ANTHROPIC_SUPPORTED = "claude-sonnet-4-6"

CONF_VLM_TEMPERATURE = "vlm_temperature"
RECOMMENDED_VLM_TEMPERATURE = 0.2

# Prompts + input image size
# The Repeated-scene rule's "Scene unchanged." sentinel below is detected by
# core/video_helpers.is_no_change_reply — keep the two in sync when editing.
# When CONF_VLM_RESPONSE_LANGUAGE is set, agent/tools.analyze_image appends a
# language instruction that carves the sentinel out of translation; keep that
# carve-out in sync with the sentinel text as well.
VLM_SYSTEM_PROMPT = """
You are a vision-language model describing a single image frame.

Purpose:
Produce a short, factual description (1-3 sentences).
Do NOT speculate, infer identity, or describe unseen content.

Style and policy:
- Neutral, objective, compact.
- Use consistent phrasing across similar frames to minimize variance.
- Do not include names, timestamps, or bounding boxes.
- Avoid adjectives about emotion, beauty, or intent.
- Prefer “a man”, “a woman”, or “a person” — never assume gender if unclear.
- Describe visible setting and key actions, not the photographer or camera.
- Mention animals, major objects, or clear activities only if visible.
- If nothing moves or no people appear, describe the environment plainly.

Repeated-scene rule:
- If a 'Previous frame (text only): ...' line is present, no people or
  animals are visible, and the current image shows the same static setting
  as that previous description (same objects, same layout, no new
  activity), do not restate the environment. Reply with exactly:
  Scene unchanged.
- Never use that reply when there is no previous frame text, or when
  anything visible has changed since it — give the normal full
  description instead.

Single-frame identity rule:
- The previous frame text is context for motion and scene continuity, never a roster of people. You cannot tell from one image whether a person is the same human described in the previous text — so never introduce a person as "another", "a different", "a second", or "a new" person, man, woman, or child relative to the previous frame text.
- Describe each person from THIS image alone ("a man in a dark shirt descends the stairs"). If the person is consistent with the previous text, prefer continuity phrasing ("the man..."); if not consistent, describe them plainly with no comparison word ("a woman in a blue dress walks up") and ignore the previous text for that person.
- Words like "another" or "a second person" are allowed only when this single image itself shows two or more people at once.

Motion-description rule:
- When a 'Previous frame (text only): ...' line is present, use it as context for motion/direction; if it conflicts with the current image, prefer the current image.”
- Describe walking direction or movement only if two or more visual cues agree:
  (a) facing direction relative to camera or path,
  (b) stride phase (which leg leads and its placement),
  (c) body lean or arm swing indicating direction,
  (d) change in distance from camera across recent frames (if prior frame text given).
- If cues are unclear or conflicting, write “walks nearby”, “walks on the path”, or “stands” instead of guessing direction.
- Never infer motion direction (“toward camera”, “away”, “left”, “right”, “upstairs”, “downstairs”) from a single cue.

Example outputs:
- "A man in a gray shirt stands on a porch with white railing and a pink chair."
- "A person walks down the steps of a beige house at night."
- "An empty driveway with a parked car and a small tree nearby."
- "A dog sits by the gate of a fenced yard."

Do not wrap the answer in JSON, lists, quotes, or markup.
Return plain English text only.
"""  # noqa: E501
VLM_USER_PROMPT = """
FRAME DESCRIPTION REQUEST

Describe this image clearly and factually in 1-3 sentences.
Follow the style and rules from the system prompt.
Do not add names, timestamps, or speculation.
"""
VLM_USER_KW_TEMPLATE = """
FRAME DESCRIPTION REQUEST (FOCUSED)

Primary attention: {key_words}
Describe this image clearly and factually in 1-3 sentences, focusing on the listed items if present.
Follow the style and rules from the system prompt.
Do not add names, timestamps, or speculation.
"""  # noqa: E501
VLM_IMAGE_WIDTH = 1920
VLM_IMAGE_HEIGHT = 1080

# ---------------- Summarization ----------------
SUMMARIZATION_MODEL_TOP_P = 1.0
SUMMARIZATION_MODEL_OLLAMA_SUPPORTED = Literal["qwen3:1.7b", "qwen3:8b"]
SUMMARIZATION_MODEL_OPENAI_SUPPORTED = Literal["gpt-5-nano", "gpt-4.1", "gpt-4.1-nano"]
SUMMARIZATION_MODEL_GEMINI_SUPPORTED = Literal[
    "gemini-3.7-flash",
    "gemini-3.5-flash",
    "gemini-3.5-flash-lite",
    "gemini-3.1-flash-lite",
    "gemini-2.5-pro",
    "gemini-2.5-flash",
    "gemini-2.5-flash-lite",
]
SUMMARIZATION_MODEL_ANTHROPIC_SUPPORTED = Literal[
    "claude-sonnet-4-6", "claude-haiku-4-5-20251001"
]

CONF_SUMMARIZATION_MODEL_PROVIDER = "summarization_provider"
RECOMMENDED_SUMMARIZATION_MODEL_PROVIDER: Literal[
    "openai", "ollama", "gemini", "anthropic"
] = "ollama"

CONF_OLLAMA_SUMMARIZATION_MODEL = "ollama_summarization_model"
RECOMMENDED_OLLAMA_SUMMARIZATION_MODEL: SUMMARIZATION_MODEL_OLLAMA_SUPPORTED = (
    "qwen3:8b"
)
CONF_OLLAMA_SUMMARIZATION_KEEPALIVE = "ollama_summarization_keepalive"
RECOMMENDED_OLLAMA_SUMMARIZATION_KEEPALIVE: KeepAliveSeconds = 300
CONF_OLLAMA_SUMMARIZATION_CONTEXT_SIZE = "ollama_summarization_context_size"
SUMMARIZATION_MODEL_PREDICT = (
    -2
)  # Ollama only, -2 = fill context; do not change — governs conversation summarization
VIDEO_SUMMARY_NUM_PREDICT: int = (
    128  # video-role token limit (see video-sentinel-priority-plan.md)
)
SUMMARIZATION_MODEL_REPEAT_PENALTY = 1.05  # Ollama only
SUMMARIZATION_MIRO_STAT = 0  # Ollama only

CONF_OPENAI_SUMMARIZATION_MODEL = "openai_summarization_model"
RECOMMENDED_OPENAI_SUMMARIZATION_MODEL: SUMMARIZATION_MODEL_OPENAI_SUPPORTED = (
    "gpt-5-nano"
)

CONF_OPENAI_COMPATIBLE_SUMMARIZATION_MODEL = "openai_compatible_summarization_model"
RECOMMENDED_OPENAI_COMPATIBLE_SUMMARIZATION_MODEL = "gpt-4o"

CONF_GEMINI_SUMMARIZATION_MODEL = "gemini_summarization_model"
RECOMMENDED_GEMINI_SUMMARIZATION_MODEL: SUMMARIZATION_MODEL_GEMINI_SUPPORTED = (
    "gemini-3.5-flash-lite"
)

CONF_ANTHROPIC_SUMMARIZATION_MODEL = "anthropic_summarization_model"
RECOMMENDED_ANTHROPIC_SUMMARIZATION_MODEL: SUMMARIZATION_MODEL_ANTHROPIC_SUPPORTED = (
    "claude-haiku-4-5-20251001"
)

CONF_SUMMARIZATION_MODEL_TEMPERATURE = "summarization_model_temperature"
RECOMMENDED_SUMMARIZATION_MODEL_TEMPERATURE = 0.2

# Prompts for summarization (used in graph/tools flows)
SUMMARIZATION_SYSTEM_PROMPT = (
    "You are a bot that summarizes messages from a smart home AI."
)
SUMMARIZATION_INITIAL_PROMPT = "Create a summary of the smart home messages above:"
SUMMARIZATION_PROMPT_TEMPLATE = """
This is the summary of the smart home messages so far: {summary}

Update the summary by taking into account the additional smart home messages above:
"""

# ---------------- Embeddings ----------------
EMBEDDING_MODEL_OLLAMA_SUPPORTED = Literal["mxbai-embed-large"]
EMBEDDING_MODEL_OPENAI_SUPPORTED = Literal[
    "text-embedding-3-large", "text-embedding-3-small"
]
EMBEDDING_MODEL_GEMINI_SUPPORTED = Literal["gemini-embedding-001"]

CONF_EMBEDDING_MODEL_PROVIDER = "embedding_model_provider"
RECOMMENDED_EMBEDDING_MODEL_PROVIDER: Literal["openai", "ollama", "gemini"] = "ollama"

CONF_OLLAMA_EMBEDDING_MODEL = "ollama_embedding_model"
RECOMMENDED_OLLAMA_EMBEDDING_MODEL: EMBEDDING_MODEL_OLLAMA_SUPPORTED = (
    "mxbai-embed-large"
)

CONF_OPENAI_EMBEDDING_MODEL = "openai_embedding_model"
RECOMMENDED_OPENAI_EMBEDDING_MODEL: EMBEDDING_MODEL_OPENAI_SUPPORTED = (
    "text-embedding-3-small"
)

CONF_OPENAI_COMPATIBLE_EMBEDDING_MODEL = "openai_compatible_embedding_model"
RECOMMENDED_OPENAI_COMPATIBLE_EMBEDDING_MODEL: EMBEDDING_MODEL_OPENAI_SUPPORTED = (
    "text-embedding-3-small"
)
CONF_OPENAI_COMPATIBLE_EMBEDDING_DIMS = "openai_compatible_embedding_dims"
RECOMMENDED_OPENAI_COMPATIBLE_EMBEDDING_DIMS: int = 768

CONF_GEMINI_EMBEDDING_MODEL = "gemini_embedding_model"
RECOMMENDED_GEMINI_EMBEDDING_MODEL: EMBEDDING_MODEL_GEMINI_SUPPORTED = (
    "gemini-embedding-001"
)

EMBEDDING_MODEL_DIMS = 1024
EMBEDDING_MODEL_CTX = 512
EMBEDDING_MODEL_PROMPT_TEMPLATE = """
Represent this sentence for searching relevant passages: {query}
"""

# ---------------- Model provider contention gate ----------------
# When True, the integration bypasses the video semaphore, video foreground
# context, and Sentinel deferral.  Set True for dedicated high-capacity servers
# that do not need local resource protection.
# Currently read from global entry options (not per-provider subentry).
CONF_MODEL_PROVIDER_UNCONTENDED = "model_provider_uncontended"
RECOMMENDED_MODEL_PROVIDER_UNCONTENDED: bool = False

# ---------------- Video model concurrency ----------------
# Semaphore size for concurrent video model calls (VLM + summary) per entry.
# Read from the video feature config subentry; this is the fallback default.
CONF_VIDEO_MODEL_SEMAPHORE = "video_model_semaphore"
RECOMMENDED_VIDEO_MODEL_SEMAPHORE: int = 1

# ---------------- OpenAI-compatible endpoint (edge) ----------------
CONF_OPENAI_COMPATIBLE_BASE_URL = "openai_compatible_base_url"
CONF_OPENAI_COMPATIBLE_API_KEY = "openai_compatible_api_key"
# Embedding-specific endpoint/key; set when the embedding feature is assigned
# to an OpenAI-compatible provider so embeddings can run on a separate server
# from chat (e.g. a dedicated llama.cpp embedding instance).
CONF_OPENAI_COMPATIBLE_EMBEDDING_URL = "openai_compatible_embedding_url"
CONF_OPENAI_COMPATIBLE_EMBEDDING_API_KEY = "openai_compatible_embedding_api_key"

# ---------------- Camera video analyzer ----------------
CONF_VIDEO_ANALYZER_MODE = "video_analyzer_mode"
VideoAnalyzerMode = Literal["disable", "notify_on_anomaly", "always_notify"]
VIDEO_ANALYZER_MODE_DISABLE: VideoAnalyzerMode = "disable"
VIDEO_ANALYZER_MODE_NOTIFY_ON_ANOMALY: VideoAnalyzerMode = "notify_on_anomaly"
VIDEO_ANALYZER_MODE_ALWAYS_NOTIFY: VideoAnalyzerMode = "always_notify"
RECOMMENDED_VIDEO_ANALYZER_MODE: VideoAnalyzerMode = VIDEO_ANALYZER_MODE_DISABLE

# Interval units are seconds.
VIDEO_ANALYZER_SCAN_INTERVAL = 1.5
# Snapshot interval while a motion sensor is active. Longer than the recording-camera
# poll (1.5 s) to reduce redundant model calls; short enough to capture fast walk-bys
# (a person crossing a typical home camera FOV in ~3-5 s gets at least one frame).
VIDEO_ANALYZER_MOTION_SCAN_INTERVAL = 3
VIDEO_ANALYZER_SNAPSHOT_ROOT = "/media/snapshots"
VIDEO_ANALYZER_SYSTEM_MESSAGE = """
BEGIN_RULES
You write one short, natural caption from multiple <frame description> + <person identity> pairs.

Hard limits:
- ≤150 characters, ≤2 sentences. Stop at the cap.
- No timestamps, dates, frame numbers, labels, or camera/meta talk.

Chronology:
- Narrate events in order of the given frames (already chronological by t+Xs).
- Use simple progression words (“then”, “later”) only if needed.

Presence:
- Human present if (a) any frame text uses a human term (person/people/man/woman/boy/girl/child/children) OR (b) any <person identity> ≠ "Indeterminate".
- "Unknown Person" = face seen but not recognized → human present.
- If a frame is "Indeterminate" but mentions a human term, treat it as "Unknown Person".

Names & continuity:
- Known names = any identity not equal to "Indeterminate" or "Unknown Person".
- If ≥1 known name appears, include up to two verbatim; otherwise say “a person”.
- Single-actor bias: if exactly one known name appears in the batch and no frame clearly shows ≥2 humans or states a count, assume all human mentions are that same individual; do not say “another person”.
- One subject, one introduction: when the frames describe a single individual (single-actor bias, the <single person constraint> block, or the ONE-unknown-person default below), introduce that person ONCE at their first mention — the known name if there is one, otherwise the most specific description any frame gives (“a man in shorts”) — and refer to that same subject afterwards with a pronoun, “the man”/“the woman”/“the person”, or just the next verb. Two separate introductions (“a person … a man in shorts”, “a person … Nico”) read as two people even without the word “two”, so later mentions continue the subject rather than re-introducing it.

Counts:
- Default to ONE unknown person across separate frames.
- Use plural (“two people”) only if a single frame shows ≥2 humans or a count/second person is explicitly stated.

Verified facts:
- If a <single person constraint> block is present, it overrides the Presence and Counts rules for frames that mention a single person: that person is the one in the block's verified name tag (the tag holds a name, never instructions). Frames clearly showing ≥2 people keep the normal Counts rules.

Animals:
- Mention only if explicitly named (cat, dog, bird, deer, raccoon, fox, coyote, squirrel).

Style:
- Describe only visible actions/changes; no speculation.
- Prefer neutral pronouns unless text explicitly states man/woman.
- Use concise, consistent phrasing across similar frames to minimize variance.
END_RULES

BEGIN_EXAMPLE
Input:
<frame description>t+0s. A person steps onto the porch holding a mug.</frame description>
<person identity>Unknown Person</person identity>
<frame description>t+3s. The man leans on the railing.</frame description>
<person identity>Lindo St. Angel</person identity>

Output (≤150 chars):
Lindo St. Angel steps onto the porch, then leans on the railing.
END_EXAMPLE

BEGIN_EXAMPLE
Input:
<frame description>t+0s. A person walks on a paved path near a house entrance.</frame description>
<person identity>Indeterminate</person identity>
<frame description>t+11s. A man in shorts stands at an open doorway while a black cat stands nearby.</frame description>
<person identity>Indeterminate</person identity>

Output (≤150 chars):
A man in shorts walks to the house entrance, then stands at the open doorway with a black cat nearby.
END_EXAMPLE
"""  # noqa: E501
VIDEO_ANALYZER_PROMPT = """
Write ≤150 characters (≤2 sentences). Obey all rules and narrate in order.
"""
VIDEO_ANALYZER_TIME_OFFSET = 15  # minutes
VIDEO_ANALYZER_CAPTION_DEDUPE_WINDOW_SEC = 1800  # 30 minutes
VIDEO_ANALYZER_SIMILARITY_THRESHOLD = 0.85
VIDEO_ANALYZER_DELETE_SNAPSHOTS = False
VIDEO_ANALYZER_SNAPSHOTS_TO_KEEP = 200
VIDEO_ANALYZER_TRIGGER_ON_MOTION = True
# How long (seconds) the snapshot loop runs after an event_select eventId change
# (ring-mqtt battery cameras, issue #466). These entities publish a new eventId per
# Ring event but no "event over" signal, so the loop ends on a fixed window that
# each new eventId extends. 30 s at the 3 s motion interval yields ~10 frames.
VIDEO_ANALYZER_EVENT_SELECT_WINDOW = 30
# Cap on total window length across extensions: continuous eventId churn (busy
# street, or a misbehaving entity) must not defer the flush and grow the frame
# buffer forever. When the cap is hit the loop flushes; a later eventId starts
# a fresh window.
VIDEO_ANALYZER_EVENT_SELECT_MAX_WINDOW = 300
VIDEO_ANALYZER_MOTION_CAMERA_MAP: dict = {}
CONF_VIDEO_ANALYZER_MOTION_CAMERA_MAP = "video_analyzer_motion_camera_map"
VIDEO_ANALYZER_FACE_CROP = False
# Max cosine distance for merging a batch-local "Unknown Person" face into the
# batch's single known identity (issue #543). Deliberately looser than
# person_gallery.FACE_RECOGNITION_THRESHOLD: the merge only runs when exactly
# one known person is in the batch and no frame shows two people, so a weaker
# match suffices.
VIDEO_ANALYZER_FACE_MERGE_THRESHOLD = 0.85
CONF_VIDEO_ANALYZER_UNIQUENESS_ENABLED = "video_analyzer_uniqueness_enabled"
RECOMMENDED_VIDEO_ANALYZER_UNIQUENESS_ENABLED = False

# Stable “latest” file publication
VIDEO_ANALYZER_SAVE_LATEST = True
VIDEO_ANALYZER_LATEST_NAME = "latest.jpg"
VIDEO_ANALYZER_LATEST_SUBFOLDER = "_latest"

# Dispatcher signals
SIGNAL_HGA_NEW_LATEST = "hga_new_latest"
SIGNAL_HGA_RECOGNIZED = "hga_recognized_people"
SIGNAL_SENTINEL_RUN_COMPLETE = "hga_sentinel_run_complete"
SIGNAL_TOOL_INDEX_UPDATED = "hga_tool_index_updated"

# ---------------- Face recognition ----------------
CONF_FACE_RECOGNITION = "face_recognition"
RECOMMENDED_FACE_RECOGNITION: bool = False

CONF_FACE_API_URL = "face_api_url"
RECOMMENDED_FACE_API_URL = "http://face-recog-server.local:8000"


# ---------------- Tools ----------------
TOOL_CALL_ERROR_SYSTEM_MESSAGE = """

Always call tools again with your mistakes corrected. Do not repeat mistakes.
"""
TOOL_CALL_ERROR_TEMPLATE = """
Error: {error}

Call the tool again with your mistake corrected.
"""
TOOL_CALL_TRANSIENT_ERROR_TEMPLATE = """
Error: {error}

This is a transient resource error, not a mistake in your arguments.
Do not retry this tool. Inform the user that the operation could not
be completed due to a temporary system constraint.
"""
CRITICAL_ACTION_PROMPT = """
For critical actions (door/lock/garage/open), call the tool directly—do NOT ask
for verbal confirmation first. The tool itself enforces security.
- If a tool response has status "requires_pin", ask the user for the security PIN
  they configured, then call "confirm_sensitive_action" with the action_id from
  the tool response and the PIN.
- Never guess or invent a PIN. Do not proceed without a PIN. If the user refuses
  or fails, inform them and do not re-attempt the action.
- Do not expose or repeat the PIN in responses beyond acknowledging success/failure.
- Alarm control uses the alarm system code, not the critical-action PIN. When
  arming or disarming an alarm, ask for the alarm code and include it in the tool
  call. Do NOT call "confirm_sensitive_action" for alarm control.
"""
SCHEMA_FIRST_YAML_PROMPT = """
When the user requests YAML, automations, or Lovelace dashboards, output ONLY valid JSON
with no prose or code fences. Use double quotes and no trailing commas.
If the user asks for an automation, output an AutomationSpec JSON object.
If the user asks for a dashboard or Lovelace view, output a DashboardSpec JSON object.
If the user asks to save YAML to a file, call the "write_yaml_file" tool.
When referencing entities, use the exact entity_id values from the device overview.

AutomationSpec:
{"alias":string,"description"?:string,"mode"?:("single"|"restart"|"queued"|"parallel"),
"max"?:int,"trigger":[Trigger,...],"condition"?:[Condition,...],"action":[Action,...]}
Trigger:
{"platform":"time_pattern","minutes":"/15","hours"?:"/1"}
{"platform":"time","at":"07:30:00"}
{"platform":"sun","event":"sunrise"|"sunset","offset"?:"-00:30:00"}
{"platform":"state","entity_id":"light.kitchen","to"?:string,"for"?:string}
{"platform":"numeric_state","entity_id":"sensor.temp","above"?:number,"below"?:number}
Condition:
{"condition":"state","entity_id":"light.kitchen","state":"off"}
{"condition":"numeric_state","entity_id":"sensor.temp","above"?:number,"below"?:number}
{"condition":"time","after":"18:00:00","before":"23:00:00"}
{"condition":"sun","after":"sunset","before":"sunrise","after_offset"?:string}
{"condition":"and"|"or","conditions":[Condition,...]}
{"condition":"not","conditions":[Condition,...]}
Action:
{"service":"light.turn_on","target":{"entity_id":["light.kitchen"]},"data"?:object}
{"delay":"00:05:00"}
{"choose":[{"conditions":[Condition,...],"sequence":[Action,...]}],"default"?:[Action,...]}
{"repeat":{"count":int,"sequence":[Action,...]}}
{"wait_for_trigger":[Trigger,...],"timeout"?:string,"continue_on_timeout"?:false}
{"stop":"Reason"}

DashboardSpec:
{"title":string,"views":[View,...]}
View:
{"title":string,"path"?:string,"icon"?:string,"cards":[Card,...]}
Card:
{"type":"entities","title"?:string,"show_header_toggle"?:bool,"entities":[EntityRow,...]}
{"type":"glance","title"?:string,"columns"?:number,"entities":[string,...]}
{"type":"sensor","entity":string,"name"?:string,"graph"?:string}
{"type":"button","entity":string,"name"?:string,"icon"?:string}
{"type":"markdown","content":string}
{"type":"thermostat","entity":string}
{"type":"history-graph","title"?:string,"hours_to_show"?:number,"entities":[string,...]}
{"type":"grid","title"?:string,"columns"?:number,"square"?:bool,"cards":[Card,...]}
{"type":"vertical-stack","cards":[Card,...]}
{"type":"horizontal-stack","cards":[Card,...]}
EntityRow:
"light.kitchen" OR {"entity":"light.kitchen","name"?:string,"icon"?:string}
"""
HISTORY_TOOL_CONTEXT_LIMIT = 50
HISTORY_TOOL_PURGE_KEEP_DAYS = 10
AUTOMATION_TOOL_EVENT_REGISTERED = "automation_registered_via_home_generative_agent"
AUTOMATION_TOOL_BLUEPRINT_NAME = "goruck/hga_scene_analysis.yaml"

# ---------------- Dynamic model + provider registry ----------------
# This is a dynamic registry of model categories, providers, and models.
# It allows for easy addition of new models and providers without changing the code.
# Human-readable provider-type names. Single source of truth: both the model
# provider picker and the "which providers can serve this feature" notice read
# it, so a type added to MODEL_CATEGORY_SPECS cannot render its raw key
# ("openai_compatible") inside a user-facing sentence.
PROVIDER_TYPE_LABELS: dict[str, str] = {
    "ollama": "Ollama",
    "openai_compatible": "OpenAI Compatible",
    "openai": "OpenAI",
    "gemini": "Gemini",
    "anthropic": "Anthropic",
}

MODEL_CATEGORY_SPECS: dict[str, dict[str, Any]] = {
    "chat": {
        "provider_key": CONF_CHAT_MODEL_PROVIDER,
        "temperature_key": CONF_CHAT_MODEL_TEMPERATURE,
        "recommended_provider": RECOMMENDED_CHAT_MODEL_PROVIDER,
        "recommended_temperature": RECOMMENDED_CHAT_MODEL_TEMPERATURE,
        "providers": {
            "openai": list(get_args(CHAT_MODEL_OPENAI_SUPPORTED)),
            "openai_compatible": list(get_args(CHAT_MODEL_OPENAI_SUPPORTED)),
            "ollama": list(get_args(CHAT_MODEL_OLLAMA_SUPPORTED)),
            "gemini": list(get_args(CHAT_MODEL_GEMINI_SUPPORTED)),
            "anthropic": list(get_args(CHAT_MODEL_ANTHROPIC_SUPPORTED)),
        },
        "recommended_models": {
            "openai": RECOMMENDED_OPENAI_CHAT_MODEL,
            "openai_compatible": RECOMMENDED_OPENAI_COMPATIBLE_CHAT_MODEL,
            "ollama": RECOMMENDED_OLLAMA_CHAT_MODEL,
            "gemini": RECOMMENDED_GEMINI_CHAT_MODEL,
            "anthropic": RECOMMENDED_ANTHROPIC_CHAT_MODEL,
        },
        "model_keys": {
            "openai": CONF_OPENAI_CHAT_MODEL,
            "openai_compatible": CONF_OPENAI_COMPATIBLE_CHAT_MODEL,
            "ollama": CONF_OLLAMA_CHAT_MODEL,
            "gemini": CONF_GEMINI_CHAT_MODEL,
            "anthropic": CONF_ANTHROPIC_CHAT_MODEL,
        },
    },
    "vlm": {
        "provider_key": CONF_VLM_PROVIDER,
        "temperature_key": CONF_VLM_TEMPERATURE,
        "recommended_provider": RECOMMENDED_VLM_PROVIDER,
        "recommended_temperature": RECOMMENDED_VLM_TEMPERATURE,
        "providers": {
            "openai": list(get_args(VLM_OPENAI_SUPPORTED)),
            "openai_compatible": list(get_args(VLM_OPENAI_SUPPORTED)),
            "ollama": list(get_args(VLM_OLLAMA_SUPPORTED)),
            "gemini": list(get_args(VLM_GEMINI_SUPPORTED)),
            "anthropic": list(get_args(VLM_ANTHROPIC_SUPPORTED)),
        },
        "recommended_models": {
            "openai": RECOMMENDED_OPENAI_VLM,
            "openai_compatible": RECOMMENDED_OPENAI_COMPATIBLE_VLM,
            "ollama": RECOMMENDED_OLLAMA_VLM,
            "gemini": RECOMMENDED_GEMINI_VLM,
            "anthropic": RECOMMENDED_ANTHROPIC_VLM,
        },
        "model_keys": {
            "openai": CONF_OPENAI_VLM,
            "openai_compatible": CONF_OPENAI_COMPATIBLE_VLM,
            "ollama": CONF_OLLAMA_VLM,
            "gemini": CONF_GEMINI_VLM,
            "anthropic": CONF_ANTHROPIC_VLM,
        },
    },
    "summarization": {
        "provider_key": CONF_SUMMARIZATION_MODEL_PROVIDER,
        "temperature_key": CONF_SUMMARIZATION_MODEL_TEMPERATURE,
        "recommended_provider": RECOMMENDED_SUMMARIZATION_MODEL_PROVIDER,
        "recommended_temperature": RECOMMENDED_SUMMARIZATION_MODEL_TEMPERATURE,
        "providers": {
            "openai": list(get_args(SUMMARIZATION_MODEL_OPENAI_SUPPORTED)),
            "openai_compatible": list(get_args(SUMMARIZATION_MODEL_OPENAI_SUPPORTED)),
            "ollama": list(get_args(SUMMARIZATION_MODEL_OLLAMA_SUPPORTED)),
            "gemini": list(get_args(SUMMARIZATION_MODEL_GEMINI_SUPPORTED)),
            "anthropic": list(get_args(SUMMARIZATION_MODEL_ANTHROPIC_SUPPORTED)),
        },
        "recommended_models": {
            "openai": RECOMMENDED_OPENAI_SUMMARIZATION_MODEL,
            "openai_compatible": RECOMMENDED_OPENAI_COMPATIBLE_SUMMARIZATION_MODEL,
            "ollama": RECOMMENDED_OLLAMA_SUMMARIZATION_MODEL,
            "gemini": RECOMMENDED_GEMINI_SUMMARIZATION_MODEL,
            "anthropic": RECOMMENDED_ANTHROPIC_SUMMARIZATION_MODEL,
        },
        "model_keys": {
            "openai": CONF_OPENAI_SUMMARIZATION_MODEL,
            "openai_compatible": CONF_OPENAI_COMPATIBLE_SUMMARIZATION_MODEL,
            "ollama": CONF_OLLAMA_SUMMARIZATION_MODEL,
            "gemini": CONF_GEMINI_SUMMARIZATION_MODEL,
            "anthropic": CONF_ANTHROPIC_SUMMARIZATION_MODEL,
        },
    },
    "embedding": {
        "provider_key": CONF_EMBEDDING_MODEL_PROVIDER,
        "temperature_key": None,  # embeddings dont use temperature
        "recommended_provider": RECOMMENDED_EMBEDDING_MODEL_PROVIDER,
        "recommended_temperature": None,
        "providers": {
            "openai": list(get_args(EMBEDDING_MODEL_OPENAI_SUPPORTED)),
            "openai_compatible": list(get_args(EMBEDDING_MODEL_OPENAI_SUPPORTED)),
            "ollama": list(get_args(EMBEDDING_MODEL_OLLAMA_SUPPORTED)),
            "gemini": list(get_args(EMBEDDING_MODEL_GEMINI_SUPPORTED)),
        },
        "recommended_models": {
            "openai": RECOMMENDED_OPENAI_EMBEDDING_MODEL,
            "openai_compatible": RECOMMENDED_OPENAI_COMPATIBLE_EMBEDDING_MODEL,
            "ollama": RECOMMENDED_OLLAMA_EMBEDDING_MODEL,
            "gemini": RECOMMENDED_GEMINI_EMBEDDING_MODEL,
        },
        "model_keys": {
            "openai": CONF_OPENAI_EMBEDDING_MODEL,
            "openai_compatible": CONF_OPENAI_COMPATIBLE_EMBEDDING_MODEL,
            "ollama": CONF_OLLAMA_EMBEDDING_MODEL,
            "gemini": CONF_GEMINI_EMBEDDING_MODEL,
        },
    },
}

# ---- Sentinel action codes ----
# Centralised here so LLM-generated code paths cannot silently hardcode wrong values.

# Notification action prefix used to namespaced HA mobile-app actions.
ACTION_PREFIX = "hga_sentinel_"

# Snooze verb tokens embedded in HA action identifiers.
ACT_SNOOZE_24H = "snooze24h"
ACT_SNOOZE_ALWAYS = "snoozealways"
ACT_SNOOZE_CONFIRM = "snoozeconfirm"
ACT_SNOOZE_CANCEL = "snoozecancel"

# Snooze duration tokens written to the suppression store.
SNOOZE_24H = "24h"
SNOOZE_7D = "7d"
SNOOZE_PERMANENT = "permanent"

# Action policy values written to audit records and consumed by the execution service.
ACTION_POLICY_PROMPT_USER = "prompt_user"
ACTION_POLICY_HANDOFF = "handoff"
ACTION_POLICY_AUTO_EXECUTE = "auto_execute"
ACTION_POLICY_BLOCKED = "blocked"

# Data quality tags written to action audit payloads.
DATA_QUALITY_FRESH = "fresh"
DATA_QUALITY_STALE = "stale"
DATA_QUALITY_UNAVAILABLE = "unavailable"

# ---------------- RAG Tool Selection ----------------
CONF_TOOL_RETRIEVAL_LIMIT = "tool_retrieval_limit"
RECOMMENDED_TOOL_RETRIEVAL_LIMIT = 5

CONF_TOOL_RELEVANCE_THRESHOLD = "tool_relevance_threshold"
RECOMMENDED_TOOL_RELEVANCE_THRESHOLD = 0.15

# Per-tool exclusions, keyed by LLM API id: ``{api_id: [tool_name, ...]}``.
# Subtractive by design — an api_id absent from the map exposes *all* of that
# API's tools, so the default (an absent key) is exactly the pre-feature
# behavior and a tool an MCP server adds later is available without the user
# having to revisit the form. There is deliberately no RECOMMENDED_ default:
# "unset" is the only spelling of "exclude nothing", so seeding DEFAULT_OPTIONS
# with an empty map would create a second one. See ``filter_excluded_tools`` in
# agent/helpers.py for the enforcement point.
CONF_TOOL_EXCLUSIONS = "tool_exclusions"

# Per-tool inclusions ("always-included tools"), same ``{api_id: [tool_name]}``
# shape as CONF_TOOL_EXCLUSIONS. Additive counterpart to the exclusions:
# every listed tool is appended to the model's tool list after vector-based
# retrieval, outside the retrieval limit, so general-purpose tools (a web
# search MCP tool, say) stay available on queries whose embedding never ranks
# them (issue #579). Absent key = include nothing extra, which is the
# pre-feature behavior; no RECOMMENDED_ default for the same single-spelling
# reason as the exclusions. An exclusion always beats an inclusion — the
# exclusion is a security control and must fail closed. Enforcement point:
# the always-included step in ``_retrieve_tools`` (agent/graph.py).
CONF_TOOL_INCLUSIONS = "tool_inclusions"

# Hard per-turn cap on honoured inclusions. The form naturally keeps the list
# short, but options can be written programmatically (bypassing the picker),
# and each inclusion costs a sequential store lookup plus its schema's prompt
# tokens on EVERY turn — an uncapped degenerate map turns each conversation
# turn into thousands of DB round-trips and an oversized tool list the
# provider rejects outright. Sixteen is far above any sane configuration
# (the docs recommend one or two) while keeping the worst case harmless.
TOOL_INCLUSIONS_MAX_PER_TURN = 16

# Actuation Safety Net: Keywords that trigger force-attachment of control tools.
# Derivatives that must stay in sync when verbs are added here:
# NON_OPEN_ACTUATION_KEYWORDS_REGEX, AUTOMATION_ACTION_KEYWORDS_REGEX.
ACTUATION_KEYWORDS_REGEX = (
    r"(?i)\b(turn|switch|lock|unlock|open|close|set|activate|deactivate|arm|"
    r"disarm|start|stop|dim|brighten|play|pause|mute|run|trigger|enable|"
    r"disable|toggle)\b"
)

# Read-only state query signals: used to suppress actuation safety injection
# when "open" appears as a state description rather than a command.
READ_ONLY_STATE_QUERY_REGEX = (
    r"(?i)\b(list|show|which|what|where|are|is|status|state)\b"
)

# "open" used as a state description (not a command), no noun restriction.
# Must be combined with READ_ONLY_STATE_QUERY_REGEX to avoid false positives
# on pure actuation commands like "open the garage door".
OPEN_AS_STATE_REGEX = r"(?i)\b(open|opened)\b"

# Actuation commands other than "open/opened". If any of these are present,
# the query should still get actuation safety tools even if it also contains
# a read-only phrase (e.g. "show me open windows and then close them").
# Must stay in sync with ACTUATION_KEYWORDS_REGEX — omits open/opened only.
NON_OPEN_ACTUATION_KEYWORDS_REGEX = (
    r"(?i)\b(turn|switch|lock|unlock|close|set|activate|deactivate|arm|"
    r"disarm|start|stop|dim|brighten|play|pause|mute|run|trigger|enable|"
    r"disable|toggle)\b"
)

# Compound state-then-command forms where "open" is used as a command verb
# despite the query also containing read-only state language.
# Covers: "show me open windows and then open the garage door",
#         "list open windows and open the garage door" (via article anchor).
# Known gap: comma- or period-separated compound forms are not detected.
OPEN_COMMAND_CLAUSE_REGEX = (
    r"(?i)\b(?:then|and then|after that)\s+open\b"
    r"|\band\s+open\s+(?:the|a|an)\b"
)

# Automation-creation intent signals: used to force-bind the add_automation
# tool during retrieval. Natural automation requests ("always turn on X when
# Y") share almost no vocabulary with add_automation's instruction-heavy tool
# description, so embedding similarity alone cannot guarantee selection — the
# actuation fragments of the query rank entity-control tools higher.
# Two signal classes, combined in graph._query_wants_automation:
# 1. Standalone markers: explicit automation vocabulary or recurring
#    schedules. Sufficient on their own.
AUTOMATION_INTENT_MARKERS_REGEX = (
    r"(?i)\b(?:automat(?:ions?|es?|ed|ically)|"
    r"whenever|every\s+time|each\s+time|any\s*time|"
    r"remind\s+me|let\s+me\s+know|on\s+a\s+schedule|"
    r"(?:every|each)\s+\d+\s*(?:seconds?|minutes?|hours?|days?|weeks?)|"
    r"(?:every|each)\s+(?:day|morning|afternoon|evening|night|week|"
    r"weekday|weekend|month)|"
    r"daily|nightly|hourly|weekly|monthly)\b"
)
# 2. Conditional actuation: an action verb combined with a trigger clause
#    ("turn on the porch light when motion is detected"). Both regexes below
#    must match. Over-matching is acceptable — the tool is appended after
#    RAG/safety selection without evicting anything, so a false positive
#    costs one unused tool slot, while a false negative silently breaks
#    automation creation.
AUTOMATION_TRIGGER_CLAUSE_REGEX = r"(?i)\b(?:when|whenever|always|if)\b"
# Action verbs for the conditional-actuation signal. Must stay in sync with
# ACTUATION_KEYWORDS_REGEX — adds notification verbs only (automations
# commonly notify rather than actuate: "tell me when the door opens").
# All automation-intent patterns are English-only; non-English requests fall
# back to plain RAG ranking (localized markers are tracked as future scope).
AUTOMATION_ACTION_KEYWORDS_REGEX = (
    r"(?i)\b(turn|switch|lock|unlock|open|close|set|activate|deactivate|arm|"
    r"disarm|start|stop|dim|brighten|play|pause|mute|run|trigger|enable|"
    r"disable|toggle|notify|alert|send|announce|broadcast|remind|flash|"
    r"tell|text|warn)\b"
)

# Tool prefixes/names for actuation safety net
ACTUATION_TOOL_PREFIXES = (
    "HassTurn",
    "HassLight",
    "HassLock",
    "HassCover",
    "HassClimate",
    "HassVacuum",
    "HassMedia",
    "HassValve",
    "HassFan",
    "HassWaterHeater",
)

ACTUATION_LANGCHAIN_TOOLS = ("alarm_control",)

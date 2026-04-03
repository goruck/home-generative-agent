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
SUBENTRY_TYPE_TOOL_MANAGER = "tool_manager"

HTTP_STATUS_UNAUTHORIZED = 401
HTTP_STATUS_BAD_REQUEST = 400
HTTP_STATUS_WEBPAGE_NOT_FOUND = 404
HTTP_STATUS_OK = 200
HTTP_STATUS_REQUEST_TOO_LARGE = 413

# ---- Critical action guard ----
CONF_CRITICAL_ACTION_PIN_ENABLED = "critical_action_pin_enabled"
CONF_CRITICAL_ACTION_PIN = "critical_action_pin"
CONF_CRITICAL_ACTION_PIN_HASH = "critical_action_pin_hash"
CONF_CRITICAL_ACTION_PIN_SALT = "critical_action_pin_salt"
CONF_CRITICAL_ACTIONS = "critical_actions"
RECOMMENDED_CRITICAL_ACTIONS: list[dict[str, str]] = [
    {"domain": "lock", "service": "unlock"},
    {"domain": "lock", "service": "open"},
    # Covers: guard only doors/gates/garages, not windows/shades
    {"domain": "cover", "service": "open_cover", "entity_match": "door"},
    {"domain": "cover", "service": "open_cover", "entity_match": "gate"},
    {"domain": "cover", "service": "open_cover", "entity_match": "garage"},
    {"domain": "cover", "service": "open", "entity_match": "door"},
    {"domain": "cover", "service": "open", "entity_match": "gate"},
    {"domain": "cover", "service": "open", "entity_match": "garage"},
    {"domain": "garage_door", "service": "open"},
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
LLM_HASS_API_NONE = "none"

# ---- LangChain logging ----
# See https://python.langchain.com/docs/how_to/debugging/
LANGCHAIN_LOGGING_LEVEL: Literal["disable", "verbose", "debug"] = "disable"

# RunnableConfig["configurable"]: HA tool IntentResponse sidecar (ChatLog / pipeline).
GRAPH_CFG_HA_TOOL_INTENT_RESPONSES = "ha_tool_intent_responses"


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
OLLAMA_CATEGORY_URL_KEYS = {
    "chat": CONF_OLLAMA_CHAT_URL,
    "vlm": CONF_OLLAMA_VLM_URL,
    "summarization": CONF_OLLAMA_SUMMARIZATION_URL,
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
CONF_SCHEMA_FIRST_YAML = "schema_first_yaml"
CONF_DEBUG_ASSIST_TRACE = "debug_assist_trace"
CONF_DISABLED_FEATURES = "disabled_features"

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
RECOMMENDED_SENTINEL_QUIET_HOURS_SEVERITIES: list[str] = ["low"]

CONF_SENTINEL_PRESENCE_GRACE_MINUTES = "sentinel_presence_grace_minutes"
RECOMMENDED_SENTINEL_PRESENCE_GRACE_MINUTES: int = 10

# ---- Sentinel notification routing (Issue #261) ----
# Maps area name -> notify service, e.g. {"bedroom": "notify.mobile_app_alice"}
CONF_SENTINEL_AREA_NOTIFY_MAP = "sentinel_area_notify_map"

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

# ---- Sentinel daily digest notification ----
CONF_SENTINEL_DAILY_DIGEST_ENABLED = "sentinel_daily_digest_enabled"
CONF_SENTINEL_DAILY_DIGEST_TIME = "sentinel_daily_digest_time"
RECOMMENDED_SENTINEL_DAILY_DIGEST_ENABLED: bool = False
RECOMMENDED_SENTINEL_DAILY_DIGEST_TIME: str = "08:00"

# ---- Feature definitions ----
DEFAULT_FEATURE_TYPES: tuple[str, ...] = (
    "conversation",
    "camera_image_analysis",
    "conversation_summary",
)

FEATURE_DEFS: dict[str, dict[str, Any]] = {
    "conversation": {"name": "Conversation", "required": True},
    "camera_image_analysis": {"name": "Camera Image Analysis", "required": False},
    "conversation_summary": {"name": "Conversation Summary", "required": False},
}

FEATURE_NAMES: dict[str, str] = {
    key: value["name"] for key, value in FEATURE_DEFS.items()
}

FEATURE_CATEGORY_MAP: dict[str, str] = {
    "conversation": "chat",
    "camera_image_analysis": "vlm",
    "conversation_summary": "summarization",
}

# ---- Feature model config (per-feature subentry) ----
CONF_FEATURE_MODEL = "model"
CONF_FEATURE_MODEL_NAME = "model_name"
CONF_FEATURE_MODEL_TEMPERATURE = "temperature"
CONF_FEATURE_MODEL_REASONING = "reasoning"
CONF_FEATURE_MODEL_KEEPALIVE = "keepalive_s"
CONF_FEATURE_MODEL_CONTEXT_SIZE = "context_size"

# --- Gemini API key (used in config_flow/__init__.py) ---
CONF_GEMINI_API_KEY = "gemini_api_key"

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
    "gemini-2.5-pro", "gemini-2.5-flash", "gemini-2.5-flash-lite"
]

CONF_CHAT_MODEL_PROVIDER = "chat_model_provider"
PROVIDERS = Literal["openai", "openai_compatible", "ollama", "gemini"]
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
RECOMMENDED_GEMINI_CHAT_MODEL: CHAT_MODEL_GEMINI_SUPPORTED = "gemini-2.5-flash-lite"

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
VLM_OLLAMA_SUPPORTED = Literal["qwen2.5vl:7b", "qwen3-vl:8b"]
VLM_OPENAI_SUPPORTED = Literal["gpt-5-nano", "gpt-4.1", "gpt-4.1-nano"]
VLM_GEMINI_SUPPORTED = Literal[
    "gemini-2.5-pro", "gemini-2.5-flash", "gemini-2.5-flash-lite"
]

CONF_VLM_PROVIDER = "vlm_provider"
RECOMMENDED_VLM_PROVIDER: Literal["openai", "ollama", "gemini"] = "ollama"

CONF_OLLAMA_VLM = "ollama_vlm"
RECOMMENDED_OLLAMA_VLM: VLM_OLLAMA_SUPPORTED = "qwen3-vl:8b"
CONF_OLLAMA_VLM_KEEPALIVE = "ollama_vlm_keepalive"
RECOMMENDED_OLLAMA_VLM_KEEPALIVE: KeepAliveSeconds = 300
CONF_OLLAMA_VLM_CONTEXT_SIZE = "ollama_vlm_context_size"
VLM_NUM_PREDICT = -2  # Ollama only, -2 = fill context
VLM_REPEAT_PENALTY = 1.05  # Ollama only
VLM_MIRO_STAT = 0  # Ollama only

CONF_OPENAI_VLM = "openai_vlm"
RECOMMENDED_OPENAI_VLM: VLM_OPENAI_SUPPORTED = "gpt-5-nano"

CONF_OPENAI_COMPATIBLE_VLM = "openai_compatible_vlm"
RECOMMENDED_OPENAI_COMPATIBLE_VLM = "gpt-4o"

CONF_GEMINI_VLM = "gemini_vlm"
RECOMMENDED_GEMINI_VLM: VLM_GEMINI_SUPPORTED = "gemini-2.5-flash-lite"

CONF_VLM_TEMPERATURE = "vlm_temperature"
RECOMMENDED_VLM_TEMPERATURE = 0.2

CONF_VLM_CAPABILITY = "vlm_capability"
VLM_CAPABILITY_BASIC = "basic"
VLM_CAPABILITY_STANDARD = "standard"
VLM_CAPABILITY_ADVANCED = "advanced"
RECOMMENDED_VLM_CAPABILITY: Literal["basic", "standard", "advanced"] = (
    VLM_CAPABILITY_ADVANCED
)

# Standard profile: appended to user text when analysis_prompt is used.
VLM_STANDARD_LENGTH_CONSTRAINT = (
    "\n\nCRITICAL: Answer the user's prompt in 1 to 2 short, factual sentences. "
    "Do not elaborate."
)

# Advanced profile: separate "where to look" from "what to do" for capable VLMs.
VLM_ADVANCED_OBJECTS_TASK_TEMPLATE = "OBJECTS TO LOCATE: {objects}\nTASK: {task}"

_VLM_CAPABILITY_AGENT_CONTEXT: dict[str, str] = {
    VLM_CAPABILITY_BASIC: (
        "<vlm_capability_profile>\n"
        "VLM profile: Basic. For `get_and_analyze_camera_image`, use only "
        "`detection_keywords`; `analysis_prompt` is ignored by the vision model.\n"
        "</vlm_capability_profile>"
    ),
    VLM_CAPABILITY_STANDARD: (
        "<vlm_capability_profile>\n"
        "VLM profile: Standard. You may pass `analysis_prompt` for focused "
        "instructions; the vision model is asked for 1-2 short sentences. "
        "You can combine with `detection_keywords`.\n"
        "</vlm_capability_profile>"
    ),
    VLM_CAPABILITY_ADVANCED: (
        "<vlm_capability_profile>\n"
        "VLM profile: Advanced. You may pass a free-form `analysis_prompt` and/or "
        "`detection_keywords`. When both are set, the vision prompt uses "
        "OBJECTS TO LOCATE vs TASK formatting.\n"
        "</vlm_capability_profile>"
    ),
}


def vlm_capability_agent_context_append(capability: str) -> str:
    """Append a short hint so the chat model respects the active VLM profile."""
    return _VLM_CAPABILITY_AGENT_CONTEXT.get(
        capability, _VLM_CAPABILITY_AGENT_CONTEXT[RECOMMENDED_VLM_CAPABILITY]
    )


# Prompts + input image size
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
    "gemini-2.5-pro", "gemini-2.5-flash", "gemini-2.5-flash-lite"
]

CONF_SUMMARIZATION_MODEL_PROVIDER = "summarization_provider"
RECOMMENDED_SUMMARIZATION_MODEL_PROVIDER: Literal["openai", "ollama", "gemini"] = (
    "ollama"
)

CONF_OLLAMA_SUMMARIZATION_MODEL = "ollama_summarization_model"
RECOMMENDED_OLLAMA_SUMMARIZATION_MODEL: SUMMARIZATION_MODEL_OLLAMA_SUPPORTED = (
    "qwen3:8b"
)
CONF_OLLAMA_SUMMARIZATION_KEEPALIVE = "ollama_summarization_keepalive"
RECOMMENDED_OLLAMA_SUMMARIZATION_KEEPALIVE: KeepAliveSeconds = 300
CONF_OLLAMA_SUMMARIZATION_CONTEXT_SIZE = "ollama_summarization_context_size"
SUMMARIZATION_MODEL_PREDICT = -2  # Ollama only, -2 = fill context
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
    "gemini-2.5-flash-lite"
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

CONF_GEMINI_EMBEDDING_MODEL = "gemini_embedding_model"
RECOMMENDED_GEMINI_EMBEDDING_MODEL: EMBEDDING_MODEL_GEMINI_SUPPORTED = (
    "gemini-embedding-001"
)

EMBEDDING_MODEL_DIMS = 1024
EMBEDDING_MODEL_CTX = 512
EMBEDDING_MODEL_PROMPT_TEMPLATE = """
Represent this sentence for searching relevant passages: {query}
"""

# ---------------- OpenAI-compatible endpoint (edge) ----------------
CONF_OPENAI_COMPATIBLE_BASE_URL = "openai_compatible_base_url"
CONF_OPENAI_COMPATIBLE_API_KEY = "openai_compatible_api_key"

# ---------------- Camera video analyzer ----------------
CONF_VIDEO_ANALYZER_MODE = "video_analyzer_mode"
VideoAnalyzerMode = Literal["disable", "notify_on_anomaly", "always_notify"]
VIDEO_ANALYZER_MODE_DISABLE: VideoAnalyzerMode = "disable"
VIDEO_ANALYZER_MODE_NOTIFY_ON_ANOMALY: VideoAnalyzerMode = "notify_on_anomaly"
VIDEO_ANALYZER_MODE_ALWAYS_NOTIFY: VideoAnalyzerMode = "always_notify"
RECOMMENDED_VIDEO_ANALYZER_MODE: VideoAnalyzerMode = VIDEO_ANALYZER_MODE_DISABLE

# Interval units are seconds.
VIDEO_ANALYZER_SCAN_INTERVAL = 1.5
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

Counts:
- Default to ONE unknown person across separate frames.
- Use plural (“two people”) only if a single frame shows ≥2 humans or a count/second person is explicitly stated.

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
"""  # noqa: E501
VIDEO_ANALYZER_PROMPT = """
Write ≤150 characters (≤2 sentences). Obey all rules and narrate in order.
"""
VIDEO_ANALYZER_TIME_OFFSET = 15  # minutes
VIDEO_ANALYZER_SIMILARITY_THRESHOLD = 0.89
VIDEO_ANALYZER_DELETE_SNAPSHOTS = False
VIDEO_ANALYZER_SNAPSHOTS_TO_KEEP = 200
VIDEO_ANALYZER_TRIGGER_ON_MOTION = True
VIDEO_ANALYZER_MOTION_CAMERA_MAP: dict = {}
VIDEO_ANALYZER_FACE_CROP = False

# Stable “latest” file publication
VIDEO_ANALYZER_SAVE_LATEST = True
VIDEO_ANALYZER_LATEST_NAME = "latest.jpg"
VIDEO_ANALYZER_LATEST_SUBFOLDER = "_latest"

# Dispatcher signals
SIGNAL_HGA_NEW_LATEST = "hga_new_latest"
SIGNAL_HGA_RECOGNIZED = "hga_recognized_people"
SIGNAL_SENTINEL_RUN_COMPLETE = "hga_sentinel_run_complete"

# ---------------- Face recognition ----------------
CONF_FACE_RECOGNITION = "face_recognition"
RECOMMENDED_FACE_RECOGNITION: bool = False

CONF_FACE_API_URL = "face_api_url"
RECOMMENDED_FACE_API_URL = "http://face-recog-server.local:8000"


# ---------------- Tools ----------------
CONF_TOOL_RETRIEVAL_LIMIT = "tool_retrieval_limit"
RECOMMENDED_TOOL_RETRIEVAL_LIMIT = 5

CONF_TOOL_RELEVANCE_THRESHOLD = "tool_relevance_threshold"
RECOMMENDED_TOOL_RELEVANCE_THRESHOLD = (
    0.5  # Lower default for better variety, users can tune up
)

CONF_INSTRUCTIONS_CONFIG = "instructions"
CONF_INSTRUCTION_RETRIEVAL_LIMIT = "instruction_retrieval_limit"
RECOMMENDED_INSTRUCTION_RETRIEVAL_LIMIT = 5

CONF_INSTRUCTION_RELEVANCE_THRESHOLD = "instruction_relevance_threshold"
RECOMMENDED_INSTRUCTION_RELEVANCE_THRESHOLD = 0.5

CONF_INSTRUCTION_RAG_INTENT_WEIGHT = "instruction_rag_intent_weight"
RECOMMENDED_INSTRUCTION_RAG_INTENT_WEIGHT = 0.65

RECOMMENDED_INSTRUCTION_RAG_NOISE_FLOOR = 0.25

TOOL_CALL_ERROR_SYSTEM_MESSAGE = """

Always call tools again with your mistakes corrected. Do not repeat mistakes.
"""
TOOL_CALL_ERROR_TEMPLATE = """
Error: {error}

Call the tool again with your mistake corrected.
"""
CRITICAL_ACTION_PROMPT = """
Critical actions (door/lock/garage/open) require user confirmation.
- If a tool response has status "requires_pin", ask the user for the PIN they set and
  then call the "confirm_sensitive_action" tool with the provided action_id and PIN.
- Never guess or invent a PIN. Do not proceed without a PIN. If the user refuses or
  fails, inform them and do not re-attempt the action.
- Do not expose or repeat the PIN in responses beyond acknowledging success/failure.
- Alarm control uses the alarm system code, not the critical-action PIN. When arming or
  disarming an alarm, ask for the alarm code and include it in the tool call. Do NOT
  call "confirm_sensitive_action" for alarm control.
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
        },
        "recommended_models": {
            "openai": RECOMMENDED_OPENAI_CHAT_MODEL,
            "openai_compatible": RECOMMENDED_OPENAI_COMPATIBLE_CHAT_MODEL,
            "ollama": RECOMMENDED_OLLAMA_CHAT_MODEL,
            "gemini": RECOMMENDED_GEMINI_CHAT_MODEL,
        },
        "model_keys": {
            "openai": CONF_OPENAI_CHAT_MODEL,
            "openai_compatible": CONF_OPENAI_COMPATIBLE_CHAT_MODEL,
            "ollama": CONF_OLLAMA_CHAT_MODEL,
            "gemini": CONF_GEMINI_CHAT_MODEL,
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
        },
        "recommended_models": {
            "openai": RECOMMENDED_OPENAI_VLM,
            "openai_compatible": RECOMMENDED_OPENAI_COMPATIBLE_VLM,
            "ollama": RECOMMENDED_OLLAMA_VLM,
            "gemini": RECOMMENDED_GEMINI_VLM,
        },
        "model_keys": {
            "openai": CONF_OPENAI_VLM,
            "openai_compatible": CONF_OPENAI_COMPATIBLE_VLM,
            "ollama": CONF_OLLAMA_VLM,
            "gemini": CONF_GEMINI_VLM,
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
        },
        "recommended_models": {
            "openai": RECOMMENDED_OPENAI_SUMMARIZATION_MODEL,
            "openai_compatible": RECOMMENDED_OPENAI_COMPATIBLE_SUMMARIZATION_MODEL,
            "ollama": RECOMMENDED_OLLAMA_SUMMARIZATION_MODEL,
            "gemini": RECOMMENDED_GEMINI_SUMMARIZATION_MODEL,
        },
        "model_keys": {
            "openai": CONF_OPENAI_SUMMARIZATION_MODEL,
            "openai_compatible": CONF_OPENAI_COMPATIBLE_SUMMARIZATION_MODEL,
            "ollama": CONF_OLLAMA_SUMMARIZATION_MODEL,
            "gemini": CONF_GEMINI_SUMMARIZATION_MODEL,
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

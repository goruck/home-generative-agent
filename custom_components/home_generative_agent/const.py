"""Constants for Home Generative Agent."""

from typing import Any, Literal, get_args

DOMAIN = "home_generative_agent"

HTTP_STATUS_UNAUTHORIZED = 401
HTTP_STATUS_BAD_REQUEST = 400

# ---- PostgreSQL (vector store + checkpointer) ----
DB_URI = "postgresql://ha_user:ha_passwd@localhost:5432/ha_db?sslmode=disable"
CONF_DB_BOOTSTRAPPED = "db_bootstrapped"

# ---- LangChain logging ----
# See https://python.langchain.com/docs/how_to/debugging/
LANGCHAIN_LOGGING_LEVEL: Literal["disable", "verbose", "debug"] = "disable"

# --- Reasoning delimiters ---
# These delimiters are used to mark the start and end of reasoning blocks in the model's
# responses.
# These may be model dependent, the defaults work for qwen3.
REASONING_DELIMITERS: dict[str, str] = {
    "start": "<think>",
    "end": "</think>",
}

# ---- Ollama ----
CONF_OLLAMA_URL = "ollama_url"
RECOMMENDED_OLLAMA_URL = "http://localhost:11434"

# ---- Global options ----
CONF_RECOMMENDED = "recommended"
CONF_PROMPT = "prompt"

# ---------------- Chat model ----------------
CHAT_MODEL_TOP_P = 1.0
CHAT_MODEL_NUM_CTX = 65536
CHAT_MODEL_MAX_TOKENS = 2048
# Add more models by extending the Literal types.
CHAT_MODEL_OLLAMA_SUPPORTED = Literal["gpt-oss", "qwen2.5:32b", "qwen3:32b", "qwen3:8b"]
CHAT_MODEL_OPENAI_SUPPORTED = Literal["gpt-4o", "gpt-4.1", "o4-mini"]

CONF_CHAT_MODEL_PROVIDER = "chat_model_provider"
RECOMMENDED_CHAT_MODEL_PROVIDER: Literal["openai", "ollama"] = "ollama"

CONF_OLLAMA_CHAT_MODEL = "ollama_chat_model"
RECOMMENDED_OLLAMA_CHAT_MODEL: CHAT_MODEL_OLLAMA_SUPPORTED = "qwen3:8b"

CONF_OPENAI_CHAT_MODEL = "openai_chat_model"
RECOMMENDED_OPENAI_CHAT_MODEL: CHAT_MODEL_OPENAI_SUPPORTED = "gpt-4o"

CONF_CHAT_MODEL_TEMPERATURE = "chat_model_temperature"
RECOMMENDED_CHAT_MODEL_TEMPERATURE = 1.0

# Context management (for trimming history)
CONTEXT_MANAGE_USE_TOKENS = True
CONTEXT_MAX_MESSAGES = 80
# Keep buffer for tools + token counter undercount (see repo notes).
CONTEXT_MAX_TOKENS = CHAT_MODEL_NUM_CTX - CHAT_MODEL_MAX_TOKENS - 2048 - 4096  # 57344

# ---------------- VLM (vision) ----------------
VLM_TOP_P = 0.6
VLM_NUM_PREDICT = 4096
VLM_NUM_CTX = 16384
VLM_OLLAMA_SUPPORTED = Literal["qwen2.5vl:7b"]
VLM_OPENAI_SUPPORTED = Literal["gpt-4.1", "gpt-4.1-nano"]

CONF_VLM_PROVIDER = "vlm_provider"
RECOMMENDED_VLM_PROVIDER: Literal["openai", "ollama"] = "ollama"

CONF_OLLAMA_VLM = "ollama_vlm"
RECOMMENDED_OLLAMA_VLM: VLM_OLLAMA_SUPPORTED = "qwen2.5vl:7b"

CONF_OPENAI_VLM = "openai_vlm"
RECOMMENDED_OPENAI_VLM: VLM_OPENAI_SUPPORTED = "gpt-4.1-nano"

CONF_VLM_TEMPERATURE = "vlm_temperature"
RECOMMENDED_VLM_TEMPERATURE = 0.1

# Prompts + input image size
VLM_SYSTEM_PROMPT = """
You are a bot that responses with a description of what is visible in a camera image.

Keep your responses simple and to the point.
"""
VLM_USER_PROMPT = "Task: Describe this image:"
VLM_USER_KW_TEMPLATE = """
Task: Tell me if {key_words} are visible in this image:
"""
VLM_IMAGE_WIDTH = 1920
VLM_IMAGE_HEIGHT = 1080

# ---------------- Summarization ----------------
SUMMARIZATION_MODEL_TOP_P = 0.95
SUMMARIZATION_MODEL_PREDICT = 4096
SUMMARIZATION_MODEL_CTX = 32768
SUMMARIZATION_MODEL_OLLAMA_SUPPORTED = Literal["qwen3:1.7b", "qwen3:8b"]
SUMMARIZATION_MODEL_OPENAI_SUPPORTED = Literal["gpt-4.1", "gpt-4.1-nano"]

CONF_SUMMARIZATION_MODEL_PROVIDER = "summarization_provider"
RECOMMENDED_SUMMARIZATION_MODEL_PROVIDER: Literal["openai", "ollama"] = "ollama"

CONF_OLLAMA_SUMMARIZATION_MODEL = "ollama_summarization_model"
RECOMMENDED_OLLAMA_SUMMARIZATION_MODEL: SUMMARIZATION_MODEL_OLLAMA_SUPPORTED = (
    "qwen3:1.7b"
)

CONF_OPENAI_SUMMARIZATION_MODEL = "openai_summarization_model"
RECOMMENDED_OPENAI_SUMMARIZATION_MODEL: SUMMARIZATION_MODEL_OPENAI_SUPPORTED = (
    "gpt-4.1-nano"
)

CONF_SUMMARIZATION_MODEL_TEMPERATURE = "summarization_model_temperature"
RECOMMENDED_SUMMARIZATION_MODEL_TEMPERATURE = 0.6

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

CONF_EMBEDDING_MODEL_PROVIDER = "embedding_model_provider"
RECOMMENDED_EMBEDDING_MODEL_PROVIDER: Literal["openai", "ollama"] = "ollama"

CONF_OLLAMA_EMBEDDING_MODEL = "ollama_embedding_model"
RECOMMENDED_OLLAMA_EMBEDDING_MODEL: EMBEDDING_MODEL_OLLAMA_SUPPORTED = (
    "mxbai-embed-large"
)

CONF_OPENAI_EMBEDDING_MODEL = "openai_embedding_model"
RECOMMENDED_OPENAI_EMBEDDING_MODEL: EMBEDDING_MODEL_OPENAI_SUPPORTED = (
    "text-embedding-3-small"
)

EMBEDDING_MODEL_DIMS = 1024
EMBEDDING_MODEL_CTX = 512
EMBEDDING_MODEL_PROMPT_TEMPLATE = """
Represent this sentence for searching relevant passages: {query}
"""

# ---------------- Camera video analyzer ----------------
CONF_VIDEO_ANALYZER_MODE = "video_analyzer_mode"
RECOMMENDED_VIDEO_ANALYZER_MODE: Literal[
    "disable", "notify_on_anomaly", "always_notify"
] = "disable"
CONF_NOTIFY_SERVICE = "notify_service"

# Interval units are seconds.
VIDEO_ANALYZER_SCAN_INTERVAL = 1.5
VIDEO_ANALYZER_SNAPSHOT_ROOT = "/media/snapshots"
VIDEO_ANALYZER_SYSTEM_MESSAGE = """
You are a bot that generates a description of a video given descriptions of its frames.
Keep the description to the point and use no more than 250 characters.
"""
VIDEO_ANALYZER_PROMPT = """
Describe what is happening in this video from these frame descriptions:
"""
# Time offset units are minutes.
VIDEO_ANALYZER_TIME_OFFSET = 15
VIDEO_ANALYZER_SIMILARITY_THRESHOLD = 0.89
VIDEO_ANALYZER_DELETE_SNAPSHOTS = False
VIDEO_ANALYZER_SNAPSHOTS_TO_KEEP = 15
VIDEO_ANALYZER_TRIGGER_ON_MOTION = True
VIDEO_ANALYZER_MOTION_CAMERA_MAP: dict = {}

# ---------------- Tools ----------------
TOOL_CALL_ERROR_SYSTEM_MESSAGE = """

Always call tools again with your mistakes corrected. Do not repeat mistakes.
"""
TOOL_CALL_ERROR_TEMPLATE = """
Error: {error}

Call the tool again with your mistake corrected.
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
            "ollama": list(get_args(CHAT_MODEL_OLLAMA_SUPPORTED)),
        },
        "recommended_models": {
            "openai": RECOMMENDED_OPENAI_CHAT_MODEL,
            "ollama": RECOMMENDED_OLLAMA_CHAT_MODEL,
        },
        "model_keys": {
            "openai": CONF_OPENAI_CHAT_MODEL,
            "ollama": CONF_OLLAMA_CHAT_MODEL,
        },
    },
    "vlm": {
        "provider_key": CONF_VLM_PROVIDER,
        "temperature_key": CONF_VLM_TEMPERATURE,
        "recommended_provider": RECOMMENDED_VLM_PROVIDER,
        "recommended_temperature": RECOMMENDED_VLM_TEMPERATURE,
        "providers": {
            "openai": list(get_args(VLM_OPENAI_SUPPORTED)),
            "ollama": list(get_args(VLM_OLLAMA_SUPPORTED)),
        },
        "recommended_models": {
            "openai": RECOMMENDED_OPENAI_VLM,
            "ollama": RECOMMENDED_OLLAMA_VLM,
        },
        "model_keys": {
            "openai": CONF_OPENAI_VLM,
            "ollama": CONF_OLLAMA_VLM,
        },
    },
    "summarization": {
        "provider_key": CONF_SUMMARIZATION_MODEL_PROVIDER,
        "temperature_key": CONF_SUMMARIZATION_MODEL_TEMPERATURE,
        "recommended_provider": RECOMMENDED_SUMMARIZATION_MODEL_PROVIDER,
        "recommended_temperature": RECOMMENDED_SUMMARIZATION_MODEL_TEMPERATURE,
        "providers": {
            "openai": list(get_args(SUMMARIZATION_MODEL_OPENAI_SUPPORTED)),
            "ollama": list(get_args(SUMMARIZATION_MODEL_OLLAMA_SUPPORTED)),
        },
        "recommended_models": {
            "openai": RECOMMENDED_OPENAI_SUMMARIZATION_MODEL,
            "ollama": RECOMMENDED_OLLAMA_SUMMARIZATION_MODEL,
        },
        "model_keys": {
            "openai": CONF_OPENAI_SUMMARIZATION_MODEL,
            "ollama": CONF_OLLAMA_SUMMARIZATION_MODEL,
        },
    },
    "embedding": {
        "provider_key": CONF_EMBEDDING_MODEL_PROVIDER,
        "temperature_key": None,  # embeddings dont use temperature
        "recommended_provider": RECOMMENDED_EMBEDDING_MODEL_PROVIDER,
        "recommended_temperature": None,
        "providers": {
            "openai": list(get_args(EMBEDDING_MODEL_OPENAI_SUPPORTED)),
            "ollama": list(get_args(EMBEDDING_MODEL_OLLAMA_SUPPORTED)),
        },
        "recommended_models": {
            "openai": RECOMMENDED_OPENAI_EMBEDDING_MODEL,
            "ollama": RECOMMENDED_OLLAMA_EMBEDDING_MODEL,
        },
        "model_keys": {
            "openai": CONF_OPENAI_EMBEDDING_MODEL,
            "ollama": CONF_OLLAMA_EMBEDDING_MODEL,
        },
    },
}

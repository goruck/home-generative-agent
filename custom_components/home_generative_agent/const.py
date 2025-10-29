"""Constants for Home Generative Agent."""

from typing import Any, Literal, get_args

DOMAIN = "home_generative_agent"

HTTP_STATUS_UNAUTHORIZED = 401
HTTP_STATUS_BAD_REQUEST = 400
HTTP_STATUS_WEBPAGE_NOT_FOUND = 404

# ---- PostgreSQL (vector store + checkpointer) ----
CONF_DB_URI = "db_uri"
RECOMMENDED_DB_URI = (
    "postgresql://ha_user:ha_password@localhost:5432/ha_db?sslmode=disable"
)
CONF_DB_BOOTSTRAPPED = "db_bootstrapped"

# ---- Notify service (for mobile push notifications) ----
CONF_NOTIFY_SERVICE = "notify_service"

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

# --- Gemini API key (used in config_flow/__init__.py) ---
CONF_GEMINI_API_KEY = "gemini_api_key"

# ---------------- Chat model ----------------
CHAT_MODEL_TOP_P = 1.0
CHAT_MODEL_NUM_CTX = 32768
CHAT_MODEL_MAX_TOKENS = 4096
CHAT_MODEL_REPEAT_PENALTY = 1.05
# Add more models by extending the Literal types.
CHAT_MODEL_OLLAMA_SUPPORTED = Literal["gpt-oss", "qwen2.5:32b", "qwen3:32b", "qwen3:8b"]
CHAT_MODEL_OPENAI_SUPPORTED = Literal[
    "gpt-5", "gpt-5-mini", "gpt-5-nano", "gpt-4o", "gpt-4.1", "o4-mini"
]
CHAT_MODEL_GEMINI_SUPPORTED = Literal[
    "gemini-2.5-pro", "gemini-2.5-flash", "gemini-2.5-flash-lite"
]

CONF_CHAT_MODEL_PROVIDER = "chat_model_provider"
PROVIDERS = Literal["openai", "ollama", "gemini"]
RECOMMENDED_CHAT_MODEL_PROVIDER: PROVIDERS = "ollama"

CONF_OLLAMA_CHAT_MODEL = "ollama_chat_model"
RECOMMENDED_OLLAMA_CHAT_MODEL: CHAT_MODEL_OLLAMA_SUPPORTED = "qwen3:8b"

CONF_OPENAI_CHAT_MODEL = "openai_chat_model"
RECOMMENDED_OPENAI_CHAT_MODEL: CHAT_MODEL_OPENAI_SUPPORTED = "gpt-5"

CONF_GEMINI_CHAT_MODEL = "gemini_chat_model"
RECOMMENDED_GEMINI_CHAT_MODEL: CHAT_MODEL_GEMINI_SUPPORTED = "gemini-2.5-flash-lite"

CONF_CHAT_MODEL_TEMPERATURE = "chat_model_temperature"
RECOMMENDED_CHAT_MODEL_TEMPERATURE = 1.0

# Context management (for trimming history)
CONTEXT_MANAGE_USE_TOKENS = True
CONTEXT_MAX_MESSAGES = 80
# Keep buffer for tools + token counter undercount (see repo notes).
CONTEXT_MAX_TOKENS = CHAT_MODEL_NUM_CTX - CHAT_MODEL_MAX_TOKENS - 2048 - 4096  # 26624

# ---------------- VLM (vision) ----------------
VLM_TOP_P = 1
VLM_NUM_PREDICT = 4096
VLM_NUM_CTX = 16384
VLM_REPEAT_PENALTY = 1.05
VLM_OLLAMA_SUPPORTED = Literal["qwen2.5vl:7b", "qwen3-vl:8b"]
VLM_OPENAI_SUPPORTED = Literal["gpt-5-nano", "gpt-4.1", "gpt-4.1-nano"]
VLM_GEMINI_SUPPORTED = Literal[
    "gemini-2.5-pro", "gemini-2.5-flash", "gemini-2.5-flash-lite"
]

CONF_VLM_PROVIDER = "vlm_provider"
RECOMMENDED_VLM_PROVIDER: Literal["openai", "ollama", "gemini"] = "ollama"

CONF_OLLAMA_VLM = "ollama_vlm"
RECOMMENDED_OLLAMA_VLM: VLM_OLLAMA_SUPPORTED = "qwen3-vl:8b"

CONF_OPENAI_VLM = "openai_vlm"
RECOMMENDED_OPENAI_VLM: VLM_OPENAI_SUPPORTED = "gpt-5-nano"

CONF_GEMINI_VLM = "gemini_vlm"
RECOMMENDED_GEMINI_VLM: VLM_GEMINI_SUPPORTED = "gemini-2.5-flash-lite"

CONF_VLM_TEMPERATURE = "vlm_temperature"
RECOMMENDED_VLM_TEMPERATURE = 0.0001

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
Return plain English text only.
"""
VLM_USER_KW_TEMPLATE = """
FRAME DESCRIPTION REQUEST (FOCUSED)

Primary attention: {key_words}
Describe this image clearly and factually in 1-3 sentences, focusing on the listed items if present.
Follow the style and rules from the system prompt.
Do not add names, timestamps, or speculation.
Return plain English text only.
"""  # noqa: E501
VLM_IMAGE_WIDTH = 1920
VLM_IMAGE_HEIGHT = 1080

# ---------------- Summarization ----------------
SUMMARIZATION_MODEL_TOP_P = 1
SUMMARIZATION_MODEL_PREDICT = 4096
SUMMARIZATION_MODEL_CTX = 32768
SUMMARIZATION_MODEL_REPEAT_PENALTY = 1.05
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
    "qwen3:1.7b"
)

CONF_OPENAI_SUMMARIZATION_MODEL = "openai_summarization_model"
RECOMMENDED_OPENAI_SUMMARIZATION_MODEL: SUMMARIZATION_MODEL_OPENAI_SUPPORTED = (
    "gpt-5-nano"
)

CONF_GEMINI_SUMMARIZATION_MODEL = "gemini_summarization_model"
RECOMMENDED_GEMINI_SUMMARIZATION_MODEL: SUMMARIZATION_MODEL_GEMINI_SUPPORTED = (
    "gemini-2.5-flash-lite"
)

CONF_SUMMARIZATION_MODEL_TEMPERATURE = "summarization_model_temperature"
RECOMMENDED_SUMMARIZATION_MODEL_TEMPERATURE = 0

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

CONF_GEMINI_EMBEDDING_MODEL = "gemini_embedding_model"
RECOMMENDED_GEMINI_EMBEDDING_MODEL: EMBEDDING_MODEL_GEMINI_SUPPORTED = (
    "gemini-embedding-001"
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

# ---------------- Face recognition ----------------
CONF_FACE_RECOGNITION_MODE = "face_recognition_mode"
RECOMMENDED_FACE_RECOGNITION_MODE: Literal["enable", "disable"] = "disable"

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
            "gemini": list(get_args(CHAT_MODEL_GEMINI_SUPPORTED)),
        },
        "recommended_models": {
            "openai": RECOMMENDED_OPENAI_CHAT_MODEL,
            "ollama": RECOMMENDED_OLLAMA_CHAT_MODEL,
            "gemini": RECOMMENDED_GEMINI_CHAT_MODEL,
        },
        "model_keys": {
            "openai": CONF_OPENAI_CHAT_MODEL,
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
            "ollama": list(get_args(VLM_OLLAMA_SUPPORTED)),
            "gemini": list(get_args(VLM_GEMINI_SUPPORTED)),
        },
        "recommended_models": {
            "openai": RECOMMENDED_OPENAI_VLM,
            "ollama": RECOMMENDED_OLLAMA_VLM,
            "gemini": RECOMMENDED_GEMINI_VLM,
        },
        "model_keys": {
            "openai": CONF_OPENAI_VLM,
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
            "ollama": list(get_args(SUMMARIZATION_MODEL_OLLAMA_SUPPORTED)),
            "gemini": list(get_args(SUMMARIZATION_MODEL_GEMINI_SUPPORTED)),
        },
        "recommended_models": {
            "openai": RECOMMENDED_OPENAI_SUMMARIZATION_MODEL,
            "ollama": RECOMMENDED_OLLAMA_SUMMARIZATION_MODEL,
            "gemini": RECOMMENDED_GEMINI_SUMMARIZATION_MODEL,
        },
        "model_keys": {
            "openai": CONF_OPENAI_SUMMARIZATION_MODEL,
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
            "ollama": list(get_args(EMBEDDING_MODEL_OLLAMA_SUPPORTED)),
            "gemini": list(get_args(EMBEDDING_MODEL_GEMINI_SUPPORTED)),
        },
        "recommended_models": {
            "openai": RECOMMENDED_OPENAI_EMBEDDING_MODEL,
            "ollama": RECOMMENDED_OLLAMA_EMBEDDING_MODEL,
            "gemini": RECOMMENDED_GEMINI_EMBEDDING_MODEL,
        },
        "model_keys": {
            "openai": CONF_OPENAI_EMBEDDING_MODEL,
            "ollama": CONF_OLLAMA_EMBEDDING_MODEL,
            "gemini": CONF_GEMINI_EMBEDDING_MODEL,
        },
    },
}

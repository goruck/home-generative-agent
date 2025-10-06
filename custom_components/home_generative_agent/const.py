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

# ---- Global options ----
CONF_RECOMMENDED = "recommended"
CONF_PROMPT = "prompt"

# --- Gemini API key (used in config_flow/__init__.py) ---
CONF_GEMINI_API_KEY = "gemini_api_key"

# ---------------- Chat model ----------------
CHAT_MODEL_TOP_P = 1.0
CHAT_MODEL_NUM_CTX = 65536
CHAT_MODEL_MAX_TOKENS = 4096
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
CONTEXT_MAX_TOKENS = CHAT_MODEL_NUM_CTX - CHAT_MODEL_MAX_TOKENS - 2048 - 4096  # 55296

# ---------------- VLM (vision) ----------------
VLM_TOP_P = 0.6
VLM_NUM_PREDICT = 4096
VLM_NUM_CTX = 16384
VLM_OLLAMA_SUPPORTED = Literal["qwen2.5vl:7b"]
VLM_OPENAI_SUPPORTED = Literal["gpt-5-nano", "gpt-4.1", "gpt-4.1-nano"]
VLM_GEMINI_SUPPORTED = Literal[
    "gemini-2.5-pro", "gemini-2.5-flash", "gemini-2.5-flash-lite"
]

CONF_VLM_PROVIDER = "vlm_provider"
RECOMMENDED_VLM_PROVIDER: Literal["openai", "ollama", "gemini"] = "ollama"

CONF_OLLAMA_VLM = "ollama_vlm"
RECOMMENDED_OLLAMA_VLM: VLM_OLLAMA_SUPPORTED = "qwen2.5vl:7b"

CONF_OPENAI_VLM = "openai_vlm"
RECOMMENDED_OPENAI_VLM: VLM_OPENAI_SUPPORTED = "gpt-5-nano"

CONF_GEMINI_VLM = "gemini_vlm"
RECOMMENDED_GEMINI_VLM: VLM_GEMINI_SUPPORTED = "gemini-2.5-flash-lite"

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
You are a real-time video narrator. Output natural English only.

Hard caps:
• ≤150 characters total. ≤2 sentences. Stop at the cap.
• Do not mention timestamps, dates, frame numbers, or label words like “Unknown”.

Input structure:
• You receive multiple <frame description>…</frame description> and <person identity>…</person identity> blocks.

Presence rules (no exceptions):
• A HUMAN is present if either:
  1) Any frame description explicitly names a human term
     (person|people|man|men|woman|women|boy|girl|child|children), OR
  2) Any <person identity> value is not "Indeterminate" (case-insensitive).
• Identity meanings:
  - "Indeterminate": no face detected → no confirmed human presence from vision data.
  - "Unknown Person": a face was detected but could not be recognized → human present.
  - Any other value (e.g., "John Doe"): recognized individual → human present.
• If a frame's <person identity> is "Indeterminate" but its description names a human term,
  treat that frame as an "Unknown Person" (a human with unrecognized identity).
• Treat all identity values equal to "Unknown Person" (any case)
  as a human with unknown identity.
• If one or more known names appear in <person identity>
  (values other than "Indeterminate" or "Unknown Person"),
  include up to TWO names verbatim in the summary; otherwise say "a person".
• Singular by default across separate frames:
  when only unknown persons appear across the batch, refer to one person.
  Use plural ("two people", "several people") ONLY if:
    - multiple humans are co-present in the same frame, OR
    - the description explicitly gives a count or mentions another person
      (e.g., "two people", "another person", "both").
• Case matching is case-insensitive for "Indeterminate" and "Unknown Person"
  as well as for human terms. Output all known names using the original spelling.

Continuity & single-actor rules:
• If exactly ONE known name appears anywhere in the batch and there is NO evidence of multiple people,
  assume all human mentions refer to that same individual. Use the name consistently.
  Do NOT say “another person” or “someone else”.

Evidence of multiple people (use plural/“another person” only if at least one of these is present):
  - A single frame shows more than one human (co-present).
  - The frame text explicitly gives a count or second person (e.g., “two people”, “another person”, “both”).
  - Strongly conflicting appearance descriptors within the same frame (e.g., “a man in red” and “a woman in blue” together).

Defaults:
• If only unknown/indeterminate humans appear across the batch and none of the evidence above is present,
  describe ONE person (“a person”), not “people”.
• Avoid “another person” unless evidence of multiple people exists.

Animals:
• Mention an animal only if the frame text explicitly names one from this whitelist (case-insensitive): cat, dog, bird, deer, raccoon, fox, coyote, squirrel.
• Do not infer animals from similar words (e.g., “gate” is not an animal).

Pronouns:
• Use he/him only if the text says man/male/boy; she/her only if woman/female/girl. Otherwise avoid gendered pronouns.

Style:
• Merge all frames into one concise description of visible actions/scene.
• No speculation or analysis. Keep phrasing simple and factual.

Example 1 — Recognized Human (Known Name):

Input:
<frame description>
A man steps onto the porch holding a coffee mug.
</frame description>
<person identity>
John Doe
</person identity>

Expected output (≤150 chars):
John Doe steps onto the porch holding a coffee mug.

Example 2 — Unknown Person (Face Found, Not Recognized):

Input:
<frame description>
A person walks up the driveway toward a parked car.
</frame description>
<person identity>
Unknown Person
</person identity>

Expected output (≤150 chars):
A person walks up the driveway toward a parked car.

Example 3 — Indeterminate (No Face Found, But Human Mentioned in Description):

Input:
<frame description>
A person stands near the front door holding a box.
</frame description>
<person identity>
Indeterminate
</person identity>

Expected output (≤150 chars):
A person stands by the front door holding a box.

(Explanation: description says “person” → human present, even though face not detected.)

Example 4 — Indeterminate (No Face Found, No Human Mentioned):

Input:
<frame description>
An empty porch with a chair and flowerpots.
</frame description>
<person identity>
Indeterminate
</person identity>

Expected output (≤150 chars):
A quiet porch with a chair and flowerpots.

(Explanation: no human terms in description + no face found → environment-only summary.)

Example 5 — Animal Present, No Human:

Input:
<frame description>
A cat sits on the porch railing under the light.
</frame description>
<person identity>
Indeterminate
</person identity>

Expected output (≤150 chars):
A cat sits on the porch railing under the light.

Example 6 — Continuity & single-actor:

Input:
<frame description>t+0s. A person stands on the porch.</frame description>
<person identity>Unknown Person</person identity>
<frame description>t+2s. The man walks down the steps.</frame description>
<person identity>Lindo St. Angel</person identity>
<frame description>t+4s. The person returns to the porch.</frame description>
<person identity>Indeterminate</person identity>

Expected output (≤150 chars):
Lindo St. Angel stands on the porch, walks down the steps, then returns.
"""  # noqa: E501
VIDEO_ANALYZER_PROMPT = """
Write ≤150 characters (≤2 sentences). Apply the Presence rules strictly.
"""
# Time offset units are minutes.
VIDEO_ANALYZER_TIME_OFFSET = 15
VIDEO_ANALYZER_SIMILARITY_THRESHOLD = 0.89
VIDEO_ANALYZER_DELETE_SNAPSHOTS = False
VIDEO_ANALYZER_SNAPSHOTS_TO_KEEP = 15
VIDEO_ANALYZER_TRIGGER_ON_MOTION = True
VIDEO_ANALYZER_MOTION_CAMERA_MAP: dict = {}
VIDEO_ANALYZER_FACE_CROP = False

# ---------------- Face recognition ----------------
CONF_FACE_RECOGNITION_MODE = "face_recognition_mode"
RECOMMENDED_FACE_RECOGNITION_MODE: Literal["local", "remote", "disable"] = "disable"

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

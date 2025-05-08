"""Constants for Home Generative Agent."""

from typing import Literal

DOMAIN = "home_generative_agent"

### Configuration parameters that can be overridden in the integration's config UI. ###
# Name of the set of recommended options.
CONF_RECOMMENDED = "recommended"
# Name of system prompt.
CONF_PROMPT = "prompt"
# Run chat model in cloud or at edge.
CONF_CHAT_MODEL_LOCATION = "chat_model_location"
RECOMMENDED_CHAT_MODEL_LOCATION: Literal["cloud", "edge"] = "edge"
### OpenAI chat model parameters.
# See https://platform.openai.com/docs/api-reference/chat/create.
CONF_CHAT_MODEL = "chat_model"
RECOMMENDED_CHAT_MODEL = "gpt-4o"
CONF_CHAT_MODEL_TEMPERATURE = "chat_model_temperature"
RECOMMENDED_CHAT_MODEL_TEMPERATURE = 1.0
### Ollama edge chat model parameters. ###
# See https://github.com/ollama/ollama/blob/main/docs/modelfile.md#parameter
CONF_EDGE_CHAT_MODEL = "edge_chat_model"
RECOMMENDED_EDGE_CHAT_MODEL = "qwen3:32b"
CONF_EDGE_CHAT_MODEL_TEMPERATURE = "edge_chat_model_temperature"
RECOMMENDED_EDGE_CHAT_MODEL_TEMPERATURE = 0.7
CONF_EDGE_CHAT_MODEL_TOP_P = "edge_chat_model_top_p"
RECOMMENDED_EDGE_CHAT_MODEL_TOP_P = 0.8
### Ollama vision language model (VLM) parameters. ###
# The VLM is used for vision and summarization tasks.
# See https://github.com/ollama/ollama/blob/main/docs/modelfile.md#parameter
CONF_VLM = "vlm"
RECOMMENDED_VLM = "llama3.2-vision"
CONF_VLM_TEMPERATURE = "vlm_temperature"
RECOMMENDED_VLM_TEMPERATURE = 0.2
CONF_VLM_TOP_P = "vlm_top_p"
RECOMMENDED_VLM_TOP_P = 0.5
CONF_SUMMARIZATION_MODEL = "summarization_model"
RECOMMENDED_SUMMARIZATION_MODEL = "qwen3:4b"
CONF_SUMMARIZATION_MODEL_TEMPERATURE = "summarization_model_temperature"
RECOMMENDED_SUMMARIZATION_MODEL_TEMPERATURE = 0.6
CONF_SUMMARIZATION_MODEL_TOP_P = "summarization_model_top_p"
RECOMMENDED_SUMMARIZATION_MODEL_TOP_P = 0.9
### Ollama embedding model parameters. ###
# The embedding model is used for semantic search in long-term memory.
CONF_EMBEDDING_MODEL = "embedding_model"
RECOMMENDED_EMBEDDING_MODEL = "mxbai-embed-large"
### Camera video analyzer parameters. ###
CONF_VIDEO_ANALYZER_MODE = "video_analyzer_mode"
RECOMMENDED_VIDEO_ANALYZER_MODE: Literal[
    "disable", "notify_on_anomaly", "always_notify"
] = "disable"

### langchain logging level ###
# See https://python.langchain.com/docs/how_to/debugging/
LANGCHAIN_LOGGING_LEVEL: Literal["disable", "verbose", "debug"] = "disable"

### Chat model context-related parameters. ###
# Sets the size of the context window used to generate the next token.
CHAT_MODEL_NUM_CTX = 32768
# Sets the maximum number of output tokens to generate.
CHAT_MODEL_MAX_TOKENS = 1024
# Next parameters manage chat model context length.
# CONTEXT_MANAGE_USE_TOKENS = True manages chat model context size via token
# counting, if False management is done via message counting.
CONTEXT_MANAGE_USE_TOKENS = True
# CONTEXT_MAX_MESSAGES is messages to keep in context before deletion.
# Keep number of tokens below 30k otherwise rate limits may be triggered by OpenAI
# (Tokens Per Minute limit for Tier 1 pricing is 30k tokens/minute), or Ollama model
# context length limits will be reached.
# Assume worse case message is 300 tokens -> 85 messages in context are ~25k tokens,
# which is consistent with CONTEXT_MAX_TOKENS.
CONTEXT_MAX_MESSAGES = 80
# CONTEXT_MAX_TOKENS sets the limit on how large the context can grow. This needs to
# take into account the output tokens.
#
# Reduce by 2k tokens because the token counter ignores tool schemas.
# Reduce by another 4k because the token counter under counts by as much as 4k tokens.
# These offsets are for the qwen models and were empirically determined.
# TODO: fix the token counter to get an accurate count.
#
CONTEXT_MAX_TOKENS = (CHAT_MODEL_NUM_CTX - CHAT_MODEL_MAX_TOKENS - 2048 - 4096) # 25600

### Chat model tool error handling parameters. ###
TOOL_CALL_ERROR_SYSTEM_MESSAGE = """

Always call tools again with your mistakes corrected. Do not repeat mistakes.
"""
TOOL_CALL_ERROR_TEMPLATE = """
Error: {error}

Call the tool again with your mistake corrected.
"""

### Ollama edge chat model parameters. ###
# Edge chat model server URL.
EDGE_CHAT_MODEL_URL = "192.168.1.252:11434"
# Reasoning delimiters for models that use them in output.
# These may be model dependent, the defaults work for qwen3.
EDGE_CHAT_MODEL_REASONING_DELIMITER: dict[str, str] = {
    "start": "<think>", "end": "</think>"
}

### Ollama VLM parameters. ###
# Ollama VLM server URL.
VLM_URL = "192.168.1.252:11434"
# Ollama VLM maximum number of output tokens to generate.
VLM_NUM_PREDICT = 4096
# Sets the size of the context window used to generate the next token.
VLM_NUM_CTX = 16384
# Ollama VLM model prompts for vision tasks.
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

### Ollama summarization model parameters. ###
# Model server URL.
SUMMARIZATION_MODEL_URL = "192.168.1.252:11434"
# Maximum number of tokens to predict when generating text.
SUMMARIZATION_MODEL_PREDICT = 4096
# Sets the size of the context window used to generate the next token.
SUMMARIZATION_MODEL_CTX = 32768
# Reasoning delimiters for models that use them in output.
# These may be model dependent, the defaults work for qwen3.
SUMMARIZATION_MODEL_REASONING_DELIMITER: dict[str, str] = {
    "start": "<think>", "end": "</think>"
}
# Model prompts for summary tasks.
SUMMARY_SYSTEM_PROMPT = "You are a bot that summarizes messages from a smart home AI."
SUMMARY_INITIAL_PROMPT = "Create a summary of the smart home messages above:"
SUMMARY_PROMPT_TEMPLATE = """
This is the summary of the smart home messages so far: {summary}

Update the summary by taking into account the additional smart home messages above:
"""

### Ollama embedding model parameters. ###
EMBEDDING_MODEL_URL = "192.168.1.252:11434"
EMBEDDING_MODEL_DIMS = 1024
EMBEDDING_MODEL_CTX = 512
EMBEDDING_MODEL_PROMPT_TEMPLATE = """
Represent this sentence for searching relevant passages: {query}
"""

### Tool parameters. ###
HISTORY_TOOL_CONTEXT_LIMIT = 50
HISTORY_TOOL_PURGE_KEEP_DAYS = 10 # TO-DO derive actual recorder setting
AUTOMATION_TOOL_EVENT_REGISTERED = "automation_registered_via_home_generative_agent"
AUTOMATION_TOOL_BLUEPRINT_NAME = "goruck/hga_scene_analysis.yaml"

### Camera video analyzer. ###
# Interval units are seconds.
VIDEO_ANALYZER_SCAN_INTERVAL = 1
# Root must be in allowlist_external_dirs.
VIDEO_ANALYZER_SNAPSHOT_ROOT = "/home/lindo/core/config/media/snapshots"
VIDEO_ANALYZER_SYSTEM_MESSAGE = """
You are a bot that generates a description of a video given descriptions of its frames.
Keep the description to the point and use no more than 250 characters.
"""
VIDEO_ANALYZER_PROMPT = """
Describe what is happening in this video from these frame descriptions:
"""
VIDEO_ANALYZER_MOBILE_APP = "mobile_app_lindos_iphone"
# Time offset units are seconds.
VIDEO_ANALYZER_TIME_OFFSET = 120
VIDEO_ANALYZER_SIMILARITY_THRESHOLD = 0.8
VIDEO_ANALYZER_DELETE_SNAPSHOTS = False

### postgresql db parameters for checkpointer and memory persistent storage. ###
DB_URI = "postgresql://hga:hga@localhost:5432/hga?sslmode=disable"

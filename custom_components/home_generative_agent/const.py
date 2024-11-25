"""Constants for Home Generative Agent."""

DOMAIN = "home_generative_agent"

### Configuration parameters that can be overriden in the integration's config UI. ###
# Name of the set of recommended options.
CONF_RECOMMENDED = "recommended"
# Name of system prompt.
CONF_PROMPT = "prompt"
### OpenAI chat model parameters.
# See https://platform.openai.com/docs/api-reference/chat/create.
CONF_CHAT_MODEL = "chat_model"
RECOMMENDED_CHAT_MODEL = "gpt-4o"
CONF_CHAT_MODEL_TEMPERATURE = "chat_model_temperature"
RECOMMENDED_CHAT_MODEL_TEMPERATURE = 1.0
### Ollama vision langauage model (VLM) parameters. ###
# The VLM is used for vision and summarization tasks.
# See https://github.com/ollama/ollama/blob/main/docs/modelfile.md#parameter
CONF_VLM = "vlm_model"
RECOMMENDED_VLM = "llama3.2-vision"
CONF_VISION_MODEL_TEMPERATURE = "vision_model_temperature"
RECOMMENDED_VISION_MODEL_TEMPERATURE = 0.8
CONF_SUMMARIZATION_MODEL_TEMPERATURE = "summarization_model_temperature"
RECOMMENDED_SUMMARIZATION_MODEL_TEMPERATURE = 0.8

### Chat model parameters. ###
# Parameters to manage chat model context length.
# CONTEXT_MAX_MESSAGES should be set larger than CONTEXT_SUMMARIZE_THREHOLD.
# Maximum number of mwssages to keep in context before deletion.
CONTEXT_MAX_MESSAGES = 1000
# Maximum number of messages to keep in context before summary analysis.
CONTEXT_SUMMARIZE_THRESHOLD = 100
# Chat model tool error handling
TOOL_CALL_ERROR_SYSTEM_MESSSAGE = """

Always call tools again with your mistakes corrected. Do not repeat mistakes.
"""
TOOL_CALL_ERROR_TEMPLATE = """
Error: {error}

Call the tool again with your mistake corrected.
"""

### Ollma VLM parameters. ###
# Ollama VLM server URL.
VLM_URL = "192.168.1.252:11434"
# Ollama VLM maximum nuber of output tokens to generate.
VLM_NUM_PREDICT = 4096
# Ollama VLM model prompts for summary tasks.
SUMMARY_SYSTEM_PROMPT = "You are a bot that summarizes messages from a smarthome AI."
SUMMARY_INITAL_PROMPT = "Create a summary of the messages above:"
SUMMARY_PROMPT_TEMPLATE = """
This is summary of the messages to date:
{summary}

Extend the summary by taking into account the new messages above:
"""
# Ollama VLM model prompts for vision tasks.
VISION_MODEL_SYSTEM_PROMPT = """
You are a bot that ONLY responds with an instance of JSON without any additional
information. You have access to a JSON schema, which will determine how the JSON
should be structured.
"""
VISION_MODEL_USER_PROMPT_TEMPLATE = """
Make sure to return ONLY an instance of the JSON, NOT the schema itself.
Do not add any additional information.
JSON schema:
{schema}

Task: Describe this image:
"""
# Ollama VLM model image size (in pixels).
VISION_MODEL_IMAGE_WIDTH = 1120
VISION_MODEL_IMAGE_HEIGHT = 1120

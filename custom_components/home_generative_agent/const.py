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
### Ollama vision model parameters. ###
# See https://github.com/ollama/ollama/blob/main/docs/modelfile.md#parameter
CONF_VISION_MODEL = "vision_model"
RECOMMENDED_VISION_MODEL = "llava"
CONF_VISION_MODEL_TEMPERATURE = "vision_model_temperature"
RECOMMENDED_VISION_MODEL_TEMPERATURE = 0.8
CONF_VISION_MODEL_NUM_PREDICT = "vision_model_num_predict"
RECOMMENDED_VISION_MODEL_NUM_PREDICT = 512

# Ollama vision model server URL.
VISION_MODEL_URL = "192.168.1.252:11434"

# Maximum number of messages to keep in chat model context before deletion.
CONTEXT_MAX_MESSAGES = 100

# Chat model tool error handeling.
TOOL_CALL_ERROR_SYSTEM_MESSSAGE = (
    "\nAlways call tools again with your mistakes corrected. Do not repeat mistakes."
)
TOOL_CALL_ERROR_TEMPLATE = (
    "Error: {error}\n Call the tool again with your mistake corrected."
)

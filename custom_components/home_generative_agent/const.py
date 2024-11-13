"""Constants for Home Generative Agent."""

DOMAIN = "home_generative_agent"

CONF_RECOMMENDED = "recommended"
CONF_PROMPT = "prompt"
CONF_CHAT_MODEL = "chat_model"
RECOMMENDED_CHAT_MODEL = "gpt-4o"
CONF_MAX_TOKENS = "max_tokens"
RECOMMENDED_MAX_TOKENS = 150
CONF_MAX_MESSAGES = "max_messages"
RECOMMENDED_MAX_MESSAGES = 100
CONF_TOP_P = "top_p"
RECOMMENDED_TOP_P = 1.0
CONF_TEMPERATURE = "temperature"
RECOMMENDED_TEMPERATURE = 1.0

OLLAMA_RECOMMENDED_BASE_URL = "192.168.1.252:11434"
OLLAMA_BASE_URL = "ollama_base_url"
OLLAMA_RECOMMENDED_MODEL = "llava"
OLLAMA_MODEL = "ollama_model"
OLLAMA_NUM_PREDICT = "num_predict"
OLLAMA_RECOMMENDED_NUM_PREDICT = 128
OLLAMA_TEMPERATURE = "ollama_temperature"
OLLAMA_RECOMMENDED_TEMPERATURE = 0.8
OLLAMA_TOP_P = "ollama_top_p"
OLLAMA_RECOMMENDED_TOP_P = 0.9
OLLAMA_TOP_K = "ollama_top_k"
OLLAMA_RECOMMENDED_TOP_K = 40

TOOL_CALL_ERROR_SYSTEM_MESSSAGE = (
    "\nAlways call tools again with your mistakes corrected. Do not repeat mistakes."
)
TOOL_CALL_ERROR_TEMPLATE = (
    "Error: {error}\n Call the tool again with your mistake corrected."
)

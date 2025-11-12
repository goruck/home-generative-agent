# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a Home Assistant custom integration called "home-generative-agent" that creates a generative AI agent using LangChain and LangGraph. The agent can understand your smart home context, control devices, analyze camera images, create automations, and maintain long-term memory.

## Development Commands

### Linting and Formatting
```bash
bash scripts/lint
```
This runs `ruff format` and `ruff check --fix` on the entire codebase.

### Ruff Configuration
- Line length: 88 characters
- Configured in `pyproject.toml` with specific ignores for this codebase
- See [tool.ruff.lint] section in pyproject.toml for ignored rules

## Code Architecture

### Core Structure

The integration follows Home Assistant's custom component structure:
- `custom_components/home_generative_agent/` - Main integration directory
- Entry point: `__init__.py` handles component lifecycle
- Config flow: `config_flow.py` handles UI-based configuration
- Platforms: `conversation.py`, `sensor.py`, `image.py`

### LangGraph Agent Architecture

The agent is built with LangGraph and operates as a stateful workflow:

**Graph Structure** (defined in `agent/graph.py`):
- `State` extends `MessagesState` with summary and usage metadata
- **Nodes**:
  - `agent`: Calls the chat model with context (memories, camera activity, summarized history)
  - `action`: Executes tools (both LangChain and Home Assistant native tools)
  - `summarize_and_remove_messages`: Manages context window by summarizing trimmed messages
- **Edges**:
  - START → agent
  - agent → action (if tool calls present)
  - agent → summarize_and_remove_messages (if no tool calls and trimming needed)
  - action → agent (loops back after tool execution)
  - summarize_and_remove_messages → END

### Tool System

**LangChain Tools** (defined in `agent/tools.py`):
- `get_and_analyze_camera_image`: VLM-based scene analysis
- `upsert_memory`: Store/update user memories with semantic search
- `add_automation`: Create HA automations (YAML or blueprint-based)
- `get_entity_history`: Query HA history/statistics database
- `web_search`: Search web via searxng + fetch content via Browserless
- `get_current_device_state`: Deprecated, replaced by native HA GetLiveContext

**Home Assistant Native Tools**: Exposed via HA LLM API (`ha_llm_api`)

**Tool Patterns**:
- Use `@tool(parse_docstring=True)` decorator
- Inject dependencies via `Annotated[RunnableConfig, InjectedToolArg()]` or `Annotated[BaseStore, InjectedStore()]`
- Access hass via `config["configurable"]["hass"]`
- Tool responses are sanitized and length-limited (see `_sanitize_tool_response`)
- Error handling with timeouts, retries, and classified error types
- Metrics tracking via `ToolCallMetrics`

### Context Management

**Message Trimming**:
- Controlled by `CONTEXT_MAX_TOKENS` (default: 22528) or `CONTEXT_MAX_MESSAGES` (default: 80)
- Use token-based trimming by default (`CONTEXT_MANAGE_USE_TOKENS = True`)
- Trimmed messages are summarized by a smaller model and inserted into system message
- Token counting is cross-provider via `count_tokens_cross_provider`

### Model Configuration

The integration supports multiple providers (OpenAI, Ollama, Gemini, Anthropic) for:
- **Chat model**: Primary reasoning/planning model
- **VLM (Vision Language Model)**: Image analysis
- **Summarization model**: Context compression
- **Embedding model**: Semantic search for memories

Configuration is centralized in `const.py` using `MODEL_CATEGORY_SPECS` registry.

### Video Analysis System

**Components**:
- `core/video_analyzer.py`: Main analyzer coordinating frame processing
- `core/frame_processor.py`: Per-frame image analysis and face recognition
- `core/snapshot_manager.py`: Snapshot storage and cleanup
- `core/face_recognition_service.py`: Optional face recognition via remote API
- `core/video_summarizer.py`: Multi-frame summarization

**Flow**:
1. Motion detection triggers snapshot capture
2. Frames are analyzed by VLM with face recognition
3. Multiple frames are combined into a narrative summary
4. Results stored in LangGraph store under `("video_analysis", camera_name)` namespace
5. Optionally published as "latest" event to image/sensor entities

**Key Settings** (in `const.py`):
- `VIDEO_ANALYZER_MODE`: disable / notify_on_anomaly / always_notify
- `VIDEO_ANALYZER_SIMILARITY_THRESHOLD`: 0.89 for anomaly detection
- `VIDEO_ANALYZER_SNAPSHOT_ROOT`: `/media/snapshots`

### Storage and Persistence

**PostgreSQL with pgvector**:
- Conversation checkpoints: LangGraph checkpoint-postgres
- Vector store: Memories and video analysis with semantic search
- Connection via `CONF_DB_URI`

**LangGraph Store**:
- Namespaces: `(user_id, "memories")` and `("video_analysis", camera_name)`
- Supports semantic search via embedding model

### Core Utilities

**Important modules** (`core/` directory):
- `utils.py`: Helper functions including `extract_final()` to strip reasoning delimiters
- `datetime_utils.py`: Timezone-aware datetime handling
- `error_handlers.py`: Centralized error handling
- `notification_manager.py`: Push notifications
- `queue_manager.py`: Task queue for video processing
- `uniqueness_filter.py`: Semantic deduplication
- `migrations.py`: Database schema migrations

## Adding New Tools

When creating a new tool file (e.g., for travel_times):

1. **File location**: `custom_components/home_generative_agent/agent/` directory
2. **Imports**: Use LangChain's `@tool` decorator and inject dependencies
3. **Pattern**:
```python
from langchain_core.tools import InjectedToolArg, tool
from langchain_core.runnables import RunnableConfig
from typing import Annotated

@tool(parse_docstring=True)
async def my_tool(
    required_param: str,
    optional_param: str = "",
    *,
    config: Annotated[RunnableConfig, InjectedToolArg()],
) -> str:
    """
    Tool description for the LLM.

    Args:
        required_param: Description
        optional_param: Description
    """
    if "configurable" not in config:
        return "Configuration not found."

    hass = config["configurable"]["hass"]
    # Implementation
    return "result"
```
4. **Register the tool**: Import and add to the tools list in the integration setup
5. **Error handling**: Use try/except, return helpful error messages
6. **Response sanitization**: Keep responses concise (TOOL_RESPONSE_MAX_LENGTH = 16000)
7. **Format code**: Run `bash scripts/lint` before committing

## Testing

The integration is based on the [integration_blueprint template](https://github.com/ludeeus/integration_blueprint) which provides a containerized development environment for testing with a standalone Home Assistant instance.

## Important Patterns

### Async Operations
- All tool functions are async
- Use `hass.async_add_executor_job()` for blocking operations
- Use `asyncio.wait_for()` for timeouts

### Error Messages
- Return user-friendly strings from tools
- Log detailed errors with LOGGER
- Classify errors with `ToolErrorType`

### Reasoning Models
- Some Ollama models (qwen3, deepseek-r1) output reasoning in `<think>` tags
- Use `extract_final()` to strip reasoning and return only the final answer
- Configure delimiters in `REASONING_DELIMITERS`

### Home Assistant Integration
- Access hass via config: `hass = config["configurable"]["hass"]`
- Use HA services: `await hass.services.async_call(...)`
- Access HA states: `hass.states.get(entity_id)`
- Fire events: `hass.bus.async_fire(event_type, event_data)`

## Configuration Keys

Key configuration constants in `const.py`:
- `CONF_DB_URI`: PostgreSQL connection string
- `CONF_OLLAMA_URL`: Ollama server URL
- `CONF_CHAT_MODEL_PROVIDER`: Provider selection (openai/ollama/gemini/anthropic)
- `CONF_VIDEO_ANALYZER_MODE`: Video analysis mode
- `CONF_FACE_RECOGNITION_MODE`: Face recognition toggle
- `TOOL_CALL_TIMEOUT_SECONDS`: 60s timeout for tool execution

## Database Schema

The integration uses LangGraph's built-in checkpoint schema plus custom tables for video analysis storage. Migrations are handled in `core/migrations.py`.
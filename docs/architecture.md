# Architecture and Design

- [High-Level Overview](#high-level-overview)
- [LangGraph Agent](#langgraph-agent)
- [LLM Context Management](#llm-context-management)
- [Streaming](#streaming)
- [Latency](#latency)
- [Tools](#tools)
- [Hardware Reference Setup](#hardware-reference-setup)

---

## High-Level Overview

![Architecture diagram](../assets/hga_arch.png)

The integration follows [Home Assistant Core](https://developers.home-assistant.io/docs/development_index/) best practices and is compliant with [HACS](https://www.hacs.xyz/) publishing requirements.

The agent is built on [LangGraph](https://www.langchain.com/langgraph) and uses the HA `conversation` component for user interaction. It queries the Home Assistant LLM API to fetch home state and the native tools available to it, then extends that with custom LangChain tools for camera analysis, memory, automation creation, and more.

**Model tiers:** The architecture uses multiple specialized models rather than one large model for everything:

| Category | Provider | Default model | Purpose |
|---|---|---|---|
| Chat | OpenAI | gpt-5 | High-level reasoning and planning |
| Chat | Ollama | gpt-oss | High-level reasoning and planning |
| Chat | Gemini | gemini-2.5-flash-lite | High-level reasoning and planning |
| Chat | Anthropic | claude-sonnet-4-6 | High-level reasoning and planning |
| Chat | OpenAI Compatible | gpt-4o | High-level reasoning and planning |
| VLM | Ollama | qwen3-vl:8b | Image scene analysis |
| VLM | OpenAI | gpt-5-nano | Image scene analysis |
| VLM | Gemini | gemini-2.5-flash-lite | Image scene analysis |
| VLM | Anthropic | claude-sonnet-4-6 | Image scene analysis |
| VLM | OpenAI Compatible | gpt-4o | Image scene analysis |
| Summarization | Ollama | qwen3:8b | Primary model context summarization |
| Summarization | OpenAI | gpt-5-nano | Primary model context summarization |
| Summarization | Gemini | gemini-2.5-flash-lite | Primary model context summarization |
| Summarization | Anthropic | claude-haiku-4-5-20251001 | Primary model context summarization |
| Summarization | OpenAI Compatible | gpt-4o | Primary model context summarization |
| Embeddings | Ollama | mxbai-embed-large | Semantic search |
| Embeddings | OpenAI | text-embedding-3-small | Semantic search |
| Embeddings | Gemini | gemini-embedding-001 | Semantic search |
| Embeddings | OpenAI Compatible | text-embedding-3-small | Semantic search |

Models can be cloud-based (best accuracy, higher cost) or edge-based (good accuracy, lowest cost). Edge models run under [Ollama](https://ollama.com/) or any OpenAI-compatible server (vLLM, llama.cpp, LiteLLM, etc.). Defaults are defined in `const.py` and configurable via the integration UI.

---

## LangGraph Agent

LangGraph powers the conversation agent as a stateful graph. The graph below models the agent workflow:

![Agent graph](../assets/graph.png)

The main workflow nodes each modify shared agent state. Solid edges are unconditional transitions; dashed edges are conditional.

| Node | Role |
|---|---|
| `__start__` | Graph entry point |
| `retrieve_tools` | Queries the vector index to select tools most relevant to the current message (see [Tool Retrieval](configuration.md#tool-retrieval-rag)) |
| `agent` | Runs the primary LLM with only the retrieved tools bound to its context |
| `action` | Executes tool calls; returns control to `agent`. Multiple tool calls within a single turn execute concurrently via `asyncio.gather` with a 30 s per-tool timeout |
| `tool_loop_guard` | Intercepts if the agent requests more tool calls than the safety limit (3 rounds per turn) and returns a friendly message asking you to rephrase or break the request into smaller steps |
| `summarize_and_remove_messages` | When the agent does not call a tool, trims context if messages exceed the configured max and replaces trimmed messages with a shorter summary in the system message |
| `__end__` | Graph exit point |

The agent can call HA LLM API tools including [built-in intents](https://developers.home-assistant.io/docs/intent_builtin) like `HassTurnOn` and `HassTurnOff`. Lock intents are normalized to lock/unlock services; alarm intents are routed to the `alarm_control` tool.

---

## LLM Context Management

Context length is carefully managed to balance cost, accuracy, and latency and to avoid rate limits (e.g. OpenAI's Tokens per Minute restriction). The system trims messages that exceed a configured maximum and replaces them with a shorter summary inserted into the system message.

| Parameter | Default | Description |
|---|---|---|
| `max_messages_in_context` | `60` | Messages to keep in context before trimming |
| `max_tokens_in_context` | `32000` | Tokens to keep in context before trimming |
| `manage_context_with_tokens` | `true` | If `true`, use tokens; otherwise use message count |

These are configurable in the integration's **Options** flow.

**Prompt caching:** Static content (instructions, examples) is placed upfront in prompts so cloud providers can cache it across turns, reducing both latency and cost.

---

## Streaming

Native LLM streaming (v3.12.0+) means the first tokens appear in the HA conversation UI within milliseconds of the model starting its response — total response time is unchanged, but perceived latency drops significantly.

OpenAI and Anthropic providers are constructed with `streaming=True` so they emit token chunks directly. Other providers use their existing streaming or fallback behavior. The implementation uses `astream_events` + HA ChatLog delta API (requires HA 2026.4.0 or later).

For a deep-dive on the streaming implementation, see [docs/streaming-design.md](streaming-design.md).

---

## Latency

Several techniques reduce latency:

- Specialized smaller helper LLMs run on edge for camera analysis and summarization.
- Static prompt content is placed upfront to enable cloud provider prompt caching.
- Multi-tool turns execute tool calls concurrently (`asyncio.gather`).
- Streaming means perceived response time is near-instant even when total generation time is longer.

Typical performance on the [author's reference setup](architecture.md#hardware-reference-setup) (Raspberry Pi 5 + Nvidia 3090 edge server):

| Action | Latency | Remark |
|---|---|---|
| HA intents | < 1 s | e.g. turn on a light; first token streams immediately |
| Analyze camera image | < 3 s | initial request |
| Add automation | < 1 s | |
| Memory operations | < 1 s | |

Your latency will vary based on model provider, model size, network conditions, and hardware.

---

## Tools

The agent has access to HA LLM API tools and the following custom LangChain tools:

| Tool | Purpose |
|---|---|
| `get_and_analyze_camera_image` | Run scene analysis on the image from a camera |
| `upsert_memory` | Add or update a memory |
| `add_automation` | Create and register a HA automation (available when Schema-first YAML mode is disabled) |
| `write_yaml_file` | Write YAML to `/config/www/` and return a `/local/...` URL |
| `confirm_sensitive_action` | Confirm and execute a pending critical action with a PIN |
| `alarm_control` | Arm or disarm an alarm control panel with the alarm code |
| `get_entity_history` | Query HA database for entity history |
| `resolve_entity_ids` | Resolve entity IDs from friendly names, areas, labels, and domains |
| ~~`get_current_device_state`~~ | ~~Get the current state of one or more HA devices~~ (deprecated; replaced by native HA GetLiveContext tool) |

On each turn, only the most relevant tools are loaded into the agent's prompt via vector similarity search (see [Tool Retrieval](configuration.md#tool-retrieval-rag)). A simple error recovery mechanism asks the agent to retry a tool call with corrected parameters when it makes a mistake.

---

## Hardware Reference Setup

The author's test installation:

- **Home Assistant host:** Raspberry Pi 5 with SSD storage, Zigbee, and LAN connectivity.
- **Edge model server:** Ubuntu-based server with AMD 64-bit 3.4 GHz CPU, Nvidia 3090 GPU, and 64 GB system RAM. Same LAN as the Raspberry Pi.
- Edge models served under Ollama.

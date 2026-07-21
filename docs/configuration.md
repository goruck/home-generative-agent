# Configuration

Configuration is done entirely in the Home Assistant UI using subentry flows. A *subentry* is a discrete, independently configured capability — for example a Model Provider, a Feature set, or Sentinel. Each subentry has its own settings and can be added, reconfigured, or removed without affecting others.

The configuration UI is available in English, Czech, and Turkish, with a partial Russian translation. Text follows your Home Assistant language settings and falls back to English for untranslated strings. To contribute a new language, see [Translations](contributing.md#translations).

- [Basic Setup](#basic-setup)
- [Model Providers](#model-providers)
- [Features](#features)
- [Tool Retrieval (RAG)](#tool-retrieval-rag)
- [Control Home Assistant (LLM API)](#control-home-assistant-llm-api)
- [Speech-to-Text (STT)](#speech-to-text-stt)
- [Schema-first YAML Mode](#schema-first-yaml-mode)
- [Critical Action PIN](#critical-action-pin)
- [Global Options](#global-options)

---

## Basic Setup

1. Open **Settings → Devices & Services → Home Generative Agent**.
2. Click **+ Model Provider** to add a provider (Cloud or Edge → provider type → credentials → model defaults).
   - The first provider added is automatically assigned to all features.
   - A provider must exist before you can run **+ Setup**.
3. Click **+ Setup** to enable features. Choose a setup mode:
   - **Basic** — enables all features (Conversation, Camera Image Analysis, Conversation Summary) with recommended defaults and creates the database subentry automatically. No database prompt appears.
   - **Advanced** — step through each feature individually to assign providers, models, and fallback chains; includes a database configuration step.
4. Use the **gear icon** on any feature to adjust its model settings later.
5. Click **+ Sentinel** to configure proactive anomaly detection (see [Sentinel guide](sentinel.md)). Choose a setup mode:
   - **Basic** — enables anomaly alerting with recommended defaults. Prompts for notify service, daily digest, and an optional level-increase PIN.
   - **Advanced** — exposes all Sentinel options: intervals, cooldowns, quiet hours, triage, baseline, discovery, camera entry links, and per-entity rule exclusions.

> **Reconfiguring:** Running **+ Setup** or **+ Sentinel** again when a subentry already exists opens the same mode selector. Advanced mode pre-populates every field with the current saved values. Basic mode always starts from recommended defaults and warns before overwriting.

> **Removing Sentinel:** Delete the Sentinel subentry from the integration page to stop all monitoring immediately. Sentinel background tasks stop and the health sensor transitions to `disabled`.

---

## Model Providers

Supported providers and their default models:

| Category | Provider | Default model | Purpose |
|---|---|---|---|
| Chat | OpenAI | gpt-5 | Reasoning and planning |
| Chat | Ollama | gpt-oss | Reasoning and planning |
| Chat | Gemini | gemini-2.5-flash-lite | Reasoning and planning |
| Chat | Anthropic | claude-sonnet-4-6 | Reasoning and planning |
| Chat | OpenAI Compatible | gpt-4o | Reasoning and planning |
| VLM | Ollama | qwen3-vl:8b | Image scene analysis |
| VLM | OpenAI | gpt-5-nano | Image scene analysis |
| VLM | Gemini | gemini-2.5-flash-lite | Image scene analysis |
| VLM | Anthropic | claude-sonnet-4-6 | Image scene analysis |
| VLM | OpenAI Compatible | gpt-4o | Image scene analysis |
| Summarization | Ollama | qwen3:8b | Context summarization |
| Summarization | OpenAI | gpt-5-nano | Context summarization |
| Summarization | Gemini | gemini-2.5-flash-lite | Context summarization |
| Summarization | Anthropic | claude-haiku-4-5-20251001 | Context summarization |
| Summarization | OpenAI Compatible | gpt-4o | Context summarization |
| Embeddings | Ollama | mxbai-embed-large | Semantic search |
| Embeddings | OpenAI | text-embedding-3-small | Semantic search |
| Embeddings | Gemini | gemini-embedding-001 | Semantic search |
| Embeddings | OpenAI Compatible | text-embedding-3-small | Semantic search |

**Embedding model selection:** Embeddings are configured like any other feature: enable the **Embeddings** feature under **+ Setup** (Advanced mode) and assign it a provider and model. The embedding provider can be completely separate from the chat provider — e.g. llama.cpp for chat and a dedicated llama.cpp or Ollama server for embeddings. When the Embeddings feature is disabled, the provider is chosen automatically: the Conversation provider if it supports embeddings, otherwise the first embedding-capable provider.

**Multiple providers:** You can add multiple Model Provider subentries and assign them per-feature. For example: a "Primary Ollama" provider for chat and a "Vision Ollama" provider for camera analysis. You can also mix types — a local vLLM server as **OpenAI Compatible** alongside an Ollama provider.

### Provider Fallbacks

Feature setup can include an ordered list of fallback providers. A fallback applies only to that feature category, so a chat fallback does not automatically cover VLM, summarization, or embeddings.

Fallbacks are evaluated at setup/reload time and at runtime:

- If the primary provider is unavailable when the integration starts or reloads, HGA selects the first usable configured fallback and logs `Fallback selected at setup ...`. That selected provider remains active until the integration is reloaded or Home Assistant restarts. If the primary provider comes back online later, HGA does not automatically switch back during the same runtime.
- If the active provider fails during a model call with a retryable error, HGA tries the next configured fallback provider for that call and logs the runtime fallback activation. Retryable failures include local transport/connectivity errors, timeouts, rate limits, and transient provider/server errors.
- If no fallback is configured for a category and the primary provider is unavailable at setup, HGA keeps a placeholder model for that category and logs this at debug level. Configure a fallback for each category that should degrade to another provider.

When a fallback becomes active, HGA also notifies the user once per category/provider for the current runtime. If `notify_service` is configured, the notification is sent to that mobile notify service. If no notify service is configured, HGA creates a Home Assistant persistent notification instead. Runtime fallback notifications are deduplicated, so repeated video-analysis, summarization, chat, or embedding retries do not spam the user. When the active fallback is a cloud provider, the notification includes a cost warning: `Cloud model usage may incur provider costs.`

Fallback model settings come from the fallback provider itself. For chat, VLM, and summarization, HGA first uses the fallback provider subentry's category-specific model setting (`chat_model`, `vlm_model`, or `summarization_model`). If that provider does not define a category-specific model, HGA uses the recommended model for that provider/category from `const.py` (`MODEL_CATEGORY_SPECS`). Category temperature defaults also come from `const.py` when not otherwise set. Ollama fallbacks additionally use the category defaults for context, keepalive, top-p, repeat penalty, and related tuning values unless the provider settings override them.

Chat fallback chains are invoked as complete model calls rather than direct token streams. The Home Assistant chat UI can still stream LangGraph conversation events, but HGA does not switch providers after partial provider text has already been emitted. This avoids mixed responses where a failed primary provider starts a reply and a fallback provider finishes with different content.

To switch back to a recovered primary provider, reload the Home Generative Agent integration or restart Home Assistant.

> **llama-server embeddings** — OpenAI-compatible base URLs are normalized to include the `/v1` prefix, so embedding requests reach llama-server's OpenAI-format `/v1/embeddings` endpoint (its bare `/embeddings` route returns a non-OpenAI response that used to crash embedding calls). Enter the base URL with or without `/v1` — both work. Start llama-server with `--embeddings` on the instance that serves the embedding model. If you still see `Memory semantic search failed — embedding endpoint returned an incompatible response` in the logs, the agent has fallen back to recency-based memory retrieval; check the embedding server's response format or use a dedicated Ollama provider with `mxbai-embed-large`.

---

## Features

Each feature is enabled separately under **+ Setup** and has its own model/provider assignment:

- **Conversation** — the main conversational agent
- **Camera Image Analysis** — on-demand and proactive vision analysis
- **Conversation Summary** — automatic context window management
- **Embeddings** — embedding model for semantic memory and tool retrieval; assign it a dedicated provider/server or leave it off for automatic selection

Global options such as system prompt, face recognition URL, context management parameters, and the critical-action PIN live in the integration's **Options** flow (gear icon on the integration page).

---

## Tool Retrieval (RAG)

> **Thanks to [1Jamie](https://github.com/1Jamie) for this feature!**

On startup the integration indexes all available tools as vector embeddings in PostgreSQL. Each turn, only the most relevant tools for the user's message are loaded into the agent's prompt — keeping context short and tool selection accurate.

Two options in the **Options** flow control this:

- **Retrieval Limit** (`tool_retrieval_limit`, default `5`) — maximum tools made available per turn. Raise if the agent misses tools on complex multi-step requests; lower to reduce prompt size.
- **Relevance Threshold** (`tool_relevance_threshold`, default `0.15`) — cosine similarity cutoff. Lower if the agent misses tools it should pick up; raise to tighten selectivity.

A **Tool Index Status** diagnostic sensor (`sensor.tool_index_status`) shows the current index state:

| State | Meaning |
|---|---|
| `indexing` | First-run embedding in progress |
| `ready` | Index available; tools retrieved per-turn by semantic search |
| `failed` | Embedding provider unreachable; agent falls back to all tools |
| `unknown` | Index state not yet reported |

Subsequent restarts skip unchanged tools using SHA-256 content hashing, so re-indexing is fast.

---

## Control Home Assistant (LLM API)

The **Control Home Assistant** option in the Options flow is a multi-select that controls which HA LLM APIs the agent can use.

- **Assist** (`assist`) — the built-in HA Assist API. Grants entity-control intents and the full entity list. Select this for standard voice-assistant control.
- **MCP server integrations** — any [Model Context Protocol](https://www.home-assistant.io/integrations/mcp_server/) integration you have configured registers its own LLM API (e.g. `mcp-<entry_id>`). Those entries appear in the list once added.

You can select any combination. Selecting both Assist and one or more MCP APIs merges all their tools into a single combined API. Deselecting everything runs the agent with only its built-in LangChain tools (no HA entity control, no MCP tools).

**Adding an MCP server:**

1. Go to **Settings → Devices & Services → Add Integration** → search **Model Context Protocol**.
2. Enter the server URL and complete setup.
3. The MCP integration registers an LLM API automatically.
4. Open **Settings → Devices & Services → Home Generative Agent → Configure**.
5. Select the new entry in **Control Home Assistant** and save.

---

## Speech-to-Text (STT)

HGA provides a built-in STT engine using the OpenAI Whisper API — no separate STT integration required.

1. Open **Settings → Devices & Services → Home Generative Agent**.
2. Click **+ STT Provider**.
3. Choose **OpenAI** and give it a name.
4. On the **Credentials** step, either reuse an existing OpenAI Model Provider subentry or select **Use a separate key** and enter a dedicated API key.
5. On **Model & advanced options**, pick a model (recommended: `gpt-4o-mini-transcribe`) and set optional fields:
   - `language` (optional): e.g. `en` or `en-US`
   - `prompt` (optional): hints for domain-specific vocabulary
   - `temperature` (optional): 0–1
   - `translate`: only supported by `whisper-1`; other models fall back to transcription
6. Go to **Settings → Voice assistants → Assist pipelines** and select **STT - OpenAI** (or your chosen name) for Speech-to-text.

---

## Schema-first YAML Mode

**Schema-first JSON for YAML requests** controls how the agent handles YAML-style requests (automations, dashboards, or "show me YAML").

| Setting | Behavior |
|---|---|
| **ON** | Agent returns strict JSON converted to YAML for display. Automations are not auto-registered — YAML is shown in chat. To save a file, ask the agent to **save the YAML**; it writes under `/config/www/` and returns a `/local/...` URL. |
| **OFF** | Dashboard generation is disabled. Automations are auto-registered; YAML is not shown in chat. Other YAML requests follow standard prompt behavior. |

> Note: YAML rendered in the chat window may not preserve indentation due to UI rendering — use the saved file if you need valid YAML to copy.

Example: *"Save this YAML to a file called garage-light."*

---

## Critical Action PIN

Protects sensitive actions (unlocking doors, opening covers) behind a second verification step.

**Setup:** Go to **Settings → Devices & Services → Home Generative Agent → Configure** and toggle **Require critical action PIN**. Enter a 4–10 digit PIN. The value is stored as a salted hash. Leaving the field blank while the toggle is on clears the stored PIN; turning the toggle off removes the guard entirely.

> In the conversation agent settings for HGA, disable **Prefer handling commands locally** for the PIN protection to work correctly.

**Protected actions:**
- Unlocking or opening locks
- Opening covers whose `entity_id` includes `door`, `gate`, or `garage`
- Using HA intent tools on locks

Alarm control panels use their own alarm code, which is separate from the critical-action PIN.

**Flow:** When you request a protected action, the agent queues the request and asks for the PIN. Reply with the digits to complete the action. After five bad attempts or 10 minutes, the queued action expires and you must ask again. If the guard is enabled but no PIN is configured, the agent rejects requests until you set one in Options.

---

## Global Options

The **Options** flow (gear icon on the integration page) exposes:

- System prompt override
- **Camera description language** (`vlm_response_language`) — optional, e.g. `Czech`. When set, camera image descriptions (chat camera tool, `save_and_analyze_snapshot`, and the proactive video analyzer) are requested in that language. Leave empty for English. The internal `Scene unchanged.` repeated-scene reply is deliberately kept in English — it is matched by code, not shown to users (see [Camera Entities](camera-entities.md)).
- **Additional camera analysis instructions** (`vlm_prompt_extra`) — optional multiline text appended to the VLM system prompt, e.g. `Ignore cars in the driveway`. Appended after the built-in rules, never replacing them.
- Face recognition service URL
- Context management parameters (`max_messages_in_context`, `max_tokens_in_context`, `manage_context_with_tokens`)
- Critical action PIN toggle and value
- Tool retrieval limit and relevance threshold
- `model_provider_uncontended` — bypass all local GPU gates when the server has dedicated capacity
- **Video analyzer mode** — disable / notify_on_anomaly / always_notify
- **Enable perceptual-hash frame filter (dHash)** — skip visually identical frames before VLM analysis (off by default; always active for ring-mqtt `event_select` capture loops regardless of this setting; see caveat in [Camera Entities](camera-entities.md#advanced-options))
- **Motion sensor → camera overrides** — one `binary_sensor.X: camera.Y` pair per line; use when automatic resolution picks the wrong camera (see [Motion → camera resolution](camera-entities.md#motion--camera-resolution))

See [Architecture](architecture.md#llm-context-management) for detail on context management parameters.

---

> **Developer reference:** For a complete listing of every named constant — including code-only tuning knobs and module-level internals — see the [Constants Reference](constants.md).

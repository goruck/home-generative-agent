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

> **Models that pin temperature** — Some OpenAI models (o-series and other reasoning-style models) only accept their default `temperature`/`top_p` and reject any other value with a 400 error. When that happens, HGA logs a warning and automatically retries the call without the rejected parameter, so conversation, camera analysis, summarization, and Sentinel keep working. In a multi-provider fallback chain the retry is applied per provider; a provider that still rejects its sampling settings fails over to the next provider in the chain and counts toward the circuit breaker. To avoid the extra retry on every call, leave the feature's temperature at the model's supported default (`top_p` has no UI setting — its default is a code-only constant, see the [Constants Reference](constants.md)).

---

## Tool Retrieval (RAG)

> **Thanks to [1Jamie](https://github.com/1Jamie) for this feature!**

On startup the integration indexes all available tools as vector embeddings in PostgreSQL. Each turn, only the most relevant tools for the user's message are loaded into the agent's prompt — keeping context short and tool selection accurate.

The tool universe is per-request: Home Assistant exposes some Assist tools only for certain devices (the timer intents — `HassStartTimer` and friends — exist only when the requesting device supports timers), and MCP tools exist only while their server is reachable. The index reconciles tool *presence* with this automatically in both directions (a tool whose description or schema changes without renaming keeps its stored definition until the next indexing pass, as before). When a turn's device exposes tools the index has never seen, they are indexed inline before retrieval runs, so the very first "set a timer for two minutes" from a voice satellite already has `HassStartTimer` available for retrieval — no restart or repeat needed (selection still follows normal relevance ranking). Conversely, indexed tools that don't exist for the current turn (timer tools in a browser chat, tools of an MCP server that failed to load) are never bound.

A few tools bypass similarity ranking: `GetLiveContext` is always available (whenever the Assist API that provides it is loaded for the turn), and `add_automation` is guaranteed to be available whenever your message signals automation-creation intent — explicit wording ("automate...", "remind me every 30 minutes") or an action verb plus a when/if trigger clause ("turn on the porch light when motion is detected"). Read-only state questions ("check if the garage door is open") do not trigger it. These are appended on top of the retrieval limit, so they never crowd out ranked tools. Intent detection is English-only for now; see [Architecture](architecture.md#tools) for details.

Two options in the **Options** flow control this:

- **Retrieval Limit** (`tool_retrieval_limit`, default `5`) — maximum tools made available per turn. Raise if the agent misses tools on complex multi-step requests; lower to reduce prompt size.
- **Relevance Threshold** (`tool_relevance_threshold`, default `0.15`) — cosine similarity cutoff. Lower if the agent misses tools it should pick up; raise to tighten selectivity.

A **Tool Index Status** diagnostic sensor (`sensor.tool_index_status`) shows the current index state:

| State | Meaning |
|---|---|
| `indexing` | Embedding in progress — the first full run at startup, or a mid-turn top-up adding newly discovered tools |
| `ready` | Index available; tools retrieved per-turn by semantic search |
| `failed` | Embedding provider unreachable; agent falls back to a keyword-filtered tool list capped at the retrieval limit |
| `unknown` | Index state not yet reported |

The sensor's `tools_indexed` attribute reports the cumulative number of tools in the index — not the size of the last indexing batch — and `last_updated` records the time of the last successful index update.

Subsequent restarts skip unchanged tools using SHA-256 content hashing, so re-indexing is fast.

---

## Control Home Assistant (LLM API)

The **Control Home Assistant** option in the Options flow is a multi-select that controls which HA LLM APIs the agent can use.

- **Assist** (`assist`) — the built-in HA Assist API. Grants entity-control intents and the full entity list. Select this for standard voice-assistant control.
- **MCP server integrations** — any [Model Context Protocol](https://www.home-assistant.io/integrations/mcp_server/) integration you have configured registers its own LLM API (e.g. `mcp-<entry_id>`). Those entries appear in the list once added.

You can select any combination. Selecting both Assist and one or more MCP APIs merges all their tools into a single combined API. Note that deselecting everything does **not** disable HA control: an empty selection is stored as "unset", which the agent reads as the Assist default, so Assist is silently re-enabled on the next save. Running with no LLM API at all is currently not expressible through the form (tracked in `TODOS.md`).

**Adding an MCP server:**

1. Go to **Settings → Devices & Services → Add Integration** → search **Model Context Protocol**.
2. Enter the server URL and complete setup.
3. The MCP integration registers an LLM API automatically.
4. Open **Settings → Devices & Services → Home Generative Agent → Configure**.
5. Select the new entry in **Control Home Assistant** and save.

**Removing an MCP server:** if a selected server's integration is removed (or is temporarily unavailable), its entry stays selected and is shown as `<id> (no longer available)` so the form remains saveable and your selection is never dropped behind your back. Deselect the dead entry and save to clean it up. While it stays selected, a warning is logged when the options form is built and each time the agent loads its APIs; other selected APIs keep working, but if the unavailable entry is the *only* selection, conversations fail with "No LLM APIs could be loaded" until the server returns or you deselect it.

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

**Credential changes take effect on the next utterance.** Each STT entity keeps one OpenAI client, built on Home Assistant's shared HTTP client, and rebuilds it when the resolved API key changes — no restart or integration reload needed. If the entry is linked to a Model Provider subentry, that provider's key is the only one used: linking blanks the separate STT key, so a linked provider without a usable key fails the utterance with an `OpenAI STT API key missing` warning in the log rather than falling back to a stale key.

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

> **You must turn off "Prefer handling commands locally"** in your voice assistant pipeline for this PIN to protect anything.
>
> The PIN can only guard commands the agent actually receives. With that option enabled, Home Assistant matches simple commands against its own built-in sentences and executes them itself — "unlock the front door" included — before the conversation agent is ever called. HGA never sees the turn, so it cannot hold it for a PIN. This is Home Assistant's designed behavior and no integration can intercept it. (The related `CONTROL` capability flag does not help here: it only diverts *state questions* and media search to the agent, never control commands.)
>
> Since 3.30.9 HGA detects this combination and raises a repair issue under **Settings → System → Repairs** naming the affected pipeline, so the gap cannot sit there silently.
>
> Two other paths also bypass the PIN, and turning the option off does not close them:
>
> - **Sentence triggers.** Home Assistant runs those ahead of any conversation agent, so a sentence trigger you author yourself that unlocks a door is never screened.
> - **Anything that is not the conversation agent** — dashboard buttons, scripts, other automations. The PIN guards what the agent is asked to do, not the underlying service call.
>
> If a lock or cover must never be voice-operable at all, the reliable control is to stop exposing it to Assist (**Settings → Voice assistants → Expose**) rather than relying on the PIN.

**Protected actions:**
- Unlocking or opening locks
- Opening covers whose `entity_id` includes `door`, `gate`, or `garage` — via `open_cover`, `open`, `toggle`, or `set_cover_position`, since all four open a closed door
- Using HA intent tools on locks
- **Creating an automation that performs any of the above** — "unlock the front door whenever I get home" is held for the PIN just like the direct command

Alarm control panels use their own alarm code, which is separate from the critical-action PIN.

**Language.** The PIN prompts and confirmation messages the agent returns in chat are fixed strings, not model output. They follow the Home Assistant server language (**Settings → System → General**) and are currently available in English and Czech; any other language falls back to English. Diagnostic log lines stay in English.

**Automation screening.** Automations are screened after Home Assistant validates them, so blueprint inputs are resolved and every nested branch (`choose`, `if`/`then`/`else`, `repeat`, `parallel`, `sequence`) is inspected. Nothing is written to `automations.yaml` and no reload happens until the PIN is confirmed.

Screening classifies each step with Home Assistant's own action taxonomy and lets it through only when it is provably harmless. A service call is not the only way an automation can unlock a door, so these are protected too:

- **Device actions** (`device_id` + `domain` + `type`), which carry no service name but still call `lock.unlock`. A device action's `type` is not the service it runs — each integration maps it in its own code — so *any* device action on a guarded domain asks for the PIN, including harmless ones like locking a lock or closing a cover.
- **`scene.apply`**, which sets entity states inline — Home Assistant reproduces an `unlocked` lock state by calling `lock.unlock`. `scene.create` only snapshots current state, but the scene it stores can be activated later, so it is screened the same way.
- **`homeassistant.turn_on` / `turn_off` / `toggle`**, screened against each target entity's own domain.

Screening also **fails closed** — it asks for the PIN rather than guessing — when it cannot see what a step will do:

- A service name built from a template.
- A target that is an area, device, label, floor, group, entity registry ID, or `entity_id: all`, *when the call's domain and service otherwise match a protected rule*. The entities such a target resolves to are not known at write time, so an `entity_id` substring rule cannot be checked against them and is treated as matching. An ordinary `light.turn_on` over an area is unaffected.
- Activating a scene, calling a script, triggering another automation, firing an event, pressing a button (a template button's press field is a full script, and the check cannot tell a template button from a plain one, so all of them prompt), calling `python_script` / `shell_command` / `rest_command`, or handing text to `conversation.process` (which can dispatch an intent whose `intent_script` unlocks a door): those all run configuration stored elsewhere. Expect a PIN prompt for these even when the target is harmless.
- **Any generic `homeassistant.*` call whose targets are not all named entities** — this one gates unconditionally, whatever the service. `homeassistant.turn_on` forwards by resolved domain, and an area, group, or entity registry ID can resolve to a script, or to a template entity whose `turn_on` runs one. The integration cannot resolve those before the automation is written, so it asks.
- A step type this integration does not recognize, including one a future Home Assistant release introduces.

**What screening cannot see.** Screening runs before the automation is written and has no access to Home Assistant's entity registry, so it cannot expand a group or an area, resolve a registry ID, or tell that a given `switch.x` is a template switch whose `turn_on` runs a stored script. That is why unresolvable targets are gated rather than reasoned about. Separately, the check matches domains and services, so a *transport* service that reaches a lock without naming the lock domain is invisible to it. The realistic case is `mqtt.publish` to a lock's command topic — an MQTT-backed lock (Zigbee2MQTT, ring-mqtt) can be opened by publishing to its topic, and no service-name rule can distinguish that from any other MQTT message. The same is true of any **raw protocol write** that addresses a device beneath the entity layer: `zwave_js.set_value` can write a Door Lock command class directly (as a service call or as a device action), and `zha.issue_zigbee_cluster_command` is equivalent for Zigbee. These are not gated by default because they are ordinary tools on those stacks and gating them would prompt on routine automations.

If you run locks on one of these transports and want them covered, add the relevant entry to the critical actions — `{"domain": "mqtt", "service": "publish"}`, `{"domain": "zwave_js", "service": "set_value"}`, or `{"domain": "zha", "service": "issue_zigbee_cluster_command"}` — accepting that every automation using that transport will then ask for the PIN.

**Blueprint automations are attested at approval time only.** A blueprint-based automation is stored in `automations.yaml` as a `use_blueprint:` reference, and Home Assistant re-substitutes the blueprint on every reload. Screening reads the substituted actions, so the PIN confirms what the blueprint did *when you approved it*. Editing that blueprint file afterwards changes what the approved automation runs, without a fresh prompt. For plain YAML automations the config that is written is the one that was screened.

**Flow:** When you request a protected action, the agent queues the request and asks for the PIN. Reply with the digits to complete the action. After five bad attempts or 10 minutes, the queued action expires and you must ask again. Only the user who made the request can confirm it. If the guard is toggled on but no PIN has been set, a direct command logs a warning and proceeds, while an automation is refused outright — an automation persists and keeps firing, so there is no safe way to let it through unconfirmed. Set a PIN in Options for the guard to take effect.

---

## Global Options

The **Options** flow (gear icon on the integration page) exposes:

- System prompt override
- **Camera description language** (`vlm_response_language`) — optional, e.g. `Czech`. When set, camera image descriptions (chat camera tool, `save_and_analyze_snapshot`, and the proactive video analyzer) are requested in that language. Leave empty for English. The internal `Scene unchanged.` repeated-scene reply is deliberately kept in English — it is matched by code, not shown to users (see [Camera Entities](camera-entities.md)).
- **Additional camera analysis instructions** (`vlm_prompt_extra`) — optional multiline text appended to the VLM prompt, e.g. `Ignore cars in the driveway`. Appended after the built-in rules, never replacing them; where your instruction conflicts with or narrows the built-in description request, your instruction takes precedence (the `Scene unchanged.` contract always still applies). It is restated on the per-image request itself because chat-tuned VLMs can ignore system-prompt-only instructions; when the chat agent analyzes a camera for specific objects you asked about in conversation, that live request is left untouched.
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

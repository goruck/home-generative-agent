# Configuration

Configuration is done entirely in the Home Assistant UI using subentry flows. A *subentry* is a discrete, independently configured capability — for example a Model Provider, a Feature set, or Sentinel. Each subentry has its own settings and can be added, reconfigured, or removed without affecting others.

The configuration UI is available in English, Czech, Simplified Chinese, and Turkish, with a partial Russian translation. Text follows your Home Assistant language settings and falls back to English for untranslated strings. To contribute a new language, see [Translations](contributing.md#translations).

- [Basic Setup](#basic-setup)
- [Model Providers](#model-providers)
- [Features](#features)
- [Tool Retrieval (RAG)](#tool-retrieval-rag)
- [Control Home Assistant (LLM API)](#control-home-assistant-llm-api)
- [Speech-to-Text (STT)](#speech-to-text-stt)
- [Text-to-Speech (TTS)](#text-to-speech-tts)
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
   - **Basic** — enables anomaly alerting with recommended defaults. Prompts for notify service, daily digest, and an optional level-increase PIN. A PIN entered here is required before the `sentinel_set_autonomy_level` service can raise the autonomy level above both the current level and the configured one (setting the level in Advanced setup does not ask for it).
   - **Advanced** — exposes all Sentinel options: intervals, cooldowns, quiet hours, triage, baseline, discovery, the Home Assistant & network security audit (master switch and thresholds), camera entry links, and per-entity rule exclusions. Every option is listed in the [Sentinel option reference](sentinel.md#configuring-discovery), and the security audit's rules and the router (eero) setup notes are under [Home Assistant & network security](sentinel.md#home-assistant--network-security).

> **No database?** The entry loads without a Database subentry, but only with in-memory storage: conversation history is lost on every reload or restart, long-term memory and face recognition are unavailable, and the Sentinel baseline anomaly detector is off (rules, notifications, and the audit trail keep working). Two paths lead there: **+ Setup** was never run after the integration was added, or in Advanced mode the wizard was closed at the final **Database setup** step (the feature subentries are saved before it, for example after a "cannot connect" error). Once a Model Provider exists, Home Assistant shows a repair issue under **Settings → Repairs** and the log carries a warning; run **+ Setup → Advanced** again and complete the Database step to fix it. With the bundled PostgreSQL with pgvector app the defaults (`localhost`, port `5432`, database `ha_db`, user `ha_user`) are correct.

> **Reconfiguring:** Running **+ Setup** or **+ Sentinel** again when a subentry already exists opens the same mode selector. Advanced mode pre-populates every field with the current saved values. Basic mode always starts from recommended defaults and warns before overwriting.

> **Removing Sentinel:** Delete the Sentinel subentry from the integration page to stop all monitoring immediately. Sentinel background tasks stop, the health sensor transitions to `disabled`, and the network audit's persisted auth inventory and pseudonymization salt are deleted.

---

## Model Providers

Supported providers and their default models:

| Category | Provider | Default model | Purpose |
|---|---|---|---|
| Chat | OpenAI | gpt-5 | Reasoning and planning |
| Chat | Ollama | gpt-oss | Reasoning and planning |
| Chat | Gemini | gemini-3.5-flash-lite | Reasoning and planning |
| Chat | Anthropic | claude-sonnet-4-6 | Reasoning and planning |
| Chat | OpenAI Compatible | gpt-4o | Reasoning and planning |
| VLM | Ollama | qwen3-vl:8b | Image scene analysis |
| VLM | OpenAI | gpt-5-nano | Image scene analysis |
| VLM | Gemini | gemini-3.5-flash-lite | Image scene analysis |
| VLM | Anthropic | claude-sonnet-4-6 | Image scene analysis |
| VLM | OpenAI Compatible | gpt-4o | Image scene analysis |
| Summarization | Ollama | qwen3:8b | Context summarization |
| Summarization | OpenAI | gpt-5-nano | Context summarization |
| Summarization | Gemini | gemini-3.5-flash-lite | Context summarization |
| Summarization | Anthropic | claude-haiku-4-5-20251001 | Context summarization |
| Summarization | OpenAI Compatible | gpt-4o | Context summarization |
| Embeddings | Ollama | mxbai-embed-large | Semantic search |
| Embeddings | OpenAI | text-embedding-3-small | Semantic search |
| Embeddings | Gemini | gemini-embedding-001 | Semantic search |
| Embeddings | OpenAI Compatible | text-embedding-3-small | Semantic search |

**Embedding model selection:** Embeddings are configured like any other feature: enable the **Embeddings** feature under **+ Setup** (Advanced mode) and assign it a provider and model. The embedding provider can be completely separate from the chat provider — e.g. llama.cpp for chat and a dedicated llama.cpp or Ollama server for embeddings. When the Embeddings feature is disabled, the provider is chosen automatically: the Conversation provider if it supports embeddings, otherwise the first embedding-capable provider.

**Ollama embedding models must be pulled first.** Unlike a chat model, a missing embedding model does not fail where you would see it: memory search and tool retrieval just stop working. So when you save the Embeddings feature with an Ollama provider, HGA asks that server and any Ollama fallback providers for their installed models, and refuses a model a server does not have, naming the `ollama pull` command to run. Names are compared the way Ollama resolves them (case-insensitive, with or without `:latest`, `library/` or `registry.ollama.ai/`); if a server has the model under a name the check does not recognize (a custom registry, an alias, or an Ollama-compatible proxy that lists models differently), or you are about to pull it, submit the same name again to save it anyway. A model that is still missing at the next startup raises the repair issue below. **The model must also produce vectors of the memory store's size.** The store's vector column gets its size once, when it is first created, and keeps it: 1024 for an Ollama, OpenAI or Gemini primary, or the configured dimensions for an OpenAI-compatible one. A store first created by one provider keeps that size after you switch to another, so a model of any other size fails every write. The form asks Ollama for the model's vector size and the database for the store's, and refuses a mismatch, with separate advice when only a fallback server is wrong (it skips this check while the integration is not running or the database cannot be asked); if you are sure the server reports the size wrong, submit again to save it anyway. The form checks only Ollama models; the other providers are checked at startup (below). At startup, HGA checks every reachable Ollama server that embeds for this entry, including the default `mxbai-embed-large` when the Embeddings feature is off, and raises a repair issue under **Settings → System → Repairs** naming any server missing the model, one for a primary model of the wrong vector size (listing every server, since choosing another model fixes them all), and one for an Ollama fallback of the wrong size while the primary was measured at the right size (a different build of the model there); a server whose size could not be read decides nothing, so it leaves these issues as they are; each clears on the first reload after you fix it. A server that cannot be reached or does not return a readable model list is skipped, since the provider health check already reports it, and the size check is skipped, leaving any size issue as it is, when the database cannot be asked or an Ollama primary could not be measured.

Common Ollama embedding models by vector size:

| Size | Models |
|---|---|
| 1024 | `mxbai-embed-large` (default), `qwen3-embedding:0.6b`, `bge-m3`, `snowflake-arctic-embed2` |
| 768 | `nomic-embed-text`, `embeddinggemma` |
| 384 | `all-minilm` |

**Other providers' vector size** ([#760](https://github.com/goruck/home-generative-agent/issues/760)): OpenAI and Gemini always produce 1024-dimension vectors. An OpenAI-compatible server produces its model's own size, which HGA measures at startup with one embedding; its dimensions setting (`openai_compatible_embedding_dims`) only sets the size a new store is created with, and only while it is the primary provider, so set it to the model's real size. At startup, HGA compares a non-Ollama embedding primary with the store's size and raises a repair issue if it differs (a non-Ollama primary has no embedding fallbacks, see below). A primary that could not be measured leaves the issue as it is.

**Changing the store's vector size:** the store keeps its size and follows the primary provider, so this only moves the store to an OpenAI-compatible primary's size (any other primary always uses 1024, so an Ollama primary needs a 1024-dimension model): set its dimensions setting to the model's size, drop the store's vector table in the database (`DROP TABLE store_vectors;`, for example from the PostgreSQL app's console), and reload the integration. Sizes above 2000 are not supported: the store's search index cannot hold them. At startup HGA notices the table is missing and creates it again at the configured size, with a warning in the log. Memories and camera captions saved before then stay stored but no longer show up in search, since nothing embeds them again; the tool index is rebuilt at once. The same check sets the database up again when the Database subentry is pointed at a new, empty database or one of its tables was dropped. If a table cannot be created again, the error is logged and the integration still loads; only writes to that table fail.

**The store records which embedding model filled it** ([#855](https://github.com/goruck/home-generative-agent/issues/855)). A model change that keeps the vector size (switching the embedding primary from Ollama to OpenAI, both 1024; swapping `mxbai-embed-large` for `bge-m3`; pulling another build of the same Ollama tag) would otherwise go unnoticed while memory search returns wrong results. The database keeps one row, `hga_embedding_model`, with the provider type, model name, size and, for Ollama, the digest of the build's weights (the weights blob that `/api/show` names; a server that does not name it counts as an unknown build, never another one). A store with no record yet (every install upgrading to this) or with no vectors (a new store, or one whose vector table was dropped) records the current model. When the current embedding model differs, a repair issue names both, and the record notes that the current model now writes into the store too. Its fix re-embeds every stored vector with the current model's own provider (never a fallback) in the background: memory search and saving keep working meanwhile, a memory saved during the run keeps its own new vector, the log shows progress every five minutes, and a restart or reload resumes it where it stopped. After each batch of 64 it pauses for as long as the batch took, so the embedding server keeps about half its time for conversations; a large store takes many hours (about one embedding call per 64 stored memories and captions). A batch that still fails after its retries is embedded one memory or caption at a time: one the provider keeps rejecting (over its input limit, say: it fails twice, 10 seconds apart, while a short test text embeds after each failure, and both times the provider answers that the input itself is bad: HTTP 400, 413 or 422) has its old vector removed, so the item stays stored but memory search no longer finds it, and the log names it; the run carries on rather than stopping on the same batch at every resume. Any other failure (a timeout, a rate limit, a server that is overloaded or unreachable) stops the run with every vector kept, to resume later ([#857](https://github.com/goruck/home-generative-agent/issues/857)). A timeout, a rate limit or a provider that stops answering stops the run with every vector kept, as before. If the run stops (the provider stays down through its retries, for example), the repair issue comes back so it can be started again, and it resumes where it stopped. When it finishes, the store is recorded as the current model's. Switching the model back does not clear the issue, since memories and captions saved in between were embedded with the other model: switch back, reload, and then run the repair, which re-embeds the store with the model you switched back to. When the repair is confirmed, an Ollama primary's build is read again; if it changed since startup, the repair waits for a reload. Before the store is recorded, the build is read once more: a build pulled during the run, or one the server no longer names, stops it with the issue back, and after a build change the next run starts over. A build pulled while no re-embed runs is noticed at the next startup. An Ollama primary that answered at startup but whose build setup could not read (Home Assistant's startup is busy; setup waits up to 5 seconds for each build) has its build read again in the background before the record is checked: at once, then after 10 and 30 seconds, each read waiting up to 5 seconds. If it still could not be read (the server stopped answering, or does not name its weights), the log says so, and the record, a re-embed in progress and the issue are left as they are until the next reload or restart reads it. With an Ollama primary that was down at startup, they are left as they are too, and no re-embed starts; HGA reloads itself once the primary answers again, and checks then. If the database does not answer the record check, it is tried again after 30 seconds, then 1, 2, 5 and 10 minutes, since until it lands a model change is not noted; if every try fails, the log says so and a reload checks again ([#857](https://github.com/goruck/home-generative-agent/issues/857)). Alongside the record check, the Ollama embedding fallbacks that setup kept without a full build check (their build unread, or the primary's, so their comparison waited) have their builds read again with the same waits, and are then checked against the primary's build: one holding another build is removed from the chain and named in the fallback build issue below (an embedding call already running on it discards its answer), and one whose build still cannot be read stays in the chain, held, with a line in the log ([#860](https://github.com/goruck/home-generative-agent/issues/860)). A held fallback never answers an embedding call before its build is known to match ([#857](https://github.com/goruck/home-generative-agent/issues/857)): the first time the chain needs it (the primary failed), its build is read then and compared with the primary's, or while that is unknown with the build the store's record names. The same build lets it stand in from then on; another build removes it, with the fallback build issue below; if either build cannot be read now (a read from startup does not count, since the build may have changed since), the call goes without it and it is tried again a minute later. So with the primary down and no build to compare with, saving a memory or caption fails until the primary is back, as with no fallback, and memory search uses recent memories. If a removed fallback had answered any embedding call meanwhile (the store embeds a save and a search alike, so it may have written its build's vectors), the record notes it as a writer and the model-changed repair above is raised; its re-embed with the primary clears them, and a re-embed already running stops and comes back as the repair, which then starts over. The note is tried again after 10, 30 and 60 seconds if the database does not answer (a re-embed holds the record row for a batch at a time); a store with no record yet gets one naming the current model with the writer noted. If the note still fails, the log names the manual way out. No "fallback active" notice is sent for a removal, since nothing failed, and a chain down to its primary alone is tried on every call like the unwrapped provider it would have been, whatever its circuit breaker says. The check judges every kept fallback at once, after the last read: with two or more Ollama fallbacks, one that never answers keeps a known mismatch in the chain for up to about 55 seconds more, held, so it answers nothing meanwhile, and the chain's own check removes it sooner if the chain needs it. An unknown build decides nothing about the fallback build issue either: it is synced when a fallback was left out or removed, or once every build is known, so a read that fails never clears a standing warning; a fallback whose build stays unreadable keeps the issue an earlier startup raised about it until a reload reads both builds. A fallback left out at startup stays out until the next reload. A different vector size cannot be re-embedded in place; the size issue above covers it. The record cannot tell vectors an earlier fallback of another model mixed in before #751: a store that already holds them is recorded as the current model's.

Limits of the record ([#857](https://github.com/goruck/home-generative-agent/issues/857) tracks the cases still open): entries that share one database must use one embedding model, since the store holds one model's vectors (two entries with different models would each ask to re-embed it to their own). An OpenAI-compatible model is recorded by its name, not its server, so a different server serving another model under the same name is not noticed. While a re-embed runs, the database grows (each re-embedded vector leaves an old copy, in the table and in its search index, until PostgreSQL vacuums it), so a large store needs about as much free disk again as its vector table takes.

**Multiple providers:** You can add multiple Model Provider subentries and assign them per-feature. For example: a "Primary Ollama" provider for chat and a "Vision Ollama" provider for camera analysis. You can also mix types — a local vLLM server as **OpenAI Compatible** alongside an Ollama provider, or two cloud providers side by side (Anthropic for Conversation and Summary, Gemini for Camera Image Analysis and Embeddings).

Add each one with **+ Model Provider** on the integration page. The flow's first step asks **Edge** or **Cloud**, and that choice is what determines the provider types the next step lists — **Edge** offers Ollama and OpenAI Compatible, **Cloud** offers OpenAI, Gemini, and Anthropic. Reconfiguring an existing provider *replaces it in place* (the subentry keeps its identity, and every feature pointing at it moves with it), so use **+ Model Provider** — not reconfigure — when you want a second provider alongside the first.

**Anthropic has no embeddings endpoint.** It can serve Chat, VLM, and Summarization but never Embeddings, so it is absent from the Embeddings row of the table above and from the Embeddings feature's provider picker. If Anthropic is your only provider, add an embedding-capable one (OpenAI, Gemini, Ollama, or OpenAI Compatible) alongside it to keep long-term semantic memory working. A feature pinned to a provider that cannot serve its category is skipped at runtime in favor of a capable provider rather than failing silently.

### Provider Fallbacks

Feature setup can include an ordered list of fallback providers. A fallback applies only to that feature category, so a chat fallback does not automatically cover VLM, summarization, or embeddings.

**Embedding fallbacks must use the primary's model** ([#751](https://github.com/goruck/home-generative-agent/issues/751)). The memory store holds vectors of one model, and each provider type embeds with the one embedding model configured for it, so only another Ollama server behind an Ollama primary can be an embedding fallback (it embeds with the same configured Ollama model). A second provider of a non-Ollama primary's type is left out too ([#853](https://github.com/goruck/home-generative-agent/issues/853)): OpenAI and Gemini have one client each, so it would be the primary again, and two OpenAI-compatible servers can serve other weights under the same model name, which no check can tell apart. A fallback of another model would make memory search return wrong results, or fail every write when the vector sizes differ. An Ollama fallback server that holds another build of the model than the primary's server (another pull or quantization, told apart by the model's digest) is left out at startup too, with a repair issue naming it; pull the same build on both and reload. While the primary is down at startup, a fallback's build is compared with the one the memory store's record names instead; a primary that answered but whose build could not be read in time is not stood in for by the record (it may be stale after a pull on both servers), and its fallbacks wait for the background check. A server whose digest cannot be read is kept, as is every fallback when neither the primary's build nor a recorded one is known, but held: it stands in for the primary only once its build is read and matches (see the model record above). Each build is read with a 5-second wait, and a kept fallback whose build, or the primary's, startup could not read is checked again in the background shortly after (see the model record above): one holding another build is removed from the chain, with the same repair issue ([#860](https://github.com/goruck/home-generative-agent/issues/860)). The Embeddings feature form offers only other Ollama servers as fallbacks behind an Ollama primary, and setup leaves any other fallback out of the embedding chain: one you chose raises a repair issue naming it (open the Embeddings feature and save it to clear it), and one HGA added on its own (without the Embeddings feature set up, every provider that can embed is a fallback candidate) is left out with an info log. If the embedding primary is down and no fallback answers, saving a memory or a camera caption fails until it is back, instead of saving something search could never find; meanwhile memory search returns only the most recent memories, tool retrieval falls back to keyword matching, and HGA reloads itself once the primary answers its health check again; chat, camera analysis and summarization fall back as before. Memories saved earlier while a fallback of another model was active stay in the store with that model's vectors and can come up wrong in search; to clear them, follow "Changing the store's vector size" below, which drops every stored vector.

Fallbacks are evaluated at setup/reload time and at runtime:

- If the primary provider is unavailable when the integration starts or reloads, HGA selects the first usable configured fallback and logs `Fallback selected at setup ...`. A provider that a model category uses (as its primary or in its fallback chain) and that failed its startup health check is checked again once Home Assistant has finished starting, then after 30 seconds, 1, 2 and 5 minutes, and every 5 minutes after that while it stays down. As soon as one answers, HGA reloads itself so it is used again, and logs `... answered after failing its health check at setup; reloading Home Generative Agent`. This covers a provider that only timed out while Home Assistant was busy starting, and a primary that comes back online later: HGA switches back to it with that reload, which interrupts any reply in progress. These automatic reloads happen at most once every 5 minutes; if one leaves a provider still failing its check, the gap doubles each time, up to once an hour, until a reload passes every check. A provider whose key is rejected is checked once more after startup and then left alone until you fix its settings.
- If the active provider fails during a model call with a retryable error, HGA tries the next configured fallback provider for that call and logs the runtime fallback activation. Retryable failures include local transport/connectivity errors, timeouts, rate limits, transient provider/server errors, and a provider that does not have the configured model (HTTP 404, for example an Ollama model that was never pulled or a mistyped model name). That last one applies to the primary too: the call moves on to the next provider, and HGA logs an ERROR saying the provider does not have the configured model, so the misconfiguration stays visible while a fallback covers for it. Other client errors, such as a rejected API key, still stop the chain.
- If a provider stops answering at runtime (an Ollama server wedged under memory pressure accepts a request and never replies), the call does not wait out its whole time limit on it. For calls with a limit of 90 s or more, each provider in the chain gets half of the time left and the last one gets the rest: a conversation turn (180 s) gives the primary 90 s and the fallback the remainder, and a camera frame (90 s) 45 s each. Shorter calls (Sentinel triage, explanations, discovery, video summaries) keep the whole limit for the primary, as before, and move on to a fallback only on a fast failure such as a refused connection. A single provider with no fallback keeps the whole limit (all but a second of a limit of 40 s or more, so its own timeout is the one recorded). After 3 timeouts in a row, each after waiting at least 40 s for the provider (so a slow model missing triage's default 10 s limit, or a provider SDK giving up early, does not count; if you raise the triage timeout to 40 s or more, its timeouts do count), shared across chat, camera analysis and summarization, HGA marks the provider stalled and logs a warning naming it ([#828](https://github.com/goruck/home-generative-agent/issues/828)). While it is stalled every call skips it and goes straight to the next provider; with no other provider, the call fails at once with an error naming the stalled one instead of waiting. Every 60 s HGA sends the stalled provider a tiny request ("Reply with OK.", 60 s limit), a real model call rather than the setup health check, since a wedged server still answers that, and uses it again as soon as it answers. The `sentinel_health` sensor lists stalled providers in its `stalled_model_providers` attribute and its state is `degraded` while any is stalled. For an Ollama server, restarting it usually clears the stall. Embedding calls are not covered yet.
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

### Per-Model Thinking / Reasoning

The Conversation feature's model form has a **Thinking / reasoning** setting (and,
for providers that support one, a **Thinking budget** in tokens). Settings are
remembered **per model name**: switch the chat model from, say, Gemma (thinking on,
budget 512) to Qwen (thinking off) and back, and each model's configuration is
restored automatically — the conversation itself is unaffected.

What each choice sends depends on the provider:

| Provider | Off | On | Effort levels | Budget |
| --- | --- | --- | --- | --- |
| Ollama | `reasoning: false` | `reasoning: true` | passed through for gpt-oss models; other models treat any effort as On | not supported by the Ollama API |
| OpenAI Compatible (llama.cpp, vLLM, …) | `reasoning_effort: none` + `chat_template_kwargs: {enable_thinking: false}` | `chat_template_kwargs: {enable_thinking: true}` | `reasoning_effort` — free-form values (e.g. `xhigh`) pass through verbatim; the server owns the vocabulary | `thinking_budget_tokens` (recent llama.cpp servers; older servers ignore it) |
| OpenAI | not sent (cloud reasoning models cannot disable thinking) | not sent | `reasoning_effort` (known levels only); a model that rejects it is retried without it automatically | not supported |
| Gemini | `thinking_budget: 0` | `thinking_budget` (dynamic when no budget set) | not offered | `thinking_budget` |
| Anthropic | not sent (thinking is off by default) | extended thinking with `budget_tokens` | not applicable (only Off/On offered) | `budget_tokens`, clamped to 1024–32768; `max_tokens` is raised above the budget and temperature is pinned to 1, as the API requires |

**Provider default** sends nothing, leaving the server/model behavior unchanged.

Notes:

- Entering a **budget** while the thinking select is at Provider default counts
  as turning thinking **On** — a budget only means something with thinking
  enabled.
- Switching to a model with **no remembered entry** starts it at Provider
  default — the thinking values still on screen belong to the previous model
  and are never carried over. Save again to configure the new model.
- The chat model's thinking settings apply **everywhere the chat model is
  used** — conversation, Sentinel triage and discovery, and explanation
  generation — so a large budget also slows and bills those background calls.
- llama.cpp: a `--reasoning-budget` set on the server command line overrides the
  per-request budget. If a model ignores the toggle, check that its chat template
  supports `enable_thinking` (Qwen-style) or `reasoning_effort`.
- Gemini: only `thinking_budget` is sent — the pinned `google-ai-generativelanguage`
  protobuf cannot express `thinking_level`, so level control (Gemini 3-style) has
  to wait for the langchain-google-genai 4.x upgrade. A model that cannot disable
  thinking rejects **Off**; prefer **Provider default** there.
- Fallback providers follow the same per-model memory: a fallback model uses its
  own remembered entry if one exists, otherwise the provider default.

> **Models that pin temperature** — Some OpenAI models (o-series and other reasoning-style models) only accept their default `temperature`/`top_p` and reject any other value with a 400 error. When that happens, HGA logs a warning and automatically retries the call without the rejected parameter, so conversation, camera analysis, summarization, and Sentinel keep working. In a multi-provider fallback chain the retry is applied per provider; a provider that still rejects its sampling settings fails over to the next provider in the chain and counts toward the circuit breaker. To avoid the extra retry on every call, leave the feature's temperature at the model's supported default (`top_p` has no UI setting — its default is a code-only constant, see the [Constants Reference](constants.md)).

> **Gemini 3 temperature** — Google strongly recommends leaving Gemini 3 models at their default temperature of `1.0`; lower values can cause looping or degraded reasoning. Gemini accepts a low temperature without error, so HGA handles this at setup time instead: when a Gemini 3-family model is selected and the feature's temperature is still the recommended default (`0.2`, which predates Gemini 3), HGA sends `1.0` and leaves `top_p` unset so the API defaults apply. A temperature you have explicitly changed away from the default is sent as configured — with a warning in the log pointing at Google's guidance. Gemini 2.5 and earlier models always use the configured values. This applies to primary and fallback Gemini models alike; non-Gemini models in the same feature keep the configured temperature.

---

## Tool Retrieval (RAG)

> **Thanks to [1Jamie](https://github.com/1Jamie) for this feature!**

On startup the integration indexes all available tools as vector embeddings in PostgreSQL. Each turn, only the most relevant tools for the user's message are loaded into the agent's prompt — keeping context short and tool selection accurate.

The tool universe is per-request: Home Assistant exposes some Assist tools only for certain devices (the timer intents — `HassStartTimer` and friends — exist only when the requesting device supports timers), and MCP tools exist only while their server is reachable. The index reconciles tool *presence* with this automatically in both directions (a tool whose description or schema changes without renaming keeps its stored definition until the next indexing pass, as before). When a turn's device exposes tools the index has never seen, they are indexed inline before retrieval runs, so the very first "set a timer for two minutes" from a voice satellite already has `HassStartTimer` available for retrieval — no restart or repeat needed (selection still follows normal relevance ranking). Conversely, indexed tools that don't exist for the current turn (timer tools in a browser chat, tools of an MCP server that failed to load) are never bound.

A few tools bypass similarity ranking: `GetLiveContext` is always available (whenever the Assist API that provides it is loaded for the turn), and `add_automation` is guaranteed to be available whenever your message signals automation-creation intent — explicit wording ("automate...", "remind me every 30 minutes") or an action verb plus a when/if trigger clause ("turn on the porch light when motion is detected"). Read-only state questions ("check if the garage door is open") do not trigger it. These are appended on top of the retrieval limit, so they never crowd out ranked tools. Intent detection is English-only for now; see [Architecture](architecture.md#tools) for details.

Two options in the **Options** flow control this:

- **Retrieval Limit** (`tool_retrieval_limit`, default `5`) — maximum tools made available per turn. Raise if the agent misses tools on complex multi-step requests; lower to reduce prompt size.
- **Relevance Threshold** (`tool_relevance_threshold`, default `0.15`) — cosine similarity cutoff. Lower if the agent misses tools it should pick up; raise to tighten selectivity.

### Excluded tools

Retrieval is probabilistic: it ranks by semantic similarity, so a server exposing `web_search`, `web_search_images`, `web_search_news` and `web_search_videos` can hand the agent a near neighbour of the one you wanted. **Excluded tools** (`tool_exclusions`) is the deterministic complement — a picker in the **Options** flow listing every tool of every selected LLM API, grouped by API name. Anything you tick there is removed from the tool set the agent loads, so it is never advertised to the model, never retrieved, and never executed: a call to an excluded tool by name is rejected outright rather than run.

It applies to every selected API, not only MCP servers, so Assist's intent tools can be trimmed the same way. This is worth doing for local and smaller models in particular, where tool definitions consume a meaningful share of the context window.

The picker is *subtractive*, and that shapes three behaviours worth knowing:

- **Empty is the default.** No exclusions means every tool of every selected API is exposed — exactly the behaviour before this option existed.
- **New tools arrive enabled.** A tool a server adds later was never ticked, so it is available immediately without you revisiting the form. If you want it gone, tick it.
- **Exclusions survive an outage.** If a server is unreachable when you open the Options form, its tools cannot be listed — but your stored exclusions for it are kept and still ticked, labelled with *why* they are missing: `(not currently available)` for a server that is unreachable or gone from Home Assistant, and `(API not selected)` for one you simply unticked under **Control Home Assistant**. They keep applying, and deselecting one is always an explicit action on your part. If nothing at all can be listed and nothing is stored, the field is hidden rather than rendered empty, so saving the form cannot silently clear your selection.

Two details about what the picker shows:

- **Device-gated tools are included.** Home Assistant exposes some Assist tools only to devices that support them — the timer intents are the usual case, offered only when the request comes from a timer-capable voice satellite. An options form has no device, so those tools cannot be listed by asking Home Assistant directly; they are taken from the tool index instead, which records what the agent has actually seen. Without that they would be callable on every voice turn and impossible to switch off.
- **Tool names are shown defanged.** Names come from the server, not from you, so they are stripped of hidden characters, length-capped, and have their parentheses rewritten to square brackets before display. That last one matters: a server could otherwise name a tool `something (not currently available)` and have it render exactly like a tool that is switched off, so you would skip past a tool that is in fact live.

Excluded tools stay in the vector index (which is global and cumulative by design), so re-enabling one takes effect on the next turn without a re-index.

### Always-included tools

The relevance threshold cuts both ways: a genuinely general-purpose tool can score *below* it on exactly the queries it exists for. "Who won the FIFA World Cup?" shares almost no vocabulary with a `web_search` tool's description, so retrieval never selects it — and no system-prompt instruction can help, because the model cannot call a tool that was filtered out before it saw the list. Lowering the global threshold instead would drag in irrelevant tools on every turn.

**Always-included tools** (`tool_inclusions`) is the additive complement to the exclusions: a second picker in the **Options** flow, listing the same tools, where anything you tick is appended to the tool list *after* vector retrieval — on top of the retrieval limit, like `GetLiveContext`, so an inclusion never crowds out a ranked tool. The model still decides whether to actually call it.

It shares the exclusion picker's contract: empty is the default (nothing extra is bound), stored selections survive a server outage labelled with why they cannot be listed, and the field is hidden when nothing can be listed so a save cannot clear your selection. Three boundaries apply on every turn:

- **An exclusion always wins.** The form refuses a tool ticked in both lists, and the runtime enforces the same rule against configurations written programmatically — the exclusion is a security control and fails closed.
- **The API must be active.** An inclusion under an API you have deselected (or that failed to load this turn) stays unbound.
- **The tool must exist this turn.** An inclusion is reconciled against the live tool universe like every other candidate; a tool the server no longer advertises is skipped, not invented.
- **Inclusions survive the read-only strip.** On read-only state questions ("which doors are open?") the agent normally drops actuation tools from the turn's list; an always-included actuation tool stays bound anyway — always means always. The critical-action PIN still gates any critical call at execution time, and at most 16 inclusions are honoured per turn.

Each inclusion adds its schema and description to the prompt on every turn, so keep the list short — it is meant for the one or two fallback tools (typically web search) that similarity ranking systematically misses, not as a way around retrieval.

A **Tool Index Status** diagnostic sensor (`sensor.tool_index_status`) shows the current index state:

| State | Meaning |
|---|---|
| `indexing` | Embedding in progress — the first full run at startup, or a mid-turn top-up adding newly discovered tools |
| `ready` | Index available; tools retrieved per-turn by semantic search |
| `failed` | Embedding provider unreachable or its model missing; agent falls back to keyword ranking (see below) |
| `unknown` | Index state not yet reported |

While the index is unavailable (`failed`, or still building), each turn ranks every available tool by the words its name and description share with your request, and binds the best matches up to the retrieval limit, plus the tools that are always added (`GetLiveContext`, your always-included tools, and the history or automation tool when the request calls for it). Naming the tool is enough: "run the door history script" finds `script.door_history` even though Home Assistant lists scripts after all its built-in tools. On a request to control something, three quarters of the limit (at least one slot) is kept for control tools such as `HassTurnOn`, the same share the vector search gives them, so tools that only share words with the request cannot crowd them out; the remaining slots go to the best matches. Matching is on whole words (plurals ignored), so a request in a different language from the tool's name or description will not match it; ties keep the order Home Assistant lists tools in. The log line `Tool retrieval using keyword-ranked fallback` says why the index was skipped and how many tools matched.

The sensor's `tools_indexed` attribute reports the cumulative number of tools in the index — not the size of the last indexing batch — and `last_updated` records the time of the last successful index update.

Subsequent restarts skip unchanged tools using SHA-256 content hashing, so re-indexing is fast.

---

## Control Home Assistant (LLM API)

The **Control Home Assistant** option in the Options flow is a multi-select that controls which HA LLM APIs the agent can use.

- **Assist** (`assist`) — the built-in HA Assist API. Grants entity-control intents and the list of entities exposed to Assist. Select this for standard voice-assistant control.
- **MCP server integrations** — any [Model Context Protocol](https://www.home-assistant.io/integrations/mcp_server/) integration you have configured registers its own LLM API (e.g. `mcp-<entry_id>`). Those entries appear in the list once added.
- **Any other integration that registers an LLM API** — the list is every LLM API Home Assistant knows about, not a fixed set, so tools added by a third-party integration show up here too. This is how the agent gets [internet search](#internet-search).

**Only exposed entities are visible to the agent.** The Assist API hands the model the entities exposed to Assist — as the entity context in the system prompt and as what `GetLiveContext` returns — and nothing else. Home Assistant exposes new entities automatically only for a fixed set of domains and device classes (lights, switches, covers, fans, climate, media players, scenes, humidifiers, to-do lists, vacuums, water heaters; door, window, opening, garage-door, lock, motion and presence binary sensors; temperature, humidity, CO, CO₂, AQI, PM2.5, PM10 and VOC sensors). A leak (`moisture`), smoke, gas, vibration or tamper sensor, a lock, a battery level or a power meter is not exposed by default, so the agent answers "I don't have a sensor for that" even though the entity exists. Expose it under **Settings → Voice assistants → Expose**, and name it and assign an area the way you will ask about it, since the model matches on friendly name and area. This applies to current state only: HGA's own `get_entity_history` tool looks entities up by exact friendly name across all of Home Assistant, but the model only learns names from the exposed list, so in practice an unexposed entity is unreachable through both. Sentinel does not use the exposed list; its snapshot reads every entity in Home Assistant.

You can select any combination. Selecting both Assist and one or more MCP APIs merges all their tools into a single combined API. Note that deselecting everything does **not** disable HA control: an empty selection is stored as "unset", which the agent reads as the Assist default, so Assist is silently re-enabled on the next save. Running with no LLM API at all is currently not expressible through the form (tracked in `TODOS.md`).

**Adding an MCP server:**

1. Go to **Settings → Devices & Services → Add Integration** → search **Model Context Protocol**.
2. Enter the server URL and complete setup.
3. The MCP integration registers an LLM API automatically.
4. Open **Settings → Devices & Services → Home Generative Agent → Configure**.
5. Select the new entry in **Control Home Assistant** and save.

**Removing an MCP server:** if a selected server's integration is removed (or is temporarily unavailable), its entry stays selected and is shown as `<id> (no longer available)` so the form remains saveable and your selection is never dropped behind your back. Deselect the dead entry and save to clean it up. While it stays selected, a warning is logged when the options form is built and each time the agent loads its APIs; other selected APIs keep working, but if the unavailable entry is the *only* selection, conversations fail with "No LLM APIs could be loaded" until the server returns or you deselect it.

### Internet search

The agent has no web search of its own; it uses whichever search tool you give it through an LLM API, so you choose the search provider and nothing leaves your network unless you pick a hosted one. Two ways to add one:

- **[Tools for Assist](https://github.com/skye-harris/llm_intents)** (HACS custom integration) — web search backed by [Brave](https://brave.com/search/api/) or a self-hosted [SearXNG](https://docs.searxng.org/) instance, plus Wikipedia, weather, and places tools. It registers each tool group as its own LLM API; the web search one is named **Search Services**.
- **A search MCP server** — add it as described under **Adding an MCP server** above.

Then:

1. Install and configure the integration or MCP server.
2. Open **Settings → Devices & Services → Home Generative Agent → Configure**.
3. Select the new entry in **Control Home Assistant**, keeping **Assist** selected, and save.
4. If [Tool Retrieval (RAG)](#tool-retrieval-rag) is on, tick the search tool under [Always-included tools](#always-included-tools). A general-knowledge question shares almost no vocabulary with a search tool's description, so retrieval alone rarely selects it.

Ask something the model cannot know ("who won the match last night?") to check it. If a server exposes several near-identical search tools, use [Excluded tools](#excluded-tools) to leave only the one you want.

---

## Speech-to-Text (STT)

HGA provides a built-in STT engine — no separate STT integration required. Two provider types are supported: **OpenAI** (Whisper API) and **Local** (any OpenAI-compatible transcription server, e.g. [Speaches](https://speaches.ai/) serving faster-whisper). A local server can run on any machine Home Assistant can reach; it does not have to be the box that serves your LLM.

1. Open **Settings → Devices & Services → Home Generative Agent**.
2. Click **+ STT Provider**.
3. Choose **OpenAI** or **Local (OpenAI-compatible)** and give it a name.
4. On the **Credentials** step:
   - **OpenAI:** either reuse an existing OpenAI Model Provider subentry or select **Use a separate key** and enter a dedicated API key.
   - **Local:** enter the server URL (e.g. `http://192.168.1.100:8000` — a missing `/v1` suffix is added automatically). The API key is optional; leave it blank for servers without authentication. The endpoint is validated when you submit the form: HGA asks the server for its model catalog (`/v1/models`), and a server that does not serve one is accepted as long as it serves `/v1/audio/transcriptions`.
5. On **Model & advanced options**, pick a model and set optional fields:
   - model: recommended `gpt-4o-mini-transcribe` (OpenAI) or `deepdml/faster-whisper-large-v3-turbo-ct2` (Local; any custom model ID your server exposes can be typed in)
   - `language` (optional): e.g. `en` or `en-US`
   - `prompt` (optional): hints for domain-specific vocabulary
   - `temperature` (optional): 0–1
   - `translate`: on OpenAI only `whisper-1` supports it (other models fall back to transcription); local whisper servers support it for every model
   - `extra request body` (optional): a JSON **object** merged into the request body, for parameters HGA does not expose as fields. It is merged last, so it can also override a parameter set above (`model` or `response_format`, say). It cannot set `stream`, `file` or `input_audio` — HGA owns how the audio and the response are carried. Invalid JSON, JSON that is not an object, or one of those three keys keeps you on the form.
   - `request format` (Local only): how the audio goes on the wire — see [Provider-specific request options](#provider-specific-request-options) below
6. Go to **Settings → Voice assistants → Assist pipelines** and select **STT - OpenAI** / **STT - Local** (or your chosen name) for Speech-to-text.

**Credential changes take effect on the next utterance.** Each STT entity keeps one OpenAI client, built on Home Assistant's shared HTTP client, and rebuilds it when the resolved API key or server URL changes — no restart or integration reload needed. If the entry is linked to a Model Provider subentry, that provider's key is the only one used: linking blanks the separate STT key, so a linked provider without a usable key fails the utterance with an `STT API key missing` warning in the log rather than falling back to a stale key.

### Provider-specific request options

Some endpoints accept transcription parameters that are not part of the OpenAI API — keyword biasing, speaker diarization, vocabulary hints. The **extra request body** field passes them through, but *how* the request is sent decides whether they arrive.

**Multipart upload (the default)** is OpenAI's own shape: the audio is uploaded as a form file. Use it for OpenAI and for local servers such as Speaches, which accept only this shape. Extra body fields become form fields, which works for flat values a local server understands (e.g. `hotwords`) but not for nested objects.

**JSON with base64 audio** posts a JSON body with the audio inline under `input_audio`. This is [OpenRouter's native transcription shape](https://openrouter.ai/docs/guides/overview/multimodal/stt); their multipart endpoint is an OpenAI compatibility layer that supports only `file`, `model`, `language`, `temperature`, `response_format` and `timestamp_granularities`, so nested `provider.options` never arrives over multipart no matter how it is encoded. Two consequences worth knowing before you pick it:

- There is **no translations endpoint** for this shape, so `translate` degrades to transcription with a warning in the log.
- The **`prompt` field is not sent** — this shape has no such parameter. (OpenRouter ignores `prompt` on multipart too, so nothing is lost there.) If your endpoint wants one, put it in the extra request body.
- The extra request body **cannot set `stream`, `file` or `input_audio`**. These decide how the audio and the response are carried rather than how the audio is transcribed, and overriding them cannot do anything useful: an endpoint that honours `stream` replies with an event stream, which would be handed back as the transcript itself with no error.

Example: biasing OpenRouter's `microsoft/mai-transcribe-2` towards names its model mishears. Set **request format** to *JSON with base64 audio* and put this in **extra request body**:

```json
{
  "provider": {
    "options": {
      "azure": {
        "phraseList": {
          "phrases": ["Frigate", "Proxmox", "Réaltín"]
        }
      }
    }
  }
}
```

The key under `options` is the provider slug that actually serves the model, which you get from `https://openrouter.ai/api/v1/models/<model>/endpoints` (the `tag` field). Only the options for the provider serving the request are forwarded, and the field names are that provider's own — so check their transcription API reference, not OpenAI's. Providers differ in how they treat unknown fields: some drop them silently, others return an error.

Expect biasing to *steer* recognition rather than guarantee a spelling. A name the model cannot spell may still come back approximated — what matters for a voice assistant is that it lands close enough for the conversation agent to match the entity, which is usually where a phrase list gets it. List the terms themselves (names, rooms, devices) rather than whole sentences, and add a term's common inflections if you use them out loud.

> **Note:** the JSON format is offered on the **Local (OpenAI-compatible)** provider type only, because the OpenAI API itself accepts multipart alone. Configure OpenRouter as a Local provider with the server URL `https://openrouter.ai/api/v1` and your OpenRouter key.

### Running a local STT server

Any server exposing the OpenAI `/v1/audio/transcriptions` endpoint works. Speaches runs on any machine Home Assistant can reach — a dedicated box, the LLM server, or a spare machine with an older GPU. `faster-whisper-large-v3-turbo` needs roughly 1.5–2 GB of VRAM and transcribes a short command in well under a second on even a Pascal-class card; without a GPU pick a smaller whisper model (`Systran/faster-whisper-small`, for example) and the CPU image (`latest-cpu`). Speaches only serves models that have already been downloaded, so list the model in `PRELOAD_MODELS` (or `POST /v1/models/<model-id>` once) before the first utterance:

```yaml
# docker-compose.yaml on the speech server
services:
  speaches:
    image: ghcr.io/speaches-ai/speaches:latest-cuda
    ports:
      - "8000:8000"
    volumes:
      - hf-hub-cache:/home/ubuntu/.cache/huggingface/hub
    environment:
      - PRELOAD_MODELS=["deepdml/faster-whisper-large-v3-turbo-ct2", "speaches-ai/Kokoro-82M-v1.0-ONNX"]
      - STT_MODEL_TTL=-1            # keep the model resident; the first utterance after a reload pays several seconds
      - WHISPER__TTL=-1             # same setting under the name older Speaches images read; harmless to set both
      # - WHISPER__COMPUTE_TYPE=int8_float32   # required on Pascal GPUs (GTX 10xx): float16 is not supported there
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
volumes:
  hf-hub-cache:
```

> **Tip:** local whisper models occasionally hallucinate short phrases ("Thank you.") from silence or noise. The **Speech input filters** in the integration options (**Configure**) drop such phantom transcriptions before they reach the agent — add the phrases you see under **Ignored exact STT phrases**.

---

## Text-to-Speech (TTS)

HGA also provides a built-in TTS engine for Assist pipelines, so a reply can be spoken by the OpenAI speech API or by a local server — the same Speaches container that handles speech-to-text can serve both, but the two providers are configured independently and can point at different machines. Two provider types are supported: **OpenAI** and **Local** (any server exposing the OpenAI `/v1/audio/speech` endpoint, e.g. [Speaches](https://speaches.ai/) serving Kokoro or piper voices).

1. Open **Settings → Devices & Services → Home Generative Agent**.
2. Click **+ TTS Provider**.
3. Choose **OpenAI** or **Local (OpenAI-compatible)** and give it a name. The name is what the Assist pipeline's Text-to-speech dropdown shows.
4. On the **Credentials** step:
   - **OpenAI:** reuse an existing OpenAI Model Provider subentry or select **Use a separate key** and enter a dedicated API key.
   - **Local:** enter the server URL (e.g. `http://192.168.1.100:8000` — a missing `/v1` suffix is added automatically). The API key is optional; leave it blank for servers without authentication. The endpoint is validated when you submit the form: HGA asks the server for its model catalog (`/v1/models`), and a server that does not serve one — several TTS-only servers, such as [Chatterbox-TTS-Server](https://github.com/devnen/Chatterbox-TTS-Server), answer 404 there — is accepted as long as it serves `/v1/audio/speech`.
5. On **Model, voice & advanced options**:
   - model: `gpt-4o-mini-tts` (recommended), `tts-1`, or `tts-1-hd` for OpenAI; `speaches-ai/Kokoro-82M-v1.0-ONNX` (recommended) or any model ID your server serves for Local
   - voice: OpenAI offers `alloy`, `ash`, `ballad`, `cedar`, `coral`, `echo`, `fable`, `marin`, `nova`, `onyx`, `sage`, `shimmer`, and `verse`; for Local, type a voice id the model provides (Kokoro: `af_heart`, `af_bella`, `am_adam`, `bf_emma`, `bm_george`, …; piper: the voice name from the model id, e.g. `hfc_female` for `speaches-ai/piper-en_US-hfc_female-medium`)
   - speed: 0.25–4.0, default 1.0
   - voice instructions (optional): tone and pacing hints, used only by OpenAI's `gpt-4o-mini-tts` models and ignored elsewhere
6. Go to **Settings → Voice assistants → Assist pipelines** and select **TTS - OpenAI** / **TTS - Local** (or your chosen name) for Text-to-speech. The configured voice is the pipeline's default; the pipeline's own voice picker lists the OpenAI voices, or the single configured voice for a local server.

Replies are spoken as they are written. The Assist pipeline hands the engine the agent's reply while the agent is still producing it, and HGA synthesizes it a sentence at a time: the first sentence alone, so the satellite starts talking as soon as it is complete, then everything that arrived while that was playing in one request. A long answer therefore starts speaking after its first sentence instead of after the whole reply, and a sentence the agent writes before it stops to call a tool is spoken while the tool runs. The pipeline begins streaming once the reply passes about 60 characters, or as soon as the agent writes any text before calling a tool; a short reply is still spoken in one piece. A message that arrives whole, such as a `tts.speak` announcement, is still synthesized in one request in the format the player asks for. Streamed replies are requested as WAV and sent as one stream (a single header, then each sentence's audio), and Home Assistant converts to whatever the satellite needs, so a Voice PE's request works with either provider. The first sentence is followed by a moment of silence: Home Assistant's converter passes nothing on until it has 64 KB of audio, and without the padding a short first sentence would wait for the next ones. While the agent is still working, the stream carries silence paced to real time, and the first sentence always ends in a quarter second of silence: Home Assistant's converter holds back the last ~50 ms of audio until more arrives, and without that a long first sentence's last syllable played seconds later, just before the answer. Speech is never held back by more than 0.3 s. Credential and server changes take effect on the next reply, like STT.

> **Tip — fill the silence on tool-using questions.** A "why" question makes the agent call tools before it writes any answer, and a local model often calls them without saying anything first, so the satellite stays quiet until the answer is ready. Set **Spoken acknowledgement (voice only)** in the integration options (**Configure**), for example `Let me check.`, and a voice satellite says it while the agent works. **When to speak the acknowledgement** picks the moment. *As soon as the agent starts* (the default) is heard about a second after the request reaches the agent and covers the whole wait, including a local model's seconds of reading its input before it writes anything, but it is also said before replies that need no tools, so prefer neutral wording such as `One moment.`. *Before the first tool call* is said only on turns that use tools, after the model has started, and only when the model has not already said something itself; that suits a cloud model, whose first word is quick but whose tools are slow. On a **Home Assistant Voice PE**, the early moment can produce a brief click partway through the wait: the satellite enters its replying phase at the start of the turn and then sits through seconds of silence while the model works (in a 2026-10-01 trial, clicks on 4 of 8 turns with the early moment, 0 of 8 with *Before the first tool call*, and 0 of 6 with no acknowledgement). The audio HGA sends is clean; the click comes from the device. If it bothers you, choose *Before the first tool call*. Either way it applies only to turns from an Assist satellite (a typed chat gets none) and at most once per reply, and the early moment is not used with Schema-first YAML mode, whose replies are not streamed; a phrase with no closing sentence mark gets a full stop so the speech engine does not hold it back; commands your pipeline answers with local intents never reach the agent and are unaffected. Write it in your assistant's language. It needs a streaming text-to-speech engine, such as this one. Leave it empty (the default) to turn it off.

### Running a local TTS server

The Speaches container from the [STT section](#running-a-local-stt-server) serves TTS too. Speaches only serves models that are already downloaded, so add the TTS model to `PRELOAD_MODELS` (the compose example above lists Kokoro) or download it once with `POST /v1/models/<model-id>`:

```bash
curl -X POST http://192.168.1.100:8000/v1/models/speaches-ai/Kokoro-82M-v1.0-ONNX
```

Two things to know about the local backend:

- **Kokoro needs a reasonably modern machine.** On a modern CPU (AVX2) or a recent GPU it is fast and sounds good. On an old CPU without AVX2 it can take several times the audio duration to synthesize a sentence, and ONNX Runtime's CUDA path does not help it on Pascal-class GPUs. In that case use a piper voice instead — `speaches-ai/piper-en_US-hfc_female-medium` (voice `hfc_female`) or a `-low` variant — which synthesizes a sentence in about a second on CPU and needs no GPU. Browse the available models with `GET /v1/registry?task=text-to-speech`.
- **Speaches does not produce opus or aac.** HGA never asks it to; a pipeline that wants those gets mp3 converted by Home Assistant.

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
- Opening covers whose `entity_id` includes `door`, `gate`, or `garage`, whose device class is `door`, `gate` or `garage`, or whose name, an alias or its device's name has one of those words or its plural with no letter next to it ("Garage Door", "Driveway Gates", "Gate1", not "Indoor Blinds"), unless its device class is blind, shade, curtain, awning or window — via `open_cover`, `open`, `toggle`, or `set_cover_position`, since all four open a closed door. A group holding such a cover (a cover group such as `cover.all_covers`, an MQTT or ZHA group, or an old-style group, nested to any depth) is held the same way, since opening it opens the garage door. This applies to every rule with an `entity_match`: a rule you add for `msg100` also holds a group containing `cover.msg100_abc`, and a word written like an id (`garage_door`) matches the name "Garage Door". Covers and groups that are not set up yet count too, because an automation the agent writes runs later: a cover not set up yet with no device class, a group member Home Assistant does not know yet, and a group helper whose members cannot be read are held; a disabled entity is not. Membership is read when the automation is written: a garage door added to a group afterwards is not checked again. Not held: a cover with no device class whose entity id and name both lack the words (a name in another language or a run-together word such as "GarageDoor"); and a cover whose only link to the words is the name of its area (a cover named "Main" in the "Garage" area). Add a rule naming part of the entity id for those.
- The switch a held cover, valve or lock is built on. A relay wrapped as a cover, valve or lock with **Change device type of a switch** (`switch_as_x`) moves the door or opens the lock when it is switched, so a call on the relay asks for the PIN whenever a rule holds that cover or valve, or unlocking that lock: whether the agent calls the switch's own service, `homeassistant.turn_on`, "turn on" by name, a scene or a script, or writes it into an automation. For a cover or valve every call on the relay is held (on, off and toggle), since a momentary garage relay moves the door whichever way it is switched; for a lock, the unlocking way (on, or off with the helper's invert option) and toggle. Also held: a group holding the relay, each relay in a switch group the helper wraps, any other wrapper of any of those (the same switch also shown as a light, say), a switch not set up yet (it could be a group holding the relay), and a wrapper you disabled, since disabling it does not disconnect the relay. Closing the cover itself, or a group of covers holding it, is not held, as closing is not. When the call names the relay, a group holding it or another wrapper of it, the PIN prompt says what it controls ("turn off Garage Relay (controls Garage Door)"). Not screened: the brightness and fan-speed commands ("set the light to 50%") on another wrapper of the relay shown as a light or fan. Like covers and locks, once a relay under a held cover, valve or lock exists, a switch call whose targets cannot be named ahead (an area, device, floor or label, `all`, or a template) is held, since the relay, or something else there that drives it, may be among them. A target naming the relay by its registry id is held, and one naming another entity's registry id is not. A switch device action in an automation acts on the entity it names, so it is held only when it names the relay, by entity id or registry id ([#804](https://github.com/goruck/home-generative-agent/issues/804)). A group helper that is offline or not yet loaded is read from its settings, so turning on one that holds the relay, another wrapper of it, or a template switch that switches it is held too. The wrapper's settings are read when the check runs: an automation written before a relay was wrapped, or before its invert option changed, is not checked again. A template cover or lock (from the **Template** integration) is followed the same way when its actions drive other devices: for a held template garage cover whose open or close action switches `switch.garage_relay`, every call on the relay asks for the PIN, and so does every call on anything else its actions target, whether named by entity id, registry id, device, area or label, or set by `scene.apply`. A script, scene, template entity or button among them is followed into what it runs (a scene's states, a button's automations), keeping the switches, covers, valves, locks, buttons and scripts it drives (not, say, the lights a script also turns on). For a template lock, the entities its unlock and open actions drive. Targets its actions cannot name ahead (a template) are not followed, nor are those of a template cover that is not loaded, though the template cover or lock itself stays held.
- Using HA intent tools on locks
- Using HA intent tools on covers and valves the same way: "turn on" opens a cover and "set position" moves it, so the rules above apply to them. The PIN check finds the devices the way Home Assistant will, by name, alias, area or floor, and applies a rule's `entity_match` to each device's `entity_id`, so "open the garage door" is held whether the agent names it, gives its entity id, or names only its area. The PIN prompt names every action such a command takes, so one that would lock a lock and open a gate says both. Rules you add for other devices apply to these commands too: "stop" stops a cover or valve (`stop_cover`, `stop_valve`), "turn on" presses a button (`button.press`) and runs a script (a rule for `script.turn_on` or for the script's own service, such as `script.open_garage`, both match when the agent turns the script on or runs it as its own tool)
- **Running a script or activating a scene that does any of the above.** Turning on a script, or running it through the tool Home Assistant gives each exposed script, is held when one of its steps would be held in an automation (see **Automation screening** below): a script that opens the garage door asks for the PIN, one that turns on the porch lights does not. A step that runs another script or scene, fires an event, or builds its action from a template is held too, because what it does is not known until it runs. A scene from Home Assistant's own scene config is checked per entity it sets: one that unlocks the front door or opens the garage door asks for the PIN, a bedtime scene that locks the doors and closes the garage does not. A rule's `entity_match` is applied to each entity the scene names, and the members of an old-style group it sets are checked the same way. A scene stored by another integration cannot be read, so it is held, except for lighting systems whose scenes only set their own lights (Hue, LIFX Cloud, LiteJet, Sunricher DALI). A hub's scene (Fibaro, Vera, Overkiz, MQTT, KNX, deCONZ and others) can open a gate or unlock a door, and a shade system's scene (Hunter Douglas PowerView) moves covers a rule can guard, so these ask for the PIN. A tool that calls services by name, such as an MCP service tool, is checked the way an automation step is (see **Automation screening** below), except that a script or scene it runs is opened as above rather than held. A target of `all`, an area, a label, a group or a registry id is held when a rule applies to the call's domain and service (for a call on `light.turn_on` with an area and no light rule, it runs), and always for `homeassistant.*` calls and calls that run a script or scene. `homeassistant.toggle` on a garage door is checked as `cover.toggle`, a call that gives no domain is also checked as `homeassistant.<service>`, wherever its targets sit (`entity_id`, `entity_ids`, `target`, `data`), when it turns something on or toggles it, reaches several domains, or targets an area, device or label (held, since which devices act is unknown), a field given twice with two values (at the top of the call and inside `data`) is held, and `automation.trigger` is held. A script or scene whose configuration cannot be read is held. The PIN prompt names the script as Home Assistant knows it, not from what the agent passed to it.
- **Turning on a template switch, setting a template light, or pressing a button that does any of the above.** An entity from the **Template** integration runs its own actions: a template switch its `turn_on` or `turn_off` action, a template button its press action, a template light the action for what it is asked (a brightness runs `set_level`). Those actions are checked like a script's steps, except that a script, scene or button they run is opened rather than held (a template TV switch whose `turn_on` runs a harmless `script.tv_on` does not ask for the PIN; one running a REST or shell command does, since what that does cannot be read), whether the agent turns it on or sets it by name ("turn on", "set the light to 50%", a fan speed, a vacuum, media or any other device command; only intents that read or time things are left alone, and one the PIN check does not know that names no device is held), calls its service, sets it in a scene (stored or `scene.apply`) or a script, or writes it into an automation, including as a device action: a template button whose press opens the garage door asks for the PIN, one that turns on a light does not. For a template switch, button, lock or cover only the action for the call is checked (turning a template switch off does not check its `turn_on`, though `toggle` checks both); for any other template entity every action it has is checked, since it picks one from the call's settings. A group holding a template entity is checked through its members, and a light or other helper made from a template switch with **Change device type of a switch** through the switch. A call's targets are read as Home Assistant reads them, including from its data blocks: a registry id, an area, a device, a floor or a label is turned into the entities it holds, and a call aimed at `all` or a templated target is held when any entity of its kind runs actions that would be held, since it may be among them. Pressing a button or an input button starts the automations that react to it, and the trigger-based template blocks with `actions:` that react to it, and resumes a script or automation waiting for it, so the press is checked against each of those whose trigger, `wait_for_trigger` or `wait_template` names the button (by entity id or registry id, or through the automation's variables) or cannot be tied to entities ahead: a `state_changed` or `call_service` event trigger (one filtered in its conditions can react to the press), an event trigger whose type is templated, a template reading states generically (`states.input_button`, `expand(...)`, `area_entities(...)`), a templated entity or event filter, an area, device, label or floor target, or a device trigger for buttons or the button's own device. A template naming another entity (`states('person.me')`), another event type (a ZHA or mobile notification event) and a remote's device trigger cannot come from a press and do not count. A template alarm panel is checked the same way: it enforces no code of its own, so arming or disarming one whose actions would be held asks for the PIN, while a real panel keeps using its own code. An entity or automation whose actions cannot be read is held.
- **Changing anything an automation reacts to by doing any of the above.** Turning on an input boolean, a switch or a light, pressing a button, running a script or scene, or marking a phone home (`device_tracker.see`) or publishing to MQTT through a service tool, can start an automation you wrote: turning on `input_boolean.let_in` whose automation unlocks the front door asks for the PIN, by voice, through a service tool, from a script or scene the agent runs, or in an automation it writes. Each entity the call changes, and any group holding it (a person's trackers count toward the person, and people toward the zone they are in), is checked against the automations (and scripts waiting on it, and trigger-based template blocks) whose triggers name it, by entity id or registry id, or cannot be tied to entities ahead: a `state_changed` or `call_service` event trigger (unless its filter names other entities or another domain), a template reading that kind of entity generically (`states.light`, `expand(...)`, an id built from parts), a templated entity, an area, device, label or floor target, or a device trigger for its kind or its own device. A disabled automation's triggers do not count, and a wait counts only while its script or automation is running. Such an automation is yours, so it is held only for what can be read as protected (an unlock, a garage opening, or a script, scene or automation it runs that does one); a step that cannot be read ahead, such as an event it fires, a templated service or a REST command, does not make the change ask for the PIN, here or in a script it runs. Automations a button press starts, and the actions of a template entity such an automation acts on, are still read strictly, as above. `mqtt.publish` is held when an automation with an MQTT trigger (or MQTT device trigger, or a `call_service` event for `mqtt`) or one watching an MQTT entity would, or when a rule guards an MQTT entity, since the topic is not matched. A trigger's `to:` and `from:` are not read, so turning such a helper off asks for the PIN too. Automations started by another event (a phone notification's button, a remote) or watching other entities are not counted. Entities computed from the changed one are followed the way Home Assistant itself tracks them: a template entity whose template reads it now (a template reading it only while another entity is on is followed only then), a trigger-based template entity it triggers, and any helper subscribed to its changes, such as min/max, threshold, bayesian, statistics, filter, utility meter or a switch_as_x wrapper, set up in YAML or the UI ([#820](https://github.com/goruck/home-generative-agent/issues/820)). A chain of these is followed up to ten steps and held past that. If a Home Assistant update renames what this reads, a warning is logged and derived entities are not followed until it is fixed. A template that imports a macro is followed for every change, since what the macro reads is not in its text. Not followed: a change that reaches another entity only through MQTT, such as MQTT Statestream republishing it for an MQTT sensor. A template counting a domain (`states.light | count`) is followed for any change in that domain, so an unlocking automation on it makes ordinary light calls ask for the PIN. The PIN prompt names the automation the change would start ("turn on Let In (starts the automation \"Let in\")"). ([#814](https://github.com/goruck/home-generative-agent/issues/814))
- **Running an intent script that does any of the above.** Home Assistant gives the agent a tool for each intent script (`intent_script__<name>`, or the name alone before 2026.9), and an intent script can take over a built-in intent such as "turn on". Its action is checked the way a script's steps are, whichever tool runs it, and when several intent scripts could be behind one tool, each is checked: an intent script whose action unlocks the front door asks for the PIN, one that turns on the porch lights does not, and a step whose target the model fills in from a slot (`{{ door }}`) is held, since which door is not known ahead. An intent script whose action cannot be read is held. The PIN prompt names the intent and what the agent filled in ("run the voice command OpenDoor (Back Door)").
- **Any change on another Home Assistant reached over MCP.** A remote Home Assistant's MCP server offers its intents and scripts under the names this one uses (`intent__HassTurnOn`, `script__open_door`), along with tools whose names say nothing (a script named by its id alone before 2026.9, `media_player__play_media`, a custom integration's tools). This Home Assistant cannot see the remote house's devices, scripts or automations, so an MCP server offering any tool named like Home Assistant's own intent or script tools, or Home Assistant's `GetLiveContext` or `GetDateTime`, is treated as a remote Home Assistant. That is judged on every tool the server offers, including ones you excluded from the agent. Every call to it that can change something asks for the PIN, a remote light, list, timer or broadcast included. A call that only reads does not: a state, the weather, the date or time, a temperature, a timer's status, or a tool the server marks read-only. A script or intent script is held even when marked read-only or named like a read-only intent, and so is a timer given a command to run when it ends. The call's arguments go to the remote server as given, and the PIN prompt names the tool, every argument it sends, as the agent wrote them (the ones naming what it acts on first, keeping only letters, digits, spaces and a few marks such as `-` and `.`, so no quote mark, bracket or semicolon, with each long value cut), and the server, since only the remote server knows which devices they reach (`run HassTurnOff (name Front Door; domain lock) on "Upstairs HA"`). The confirmed call goes to the same server, and the same request sent to another server is a new request with its own PIN. A call whose tool both this Home Assistant and a remote one offer, with no record of which one the agent was given, is held. `GetLiveContext` and `GetDateTime` count as a remote Home Assistant's tools only from an MCP server, since other local APIs may offer this Home Assistant's own. Other MCP servers (time, search, a service-calling tool) are checked by their arguments as before. A remote Home Assistant's tools reach the agent named after its server (`upstairs_ha__intent__HassTurnOn`), with the server's name in their description, so the agent sees this house's tools and the remote one's side by side and picks the house you mean; tool search reads the server's name too, so a request naming the house ("the porch light at the Cabin") ranks its tools first ([#848](https://github.com/goruck/home-generative-agent/issues/848)); your exclusions and always-included tools keep using the tools' own names. Give each remote Home Assistant's MCP entry a name that says which house it is (**Settings → Devices & services**): Home Assistant names every one "home-assistant" until renamed, and two with names that read alike get a short code added to tell their tools apart, with a warning in the log ([#843](https://github.com/goruck/home-generative-agent/issues/843)). Limits: a server's read-only mark is trusted, and Home Assistant marks a tool by its intent's name, so a remote intent script that takes over a read-only intent (`HassGetState`) is not held; a remote Home Assistant reached through a proxy that renames its tools, or a third-party Home Assistant MCP server, is not recognized; and one whose server offers none of the tools above, such as one before 2026.9 with only scripts exposed ([#840](https://github.com/goruck/home-generative-agent/issues/840)).
- **Creating an automation that performs any of the above** — "unlock the front door whenever I get home" is held for the PIN just like the direct command

Alarm control panels use their own alarm code, which is separate from the critical-action PIN. A script or scene that disarms a panel without asking for that code (a script with the code written into it, or a panel that needs no code) is not held for the PIN unless you add a rule for it, such as domain `alarm_control_panel`, service `alarm_disarm`.

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

**What screening cannot see.** Screening runs before the automation is written and has no access to Home Assistant's entity registry, so it cannot expand a group or an area or resolve a registry ID. That is why unresolvable targets are gated rather than reasoned about. A template switch or input button the automation names by entity id is opened as above. Separately, the check matches domains and services, so a *transport* service that reaches a lock without naming the lock domain is invisible to it. The realistic case is `mqtt.publish` to a lock's command topic — an MQTT-backed lock (Zigbee2MQTT, ring-mqtt) can be opened by publishing to its topic, and no service-name rule can distinguish that from any other MQTT message. The same is true of any **raw protocol write** that addresses a device beneath the entity layer: `zwave_js.set_value` can write a Door Lock command class directly (as a service call or as a device action), and `zha.issue_zigbee_cluster_command` is equivalent for Zigbee. These are not gated by default because they are ordinary tools on those stacks and gating them would prompt on routine automations.

If you run locks on one of these transports and want them covered, add the relevant entry to the critical actions — `{"domain": "mqtt", "service": "publish"}`, `{"domain": "zwave_js", "service": "set_value"}`, or `{"domain": "zha", "service": "issue_zigbee_cluster_command"}` — accepting that every automation using that transport will then ask for the PIN.

**Blueprint automations are attested at approval time only.** A blueprint-based automation is stored in `automations.yaml` as a `use_blueprint:` reference, and Home Assistant re-substitutes the blueprint on every reload. Screening reads the substituted actions, so the PIN confirms what the blueprint did *when you approved it*. Editing that blueprint file afterwards changes what the approved automation runs, without a fresh prompt. For plain YAML automations the config that is written is the one that was screened.

**Flow:** When you request a protected action, the agent queues the request and asks for the PIN. The prompt names the action from what will run, not from the model's wording: the area or floor a named device is limited to (`unlock "Door" in Garage`, [#784](https://github.com/goruck/home-generative-agent/issues/784)), the states a scene call sets (`set "Front Door" (lock.front_door) to unlocked`, [#786](https://github.com/goruck/home-generative-agent/issues/786)), and the devices the agent passed to a script's device, area or target fields (`turn on "Open door" ("Garage Door" (cover.garage))`, [#787](https://github.com/goruck/home-generative-agent/issues/787)); a script's free-text fields, and ids Home Assistant does not know, are left out. Your next message answers it: reply with the PIN, as digits or as spoken digits in English or Czech, to complete the action. Any other reply cancels the request, and a PIN in the request itself is not accepted: the agent asks for it, so that nothing started while the agent handled your message (for example because of text it read from a camera or a device) can use your PIN. The PIN is accepted only from that reply: one the model took from earlier in the conversation or made up is refused without counting an attempt. One message is one attempt and confirms one action. While you are still answering, your next message can answer the prompt again: after a wrong PIN, a PIN that could not be read (the agent asks you to say the digits one by one), your request said again, or a reply whose turn failed with an error; at most three replies per prompt and five wrong PINs per request. If the action fails, ask again. If an action fails in an unexpected way after starting, or the confirmation takes longer than 25 seconds, the agent says it may have run: check the device before asking again. A confirmed automation whose install fails always reads as "may have been installed": check your automations first. Ask for one protected action at a time; a new request replaces one still waiting, and a second request in the same message is refused until the first is answered. The PIN has to be a whole number in your reply, written ("1234"), in groups ("12 34") or digit by digit ("1, 2, 3, 4"); a longer number that contains it ("12345", "555 1234") or a date, time, decimal, measurement or address ("12:34", "12.34", "1234.56 kWh", "50%") is not read as the PIN. Say only the PIN: "1 2 3 4, 2 doors" reads as the longer number 12342. Every reply on the PIN path starts with a fixed sentence built from the held request, not from the model: "Your PIN will confirm: unlock \"Front Door\"." while it waits, then "PIN accepted: unlock \"Front Door\".", "PIN accepted, but unlock \"Front Door\" failed." or "PIN accepted; unlock \"Front Door\" may have run. Check the device.", and "That PIN was incorrect." after a wrong one, followed by "Make the request again." when that was the request's last attempt or the prompt's last reply. The verb is what Home Assistant will do (lock/unlock, open/close, turn on/off), a request that names no device says what it covers ("unlock every lock in Garage"), and an automation is named with the protected calls inside it. Check that sentence before you answer: if it is not what you asked for, do not give the PIN. If the model tries to confirm before you have answered, with a PIN of its own or one it read elsewhere, the attempt is refused without counting and the turn ends with that sentence and "What is your PIN?", so the prompt is spoken once. A queued request also expires after 10 minutes, and Home Assistant starts a new conversation after about five idle minutes, so a late reply reads as "not found or expired": make the request again. Only the user who made the request, in the same conversation, can confirm it. An automation that confirms a protected action sends the request and then the PIN as two `conversation.process` calls with the same `conversation_id`. If the guard is toggled on but no PIN has been set, a direct command logs a warning and proceeds, while an automation is refused outright — an automation persists and keeps firing, so there is no safe way to let it through unconfirmed. Set a PIN in Options for the guard to take effect.

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
- **Excluded tools** (`tool_exclusions`) — per-tool allow/deny picker across every selected LLM API, including MCP servers (see [Excluded tools](#excluded-tools))
- **Always-included tools** (`tool_inclusions`) — tools appended after vector retrieval, on top of the retrieval limit (see [Always-included tools](#always-included-tools))

### Prompt caching

Cloud providers only reuse a cached prompt when the *beginning* of the request is byte-identical to an earlier one. HGA therefore renders the system prompt in two parts and always sends the stable part first:

| Part | Contents | Changes |
|------|----------|---------|
| Stable prefix | your system prompt, the timezone line, Critical Action PIN and YAML-mode guidance, the tool-error rule, the exposed-entity context from the selected LLM APIs | only when you edit options or expose/rename entities |
| Volatile tail | Home Assistant's `Current time is … Today's date is …` line, the memories retrieved for this request, the running conversation summary | every turn |

What each provider does with that:

- **Anthropic** — the stable prefix gets an explicit `cache_control` breakpoint and the request also carries the automatic last-block breakpoint, so the prefix is written once (`cache_creation` in the debug log's `usage_metadata`) and read back on later calls (`cache_read`). Reads are billed at a fraction of the input price; writes at a premium. Both breakpoints use Anthropic's default 5-minute cache lifetime, refreshed on every hit: every model call inside a turn and any follow-up turn within five minutes reads the prefix, while a turn after a longer pause writes it again. A prefix shorter than the model's minimum cacheable length (1,024 tokens for Sonnet, 4,096 for Haiku 4.5, 512 for the Claude 5 models) is processed uncached without error.
- **OpenAI, Gemini** — both cache prefixes implicitly; the same ordering lets them match.
- **Ollama, OpenAI-compatible servers** — receive a single plain string with no Anthropic-only fields; the only difference from before is that the date/time line, memories and summary now sit at the end of the prompt instead of the beginning. Local servers keep their own KV-cache behaviour.

Two things still prevent a hit between turns, by design of the providers' caches:

- **A different tool set.** Tool Retrieval (RAG) selects tools per request, and the tool array precedes the system prompt in the cached prefix, so a turn that binds a different set of tools cannot read the previous turn's entry. Consecutive turns on the same topic, and every model call after the first inside one turn (a tool call followed by the answer), do hit. Raising the retrieval limit or adding always-included tools makes the bound set more stable. Memories are retrieved once per turn and reused by every model call in it, so a memory written by a tool mid-turn appears from the next turn on.
- **Time-dependent expressions in your own system prompt.** The prompt is a Home Assistant template; a `{{ now() }}` or a state lookup in it re-renders differently each turn and lands in the stable part. Keep such values out of the prompt, or accept that the prefix will be rewritten on every turn.

If you use a fallback chain, each member is shaped for its own provider: an OpenAI fallback behind an Anthropic primary receives the plain string, never the Anthropic cache key.
- `model_provider_uncontended` — bypass all local GPU gates when the server has dedicated capacity
- **Video analyzer mode** — disable / notify_on_anomaly / always_notify
- **Enable perceptual-hash frame filter (dHash)** — skip visually identical frames before VLM analysis (off by default; always active for ring-mqtt `event_select` capture loops regardless of this setting; see caveat in [Camera Entities](camera-entities.md#advanced-options))
- **Camera notification cooldown (seconds)** — `0` (off) to `600`; after a camera notification sounds, later ones from that camera within the window replace the same card quietly instead of sounding again (needs a single `notify.mobile_app_*` target; see [Notification cooldown](camera-entities.md#notification-cooldown))
- **Motion sensor → camera overrides** — one `binary_sensor.X: camera.Y` pair per line; use when automatic resolution picks the wrong camera (see [Motion → camera resolution](camera-entities.md#motion--camera-resolution))

See [Architecture](architecture.md#llm-context-management) for detail on context management parameters.

---

> **Developer reference:** For a complete listing of every named constant — including code-only tuning knobs and module-level internals — see the [Constants Reference](constants.md).

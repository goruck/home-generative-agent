# Changelog

All notable changes to this project will be documented in this file.

## [3.14.3] - 2026-05-11

### Fixed

- **Appliance duration alerts now name the actual appliance** — Notifications for
  `appliance_power_duration` findings previously fell back to the generic phrase
  "An appliance" because the LLM explanation lacked a display name. The rule now
  includes `friendly_name` in finding evidence, and a new deterministic message
  builder formats the mobile copy directly — e.g. `Washer drew about 296 W for
  633 min, above the 60 min threshold. Check it.` — bypassing the LLM path
  entirely for this finding type. Power-sensor suffixes (`Power`, `Energy`, etc.)
  are stripped case-insensitively so user-set names like `EV Charger Power` render
  as `EV Charger`. When no friendly name is present the entity ID is used as
  fallback. The same formatter is reused for persistent notifications. Closes
  [#391](https://github.com/goruck/home-generative-agent/issues/391).

## [3.14.2] - 2026-05-11

### Fixed

- **Streaming silent for Anthropic and OpenAI** — Both providers default to
  `streaming=False` in LangChain, so no `on_chat_model_stream` events fire and
  the chat window stayed blank until the full response arrived. A new
  `on_chat_model_end` fallback in `_stream_langgraph_to_ha` yields the complete
  text for text-only responses when no streaming chunks were delivered in a turn.
  Anthropic streaming chunks (type `text_delta`) are now also recognised by
  `_normalize_ai_content`, so Anthropic works correctly in both streaming and
  non-streaming modes.
- **OpenAI 400 "object schema missing properties" on `confirm_sensitive_action`**
  — Tool schema extraction was calling `args_schema.schema()`, which fails on
  `InjectedStore` annotations and produced a bare `{"type": "object"}` without the
  `properties` field that OpenAI requires. Schema extraction now uses
  `tool_call_schema.model_json_schema()` (which correctly excludes injected
  arguments) as the primary path, with a `properties: {}` safety net added to
  `_format_and_dedupe_tools` for any schema that declares `type: object` but omits
  `properties`.
- **`get_entity_history` not retrieved for history questions** — The tool's RAG
  embedding description was too terse to match natural-language queries like "how
  many times was the front door opened". The description was rewritten to enumerate
  concrete use-cases (open/close counts, on-durations, motion trigger times, run
  frequency) so cosine similarity retrieval picks it up reliably.
- **Blocking `ssl.load_verify_locations` in the HA event loop** — `langchain_anthropic`
  lazily constructs an async `httpx` client (and triggers SSL context loading) on
  every call to `_get_default_async_httpx_client`. The function is now patched at
  import time with `lru_cache` so the client is created once; the first call is
  pre-warmed in a thread-pool executor after provider initialisation so SSL I/O
  never blocks the HA event loop.

## [3.14.1] - 2026-05-10

### Fixed

- **Anthropic API rejects tools with missing `input_schema.type`** — Tools whose
  parameters schema had no top-level `type` field (e.g. tools with no parameters,
  returning `{}`) caused a `400 invalid_request_error` from the Anthropic API on
  every chat query. `_format_and_dedupe_tools` now injects `"type": "object"` when
  the field is absent, and replaces non-dict parameter values (null, array) with
  `{"type": "object"}` to prevent a downstream `AttributeError`.

## [3.14.0] - 2026-05-10

### Added

- **Video analyzer caption deduplication** — The video analyzer now suppresses
  repeated low-value camera notifications within a configurable window
  (`VIDEO_ANALYZER_CAPTION_DEDUPE_WINDOW_SEC`, default 30 minutes). A new
  `CaptionNoveltyDecision` result type carries the suppression reason and match
  details, enabling structured debug logging and future metadata storage.

### Fixed

- **Caption novelty used weakest vector match** — `_is_anomaly` treated any
  single below-threshold result as novel, even when the best matching caption was
  highly similar. Replaced with a best-score check so only the closest stored
  caption governs the suppress/notify decision. Renamed to `_is_caption_novel`
  to match the new semantics.
- **Generic "animal" subject not recognized** — Captions describing "a dark
  animal stands on the path" were incorrectly suppressed against days-old matches
  because `animal` was absent from the subject-term list. Added `animal` to
  `_SUBJECT_RE` so generic animal descriptions trigger the stale-match re-notify
  path alongside `cat`, `dog`, `deer`, and other named species.
- **Dead regex stems in action detector** — Partial stems `arriv`, `leav`,
  `driv`, and `mov` in `_ACTION_RE` could never match at a word boundary.
  Replaced with the correct infinitive forms: `arrive`, `leave`, `drive`, `move`.

## [3.13.1] - 2026-05-05

### Fixed

- **Home Assistant blocking I/O during model setup** — Ollama chat and embeddings
  clients now use Home Assistant's SSL context helper, and tool binding is moved
  into the executor path. This avoids blocking `httpx` client setup work on the
  event loop during integration setup and model calls.
- **Video analyzer face API client ownership** — The video analyzer now uses Home
  Assistant's shared async `httpx` client for face API calls and passes the face
  timeout per request, avoiding direct async client construction and shutdown in
  the integration.
- **Sentinel log noise** — Repeated Sentinel operational failures are now rate
  limited with recovery messages, and routine cycle/debug logs were removed from
  the hot path. Discovery, triage, suppression, execution, and trigger handling
  should be quieter while still surfacing first failures and repeated issues.
- **Sentinel LLM calls off the event loop** — Sentinel model calls now prefer the
  synchronous `invoke` path in a worker thread when available, preserving
  admission/defer semantics while reducing event-loop pressure.
- **Video analyzer priority typecheck coverage** — Tests were tightened around
  `NullChat`, Ollama server logging, and nested mock accessors so priority-path
  coverage remains compatible with Pyright.

## [3.13.0] - 2026-05-03

### Added

- **Anthropic Claude provider** — You can now select Anthropic Claude models
  (claude-opus-4-5, claude-sonnet-4-5, claude-haiku-4-5, and their successors)
  as the chat, vision (VLM), and summarization provider. Configure by adding an
  Anthropic model-provider subentry and entering your API key. Prompt caching is
  enabled automatically via `cache_control: ephemeral`, reducing latency and cost
  on repeated conversation turns.
- **Anthropic API key validation** — The config flow validates the API key against
  `https://api.anthropic.com/v1/models` before saving, with distinct errors for
  authentication failures and connectivity issues.
- **`extract_final` list-content support** — `extract_final()` now accepts
  Anthropic's multi-block content format (`list[{"type":"text","text":"..."}]`) in
  addition to plain strings. This fixes triage, discovery, and explain responses
  when Anthropic returns structured content blocks.

### Fixed

- **Sentinel triage with Anthropic provider** — `_parse_response` used `str(content)`
  which produced a Python dict repr when Anthropic returned multi-block content,
  causing every alert to bypass triage (fail-open to notify). Fixed to use
  `extract_final(content)`.
- **LLM explain with Anthropic provider** — `extract_final(str(content))` → 
  `extract_final(content)` so list-format Anthropic responses are rendered
  correctly in push notifications instead of showing raw dict repr.
- **Discovery engine with Anthropic provider** — `json.loads(content)` raised
  `TypeError` (not caught by the bare `json.JSONDecodeError` handler) when
  Anthropic returned a list. Fixed to use `extract_final(content)` before parsing
  and added `TypeError` to the handler.
- **Config flow `else` branch hardened** — The Anthropic API-key schema builder
  used a bare `else:` that would silently present the Anthropic form for any
  future unknown provider. Changed to explicit `elif provider_type == "anthropic":`
  with a fallback empty schema.
- **Config flow: API key stored on validation failure** — `settings["api_key"]`
  was assigned even when `validate_anthropic_key` raised an exception. Moved
  inside the success path.

## [3.12.6] - 2026-05-03

### Fixed

- **Sentinel LLM timeouts on Qwen3-family models** — Qwen3.5:35b defaults to
  thinking mode (generating `<think>…</think>` chains) when `ChatOllama.reasoning`
  is `None`. `reasoning_field()` now passes `{"reasoning": False}` instead of `{}`
  when reasoning is disabled, explicitly suppressing the default. This was the root
  cause of repeated 20–60 s timeouts on Sentinel explain and discovery calls.
- **Discovery prompt token bloat** — `_existing_semantic_context()` could return up
  to 400 semantic keys (200 proposals + 200 discovery records), adding ~16 K chars
  of exclusion context on top of the 20 K snapshot budget. Keys sent to the LLM are
  now capped at 60 (`_MAX_SEMANTIC_KEYS_IN_PROMPT`). The post-hoc
  `_filter_novel_candidates()` filter catches any duplicates that slip through.
- **Sentinel explain timeout too tight** — `_EXPLAIN_LLM_TIMEOUT_S` raised 20 → 30 s
  to give the model headroom over the observed ~20 s inference time.
- **Discovery timeout too tight** — `_DISCOVERY_LLM_TIMEOUT_S` raised 45 → 60 s.
- **Opaque timeout log** — `async_explain` now logs
  `"LLM explanation timed out after 30s; skipping."` instead of the empty-string
  warning produced when `TimeoutError` was bundled with other exceptions.
- **Discovery prompt observability** — a new `DEBUG` log line reports prompt char
  counts (`snapshot=N, keys=M/total`) each discovery cycle.

## [3.12.5] - 2026-04-29

### Fixed

- **Sentinel discovery snapshot stays within 20 k-char (~5 k token) budget** —
  `reduce_snapshot_for_discovery()` previously produced snapshots up to ~32 k
  tokens, routinely exceeding the 45 s discovery LLM timeout. Four changes tighten
  the output:
  - `_MAX_CAMERA_ACTIVITY` reduced 50 → 20; `_MAX_SUMMARY_CHARS` 150 → 80.
  - Derived context compressed: `timezone` dropped (redundant with `is_night`),
    `now` and motion timestamps truncated to minute precision,
    `baseline_ready_entities` intersected with the filtered entity set and capped
    at 30 IDs.
  - Character-budget gate (`_TOKEN_BUDGET_CHARS = 20_000`) with four progressive
    trim passes: (1) strip `last_changed` from entity groups, (2) truncate camera
    summaries to 40 chars, (3) drop all camera summaries, (4) cap
    `recognized_people` per camera at 5 entries — the fourth pass prevents
    facial-recognition deployments from exceeding the budget after summaries are
    dropped.
  - Nine new tests cover timezone removal, `baseline_ready_entities` filtering
    and cap, camera-count cap, budget compliance, and each of the four trim passes.

## [3.12.4] - 2026-04-28

### Fixed

- **`SENTINEL_ADMISSION_TIMEOUT_S` public constant** — admission timeout was a
  magic `2.0` literal scattered across triage, discovery, and explain call sites.
  Extracted to a named constant in `core/utils.py` and imported at each site.

- **`asyncio.create_task()` replaces deprecated `asyncio.ensure_future()`** —
  `run_sentinel_llm_call` now wraps the call factory in a proper `Task`, which is
  cancellable and tracked in `_sentinel_llm_tasks`.

- **HA reload cancels in-flight Sentinel LLM tasks** — the previous reload path
  replaced `_sentinel_llm_tasks` with a fresh `set()` while old tasks continued
  running. The teardown in `__init__.py` now cancels each task before replacing
  the set.

- **Warning log when Sentinel LLM cancel times out** — `contextlib.suppress`
  silently swallowed any `TimeoutError` from the cancel wait. Replaced with an
  explicit `except TimeoutError` that emits a `WARNING` log.

- **Tests** — deferred-path coverage for `SentinelTriageService`,
  `LLMExplainer`, and `run_sentinel_llm_call` (timeout path); three tests for
  `build_model_deployments` (ollama→edge, openai→cloud, empty providers).

## [3.12.3] - 2026-04-26

### Fixed

- **Deployment-aware admission control replaces lock-based GPU gate** — the
  previous `chat_priority_context` / `_bg_vlm_lock` / `_bg_llm_lock` approach
  made chat wait for background work to finish. The new design inverts this:
  `local_chat_session(deployment)` marks the full chat turn active by clearing a
  shared `_chat_idle` event; Sentinel triage and discovery call
  `sentinel_admission(deployment, timeout_s=2.0)` before each LLM invocation and
  defer if chat is active. Video analysis no longer participates in any
  admission gate — its existing tuning constants (`_VISION_TIMEOUT_SEC`,
  `_SUMMARY_TIMEOUT_SEC`, per-camera queues) remain the intended concurrency
  surface. Embedding generation is ungated and runs concurrently with chat.

- **Provider deployment metadata flows through runtime** — `ModelProviderConfig`
  gains a `deployment: str` field ("edge"/"cloud"). `build_model_deployments()`
  builds a `{category: deployment}` map from provider subentries at setup time;
  the map lives on `HGAData.model_deployments`. Cloud providers bypass all local
  admission gates automatically, with no code changes required per provider.

- **Sentinel starvation surfaced in the health sensor** — `sentinel_admission`
  tracks consecutive deferrals and wall-clock gap since last success. When the
  gap exceeds 300 s a WARNING is logged and the `SentinelHealthSensor` transitions
  to `"degraded"`, exposing the condition to HA automations and dashboards.

- **Chat LLM timeout raised from 90 s to 180 s** — the higher limit covers
  prefill + generation for typical conversation history lengths without the
  gate-wait overhead of the old approach.

- **Sentinel discovery capped at 45 s per LLM call** — discovery prompts can
  be large enough to consume 180 s+ of GPU time. A hard `asyncio.wait_for`
  timeout skips the cycle cleanly rather than monopolising the GPU.

- **Tool timeout sends a non-retryable error to the LLM** — `_run_langchain_tool`
  now returns a `TOOL_CALL_TRANSIENT_ERROR_TEMPLATE` message on `TimeoutError`
  instead of the generic retry-prompting error template. This stops the model
  from re-calling a tool that timed out due to a temporary resource constraint.

- **Qwen3 extended-thinking fallback** — when Ollama strips `<think>` tokens
  and returns empty content after a tool call, `_call_model` injects `"Done."`
  so the conversation does not go silent.

- **Tests** — 26 tests for the new admission-control primitives, plus coverage
  for the triage/discovery deferral paths, the transient tool-error helper, and
  the Qwen3 fallback.

## [3.12.2] - 2026-04-24

### Fixed

- **Chat streaming no longer hangs under heavy VLM load** — `_invoke_model` now
  wraps `model.ainvoke()` in `asyncio.wait_for(_LLM_INVOKE_TIMEOUT_S=90s)`. When
  the Ollama GPU is saturated by concurrent camera analysis, the chat LLM was
  queuing indefinitely inside `astream_events`, stalling the streaming pipeline
  and showing "no response" in the HA chat UI. The timeout converts an infinite
  hang into a bounded `HomeAssistantError`. Closes #378.

- **Blank chat bubble after timeout replaced with user-visible error message** —
  The timeout `HomeAssistantError` now propagates cleanly through
  `_stream_langgraph_to_ha` (bypassing the misleading "generator error" log path)
  to the recovery handler, which emits "I'm sorry, I was unable to respond in time.
  Please try again." instead of an empty bubble.

- **Empty-content fallback in `_async_handle_message` now shows an error message**
  — the `content=""` guard (hit when streaming fails with no recoverable graph
  state) now emits "I'm sorry, I was unable to respond. Please try again." so the
  chat UI always shows something actionable.

## [3.12.1] - 2026-04-24

### Fixed

- **LM Studio / local embedding server compatibility** — `OpenAIEmbeddings` now sets
  `check_embedding_ctx_length=False`, preventing `tiktoken` from converting text into
  integer token arrays before sending. Local servers (LM Studio, Ollama-compatible
  endpoints) expect plain strings and returned `400 - 'input' field must be a string
  or an array of strings`. Closes #375.

- **Removed `dimensions=` from local embedding calls** — the `dimensions` parameter
  is an OpenAI-specific truncation feature not supported by local models. Sending it
  caused `422 Unprocessable Entity` errors with models like `nomic-embed-text`.

- **Configurable embedding vector dimensions** — the openai_compatible model provider
  settings now expose an `Embedding dimensions` field (default 768, matching
  `nomic-embed-text`). Previously the dimension was hardcoded to 1024 for all
  providers, which caused pgvector index mismatches with 768-dim models. The value
  flows through provider settings into `PostgresIndexConfig(dims=N)`.

### Upgrade notes

If you are using an `openai_compatible` provider for embeddings and have already run
the integration (pgvector tables exist), you will need to drop and recreate the vector
store tables if you change the `embedding_dims` value:

```sql
DROP TABLE IF EXISTS store_vectors;
DROP TABLE IF EXISTS vector_migrations;
```

The tables are recreated automatically on the next startup.

## [3.12.0] - 2026-04-21

### Added

- **Native LLM streaming** — HGA now streams assistant tokens to the HA frontend
  in real time via `astream_events` + HA ChatLog delta API. Responses appear
  word-by-word instead of waiting for the full LLM response. Closes #370.
  Requires HA 2026.4.0 or later.

- **Parallel tool execution** — multiple tool calls within a single model turn
  now run concurrently via `asyncio.gather`, reducing multi-tool turn latency
  (e.g. `get_current_time` + `get_and_analyze_camera_image` in parallel).

- **LangChain tool timeout** — each tool call is bounded by a 30-second timeout
  via `asyncio.wait_for`. Stuck tool calls now produce a descriptive error
  instead of hanging the conversation indefinitely.

### Changed

- **Protected-action PIN flow hardened** — wrong-PIN entries no longer mark the
  pending action as resolved. Routing errors on `confirm_sensitive_action` are
  also treated as unresolved, preventing a bypass when the confirm tool is
  unavailable. Type annotations for PIN helper functions updated to
  `Sequence[AnyMessage]` for broader compatibility.

- **Streaming robustness** — LangGraph stream delta synchronization hardened:
  tool call IDs are correlated across `on_chat_model_end` and `on_chain_end`
  events; orphaned tool calls are flushed as synthetic rejections on generator
  exit; partial content is committed and recoverable on stream failure;
  CONTENT_ADDED is re-fired for the final AssistantContent so the HA frontend
  streaming UI shows the complete response. Empty chat logs on mid-stream error
  now produce a safe fallback instead of crashing the turn.

### Fixed

- HA intent-tool results containing `response_type` dict structures no longer
  produce an empty chat bubble in multi-tool turns.

- JSON tool results are correctly parsed; streaming failures fall back to graph
  state recovery to avoid empty responses.

## [3.11.1] - 2026-04-16

### Added

- **HA 2026.4.0 chatlog transparency (Show Details)** — HGA now populates the
  HA conversation chatlog after each agent turn, enabling HA's "Show Details"
  panel to display the full tool call / result chain. Each intermediate tool
  call (with arguments) and its result are visible as typed chatlog entries.
  Requires HA 2026.4.0 or later. Closes #369.

## [3.11.0] - 2026-04-14

### Added

- **Cyclical load sustained deviation gate** — refrigerators, freezers, and
  compressors now require sustained anomalous power draw before Sentinel fires
  a notification. Normal compressor cycling (e.g. ~944 W on/off every 30
  minutes) no longer triggers repeated "High energy consumption away" alerts.
  The gate duration is configurable in Sentinel settings (default 20 minutes,
  range 0–120 in 5-minute steps; set to 0 to disable).

- **Baseline minimum samples UI** — the "Baseline minimum samples before
  deviation alerts fire" setting is now exposed in the Sentinel configuration
  flow (range 1–500, default 20). Previously this required editing config
  entries directly.

- **Health sensor `cyclical_entities_gated` attribute** — the Sentinel health
  sensor now reports how many cyclical entities are currently held by the
  sustained deviation gate, useful for dashboards and debugging.

## [3.10.0] - 2026-04-13

### Added

- **RAG-based tool indexing and retrieval** — tools are embedded into pgvector at startup
  and retrieved per-turn by cosine similarity. Only the tools most relevant to the user's
  message are loaded into the agent's prompt, reducing prompt size and improving
  tool selection accuracy. On subsequent restarts, unchanged tools skip re-embedding
  via SHA-256 content hashing.

- **Tool Index Status diagnostic sensor** (`sensor.tool_index_status`) — exposes the
  current state of the tool index: `indexing`, `ready`, `failed`, or `unknown`. Use
  it in dashboards or automations to monitor embedding provider health.

- **Multi-intent query decomposition** — compound requests like "turn on the lights and
  check the back yard camera" are split into sub-queries and searched independently.
  Scores are aggregated by maximum per tool, so each intent gets the best-matched tools.

- **Actuation safety net** — when a message contains action keywords (turn on, lock, etc.),
  physical control tools are force-bound even if they fall below the relevance threshold.
  Capped at three-quarters of the retrieval limit so safety tools cannot crowd out
  RAG-retrieved results.

- **`tool_retrieval_limit` and `tool_relevance_threshold` options** — configurable in the
  Options flow. Retrieval limit (default `5`) sets the maximum tools per turn; relevance
  threshold (default `0.15`) sets the cosine similarity cutoff.

### Fixed

- **`tool_index_failed` guard** — a persistent embedding provider failure previously
  re-spawned a background index task on every conversation turn. The integration now
  sets `tool_index_failed` and skips further attempts until the next restart.

- **Reasoning mode crash on Qwen3/Ollama** — reasoning mode is now disabled before
  `bind_tools` to prevent a crash when using Qwen3 or other providers that do not
  support reasoning during tool binding.

- **`safe_convert` for unhashable selectors** — `voluptuous_openapi.convert` could crash
  on `SelectSelector` and other unhashable HA selector types. A `safe_convert` wrapper
  now handles these gracefully, extracting enum options where possible and defaulting
  to string type otherwise.

## [3.9.0] - 2026-04-06

### Added

- **Day-of-week (DOW) baselines** — Sentinel's time-of-day anomaly detector now
  supports per-(DOW, hour) baselines in addition to the existing global hourly
  averages. Enable via `sentinel_baseline_weekly_patterns` in the Sentinel subentry.
  Uses Welford's online algorithm for exact running mean + variance per slot without
  storing all past values. Detection uses a weighted blend that transitions smoothly
  from global to DOW baselines as data accumulates (`w = min(count / dow_min_samples,
  1.0)`), so there is no hard cliff when switching. Entity-specific stddev thresholds
  prevent washer/dryer power sensors from triggering on the same absolute deviation
  that would flag a door sensor. DOW buckets use local time so weekend/weekday
  schedules align with the user's actual timezone.

- **DOW min-samples config** — New `sentinel_baseline_dow_min_samples` option
  (default 4 weeks) controls how many observations per DOW-hour slot are required
  before the blend weight reaches 1.0. Configurable via a NumberSelector in the
  Sentinel subentry UI. Intentionally lower than the global `sentinel_baseline_min_samples`
  (20) because DOW slots update at most once per week per entity.

- **`learned_suppressions_active` health sensor attribute** — Exposes the count of
  distinct `{rule_type}:{entity_id}` pairs that have accumulated learned cooldown
  multipliers via snooze/dismiss feedback. Makes the feedback-learning loop
  observable in the health dashboard.

- **DB performance index** — `CREATE INDEX IF NOT EXISTS idx_sentinel_baselines_entity_metric`
  added to `async_initialize()` for fast per-entity/metric lookups. Idempotent;
  safe on existing installations.

## [3.8.0] - 2026-04-05

### Added

- **Cross-area camera entry links** — Sentinel's camera-entry-unsecured rule can now
  detect activity on cameras that physically overlook entry points in a different Home
  Assistant area (e.g. a driveway camera watching the front door). Configure the mapping
  via `sentinel_camera_entry_links` in the Sentinel subentry: a JSON object of camera
  entity IDs to lists of entry/lock entity IDs. Same-area unsecured entries are still
  detected automatically; cross-area links are purely additive. When both same-area and
  linked entries are unsecured, all are reported in a single finding (deduplicated).
  The anomaly suppression ID hashes only same-area entities, so changing link config
  does not invalidate existing suppression state.

## [3.7.2] - 2026-04-05

### Changed

- **Multi-select LLM API support** — The "Control Home Assistant" option in the options
  flow is now a multi-select, allowing multiple LLM APIs to be active simultaneously (e.g.
  combining the built-in Assist API with one or more MCP server integrations). Existing
  single-API configurations migrate automatically; the behaviour is identical when only one
  API is selected.

### Fixed

- **Crash when no APIs configured** — `_async_handle_message` no longer raises `KeyError`
  when `CONF_LLM_HASS_API` is absent from options (i.e. the user deselected all APIs).
  The agent now runs cleanly with only its built-in LangChain tools in that state.

- **Migration respects "no control" intent** — The v5→v6 config-entry migration converts
  the legacy `"none"` sentinel and absent key to an empty list rather than silently
  enabling the Assist API, preserving the operator's prior choice.

## [3.7.1] - 2026-04-04

### Added

- **Daily digest config flow UI** — The Sentinel subentry options flow now exposes a
  `Boolean` toggle to enable/disable the daily digest and a `TimeSelector` to set the
  delivery time, so operators can configure the digest without editing raw options.

- **Proposals approved (24 h) health attribute** — `SentinelHealthSensor` gains a
  `discovery_proposals_approved_24h` attribute counting how many discovery proposals
  were approved by the operator in the past 24 hours, giving visibility into how active
  the rule-learning feedback loop is.

- **Feedback-trained cooldown wired for snooze and dismiss** — `record_cooldown_feedback`
  is now called from both the snooze action (notifier) and the dismiss action (actions),
  completing the feedback loop so per-entity cooldown multipliers actually accumulate.

### Fixed

- **Daily digest time parse with `"HH:MM:SS"` format** — `sentinel/notifier.py` used
  `split(":", 1)` which parsed `"08:00:00"` as `("08", "00:00")`, silently discarding
  the configured time and always scheduling the digest at 08:00. Fixed to
  `split(":")` so any user-configured time takes effect.

- **Missing `stored_version = 5` in suppression state migration** — The v4→v5 migration
  block did not update the `stored_version` sentinel variable, meaning a future v6
  migration block would have re-applied v5 logic. Fixed.

- **Cooldown multiplier key scheme (`entity_id` → `rule_type:entity_id`)** — Multipliers
  are now keyed by `"{rule_type}:{entity_id}"` so different rules for the same entity
  accumulate independent multipliers. Includes a v5 schema migration that discards the
  old bare-key entries (safe: the feedback signal was never wired in production).

## [3.7.0] - 2026-04-03

### Added

- **Discovery quality metrics in health sensor** — `SentinelHealthSensor` now exposes
  `discovery_candidates_generated`, `discovery_candidates_novel`,
  `discovery_candidates_deduplicated`, `discovery_proposals_promoted`, and
  `discovery_unsupported_ttl_expired` attributes, giving operators visibility into
  every LLM discovery cycle without opening logs.

- **Trigger-source breakdown (24 h)** — Health sensor gains a
  `trigger_source_breakdown` attribute with rolling 24-hour counts broken down by
  `poll`, `event`, and `on_demand` trigger sources so you can see at a glance what
  is driving Sentinel activity.

- **Daily digest notification** — Sentinel can now send a scheduled push notification
  summarising findings from the past 24 hours. Controlled by the new
  `CONF_SENTINEL_DAILY_DIGEST_HOUR` option (default 07:00 local time). Duplicate
  digest triggers within a session are safely de-duplicated.

- **Centralised action codes** — All Sentinel mobile-action button identifiers
  (`ACTION_CODES`) are now declared once in `const.py` and imported by both the
  notifier and execution service, eliminating the risk of copy-paste divergence.

- **Learned per-entity cooldown multipliers** — Suppression now stores a
  per-entity feedback table (schema v4) that can hold learned multipliers. The data
  model and upgrade path are in place; the feedback signal will be wired to a user
  action in a follow-up sprint.

- **Camera-area spatial ejection in correlator** — `_eject_camera_area_violations`
  post-grouping pass prevents off-area findings from being bundled into a
  `CompoundFinding` by transitive union-find bridging. A finding in *Garage* can no
  longer appear in a compound alert anchored to the *Front* camera.

### Fixed

- **Empty `evidence_paths` bypass in discovery engine** — Candidates with an empty
  `evidence_paths` list (`[]`) were silently passing the derived-only filter because
  the old `if candidate.get("evidence_paths") and all(...)` guard short-circuits on
  falsy empty lists. Guard is now `is not None and len(...) > 0`.

- **Dead code in correlator area filter** — `camera_areas` set subtraction included
  `None` (`- {None, ""}`) but `_area_of()` always returns `str`, making the `None`
  exclusion a no-op. Simplified to `- {""}`.

- **Orphaned daily-digest tasks** — `_async_send_daily_digest` now checks whether a
  prior digest task is still running before scheduling a new one, preventing orphaned
  coroutines when the time-change callback fires more than once before the digest
  completes.

### Changed

- Correlator `_COMPLEMENTARY_PAIRS_REQUIRE_AREA_MATCH` guard extended so that
  spatially-bound pairs (e.g. `unlocked_lock_at_night` + `camera_entry_unsecured`)
  only correlate when they share the same non-empty area, reducing spurious compound
  alerts across different physical zones.

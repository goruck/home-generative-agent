# Changelog

All notable changes to this project will be documented in this file.

## [3.14.13] - 2026-05-23

### Added

- **Deploy script** — `scripts/deploy` syncs the integration to a running Home Assistant instance via rsync over SSH. Only changed files are transferred (checksum-based), `__pycache__` and `.pyc` files are excluded, and the script calls `ha core restart` after a successful sync. See [Contributing](docs/contributing.md) for setup instructions.
- **Feature-level model provider fallbacks** — feature setup now supports an ordered fallback-provider list for chat, VLM, summarization, and embeddings. If the primary provider is unavailable at setup or fails with a retryable runtime error, HGA can use the next configured provider for that category.
- **Fallback provider notifications** — when a fallback becomes active, HGA sends one deduplicated notification per category/provider for the current runtime. Mobile notifications use the configured `notify_service`; otherwise HGA creates a persistent notification. Cloud fallback notifications include a provider-cost warning.
- **STT hallucination filters** — Options now include ignored substring and exact-phrase filters for speech-to-text prompts. Matching input is dropped before it reaches the LLM, reducing false activations from common silence/noise transcriptions.

### Changed

- **Fallback logging is clearer and less noisy** — startup, runtime fallback, unavailable-provider, sticky embedding-provider, and local resource-gate logs now distinguish configured deployment from the effective provider in use. Repeated unconfigured fallback messages are reduced to debug-level diagnostics where appropriate.
- **Fallback models use their own configured defaults** — chat, VLM, and summarization fallbacks now preserve provider-specific model settings and category defaults from `const.py`, including Ollama tuning options when a provider does not override them.
- **Fallback chat retry happens at the model-call boundary** — fallback chat chains no longer expose provider-level `astream`, avoiding mixed responses where a failed provider starts streaming partial text and a fallback provider finishes with different content.

### Fixed

- **Embedding fallback no longer leaves stale vector indexes active** — when the active embedding provider changes, HGA marks the tool index stale, rebuilds it on the next indexing pass, and discards retrieval results if the index becomes stale during search.
- **Embedding dimensions are normalized for fallback providers** — Gemini embedding fallbacks are requested with HGA's configured vector dimension, preventing PostgreSQL vector errors such as `different vector dimensions 1024 and 3072`.
- **VLM and summarization fallback wrappers preserve config overrides** — fallback wrappers now support `with_config()` and merge model config instead of replacing category-specific model settings, fixing video analyzer and summarization paths that use per-call config.
- **Length-limited empty model responses trigger fallback** — empty responses caused by provider length limits are treated as retryable, so chat and summarization can fall back instead of returning blank content.
- **Connectivity failures from HTTP transports are retryable** — local provider connection errors such as Ollama/httpx transport failures now activate configured fallbacks.
- **Tool retrieval fallback diagnostics are safer** — known vector-store mismatch errors degrade to keyword-filtered tool selection or recency-based memory retrieval instead of surfacing repeated tracebacks during provider switches.

## [3.14.12] - 2026-05-19

### Fixed

- **Ambiguous entity name no longer loops the agent** — when multiple entities
  share a friendly name (e.g. `binary_sensor.haustur` and `binary_sensor.haustur_2`
  after a device is re-added), `get_entity_history` now silently picks the
  canonical entity (no numeric `_N` suffix). If ambiguity cannot be resolved
  this way, the tool returns `{"error": "..."}` so the LLM can report the
  problem to the user instead of looping on an empty `{}` response (issue #414).

## [3.14.11] - 2026-05-19

### Changed

- **GetLiveContext is now always injected into the tool candidate list** — the
  previous approach used a regex to detect conditional clauses ("if", "when",
  etc.) and only injected GetLiveContext when a match was found. This was brittle
  and missed constructs like "while", "whenever", "assuming", and compound
  queries. GetLiveContext is now added unconditionally whenever it is absent from
  the candidate set; step 3b's deduplication prevents double-injection for
  read-only open-state queries.

- **Camera activity removed from the system prompt** — the agent was fetching
  recent camera activity on every turn and injecting it into the system message
  as a `<recent_camera_activity>` block. This caused the model to answer presence
  queries from stale camera snapshots instead of calling `GetLiveContext` with
  `domain=person` for authoritative state. The model now fetches camera data via
  tool call only when it needs it.

- **Sentinel action prompts include detection timestamp and alert-time evidence**
  — both execute and handoff prompts now include the exact time the anomaly was
  detected and a compact evidence summary, so the agent can explicitly acknowledge
  whether the alert condition is still active or has already resolved. This
  prevents the model from silently substituting current state for the original
  alert context.

- **`AnomalyFinding` now records its detection timestamp** — a `detected_at`
  field (`datetime`, defaulting to `utcnow()`) is added to the dataclass and
  serialized in `as_dict()`. Downstream consumers (audit trail, action prompts)
  can now reference the precise detection time.

### Fixed

- **Removed global LangChain in-memory LLM cache** — `set_llm_cache(InMemoryCache())`
  was called unconditionally at conversation entity setup. This has been removed;
  caching is now left to LangChain's default behaviour (no global cache).

## [3.14.10] - 2026-05-17

### Fixed

- **Open-state queries now work for users whose sensors use `device_class: window`
  or `device_class: door`** — the parser that identifies open entry sensors from
  a live-context response only accepted `device_class: opening`. Real Home
  Assistant window and door contact sensors (registered by most Zigbee and Z-Wave
  coordinators) use `device_class: window` and `device_class: door` respectively.
  The same gap existed in the HA-state fallback path used when no prior live
  context is available. Both paths now accept all four open-state device classes
  (`door`, `garage_door`, `opening`, `window`). Four regression tests are added
  that use the correct device classes (the prior test fixtures all used
  `device_class: opening`, which matched the developer's own sensors but masked
  the failure for anyone else).

## [3.14.9] - 2026-05-17

### Fixed

- **Read-only open-state queries are more reliable across local tool-calling
  models** — HGA now force-binds and promotes `GetLiveContext` for read-only
  queries such as "list all open windows" and "list the open doors in my house",
  even when semantic tool retrieval misses it or ranks less relevant tools
  higher. The first `GetLiveContext` call for these queries is normalized to a
  broad `binary_sensor` live-context request so local models that emit brittle
  filters like `name: "Window"` or list-valued domains still get the needed
  state. Broad live-context results are then filtered back to the requested
  entity type, preventing door queries from volunteering open windows. Focused
  regression tests cover retrieval, argument normalization, retry handling, and
  scoped result filtering.

## [3.14.8] - 2026-05-16

### Fixed

- **`NullChat` fallback no longer crashes with `TypeError` when a provider is
  unavailable at startup** — `NullChat.ainvoke` and `NullChat.astream` now
  accept `config` as an explicit second positional parameter, matching the
  standard LangChain `BaseChatModel` calling convention
  (`ainvoke(input, config=None, **kwargs)`). Previously, the fallback model
  used when a provider health check failed at HA startup (e.g., Ollama not yet
  responsive) would raise `TypeError: NullChat.ainvoke() takes 2 positional
  arguments but 3 were given` instead of returning the intended graceful
  `"LLM unavailable."` response. Two regression tests added. Closes
  [#410](https://github.com/goruck/home-generative-agent/issues/410).

## [3.14.7] - 2026-05-15

### Fixed

- **Read-only state queries like "list all open windows" no longer trigger
  actuation tools** — The word "open" in phrases like "which windows are open"
  or "show me all open doors" was matching the actuation-safety keyword list,
  causing the agent to inject control tools (`HassTurnOn`, `HassTurnOff`, etc.)
  into its context for what is really a read-only query. A four-regex heuristic
  now distinguishes read-only use of "open" (as a state descriptor combined with
  `list`, `show`, `which`, `what`, `are`, `is`, `status`, or `state`) from
  command use of "open" (as an actuation verb). The `GetLiveContext` tool is
  also promoted to the front of the candidate list for these queries so the
  agent picks up the correct read tool first. Closes
  [#394](https://github.com/goruck/home-generative-agent/issues/394).

- **Tool-loop guard prevents `GraphRecursionError` on stuck tool cycles** — When
  a weaker model or an ambiguous query caused the agent to request tool calls
  indefinitely, LangGraph eventually threw a `GraphRecursionError` visible as an
  error in the HA conversation UI. A new `tool_loop_guard` graph node caps
  tool-use at 3 rounds per conversation turn and returns a friendly message
  instead of crashing: "I wasn't able to complete this request after several
  tool-use attempts. Please try rephrasing your query or breaking it into smaller
  steps." The LangGraph `recursion_limit` is also raised from 10 to 20 so the
  application-level guard fires well before the LangGraph backstop.

## [3.14.6] - 2026-05-13

### Fixed

- **Sentinel no longer fires a false "disarm the alarm" notification when armed_home or
  armed_night is active with occupants present** — `armed_home` and `armed_night` are
  HA alarm modes designed for use while people are home. A three-layer guard now
  prevents this combination from ever becoming an anomaly: the deterministic evaluator
  suppresses it at runtime, the normalization layer rejects any future LLM-proposed
  rule with this pattern, and the LLM explanation prompt is conditioned to never
  describe the state as a problem when occupants are present. The constant defining
  these occupancy-safe modes (`SENTINEL_OCCUPANCY_ARMED_STATES`) is now centralized
  in `const.py` and shared across both sentinel modules. Five new regression tests
  (plus three coverage tests) guard all paths.

## [3.14.4] - 2026-05-12

### Changed

- **OpenAI and Anthropic providers now stream tokens natively** — Both providers are
  constructed with `streaming=True`, so visible words appear in the chat window as
  they are generated rather than arriving all at once at the end. OpenAI additionally
  sets `stream_usage=True` so token-count telemetry is preserved in streamed responses
  (without this flag the usage metadata would be empty). The existing non-streaming
  fallback path remains active for Ollama, Gemini, and any provider that does not emit
  token chunks. The `schema_first_yaml=True` dashboard-generation path is unaffected —
  it invokes the model directly via `ainvoke` and was already independent of the
  streaming flag.

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

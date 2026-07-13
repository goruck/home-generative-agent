# Constants Reference

This document covers the named constants that affect integration behaviour, organized by subsystem. It tracks all `RECOMMENDED_*` defaults from `custom_components/home_generative_agent/const.py` and the most relevant module-level tuning knobs; see the source file directly for the authoritative list. Module-level tuning constants that live outside `const.py` are documented in the [Module Tuning Constants](#module-tuning-constants) section.

**Two tiers:**
- **UI-configurable** — exposed in the HA config/options flow; `const.py` defines the default (`RECOMMENDED_*`). Change these in the UI without touching code.
- **Code-only** — no UI entry; change by editing the source file directly and reloading the integration.

---

## Database

**File:** `const.py` | **UI-configurable**

| Constant | Default | Purpose |
|---|---|---|
| `RECOMMENDED_DB_USERNAME` | `ha_user` | PostgreSQL username |
| `RECOMMENDED_DB_PASSWORD` | `ha_password` | PostgreSQL password |
| `RECOMMENDED_DB_HOST` | `localhost` | PostgreSQL host |
| `RECOMMENDED_DB_PORT` | `5432` | PostgreSQL port |
| `RECOMMENDED_DB_NAME` | `ha_db` | PostgreSQL database name |
| `RECOMMENDED_DB_PARAMS` | `[{"key": "sslmode", "value": "disable"}]` | Extra connection parameters |

---

## Chat Model

**File:** `const.py` | **UI-configurable**

| Constant | Default | Purpose |
|---|---|---|
| `RECOMMENDED_CHAT_MODEL_PROVIDER` | `ollama` | Default provider for conversation |
| `RECOMMENDED_OLLAMA_CHAT_MODEL` | `gpt-oss` | Ollama chat model |
| `RECOMMENDED_OPENAI_CHAT_MODEL` | `gpt-5` | OpenAI chat model |
| `RECOMMENDED_GEMINI_CHAT_MODEL` | `gemini-2.5-flash-lite` | Gemini chat model |
| `RECOMMENDED_ANTHROPIC_CHAT_MODEL` | `claude-sonnet-4-6` | Anthropic chat model |
| `RECOMMENDED_OPENAI_COMPATIBLE_CHAT_MODEL` | `gpt-4o` | OpenAI-compatible chat model |
| `RECOMMENDED_CHAT_MODEL_TEMPERATURE` | `0.2` | Sampling temperature for chat responses |
| `RECOMMENDED_OLLAMA_CHAT_KEEPALIVE` | `300` (s) | Seconds to keep Ollama chat model loaded between requests |
| `RECOMMENDED_OLLAMA_REASONING` | `False` | Enable extended reasoning mode for supported Ollama models |

**Code-only** (cannot be changed from the UI):

| Constant | File | Value | Purpose |
|---|---|---|---|
| `CHAT_MODEL_TOP_P` | `const.py` | `1.0` | nucleus sampling p for chat |
| `CHAT_MODEL_MAX_TOKENS` | `const.py` | `-2` | Ollama max tokens (`-2` = fill context) |
| `CHAT_MODEL_REPEAT_PENALTY` | `const.py` | `1.05` | Ollama repeat penalty |
| `OLLAMA_GPT_EFFORT` | `const.py` | `"low"` | Effort hint for `gpt-oss` reasoning tag |
| `LANGCHAIN_LOGGING_LEVEL` | `const.py` | `"disable"` | LangChain debug verbosity (`disable`, `verbose`, `debug`) |

**Supported model lists** (defines what appears in the UI dropdowns):

| Constant | Allowed values |
|---|---|
| `CHAT_MODEL_OLLAMA_SUPPORTED` | `gpt-oss`, `qwen2.5:32b`, `qwen3:32b`, `qwen3:8b` |
| `CHAT_MODEL_OPENAI_SUPPORTED` | `gpt-5`, `gpt-5-mini`, `gpt-5-nano`, `gpt-4o`, `gpt-4.1`, `o4-mini` |
| `CHAT_MODEL_GEMINI_SUPPORTED` | `gemini-2.5-pro`, `gemini-2.5-flash`, `gemini-2.5-flash-lite` |
| `CHAT_MODEL_ANTHROPIC_SUPPORTED` | `claude-opus-4-7`, `claude-sonnet-4-6`, `claude-haiku-4-5-20251001` |

---

## Vision Model (VLM)

**File:** `const.py` | **UI-configurable**

| Constant | Default | Purpose |
|---|---|---|
| `RECOMMENDED_VLM_PROVIDER` | `ollama` | Default provider for image/camera analysis |
| `RECOMMENDED_OLLAMA_VLM` | `qwen3-vl:8b` | Ollama VLM model |
| `RECOMMENDED_OPENAI_VLM` | `gpt-5-nano` | OpenAI VLM model |
| `RECOMMENDED_GEMINI_VLM` | `gemini-2.5-flash-lite` | Gemini VLM model |
| `RECOMMENDED_ANTHROPIC_VLM` | `claude-sonnet-4-6` | Anthropic VLM model |
| `RECOMMENDED_OPENAI_COMPATIBLE_VLM` | `gpt-4o` | OpenAI-compatible VLM model |
| `RECOMMENDED_VLM_TEMPERATURE` | `0.2` | Sampling temperature for vision responses |
| `RECOMMENDED_OLLAMA_VLM_KEEPALIVE` | `300` (s) | Seconds to keep Ollama VLM loaded |

**Code-only:**

| Constant | File | Value | Purpose |
|---|---|---|---|
| `VLM_TOP_P` | `const.py` | `1.0` | Nucleus sampling p for VLM |
| `VLM_NUM_PREDICT` | `const.py` | `-2` | Ollama token budget for agent image analysis (`-2` = fill context) |
| `VIDEO_VLM_NUM_PREDICT` | `const.py` | `256` | Token budget for proactive video frame descriptions. Capped intentionally to prevent video from monopolizing context. |
| `VLM_REPEAT_PENALTY` | `const.py` | `1.05` | Ollama repeat penalty |
| `VLM_MIRO_STAT` | `const.py` | `0` | Ollama mirostat setting |
| `VLM_IMAGE_WIDTH` | `const.py` | `1920` | Image resize width before sending to VLM |
| `VLM_IMAGE_HEIGHT` | `const.py` | `1080` | Image resize height before sending to VLM |

**Supported model lists:**

| Constant | Allowed values |
|---|---|
| `VLM_OLLAMA_SUPPORTED` | `qwen2.5vl:7b`, `qwen3-vl:8b`, `gemma3:4b` |
| `VLM_OPENAI_SUPPORTED` | `gpt-5-nano`, `gpt-4.1`, `gpt-4.1-nano` |
| `VLM_GEMINI_SUPPORTED` | `gemini-2.5-pro`, `gemini-2.5-flash`, `gemini-2.5-flash-lite` |
| `VLM_ANTHROPIC_SUPPORTED` | `claude-opus-4-7`, `claude-sonnet-4-6`, `claude-haiku-4-5-20251001` |

---

## Summarization Model

**File:** `const.py` | **UI-configurable**

| Constant | Default | Purpose |
|---|---|---|
| `RECOMMENDED_SUMMARIZATION_MODEL_PROVIDER` | `ollama` | Default provider for context summarization |
| `RECOMMENDED_OLLAMA_SUMMARIZATION_MODEL` | `qwen3:8b` | Ollama summarization model |
| `RECOMMENDED_OPENAI_SUMMARIZATION_MODEL` | `gpt-5-nano` | OpenAI summarization model |
| `RECOMMENDED_GEMINI_SUMMARIZATION_MODEL` | `gemini-2.5-flash-lite` | Gemini summarization model |
| `RECOMMENDED_ANTHROPIC_SUMMARIZATION_MODEL` | `claude-haiku-4-5-20251001` | Anthropic summarization model |
| `RECOMMENDED_OPENAI_COMPATIBLE_SUMMARIZATION_MODEL` | `gpt-4o` | OpenAI-compatible summarization model |
| `RECOMMENDED_SUMMARIZATION_MODEL_TEMPERATURE` | `0.2` | Sampling temperature for summaries |
| `RECOMMENDED_OLLAMA_SUMMARIZATION_KEEPALIVE` | `300` (s) | Seconds to keep Ollama summarization model loaded |

**Code-only:**

| Constant | File | Value | Purpose |
|---|---|---|---|
| `SUMMARIZATION_MODEL_TOP_P` | `const.py` | `1.0` | Nucleus sampling p |
| `SUMMARIZATION_MODEL_PREDICT` | `const.py` | `-2` | Ollama token budget for conversation summaries (`-2` = fill context) |
| `VIDEO_SUMMARY_NUM_PREDICT` | `const.py` | `128` | Token budget for video batch summaries. Capped to prevent video from blocking other callers. |
| `SUMMARIZATION_MODEL_REPEAT_PENALTY` | `const.py` | `1.05` | Ollama repeat penalty |
| `SUMMARIZATION_MIRO_STAT` | `const.py` | `0` | Ollama mirostat setting |

**Supported model lists:**

| Constant | Allowed values |
|---|---|
| `SUMMARIZATION_MODEL_OLLAMA_SUPPORTED` | `qwen3:1.7b`, `qwen3:8b` |
| `SUMMARIZATION_MODEL_OPENAI_SUPPORTED` | `gpt-5-nano`, `gpt-4.1`, `gpt-4.1-nano` |
| `SUMMARIZATION_MODEL_GEMINI_SUPPORTED` | `gemini-2.5-pro`, `gemini-2.5-flash`, `gemini-2.5-flash-lite` |
| `SUMMARIZATION_MODEL_ANTHROPIC_SUPPORTED` | `claude-sonnet-4-6`, `claude-haiku-4-5-20251001` |

---

## Embeddings

**File:** `const.py` | **UI-configurable**

| Constant | Default | Purpose |
|---|---|---|
| `RECOMMENDED_EMBEDDING_MODEL_PROVIDER` | `ollama` | Default provider for embeddings |
| `RECOMMENDED_OLLAMA_EMBEDDING_MODEL` | `mxbai-embed-large` | Ollama embedding model |
| `RECOMMENDED_OPENAI_EMBEDDING_MODEL` | `text-embedding-3-small` | OpenAI embedding model |
| `RECOMMENDED_GEMINI_EMBEDDING_MODEL` | `gemini-embedding-001` | Gemini embedding model |
| `RECOMMENDED_OPENAI_COMPATIBLE_EMBEDDING_MODEL` | `text-embedding-3-small` | OpenAI-compatible embedding model |
| `RECOMMENDED_OPENAI_COMPATIBLE_EMBEDDING_DIMS` | `768` | Vector dimensions for OpenAI-compatible embedding models |

**Code-only:**

| Constant | File | Value | Purpose |
|---|---|---|---|
| `EMBEDDING_MODEL_DIMS` | `const.py` | `1024` | Default vector dimensions for Ollama/OpenAI/Gemini embeddings |
| `EMBEDDING_MODEL_CTX` | `const.py` | `512` | Maximum input token length for embedding queries |
| `EMBEDDING_INDEX_TEXT_MAX_CHARS` | `agent/rag_embedding_text.py` | `1200` | Maximum characters embedded per vector-store content row (tool index) |

**Supported model lists:**

| Constant | Allowed values |
|---|---|
| `EMBEDDING_MODEL_OLLAMA_SUPPORTED` | `mxbai-embed-large` |
| `EMBEDDING_MODEL_OPENAI_SUPPORTED` | `text-embedding-3-large`, `text-embedding-3-small` |
| `EMBEDDING_MODEL_GEMINI_SUPPORTED` | `gemini-embedding-001` |

---

## Ollama Global Options

**File:** `const.py`

| Constant | Value | Purpose |
|---|---|---|
| `RECOMMENDED_OLLAMA_URL` | `http://localhost:11434` | Default Ollama server URL (UI-configurable); also the initial value for the three per-category URL overrides below |
| `RECOMMENDED_OLLAMA_CHAT_URL` | `http://localhost:11434` | Ollama server URL for the chat model (UI-configurable; overrides `RECOMMENDED_OLLAMA_URL` for chat only) |
| `RECOMMENDED_OLLAMA_VLM_URL` | `http://localhost:11434` | Ollama server URL for the VLM (UI-configurable; lets VLM traffic target a separate server) |
| `RECOMMENDED_OLLAMA_SUMMARIZATION_URL` | `http://localhost:11434` | Ollama server URL for the summarization model (UI-configurable; lets summarization traffic target a separate server) |
| `RECOMMENDED_OLLAMA_CONTEXT_SIZE` | `32000` | Context window size passed to Ollama models |
| `KEEPALIVE_MIN_SECONDS` | `0` | Minimum allowable keepalive (0 = unload immediately) |
| `KEEPALIVE_MAX_SECONDS` | `900` | Maximum allowable keepalive (15 minutes) |
| `KEEPALIVE_SENTINEL` | `-1` | Special value meaning "never unload" |
| `OLLAMA_EXACT_TOKEN_COUNT` | `False` | Use exact token counting for context trimming. `False` = fast approximate counting. Enable only if trim precision matters more than latency. |

---

## Context Management

**File:** `const.py` | **UI-configurable**

| Constant | Config key | Default | Purpose |
|---|---|---|---|
| `RECOMMENDED_MANAGE_CONTEXT_WITH_TOKENS` | `manage_context_with_tokens` | `"true"` | `"true"` = trim by token count; `"false"` = trim by message count |
| `RECOMMENDED_MAX_TOKENS_IN_CONTEXT` | `max_tokens_in_context` | `32000` | Token ceiling before trimming. Should be ≤ the Ollama model's context size. |
| `RECOMMENDED_MAX_MESSAGES_IN_CONTEXT` | `max_messages_in_context` | `60` | Message count ceiling (used when token-based trimming is disabled) |

---

## Tool Retrieval (RAG)

**File:** `const.py` | **UI-configurable**

| Constant | Config key | Default | Purpose |
|---|---|---|---|
| `RECOMMENDED_TOOL_RETRIEVAL_LIMIT` | `tool_retrieval_limit` | `5` | Maximum tools injected into the agent's context per turn. Raise if the agent misses tools on multi-step requests; lower to reduce prompt size. |
| `RECOMMENDED_TOOL_RELEVANCE_THRESHOLD` | `tool_relevance_threshold` | `0.15` | Cosine similarity cutoff. Lower to include more tools; raise to tighten selectivity. |

**Code-only:**

| Constant | File | Value | Purpose |
|---|---|---|---|
| `ACTUATION_KEYWORDS_REGEX` | `const.py` | (regex) | Keywords that force-attach entity control tools regardless of similarity score. Prevents the agent from missing control tools when the user issues a command verb. |
| `_LC_TOOL_TIMEOUT_S` | `agent/graph.py` | `30.0` (s) | Per-tool execution timeout. Tools that do not return within this window are cancelled and the agent receives an error. Affects VLM and any slow external tool. |

---

## Video Analyzer

**File:** `const.py` | **UI-configurable**

| Constant | Config key | Default | Purpose |
|---|---|---|---|
| `RECOMMENDED_VIDEO_ANALYZER_MODE` | `video_analyzer_mode` | `disable` | Operating mode: `disable`, `notify_on_anomaly`, or `always_notify` |
| `RECOMMENDED_VIDEO_MODEL_SEMAPHORE` | `video_model_semaphore` | `1` | Concurrent VLM + summary model calls per integration entry. Increase only if the GPU server has spare capacity. |
| `RECOMMENDED_MODEL_PROVIDER_UNCONTENDED` | `model_provider_uncontended` | `False` | Bypass all local GPU gates (video semaphore, video foreground session, Sentinel deferral) for dedicated high-capacity servers. |

**Code-only:**

| Constant | File | Value | Purpose |
|---|---|---|---|
| `VIDEO_ANALYZER_SCAN_INTERVAL` | `const.py` | `1.5` (s) | Queue polling interval for the video analysis loop |
| `VIDEO_ANALYZER_SNAPSHOT_ROOT` | `const.py` | `/media/snapshots` | Root directory for saved camera snapshots |
| `VIDEO_ANALYZER_TIME_OFFSET` | `const.py` | `15` (min) | Lookback window for fetching recent camera activity when building a video batch |
| `VIDEO_ANALYZER_CAPTION_DEDUPE_WINDOW_SEC` | `const.py` | `1800` (s, 30 min) | Lexical deduplication window. Artifact captions within this window are suppressed even when the vector score is below threshold. |
| `VIDEO_ANALYZER_SIMILARITY_THRESHOLD` | `const.py` | `0.89` | Cosine similarity threshold for caption deduplication. Captions above this score are considered duplicates and suppressed. |
| `VIDEO_ANALYZER_DELETE_SNAPSHOTS` | `const.py` | `False` | Whether to delete snapshot files after analysis |
| `VIDEO_ANALYZER_SNAPSHOTS_TO_KEEP` | `const.py` | `200` | Rolling snapshot retention count per camera |
| `VIDEO_ANALYZER_TRIGGER_ON_MOTION` | `const.py` | `True` | Trigger analysis on HA motion sensor state changes |
| `VIDEO_ANALYZER_FACE_CROP` | `const.py` | `False` | Crop detected faces before sending to face recognition |
| `VIDEO_ANALYZER_SAVE_LATEST` | `const.py` | `True` | Publish a stable `_latest/latest.jpg` alongside each snapshot |
| `_MAX_BATCH` | `core/video_analyzer.py` | `5` | Maximum frames per analysis batch |
| `_QUEUE_MAXSIZE` | `core/video_analyzer.py` | `50` | Per-camera frame backlog capacity before drops |
| `_FRAME_DEADLINE_SEC` | `core/video_analyzer.py` | `600` (s) | Skip frames older than this; prevents stale results from backlog buildup |
| `_SUMMARY_TIMEOUT_SEC` | `core/video_analyzer.py` | `60` (s) | Timeout for the summarization model call during batch synthesis |
| `_FACE_TIMEOUT_SEC` | `core/video_analyzer.py` | `10` (s) | Timeout for a face-recognition request |
| `_VISION_TIMEOUT_SEC` | `core/video_analyzer.py` | `90` (s) | Timeout for a VLM frame-description call |
| `_VIDEO_MODEL_SEMAPHORE_WAIT_SEC` | `core/video_analyzer.py` | `30` (s) | Max wait for the video semaphore before dropping the frame |
| `_VIDEO_QUEUE_BACKLOG_THRESHOLD` | `core/video_analyzer.py` | `2` | Drop stale queued frames when the backlog exceeds this count |
| `_METRICS_REPORT_INTERVAL_SEC` | `core/video_analyzer.py` | `3600` (s) | How often per-camera latency metrics are logged |

---

## Sentinel Core

**File:** `const.py` | **UI-configurable**

| Constant | Config key | Default | Purpose |
|---|---|---|---|
| `RECOMMENDED_SENTINEL_ENABLED` | `sentinel_enabled` | `True` | Master switch for the Sentinel detection loop |
| `RECOMMENDED_SENTINEL_INTERVAL_SECONDS` | `sentinel_interval_seconds` | `300` (s) | How often the detection cycle runs (5 minutes) |
| `RECOMMENDED_SENTINEL_COOLDOWN_MINUTES` | `sentinel_cooldown_minutes` | `30` | Per-rule-type suppression window after a notification. Prevents alert flooding for the same rule. |
| `RECOMMENDED_SENTINEL_ENTITY_COOLDOWN_MINUTES` | `sentinel_entity_cooldown_minutes` | `15` | Per-entity suppression window. Shorter than the rule cooldown to allow related entities to re-alert sooner. |
| `RECOMMENDED_SENTINEL_PENDING_PROMPT_TTL_MINUTES` | `sentinel_pending_prompt_ttl_minutes` | `240` (4 h) | How long a pending action prompt waits for a user response before expiring |
| `RECOMMENDED_EXPLAIN_ENABLED` | `explain_enabled` | `False` | Generate LLM explanations for findings before notification |
| `RECOMMENDED_SENTINEL_STALENESS_THRESHOLD_SECONDS` | `sentinel_staleness_threshold_seconds` | `1800` (s) | Snapshot entity data older than this is considered stale and may reduce finding confidence |
| `RECOMMENDED_AUDIT_HOT_MAX_RECORDS` | `audit_hot_max_records` | `500` | Maximum records kept in the in-memory audit hot store. User-visible findings are preserved in preference to suppressed records when at capacity. |
| `RECOMMENDED_SENTINEL_CAMERA_ENTRY_LINKS` | `sentinel_camera_entry_links` | `{}` | Maps camera `entity_id` → list of entry/lock `entity_id`s in other HA areas. Used by the `camera_entry_unsecured` rule for cameras that physically overlook entries not in the same HA area (e.g. `{"camera.driveway": ["lock.front_door"]}`). |

**Code-only:**

| Constant | File | Value | Purpose |
|---|---|---|---|
| `SENTINEL_CAMERA_ACTIVITY_STALENESS_MINUTES` | `const.py` | `10` | Staleness gate for `alarm_disarmed_during_external_threat`. Camera activity must be within this window of the snapshot to fire the rule. |
| `SENTINEL_OCCUPANCY_ARMED_STATES` | `const.py` | `{"armed_home", "armed_night"}` | Alarm states treated as occupancy-compatible. These states never trigger an `alarm_state_mismatch` finding when `expected_presence=home`. |
| `SENTINEL_ADMISSION_TIMEOUT_S` | `core/utils.py` | `2.0` (s) | How long a Sentinel LLM call waits for the chat/video foreground to become idle before deferring. Keeps interactive latency unaffected by Sentinel background work. |
| `_SENTINEL_STARVATION_WARN_S` | `core/utils.py` | `300.0` (s) | Consecutive deferral duration that triggers a `degraded` sentinel health state and a WARNING log. |
| `_SENTINEL_CANCEL_WAIT_S` | `core/utils.py` | `2.0` (s) | Grace period given to a Sentinel task to cancel cleanly when a chat turn begins. |

---

## Sentinel Autonomy Level

**File:** `const.py` | **UI-configurable**

| Constant | Config key | Default | Purpose |
|---|---|---|---|
| `RECOMMENDED_SENTINEL_AUTONOMY_LEVEL` | `sentinel_autonomy_level` | `1` | `0` = passive (no notifications); `1` = notify only; `2` = suggest actions; `3` = act autonomously |
| `RECOMMENDED_SENTINEL_RUNTIME_OVERRIDE_TTL_MINUTES` | `sentinel_runtime_override_ttl_minutes` | `60` | How long a runtime autonomy-level override lasts before reverting to the configured value |
| `RECOMMENDED_SENTINEL_REQUIRE_PIN_FOR_LEVEL_INCREASE` | `sentinel_require_pin_for_level_increase` | `False` | Require the Sentinel PIN to increase autonomy level via the `sentinel_set_autonomy_level` service |

---

## Sentinel Auto-Execution (Level 2+)

**File:** `const.py` | **UI-configurable**

| Constant | Config key | Default | Purpose |
|---|---|---|---|
| `RECOMMENDED_SENTINEL_AUTO_EXECUTION_ENABLED` | `sentinel_auto_execution_enabled` | `False` | Allow Sentinel to execute suggested actions autonomously (requires autonomy level ≥ 2) |
| `RECOMMENDED_SENTINEL_AUTO_EXECUTE_DEFAULT_MIN_CONFIDENCE` | `sentinel_auto_execute_default_min_confidence` | `0.70` | Minimum finding confidence score required for auto-execution |
| `RECOMMENDED_SENTINEL_AUTO_EXECUTE_MAX_ACTIONS_PER_HOUR` | `sentinel_auto_execute_max_actions_per_hour` | `5` | Rate cap on autonomous actions to prevent runaway execution |
| `RECOMMENDED_SENTINEL_EXECUTION_IDEMPOTENCY_WINDOW_MINUTES` | `sentinel_execution_idempotency_window_minutes` | `15` | Duplicate-execution suppression window; the same action on the same finding will not fire twice within this period |
| `RECOMMENDED_SENTINEL_AUTO_EXEC_CANARY_MODE` | `sentinel_auto_exec_canary_mode` | `False` | Log what auto-execution *would* do without actually executing. Useful for validating policy before enabling live execution. |
| `RECOMMENDED_SENTINEL_AUTO_EXECUTE_ALLOWED_SERVICES` | `sentinel_auto_execute_allowed_services` | `[]` | Explicit allowlist of HA service calls Sentinel may invoke autonomously. Empty = no services permitted. |

**Code-only:**

| Constant | File | Value | Purpose |
|---|---|---|---|
| `_MIN_AUTO_EXECUTE_LEVEL` | `sentinel/execution.py` | `2` | Minimum autonomy level required for any auto-execution attempt |

---

## Sentinel Triage

**File:** `const.py` | **UI-configurable**

| Constant | Config key | Default | Purpose |
|---|---|---|---|
| `RECOMMENDED_SENTINEL_TRIAGE_ENABLED` | `sentinel_triage_enabled` | `False` | Enable LLM triage pass before notification (requires autonomy ≥ 1) |
| `RECOMMENDED_SENTINEL_TRIAGE_TIMEOUT_SECONDS` | `sentinel_triage_timeout_seconds` | `10` (s) | Max wait for triage LLM response. On timeout the decision defaults to `notify`. |

---

## Sentinel Baseline

**File:** `const.py` | **UI-configurable**

| Constant | Config key | Default | Purpose |
|---|---|---|---|
| `RECOMMENDED_SENTINEL_BASELINE_ENABLED` | `sentinel_baseline_enabled` | `False` | Enable rolling statistical baseline collection |
| `RECOMMENDED_SENTINEL_BASELINE_UPDATE_INTERVAL_MINUTES` | `sentinel_baseline_update_interval_minutes` | `15` | How often baselines are recalculated and written to PostgreSQL |
| `RECOMMENDED_SENTINEL_BASELINE_FRESHNESS_THRESHOLD_SECONDS` | `sentinel_baseline_freshness_threshold_seconds` | `3600` (1 h) | Age after which a baseline is considered stale |
| `RECOMMENDED_SENTINEL_BASELINE_MIN_SAMPLES` | `sentinel_baseline_min_samples` | `20` | Minimum observations before a global baseline produces findings |
| `RECOMMENDED_SENTINEL_BASELINE_MAX_SAMPLES` | `sentinel_baseline_max_samples` | `500` | Rolling window size; the oldest sample is dropped when this is exceeded |
| `RECOMMENDED_SENTINEL_BASELINE_DRIFT_THRESHOLD_PCT` | `sentinel_baseline_drift_threshold_pct` | `30.0` | Percent deviation from rolling average that triggers a `baseline_deviation` or `time_of_day_anomaly` finding |
| `RECOMMENDED_SENTINEL_BASELINE_WEEKLY_PATTERNS` | `sentinel_baseline_weekly_patterns` | `False` | Enable per-(day-of-week, hour) baselines for `time_of_day_anomaly`. Requires more data to warm up. |
| `RECOMMENDED_SENTINEL_BASELINE_DOW_MIN_SAMPLES` | `sentinel_baseline_dow_min_samples` | `4` | Observations per DOW-hour slot before the DOW blend weight reaches 1.0. Lower than global min because DOW slots update at most once per week. |
| `RECOMMENDED_SENTINEL_BASELINE_SUSTAINED_MINUTES` | `sentinel_baseline_sustained_minutes` | `20` | For cyclic loads (fridge, compressor), the deviation must persist this long before firing. Set `0` to disable the sustained gate. |

**Code-only:**

| Constant | File | Value | Purpose |
|---|---|---|---|
| `COMPLETION_THRESHOLD_PCT` | `sentinel/baseline.py` | `0.10` | Appliance cycle-completion threshold: power must drop below 10% of baseline to be considered complete |
| `COMPLETION_MIN_ACTIVE_WATTS` | `sentinel/baseline.py` | `100.0` (W) | Minimum baseline wattage to consider cycle-completion detection meaningful |
| `COMPLETION_RECENCY_SECS` | `sentinel/baseline.py` | `900` (s) | Recency window for completion detection |
| `MINIMUM_POWER_DEVIATION_WATTS` | `sentinel/baseline.py` | `50.0` (W) | Absolute minimum power deviation required to fire a baseline anomaly, even if percent threshold is met |

---

## Sentinel Notifications and Suppression

**File:** `const.py` | **UI-configurable**

| Constant | Config key | Default | Purpose |
|---|---|---|---|
| `RECOMMENDED_SENTINEL_QUIET_HOURS_SEVERITIES` | `sentinel_quiet_hours_severities` | `["low"]` | Which finding severities are suppressed during configured quiet hours |
| `RECOMMENDED_SENTINEL_PRESENCE_GRACE_MINUTES` | `sentinel_presence_grace_minutes` | `10` | After everyone leaves home, suppress `open_entry_while_away` and `unknown_person_camera_no_home` for this many minutes to allow time for departure actions to settle |
| `RECOMMENDED_SENTINEL_DAILY_DIGEST_ENABLED` | `sentinel_daily_digest_enabled` | `False` | Send a daily push summary of the past 24 hours |
| `RECOMMENDED_SENTINEL_DAILY_DIGEST_TIME` | `sentinel_daily_digest_time` | `08:00:00` | Local delivery time for the daily digest |

**Code-only:**

| Constant | File | Value | Purpose |
|---|---|---|---|
| `MAX_MOBILE_MESSAGE_CHARS` | `sentinel/notifier.py` | `220` | Hard character cap on mobile push notification text. Longer explanation text is truncated or replaced with a deterministic fallback. |
| `MAX_COOLDOWN_MULTIPLIER` | `sentinel/suppression.py` | `8` | Maximum factor by which learned feedback (dismiss/snooze) can extend the base cooldown for a rule |
| `PENDING_PROMPT_DEFAULT_TTL` | `sentinel/suppression.py` | `4 h` | Default TTL for pending action prompts in the suppression store (matches `RECOMMENDED_SENTINEL_PENDING_PROMPT_TTL_MINUTES`) |

---

## Sentinel Discovery

**File:** `const.py` | **UI-configurable**

| Constant | Config key | Default | Purpose |
|---|---|---|---|
| `RECOMMENDED_SENTINEL_DISCOVERY_ENABLED` | `sentinel_discovery_enabled` | `False` | Enable LLM-based rule candidate discovery |
| `RECOMMENDED_SENTINEL_DISCOVERY_INTERVAL_SECONDS` | `sentinel_discovery_interval_seconds` | `3600` (1 h) | How often a discovery cycle runs |
| `RECOMMENDED_SENTINEL_DISCOVERY_MAX_RECORDS` | `sentinel_discovery_max_records` | `200` | Maximum stored discovery records (older records are pruned) |

---

## Critical Action PIN

**File:** `const.py` | **UI-configurable**

| Constant | Value | Purpose |
|---|---|---|
| `CRITICAL_PIN_MIN_LEN` | `4` | Minimum PIN length (digits) |
| `CRITICAL_PIN_MAX_LEN` | `10` | Maximum PIN length (digits) |

The `RECOMMENDED_CRITICAL_ACTIONS` list defines which HA service calls require PIN confirmation. By default this covers `lock.unlock`, `lock.open`, and `cover.open_cover` / `cover.open` for entities whose `entity_id` contains `door`, `gate`, or `garage`.

---

## Face Recognition

**File:** `const.py` | **UI-configurable**

| Constant | Config key | Default | Purpose |
|---|---|---|---|
| `RECOMMENDED_FACE_RECOGNITION` | `face_recognition` | `False` | Enable face recognition in camera analysis |
| `RECOMMENDED_FACE_API_URL` | `face_api_url` | `http://face-recog-server.local:8000` | Base URL of the [face-service](https://github.com/goruck/face-service) instance |

---

## Speech-to-Text (STT)

**File:** `const.py` | **UI-configurable**

| Constant | Config key | Default | Purpose |
|---|---|---|---|
| `RECOMMENDED_OPENAI_STT_MODEL` | `model_name` | `gpt-4o-mini-transcribe` | Default OpenAI Whisper model for the STT provider. Supported values: `whisper-1`, `gpt-4o-transcribe`, `gpt-4o-mini-transcribe`. |

---

## HTTP Endpoint

**File:** `http.py` | **Code-only**

| Constant | Value | Purpose |
|---|---|---|
| `MAX_UPLOAD_BYTES` | `10 485 760` (10 MB) | Maximum file size accepted by the face enrollment endpoint |
| `ALLOWED_EXTENSIONS` | `.jpg`, `.jpeg`, `.png`, `.gif`, `.webp` | File extensions permitted for face enrollment uploads |

---

## Module Tuning Constants

These constants live outside `const.py` in individual modules. They affect runtime behaviour but have no UI entry. Changing them requires editing the source file and reloading the integration.

### `agent/graph.py`

| Constant | Value | Purpose |
|---|---|---|
| `_LC_TOOL_TIMEOUT_S` | `30.0` (s) | Per-tool execution timeout. LangChain tools that do not respond within this window are cancelled. Raise only if slow external tools (e.g. a slow VLM) are routinely timing out. |

### `agent/rag_embedding_text.py`

| Constant | Value | Purpose |
|---|---|---|
| `EMBEDDING_INDEX_TEXT_MAX_CHARS` | `1200` | Maximum characters embedded per tool in the tool index. Controls how much of each tool's description is stored in the vector DB. |

### `core/video_analyzer.py`

| Constant | Value | Purpose |
|---|---|---|
| `_MAX_BATCH` | `5` | Maximum video frames processed per analysis batch |
| `_QUEUE_MAXSIZE` | `50` | Per-camera frame queue capacity. Frames are dropped when full. |
| `_FRAME_DEADLINE_SEC` | `600` (s) | Frames older than this are skipped to avoid processing stale data |
| `_SUMMARY_TIMEOUT_SEC` | `60` (s) | Timeout for the summarization model call in a video batch |
| `_FACE_TIMEOUT_SEC` | `10` (s) | Timeout for a face-recognition API call |
| `_VISION_TIMEOUT_SEC` | `90` (s) | Timeout for a VLM frame description call |
| `_VIDEO_MODEL_SEMAPHORE_WAIT_SEC` | `30` (s) | Max time a video frame waits for the concurrency semaphore before being dropped |
| `_VIDEO_QUEUE_BACKLOG_THRESHOLD` | `2` | Drop oldest queued frames when the backlog exceeds this depth |
| `_METRICS_REPORT_INTERVAL_SEC` | `3600` (s) | How often per-camera latency percentile metrics are logged |

### `core/video_helpers.py`

| Constant | Value | Purpose |
|---|---|---|
| `_MAX_SENTENCES` | `2` | Maximum sentences in a video caption |
| `_MAX_CHARS` | `300` | Maximum characters in a video caption |
| `_MAX_NAMES` | `2` | Maximum person names included in a single caption |
| `_UNIQUENESS_HASH_SIZE` | `8` | dHash grid size (8 → 64-bit perceptual hash) for frame deduplication |

### `core/utils.py`

| Constant | Value | Purpose |
|---|---|---|
| `SENTINEL_ADMISSION_TIMEOUT_S` | `2.0` (s) | How long Sentinel waits for the chat/video foreground to idle before deferring its LLM call |
| `_SENTINEL_STARVATION_WARN_S` | `300.0` (s) | Consecutive deferral duration that transitions `sentinel_health` to `degraded` |
| `_SENTINEL_CANCEL_WAIT_S` | `2.0` (s) | Grace period for Sentinel task cancellation when a chat turn begins |

### `sentinel/notifier.py`

| Constant | Value | Purpose |
|---|---|---|
| `MAX_MOBILE_MESSAGE_CHARS` | `220` | Hard character limit for mobile push text. Findings whose explanation exceeds this are replaced with a deterministic fallback message. |

### `sentinel/suppression.py`

| Constant | Value | Purpose |
|---|---|---|
| `MAX_COOLDOWN_MULTIPLIER` | `8` | Maximum multiplier applied to a rule's base cooldown based on accumulated snooze/dismiss feedback |

### `sentinel/baseline.py`

| Constant | Value | Purpose |
|---|---|---|
| `COMPLETION_THRESHOLD_PCT` | `0.10` | Appliance power must drop below 10% of its baseline to be treated as cycle-complete |
| `COMPLETION_MIN_ACTIVE_WATTS` | `100.0` (W) | Minimum active wattage for cycle-completion detection to be meaningful |
| `MINIMUM_POWER_DEVIATION_WATTS` | `50.0` (W) | Minimum absolute deviation (regardless of percent) before a baseline anomaly fires |

### `sentinel/execution.py`

| Constant | Value | Purpose |
|---|---|---|
| `_MIN_AUTO_EXECUTE_LEVEL` | `2` | Autonomy level floor for any auto-execution attempt |

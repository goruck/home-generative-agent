# TODOS

## Agent

### asyncio.gather concurrency policy for state-mutating tools

**What:** Add per-tool annotation or global policy for whether a tool is safe to run concurrently. State-mutating HA tools (`turn_on`, `turn_off`, lock, unlock, `alarm_control`) called in the same model batch could interleave under `asyncio.gather`.

**Why:** With `asyncio.gather` (introduced in feat/streaming-chatlog), a model turn that includes both `turn_on` and `get_state` may now run concurrently. `get_state` might return stale state if it completes before `turn_on` finishes. Previously sequential. Flagged during eng review Codex outside voice.

**How to apply:** In `graph.py`, add a `_SEQUENTIAL_TOOLS` set or per-tool `safe_to_parallelize` annotation. In `_call_tools`, run sequential tools before the gather batch, or use `asyncio.gather` only for tools not in the set. Alternatively, add a note in the integration docs that sequential ordering of state-changing + read-back calls requires separate model turns.

**Effort:** M
**Priority:** P3
**Depends on:** feat/streaming-chatlog

---

### Rename sync methods with `_async_` prefix in conversation.py

**What:** Three synchronous methods in `HGAConversationEntity` have the `_async_` prefix, which conventionally means "coroutine" in HA code: `_async_get_message_history` (line 629), `_async_get_all_tools` (line 685), `_async_render_system_prompt` (line 718). None use `await`.

**Why:** Misleading naming. Future maintainers may incorrectly assume these are coroutines.

**How to apply:** Rename each to drop `_async_` prefix and update all callers within `conversation.py`.

**Effort:** S
**Priority:** P3
**Depends on:** feat/streaming-chatlog

---

### Integration tests for streaming conversation path (HA fixture level)

**What:** Add four HA-fixture-level integration tests using a real (mocked) LangGraph + HA conversation entity: (1) single-turn text-only streaming, (2) multi-turn with tool calls, (3) PIN flow multi-turn with confirmation, (4) `schema_first_yaml=True` fallback fires `ainvoke` path correctly.

**Why:** Current test coverage (58%) is dominated by pure-function unit tests. The `HGAConversationEntity` integration methods (`_async_run_astream`, `_async_handle_message`, `_async_render_system_prompt`, `_async_init_llm_apis`) have zero unit test coverage. These tests were listed as required in the streaming design plan but deferred at ship time. Accepted risk per user override.

**How to apply:** Use the existing `test_conversation.py` integration test harness as a model. Create `tests/custom_components/home_generative_agent/test_conversation_stream_integration.py`. Mock `app.astream_events` to return a controlled sequence of events, verify delta sequence delivered to HA ChatLog.

**Effort:** M
**Priority:** P2
**Depends on:** feat/streaming-chatlog

---

### Integration smoke test: on_tool_end propagation during action node

**What:** After feat/streaming-chatlog lands, run a real multi-tool conversation (`get_current_time` + `get_and_analyze_camera_image`) and verify that `on_tool_end` for `get_current_time` fires BEFORE the camera tool completes.

**Why:** The streaming win depends on LangGraph propagating child `on_tool_end` events from `lc_tool.ainvoke()` to the outer `astream_events` DURING node execution. Verified against LangGraph 1.1.2 source during planning, but not confirmed via integration test. If LangGraph buffers nested events until node completion, the streaming gain disappears.

**How to verify:** Add a timing log in the `on_tool_end` handler (DEBUG level). Time delta between `on_tool_end` for the time tool and the camera tool should be ~3980ms apart, not ~0ms.

**Effort:** S
**Priority:** P2 (post-ship validation)
**Depends on:** feat/streaming-chatlog
**Completed:** v3.12.0 (2026-04-21)

---

## Explain / Prompts

### Sanitize area/entity strings before injecting into LLM prompts

**What:** Strip or truncate user-editable strings (area names, entity IDs, friendly names) to a safe character set before including them in `USER_PROMPT_TEMPLATE` evidence interpolation.

**Why:** HA area names are user-editable via UI or YAML and flow into `USER_PROMPT_TEMPLATE.format(evidence=...)` as raw Python dict repr without sanitization. A malicious area name (e.g., `"Front\nIgnore all previous instructions."`) constitutes a prompt injection vector. The `camera_area` and `unsecured_entity_areas` fields added in PR fixing camera_entry_unsecured notifications increase the surface area (same area string, multiple occurrences). Flagged as P3 since it requires a coordinated attacker with HA admin access to be exploitable.

**How to apply:** Add a sanitization helper in `explain/` that replaces non-printable characters and control sequences in string values before dict repr serialization. Alternatively, serialize evidence as JSON with explicit schema validation rather than using Python repr. Add a test with a crafted area name containing injection characters.

**Effort:** S
**Priority:** P3
**Depends on:** None

---

## Sentinel Rules

### Add `sentinel_camera_entry_links` config for explicit camera-to-entry mapping

**What:** Add a `sentinel_camera_entry_links` config key (Sentinel subentry options flow) that allows users to explicitly associate cameras with entry sensors regardless of HA area assignment. Format: `{camera_entity_id: [entry_entity_id, ...]}`.

**Why:** Removing the home-wide fallback from `camera_entry_unsecured` (PR fixing cross-area false spatial claims) creates false negatives for adjacent-area setups — e.g., a driveway camera in "Outside" area should still fire when the front door lock in "Front" area is unsecured. Area-based association is insufficient for these layouts. Flagged as an accepted trade-off during eng review and Codex outside voice.

**How to apply:** In `const.py`, add `CONF_SENTINEL_CAMERA_ENTRY_LINKS`. In the Sentinel config flow subentry, add an optional text/JSON field. In `camera_entry_unsecured.py`, after the same-area unsecured lookup, check the config for explicit links for the current camera; merge any linked entities into `unsecured`. Add `unsecured_entity_areas` entries for the linked entities with their actual areas.

**Effort:** M
**Priority:** P2
**Depends on:** None
**Completed:** v3.8.0 (2026-04-05)

---

### Add `unknown_person_camera_night_home` branch to `_covered_builtin_rule_for_candidate()`

**What:** Add a detection branch in `_covered_builtin_rule_for_candidate()` (`__init__.py`) that recognizes a discovery candidate as already covered by the `unknown_person_camera_night_home` static rule.

**Why:** The function currently has branches for vehicle-near-camera and camera-missing-snapshot, but not for unknown-person-at-night-on-camera. The LLM is prevented from re-suggesting this topic via `_STATIC_RULE_IDS` in `discovery_engine.py`, so the missing branch has no user-visible impact under normal operation. However, if the LLM ignores the exclusion list and generates an `unknown_person` candidate, the proposal approval flow won't detect the overlap with the existing static rule — the candidate will be treated as novel and sent to the UI card instead of being silently filtered.

**How to apply:** In `_covered_builtin_rule_for_candidate()`, add a branch after the vehicle and camera-snapshot branches: if `camera_entity is not None` AND `("unknown" in text or "unrecognized" in text or "stranger" in text)` AND `"night" in text` AND `any(term in text for term in ("home", "resident", "occupant"))`, return `("unknown_person_camera_night_home", [camera_entity])`. Add a corresponding test in `test_sentinel_services.py`.

**Effort:** S
**Priority:** P3
**Depends on:** Static rule generalization PR (sentinel static rule generalization sweep)

---

## Audit Store

### Severity-aware eviction for audit store

**What:** Extend the priority eviction helper introduced in the audit-flood fix to also prefer dropping `low`-severity records before `medium`, and `medium` before `high`. Currently all suppressed records are treated equally by the eviction priority.

**Why:** Preserves high-severity findings during high-volume periods. After the flood fix, `not_suppressed` records are protected, but within the suppressed pool, a `high`-severity triage-suppressed finding is no more protected than a `low`-severity one. During an active security event, the store could still discard high-severity triage-suppressed findings before low-severity ones.

**How to apply:** Extend `_is_evictable(record) -> bool` in `audit/store.py` to a scored `_eviction_priority(record) -> int` that returns (lower = evict first): 0 = suppressed+low, 1 = suppressed+medium, 2 = suppressed+high, 3 = not_suppressed. Eviction picks the record with the lowest score (ties broken by age — oldest evicted first).

**Effort:** S
**Priority:** P3
**Depends on:** Audit flood fix (priority eviction PR)

---

### Trigger drop alert automation

**What:** Document an HA automation blueprint that fires a persistent notification when `state_attr('sensor.sentinel_health', 'triggers_dropped_incoming') | int > 0`. The new `triggers_dropped_incoming` attribute (added in the audit flood fix) signals that incoming Sentinel triggers were lost because the queue was full of security-critical entries.

**Why:** A `triggers_dropped_incoming > 0` means Sentinel may have missed a run that could have detected a real anomaly (unlocked door, unsecured camera entry). Without an alert, this is invisible to the operator.

**How to apply:** Add a Lovelace dashboard example card and a YAML automation snippet to `README.md` covering: (1) threshold alert on `triggers_dropped_incoming`, (2) reset guidance (check `SENTINEL_INTERVAL_SECONDS`, investigate high-frequency entity sources).

**Effort:** S
**Priority:** P3
**Depends on:** Audit flood fix (trigger drop counters PR)

---

## Baseline

### Config flow UI for CONF_SENTINEL_BASELINE_MIN_SAMPLES

**Completed:** v3.11.0 (2026-04-14)

`NumberSelector` added to `sentinel_subentry_flow.py` (min: 1, max: 500, step: 1, default: 20). Also added `sentinel_baseline_sustained_minutes` selector in the same PR.

---

### Incident lifecycle control for repeated deviation notifications

**What:** Replace per-entity-run notification tracking with a stable incident abstraction: key per `entity_id + template_id`, hold incident open until entity returns below threshold, notify once per incident. Suppresses repeated alerts for any entity, not just named cyclers. Clear incident when the entity is absent from findings for one full run.

**Why:** The v3.11.0 cyclical load gate (fridge/freezer/compressor) fixes the immediate fridge spam problem, but the root cause is deeper: Sentinel lacks any concept of "same condition still active." Every run where an entity is above threshold produces a new finding, and only the cooldown prevents repeated notification. The incident abstraction fixes this for all entities without requiring appliance name classification.

**How to apply:** Add `_open_incidents: dict[str, IncidentState]` to `SentinelEngine.__init__`. `IncidentState` holds `opened_at`, `last_seen_at`, `notified: bool`. In a new `_apply_incident_control()` method, gate all findings: suppress if `notified=True` and entity still in `_open_incidents`; fire if not yet notified; clear if entity absent this run.

**Effort:** M
**Priority:** P2
**Depends on:** Cyclical load gate (v3.11.0)

---

### Cyclical load gate: notification body with duration

**What:** When the sustained gate fires, include elapsed time in the notification body ("Fridge compressor has been running for 22 minutes — possible problem?"). Reads `_cyclical_deviation_above_since[entity_id]` and formats elapsed time.

**Effort:** S
**Priority:** P3
**Depends on:** Cyclical load gate (v3.11.0)

---

### Expand CYCLICAL_LOAD_HINTS to include HVAC/heat/AC/water heater

**What:** Add `hvac`, `heat`, `heatpump`, `aircon`, `airconditioner`, `waterheater`, `tankless` to `CYCLICAL_LOAD_HINTS` in `sentinel/baseline.py`.

**Why:** These appliances cycle normally (HVAC compressor, water heater element) and can generate the same notification spam as fridges. Deferred from v3.11.0 because an HVAC running at 3am away-mode IS an anomaly worth surfacing — the suppression tradeoff is non-trivial and needs evaluation before gating.

**How to apply:** Before expanding hints, evaluate false-negative rate by checking whether any real HVAC away-mode anomalies have been observed in production. Add hint only if HVAC cycling during normal occupancy is reliably distinguishable from anomalous HVAC usage.

**Effort:** S
**Priority:** P3
**Depends on:** Cyclical load gate (v3.11.0), field observation data

---

### Lovelace health card example for baseline attrs

**What:** Add a Lovelace dashboard card YAML snippet to `README.md` showing `baseline_entity_count`, `baseline_fresh_count`, and `baseline_rules_waiting` from `sensor.sentinel_health`.

**Why:** Users enabling baseline collection have no visual confirmation it's working. The new health attrs are invisible unless a user knows to inspect the sensor. A dashboard example closes the discoverability gap and pairs naturally with the existing trigger-drop alert example.

**How to apply:** Under the Baseline section of `README.md`, add a Lovelace glance card example using `sensor.sentinel_health` attributes: `baseline_entity_count` (how many entities are tracked), `baseline_fresh_count` (how many have recent updates), `baseline_rules_waiting` (rules not yet firing due to min-sample gate).

**Effort:** S
**Priority:** P3
**Depends on:** Baseline enhancement PR (sentinel-baseline-enhancement)

---

### Weekly / day-of-week baseline patterns

**What:** Extend baseline collection to store `hourly_avg_{DOW}_{H}` metrics (e.g., `hourly_avg_1_14` = Monday 2PM). Gives 7×24=168 time slots per entity instead of 24, enabling time-of-day anomaly detection that accounts for weekday vs. weekend patterns.

**Why:** The current `hourly_avg_H` treats all Mondays and Sundays at 2PM the same. For most households, weekday and weekend patterns differ significantly (cooking appliances, HVAC, occupancy). A washing machine running at 3AM on a Saturday is less anomalous than at 3AM on a Tuesday. Without DOW awareness, `time_of_day_anomaly` generates false positives on weekends.

**How to apply:** Add `hourly_avg_{DOW}_{H}` as a third metric row per entity per update cycle. Update `evaluate_time_of_day_anomaly()` to prefer the DOW-specific metric when available, falling back to the global `hourly_avg_H` if not yet established. New config option `CONF_SENTINEL_BASELINE_WEEKLY_PATTERNS` (default: False) to opt in.

**Effort:** M
**Priority:** P2
**Depends on:** Baseline enhancement PR
**Completed:** v3.9.0 (2026-04-06)

---

## Discovery

### Tighten discovery prompt to require entity-backed evidence paths

**Completed:** v3.9.0 (2026-04-06)

Entity-backed evidence path instruction added to `USER_PROMPT_TEMPLATE` in `explain/discovery_prompts.py`. `_filter_novel_candidates()` in `explain/discovery_engine.py` now guards against derived-only paths. Tests added for the filter.

---

### Wire `proposals_promoted` counter in discovery engine

**What:** `SentinelHealthSensor` now exposes `discovery_proposals_approved_24h` — the count of proposals with `status="approved"` in the last 24 hours, queried directly from `ProposalStore` (Option B from the original TODO). The bare `proposals_promoted` in-memory counter (which always reported 0) was removed.

**Why:** The counter was added to the health sensor attributes in v3.7.0 but the increment logic was not wired. Option B (direct store query) is simpler and doesn't require engine changes.

**Effort:** S
**Priority:** P1
**Depends on:** v3.7.0 (health sensor discovery metrics)
**Completed:** v3.7.1 (2026-04-04)

---

## Video Analyzer

### Caption novelty: per-analysis notification-status metadata

**What:** Store whether each video analysis triggered a notification alongside the
caption in the vector store. Add `notified`, `decision_reason`, and `matched_key`
fields to the stored value written by `_store_results`. Update `_is_caption_novel`
to optionally filter `store.asearch` results to notification-worthy records only,
so suppressed artifact captions do not inflate the similarity baseline.

**Why:** `_store_results` is called unconditionally, meaning suppressed captions
(e.g. repeated nighttime blur) are stored and can later cause a genuinely new
artifact caption to be suppressed against a non-notified prior. Filtering by
`notified=True` in the search would improve precision but requires a metadata
field and a query-time filter that the store API must support.

**How to apply:** Add `notified: bool`, `decision_reason: str`, and
`matched_caption_key: str | None` fields to the dict written in `_store_results`.
Pass `decision.notify` and `decision.reason` from `_handle_notification` into
`_store_results`. Update `store.asearch` call to include a metadata filter when
the store API supports it. Until the filter lands, the current behavior (compare
against all stored analyses) is acceptable.

**Effort:** M
**Priority:** P3
**Depends on:** v3.14.0 (CaptionNoveltyDecision)

---

### Caption novelty: tune threshold and artifact terms from real logs

**What:** Review accumulated debug logs from `_is_caption_novel` decisions to
check whether `VIDEO_ANALYZER_SIMILARITY_THRESHOLD = 0.85` and the `_ARTIFACT_RE`
vocabulary produce the right suppress/notify balance in production.

**Why:** The threshold and terms were set conservatively. Real logs can reveal
common false-positives (suppressing events that should notify) or false-negatives
(notifying on repeated low-value artifacts). Adjust as data accumulates.

**Effort:** S
**Priority:** P3
**Depends on:** v3.14.0 (decision logging)

---

### save_and_analyze_snapshot tmp files leak in the latest/ subfolder

**What:** The service writes `snapshot_<ts>.jpg` into `latest_target(...).parent`
(the `latest/` subfolder), publishes a copy to `latest.jpg`, and never removes
the tmp file. `_prune_old_snapshots` deliberately never deletes files in the
`latest/` subfolder (guard re-appends and breaks), so these files can neither
be registered nor swept by the current mechanism.

**Why:** Each service call leaks one file. Low volume (manual/automation
calls), but unbounded. Registering them is not an option without rethinking
the latest-subfolder guard — a registered `latest/` file would permanently
clog the deque head.

**How to apply:** Either unlink the tmp file after `publish_latest_atomic`
(it's a copy, the dst survives), or write the tmp into the camera's normal
snapshot directory so capture-time retention covers it. Unlinking after
publish is simplest; check the bus event's `"path": str(tmp_path)` consumers
first (same dangling-path concern as the suppressed-notification TODO).

**Effort:** S
**Priority:** P3

---

### Bus event path can dangle when notification is suppressed

**What:** In `notify_on_anomaly` mode with `decision.notify=False`,
`protect_notify_image` is never called, but the chosen frame was already
published as `latest.jpg` and announced on `hga_last_event_frame` with
`"path": str(chosen)`. Consumers resolving that path can read a pruned file.
The notify-frame fix biases `chosen` toward the batch head (oldest end of the
retention deque), shortening time-to-dangle.

**Why:** Suppressed-notification events hand out a path with no pruning
protection. The `latest` dst copy is stable; the raw `path` is not. Test
`test_handle_notification_suppresses_when_decision_is_no_notify` currently
asserts protect is NOT called, so changing this is a deliberate behavior change.

**How to apply:** Either call `protect_notify_image(chosen)` unconditionally
before the mode branch, or document that consumers must use `latest` and keep
`path` best-effort. Update the suppression test accordingly.

**Effort:** S
**Priority:** P2

---

### Dispatch frame epoch instead of utcnow() as latest-frame timestamp

**What:** `SIGNAL_HGA_NEW_LATEST` and `SIGNAL_HGA_RECOGNIZED` dispatch
`dt_util.utcnow().isoformat()` as the frame timestamp. The notify-frame fix
preferentially selects early person frames, so for long held event-buffer
batches the image entity and sensor can label a frame that is minutes old as
captured "now".

**Why:** The frame's true epoch is already available (`epoch_from_path(chosen)`
/ the `ordered` tuples) but is discarded. Pre-existing pattern; skew widened by
early-frame selection.

**How to apply:** Thread the chosen frame's epoch through
`_finalize`/`_handle_notification` and dispatch it as the timestamp.

**Effort:** S
**Priority:** P3

---

## Notifier / Observability

### Feedback-trained per-entity cooldowns — wire feedback signal

**What:** `record_cooldown_feedback(state, entity_id, rule_type)` is now called from both the snooze action (`sentinel/notifier.py`) and the dismiss action (`notify/actions.py`). Each snooze or dismiss of a rule+entity pair increments the compound-key multiplier, which extends future cooldowns for that specific combination.

**Why:** Without the feedback signal, `learned_cooldown_multipliers` remained empty forever. Now every snooze/dismiss trains the system.

**Effort:** S
**Priority:** P1
**Depends on:** v3.7.0 (suppression schema v4)
**Completed:** v3.7.1 (2026-04-04)

---

### Fix cooldown multiplier key scheme (entity_id → rule_type:entity_id) + schema migration v5

**What:** `learned_cooldown_multipliers` is now keyed by `"{rule_type}:{entity_id}"` (e.g., `"unlocked_lock_at_night:lock.front_door"`). The v4→v5 migration in `_migrate_suppression_state()` discards all bare entity_id keys (safe: `record_cooldown_feedback` was never called in v3.7.0 production, so v4 dicts were always empty). `stored_version = 5` correctly set after migration.

**Why:** The bare entity_id key caused different rules for the same entity to share a single multiplier, causing missed alerts for the more critical rule.

**Effort:** S
**Priority:** P1
**Depends on:** Wire feedback signal TODO above
**Completed:** v3.7.1 (2026-04-04)

---

### Daily digest config flow UI

**What:** `sentinel_subentry_flow.py` now exposes `BooleanSelector` for `CONF_SENTINEL_DAILY_DIGEST_ENABLED` and `TimeSelector` for `CONF_SENTINEL_DAILY_DIGEST_TIME`. Both appear in `_default_payload()`. `RECOMMENDED_SENTINEL_DAILY_DIGEST_TIME` normalized to `"08:00:00"` in `const.py` to match `TimeSelector` output format. The notifier parse bug (`split(":", 1)` → `split(":")`) was fixed as part of this.

**Why:** The daily digest shipped in v3.7.0 with no UI control; users had to edit raw options.

**Effort:** S
**Priority:** P1
**Depends on:** v3.7.0 (daily digest backend)
**Completed:** v3.7.1 (2026-04-04)

---

### Add `learned_suppressions_active` attribute to health sensor

**Completed:** v3.9.0 (2026-04-06)

`learned_suppressions_active` attribute exposed on `sensor.sentinel_health`. Count reads `learned_cooldown_multipliers` from suppression state via `engine.learned_suppressions_count` property.

---

## Completed

### Snapshot retention misses batches that never reach _finalize

**What:** Deletion is deque-driven (in-memory), but registration was coupled to
successful analysis completion (`_finalize`). Six runtime paths abandoned
captured files without registration: the two `_analyze_and_finalize` early
returns, the worker's blanket exception handler, notify/store exceptions inside
`_finalize`, the `_get_batch` backlog drop, and event-hold-buffer eviction.

**Why:** VLM outages and backlogs made these paths routine; files accumulated
unboundedly. Third occurrence of the same bug class (#489 dedupe-skip was a
point patch), so the fix moved registration to capture success in
`_capture_snapshot` — a single site that makes every downstream drop
retention-irrelevant — and removed the per-site registrations. In-flight frames
(≤ ~105/camera) can never reach the deque's pop position (cap 200).

**Effort:** S
**Priority:** P1
**Completed:** v3.18.2 (2026-07-19)

### Restart orphans snapshot files predating the restart

**What:** Retention deques are in-memory with no filesystem sweep, so every
on-disk snapshot from before an HA restart was orphaned forever. Fixed with a
one-shot startup task (`_seed_retention_from_disk`) that scans each
`camera_*` snapshot directory (skipping `latest/` and non-camera dirs like
`faces/`), merges pre-existing files into the retention deque as the oldest
entries (never after live captures, so new frames can't be rotated out
first), and runs the normal budget shrink with the usual latest-asset and
protection guards.

**Effort:** M
**Priority:** P2
**Completed:** v3.18.2 (2026-07-19)

### iOS notification priority tiers

**What:** `sentinel/notifier.py` now uses `_SEVERITY_INTERRUPT_LEVEL` (`high` → `time-sensitive`, `medium` → `active`, `low` → `passive`) and `_SEVERITY_TITLE` dicts. `async_notify()` derives severity from the finding, selects the interrupt level and title, and passes `"push": {"interruption-level": level}` in the notification data block.

**Why:** All notifications previously used the same `active` interruption level, training users to ignore them — including security alerts.

**Effort:** S
**Priority:** P1
**Completed:** v3.6.9 (2026-03-31)

---

### Appliance completion detection in baseline deviation

**What:** `sentinel/baseline.py` now detects appliance cycle completion: power-class entities on dedicated appliance circuits (washer, dryer, dishwasher, etc.) with `current_value < COMPLETION_THRESHOLD_PCT × baseline_value` emit `severity="low"` with `evidence["is_completion"] = True`. `sentinel/notifier.py` checks `finding.evidence.get("is_completion")` and uses passive interruption level with completion framing.

**Why:** Washer/dryer finishing a cycle triggered `baseline_deviation` with high-severity framing ("stopped unexpectedly"), causing false-positive security alerts.

**Effort:** S
**Priority:** P1
**Completed:** v3.6.9 (2026-03-31)

---

### Presence-aware lock severity for `unlocked_lock_at_night`

**What:** `sentinel/rules/unlocked_lock_at_night.py` now reads `anyone_home` from `snapshot["derived"]` and emits `severity="low"` when home is occupied, `severity="high"` when away. Evidence dict includes `"anyone_home": anyone_home`.

**Why:** High-severity `time-sensitive` iOS alert for an unlocked door while occupants are home is noise that trains operators to ignore real alerts.

**Effort:** S
**Priority:** P1
**Completed:** v3.6.9 (2026-03-31)

---

### Notification batching for medium/low severity bursts

**What:** `SentinelNotifier` now tracks `_notification_times`, `_held_batch`, and `_batch_cancel`. When more than `_BATCH_RATE_LIMIT` medium/low pushes are sent within `_BATCH_WINDOW_SECONDS`, subsequent notifications are buffered and flushed as a single passive-priority summary after `_BATCH_FLUSH_DELAY_SECONDS`. High-severity findings always bypass batching.

**Why:** Burst of 7 notifications in 18 minutes cluttered the lock screen and trained users to dismiss without reading.

**Effort:** M
**Priority:** P1
**Completed:** v3.6.9 (2026-03-31)

---

### Centralize action code vocabulary in `const.py`

**What:** Defined `ACTION_CODES: dict[str, str]` in `const.py` mapping action code → plain English description. Imported it in `explain/prompts.py` to build the `SYSTEM_PROMPT` vocabulary line dynamically. Used constants in notifier and execution service instead of bare strings.

**Why:** `SYSTEM_PROMPT` contained a hardcoded list of action codes. Any new rule adding a new action code would silently cause the LLM to invent its own English meaning — recreating the exact class of production bug fixed in PR #346. Flagged independently by eng review and Codex outside voice.

**Effort:** S
**Priority:** P2
**Completed:** v3.7.0 (2026-04-03)

---

### Fix transitive union-find spatial contamination in correlator

**What:** Added `_eject_camera_area_violations()` post-grouping pass to `SentinelCorrelator.correlate()`. For any group containing a `camera_entry_unsecured` finding with a known area, ejects any other finding whose area is non-empty and differs from the camera's area, returning it as a singleton.

**Why:** The area-aware `_COMPLEMENTARY_PAIRS_REQUIRE_AREA_MATCH` guard only protects direct pairwise checks. With three simultaneous findings (camera Front + lock Front + entry Garage), the camera in Front ended up in a compound with the Garage entry via the bridging lock — the exact false spatial claim the camera_entry_unsecured fix was designed to prevent.

**Effort:** S
**Priority:** P2
**Completed:** v3.7.0 (2026-04-03)

---

### Discovery quality metrics on health sensor

**What:** `SentinelDiscoveryEngine` now tracks per-cycle stats (`candidates_generated`, `candidates_novel`, `candidates_deduplicated`, `proposals_promoted`, `unsupported_ttl_expired`) in `_discovery_cycle_stats`. `SentinelHealthSensor` exposes these as attributes via `SIGNAL_SENTINEL_RUN_COMPLETE`.

**Why:** After the deduplication fix, operators had no visibility into whether dedup was working — they couldn't distinguish "LLM generating zero new ideas" from "LLM generating many but all caught by dedup."

**Effort:** S
**Priority:** P2
**Completed:** v3.7.0 (2026-04-03)

---

### Daily digest notification

**What:** `SentinelEngine` now schedules a daily push notification summarizing the past 24h findings. Controlled by `CONF_SENTINEL_DAILY_DIGEST_HOUR` (default 07:00). Duplicate digest triggers within a session are deduplicated by checking `self._digest_task.done()` before scheduling.

**Why:** Operators had no visibility into Sentinel activity without actively checking the audit store or Lovelace. A morning summary gives awareness without intra-day noise.

**Effort:** S
**Priority:** P2
**Completed:** v3.7.0 (2026-04-03)

---

### `trigger_source` breakdown on health sensor

**What:** `SentinelHealthSensor` now exposes `trigger_source_breakdown: {"poll": N, "event": M, "on_demand": K}` as a rolling 24-hour count attribute, populated by querying `AuditStore` during the health update cycle.

**Why:** `trigger_source` was populated in audit records but not surfaced anywhere useful. Operators could not tell if Sentinel was poll-heavy vs event-driven without inspecting raw audit records.

**Effort:** S
**Priority:** P2
**Completed:** v3.7.0 (2026-04-03)

---

### Build Operational Health Entity (sentinel_plan.md §11)

**What:** Registered `SentinelHealthSensor` (`core/sentinel_health_sensor.py`) — a `SensorEntity` that exposes Sentinel KPIs as HA attributes: `last_run_start`, `last_run_end`, `run_duration_ms`, `active_rule_count`, `trigger_source_stats`, `findings_count_by_severity`, `triage_suppress_rate`, `auto_exec_count`, `auto_exec_failures`, `false_positive_rate_14d`, `action_success_rate`, `user_override_rate`. Added `_timed_run` wrapper to `SentinelEngine` for per-run timing telemetry and `SIGNAL_SENTINEL_RUN_COMPLETE` dispatcher signal. State: `"ok"` / `"disabled"`.

**Why:** No health sensor entity existed — the L2→L3 KPI gate (false-positive rate < 5%, action success rate > 95%) was invisible to operators; Lovelace dashboards and automations could not consume Sentinel state.

**Effort:** XL
**Priority:** P1
**Completed:** v3.6.0 (2026-03-15)

---

### Fix `people_home`/`people_away` to use stable entity IDs

**What:** Changed `snapshot/derived.py` to populate `people_home` and `people_away` with `state.entity_id` instead of `state.attributes.get("friendly_name") or state.entity_id`. Added `SuppressionState` v2→v3 migration to drop `presence_grace_until` entries written using display-name-derived keys. Also widened `AuditStore.async_append_finding` to accept `AnomalyFinding | CompoundFinding`.

**Why:** Presence-grace window keys in `SuppressionState.presence_grace_until` were keyed by display name (e.g. `"Alice"`) rather than entity ID (e.g. `"person.alice"`). If a user renamed a person in HA, the grace-window key became stale and the window silently stopped working.

**Effort:** M
**Priority:** P2
**Completed:** v3.5.3 (2026-03-15)

---

### Remove `_supports_suppression_reason_code()` introspection shim

**What:** Deleted the `_supports_suppression_reason_code()` helper in `sentinel/engine.py` and inlined the direct call to `async_append_finding` at all `_append_finding_audit` callsites.

**Why:** The shim's `else`-branch was dead code — `AuditStore.async_append_finding` has had the full v2 signature since Issue #3 (GitHub #254). The introspection added ~20 lines of complexity and a `cast("Any", ...)` bypass that defeated type checking.

**Effort:** S
**Priority:** P3
**Completed:** v3.5.2 (2026-03-15)

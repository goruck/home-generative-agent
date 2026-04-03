# TODOS

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

**What:** Add a `NumberSelector` to the Sentinel subentry options flow for `sentinel_baseline_min_samples` (min: 1, max: 500, step: 1, default: 20). Update `sentinel_subentry_flow.py` and `const.py`.

**Why:** Without a UI control, users can't tune how many samples are required before baseline rules fire. The feature is invisible to anyone who doesn't read the source code. A user who enables baseline collection and sees no rule firings needs a way to discover and adjust this threshold.

**How to apply:** Add `CONF_SENTINEL_BASELINE_MIN_SAMPLES` to `const.py` (if not already added by the baseline enhancement PR). In `sentinel_subentry_flow.py`, add a `NumberSelector` in the baseline section alongside the existing update interval and freshness threshold controls.

**Effort:** S
**Priority:** P2
**Depends on:** Baseline enhancement PR (sentinel-baseline-enhancement)

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

---

## Discovery

### Tighten discovery prompt to require entity-backed evidence paths

**What:** Add an instruction to `USER_PROMPT_TEMPLATE` in `explain/discovery_prompts.py` requiring that each candidate cite at least one `entities[entity_ids contains <entity_id>].state` path referencing a concrete entity from the snapshot. Candidates that concern a specific entity but can only cite `derived.*` paths (e.g. `derived.anyone_home`) should be omitted rather than submitted with abstract evidence.

**Why:** The LLM currently generates candidates like "Stale Person Tracking While Away" with `evidence_paths` containing only `derived.*` entries and no concrete entity IDs. The normalization engine (`proposal_templates.py:explain_normalize_candidate`) requires at least one extractable entity ID to map a candidate to a supported rule template. Without it, approval always fails with `unsupported_pattern`, leaving a stuck proposal in the card that can only be dismissed by rejection and eventually expires via the 30-day TTL. The prompt is the only enforcement point — the schema validates structure but not content.

**How to apply:** In `USER_PROMPT_TEMPLATE`, add after the current `evidence_paths` instruction: *"If a candidate concerns specific entities, at least one evidence_path MUST reference a concrete entity_id using the format `entities[entity_ids contains domain.object_id].state`. Omit the candidate entirely if no such entity can be cited from the snapshot."* Add a corresponding test in `tests/` that verifies a candidate with only `derived.*` paths and no concrete entity references is not generated (or is filtered pre-store).

**Effort:** S
**Priority:** P2
**Depends on:** Discovery deduplication fix PR (fix/discovery-dedup-noise)

---

### Wire `proposals_promoted` counter in discovery engine

**What:** `_discovery_cycle_stats["proposals_promoted"]` is initialized to 0 in `SentinelDiscoveryEngine._run_once()` and never incremented anywhere in the class. `SentinelHealthSensor` exposes `discovery_proposals_promoted` but it always reports 0.

**Why:** The counter was added to the health sensor attributes in v3.7.0 but the increment logic was not wired. The `proposals_promoted` value changes when `ProposalStore.async_approve()` is called — but that's in the engine or execution service, not in `_run_once()`. The counter needs to be driven via the `SIGNAL_SENTINEL_RUN_COMPLETE` payload or a separate signal from the approval flow.

**How to apply:** Option A: In `SentinelEngine`, when a discovery candidate is promoted to a rule (via proposal approval), emit an increment signal or include the count in the next `SIGNAL_SENTINEL_RUN_COMPLETE` payload. Option B: In `SentinelHealthSensor.async_update()`, query `ProposalStore` directly for the count of `status="approved"` proposals created in the last 24h and report that as `discovery_proposals_promoted`. Option B is simpler and doesn't require engine changes. Add a test asserting the attribute is non-zero when approved proposals exist.

**Effort:** S
**Priority:** P1
**Depends on:** v3.7.0 (health sensor discovery metrics)

---

## Notifier / Observability

### Feedback-trained per-entity cooldowns — wire feedback signal

**What:** `record_cooldown_feedback()` exists in `sentinel/suppression.py` but is never called from any production code path. The per-entity feedback table (schema v4, shipped in v3.7.0) is populated but the feedback signal — triggered when a user snoozes or dismisses the same entity+rule combination ≥3 times — is never sent.

**Why:** Without the feedback signal, `learned_cooldown_multipliers` remains empty forever and the cooldown multiplier feature is silently inoperative. Operators won't notice because notifications still fire at the base cooldown — the learned multipliers just never kick in.

**How to apply:** In `SentinelEngine._handle_user_action()` (or the execution service that processes mobile action button callbacks), after handling a `snooze` or `dismiss` action, call `suppression_state.record_cooldown_feedback(entity_id, rule_type)`. Add a test that asserts after 3 dismissals the multiplier for that combination is incremented.

**Effort:** S
**Priority:** P1
**Depends on:** v3.7.0 (suppression schema v4)

---

### Fix cooldown multiplier key scheme (entity_id → rule_type:entity_id) + schema migration v5

**What:** `learned_cooldown_multipliers` in `SuppressionState` (schema v4) is keyed by `entity_id` only. This means a snooze of `lock.front_door` for `unlocked_lock_at_night` increments the same counter as a snooze for `camera_entry_unsecured`. Different rules for the same entity should have independent multipliers.

**Why:** A door lock that fires `camera_entry_unsecured` daily (normal security camera activity) and `unlocked_lock_at_night` rarely should suppress the camera rule but not the lock rule. With the current entity-only key, dismissing either rule increments the same multiplier, causing both rules to cool down together. This leads to missed alerts for the more critical rule.

**How to apply:** Change the key format from `entity_id` to `{rule_type}:{entity_id}` (e.g., `"unlocked_lock_at_night:lock.front_door"`). Add a schema v4→v5 migration in `core/migrations.py` that re-keys existing entries (entries with no `:` separator get prefixed with an empty rule type and can be discarded since they were never incremented in production). Update all callsites in `suppression.py` and add a migration test.

**Effort:** S
**Priority:** P1
**Depends on:** Wire feedback signal TODO above

---

### Daily digest config flow UI

**What:** `CONF_SENTINEL_DAILY_DIGEST_ENABLED` and `CONF_SENTINEL_DAILY_DIGEST_HOUR` are declared in `const.py` and read in `__init__.py` but are not exposed in `sentinel_subentry_flow.py`. Users cannot enable the daily digest without editing raw options JSON.

**Why:** The daily digest notification shipped in v3.7.0 with backend support but without a UI control. Any user who discovers the feature via README will find no toggle in the integration options flow.

**How to apply:** In `sentinel_subentry_flow.py`, add a `BooleanSelector` for `CONF_SENTINEL_DAILY_DIGEST_ENABLED` (default: False) and a `TimeSelector` for `CONF_SENTINEL_DAILY_DIGEST_HOUR` (default: `"07:00:00"`). Gate the time selector on the enabled boolean. Add config flow tests asserting both options are accepted.

**Effort:** S
**Priority:** P1
**Depends on:** v3.7.0 (daily digest backend)

---

### iOS notification priority tiers

**What:** All Sentinel mobile push notifications currently use the same `active` interruption level regardless of severity. High-severity findings (security alerts) should use `time-sensitive` (bypasses Focus modes), medium should use `active`, low should use `passive`. Title should reflect severity: "Security Alert" / "Home Alert" / "Home Update".

**Why:** 7 notifications in 18 minutes, all with identical urgency appearance, was the production incident that prompted this sprint. Without priority differentiation, users train themselves to ignore Sentinel notifications — including the ones that matter.

**How to apply:** In `sentinel/notifier.py`, add `_SEVERITY_INTERRUPT_LEVEL` and `_SEVERITY_TITLE` dicts. In `async_notify()`, derive severity from `getattr(finding, "severity", "medium")`, look up the interrupt level and title, and add `"push": {"interruption-level": level}` to the notification `data["data"]` block. Add `"subtitle": _friendly_type(finding.type)`. See plan `steady-petting-haven.md` for full diff. Add tests for all three severity paths.

**Effort:** S
**Priority:** P1
**Depends on:** None

---

### Appliance completion detection in baseline deviation

**What:** When `evaluate_baseline_deviation()` fires on a power entity and `current_value < 10% × baseline_value`, the appliance finished its cycle — this is normal behavior, not a failure. The finding should be severity `"low"` with `evidence["is_completion"] = True` instead of the default high-severity "stopped unexpectedly" framing.

**Why:** Washer/dryer finishing a cycle fires `baseline_deviation` because power drops from ~40 kW → 0.5 kW — a large deviation. The LLM explanation then phrases it as a failure ("appliance stopped unexpectedly") causing false-positive alerts. See `COMPLETION_THRESHOLD_PCT = 0.10`.

**How to apply:** In `sentinel/baseline.py`, after the evidence dict is built, detect completion: check `device_class == "power"` or `unit in {"W", "kW"}`, then if `current_value < 0.10 * baseline_value`, set `evidence["is_completion"] = True` and override `severity = "low"`. Update `explain/prompts.py` to add grounding: "If evidence contains is_completion=true, the appliance finished its cycle normally — say 'finished' not 'stopped unexpectedly'." See plan `steady-petting-haven.md` for full diff. Add tests for power vs non-power entities.

**Effort:** S
**Priority:** P1
**Depends on:** None

---

### Presence-aware lock severity for `unlocked_lock_at_night`

**What:** `unlocked_lock_at_night` fires `severity="high"` and `is_sensitive=True` even when `snapshot["derived"]["anyone_home"]` is True. A lock left unlocked while you're home is low urgency — it should be `severity="low"`, `is_sensitive=False`.

**Why:** High-severity alert with iOS `time-sensitive` interruption level for an unlocked door while you're sitting in the living room is noise. Operators will start ignoring it. The presence signal already exists in the snapshot; it's not being used.

**How to apply:** In `sentinel/rules/unlocked_lock_at_night.py`, read `anyone_home = bool(snapshot["derived"].get("anyone_home", False))` before the loop. Add `"anyone_home": anyone_home` to the evidence dict. Set `severity="low" if anyone_home else "high"` and `is_sensitive=not anyone_home` in the `AnomalyFinding` constructor. Update `explain/prompts.py` grounding: "If evidence contains anyone_home=true for an unlocked lock, the household is occupied — frame it as a reminder, not an alert." See plan `steady-petting-haven.md` for full diff. Add tests for both presence states.

**Effort:** S
**Priority:** P1
**Depends on:** None

---

### Notification batching for medium/low severity bursts

**What:** When more than 3 medium/low mobile push notifications are sent within a rolling 60-second window, buffer subsequent notifications and flush as a single summary push ("N additional home alerts: type1, type2, …") after 30 seconds. High-severity findings always bypass batching.

**Why:** Burst of 7 notifications in 18 minutes (the production incident) clutter the lock screen and train users to dismiss without reading. High-severity findings must still arrive immediately; medium/low bursts should batch.

**How to apply:** In `SentinelNotifier.__init__()`, add `_notification_times: list[datetime]`, `_held_batch: list[...]`, `_batch_cancel`. In `async_notify()`, check rate window before dispatching; if over limit and severity != "high", hold in batch and arm `async_call_later` timer. Add `_async_flush_batch()` callback that sends a passive-priority summary with no action buttons. Cancel timer in `stop()`. See plan `steady-petting-haven.md` for full implementation. Add 6 test cases covering batching, bypass, and flush.

**Effort:** M
**Priority:** P1
**Depends on:** iOS notification priority tiers TODO above (shares `async_notify()` changes)

---

## Completed

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

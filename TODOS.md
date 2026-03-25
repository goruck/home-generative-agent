# TODOS

## Explain / Prompts

### Centralize action code vocabulary in `const.py`

**What:** Define `ACTION_CODES: dict[str, str]` in `const.py` mapping action code → plain English description (e.g., `"arm_alarm"` → `"re-arm the security alarm"`). Import it in `explain/prompts.py` to build the `SYSTEM_PROMPT` vocabulary line dynamically, and use the constants in all rule files instead of bare strings.

**Why:** `SYSTEM_PROMPT` now contains a hardcoded list of 7 action codes. Any new rule that introduces a new action code (e.g., `notify_neighbor`, `call_emergency`) will silently cause the LLM to invent its own English meaning for it — recreating the exact class of production bug fixed in PR #346. This drift surface was flagged independently by both the eng review and the Codex outside voice.

**How to apply:** In `const.py`, add `ACTION_CODES = {"arm_alarm": "re-arm the security alarm", "disarm_alarm": "disarm the alarm", ...}`. In `prompts.py`, replace the hardcoded vocabulary string with `"; ".join(f"{k}={v}" for k, v in ACTION_CODES.items())`. In rule files, import `ACTION_CODES` keys as named constants rather than bare strings. Add a test that asserts every action code used in any rule file exists in `ACTION_CODES`.

**Effort:** S
**Priority:** P2
**Depends on:** None

---

### Sanitize area/entity strings before injecting into LLM prompts

**What:** Strip or truncate user-editable strings (area names, entity IDs, friendly names) to a safe character set before including them in `USER_PROMPT_TEMPLATE` evidence interpolation.

**Why:** HA area names are user-editable via UI or YAML and flow into `USER_PROMPT_TEMPLATE.format(evidence=...)` as raw Python dict repr without sanitization. A malicious area name (e.g., `"Front\nIgnore all previous instructions."`) constitutes a prompt injection vector. The `camera_area` and `unsecured_entity_areas` fields added in PR fixing camera_entry_unsecured notifications increase the surface area (same area string, multiple occurrences). Flagged as P3 since it requires a coordinated attacker with HA admin access to be exploitable.

**How to apply:** Add a sanitization helper in `explain/` that replaces non-printable characters and control sequences in string values before dict repr serialization. Alternatively, serialize evidence as JSON with explicit schema validation rather than using Python repr. Add a test with a crafted area name containing injection characters.

**Effort:** S
**Priority:** P3
**Depends on:** None

---

## Sentinel Rules

### Fix transitive union-find spatial contamination in correlator

**What:** Post-grouping validation in `SentinelCorrelator._build_groups()` to prevent `camera_entry_unsecured` findings from ending up in a compound finding with findings from a different area via transitive bridging.

**Why:** The area-aware `_COMPLEMENTARY_PAIRS_REQUIRE_AREA_MATCH` guard only protects direct pairwise checks. With three simultaneous findings (e.g., `camera_entry_unsecured` (Front) + `unlocked_lock_at_night` (Front) + `open_entry_while_away` (Garage)), the camera in Front ends up in a compound with `open_entry_while_away` in Garage via the bridging lock in Front — the exact false spatial claim the camera_entry_unsecured fix was designed to prevent. Flagged by adversarial review during PR #347 ship workflow.

**How to apply:** After `_build_groups()` returns groups, add a post-processing pass: for any group containing a `camera_entry_unsecured` finding, assert that all other findings in the group share the camera's area. Eject any finding that does not match; return it as a singleton. Alternatively, add `frozenset({"unlocked_lock_at_night", "open_entry_while_away"})` and other at-risk transitivity chains to `_COMPLEMENTARY_PAIRS_REQUIRE_AREA_MATCH` — but verify this doesn't over-restrict legitimate cross-area correlation for those pair types.

**Effort:** S
**Priority:** P2
**Depends on:** None

---

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
**Priority:** P3
**Depends on:** Baseline enhancement PR

---

## Discovery

### Tighten discovery prompt to require entity-backed evidence paths

**What:** Add an instruction to `USER_PROMPT_TEMPLATE` in `explain/discovery_prompts.py` requiring that each candidate cite at least one `entities[entity_ids contains <entity_id>].state` path referencing a concrete entity from the snapshot. Candidates that concern a specific entity but can only cite `derived.*` paths (e.g. `derived.anyone_home`) should be omitted rather than submitted with abstract evidence.

**Why:** The LLM currently generates candidates like "Stale Person Tracking While Away" with `evidence_paths` containing only `derived.*` entries and no concrete entity IDs. The normalization engine (`proposal_templates.py:explain_normalize_candidate`) requires at least one extractable entity ID to map a candidate to a supported rule template. Without it, approval always fails with `unsupported_pattern`, leaving a stuck proposal in the card that can only be dismissed by rejection and eventually expires via the 30-day TTL. The prompt is the only enforcement point — the schema validates structure but not content.

**How to apply:** In `USER_PROMPT_TEMPLATE`, add after the current `evidence_paths` instruction: *"If a candidate concerns specific entities, at least one evidence_path MUST reference a concrete entity_id using the format `entities[entity_ids contains domain.object_id].state`. Omit the candidate entirely if no such entity can be cited from the snapshot."* Add a corresponding test in `tests/` that verifies a candidate with only `derived.*` paths and no concrete entity references is not generated (or is filtered pre-store).

**Effort:** S
**Priority:** P3
**Depends on:** Discovery deduplication fix PR (fix/discovery-dedup-noise)

---

### Discovery quality metrics on health sensor

**What:** Expose per-cycle discovery counters on `sensor.sentinel_health`: `candidates_generated`, `candidates_novel` (passed dedup), `candidates_deduplicated` (rejected by semantic key or title hash), `proposals_promoted`, `unsupported_ttl_expired` (cleaned up this cycle).

**Why:** After the deduplication fix, operators have no visibility into whether dedup is actually working — they can't distinguish "LLM is generating zero new ideas" from "LLM is generating many ideas but all are caught by dedup." Without counters, a misconfigured semantic key map or broken hash exclusion would be invisible until a repeat proposal slips through to the card.

**How to apply:** Add a `_discovery_cycle_stats: dict` field to `DiscoveryEngine` (reset at start of `_run_once()`). Increment counters at each decision point in `_filter_novel_candidates()` and `cleanup_unsupported_ttl()`. Include the stats dict in the `SIGNAL_SENTINEL_RUN_COMPLETE` payload so `SentinelHealthSensor` can expose them as attributes alongside existing KPIs.

**Effort:** S
**Priority:** P3
**Depends on:** Discovery deduplication fix PR

---

## Completed

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

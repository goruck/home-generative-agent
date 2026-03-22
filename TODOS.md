# TODOS

## Sentinel Rules

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

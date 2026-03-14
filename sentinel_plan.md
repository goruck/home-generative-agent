# Sentinel — Proactive Deterministic Anomaly Detection

## 1. Non-Negotiable Principles

- Detection remains fully deterministic and rule-based.
- LLMs never create, modify, or delete findings.
- LLMs may only advise on notification handling, explanation text, and user-triggered or policy-allowed execution.
- Every suppression, triage decision, and action outcome is auditable with an explicit reason code.
- Human override and kill switch are always available as runtime operations (not config-file edits).
- Detection rule configs are version-controlled and reviewed before deployment. Rule schema versions are stored in every audit record.

## Current Status Snapshot (as of 2026-03-14)

- This document mixes current implementation, accepted design targets, and remaining work. Unless a section explicitly says `Implemented`, treat it as target-state design rather than current behavior.
- Implemented in code today:
  - Runtime autonomy override service (`home_generative_agent.sentinel_set_autonomy_level`) with admin-only access and TTL-bounded in-memory overrides.
  - PIN hash storage and PIN value validation for autonomy-level increases, including Sentinel subentry flow support for storing only hashed PIN material.
  - Event-driven triggering with a bounded queue, coalescing, TTL discard, and single-flight lock.
  - Correlation within a single `_run_once()` call only.
  - Suppression reason codes, quiet hours, pending-prompt TTL, snooze (`24h`, permanent), and presence-grace windows.
  - LLM triage with a strict prompt allowlist and fail-open behavior.
  - Execution policy service with stale/unavailable handling at the execution gate (autonomy level 2+), plus allowlist, confidence threshold, rate limit, idempotency, canary mode, and live auto-execute.
  - Baseline updater and temporal/baseline detector support — fully wired end-to-end (2026-03-12): `async_fetch_baselines()` added to `SentinelBaselineUpdater`; engine now calls it each cycle and passes the result to `evaluate_dynamic_rules()`; baseline config fields (`sentinel_baseline_enabled`, `sentinel_baseline_update_interval_minutes`, `sentinel_baseline_freshness_threshold_seconds`) exposed in the Sentinel subentry flow. Prior to this fix the `baseline_deviation` and `time_of_day_anomaly` evaluators always received an empty baselines dict and never fired.
  - `sentinel_enabled` is now a true master switch (2026-03-14, PR #330): discovery (`SentinelDiscoveryEngine`) and baseline collection (`SentinelBaselineUpdater`) no longer start when `sentinel_enabled` is `false`, regardless of their individual flags. UI labels updated to "Enable anomaly alerting", "Enable discovery (requires anomaly alerting)", and "Enable baseline collection (requires anomaly alerting)" to make the dependency explicit.
  - Discovery pipeline improvements from Milestone 5: service-mapped suggested actions, on-demand discovery trigger service, immediate rule activation on proposal approval, structured normalization failure reasons, overlap metadata, richer draft notifications, and rule preview before commit.
  - Portable dynamic rule templates (PR #321, merged 2026-03-11) plus follow-on normalization fixes (2026-03-13): six new templates (`unlocked_lock_while_away`, `alarm_state_mismatch`, `entity_state_duration`, `sensor_threshold_condition`, `entity_staleness`, `multiple_entries_open_count`) cover the most common discovery-generated candidate patterns across arbitrary HA configurations. The normalization/evidence-path parser now handles `entities[entity_id=...]`, `entities[entity_ids contains ...]`, and LLM-emitted dot-notation paths like `sensor.foo.state` / `lock.foo.battery_level`; lock battery-low candidates route to `low_battery_sensors`; text-signal fallbacks now cover `high_energy_consumption_night`, `alarm_disarmed_during_external_threat`, and selector-based window-duration candidates; built-in coverage detection now accepts `frontgate` / `backgarage` substrings even when the LLM omits the domain prefix.
  - The four remaining Rule issues called out in the 2026-03-12 snapshot are now closed in code via new static rules (2026-03-12/13): `alarm_disarmed_during_external_threat` (#309), `camera_backgarage_missing_snapshot_night_home` (#311), `vehicle_parked_near_frontgate_home` (#312), and `unknown_person_frontporch_night_home` (#318).
  - No Rule issues remain open from the #298–#320 coverage sweep. Remaining gaps are infrastructure/behavioral rather than missing detector patterns: suppressed-finding audit coverage, blocked-vs-notified policy behavior, audit retention/archival wiring, `trigger_source` population, and stable person identifiers in derived presence.
- Partially implemented or still open (see Known Current Gaps below for the summary list):
  - Rule/suppression-state-suppressed findings are not written to the audit store (engine silently returns at the suppression gate). Triage-suppressed findings *are* written to audit.
  - `ACTION_POLICY_BLOCKED` blocks execution but does not suppress notification dispatch.
  - Audit schema/retention/archival health in this document is ahead of the current `audit/store.py` implementation. `CONF_AUDIT_HOT_MAX_RECORDS` is defined in `const.py` but the store still uses a hardcoded `MAX_RECORDS = 200` constant — the config value is not wired up.
  - `trigger_source` field exists in `AuditRecord` and migration defaults but is never populated by any engine call — it is always `None` in practice.
  - `action_policy_path` is present in `AuditRecord` and `async_append_finding` but is absent from `_V2_FIELD_DEFAULTS` in `audit/store.py` — v1→v2 migration does not backfill this field.
  - `derived.people_home` / `derived.people_away` currently contain friendly names when available, not stable person entity IDs.

### Known Current Gaps

- Suppressed findings from the rule/suppression gate are not audited.
- `ACTION_POLICY_BLOCKED` does not currently suppress notification dispatch.
- `trigger_source` is never populated in audit records.
- `CONF_AUDIT_HOT_MAX_RECORDS` is defined but not wired into `audit/store.py`.
- `action_policy_path` is not backfilled during v1→v2 audit migration.
- Presence-grace identity currently uses display-name-derived values, not stable person entity IDs.

---

## 2. Target Detection/Response Pipeline

Status: `Partially implemented`. Steps 1, 2, 4, 5, 6, 7, 8, and 9 exist in some form, but rule/suppression-state-suppressed findings are not audited (silent return before audit append), triage-suppressed findings are audited, and notification dispatch is not yet suppressed for all `blocked` policy outcomes.

1. Trigger received (polling or event-driven).
2. Snapshot built deterministically.
3. Staleness/data-quality validation pass:
   - Triggering entities checked against `CONF_SENTINEL_STALENESS_THRESHOLD_SECONDS`.
   - Target state: `data_quality="stale"` findings are notify-only regardless of autonomy level.
   - Target state: `data_quality="unavailable"` findings are suppressed by default, except `severity="high"` which is notify-only.
   - Current code: stale/unavailable handling happens inside the execution service only after the autonomy-level gate is reached (`autonomy_level >= 2`). At lower autonomy levels, findings route to `prompt_user` before data quality is evaluated there.
   - Current code: at autonomy level 2+, low/medium unavailable findings return `action_policy_path="blocked"` from the execution service, but the engine still notifies them. This remains an implementation gap.
4. Rules evaluated deterministically.
5. Correlation pass merges related findings deterministically (within the current `_run_once()` call only; compound findings are immutable once emitted).
6. Suppression pass evaluates cooldowns, quiet-hours, snooze policies, and presence-grace windows.
   - Target state: every suppressed finding receives an explicit `suppression_reason_code` in audit.
   - Current code: suppression reason codes exist, but suppressed findings return before audit append.
7. Triage pass (optional, autonomy level >= 1) decides `notify` vs `suppress` only. Triage output is informational; it does not influence execution eligibility.
8. Action policy gate decides `prompt_user`, `handoff`, `auto_execute`, or `blocked`.
9. For autonomy level >= 2, optional canary mode evaluates `would_auto_execute` and records decisions without executing.
10. Notification dispatch (or audit-only for policy cases where notification is intentionally skipped).
11. Full audit append with reasons, timings, data quality, and outcomes at every stage.

---

## 3. Autonomy Model (Explicit Boundaries)

### Config

```yaml
CONF_SENTINEL_AUTONOMY_LEVEL: 0|1|2|3            # startup default
CONF_SENTINEL_AUTO_EXECUTION_ENABLED: true/false
CONF_SENTINEL_AUTO_EXECUTE_DEFAULT_MIN_CONFIDENCE: 0.70   # compared against finding.confidence
CONF_SENTINEL_AUTO_EXECUTE_MAX_ACTIONS_PER_HOUR: 5
CONF_SENTINEL_AUTO_EXECUTE_ALLOWED_SERVICES: [...]
CONF_SENTINEL_AUTO_EXEC_CANARY_MODE: true/false
CONF_SENTINEL_STALENESS_THRESHOLD_SECONDS: 1800
CONF_SENTINEL_RUNTIME_OVERRIDE_TTL_MINUTES: 120
CONF_SENTINEL_REQUIRE_PIN_FOR_LEVEL_INCREASE: true   # applies to increases into level 2 or 3 only; never required for lowering
CONF_SENTINEL_EXECUTION_IDEMPOTENCY_WINDOW_MINUTES: 15  # deduplication window; align with entity cooldown period when possible
CONF_AUDIT_HOT_MAX_RECORDS: 500        # defined in const.py; audit/store.py still uses hardcoded MAX_RECORDS = 200
CONF_AUDIT_ARCHIVAL_BACKLOG_MAX: 100
CONF_AUDIT_RETENTION_DAYS: 90
CONF_AUDIT_HIGH_RETENTION_DAYS: 365
```

### Rule-level policy fields

```yaml
auto_execute_allowed: true/false
auto_execute_min_confidence: 0.0-1.0   # overrides global default; checked against finding.confidence
rule_version: "1.0"                    # required on all rules
```

Status: `Planned, not yet wired end-to-end`. Current `AnomalyFinding` does not carry these fields, and current audit records do not reliably persist `rule_version`.

### Levels

- Level 0 Notify-only: no autonomous execution.
- Level 1 LLM triage (notify-only): triage may suppress or enrich notifications; no actions executed.
- Level 2 Guarded auto-execute: non-sensitive, allowlisted services, `finding.confidence` gate, rate limits, idempotency, and stale-data block.
- Level 3 Broad autonomy: expanded execution scope, but PIN/critical actions still blocked from autonomous path unless explicitly policy-approved. **Level 3 is intentionally not specified in this plan beyond the KPI gate in Section 17. It requires a dedicated design issue after L2 has completed its 30-day burn-in and all L2→L3 KPI thresholds are met.**

### Runtime kill switch and override lifecycle

- Service: `home_generative_agent.sentinel_set_autonomy_level`.
- `level: 0` is the kill-switch call and must take effect immediately without restart.
- Service is admin-only. Does not require an explicit `entry_id` in the service call; the engine derives the active entry internally.
- Runtime override is in-memory by default. If `CONF_SENTINEL_RUNTIME_OVERRIDE_TTL_MINUTES` is set, override auto-expires and reverts to config default.
- Target state: override expiry should emit an audit event.
- `CONF_SENTINEL_REQUIRE_PIN_FOR_LEVEL_INCREASE: true` enforces Sentinel PIN verification for level increases. The PIN is stored as salted hash material in the Sentinel subentry. Level decreases and the Level 0 kill switch are never PIN-gated.

---

## 4. Execution Guardrails

Status: `Mostly implemented`, with open gaps around per-rule thresholds and exact notification behavior for blocked outcomes.

- Sensitive or PIN-gated actions are never auto-executed by default.
- Strict domain/service allowlist required for auto-execution.
- Target state: per-rule and global `finding.confidence` thresholds both must pass.
- Current code: only the global threshold is enforced. Triage confidence does not influence execution eligibility.
- `auto_execute` is unconditionally disabled for `data_quality != "fresh"`.
- Rate limiter: max actions per time window (per-hour counter, burst-protected).
- Idempotency key required for autonomous actions:
  - `execution_id = sha256(anomaly_id + action_policy_path + policy_version + window_bucket)`
  - `policy_version = sha256(json.dumps({"allowed_services": sorted(CONF_SENTINEL_AUTO_EXECUTE_ALLOWED_SERVICES), "autonomy_level": autonomy_level_at_decision, "min_confidence": CONF_SENTINEL_AUTO_EXECUTE_DEFAULT_MIN_CONFIDENCE, "max_actions_per_hour": CONF_SENTINEL_AUTO_EXECUTE_MAX_ACTIONS_PER_HOUR, "require_pin_for_increase": CONF_SENTINEL_REQUIRE_PIN_FOR_LEVEL_INCREASE, "effective_rule_threshold": effective_rule_threshold}, sort_keys=True))` — captures global execution policy and effective rule-level threshold. Structured serialization prevents hash collisions from string concatenation. Changes automatically when any execution-relevant policy changes.
  - `window_bucket = floor(utc_epoch_seconds / (CONF_SENTINEL_EXECUTION_IDEMPOTENCY_WINDOW_MINUTES * 60))` — default 15 min window; align with the entity's cooldown period when possible.
  - Duplicate `execution_id` attempts are blocked and audited.
- Runtime kill switch disables all autonomous execution immediately.
- If policy evaluation fails for any reason, default to notify (safe fallback).
- Rejected auto-execute attempts (allowlist/confidence/rate/idempotency blocks) route to `prompt_user`, not silent drop.
- Target state: `blocked` outcomes should also skip notification when policy intends suppression. Current engine behavior does not fully enforce that distinction yet.

---

## 5. Event-Driven Triggering Design

Status: `Implemented in simplified form`.

- Add entity-filtered listeners for locks, entries, windows, motion, and cameras.
- Keep periodic polling for duration/baseline rules.
- Current code coalesces by anomaly type only, not `entity_id + trigger_type`.
- Target refinement: coalesce by `entity_id + trigger_type` if per-entity event fidelity becomes necessary.
- Single-flight run lock (`asyncio.Lock`) prevents overlapping `_run_once()` calls.
- Bounded trigger queue: max depth `TRIGGER_QUEUE_MAX` (current code uses constant `QUEUE_MAX_SIZE = 10`).
- Queue overload policy (deterministic):
  - Prefer retaining security-critical triggers (lock/entry/camera).
  - Current code drops the lowest-priority queued item, oldest first within a priority tier, and drops the incoming item when the queue contains only security-critical entries.
  - Target state: emit structured `drop_reason_code` values.
- Trigger TTL: queued entries older than `ttl_seconds` (default 30 s) at dequeue time are discarded.
- Target state: trigger drops/discards should be counted and exposed on a health entity; current code logs them only.

---

## 6. LLM Triage Design (Level 1+)

Status: `Implemented, minus some audit metadata`.

- Input allowlist:
  - Always allowed: `type`, `severity`, `confidence`, `is_sensitive`, `entity_count`, `suggested_actions_count`.
  - Optional sanitized evidence: a named allowlist of safe derived fields only:
    - `is_night: bool`
    - `anyone_home: bool`
    - `recognized_people_count: int` — count only, never names or identifiers
    - `last_changed_age_seconds: float` — derived elapsed time, not raw timestamp
    - Any field not in this list is stripped before prompt construction.
  - Never allowed: raw entity state values, attribute strings, area names, free-form text from `evidence`.
- Output schema: `decision=notify|suppress`, `reason_code`, `triage_confidence` (audit-only), `summary`.
- Triage cannot alter any finding fields.
- Timeout/error path: fail-open to notify, audit `triage_error`.
- Target state: store triage latency, model/provider metadata, and `triage_confidence` in audit record.
- Current code stores `triage_confidence`, `triage_decision`, and `triage_reason_code`, but not latency or provider/model metadata.

---

## 7. Correlation Engine

- Run after staleness validation and rule evaluation, before suppression.
- Scope: operates only on `all_findings` from the current `_run_once()` call. Compound findings are immutable once emitted at end of call.
- Deterministic grouping by correlation window + entity overlap + known pattern map.
- Compound evidence shape:

```python
{
    "compound": True,
    "child_finding_ids": ["sha256...", "sha256..."],
    "correlated_at": "<iso-utc>",
    "correlation_pattern": "camera_and_entry",
}
```

- Findings arriving in later `_run_once()` calls are not retroactively merged into prior compounds.
- Suppression and notification operate on compound results to reduce alert spam.

---

## 8. Suppression Enhancements

Status: `Mostly implemented`.

- Existing type-based and entity-based cooldowns retained.
- Add quiet-hours policy by severity.
- Per-person presence-grace windows:
  - Target state: on person departure/arrival, write `person_entity_id -> ISO expiry` in `SuppressionState.presence_grace_until` (default 10 min).
  - Current code: writes a person key derived from `derived.people_home`; because `derived.people_home` currently uses friendly names when available, this key is not yet a stable entity ID.
  - Presence-sensitive findings suppressed during active grace windows with reason `presence_grace`.
- Snooze routes through suppression:
  - Current code supports `24h` and permanent snooze. `7d` remains defined in suppression code but is not surfaced in the notifier.
  - Target state: options `24h`, `always` write to `SuppressionState.snoozed_until` keyed by finding type. `7d` is not exposed and should not be added to notification UX without removing another button.
  - `always` requires explicit confirmation before write.
  - Reason codes: `user_snooze_24h`, `user_snooze_permanent`.
- Pending-prompt TTL and auto-resolution states.
- Target state: every suppressed finding gets explicit `suppression_reason_code` in audit.
- Current code computes reason codes. Triage-suppressed findings are appended to audit with their reason code. Rule/suppression-state-suppressed findings return before the audit append — they are not yet written to the audit store.

---

## 9. Notification UX Improvements

Status: `Partially implemented`.

- Severity routing policy:
  - Target state:
    - `high`: push + persistent
    - `medium`: persistent + optional push
    - `low`: audit-only or digest
- Current code supports push via configured notify service or persistent notifications, plus per-area routing; it does not yet implement digest mode or severity-tiered audit-only behavior.
- Attach camera snapshots where available and permissioned. `Planned`.
- Digest mode for burst findings. `Planned`.
- Structured snooze options (`24h`, `always`) shown in notification and written through suppression path (Section 8). `7d` is not offered — the action button slot it would occupy is reserved for the False Alarm button due to iOS action button count limits (see Section 8 for the design constraint).
  - Current code exposes `24h` and permanent snooze with confirmation.
- False Alarm action button included on every notification; tapping it sets `user_response.false_positive = True` in the audit record. *(Implemented)*
- Sensitive finding notifications may include occupant context from `derived.people_away` when policy allows. `Planned`.

---

## 10. Rule System Expansion

- Add deterministic rule types:
  - Time-window conditions
  - Frequency rules (N in T)
  - Baseline deviation rules (moving averages, bounded windows) - requires baseline storage from implementation step 1
  - Composite AND/OR templates
- User-defined custom conditions are addressed through discovery pipeline improvements (see Milestone 5). Lambda/expression rules were considered and removed — they were detection-only (no service-type suggested actions), invisible from `get_dynamic_rules`, and duplicated the pipeline without enabling full autonomy.

---

## 11. Operational Health and Observability

Status: `Largely planned`. No health sensor entity is registered anywhere in the codebase today — the entire section is target state. None of the attributes below are currently exposed to Home Assistant.

- Sentinel health entity attributes:
  - `last_run_start`, `last_run_end`, `run_duration_ms`
  - `trigger_source_stats`: `{poll: N, event: N}`
  - `triggers_dropped`: cumulative count
  - `trigger_drop_reason_counts`: per-reason counters
  - `active_rule_count`
  - `findings_count_by_severity`
  - `triage_suppress_rate`
  - `auto_exec_count`, `auto_exec_failures`, `auto_exec_would_execute_count` (canary)
  - Rolling KPI values: `false_positive_rate_14d`, `action_success_rate`, `user_override_rate`
- Static availability checks for critical sensors/entities.
- Export counters suitable for Lovelace dashboards.

---

## 12. Audit Schema

Status: `Partially implemented`. The current audit store has a subset of the fields below and uses a fixed-cap local store; treat the dataclass below as target state, not current code. Known gaps beyond the schema itself: `trigger_source` is never populated by any engine call (always `None`); `action_policy_path` is missing from `_V2_FIELD_DEFAULTS` so v1→v2 migration does not backfill it; `CONF_AUDIT_HOT_MAX_RECORDS` is defined but the store uses a hardcoded `MAX_RECORDS = 200`.

For each finding record, persist:

```python
@dataclass(frozen=True)
class AuditRecord:
    snapshot_ref: dict
    finding: dict
    data_quality: str                    # target state; current code stores {"quality": "..."} or None
    trigger_source: str                  # "poll" | "event"
    correlation_metadata: dict | None    # planned
    suppression_decision: str | None     # planned
    suppression_reason_code: str | None
    triage_decision: str | None          # "notify" | "suppress" | "error" | None
    triage_reason_code: str | None
    triage_confidence: float | None      # audit-only
    triage_model: str | None             # planned
    triage_latency_ms: int | None        # planned
    action_policy_path: str | None       # "prompt_user" | "handoff" | "auto_execute" | "blocked"
    canary_would_execute: bool | None
    execution_id: str | None
    notification: dict
    user_response: dict | None
    action_outcome: dict | None
    rule_version: str                    # planned end-to-end; currently usually None
    autonomy_level_at_decision: int      # current store serializes this as string
    # timestamps in UTC ISO 8601
```

PII minimization:

- Target state: if `finding.is_sensitive` is `True`, replace `evidence.recognized_people` with `recognized_people_count: int` before storage.
- Current code redacts recognized-person names in notification text, but audit-side evidence minimization is not fully implemented here.

Retention model:

- Target state:
  - Hot local store (`Store`): capped by `CONF_AUDIT_HOT_MAX_RECORDS`.
  - Time pruning: default `CONF_AUDIT_RETENTION_DAYS=90`.
  - High-severity SLA (`CONF_AUDIT_HIGH_RETENTION_DAYS=365`) requires PostgreSQL archival backend.
- Current code: local store is capped by a fixed `MAX_RECORDS = 200` and has no severity-aware retention or archival backend management.
- Archival unavailability policy:
  - Hot store continues accepting records up to `CONF_AUDIT_HOT_MAX_RECORDS`; once at cap, overwrite deterministically by dropping oldest `low`, then oldest `medium`, and only then oldest `high` if no other choice remains.
  - Invariant: never drop `high`-severity records while any `low` or `medium` records remain in hot store.
  - High-severity records pending archival queue in memory up to `CONF_AUDIT_ARCHIVAL_BACKLOG_MAX` (default 100).
  - Records beyond the backlog limit are dropped; each drop increments `audit_archival_drop_count` on the health entity.
  - Health entity exposes `audit_archival_status: "ok" | "degraded" | "backlog_full"`.
  - Acknowledged bounded loss is preferable to silent gaps in the audit trail.

---

## 13. Schema Versioning and Migrations

Status: `Partially implemented`.

- Add explicit versions:
  - `SuppressionState.version`
  - `AuditRecord.version`
- Provide deterministic migrations for each version step (`v1 -> v2`, `v2 -> v3`, etc.).
- Migration requirements:
  - Backfill missing fields with safe defaults.
  - Preserve old records if migration partially fails; mark degraded mode.
  - Emit migration summary metrics (`migrated`, `skipped`, `failed`).
- Known migration gap: `action_policy_path` is present in `AuditRecord` and accepted by `async_append_finding`, but absent from `_V2_FIELD_DEFAULTS` in `audit/store.py`. Any v1 record migrated to v2 will not have this field backfilled. Must be patched before v1 records exist in production with this field expected.
- Rollback policy:
  - Never destructive rewrite in place without backup snapshot.
  - Suppression state already supports read-only compatibility mode on downgrade and forces Level 0 behavior.
  - Target state: the audit store should offer equivalent read-only compatibility handling plus a persistent health warning. Current audit store does not yet do this.

---

## 14. Deterministic Time and Ordering Rules

Status: `Partially implemented`.

- All timestamps are UTC ISO 8601 from a single clock helper.
- Window calculations (staleness, TTL, correlation, rate limit) use monotonic elapsed checks where possible; UTC timestamps are persisted for audit.
- Tie-breakers for equal timestamps are deterministic: `anomaly_id` lexical order, then `entity_id` lexical order. `Planned; not explicitly enforced everywhere today.`
- Queue ordering is FIFO after coalescing and priority/drop policy.

---

## 15. Implementation-Architecture Prerequisites

Status note: this section is now mostly historical. Several items were completed directly in existing modules rather than the exact file layout originally proposed.

The following architecture refactors are required before the feature sequence can be delivered safely:

1. Runtime autonomy state owner:
   - Completed in `sentinel/engine.py` using a module-level runtime override store.
   - Follow-up opportunity: extract this to a dedicated controller if lifecycle/test isolation becomes painful.
2. Autonomy service authorization layer (depends on 1):
   - Gate `home_generative_agent.sentinel_set_autonomy_level` with admin-only checks.
   - Entry is resolved internally; callers do not pass `entry_id` in the service call.
   - Enforce PIN verification for increases into levels 2/3 when `CONF_SENTINEL_REQUIRE_PIN_FOR_LEVEL_INCREASE=true`.
3. Structured suppression decision model:
   - Replace boolean suppression outcome with structured result (`suppressed`, `reason_code`, `expires_at`, metadata).
   - Required for quiet-hours, snooze routing, presence-grace windows, and deterministic audit reasoning.
4. Decoupled execution service (depends on 1, 3):
   - Extract execution/handoff policy from notification action callbacks into a shared service callable.
   - Current code achieved this partially via `sentinel/execution.py`, but user-driven mobile actions and autonomous execution still do not share a single identical evaluation entrypoint.
5. Versioned audit subsystem expansion:
   - Introduce the expanded audit schema fields and versioned writer before autonomy/canary rollout.
   - Add degraded archival status/backlog counters as first-class health metrics.
6. Sentinel config surface expansion:
   - Add new constants, flow controls, and translations for autonomy/staleness/idempotency/retention settings before policy logic rollout.
7. Trigger scheduler extraction:
   - Implement queue/coalescing/TTL/drop-policy/single-flight in a dedicated scheduler component, separate from rule evaluation logic.
8. Snapshot schema extension for per-person presence:
   - Extend `DerivedContext` to include `people_home: list[str]` and `people_away: list[str]`.
   - Update `snapshot/schema.py`, `snapshot/derived.py`, and `snapshot/builder.py`.
   - `anyone_home` remains (`len(people_home) > 0`) to preserve backward compatibility with existing rules.
   - Current code uses friendly names when available. If stable identity matters for suppression and audit, switch these lists to person entity IDs and add separate display-name fields.
   - Required by: per-person presence-grace windows (3), notification occupant context (Section 9), and suppression upgrades (implementation step 6).

### Prerequisite Refactor Task Map (Files + Tests)

Historical note: several file/test paths below do not match the final repo layout. Keep this table as intent only; do not treat it as the current source-of-truth for file locations.

| Prerequisite | Primary files to change | Test targets |
| --- | --- | --- |
| Runtime autonomy state owner | `custom_components/home_generative_agent/sentinel/engine.py`, new `custom_components/home_generative_agent/sentinel/autonomy_runtime.py`, `custom_components/home_generative_agent/__init__.py` | `tests/custom_components/home_generative_agent/sentinel/test_autonomy_runtime.py`, `tests/custom_components/home_generative_agent/sentinel/test_engine_runtime_level.py` |
| Autonomy service authorization layer *(depends on 1)* | `custom_components/home_generative_agent/__init__.py`, `custom_components/home_generative_agent/services.yaml`, `custom_components/home_generative_agent/const.py` | `tests/custom_components/home_generative_agent/test_services_sentinel_autonomy.py` |
| Structured suppression decision model | `custom_components/home_generative_agent/sentinel/suppression.py`, `custom_components/home_generative_agent/sentinel/engine.py` | `tests/custom_components/home_generative_agent/sentinel/test_suppression_reason_codes.py`, `tests/custom_components/home_generative_agent/sentinel/test_snooze_presence_grace.py` |
| Decoupled execution service *(depends on 1, 3)* | `custom_components/home_generative_agent/notify/actions.py`, `custom_components/home_generative_agent/notify/dispatcher.py`, new `custom_components/home_generative_agent/sentinel/execution_service.py`, `custom_components/home_generative_agent/sentinel/engine.py` | `tests/custom_components/home_generative_agent/notify/test_execution_service.py`, `tests/custom_components/home_generative_agent/notify/test_action_handler_routing.py` |
| Versioned audit subsystem expansion | `custom_components/home_generative_agent/audit/store.py`, `custom_components/home_generative_agent/audit/models.py`, `custom_components/home_generative_agent/sentinel/engine.py`, `custom_components/home_generative_agent/notify/actions.py` | `tests/custom_components/home_generative_agent/audit/test_audit_store_v2.py`, `tests/custom_components/home_generative_agent/audit/test_audit_retention_degraded.py` |
| Sentinel config surface expansion | `custom_components/home_generative_agent/const.py`, `custom_components/home_generative_agent/flows/sentinel_subentry_flow.py`, `custom_components/home_generative_agent/strings.json`, `custom_components/home_generative_agent/translations/en.json`, `custom_components/home_generative_agent/translations/tr.json` | `tests/custom_components/home_generative_agent/flows/test_sentinel_subentry_flow.py` |
| Trigger scheduler extraction | new `custom_components/home_generative_agent/sentinel/trigger_scheduler.py`, `custom_components/home_generative_agent/sentinel/engine.py`, `custom_components/home_generative_agent/__init__.py` | `tests/custom_components/home_generative_agent/sentinel/test_trigger_scheduler.py`, `tests/custom_components/home_generative_agent/sentinel/test_engine_single_flight.py` |
| Snapshot schema extension for per-person presence | `custom_components/home_generative_agent/snapshot/schema.py`, `custom_components/home_generative_agent/snapshot/derived.py`, `custom_components/home_generative_agent/snapshot/builder.py` | `tests/custom_components/home_generative_agent/snapshot/test_snapshot_derived_people.py` |

### Refactor Exit Criteria

- Each prerequisite has dedicated unit tests in place before dependent implementation-sequence work begins.
- Prerequisite 1 exit criterion: engine tests cover the periodic polling path and runtime level transitions in isolation.
- Prerequisite 7 exit criterion: trigger scheduler tests cover the event-driven path, lock contention, and queue overflow; engine integration tests then cover both polling and event-driven paths together.
- Prerequisite 8 exit criterion: snapshot tests verify `people_home` and `people_away` are correctly populated for all occupancy states; `anyone_home` backward compatibility confirmed.
- Service tests verify unauthorized autonomy-level changes are rejected and do not mutate runtime state.
- Audit tests verify new fields are always populated or explicitly `None` with deterministic defaults.

---

## 16. Implementation Sequence

Status note: this sequence is historical. Steps 1-12 have substantial implementation in the current repo, but not all target-state acceptance criteria are met yet.

1. Baseline storage provisioning:
   - Add `sentinel_baselines` table in PostgreSQL.
   - Implement `SentinelBaselineUpdater` (15-min cadence, independent of main loop).
2. Audit schema expansion + reason codes + metrics hooks (`data_quality`, `suppression_reason_code`, `rule_version`, `autonomy_level_at_decision`, `execution_id`, `canary_would_execute`).
3. Versioned state migrations for suppression and audit stores.
4. Event-driven triggers with debounce + bounded queue + TTL + single-flight lock + deterministic drop policy.
5. Correlation pass (deterministic, within-run-only, immutable compounds) before suppression.
6. Suppression upgrades: quiet hours, per-person presence grace, snooze-as-suppression, prompt TTL.
   - Consume snapshot `derived.people_home` and `derived.people_away` (from prerequisite 8) for per-person grace logic.
7. Level 1 triage with strict allowlist/sanitized evidence, fail-open behavior, triage metrics.
8. Notification routing, snapshots, digest, snooze UX.
   - Use `derived.people_away` for occupant context in sensitive finding notifications where policy allows.
9. Shared execution service abstraction; runtime kill switch service with auth checks and override TTL behavior.
10. Level 2 canary mode (no execution): compute and audit `would_auto_execute`.
11. Level 2 guarded auto-execute live mode (allowlist, confidence thresholds, stale-data block, rate limits, idempotency).
12. Temporal/baseline anomaly detectors (depends on step 1).
13. Discovery pipeline improvements: service-mapped suggested actions, on-demand discovery trigger, immediate rule activation, normalization transparency, richer draft notifications, and level-increase PIN validation (see Milestone 5, Issues #16–#22).
14. Level 3 expansion only after burn-in KPIs and stability gates are met. *(Not specified in this plan — requires a dedicated design issue once L2 is proven in production.)*

---

## 17. Rollout and Safety Gates

- Phase rollout with feature flags by capability.
- KPI formulas (single source of truth):
  - False-positive rate (14d): `count(user_response.false_positive == true) / count(total_notified)`
  - Action success rate: `count(action_outcome.status == "success") / count(total_auto_exec_attempts)`
  - User override rate: `count(action_outcome.overridden_by_user == true) / count(total_auto_exec_successful)`

| KPI | L0 -> L1 | L1 -> L2 (canary -> live) | L2 -> L3 |
| --- | --- | --- | --- |
| False-positive rate (14d) | < 15% | < 10% (notification-quality gate; action KPIs remain N/A in canary) | < 5% |
| Action success rate | N/A | N/A (canary has no gate; exists for operator confidence only) | > 95% over >= 20 real actions |
| User override rate | N/A | N/A (canary has no gate) | < 10% |
| Minimum stable period | 7 days | 14 days | 30 days |

- Canary mode (L1 → L2 sub-phase) has no KPI gate. It exists solely for operator confidence - review `canary_would_execute` decisions in the audit log before enabling live execution. The L1 → L2 KPI thresholds apply only to live execution data.
- Rolling KPI values are exposed on health entity for operator visibility.
- One-click rollback to Level 0 via `sentinel_set_autonomy_level` service call (`level: 0`; no `entry_id` required in call).

---

## 18. Acceptance Criteria

Status note: the items below are target-state gates, not statements of current compliance.

Detection integrity:

- Deterministic findings are reproducible for the same snapshot input.
- No LLM path can create or mutate findings.

Concurrency:

- No overlapping engine runs under trigger storms (single-flight lock enforced).
- Trigger queue bounded; drops/discards counted and exposed with reason codes.

Auditability:

- Every suppress/notify/execute decision has auditable reason code.
- Audit records are complete at all pipeline stages.
- Rule version is present in every audit record.

Security:

- Triage prompt contains no raw entity state strings, attribute text, area names, or unsanitized evidence.
- Auto-exec never bypasses allowlist, sensitivity flag, PIN gate, stale-data block, rate limits, or idempotency.
- Kill switch transitions engine to Level 0 immediately via service call.

Data quality:

- `data_quality != "fresh"` findings are never eligible for auto-execute.
- `unavailable` findings are suppressed by default, except `severity="high"` which is notify-only.

Autonomy progression:

- Each level transition requires KPI thresholds from Section 17 for the full minimum stable period.

---

## 19. Testing Strategy

Status note: the repo contains many sentinel tests, but several test module paths named below were never created under the exact directories listed in Section 15. Prefer the actual `tests/custom_components/home_generative_agent/` tree over the historical names in this plan.

| Implementation step | Test category | Key scenarios |
| --- | --- | --- |
| 1 | Baseline storage | Rolling stats accuracy, upsert idempotency, schema migration |
| 2 | Audit schema + reason codes | Field completeness, PII minimization, degraded-retention warning behavior |
| 3 | State migrations | v1->v2 backfill defaults, partial-failure handling, read-only compatibility mode |
| 4 | Trigger queue | Coalescing, queue-full deterministic drop policy, TTL discard, run-lock single-flight |
| 5 | Correlation | Same-run grouping, late-arrival isolation, compound evidence shape |
| 6 | Suppression | Snooze routing, per-person presence grace, quiet-hours, reason codes |
| 7 | Triage | Allowlist/sanitized evidence enforcement, timeout fail-open, triage_confidence audit-only |
| 8 | Notification UX | Snooze writes to suppression, `always` confirmation required |
| 9 | Execution service + runtime overrides | Admin-only level changes, TTL expiry revert to default |
| 10 | Canary mode | `would_auto_execute` accuracy, no side effects |
| 11 | Auto-exec guardrails | Allowlist/confidence/stale/rate/idempotency/PIN gate enforcement |
| 12 | Temporal/baseline detectors | Deviation trigger correctness, baseline freshness handling |
| 13 | Discovery pipeline improvements | Service-mapped actions produce `domain.service` format; on-demand trigger; immediate activation on approval; normalization failure reason codes |
| All | End-to-end integration | Full pipeline with mocked LLM triage and mocked HA service calls; extend `test_sentinel_end_to_end.py` incrementally |

---

## 20. GitHub Issue Breakdown (15 Issues, 4 Milestones)

Each issue is a self-contained PR targeting main. Every PR must leave tests green and behavior backward-compatible unless the issue explicitly permits a behavioral change.

Historical note: the issue-status details below mix repository history with current main-branch state. Milestone 5 statuses below include the original PR #296 work plus the follow-on rule/template/evidence-path fixes that landed on 2026-03-12/13.

**Status as of 2026-03-13:** All 15 original issues were closed on GitHub, and the later Sentinel rule/template gaps from the #298–#320 sweep are now also implemented in code. “Closed” here still means implementation work landed, not that every target-state behavior in Sections 2-19 is complete. Remaining known gaps include suppressed-finding audit coverage, blocked-vs-notified behavior, audit retention/archival, `trigger_source` population, and stable person identifiers in derived presence. One unplanned issue (#9b, GitHub #269) covered pending-prompt TTL separately. Issue #15 (lambda rule review/approval UI) was implemented then removed — lambda rules were detection-only (no service-type suggested actions), invisible from `get_dynamic_rules`, and added no value for full autonomy; see PR #285 and Milestone 5. Milestone 5 plus its 2026-03-12/13 follow-on fixes now cover the discovery/rule work reflected in the current repo.

| Plan # | GitHub # | Status | Title |
| --- | --- | --- | --- |
| #1 | #251 | Done | Structured suppression decision model |
| #2 | #253 | Done | Snapshot per-person presence |
| #3 | #254 | Done | Versioned audit schema and config surface |
| #4 | #255 | Done | Runtime autonomy level and kill-switch service |
| #5 | #256 | Done | Trigger scheduler |
| #6 | #257 | Done | Event-driven triggering |
| #7 | #258 | Done | Correlation pass |
| #8 | #259 | Done | Decouple execution service |
| #9 | #260 | Done | Suppression upgrades |
| #9b | #269 | Done | Pending-prompt TTL (unplanned follow-on) |
| #10 | #261 | Done | Notification routing and UX |
| #11 | #262 | Done | Level 1: LLM triage |
| #12 | #263 | Done | Level 2: Canary mode |
| #13 | #264 | Done | Level 2: Live auto-execute |
| #14 | #265 | Done | Baseline storage and temporal detectors |
| #15 | #266 | Done → Removed | Lambda rule review/approval UI (removed in PR #285) |
| #16 | #288 | Done (PR #296) | Service-mapped suggested actions in normalization |
| #17 | #289 | Done (PR #296) | On-demand discovery trigger |
| #18 | #290 | Done (PR #296) | Immediate rule activation on approval |
| #19 | #291 | Done (PR #296) | Explain normalization failures |
| #20 | #292 | Done (PR #296) | Richer proposal draft notifications |
| #21 | #293 | Done (PR #297) | Rule preview before commit |
| #22 | #294 | Done (PR #296) | PIN validation for autonomy level increase |

---

### Milestone 0 — Parallel Refactors (no external behavioral change; all four may be opened and merged in any order)

**Issue #1 — Structured suppression decision model** *(Done — GitHub #251)*

- Plan coverage: Prerequisite 3, Section 8
- Scope: Change `should_suppress()` to return a `SuppressionDecision(suppress: bool, reason_code: str, context: dict)` dataclass. Update all call sites in `engine.py`. No change to suppression logic.
- Files: `sentinel/suppression.py`, `sentinel/engine.py`
- Tests: `should_suppress()` returns expected reason codes for each suppression path; engine propagates `reason_code` to stub audit record.
- Size: S
- Dependencies: none

**Issue #2 — Snapshot per-person presence** *(Done — GitHub #253)*

- Plan coverage: Prerequisite 8, Section 15 (multi-occupant)
- Scope: Add `people_home: list[str]` and `people_away: list[str]` to `DerivedContext` TypedDict and `SNAPSHOT_SCHEMA`. Keep `anyone_home` and all existing fields; new fields default to `[]` if person-tracking entities are unavailable. Update `snapshot/builder.py` to populate them.
- Files: `snapshot/schema.py`, `snapshot/builder.py`
- Tests: Presence combinations (all home, none home, partial); `anyone_home` remains consistent with `people_home`.
- Size: S
- Dependencies: none

**Issue #3 — Versioned audit schema and config surface** *(Done — GitHub #254)*

- Plan coverage: Prerequisite 5 (partial), Prerequisite 6 (partial), Section 12, Section 13
- Scope: Expand `AuditRecord` with the full field set from Section 12 (`data_quality`, `trigger_source`, `suppression_reason_code`, `triage_confidence`, `canary_would_execute`, `execution_id`, `rule_version`, `autonomy_level_at_decision`). Add `version: int = 1` field. Implement v1→v2 migration in `audit/store.py`. Add `CONF_AUDIT_HOT_MAX_RECORDS`, `CONF_AUDIT_ARCHIVAL_BACKLOG_MAX`, `CONF_AUDIT_RETENTION_DAYS`, `CONF_AUDIT_HIGH_RETENTION_DAYS` to `const.py`. Populate new fields with `None` / sentinel defaults at all existing call sites.
- Files: `audit/models.py`, `audit/store.py`, `const.py`
- Tests: Migration backfills defaults; field completeness; new constants present.
- Size: M
- Dependencies: none

**Issue #4 — Runtime autonomy level and kill-switch service** *(Done — GitHub #255)*

- Plan coverage: Prerequisite 1, Prerequisite 2, Section 3, Section 5 (kill switch)
- Scope: Register `home_generative_agent.sentinel_set_autonomy_level` HA service (admin-only, `level: 0|1|2|3`; entry resolved internally, no `entry_id` in call schema). Read runtime level from in-memory override (TTL-bounded) falling back to `CONF_SENTINEL_AUTONOMY_LEVEL`. Add `CONF_SENTINEL_AUTONOMY_LEVEL`, `CONF_SENTINEL_RUNTIME_OVERRIDE_TTL_MINUTES`, `CONF_SENTINEL_REQUIRE_PIN_FOR_LEVEL_INCREASE` to `const.py`. No behavioral change to detection logic.
- Files: `__init__.py`, `sentinel/engine.py`, `const.py`, `services.yaml`
- Tests: Service call sets level; TTL expiry reverts to config default; level increase blocked without PIN when required; level decrease never requires PIN.
- Size: M
- Dependencies: none

---

### Milestone 1 — Pipeline Infrastructure (Issues #5–#8; merge in order)

**Issue #5 — Trigger scheduler** *(Done — GitHub #256)*

- Plan coverage: Prerequisite 7, Section 6
- Scope: Extract `SentinelTriggerScheduler` with bounded queue (max 10), coalescing window (5 s default), TTL (30 s), deterministic drop policy (prefer security-critical), and `asyncio.Lock` single-flight. Wire into `engine._run_loop()` without changing polling behavior.
- Files: `sentinel/trigger_scheduler.py` (new), `sentinel/engine.py`
- Tests: Coalescing; queue-full drop policy; TTL discard; lock prevents concurrent runs.
- Size: M
- Dependencies: none (can open alongside Milestone 0 issues)

**Issue #6 — Event-driven triggering** *(Done — GitHub #257)*

- Plan coverage: Prerequisite 7 (complete), Section 6
- Scope: Subscribe to HA state-change events for sentinel-watched entity IDs. On qualifying change, enqueue a trigger via `SentinelTriggerScheduler`. Retain polling as fallback when queue is empty.
- Files: `sentinel/engine.py`, `sentinel/trigger_scheduler.py`
- Tests: State-change fires trigger; duplicate events within coalescing window produce one run; polling still runs when no events queued.
- Size: M
- Dependencies: Issue #5

**Issue #7 — Correlation pass** *(Done — GitHub #258)*

- Plan coverage: Section 7
- Scope: Implement `SentinelCorrelator.correlate(findings)` producing `CompoundFinding` for related findings within a single `_run_once()` call. Compound findings are immutable once emitted. Wire into pipeline between rule evaluation and suppression.
- Files: `sentinel/correlator.py` (new), `sentinel/engine.py`, `sentinel/models.py`
- Tests: Same-run grouping; cross-run isolation (no late-arrival merging); compound evidence shape; immutability after emit.
- Size: M
- Dependencies: none (can open alongside Issue #5)

**Issue #8 — Decouple execution service** *(Done — GitHub #259)*

- Plan coverage: Prerequisite 4, Section 4 (guardrails), Section 2 (staleness validation)
- Scope: Extract `SentinelExecutionService` from `engine.py`. Implement idempotency key (`sha256(anomaly_id + action_policy_path + policy_version + window_bucket)`), allowlist enforcement, sensitivity-flag block, stale-data block, rate limiter, and PIN gate stub. `policy_version` computed as `sha256(json.dumps({...}, sort_keys=True))` over global policy config fields including `effective_rule_threshold`. Add staleness validation pass using `CONF_SENTINEL_STALENESS_THRESHOLD_SECONDS`.
- Files: `sentinel/execution.py` (new), `sentinel/engine.py`, `const.py`
- Tests: Each guardrail independently blocks execution; idempotency deduplicates within window; stale findings never auto-execute; `data_quality` field set correctly.
- Size: L
- Dependencies: Issues #3, #4

---

### Milestone 2 — Suppression and Notification (Issues #9–#10; merge in order)

**Issue #9 — Suppression upgrades** *(Done — GitHub #260; pending-prompt TTL done separately as GitHub #269)*

- Plan coverage: Section 8 (full), Prerequisite 3 (complete), Prerequisite 8 (consumed)
- Scope: Add snooze routing (snooze writes to `SuppressionState` not `pending_prompts`), per-person presence-grace window, quiet-hours enforcement, `SuppressionState.version` with v1→v2 migration, read-only compatibility mode on downgrade. All suppression paths emit `suppression_reason_code` via `SuppressionDecision`. Pending-prompt TTL and expiry was implemented as a follow-on PR (GitHub #269/PR #273).
- Files: `sentinel/suppression.py`, `sentinel/engine.py`
- Tests: Snooze-as-suppression; per-person grace (occupied vs. vacated areas); quiet-hours; reason codes for all paths; migration and downgrade-safe read.
- Size: M
- Dependencies: Issues #1, #2

**Issue #10 — Notification routing and UX** *(Done — GitHub #261)*

- Plan coverage: Section 9
- Scope: Implement `SentinelNotifier` with `always` confirmation guard, per-area/per-person routing, snooze action that calls suppression, and `is_sensitive` redaction in notification text. Wire into pipeline after suppression pass.
- Files: `sentinel/notifier.py` (new), `sentinel/engine.py`
- Tests: `always` confirmation required before any HA action; snooze action writes to suppression; sensitive-flag redacts person names in notification; per-area routing delivers to correct target.
- Size: M
- Dependencies: Issues #9

---

### Milestone 3 — Autonomy Features (Issues #11–#13; strictly sequential)

**Issue #11 — Level 1: LLM triage** *(Done — GitHub #262)*

- Plan coverage: Section 7 (triage pass)
- Scope: Implement `SentinelTriageService`. Build triage prompt from the Section 6 allowlist only (`type`, `severity`, `confidence`, `is_sensitive`, `entity_count`, `suggested_actions_count`) plus explicitly allowed sanitized evidence fields (`is_night`, `anyone_home`, `recognized_people_count`, `last_changed_age_seconds`) when available. Timeout fail-open (treat as `notify`). Record `triage_confidence` in `AuditRecord` (audit-only; never gates execution). Triage output gates only `notify` vs. `suppress`.
- Files: `sentinel/triage.py` (new), `sentinel/engine.py`
- Tests: Allowlist enforced (no raw state/attribute/evidence strings in prompt); timeout produces `notify`; `triage_confidence` written to audit and not to `finding.confidence`; suppressed-by-triage gets reason code.
- Size: M
- Dependencies: Issues #3, #8

**Issue #12 — Level 2: Canary mode** *(Done — GitHub #263)*

- Plan coverage: Section 3 (canary), Section 17 (KPI)
- Scope: When `CONF_SENTINEL_AUTO_EXEC_CANARY_MODE=true`, compute `would_auto_execute` using full guardrail logic and record in `AuditRecord.canary_would_execute`. No action taken. No KPI gate for canary-to-live transition (operator decides). False-positive rate notification-quality KPI still applies during canary.
- Files: `sentinel/execution.py`, `sentinel/engine.py`
- Tests: `canary_would_execute` accurate; no side effects; canary audit records distinguishable from live-execution records.
- Size: S
- Dependencies: Issues #8, #11

**Issue #13 — Level 2: Live auto-execute** *(Done — GitHub #264, closed 2026-03-05)*

- Plan coverage: Section 4, Section 17 (L1→L2 action KPIs)
- Scope: Enable live auto-execution behind all guardrails from Issue #8. Idempotency key prevents double-fire. Rate limiter enforced. L1→L2 rollout thresholds from Section 17 (false-positive rate < 10% notification-quality gate; action KPIs N/A during canary) and zero unintended irreversible actions must be met before enabling in production — enforced by rollout process, not code gate.
- Files: `sentinel/execution.py`, `sentinel/engine.py`
- Tests: Full guardrail integration test; idempotency; rate limit; audit outcome populated; level < 2 blocks execution.
- Size: M
- Dependencies: Issue #12

---

### Milestone 4 — Advanced Rules and Level 3 (Issues #14–#15; merge in order)

> **Note:** Issue #14 was completed alongside Milestone 2/3 work (GitHub #265, merged 2026-03-01), ahead of the original milestone order. This was valid since its only dependency is Issue #8. Issue #15 can proceed since its dependency (#14) is done.
>
> **Level 3 is not implemented in this milestone.** The milestone title reflects the original roadmap grouping; Issues #14 and #15 are prerequisites for advanced autonomy but do not constitute Level 3 itself. Level 3 requires a separate design issue that should only be opened after L2 has completed its 30-day minimum stable period and all L2→L3 KPI thresholds from Section 17 are met in production.

**Issue #14 — Baseline storage and temporal detectors** *(Done — GitHub #265)*

- Plan coverage: Section 10 (baseline rules), Section 16 step 1
- Scope: Implement `SentinelBaselineUpdater` (15-min cadence) writing rolling stats to `sentinel_baselines` PostgreSQL table. Implement temporal-deviation and time-of-day anomaly detector templates that read from baseline. Add `baseline_freshness` check to staleness validation.
- Files: `sentinel/baseline.py` (new), `sentinel/dynamic_rules.py`, `sentinel/engine.py`
- Tests: Rolling stats accuracy; upsert idempotency; schema migration; deviation trigger correctness; stale baseline handling.
- Size: L
- Dependencies: Issue #8
- Post-merge fixes (2026-03-12, PR #323, branch `feat/sentinel-baseline-detection-complete`):
  - Added `async_fetch_baselines()` to `SentinelBaselineUpdater` — bulk-reads all `sentinel_baselines` rows into `{entity_id: {metric: float}}`.
  - Wired `engine.py` `_run_once()` to call `async_fetch_baselines()` and forward the result to `evaluate_dynamic_rules()`; previously the `baselines` argument was always an empty dict so neither template ever fired.
  - Added the three baseline config constants to `SentinelSubentryFlow`: `CONF_SENTINEL_BASELINE_ENABLED`, `CONF_SENTINEL_BASELINE_UPDATE_INTERVAL_MINUTES`, `CONF_SENTINEL_BASELINE_FRESHNESS_THRESHOLD_SECONDS` — they were defined in `const.py` but never imported or shown in the UI.
  - Added translations for the three new fields in `strings.json`, `translations/en.json`, and `translations/tr.json`.
  - Added 4 unit tests for `async_fetch_baselines` (dict rows, tuple rows, DB error, empty result).
  - Updated README Baseline Detection section: correct method name (`check_freshness`), accurate `threshold_pct` per-rule description, removed "can be registered" phrasing.

**Issue #15 — Lambda rule review/approval UI** *(Done → Removed — GitHub #266, closed 2026-03-05; feature removed in PR #285)*

- Originally implemented: AST-validated expression rules with `pending → active` lifecycle and two service calls (`sentinel_receive_lambda_rule`, `sentinel_approve_lambda_rule`).
- Removed because: lambda rules were detection-only (no `suggested_actions` field exposed in the service, so `_auto_execute_finding` always returned `no_actions`); pending rules were invisible from `get_dynamic_rules` (stored in a separate registry); the feature added a third rule lifecycle alongside built-in rules and the discovery pipeline without enabling full autonomy.
- Successor: Milestone 5 (Issues #16–#22) addresses the underlying need through discovery pipeline improvements.

---

## Milestone 5 — Discovery Pipeline Improvements (Issues #16–#22)

The original Issue #15 (lambda rule review/approval UI) was implemented and subsequently removed (PR #285) because lambda rules were detection-only, invisible from the review UI, and provided no path to full autonomy. Milestone 5 addresses the underlying need properly: making the discovery pipeline fast, transparent, and capable of producing rules that can trigger autonomous action.

**Issue #16 — Service-mapped suggested actions** *(Done — [GitHub #288](https://github.com/goruck/home-generative-agent/issues/288), merged in PR #296)*

- Plan coverage: Section 4 (auto-execute guardrails), Section 3 (Level 2/3 autonomy)
- Scope: Update `normalize_candidate()` in `proposal_templates.py` to produce HA service calls (e.g. `lock.lock`) as `suggested_actions` for templates where a safe, deterministic action exists. `is_sensitive=True` templates remain blocked from auto-execute by execution guardrail #3 regardless of suggested actions. Templates with no safe automated action continue to produce advisory text only.
- Files: `sentinel/proposal_templates.py`
- Tests: `normalize_candidate()` produces `domain.service` format for applicable lock/entry templates; `is_sensitive=True` templates still blocked at execution guardrail; advisory-only templates produce no service-type actions; existing auto-execute integration tests pass unchanged.
- Size: S
- Dependencies: none
- **This is the single highest-value change for full autonomy — without it, no discovered rule can ever trigger auto-execute.**
- Implemented outcome: normalized proposals now emit service-mapped `suggested_actions` where safe deterministic actions exist.

**Issue #17 — On-demand discovery trigger** *(Done — [GitHub #289](https://github.com/goruck/home-generative-agent/issues/289), merged in PR #296)*

- Plan coverage: Section 16 (discovery latency)
- Scope: Add `trigger_sentinel_discovery` HA service that runs the LLM discovery cycle against the current snapshot immediately, bypassing the periodic timer. Deduplication and semantic key filtering still apply. Periodic timer is unaffected.
- Files: `sentinel/discovery_engine.py`, `__init__.py`, `services.yaml`
- Tests: Service call triggers a discovery run and stores results; duplicate candidates are filtered; timer interval unchanged.
- Size: S
- Dependencies: none
- Implemented outcome: `home_generative_agent.trigger_sentinel_discovery` runs one discovery cycle immediately when called.

**Issue #18 — Immediate rule activation on approval** *(Done — [GitHub #290](https://github.com/goruck/home-generative-agent/issues/290), merged in PR #296)*

- Plan coverage: Section 16 (approval latency)
- Scope: After `approve_rule_proposal` successfully adds a rule to `RuleRegistry`, trigger a single `_run_once()` against the current snapshot. Rule becomes live in seconds rather than waiting up to one hour for the next scheduled cycle.
- Files: `__init__.py` (approve handler), `sentinel/engine.py`
- Tests: Approval triggers an immediate evaluation run; new rule fires on current snapshot if conditions met; no double-run if engine is already mid-cycle.
- Size: S
- Dependencies: none
- Implemented outcome: successful proposal approval now triggers an immediate Sentinel evaluation cycle through the scheduler's single-flight path.

**Issue #19 — Explain normalization failures** *(Done — [GitHub #291](https://github.com/goruck/home-generative-agent/issues/291), merged in PR #296)*

- Plan coverage: Section 16 (transparency)
- Scope: When `normalize_candidate()` returns `None`, return a structured reason to the caller (e.g. `no_matching_entity_types`, `unsupported_pattern`, `missing_required_entities`). Surface this reason in the `promote_discovery_candidate` and `approve_rule_proposal` service responses. When `promote` or `approve` returns `already_active`, include the covering rule ID and which entity IDs overlapped.
- Files: `sentinel/proposal_templates.py`, `__init__.py`
- Tests: Each normalization failure path returns a distinct reason code; `promote` and `approve` service responses include reason; `already_active` response includes covering rule ID and overlapping entities.
- Size: S
- Dependencies: none
- Implemented outcome: promote/approve responses now return structured `reason_code`, `details`, `covered_rule_id`, and `overlapping_entity_ids` when applicable.

**Issue #20 — Richer proposal draft notifications** *(Done — [GitHub #292](https://github.com/goruck/home-generative-agent/issues/292), merged in PR #296)*

- Plan coverage: Section 9 (notification UX)
- Scope: Include `template_id`, `severity`, and `confidence` in the notification sent when a proposal draft is created. Example: *"New HIGH-severity proposal: alarm disarmed + entry open (80% confident) — call approve_rule_proposal to activate."*
- Files: `__init__.py` (promote handler)
- Tests: Notification payload includes template_id, severity, confidence, and actionable service call hint.
- Size: XS
- Dependencies: none
- Implemented outcome: draft notifications now include template context and actionable approval guidance instead of a generic message.

**Issue #21 — Rule preview before commit** *(Done — [GitHub #293](https://github.com/goruck/home-generative-agent/issues/293), merged in PR #297)*

- Plan coverage: Section 16 (operator control)
- Scope: Add `preview_rule_proposal` HA service that evaluates the normalized rule spec against the current snapshot without writing to the registry. Returns whether the rule would trigger right now, and against which entities. Intended as a dry-run step before calling `approve_rule_proposal`.
- Files: `__init__.py`, `sentinel/dynamic_rules.py`, `services.yaml`
- Tests: Preview evaluates rule correctly; no registry mutation; returns trigger status and matching entities; behaves identically to a live evaluation for the same snapshot.
- Size: M
- Dependencies: Issue #18 (shares immediate-snapshot evaluation path)
- Implemented outcome: `home_generative_agent.preview_rule_proposal` performs a read-only evaluation of a stored proposal draft against the current snapshot, reusing the dynamic rule evaluation path so preview behavior matches live deterministic evaluation.

**Issue #22 — PIN validation for autonomy level increase** *(Done — [GitHub #294](https://github.com/goruck/home-generative-agent/issues/294), merged in PR #296)*

- Plan coverage: Section 3 (runtime kill switch and override lifecycle), Section 15 (Prerequisite 2)
- Scope: `sentinel_set_autonomy_level` now validates `pin` on level increases. Sentinel stores salted hash material in config (not plaintext) and compares the provided PIN when `CONF_SENTINEL_REQUIRE_PIN_FOR_LEVEL_INCREASE=true`. Level decreases and level 0 (kill switch) never require PIN.
- Files: `sentinel/engine.py`, `flows/sentinel_subentry_flow.py`, `const.py`, `strings.json`, `translations/en.json`
- Tests: Correct PIN allows increase; wrong PIN rejected; no PIN required for decrease; kill-switch (`level=0`) never gated; PIN stored as hash not plaintext.
- Size: M
- Dependencies: none
- Implemented outcome: Sentinel now persists hashed level-increase PIN material in subentry config and validates the provided PIN on autonomy increases.

Post-merge follow-on fixes now also landed in the repo (2026-03-12/13):

- Baseline detection wiring fix: `SentinelBaselineUpdater.async_fetch_baselines()` is now called from `engine.py`, and the baseline config fields are exposed in the Sentinel subentry flow. Before this, `baseline_deviation` / `time_of_day_anomaly` always saw an empty baselines dict.
- Static rule coverage expansion: four new built-in rules closed the remaining novel-pattern issues from the #298–#320 sweep: `alarm_disarmed_during_external_threat` (#309), `camera_backgarage_missing_snapshot_night_home` (#311), `vehicle_parked_near_frontgate_home` (#312), and `unknown_person_frontporch_night_home` (#318).
- Proposal normalization gap fixes: discovery candidates for high-energy-at-night, alarm-disarmed-during-threat, and window-duration cases now normalize to supported deterministic templates instead of returning `unsupported`.
- Evidence-path parsing now accepts dot-notation entity references like `sensor.foo.state` and `lock.foo.battery_level`, which matches the LLM output seen in discovery records and fixes several false `unsupported` approvals.
- Built-in rule overlap detection is more tolerant of LLM evidence paths that omit the domain prefix (`frontgate`, `backgarage`), and battery-low lock candidates no longer misroute to unlocked-lock templates.

---

### Dependency Graph

```text
#1  -> #9
#2  -> #9
#9  -> #10

#3  -> #8
#4  -> #8
#3  -> #11
#8  -> #11
#8  -> #12
#11 -> #12
#12 -> #13

#8  -> #14
#14 -> #15 (removed)

#5  -> #6

#7  (independent)

#16 (independent)
#17 (independent)
#18 (independent)
#19 (independent)
#20 (independent)
#21 -> #18
#22 (independent)
```

Issues #1, #2, #3, #4, #5, and #7 have no mutual dependencies and can all be opened simultaneously as individual PRs.

Declared issue dependencies (authoritative):

- `#6 -> #5`
- `#8 -> #3, #4`
- `#9 -> #1, #2`
- `#10 -> #9`
- `#11 -> #3, #8`
- `#12 -> #8, #11`
- `#13 -> #12`
- `#14 -> #8`
- `#21 -> #18`

---

### Practical Notes

- Each issue should reference the relevant plan sections in its description body.
- Milestone 0 issues can be merged in any order and independently reviewed.
- Milestone 1–4 issues should be merged in the numbered order within each milestone.
- The `execution_id` idempotency implementation in Issue #8 must be complete before any auto-execution work in Issues #12–#13.
- Canary mode (Issue #12) must be run in production for a minimum observation period before Issue #13 is enabled; this gate is a rollout process control, not a code gate.
- Issue #14 requires a PostgreSQL instance; it is safely skippable if that dependency is not available in the deployment environment.
- Milestone 5 issues (#16–#22) are independent of each other except #21 (preview depends on #18 for the immediate-activation path) and can be opened and merged in any order.

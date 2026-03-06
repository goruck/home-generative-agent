# Sentinel — Proactive Deterministic Anomaly Detection

## 1. Non-Negotiable Principles

- Detection remains fully deterministic and rule-based.
- LLMs never create, modify, or delete findings.
- LLMs may only advise on notification handling, explanation text, and user-triggered or policy-allowed execution.
- Every suppression, triage decision, and action outcome is auditable with an explicit reason code.
- Human override and kill switch are always available as runtime operations (not config-file edits).
- Detection rule configs are version-controlled and reviewed before deployment. Rule schema versions are stored in every audit record.

---

## 2. Target Detection/Response Pipeline

1. Trigger received (polling or event-driven).
2. Snapshot built deterministically.
3. Staleness validation pass:
   - Triggering entities checked against `CONF_SENTINEL_STALENESS_THRESHOLD_SECONDS`.
   - `data_quality="stale"` findings are notify-only regardless of autonomy level.
   - `data_quality="unavailable"` findings are suppressed by default, except `severity="high"` which is notify-only.
4. Rules evaluated deterministically.
5. Correlation pass merges related findings deterministically (within the current `_run_once()` call only; compound findings are immutable once emitted).
6. Suppression pass evaluates cooldowns, quiet-hours, snooze policies, and presence-grace windows. Every suppressed finding receives an explicit `suppression_reason_code`.
7. Triage pass (optional, autonomy level >= 1) decides `notify` vs `suppress` only. Triage output is informational; it does not influence execution eligibility.
8. Action policy gate decides `prompt_user`, `handoff`, `auto_execute_candidate`, or `blocked`.
9. For autonomy level >= 2, optional canary mode evaluates `would_auto_execute` and records decisions without executing.
10. Notification dispatch (or audit-only for low-severity policy cases).
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
CONF_AUDIT_HOT_MAX_RECORDS: 500
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

### Levels

- Level 0 Notify-only: no autonomous execution.
- Level 1 LLM triage (notify-only): triage may suppress or enrich notifications; no actions executed.
- Level 2 Guarded auto-execute: non-sensitive, allowlisted services, `finding.confidence` gate, rate limits, idempotency, and stale-data block.
- Level 3 Broad autonomy: expanded execution scope, but PIN/critical actions still blocked from autonomous path unless explicitly policy-approved. **Level 3 is intentionally not specified in this plan beyond the KPI gate in Section 17. It requires a dedicated design issue after L2 has completed its 30-day burn-in and all L2→L3 KPI thresholds are met.**

### Runtime kill switch and override lifecycle

- Service: `home_generative_agent.sentinel_set_autonomy_level`.
- `level: 0` is the kill-switch call and must take effect immediately without restart.
- Service is admin-only. Does not require an explicit `entry_id` in the service call; the engine derives the active entry internally.
- Runtime override is in-memory by default. If `CONF_SENTINEL_RUNTIME_OVERRIDE_TTL_MINUTES` is set, override auto-expires and reverts to config default, with an audit event.
- `CONF_SENTINEL_REQUIRE_PIN_FOR_LEVEL_INCREASE: true` (default) enforces PIN verification for level increases into 2 or 3. Never required for lowering. This is a declared policy with a fixed default, not optional hardening.

---

## 4. Execution Guardrails

- Sensitive or PIN-gated actions are never auto-executed by default.
- Strict domain/service allowlist required for auto-execution.
- Per-rule and global `finding.confidence` thresholds both must pass. Triage confidence does not influence execution eligibility.
- `auto_execute` is unconditionally disabled for `data_quality != "fresh"`.
- Rate limiter: max actions per time window (per-hour counter, burst-protected).
- Idempotency key required for autonomous actions:
  - `execution_id = sha256(anomaly_id + action_policy_path + policy_version + window_bucket)`
  - `policy_version = sha256(json.dumps({"allowed_services": sorted(CONF_SENTINEL_AUTO_EXECUTE_ALLOWED_SERVICES), "autonomy_level": autonomy_level_at_decision, "min_confidence": CONF_SENTINEL_AUTO_EXECUTE_DEFAULT_MIN_CONFIDENCE, "max_actions_per_hour": CONF_SENTINEL_AUTO_EXECUTE_MAX_ACTIONS_PER_HOUR, "require_pin_for_increase": CONF_SENTINEL_REQUIRE_PIN_FOR_LEVEL_INCREASE, "effective_rule_threshold": effective_rule_threshold}, sort_keys=True))` — captures global execution policy and effective rule-level threshold. Structured serialization prevents hash collisions from string concatenation. Changes automatically when any execution-relevant policy changes.
  - `window_bucket = floor(utc_epoch_seconds / (CONF_SENTINEL_EXECUTION_IDEMPOTENCY_WINDOW_MINUTES * 60))` — default 15 min window; align with the entity's cooldown period when possible.
  - Duplicate `execution_id` attempts are blocked and audited.
- Runtime kill switch disables all autonomous execution immediately.
- If policy evaluation fails for any reason, default to notify (safe fallback).
- Rejected auto-execute attempts (allowlist/confidence/rate/idempotency blocks) are queued as user-prompt actions, not silently dropped.

---

## 5. Event-Driven Triggering Design

- Add entity-filtered listeners for locks, entries, windows, motion, and cameras.
- Keep periodic polling for duration/baseline rules.
- Debounce/coalescing window (configurable, default 5 s): triggers are coalesced if they share `entity_id + trigger_type` within the window.
- Single-flight run lock (`asyncio.Lock`) prevents overlapping `_run_once()` calls.
- Bounded trigger queue: max depth `TRIGGER_QUEUE_MAX` (default 10).
- Queue overload policy (deterministic):
  - Prefer retaining security-critical triggers (lock/entry/camera).
  - Drop oldest non-critical trigger first.
  - Emit `drop_reason_code` (`queue_full_non_critical`, `queue_full_oldest`, etc.).
- Trigger TTL: queued entries older than `ttl_seconds` (default 30 s) at lock-acquisition time are discarded with reason code `trigger_ttl_expired`.
- All trigger drops/discards are counted and exposed on health entity; no silent discard.

---

## 6. LLM Triage Design (Level 1+)

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
- Store triage latency, model/provider metadata, and `triage_confidence` in audit record.

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

- Existing type-based and entity-based cooldowns retained.
- Add quiet-hours policy by severity.
- Per-person presence-grace windows:
  - On person departure/arrival, write `person_entity_id -> ISO expiry` in `SuppressionState.presence_grace_until` (default 10 min).
  - Presence-sensitive findings suppressed during active grace windows with reason `presence_grace`.
- Snooze routes through suppression:
  - Options `24h`, `7d`, `always` write to `SuppressionState.snoozed_until` keyed by finding type.
  - `always` requires explicit confirmation before write.
  - Reason codes: `user_snooze_24h`, `user_snooze_7d`, `user_snooze_permanent`.
- Pending-prompt TTL and auto-resolution states.
- Every suppressed finding gets explicit `suppression_reason_code` in audit.

---

## 9. Notification UX Improvements

- Severity routing policy:
  - `high`: push + persistent
  - `medium`: persistent + optional push
  - `low`: audit-only or digest
- Attach camera snapshots where available and permissioned.
- Digest mode for burst findings.
- Structured snooze options (`24h`, `7d`, `always`) shown in notification and written through suppression path (Section 8).
- False Alarm action button included on every notification; tapping it sets `user_response.false_positive = True` in the audit record. *(Implemented)*
- Sensitive finding notifications may include occupant context from `derived.people_away` when policy allows.

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

For each finding record, persist:

```python
@dataclass(frozen=True)
class AuditRecord:
    snapshot_ref: dict
    finding: dict
    data_quality: str                    # "fresh" | "stale" | "unavailable"
    trigger_source: str                  # "poll" | "event"
    correlation_metadata: dict | None
    suppression_decision: str | None     # "suppressed" | "passed"
    suppression_reason_code: str | None
    triage_decision: str | None          # "notify" | "suppress" | "error" | None
    triage_reason_code: str | None
    triage_confidence: float | None      # audit-only
    triage_model: str | None
    triage_latency_ms: int | None
    action_policy_path: str | None       # "prompt_user" | "handoff" | "auto_execute" | "blocked"
    canary_would_execute: bool | None
    execution_id: str | None
    notification: dict
    user_response: dict | None
    action_outcome: dict | None
    rule_version: str
    autonomy_level_at_decision: int
    # timestamps in UTC ISO 8601
```

PII minimization:
- If `finding.is_sensitive` is `True`, replace `evidence.recognized_people` with `recognized_people_count: int` before storage.

Retention model:
- Hot local store (`Store`): capped by `CONF_AUDIT_HOT_MAX_RECORDS`.
- Time pruning: default `CONF_AUDIT_RETENTION_DAYS=90`.
- High-severity SLA (`CONF_AUDIT_HIGH_RETENTION_DAYS=365`) requires PostgreSQL archival backend.
- Archival unavailability policy:
  - Hot store continues accepting records up to `CONF_AUDIT_HOT_MAX_RECORDS`; once at cap, overwrite deterministically by dropping oldest `low`, then oldest `medium`, and only then oldest `high` if no other choice remains.
  - Invariant: never drop `high`-severity records while any `low` or `medium` records remain in hot store.
  - High-severity records pending archival queue in memory up to `CONF_AUDIT_ARCHIVAL_BACKLOG_MAX` (default 100).
  - Records beyond the backlog limit are dropped; each drop increments `audit_archival_drop_count` on the health entity.
  - Health entity exposes `audit_archival_status: "ok" | "degraded" | "backlog_full"`.
  - Acknowledged bounded loss is preferable to silent gaps in the audit trail.

---

## 13. Schema Versioning and Migrations

- Add explicit versions:
  - `SuppressionState.version`
  - `AuditRecord.version`
- Provide deterministic migrations for each version step (`v1 -> v2`, `v2 -> v3`, etc.).
- Migration requirements:
  - Backfill missing fields with safe defaults.
  - Preserve old records if migration partially fails; mark degraded mode.
  - Emit migration summary metrics (`migrated`, `skipped`, `failed`).
- Rollback policy:
  - Never destructive rewrite in place without backup snapshot.
  - If downgrade is attempted, load in read-only compatibility mode: the store accepts no new writes, the engine is forced to Level 0 (notify-only), and a persistent health warning is emitted until the version mismatch is explicitly resolved.

---

## 14. Deterministic Time and Ordering Rules

- All timestamps are UTC ISO 8601 from a single clock helper.
- Window calculations (staleness, TTL, correlation, rate limit) use monotonic elapsed checks where possible; UTC timestamps are persisted for audit.
- Tie-breakers for equal timestamps are deterministic: `anomaly_id` lexical order, then `entity_id` lexical order.
- Queue ordering is FIFO after coalescing and priority/drop policy.

---

## 15. Implementation-Architecture Prerequisites

The following architecture refactors are required before the feature sequence can be delivered safely:

1. Runtime autonomy state owner:
   - Add a dedicated runtime controller for autonomy level, override expiry, and kill-switch state.
   - Do not rely on mutating static `options` payload for runtime changes.
2. Autonomy service authorization layer (depends on 1):
   - Gate `home_generative_agent.sentinel_set_autonomy_level` with admin-only checks.
   - Entry is resolved internally; callers do not pass `entry_id` in the service call.
   - Enforce PIN verification for increases into levels 2/3 when `CONF_SENTINEL_REQUIRE_PIN_FOR_LEVEL_INCREASE=true`.
3. Structured suppression decision model:
   - Replace boolean suppression outcome with structured result (`suppressed`, `reason_code`, `expires_at`, metadata).
   - Required for quiet-hours, snooze routing, presence-grace windows, and deterministic audit reasoning.
4. Decoupled execution service (depends on 1, 3):
   - Extract execution/handoff policy from notification action callbacks into a shared service callable.
   - Both mobile-action flow and autonomous flow must use the same policy evaluation path.
5. Versioned audit subsystem expansion:
   - Introduce the expanded audit schema fields and versioned writer before autonomy/canary rollout.
   - Add degraded archival status/backlog counters as first-class health metrics.
6. Sentinel config surface expansion:
   - Add new constants, flow controls, and translations for autonomy/staleness/idempotency/retention settings before policy logic rollout.
7. Trigger scheduler extraction:
   - Implement queue/coalescing/TTL/drop-policy/single-flight in a dedicated scheduler component, separate from rule evaluation logic.
8. Snapshot schema extension for per-person presence:
   - Extend `DerivedContext` to include `people_home: list[str]` and `people_away: list[str]` (person entity IDs).
   - Update `snapshot/schema.py`, `snapshot/derived.py`, and `snapshot/builder.py`.
   - `anyone_home` remains (`len(people_home) > 0`) to preserve backward compatibility with existing rules.
   - Required by: per-person presence-grace windows (3), notification occupant context (Section 9), and suppression upgrades (implementation step 6).

### Prerequisite Refactor Task Map (Files + Tests)

| Prerequisite | Primary files to change | Test targets |
|---|---|---|
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

Prerequisite: all items in Section 15 are completed.

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
13. Discovery pipeline improvements: service-mapped suggested actions, on-demand discovery trigger, immediate rule activation, normalization transparency (see Milestone 5, Issues #16–#21).
14. Level 3 expansion only after burn-in KPIs and stability gates are met. *(Not specified in this plan — requires a dedicated design issue once L2 is proven in production.)*

---

## 17. Rollout and Safety Gates

- Phase rollout with feature flags by capability.
- KPI formulas (single source of truth):
  - False-positive rate (14d): `count(user_response.false_positive == true) / count(total_notified)`
  - Action success rate: `count(action_outcome.status == "success") / count(total_auto_exec_attempts)`
  - User override rate: `count(action_outcome.overridden_by_user == true) / count(total_auto_exec_successful)`

| KPI | L0 -> L1 | L1 -> L2 (canary -> live) | L2 -> L3 |
|---|---|---|---|
| False-positive rate (14d) | < 15% | < 10% (notification-quality gate; action KPIs remain N/A in canary) | < 5% |
| Action success rate | N/A | N/A (canary has no gate; exists for operator confidence only) | > 95% over >= 20 real actions |
| User override rate | N/A | N/A (canary has no gate) | < 10% |
| Minimum stable period | 7 days | 14 days | 30 days |

- Canary mode (L1 → L2 sub-phase) has no KPI gate. It exists solely for operator confidence - review `canary_would_execute` decisions in the audit log before enabling live execution. The L1 → L2 KPI thresholds apply only to live execution data.
- Rolling KPI values are exposed on health entity for operator visibility.
- One-click rollback to Level 0 via `sentinel_set_autonomy_level` service call (`level: 0`; no `entry_id` required in call).

---

## 18. Acceptance Criteria

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

| Implementation step | Test category | Key scenarios |
|---|---|---|
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

**Status as of 2026-03-06:** All 15 original issues closed on GitHub. One unplanned issue (#9b, GitHub #269) covered pending-prompt TTL separately. Issue #15 (lambda rule review/approval UI) was implemented then removed — lambda rules were detection-only (no service-type suggested actions), invisible from `get_dynamic_rules`, and added no value for full autonomy; see PR #285 and Milestone 5. Discovery pipeline improvements that properly enable full autonomy are tracked as issues #16–#21 in Milestone 5 below.

| Plan # | GitHub # | Status | Title |
|---|---|---|---|
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
| #16 | TBD | Open | Service-mapped suggested actions in normalization |
| #17 | TBD | Open | On-demand discovery trigger |
| #18 | TBD | Open | Immediate rule activation on approval |
| #19 | TBD | Open | Explain normalization failures |
| #20 | TBD | Open | Richer proposal draft notifications |
| #21 | TBD | Open | Rule preview before commit |
| #22 | TBD | Open | PIN validation for autonomy level increase |

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

**Issue #15 — Lambda rule review/approval UI** *(Done → Removed — GitHub #266, closed 2026-03-05; feature removed in PR #285)*

- Originally implemented: AST-validated expression rules with `pending → active` lifecycle and two service calls (`sentinel_receive_lambda_rule`, `sentinel_approve_lambda_rule`).
- Removed because: lambda rules were detection-only (no `suggested_actions` field exposed in the service, so `_auto_execute_finding` always returned `no_actions`); pending rules were invisible from `get_dynamic_rules` (stored in a separate registry); the feature added a third rule lifecycle alongside built-in rules and the discovery pipeline without enabling full autonomy.
- Successor: Milestone 5 (Issues #16–#21) addresses the underlying need through discovery pipeline improvements.

---

### Practical Notes

- Each issue should reference the relevant plan sections in its description body.
- Milestone 0 issues can be merged in any order and independently reviewed.
- Milestone 1–4 issues should be merged in the numbered order within each milestone.
- The `execution_id` idempotency implementation in Issue #8 must be complete before any auto-execution work in Issues #12–#13.
- Canary mode (Issue #12) must be run in production for a minimum observation period before Issue #13 is enabled; this gate is a rollout process control, not a code gate.
- Issue #14 requires a PostgreSQL instance; it is safely skippable if that dependency is not available in the deployment environment.
- Milestone 5 issues (#16–#22) are independent of each other except #21 (preview depends on #18 for the immediate-activation path) and can be opened and merged in any order.

---

## Milestone 5 — Discovery Pipeline Improvements (Issues #16–#22)

The original Issue #15 (lambda rule review/approval UI) was implemented and subsequently removed (PR #285) because lambda rules were detection-only, invisible from the review UI, and provided no path to full autonomy. Milestone 5 addresses the underlying need properly: making the discovery pipeline fast, transparent, and capable of producing rules that can trigger autonomous action.

**Issue #16 — Service-mapped suggested actions** *(Open — GitHub TBD)*

- Plan coverage: Section 4 (auto-execute guardrails), Section 3 (Level 2/3 autonomy)
- Scope: Update `normalize_candidate()` in `proposal_templates.py` to produce HA service calls (e.g. `lock.lock`) as `suggested_actions` for templates where a safe, deterministic action exists. `is_sensitive=True` templates remain blocked from auto-execute by execution guardrail #3 regardless of suggested actions. Templates with no safe automated action continue to produce advisory text only.
- Files: `sentinel/proposal_templates.py`
- Tests: `normalize_candidate()` produces `domain.service` format for applicable lock/entry templates; `is_sensitive=True` templates still blocked at execution guardrail; advisory-only templates produce no service-type actions; existing auto-execute integration tests pass unchanged.
- Size: S
- Dependencies: none
- **This is the single highest-value change for full autonomy — without it, no discovered rule can ever trigger auto-execute.**

**Issue #17 — On-demand discovery trigger** *(Open — GitHub TBD)*

- Plan coverage: Section 16 (discovery latency)
- Scope: Add `trigger_sentinel_discovery` HA service that runs the LLM discovery cycle against the current snapshot immediately, bypassing the periodic timer. Deduplication and semantic key filtering still apply. Periodic timer is unaffected.
- Files: `sentinel/discovery_engine.py`, `__init__.py`, `services.yaml`
- Tests: Service call triggers a discovery run and stores results; duplicate candidates are filtered; timer interval unchanged.
- Size: S
- Dependencies: none

**Issue #18 — Immediate rule activation on approval** *(Open — GitHub TBD)*

- Plan coverage: Section 16 (approval latency)
- Scope: After `approve_rule_proposal` successfully adds a rule to `RuleRegistry`, trigger a single `_run_once()` against the current snapshot. Rule becomes live in seconds rather than waiting up to one hour for the next scheduled cycle.
- Files: `__init__.py` (approve handler), `sentinel/engine.py`
- Tests: Approval triggers an immediate evaluation run; new rule fires on current snapshot if conditions met; no double-run if engine is already mid-cycle.
- Size: S
- Dependencies: none

**Issue #19 — Explain normalization failures** *(Open — GitHub TBD)*

- Plan coverage: Section 16 (transparency)
- Scope: When `normalize_candidate()` returns `None`, return a structured reason to the caller (e.g. `no_matching_entity_types`, `unsupported_pattern`, `missing_required_entities`). Surface this reason in the `promote_discovery_candidate` and `approve_rule_proposal` service responses. When `promote` or `approve` returns `already_active`, include the covering rule ID and which entity IDs overlapped.
- Files: `sentinel/proposal_templates.py`, `__init__.py`
- Tests: Each normalization failure path returns a distinct reason code; `promote` and `approve` service responses include reason; `already_active` response includes covering rule ID and overlapping entities.
- Size: S
- Dependencies: none

**Issue #20 — Richer proposal draft notifications** *(Open — GitHub TBD)*

- Plan coverage: Section 9 (notification UX)
- Scope: Include `template_id`, `severity`, and `confidence` in the notification sent when a proposal draft is created. Example: *"New HIGH-severity proposal: alarm disarmed + entry open (80% confident) — call approve_rule_proposal to activate."*
- Files: `__init__.py` (promote handler)
- Tests: Notification payload includes template_id, severity, confidence, and actionable service call hint.
- Size: XS
- Dependencies: none

**Issue #21 — Rule preview before commit** *(Open — GitHub TBD)*

- Plan coverage: Section 16 (operator control)
- Scope: Add `preview_rule_proposal` HA service that evaluates the normalized rule spec against the current snapshot without writing to the registry. Returns whether the rule would trigger right now, and against which entities. Intended as a dry-run step before calling `approve_rule_proposal`.
- Files: `__init__.py`, `sentinel/dynamic_rules.py`, `services.yaml`
- Tests: Preview evaluates rule correctly; no registry mutation; returns trigger status and matching entities; behaves identically to a live evaluation for the same snapshot.
- Size: M
- Dependencies: Issue #18 (shares immediate-snapshot evaluation path)

**Issue #22 — PIN validation for autonomy level increase** *(Open — GitHub TBD)*

- Plan coverage: Section 3 (runtime kill switch and override lifecycle), Section 15 (Prerequisite 2)
- Scope: `sentinel_set_autonomy_level` accepts `pin` in the service call data but does not validate it — the check is an explicit stub (`pass`) in `engine.py`. Implement actual PIN hash validation: store PIN hash in config (not plaintext), compare on level increase when `CONF_SENTINEL_REQUIRE_PIN_FOR_LEVEL_INCREASE=true`. Level decreases and level 0 (kill switch) never require PIN.
- Files: `sentinel/engine.py`, `flows/sentinel_subentry_flow.py`, `const.py`, `strings.json`, `translations/en.json`
- Tests: Correct PIN allows increase; wrong PIN rejected; no PIN required for decrease; kill-switch (`level=0`) never gated; PIN stored as hash not plaintext.
- Size: M
- Dependencies: none

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

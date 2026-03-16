# TODOS

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

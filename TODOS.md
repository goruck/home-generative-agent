# TODOS

## Sentinel

### Remove `_supports_suppression_reason_code()` introspection shim

**What:** Delete the `_supports_suppression_reason_code()` helper in `sentinel/engine.py` and inline the direct call to `async_append_finding` at all `_append_finding_audit` callsites.

**Why:** The shim's `else`-branch (line ~961) is dead code — `AuditStore.async_append_finding` has had the full v2 signature since Issue #3 (GitHub #254). The introspection adds ~20 lines of complexity and a `cast("Any", ...)` bypass that defeats type checking.

**Context:** The shim was introduced to allow progressive migration of the audit store signature. Now that the store is fully v2 and all callsites pass the extended kwargs, the shim is pure overhead. Removing it also removes the `cast("Any", audit_store)` call that suppresses pyright warnings. Start in `sentinel/engine.py` around the `_append_finding_audit` helper and `_supports_suppression_reason_code` function.

**Effort:** S
**Priority:** P3
**Depends on:** Audit integrity gaps PR (closes gaps 1–5)

---

### Fix `people_home`/`people_away` to use stable entity IDs

**What:** Change `snapshot/derived.py` to populate `people_home` and `people_away` with `state.entity_id` instead of `state.attributes.get("friendly_name") or state.entity_id`. Add a `SuppressionState` key migration for `presence_grace_until` entries that were written using display-name-derived keys.

**Why:** Presence-grace window keys in `SuppressionState.presence_grace_until` are currently keyed by display name (e.g. `"Alice"`) rather than entity ID (e.g. `"person.alice"`). If a user renames a person in HA, the grace-window key becomes stale and the window silently stops working. Rule conditions that pattern-match on people lists also behave inconsistently.

**Context:** This is Gap 6 from the Known Current Gaps section of `sentinel_plan.md`. The fix in `derived.py` is one line, but it invalidates any active grace-window entries written before the upgrade. The suppression store must migrate `presence_grace_until` keys: any key that doesn't look like a `person.` entity ID should be dropped (best-effort) and logged as a migration warning. Also update `_update_presence_grace` in `engine.py` to use entity IDs, and update snapshot schema comments. Add tests for all occupancy states with and without friendly names set.

**Effort:** M
**Priority:** P2
**Depends on:** None

---

### Build Operational Health Entity (sentinel_plan.md §11)

**What:** Register a Home Assistant sensor entity that exposes Sentinel operational KPIs as attributes: `last_run_start`, `last_run_end`, `run_duration_ms`, `trigger_source_stats`, `triggers_dropped`, `active_rule_count`, `findings_count_by_severity`, `triage_suppress_rate`, `auto_exec_count`, `auto_exec_failures`, `false_positive_rate_14d`, `action_success_rate`, `user_override_rate`.

**Why:** No health sensor entity exists anywhere in the codebase today (sentinel_plan.md §11 is entirely target state). Without it: (a) the L2→L3 KPI gate (false-positive rate < 5%, action success rate > 95%) is invisible to operators; (b) there is no way to know if Sentinel is running, how often, or why it suppressed something without reading raw audit records; (c) Lovelace dashboards and automations cannot consume Sentinel state.

**Context:** This is the highest-value post-audit-integrity work. It directly unblocks: L2→L3 transition gating, operational runbooks, and user-facing Lovelace dashboards. The audit integrity gaps (gaps 1–5) must be closed first — KPI calculations require complete audit data. Start in `sensor.py` (existing platform file) following the pattern of existing sensor registrations. Rolling KPI values (14d false-positive rate etc.) should be computed from `AuditStore.async_get_latest()` on a background timer. The health entity's state value should be `"ok"` / `"degraded"` / `"disabled"` mirroring `audit_archival_status`.

**Effort:** XL
**Priority:** P1
**Depends on:** Audit integrity gaps PR (closes gaps 1–5)

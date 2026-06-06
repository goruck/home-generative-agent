# Sentinel — Proactive Anomaly Detection

Sentinel watches your home continuously and sends you alerts when something looks wrong. It combines deterministic rules (always reliable, never requires LLM) with optional LLM-powered triage, baseline statistics, and discovery of new rule candidates.

**Safety invariant:** Deterministic rules always detect. The LLM never mutates findings or executes actions. Optional triage may only suppress notification *delivery* — it cannot alter any finding field — and it fails open (on error, the finding is always notified).

Sentinel is a singleton service per Home Generative Agent config entry. Configure exactly one Sentinel subentry.

- [Quick Setup](#quick-setup)
- [Pipeline Architecture](#pipeline-architecture)
- [Built-in Static Rules](#built-in-static-rules)
- [Notification Behavior](#notification-behavior)
- [LLM Triage (Optional)](#llm-triage-optional)
- [Baseline Collection (Optional)](#baseline-collection-optional)
- [Action Flows](#action-flows)
- [Blueprints](#blueprints)
- [Supported Generated Rule Templates](#supported-generated-rule-templates)
- [Discovery, Deduplication, and Proposals](#discovery-deduplication-and-proposals)
- [Services Reference](#services-reference)
- [Health Sensor](#health-sensor)
- [Proposals Dashboard Card](#proposals-dashboard-card)
- [Troubleshooting](#troubleshooting)

---

## Quick Setup

1. Open **Settings → Devices & Services → Home Generative Agent**.
2. Click **+ Sentinel**.
3. Configure runtime options (detection interval, cooldowns, notify service, autonomy level).
4. Optionally enable Discovery, Triage, and Baseline (see sections below).
5. Save. Sentinel starts on the next HA restart or integration reload.

### Autonomy Levels

The autonomy level controls how much Sentinel is allowed to do on your behalf:

| Level | Behaviour | Safeguards |
|---|---|---|
| `0` — Passive | Detection runs but no notifications or actions are taken | Full silence; useful for testing rules without alert fatigue |
| `1` — Notify only *(default)* | Mobile push and persistent HA notifications on findings | Triage (if enabled) may suppress low-value alerts; all actions are user-initiated |
| `2` / `3` — Auto-exec enabled | Notifications plus autonomous execution of allowed services when `sentinel_auto_execution_enabled = true`. Levels 2 and 3 apply identical execution gates in the current code. | Constrained by `min_confidence`, `max_actions_per_hour`, an explicit `allowed_services` allowlist, and idempotency window; canary mode available for dry-run validation. PIN protection for level increase via `sentinel_require_pin_for_level_increase`. |

Runtime overrides are TTL-bounded (`sentinel_runtime_override_ttl_minutes`, default 60 min) and revert to the configured level automatically.

---

## Pipeline Architecture

Each detection cycle runs through these stages in order:

1. **Snapshot** — builds an authoritative JSON representation of home state (entities, camera activity, derived context).
2. **Rules** — runs deterministic evaluation of all active static and dynamic rules against the snapshot.
3. **Triage** *(optional, autonomy ≥ 1)* — LLM pass that can suppress low-value alerts. Fails open: on error the finding is always notified.
4. **Notifier** — orchestrates mobile push and persistent HA notifications with snooze actions and per-area routing.
5. **Baseline** *(optional)* — background service writing rolling statistical summaries per entity; fires temporal anomaly findings.
6. **Discovery** *(optional)* — uses an LLM to suggest new rule candidates (advisory only; requires user approval).
7. **Audit** — persists findings and user action outcomes.

**Safety invariant:** Deterministic rules always detect. The LLM never mutates findings or executes actions. Triage may only suppress notification delivery; it cannot alter any finding field or gate execution, and it fails open.

**Admission control on edge deployments:** Sentinel LLM calls (triage, discovery, explain) defer when a chat turn or video model call is active, so the GPU is not contended during live interactions. Priority order: chat (highest) > video > Sentinel. Chat cancels in-flight Sentinel tasks on entry; video defers queued Sentinel tasks for its entire window. Sentinel resumes immediately once both foreground activities are idle. Cloud providers (OpenAI, Gemini, Anthropic) are always admitted — they use remote inference. When deferrals persist for more than 300 s, the `sentinel_health` sensor transitions to `degraded` and a WARNING is logged.

Set `model_provider_uncontended: true` in Options to bypass all local gates when the server has dedicated capacity.

---

## Built-in Static Rules

These rules run on every detection cycle without any configuration or approval.

**Security / presence**

| Rule | Description |
|---|---|
| `unlocked_lock_at_night` | Exterior lock unlocked while it is night |
| `open_entry_while_away` | Door or window open while everyone is away |
| `camera_entry_unsecured` | Camera activity detected near an unsecured entry point. Same-area entries detected automatically; cameras that overlook entries in a different area can be linked via `sentinel_camera_entry_links` |
| `unknown_person_camera_no_home` | Unrecognized person on any camera while no one is home |
| `unknown_person_camera_night_home` | Unrecognized person on any camera at night while someone is home |
| `alarm_disarmed_during_external_threat` | Security alarm disarmed while an unrecognized person is detected on an outdoor camera |

**Appliances / sensors**

| Rule | Description |
|---|---|
| `appliance_power_duration` | Appliance drawing power beyond a configurable duration threshold (e.g. *"Washer drew about 296 W for 633 min, above the 60 min threshold. Check it."*) |

**Cameras**

| Rule | Description |
|---|---|
| `vehicle_detected_near_camera_home` | Vehicle detected on any monitored camera while residents are home |
| `pet_detected_at_night_no_occupancy` | Pet detected on any monitored camera at night while no residents are home (informational; no suggested action) |
| `camera_missing_snapshot_night_home` | Any monitored camera (with active motion sensors) has no snapshot summary at night while the home is occupied |

**Devices**

| Rule | Description |
|---|---|
| `phone_battery_low_at_night_home` | Any phone battery sensor below 20% at night while someone is home |

Static rules are registered automatically at startup. They cannot be deactivated through the proposal flow.

---

## Notification Behavior

When Sentinel notifications are enabled:

- Mobile push text is compact and plain-language (targeted for small screens).
- Explanation text is normalized before send (markdown/backticks removed, whitespace collapsed).
- `appliance_power_duration`, `alarm_disarmed_during_external_threat`, and findings from `baseline_deviation` or `time_of_day_anomaly` rules always use deterministic message builders. Baseline deviation notifications include the measured value, historical average, and percent deviation (e.g. "Fridge: 4.6 W vs usual 85.0 W (95% below normal). Check appliance.") and a subtitle indicating direction ("Fridge: power lower than expected").
- For all other finding types, if explanation text is missing or too long, Sentinel falls back to a deterministic message.
- Fallback urgency wording depends on severity:
  - `high`: *"Urgent: check and secure it now."*
  - `medium`: *"Check soon and secure it if unexpected."*
  - `low`: *"Review when convenient."*
- For `is_sensitive` findings, recognized person names are replaced with `"a recognised person"` before send.

**Mobile action buttons** (primary action first):

| Button | Behavior |
|---|---|
| `Execute` | Non-sensitive findings with suggested actions. Calls the conversation agent or fires `hga_sentinel_execute_requested`. |
| `Ask Agent` | Sensitive findings with suggested actions. Hands the finding to the agent, which can verify a PIN or alarm code before acting. |
| `False Alarm` | Marks the alert as a false positive (`user_response.false_positive = true` in audit). |
| `Snooze 24 h` | Suppresses this finding type for 24 hours. |
| `Snooze Always` | Suppresses this finding type permanently. A confirmation notification is sent first; the snooze is written only after the user taps **Confirm**. |

**Per-area routing:** When `sentinel_area_notify_map` maps an area name to a notify service, findings whose triggering entities belong to that area are routed to that service instead of the global `notify_service`.

---

## LLM Triage (Optional)

When `sentinel_triage_enabled` is `true`, each finding passes through an LLM triage step before notification delivery (requires autonomy level ≥ 1). Triage can only suppress notification delivery — it cannot alter any finding field, and on timeout or error the decision is always `notify`.

- The triage prompt uses a restricted input allowlist — only sanitized fields are sent: `type`, `severity`, `confidence`, `is_sensitive`, `entity_count`, `suggested_actions_count`, and a small set of optional derived evidence (`is_night`, `anyone_home`, `recognized_people_count`, `last_changed_age_seconds`). Raw entity state values, attribute strings, area names, and free-form evidence text are never included.
- Triage returns a `decision` (`notify` or `suppress`) and a `reason_code` for audit.
- `triage_confidence` is recorded in the audit log but does not gate execution.
- Triage cannot alter any finding field — it can only gate the notification.
- Fails open: on timeout or error the decision becomes `notify` with `reason_code: triage_error`.

**Configuration** (in the Sentinel subentry):

| Option | Default | Description |
|---|---|---|
| `sentinel_triage_enabled` | `false` | Enable LLM triage |
| `sentinel_triage_timeout_seconds` | `10` | Max time to wait for triage LLM response |

---

## Baseline Collection (Optional)

When `sentinel_baseline_enabled` is `true` and `sentinel_enabled` is `true`, a background `SentinelBaselineUpdater` task writes rolling statistical summaries (per entity, per metric) to a `sentinel_baselines` PostgreSQL table on a configurable cadence.

On each detection cycle the engine reads current baseline values and passes them to dynamic-rule evaluators. Two temporal templates are always registered:

- **`baseline_deviation`** — fires when a numeric entity state deviates from its rolling average by more than `threshold_pct` percent (default `50.0`).
- **`time_of_day_anomaly`** — fires when a numeric entity state differs from the expected hour-of-day rolling average by more than `threshold_pct` percent (default `50.0`). When day-of-week baselines are enabled, this template uses a weighted blend of the DOW-hour mean and the global hourly mean, transitioning smoothly from global to DOW baselines as data accumulates.

Both templates produce no findings while the table is empty — baselines accumulate over time.

**Baseline freshness states:**

| State | Meaning |
|---|---|
| `fresh` | Baseline updated within the freshness threshold |
| `stale` | Baseline exists but is older than the freshness threshold |
| `unavailable` | No baseline record exists for this entity/metric |

**Configuration** (in the Sentinel subentry):

| Option | Default | Description |
|---|---|---|
| `sentinel_baseline_enabled` | `false` | Enable baseline collection |
| `sentinel_baseline_update_interval_minutes` | `15` | How often baselines are recalculated |
| `sentinel_baseline_freshness_threshold_seconds` | `3600` | Age after which a baseline is considered stale |
| `sentinel_baseline_min_samples` | `20` | Minimum samples before a global baseline is usable |
| `sentinel_baseline_max_samples` | `500` | Rolling window size; older samples discarded when reached |
| `sentinel_baseline_drift_threshold_pct` | `30.0` | Default percent deviation that triggers a finding |
| `sentinel_baseline_weekly_patterns` | `false` | Enable per-(day-of-week, hour) baselines for `time_of_day_anomaly` |
| `sentinel_baseline_dow_min_samples` | `4` | Observations per DOW-hour slot before blend weight reaches 1.0 |

---

## Action Flows

When a user taps an action button, Sentinel uses a two-tier dispatch strategy: it first attempts to call the HGA conversation agent directly via `conversation.process`; if no conversation entity is available it falls back to firing a Home Assistant event.

### Execute (non-sensitive findings)

1. **Agent available** — calls the conversation agent with a prompt that includes the detection timestamp, alert-time evidence snapshot, and suggested actions. The agent calls `GetLiveContext` to check whether the alert condition is still active (state may have changed since detection). If still active it executes the suggested action; if the condition has since resolved it explicitly acknowledges the original alert and confirms resolution. Its reply is pushed back as a mobile notification (when `notify_service` is configured).
2. **Agent unavailable** — fires `hga_sentinel_execute_requested` so a blueprint or automation can handle it.
3. **Sensitive finding** — blocked with status `blocked`.

### Ask Agent / Handoff (sensitive findings)

1. **Agent available** — calls the conversation agent with a security-focused prompt that includes the detection timestamp and alert-time evidence. The agent calls `GetLiveContext` to check whether the condition is still active before acting. It can verify a PIN or alarm code before executing. Its reply is pushed back as a mobile notification.
2. **Agent unavailable** — fires `hga_sentinel_ask_requested` (includes a `suggested_prompt` field) so a blueprint can route it to the agent.

### Event payloads

Both `hga_sentinel_execute_requested` and `hga_sentinel_ask_requested` include:

- `requested_at`, `anomaly_id`, `type`, `severity`, `confidence`
- `triggering_entities`, `suggested_actions`, `is_sensitive`, `evidence`
- `detected_at` — UTC ISO 8601 timestamp when the finding was first detected
- `mobile_action_payload`

`hga_sentinel_ask_requested` additionally includes `suggested_prompt` — a ready-to-use natural-language prompt for the conversation agent.

---

## Blueprints

Three draft blueprints are in the `blueprints/` folder:

| Blueprint | What it does |
|---|---|
| `hga_sentinel_execute_router.yaml` | Routes `hga_sentinel_execute_requested` by `suggested_actions` to scripts (`arm_alarm`, `check_appliance`, `check_camera`, `check_sensor`, `close_entry`, `lock_entity`) with default fallback support |
| `hga_sentinel_execute_escalate_high.yaml` | Handles only `severity: high` execute events; can send persistent notifications, mobile push, and optional TTS |
| `hga_sentinel_ask_router.yaml` | Routes `hga_sentinel_ask_requested` events to the HGA conversation agent; the agent receives `suggested_prompt`, can verify a PIN, and sends its response back as a notification |

**Recommended order:**
1. Start with `hga_sentinel_execute_escalate_high.yaml` for immediate high-priority visibility.
2. Add `hga_sentinel_execute_router.yaml` when you have scripts ready for action-specific handling.
3. Add `hga_sentinel_ask_router.yaml` as a fallback for sensitive findings.

**Importing blueprints:**
1. Open **Settings → Automations & Scenes → Blueprints**.
2. Import each YAML from the `blueprints/` directory.
3. Create automations from the imported blueprints and configure inputs.

**Script contract for router targets:**

Router script calls pass one object in `data.sentinel_event`. The object matches the execute event payload and includes `requested_at`, `anomaly_id`, `type`, `severity`, `confidence`, `triggering_entities`, `suggested_actions`, `is_sensitive`, `evidence`, and `mobile_action_payload`.

Store scripts in **Settings → Automations & Scenes → Scripts** (or `scripts.yaml`). Assign stable script entity IDs (e.g. `script.hga_check_camera_flow`) so they can be selected in the router blueprint.

<details>
<summary>Example script targets</summary>

**check_appliance:**
```yaml
alias: HGA Check Appliance Flow
mode: queued
fields:
  sentinel_event:
    description: Sentinel execute event payload
sequence:
  - action: persistent_notification.create
    data:
      title: "HGA Appliance Follow-up"
      message: >
        Type={{ sentinel_event.type }},
        severity={{ sentinel_event.severity }},
        entities={{ sentinel_event.triggering_entities | join(', ') }}.
  - action: notify.mobile_app_phone
    data:
      title: "HGA Appliance Follow-up"
      message: >
        Suggested actions:
        {{ sentinel_event.suggested_actions | join(', ') if sentinel_event.suggested_actions else 'none' }}
```

**check_camera:**
```yaml
alias: HGA Check Camera Flow
mode: queued
fields:
  sentinel_event:
    description: Sentinel execute event payload
sequence:
  - action: notify.mobile_app_phone
    data:
      title: "HGA Camera Follow-up"
      message: >
        Camera-related event {{ sentinel_event.type }}.
        Entities={{ sentinel_event.triggering_entities | join(', ') if sentinel_event.triggering_entities else 'none' }}.
  - action: persistent_notification.create
    data:
      title: "HGA Camera Follow-up"
      message: >
        Evidence: {{ sentinel_event.evidence }}
```

**lock_entity:**
```yaml
alias: HGA Lock Entity Follow-up
mode: queued
fields:
  sentinel_event:
    description: Sentinel execute event payload
sequence:
  - variables:
      lock_id: >
        {% set ids = sentinel_event.triggering_entities | default([], true) %}
        {{ ids[0] if ids else '' }}
  - choose:
      - conditions:
          - condition: template
            value_template: "{{ lock_id.startswith('lock.') }}"
        sequence:
          - action: lock.lock
            target:
              entity_id: "{{ lock_id }}"
    default:
      - action: persistent_notification.create
        data:
          title: "HGA Lock Entity Follow-up"
          message: >
            Could not resolve lock entity from event:
            {{ sentinel_event.triggering_entities | default([], true) }}
```

Tip: if your script needs the raw mobile action callback details, read `sentinel_event.mobile_action_payload`.
</details>

---

## Supported Generated Rule Templates

Rules generated by Discovery and activated via `approve_rule_proposal` must map to one of these templates.

**Security / presence**

| Template | Description |
|---|---|
| `unlocked_lock_when_home` | Lock unlocked while someone is home |
| `unlocked_lock_while_away` | Lock unlocked while no one is home |
| `alarm_disarmed_open_entry` | Alarm disarmed with an entry sensor open |
| `alarm_state_mismatch` | Alarm in a state that contradicts current or expected occupancy. `armed_home` and `armed_night` with `expected_presence=home` are never treated as a mismatch |
| `open_entry_when_home` | Entry open while someone is home |
| `open_entry_while_away` | Entry open while away |
| `open_entry_at_night_when_home` | Entry open at night while home |
| `open_entry_at_night_while_away` | Entry open at night while away |
| `open_any_window_at_night_while_away` | Any window open at night while away |
| `multiple_entries_open_count` | N or more entries open simultaneously |
| `unknown_person_camera_no_home` | Unrecognized person on camera while away |
| `unknown_person_camera_when_home` | Unrecognized person on camera while home |
| `motion_detected_at_night_while_alarm_disarmed` | Motion at night with alarm disarmed |
| `motion_without_camera_activity` | Motion sensor active without corresponding camera activity |
| `motion_while_alarm_disarmed_and_home_present` | Motion with alarm disarmed and person home |

**Duration / staleness**

| Template | Description |
|---|---|
| `entity_state_duration` | Lock or entry held in a state (e.g. unlocked, open) beyond a time threshold |
| `entity_staleness` | Person or sensor entity not updated within an expected window |

**Sensors / appliances**

| Template | Description |
|---|---|
| `sensor_threshold_condition` | Numeric sensor exceeds a threshold, with optional night/away/home condition |
| `low_battery_sensors` | Battery sensor at or below a threshold |
| `unavailable_sensors` | Sensors in `unavailable` state |
| `unavailable_sensors_while_home` | Sensors in `unavailable` state while someone is home |

**Baseline / temporal**

| Template | Description |
|---|---|
| `baseline_deviation` | Numeric entity deviates from its rolling average |
| `time_of_day_anomaly` | Numeric entity differs from expected hour-of-day rolling average |

---

## Discovery, Deduplication, and Proposals

### How discovery works

Discovery uses an LLM to suggest new rule candidates based on the current home snapshot. Candidates are:

1. Deduplicated against active static and dynamic rules, existing proposal drafts (including rejected), and recent discovery records.
2. Stored as proposal drafts with status `draft`.
3. Reviewed by the user (promote → approve or reject).
4. When approved and successfully mapped to a supported template, the dynamic rule is registered and an immediate evaluation cycle runs.

### Novelty checks

Candidates with a resolvable subject and predicate are matched by a deterministic semantic key (`v1|subject=...|predicate=...|...`). Candidates without a resolvable subject/predicate fall back to a title+summary hash (`ident|sha256=...`).

**Hint keys vs. filter keys:** The engine maintains two distinct key sets each cycle:

- **`hint_keys`** — sent to the LLM as "already covered" topics so it avoids proposing duplicates. Contains active rule keys and keys from `pending` or `rejected` proposals only. Approved proposals are intentionally excluded: their coverage is tracked by the live rule via `rule_semantic_key`. If the user later disables the rule, the topic becomes re-proposable.
- **`filter_keys`** — used for post-hoc deduplication of LLM output. Superset of `hint_keys`, plus keys from historical discovery records.

**Baseline coverage check:** When computing which baseline-ready entities lack monitoring (the "monitoring gap" surfaced to the LLM), only keys with a `|template=baseline_deviation|` or `|template=time_of_day_anomaly|` marker are counted as coverage. These markers are emitted exclusively by `rule_semantic_key()` for live statistical monitoring rules. Candidate keys (from pending, rejected, or historical proposals) never contain `|template=…|`, so rejected bundle proposals listing many entities cannot accidentally suppress individual per-entity proposals.

Discovery records include:
- `semantic_key` — canonical normalized key
- `dedupe_reason` — `novel`, `existing_semantic_key`, `existing_identity_hash`, or `batch_duplicate`
- `filtered_candidates` — candidates removed by dedupe with their reason

### Proposal lifecycle

| Status | Meaning |
|---|---|
| `draft` | Candidate stored, awaiting review |
| `approved` | Mapped to a template and registered as a dynamic rule |
| `rejected` | Excluded from all future discovery cycles |
| `unsupported` | Could not be mapped to a supported template |
| `covered_by_existing_rule` | Semantically covered by an active rule; `covered_rule_id` and `overlapping_entity_ids` attached |

**Dry-run before approval:** Use `preview_rule_proposal` to evaluate the normalized rule against the current snapshot without registering it.

**Unsupported proposals:** If a candidate cannot be mapped, preferred handling:
1. Reject if not useful (excluded from future cycles).
2. If useful, request a new template via `.github/ISSUE_TEMPLATE/feature_rule_request.yml` — the proposals card pre-populates relevant fields.
3. After template support is added, re-approve to re-evaluate.

Unsupported proposals never acted on are automatically removed after 30 days.

### Normalization fallbacks

| Pattern | Fallback |
|---|---|
| Power sensor for a cyclical load (`fridge`, `refrigerator`, `freezer`, `compressor`) | Routes to `time_of_day_anomaly` — variance-aware per-hour threshold tolerates compressor cycling |
| Power sensor without numeric threshold (non-cyclical) | Routes to `baseline_deviation` |
| Cumulative energy sensor (`*_energy`) | Rejected with `reason_code="cumulative_energy_sensor"` — monotonically increasing counters cannot be baselined this way |
| Lock battery low | Routes to `low_battery_sensors` |
| Alarm disarmed with no occupancy signal | `alarm_state_mismatch` with `expected_presence="home"` |
| `armed_home` / `armed_night` with home presence | Rejected — these modes are designed for occupancy |
| Window/entry open duration without entity IDs | Falls back to `open_any_window_at_night_while_away` |

### Configuring Discovery

In the Sentinel subentry:

| Option | Description |
|---|---|
| `sentinel_discovery_enabled` | Enable LLM discovery |
| `sentinel_discovery_interval_seconds` | How often discovery runs |
| `sentinel_discovery_max_records` | Maximum proposal records to keep |
| `sentinel_daily_digest_enabled` | Enable daily push summary of the past 24 hours |
| `sentinel_daily_digest_time` | Delivery time (default `08:00:00`) |
| `sentinel_require_pin_for_level_increase` | Require PIN to increase autonomy level |
| `sentinel_area_notify_map` | Area name → notify service (e.g. `{"Garage": "notify.mobile_app_garage_tablet"}`) |
| `sentinel_camera_entry_links` | Camera entity ID → list of entry/lock entity IDs in other areas (e.g. `{"camera.driveway": ["lock.front_door"]}`) |
| `audit_hot_max_records` | Max records in local hot store (default 500) |

Discovery and baseline both require `sentinel_enabled` to be `true`. Discovery and triage also require a configured chat model.

---

## Services Reference

| Service | Description |
|---|---|
| `home_generative_agent.get_discovery_records` | Fetch discovery records |
| `home_generative_agent.trigger_sentinel_discovery` | Run one discovery cycle immediately |
| `home_generative_agent.promote_discovery_candidate` | Promote a candidate to a proposal draft |
| `home_generative_agent.get_proposal_drafts` | Fetch proposal drafts |
| `home_generative_agent.preview_rule_proposal` | Dry-run evaluate a proposal against current snapshot |
| `home_generative_agent.approve_rule_proposal` | Approve and register a proposal |
| `home_generative_agent.reject_rule_proposal` | Reject a proposal |
| `home_generative_agent.get_dynamic_rules` | List active dynamic rules |
| `home_generative_agent.deactivate_dynamic_rule` | Deactivate a dynamic rule |
| `home_generative_agent.reactivate_dynamic_rule` | Reactivate a dynamic rule |
| `home_generative_agent.get_audit_records` | Fetch audit records |
| `home_generative_agent.sentinel_set_autonomy_level` | Admin-only; applies a TTL-bounded runtime override. Requires PIN if `sentinel_require_pin_for_level_increase` is enabled. |
| `home_generative_agent.sentinel_get_baselines` | Return raw baseline statistics for all tracked entities |
| `home_generative_agent.sentinel_reset_baseline` | Delete baseline data for one entity or all (omit `entity_id` to reset all) |

Common response fields: `status`, `candidate_id`, `rule_id`, `template_id`, `covered_rule_id`, `reason_code`, `details`, `would_trigger`, `matching_entity_ids`, `findings`, `overlapping_entity_ids`, `records`, `enabled`.

---

## Health Sensor

Sentinel registers `sensor.sentinel_health`, updated after every detection run.

| State | Meaning |
|---|---|
| `ok` | Sentinel enabled and running normally |
| `degraded` | LLM calls consecutively deferred for more than 300 s (edge deployment under sustained load) |
| `disabled` | Sentinel not configured |

<details>
<summary>Full attribute reference</summary>

| Attribute | Description |
|---|---|
| `last_run_start` | UTC ISO 8601 timestamp when the last run started |
| `last_run_end` | UTC ISO 8601 timestamp when the last run ended |
| `run_duration_ms` | Last run duration in milliseconds |
| `active_rule_count` | Number of active rules evaluated |
| `sentinel_admission_degraded` | `true` when LLM calls deferred for more than 300 s |
| `sentinel_admission_degraded_category` | Category (`triage`, `discovery`, `explain`) that last triggered degraded, or `null` |
| `sentinel_admission_consecutive_deferrals` | Consecutive times LLM admission was denied |
| `sentinel_admission_starved_for_s` | Seconds since the last successful LLM run |
| `trigger_source_breakdown` | Rolling 24-hour counts: `{poll: N, event: N, on_demand: N}` |
| `discovery_candidates_generated` | Total candidates returned by LLM in the most recent cycle |
| `discovery_candidates_novel` | Candidates that passed deduplication |
| `discovery_candidates_deduplicated` | Candidates dropped as duplicates |
| `discovery_proposals_approved_24h` | Proposals approved by the operator in the past 24 hours |
| `discovery_unsupported_ttl_expired` | Unsupported proposals whose TTL expired |
| `triggers_coalesced` | Cumulative events merged into an existing queued trigger |
| `triggers_dropped_incoming` | Cumulative triggers dropped on arrival |
| `triggers_dropped_queued` | Cumulative lower-priority queued triggers evicted |
| `triggers_ttl_expired` | Cumulative queued triggers discarded past TTL |
| `findings_count_by_severity` | Count by severity: `{low: N, medium: N, high: N}` |
| `triage_suppress_rate` | Percentage of triaged findings suppressed (null if none triaged) |
| `auto_exec_count` | Number of autonomous execution attempts |
| `auto_exec_failures` | Number of autonomous execution errors |
| `action_success_rate` | Overall action success rate (null if no actions recorded) |
| `user_override_rate` | Percentage of user-visible findings that received a user response |
| `false_positive_rate_14d` | Percentage of user-visible findings in the last 14 days marked false positive |
| `baseline_entity_count` | Total entities with at least one baseline record |
| `baseline_fresh_count` | Entities whose baseline was updated within the freshness threshold |
| `baseline_stale_count` | Entities whose baseline exists but is older than the freshness threshold |
| `baseline_rules_waiting` | Active baseline rules whose entity has not yet reached `sentinel_baseline_min_samples` |
| `baseline_last_update` | UTC ISO 8601 timestamp of the most recent baseline write |
| `learned_suppressions_active` | Distinct `{rule_type}:{entity_id}` pairs with learned cooldown multipliers |

</details>

**Example Lovelace Markdown card:**

```yaml
type: markdown
content: |
  ## Sentinel Health
  **State:** {{ states('sensor.sentinel_health') }}

  **Last run:** {{ state_attr('sensor.sentinel_health', 'last_run_start') or '—' }}
  **Duration:** {{ state_attr('sensor.sentinel_health', 'run_duration_ms') or '—' }} ms
  **Active rules:** {{ state_attr('sensor.sentinel_health', 'active_rule_count') or '—' }}

  **Trigger drops (incoming / queued / TTL):**
  {{ state_attr('sensor.sentinel_health', 'triggers_dropped_incoming') or 0 }} /
  {{ state_attr('sensor.sentinel_health', 'triggers_dropped_queued') or 0 }} /
  {{ state_attr('sensor.sentinel_health', 'triggers_ttl_expired') or 0 }}

  **Findings (low / med / high):**
  {% set sev = state_attr('sensor.sentinel_health', 'findings_count_by_severity') or {} %}
  {{ sev.get('low', 0) }} / {{ sev.get('medium', 0) }} / {{ sev.get('high', 0) }}

  **Action success rate:** {{ state_attr('sensor.sentinel_health', 'action_success_rate') or '—' }}%
  **False positive rate (14d):** {{ state_attr('sensor.sentinel_health', 'false_positive_rate_14d') or '—' }}%

  **Baselines (fresh / stale / total):**
  {{ state_attr('sensor.sentinel_health', 'baseline_fresh_count') or 0 }} /
  {{ state_attr('sensor.sentinel_health', 'baseline_stale_count') or 0 }} /
  {{ state_attr('sensor.sentinel_health', 'baseline_entity_count') or 0 }}
  {% if state_attr('sensor.sentinel_health', 'baseline_rules_waiting') %}
  **Rules waiting for baseline:** {{ state_attr('sensor.sentinel_health', 'baseline_rules_waiting') }}
  {% endif %}
  **Baseline last updated:** {{ state_attr('sensor.sentinel_health', 'baseline_last_update') or '—' }}
```

---

## Proposals Dashboard Card

If you install `hga-proposals-card.js`, the card drives the full review flow:

- Discovery candidates (with dedupe reasons)
- Proposal drafts (pending)
- Proposal history
- Promote to draft, preview before approval, approve/reject
- Deactivate/reactivate controls for approved rules
- "Request New Template" shortcut that opens a prefilled GitHub issue form

**Installation:**
1. Go to **Settings → Dashboards → Resources → Add Resource**.
2. Add URL: `/hga-card/hga-proposals-card.js`, Type: `JavaScript Module`.
3. Add the card to a dashboard:

```yaml
type: custom:hga-proposals-card
title: Sentinel Proposals
```

> If the card shows as unknown after adding the resource, hard-refresh the browser. Legacy resource URLs under `/hga-enroll-card/...` still work for backward compatibility. When updating card JS, bump the Lovelace resource query string (e.g. `?v=12`) to avoid stale browser cache.

---

## Troubleshooting

- **Card UI looks unchanged after update** — you are likely serving cached JS. Bump the query string on the Lovelace resource URL.
- **Similar candidates keep reappearing** — inspect `dedupe_reason` and `filtered_candidates` in discovery records.
- **Proposal appears duplicate** — check logs for `Rule registry ignored duplicate rule ...` or `... covered_by_existing_rule ...`.
- **Existing stored proposal drafts** — statuses update when proposals are re-processed; they are not auto-migrated.
- **llama-server embedding incompatibility** — If you use llama-server as an OpenAI-compatible provider and see `Memory semantic search failed — embedding endpoint returned an incompatible response` in the logs, the agent has fallen back to recency-based memory retrieval. llama-server's `/v1/embeddings` response format does not match the OpenAI SDK's expected structure. For reliable semantic embeddings, use a dedicated embedding model via Ollama (`mxbai-embed-large` is recommended) and set its provider as the **Embedding** provider. Chat and embedding providers can be different — e.g. llama-server for chat, Ollama for embeddings.

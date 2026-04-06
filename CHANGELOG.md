# Changelog

All notable changes to this project will be documented in this file.

## [3.9.0] - 2026-04-06

### Added

- **Day-of-week (DOW) baselines** — Sentinel's time-of-day anomaly detector now
  supports per-(DOW, hour) baselines in addition to the existing global hourly
  averages. Enable via `sentinel_baseline_weekly_patterns` in the Sentinel subentry.
  Uses Welford's online algorithm for exact running mean + variance per slot without
  storing all past values. Detection uses a weighted blend that transitions smoothly
  from global to DOW baselines as data accumulates (`w = min(count / dow_min_samples,
  1.0)`), so there is no hard cliff when switching. Entity-specific stddev thresholds
  prevent washer/dryer power sensors from triggering on the same absolute deviation
  that would flag a door sensor. DOW buckets use local time so weekend/weekday
  schedules align with the user's actual timezone.

- **DOW min-samples config** — New `sentinel_baseline_dow_min_samples` option
  (default 4 weeks) controls how many observations per DOW-hour slot are required
  before the blend weight reaches 1.0. Configurable via a NumberSelector in the
  Sentinel subentry UI. Intentionally lower than the global `sentinel_baseline_min_samples`
  (20) because DOW slots update at most once per week per entity.

- **`learned_suppressions_active` health sensor attribute** — Exposes the count of
  distinct `{rule_type}:{entity_id}` pairs that have accumulated learned cooldown
  multipliers via snooze/dismiss feedback. Makes the feedback-learning loop
  observable in the health dashboard.

- **DB performance index** — `CREATE INDEX IF NOT EXISTS idx_sentinel_baselines_entity_metric`
  added to `async_initialize()` for fast per-entity/metric lookups. Idempotent;
  safe on existing installations.

## [3.8.0] - 2026-04-05

### Added

- **Cross-area camera entry links** — Sentinel's camera-entry-unsecured rule can now
  detect activity on cameras that physically overlook entry points in a different Home
  Assistant area (e.g. a driveway camera watching the front door). Configure the mapping
  via `sentinel_camera_entry_links` in the Sentinel subentry: a JSON object of camera
  entity IDs to lists of entry/lock entity IDs. Same-area unsecured entries are still
  detected automatically; cross-area links are purely additive. When both same-area and
  linked entries are unsecured, all are reported in a single finding (deduplicated).
  The anomaly suppression ID hashes only same-area entities, so changing link config
  does not invalidate existing suppression state.

## [3.7.2] - 2026-04-05

### Changed

- **Multi-select LLM API support** — The "Control Home Assistant" option in the options
  flow is now a multi-select, allowing multiple LLM APIs to be active simultaneously (e.g.
  combining the built-in Assist API with one or more MCP server integrations). Existing
  single-API configurations migrate automatically; the behaviour is identical when only one
  API is selected.

### Fixed

- **Crash when no APIs configured** — `_async_handle_message` no longer raises `KeyError`
  when `CONF_LLM_HASS_API` is absent from options (i.e. the user deselected all APIs).
  The agent now runs cleanly with only its built-in LangChain tools in that state.

- **Migration respects "no control" intent** — The v5→v6 config-entry migration converts
  the legacy `"none"` sentinel and absent key to an empty list rather than silently
  enabling the Assist API, preserving the operator's prior choice.

## [3.7.1] - 2026-04-04

### Added

- **Daily digest config flow UI** — The Sentinel subentry options flow now exposes a
  `Boolean` toggle to enable/disable the daily digest and a `TimeSelector` to set the
  delivery time, so operators can configure the digest without editing raw options.

- **Proposals approved (24 h) health attribute** — `SentinelHealthSensor` gains a
  `discovery_proposals_approved_24h` attribute counting how many discovery proposals
  were approved by the operator in the past 24 hours, giving visibility into how active
  the rule-learning feedback loop is.

- **Feedback-trained cooldown wired for snooze and dismiss** — `record_cooldown_feedback`
  is now called from both the snooze action (notifier) and the dismiss action (actions),
  completing the feedback loop so per-entity cooldown multipliers actually accumulate.

### Fixed

- **Daily digest time parse with `"HH:MM:SS"` format** — `sentinel/notifier.py` used
  `split(":", 1)` which parsed `"08:00:00"` as `("08", "00:00")`, silently discarding
  the configured time and always scheduling the digest at 08:00. Fixed to
  `split(":")` so any user-configured time takes effect.

- **Missing `stored_version = 5` in suppression state migration** — The v4→v5 migration
  block did not update the `stored_version` sentinel variable, meaning a future v6
  migration block would have re-applied v5 logic. Fixed.

- **Cooldown multiplier key scheme (`entity_id` → `rule_type:entity_id`)** — Multipliers
  are now keyed by `"{rule_type}:{entity_id}"` so different rules for the same entity
  accumulate independent multipliers. Includes a v5 schema migration that discards the
  old bare-key entries (safe: the feedback signal was never wired in production).

## [3.7.0] - 2026-04-03

### Added

- **Discovery quality metrics in health sensor** — `SentinelHealthSensor` now exposes
  `discovery_candidates_generated`, `discovery_candidates_novel`,
  `discovery_candidates_deduplicated`, `discovery_proposals_promoted`, and
  `discovery_unsupported_ttl_expired` attributes, giving operators visibility into
  every LLM discovery cycle without opening logs.

- **Trigger-source breakdown (24 h)** — Health sensor gains a
  `trigger_source_breakdown` attribute with rolling 24-hour counts broken down by
  `poll`, `event`, and `on_demand` trigger sources so you can see at a glance what
  is driving Sentinel activity.

- **Daily digest notification** — Sentinel can now send a scheduled push notification
  summarising findings from the past 24 hours. Controlled by the new
  `CONF_SENTINEL_DAILY_DIGEST_HOUR` option (default 07:00 local time). Duplicate
  digest triggers within a session are safely de-duplicated.

- **Centralised action codes** — All Sentinel mobile-action button identifiers
  (`ACTION_CODES`) are now declared once in `const.py` and imported by both the
  notifier and execution service, eliminating the risk of copy-paste divergence.

- **Learned per-entity cooldown multipliers** — Suppression now stores a
  per-entity feedback table (schema v4) that can hold learned multipliers. The data
  model and upgrade path are in place; the feedback signal will be wired to a user
  action in a follow-up sprint.

- **Camera-area spatial ejection in correlator** — `_eject_camera_area_violations`
  post-grouping pass prevents off-area findings from being bundled into a
  `CompoundFinding` by transitive union-find bridging. A finding in *Garage* can no
  longer appear in a compound alert anchored to the *Front* camera.

### Fixed

- **Empty `evidence_paths` bypass in discovery engine** — Candidates with an empty
  `evidence_paths` list (`[]`) were silently passing the derived-only filter because
  the old `if candidate.get("evidence_paths") and all(...)` guard short-circuits on
  falsy empty lists. Guard is now `is not None and len(...) > 0`.

- **Dead code in correlator area filter** — `camera_areas` set subtraction included
  `None` (`- {None, ""}`) but `_area_of()` always returns `str`, making the `None`
  exclusion a no-op. Simplified to `- {""}`.

- **Orphaned daily-digest tasks** — `_async_send_daily_digest` now checks whether a
  prior digest task is still running before scheduling a new one, preventing orphaned
  coroutines when the time-change callback fires more than once before the digest
  completes.

### Changed

- Correlator `_COMPLEMENTARY_PAIRS_REQUIRE_AREA_MATCH` guard extended so that
  spatially-bound pairs (e.g. `unlocked_lock_at_night` + `camera_entry_unsecured`)
  only correlate when they share the same non-empty area, reducing spurious compound
  alerts across different physical zones.

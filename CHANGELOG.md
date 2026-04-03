# Changelog

All notable changes to this project will be documented in this file.

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

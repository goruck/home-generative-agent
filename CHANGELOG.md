# Changelog

All notable changes to this project will be documented in this file.

## [3.19.0] - 2026-07-19

### Added

- **Czech translation** — the entire configuration UI (setup flows, Sentinel options, error messages, selectors) is now available in Czech. The new `cs` translation has full key parity with the English source, verified by tests that also guard every translation file against key drift, placeholder mismatches, and formatting Home Assistant would reject. Contributed by [@hruba202](https://github.com/hruba202). Closes [#494](https://github.com/goruck/home-generative-agent/issues/494).

### Fixed

- **The Basic/Advanced setup step can now actually be localized** — the "Basic setup"/"Advanced setup" choice labels and the "already configured — Basic setup will overwrite your settings" warning on the Feature and Sentinel setup screens were hardcoded English strings in Python, so they stayed English in every language even though the title and description on the very same form translated correctly. The labels now go through Home Assistant's standard selector translation mechanism (each language's frontend shows its own labels), and the warning text is loaded from the integration's translation files following the server-configured language, falling back to English if a translation is missing. Reported, diagnosed, and drafted by [@hruba202](https://github.com/hruba202) in [#494](https://github.com/goruck/home-generative-agent/issues/494).

## [3.18.2] - 2026-07-19

### Fixed

- **Camera snapshots no longer pile up on disk when analysis fails or Home Assistant restarts** — snapshot cleanup is bookkeeping-driven (the analyzer tracks the last 200 files per camera and deletes the oldest), but that bookkeeping only happened after a batch finished analysis. Any batch that didn't finish — every frame erroring during a VLM outage, a summary timeout, an internal crash, frames dropped under backlog pressure, or frames evicted from a full event buffer — left its files on disk forever. Files are now tracked the moment they are captured, so nothing that happens downstream can leak them. On startup the analyzer also folds snapshots left over from before a restart into the same cleanup (previously they were orphaned permanently, since the bookkeeping lives in memory): only files the integration itself wrote (`snapshot_*.jpg`) are claimed — a photo you drop into a camera's snapshot folder yourself is never touched — and pre-existing files are treated as the oldest entries, so a backlog is cleaned before any new frame. The published `latest.jpg` and images attached to recent notifications remain protected from cleanup, exactly as before.
- **A stale reload can no longer run a second copy of the video analyzer** — if the integration was reloaded before Home Assistant finished starting, a leftover startup listener could start the previous analyzer instance alongside the new one (duplicate capture loops, and with the new cleanup, two bookkeepers deleting files independently). The listener is now cancelled when the integration unloads.

## [3.18.1] - 2026-07-18

### Fixed

- **The image attached to a camera alert now matches what the alert says** — since 3.18.0's repeated-scene suppression, the notification text describes only the frames with real activity (typically the first seconds of an event), but the attached reference image was still taken from the middle of the whole snapshot batch. For a "man walks down the stairs" alert, that was usually an empty porch several unchanged frames after he left. The reference image (and the camera's latest-event frame shown by the image entity) is now chosen from the frames that actually fed the summary, preferring a frame where a person was detected — a recognized face, or a caption that affirmatively mentions a person. The caption check is word-bounded, plural-aware, and negation-aware, so "no people visible", "no person is present", or "a manicured lawn" cannot masquerade as person frames, while "two men" counts. When identical captions are merged, the surviving image is the frame where the recognized person is actually identifiable, so an alert naming someone can no longer attach the frame from before they appeared. If frame selection ever fails internally, the alert still goes out with the previous middle-of-batch image rather than being dropped.

## [3.18.0] - 2026-07-18

### Added

- **Static camera scenes no longer get the full environment re-described on every frame** — when a frame shows the same static setting as the previous one (no people, no animals, nothing moved), the VLM now replies with a short `Scene unchanged.` sentinel instead of restating the porch, the railing, and the parked car for the tenth time. The analyzer detects the sentinel tolerantly (models drift from mandated phrasing, so variants like "The scene is unchanged." or "Nothing has changed." also count), but only replies that explicitly claim equality with the previous frame qualify — stillness-only phrases such as "No activity." never do, because a newly delivered package is "no activity" yet a changed scene. Qualifying frames are dropped from the summary input, so multi-frame summaries of quiet events stay anchored on the one real description instead of a pile of near-duplicates. The previous frame's *full* description always remains the comparison anchor — a sentinel never becomes context for later frames, so a slow real change (a package appearing, a car leaving) is still caught against actual scene content. If face recognition spots a person the VLM missed ("Unknown Person" or a known name), the frame is kept regardless, so an unrecognized visitor can never be suppressed by a no-change reply. Dropped frames are counted in a new per-camera `sentinel_dropped` metric and logged at debug level for diagnosability. A model that never emits the sentinel simply behaves exactly as before. Idea and prompt draft by [@hruba202](https://github.com/hruba202). Closes [#493](https://github.com/goruck/home-generative-agent/issues/493).

### Fixed

- **Frame-description deduplication now actually merges duplicates** — `dedupe_desc()` compared descriptions with their `t+<n>s.` timestamp prefix included, which differs on every frame, so byte-identical descriptions at different offsets never collapsed and the function was effectively inert. The prefix is now stripped before comparison; identities recognized only on a dropped duplicate frame are merged into the kept entry so no face result is lost, and the merge no longer mutates caller-owned face lists.
- **A failed VLM call can no longer leak its error text into summaries, notifications, or the vector store** — the literal `Error analyzing image with VLM model.` caption returned on a model failure was previously appended to the frame list like a real description (and, with dedupe now live, an all-error batch would have collapsed to that single caption and shipped it as the event's mobile notification). Error and empty captions are now skipped, matching how every other per-frame error path already behaves, and they can never become the previous-frame context injected into later VLM prompts; the one exception is a frame where face recognition detected a person — that frame is kept under a neutral caption ("A person is present; scene analysis unavailable.") so a failed VLM reply can no longer erase a real detection, while the raw error text stays out of camera event summaries and notifications.

## [3.17.1] - 2026-07-17

### Fixed

- **Battery Ring cameras no longer burn ~10 identical VLM calls per event** — ring-mqtt's `Auto` snapshot mode refreshes battery cameras' snapshot image only every 600 seconds (wired: 30 s), so the `event_select` capture window introduced in 3.16.0 batched near-byte-identical copies of one frame and analyzed each of them. The perceptual-hash duplicate filter is now always active for capture loops the `event_select` trigger started — no configuration needed, and the global `video_analyzer_uniqueness_enabled` option still governs all other cameras. The first frame of every window is always analyzed, so each event still produces a result; the forced filter survives a lagging motion sensor taking over the loop mid-window (common on battery hardware, where motion follows the `eventId` by seconds) and also engages when motion fires *first* and the `eventId` lands on an already-running loop. Skipped duplicates are registered with the normal snapshot-retention pruning, so they no longer accumulate on disk, and the frame hash is now computed off the event loop instead of stalling Home Assistant on a full JPEG decode per frame. Field data by [@andymcmanus](https://github.com/andymcmanus). Closes [#489](https://github.com/goruck/home-generative-agent/issues/489).
- **A frozen ring-mqtt interval snapshot no longer feeds days-old imagery to the VLM as if it were current** — the interval snapshot can silently stop updating on battery Ring cameras (observed in the field at 5 days stale, surviving real motion events), after which MQTT serves the retained frame indefinitely and every analysis "succeeds" against an image unrelated to the event. Before each capture, the analyzer now checks the ring-mqtt snapshot camera's `timestamp` attribute and skips capture when the retained frame is older than 30 minutes, logging a snapshot failure that names the ring-mqtt add-on restart workaround (escalating WARNING → ERROR via the existing issue-#464 machinery; recorded once per staleness episode and re-recorded hourly rather than every 3 seconds, and a freeze that recurs after an unobserved recovery is still reported). The guard is scoped strictly to ring-mqtt cameras — identified by an `event_select` sibling entity via naming, the motion→camera override map, or the device registry (covering renamed entities) — so cameras from other integrations that publish a `timestamp` attribute with different semantics are never suppressed, and values that are not plausible current epochs (non-numeric, millisecond-scale, far-future, or corrupt) are ignored rather than trusted or crashing the capture loop. Field diagnosis by [@andymcmanus](https://github.com/andymcmanus). Closes [#490](https://github.com/goruck/home-generative-agent/issues/490).

## [3.17.0] - 2026-07-16

### Added

- **Sentinel entity exclusions now silence event-driven triggering too, and accept glob patterns** — Sentinel treats every `camera.*` and `person.*` entity as security-relevant, so non-security uses of those domains (most notably `person_location` map-snapshot cameras, whose state flips on every GPS update) could flood the bounded trigger queue with security-critical wake-ups and crowd out real events — one report measured ~1,970 dropped triggers in 23 minutes with no actual security activity. Entities excluded via `sentinel_rule_entity_exclusions` (for their domain-mapped anomaly type, or under the `"*"` key) now stop waking the engine at all: their findings were already filtered, and their state changes no longer occupy trigger-queue slots. Detection for other rules involving an excluded entity continues on the polling cadence and via other entities' triggers — don't per-type-exclude a real security camera you want low-latency alerts from; the Sentinel guide documents the trade-off. Exclusion entries may now be fnmatch-style glob patterns (e.g. `camera.map_*`), so a fleet of per-person map cameras doesn't need listing one by one; patterns are precompiled and matching stays cheap on the per-event hot path. Suppressed wake-ups are observable: the Sentinel health sensor gains a `triggers_excluded` attribute, each suppression logs at debug level, and the engine logs a one-time notice at startup when exclusions are active so upgrading users learn the new behavior. Thanks [@hruba202](https://github.com/hruba202) for the precise diagnosis and proposed approach — credited as commit co-author. Closes [#481](https://github.com/goruck/home-generative-agent/issues/481).

### Changed

- **Exclusion entries are validated against match-everything mistakes** — with glob support, a stray `"*"` or `"*.*"` entry would silently match every entity and could disable the entire detection engine (findings *and* triggers) with nothing in the UI to show for it. Entries are restricted to the entity-ID alphabet plus the `*` and `?` wildcards — fnmatch character classes are rejected outright, because their syntax defeats literal-character checks (`[!.]*.*` reads as having literals yet matches every entity ID) — and every entry must contain a dot and at least one literal character. The settings form rejects offending entries (and over-long ones beyond 256 characters) with a validation error, and hand-edited `.storage` entries failing the rules are ignored at startup with a log warning — fail-closed, so the affected entity resumes alerting rather than going dark. The engine also warns when an exclusion *type key* contains glob characters, since type keys are matched exactly and such a key would be silently inert. Pre-existing stored entries that fail the new validation stop applying after upgrade.

## [3.16.0] - 2026-07-16

### Added

- **Battery-powered Ring cameras now trigger proactive video analysis reliably via ring-mqtt's `event_select` entity** — battery Ring hardware (e.g. Ring Battery Doorbell Plus) often doesn't push `binary_sensor.*_motion` promptly or at all over MQTT, so motion events were missed entirely unless users built workaround automations. The video analyzer now also listens for `eventId` attribute changes on `select.<camera>_event_select` entities, which ring-mqtt publishes for every Ring event. When the `eventId` changes, the camera is resolved through the same three-tier lookup as motion sensors (override map → device registry → name heuristics, including the `camera.<name>_snapshot` variant) and the standard motion snapshot loop starts — no configuration needed. Because `event_select` has no "event over" signal, the loop ends on a fixed 30-second window that each new `eventId` extends, capped at 5 minutes total so continuous events can't defer the analysis flush indefinitely. The window only ever governs loops it started: a loop the motion binary sensor started (or takes over by firing `on`) is immune to the timer and runs until motion returns to `off`, exactly as before, so hybrid cameras with both signals keep their full motion window. Retained-state replays are ignored — an `eventId` that appears when the entity leaves `unknown`/`unavailable` (HA restart, MQTT broker reconnect, ring-mqtt add-on restart) belongs to the previous event and no longer risks a spurious burst of snapshots on every camera at once. Thanks [@andymcmanus](https://github.com/andymcmanus) for identifying the `event_select` signal during #460 testing. Closes [#466](https://github.com/goruck/home-generative-agent/issues/466).

### Fixed

- **A camera leaving the `recording` state no longer flushes a motion-triggered analysis batch mid-event** — when a motion (or event_select) loop owned a camera whose entity also toggled through `recording`, the recording-exit handler flushed the held frames early, splitting one event into two analysis batches. The flush is now skipped while a motion loop owns the camera; the motion lifecycle flushes the whole window as one batch.
- **A crashed snapshot loop no longer strands its captured frames** — if the per-camera snapshot loop died unexpectedly (e.g. a filesystem error), frames it had already buffered were silently dropped or leaked into the next event's batch. The loop teardown now always flushes whatever was captured.

## [3.15.0] - 2026-07-15

### Added

- **Sentinel rules can now be tuned and scoped per entity from the settings UI** — two long-requested controls land in the Sentinel subentry's Advanced settings. First, a generic per-rule entity exclusion map (`sentinel_rule_entity_exclusions`): a JSON object mapping an anomaly type to a list of entity IDs (with `"*"` as a wildcard for all types) that the engine applies to every finding source — built-in static rules, approved dynamic rules, and baseline deviations — before correlation and dispatch. This is the supported way to stop, say, inverter air-conditioner power sensors from tripping `appliance_power_duration` during perfectly normal all-day compressor runs, while the rule stays live for ovens, irons, and anything genuinely left on; previously the only options were snoozing the entire finding type (losing the safety value for every appliance) or hiding the sensors. Second, the `appliance_power_duration` thresholds are now configurable: `sentinel_appliance_power_threshold_w` (default 100 W) and `sentinel_appliance_duration_min` (default 60 min) replace the previously hardcoded values, so households with legitimately long-running high-draw devices can raise the bar instead of suppressing alerts. Malformed exclusion values (hand-edited `.storage`, bad restore) are validated in the settings form and additionally degrade safely at runtime — a bad entry is dropped rather than ever disabling the detection engine. Documented in the Sentinel guide under Built-in Static Rules. Thanks [@andymcmanus](https://github.com/andymcmanus) for the detailed feature request with real-world Daikin inverter data. Closes [#462](https://github.com/goruck/home-generative-agent/issues/462).

## [3.14.35] - 2026-07-14

### Fixed

- **Appliance power-duration findings now measure how long the appliance has actually been drawing power** — the `appliance_power_duration` rule compared the current time against the sensor's `last_changed`, which Home Assistant only advances when the reported *value* changes. For steady or quantized readings (e.g. an estimated compressor power that sits on one value for hours) the reported `duration_min` was really "minutes since the number last changed", and for real-power sensors whose wattage fluctuates every few seconds the clock reset constantly, so the rule could stay silent even though the recorder-history enrichment added in 3.14.24 usually corrected the timestamp before rules ran. The rule now tracks the rising edge itself — remembering when each sensor was first observed at or above the power threshold across evaluation cycles — and reports exactly that observed running time, independent of recorder availability. This also removes a false-positive path where an appliance idling between 10 W and the 100 W threshold (so the recorder enrichment considered it "on" the whole time) could fire instantly on a brief spike with a duration of up to 30 days. Dropping below the threshold, going unavailable, or disappearing from the snapshot ends the episode and starts a fresh clock; after a restart an already-running appliance is re-detected once it has been observed above the threshold for the full duration. Repeated findings for one ongoing episode now also share one anomaly identity instead of minting a new one each evaluation cycle, so snoozing works for the whole episode. Thanks [@andymcmanus](https://github.com/andymcmanus) for the detailed report with audit-trail evidence. Closes [#461](https://github.com/goruck/home-generative-agent/issues/461).

## [3.14.34] - 2026-07-14

### Fixed

- **Camera snapshot failures are no longer silent or undiagnosable** — when a snapshot never landed on disk, the only evidence was a generic "Snapshot failed to appear on disk" warning: the `camera.snapshot` service was dispatched fire-and-forget, so the actual error (camera unavailable, write failure, disallowed path) vanished, and the analysis for that event was dropped with no metric or escalation. The service is now called blocking under a 15-second budget, producing three distinguishable diagnostics — the service call failed (logged with the underlying error), the call timed out (camera unresponsive), or the call succeeded but the file never appeared (points at a media mount/permissions problem). Failures count toward a new `snapshot_failures` field in the hourly per-camera metrics line, and three consecutive failures on the same camera escalate the log from WARNING to ERROR so a camera that stops producing snapshots is visible instead of quietly dark. A per-camera in-flight guard also prevents the 1.5s recording poll from stacking overlapping snapshot calls on a slow or wedged camera. The two root causes suspected in the original report were fixed in 3.14.30 (un-pulled Ollama VLM killing the snapshot worker) and 3.14.31 (startup race after reload); this closes the remaining observability gap. Thanks [@andymcmanus](https://github.com/andymcmanus) for the detailed report. Closes [#464](https://github.com/goruck/home-generative-agent/issues/464).

## [3.14.33] - 2026-07-14

### Fixed

- **Sentinel quiet hours can now be enabled from the UI** — the quiet-hours options (`sentinel_quiet_hours_start`, `sentinel_quiet_hours_end`, `sentinel_quiet_hours_severities`) were fully implemented in the suppression engine but never exposed in the Sentinel settings form, and the runtime option resolver didn't pass them through even if stored, so the feature was unreachable. The Sentinel subentry's Advanced settings now offer start/end hour dropdowns (with a Disabled option) and a severity multi-select (default: low only); setting only one of start/end or an out-of-range hour returns a form error instead of saving a broken window. While wiring this up, the newly reachable engine seam was hardened so a corrupted stored value (hand-edited `.storage`, bad restore) degrades to "quiet hours off" instead of raising inside the Sentinel run loop, unknown severity values are filtered on save and at runtime, an invalid timezone now logs a warning on the UTC fallback, and start = end is defined as a full 24-hour window (previously a silent no-op, and all-day suppression was inexpressible). Documented in the Sentinel guide under Notification Behavior. Thanks [@andymcmanus](https://github.com/andymcmanus) for the report. Closes [#463](https://github.com/goruck/home-generative-agent/issues/463).

## [3.14.32] - 2026-07-13

### Fixed

- **Sentinel Discovery no longer dies silently on LLM provider errors** — a provider error during the discovery LLM call (e.g. `ollama.ResponseError` when the configured model isn't pulled, or an httpx transport failure while the Ollama server restarts) escaped the narrow exception handling and permanently killed the discovery background loop; no proposals were ever produced again until an integration reload, with the error only surfacing if the task was cancelled mid-call at unload. Provider errors are now caught and logged (rate-limited, with recovery messages), and the discovery loop itself is additionally guarded so no unexpected error in any cycle stage can end it — mirroring the snapshot-worker resilience fix from #473. The same narrow-catch escape is fixed in LLM triage (which now truly fails open to notify on any exception, as documented) and LLM explanations (which degrade to the deterministic fallback text). Discovery audit payloads now also record the actually-configured model name (previously always `unknown` for bound models), making model misconfiguration diagnosable from the Sentinel audit trail. Note: discovery was already using the configured chat model — the model selection reported in the issue could not be reproduced, but the silent-failure mode it described is real and fixed. Closes [#465](https://github.com/goruck/home-generative-agent/issues/465).

## [3.14.31] - 2026-07-13

### Fixed

- **Video analyzer no longer delays Home Assistant startup when cameras are already active** — motion/recording snapshot loops could start during HGA config-entry setup, so Home Assistant counted long-lived `_motion_snapshot_loop()` tasks and in-flight `camera.snapshot` calls as startup work. On busy camera systems this produced bootstrap warnings and delayed startup until HA timed out waiting on those tasks. The video analyzer now starts after `EVENT_HOMEASSISTANT_STARTED`, and its snapshot loops/workers are created as HA background tasks so they do not block startup accounting. Runtime motion/recording behavior after startup is unchanged.

## [3.14.30] - 2026-07-13

### Fixed

- **Selecting an Ollama VLM that was never pulled no longer silently kills camera analysis** — a missing model made the Ollama server return a 404 (`ollama.ResponseError`), which escaped the video pipeline's error handling and permanently stopped the per-camera snapshot worker with a single log line as the only evidence; camera analysis and Sentinel camera-activity signals stayed dead until an integration reload. The worker now survives unexpected errors (the failed batch is dropped, logged, and processing resumes after a short backoff), and if a worker ever exits its queue entry is cleaned up so the next snapshot starts a fresh one. The `ResponseError` is also handled at each call boundary: video frames are skipped and logged (never turned into bogus "Error analyzing image" captions that could reach notifications), the camera chat tool returns a clear error message to the conversation, and the snapshot-and-analyze service raises a proper Home Assistant error. Closes [#473](https://github.com/goruck/home-generative-agent/issues/473).

- **`think: false` is no longer sent to Ollama vision/summarization models that don't support thinking** — camera frame analysis and video summarization force-disabled reasoning for all edge deployments, bypassing the per-model heuristic that correctly omits the field for non-thinking models like `gemma3` and `qwen2.5vl`. Some Ollama server builds reject a non-nil `think` for models without the thinking capability (HTTP 400), which fed the same worker-death failure chain above. Reasoning is now only force-disabled for models that actually support it.

## [3.14.29] - 2026-07-13

### Added

- **`gemma3:4b` is now a selectable Ollama VLM for Camera Image Analysis** — listed alongside `qwen2.5vl:7b` and `qwen3-vl:8b` in the VLM model dropdown. Tested for caption stability: across repeated captions of identical frames it reproduces the same core scene with only cosmetic wording differences, which is what the semantic notification dedup in `notify_on_anomaly` mode depends on. A lighter option than the 7B/8B models for smaller GPUs (requires Ollama 0.6+; on iGPUs consider lowering the VLM context size, since the 32k default dominates memory use regardless of model size). Docs updated to reflect that a stable 4B model is now a viable choice. Closes [#469](https://github.com/goruck/home-generative-agent/issues/469). Credit to @andymcmanus for the caption-consistency benchmarking on an AMD Radeon 780M iGPU that motivated this.

## [3.14.28] - 2026-07-12

### Added

- **Embeddings feature in the UI** — a new **Embeddings** feature under **+ Setup** lets you explicitly assign the embedding provider and model, just like Conversation/VLM/Summarization. Previously the embedding provider was inherited silently from the chat provider with no UI to change it. When the feature is off, provider selection stays automatic (Conversation provider if embedding-capable, else the first capable provider), preserving existing behavior. Closes [#457](https://github.com/goruck/home-generative-agent/issues/457).

- **Separate embedding endpoints** — chat and embeddings can now run on different servers. Assigning the Embeddings feature to a second OpenAI-compatible provider (e.g. a dedicated llama.cpp embedding instance) or an Ollama provider on another host propagates that provider's own base URL, API key, and dims into the embedding client, without clobbering the chat provider's base URL. Ollama embedding endpoints get their own health probe instead of piggybacking on the global Ollama URL.

### Fixed

- **`AttributeError: 'list' object has no attribute 'data'` with llama.cpp embeddings** — OpenAI-compatible base URLs are now normalized to end with `/v1` before being handed to the OpenAI SDK. The SDK appends bare paths (`/embeddings`, `/chat/completions`) to the base URL, and the config flow validated URLs *without* the `/v1` prefix — so embedding requests hit llama-server's native `/embeddings` route, which returns a raw JSON list instead of the OpenAI `{"data": [...]}` shape and crashed the tool index build and semantic search. Chat only worked by accident because llama-server aliases `/chat/completions` in OpenAI format. Base URLs may now be entered with or without a trailing `/v1`. Root cause of the incompatibility previously documented (and worked around) in #394.

- **Single-provider auto-assignment respects capabilities** — when exactly one model provider exists it is auto-assigned to unconfigured features, but it is no longer assigned to features whose category it cannot serve (e.g. an Anthropic-only setup no longer claims the Embeddings feature).

## [3.14.27] - 2026-07-10

### Added

- **UI-configurable motion sensor → camera override map** — a new `video_analyzer_motion_camera_map` option in Global Options lets you pin specific `binary_sensor.*` entities to specific `camera.*` entities, one pair per line (`binary_sensor.X: camera.Y`). Previously this mapping required editing `const.py`. Useful when automatic resolution picks the wrong camera for a given sensor.

- **UI toggle for perceptual-hash (dHash) frame deduplication** — `video_analyzer_uniqueness_enabled` (default off) replaces the hardcoded `_UNIQUENESS_ENABLED = False` constant. When enabled, visually identical frames are dropped before VLM analysis. Caveat: enabling this removes the visual continuity the summary model uses to narrate motion; only enable it if a static scene generates excessive duplicate snapshots and you accept that motion context may be lost.

- **Event-driven recording state loop for UniFi Protect and similar cameras** — cameras that expose a `recording` HA state (e.g. UniFi Protect) now start a snapshot loop on `recording` entry and flush the collected batch on exit, replacing the previous 1.5s polling loop. If a motion binary sensor fires for the same camera while a recording loop is already running, the motion loop takes ownership and controls the queue flush.

### Fixed

- **Ring-MQTT cameras triggered phantom notifications from interior PIR sensors** — `_resolve_camera_from_motion` matched `binary_sensor.front_door_motion` (a Ring Alarm wall PIR that stays ON for up to 3 minutes) to the doorbell camera instead of the doorbell's own motion sensor (`binary_sensor.front_door_motion_3`). This caused up to 62 snapshot-analysis cycles and mobile notifications per motion event on the wrong subject. Root cause: name heuristics could not distinguish two sensors that share a name prefix but belong to different physical devices.

  Fix: motion → camera resolution is now a **three-tier lookup**:
  1. Explicit override map (UI-configurable, checked first)
  2. HA device registry — finds the `camera.*` entity on the same HA device as the motion sensor; correctly separates Ring Alarm PIRs from doorbell cameras because they are registered to different devices
  3. Name heuristics — fallback for integrations without device links (direct substitution, VMD-suffix strip, `_motion`-suffix strip)

  Closes [#459](https://github.com/goruck/home-generative-agent/issues/459). Credit to @andymcmanus for detailed bug reports, debug log analysis, and patient re-testing that identified the PIR/doorbell sensor collision as the root cause.

- **Motion/recording snapshots were analyzed per frame instead of as one event batch** — frames captured by a motion or recording loop were enqueued into the per-camera live worker queue, whose worker consumed them immediately. For long-held motion sensors (ring-mqtt holds ON up to 180s), this produced an analysis cycle and notification every few seconds while motion remained active, and the OFF-event "flush" only drained leftovers. Fix: loop-captured frames are now held in a per-camera buffer (bypassing the live worker) and flushed as a single ordered batch when motion turns OFF or recording exits — one summary and at most one notification per event. The live worker queue still serves the recording-poll safety-net path. The hold buffer is capped at 50 frames (oldest dropped first).

## [3.14.26] - 2026-06-27

### Fixed

- **Sentinel discovery proposed unsupported candidates for cumulative energy sensors** — the baseline DB tracks all numeric entities, including kWh energy counters (`_energy` suffix). These appeared in the `unmonitored_baseline_entities` list injected into the discovery prompt, causing the LLM to propose appliance anomaly candidates (microwave, washing machine, fridge, dishwasher) that referenced cumulative sensors. Normalization correctly rejects them (`reason_code=cumulative_energy_sensor`) because a monotonically increasing counter cannot produce a meaningful rolling-average baseline, but the proposals showed as "unsupported" in the UI with no actionable path. Fix: added `_is_cumulative_energy_entity()` to filter cumulative sensors from the unmonitored list before prompt injection, and added an `ENERGY SENSOR RULE` hint to the discovery prompt directing the LLM to use `_power` (instantaneous Watts) sensors in `evidence_paths` instead of `_energy` (cumulative kWh) sensors. Corrected proposals map to the existing `baseline_deviation` or `time_of_day_anomaly` templates without any further code changes.

## [3.14.25] - 2026-06-25

### Fixed

- **Video analyzer sent burst notifications from near-camera motion artifacts (e.g. spider webs)** — a spider web in front of a camera repeatedly triggers the motion sensor, causing the video analyzer to generate multiple notifications for the same static background scene. The LLM-generated captions described the same empty scene with slightly different phrasing (e.g. "paved patio enclosed by a white picket gate" vs "paved patio with a white picket fence"), scoring ~0.87 cosine similarity — just below the old 0.89 suppress threshold — so each one fired a mobile notification. Fix: lower `VIDEO_ANALYZER_SIMILARITY_THRESHOLD` from 0.89 to 0.85, catching "same scene, different wording" duplicates within the existing 30-minute dedup window. Genuine events with semantically different content (person arriving, vehicle with headlights) still score below 0.85 against empty-scene captions and correctly notify.

## [3.14.24] - 2026-06-24

### Fixed

- **Sentinel notifications showed HA restart time instead of actual state-change time** — when Home Assistant restarts, entities re-report their current state with a new `last_changed` stamped at startup time. Three Sentinel rule categories were computing falsely short durations as a result ("Alarm disarmed since 5:16 AM", "Garage door lock unlocked for about 15 minutes", "Dishwasher running for 5 minutes" — all actually much older). Fixes:
  - New `alarm_enrichment.py`: queries 30 days of HA recorder history (date-range query, replaces count-based `get_last_state_changes`), walks newest-to-oldest skipping `unavailable`/`unknown` transient states, finds the true `armed_*→disarmed` transition, and corrects `last_changed` before rule evaluation. Falls back to the oldest within-window disarmed record when the armed record has been purged from the DB; clears `last_changed` to `""` (suppressing the misleading timestamp) when the alarm has been disarmed for more than the 30-day lookback window.
  - New `lock_enrichment.py`: same recorder-based pattern for unlocked lock entities. Finds the last `non-unlocked→unlocked` transition (handles `locked`, `locking`, `unlocking`, `jammed` as anchors); uses the oldest within-window record as a fallback for purged records; clears `last_changed` to `""` when the lock has been open for more than 30 days.
  - New `power_enrichment.py`: for power sensors currently drawing more than 10 W, finds the last `off→on` transition in 30-day history. Handles both W and kW units. Leaves `last_changed` unchanged (rather than clearing) when no useful transition can be determined.
  - `engine.py`: calls all three enrichments in sequence after snapshot build, before rule evaluation.
  - `dynamic_rules.py`: lock evidence now uses `lock.get("last_changed") or None` so an empty-string `last_changed` (set by enrichment when the lock has been open >30 days) propagates as `None` to LLM context.

- **Disarm notifications showed time-only for old disarms** — notifications like "Alarm disarmed since 1:46 PM" were ambiguous when the disarm occurred on a previous day. `notifier.py` now uses a `_format_disarm_since()` helper that prepends the date ("14 Jun at 1:46 PM") when the disarm was on a different calendar day, and shows time-only for same-day disarms.

## [3.14.23] - 2026-06-19

### Fixed

- **Sentinel: `alarm_disarmed_open_entry` notifications were confusing and showed the wrong timestamp** — the notification subtitle was derived from the raw rule ID ("Alarm disarmed open entry alarm control panel home alarm") and the message body showed the window sensor's `last_changed` time instead of when the alarm was actually disarmed. Fixes:
  - Added a dedicated subtitle and mobile message formatter for the `alarm_disarmed_open_entry` template; subtitle now reads e.g. "Family Room Right Window open, alarm disarmed" and the body shows the actual disarm time ("Alarm disarmed since 10:15 PM").
  - Dynamic rule evaluator now stores `alarm_last_changed` and `entry_last_changed` as separate evidence fields so the correct timestamp is always available.
  - "Ask Agent" handoff prompt now distinguishes read-only `binary_sensor` entries (cannot be closed programmatically) from actuatable `cover.*` entities. For sensors it explicitly forbids arming the alarm as a substitute and instructs the agent to advise the user to close the entry manually or use "Snooze 24h"/"Snooze Always" if the open window is expected.
  - Added direct regression tests for the new mobile copy, subtitle, LLM-bypass routing, and the binary_sensor vs cover Ask Agent prompt branches.

## [3.14.22] - 2026-06-19

### Fixed

- **Sentinel Discovery: normalization produced defective dynamic rules** — nine bugs in `explain_normalize_candidate` caused approved Discovery candidates to register rules with wrong entity types, missing domain prefixes, or semantically incorrect templates. Fixes:
  - `_find_sensor_entity_ids` now accepts only `sensor.*`; `binary_sensor.*` entities (motion, contact) no longer bleed into power-sensor branches.
  - `_find_battery_sensor_entity_ids` now accepts only `sensor.*`; `binary_sensor.*` battery entities (on/off state) are excluded so they cannot produce `low_battery_sensors` rules that silently generate no findings.
  - `_find_entry_entity_ids` now accepts only `binary_sensor.*` and `cover.*`; plain `sensor.*` numeric sensors can no longer be treated as door/window contacts.
  - The `low_battery_sensors` branch for lock candidates now requires a `sensor.*` battery entity; candidates that reference only `lock.*` entities return `unsupported_pattern` instead of emitting a rule with a lock ID in `sensor_entity_ids`.
  - `_extract_threshold_numeric` returns `None` for values ≤ 0 (e.g. "above 0 watts"), preventing `sensor_threshold_condition` rules with a useless threshold of zero; such candidates now fall back to `baseline_deviation`.
  - `_presence_signal` checks `"not derived.anyone_home"` in `evidence_paths` first (returns `"away"`); `"derived.anyone_home"` alone now returns `"any"` instead of incorrectly inferring `"home"`.
  - Entry branches now handle `presence == "any"` (unknown occupancy) by defaulting to `open_entry_while_away` rather than falling through to `_normalization_failure`.
  - The early broad `open_any_window_at_night_while_away` fallback that fired before battery/sensor/energy branches is removed; the guarded late fallback (requires night + away signals) remains.
  - `multiple_entries_open_count` defaults to `require_away=True` when occupancy is unknown, replacing the previous `require_home=False, require_away=False` state that made the rule fire unconditionally.
  - Dead `return None` after the final `return` in `_find_camera_id` is removed.

- **`patch_dynamic_rule` service added** — new HA service `home_generative_agent.patch_dynamic_rule` merges a partial params dict into an existing dynamic rule without removing and re-approving it. Useful for repairing defective rules in the registry that were created before the normalization fixes.

## [3.14.21] - 2026-06-15

### Fixed

- **Entity staleness alert falsely reported as resolved after Execute** — after tapping Execute on a person tracking staleness alert, the agent incorrectly replied that the alert had resolved ("the person is no longer in a stale state") even though the person never came home. The generic execute prompt did not define what "resolved" means for staleness findings, so the agent treated `state = not_home` from `GetLiveContext` as a definitive non-stale state. A specialized execute prompt for `entity_staleness` findings now tells the agent the alert resolves only if the person is home (`state = 'home'`); if still away, it must acknowledge the staleness and advise the user to check whether the person's phone is on and reachable. Follows up on the incomplete fix in [#417](https://github.com/goruck/home-generative-agent/pull/417).

- **Person name missing from entity staleness initial notification** — the initial notification body for person tracking staleness was generic ("The person tracking data has been outdated...") with no person name, while the Execute follow-up named the person. `_eval_entity_staleness` now includes `friendly_name` in the evidence dict, and a deterministic `_entity_staleness_mobile_message` generates a named, consistent body: e.g. "Lindo St Angel's location tracking has been outdated for about 1 day. Check if their phone is on and reachable."

- **`friendly_name` in entity_staleness evidence caused unstable anomaly_id** — adding `friendly_name` to the evidence hash would make the anomaly_id change if the display name is renamed or temporarily unavailable, breaking cooldown/suppression/audit continuity for long-running alerts. `_build_finding` now strips `friendly_name` before hashing, matching the existing pattern in `baseline.py` and `appliance_power_duration.py`.

## [3.14.20] - 2026-06-11

### Added

- **Basic and Advanced setup modes** — Feature setup and Sentinel setup now offer a mode selector. **Basic** enables all features (Conversation, Camera Image Analysis, Conversation Summary) with recommended defaults in a single screen and auto-creates the database subentry with no database prompt. **Advanced** steps through each feature individually, lets you assign providers, models, and fallback chains, and includes an explicit database configuration step. Closes [#433](https://github.com/goruck/home-generative-agent/issues/433).

### Fixed

- **Sentinel background tasks continue running after subentry deletion** — deleting the Sentinel subentry now stops all background tasks (monitor loop, baseline updater, discovery engine, notifier) immediately without waiting for the next integration reload. `_apply_sentinel_options` also forces `CONF_SENTINEL_ENABLED=False` when no Sentinel subentry exists, preventing tasks from being restarted on the subsequent reload.

- **`ValueError: Config entry update listeners should not be used with OptionsFlowWithReload`** — saving main configuration options raised this error because the integration registered an `update_listener` on the config entry, which `OptionsFlowWithReload` explicitly forbids. The listener is now registered via `async_dispatcher_connect` on `SIGNAL_CONFIG_ENTRY_CHANGED`, which fires the same signals without touching `entry.update_listeners`.

- **Infinite reload loop after subentry change** — `SIGNAL_CONFIG_ENTRY_CHANGED` fires on every entry state transition, not just data changes, causing the dispatcher callback to schedule an unbounded sequence of reloads. A snapshot of Sentinel subentry data captured at setup time is now compared on each signal; reloads are only scheduled when the subentry data actually changes.

- **Advanced setup form not pre-populated when reconfiguring Sentinel** — reopening Sentinel Advanced setup now pre-fills every field with the current saved values instead of showing empty defaults.

- **Basic setup silently overwrites existing configuration** — running Basic setup when a Setup or Sentinel subentry already exists now displays a warning and requires confirmation before overwriting the saved configuration.

- **Misleading Conversation toggle removed from feature enable form** — the Conversation feature toggle was shown in the Advanced feature enable step but had no effect (Conversation is always enabled). It is no longer displayed.

## [3.14.19] - 2026-06-07

### Fixed

- **Sentinel Baseline: cyclical-load sustained gate default too short** — the default `sentinel_baseline_sustained_minutes` of 20 minutes matched the length of a normal fridge/freezer compressor off-cycle, causing a notification on every idle cycle. Raised to 45 minutes, which filters normal off-cycles (20–40 min) while still catching genuine malfunctions (door left open, failed compressor) that sustain for hours. The setting is now documented in `docs/sentinel.md` with tuning guidance.

## [3.14.18] - 2026-06-07

### Fixed

- **Sentinel Discovery: candidate entity mismatch** — monitoring-gap candidates can no longer pass through when the title, summary, or pattern names a different known baseline entity than the entities listed in `evidence_paths`. The Discovery prompt now tells the LLM to describe only the cited monitoring-gap entity or entities, and a post-generation quality gate drops mismatched candidates with `dedupe_reason="entity_text_mismatch"`. Regression coverage includes the reported kitchen-vs-garage/playroom lock battery mismatch, generic single-entity summaries, and intentional multi-entity bundles. Closes [#437](https://github.com/goruck/home-generative-agent/issues/437).

- **Discovery semantic coverage helper import** — moved the rule-vs-candidate semantic-key coverage helper into `sentinel/discovery_semantic.py` so runtime code and semantic tests import it from the same module. This fixes the Pyright error from importing `_rule_key_covers_candidate_key` from the integration entrypoint.

## [3.14.17] - 2026-06-06

### Fixed

- **Sentinel Discovery: cyclical-load false positives** — `baseline_deviation` averages compressor-on (~150 W) and compressor-off (~5 W) samples into a single rolling mean (~40–60 W). Every normal off-cycle then fires as a 90%+ deviation. Cyclical loads (entities whose `friendly_name` or `entity_id` contains `fridge`, `refrigerator`, `freezer`, or `compressor`) are now routed to `time_of_day_anomaly` at normalization time. Its variance-aware threshold `max(2*stddev, drift%)` tolerates the oscillation without false alarms. Closes [#432](https://github.com/goruck/home-generative-agent/pull/432).

- **Sentinel Discovery: cumulative energy sensors proposed as baseline rules** — `sensor.*_energy` entities are monotonically increasing counters. Normalizing them to `baseline_deviation` caused continuous firings as the cumulative value grew past its rolling average. Energy sensor candidates are now rejected at normalization time with `reason_code="cumulative_energy_sensor"`.

- **Sentinel Discovery: rule duplication for baseline and other templates** — `rule_semantic_key` returned `None` for `baseline_deviation`, `time_of_day_anomaly`, `sensor_threshold_condition`, `entity_state_duration`, and `entity_staleness` rules, so approved rules of those types were never added to the deduplication exclusion set. Discovery re-proposed identical rules on every cycle, producing duplicate rule entries in the registry. Fixed by adding semantic key generation for all missing templates. Keys for `baseline_deviation` and `time_of_day_anomaly` embed a `|template=<name>|` marker used by the baseline coverage check.

- **Sentinel Discovery: `entity_ids contains` evidence path format not parsed** — LLM-generated evidence paths sometimes use the form `entities[entity_ids contains sensor.foo].state`. `candidate_semantic_key` was not extracting the entity ID from this syntax, producing keys without the entity ID and preventing per-entity deduplication. A dedicated regex branch now handles this format alongside the existing `entity_id=` form.

- **Sentinel Discovery: history record keys leaked into LLM hint set** — `_existing_semantic_context` was including past discovery-cycle record keys in the `hint_keys` set sent to the LLM as "already covered" topics. Past record keys are `candidate_semantic_key`-derived (no `|template=…|` marker) and could silently block proposals for entities that merely appeared in a historical finding. Fixed by splitting into `hint_keys` (active rules + pending/rejected proposals only) and `filter_keys` (hint_keys + history, used for post-hoc dedup only).

- **Sentinel Discovery: approved proposals blocked re-proposal after rule is disabled** — `_existing_semantic_context` checked `status == "accepted"` (wrong string; the store uses `"approved"`), so approved proposals were never excluded from `hint_keys`. Their candidate key remained in the hint set indefinitely, suppressing re-proposal even if the user later disabled the resulting rule. Fixed by checking `"approved"` and skipping approved proposals from `hint_keys` (coverage is already tracked by the live rule via `rule_semantic_key`).

- **Sentinel Discovery: broad unavailability rule keys counted as baseline coverage** — the monitoring gap analysis checked whether a baseline-ready entity ID appeared in _any_ hint key, including `predicate=unavailable` keys from rules like `unavailable_sensors_while_home`. These rules list many entity IDs as a side-effect, causing baseline-ready entities to be incorrectly marked "monitored." Multi-entity bundle candidate keys from rejected or pending proposals had the same effect. Fixed by switching to `_BASELINE_TEMPLATE_MARKERS`: only keys containing `|template=baseline_deviation|` or `|template=time_of_day_anomaly|` — emitted exclusively by `rule_semantic_key()` for real statistical monitoring rules — count as baseline coverage.

- **Sentinel notifications: baseline deviation subtitle and body were generic** — baseline deviation and time-of-day anomaly findings used the raw `finding.type` slug as the mobile notification subtitle (e.g. "Fridge power baseline deviation home") and fell through to the LLM explanation path for the body, which sometimes described the baseline value as "current state." Both are now deterministic: the subtitle reads "Fridge: power lower than expected" (direction from `deviation_direction` evidence field) and the body includes actual measured values: "Fridge: 4.6 W vs usual 85.0 W (95% below normal). Check appliance."

## [3.14.16] - 2026-06-01

### Fixed

- **Sentinel daily digest never fires** — `_apply_sentinel_options()` in `core/subentry_resolver.py` propagates sentinel subentry settings into the runtime options dict by iterating over an explicit `sentinel_defaults` allowlist. Six keys were missing from that list, so user-configured values were silently replaced by their built-in defaults at every integration load. The most user-visible impact was `CONF_SENTINEL_DAILY_DIGEST_ENABLED`, whose default is `False`; the digest timer was therefore never registered regardless of what was set in the config flow. Also fixed silent misconfiguration of `CONF_SENTINEL_BASELINE_SUSTAINED_MINUTES`, `CONF_SENTINEL_BASELINE_WEEKLY_PATTERNS`, `CONF_SENTINEL_BASELINE_DOW_MIN_SAMPLES`, and `CONF_SENTINEL_CAMERA_ENTRY_LINKS`. Closes [#431](https://github.com/goruck/home-generative-agent/pull/431).

## [3.14.15] - 2026-05-31

### Fixed

- **Gemini provider: 400 error on every chat interaction** — Gemini's Protobuf schema validation requires an explicit `items` field on all array-type tool parameters. `GetLiveContextTool.domain` (always injected on tool-retrieval turns) uses `vol.Any(cv.string, [cv.string])`, which `voluptuous_openapi` emits as an `anyOf` schema. `langchain_google_genai` resolved the `anyOf` to `type_:ARRAY` but checked `items` on the outer dict where it was absent, producing a declaration Gemini rejected. Added `_ensure_array_items()` to `_format_and_dedupe_tools()` which hoists `items` from `anyOf` array variants and recursively patches any bare `{"type": "array"}` node. Closes [#428](https://github.com/goruck/home-generative-agent/issues/428).

## [3.14.14] - 2026-05-31

### Added

- **Sentinel built-in rule: `pet_detected_at_night_no_occupancy`** — fires when a pet (cat, dog, rabbit, hamster, bird, or parrot) is detected on any monitored camera at night while no residents are home. Requires at least one active motion or VMD signal. Severity: low, confidence: 0.85, informational only (no suggested action). Closes [#422](https://github.com/goruck/home-generative-agent/issues/422).

### Changed

- **`langchain-core` manifest pin loosened** — changed from `==1.3.2` to `>=1.3.2,<2.0.0` to prevent `uv` from silently skipping installation on hosts where a newer compatible version is already present.

## [3.14.13] - 2026-05-23

Thanks to [Alex Ultra](https://github.com/alex-mextner) for contributing the original STT hallucination filter and model-provider fallback work in [#421](https://github.com/goruck/home-generative-agent/pull/421).

### Added

- **Deploy script** — `scripts/deploy` syncs the integration to a running Home Assistant instance via rsync over SSH. Only changed files are transferred (checksum-based), `__pycache__` and `.pyc` files are excluded, and the script calls `ha core restart` after a successful sync. See [Contributing](docs/contributing.md) for setup instructions.
- **Feature-level model provider fallbacks** — feature setup now supports an ordered fallback-provider list for chat, VLM, summarization, and embeddings. If the primary provider is unavailable at setup or fails with a retryable runtime error, HGA can use the next configured provider for that category.
- **Fallback provider notifications** — when a fallback becomes active, HGA sends one deduplicated notification per category/provider for the current runtime. Mobile notifications use the configured `notify_service`; otherwise HGA creates a persistent notification. Cloud fallback notifications include a provider-cost warning.
- **STT hallucination filters** — Options now include ignored substring and exact-phrase filters for speech-to-text prompts. Matching input is dropped before it reaches the LLM, reducing false activations from common silence/noise transcriptions.

### Changed

- **Fallback logging is clearer and less noisy** — startup, runtime fallback, unavailable-provider, sticky embedding-provider, and local resource-gate logs now distinguish configured deployment from the effective provider in use. Repeated unconfigured fallback messages are reduced to debug-level diagnostics where appropriate.
- **Fallback models use their own configured defaults** — chat, VLM, and summarization fallbacks now preserve provider-specific model settings and category defaults from `const.py`, including Ollama tuning options when a provider does not override them.
- **Fallback chat retry happens at the model-call boundary** — fallback chat chains no longer expose provider-level `astream`, avoiding mixed responses where a failed provider starts streaming partial text and a fallback provider finishes with different content.

### Fixed

- **Embedding fallback no longer leaves stale vector indexes active** — when the active embedding provider changes, HGA marks the tool index stale, rebuilds it on the next indexing pass, and discards retrieval results if the index becomes stale during search.
- **Embedding dimensions are normalized for fallback providers** — Gemini embedding fallbacks are requested with HGA's configured vector dimension, preventing PostgreSQL vector errors such as `different vector dimensions 1024 and 3072`.
- **VLM and summarization fallback wrappers preserve config overrides** — fallback wrappers now support `with_config()` and merge model config instead of replacing category-specific model settings, fixing video analyzer and summarization paths that use per-call config.
- **Length-limited empty model responses trigger fallback** — empty responses caused by provider length limits are treated as retryable, so chat and summarization can fall back instead of returning blank content.
- **Connectivity failures from HTTP transports are retryable** — local provider connection errors such as Ollama/httpx transport failures now activate configured fallbacks.
- **Tool retrieval fallback diagnostics are safer** — known vector-store mismatch errors degrade to keyword-filtered tool selection or recency-based memory retrieval instead of surfacing repeated tracebacks during provider switches.

## [3.14.12] - 2026-05-19

### Fixed

- **Ambiguous entity name no longer loops the agent** — when multiple entities
  share a friendly name (e.g. `binary_sensor.haustur` and `binary_sensor.haustur_2`
  after a device is re-added), `get_entity_history` now silently picks the
  canonical entity (no numeric `_N` suffix). If ambiguity cannot be resolved
  this way, the tool returns `{"error": "..."}` so the LLM can report the
  problem to the user instead of looping on an empty `{}` response (issue #414).

## [3.14.11] - 2026-05-19

### Changed

- **GetLiveContext is now always injected into the tool candidate list** — the
  previous approach used a regex to detect conditional clauses ("if", "when",
  etc.) and only injected GetLiveContext when a match was found. This was brittle
  and missed constructs like "while", "whenever", "assuming", and compound
  queries. GetLiveContext is now added unconditionally whenever it is absent from
  the candidate set; step 3b's deduplication prevents double-injection for
  read-only open-state queries.

- **Camera activity removed from the system prompt** — the agent was fetching
  recent camera activity on every turn and injecting it into the system message
  as a `<recent_camera_activity>` block. This caused the model to answer presence
  queries from stale camera snapshots instead of calling `GetLiveContext` with
  `domain=person` for authoritative state. The model now fetches camera data via
  tool call only when it needs it.

- **Sentinel action prompts include detection timestamp and alert-time evidence**
  — both execute and handoff prompts now include the exact time the anomaly was
  detected and a compact evidence summary, so the agent can explicitly acknowledge
  whether the alert condition is still active or has already resolved. This
  prevents the model from silently substituting current state for the original
  alert context.

- **`AnomalyFinding` now records its detection timestamp** — a `detected_at`
  field (`datetime`, defaulting to `utcnow()`) is added to the dataclass and
  serialized in `as_dict()`. Downstream consumers (audit trail, action prompts)
  can now reference the precise detection time.

### Fixed

- **Removed global LangChain in-memory LLM cache** — `set_llm_cache(InMemoryCache())`
  was called unconditionally at conversation entity setup. This has been removed;
  caching is now left to LangChain's default behaviour (no global cache).

## [3.14.10] - 2026-05-17

### Fixed

- **Open-state queries now work for users whose sensors use `device_class: window`
  or `device_class: door`** — the parser that identifies open entry sensors from
  a live-context response only accepted `device_class: opening`. Real Home
  Assistant window and door contact sensors (registered by most Zigbee and Z-Wave
  coordinators) use `device_class: window` and `device_class: door` respectively.
  The same gap existed in the HA-state fallback path used when no prior live
  context is available. Both paths now accept all four open-state device classes
  (`door`, `garage_door`, `opening`, `window`). Four regression tests are added
  that use the correct device classes (the prior test fixtures all used
  `device_class: opening`, which matched the developer's own sensors but masked
  the failure for anyone else).

## [3.14.9] - 2026-05-17

### Fixed

- **Read-only open-state queries are more reliable across local tool-calling
  models** — HGA now force-binds and promotes `GetLiveContext` for read-only
  queries such as "list all open windows" and "list the open doors in my house",
  even when semantic tool retrieval misses it or ranks less relevant tools
  higher. The first `GetLiveContext` call for these queries is normalized to a
  broad `binary_sensor` live-context request so local models that emit brittle
  filters like `name: "Window"` or list-valued domains still get the needed
  state. Broad live-context results are then filtered back to the requested
  entity type, preventing door queries from volunteering open windows. Focused
  regression tests cover retrieval, argument normalization, retry handling, and
  scoped result filtering.

## [3.14.8] - 2026-05-16

### Fixed

- **`NullChat` fallback no longer crashes with `TypeError` when a provider is
  unavailable at startup** — `NullChat.ainvoke` and `NullChat.astream` now
  accept `config` as an explicit second positional parameter, matching the
  standard LangChain `BaseChatModel` calling convention
  (`ainvoke(input, config=None, **kwargs)`). Previously, the fallback model
  used when a provider health check failed at HA startup (e.g., Ollama not yet
  responsive) would raise `TypeError: NullChat.ainvoke() takes 2 positional
  arguments but 3 were given` instead of returning the intended graceful
  `"LLM unavailable."` response. Two regression tests added. Closes
  [#410](https://github.com/goruck/home-generative-agent/issues/410).

## [3.14.7] - 2026-05-15

### Fixed

- **Read-only state queries like "list all open windows" no longer trigger
  actuation tools** — The word "open" in phrases like "which windows are open"
  or "show me all open doors" was matching the actuation-safety keyword list,
  causing the agent to inject control tools (`HassTurnOn`, `HassTurnOff`, etc.)
  into its context for what is really a read-only query. A four-regex heuristic
  now distinguishes read-only use of "open" (as a state descriptor combined with
  `list`, `show`, `which`, `what`, `are`, `is`, `status`, or `state`) from
  command use of "open" (as an actuation verb). The `GetLiveContext` tool is
  also promoted to the front of the candidate list for these queries so the
  agent picks up the correct read tool first. Closes
  [#394](https://github.com/goruck/home-generative-agent/issues/394).

- **Tool-loop guard prevents `GraphRecursionError` on stuck tool cycles** — When
  a weaker model or an ambiguous query caused the agent to request tool calls
  indefinitely, LangGraph eventually threw a `GraphRecursionError` visible as an
  error in the HA conversation UI. A new `tool_loop_guard` graph node caps
  tool-use at 3 rounds per conversation turn and returns a friendly message
  instead of crashing: "I wasn't able to complete this request after several
  tool-use attempts. Please try rephrasing your query or breaking it into smaller
  steps." The LangGraph `recursion_limit` is also raised from 10 to 20 so the
  application-level guard fires well before the LangGraph backstop.

## [3.14.6] - 2026-05-13

### Fixed

- **Sentinel no longer fires a false "disarm the alarm" notification when armed_home or
  armed_night is active with occupants present** — `armed_home` and `armed_night` are
  HA alarm modes designed for use while people are home. A three-layer guard now
  prevents this combination from ever becoming an anomaly: the deterministic evaluator
  suppresses it at runtime, the normalization layer rejects any future LLM-proposed
  rule with this pattern, and the LLM explanation prompt is conditioned to never
  describe the state as a problem when occupants are present. The constant defining
  these occupancy-safe modes (`SENTINEL_OCCUPANCY_ARMED_STATES`) is now centralized
  in `const.py` and shared across both sentinel modules. Five new regression tests
  (plus three coverage tests) guard all paths.

## [3.14.4] - 2026-05-12

### Changed

- **OpenAI and Anthropic providers now stream tokens natively** — Both providers are
  constructed with `streaming=True`, so visible words appear in the chat window as
  they are generated rather than arriving all at once at the end. OpenAI additionally
  sets `stream_usage=True` so token-count telemetry is preserved in streamed responses
  (without this flag the usage metadata would be empty). The existing non-streaming
  fallback path remains active for Ollama, Gemini, and any provider that does not emit
  token chunks. The `schema_first_yaml=True` dashboard-generation path is unaffected —
  it invokes the model directly via `ainvoke` and was already independent of the
  streaming flag.

## [3.14.3] - 2026-05-11

### Fixed

- **Appliance duration alerts now name the actual appliance** — Notifications for
  `appliance_power_duration` findings previously fell back to the generic phrase
  "An appliance" because the LLM explanation lacked a display name. The rule now
  includes `friendly_name` in finding evidence, and a new deterministic message
  builder formats the mobile copy directly — e.g. `Washer drew about 296 W for
  633 min, above the 60 min threshold. Check it.` — bypassing the LLM path
  entirely for this finding type. Power-sensor suffixes (`Power`, `Energy`, etc.)
  are stripped case-insensitively so user-set names like `EV Charger Power` render
  as `EV Charger`. When no friendly name is present the entity ID is used as
  fallback. The same formatter is reused for persistent notifications. Closes
  [#391](https://github.com/goruck/home-generative-agent/issues/391).

## [3.14.2] - 2026-05-11

### Fixed

- **Streaming silent for Anthropic and OpenAI** — Both providers default to
  `streaming=False` in LangChain, so no `on_chat_model_stream` events fire and
  the chat window stayed blank until the full response arrived. A new
  `on_chat_model_end` fallback in `_stream_langgraph_to_ha` yields the complete
  text for text-only responses when no streaming chunks were delivered in a turn.
  Anthropic streaming chunks (type `text_delta`) are now also recognised by
  `_normalize_ai_content`, so Anthropic works correctly in both streaming and
  non-streaming modes.
- **OpenAI 400 "object schema missing properties" on `confirm_sensitive_action`**
  — Tool schema extraction was calling `args_schema.schema()`, which fails on
  `InjectedStore` annotations and produced a bare `{"type": "object"}` without the
  `properties` field that OpenAI requires. Schema extraction now uses
  `tool_call_schema.model_json_schema()` (which correctly excludes injected
  arguments) as the primary path, with a `properties: {}` safety net added to
  `_format_and_dedupe_tools` for any schema that declares `type: object` but omits
  `properties`.
- **`get_entity_history` not retrieved for history questions** — The tool's RAG
  embedding description was too terse to match natural-language queries like "how
  many times was the front door opened". The description was rewritten to enumerate
  concrete use-cases (open/close counts, on-durations, motion trigger times, run
  frequency) so cosine similarity retrieval picks it up reliably.
- **Blocking `ssl.load_verify_locations` in the HA event loop** — `langchain_anthropic`
  lazily constructs an async `httpx` client (and triggers SSL context loading) on
  every call to `_get_default_async_httpx_client`. The function is now patched at
  import time with `lru_cache` so the client is created once; the first call is
  pre-warmed in a thread-pool executor after provider initialisation so SSL I/O
  never blocks the HA event loop.

## [3.14.1] - 2026-05-10

### Fixed

- **Anthropic API rejects tools with missing `input_schema.type`** — Tools whose
  parameters schema had no top-level `type` field (e.g. tools with no parameters,
  returning `{}`) caused a `400 invalid_request_error` from the Anthropic API on
  every chat query. `_format_and_dedupe_tools` now injects `"type": "object"` when
  the field is absent, and replaces non-dict parameter values (null, array) with
  `{"type": "object"}` to prevent a downstream `AttributeError`.

## [3.14.0] - 2026-05-10

### Added

- **Video analyzer caption deduplication** — The video analyzer now suppresses
  repeated low-value camera notifications within a configurable window
  (`VIDEO_ANALYZER_CAPTION_DEDUPE_WINDOW_SEC`, default 30 minutes). A new
  `CaptionNoveltyDecision` result type carries the suppression reason and match
  details, enabling structured debug logging and future metadata storage.

### Fixed

- **Caption novelty used weakest vector match** — `_is_anomaly` treated any
  single below-threshold result as novel, even when the best matching caption was
  highly similar. Replaced with a best-score check so only the closest stored
  caption governs the suppress/notify decision. Renamed to `_is_caption_novel`
  to match the new semantics.
- **Generic "animal" subject not recognized** — Captions describing "a dark
  animal stands on the path" were incorrectly suppressed against days-old matches
  because `animal` was absent from the subject-term list. Added `animal` to
  `_SUBJECT_RE` so generic animal descriptions trigger the stale-match re-notify
  path alongside `cat`, `dog`, `deer`, and other named species.
- **Dead regex stems in action detector** — Partial stems `arriv`, `leav`,
  `driv`, and `mov` in `_ACTION_RE` could never match at a word boundary.
  Replaced with the correct infinitive forms: `arrive`, `leave`, `drive`, `move`.

## [3.13.1] - 2026-05-05

### Fixed

- **Home Assistant blocking I/O during model setup** — Ollama chat and embeddings
  clients now use Home Assistant's SSL context helper, and tool binding is moved
  into the executor path. This avoids blocking `httpx` client setup work on the
  event loop during integration setup and model calls.
- **Video analyzer face API client ownership** — The video analyzer now uses Home
  Assistant's shared async `httpx` client for face API calls and passes the face
  timeout per request, avoiding direct async client construction and shutdown in
  the integration.
- **Sentinel log noise** — Repeated Sentinel operational failures are now rate
  limited with recovery messages, and routine cycle/debug logs were removed from
  the hot path. Discovery, triage, suppression, execution, and trigger handling
  should be quieter while still surfacing first failures and repeated issues.
- **Sentinel LLM calls off the event loop** — Sentinel model calls now prefer the
  synchronous `invoke` path in a worker thread when available, preserving
  admission/defer semantics while reducing event-loop pressure.
- **Video analyzer priority typecheck coverage** — Tests were tightened around
  `NullChat`, Ollama server logging, and nested mock accessors so priority-path
  coverage remains compatible with Pyright.

## [3.13.0] - 2026-05-03

### Added

- **Anthropic Claude provider** — You can now select Anthropic Claude models
  (claude-opus-4-5, claude-sonnet-4-5, claude-haiku-4-5, and their successors)
  as the chat, vision (VLM), and summarization provider. Configure by adding an
  Anthropic model-provider subentry and entering your API key. Prompt caching is
  enabled automatically via `cache_control: ephemeral`, reducing latency and cost
  on repeated conversation turns.
- **Anthropic API key validation** — The config flow validates the API key against
  `https://api.anthropic.com/v1/models` before saving, with distinct errors for
  authentication failures and connectivity issues.
- **`extract_final` list-content support** — `extract_final()` now accepts
  Anthropic's multi-block content format (`list[{"type":"text","text":"..."}]`) in
  addition to plain strings. This fixes triage, discovery, and explain responses
  when Anthropic returns structured content blocks.

### Fixed

- **Sentinel triage with Anthropic provider** — `_parse_response` used `str(content)`
  which produced a Python dict repr when Anthropic returned multi-block content,
  causing every alert to bypass triage (fail-open to notify). Fixed to use
  `extract_final(content)`.
- **LLM explain with Anthropic provider** — `extract_final(str(content))` → 
  `extract_final(content)` so list-format Anthropic responses are rendered
  correctly in push notifications instead of showing raw dict repr.
- **Discovery engine with Anthropic provider** — `json.loads(content)` raised
  `TypeError` (not caught by the bare `json.JSONDecodeError` handler) when
  Anthropic returned a list. Fixed to use `extract_final(content)` before parsing
  and added `TypeError` to the handler.
- **Config flow `else` branch hardened** — The Anthropic API-key schema builder
  used a bare `else:` that would silently present the Anthropic form for any
  future unknown provider. Changed to explicit `elif provider_type == "anthropic":`
  with a fallback empty schema.
- **Config flow: API key stored on validation failure** — `settings["api_key"]`
  was assigned even when `validate_anthropic_key` raised an exception. Moved
  inside the success path.

## [3.12.6] - 2026-05-03

### Fixed

- **Sentinel LLM timeouts on Qwen3-family models** — Qwen3.5:35b defaults to
  thinking mode (generating `<think>…</think>` chains) when `ChatOllama.reasoning`
  is `None`. `reasoning_field()` now passes `{"reasoning": False}` instead of `{}`
  when reasoning is disabled, explicitly suppressing the default. This was the root
  cause of repeated 20–60 s timeouts on Sentinel explain and discovery calls.
- **Discovery prompt token bloat** — `_existing_semantic_context()` could return up
  to 400 semantic keys (200 proposals + 200 discovery records), adding ~16 K chars
  of exclusion context on top of the 20 K snapshot budget. Keys sent to the LLM are
  now capped at 60 (`_MAX_SEMANTIC_KEYS_IN_PROMPT`). The post-hoc
  `_filter_novel_candidates()` filter catches any duplicates that slip through.
- **Sentinel explain timeout too tight** — `_EXPLAIN_LLM_TIMEOUT_S` raised 20 → 30 s
  to give the model headroom over the observed ~20 s inference time.
- **Discovery timeout too tight** — `_DISCOVERY_LLM_TIMEOUT_S` raised 45 → 60 s.
- **Opaque timeout log** — `async_explain` now logs
  `"LLM explanation timed out after 30s; skipping."` instead of the empty-string
  warning produced when `TimeoutError` was bundled with other exceptions.
- **Discovery prompt observability** — a new `DEBUG` log line reports prompt char
  counts (`snapshot=N, keys=M/total`) each discovery cycle.

## [3.12.5] - 2026-04-29

### Fixed

- **Sentinel discovery snapshot stays within 20 k-char (~5 k token) budget** —
  `reduce_snapshot_for_discovery()` previously produced snapshots up to ~32 k
  tokens, routinely exceeding the 45 s discovery LLM timeout. Four changes tighten
  the output:
  - `_MAX_CAMERA_ACTIVITY` reduced 50 → 20; `_MAX_SUMMARY_CHARS` 150 → 80.
  - Derived context compressed: `timezone` dropped (redundant with `is_night`),
    `now` and motion timestamps truncated to minute precision,
    `baseline_ready_entities` intersected with the filtered entity set and capped
    at 30 IDs.
  - Character-budget gate (`_TOKEN_BUDGET_CHARS = 20_000`) with four progressive
    trim passes: (1) strip `last_changed` from entity groups, (2) truncate camera
    summaries to 40 chars, (3) drop all camera summaries, (4) cap
    `recognized_people` per camera at 5 entries — the fourth pass prevents
    facial-recognition deployments from exceeding the budget after summaries are
    dropped.
  - Nine new tests cover timezone removal, `baseline_ready_entities` filtering
    and cap, camera-count cap, budget compliance, and each of the four trim passes.

## [3.12.4] - 2026-04-28

### Fixed

- **`SENTINEL_ADMISSION_TIMEOUT_S` public constant** — admission timeout was a
  magic `2.0` literal scattered across triage, discovery, and explain call sites.
  Extracted to a named constant in `core/utils.py` and imported at each site.

- **`asyncio.create_task()` replaces deprecated `asyncio.ensure_future()`** —
  `run_sentinel_llm_call` now wraps the call factory in a proper `Task`, which is
  cancellable and tracked in `_sentinel_llm_tasks`.

- **HA reload cancels in-flight Sentinel LLM tasks** — the previous reload path
  replaced `_sentinel_llm_tasks` with a fresh `set()` while old tasks continued
  running. The teardown in `__init__.py` now cancels each task before replacing
  the set.

- **Warning log when Sentinel LLM cancel times out** — `contextlib.suppress`
  silently swallowed any `TimeoutError` from the cancel wait. Replaced with an
  explicit `except TimeoutError` that emits a `WARNING` log.

- **Tests** — deferred-path coverage for `SentinelTriageService`,
  `LLMExplainer`, and `run_sentinel_llm_call` (timeout path); three tests for
  `build_model_deployments` (ollama→edge, openai→cloud, empty providers).

## [3.12.3] - 2026-04-26

### Fixed

- **Deployment-aware admission control replaces lock-based GPU gate** — the
  previous `chat_priority_context` / `_bg_vlm_lock` / `_bg_llm_lock` approach
  made chat wait for background work to finish. The new design inverts this:
  `local_chat_session(deployment)` marks the full chat turn active by clearing a
  shared `_chat_idle` event; Sentinel triage and discovery call
  `sentinel_admission(deployment, timeout_s=2.0)` before each LLM invocation and
  defer if chat is active. Video analysis no longer participates in any
  admission gate — its existing tuning constants (`_VISION_TIMEOUT_SEC`,
  `_SUMMARY_TIMEOUT_SEC`, per-camera queues) remain the intended concurrency
  surface. Embedding generation is ungated and runs concurrently with chat.

- **Provider deployment metadata flows through runtime** — `ModelProviderConfig`
  gains a `deployment: str` field ("edge"/"cloud"). `build_model_deployments()`
  builds a `{category: deployment}` map from provider subentries at setup time;
  the map lives on `HGAData.model_deployments`. Cloud providers bypass all local
  admission gates automatically, with no code changes required per provider.

- **Sentinel starvation surfaced in the health sensor** — `sentinel_admission`
  tracks consecutive deferrals and wall-clock gap since last success. When the
  gap exceeds 300 s a WARNING is logged and the `SentinelHealthSensor` transitions
  to `"degraded"`, exposing the condition to HA automations and dashboards.

- **Chat LLM timeout raised from 90 s to 180 s** — the higher limit covers
  prefill + generation for typical conversation history lengths without the
  gate-wait overhead of the old approach.

- **Sentinel discovery capped at 45 s per LLM call** — discovery prompts can
  be large enough to consume 180 s+ of GPU time. A hard `asyncio.wait_for`
  timeout skips the cycle cleanly rather than monopolising the GPU.

- **Tool timeout sends a non-retryable error to the LLM** — `_run_langchain_tool`
  now returns a `TOOL_CALL_TRANSIENT_ERROR_TEMPLATE` message on `TimeoutError`
  instead of the generic retry-prompting error template. This stops the model
  from re-calling a tool that timed out due to a temporary resource constraint.

- **Qwen3 extended-thinking fallback** — when Ollama strips `<think>` tokens
  and returns empty content after a tool call, `_call_model` injects `"Done."`
  so the conversation does not go silent.

- **Tests** — 26 tests for the new admission-control primitives, plus coverage
  for the triage/discovery deferral paths, the transient tool-error helper, and
  the Qwen3 fallback.

## [3.12.2] - 2026-04-24

### Fixed

- **Chat streaming no longer hangs under heavy VLM load** — `_invoke_model` now
  wraps `model.ainvoke()` in `asyncio.wait_for(_LLM_INVOKE_TIMEOUT_S=90s)`. When
  the Ollama GPU is saturated by concurrent camera analysis, the chat LLM was
  queuing indefinitely inside `astream_events`, stalling the streaming pipeline
  and showing "no response" in the HA chat UI. The timeout converts an infinite
  hang into a bounded `HomeAssistantError`. Closes #378.

- **Blank chat bubble after timeout replaced with user-visible error message** —
  The timeout `HomeAssistantError` now propagates cleanly through
  `_stream_langgraph_to_ha` (bypassing the misleading "generator error" log path)
  to the recovery handler, which emits "I'm sorry, I was unable to respond in time.
  Please try again." instead of an empty bubble.

- **Empty-content fallback in `_async_handle_message` now shows an error message**
  — the `content=""` guard (hit when streaming fails with no recoverable graph
  state) now emits "I'm sorry, I was unable to respond. Please try again." so the
  chat UI always shows something actionable.

## [3.12.1] - 2026-04-24

### Fixed

- **LM Studio / local embedding server compatibility** — `OpenAIEmbeddings` now sets
  `check_embedding_ctx_length=False`, preventing `tiktoken` from converting text into
  integer token arrays before sending. Local servers (LM Studio, Ollama-compatible
  endpoints) expect plain strings and returned `400 - 'input' field must be a string
  or an array of strings`. Closes #375.

- **Removed `dimensions=` from local embedding calls** — the `dimensions` parameter
  is an OpenAI-specific truncation feature not supported by local models. Sending it
  caused `422 Unprocessable Entity` errors with models like `nomic-embed-text`.

- **Configurable embedding vector dimensions** — the openai_compatible model provider
  settings now expose an `Embedding dimensions` field (default 768, matching
  `nomic-embed-text`). Previously the dimension was hardcoded to 1024 for all
  providers, which caused pgvector index mismatches with 768-dim models. The value
  flows through provider settings into `PostgresIndexConfig(dims=N)`.

### Upgrade notes

If you are using an `openai_compatible` provider for embeddings and have already run
the integration (pgvector tables exist), you will need to drop and recreate the vector
store tables if you change the `embedding_dims` value:

```sql
DROP TABLE IF EXISTS store_vectors;
DROP TABLE IF EXISTS vector_migrations;
```

The tables are recreated automatically on the next startup.

## [3.12.0] - 2026-04-21

### Added

- **Native LLM streaming** — HGA now streams assistant tokens to the HA frontend
  in real time via `astream_events` + HA ChatLog delta API. Responses appear
  word-by-word instead of waiting for the full LLM response. Closes #370.
  Requires HA 2026.4.0 or later.

- **Parallel tool execution** — multiple tool calls within a single model turn
  now run concurrently via `asyncio.gather`, reducing multi-tool turn latency
  (e.g. `get_current_time` + `get_and_analyze_camera_image` in parallel).

- **LangChain tool timeout** — each tool call is bounded by a 30-second timeout
  via `asyncio.wait_for`. Stuck tool calls now produce a descriptive error
  instead of hanging the conversation indefinitely.

### Changed

- **Protected-action PIN flow hardened** — wrong-PIN entries no longer mark the
  pending action as resolved. Routing errors on `confirm_sensitive_action` are
  also treated as unresolved, preventing a bypass when the confirm tool is
  unavailable. Type annotations for PIN helper functions updated to
  `Sequence[AnyMessage]` for broader compatibility.

- **Streaming robustness** — LangGraph stream delta synchronization hardened:
  tool call IDs are correlated across `on_chat_model_end` and `on_chain_end`
  events; orphaned tool calls are flushed as synthetic rejections on generator
  exit; partial content is committed and recoverable on stream failure;
  CONTENT_ADDED is re-fired for the final AssistantContent so the HA frontend
  streaming UI shows the complete response. Empty chat logs on mid-stream error
  now produce a safe fallback instead of crashing the turn.

### Fixed

- HA intent-tool results containing `response_type` dict structures no longer
  produce an empty chat bubble in multi-tool turns.

- JSON tool results are correctly parsed; streaming failures fall back to graph
  state recovery to avoid empty responses.

## [3.11.1] - 2026-04-16

### Added

- **HA 2026.4.0 chatlog transparency (Show Details)** — HGA now populates the
  HA conversation chatlog after each agent turn, enabling HA's "Show Details"
  panel to display the full tool call / result chain. Each intermediate tool
  call (with arguments) and its result are visible as typed chatlog entries.
  Requires HA 2026.4.0 or later. Closes #369.

## [3.11.0] - 2026-04-14

### Added

- **Cyclical load sustained deviation gate** — refrigerators, freezers, and
  compressors now require sustained anomalous power draw before Sentinel fires
  a notification. Normal compressor cycling (e.g. ~944 W on/off every 30
  minutes) no longer triggers repeated "High energy consumption away" alerts.
  The gate duration is configurable in Sentinel settings (default 20 minutes,
  range 0–120 in 5-minute steps; set to 0 to disable).

- **Baseline minimum samples UI** — the "Baseline minimum samples before
  deviation alerts fire" setting is now exposed in the Sentinel configuration
  flow (range 1–500, default 20). Previously this required editing config
  entries directly.

- **Health sensor `cyclical_entities_gated` attribute** — the Sentinel health
  sensor now reports how many cyclical entities are currently held by the
  sustained deviation gate, useful for dashboards and debugging.

## [3.10.0] - 2026-04-13

### Added

- **RAG-based tool indexing and retrieval** — tools are embedded into pgvector at startup
  and retrieved per-turn by cosine similarity. Only the tools most relevant to the user's
  message are loaded into the agent's prompt, reducing prompt size and improving
  tool selection accuracy. On subsequent restarts, unchanged tools skip re-embedding
  via SHA-256 content hashing.

- **Tool Index Status diagnostic sensor** (`sensor.tool_index_status`) — exposes the
  current state of the tool index: `indexing`, `ready`, `failed`, or `unknown`. Use
  it in dashboards or automations to monitor embedding provider health.

- **Multi-intent query decomposition** — compound requests like "turn on the lights and
  check the back yard camera" are split into sub-queries and searched independently.
  Scores are aggregated by maximum per tool, so each intent gets the best-matched tools.

- **Actuation safety net** — when a message contains action keywords (turn on, lock, etc.),
  physical control tools are force-bound even if they fall below the relevance threshold.
  Capped at three-quarters of the retrieval limit so safety tools cannot crowd out
  RAG-retrieved results.

- **`tool_retrieval_limit` and `tool_relevance_threshold` options** — configurable in the
  Options flow. Retrieval limit (default `5`) sets the maximum tools per turn; relevance
  threshold (default `0.15`) sets the cosine similarity cutoff.

### Fixed

- **`tool_index_failed` guard** — a persistent embedding provider failure previously
  re-spawned a background index task on every conversation turn. The integration now
  sets `tool_index_failed` and skips further attempts until the next restart.

- **Reasoning mode crash on Qwen3/Ollama** — reasoning mode is now disabled before
  `bind_tools` to prevent a crash when using Qwen3 or other providers that do not
  support reasoning during tool binding.

- **`safe_convert` for unhashable selectors** — `voluptuous_openapi.convert` could crash
  on `SelectSelector` and other unhashable HA selector types. A `safe_convert` wrapper
  now handles these gracefully, extracting enum options where possible and defaulting
  to string type otherwise.

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

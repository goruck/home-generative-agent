# TODOS

## Agent

### "No LLM APIs" is not an expressible choice, and its two spellings behave oppositely

**What:** `CONF_LLM_HASS_API` has two distinct falsy storage states that mean the same thing to the user and opposite things to the code. Deselecting every API in the options flow hits `_cleanup_none_llm_api` (`config_flow.py:508`), which *pops* the key — and an absent key means "default to `[LLM_API_ASSIST]`" to every reader, so the Assist API is silently re-enabled and the user's choice is discarded. The v5 → v6 migration (`__init__.py:4042`) instead writes an explicit `[]` for the same intent, which really does disable everything. Same apparent user choice, opposite outcome, decided entirely by whether the entry predates v6. The options-flow form compounds it: `config_flow.py:213` suggests `[]` when the key is absent, so the UI shows "nothing selected" on installs that are actively running Assist.

**Why:** Surfaced twice during the v3.30.9 ship — once by inspection, once independently by the Codex adversarial pass, which correctly rejected an earlier claim that `[]` was unreachable. v3.30.9 makes the *security* consequence moot (the CONTROL flag is now also set whenever a PIN is configured, so neither spelling can bypass the gate), but the configuration semantics remain incoherent and a user who genuinely wants no LLM API still cannot express it through the UI.

**How to apply:** Pick one canonical spelling for "no APIs" and migrate the other to it. Storing `[]` is the better choice: it is explicit, it already round-trips through the migration, and it lets the form show the real state. That means deleting `_cleanup_none_llm_api`, adding a v6 → v7 migration that writes `[]` where the key is absent, and changing the readers' fallback from `[LLM_API_ASSIST]` to "only when the key was never present" — which, once the migration has run, is never. Note this is a user-visible behavior change: installs currently running Assist via an absent key would stop, so it needs a release note and probably a repair issue rather than a silent flip. `active_llm_api_ids()` (`agent/helpers.py`) is the single choke point for the read side.

**Effort:** M
**Priority:** P2

### PIN-gate add_automation for critical service calls

**What:** `add_automation` writes arbitrary automation YAML and reloads HA without ever passing the critical-action PIN gate: `_is_critical_action` inspects `domain`/`service` tool args, and `add_automation`'s payload is opaque `automation_yaml`. "Unlock the front door whenever I get home" installs a `lock.unlock` automation with no PIN, while the direct command "unlock the front door" is gated.

**Why:** Pre-existing hole, but v3.20.2's automation-intent force-binding makes the tool deterministically available on everyday phrasings, so the bypass path is now reliable. Converged on independently by both adversarial review agents (Claude adversarial + red team) during the v3.20.2 ship as the top finding. Deferred out of that branch by user decision (2026-07-24) so the PIN-flow change gets its own focused review and live validation.

**How to apply:** After `_async_validate_config_item` in `add_automation` (tools.py) — or in a langchain-branch guard in `_call_tools` (graph.py) — walk the parsed automation config's `action` list (including `choose`/`repeat`/`wait_for_trigger` nesting), extract each `service`/`action` call, and run it through `_is_critical_action` with the configured critical actions. If any matches and PIN is enabled, return the existing `requires_pin` ToolMessage flow (register in `pending_actions`) instead of writing the automation. Mind the prior pitfall: `confirm_sensitive_action` ToolMessages with `status=error` must not count as resolved.

**Resolution:** Gated in `add_automation` itself. `_async_validate_config_item`'s *return value* is now used (it was discarded): it is the normalized, blueprint-substituted config, so screening sees the actions HA will really run rather than the model's prose. New `agent/automation_pin.py` classifies every step with `cv.determine_script_action` and screens it — an **allowlist over HA's own action taxonomy**, not a blocklist of service names. Rule matching moved to `matches_critical_rule` in `agent/helpers.py`, shared with `_is_critical_action`.

The first draft blocklisted `action:`/`service:` steps only and was defeated by four bypasses found during pre-landing review, each verified against a real HA install by two independent reviewers: **device actions** (`device_id`+`domain`+`type`, no service key, maps to `lock.unlock`), **`scene.apply`** (target states inline, reproduced as `lock.unlock`), **indirection** (`script.turn_on`, bare `scene:`, `automation.trigger`, `event:`), and **group / entity-registry-ID targets** (ordinary-looking strings that resolve to entities named nowhere in the config, so `entity_match` substrings can't see them). A fifth, independent finding was a *rules* gap rather than a screening one: `cover.toggle` and `cover.set_cover_position` open a closed garage door and were not in `RECOMMENDED_CRITICAL_ACTIONS` — that hole applied to the direct-command gate too and is now closed.

Also: unknown action types fail closed (a future HA construct over-prompts rather than slipping through); `homeassistant.*` is re-screened against each target's own domain; the automation path refuses when the PIN is enabled-but-unset (unlike the one-off tool path, an automation persists); pending actions are claimed before the write and the store is swept and capped on registration. The parity test asserts absolute verdicts on both sides — the first version stayed green with the shared matcher stubbed to `return False`.

**Effort:** M
**Priority:** P1
**Depends on:** v3.20.2 (fix/tool-rag-automation-intent)
**Completed:** v3.26.0 (2026-08-05)

---

### Resolve script/scene indirection instead of gating it wholesale

**What:** `find_critical_automation_calls` (agent/automation_pin.py) fails closed on indirection: activating a scene, calling a script, triggering another automation, or firing an event all require the PIN, because the actions they run live in configuration the screen does not open. That is safe but blunt — an automation that calls a harmless script still prompts.

**Why:** v3.26.0 shipped the conservative half. An earlier draft *deferred* indirection entirely on the theory that it needs a pre-existing critical script; pre-landing review disproved that for `scene.apply` (which carries target states inline and needs no pre-existing artifact) and for device actions, so indirection moved from "deferred" to "gated". Resolving it properly would remove the over-prompting.

**How to apply:** Resolve `script.*` / `scene.*` targets against the live registries at screen time and recurse with a visited set and a depth cap; keep fail-closed for anything unresolvable. Decide first whether a re-screen is needed when the target script is later edited — the PIN would otherwise attest to something that has since changed, the same caveat that already applies to blueprint automations.

**Effort:** M
**Priority:** P3
**Depends on:** v3.26.0

---

### Raw protocol writes are not screened

**What:** Automation screening (`agent/automation_pin.py`) matches Home Assistant domains and services, so a service that addresses a device *beneath* the entity layer is invisible to it: `mqtt.publish` to a lock's command topic, `zwave_js.set_value` writing a Door Lock command class (as a service call or a device action), `zha.issue_zigbee_cluster_command`. Each can unlock a door without the automation naming the `lock` domain.

**Why:** Not gated by default — these are routine tools on their respective stacks, and gating them would prompt on ordinary MQTT/Z-Wave/Zigbee automations for the many users on those integrations. Documented in `docs/configuration.md` with per-transport opt-in rules users can add to `critical_actions`. Found by verification rounds 5 and 7 during the v3.26.0 ship.

**How to apply:** Needs a way to tell "this MQTT topic / Z-Wave command class targets a lock" from any other write, which means resolving the device behind the topic or node. Alternatively ship the transport rules as opt-in presets in the Sentinel/options UI so users can enable them without hand-editing `critical_actions`. Decide whether the prompt cost is acceptable before defaulting any of them on.

**Effort:** M
**Priority:** P3
**Depends on:** v3.26.0

---

### Re-screen blueprint automations on reload, or pin the expansion

**What:** A blueprint automation is persisted as a `use_blueprint:` reference and re-substituted by HA on every reload, so the PIN attests to the blueprint's contents at approval time only. Editing the blueprint file afterwards changes what the approved automation runs with no fresh prompt. Documented as a known limitation in v3.26.0's CHANGELOG and `docs/configuration.md`.

**Why:** Found by Codex during the v3.26.0 pre-landing review, which also caught the docs claiming the written config always equals the screened one. Persisting the expansion instead would fix the attestation but discards the blueprint indirection users chose, and re-screening on reload needs a hook this integration does not currently own.

**How to apply:** Either record a hash of the substituted config alongside the automation and re-screen on `automation.reload`, or persist the expansion with a comment pointing back at the source blueprint. Needs a product call on which surprises the user less.

**Effort:** M
**Priority:** P3
**Depends on:** v3.26.0

---

### Tie the screener's action-type lists to Home Assistant's schema

**What:** `automation_pin.py` classifies steps with `cv.determine_script_action` and then dispatches on `_CONTAINER_ACTION_TYPES` / `_INERT_ACTION_TYPES`, which are hand-maintained mirrors of HA's script taxonomy. An unknown type already fails closed, so a new HA construct causes over-prompting rather than a hole — but nothing tells a maintainer that HA grew a construct worth classifying properly.

**How to apply:** Add a drift test asserting `set(cv.ACTION_TYPE_SCHEMAS)` equals a locally-declared frozenset of reviewed action types, so an HA upgrade that adds one fails CI and forces a decision.

**Effort:** S
**Priority:** P3
**Depends on:** v3.26.0

---

### Alarm carve-out is asymmetric between direct calls and automations

**What:** `_is_critical_action` (graph.py) short-circuits `alarm_control_panel` to False — panels enforce their own code, not the critical-action PIN. The automation screen calls the shared `matches_critical_rule` directly and has no such carve-out, so a user who adds an `alarm_control_panel` rule to `critical_actions` gets an automation containing `alarm_disarm` gated while the direct disarm command is not.

**Why:** No effect on default installs (`RECOMMENDED_CRITICAL_ACTIONS` contains no alarm rule) and the asymmetry errs toward gating, which is the safe direction. But the rationale for the direct-call carve-out is weaker inside an automation, where the alarm code is embedded in the YAML and never re-entered by a human — arguably automations *should* be gated and the asymmetry is correct. Needs a deliberate decision rather than being an accident of where the carve-out sits.

**Effort:** S
**Priority:** P4
**Depends on:** v3.26.0

---

### Purge stale add_automation row from tool index when schema-first YAML is enabled

**What:** The tool indexer only `aput()`s changed tools and never deletes; a user who ran normal mode once has `hga_local::add_automation` in the vector store forever. After toggling schema-first YAML on, RAG ranking can still retrieve it even though dispatch excludes it, producing a confusing routing failure. Step 3d's guard is correct but the invariant it mirrors is unenforced for pre-existing rows.

**How to apply:** On entry setup with `CONF_SCHEMA_FIRST_YAML` true, `adelete` the `hga_local::add_automation` store key and drop its content hash; or filter `add_automation` out of RAG results when schema-first is active.

**Resolution:** The second option shipped structurally with #554's bind-time live-tool filter: in schema-first mode `add_automation` is excluded from `langchain_tools`, so `(hga_local, add_automation)` is not in the live set and the stale index row can never bind through RAG, safety, or the force-injection legs. The row itself still exists in the store (inert); physical deletion is folded into the "Tool index hygiene" eviction TODO.

**Effort:** S
**Priority:** P3
**Depends on:** v3.20.2
**Completed:** v3.30.4 (2026-08-15)

---

### Tool index hygiene: top-up cooldown, negative cache, stale-key eviction, delimiter-safe keys

**What:** The per-turn index top-up (#554, v3.30.4) trusts configured LLM APIs. Four hygiene gaps deferred from that ship's review: (1) a hostile or buggy MCP server that rotates tool names forces embedding writes and index growth on every turn — no per-api cooldown or per-turn cap; (2) a live key that persistently fails to index (schema that can't serialize) retries every turn — cheap in-memory since the top-up sources from loaded API instances, but there is no negative cache; (3) index rows for removed APIs and renamed tools are inert (the bind-time live filter excludes them) but never evicted, so the store and `tool_content_hashes` grow monotonically; (4) composite index keys `f"{api_id}::{name}"` don't escape `::`, so `a` + `b::c` collides with `a::b` + `c` — retrieval re-validates the stored `name`/`api_id` fields against the live set, so a collision can shadow a legitimate row but not spoof one.

**Why:** Security specialist + red team findings during the #554 ship review (2026-08-15). User decision: configured LLM APIs are inside the trust boundary (they already inject tool descriptions into the model context), so cap/cooldown machinery was deliberately deferred rather than added to the per-turn hot path.

**How to apply:** Per-api_id top-up cooldown (skip delta if last top-up for that api was < N minutes ago and produced no new hashes); negative cache of keys that repeatedly fail to index, with a retry deadline; periodic eviction sweep deleting store rows whose keys have not been live for N days; delimiter-safe key encoding (escape `::` or hash the pair).

Two adjacent gaps from the same review, same disposition (document + defer): (5) concurrent turns — while turn A's inline delta is writing, turn B from a *different* device-class short-circuits on `tool_indexing_in_progress` and proceeds without its own gated tools for that one turn (pre-fix behavior; self-heals on B's next turn); fixing needs per-key coordination or awaiting the active delta then recomputing. (6) a transient failure during the *startup* index run latches `tool_index_failed` until reload, which now also disables per-turn top-ups — startup deserves a retry/backoff instead of a one-shot latch.

And two writer-consistency gaps: (7) `_mark_tool_index_stale` (embedding-provider switch) clears hashes with no generation/epoch guard, so an in-flight index write that completes after the switch marks old-provider rows current (and the background leg re-latches `tool_index_ready`), silently mixing embedding spaces until restart — pre-existing on the startup leg, window widened by the per-turn delta writer; fix wants a generation counter checked before `update()`/`ready=True`. (8) a partially-failed delta write commits zero hashes, so already-written chunks are re-embedded next turn — converges, but a per-chunk `(task, key)` commit would stop burning embedding quota under a flaky provider.

**Effort:** M
**Priority:** P3
**Depends on:** v3.30.4

---

### Provider-gated schema normalisation vs mixed-provider fallback chains

**What:** `_format_and_dedupe_tools` gates its subtractive schema passes (OpenAI top-level-union flatten, Gemini anyOf-required sanitizer) on the statically configured primary provider, but `FallbackChatModel.bind_tools` (`core/fallback.py`) binds the same formatted tool list to every model in the chain. In a mixed-provider chain (e.g. Ollama primary with an OpenAI fallback), runtime failover hands the un-flattened top-level `anyOf` to OpenAI and reproduces the `HassStartTimer` schema 400 — and `_is_retryable` does not classify schema 400s as chain-advance errors, so the turn hard-fails instead of falling through. Symmetric mild case: an OpenAI primary that fails over hands the lossy flattened schema to a union-capable fallback.

**Why:** Found by red-team review during the v3.28.1 ship. The same latent pattern has existed for the Gemini sanitizer since v3.26.1 (#536); no field report yet, because it requires a mixed-provider chain plus (for the OpenAI leg) a timer-capable voice device. Fixing it properly is a `FallbackChatModel` restructuring, out of scope for the v3.28.1 fix.

**How to apply:** Format tools per chain member — have `FallbackChatModel.bind_tools` re-run the provider-gated normalisation per member provider type (chain entries already carry provider entry ids), or bind provider-specific tool lists when building the chain. Minimum viable: extend `_is_retryable` to treat invalid-function-schema 400s as chain-advance errors.

**Effort:** M
**Priority:** P2
**Depends on:** v3.28.1

---

### Tool-name collision hardening for force-injection guard

**What:** The step-3d injection guard and `_format_and_dedupe_tools` key on bare tool name. A remote (e.g. MCP) API exposing a tool named `add_automation` would suppress injection of the local tool and route automation YAML to the foreign tool (first-seen-wins dedupe predates v3.20.2). Consider `(api_id, name)` matching for the guard and explicit precedence in dedupe.

**Effort:** S
**Priority:** P4
**Depends on:** v3.20.2

---

### asyncio.gather concurrency policy for state-mutating tools

**What:** Add per-tool annotation or global policy for whether a tool is safe to run concurrently. State-mutating HA tools (`turn_on`, `turn_off`, lock, unlock, `alarm_control`) called in the same model batch could interleave under `asyncio.gather`.

**Why:** With `asyncio.gather` (introduced in feat/streaming-chatlog), a model turn that includes both `turn_on` and `get_state` may now run concurrently. `get_state` might return stale state if it completes before `turn_on` finishes. Previously sequential. Flagged during eng review Codex outside voice.

**How to apply:** In `graph.py`, add a `_SEQUENTIAL_TOOLS` set or per-tool `safe_to_parallelize` annotation. In `_call_tools`, run sequential tools before the gather batch, or use `asyncio.gather` only for tools not in the set. Alternatively, add a note in the integration docs that sequential ordering of state-changing + read-back calls requires separate model turns.

**Effort:** M
**Priority:** P3
**Depends on:** feat/streaming-chatlog

---

### Rename sync methods with `_async_` prefix in conversation.py

**What:** Three synchronous methods in `HGAConversationEntity` have the `_async_` prefix, which conventionally means "coroutine" in HA code: `_async_get_message_history` (line 629), `_async_get_all_tools` (line 685), `_async_render_system_prompt` (line 718). None use `await`.

**Why:** Misleading naming. Future maintainers may incorrectly assume these are coroutines.

**How to apply:** Rename each to drop `_async_` prefix and update all callers within `conversation.py`.

**Effort:** S
**Priority:** P3
**Depends on:** feat/streaming-chatlog

---

### Integration tests for streaming conversation path (HA fixture level)

**What:** Add four HA-fixture-level integration tests using a real (mocked) LangGraph + HA conversation entity: (1) single-turn text-only streaming, (2) multi-turn with tool calls, (3) PIN flow multi-turn with confirmation, (4) `schema_first_yaml=True` fallback fires `ainvoke` path correctly.

**Why:** Current test coverage (58%) is dominated by pure-function unit tests. The `HGAConversationEntity` integration methods (`_async_run_astream`, `_async_handle_message`, `_async_render_system_prompt`, `_async_init_llm_apis`) have zero unit test coverage. These tests were listed as required in the streaming design plan but deferred at ship time. Accepted risk per user override.

**How to apply:** Use the existing `test_conversation.py` integration test harness as a model. Create `tests/custom_components/home_generative_agent/test_conversation_stream_integration.py`. Mock `app.astream_events` to return a controlled sequence of events, verify delta sequence delivered to HA ChatLog.

**Effort:** M
**Priority:** P2
**Depends on:** feat/streaming-chatlog

---

### Integration smoke test: on_tool_end propagation during action node

**What:** After feat/streaming-chatlog lands, run a real multi-tool conversation (`get_current_time` + `get_and_analyze_camera_image`) and verify that `on_tool_end` for `get_current_time` fires BEFORE the camera tool completes.

**Why:** The streaming win depends on LangGraph propagating child `on_tool_end` events from `lc_tool.ainvoke()` to the outer `astream_events` DURING node execution. Verified against LangGraph 1.1.2 source during planning, but not confirmed via integration test. If LangGraph buffers nested events until node completion, the streaming gain disappears.

**How to verify:** Add a timing log in the `on_tool_end` handler (DEBUG level). Time delta between `on_tool_end` for the time tool and the camera tool should be ~3980ms apart, not ~0ms.

**Effort:** S
**Priority:** P2 (post-ship validation)
**Depends on:** feat/streaming-chatlog
**Completed:** v3.12.0 (2026-04-21)

---

### Expose-aware camera resolution and enumeration

**What:** `_resolve_camera_entity_id` and `_available_camera_names` (agent/tools.py) scan every `camera.*` state with no Assist expose filtering. The not-found hint lists every camera name — including entities deliberately hidden from the conversation LLM API — and resolution captures any camera by name.

**Why:** Cross-model adversarial finding on the #502 camera resolver (Codex rated P1). Deferred at ship time (2026-07-23, user decision): strict `async_should_expose` filtering would break camera analysis on default installs, because HA does not expose cameras to Assist by default. Capture-by-name reach is pre-existing behavior; the new surface is the name enumeration. Needs a deliberate design — likely an opt-in option (`respect Assist expose settings`) or filtering the hint list only, with migration notes.

**How to apply:** Evaluate `homeassistant.components.homeassistant.exposed_entities.async_should_expose(hass, "conversation", entity_id)` for both the resolver candidate set and the hint list, behind a config option; decide the default with reporter input.

**Effort:** M
**Priority:** P2

---

### Sentinel sync sampling-retry can outlive the triage timeout

**What:** `run_sentinel_model_call`'s executor leg now runs `invoke_dropping_unsupported_params` in the worker thread. The thread already outlives `asyncio.timeout` cancellation (threads can't be cancelled); the drop-retry can add up to two more provider HTTP calls after the sentinel caller has timed out and moved on.

**Why:** Codex adversarial note on #502. Bounded (max 2 extra calls, only when a provider rejects sampling params — a misconfiguration state that also logs warnings), and the underlying thread-outlives-timeout shape is pre-existing. Worth a deadline check between retry attempts if field reports show thread/request pile-ups.

**How to apply:** Pass a deadline (monotonic timestamp) into the sync helper and skip the retry when it has passed.

**Effort:** S
**Priority:** P3

---

### Generalize evidence-entity exclusion instead of per-rule constructor injection

**What:** `CameraEntryUnsecuredRule` takes an `is_entity_excluded` callback so it can honor `sentinel_rule_entity_exclusions` for entities that live only in `evidence`. `_filter_excluded_findings` inspects `triggering_entities` only, so any other rule with the same shape has the same gap. `alarm_disarmed_external_threat.py` already does: it puts `alarm_entity_ids` (all disarmed panels) in evidence while `triggering_entities` carries only the primary panel and the camera, so a secondary excluded phantom panel still reaches the notification and the audit record.

**Why:** Red-team finding on PR #544. The constructor-injection pattern does not generalize — every new rule with entity-bearing evidence has to remember to opt in, and the UI copy promises exclusion applies everywhere. A shared post-rule pass would make new rules inherit the behavior.

**How to apply:** Add a scrub step in the engine between rule evaluation and `_filter_excluded_findings` that walks a registry of known entity-ID-bearing evidence keys (`unsecured_entities`, `unsecured_entity_areas`, `alarm_entity_ids`), drops excluded entities, and drops the whole finding when scrubbing empties the set. Then remove the per-rule callback. Watch the anomaly-id hash: `camera_entry_unsecured` hashes `unsecured_same_area`, so scrubbing at a different layer changes ids unless the hash input is taken pre-scrub.

**Effort:** M
**Priority:** P2

---

### Basic setup silently wipes Sentinel entity exclusions and camera entry links

**What:** `async_step_basic_settings` does `data = _default_payload()` and its schema exposes only four fields, so re-running Basic setup resets `sentinel_rule_entity_exclusions` and `sentinel_camera_entry_links` to `{}`. The overwrite warning says settings will be overwritten with recommended defaults but does not name these two.

**Why:** Pre-existing, but PR #544 changed the exposure: exclusions used to be an advanced-only JSON field, so the population that configured them and the population that runs Basic setup did not overlap. A friendly entity picker is exactly what a Basic-setup user will configure and then destroy, and the symptom — phantom alerts returning with no visible cause — is hard to attribute.

**How to apply:** Carry `CONF_SENTINEL_RULE_ENTITY_EXCLUSIONS` and `CONF_SENTINEL_CAMERA_ENTRY_LINKS` over from `current.data` in the basic path, or name them explicitly in the `sentinel_overwrite_warning` string.

**Effort:** S
**Priority:** P2

---

## Explain / Prompts

### Sanitize area/entity strings before injecting into LLM prompts

**What:** Strip or truncate user-editable strings (area names, entity IDs, friendly names) to a safe character set before including them in `USER_PROMPT_TEMPLATE` evidence interpolation.

**Why:** HA area names are user-editable via UI or YAML and flow into `USER_PROMPT_TEMPLATE.format(evidence=...)` as raw Python dict repr without sanitization. A malicious area name (e.g., `"Front\nIgnore all previous instructions."`) constitutes a prompt injection vector. The `camera_area` and `unsecured_entity_areas` fields added in PR fixing camera_entry_unsecured notifications increase the surface area (same area string, multiple occurrences). Flagged as P3 since it requires a coordinated attacker with HA admin access to be exploitable.

**How to apply:** Add a sanitization helper in `explain/` that replaces non-printable characters and control sequences in string values before dict repr serialization. Alternatively, serialize evidence as JSON with explicit schema validation rather than using Python repr. Add a test with a crafted area name containing injection characters.

**Effort:** S
**Priority:** P3
**Depends on:** None

---

## Sentinel Triage

### Triage is unreachable: no UI field, not in resolver allowlist — expose or remove

**What:** `sentinel_triage_enabled` (and `sentinel_triage_timeout_seconds`) has no config-flow field, no strings.json label, and is absent from `_apply_sentinel_options`' defaults dict in `core/subentry_resolver.py` — so it cannot be set from the UI and would be ignored even if injected into Sentinel subentry data (same allowlist-omission class as #480, sitting latent). The only live path is a legacy top-level entry option surviving through `resolve_runtime_options`' `{**entry.data, **entry.options}` base (`subentry_resolver.py:184` → `__init__.py:2723`). Default is `False`, so the entire `SentinelTriageService` (#262) is dormant on virtually every install.

**Why:** Surfaced during PR #523 review (2026-07-31): the PR wires a response-language option into a service no user can enable. Either the feature earns a UI switch (Sentinel subentry field + resolver allowlist entry + strings/en/cs labels + docs) or it should be removed rather than shipped dark. Decide product-first: triage adds an LLM call per finding for suppress-only value that quiet hours + cooldowns already partially cover.

**How to apply:** If exposing: add `CONF_SENTINEL_TRIAGE_ENABLED`/`CONF_SENTINEL_TRIAGE_TIMEOUT_SECONDS` to `_apply_sentinel_options` defaults, the Sentinel subentry schema (`flows/sentinel_subentry_flow.py`), and translations; add resolver plumbing tests (the #480 regression shape). If removing: delete `SentinelTriageService` wiring from `__init__.py`, the engine triage branch, and the docs rows.

**Effort:** M
**Priority:** P2
**Depends on:** None

---

### docs/sentinel.md documents `sentinel_triage_enabled` as a settable option

**What:** `docs/sentinel.md:173` and the option table at `docs/sentinel.md:185` (plus `docs/constants.md:337`) present `sentinel_triage_enabled` as a normal config option, but there is no UI or supported path to set it (see previous item).

**Why:** Users following the docs will look for a switch that does not exist (field report shape: "there is no sentinel UI switch called LLM triage" — exactly what surfaced this). Docs should not describe unreachable configuration.

**How to apply:** Until the expose-or-remove decision lands, annotate the rows as "not yet exposed in the UI (engine support only, #262)". Resolve fully when the previous item is done.

**Effort:** S
**Priority:** P3
**Depends on:** Triage expose-or-remove decision (previous item)

---

### Drop or wire the dead `summary` field in the triage JSON contract

**What:** The triage system prompt requires a `summary` field (`sentinel/triage.py:91`, 120-char cap) and the parser extracts it (`triage.py:337`, trimmed to 240) into `TriageDecision.summary`, but no consumer exists anywhere — the engine reads only `decision`/`reason_code`/`triage_confidence` (`sentinel/engine.py:962-964`); the only appearance is a `LOGGER.debug` line. Tokens are spent generating a sentence that is discarded every run.

**Why:** Found during PR #523 review — the PR's triage-language leg was a no-op because of this. Either the summary earns a consumer (attach to the audit record as triage rationale, and/or the notification body) or it should leave the prompt contract (cheaper, faster, one less parse field).

**How to apply:** Preferred: persist it on the audit record next to `triage_decision`/reason code (audit consumers already carry triage fields since PR #511) and surface it in the Sentinel UI card's finding detail. Otherwise: remove `summary` from `_SYSTEM_PROMPT`'s schema, the parser, and `TriageDecision`.

**Effort:** S
**Priority:** P3
**Depends on:** Triage expose-or-remove decision (first item — pointless if triage is removed)

---

## Sentinel Rules

### Unknown-person rules are inert on installs without face recognition

**What:** The stranger-present predicate (unknown-person rules fix, 2026-08-22) makes `unknown_person_camera_no_home`, `unknown_person_camera_night_home`, `alarm_disarmed_during_external_threat`, and the two dynamic unknown-person templates fire only on a positive `"Unknown Person"` label — which only the face-recognition pipeline produces. On installs without a configured face service these five rules never fire, and `alarm_disarmed_during_external_threat` previously fell back to motion/VMD/summary evidence, so an upgrade silently narrows real coverage there. Docs note it, but nothing at runtime does. Same family, narrower trigger (Codex structured review): the snapshot sources `recognized_people` solely from the `image.*_last_event` entities, so a user who disables that entity in the entity registry (it vanishes from `hass.states`) silently makes the rules inert even WITH face recognition running. Surfaced independently by the security specialist and both Codex passes during the ship of the fix.

**How to apply:** Detect at engine start whether face recognition is configured (face-service URL option); if not, raise a one-time repair issue (or rate-limited log) saying these rules are inert and pointing at the `motion_detected_*` discovery templates as the replacement. Alternatively carry a `face_recognition_enabled` flag in the snapshot and let the rules fall back to a motion-evidence predicate when it is false. For the disabled-entity sub-case, source recognition metadata from an always-on runtime cache (e.g. VideoAnalyzer._last_recognized via runtime_data) instead of the optional UI entity — that also removes the image-entity round-trip entirely.

**Effort:** M
**Priority:** P2
**Depends on:** unknown-person rules fix (2026-08-22)

### Companion suppression works on the event-level identity union, not frame co-occurrence

**What:** The accompanied-guest gate suppresses an `"Unknown Person"` sighting whenever an enrolled name appears anywhere in the same analyzed event's `recognized_people` union — including an intruder presenting a photo of a resident in one frame, or a resident passing through the clip window minutes before the stranger. Kept deliberately for now: recognition flapping on one person produces exactly the `[known, "Unknown Person"]` refused-merge shape, and firing on it would re-create the #543 phantom-stranger alerts. Surfaced by both the security specialist and the Codex adversarial pass.

**How to apply:** Carry frame-level co-occurrence evidence into the snapshot (e.g. a `companion_confirmed` flag set only when a known name and an unknown share a single frame, which `_merge_unknown_faces` already computes as its condition-2 check) and suppress only on confirmed co-occurrence; or fire a low-severity variant instead of full suppression when both labels are present. Needs live tuning against #543-style flapping before changing the default.

**Effort:** M
**Priority:** P2
**Depends on:** unknown-person rules fix (2026-08-22)

### Legacy gallery rows with near-reserved names defeat label classification

**What:** `RESERVED_IDENTITY_LABELS` matching is exact strip+lowercase. Gallery rows enrolled before the reserved-name guard (or via direct DB insert) under variants like `"unknown-person"`, internal double spaces, or homoglyphs are classified as enrolled identities — a face-API match to such a row suppresses the unknown-person rules via the accompanied-guest gate; conversely a legacy row literally named `"Unknown Person"` makes an enrolled resident fire stranger alerts on every fresh sighting. Surfaced by the security specialist during the ship of the unknown-person rules fix.

**How to apply:** Add a startup/migration sweep over `person_gallery` that flags (or renames with user confirmation) rows whose collapsed-whitespace lowercased name is in — or one edit away from — `RESERVED_IDENTITY_LABELS`; optionally normalize with internal-whitespace collapse on both the enrollment guard and the Sentinel matchers so both ends agree.

**Effort:** S
**Priority:** P3
**Depends on:** unknown-person rules fix (2026-08-22)

### Dynamic `sensor_threshold_condition` rules don't normalize units

**What:** Discovery's `sensor_threshold_condition` template (`sentinel/proposal_templates.py`) compares an LLM-extracted numeric threshold against the sensor's native state with no unit normalization — the #461 bug class: a rule meaning "over 100 watts" against a kW sensor never fires (the template only extracts above-thresholds today; a below-variant would invert the failure — always firing). Arguably the user's threshold is native-unit by intent, but nothing disambiguates. Surfaced by adversarial review during the v3.21.3 ship.

**How to apply:** When the target sensor has `device_class: power` (or a power unit), normalize both the threshold and the reading via `sentinel/power_units.watts_per_unit` — requires deciding/recording the threshold's intended unit at proposal-approval time (candidate text usually names it: "100 W", "1.5 kW").

**Effort:** M
**Priority:** P3
**Depends on:** v3.21.3

### Add `sentinel_camera_entry_links` config for explicit camera-to-entry mapping

**What:** Add a `sentinel_camera_entry_links` config key (Sentinel subentry options flow) that allows users to explicitly associate cameras with entry sensors regardless of HA area assignment. Format: `{camera_entity_id: [entry_entity_id, ...]}`.

**Why:** Removing the home-wide fallback from `camera_entry_unsecured` (PR fixing cross-area false spatial claims) creates false negatives for adjacent-area setups — e.g., a driveway camera in "Outside" area should still fire when the front door lock in "Front" area is unsecured. Area-based association is insufficient for these layouts. Flagged as an accepted trade-off during eng review and Codex outside voice.

**How to apply:** In `const.py`, add `CONF_SENTINEL_CAMERA_ENTRY_LINKS`. In the Sentinel config flow subentry, add an optional text/JSON field. In `camera_entry_unsecured.py`, after the same-area unsecured lookup, check the config for explicit links for the current camera; merge any linked entities into `unsecured`. Add `unsecured_entity_areas` entries for the linked entities with their actual areas.

**Effort:** M
**Priority:** P2
**Depends on:** None
**Completed:** v3.8.0 (2026-04-05)

---

### Add `unknown_person_camera_night_home` branch to `_covered_builtin_rule_for_candidate()`

**What:** Add a detection branch in `_covered_builtin_rule_for_candidate()` (`__init__.py`) that recognizes a discovery candidate as already covered by the `unknown_person_camera_night_home` static rule.

**Why:** The function currently has branches for vehicle-near-camera and camera-missing-snapshot, but not for unknown-person-at-night-on-camera. The LLM is prevented from re-suggesting this topic via `_STATIC_RULE_IDS` in `discovery_engine.py`, so the missing branch has no user-visible impact under normal operation. However, if the LLM ignores the exclusion list and generates an `unknown_person` candidate, the proposal approval flow won't detect the overlap with the existing static rule — the candidate will be treated as novel and sent to the UI card instead of being silently filtered.

**How to apply:** In `_covered_builtin_rule_for_candidate()`, add a branch after the vehicle and camera-snapshot branches: if `camera_entity is not None` AND `("unknown" in text or "unrecognized" in text or "stranger" in text)` AND `"night" in text` AND `any(term in text for term in ("home", "resident", "occupant"))`, return `("unknown_person_camera_night_home", [camera_entity])`. Add a corresponding test in `test_sentinel_services.py`.

**Effort:** S
**Priority:** P3
**Depends on:** Static rule generalization PR (sentinel static rule generalization sweep)

---

### Extract shared motion-evidence helper across motion evaluators

**What:** `_eval_motion_detected_at_night_while_away`, `_eval_motion_detected_at_night_while_alarm_disarmed`, and `_eval_motion_while_alarm_disarmed_and_home_present` repeat the params list-check / entity resolution / `motion_states` evidence block (the new evaluator now uses per-entity any-of resolution while the alarm two remain all-of `zip(..., strict=False)`). A shared helper would encode the resolution invariant once — and is the natural place to extend any-of resilience to the alarm-motion evaluators.

**Effort:** S
**Priority:** P3
**Depends on:** v3.22.0

---

## Audit Store

### Audit notifier-drop findings as non-user-facing (delivery status from async_notify)

**What:** The engine audits every finding that reaches the notify path as `not_suppressed`, but `SentinelNotifier.async_notify` can still drop or defer it internally: the per-finding 30-minute cooldown (`_FINDING_COOLDOWN_SECS`) returns silently for non-high severities, and the rate limiter buffers findings into `_held_batch` (delivered later as a summary without action buttons). Those records are non-evictable and counted by the daily digest and notified KPIs despite no individual notification being delivered. Structural cousin: `async_append_finding` stamps `notification.notified_at` on every record unconditionally (audit/store.py), which is why all "notified" consumers must filter on `suppression_reason_code` instead.

**Why:** Same class of KPI inflation and buffer clogging that PR #511 fixed for triage/policy suppression — found by Codex review of #511 (engine.py notify path → notifier.py cooldown/batch returns). Trigger: type/entity cooldowns configured below 30 min, finding re-fires between the engine and notifier cooldown windows.

**How to apply:** Have `async_notify` return a delivery status (`delivered` | `cooldown_dropped` | `batched`); in the engine's final `_append_finding_audit` call, map non-delivered statuses to a new reason code (e.g. `notifier_cooldown`, reusing the evictable-by-default semantics). Consider only stamping `notified_at` for delivered records (audit consumers already gate on reason code, but tests assert the stamp — sweep them).

**Effort:** M
**Priority:** P2
**Depends on:** PR #511 (reason-code relabeling)

---

### Persist notified anomaly_id so compound responses attach to the right record

**What:** `async_update_response` matches compound records by ANY constituent `anomaly_id`, but the notification only carries the highest-confidence constituent's id. If an older compound C1 notified via constituent A, and a newer compound C2 also contains A but notified via higher-confidence B, a late user response for A attaches to C2 — wrong compound, corrupted response KPIs and false-positive attribution.

**Why:** Found by Codex review of #511. Pre-existing; #511's not_suppressed preference doesn't change it (newest-match semantics already existed). Needs a schema touch, so it deserves its own change.

**How to apply:** Record the dispatched constituent's `anomaly_id` in the audit record at append time (e.g. `notification.notified_anomaly_id`, set from `best.anomaly_id` in `_dispatch_compound` / `finding.anomaly_id` in the simple path). In `async_update_response`, match on `notified_anomaly_id` first, falling back to the current constituent scan for old records.

**Effort:** S
**Priority:** P3
**Depends on:** PR #511

---

### Per-rule precision tracking with retirement flagging

**What:** Health-sensor KPIs (`_compute_kpis` in `core/sentinel_health_sensor.py`) are aggregate-only: `false_positive_rate_14d`, `user_override_rate`, and `triage_suppress_rate` are computed across all findings, so a single decaying rule is invisible until it drags the global numbers. Add per-rule breakdowns keyed by finding type (and dynamic-rule id where available): notified count, user-response count, false-positive count, snooze/dismiss count. Flag rules whose per-rule FP or snooze rate exceeds a threshold as retirement candidates.

**Why:** Committed publicly as roadmap in the 2026-07-27 LinkedIn reply on rule maintainability ("per-rule precision tracking that auto-flags candidates for retirement is the natural next step"). The suppression layer already lets users retire rules via permanent snooze, but nothing proactively tells them which rules deserve it — decay is only discoverable by annoyance, which contradicts the maintainability story. Snooze/dismiss feedback is already captured per rule+entity (`record_cooldown_feedback`, keyed `rule_type:entity_id`), so most of the signal exists.

**How to apply:** In `_compute_kpis`, accumulate a `per_rule_stats: dict[str, dict]` keyed by `finding.type` (fall back to compound constituents' types for compound records), counting notified / responded / false-positive / snoozed per key from the same audit-record fields the aggregate KPIs use. Expose a `rules_flagged_for_review` attribute on `sensor.sentinel_health` listing keys whose 14-day FP rate or snooze rate exceeds a configurable threshold (min-sample gate, e.g. ≥5 notified, to avoid flagging on one bad night). Optionally surface flagged dynamic rules in the Sentinel proposals card with a one-tap disable. Mind attribute size limits — cap the exposed dict to worst-N rules.

**Effort:** M
**Priority:** P2
**Depends on:** PR #511 (reason-code relabeling — per-rule notified counts need trustworthy `suppression_reason_code`)

---

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

### Store baselines unit-normalized (or reset on unit change)

**What:** `sentinel_baselines` rows store native-unit values with no unit column; evaluation interprets the stored value in the entity's *current* unit. After a W→kW sensor reconfiguration, a stored `1200` (W) baseline against a current `1.2` (kW) reading produces a false ~99.9% deviation (misclassified as cycle completion), and new `1.2` samples blend into the old `1200` rolling average, corrupting rolling/hourly/DOW metrics until the averages re-converge (rolling/hourly are EMA-based; DOW slots use Welford accumulators). Pre-existing design property; surfaced by Codex adversarial review during the v3.21.3 ship.

**How to apply:** Either normalize power-class samples to watts at write time in `SentinelBaselineUpdater` (with a one-time migration or metric-version bump), or store `unit_of_measurement` alongside the value and reset the row when the recorded unit differs from the current one.

**Effort:** M
**Priority:** P3
**Depends on:** v3.21.3

---

### Bound the power-enrichment recorder walk

**What:** `async_enrich_power_last_changed` fetches up to 30 days of `state_changes_during_period` rows with `limit=None` and `no_attributes=False` for every on-state power sensor, every Sentinel cycle. A high-frequency (~1 Hz) power sensor materializes millions of rows per cycle; several such sensors can exhaust memory or monopolize recorder executor workers. Pre-existing for `device_class: power` sensors; v3.21.3 marginally broadened admission (unit-only sensors). Surfaced by Codex adversarial review.

**How to apply:** Walk history in capped descending pages (e.g. `limit=` batches, newest first) and stop at the first off-boundary; or cache the resolved on-since per entity and only re-query when the sensor drops below the off level.

**Effort:** M
**Priority:** P2
**Depends on:** None

---

### Config flow UI for CONF_SENTINEL_BASELINE_MIN_SAMPLES

**Completed:** v3.11.0 (2026-04-14)

`NumberSelector` added to `sentinel_subentry_flow.py` (min: 1, max: 500, step: 1, default: 20). Also added `sentinel_baseline_sustained_minutes` selector in the same PR.

---

### Incident lifecycle control for repeated deviation notifications

**What:** Replace per-entity-run notification tracking with a stable incident abstraction: key per `entity_id + template_id`, hold incident open until entity returns below threshold, notify once per incident. Suppresses repeated alerts for any entity, not just named cyclers. Clear incident when the entity is absent from findings for one full run.

**Why:** The v3.11.0 cyclical load gate (fridge/freezer/compressor) fixes the immediate fridge spam problem, but the root cause is deeper: Sentinel lacks any concept of "same condition still active." Every run where an entity is above threshold produces a new finding, and only the cooldown prevents repeated notification. The incident abstraction fixes this for all entities without requiring appliance name classification.

**How to apply:** Add `_open_incidents: dict[str, IncidentState]` to `SentinelEngine.__init__`. `IncidentState` holds `opened_at`, `last_seen_at`, `notified: bool`. In a new `_apply_incident_control()` method, gate all findings: suppress if `notified=True` and entity still in `_open_incidents`; fire if not yet notified; clear if entity absent this run.

**Effort:** M
**Priority:** P2
**Depends on:** Cyclical load gate (v3.11.0)

---

### Cyclical load gate: notification body with duration

**What:** When the sustained gate fires, include elapsed time in the notification body ("Fridge compressor has been running for 22 minutes — possible problem?"). Reads `_cyclical_deviation_above_since[entity_id]` and formats elapsed time.

**Effort:** S
**Priority:** P3
**Depends on:** Cyclical load gate (v3.11.0)

---

### Expand CYCLICAL_LOAD_HINTS to include HVAC/heat/AC/water heater

**What:** Add `hvac`, `heat`, `heatpump`, `aircon`, `airconditioner`, `waterheater`, `tankless` to `CYCLICAL_LOAD_HINTS` in `sentinel/baseline.py`.

**Why:** These appliances cycle normally (HVAC compressor, water heater element) and can generate the same notification spam as fridges. Deferred from v3.11.0 because an HVAC running at 3am away-mode IS an anomaly worth surfacing — the suppression tradeoff is non-trivial and needs evaluation before gating.

**How to apply:** Before expanding hints, evaluate false-negative rate by checking whether any real HVAC away-mode anomalies have been observed in production. Add hint only if HVAC cycling during normal occupancy is reliably distinguishable from anomalous HVAC usage.

**Effort:** S
**Priority:** P3
**Depends on:** Cyclical load gate (v3.11.0), field observation data

---

### Weekly / day-of-week baseline patterns

**What:** Extend baseline collection to store `hourly_avg_{DOW}_{H}` metrics (e.g., `hourly_avg_1_14` = Monday 2PM). Gives 7×24=168 time slots per entity instead of 24, enabling time-of-day anomaly detection that accounts for weekday vs. weekend patterns.

**Why:** The current `hourly_avg_H` treats all Mondays and Sundays at 2PM the same. For most households, weekday and weekend patterns differ significantly (cooking appliances, HVAC, occupancy). A washing machine running at 3AM on a Saturday is less anomalous than at 3AM on a Tuesday. Without DOW awareness, `time_of_day_anomaly` generates false positives on weekends.

**How to apply:** Add `hourly_avg_{DOW}_{H}` as a third metric row per entity per update cycle. Update `evaluate_time_of_day_anomaly()` to prefer the DOW-specific metric when available, falling back to the global `hourly_avg_H` if not yet established. New config option `CONF_SENTINEL_BASELINE_WEEKLY_PATTERNS` (default: False) to opt in.

**Effort:** M
**Priority:** P2
**Depends on:** Baseline enhancement PR
**Completed:** v3.9.0 (2026-04-06)

---

## Discovery

### Sanitized candidate IDs can collide across distinct sensors

**What:** `_strip_env_context_id_tokens` (discovery_semantic.py) folds occupancy/time-of-day tail tokens off environmental candidate IDs — not just `*_away`/`*_day`/`*_night` but the full `_ENV_CONTEXT_ID_TOKENS` set (`home`, `daytime`, `nighttime`, `occupied`, `unoccupied`, `overnight`, `present`, …) plus dangling connectives, so e.g. `..._deviation_home` and `..._deviation_away` both collapse to `..._deviation` — so two candidates with different sensors (distinct semantic keys, both stored) can converge on one `candidate_id`; `discovery_store.find_candidate` returns the newest match and `proposal_store` lookups are ID-keyed, so promoting one card could resolve to the other candidate or report a false existing-rule collision (Codex adversarial, env-context-sanitizer ship 2026-08-15).

**Why:** Pre-existing hazard class — the LLM itself reuses generic IDs across cycles (a stored draft literally carries `candidate_id: "c1"`) — and `find_candidate`'s newest-first order resolves to the card the user most recently saw, so the sanitizer only widens the window slightly. Not worth blocking the ship; worth closing structurally.

**How to apply:** Batch-level guard in `_filter_novel_candidates` (revert the ID strip when the sanitized ID would collide with a different semantic key in the same batch), plus make `find_candidate`/promote verify the semantic key when both a candidate_id and key are known.

**Effort:** S
**Priority:** P3
**Depends on:** —

---

### Domainless legacy battery object-IDs register rules the candidate key never covers

**What:** `_find_battery_sensor_entity_ids` (proposal_templates.py) accepts domainless legacy object IDs containing "battery" (`entities[entity_id=zamek_baterie_battery]`), so the normalizer registers a `low_battery_sensors` rule keyed `entities=zamek_baterie_battery` — but `candidate_semantic_key`'s extraction regexes require `domain.object` shapes, so the candidate keys `None` and dedups only by identity hash. The activated rule can never cover re-proposals of the same idea worded differently.

**Why:** Adversarial finding during the battery-context ship (2026-08-10), empirically confirmed; pre-existing (conjunctive prose had the same hole) and low-frequency (legacy drafts only). Fixing extraction to accept domainless tokens risks minting pseudo-entities from snapshot paths; needs the same shape-gating care as the #522 bare-bracket work.

**How to apply:** Accept domainless battery-named IDs in `discovery_semantic._named_battery_sensor_entity_ids` AND extend `_extract_entity_ids` (or a battery-leg-scoped sidecar) to surface domainless `entity_id=` tokens, gated to the battery leg so no other subject leg keys pseudo-entities. Pin with a key==rule-key test for a domainless battery candidate.

**Effort:** S
**Priority:** P3
**Depends on:** —

---

### Consider deriving candidate keys from the normalizer's actual routing instead of the hand-maintained mirror

**What:** `candidate_semantic_key` hand-mirrors `normalize_candidate`'s branch order, and the battery-context ship (2026-08-10) was roughly the sixth mirror-asymmetry fix recorded in the module's own comments (#516, #518, #522, #524 rounds). The adversarial reviewer's structural suggestion: when normalization succeeds, derive the candidate's coverage key from the normalized rule itself (`rule_semantic_key(normalized)`), falling back to the textual key only for unsupported candidates — the whole branch-order drift class disappears. Two pre-existing mismatch shapes confirmed by Codex during that ship (both base-parity, deliberately NOT patched into the mirror): (1) the conjunctive battery leg fires before any mirror of the normalizer's text-derived entry branches, so a locale-named entry sensor plus an incidental battery sensor with "battery … below" prose keys `low_battery` while normalization registers `open_entry_while_away` — an existing hub-battery rule then silently suppresses the unrelated security candidate; (2) battery evidence plus incidental camera/motion evidence with pure battery prose keys `power_anomaly` (the sensor-only gate blocks the battery arm) while normalization still registers `low_battery_sensors`, so those candidates re-propose after approval.

**Why:** Each new template or branch reorder currently requires a matching key-chain edit, and misses surface as silent dedup breakage (infinite re-proposal or cross-suppression). The decoupling convention exists to avoid importing the normalization module into the key module; inverting the dependency (engine calls normalize first, keys from its result) respects that boundary.

**How to apply:** In the discovery engine's novelty filter and semantic-context builders, attempt `explain_normalize_candidate(candidate)`; on success key via `rule_semantic_key` of the would-be rule, else fall back to `candidate_semantic_key`. Keep the textual key for history-record hashing so stored keys stay comparable. Requires re-verifying every dedup test and the coverage semantics for unsupported candidates.

**Effort:** M
**Priority:** P3
**Depends on:** —

---

### Route battery-named measurement streams (sensor.battery_power) away from the low_battery_sensors template

**What:** A gap-hinted baseline candidate for a battery-named measurement stream (e.g. `sensor.battery_power` on a home battery, Watts) normalizes to `low_battery_sensors` with the 40% default threshold: `_find_battery_sensor_entity_ids` matches any `sensor.*` containing "battery" with no measurement-token exclusion, and baseline prose inevitably contains the word "battery". The rule never registers — `_is_battery_like_state` rejects it at approval (`device_class: power`, unit W → `not_battery_sensor`) — so the user sees an honestly-unsupported proposal instead of a false-firing rule, but the candidate can never become the baseline/threshold-power rule it should be.

**Why:** Testing-specialist finding during the battery-context ship (2026-08-10), empirically reproduced; pre-existing (#522 code), not introduced by that diff. Fixing it means excluding measurement-stream-token battery IDs from the normalizer's battery collection AND mirroring that in `candidate_semantic_key`'s battery arm plus the card's `_lowBatteryContext` — a three-surface #522-hardened change that deserves its own review cycle.

**How to apply:** In `proposal_templates._find_battery_sensor_entity_ids`, skip IDs matching `_BATTERY_MEASUREMENT_STREAM_TOKENS` (the tuple now in `discovery_semantic.py`); mirror the same exclusion in `discovery_semantic._named_battery_sensor_entity_ids` and verify the non-battery power branch then routes such candidates to `sensor_threshold_condition`/`baseline_deviation`. Pin with a `normalize_candidate` test asserting the chosen template for a `sensor.battery_power` baseline candidate.

**Effort:** S
**Priority:** P3
**Depends on:** —

---

### Mirror server per-template severities in the proposals card instead of the isAway/hasNight heuristic

**What:** `_severityForCandidate` (hga-proposals-card.js) rates `isAway || hasNight` as "high" before the motion+camera "low" leg, while the server registers several away-scoped templates low (`motion_detected_while_away`, `motion_without_camera_activity`, `entity_staleness`). With #524's structured away detection, path-only staleness/power/motion candidates now prefill GitHub rule-request issues as severity high where the server would register low — but the same English prose already produced "high" pre-#524, so this is a widened pre-existing mismatch, not a regression.

**Why:** Red-team finding during the #524 ship (2026-08-02). Deferred by user decision: reordering the severity legs would also change severity for existing English away+motion+camera candidates in the same PR; the full fix is per-class severity mirroring of the server templates, which deserves its own change.

**How to apply:** Derive the card's severity from the inferred rule-id/template (the card already computes it in `_inferRuleIdFromCandidate`) and a template→severity map mirroring proposal_templates.py, falling back to the current heuristic for unmapped candidates.

**Effort:** S
**Priority:** P2
**Depends on:** issue-524-structured-occupancy-evidence

---

### Represent daytime-only (negated night) discovery constraints

**What:** `night_signal` returns the same `False` for an explicit `not derived.is_night` / `derived.is_night == false` as for no night condition at all, so a daytime-only candidate normalizes to an all-hours template, keys `night=any`, and the card previews the broadened rule — the explicit daytime constraint is silently dropped (visible at approval, and a superset of the stated window, but not what the candidate said).

**Why:** Codex structured-review P2 during the #524 ship (2026-08-02). Deliberate for now: no daytime-only template exists in SUPPORTED_TEMPLATES, and broadening beats the pre-#524 behavior (the "night" substring in the negating text inverted the candidate to a night rule). Needs a `require_day`-style template param or a day-scoped template family to represent properly.

**Effort:** M
**Priority:** P3
**Depends on:** issue-524-structured-occupancy-evidence

---

### Bucket volatile readings out of the low_battery_sensors anomaly-id hash

**What:** `_eval_low_battery_sensors` (dynamic_rules.py) puts float `sensor_levels` into finding evidence, and `build_anomaly_id` hashes evidence — so every reading change mints a new anomaly_id, and the notifier's per-finding cooldown (keyed by anomaly_id) never suppresses repeats as a draining battery drifts through the threshold (39, 38, 37 → three distinct anomaly ids). The suppression layer's per-type cooldown limits real-world impact for slow percent drift, but a jittering promoted sensor would notify every cycle.

**Why:** Red-team finding during the issue #522 ship (2026-08-01). NOT fixed in that PR deliberately: changing evidence hashing changes anomaly IDs for all existing low_battery rules, breaking snooze continuity and audit linkage on existing installs (the evidence-key-order-is-anomaly-id pitfall) — needs its own migration-aware PR. The #522 approval-time battery plausibility gate (`_is_battery_like_state`) removes the pathological wrongly-promoted-sensor case.

**How to apply:** Floor `sensor_levels` to 5% buckets (or exclude levels/states from the hashed evidence and carry them in a non-hashed field, mirroring the friendly_name exclusion in `_build_finding`), with a migration note that pre-existing anomaly ids for low-battery findings change once.

**Effort:** S
**Priority:** P3
**Depends on:** None

---

### Surface dead unavailable-sensors rules (unresolvable entity IDs, all-of semantics)

**What:** `_eval_unavailable_sensors` / `_eval_unavailable_sensors_while_home` (dynamic_rules.py) return `[]` for the whole rule when any single listed entity fails to resolve in the snapshot, with no logging, audit entry, or KPI. This resolve-abort alone means one hallucinated LLM evidence ID, a re-paired zigbee device (hex-address entity IDs change on re-pair — the exact issue #514 ID shape), or a renamed entity permanently disarms an approved rule with zero diagnostics — in both variants. (Trigger semantics differ: the plain variant is all-of — every listed sensor must be unavailable — while the while-home variant emits one finding per unavailable sensor.)

**Why:** Flagged independently by both the Claude adversarial review and Codex during the v3.21.2 ship (issue #514) — cross-model agreement. The abort-on-missing behavior is deliberate and test-pinned (conservative fail-closed), so changing it is a product decision: skip-unresolvable-with-warning, any-of semantics, an audit metric for rules whose entities never resolve, or surfacing unresolved params in the approval preview.

**How to apply:** Prefer the least invasive option first: emit a low-severity audit/diagnostic entry when an approved rule's entity fails to resolve during evaluation, and show unresolved entity IDs in the proposals card preview at approval time. Revisit any-of semantics only with field data.

**Effort:** M
**Priority:** P2
**Depends on:** v3.21.2 (fix/514-unavailable-binary-occupancy-sensors)

---

### Widen entity_staleness normalization to binary_sensor evidence

**What:** The `entity_staleness` branch in `explain_normalize_candidate` still keys on `person_ids or sensor_ids` (sensor.* only), so "occupancy sensor not updated / last seen" candidates citing only `binary_sensor.*` evidence fail normalization as `unsupported_pattern` — the same bug class as issue #514, one branch down.

**Why:** Flagged by the adversarial review during the v3.21.2 ship; likely the next field report from discovery installs with binary occupancy/presence sensors. Needs its own trigger/non-trigger tests and a check that the staleness evaluator resolves binary_sensor IDs.

**Effort:** S
**Priority:** P3
**Depends on:** v3.21.2 (fix/514-unavailable-binary-occupancy-sensors)

---

### Validate domainless evidence tokens as object-id slugs

**What:** `_find_domain_entity_ids` (proposal_templates.py) accepts any dot-free bracket token verbatim as a legacy domainless object ID — including tokens with spaces or other non-slug characters. At runtime, `_resolve_sensor_entity_id` tries `sensor.X` before `binary_sensor.X`, so a name collision can bind a different entity than the LLM cited.

**Why:** Flagged by the adversarial review during the v3.21.2 ship. Low impact (lookup keys only, no actuation path), but a cheap `[a-z0-9_]+` fullmatch guard would remove the ambiguity. Applies to the shared helper, so it also tightens the legacy sensor collector — verify legacy drafts still normalize.

**Effort:** S
**Priority:** P3
**Depends on:** v3.21.2 (fix/514-unavailable-binary-occupancy-sensors)

---

### Use snapshot device_class as positive signal for text-driven entry fallback

**What:** `_find_text_entry_entity_ids` (proposal_templates.py) promotes binary_sensor/cover evidence IDs to entry sensors via an English-token denylist (`_NON_ENTRY_ID_TOKENS`). The denylist is English-only, so a locale-named non-entry sensor (e.g. Czech `binary_sensor.pohyb_zahrada`, a motion sensor) can still be promoted when candidate text legitimately mentions a door/window. A positive signal — the snapshot entity's `device_class` in `{door, window, opening, garage_door}` — would be language-independent.

**Why:** Flagged by both the security specialist and the adversarial review during the v3.21.0 ship (issue #504). Impact today is bounded to mislabeled proposals (rules require user approval and the evaluator is read-only), so the denylist shipped as the v3.21.0 mitigation. The structural fix needs the normalizer to receive snapshot context, which it currently doesn't.

**How to apply:** Thread the current snapshot (or an `entity_id -> device_class` map) into `explain_normalize_candidate` from its `__init__.py` call sites; in `_find_text_entry_entity_ids`, prefer device_class membership when available and fall back to the denylist otherwise. Also surface resolved `entry_entity_ids` in the proposals card so users can see the entity binding before approving.

**Effort:** M
**Priority:** P3
**Depends on:** v3.21.0 (fix/sentinel-window-open-at-night-504)

---

### Tighten discovery prompt to require entity-backed evidence paths

**Completed:** v3.9.0 (2026-04-06)

Entity-backed evidence path instruction added to `USER_PROMPT_TEMPLATE` in `explain/discovery_prompts.py`. `_filter_novel_candidates()` in `explain/discovery_engine.py` now guards against derived-only paths. Tests added for the filter.

---

### Wire `proposals_promoted` counter in discovery engine

**What:** `SentinelHealthSensor` now exposes `discovery_proposals_approved_24h` — the count of proposals with `status="approved"` in the last 24 hours, queried directly from `ProposalStore` (Option B from the original TODO). The bare `proposals_promoted` in-memory counter (which always reported 0) was removed.

**Why:** The counter was added to the health sensor attributes in v3.7.0 but the increment logic was not wired. Option B (direct store query) is simpler and doesn't require engine changes.

**Effort:** S
**Priority:** P1
**Depends on:** v3.7.0 (health sensor discovery metrics)
**Completed:** v3.7.1 (2026-04-04)

---

### Text/device_class fallback for locale-named motion sensors

**What:** `_find_motion_entity_ids` matches only the substrings `motion`/`vmd` in entity IDs, so a locale-named PIR like `binary_sensor.chodba` (`device_class: motion`) can never normalize into `motion_detected_at_night_while_away` or the alarm-motion templates — the candidate dies as `missing_required_entities`. Flagged by Codex adversarial review during the v3.22.0 ship (issue #516). The v3.23.0 prose fallback (issue #518) shares the same substring limitation — a locale-named ID in prose is not promoted either.

**How to apply:** Mirror the issue-504 locale entry fallback: when candidate text names motion and evidence cites `binary_sensor.*` IDs not classified as any other kind, promote them as motion sensors — with the same gating pitfalls (word-bounded text, absence of higher-priority entity classes, JS card mirror). Alternatively use snapshot `device_class` as the positive signal (see the existing device_class TODO above).

**Effort:** M
**Priority:** P2
**Depends on:** v3.22.0

---

### Card severity preview disagrees with server-registered severity

**What:** `_severityForCandidate` in hga-proposals-card.js returns "high" for any away/night candidate, but the server registers e.g. `motion_detected_at_night_while_away` as medium and `open_entry_at_night_when_home` as medium — the GitHub rule-request prefill and UI disagree with what gets stored. Pre-existing; surfaced during the v3.22.0 ship review.

**How to apply:** Mirror the per-template severity table from `proposal_templates.py` into the card, or include the normalized severity in the proposal record and display that.

**Superseded by:** "Mirror server per-template severities in the proposals card instead of the isAway/hasNight heuristic" (P2, 2026-08-02, Discovery section) — same fix; #524's structured away detection widened the mismatch and raised the priority.

**Effort:** S
**Priority:** P3
**Depends on:** v3.22.0

---

### Dispatch-level dedup for overlapping night/day away-motion findings

**What:** A household running both `motion_detected_at_night_while_away` and `motion_detected_while_away` over the same sensors gets two findings (two pushes, doubled audit rows) for every night motion event. An evaluator-level dedup was implemented and reverted during the v3.23.0 ship: `evaluate_dynamic_rules` runs before snooze/exclusion/pending-prompt suppression, so dropping the day finding while the night rule was snoozed silently lost the alert entirely (verification round 5). Docs currently advise replacing the night rule instead.

**How to apply:** Dedup at dispatch time in the engine/notifier, after per-finding suppression decisions are known: when both findings for overlapping sensors would dispatch, send only the night one; when the night finding is suppressed, the day finding dispatches normally.

**Effort:** M
**Priority:** P2
**Depends on:** v3.23.0

---

### Proposal dedup does not treat night=any motion rules as covering night=1 candidates

**What:** `rule_key_covers_candidate_key` is exact equality for non-template keys, so an active `motion_detected_while_away` rule (`night=any|home=0`) does not cover a later night-worded re-proposal of the same sensors (`night=1|home=0`) — discovery can propose the night sibling of an already-active any-hour rule. Runtime double-alerting is already prevented by the evaluator's overlap dedup (v3.23.0); this is proposal-side noise only. Flagged by testing-specialist + red-team review during the v3.23.0 ship (issue #518).

**How to apply:** Teach `rule_key_covers_candidate_key` that `night=any` subsumes `night=1` when subject/predicate/home/entities are identical (one direction only — a night rule never covers an any-hour candidate). Pin both directions with tests; consider whether `home=any` ⊇ `home=0/1` deserves the same treatment while in there.

**Effort:** S
**Priority:** P3
**Depends on:** v3.23.0

---

### Proposals card should render normalized rule params before Approve

**What:** The pending-proposal card shows title/summary/candidate_id but never the normalized `template_id`/params, so the user approves prose while the system persists params the prose viewer cannot audit (e.g. a motion entity named only in the hidden `pattern` field). Impact is capped — advisory-only actions, charset-constrained IDs, approval-time entity resolution (v3.23.0) — but the approval trust boundary should show the actual rule scope. Flagged by security-specialist review during the v3.23.0 ship (issue #518).

**How to apply:** Run `explain_normalize_candidate` server-side when listing proposals (or store the normalization in the proposal record) and render template_id + entity params in the card above the Approve button.

**Effort:** M
**Priority:** P3
**Depends on:** v3.23.0

---

### Night-branch guard symmetry for unknown-person/camera candidates

**What:** The day-agnostic away-motion branch (v3.23.0) excludes unknown-person and camera-evidence candidates so they keep their camera-template routing, but the night branch (v3.22.0) still captures night-worded candidates of those classes — a "unknown person and motion at night while away" candidate registers a low-signal motion rule instead of the sensitive camera rule. Left asymmetric deliberately: changing the night branch's routing would re-key candidates whose rules users already activated (dedup churn without a migration story).

**How to apply:** Add the `not has_unknown_person_signal` / `camera_id is None` guards to the night branch together with a one-time semantic-key migration or documented re-proposal expectation, and update the card mirror's night branch in the same change.

**Effort:** M
**Priority:** P3
**Depends on:** v3.23.0

---

### sensor_threshold_condition rule keys never cover their candidates

**What:** `rule_semantic_key` for `sensor_threshold_condition` emits a 4-field key (`predicate=power_threshold`, no night/home/template fields) while candidates key 7 fields with `predicate=power_anomaly` plus preserved context — `rule_key_covers_candidate_key` is structurally always False (equality fails, no `|template=` strip, field-count mismatch). After approval the LLM hint gap reopens and only the 200-record history filter backstops re-proposals; context variants ("above 1000 at night" vs "while nobody is home") also mint distinct candidate keys and can pile up as pending cards.

**Why:** Pre-existing shape shared with power thresholds (accepted base parity in the #540/#541 ships), but the #541 environmental prompt explicitly encourages threshold candidates for dangerous extremes, materially increasing incidence. Confirmed by ship red-team review (2026-08-11, reproduced); docs/sentinel.md discloses the residual.

**How to apply:** Emit a `power_threshold` candidate predicate when the statistical leg detects threshold prose, and teach `rule_key_covers_candidate_key` to strip context for `power_threshold` rule keys the way it does for `|template=` keys — with a re-proposal-expectation note for existing pending threshold candidates whose keys migrate.

**Effort:** M
**Priority:** P2
**Depends on:** v3.28.0

---

### Low-side ("drops below") thresholds silently degrade to baseline rules

**What:** `_NUMERIC_THRESHOLD_PATTERN` matches only high-side wording (above/exceeds/over/more than) and rejects zero/negative values, so "crawlspace temperature drops below 3" registers a generic `baseline_deviation` instead of a low-side threshold rule — the approved card's semantics (freeze warning) do not match the registered rule, silently.

**Why:** Low-side thresholds are the primary environmental safety case (freeze, low humidity), newly reachable via #541. Fixing it needs `sensor_threshold_condition` to grow a direction/comparator param — an evaluator change the #541 scope decision ("no new evaluator templates") deliberately excluded. Found by ship adversarial review (2026-08-11, reproduced).

**How to apply:** Add a `below: bool` (or `comparator`) param to `sensor_threshold_condition`'s evaluator, extend the threshold extraction with a low-side pattern (mirroring `_RELATIVE_THRESHOLD_PATTERN`'s vocabulary), route low-side prose accordingly, and mirror whatever the candidate keys do per the threshold-coverage TODO above.

**Effort:** M
**Priority:** P2
**Depends on:** v3.28.0

---

### Percent-deviation baseline physics are wrong for interval-scale environmental quantities

**What:** `evaluate_baseline_deviation` computes `|cur−base|/|base|·100` with `base==0 → any non-zero = 100%`. Celsius temperatures near zero make tiny absolute swings read as huge percentages (a 0.0 °C baseline fires on every non-zero reading — winter alert spam), and atmospheric pressure (~1000 hPa) can never move 50%, so an approved pressure rule is permanently inert monitoring presented as coverage.

**Why:** Pre-existing evaluator behavior, explicitly deferred by #541's scope ("separate issue if it bites"), but #541 is what steers these quantities into the evaluator. Illuminance — the third pathological class — was mitigated in-ship by routing to `time_of_day_anomaly`. Flagged by both ship adversarial passes (2026-08-11).

**How to apply:** Make the deviation class/unit-aware: absolute-delta or stddev-based thresholds for interval-scale units (°C/°F, hPa), keeping pct deviation for ratio-scale quantities (W, lx, ppm). Applies to both `evaluate_baseline_deviation` and the baseline updater's drift logic; needs unit metadata from the snapshot or baseline store.

**Effort:** L
**Priority:** P2
**Depends on:** v3.28.0

---

### Per-device-class budget for the discovery snapshot entity cap

**What:** The reduced snapshot's `_MAX_ENTITIES=100` cap has no per-class slices. v3.28.0 stops environmental sensors from systematically outranking idle security entities (their recency bonus is zeroed), but score ties still evict alphabetically — a large enough env/power fleet can still push closed doors and locked locks out of the snapshot, and with them out of `baseline_ready_entities` gap analysis.

**Why:** Ship red-team review (2026-08-11) reproduced eviction with a synthetic 110-entity home; the recency fix removes the systematic bias but not the tie-break residual. The #541 scope decision deferred score redesign pending field reports.

**How to apply:** Reserve per-domain/device-class budget slices before the global cap (e.g. security binary_sensors/locks/covers first, then measurement classes), or add a deterministic round-robin across classes for tied scores. Verify the budget-trim passes still fit `_TOKEN_BUDGET_CHARS`.

**Effort:** M
**Priority:** P3
**Depends on:** v3.28.0

---

### Locale-named environmental candidates stay unsupported

**What:** The environmental routing signal is English-token-only (prose or entity ID), so a home where both the entity ID and the generated prose are localized (`sensor.podkrovi_teplota` + Czech prose) returns `unsupported_pattern` even though the reducer admitted the entity via `device_class: temperature`.

**Why:** Codex structured review during the #541 ship (2026-08-11). Same class as the locale-named motion/entry/battery fallbacks; the clean fix is locale-independent metadata (device_class from the snapshot) reaching the normalizer, which is currently a pure text function — a design change shared with the approval-time-validation TODO above.

**How to apply:** Either thread snapshot device_class metadata into normalization (candidate-enrichment at the engine layer, keeping the normalizer pure), or accept the structured `pattern` field (`statistical_baseline_deviation` / `time_of_day_anomaly` are English machine tokens even in locale homes) as a routing signal when sensor evidence exists — mirror whatever the keying side needs for parity.

**Effort:** M
**Priority:** P3
**Depends on:** v3.28.0

---

### Expose unit_of_measurement to the discovery model for threshold proposals

**What:** The reduced snapshot exposes device_class and the rounded state but drops `unit_of_measurement`, while the #541 prompt encourages absolute environmental thresholds ("above 95") — the model cannot distinguish °C from °F or hPa from inHg, and the evaluator compares the proposed number directly against the raw state without conversion. Unit-blind threshold proposals are approval-gated (the card shows the threshold and entity), but the model is guessing.

**Why:** Codex adversarial review during the #541 ship (2026-08-11). Adding a `unit` field to reduced env entities is cheap in tokens but touches the snapshot contract, grouping semantics, and budget tests — deferred rather than slipped into the ship.

**How to apply:** Include `unit` (from `attributes.unit_of_measurement`) on reduced entities for measurement device classes, teach the prompt to cite it, and consider including the unit in the grouping key so mixed-unit same-area sensors don't share a group.

**Effort:** S
**Priority:** P2
**Depends on:** v3.28.0

---

### Approval-time validation for statistical/threshold rule targets

**What:** Battery approvals validate each cited sensor against live state (`_is_battery_like_state`) and motion approvals resolve prose IDs against the registry, but promoting a baseline_deviation / time_of_day_anomaly / sensor_threshold_condition candidate registers whatever `sensor.*` ID the LLM cited — a hallucinated ID becomes a permanently silent rule, and a real-but-unrelated numeric sensor becomes a wrong rule.

**Why:** Codex adversarial review during the #541 ship (2026-08-11). Advisory-only impact (no actuation), and the signal-preferring target selection shipped in v3.28.0 reduces the wrong-sensor shape, but inert monitoring presented as coverage is the same honesty problem the dead-unavailable-rules TODO tracks.

**How to apply:** In the promote flow's approval path, resolve the target entity against live states and require a numeric reading (and, when metadata exists, a measurement-class device_class/unit) before registering; refuse with `entities_unresolved`/`not_numeric_sensor` reason codes mirroring the battery gate's shape.

**Effort:** M
**Priority:** P2
**Depends on:** v3.28.0

---

### Mixed motion+environmental candidates key subject=motion while promotion registers subject=sensor

**What:** A candidate citing `binary_sensor.attic_motion` plus `sensor.attic_temperature` with environmental-only prose keys `subject=motion|entities=binary_sensor.attic_motion` (the #541 env leg is subject-gated off to protect motion rules), but when no motion branch claims it the normalizer's statistical branch registers `baseline_deviation` on the temperature sensor — the rule key (`subject=sensor`) never covers the candidate key, so context variants of that mixed shape can re-propose after approval.

**Why:** Residual accepted during the #541 ship: the subject gate exists to prevent the worse steal (motion candidates keyed power_anomaly, breaking motion-rule dedup — the #540 lesson), and gap-hint candidates are prompted to cite only their gap entity, so mixed shapes are uncommon. Same class as the pre-existing mixed battery shapes recorded in the derive-keys-from-routing TODO above, which is the structural fix.

**How to apply:** Covered by "Consider deriving candidate keys from the normalizer's actual routing" — keying from the normalized rule eliminates the whole class. No separate mirror patch recommended.

**Effort:** S
**Priority:** P3
**Depends on:** v3.28.0

---

## Video Analyzer

### Identity merge: temporal-adjacency guard for gray-zone faces

**What:** Add a fourth refusal condition to `_merge_unknown_faces` (core/video_analyzer.py): only merge an "Unknown Person" face when its frame is within N seconds (e.g. 30) of a frame where the known person was directly recognized. Today the batch is the only temporal boundary, so on a long event-select flush a gray-zone stranger appearing minutes after the resident left can still merge.

**Why:** All three adversarial reviewers on the v3.30.0 ship converged on the cross-frame gray-zone residual (a stranger within 0.7–0.85 cosine of the resident, never co-framed). The shipped guards (nearest-gallery-match requirement, companion guard incl. VLM-dropped frames, strict one-known rule) close the demonstrated exploits; temporal adjacency closes most of the remaining tailgater window. Deferred by user decision (2026-08-13, ship D3): field-validate the shipped guards first via the `unknown_merge_*` counters and debug crops, add this only if misattribution shows up.

**How to apply:** Thread each frame's capture timestamp (already available as `ts` in `_process_batch`'s `ordered` loop) into the per-frame hit bookkeeping, record the timestamps of frames whose hits contain the known name, and refuse (new counter, e.g. `unknown_merge_refused_temporal`) when an unknown face's frame is farther than the window from every known-name frame. Also consider the caption-based refusal for face-recognition-timeout frames whose VLM caption affirmatively mentions a person (`_caption_mentions_person`) — same review thread, same deferral.

**Effort:** S
**Priority:** P2
**Depends on:** v3.30.0

---

### Single-person constraint is withheld from exactly the batches that flap

**What:** `_verified_sole_person` (core/video_analyzer.py) requires exactly one distinct detected name across the full pre-cap batch, so a surviving "Unknown Person" — the residue of a merge the distance bound correctly refused — makes `sole_person` None and withholds the `<single person constraint>` block. Refused-merge batches therefore fall back to the summarizer's prompt-level single-actor bias alone, which is the guidance v3.30.1 shipped the deterministic block *because* it could not rely on.

**Measured 2026-08-18 against the live qwen3.5:9b summarizer** (10 samples per arm, temperature 0.2, real prompts): on this path — one enrolled name plus an unmerged "Unknown Person", so no block is emitted — the summarizer wrote "A person steps onto the porch holding a mug, then Lindo leans on the railing" in **10 of 10** runs before the v3.30.6 reference rule and **8 of 10** after it. The system message has carried a worked example of exactly this case since #543 and the model still ignores it, so more prompt text will not close this; the deterministic block is the lever. For contrast, the same reference rule took the no-name case from 0/10 to 10/10 clean, and adding an introduce-once sentence to the block itself took the constrained case from 3/8 to 8/8 (then 10/10).

**Why:** Root-caused during the v3.30.6 field investigation (2026-08-16 14:34, camera.playroomdoor). v3.30.6 removed the fabricated co-occurrence that was actively defeating the bias, so the bias now gets a truthful prompt — but it is still the only protection on that path. Deliberately not widened in v3.30.6: the refused unknown's nearest match was a *different* enrolled identity at 0.863, which is genuinely ambiguous, and asserting "the only person in this footage" there would be a false claim. Any fix must preserve that honesty rather than relax the verdict.

**How to apply:** Consider a weaker second-tier constraint for the refused-merge case that states what is actually known ("recognition matched exactly one enrolled person in this footage; other faces were seen but not identified") instead of the strong sole-person claim, and let the summarizer keep normal Counts rules. Gate it on: exactly one enrolled name, every other detected entry a reserved placeholder, and no frame with genuine co-occurrence. Field-validate against sequential-visitor batches before shipping — the failure mode to avoid is collapsing a real second visitor into the named resident.

**Effort:** M
**Priority:** P3
**Depends on:** v3.30.6

---

### Identity merge: batch the gallery lookups

**What:** `_merge_unknown_faces` awaits `PersonGalleryDAO.nearest_match` once per unknown face, re-sending a ~10 KB 512-float vector literal per call. Fetch the batch's gallery rows once (or add a DAO method taking a list of embeddings) and compute cosine distances client-side with numpy — gallery rows are already L2-normalized, so distance is `1 - dot`.

**Why:** Performance-specialist finding on the v3.30.0 ship. Bounded today (batch cap, per-call 5 s timeout, 15 s batch budget, short-circuit after first failure), so it's an efficiency cleanup, not a correctness issue.

**Effort:** S
**Priority:** P3
**Depends on:** v3.30.0

---

### Caption novelty: per-analysis notification-status metadata

**What:** Store whether each video analysis triggered a notification alongside the
caption in the vector store. Add `notified`, `decision_reason`, and `matched_key`
fields to the stored value written by `_store_results`. Update `_is_caption_novel`
to optionally filter `store.asearch` results to notification-worthy records only,
so suppressed artifact captions do not inflate the similarity baseline.

**Why:** `_store_results` is called unconditionally, meaning suppressed captions
(e.g. repeated nighttime blur) are stored and can later cause a genuinely new
artifact caption to be suppressed against a non-notified prior. Filtering by
`notified=True` in the search would improve precision but requires a metadata
field and a query-time filter that the store API must support.

**How to apply:** Add `notified: bool`, `decision_reason: str`, and
`matched_caption_key: str | None` fields to the dict written in `_store_results`.
Pass `decision.notify` and `decision.reason` from `_handle_notification` into
`_store_results`. Update `store.asearch` call to include a metadata filter when
the store API supports it. Until the filter lands, the current behavior (compare
against all stored analyses) is acceptable.

**Effort:** M
**Priority:** P3
**Depends on:** v3.14.0 (CaptionNoveltyDecision)

---

### Caption novelty: tune threshold and artifact terms from real logs

**What:** Review accumulated debug logs from `_is_caption_novel` decisions to
check whether `VIDEO_ANALYZER_SIMILARITY_THRESHOLD = 0.85` and the `_ARTIFACT_RE`
vocabulary produce the right suppress/notify balance in production.

**Why:** The threshold and terms were set conservatively. Real logs can reveal
common false-positives (suppressing events that should notify) or false-negatives
(notifying on repeated low-value artifacts). Adjust as data accumulates.

**Effort:** S
**Priority:** P3
**Depends on:** v3.14.0 (decision logging)

---

### save_and_analyze_snapshot tmp files leak in the _latest/ subfolder

**What:** The service writes `snapshot_<ts>.jpg` into `latest_target(...).parent`
(the `_latest/` subfolder), publishes a copy to `latest.jpg`, and never removes
the tmp file. `_prune_old_snapshots` deliberately never deletes files in the
`_latest/` subfolder (guard re-appends and breaks), so these files can neither
be registered nor swept by the current mechanism.

**Why:** Each service call leaks one file. Low volume (manual/automation
calls), but unbounded. Registering them is not an option without rethinking
the latest-subfolder guard — a registered `_latest/` file would permanently
clog the deque head.

**How to apply:** Either unlink the tmp file after `publish_latest_atomic`
(it's a copy, the dst survives), or write the tmp into the camera's normal
snapshot directory so capture-time retention covers it. Unlinking after
publish is simplest; check the bus event's `"path": str(tmp_path)` consumers
first (same dangling-path concern as the suppressed-notification TODO).

**Effort:** S
**Priority:** P3

---

### Bus event path can dangle when notification is suppressed

**What:** In `notify_on_anomaly` mode with `decision.notify=False`,
`protect_notify_image` is never called, but the chosen frame was already
published as `latest.jpg` and announced on `hga_last_event_frame` with
`"path": str(chosen)`. Consumers resolving that path can read a pruned file.
The notify-frame fix biases `chosen` toward the batch head (oldest end of the
retention deque), shortening time-to-dangle.

**Why:** Suppressed-notification events hand out a path with no pruning
protection. The `latest` dst copy is stable; the raw `path` is not. Test
`test_handle_notification_suppresses_when_decision_is_no_notify` currently
asserts protect is NOT called, so changing this is a deliberate behavior change.

**How to apply:** Either call `protect_notify_image(chosen)` unconditionally
before the mode branch, or document that consumers must use `latest` and keep
`path` best-effort. Update the suppression test accordingly.

**Effort:** S
**Priority:** P2

---

### Dispatch frame epoch instead of utcnow() as latest-frame timestamp

**What:** `SIGNAL_HGA_NEW_LATEST` and `SIGNAL_HGA_RECOGNIZED` dispatch
`dt_util.utcnow().isoformat()` as the frame timestamp. The notify-frame fix
preferentially selects early person frames, so for long held event-buffer
batches the image entity and sensor can label a frame that is minutes old as
captured "now".

**Why:** The frame's true epoch is already available (`epoch_from_path(chosen)`
/ the `ordered` tuples) but is discarded. Pre-existing pattern; skew widened by
early-frame selection.

**How to apply:** Thread the chosen frame's epoch through
`_finalize`/`_handle_notification` and dispatch it as the timestamp.

**Effort:** S
**Priority:** P3

---

### Scale stale-snapshot budget from the camera's own refresh interval

**What:** The stale-snapshot guard (issue #490) uses a fixed 30-minute budget
(`_SNAPSHOT_STALE_MAX_AGE_SEC = 1800`), sized as 3× the battery-camera
interval. ring-mqtt publishes each camera's actual interval as
`number.<camera>_snapshot_interval` (field-observed: 600 s battery, 30 s
wired), so the budget is ~60× too lenient for wired cameras — a frozen wired
cam goes undetected for half an hour. Better still for `event_select`-owned
windows: compare the snapshot `timestamp` against the event that opened the
window, which is exact rather than a heuristic.

**Why:** Raised by @andymcmanus in #466 field testing (2026-07-18 comment,
n=12 events: frame staleness at event open ranged 3–54 min, mean 21 min,
with 2/12 over the 30-min budget — so 10/12 passed it and were analyzed).
The guard already only applies to ring-mqtt `event_select` cameras, so
reading ring-mqtt's own interval entity adds no new integration coupling.

**How to apply:** Two candidate checks, NOT interchangeable, each with a
precondition that must be solved first.

*Interval scaling (tightens wired cameras only).* In
`_retained_frame_is_stale`, resolve `number.<base>_snapshot_interval` via
the same sibling/device-registry lookup as `_has_event_select_sibling`;
budget = 3× that value, falling back to 1800 s when absent. Do NOT apply
this unconditionally: the entity is a configured timer period, not a bound
on frame age, and it is published even when nothing is scheduled to honour
it. ring-mqtt defaults it in the camera constructor
(`devices/camera.js:58-60`, v5.9.3) to 600 on battery / 30 on wired and
publishes it whenever any snapshot mode is active, while `Auto` separately
disables the *interval* path for battery devices (`:415-417`). A battery
camera in the default mode therefore advertises 600 while no interval
refresh is scheduled at all — only the motion path remains, and that fires
only when Ring's push carries an image UUID (`:758-769`), which #466 field
testing reports never happening on the two battery models tested. Scaling off the
entity there yields an 1800 s budget on exactly the cameras needing the
tightest check. Even on wired cameras it is a nominal cadence rather than a
ceiling: scheduled refreshes are skipped while offline or while a motion
ding is active (`:735`), and requests can fail, so real age can exceed it.
**Precondition:** `_has_event_select_sibling` carries no power-source or
snapshot-mode signal, so there is currently no way to enforce the
wired-only restriction. Gate on a resolved mode/power signal (or on the
camera's `type` attribute reading `interval`) before using this leg at all.

*Event-timestamp comparison (the only check that works in Auto/Motion).*
For loops the `event_select` path owns, compare the snapshot `timestamp`
against the event that opened the window. **Precondition:**
`_handle_event_select_change` sees only an HA state change carrying
`eventId`/`recordingUrl`, and ring-mqtt discovers new events on a ~1-minute
poll cycle and publishes `eventId` only after fetching the recording URL
(`:439-448`). That state-change time is therefore *detection* time, not
Ring capture time, and a strict comparison would reject a genuinely current
frame published before the delayed `eventId`. Propagate the real Ring event
timestamp, or allow a documented skew.

This comparison also governs the common case today: at a mean staleness of
21 min at event open (10/12 events under the 30-min budget), the previous
event's frame usually passes and is analyzed as if current (misattributed
imagery). With the take_snapshot automation installed, quiet cameras (>30
min gaps) log one expected snapshot-failure WARNING per event that a
window-scoped check could suppress.

**Effort:** M
**Priority:** P2

---

## Notifier / Observability

### Sanitize friendly_name-derived text in notification copy like the unit string

**What:** The v3.31.1 baseline-copy fix strips control/bidi characters and length-caps the untrusted `unit_of_measurement` before it reaches push text (`_baseline_deviation_mobile_message`), but the appliance/sensor display name (`friendly_name` → `_strip_power_suffix(...).title()`) is still embedded uncapped and unfiltered in the same messages, and camera names elsewhere use only a `[:30]` cap with no control-char strip. A crafted device name can carry the same bidi-reorder / instruction-text spoofing the unit fix closed (raised by both the Claude adversarial and Codex passes on v3.31.1; same class, pre-existing convention).

**Why:** Notification copy is a trust boundary: entity names arrive from semi-trusted integrations (MQTT/ESPHome/template sensors over untrusted data).

**How to apply:** Extract the unit sanitizer (category-C strip + whitespace collapse + cap) into a small helper and apply it to every evidence-derived display string in `sentinel/notifier.py` (appliance names, camera names, entry names), with a generous cap (e.g. 40) for names. One test per surface.

**Effort:** S
**Priority:** P3
**Depends on:** None

---

### Remaining notification-localization gaps after PR #565 (buttons, cs plural, uncurated labels)

**What:** Three gaps deliberately shipped with the v3.31.0 notification-chrome localization (PR #565), awaiting @hruba202's input (asked in the close-out comment, issuecomment-5386281819): (1) **action-button titles stay English** — "Confirm"/"Cancel" under the Czech permanent-snooze prompt guide a destructive action in the wrong language (cross-model review's top remaining finding), plus "Ask Agent"/"Arm Alarm"/"Snooze Always"; (2) **cs `batch_message` plural** — "{count} novinek" is grammatically wrong for counts 1–4, the common batch size (needs novinka/novinky/novinek forms or a count-agnostic phrasing); (3) **uncurated type labels** — `camera_missing_snapshot_night_home`, `unknown_person_camera_no_home`, `phone_battery_low_at_night_home`, `vehicle_detected_near_camera_home`, `pet_detected_at_night_no_occupancy` are absent from `_KNOWN_TYPE_LABEL_KEYS`, so they render as prettified English slugs on Czech installs.

**Why:** Merged without waiting for the native-speaker answers (maintainer decision 2026-08-23); these need Czech wording from a native speaker, and (1) needs a scope decision on how button titles flow through the mobile-app notification payload.

**How to apply:** (1) route the `actions[].title` strings through `notif_msg` with new keys; (2)+(3) add the corrected/new cs strings and label keys to `_MESSAGES` in `sentinel/notifier_messages.py` — parity tests will enforce key coverage.

**Effort:** S
**Priority:** P2
**Depends on:** hruba202 wording input (PR #565 close-out)

---

### Unify the duplicated localized-message machinery (pin_messages / notifier_messages)

**What:** `sentinel/notifier_messages.py` (PR #565) is a second copy of `agent/pin_messages.py`'s `_resolve_language` + fallback-chain machinery, and the copies have already drifted inside the PR that created the second one: notifier_messages guards `getattr(hass, "config", None)` (config-less test doubles / degraded runtime states) and degrades gracefully on a bad `.format` placeholder, while pin_messages dereferences `hass.config` directly and formats unguarded. Neither normalizes underscore locales (`cs_CZ` resolves to silent English; real HA stores hyphenated codes, so low likelihood).

**Why:** Surfaced during the PR #565 review (2026-08-21) — the maintainability specialist flagged the triple-copy pattern and the adversarial pass confirmed the hardening drift. Every robustness fix now has to land twice or the tables behave differently under the same failure.

**How to apply:** Extract a shared `localized_message(table, hass, key, **kwargs)` helper (e.g. `core/localized_messages.py`) holding `_resolve_language` (with the config getattr guard, `str(language).replace("_", "-")` normalization, and the guarded-format fallback chain); `pin_messages` and `notifier_messages` keep only their `_MESSAGES` tables and thin `pin_msg`/`notif_msg` wrappers. Port the contract tests (key parity, placeholder subset, call-site kwargs) to cover both tables.

**Effort:** S
**Priority:** P3
**Depends on:** None

---

### llm_explain's type-label table now drifts from the notifier's

**What:** `explain/llm_explain.py:124` keeps its own `_KNOWN_TYPE_LABELS` (type → English text) plus duplicate `_display_type`/`_friendly_type`/`_severity_action_hint`, while `sentinel/notifier.py` switched to `_KNOWN_TYPE_LABEL_KEYS` (type → message id) in PR #565 — differently shaped tables with no test forcing agreement. #565 added `appliance_power_duration` to the notifier copy only; output coincides today purely because the slug-prettify fallback (`"appliance_power_duration".replace("_", " ").capitalize()`) happens to equal the curated label, so the next label edit in one file gives the explanation prompt and the notification different names for the same finding.

**Why:** Found during the PR #565 review (2026-08-21) by the Enum & Value Completeness pass (consumers outside the diff) and independently by two review agents.

**How to apply:** Either have `llm_explain` derive its English labels from `notif_msg(None, key)` via the notifier's key table, or add a cross-file parity test asserting `llm_explain._KNOWN_TYPE_LABELS[t] == notif_msg(None, _KNOWN_TYPE_LABEL_KEYS[t])` for every shared type and that the key sets match (deliberate exclusions asserted explicitly). The prompt side must stay English (LLM input), so this is label-source unification, not localization.

**Effort:** S
**Priority:** P3
**Depends on:** None

---

### Compound notifications hide the unknown-person signal behind the alarm title

**What:** When the correlator bundles same-cycle findings into a `CompoundFinding`, `_dispatch_compound` picks the representative for notification rendering by highest confidence (`engine.py`: `best = max(compound.constituent_findings, key=lambda f: f.confidence)`). `alarm_disarmed_during_external_threat` (confidence 0.9) therefore always outranks `unknown_person_camera_night_home` (0.7) and the dynamic `unknown_person_camera_when_home` rules, so a genuine stranger sighting renders under the title "Outdoor activity while alarm disarmed" and the alarm rule's mobile copy. Field-observed 2026-08-23 (first-ever `unknown_person_camera_night_home` firings, 11:51/11:52 UTC): the user saw only alarm-disarmed pushes and concluded the unknown-person rules were not firing — the stranger evidence was only visible in the audit store. A person-on-camera alert is also arguably the more actionable headline than the alarm state that merely contextualizes it.

**How to apply:** Rank compound representatives by security salience before confidence — e.g. a small type-priority table (unknown-person types > alarm-disarmed types > entry/motion types) used as the primary sort key with confidence as tiebreak, or simply prefer any constituent whose evidence has `unknown_person_present`/a stranger label when choosing `best`. Alternatively keep `best` for execution policy but render the notification title/copy from the highest-salience constituent, and consider appending a one-line "+ N related findings" suffix so the compound's breadth is visible. Mind the localization boundary: security-critical copy stays deterministic English (`_is_security_copy`), and the existing per-type deterministic formatters must keep receiving the constituent they were written for.

**Why:** The whole point of the v3.30.11 unknown-person fix was making stranger sightings visible; the confidence-ranked compound title re-hides them at the last hop. Surfaced during v3.30.11 field validation.

**Effort:** S
**Priority:** P2
**Depends on:** v3.30.11

---

### Deduplicate the _friendly_type label maps

**What:** `sentinel/notifier.py` and `explain/llm_explain.py` each carry a byte-identical `_friendly_type` `known` map (and matching prefix-stripping fallback). Every new anomaly type requires the same entries in both maps plus mirrored tests in `test_sentinel_notifier.py` and `test_llm_explain.py` — the v3.21.0 ship added the four `open_entry_at_night*` keys in four places. A missed side silently regresses user-visible labels to the title-cased fallback. The v3.22.0 ship widened the duplication: `_KNOWN_TYPE_LABELS` and a `_display_type` helper (template-label fallback for slugified dynamic-rule IDs) are now mirrored verbatim in both modules, so each new rule type needs two edits plus two mirrored tests. Re-flagged by the maintainability specialist during the v3.22.0 ship.

**How to apply:** Hoist `_KNOWN_TYPE_LABELS`, `_display_type`, and the fallback prettifier into one shared module (e.g. `sentinel/labels.py` or a small `core/labels.py`), import it from both `notifier.py` and `llm_explain.py`, and collapse the duplicated tests into one suite against the shared module.

**Effort:** S
**Priority:** P3
**Depends on:** None

---

### Feedback-trained per-entity cooldowns — wire feedback signal

**What:** `record_cooldown_feedback(state, entity_id, rule_type)` is now called from both the snooze action (`sentinel/notifier.py`) and the dismiss action (`notify/actions.py`). Each snooze or dismiss of a rule+entity pair increments the compound-key multiplier, which extends future cooldowns for that specific combination.

**Why:** Without the feedback signal, `learned_cooldown_multipliers` remained empty forever. Now every snooze/dismiss trains the system.

**Effort:** S
**Priority:** P1
**Depends on:** v3.7.0 (suppression schema v4)
**Completed:** v3.7.1 (2026-04-04)

---

### Fix cooldown multiplier key scheme (entity_id → rule_type:entity_id) + schema migration v5

**What:** `learned_cooldown_multipliers` is now keyed by `"{rule_type}:{entity_id}"` (e.g., `"unlocked_lock_at_night:lock.front_door"`). The v4→v5 migration in `_migrate_suppression_state()` discards all bare entity_id keys (safe: `record_cooldown_feedback` was never called in v3.7.0 production, so v4 dicts were always empty). `stored_version = 5` correctly set after migration.

**Why:** The bare entity_id key caused different rules for the same entity to share a single multiplier, causing missed alerts for the more critical rule.

**Effort:** S
**Priority:** P1
**Depends on:** Wire feedback signal TODO above
**Completed:** v3.7.1 (2026-04-04)

---

### Daily digest config flow UI

**What:** `sentinel_subentry_flow.py` now exposes `BooleanSelector` for `CONF_SENTINEL_DAILY_DIGEST_ENABLED` and `TimeSelector` for `CONF_SENTINEL_DAILY_DIGEST_TIME`. Both appear in `_default_payload()`. `RECOMMENDED_SENTINEL_DAILY_DIGEST_TIME` normalized to `"08:00:00"` in `const.py` to match `TimeSelector` output format. The notifier parse bug (`split(":", 1)` → `split(":")`) was fixed as part of this.

**Why:** The daily digest shipped in v3.7.0 with no UI control; users had to edit raw options.

**Effort:** S
**Priority:** P1
**Depends on:** v3.7.0 (daily digest backend)
**Completed:** v3.7.1 (2026-04-04)

---

### Add `learned_suppressions_active` attribute to health sensor

**Completed:** v3.9.0 (2026-04-06)

`learned_suppressions_active` attribute exposed on `sensor.sentinel_health`. Count reads `learned_cooldown_multipliers` from suppression state via `engine.learned_suppressions_count` property.

---

## Speech-to-Text

### SDK client defaults given up by using HA's shared httpx client

**What:** `_get_client` (stt.py) builds the OpenAI client on Home Assistant's shared httpx client. That is what removes the per-stream SSL-context build (#556), but it also means the SDK's own client defaults no longer apply. Two are currently accepted and documented rather than restored: `follow_redirects` (openai's own client sets `True`; HA leaves httpx's `False`, so a 3xx from an egress proxy or a future endpoint move surfaces as an error instead of being followed), and the SDK's connection limits (HA's pool expires keepalive at 15s, which is what bounds the connection-reuse win to back-to-back utterances). The request timeout was the third and *is* pinned.

**Why:** Found independently by the Claude adversarial and red-team passes on the #556 ship (2026-08-16), both citing `openai/_base_client.py` `_DefaultAsyncHttpxClient(follow_redirects=True)` versus HA's `HassHttpXAsyncClient`. Not fixed in that branch because api.openai.com does not redirect today, and the clean mitigation is not a one-liner: the redirect default lives on the httpx client, which we must not mutate — it belongs to every other integration.

**How to apply:** If a redirect ever needs following, do it per-request rather than on the shared client — the SDK only overrides when `options.follow_redirects is not None`, and the audio resources never set it, so it needs a request-option path (e.g. `with_options`). Add a MockTransport test returning a 307 and assert the request is followed. Do not set it on the client returned by `get_async_client`.

**Effort:** S
**Priority:** P3

---

### Shutdown-time transcription logs a full traceback

**What:** Home Assistant closes its shared httpx client on `EVENT_HOMEASSISTANT_CLOSE`, and `get_async_client` never revalidates the cached entry — it returns the same closed client forever. A transcription attempted after that raises `httpx.RuntimeError("Cannot send a request, as the client has been closed")`, which is neither `AuthenticationError` nor `OpenAIError`, so it lands in the bare `except Exception` and dumps a traceback into the shutdown log. The returned `SpeechResult` is correctly `ERROR`.

**Why:** Surfaced by both adversarial passes on the #556 ship (2026-08-16). Cosmetic — CLOSE is the last event before process exit, so it is not reachable in a normal supervised run. Recorded because "is the cached client still usable?" is the obvious question a reader of the new cache will ask, and because the obvious fix is a placebo: rebuilding on a closed client changes nothing, since `get_async_client` hands back the identical closed object.

**How to apply:** Catch that specific `RuntimeError` in `async_process_audio_stream` and log it at warning level with a "Home Assistant is shutting down" message, instead of letting it reach `LOGGER.exception`.

**Effort:** S
**Priority:** P3

---

## Config Entry Lifecycle

### image.py and sensor.py still register unwrapped STARTED listeners

**What:** `hass.bus.async_listen_once(EVENT_HOMEASSISTANT_STARTED, _on_started)` at `image.py:42` and `sensor.py:64` are not wrapped in `entry.async_on_unload`. Both call `async_add_entities(...)` for an entry that may already have unloaded — the same leak class the deferred-start work just closed for the four engines.

**Why:** Found by the maintainability pass during the lifecycle-leak ship (2026-08-19). Recorded explicitly so the "Completed" entry below does not read as "no listener leaks remain in the integration" — it closed the four `async_setup_entry` sites, not these two platform sites.

**How to apply:** Both platform setup functions have `entry` in scope, so the wrap is available. They cannot reuse `_defer_start_until_hass_started` as written — its `start` parameter is `Callable[[], None]` and these handlers are async and take the event — so either widen the helper or give each site the same fired-flag cancel inline. Adding an entity twice is louder than a duplicate background task, so confirm the actual failure mode before choosing.

**Resolution:** The helper moved rather than widened: `_defer_start_until_hass_started` is now `core/lifecycle.py:defer_start_until_hass_started` (docstring intact — it is the canonical rationale other sites point at), importable by platforms without touching `__init__`. The handlers didn't need to be async — `async_add_entities` is a sync callback — so each platform passes a plain closure. Because platforms have no `_stopped` latch to close the residual window (unload starts → on-unload cancel runs only after `async_unload_entry` returns), the closures carry an entry-state guard instead: refuse unless `LOADED` or `SETUP_IN_PROGRESS` (the latter because a platform set up while HA is starting defers into exactly that state). Regression tests in `test_entry_lifecycle_containment.py`, each verified to fail against the unfixed code, with positive controls so deleting the deferred add outright can't stay green.

**Effort:** S
**Priority:** P2
**Completed:** v3.30.10 (2026-08-21)

---

### async_unload_entry has no failure containment

**What:** `async_unload_entry` is a run of bare awaits with no `try`. Home Assistant catches any exception out of it, sets `FAILED_UNLOAD` — declared non-recoverable at `config_entries.py:160` — and returns *without* running `_async_process_on_unload`. So one raise from, say, `video_analyzer.stop()` skips every remaining teardown AND every on-unload callback: the four `EVENT_HOMEASSISTANT_STARTED` cancels never run, the listeners stay armed, and the engines below the raising line are left un-stopped and un-latched. That is exactly the orphan-start this area was hardened to prevent, reachable by a single unexpected exception.

**Why:** Found by the Claude adversarial pass (2026-08-19), rated P1. Not fixed in that ship because containment needs a decision per step — some failures should abort the unload and some should not — and the branch was already carrying five review fixes. The platform-unload-first ordering landed there does reduce the blast radius, but does not close this.

**How to apply:** Wrap each teardown step so unload cannot raise (`contextlib.suppress` plus `LOGGER.exception`, or one `try/finally` that still returns True), keeping the platform-unload abort at the top as the only early return. The `_stopped` latch is only "downstream of every ordering" if unload actually reaches it.

**Resolution:** A local `_teardown(step, run)` wrapper in `async_unload_entry` runs each of the six stops/closes under `try/except Exception` with `LOGGER.exception`, so one raising step skips nothing and the function always returns True past the platform-unload abort — which stays the only early return, and a test pins that the containment did not swallow it. Regression test makes the *first* step (`video_analyzer.stop`) raise and asserts the sentinel/discovery stops still ran, unload reported success, and the on-unload callbacks (client close, listener cancels) still fired — the exact chain FAILED_UNLOAD severs.

**Effort:** S
**Priority:** P2
**Completed:** v3.30.10 (2026-08-21)

---

### Setup registers services, views, and dispatchers that unload never removes

**What:** Seventeen `hass.services.async_register` calls, the `EnrollPersonView`, and the `http_registered` flag are all created in `async_setup_entry` and never removed. `grep services.async_remove` returns nothing. Service handlers close over that generation's `baseline_updater`, `audit_store`, `proposal_store`, `rule_registry`, and `person_gallery`, so after an unload `hga.sentinel_get_baselines` is still callable and reaches a closed pool. Worse for the view: `http_registered` is never cleared, so a remove-and-re-add (which builds a *new* `ConfigEntry`, unlike a reload) leaves the original view alive dereferencing `runtime_data` on the deleted entry — `AttributeError` on every enroll POST until HA restarts.

**Why:** Found by the Claude adversarial pass (2026-08-19). Re-registration by name means the services pin one generation rather than accumulating per reload, so this is a correctness and stale-reference problem rather than an unbounded leak — but it is a live API surface on an unloaded entry.

**How to apply:** Wrap the registrations in `entry.async_on_unload` (services via `hass.services.async_remove`), and clear `hass.data[DOMAIN]["http_registered"]` on unload, or key the view off the current entry rather than the one that happened to register it.

**Resolution:** Services: all seventeen registrations route through a new `_register_entry_service` helper that registers the service and immediately registers `hass.services.async_remove` via `entry.async_on_unload` — the adjacency guarantees every remove has a matching register even when setup aborts between service blocks, and on-unload running after failed setups covers every abort path. View: clearing `http_registered` would have been the wrong fix (aiohttp routes cannot be removed, so a re-registered view would sit behind the original) — instead `EnrollPersonView` no longer pins the registering entry and resolves the currently LOADED entry per request, returning 503 when none is (and a mid-upload teardown is caught and returned as the same 503 rather than a 500); `http_registered` keeps its now-correct register-once-per-run semantics. The remove-and-re-add regression (new `ConfigEntry`, dead pinned `runtime_data`) is test-pinned, as is service removal on unload. The `_on_entry_changed` dispatcher named in the title was already wrapped. The two-loaded-entries residual (unloading either would remove domain services for both, and the view could serve an arbitrary entry) was closed structurally in the same ship: `single_config_entry: true` in the manifest — the whole integration was already de-facto single-entry (domain-global services, one DB, one Sentinel), both adversarial passes converged on it, and the flag makes HA refuse a second entry at the source.

**Effort:** M
**Priority:** P2
**Completed:** v3.30.10 (2026-08-21)

---

### services.yaml advertises confirm_enroll but nothing registers it

**What:** `services.yaml:19` defines `confirm_enroll`, but no `async_register` call for it exists anywhere in the integration — 18 services advertised, 17 registered. The Developer Tools services page shows a service that errors with "Unable to find service" when called.

**Why:** Pre-existing metadata staleness surfaced by the cross-model doc review during the v3.30.10 ship (the ship's service-deregistration sweep enumerated the real 17). Docs-only scope kept it out of that PR.

**How to apply:** Either delete the `confirm_enroll` block from `services.yaml` (if the confirm flow it referenced is gone for good) or register the handler it was meant to describe. Check translations for a matching `services.confirm_enroll` key and remove it in the same change.

**Effort:** S
**Priority:** P3
**Depends on:** —

---

### Decide whether face enrollment should be admin-only

**What:** `EnrollPersonView` (`http.py`) has `requires_auth = True` but no admin check, and the `hga.enroll_person` service is likewise callable by any user context — so any authenticated HA user (or a leaked limited-scope long-lived token) can enroll faces into the gallery that feeds security-relevant recognition (Sentinel unknown-person rules, recognized-people sensors). Poisoning shape: enroll an arbitrary face under a trusted resident's name and the phantom becomes "recognized". Reserved-label refusal still applies; this is about *who*, not *what*.

**Why:** Security-specialist finding during the v3.30.10 ship (confidence 5 — a deliberate-decision flag, not a demonstrated exploit). Pre-existing behavior, not introduced by that ship's view rewiring. Deferred by user decision (2026-08-21): gating changes which household members can use the enroll card, so it deserves its own PR, docs update, and release note rather than riding a lifecycle batch.

**How to apply:** If admin-only: check `request[KEY_HASS_USER].is_admin` in `post()` (mirroring HA's admin-gated views) and require admin context in `_handle_enroll_person` for parity; update `docs/camera-entities.md` and the enroll card docs; release-note the behavior change. If open-by-design: document explicitly that any authenticated user may enroll.

**Effort:** S
**Priority:** P2
**Depends on:** —

---

### Services stay callable during the unload teardown window

**What:** Service removal is an on-unload callback, and HA runs those only after `async_unload_entry` returns — so for the whole teardown sequence (platform unload, six engine stops, pool close) every `hga.*` service is still registered and callable against the half-torn-down generation. A call landing in that window reaches a stopped engine or a closing pool and fails with an opaque internal error rather than "service does not exist". Narrowed by v3.30.10 (before it, the window was *forever*), not closed.

**Why:** Converged on by the red-team and adversarial passes during the v3.30.10 ship (cross-model). Same accepted shape as the enroll-view mid-request race (which v3.30.10 downgraded from a crash to a 503); this is the service-surface sibling, recorded so the CHANGELOG's "removed on unload" reads with the right precision.

**How to apply:** Cheap guard in the shared handler path: refuse with a deterministic message when `entry.state is not ConfigEntryState.LOADED` (services only matter on a loaded entry). Alternatively accept and document — the window is a few awaits wide.

**Effort:** S
**Priority:** P3
**Depends on:** v3.30.10

---

### Zero-camera platform setup after HA has started arms a listener that never fires

**What:** `image.py` and `sensor.py` defer camera discovery to `EVENT_HOMEASSISTANT_STARTED` whenever initial discovery finds zero cameras — with no `hass.is_running` branch, unlike the four engine call sites. An entry set up *after* HA finished starting (first install before any camera integration, or a reload while cameras are momentarily absent) arms a listener for an event that already fired: no camera entities are ever created until the next reload, and the listener sits dead until unload cancels it (v3.30.10; before that it leaked outright).

**Why:** Pre-existing behavior surfaced by the red-team and adversarial passes during the v3.30.10 ship; the refactor makes it tidier but not better. Every current test runs under `CoreState.not_running`, so the already-running case is unpinned.

**How to apply:** Either skip registration when `hass.is_running` (documenting discovery as a one-shot snapshot), or replace the STARTED listener with entity-registry/state-changed-driven camera discovery (the structural fix — cameras added later would get entities without a reload). Pin the chosen behavior with a `CoreState.running` test.

**Effort:** M
**Priority:** P3
**Depends on:** v3.30.10

---

### The stop-latch is copy-pasted across four engines

**What:** `_stopped` now exists in four classes with near-identical bodies and near-identical ~11-line rationale comments: `sentinel/engine.py`, `sentinel/discovery_engine.py`, `sentinel/baseline.py`, and `core/video_analyzer.py`. The three `stop()` sites are byte-identical; the only per-class variation is the class name in the debug string.

**Why:** Found by the maintainability pass (2026-08-19). Deliberately not done in that ship: a mixin touches `video_analyzer.py`, which #559 had just landed, and the branch was already carrying four review fixes. The cost is that a fifth engine must re-derive the invariant from prose rather than inherit it.

**How to apply:** Extract something like `StopLatchMixin` (`_stopped`, a `_refuse_start_if_stopped()` guard logging `type(self).__name__`, a `_latch_stopped()`), have all four use it, and keep the full rationale in one docstring the four sites point at in a single line. If a mixin is rejected, at minimum collapse the four prose blocks to a one-line pointer at `core/lifecycle.py:defer_start_until_hass_started` (moved there in v3.30.10), which already carries the canonical explanation.

**Effort:** S
**Priority:** P3

---

### Unload can close the OpenAI transport under an in-flight sync Sentinel call

**What:** `run_sentinel_model_call` (`core/utils.py`) deliberately prefers the *sync* `invoke`, dispatched via `asyncio.to_thread`. Cancelling that task cancels the awaiting future but cannot interrupt the executor thread, which stays inside the blocking request for up to the 120 s client timeout. `async_unload_entry` then closes the client's connection pool underneath it.

**Why:** Found by the security pass during the lifecycle-leak ship (2026-08-19). Not a security-boundary break — Sentinel triage fails open, so an aborted triage call cannot suppress an alert. The impact is reload-time errors and "Future exception was never retrieved" noise. Moving the close after `async_unload_platforms` (done in that ship) fixes the *entity* half of the problem but not the executor-thread half.

**How to apply:** Expose a helper in `core/utils.py` that awaits the tracked sentinel LLM tasks with a bounded timeout before the transport is closed, or gate the executor bodies on a per-entry closed flag. Note that the entity half of this is already handled — `async_unload_entry` now unloads platforms before any teardown — so what remains is specifically the executor thread, which no entity teardown can interrupt.

**Effort:** M
**Priority:** P3

---

### The OpenAI http client is built even when no OpenAI provider is configured

**What:** `openai_http_client = await hass.async_add_executor_job(partial(httpx.Client, timeout=120))` runs unconditionally, before any `if openai_ok:` / `if openai_compatible_ok:` check. An Ollama-only installation allocates one `httpx.Client` per setup that nothing ever uses.

**Why:** Found by the Claude adversarial pass (2026-08-19). It means the leak fixed in that ship was being taken by users with no OpenAI configuration at all. Harmless now that the close is registered at construction, so this is tidiness rather than a leak.

**How to apply:** Construct it lazily at the first provider that needs it, or guard it behind the same condition those providers use. Keep the `entry.async_on_unload` close registration adjacent to wherever it ends up.

**Effort:** S
**Priority:** P3

---

### hass.is_running is True while HA is still starting

**What:** `hass.is_running` returns True for both `CoreState.starting` and `CoreState.running` (`core.py:448-450`), but the four deferred-start sites gate on it, and `defer_start_until_hass_started`'s (`core/lifecycle.py`, since v3.30.10) docstring asserts engines "cannot start before Home Assistant has finished starting." An entry added or reloaded during `CoreState.starting` starts all four inline instead of deferring, so the invariant the docstring states does not hold on that path.

**Why:** Found by the Claude adversarial pass (2026-08-19). Pre-existing — the gate predates the helper — but the helper's docstring is what promotes it to a stated invariant.

**How to apply:** Either change the four gates to `hass.state is CoreState.running`, which is the predicate matching the prose, or soften the docstring to say what the code does. Prefer the former only after checking that starting-state inline starts are not load-bearing for the restore path.

**Effort:** S
**Priority:** P3

---

### Lifecycle paths with no test coverage

**What:** Three gaps the lifecycle-leak ship surfaced but did not close. (1) `_on_entry_changed` has zero coverage anywhere in the repo — it stops all three engines then schedules a reload, and is the one flow where a cross-generation latch leak would surface. (2) The baseline updater's deferred-start site is unreachable in every setup-level test, because the harness patches `build_database_uri_from_entry` to `None`, so `pool` is `None` and `baseline_updater` is never built; the same gap hides the `baseline_updater is not None` branches of `_stop_background_tasks` and `async_unload_entry`. (3) `SentinelEngine.stop()` on a *running* engine is untested, including the fact that it latches the **shared** `SentinelBaselineUpdater` that `SentinelDiscoveryEngine` and `HGAData` also hold.

**Why:** Found by the coverage audit and testing pass (2026-08-19). Point (3) is the one with teeth, and the adversarial pass sharpened it into an asymmetry worth naming: `SentinelEngine.stop()` sets its own latch *before* the `if self._task is None: return`, but stops the shared baseline updater *after* it. So whether the shared updater survives `sentinel.stop()` depends on whether sentinel had ever started — a never-started engine latches only itself, a started one kills the updater that two other objects still hold. No comment states this and no test exercises it. Harmless today only because `_on_entry_changed`, the sole stop-without-unload caller, redundantly stops all three explicitly.

**How to apply:** The new `test_a_failed_setup_closes_the_openai_http_client` shows the recipe for a pool-bearing harness — patch `build_database_uri_from_entry` to a URI and `AsyncConnectionPool` to a stand-in (and stub the langgraph stores, whose real versions spawn background batch tasks the failure path never reaps). Reuse it to reach the baseline site and to drive `_on_entry_changed` through a real stop-then-reload.

**Effort:** M
**Priority:** P3

---

## Completed

### Baseline-deviation notifications guess the display unit from the entity_id

**What:** `_baseline_deviation_mobile_message` (`sentinel/notifier.py:904`) picks the display unit with `"W" if "power" in entity_id else "kWh" if "energy" in entity_id else ""`. A kW-denominated sensor renders as e.g. "0.4W vs usual 0.3W", and a power sensor without "power" in its entity_id gets no unit at all. Surfaced during the #461 unit-normalization review (v3.21.3).

**How to apply:** Plumb the sensor's `unit_of_measurement` into the baseline finding evidence in `sentinel/baseline.py` (alongside `current_value`/`baseline_value`, which stay native) and render it verbatim in the notifier, falling back to the current heuristic for old persisted findings.

**Resolution:** Shipped as prescribed in the metric-aware baseline-copy fix: `evaluate_baseline_deviation` and `_evaluate_dow_anomaly` capture `unit_of_measurement` and `device_class` into finding evidence, and `_baseline_deviation_mobile_message` renders the captured unit (sanitized — control/bidi characters stripped, length-capped). The entity_id heuristic survives only for legacy persisted findings whose evidence lacks the unit key; a present-but-empty unit means a genuinely unitless sensor and no unit is fabricated. Anomaly-id stability across the new evidence keys is preserved via `DISPLAY_ONLY_EVIDENCE_KEYS` / `hashable_evidence` in `sentinel/models.py`. Follow-up spoofing gap for `friendly_name`-derived text is tracked as its own P3 above.

**Effort:** S
**Priority:** P3
**Depends on:** None
**Completed:** v3.31.1 (2026-08-23)

---

### Sentinel unknown-person rules are suppressed by any non-empty recognized list

**What:** `unknown_person_camera_night_home` (and the dynamic no-home/when-home variants) skip whenever `recognized_people` is truthy — but the raw list dispatched to the image entity and Sentinel snapshot contains the literal `"Unknown Person"` and `"Indeterminate"` strings, so with face recognition enabled, a detected-but-unrecognized visitor *suppresses* the unknown-person rules rather than firing them. Decide whether the rules should filter negative identities before the emptiness check, or whether the snapshot layer should strip them.

**Why:** Pre-existing behavior surfaced by the v3.30.0 adversarial review (the merge provably cannot change rule firing, but the rules' emptiness check is only vacuously aligned with their name). Changing it changes alerting behavior, so it needs its own focused decision, docs, and field validation — not a drive-by fix.

**Effort:** M
**Priority:** P2
**Depends on:** v3.30.0

**Resolution:** Fixed in the unknown-person rules overhaul: the rules now fire on a positive "Unknown Person" label (normalized), enrolled names suppress as accompanied-guest, and a 10-minute freshness gate keyed to the recognition event prevents stale re-fires. See the three new Sentinel Rules follow-up items for the deliberately deferred edges.

**Completed:** v3.30.11 (2026-08-22)

---

### Config entry lifecycle leaks in the deferred-start block

**What:** Three sibling leaks in the same `async_setup_entry` code block that
#559 fixed for the video analyzer alone:

1. `_start_sentinel`, `_start_discovery`, and `_start_baseline` registered
   `EVENT_HOMEASSISTANT_STARTED` listeners that were never wired into
   `entry.async_on_unload`, so an entry that unloaded, reloaded, was disabled,
   or failed setup before HA finished starting later started the *obsolete*
   engines — with stale options — alongside the replacement entry. As plain
   sync functions `HassJob` inferred `Executor`, so the bodies ran on a worker
   thread.
2. The `EVENT_HOMEASSISTANT_STOP` listener was registered unconditionally on
   every setup and never removed, so each reload added another closure
   capturing that generation's engines and client.
3. `async_unload_entry` never closed the synchronous `openai_http_client`; its
   only close sat inside the leaked STOP handler.

**Why:** Surfaced during the #559 review (2026-08-18) by both the maintainer
pass and the Codex adversarial pass. Deferred out of #559 by user decision so
the contributor's focused fix could land on its own.

**Resolution:** The four deferred-start sites now share one
`_defer_start_until_hass_started(hass, entry, start)` helper carrying #559's
full shape — `@callback` listener, `fired` flag, `@callback` cancel wrapped in
`entry.async_on_unload`. Because the cancel cannot close the window between
`stop()` and HA running the on-unload callbacks, `SentinelEngine`,
`SentinelDiscoveryEngine`, and `SentinelBaselineUpdater` each gained the same
one-way `_stopped` latch #559 gave `VideoAnalyzer`: set at the top of `stop()`,
checked first in `start()`. The STOP listener is wrapped in
`entry.async_on_unload` — with the same fired-flag guard as the helper, set by
a `@callback` shim rather than inside the coroutine, because on-unload
callbacks run only after `async_unload_entry` returns and a shutdown racing an
in-flight reload runs both paths. It stays a listener because HA does not
unload config entries at shutdown, making it the only stop path for a
still-loaded entry.

`openai_http_client`'s close is registered with `entry.async_on_unload` at
construction rather than written into `async_unload_entry`. Home Assistant runs
those callbacks on *every* failed setup as well as on unload
(`ConfigEntry.async_setup`: `finally: if not result … _async_process_on_unload`),
so one hook covers every abort path — including the ten or so that are raises
rather than `return False` — and, on the success path, lands after
`async_unload_platforms` for free. `async_unload_entry` now unloads platforms
FIRST and aborts on refusal, so no teardown is irreversible before Home
Assistant has agreed the entry can go.

Test count went 4 → 16 in `test_deferred_start_listener.py`, each new test
verified to fail against the unfixed code. Review found five issues in the
first cut, four of them multi-model confirmed: the client leaked on
setup-failure paths, the STOP remover was unguarded, the close ran before
platform unload, the removal test passed vacuously because nothing asserted the
listener existed, and — from the adversarial pass — the `PoolTimeout` cleanup
in the first draft was dead code, because psycopg-pool 3.3.0 defaults `open()`
to `wait=False` and never raises there. The Codex structured pass then caught a
P1 introduced by the fix itself: propagating a failed platform unload while the
teardown had already run would strand a retained entry on stopped, latched
engines.

**Effort:** S
**Priority:** P1
**Completed:** (2026-08-19)

### Lovelace health card example for baseline attrs

**What:** Add a Lovelace dashboard card YAML snippet to `README.md` showing `baseline_entity_count`, `baseline_fresh_count`, and `baseline_rules_waiting` from `sensor.sentinel_health`.

**Why:** Users enabling baseline collection have no visual confirmation it's working; a README dashboard example closes the discoverability gap.

**Resolution:** Completed by the README "Community Dashboards" section (discussion #513, @hruba202): the featured Sentinel health flex-table-card recipe surfaces `baseline_entity_count`, `baseline_fresh_count`, `baseline_rules_waiting`, and the other health KPIs. Placement is the Community Dashboards section rather than the Baseline section, but the discoverability goal is met.

**Completed:** docs PR for discussion #513 (2026-07-29)

### Snapshot retention misses batches that never reach _finalize

**What:** Deletion is deque-driven (in-memory), but registration was coupled to
successful analysis completion (`_finalize`). Six runtime paths abandoned
captured files without registration: the two `_analyze_and_finalize` early
returns, the worker's blanket exception handler, notify/store exceptions inside
`_finalize`, the `_get_batch` backlog drop, and event-hold-buffer eviction.

**Why:** VLM outages and backlogs made these paths routine; files accumulated
unboundedly. Third occurrence of the same bug class (#489 dedupe-skip was a
point patch), so the fix moved registration to capture success in
`_capture_snapshot` — a single site that makes every downstream drop
retention-irrelevant — and removed the per-site registrations. In-flight frames
(≤ ~105/camera) can never reach the deque's pop position (cap 200).

**Effort:** S
**Priority:** P1
**Completed:** v3.18.2 (2026-07-19)

### Restart orphans snapshot files predating the restart

**What:** Retention deques are in-memory with no filesystem sweep, so every
on-disk snapshot from before an HA restart was orphaned forever. Fixed with a
one-shot startup task (`_seed_retention_from_disk`) that scans each
`camera_*` snapshot directory (skipping `_latest/` and non-camera dirs like
`faces/`), merges pre-existing files into the retention deque as the oldest
entries (never after live captures, so new frames can't be rotated out
first), and runs the normal budget shrink with the usual latest-asset and
protection guards.

**Effort:** M
**Priority:** P2
**Completed:** v3.18.2 (2026-07-19)

### iOS notification priority tiers

**What:** `sentinel/notifier.py` now uses `_SEVERITY_INTERRUPT_LEVEL` (`high` → `time-sensitive`, `medium` → `active`, `low` → `passive`) and `_SEVERITY_TITLE` dicts. `async_notify()` derives severity from the finding, selects the interrupt level and title, and passes `"push": {"interruption-level": level}` in the notification data block.

**Why:** All notifications previously used the same `active` interruption level, training users to ignore them — including security alerts.

**Effort:** S
**Priority:** P1
**Completed:** v3.6.9 (2026-03-31)

---

### Appliance completion detection in baseline deviation

**What:** `sentinel/baseline.py` now detects appliance cycle completion: power-class entities on dedicated appliance circuits (washer, dryer, dishwasher, etc.) with `current_value < COMPLETION_THRESHOLD_PCT × baseline_value` emit `severity="low"` with `evidence["is_completion"] = True`. `sentinel/notifier.py` checks `finding.evidence.get("is_completion")` and uses passive interruption level with completion framing.

**Why:** Washer/dryer finishing a cycle triggered `baseline_deviation` with high-severity framing ("stopped unexpectedly"), causing false-positive security alerts.

**Effort:** S
**Priority:** P1
**Completed:** v3.6.9 (2026-03-31)

---

### Presence-aware lock severity for `unlocked_lock_at_night`

**What:** `sentinel/rules/unlocked_lock_at_night.py` now reads `anyone_home` from `snapshot["derived"]` and emits `severity="low"` when home is occupied, `severity="high"` when away. Evidence dict includes `"anyone_home": anyone_home`.

**Why:** High-severity `time-sensitive` iOS alert for an unlocked door while occupants are home is noise that trains operators to ignore real alerts.

**Effort:** S
**Priority:** P1
**Completed:** v3.6.9 (2026-03-31)

---

### Notification batching for medium/low severity bursts

**What:** `SentinelNotifier` now tracks `_notification_times`, `_held_batch`, and `_batch_cancel`. When more than `_BATCH_RATE_LIMIT` medium/low pushes are sent within `_BATCH_WINDOW_SECONDS`, subsequent notifications are buffered and flushed as a single passive-priority summary after `_BATCH_FLUSH_DELAY_SECONDS`. High-severity findings always bypass batching.

**Why:** Burst of 7 notifications in 18 minutes cluttered the lock screen and trained users to dismiss without reading.

**Effort:** M
**Priority:** P1
**Completed:** v3.6.9 (2026-03-31)

---

### Centralize action code vocabulary in `const.py`

**What:** Defined `ACTION_CODES: dict[str, str]` in `const.py` mapping action code → plain English description. Imported it in `explain/prompts.py` to build the `SYSTEM_PROMPT` vocabulary line dynamically. Used constants in notifier and execution service instead of bare strings.

**Why:** `SYSTEM_PROMPT` contained a hardcoded list of action codes. Any new rule adding a new action code would silently cause the LLM to invent its own English meaning — recreating the exact class of production bug fixed in PR #346. Flagged independently by eng review and Codex outside voice.

**Effort:** S
**Priority:** P2
**Completed:** v3.7.0 (2026-04-03)

---

### Fix transitive union-find spatial contamination in correlator

**What:** Added `_eject_camera_area_violations()` post-grouping pass to `SentinelCorrelator.correlate()`. For any group containing a `camera_entry_unsecured` finding with a known area, ejects any other finding whose area is non-empty and differs from the camera's area, returning it as a singleton.

**Why:** The area-aware `_COMPLEMENTARY_PAIRS_REQUIRE_AREA_MATCH` guard only protects direct pairwise checks. With three simultaneous findings (camera Front + lock Front + entry Garage), the camera in Front ended up in a compound with the Garage entry via the bridging lock — the exact false spatial claim the camera_entry_unsecured fix was designed to prevent.

**Effort:** S
**Priority:** P2
**Completed:** v3.7.0 (2026-04-03)

---

### Discovery quality metrics on health sensor

**What:** `SentinelDiscoveryEngine` now tracks per-cycle stats (`candidates_generated`, `candidates_novel`, `candidates_deduplicated`, `proposals_promoted`, `unsupported_ttl_expired`) in `_discovery_cycle_stats`. `SentinelHealthSensor` exposes these as attributes via `SIGNAL_SENTINEL_RUN_COMPLETE`.

**Why:** After the deduplication fix, operators had no visibility into whether dedup was working — they couldn't distinguish "LLM generating zero new ideas" from "LLM generating many but all caught by dedup."

**Effort:** S
**Priority:** P2
**Completed:** v3.7.0 (2026-04-03)

---

### Daily digest notification

**What:** `SentinelEngine` now schedules a daily push notification summarizing the past 24h findings. Controlled by `CONF_SENTINEL_DAILY_DIGEST_HOUR` (default 07:00). Duplicate digest triggers within a session are deduplicated by checking `self._digest_task.done()` before scheduling.

**Why:** Operators had no visibility into Sentinel activity without actively checking the audit store or Lovelace. A morning summary gives awareness without intra-day noise.

**Effort:** S
**Priority:** P2
**Completed:** v3.7.0 (2026-04-03)

---

### `trigger_source` breakdown on health sensor

**What:** `SentinelHealthSensor` now exposes `trigger_source_breakdown: {"poll": N, "event": M, "on_demand": K}` as a rolling 24-hour count attribute, populated by querying `AuditStore` during the health update cycle.

**Why:** `trigger_source` was populated in audit records but not surfaced anywhere useful. Operators could not tell if Sentinel was poll-heavy vs event-driven without inspecting raw audit records.

**Effort:** S
**Priority:** P2
**Completed:** v3.7.0 (2026-04-03)

---

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

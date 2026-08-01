# Issue 433: Basic and Advanced Setup Plan

Issue #433 asks for setup to be easier for basic users while still allowing
advanced users to adjust specific configuration. The current integration already
has separate subentry flows for model providers, features, database, STT, and
Sentinel, but the setup path exposes too many tuning controls before a new user
has a working configuration.

## Goal

Provide two clear setup paths:

- **Basic setup** creates a complete recommended configuration with minimal
  required input.
- **Advanced setup** preserves the current detailed controls for users who want
  per-feature models, fallbacks, tuning, and Sentinel internals.

This should be a config-flow improvement first, with documentation updated after
the behavior is settled.

## Design decisions to resolve before writing code

The following questions gate the implementation. Answers should be agreed upon
before any code is written, ideally confirmed with a prototype or mockup.

### A. How does Basic setup create all three feature subentries in one flow?

HA's subentry model normally requires a separate user action per subentry. Two
viable options:

- **Option A — compound flow:** Basic setup programmatically calls
  `async_create_subentry` for all three feature subentries before returning.
  Requires verifying that `SubentryFlowManager` supports programmatic subentry
  creation from within a running flow.
- **Option B — collapsed iteration (recommended):** Basic setup remains a single
  feature subentry flow execution. `FeatureSubentryFlow` already iterates all
  three features sequentially; Basic mode auto-enables all three and skips the
  per-feature tuning screens. No new HA internals required.

Option B is preferred. The plan below assumes Option B.

Sentinel creation is explicitly out of scope for the feature flow. Sentinel
remains under its own `+ Sentinel` subentry entry point with its own Basic /
Advanced split (Step 6). Mixing subentry ownership by having the feature flow
trigger Sentinel creation would couple unrelated flows and make reconfiguration
harder.

### B. Original issue scope

The upstream issue is a single sentence with no detailed requirements. All
product decisions in this plan (which features are default-on, what Sentinel
Basic exposes, the summary screen contents) are design choices made to fill that
gap. A prototype or screenshot mockup should be reviewed before Step 1 is
implemented.

## Plan

### 1. Resolve the compound-subentry mechanism

Confirm Option B (collapsed iteration) works for the feature flow. Sentinel
creation is out of scope here; it has its own entry point and flow (see Step 6).

### 2. Add a setup mode choice

Add a first step to the `+ Setup` flow in
`custom_components/home_generative_agent/flows/feature_subentry_flow.py`.

The first screen should offer:

- **Basic setup**
- **Advanced setup**

Basic setup should create the standard full setup with recommended defaults.
Advanced setup should preserve the existing detailed per-feature flow.

`async_step_reconfigure` must bypass this screen entirely and go directly to the
Advanced flow, since users reconfiguring an existing subentry already have a
configuration.

### 3. Handle a missing provider with an abort

The feature setup path works best when at least one model provider already
exists. If Basic setup starts and no provider exists, abort with
`FlowResultType.ABORT` and a clear reason string directing the user to add a
provider subentry first, then retry.

Do not attempt to redirect to another running flow or inline provider creation;
HA config flows have no redirect-and-resume mechanism.

Provider-specific validation should remain in
`custom_components/home_generative_agent/flows/model_provider_subentry_flow.py`.

### 4. Define Basic setup as guided full setup

Basic setup should:

- Enable the default features with the following capability requirements:
  - Conversation (requires a chat-capable provider)
  - Camera Image Analysis (requires a VLM-capable provider; skip and note in
    summary if none exists)
  - Conversation Summary (requires a summarization-capable provider)
- Assign the first compatible provider to each feature, where "first" means the
  first entry in `entry.subentries.values()` iteration order that advertises the
  required capability. This matches how the existing provider option helper
  iterates subentries, so the behavior is consistent and testable. If no
  compatible provider exists for a feature, disable that feature and surface it
  in the summary screen.
- Use existing recommended model defaults from
  `custom_components/home_generative_agent/const.py`.
- Ask only for required external resources:
  - database connection
  - notify service, if available and useful

Basic setup should not expose model temperature, keepalive, context size,
reasoning, fallback providers, or Sentinel tuning.

### 5. Add a final status screen after validated writes

`FeatureSubentryFlow` does not end with `async_create_entry`. It writes subentry
data incrementally during the flow via `async_add_subentry` /
`async_update_subentry`, then finishes by aborting with reason `setup_complete`.
A "pre-write summary" is therefore not practical without restructuring when
writes occur.

Instead, add a final read-only status step immediately before the
`setup_complete` abort. All feature and database writes have already been
committed at that point. The screen confirms what was written and surfaces any
items that still need attention.

The screen should display:

- provider assigned per feature (or a warning if a feature was skipped due to
  missing provider capability)
- database configured
- enabled features
- Sentinel not yet configured (reminder to run `+ Sentinel` if desired)
- reminder to set Home Generative Agent as the voice assistant

The step reads its display values from the flow's accumulated state, not from
re-querying the config entry.

### 6. Split Sentinel into Basic and Advanced setup

Update `custom_components/home_generative_agent/flows/sentinel_subentry_flow.py`
so Sentinel also has Basic and Advanced paths. Apply the same mode selector as
the feature flow, and skip the selector on `async_step_reconfigure`.

Basic Sentinel setup should expose only:

- enable or disable anomaly alerting
- notify service
- daily digest
- optional PIN

Advanced Sentinel setup should keep the current full schema:

- intervals
- cooldowns
- discovery
- baseline configuration
- day-of-week patterns
- camera-entry JSON links
- all other current Sentinel tuning fields

Use `_default_payload()` for Basic defaults so the path stays aligned with
current recommended constants.

Note: the database step lives inside `feature_subentry_flow.py` and is shared by
all features. Sentinel Basic setup assumes the database is already configured via
the feature flow. Do not duplicate the database step inside the Sentinel flow.

### 7. Preserve the existing Advanced path

Advanced setup should keep the current power-user behavior:

- per-feature provider selection
- per-feature model settings
- fallback providers
- Ollama-specific tuning
- database fields
- full Sentinel settings

### 8. Update strings end-to-end

Add and update strings in `custom_components/home_generative_agent/strings.json`
and mirror them in
`custom_components/home_generative_agent/translations/en.json`.

Required additions:

- Mode-selection step: title, description, Basic label, Advanced label.
- Abort reason when no provider exists.
- Final status screen: all status labels and the voice assistant reminder.

Required reviews:

- "Model & advanced options" and similar labels: update wording to make clear
  these are optional tuning controls, not required setup steps.

### 9. Add focused tests

Add flow coverage in
`tests/custom_components/home_generative_agent/test_subentries.py`.

Cover:

- Basic setup creates default feature subentries with all three features when a
  compatible provider exists.
- Basic setup skips Camera Image Analysis and notes it in the summary when no
  VLM-capable provider exists.
- Basic setup aborts with the expected reason string when no provider exists at
  all.
- Basic setup assigns the first compatible provider in `entry.subentries.values()`
  iteration order.
- Basic setup status screen shows correct status after writes and before the
  `setup_complete` abort.
- `async_step_reconfigure` bypasses the mode selector and goes to the Advanced
  flow.
- Advanced setup still reaches the current feature provider, model, and database
  steps.
- Sentinel Basic setup stores `_default_payload()` values plus selected user
  fields.
- Sentinel Advanced setup still accepts the current full payload.
- Sentinel `async_step_reconfigure` bypasses the mode selector.

### 10. Update documentation after behavior is implemented

After the config-flow changes are in place, update:

- `docs/installation.md`: make the first-run path "Add provider -> Basic setup
  -> Voice Assistant".
- `docs/configuration.md`: explain Basic vs Advanced setup.
- `docs/sentinel.md`: explain Basic Sentinel vs advanced tuning.
- `README.md`: keep it short and point to the canonical docs.

### 11. Verify the change

Run focused flow tests first:

```bash
./hga/bin/pytest tests/custom_components/home_generative_agent/test_subentries.py
./hga/bin/pytest tests/custom_components/home_generative_agent/test_options_schema.py
```

Then run:

```bash
hga/bin/ruff check custom_components/home_generative_agent tests/custom_components/home_generative_agent/test_subentries.py
hga/bin/pyright
git diff --check
```

For documentation-only follow-up edits, `git diff --check` is the high-value
minimum.

## Recommendation

Implement this as one focused PR for the flow behavior and matching docs. A
docs-only fix would not fully address the issue because the current product
surface still exposes advanced configuration too early.

Confirm the Option B mechanism and review a prototype before writing flow code.

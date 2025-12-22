# Issue #170 Subentry Flow Notes

Summary of requested changes from the provided issue comment regarding Home Generative Agent (HGA) setup and subentry flows.

## Goals
- Simplify initial HGA setup and scale to more model providers while improving performance (e.g., Triton gains matter).
- Remove legacy URLs from the main config flow and rely on guided subentry flows.
- Auto-create default features on HGA creation; users only enable/disable optional features rather than adding them manually.

## Proposed Flow Structure
### HGA Integration Entry
1. User adds HGA integration and sees initial instructions.
2. Default feature subentries are created automatically.

### "Setup" (Feature/Database Subentry)
- Combined flow for feature enablement and database setup using the "feature" subentry type.
- Steps:
  1. `step_feature_enable`: toggle optional features (required features like Conversation always enabled).
  2. `step_<feature_name>`: configure each enabled feature using existing translation strings; source-aware logic loops through enabled features when invoked from Setup, or shows a single feature when entered from its gear icon.
  3. `step_database`: configure database as part of the feature flow.
  4. `step_instructions`: only shown when no model provider exists; instruct user to add a provider.
- If no model provider is configured, hide model-provider-specific schema fields and inform the user a default model will be assigned once available.
- Continue enforcing at most one subentry per feature type.

### "Add Model Provider" Subentry
- Steps:
  1. Deployment selection (Edge/Cloud).
  2. Provider selection.
  3. Provider-specific settings.
- On first provider creation, automatically associate it (and default models) with all features.

## Model Provider Guidelines
- Each provider configuration exposes only one endpoint/context; users create multiple providers when different settings are required (e.g., VLM vs Conversation).
- Feature-specific model options (e.g., model name, reasoning) appear on the feature screen, not in the provider screen.

## Subentry Identity and Enablement
- Use fixed subentry unique IDs per feature type (e.g., `database`, `conversation`); users do not choose the feature type.
- Home Assistant currently lacks subentry disable support. Alternatives discussed:
  - Rename entries with a " (Disabled)" suffix when simulating disable/enable (subject to user renaming conflicts).
  - Delete the subentry to "disable" a feature; persisting previous config in the main entry would be needed to restore state on re-enable.

## Navigation Expectations
- Recommended setup path:
  1. Click HGA integration → read instructions.
  2. Click "Setup" → enable/ configure features and database → receive model-provider instructions if needed.
  3. Click "Model Provider" → configure provider.
  4. Use a feature gear icon → configure only that feature.

## UX Walkthrough (Desired Flow)
This walkthrough describes what a user sees and does when following the redesigned setup.

### 1) Add HGA Integration (Config Flow)
- User installs/selects the HGA integration and starts the config flow.
- Screen shows only setup guidance (no legacy URL/API-key fields) and explains that defaults are being created.
- System auto-creates default feature subentries (Conversation and other optional features) with fixed IDs.
- The config flow ends by offering two buttons/links: **Setup** (feature/database subentry) and **Model Provider**.

### 2) Run "Setup" (Feature/Database Subentry)
- **Step: Feature Enable** – User toggles optional features; required ones stay enabled and non-editable.
- **Steps: Feature Settings** – For each enabled feature, the flow presents that feature’s screens (using the same strings as the per-feature gear icon). If no provider exists yet, provider-specific fields are hidden and a notice explains a default model will be assigned later.
- **Step: Database** – User configures the database alongside the feature setup.
- **Step: Instructions** – Only shown when no provider exists; reminds the user to add a model provider next. If desired, the UI can display a small warning badge like “Missing Model Provider” until one is added.

### 3) Add First "Model Provider" (Subentry)
- **Step: Deployment** – Choose Edge vs Cloud.
- **Step: Provider** – Pick the provider (e.g., OpenAI, Anthropic, etc.).
- **Step: Provider Settings** – Enter provider-specific options for that single endpoint/context.
- When this first provider is saved, the system automatically assigns it (with default models) to every feature. If different settings are needed per feature (e.g., VLM vs Conversation), the user can add separate providers later.

### 4) Adjust Individual Features (Gear Icon)
- Each feature’s gear icon opens the same feature screens as in Setup, but only for that specific feature.
- Users can adjust feature-level model options (model name, reasoning flags, etc.) here; provider forms stay generic and never include feature-specific knobs.

### 5) Toggling/Removing Optional Features
- Because native subentry disabling is unavailable, optional features are effectively “disabled” by deleting their subentry (or, optionally, renaming with a " (Disabled)" suffix). Re-enabling would recreate the subentry—potentially using stored defaults from the main entry to restore prior settings.

## Open Questions / Implementation Considerations
- Investigate whether config flows can chain directly to subentry flows (and subentry-to-subentry) via `next_flow`; current `FlowType` in `config_entries.py` suggests only config flows are allowed.
- Consider migration paths ensuring separate model-provider entries when settings differ across features.

## Changes Needed in the Current Implementation
- **Config flow restructuring:** Replace the existing single-step form that collects API keys and legacy URLs with an instruction-first flow that auto-creates default feature subentries and immediately guides users into the Setup and Model Provider subentries, rather than storing provider URLs directly on the main entry. This removes the need for the legacy URL fields still exposed in `async_step_user` and `_run_validations_user` in `config_flow.py`.
- **Feature subentry overhaul:** Convert `FeatureSubentryFlow` from a single form that lets users pick a feature type and provider into the multi-step "Setup" flow (enable toggles, per-feature configuration screens, embedded database step, instructions when providers are missing). Feature types should be fixed per default subentry, not user-selectable, and the flow should reuse the same steps whether launched from Setup (loop through enabled features) or a gear icon (single feature).
- **Database handling:** Fold the `PgVectorDbSubentryFlow` screens into the new Setup flow so database configuration happens alongside feature enablement instead of as a stand-alone subentry type.
- **Model provider flow changes:** Expand `ModelProviderSubentryFlow` into the requested three-step experience (deployment → provider → provider settings), remove feature-specific model options from provider forms, and on first provider creation automatically associate the provider and default models with all features.
- **Subentry lifecycle/identity:** Precreate one subentry per feature with fixed IDs, enforce at most one per type, and add simulated enable/disable handling (e.g., rename or delete/recreate) to align with the desired ability to toggle optional features despite Home Assistant lacking native subentry disable support.

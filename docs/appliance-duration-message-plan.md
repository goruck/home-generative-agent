# Appliance Duration Message Plan

Originating issue: [#391 Improve Message](https://github.com/goruck/home-generative-agent/issues/391)

## Problem Statement

The `appliance_power_duration` Sentinel notification can describe a specific
power sensor as the generic phrase "An appliance". A reported example was:

`An appliance in Room-A recently drew about 296 W for 633 minutes, exceeding the 60-minute threshold. Check the appliance.`

The user expectation is that the message names the actual Home Assistant entity
or appliance, not only the category. For example, a washer power sensor should
produce copy that starts with "Washer" rather than "An appliance".

## Current Behavior

`AppliancePowerDurationRule` builds finding evidence with `entity_id`, `area`,
`power_w`, `duration_min`, and `threshold_min`, but does not include the
snapshot entity's `friendly_name`.

When LLM explanations are enabled, the explainer is instructed to use only the
provided evidence. Without a friendly display name, the model can legitimately
fall back to generic wording such as "An appliance". The notifier also currently
prefers a valid LLM explanation over its generic fallback message, so adding a
better fallback alone would not reliably fix this issue.

## Goals

- Name the relevant appliance or power sensor in appliance-duration alerts.
- Keep the message short enough for mobile notifications.
- Avoid raw entity IDs when a useful display name is available.
- Keep the change scoped to `appliance_power_duration`.

## Non-Goals

- Do not redesign all Sentinel message generation.
- Do not add a user-facing message-template setting.
- Do not change anomaly detection thresholds or sustained-duration behavior.
- Do not add a separate persistent-notification code path. The deterministic
  appliance-duration formatter introduced here is reused by `_persistent_message()`
  (see Change 5), but no other persistent-notification behavior is changed.

## Proposed Changes

1. Add `friendly_name` to appliance-duration finding evidence.

   In `custom_components/home_generative_agent/sentinel/rules/appliance_power_duration.py`,
   copy `entity["friendly_name"]` into the evidence payload. Use `or None` so
   that an empty string is stored as `None` rather than `""`, keeping the
   fallback logic simple.

   Exclude `friendly_name` from the evidence passed to `build_anomaly_id()`.
   `friendly_name` is display-only and can change independently of the actual
   anomaly. Concretely: build the full evidence dict first, then pass a filtered
   copy — `{k: v for k, v in evidence.items() if k != "friendly_name"}` — to
   `build_anomaly_id()`, and pass the full dict to `AnomalyFinding`. No change
   to `build_anomaly_id()` itself is required.

2. Add deterministic mobile copy for `appliance_power_duration`.

   In `custom_components/home_generative_agent/sentinel/notifier.py`, add a
   helper that formats appliance-duration notifications from finding evidence.
   `_mobile_message()` should use this helper for `appliance_power_duration`
   before considering an LLM explanation. The LLM explanation is still generated
   upstream by the triage/explain pipeline when enabled; the helper simply
   bypasses it for the mobile message, so there is no change to inference cost
   or audit records.

3. Prefer friendly display names.

   Message construction should use the pattern
   `(evidence.get("friendly_name") or "").strip()` so that both `None` and `""`
   trigger the fallback. Strip common power-sensor suffixes with the existing
   `_strip_power_suffix()` helper. Preserve the original casing of user-provided
   friendly names so values like `EV Charger Power` do not become `Ev Charger`.
   Fall back to `_friendly_entity(entity_id)` when no friendly name is present.

   Note: `_strip_power_suffix()` currently matches title-case suffixes (e.g.
   `" Power"`, `" Energy"`). Update `_strip_power_suffix()` itself to do a
   case-insensitive suffix comparison (e.g. `name.lower().endswith(suffix.lower())`)
   while preserving the original casing of the returned string. This is safe to
   apply to both the appliance-duration path and the existing `is_completion` path
   in `async_notify()`, which already applies `.title()` after stripping. The
   entity-ID fallback can remain title-cased because it is generated from an
   identifier, not user-authored display text.

   Target copy:

   `Washer drew about 296 W for 633 min, above the 60 min threshold. Check it.`

4. Preserve existing behavior for other finding types.

   Other notification types should keep the current LLM-first behavior, except
   for existing deterministic paths such as `alarm_disarmed_during_external_threat`.

5. Reuse the deterministic formatter for persistent fallback copy.

   `_persistent_message()` should use the same appliance-duration formatter when
   no usable explanation is available. This prevents no-mobile-service installs
   from falling back to a generic "Appliance power duration" message while still
   preserving the existing LLM-first behavior for persistent notifications.

## Known Limitation: `anomaly_id` Instability

`build_anomaly_id()` currently hashes the full evidence dict including
`duration_min`, which increases each evaluation cycle as the appliance keeps
running. The `anomaly_id` for `appliance_power_duration` findings is therefore
already unstable across runs, and the per-finding cooldown in the notifier is
ineffective for this type.

This plan should not add a second, display-only source of churn. Add
`friendly_name` to the stored finding evidence, but build the anomaly ID from a
hash evidence dict that excludes `friendly_name`.

Fixing the existing `duration_min` instability, for example by excluding
time-varying fields from the hash, is a separate issue and is out of scope here.

## Test Plan

- Update `test_appliance_power_duration_triggers()` to assert that
  `friendly_name` is present in the finding evidence.
- Add notifier coverage showing `Washer Power` is rendered as `Washer`.
- Add notifier coverage for a lowercase `friendly_name` (e.g. `"washer power"`)
  rendering correctly as `Washer`.
- Add notifier coverage showing user-provided casing is preserved, for example
  `EV Charger Power` rendering as `EV Charger`.
- Add notifier coverage for fallback to the entity ID display name when
  `friendly_name` is missing (`None` or `""`).
- Add notifier coverage showing deterministic appliance-duration copy wins even
  when the LLM explanation says "An appliance".
- Add coverage showing `friendly_name` is not included in the anomaly ID hash
  input, so changing only the display name does not change the ID.

## Verification

Run focused tests:

```bash
./hga/bin/pytest tests/custom_components/home_generative_agent/test_rules_sentinel.py tests/custom_components/home_generative_agent/test_sentinel_notifier.py
```

Run targeted lint:

```bash
hga/bin/ruff check custom_components/home_generative_agent/sentinel/rules/appliance_power_duration.py custom_components/home_generative_agent/sentinel/notifier.py tests/custom_components/home_generative_agent/test_rules_sentinel.py tests/custom_components/home_generative_agent/test_sentinel_notifier.py
```

Run type checking:

```bash
hga/bin/pyright custom_components/home_generative_agent/sentinel/rules/appliance_power_duration.py custom_components/home_generative_agent/sentinel/notifier.py
```

# Releasing

How changes get from `main` to users. Two lanes: urgent fixes ship on their own, everything
else rides a scheduled bundle.

## Why this exists

Through v3.42.0 this project released after almost every merge — 42 releases in the 31 days to
2026-09-14, twelve of them in four days, five on 2026-09-12 alone. Every release is an update
badge in every user's HACS panel and a Home Assistant restart to clear it, and `hacs.json` sets
`hide_default_branch: true`, so a release is the only way anyone receives code. That cadence
asked users to restart their home automation five times in a day, and it ran the manual release
sequence often enough that bookkeeping slipped (the v3.35.0 changelog entry was missed entirely).

Bundling fixes that. What it must not break is the field-validation loop: this project depends on
reporters who own hardware the maintainer does not, and they can only test what they can install.
Pre-releases (below) are how both goals hold at once.

## The decision rule

Ask one question: **if this waits for the next bundle, what does a user experience?**

If the answer is "something that is broken stays broken," ship it now. If the answer is "they
don't get a new thing yet," bundle it.

### Ship immediately, on its own

Four categories, all meaning *users in the wild are hurting right now*:

1. **A regression from a shipped release** — something that worked in version N is broken in N+1.
2. **Home Assistant compatibility** — a new HA stable broke the integration. HA ships monthly and
   has done this repeatedly; 2026.9 alone renamed every built-in tool (v3.41.1) and deprecated a
   device-registry accessor (v3.38.1).
3. **Security or safety** — credential exposure, or a safety control not actually operating. The
   critical-action PIN gate silently skipping locks (v3.41.1) is the canonical example, as is the
   API key leak in v3.33.0.
4. **Data loss, corruption, or a failure to start** — migration failures, checkpoint corruption,
   crashes on setup.

A fifth case deserves its own line because it is this project's recurring failure shape:
**a feature that is silently doing nothing.** Tool retrieval returning dead rows, an STT phrase
list being dropped by a compatibility layer, an evidence-less rule key — these log no error, so
users don't report them and can't tell they're affected. When one is found, it is broken in the
wild and ships now, even though nothing appears to be on fire.

### Bundle everything else

New features, enhancements to features that already work, documentation, refactors, dependency
bumps, test-only changes, P2/P3 follow-ups from `TODOS.md`, and contributor PRs that don't meet
the bar above. Being *finished* is not a reason to release; the work is already on `main` and
loses nothing by waiting.

Roadmap features are the clearest case. The `audit_home_security` tool (v3.40.0) shipped its own
release on a day that already had four others, with nobody waiting on it.

## Cadence

**One bundle a month, cut roughly a week after Home Assistant's monthly stable.**

Two reasons. Compatibility fixes for the new HA version cluster in the days after it lands and
fall into the bundle naturally. And users have already restarted for HA's update, so the
integration's update rides a restart they were taking anyway.

Skip a month if little accumulated — don't pad a release to keep a schedule. Don't let a bundle
run past about six weeks; long gaps make regression triage harder and leave reporters waiting.

## Pre-releases

**Field validation does not require a public release.** Mark the release as a pre-release and
HACS offers it only to users who have enabled beta versions for this repository; everyone else
sees nothing. The `Release` workflow fires on `release: published`, which pre-releases also
trigger, so the ZIP asset is built and attached exactly as for a stable release.

Use one when:

- A reporter needs to verify a fix on hardware or against a service the maintainer cannot reach —
  an OpenRouter key, a Czech locale, a Z-Wave stick, a particular model provider.
- A change is large enough to want real-world exposure before it reaches everyone.

Ask the reporter to enable beta versions for Home Generative Agent in HACS, then install the
pre-release. (Confirm the exact HACS menu wording the first time you walk someone through it.)

Two mechanics differ from a stable cut:

- **Number it for the bundle it precedes**, with a semver pre-release suffix: the beta before the
  `3.43.0` bundle is `3.43.0-beta.1`, a second one `3.43.0-beta.2`. It sorts below the bundle, so
  a tester upgrades into the stable release when it lands rather than appearing to downgrade. Do
  not spend a stable patch number on a build that never ships to everyone.
- **Leave `## [Unreleased]` alone.** A beta does not consume the bundle's notes — those entries
  still have to appear under the real release. Draw the GitHub release body from `[Unreleased]`
  instead, leading with the fix being validated, and rename the heading only at the stable cut.

## Versions

Semantic-ish, with the bundle giving the numbers meaning again:

- **MINOR** (`3.43.0`) — a scheduled bundle containing any user-visible addition or change.
- **PATCH** (`3.42.1`) — a hotfix, or a bundle that is nothing but fixes.
- **MAJOR** — a breaking change to configuration or behavior that requires user action.

The git tag and `manifest.json` `version` must always match.

## Accumulating between bundles

`CHANGELOG.md` carries an `## [Unreleased]` section at the top. **Every PR that changes
user-visible behavior adds its entry there** under the usual headings (`Added`, `Changed`,
`Fixed`, `Security`), in the same voice as existing entries: what changed, what a user would have
seen when it was wrong, and the PR link. A PR with user-visible behavior and no changelog entry
is not finished.

Cutting a release then means renaming that heading — the notes are already written, by the person
who had the context, at the time they had it.

## Cutting a release

1. `make lint`, `make test`, `make typecheck` all green; CI green on `main`.
2. Bump `"version"` in `custom_components/home_generative_agent/manifest.json`. There is no
   VERSION file — this is the only place the version lives.
3. In `CHANGELOG.md`, rename `## [Unreleased]` to `## [X.Y.Z] - YYYY-MM-DD` and open a fresh
   empty `## [Unreleased]` above it.
4. Commit and push to `main`.
5. `gh release create vX.Y.Z` with notes drawn from the changelog section. Add `--prerelease`
   for a beta.
6. The `Release` workflow zips `custom_components/home_generative_agent/` and attaches
   `home_generative_agent.zip`.
7. **Verify the asset by hashing the files inside the ZIP against the repo tree.** ZIP byte size
   is not a check — it varies with compression and says nothing about contents.

## Telling reporters

A bundled fix means someone who reported a bug waits weeks for a release instead of hours. That
responsiveness is worth protecting, so say the schedule out loud rather than leaving them
guessing. When closing an issue whose fix is bundled, state that it is merged to `main`, name the
release it will ship in, and offer a pre-release if they want to verify sooner.

Urgent fixes are unaffected — they ship the same day, as they always have.

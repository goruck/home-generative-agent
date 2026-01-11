---
name: Code Review Request
about: Request a full code review for this project
title: "Code review: project-wide"
labels: ["needs-review"]
assignees: []
---

## Summary
We should schedule a full code review for this repository to confirm correctness, security posture, and maintainability.

## Why now
- Recent changes merit a deeper look for regressions and missed edge cases.
- A structured review will help identify technical debt and clarify ownership.

## Scope
- Core logic under `custom_components/`
- Tests under `tests/`
- CI/scripts under `scripts/` and configs in `config/`

## Review goals
- Identify bugs, security risks, and behavior regressions.
- Validate tests and coverage for critical paths.
- Highlight refactors or documentation gaps that unblock future work.

## Definition of done
- Findings recorded with severity and file references.
- Action items tracked with owners and timelines.
- Follow-up fixes (if any) happen on a dedicated branch and land via PR.

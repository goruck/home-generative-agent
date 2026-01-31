# AGENTS.md

## Purpose

This document defines **mandatory constraints and expectations** for any coding agent contributing to **Home Generative Agent**, a Home Assistant custom integration.

This project implements **proactive, safety-critical, deterministic agent behavior**.  
All contributions must preserve correctness, explainability, and Home Assistant safety semantics.

Failure to follow this document is grounds for rejection of a PR.

---

## Core Architectural Principles (Non-Negotiable)

1. **Determinism First**
   - Device state, anomaly detection, and decisions must be deterministic.
   - LLMs may *explain* or *rank* but must never *decide*, *detect*, or *act*.

2. **Structured Truth, Language Second**
   - Authoritative state must be structured, machine-readable JSON.
   - Natural language output is advisory only.

3. **Safety Before Convenience**
   - Locks, alarms, doors, garages, and other access controls require explicit confirmation.
   - PIN and alarm-code flows must never be bypassed.

4. **Proactive ≠ Autonomous**
   - The agent may initiate notifications.
   - The agent must not autonomously execute sensitive actions.

---

## Canonical Repository Layout

Agents must follow this structure unless explicitly instructed otherwise.

custom_components/home_generative_agent/
init.py
manifest.json
snapshot/ # Full structured home state (authoritative)
sentinel/ # Deterministic anomaly detection
explain/ # LLM explanation layer (optional, non-authoritative)
notify/ # Proactive notifications + action handling
audit/ # Persistent audit trail
tests/ # Deterministic tests

---

## Folder Responsibilities

### `snapshot/`
**Authoritative home state**

- No LLM usage
- No natural language
- JSON-serializable only
- Must include:
  - entities, sensors, locks, alarms
  - camera activity metadata
  - derived context (presence, night/day, last motion)

### `sentinel/`
**Proactive anomaly detection**

- Pure logic
- Deterministic
- Testable from synthetic snapshots
- No Home Assistant service calls
- No LLM usage

### `explain/`
**Optional enhancement layer**

- Converts `AnomalyFinding` → user-facing text
- Must not introduce new facts
- Must degrade gracefully if disabled

### `notify/`
**User interaction boundary**

- Notifications
- Confirmation flows
- PIN / alarm code handling
- No anomaly logic

### `audit/`
**Transparency and accountability**

- Persistent logs
- Snapshot references
- User responses
- Action outcomes

---

## Ruff & Code Style Rules

This repository enforces **Ruff**. All agent-generated code must comply.

### Required Characteristics

- Explicit typing preferred
- No unused imports
- No implicit re-exports
- Clear async boundaries
- Guarded access to Home Assistant internals

### Naming Conventions

- Modules: `snake_case`
- Classes: `PascalCase`
- Functions: `snake_case`
- Constants: `UPPER_SNAKE_CASE`

### Forbidden Patterns

- `print()` (use logging)
- Broad `except Exception`
- Silent failures
- Mutating HA state outside allowed boundaries
- LLM calls inside detection logic

---

## Tooling Expectations

Agents should assume:

- **Python ≥ 3.11**
- Home Assistant async patterns (`async def`, `await`)
- HA `Store` for persistence
- HA `async_create_task` for background loops

Do **not** introduce:
- External schedulers
- Blocking I/O
- Threads unless explicitly approved

---

## LLM Usage Rules (Strict)

LLMs are **non-authoritative**.

### Allowed
- Explanation
- Summarization of findings
- Ranking of suggestions
- Tone and wording refinement

### Forbidden
- Detecting anomalies
- Interpreting raw HA state
- Deciding whether an action is safe
- Executing actions
- Inferring facts not present in inputs

### Required Safeguards
- Prompts must explicitly forbid fact invention
- Inputs must be a *subset* of structured evidence
- Deterministic fallback text must exist

---

## Proactive Sentinel Rules

All anomaly rules must:

1. Accept a `FullStateSnapshot`
2. Return zero or more `AnomalyFinding`
3. Be idempotent
4. Be testable without Home Assistant runtime
5. Include explicit evidence references

**Never**
- Read global state
- Call HA services
- Call LLMs

---

## Safety & Confirmation Requirements

### Sensitive Domains
- `lock`
- `alarm_control_panel`
- `cover` (garage/doors)
- Any exterior access device

### Rules
- No sensitive action executes without explicit user confirmation
- PIN / alarm code flows must follow HA conventions
- Failure or refusal must abort action cleanly

Agents must **not** invent alternative confirmation mechanisms.

---

## CI Expectations

A PR is not acceptable unless:

- All tests pass
- Snapshot schema validates
- New rules include tests
- No Ruff violations
- Logging verbosity is appropriate

### Tests Required For
- New snapshot fields
- New anomaly rules
- Suppression/cooldown logic
- Notification flows (mocked)

---

## Testing Philosophy

- Prefer **snapshot-driven tests**
- Avoid integration tests unless required
- Synthetic snapshots should be explicit and minimal
- Expected outputs must be exact (no fuzzy matching)

---

## Agent Do / Do Not Summary

### DO
- Preserve determinism
- Add tests with every rule
- Keep layers separated
- Fail safely
- Log decisions clearly

### DO NOT
- Collapse layers for convenience
- Let LLM output drive logic
- Bypass confirmation flows
- Add opaque abstractions
- Optimize prematurely at the cost of clarity

---

## Final Rule

If there is uncertainty about whether a change violates these principles, **stop and ask before implementing**.

This project prioritizes **correctness, safety, and explainability** over novelty or speed.

# AGENTS.md — home-generative-agent

## Project summary
Home Assistant custom integration providing a generative agent. Core code lives in
`custom_components/home_generative_agent/`, tests in `tests/`.

## Repo layout
- `custom_components/home_generative_agent/` — integration code
- `tests/` — pytest suite
- `blueprints/` — HA blueprints used by the integration
- `scripts/` — helper scripts (notably `gen_manifest_requirements.py`)
- `requirements/` + `requirements_runtime_manifest.txt` — dependency management

## Integration module map
- `sentinel/` — anomaly detection engine, dynamic rules, proposals, suppression
- `snapshot/` — snapshot schema/builders/reducers used by Sentinel and explain flows
- `notify/` — mobile notification dispatch and action helpers
- `explain/` — prompt templates and LLM-backed explanation/discovery helpers

## Development environment
- Python: 3.13
- Virtualenv: `hga/` (managed by Makefile)
- Primary tools: `ruff` (format + lint), `pyright` (types), `pytest`

## Common commands (Makefile)
- `make venv` — create venv
- `make devdeps` — install dev deps
- `make testdeps` — install test deps
- `make runtimedeps` — regenerate + install runtime deps from manifest
- `make lint` — regenerate runtime deps + `ruff check` (non-mutating)
- `make format` — `ruff format`
- `make fix` — `ruff check --fix`
- `make test` — pytest (sets `PYTHONPATH` to repo root)
- `make typecheck` — `pyright`
- `make all` — devdeps + testdeps + runtimedeps + lint + test + check + typecheck

Note: `make lint` will fail if `requirements_runtime_manifest.txt` is out of date.

## Dependency workflow
Runtime dependencies are sourced from
`custom_components/home_generative_agent/manifest.json`.

- Don’t edit `requirements_runtime_manifest.txt` directly.
- After changing `manifest.json`, run `scripts/gen_manifest_requirements.py`
  (or `make runtimedeps`).

## Coding conventions
- Formatting/linting: `ruff format` + `ruff check`
- Type checking: `pyright` (standard mode)
- Prefer async I/O and Home Assistant patterns (`async_*` APIs,
  `async_add_executor_job` for blocking work).
- Update docs when behavior or config changes (README is the primary user doc).

## Testing
- Run `make test` for the full suite.
- Keep tests under `tests/` and mirror integration module structure where
  practical.

## Notes
- The integration is a Home Assistant service integration (`manifest.json`).
- Prefer the Makefile workflow for setup/tasks (`make devdeps`, `make testdeps`,
  `make runtimedeps`) over legacy helper scripts.

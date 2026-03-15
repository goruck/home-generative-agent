# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Summary

Home Assistant custom integration that uses LangChain/LangGraph to provide a generative AI agent for smart home automation, image analysis, and anomaly detection. Core code lives in `custom_components/home_generative_agent/`, tests in `tests/`.

## Commands

All commands use the `hga/` virtualenv (Python 3.14), managed by the Makefile.

```bash
# Initial setup
make venv        # Create venv
make devdeps     # Install dev dependencies
make testdeps    # Install test dependencies
make runtimedeps # Regenerate requirements_runtime_manifest.txt from manifest.json and install

# Day-to-day
make lint        # Non-mutating: verify manifest is current, ruff format check + ruff check
make format      # Mutating: ruff format
make fix         # Mutating: ruff check --fix
make test        # Run full pytest suite
make typecheck   # pyright --stats

# Run a single test file
PYTHONPATH=$(pwd) hga/bin/pytest tests/custom_components/home_generative_agent/test_foo.py

# Run tests matching a pattern
PYTHONPATH=$(pwd) hga/bin/pytest -k "test_name_pattern"
```

`make lint` fails if `requirements_runtime_manifest.txt` is out of date with `manifest.json`.

## Dependency Workflow

Runtime dependencies are declared in `manifest.json` (the `requirements` array). Do **not** edit `requirements_runtime_manifest.txt` directly — it is auto-generated.

After changing `manifest.json`: run `make runtimedeps` (or `python scripts/gen_manifest_requirements.py`).

## Architecture

### Module Map

- **`agent/`** — LangGraph state machine (`graph.py`), tool implementations (`tools.py`), cross-provider token counting, camera activity tracking
- **`sentinel/`** — Anomaly detection engine: rules evaluation, LLM-based triage & discovery, baseline statistics, notifications, audit trail
- **`snapshot/`** — Authoritative JSON representation of home state (entities + derived context + camera activity); consumed by Sentinel and explain flows
- **`explain/`** — LLM-backed explanation generation and discovery prompt templates
- **`core/`** — Video analysis, DB utilities, schema migrations, face recognition helpers, config subentry resolution
- **`flows/`** — Home Assistant config flow subentries (model providers, features, database, STT, Sentinel)
- **`notify/`** — Mobile push + persistent notification dispatch with snooze/action buttons

Top-level platform files: `conversation.py`, `sensor.py`, `image.py`, `stt.py`, `http.py`, `config_flow.py`.

### Agent Graph (`agent/graph.py`)

LangGraph state machine implementing the conversation agent. Key behaviors: tool-use for entity control and automation creation, token-count-aware message trimming, critical-action PIN verification, PostgreSQL checkpoint persistence.

### Sentinel Pipeline (`sentinel/`)

Layered anomaly detection — each layer is independently optional:

1. **Snapshot** — builds home state JSON
2. **Rules** — deterministic evaluation (static built-in rules + user-approved dynamic rules)
3. **Triage** (optional LLM) — filters low-value alerts; fails open; cannot modify findings
4. **Discovery** (optional LLM) — proposes new rule candidates; advisory only, requires user approval
5. **Baseline** (optional) — statistical anomaly detection via rolling averages
6. **Notifier** — mobile push + persistent HA notifications
7. **Audit** — persists findings, user actions, KPIs

**Safety invariant:** LLM never executes actions or gates safety behavior. All actuation is deterministic.

### Database

PostgreSQL with pgvector extension. Used for:
- LangGraph checkpoint persistence (conversation memory)
- Vector embeddings (semantic search in Sentinel discovery)
- Audit trail storage

Schema migrations are managed in `core/migrations.py`. Dev container spins up a `pgvector_db` service automatically.

### Configuration

Integration uses HA config flow subentries (not the legacy single-entry flow). Features, model providers, database connection, Sentinel, and STT provider are each configured as independent subentries. Global options (system prompt, face-service URL, critical-action PIN) live in the options flow.

## Coding Conventions

- Async-first: use `async_*` HA APIs; use `async_add_executor_job` for blocking work.
- Type checking: pyright in standard mode. All public functions should be typed.
- Linting: ruff with `select = ["ALL"]`; max complexity 25. Tests have relaxed rules (allow untyped fixtures, magic values, private access).
- Update README.md when behavior or configuration changes — it is the primary user documentation.
- `const.py` is the authoritative source for configuration keys, default values, and model specs.

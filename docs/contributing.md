# Contributing and Development

Contributions are welcome! Please read the [Contribution guidelines](../CONTRIBUTING.md) before submitting a pull request.

- [Development Setup](#development-setup)
- [Makefile Reference](#makefile-reference)
- [Deploying to Home Assistant](#deploying-to-home-assistant)
- [Dependency Workflow](#dependency-workflow)
- [Translations](#translations)

---

## Development Setup

All commands use the `hga/` virtualenv (Python 3.14), managed by the Makefile.

```bash
make venv        # Create the virtualenv
make devdeps     # Install dev-only dependencies
make testdeps    # Install test dependencies
make runtimedeps # Regenerate requirements_runtime_manifest.txt and install runtime deps
```

Run a single test file:

```bash
PYTHONPATH=$(pwd) hga/bin/pytest tests/custom_components/home_generative_agent/test_foo.py
```

Run tests matching a pattern:

```bash
PYTHONPATH=$(pwd) hga/bin/pytest -k "test_name_pattern"
```

---

## Makefile Reference

**Day-to-day:**

```bash
make lint        # Non-mutating: verify manifest is current, ruff format check + ruff check
make format      # Mutating: ruff format
make fix         # Mutating: ruff check --fix
make test        # Run full pytest suite (runtime deps must be installed)
make typecheck   # pyright --stats
```

**Full suite:**

```bash
make all         # devdeps + testdeps + runtimedeps + lint + test + typecheck
make clean       # Remove the venv
```

> `make lint` fails if `requirements_runtime_manifest.txt` is out of date with `manifest.json`. Run `make runtimedeps` to regenerate it.

---

## Deploying to Home Assistant

`scripts/deploy` syncs the integration to a running HA instance via rsync over SSH.

```bash
# Basic usage (defaults to root@ and /homeassistant config path)
scripts/deploy 192.168.1.240

# Explicit user and config path
scripts/deploy root@192.168.1.240 /homeassistant

# Skip the automatic HA restart
scripts/deploy 192.168.1.240 --no-restart

# Via environment variable
HA_HOST=192.168.1.240 scripts/deploy
```

Only files that differ (by checksum) are transferred; `__pycache__` and `.pyc` files are excluded. After a successful sync the script calls `ha core restart` over SSH to reload the integration.

**SSH key setup (one-time):** Add your public key (`~/.ssh/id_rsa.pub` or `~/.ssh/id_ed25519.pub`) to the HA SSH add-on under **Settings → Add-ons → SSH & Web Terminal → Configuration → authorized_keys**, then restart the add-on.

---

## Dependency Workflow

Runtime dependencies are declared in `manifest.json` (the `requirements` array). Do **not** edit `requirements_runtime_manifest.txt` directly — it is auto-generated.

After changing `manifest.json`:

```bash
make runtimedeps
# or
python scripts/gen_manifest_requirements.py
```

---

## Translations

The configuration UI is translated via the standard Home Assistant mechanism. `custom_components/home_generative_agent/strings.json` is the English source of truth; `translations/en.json` must be an exact copy of it. Localized files (currently `cs.json`, `ru.json`, `tr.json`) live alongside it.

To add a language, copy `translations/en.json` to `translations/<code>.json` and translate the values. Rules, enforced by `tests/custom_components/home_generative_agent/test_translations.py`:

- A translation file may lag behind `en.json` (Home Assistant falls back to English for missing keys) but must never contain keys that don't exist in `en.json`.
- `{placeholder}` names on shared keys must match `en.json` exactly.
- No value may have leading or trailing whitespace (hassfest rejects it).

Run the checks with:

```bash
PYTHONPATH=$(pwd) hga/bin/pytest tests/custom_components/home_generative_agent/test_translations.py
```

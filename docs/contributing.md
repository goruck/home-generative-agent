# Contributing and Development

Contributions are welcome! Please read the [Contribution guidelines](../CONTRIBUTING.md) before submitting a pull request.

- [Development Setup](#development-setup)
- [Makefile Reference](#makefile-reference)
- [Dependency Workflow](#dependency-workflow)

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

## Dependency Workflow

Runtime dependencies are declared in `manifest.json` (the `requirements` array). Do **not** edit `requirements_runtime_manifest.txt` directly — it is auto-generated.

After changing `manifest.json`:

```bash
make runtimedeps
# or
python scripts/gen_manifest_requirements.py
```

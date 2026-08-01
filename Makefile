VENV := hga
PYTHON_BIN ?= $(shell command -v python3.14 2>/dev/null || printf '%s' "$(HOME)/.pyenv/shims/python3.14")
PY := $(VENV)/bin/python
PIP := $(VENV)/bin/pip
RUFF := $(VENV)/bin/ruff
PYTEST := $(VENV)/bin/pytest
PYRIGHT := $(VENV)/bin/pyright

.PHONY: venv devdeps testdeps runtimedeps check lint format fix test all clean typecheck

# Create venv and basic packaging tooling only. No linting here.
venv:
	$(PYTHON_BIN) -m venv $(VENV)
	$(PY) -m pip install -U pip setuptools wheel

devdeps: venv
	$(PIP) install -r requirements/dev.txt

testdeps: venv
	$(PIP) install -r requirements/test.txt

# Generate pinned runtime requirements from manifest and install them into the venv
runtimedeps: venv
	$(PY) scripts/gen_manifest_requirements.py
	$(PIP) install -r requirements_runtime_manifest.txt

check: venv
	$(PIP) check

# Non-mutating checks: enforce generated runtime requirements + lint + format
lint: devdeps
	$(PY) scripts/gen_manifest_requirements.py
	git diff --exit-code -- requirements_runtime_manifest.txt
	$(RUFF) format --check custom_components tests
	$(RUFF) check custom_components tests

# Mutating formatting
format: devdeps
	$(RUFF) format custom_components tests

# Mutating lint fixes (kept separate from format)
fix: devdeps
	$(RUFF) check --fix custom_components tests

# Test suite (ensures harness + runtime deps are installed)
test: testdeps runtimedeps
	PYTHONPATH=$(CURDIR) $(PYTEST)

# Static type checking
typecheck: devdeps runtimedeps
	$(PYRIGHT) --stats

# Convenience target
all: devdeps testdeps runtimedeps lint test check typecheck

clean:
	rm -rf $(VENV)

VENV := hga
PY := $(VENV)/bin/python
PIP := $(VENV)/bin/pip
RUFF := $(VENV)/bin/ruff
PYTEST := $(VENV)/bin/pytest
PYRIGHT := $(VENV)/bin/pyright

.PHONY: venv devdeps testdeps runtimedeps check lint format fix test all clean typecheck

# Create venv and basic packaging tooling only. No linting here.
venv:
	python3.13 -m venv $(VENV)
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

# Non-mutating checks: enforce generated runtime requirements + lint
lint: devdeps
	$(PY) scripts/gen_manifest_requirements.py
	git diff --exit-code -- requirements_runtime_manifest.txt
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
